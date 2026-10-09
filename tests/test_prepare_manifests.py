from pathlib import Path

from PIL import Image
import pytest

import prepare_manifests
from annotations import IMAGE_OBJECT_FIELDS, read_csv, write_csv


def test_prepare_manifests_creates_instances_reuses_metadata_and_preserves_annotations(
    tmp_path, monkeypatch
):
    images = tmp_path / "images"
    raw = tmp_path / "raw-images"
    (images / "7").mkdir(parents=True)
    (raw / "7").mkdir(parents=True)
    manifest = tmp_path / "image_manifest.csv"
    object_manifest = tmp_path / "image_object_manifest.csv"
    fields = [
        "Image ID",
        "Capture Session ID",
        "Filepath",
        "Sample ID",
        "Angle",
        "Lights",
        *prepare_manifests.METADATA_COLUMNS,
    ]
    write_csv(manifest, [], fields)
    write_csv(object_manifest, [], IMAGE_OBJECT_FIELDS)
    metadata = dict.fromkeys(prepare_manifests.METADATA_COLUMNS, "known")
    metadata["Time Taken"] = "2026-09-01T12:00:00-04:00"
    monkeypatch.setattr(prepare_manifests, "extract_metadata", lambda path: metadata)
    answers = iter(["Scissors", "Iris", "0.8"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))
    for index in range(8):
        Image.new("RGB", (8, 8)).save(images / "7" / f"{index:02}.jpg")
        Image.new("RGB", (8, 8)).save(raw / "7" / f"{index:02}.jpg")
    prepare_manifests.main(images, raw, manifest)
    rows = read_csv(manifest)
    objects = read_csv(object_manifest)
    assert len(rows) == len(objects) == 8
    assert not {"Tool ID", "Tool Class", "Tool Name", "Confidence"} & rows[0].keys()
    assert {row["Angle"] for row in rows} == {
        str(angle) for angle in prepare_manifests.ANGLES
    }
    assert all(row["Capture Session ID"] for row in rows)
    assert all(obj["Tool ID"] == "7" and obj["Confidence"] == "0.8" for obj in objects)
    assert all(not obj["Reviewed"] and not obj["Mask Path"] for obj in objects)
    objects[0].update(
        {
            "Mask Path": "masks/images/7/00.npz",
            "Reviewed": "TRUE",
            "Annotation Confidence": "0.9",
        }
    )
    write_csv(object_manifest, objects, IMAGE_OBJECT_FIELDS)
    preserved = dict(objects[0])
    for index in range(8, 16):
        Image.new("RGB", (8, 8)).save(images / "7" / f"{index:02}.jpg")
        Image.new("RGB", (8, 8)).save(raw / "7" / f"{index:02}.jpg")
    answers = iter(["y"])
    prepare_manifests.main(images, raw, manifest)
    updated = read_csv(object_manifest)
    assert len(read_csv(manifest)) == len(updated) == 16
    assert updated[0] == preserved
    assert all(
        obj["Tool Name"] == "iris" and obj["Confidence"] == "0.8" for obj in updated
    )
    edited = read_csv(manifest)
    for row in edited[:8]:
        row["Sample ID"] = "50"
    edited[0]["Angle"], edited[1]["Angle"] = edited[1]["Angle"], edited[0]["Angle"]
    write_csv(manifest, edited, edited[0].keys())
    prepare_manifests.main(images, raw, manifest)
    reordered = read_csv(manifest)
    assert reordered[0]["Sample ID"] == "50"
    assert reordered[0]["Filepath"].endswith("00.jpg")
    assert reordered[0]["Angle"] == "45"
    assert read_csv(object_manifest)[0]["Sample ID"] == "50"
    assert all(row["Tool IDs"] == "7" for row in reordered)
    before = (manifest.read_bytes(), object_manifest.read_bytes())
    prepare_manifests.main(images, raw, manifest)
    assert before == (manifest.read_bytes(), object_manifest.read_bytes())


def test_multi_intake_reports_undeclared_images_without_inference(
    tmp_path, monkeypatch, capsys
):
    images = tmp_path / "images"
    raw = tmp_path / "raw-images"
    setup = images / "multi" / "setup-001"
    originals = raw / "multi" / "setup-001"
    setup.mkdir(parents=True)
    originals.mkdir(parents=True)
    for index in range(8):
        Image.new("RGB", (8, 8)).save(setup / f"{index:02}.jpg")
        Image.new("RGB", (8, 8)).save(originals / f"{index:02}.jpg")
    manifest = tmp_path / "image_manifest.csv"
    object_manifest = tmp_path / "image_object_manifest.csv"
    fields = [
        "Image ID",
        "Capture Session ID",
        "Filepath",
        "Sample ID",
        "Angle",
        "Lights",
        *prepare_manifests.METADATA_COLUMNS,
    ]
    rows = []
    for index in range(3):
        row = dict.fromkeys(fields, "")
        row.update(
            {
                "Image ID": str(index),
                "Filepath": f"images/multi/setup-001/{index:02}.jpg",
            }
        )
        rows.append(row)
    rows[0].update({"Sample ID": "100", "Angle": "90"})
    write_csv(manifest, rows, fields)
    objects = [
        {
            "Image ID": "0",
            "Instance ID": "a",
            "Tool ID": "12",
            "Tool Class": "scissors",
            "Tool Name": "iris",
            "Confidence": "0.9",
            "Mask Path": "masks/images/multi/setup-001/00.npz",
            "Mask Index": "0",
            "Reviewed": "TRUE",
        },
        {
            "Image ID": "0",
            "Instance ID": "b",
            "Tool ID": "13",
            "Tool Class": "forceps",
            "Tool Name": "fine",
            "Confidence": "0.8",
        },
        {
            "Image ID": "1",
            "Instance ID": "c",
            "Tool ID": "12",
            "Tool Class": "scissors",
            "Tool Name": "iris",
            "Confidence": "0.9",
        },
    ]
    write_csv(object_manifest, objects, IMAGE_OBJECT_FIELDS)
    metadata = dict.fromkeys(prepare_manifests.METADATA_COLUMNS, "known")
    metadata["Time Taken"] = "2026-09-01T12:00:00-04:00"
    sources = []

    def extract(path):
        sources.append(path)
        return metadata

    monkeypatch.setattr(prepare_manifests, "extract_metadata", extract)
    monkeypatch.setattr(
        "builtins.input", lambda _: pytest.fail("Prompted for multi tool identity")
    )
    prepare_manifests.main(images, raw, manifest)
    output = capsys.readouterr().out
    assert "Multi-tool images require explicit instance declarations" in output
    assert "images/multi/setup-001/02.jpg" in output  # Registered with no instances.
    assert "images/multi/setup-001/03.jpg" in output  # Not in the manifest yet.
    assert "annotate.py declare" in output
    assert "No new tool IDs found." in output
    updated = {row["Image ID"]: row for row in read_csv(manifest)}
    assert len(updated) == 3  # No automatic intake of multi files.
    assert updated["0"]["Tool IDs"] == "12;13"
    assert updated["0"]["Sample ID"] == "100"
    assert updated["0"]["Angle"] == "90"
    assert not updated["1"]["Sample ID"] and not updated["1"]["Angle"]
    assert all(row["Make"] == "known" for row in updated.values())
    assert set(sources) == {originals / f"{index:02}.jpg" for index in range(3)}
    refreshed = read_csv(object_manifest)
    assert len(refreshed) == 3
    reviewed = next(row for row in refreshed if row["Instance ID"] == "a")
    assert reviewed["Mask Path"] == objects[0]["Mask Path"]
    assert reviewed["Reviewed"] == "TRUE"
    assert reviewed["Sample ID"] == "100"
    assert {row["Tool ID"] for row in refreshed} == {"12", "13"}


def test_undeclared_multi_folder_does_not_interrupt_single_tool_intake(
    tmp_path, monkeypatch, capsys
):
    images = tmp_path / "images"
    raw = tmp_path / "raw-images"
    for root in (images, raw):
        (root / "7").mkdir(parents=True)
        (root / "multi" / "setup-001").mkdir(parents=True)
        Image.new("RGB", (8, 8)).save(root / "7" / "tool.jpg")
        Image.new("RGB", (8, 8)).save(root / "multi" / "setup-001" / "tools.jpg")
    manifest = tmp_path / "image_manifest.csv"
    object_manifest = tmp_path / "image_object_manifest.csv"
    fields = [
        "Image ID",
        "Capture Session ID",
        "Filepath",
        "Sample ID",
        "Angle",
        *prepare_manifests.METADATA_COLUMNS,
    ]
    write_csv(manifest, [], fields)
    write_csv(object_manifest, [], IMAGE_OBJECT_FIELDS)
    metadata = dict.fromkeys(prepare_manifests.METADATA_COLUMNS, "")
    monkeypatch.setattr(prepare_manifests, "extract_metadata", lambda _: metadata)
    answers = iter(["scissors", "iris", "0.8"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    prepare_manifests.main(images, raw, manifest)
    output = capsys.readouterr().out
    assert "images/multi/setup-001/tools.jpg" in output
    assert "New tool IDs found: 7" in output
    assert len(read_csv(manifest)) == 1
    assert read_csv(object_manifest)[0]["Tool ID"] == "7"
