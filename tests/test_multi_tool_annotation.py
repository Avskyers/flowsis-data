import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import annotate
import verify_data
from annotations import IMAGE_OBJECT_FIELDS, read_csv, write_csv
from image_review import Candidate, review_image_instances
from masks import load_mask_bundle


def instance(identity, tool_id, tool_class="scissors"):
    return {
        "Image ID": "image-1",
        "Instance ID": identity,
        "Tool ID": tool_id,
        "Tool Class": tool_class,
        "Tool Name": f"tool {tool_id}",
        "Confidence": "0.8",
    }


def candidates():
    masks = np.zeros((3, 10, 12), dtype=np.bool_)
    masks[0, 1:4, 2:5] = True
    masks[1, 3:8, 3:8] = True  # Crosses the other tool's visible region.
    masks[2, 7:9, 2:5] = True
    return [
        Candidate(mask, 0.9, "scissors", f"candidate-{index}")
        for index, mask in enumerate(masks)
    ]


def interactive_review(
    monkeypatch, tmp_path, commands, rows=None, initial=None, predict=None
):
    commands = iter(commands)
    monkeypatch.setattr("builtins.input", lambda _: next(commands))
    log = annotate.ReviewLog(tmp_path / "review.jsonl")
    result = review_image_instances(
        Image.new("RGB", (12, 10)),
        candidates() if initial is None else initial,
        rows or [instance("a", "12"), instance("b", "13")],
        key="image-1",
        show=lambda *args: None,
        predict=predict or (lambda _: []),
        log=log.append,
    )
    return result, [json.loads(line) for line in log.path.read_text().splitlines()]


def test_grouping_preserves_gaps_and_source_history(monkeypatch, tmp_path, capsys):
    result, records = interactive_review(
        monkeypatch,
        tmp_path,
        [
            "a",
            "g a 0,2",
            "v a partial",
            "g b 1",
            "a",
        ],
    )
    assert "Resolve tool a" in capsys.readouterr().out
    merged = result.selected[0]
    np.testing.assert_array_equal(
        merged.mask, candidates()[0].mask | candidates()[2].mask
    )
    assert not merged.mask[5, 3]
    assert merged.visibility == "partial"
    grouped = next(
        record for record in records if record.get("action") == "assign_fragments"
    )
    assert [part["candidate_id"] for part in grouped["assignments"]["a"]] == [
        "candidate-0",
        "candidate-2",
    ]


def test_reprompt_preserves_assignments_and_resolves_extra_tool(monkeypatch, tmp_path):
    result, records = interactive_review(
        monkeypatch,
        tmp_path,
        [
            "g a 0,2",
            "v b occluded",
            "x 1",
            "p forceps",
            'n c 14 "forceps" "fine forceps" 0.9',
            "g c 3",
            "a",
        ],
        predict=lambda _: [candidates()[1]],
    )
    assert [part.identity for part in result.selected] == ["a", "b", "c"]
    assert result.selected[1].visibility == "occluded"
    assert not result.selected[1].mask.any()
    assert records[-1]["assignments"]["a"][0]["index"] == 0
    assert result.instances[2]["Tool Class"] == "forceps"


def test_conflicting_assignments_and_unresolved_candidates_block_acceptance(
    monkeypatch, tmp_path, capsys
):
    result, _ = interactive_review(
        monkeypatch,
        tmp_path,
        [
            "g a 0",
            "g b 0",
            "v b absent",
            "a",
            "i 1",
            "x 2",
            "a",
        ],
    )
    output = capsys.readouterr().out
    assert "Clear the other tool" in output
    assert "Assign, exclude, or ignore" in output
    assert result.selected[1].visibility == "absent"
    np.testing.assert_array_equal(result.ignored[0].mask, candidates()[1].mask)


def setup_manifest(tmp_path, rows):
    (tmp_path / "images").mkdir()
    Image.new("RGB", (12, 10)).save(tmp_path / "images/tools.jpg")
    write_csv(
        tmp_path / "image_manifest.csv",
        [{"Image ID": "image-1", "Filepath": "images/tools.jpg"}],
        ["Image ID", "Filepath"],
    )
    write_csv(tmp_path / "image_object_manifest.csv", rows, IMAGE_OBJECT_FIELDS)


def test_cli_saves_overlapping_tools_and_resumes(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    setup_manifest(
        tmp_path,
        [instance("a", "12"), instance("b", "13"), instance("c", "14", "forceps")],
    )
    monkeypatch.setattr(sys, "argv", ["annotate.py", "images"])
    prompts_seen = []

    class Backend:
        def __init__(self, *args, **kwargs):
            self.refine_crop = kwargs["refine_crop"]

        def predict(self, images, prompts):
            assert not self.refine_crop
            prompts_seen.extend(prompts)
            return [candidates() if prompt == "scissors" else [] for prompt in prompts]

    monkeypatch.setattr(annotate, "Sam3ImageBackend", Backend)
    monkeypatch.setattr(annotate, "show_candidates", lambda *args: None)
    commands = iter(["g a 0,2", "g b 1", "v c occluded", "a"])
    monkeypatch.setattr("builtins.input", lambda _: next(commands))
    annotate.main()
    assert prompts_seen == ["scissors", "forceps"]  # Same-class tools share inference.
    bundle = load_mask_bundle("masks/images/tools.npz")
    assert bundle.identities == ("a", "b")
    assert np.any(bundle.masks[0] & bundle.masks[1])
    rows = read_csv("image_object_manifest.csv")
    assert [row["Tool ID"] for row in rows] == ["12", "13", "14"]
    assert rows[2]["Visibility"] == "occluded"
    assert rows[2]["Mask Index"] == "-1"
    monkeypatch.setattr(verify_data, "ROOT", tmp_path)
    assert not verify_data.validate_image_annotations(
        read_csv("image_manifest.csv")[0], rows
    )
    rows[0]["Mask Index"] = "1"
    assert verify_data.validate_image_annotations(
        read_csv("image_manifest.csv")[0], rows
    )
    annotate.main()
    assert prompts_seen == ["scissors", "forceps"]


@pytest.mark.parametrize("flag", ["--single-tool", "--auto-accept"])
def test_unsafe_automatic_multi_tool_modes_fail_before_model_load(
    monkeypatch, tmp_path, flag
):
    monkeypatch.chdir(tmp_path)
    setup_manifest(tmp_path, [instance("a", "12"), instance("b", "13")])
    monkeypatch.setattr(sys, "argv", ["annotate.py", "images", flag])
    monkeypatch.setattr(
        annotate,
        "Sam3ImageBackend",
        lambda *args, **kwargs: pytest.fail("Loaded model"),
    )
    with pytest.raises(ValueError):
        annotate.main()


def test_high_iou_candidates_survive_until_review(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    setup_manifest(tmp_path, [instance("a", "12"), instance("b", "13")])
    monkeypatch.setattr(sys, "argv", ["annotate.py", "images"])
    mask = candidates()[0].mask

    class Backend:
        def __init__(self, *args, **kwargs):
            pass

        def predict(self, images, prompts):
            return [
                [Candidate(mask.copy(), 0.9, prompt, str(index)) for index in range(2)]
                for prompt in prompts
            ]

    displayed = []
    monkeypatch.setattr(annotate, "Sam3ImageBackend", Backend)
    monkeypatch.setattr(
        annotate,
        "show_candidates",
        lambda image, parts, key: displayed.append(len(parts)),
    )
    commands = iter(["g a 0", "g b 1", "a"])
    monkeypatch.setattr("builtins.input", lambda _: next(commands))
    annotate.main()
    assert displayed[0] == 2
    bundle = load_mask_bundle("masks/images/tools.npz")
    np.testing.assert_array_equal(bundle.masks[0], bundle.masks[1])


def test_declare_sample_preserves_metadata_and_reopen_clears_annotations(
    monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    setup_manifest(tmp_path, [instance("a", "12")])
    images = read_csv("image_manifest.csv")
    images[0]["Sample ID"] = "4"
    images[0]["Angle"] = "0"
    write_csv("image_manifest.csv", images, images[0].keys())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "annotate.py",
            "declare",
            "--sample-id",
            "4",
            "--tool-id",
            "13",
            "--tool-class",
            "forceps",
            "--tool-name",
            "fine",
            "--confidence",
            "0.7",
        ],
    )
    annotate.main()
    assert read_csv("image_manifest.csv")[0]["Tool IDs"] == "12;13"
    assert read_csv("image_object_manifest.csv")[0]["Instance ID"] == "a"
    log = annotate.ReviewLog(Path("image_review.jsonl"))
    log.append("image-1", "accepted")
    argv = [
        "annotate.py",
        "declare",
        "--sample-id",
        "4",
        "--tool-id",
        "14",
        "--tool-class",
        "forceps",
        "--tool-name",
        "large",
        "--confidence",
        "0.8",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(ValueError, match="--reopen"):
        annotate.main()
    monkeypatch.setattr(sys, "argv", [*argv, "--reopen"])
    annotate.main()
    assert not annotate.ReviewLog(Path("image_review.jsonl")).completed("image-1")
    assert len(read_csv("image_object_manifest.csv")) == 3


def test_resolved_rows_follow_mask_order_and_reject_missing_identity():
    rows = [instance("a", "12"), instance("b", "13")]
    selected = [
        Candidate(candidates()[1].mask, 0.9, "scissors", "b"),
        Candidate(candidates()[0].mask, 0.9, "scissors", "a"),
    ]
    result = annotate.resolved_image_rows(rows, selected, Path("mask.npz"))
    assert [row["Mask Index"] for row in result] == [1, 0]
    with pytest.raises(ValueError, match="Every declared instance"):
        annotate.resolved_image_rows(rows, selected[:1], Path("mask.npz"))


def test_explicit_empty_image_retains_resolved_tool_metadata(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    setup_manifest(tmp_path, [instance("a", "12"), instance("b", "13")])
    monkeypatch.setattr(sys, "argv", ["annotate.py", "images"])

    class Backend:
        def __init__(self, *args, **kwargs):
            pass

        def predict(self, images, prompts):
            return [[] for _ in images]

    monkeypatch.setattr(annotate, "Sam3ImageBackend", Backend)
    monkeypatch.setattr(annotate, "show_candidates", lambda *args: None)
    monkeypatch.setattr("builtins.input", lambda _: "e")
    annotate.main()
    bundle = load_mask_bundle("masks/images/tools.npz")
    assert bundle.masks.shape == (0, 10, 12)
    rows = read_csv("image_object_manifest.csv")
    assert [row["Tool ID"] for row in rows] == ["12", "13"]
    assert all(row["Visibility"] == "absent" for row in rows)
    assert all(row["Annotation Complete"] == "TRUE" for row in rows)
    assert all(row["Reviewed"] == "FALSE" for row in rows)
    monkeypatch.setattr(verify_data, "ROOT", tmp_path)
    assert not verify_data.validate_image_annotations(
        read_csv("image_manifest.csv")[0], rows
    )


def test_declare_new_image_and_reject_duplicate_tool(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    setup_manifest(tmp_path, [instance("a", "12")])
    Image.new("RGB", (12, 10)).save("images/new.jpg")
    argv = [
        "annotate.py",
        "declare",
        "--filepath",
        "images/new.jpg",
        "--tool-id",
        "13",
        "--tool-class",
        "forceps",
        "--tool-name",
        "fine",
        "--confidence",
        "0.7",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    annotate.main()
    rows = read_csv("image_manifest.csv")
    assert len(rows) == 2
    added = next(row for row in rows if row["Filepath"] == "images/new.jpg")
    assert added["Tool IDs"] == "13"
    assert added["Capture Session ID"]
    before = Path("image_object_manifest.csv").read_text()
    with pytest.raises(ValueError, match="unique"):
        annotate.main()
    assert Path("image_object_manifest.csv").read_text() == before
