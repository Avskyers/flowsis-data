from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image

from annotations import (
    CAPTURE_SESSION_FIELD,
    IMAGE_ID_FIELD,
    IMAGE_OBJECT_FIELDS,
    canonical_tool_label,
    extend_image_manifest,
    image_mask_path,
    load_taxonomy_aliases,
    normalize_tool_name,
    read_csv,
    stable_media_id,
    write_csv,
    write_image_manifests,
    validate_instance_metadata,
)
from masks import mask2xywh, save_mask_bundle
from image_review import Candidate, review_image_instances


class ReviewLog:
    """Append-only decisions; the latest decision for a key wins on resume."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.decisions: dict[str, dict[str, Any]] = {}
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    record = json.loads(line)
                    self.decisions[str(record["key"])] = record

    def completed(self, key: str) -> bool:
        return self.decisions.get(key, {}).get("decision") in {
            "accepted",
            "rejected",
        }

    def append(self, key: str, decision: str, **values: Any) -> None:
        record = {"key": key, "decision": decision, **values}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, sort_keys=True, default=str) + "\n")
            file.flush()
        self.decisions[key] = record


def _as_numpy(value: Any) -> NDArray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def extract_candidates(
    output: Mapping[str, Any] | None,
    *,
    prompt: str,
    identity_prefix: str = "instance",
    keep_empty: bool = False,
) -> list[Candidate]:
    if output is None or output.get("masks") is None:
        return []
    masks = _as_numpy(output["masks"])
    if masks.ndim == 2:
        masks = masks[None]
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim != 3:
        raise ValueError(f"Expected candidate masks [N,H,W], got {masks.shape}.")
    scores_value = output.get("scores")
    scores = (
        np.ones(len(masks), dtype=np.float32)
        if scores_value is None
        else _as_numpy(scores_value).reshape(-1)
    )
    object_ids = output.get("object_ids")
    identities = (
        [f"{identity_prefix}-{index}" for index in range(len(masks))]
        if object_ids is None
        else [str(value) for value in _as_numpy(object_ids).reshape(-1).tolist()]
    )
    if len(scores) != len(masks) or len(identities) != len(masks):
        raise ValueError("SAM3 masks, scores, and identities must align.")
    return [
        Candidate(mask.astype(np.bool_, copy=False), float(score), prompt, identity)
        for mask, score, identity in zip(masks, scores, identities, strict=True)
        if keep_empty or bool(mask.any())
    ]


def mask_iou(left: NDArray[np.bool_], right: NDArray[np.bool_]) -> float:
    intersection = np.count_nonzero(left & right)
    union = np.count_nonzero(left | right)
    return intersection / union if union else 0.0


def deduplicate_candidates(
    candidates: Iterable[Candidate], *, threshold: float = 0.9
) -> list[Candidate]:
    kept: list[Candidate] = []
    for candidate in sorted(
        candidates,
        key=lambda item: (item.identity.startswith("instance-"), item.score),
        reverse=True,
    ):
        if any(
            mask_iou(candidate.mask, existing.mask) >= threshold for existing in kept
        ):
            continue
        kept.append(candidate)
    return kept


def image_prompt(row: Mapping[str, str], style: str, aliases: Mapping[str, str]) -> str:
    tool_class = " ".join(row.get("Tool Class", "").strip().split())
    tool_name = normalize_tool_name(row.get("Tool Name", ""), aliases)
    if style == "tool-class":
        prompt = tool_class or tool_name
    elif style == "name-and-class":
        prompt = canonical_tool_label(tool_name, tool_class, aliases)
    else:
        raise ValueError(f"Unknown image prompt style {style!r}.")
    if not prompt:
        raise ValueError("An image row must have a Tool Class or Tool Name prompt.")
    return prompt


def show_candidates(
    image: Image.Image,
    candidates: Sequence[Candidate],
    title: str,
    *,
    focus: bool = False,
) -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 8))
    axis.imshow(image)
    colors = plt.get_cmap("tab20")
    boxes: list[list[int]] = []
    for index, candidate in enumerate(candidates):
        overlay = np.zeros((*candidate.mask.shape, 4), dtype=np.float32)
        overlay[candidate.mask] = (*colors(index % 20)[:3], 0.48)
        axis.imshow(overlay, interpolation="nearest")
        box = mask2xywh(candidate.mask)
        if box is not None:
            boxes.append(box)
            x, y, width, height = box
            axis.text(
                x,
                y,
                f"{index}: {candidate.prompt} {candidate.score:.2f}",
                color="white",
                bbox={"facecolor": "black", "alpha": 0.7},
            )
            axis.add_patch(
                plt.Rectangle((x, y), width, height, fill=False, color=colors(index))
            )
    if boxes and focus:
        left = min(box[0] for box in boxes)
        top = min(box[1] for box in boxes)
        right = max(box[0] + box[2] for box in boxes)
        bottom = max(box[1] + box[3] for box in boxes)
        padding = max(1, int(np.ceil(0.2 * max(right - left, bottom - top))))
        axis.set_xlim(
            max(0, left - padding) - 0.5,
            min(image.width, right + padding) - 0.5,
        )
        axis.set_ylim(
            min(image.height, bottom + padding) - 0.5,
            max(0, top - padding) - 0.5,
        )
    axis.set_title(title)
    axis.axis("off")
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.001)


def _parse_indices(text: str, count: int) -> list[int]:
    if not text.strip():
        return list(range(count))
    indices = sorted({int(value) for value in text.split(",")})
    if any(index < 0 or index >= count for index in indices):
        raise ValueError(f"Candidate index must be between 0 and {count - 1}.")
    return indices


def review_candidates(
    image: Image.Image,
    candidates: Sequence[Candidate],
    *,
    key: str,
    auto_accept: bool,
    allow_occluded: bool = False,
    review_log: ReviewLog | None = None,
) -> tuple[str, list[Candidate], list[Candidate], str | None, str]:
    """Return decision, selected/ignored candidates, new prompt, and visibility."""
    import matplotlib.pyplot as plt

    if auto_accept:
        return "accepted", list(candidates), [], None, "clear"
    visibility = "clear"
    visibility_by_index = {index: "clear" for index in range(len(candidates))}
    ignored_indices: set[int] = set()
    while True:
        show_candidates(image, candidates, key, focus=True)
        print(
            "a [0,2]=accept, e=accept empty, r=reject, s=skip, "
            "p TEXT=re-prompt, i [0,2]=mark ignore regions, "
            "v STATE [0,2]=set visibility, "
            + ("o [0,2]=retain occluded tracks, " if allow_occluded else "")
            + "q=quit"
        )
        response = input("review> ").strip()
        plt.close("all")
        command, _, argument = response.partition(" ")
        try:
            if command == "a":
                accepted_indices = set(_parse_indices(argument, len(candidates)))
                return (
                    "accepted",
                    [
                        replace(
                            candidates[index],
                            visibility=visibility_by_index[index],
                        )
                        for index in sorted(accepted_indices - ignored_indices)
                    ],
                    [candidates[index] for index in sorted(ignored_indices)],
                    None,
                    visibility,
                )
            if command == "e":
                return (
                    "accepted",
                    [],
                    [candidates[index] for index in sorted(ignored_indices)],
                    None,
                    visibility,
                )
            if command == "i":
                ignored_indices = set(_parse_indices(argument, len(candidates)))
                if review_log is not None:
                    review_log.append(
                        key,
                        "in_progress",
                        action="ignore",
                        indices=sorted(ignored_indices),
                    )
                continue
            if command == "o" and allow_occluded:
                selected = [
                    replace(candidates[index], visibility="occluded")
                    for index in _parse_indices(argument, len(candidates))
                ]
                return "accepted", selected, [], None, "occluded"
            if command == "r":
                return "rejected", [], [], None, visibility
            if command == "s":
                return "skipped", [], [], None, visibility
            if command == "p" and argument:
                if review_log is not None:
                    review_log.append(
                        key, "in_progress", action="reprompt", prompt=argument
                    )
                return "reprompt", [], [], argument, visibility
            if command == "v":
                state, _, indices_text = argument.partition(" ")
                if state not in {"clear", "partial", "severe"}:
                    raise ValueError("Visibility must be clear, partial, or severe.")
                indices = _parse_indices(indices_text, len(candidates))
                for index in indices:
                    visibility_by_index[index] = state
                visibility = state
                if review_log is not None:
                    review_log.append(
                        key,
                        "in_progress",
                        action="visibility",
                        visibility=state,
                        indices=indices,
                    )
                continue
            if command == "q":
                return "quit", [], [], None, visibility
        except ValueError as error:
            print(error)
            continue
        print("Invalid review command.")


def merge_candidates(candidates: list[Candidate]) -> list[Candidate]:
    """Union one prompt's tool fragments, keeping the strongest candidate's metadata."""
    if len(candidates) < 2:
        return candidates
    strongest = max(candidates, key=lambda candidate: candidate.score)
    return [
        replace(
            strongest,
            mask=np.logical_or.reduce([candidate.mask for candidate in candidates]),
        )
    ]


class Sam3ImageBackend:
    def __init__(
        self,
        model_name: str,
        device: torch.device,
        threshold: float,
        *,
        refine_crop: bool = True,
        single_tool: bool = False,
    ) -> None:
        from transformers import Sam3Model, Sam3Processor

        self.model = Sam3Model.from_pretrained(model_name).to(device).eval()
        self.processor = Sam3Processor.from_pretrained(model_name)
        self.device = device
        self.threshold = threshold
        self.refine_crop = refine_crop
        self.single_tool = single_tool

    def predict(
        self, images: Sequence[Image.Image], prompts: Sequence[str]
    ) -> list[list[Candidate]]:
        candidates = self._predict(images, prompts)
        if self.single_tool:
            candidates = [merge_candidates(result) for result in candidates]
        if not self.refine_crop:
            return candidates

        crops: list[Image.Image] = []
        crop_prompts: list[str] = []
        regions: list[tuple[int, int, int, int, int]] = []
        for index, (image, prompt, initial) in enumerate(
            zip(images, prompts, candidates, strict=True)
        ):
            if len(initial) != 1:
                continue
            box = mask2xywh(initial[0].mask)
            if box is None:
                continue
            x, y, width, height = box
            padding = max(1, int(np.ceil(0.2 * max(width, height))))
            left, top = max(0, x - padding), max(0, y - padding)
            right = min(image.width, x + width + padding)
            bottom = min(image.height, y + height + padding)
            if (left, top, right, bottom) == (0, 0, image.width, image.height):
                continue
            crops.append(image.crop((left, top, right, bottom)))
            crop_prompts.append(prompt)
            regions.append((index, left, top, right, bottom))

        if not crops:
            return candidates
        refined = self._predict(crops, crop_prompts)
        if self.single_tool:
            refined = [merge_candidates(result) for result in refined]
        for (index, left, top, right, bottom), result in zip(
            regions, refined, strict=True
        ):
            result = deduplicate_candidates(result)
            if len(result) != 1:
                continue
            candidate = result[0]
            mask = candidate.mask
            image = images[index]
            if (
                (left > 0 and mask[:, 0].any())
                or (top > 0 and mask[0, :].any())
                or (right < image.width and mask[:, -1].any())
                or (bottom < image.height and mask[-1, :].any())
            ):
                # A mask touching an interior crop edge may cut off the tool.
                continue
            full_mask = np.zeros((image.height, image.width), dtype=np.bool_)
            full_mask[top:bottom, left:right] = mask
            original = candidates[index][0]
            candidates[index] = [
                replace(candidate, mask=full_mask, identity=original.identity)
            ]
        return candidates

    @torch.inference_mode()
    def _predict(
        self, images: Sequence[Image.Image], prompts: Sequence[str]
    ) -> list[list[Candidate]]:
        inputs = self.processor(
            images=list(images), text=list(prompts), return_tensors="pt"
        ).to(self.device)
        outputs = self.model(**inputs)
        results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=self.threshold,
            mask_threshold=0.5,
            target_sizes=inputs["original_sizes"].tolist(),
        )
        return [
            extract_candidates(result, prompt=prompt)
            for result, prompt in zip(results, prompts, strict=True)
        ]


def _visibility_flags(visibility: str) -> dict[str, str]:
    severe = visibility == "severe"
    visible = visibility not in {"occluded", "absent"}
    return {
        "Visibility": visibility,
        "Reviewed": "TRUE",
        "Annotation Complete": "TRUE",
        "Mask Valid": str(visible).upper(),
        "Class Valid": str(visible and not severe).upper(),
        "Base Valid": str(visible and not severe).upper(),
        "Visible": str(visible).upper(),
        "Annotation Confidence": "1.0",
    }


def _replace_scope(
    rows: list[dict[str, object]],
    replacements: list[dict[str, object]],
    *,
    scope: Mapping[str, object],
) -> list[dict[str, object]]:
    retained = [
        row
        for row in rows
        if not all(str(row.get(key, "")) == str(value) for key, value in scope.items())
    ]
    return retained + replacements


def resolved_image_rows(
    instances: Sequence[Mapping[str, str]],
    selected: Sequence[Candidate],
    mask_path: Path,
) -> list[dict[str, object]]:
    """Produce rows and bundle indices from explicitly resolved tool identities."""
    validate_instance_metadata(instances)
    by_id = {candidate.identity: candidate for candidate in selected}
    if len(by_id) != len(selected) or set(by_id) != {
        row["Instance ID"] for row in instances
    }:
        raise ValueError(
            "Every declared instance must have exactly one resolved annotation."
        )
    rows = []
    indices = {
        candidate.identity: index
        for index, candidate in enumerate(
            candidate for candidate in selected if candidate.mask.any()
        )
    }
    for source in instances:
        candidate = by_id[source["Instance ID"]]
        box = mask2xywh(candidate.mask)
        if candidate.visibility not in {
            "clear",
            "partial",
            "severe",
            "occluded",
            "absent",
        }:
            raise ValueError("Invalid resolved visibility.")
        visible = candidate.visibility in {"clear", "partial", "severe"}
        if visible != (box is not None):
            raise ValueError(
                "Visible tools require masks; absent or occluded tools require empty masks."
            )
        row: dict[str, object] = {
            field: source.get(field, "") for field in IMAGE_OBJECT_FIELDS
        }
        row.update(
            {
                "BBox X": box[0] if box else "",
                "BBox Y": box[1] if box else "",
                "BBox Width": box[2] if box else "",
                "BBox Height": box[3] if box else "",
                "Mask Path": str(mask_path) if visible else "",
                "Mask Index": indices[candidate.identity] if visible else -1,
                **_visibility_flags(candidate.visibility),
            }
        )
        if not visible:
            # The existing training builder consumes Reviewed mask rows only.
            # Retain resolution and identity without presenting a positive annotation.
            row["Reviewed"] = "FALSE"
        rows.append(row)
    return rows


def annotate_images(args: argparse.Namespace) -> None:
    manifest = extend_image_manifest(args.manifest)
    rows = read_csv(manifest)
    mask_paths = [image_mask_path(args.mask_dir, row["Filepath"]) for row in rows]
    if len(set(mask_paths)) != len(mask_paths):
        raise ValueError("Image filenames map to duplicate mask paths.")
    object_rows: list[dict[str, object]] = (
        list(read_csv(args.object_manifest)) if args.object_manifest.is_file() else []
    )
    sources: dict[str, list[dict[str, str]]] = {}
    for obj in object_rows:
        sources.setdefault(str(obj[IMAGE_ID_FIELD]), []).append(
            {field: str(value) for field, value in obj.items()}
        )
    review = ReviewLog(args.review_log)
    for row in rows:
        if review.completed(row[IMAGE_ID_FIELD]):
            continue
        instances = sources.get(row[IMAGE_ID_FIELD], [])
        if not instances:
            raise ValueError(
                f"{row['Filepath']} has no declared instances. Run prepare_manifests.py or annotate.py declare."
            )
        validate_instance_metadata(instances)
        if len(instances) > 1 and args.single_tool:
            raise ValueError(
                f"{row['Filepath']}: --single-tool cannot merge multiple declared tools."
            )
        if len(instances) > 1 and args.auto_accept:
            raise ValueError(
                "Multi-tool images require explicit review; --auto-accept cannot assign identities."
            )
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    backend = Sam3ImageBackend(
        args.model,
        device,
        args.threshold,
        refine_crop=args.refine_crop,
        single_tool=args.single_tool,
    )
    root = manifest.parent
    aliases = load_taxonomy_aliases(args.aliases_json)
    missing = [
        row["Filepath"] for row in rows if not (root / row["Filepath"]).is_file()
    ]
    if missing:
        print(
            f"Skipping {len(missing)} manifest images that are not present below {root}."
        )
    queue = [
        row
        for row in rows
        if not review.completed(row[IMAGE_ID_FIELD])
        and (root / row["Filepath"]).is_file()
    ]
    queue.sort(
        key=lambda row: (
            review.decisions.get(row[IMAGE_ID_FIELD], {}).get("decision") == "skipped"
        )
        != args.skipped_first
    )
    for offset in range(0, len(queue), args.batch_size):
        pending = queue[offset : offset + args.batch_size]
        images = [Image.open(root / row["Filepath"]).convert("RGB") for row in pending]
        prompt_lists = []
        for row in pending:
            prompt_lists.append(
                list(
                    dict.fromkeys(
                        [
                            image_prompt(instance, args.image_prompt, aliases)
                            for instance in sources[row[IMAGE_ID_FIELD]]
                        ]
                        + [
                            normalize_tool_name(prompt, aliases)
                            for prompt in [
                                *(args.discovery_prompt or []),
                                *(args.canonical_prompt or []),
                            ]
                        ]
                    )
                )
            )
        # A mixed batch also keeps single-tool candidates unchanged for review.
        multi_batch = any(len(sources[row[IMAGE_ID_FIELD]]) > 1 for row in pending)
        original_refine = getattr(backend, "refine_crop", args.refine_crop)
        if multi_batch:
            backend.refine_crop = False
        predicted = backend.predict(images, [prompts[0] for prompts in prompt_lists])
        for index, prompts in enumerate(prompt_lists):
            for prompt in prompts[1:]:
                predicted[index].extend(backend.predict([images[index]], [prompt])[0])
        for row, image, initial, prompts in zip(
            pending, images, predicted, prompt_lists, strict=True
        ):
            key = row[IMAGE_ID_FIELD]
            instances = sources[key]
            active_prompt = prompts[0]
            candidates = [
                replace(candidate, identity=f"candidate-{index}")
                for index, candidate in enumerate(initial)
            ]
            explicit = (
                len(instances) > 1
                or len(candidates) > 1
                or len(prompts) > 1
                or (not args.single_tool and not args.auto_accept)
            )
            while not explicit:
                decision, selected, ignored, replacement_prompt, _ = review_candidates(
                    image,
                    candidates,
                    key=key,
                    auto_accept=args.auto_accept,
                    review_log=review,
                )
                if decision != "reprompt":
                    break
                assert replacement_prompt is not None
                active_prompt = normalize_tool_name(replacement_prompt, aliases)
                candidates = backend.predict([image], [active_prompt])[0]
                explicit = len(candidates) > 1
            if explicit:
                if args.auto_accept:
                    raise ValueError(
                        "Ambiguous candidates require explicit assignment; disable --auto-accept."
                    )
                for left in range(len(candidates)):
                    for right in range(left + 1, len(candidates)):
                        if (
                            mask_iou(candidates[left].mask, candidates[right].mask)
                            >= 0.9
                        ):
                            print(
                                f"Possible duplicate candidates: {left}, {right}. Both retained for review."
                            )
                # Reprompting appends candidates without replacing assigned fragments.
                backend.refine_crop = False
                result = review_image_instances(
                    image,
                    candidates,
                    instances,
                    key=key,
                    show=show_candidates,
                    predict=lambda prompt: backend.predict(
                        [image], [normalize_tool_name(prompt, aliases)]
                    )[0],
                    log=review.append,
                )
                decision, selected, ignored = (
                    result.decision,
                    result.selected,
                    result.ignored,
                )
                instances = result.instances
            elif decision == "accepted" and selected:
                selected = [replace(selected[0], identity=instances[0]["Instance ID"])]
            backend.refine_crop = False if multi_batch else original_refine
            if decision == "quit":
                return
            if decision == "accepted":
                mask_path = image_mask_path(args.mask_dir, row["Filepath"])
                normalized = [
                    {
                        **source,
                        "Tool Name": normalize_tool_name(source["Tool Name"], aliases),
                    }
                    for source in instances
                ]
                if explicit or selected:
                    replacements = resolved_image_rows(normalized, selected, mask_path)
                else:
                    # Preserve legacy accepted-empty semantics for single-tool captures.
                    replacements = [
                        {
                            field: (
                                source.get(field, "")
                                if field
                                in {
                                    IMAGE_ID_FIELD,
                                    "Instance ID",
                                    "Tool ID",
                                    "Tool Class",
                                    "Tool Name",
                                    "Confidence",
                                }
                                else ""
                            )
                            for field in IMAGE_OBJECT_FIELDS
                        }
                        for source in normalized
                    ]
                visible = [candidate for candidate in selected if candidate.mask.any()]
                masks = (
                    np.stack([candidate.mask for candidate in visible])
                    if visible
                    else np.zeros((0, image.height, image.width), dtype=np.bool_)
                )
                save_mask_bundle(
                    mask_path,
                    masks,
                    [candidate.identity for candidate in visible],
                    ignore=(
                        np.logical_or.reduce([candidate.mask for candidate in ignored])
                        if ignored
                        else None
                    ),
                )
                object_rows = _replace_scope(
                    object_rows, replacements, scope={IMAGE_ID_FIELD: key}
                )
                write_image_manifests(manifest, rows, args.object_manifest, object_rows)
            review.append(
                key,
                decision,
                prompt=active_prompt,
                instances=[
                    {
                        "instance_id": candidate.identity,
                        "visibility": candidate.visibility,
                    }
                    for candidate in selected
                ],
            )
        backend.refine_crop = original_refine


def declare_instances(args: argparse.Namespace) -> None:
    """Explicit intake for an image or an existing rotation sample."""
    manifest = extend_image_manifest(args.manifest)
    images = read_csv(manifest)
    objects = read_csv(args.object_manifest) if args.object_manifest.is_file() else []
    review = ReviewLog(args.review_log)
    if args.sample_id is not None:
        targets = [row for row in images if row.get("Sample ID") == args.sample_id]
        if not targets:
            raise ValueError(f"No images have Sample ID {args.sample_id}.")
    else:
        targets = []
        by_path = {row["Filepath"]: row for row in images}
        for filepath in dict.fromkeys(args.filepath):
            image_mask_path(Path("masks/images"), filepath)
            if (
                not filepath.startswith("images/")
                or not (manifest.parent / filepath).is_file()
            ):
                raise ValueError(
                    f"Expected an existing image below images/: {filepath}"
                )
            if filepath not in by_path:
                row = {
                    "Filepath": filepath,
                    IMAGE_ID_FIELD: stable_media_id("image", filepath),
                }
                images.append(row)
                by_path[filepath] = row
            targets.append(by_path[filepath])
    additions = []
    for image in targets:
        key = image[IMAGE_ID_FIELD]
        existing = [row for row in objects if row[IMAGE_ID_FIELD] == key]
        if review.completed(key) or any(
            row.get("Reviewed", "").upper() == "TRUE" for row in existing
        ):
            if not args.reopen:
                raise ValueError(
                    "Image already reviewed. Use --reopen to clear annotations and review again."
                )
        instance = {
            IMAGE_ID_FIELD: key,
            "Instance ID": args.instance_id
            or stable_media_id("instance", f"{key}/{args.tool_id}"),
            "Tool ID": args.tool_id,
            "Tool Class": " ".join(args.tool_class.strip().split()),
            "Tool Name": " ".join(args.tool_name.strip().split()),
            "Confidence": str(args.confidence),
        }
        validate_instance_metadata([*existing, instance])
        additions.append(instance)
    target_ids = {row[IMAGE_ID_FIELD] for row in targets}
    if args.reopen:
        identity_fields = {
            IMAGE_ID_FIELD,
            "Instance ID",
            "Tool ID",
            "Tool Class",
            "Tool Name",
            "Confidence",
        }
        objects = [
            (
                {
                    field: row.get(field, "") if field in identity_fields else ""
                    for field in IMAGE_OBJECT_FIELDS
                }
                if row[IMAGE_ID_FIELD] in target_ids
                else row
            )
            for row in objects
        ]
    write_image_manifests(
        manifest, images, args.object_manifest, [*objects, *additions]
    )
    extend_image_manifest(manifest)
    for instance in additions:
        review.append(
            instance[IMAGE_ID_FIELD], "pending", action="declare", instance=instance
        )
    print(f"Declared tool {args.tool_id} in {len(targets)} image(s).")


def _shared(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default="facebook/sam3")
    parser.add_argument("--device")
    parser.add_argument("--auto-accept", action="store_true")
    parser.add_argument("--aliases-json", type=Path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and review SAM3 annotations."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    declare = commands.add_parser(
        "declare", help="Declare a physical tool in images or a rotation sample."
    )
    declare.add_argument("--manifest", type=Path, default=Path("image_manifest.csv"))
    declare.add_argument(
        "--object-manifest", type=Path, default=Path("image_object_manifest.csv")
    )
    declare.add_argument("--review-log", type=Path, default=Path("image_review.jsonl"))
    target = declare.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "--filepath",
        action="append",
        help="Image path; may be repeated. Adds new images to the manifest.",
    )
    target.add_argument(
        "--sample-id", help="Declare the tool in every image of this existing sample."
    )
    declare.add_argument(
        "--instance-id", help="Optional per-image instance identifier."
    )
    declare.add_argument("--tool-id", required=True)
    declare.add_argument("--tool-class", required=True)
    declare.add_argument("--tool-name", required=True)
    declare.add_argument("--confidence", required=True, type=float)
    declare.add_argument(
        "--reopen",
        action="store_true",
        help="Clear previous annotations for targeted images and reopen review.",
    )
    images = commands.add_parser("images")
    _shared(images)
    images.add_argument("--manifest", type=Path, default=Path("image_manifest.csv"))
    images.add_argument(
        "--object-manifest",
        type=Path,
        default=Path("image_object_manifest.csv"),
    )
    images.add_argument("--mask-dir", type=Path, default=Path("masks/images"))
    images.add_argument("--review-log", type=Path, default=Path("image_review.jsonl"))
    images.add_argument(
        "--skipped-first",
        action="store_true",
        help="Review previously skipped images first (default: skipped images last).",
    )
    images.add_argument("--batch-size", type=int, default=4)
    images.add_argument("--threshold", type=float, default=0.5)
    images.add_argument(
        "--single-tool",
        action="store_true",
        help=(
            "For images containing one tool, merge masks from each prompt into "
            "one object in both inference passes (useful for open scissors)."
        ),
    )
    images.add_argument(
        "--refine-crop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Refine single-tool masks with a second SAM3 pass on a padded crop "
            "(default: enabled). Keep the original prediction if refinement is "
            "ambiguous or touches an interior crop edge."
        ),
    )
    images.add_argument(
        "--image-prompt",
        choices=("tool-class", "name-and-class"),
        default="tool-class",
        help=(
            "Prompt SAM3 with Tool Class (default), or combine Tool Name and "
            "Tool Class."
        ),
    )
    images.add_argument(
        "--discovery-prompt",
        action="append",
        help="Broad prompt to discover additional instruments in multi-tool images.",
    )
    images.add_argument(
        "--canonical-prompt",
        action="append",
        help="Canonical tool-name prompt to add to multi-tool discovery.",
    )

    from video_annotation import add_video_commands

    add_video_commands(commands)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "declare":
        declare_instances(args)
    elif args.command == "images":
        if args.batch_size <= 0:
            raise ValueError("batch-size must be positive.")
        annotate_images(args)
    else:
        from video_annotation import annotate_videos, prepare_videos, verify_videos

        if args.command == "prepare-videos":
            prepare_videos(args)
        elif args.command == "verify-videos":
            verify_videos(args)
        elif args.command == "export-video-training":
            from video_export import export_video_training

            export_video_training(args)
        else:
            annotate_videos(args)


if __name__ == "__main__":
    main()
