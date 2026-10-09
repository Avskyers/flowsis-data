"""Frame review, bounded correction propagation, and the video CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from annotations import (
    canonical_tool_label,
    load_taxonomy_aliases,
    normalize_tool_name,
    read_csv,
    write_csv,
)
from image_review import Candidate
from masks import load_mask_bundle
from video_backend import Sam3VideoBackend
from video_data import (
    FRAME_FIELDS,
    Mask,
    VideoLog,
    file_hash,
    prepare_videos,
    read_frame,
    read_mask,
    resolve_path,
    save_frame_mask,
    validate_video_frames,
)
from video_ui import collect_clicks, preview_frames, select_region, show_frame


def merge_region(
    original: Mask, replacement: Mask, region: tuple[int, int, int, int] | None
) -> Mask:
    if replacement.shape != original.shape:
        raise ValueError("Replacement mask dimensions differ from the frame.")
    if region is None:
        return replacement.copy()
    left, top, right, bottom = region
    if not (
        0 <= left < right <= original.shape[1]
        and 0 <= top < bottom <= original.shape[0]
    ):
        raise ValueError("Region must be inside the frame.")
    result = original.copy()
    result[top:bottom, left:right] = replacement[top:bottom, left:right]
    return result


class VideoReviewer:
    def __init__(
        self,
        args: argparse.Namespace,
        video: dict[str, str],
        all_frames: list[dict[str, str]],
        backend: Sam3VideoBackend,
    ):
        if args.chunk_size < 2 or args.context_frames < 0 or args.start_frame < 0:
            raise ValueError(
                "chunk-size must be at least 2; context/start frame must be nonnegative."
            )
        self.args = args
        self.video = video
        self.root = args.manifest.parent.resolve()
        self.all_frames = all_frames
        self.frames = sorted(
            (row for row in all_frames if row["Video ID"] == video["Video ID"]),
            key=lambda row: int(row["Frame Index"]),
        )
        validate_video_frames(self.root, video, self.frames, check_files=False)
        self.backend = backend
        self.log = VideoLog(args.review_log)
        self.aliases = load_taxonomy_aliases(args.aliases_json)
        self.inspected: dict[int, str] = {}
        self.click_history: dict[
            tuple[int, tuple[int, int, int, int] | None], list[tuple[float, float, int]]
        ] = {}

    def flush(self) -> None:
        write_csv(self.args.frame_manifest, self.all_frames, FRAME_FIELDS)

    def editable(self, indices: list[int]) -> None:
        protected = [
            index
            for index in indices
            if self.frames[index].get("Review Status") in {"accepted", "rejected"}
        ]
        if protected:
            raise ValueError(
                f"Frames {protected[:10]} are reviewed. Explicitly reopen the desired range first."
            )

    def reopen(self, start: int, stop: int) -> None:
        self.check_range(start, stop)
        for row in self.frames[start : stop + 1]:
            row.update(
                {
                    "Review Status": "pending",
                    "Reviewed": "FALSE",
                    "Annotation Complete": "FALSE",
                    "Mask Valid": "FALSE",
                    "Class Valid": "FALSE",
                    "Base Valid": "FALSE",
                    "Annotation Confidence": "",
                }
            )
            self.log.append(row["Frame ID"], "pending", action="reopen")
        self.flush()

    def check_range(self, start: int, stop: int) -> None:
        if not 0 <= start <= stop < len(self.frames):
            raise ValueError(
                f"Use an inclusive frame range within 0..{len(self.frames)-1}."
            )

    def mask(self, index: int) -> Mask:
        return read_mask(
            self.root,
            self.frames[index],
            (int(self.video["Height"]), int(self.video["Width"])),
        )

    def save(
        self,
        index: int,
        mask: Mask,
        decision: str,
        *,
        visibility: str = "",
        anchor: bool = False,
        action: str = "review",
        **details: object,
    ) -> None:
        self.editable([index])
        row = self.frames[index]
        save_frame_mask(
            self.root,
            row,
            self.video,
            self.args.mask_dir,
            mask,
            decision=decision,
            visibility=visibility,
            anchor=anchor,
        )
        self.flush()
        self.log.append(
            row["Frame ID"],
            decision,
            action=action,
            extraction_id=self.video["Extraction ID"],
            visibility=visibility,
            mask_path=row["Mask Path"],
            **details,
        )

    def preview(self, start: int, stop: int) -> None:
        self.check_range(start, stop)
        for first in range(start, stop + 1, 12):
            indices = list(range(first, min(stop + 1, first + 12)))
            items = [
                (
                    read_frame(self.root, self.frames[index]),
                    self.mask(index),
                    f"Frame {index}",
                )
                for index in indices
            ]
            preview_frames(items)
            for index in indices:
                path = self.frames[index].get("Mask Path")
                if path:
                    self.inspected[index] = file_hash(resolve_path(self.root, path))

    def accept_range(self, start: int, stop: int) -> None:
        self.check_range(start, stop)
        indices = list(range(start, stop + 1))
        self.editable(indices)
        for index in indices:
            row = self.frames[index]
            path = row.get("Mask Path")
            if not path or self.inspected.get(index) != file_hash(
                resolve_path(self.root, path)
            ):
                raise ValueError(
                    f"Preview the current mask for frame {index} before range acceptance."
                )
            if not self.mask(index).any():
                raise ValueError(
                    f"Frame {index} has no visible mask; resolve absent/occluded explicitly."
                )
        for index in indices:
            self.save(
                index,
                self.mask(index),
                "accepted",
                visibility=self.frames[index].get("Visibility") or "clear",
            )

    def propagate(
        self, anchor_index: int, start: int, stop: int, direction: str
    ) -> None:
        self.check_range(start, stop)
        if not start <= anchor_index <= stop or direction not in {
            "forward",
            "backward",
            "both",
        }:
            raise ValueError(
                "Select forward/backward/both and a range containing the current anchor frame."
            )
        target_start = anchor_index if direction == "forward" else start
        target_stop = anchor_index if direction == "backward" else stop
        self.editable(list(range(target_start, target_stop + 1)))
        seed = self.mask(anchor_index)
        if not seed.any():
            raise ValueError(
                "Correct and select a nonempty anchor mask before propagation."
            )
        # Persist the explicitly selected anchor independently of pending propagation output.
        self.save(anchor_index, seed, "pending", anchor=True, action="anchor")
        written: set[int] = set()
        for reverse in (
            [False, True] if direction == "both" else [direction == "backward"]
        ):
            current, carry = anchor_index, seed
            terminal = target_start if reverse else target_stop
            while True:
                first = (
                    max(terminal, current - self.args.chunk_size + 1)
                    if reverse
                    else current
                )
                last = (
                    current
                    if reverse
                    else min(terminal, current + self.args.chunk_size - 1)
                )
                window_start = max(0, first - self.args.context_frames)
                window_stop = min(len(self.frames) - 1, last + self.args.context_frames)
                images = [
                    read_frame(self.root, row)
                    for row in self.frames[window_start : window_stop + 1]
                ]
                anchors = {}
                for index in range(window_start, window_stop + 1):
                    path = self.frames[index].get("Anchor Path")
                    if path:
                        bundle = load_mask_bundle(resolve_path(self.root, path))
                        if bundle.masks.shape != (1, images[0].height, images[0].width):
                            raise ValueError(
                                "Correction anchor has invalid mask dimensions."
                            )
                        anchors[index - window_start] = bundle.masks[0]
                anchors[current - window_start] = carry
                outputs = dict(
                    self.backend.track(
                        images,
                        anchors,
                        current - window_start,
                        (first if reverse else last) - window_start,
                        reverse,
                    )
                )
                # The backend's reverse iterator is empty at local frame zero.
                outputs[current - window_start] = carry
                for index in range(first, last + 1):
                    if index - window_start not in outputs:
                        raise ValueError(
                            f"Tracker did not return frame {index}; correct it before continuing."
                        )
                    if index not in written:
                        self.save(
                            index,
                            outputs[index - window_start],
                            "pending",
                            action="propagate",
                            anchor_frame=anchor_index,
                            frame_range=[target_start, target_stop],
                            direction=direction,
                        )
                        written.add(index)
                next_index = first if reverse else last
                if next_index == terminal:
                    break
                carry = outputs[next_index - window_start]
                if not carry.any():
                    raise ValueError(
                        f"Tracking lost the tool at frame {next_index}. Add a correction there."
                    )
                current = next_index
        print(f"Generated {len(written)} pending masks. Review them before acceptance.")

    def run(self) -> bool:
        start = self.args.start_frame
        stop = (
            self.args.end_frame
            if self.args.end_frame is not None
            else len(self.frames) - 1
        )
        self.check_range(start, stop)
        current = next(
            (
                index
                for index in range(start, stop + 1)
                if self.frames[index].get("Review Status")
                not in {"accepted", "rejected"}
            ),
            None,
        )
        if current is None:
            print(f"{self.video['Filepath']}: selected frames are already reviewed.")
            return True
        prompt = self.args.prompt or canonical_tool_label(
            self.video["Tool Name"], self.video["Tool Class"], self.aliases
        )
        region: tuple[int, int, int, int] | None = None
        candidates: list[Candidate] = []
        visited: set[int] = set()
        while current is not None:
            row = self.frames[current]
            image = read_frame(self.root, row)
            mask = self.mask(current)
            if self.args.auto_accept:
                if not mask.any():
                    raise ValueError(
                        "--auto-accept requires existing nonempty predictions; missing masks are unresolved."
                    )
                self.save(
                    current, mask, "accepted", visibility="clear", action="auto_accept"
                )
                command, argument = "next", ""
            else:
                show_frame(
                    image,
                    mask,
                    f"{self.video['Tool Name']} ({self.video['Tool ID']}) | "
                    f"frame {current} @ {row['Timestamp Seconds']}s | {row.get('Review Status')} | region {region}",
                    candidates,
                )
                if row.get("Mask Path"):
                    self.inspected[current] = file_hash(
                        resolve_path(self.root, row["Mask Path"])
                    )
                print(
                    "p [TEXT]=text candidates; g 0,2=select/group; click=positive/negative mouse clicks; "
                    "box=box prompt; region [L T R B]=repair region; whole=whole-tool mode; "
                    "prop forward|backward|both START END; a=accept; v clear|partial|severe; "
                    "absent / occluded=accept nonvisible; preview START END; accept START END; reopen START END; jump FRAME; "
                    "next / s=skip; r=reject; q=quit"
                )
                command, _, argument = input("video> ").strip().partition(" ")
            try:
                if command == "q":
                    return False
                if command == "jump":
                    index = int(argument)
                    self.check_range(index, index)
                    current, region, candidates = index, None, []
                    continue
                if command in {"preview", "accept"}:
                    first, last = map(int, argument.split())
                    (self.preview if command == "preview" else self.accept_range)(
                        first, last
                    )
                    continue
                if command == "reopen":
                    first, last = map(int, argument.split())
                    self.reopen(first, last)
                    continue
                if command == "whole":
                    region = None
                    continue
                if command == "region":
                    if argument:
                        values = [int(value) for value in argument.split()]
                        if len(values) != 4:
                            raise ValueError("Region requires L T R B coordinates.")
                        selected_region = (values[0], values[1], values[2], values[3])
                        merge_region(mask, mask, selected_region)
                        region = selected_region
                    else:
                        region = select_region(image)
                    continue
                if command in {
                    "p",
                    "g",
                    "click",
                    "box",
                    "prop",
                    "a",
                    "v",
                    "absent",
                    "occluded",
                    "r",
                }:
                    self.editable([current])
                if command == "p":
                    prompt = normalize_tool_name(argument or prompt, self.aliases)
                    view = image.crop(region) if region else image
                    parts = self.backend.text(view, prompt)
                    candidates = []
                    for part in parts:
                        full = np.zeros((image.height, image.width), dtype=np.bool_)
                        if region:
                            left, top, right, bottom = region
                            full[top:bottom, left:right] = part.mask
                        else:
                            full = part.mask
                        candidates.append(
                            Candidate(full, part.score, prompt, part.identity)
                        )
                    self.log.append(
                        row["Frame ID"],
                        "pending",
                        action="text_prompt",
                        prompt=prompt,
                        region=region,
                    )
                    continue
                if command == "g":
                    selected = sorted({int(value) for value in argument.split(",")})
                    if not selected or any(
                        index < 0 or index >= len(candidates) for index in selected
                    ):
                        raise ValueError("Select valid candidate indices.")
                    replacement = np.logical_or.reduce(
                        [candidates[index].mask for index in selected]
                    )
                    self.save(
                        current,
                        merge_region(mask, replacement, region),
                        "pending",
                        anchor=True,
                        action="group_fragments",
                        candidate_indices=selected,
                        prompt=prompt,
                        region=region,
                    )
                    candidates = []
                    continue
                if command in {"click", "box"}:
                    view = image.crop(region) if region else image
                    view_mask = (
                        mask[region[1] : region[3], region[0] : region[2]]
                        if region
                        else mask
                    )
                    points = (
                        collect_clicks(
                            view,
                            view_mask,
                            self.click_history.get((current, region), []),
                        )
                        if command == "click"
                        else []
                    )
                    box = select_region(view) if command == "box" else None
                    if points is None or (not points and box is None):
                        continue
                    replacement = self.backend.clicks(view, points, box)
                    full = np.zeros_like(mask)
                    if region:
                        full[region[1] : region[3], region[0] : region[2]] = replacement
                    else:
                        full = replacement
                    self.save(
                        current,
                        merge_region(mask, full, region),
                        "pending",
                        anchor=True,
                        action="click_correction",
                        points=points,
                        box=box,
                        region=region,
                    )
                    if command == "click":
                        self.click_history[(current, region)] = points
                    candidates = []
                    continue
                if command == "prop":
                    direction, first_text, last_text = argument.split()
                    self.propagate(current, int(first_text), int(last_text), direction)
                    continue
                if command == "v":
                    if argument not in {"clear", "partial", "severe"}:
                        raise ValueError(
                            "Visible state must be clear, partial, or severe."
                        )
                    row["Visibility"] = argument
                    self.flush()
                    continue
                if command in {"a", "absent", "occluded"}:
                    state = row.get("Visibility") or "clear"
                    if command != "a":
                        state = command
                        mask = np.zeros_like(mask)
                    self.save(
                        current,
                        mask,
                        "accepted",
                        visibility=state,
                        anchor=bool(mask.any()),
                    )
                elif command == "r":
                    row.update(
                        {
                            "Review Status": "rejected",
                            "Reviewed": "FALSE",
                            "Annotation Complete": "FALSE",
                        }
                    )
                    self.flush()
                    self.log.append(row["Frame ID"], "rejected", action="reject")
                elif command not in {"next", "s"}:
                    raise ValueError("Unknown command.")
                elif not self.args.auto_accept:
                    self.log.append(
                        row["Frame ID"],
                        row.get("Review Status") or "pending",
                        action="skip",
                    )
                visited.add(current)
                current = next(
                    (
                        index
                        for index in range(start, stop + 1)
                        if index not in visited
                        and self.frames[index].get("Review Status")
                        not in {"accepted", "rejected"}
                    ),
                    None,
                )
                region, candidates = None, []
            except (ValueError, OSError) as error:
                print(error)
        return True


def annotate_videos(args: argparse.Namespace) -> None:
    videos = read_csv(args.manifest)
    all_frames = read_csv(args.frame_manifest)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    backend = Sam3VideoBackend(args.model, device, args.threshold)
    selected = [
        video
        for video in videos
        if not args.video_id or video["Video ID"] == args.video_id
    ]
    if not selected:
        raise ValueError("No matching prepared video. Run prepare-videos first.")
    for video in selected:
        root = args.manifest.parent.resolve()
        if file_hash(resolve_path(root, video["Filepath"])) != video["Source SHA256"]:
            raise ValueError("Source MP4 changed; reextract before annotating.")
        if not VideoReviewer(args, video, all_frames, backend).run():
            return


def verify_videos(args: argparse.Namespace) -> None:
    videos = read_csv(args.manifest)
    frames = read_csv(args.frame_manifest)
    known = {video["Video ID"] for video in videos}
    if len(known) != len(videos):
        raise ValueError("Video IDs must be unique.")
    if args.video_id and args.video_id not in known:
        raise ValueError("Unknown Video ID.")
    if any(row["Video ID"] not in known for row in frames):
        raise ValueError("Frame manifest refers to an unknown video.")
    log = VideoLog(args.review_log)
    for video in videos:
        if args.video_id and args.video_id != video["Video ID"]:
            continue
        selected = [row for row in frames if row["Video ID"] == video["Video ID"]]
        validate_video_frames(args.manifest.parent.resolve(), video, selected)
        for row in selected:
            if (
                row.get("Review Status") == "accepted"
                and log.decisions.get(row["Frame ID"]) != "accepted"
            ):
                raise ValueError(
                    f"Review log disagrees with accepted frame {row['Frame ID']}."
                )
        print(
            f"Verified {video['Filepath']}: {len(selected)} frame files and their annotations."
        )


def add_video_commands(
    commands: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    prepare = commands.add_parser(
        "prepare-videos", help="Extract single-tool MP4s into individual PNG frames."
    )
    review = commands.add_parser(
        "videos",
        help="Review frame masks with mouse corrections and bounded propagation.",
    )
    verify = commands.add_parser(
        "verify-videos", help="Validate extracted frames and video annotations."
    )
    export = commands.add_parser(
        "export-video-training",
        help="Export reviewed frames for the existing flowsis image training reader.",
    )
    for parser in (prepare, review, verify, export):
        parser.add_argument("--manifest", type=Path, default=Path("video_manifest.csv"))
        parser.add_argument(
            "--frame-manifest", type=Path, default=Path("video_frame_manifest.csv")
        )
        parser.add_argument(
            "--review-log", type=Path, default=Path("video_review.jsonl")
        )
        parser.add_argument("--video-id")
    prepare.add_argument("--video-dir", type=Path, default=Path("videos"))
    prepare.add_argument("--frame-dir", type=Path, default=Path("frames/videos"))
    prepare.add_argument(
        "--image-object-manifest", type=Path, default=Path("image_object_manifest.csv")
    )
    prepare.add_argument(
        "--tool-id",
        help="Explicit known physical tool ID when videos are not in ID subfolders.",
    )
    prepare.add_argument("--capture-session-id", default="")
    prepare.add_argument("--condition", default="")
    prepare.add_argument(
        "--reextract",
        action="store_true",
        help="Invalidate existing masks/review and replace extracted frames.",
    )
    review.add_argument("--model", default="facebook/sam3")
    review.add_argument("--device")
    review.add_argument("--threshold", type=float, default=0.5)
    review.add_argument("--mask-dir", type=Path, default=Path("masks/videos"))
    review.add_argument("--aliases-json", type=Path)
    review.add_argument(
        "--prompt",
        help="Optional text-prompt override; tool identity remains unchanged.",
    )
    review.add_argument("--start-frame", type=int, default=0)
    review.add_argument("--end-frame", type=int, help="Inclusive final frame index.")
    review.add_argument("--chunk-size", type=int, default=64)
    review.add_argument("--context-frames", type=int, default=8)
    review.add_argument(
        "--auto-accept",
        action="store_true",
        help="Explicitly accept existing nonempty pending masks without human review.",
    )
    export.add_argument(
        "--replace",
        action="store_true",
        help="Replace a previous training export after annotation changes.",
    )
    export.add_argument("--output-dir", type=Path, default=Path("video_training"))
