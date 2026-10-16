"""Explicit image-instance assignment in the Matplotlib/terminal reviewer."""

from __future__ import annotations

import shlex
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from annotations import (
    canonical_tool_label,
    stable_media_id,
    validate_instance_metadata,
)
from image_ui import ToolPreview


@dataclass(frozen=True)
class Candidate:
    mask: NDArray[np.bool_]
    score: float
    prompt: str
    identity: str
    visibility: str = "clear"


@dataclass
class ImageReview:
    decision: str
    instances: list[dict[str, str]]
    selected: list[Candidate]
    ignored: list[Candidate]


def review_image_instances(
    image: Image.Image,
    candidates: Sequence[Candidate],
    instances: Sequence[dict[str, str]],
    *,
    key: str,
    show: Callable[[Image.Image, Sequence[Candidate], str], None],
    predict: Callable[[str], list[Candidate]],
    log: Callable[..., None],
    refine: Callable[[Candidate, str], tuple[Candidate | None, str]] | None = None,
    show_tools: (
        Callable[[Image.Image, Sequence[Candidate], str, Sequence[ToolPreview]], None]
        | None
    ) = None,
) -> ImageReview:
    import matplotlib.pyplot as plt

    candidates = list(candidates)
    instances = [dict(row) for row in instances]
    assignments: dict[str, list[int]] = {}
    states: dict[str, str] = {}
    excluded: set[int] = set()
    ignored: set[int] = set()
    refinements: dict[str, Candidate] = {}

    def assigned_candidate(row: dict[str, str], *, original: bool = False) -> Candidate:
        identity = row["Instance ID"]
        if not original and identity in refinements:
            return replace(
                refinements[identity], visibility=states.get(identity, "clear")
            )
        fragments = [candidates[index] for index in assignments.get(identity, [])]
        mask = (
            np.logical_or.reduce([part.mask for part in fragments])
            if fragments
            else np.zeros((image.height, image.width), dtype=np.bool_)
        )
        return Candidate(
            mask,
            max((part.score for part in fragments), default=0),
            row["Tool Class"],
            identity,
            states.get(identity, "clear"),
        )

    def indices(text: str) -> list[int]:
        values = sorted({int(value) for value in text.split(",")})
        if not values or any(value < 0 or value >= len(candidates) for value in values):
            raise ValueError("Provide valid comma-separated candidate indices.")
        return values

    def record(action: str) -> None:
        log(
            key,
            "in_progress",
            action=action,
            assignments={
                identity: [
                    {
                        "index": index,
                        "candidate_id": candidates[index].identity,
                        "prompt": candidates[index].prompt,
                        "score": candidates[index].score,
                    }
                    for index in values
                ]
                for identity, values in assignments.items()
            },
            visibility=dict(states),
            excluded=sorted(excluded),
            ignored=sorted(ignored),
            refinements={
                identity: {"prompt": candidate.prompt, "score": candidate.score}
                for identity, candidate in refinements.items()
            },
        )

    while True:
        labels = []
        tools = {row["Instance ID"]: row["Tool ID"] for row in instances}
        for index, candidate in enumerate(candidates):
            owner = next(
                (
                    identity
                    for identity, values in assignments.items()
                    if index in values
                ),
                "unassigned",
            )
            if owner in tools:
                owner = f"tool {tools[owner]}"
            if index in ignored:
                owner = "ignore"
            elif index in excluded:
                owner = "excluded"
            labels.append(replace(candidate, prompt=f"{candidate.prompt} [{owner}]"))
        if show_tools is None:
            show(image, labels, key)
        else:
            previews = [
                ToolPreview(
                    row["Tool ID"],
                    canonical_tool_label(row["Tool Name"], row["Tool Class"], {}),
                    assigned_candidate(row).mask,
                    (
                        assigned_candidate(row, original=True).mask
                        if row["Instance ID"] in refinements
                        else None
                    ),
                )
                for row in instances
            ]
            show_tools(image, labels, key, previews)
        print(f"Mask candidates: {len(candidates)} (numbers match the preview)")
        for index, candidate in enumerate(labels):
            print(f"  {index}: {candidate.prompt} | score {candidate.score:.2f}")
        if not candidates:
            prompt = instances[0]["Tool Class"]
            print(
                f"No masks were predicted. Try 'p {prompt}' or a more specific prompt. "
                "Do not accept an empty annotation for a visible tool."
            )
        print("Declared tools (use Tool ID in commands):")
        for row in instances:
            identity = row["Instance ID"]
            print(
                f"  tool {row['Tool ID']} | {row['Tool Class']} | "
                f"{row['Tool Name']} | confidence {row['Confidence']} | "
                f"{states.get(identity, 'unresolved')} | masks {assignments.get(identity, [])}"
                + (
                    " | refined (restore TOOL_ID to undo)"
                    if identity in refinements
                    else ""
                )
            )
        if len(instances) == 1 and candidates:
            identity = instances[0]["Instance ID"]
            available = [
                index
                for index, candidate in enumerate(candidates)
                if candidate.mask.any()
                and index not in excluded
                and index not in ignored
            ]
            if available and identity not in assignments and identity not in states:
                print(
                    f"If candidate {available[0]} covers this tool, enter "
                    f"'g {instances[0]['Tool ID']} {available[0]}', then 'a'. "
                    "Group multiple pieces with comma-separated candidate numbers."
                )
        print(
            "g TOOL_ID 0,2=assign/group; u TOOL_ID=clear; "
            "v TOOL_ID clear|partial|severe|occluded|absent; "
            "refine TOOL_ID [TEXT]|all=crop refinement; restore TOOL_ID|all=original mask; "
            "x 0,2=exclude; i 0,2=ignore; p TEXT=add predictions; "
            'n TOOL_ID "CLASS" "NAME" CONFIDENCE=declare extra tool; '
            "a=accept; e=all absent; s=skip; r=reject; q=quit"
        )
        response = input("review> ").strip()
        plt.close("all")
        command, _, argument = response.partition(" ")
        try:
            if command in {"q", "s", "r"}:
                record(command)
                return ImageReview(
                    {"q": "quit", "s": "skipped", "r": "rejected"}[command],
                    instances,
                    [],
                    [],
                )
            if command == "n":
                fields = shlex.split(argument)
                if len(fields) == 4:
                    tool_id, tool_class, name, confidence = fields
                    identity = stable_media_id("instance", f"{key}/{tool_id}")
                elif len(fields) == 5:
                    identity, tool_id, tool_class, name, confidence = fields
                else:
                    raise ValueError('Use: n TOOL_ID "CLASS" "NAME" CONFIDENCE.')
                row = {
                    "Image ID": key,
                    "Instance ID": identity,
                    "Tool ID": tool_id,
                    "Tool Class": tool_class,
                    "Tool Name": name,
                    "Confidence": confidence,
                }
                validate_instance_metadata([*instances, row])
                instances.append(row)
                log(key, "in_progress", action="declare", instance=row)
            elif command == "p" and argument:
                additions = predict(argument)
                start = len(candidates)
                print(
                    f"Prompt {argument!r} returned {len(additions)} mask candidate(s)."
                )
                candidates.extend(
                    replace(candidate, identity=f"candidate-{start + index}")
                    for index, candidate in enumerate(additions)
                )
                log(
                    key,
                    "in_progress",
                    action="reprompt",
                    prompt=argument,
                    candidate_indices=list(range(start, len(candidates))),
                )
            elif command in {"refine", "restore"}:
                target, _, prompt_override = argument.partition(" ")
                targets = (
                    instances
                    if target == "all"
                    else [row for row in instances if row["Tool ID"] == target]
                )
                if not targets:
                    raise ValueError(f"Use {command} TOOL_ID or {command} all.")
                if target == "all" and prompt_override:
                    raise ValueError(
                        "Prompt overrides apply to one Tool ID, not all tools."
                    )
                if command == "refine" and refine is None:
                    raise ValueError("Crop refinement is unavailable in this reviewer.")
                if (
                    command == "refine"
                    and target != "all"
                    and not assignments.get(targets[0]["Instance ID"])
                ):
                    raise ValueError(
                        f"Assign tool {target} a mask with 'g {target} INDEX' before refinement."
                    )
                for row in targets:
                    identity = row["Instance ID"]
                    if command == "restore":
                        refinements.pop(identity, None)
                        print(
                            f"Tool {row['Tool ID']}: restored original assigned mask."
                        )
                    elif assignments.get(identity):
                        assert refine is not None
                        prompt = prompt_override or canonical_tool_label(
                            row["Tool Name"], row["Tool Class"], {}
                        )
                        replacement, message = refine(assigned_candidate(row), prompt)
                        if replacement is not None:
                            if (
                                replacement.mask.shape != (image.height, image.width)
                                or not replacement.mask.any()
                            ):
                                raise ValueError(
                                    "Refinement must return a nonempty full-image mask."
                                )
                            refinements[identity] = replace(
                                replacement, identity=identity
                            )
                        print(f"Tool {row['Tool ID']}: {message}")
                        log(
                            key,
                            "in_progress",
                            action="crop_refinement",
                            tool_id=row["Tool ID"],
                            instance_id=identity,
                            prompt=prompt,
                            applied=replacement is not None,
                            message=message,
                        )
                    else:
                        print(
                            f"Tool {row['Tool ID']}: no visible mask assigned; skipped refinement."
                        )
                record(command)
            elif command in {"g", "u", "v"}:
                target, _, value = argument.partition(" ")
                target_row = next(
                    (row for row in instances if row["Tool ID"] == target), None
                )
                if target_row is None:
                    # Accept old commands while keeping Tool IDs the primary interface.
                    target_row = next(
                        (row for row in instances if row["Instance ID"] == target), None
                    )
                if target_row is None:
                    raise ValueError(
                        "Unknown Tool ID. Use an ID from the declared tools list."
                    )
                row = target_row
                identity = row["Instance ID"]
                if command == "g":
                    if not value.strip():
                        raise ValueError(
                            f"Provide a candidate number: g {row['Tool ID']} 0 (or 0,2 to group pieces)."
                        )
                    values = indices(value)
                    if any(not candidates[index].mask.any() for index in values):
                        raise ValueError("Cannot assign an empty mask as visible.")
                    if any(
                        set(values) & set(assigned)
                        for owner, assigned in assignments.items()
                        if owner != identity
                    ):
                        raise ValueError(
                            "Clear the other tool's assignment before reusing a candidate."
                        )
                    assignments[identity] = values
                    refinements.pop(identity, None)
                    excluded.difference_update(values)
                    ignored.difference_update(values)
                    if states.get(identity) not in {"clear", "partial", "severe"}:
                        states[identity] = "clear"
                elif command == "u":
                    refinements.pop(identity, None)
                    assignments.pop(identity, None)
                    states.pop(identity, None)
                else:
                    if value not in {
                        "clear",
                        "partial",
                        "severe",
                        "occluded",
                        "absent",
                    }:
                        raise ValueError("Invalid visibility state.")
                    states[identity] = value
                    if value in {"occluded", "absent"}:
                        refinements.pop(identity, None)
                        assignments.pop(identity, None)
                record(
                    {
                        "g": "assign_fragments",
                        "u": "clear_assignment",
                        "v": "visibility",
                    }[command]
                )
            elif command in {"x", "i"}:
                values = indices(argument)
                if any(
                    set(values) & set(assigned) for assigned in assignments.values()
                ):
                    raise ValueError(
                        "Clear the tool assignment before excluding its candidates."
                    )
                excluded.difference_update(values)
                ignored.difference_update(values)
                (ignored if command == "i" else excluded).update(values)
                record("ignore" if command == "i" else "exclude")
            elif command in {"a", "e"}:
                if command == "e":
                    refinements.clear()
                    assignments.clear()
                    states = {row["Instance ID"]: "absent" for row in instances}
                    excluded.update(set(range(len(candidates))) - ignored)
                validate_instance_metadata(instances)
                for row in instances:
                    identity = row["Instance ID"]
                    state = states.get(identity)
                    if state is None or (
                        state in {"clear", "partial", "severe"}
                        and not assignments.get(identity)
                    ):
                        if not candidates:
                            raise ValueError(
                                f"Resolve tool {row['Tool ID']} before acceptance. No mask candidates "
                                f"are available; try 'p {row['Tool Class']}' or a more specific prompt. "
                                f"Use 'v {row['Tool ID']} absent' or 'v {row['Tool ID']} occluded' "
                                "only if the tool is absent or fully hidden."
                            )
                        raise ValueError(
                            f"Resolve tool {row['Tool ID']} before acceptance. Assign its mask with "
                            f"'g {row['Tool ID']} INDEX' (comma-separated indices combine pieces), "
                            f"or explicitly mark it 'v {row['Tool ID']} absent' or "
                            f"'v {row['Tool ID']} occluded' if appropriate."
                        )
                used = {index for values in assignments.values() for index in values}
                unresolved = set(range(len(candidates))) - used - excluded - ignored
                if unresolved:
                    raise ValueError(
                        f"Assign, exclude, or ignore candidates {sorted(unresolved)}."
                    )
                selected = [assigned_candidate(row) for row in instances]
                record("resolved")
                return ImageReview(
                    "accepted",
                    instances,
                    selected,
                    [candidates[index] for index in sorted(ignored)],
                )
            else:
                raise ValueError("Invalid review command.")
        except ValueError as error:
            print(error)
