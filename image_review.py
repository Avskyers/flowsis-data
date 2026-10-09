"""Explicit image-instance assignment in the Matplotlib/terminal reviewer."""

from __future__ import annotations

import shlex
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from annotations import validate_instance_metadata


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
) -> ImageReview:
    import matplotlib.pyplot as plt

    candidates = list(candidates)
    instances = [dict(row) for row in instances]
    assignments: dict[str, list[int]] = {}
    states: dict[str, str] = {}
    excluded: set[int] = set()
    ignored: set[int] = set()

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
        )

    while True:
        labels = []
        for index, candidate in enumerate(candidates):
            owner = next(
                (
                    identity
                    for identity, values in assignments.items()
                    if index in values
                ),
                "unassigned",
            )
            if index in ignored:
                owner = "ignore"
            elif index in excluded:
                owner = "excluded"
            labels.append(replace(candidate, prompt=f"{candidate.prompt} [{owner}]"))
        show(image, labels, key)
        print("Declared tools (use Instance ID in commands):")
        for row in instances:
            identity = row["Instance ID"]
            print(
                f"  {identity}: tool {row['Tool ID']} | {row['Tool Class']} | "
                f"{row['Tool Name']} | confidence {row['Confidence']} | "
                f"{states.get(identity, 'unresolved')} | masks {assignments.get(identity, [])}"
            )
        print(
            "g INSTANCE 0,2=assign/group; u INSTANCE=clear; "
            "v INSTANCE clear|partial|severe|occluded|absent; "
            "x 0,2=exclude; i 0,2=ignore; p TEXT=add predictions; "
            'n INSTANCE TOOL_ID "CLASS" "NAME" CONFIDENCE=declare extra tool; '
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
                identity, tool_id, tool_class, name, confidence = shlex.split(argument)
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
            elif command in {"g", "u", "v"}:
                identity, _, value = argument.partition(" ")
                if identity not in {row["Instance ID"] for row in instances}:
                    raise ValueError("Unknown Instance ID.")
                if command == "g":
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
                    excluded.difference_update(values)
                    ignored.difference_update(values)
                    if states.get(identity) not in {"clear", "partial", "severe"}:
                        states[identity] = "clear"
                elif command == "u":
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
                        raise ValueError(f"Resolve tool {identity} before acceptance.")
                used = {index for values in assignments.values() for index in values}
                unresolved = set(range(len(candidates))) - used - excluded - ignored
                if unresolved:
                    raise ValueError(
                        f"Assign, exclude, or ignore candidates {sorted(unresolved)}."
                    )
                selected = []
                for row in instances:
                    identity = row["Instance ID"]
                    values = assignments.get(identity, [])
                    fragments = [candidates[index] for index in values]
                    mask = (
                        np.logical_or.reduce([part.mask for part in fragments])
                        if fragments
                        else np.zeros((image.height, image.width), dtype=np.bool_)
                    )
                    selected.append(
                        Candidate(
                            mask,
                            max((part.score for part in fragments), default=0),
                            row["Tool Class"],
                            identity,
                            states[identity],
                        )
                    )
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
