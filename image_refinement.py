"""Refine one assigned tool on original pixels and restore full-image coordinates."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from masks import mask2xywh

if TYPE_CHECKING:
    from image_review import Candidate


def tool_crop(
    mask: NDArray[np.bool_], padding: float = 0.2
) -> tuple[int, int, int, int]:
    box = mask2xywh(mask)
    if box is None:
        raise ValueError("Assign a nonempty mask before refining this tool.")
    x, y, width, height = box
    margin = max(1, int(np.ceil(padding * max(width, height))))
    return (
        max(0, x - margin),
        max(0, y - margin),
        min(mask.shape[1], x + width + margin),
        min(mask.shape[0], y + height + margin),
    )


def refine_tool(
    image: Image.Image,
    seed: Candidate,
    prompt: str,
    predict: Callable[[Image.Image, str], list[Candidate]],
) -> tuple[Candidate | None, str]:
    """Select a crop prediction by agreement with the assigned tool, never by class alone."""
    if seed.mask.shape != (image.height, image.width) or not seed.mask.any():
        raise ValueError("Assign a nonempty, full-image mask before refinement.")
    previous_bounds = None
    for padding in (0.2, 0.4, 0.8):
        bounds = tool_crop(seed.mask, padding)
        if bounds == previous_bounds:
            break
        previous_bounds = bounds
        left, top, right, bottom = bounds
        crop = image.crop(bounds)
        reference = seed.mask[top:bottom, left:right]
        ranked = []
        for candidate in predict(crop, prompt):
            if candidate.mask.shape != reference.shape:
                raise ValueError(
                    "Crop prediction dimensions do not match the original image crop."
                )
            intersection = int(np.count_nonzero(reference & candidate.mask))
            union = int(np.count_nonzero(reference | candidate.mask))
            overlap = intersection / union if union else 0.0
            coverage = intersection / int(np.count_nonzero(reference))
            ranked.append((overlap, coverage, candidate))
        ranked.sort(key=lambda item: item[0], reverse=True)
        if not ranked or ranked[0][0] < 0.25 or ranked[0][1] < 0.5:
            return (
                None,
                "No prediction reliably matches the assigned tool; previous mask retained.",
            )
        if len(ranked) > 1 and ranked[0][0] - ranked[1][0] < 0.1:
            return (
                None,
                "Ambiguous crop predictions match this tool; previous mask retained.",
            )
        candidate = ranked[0][2]
        mask = candidate.mask
        clipped = (
            (left > 0 and mask[:, 0].any())
            or (top > 0 and mask[0, :].any())
            or (right < image.width and mask[:, -1].any())
            or (bottom < image.height and mask[-1, :].any())
        )
        if clipped:
            continue
        full_mask = np.zeros_like(seed.mask)
        full_mask[top:bottom, left:right] = mask
        return replace(
            candidate,
            mask=full_mask,
            identity=seed.identity,
            visibility=seed.visibility,
        ), (f"Refined on crop {bounds}; inspect the tool panel before accepting.")
    return (
        None,
        "Refinement touches an interior crop edge after expansion; previous mask retained.",
    )
