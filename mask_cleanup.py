"""Optional cleanup of reviewed binary masks at their saved pixel resolution."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray


def parse_cleanup(text: str) -> tuple[str, int]:
    """Normalize an optional operation and pixel size at the command boundary."""
    parts = text.split()
    if len(parts) > 2:
        raise ValueError("Use [small|holes|specks|open|close|smooth] [SIZE].")
    operation = parts[0] if parts else "small"
    if operation not in {"small", "holes", "specks", "open", "close", "smooth"}:
        raise ValueError(
            "Cleanup operation must be small, holes, specks, open, close, or smooth."
        )
    size = (
        int(parts[1])
        if len(parts) == 2
        else (3 if operation in {"open", "close", "smooth"} else 16)
    )
    if size < 1 or (operation in {"open", "close", "smooth"} and size % 2 == 0):
        raise ValueError("Use a positive size; morphology kernel widths must be odd.")
    return operation, size


def clean_mask(
    mask: NDArray[np.bool_],
    operation: str,
    size: int,
    *,
    protected: NDArray[np.bool_] | None = None,
) -> NDArray[np.bool_]:
    """Remove small regions or apply morphology, without claiming protected pixels.

    Area thresholds are inclusive. Foreground uses 8-connectivity; holes use
    4-connectivity and never include background connected to an image edge.
    Existing overlaps are preserved, but cleanup cannot introduce new ones.
    """
    parse_cleanup(f"{operation} {size}")
    if mask.ndim != 2 or mask.dtype != np.bool_:
        raise ValueError("Cleanup requires a two-dimensional boolean mask.")
    if protected is not None and (
        protected.shape != mask.shape or protected.dtype != np.bool_
    ):
        raise ValueError("Protected pixels must be a boolean mask of the same shape.")
    result = mask.copy()
    if operation in {"small", "specks"}:
        _, labels, stats, _ = cv2.connectedComponentsWithStats(
            result.astype(np.uint8), connectivity=8
        )
        keep = stats[:, cv2.CC_STAT_AREA] > size
        keep[0] = False
        result = keep[labels]
    if operation in {"small", "holes"}:
        _, labels, stats, _ = cv2.connectedComponentsWithStats(
            (~result).astype(np.uint8), connectivity=4
        )
        fill = stats[:, cv2.CC_STAT_AREA] <= size
        fill[0] = False
        exterior = np.unique(
            np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1]))
        )
        fill[exterior] = False
        result |= fill[labels]
    if operation in {"open", "close", "smooth"}:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        # Explicit background padding keeps closing from eroding the image edge.
        padding = size // 2 + 1
        pixels = np.pad(result.astype(np.uint8), padding)
        if operation in {"open", "smooth"}:
            pixels = cv2.morphologyEx(
                pixels,
                cv2.MORPH_OPEN,
                kernel,
                borderType=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
        if operation in {"close", "smooth"}:
            pixels = cv2.morphologyEx(
                pixels,
                cv2.MORPH_CLOSE,
                kernel,
                borderType=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
        result = pixels[padding:-padding, padding:-padding].astype(np.bool_)
    if protected is not None:
        result &= ~protected | mask
    return result
