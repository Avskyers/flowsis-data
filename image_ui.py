"""Combined multi-tool overview and original-pixel crop panels."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from image_refinement import tool_crop
from masks import mask2xywh

if TYPE_CHECKING:
    from image_review import Candidate


@dataclass(frozen=True)
class ToolPreview:
    tool_id: str
    label: str
    mask: NDArray[np.bool_]
    original: NDArray[np.bool_] | None = None


def show_tool_review(
    image: Image.Image,
    candidates: Sequence[Candidate],
    title: str,
    tools: Sequence[ToolPreview],
) -> None:
    import matplotlib.pyplot as plt

    columns = min(2, max(1, len(tools)))
    rows = max(1, math.ceil(len(tools) / columns))
    figure = plt.figure(figsize=(6 * (columns + 1), 5 * rows))
    grid = figure.add_gridspec(rows, columns + 1, width_ratios=[1.3, *([1] * columns)])
    overview = figure.add_subplot(grid[:, 0])
    overview_pixels = np.array(image.convert("RGB"))
    figure.suptitle(title, fontsize=11, y=0.98)
    overview.set_title("Overview — numbered mask candidates", fontsize=10)
    overview.axis("off")
    for index, candidate in enumerate(candidates):
        box = mask2xywh(candidate.mask)
        if box is None:
            continue
        x, y, width, height = box
        overview.add_patch(
            plt.Rectangle(
                (x, y), width, height, fill=False, edgecolor="gray", linestyle=":"
            )
        )
        overview.text(
            x,
            y,
            f"{index}: {candidate.prompt.split(' [', 1)[0]}",
            fontsize=8,
            color="white",
            bbox={"facecolor": "black", "alpha": 0.7},
        )
        # Unassigned candidates remain visible until assigned, excluded, or ignored.
        if "[unassigned]" in candidate.prompt:
            overview_pixels[candidate.mask] = (
                0.7 * overview_pixels[candidate.mask] + 0.3 * 178
            ).astype(np.uint8)
    colors = plt.get_cmap("tab10")
    for index, tool in enumerate(tools):
        color = colors(index % 10)[:3]
        axis = figure.add_subplot(grid[index // columns, 1 + index % columns])
        axis.set_title(f"Tool {tool.tool_id} — {tool.label}", color=color, fontsize=10)
        axis.axis("off")
        if not tool.mask.any():
            axis.text(
                0.5,
                0.5,
                "No visible mask assigned",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
            continue
        overview_pixels[tool.mask] = (
            0.55 * overview_pixels[tool.mask] + 0.45 * np.array(color) * 255
        ).astype(np.uint8)
        overview_box = mask2xywh(tool.mask)
        assert overview_box is not None
        overview.text(
            overview_box[0],
            overview_box[1] + overview_box[3],
            f"tool {tool.tool_id}",
            color=color,
            bbox={"facecolor": "black", "alpha": 0.7},
        )
        extent = tool.mask if tool.original is None else tool.mask | tool.original
        left, top, right, bottom = tool_crop(extent)
        axis.imshow(image.crop((left, top, right, bottom)))
        crop_mask = tool.mask[top:bottom, left:right]
        overlay = np.zeros((*crop_mask.shape, 4), dtype=np.uint8)
        overlay[crop_mask] = (*[round(channel * 255) for channel in color], 115)
        axis.imshow(overlay, interpolation="nearest")
        if tool.original is not None and not np.array_equal(tool.original, tool.mask):
            axis.contour(
                tool.original[top:bottom, left:right],
                levels=[0.5],
                colors=["gold"],
                linestyles="dashed",
                linewidths=1,
            )
            axis.text(
                0.02,
                0.02,
                "Color: current mask | dashed gold: original",
                transform=axis.transAxes,
                fontsize=8,
                bbox={"facecolor": "white", "alpha": 0.8},
            )
    overview.imshow(overview_pixels)
    figure.subplots_adjust(
        left=0.03, right=0.98, bottom=0.06, top=0.88, wspace=0.15, hspace=0.3
    )
    plt.show(block=False)
    plt.pause(0.001)
