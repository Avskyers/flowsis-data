"""Mouse controls and frame previews for the video annotation prototype."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from image_review import Candidate
from video_data import Mask

if TYPE_CHECKING:
    from matplotlib.backend_bases import KeyEvent, MouseEvent


def show_frame(
    image: Image.Image, mask: Mask, title: str, candidates: list[Candidate]
) -> None:
    import matplotlib.pyplot as plt

    plt.close("all")
    figure, axis = plt.subplots(figsize=(9, 8))
    axis.imshow(image)
    if mask.any():
        overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
        overlay[mask] = (0.0, 0.8, 0.3, 0.4)
        axis.imshow(overlay)
    for index, candidate in enumerate(candidates):
        overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
        overlay[candidate.mask] = (*plt.get_cmap("tab10")(index % 10)[:3], 0.35)
        axis.imshow(overlay)
        rows, columns = np.nonzero(candidate.mask)
        if len(rows):
            axis.text(
                int(columns.min()),
                int(rows.min()),
                f"{index}: {candidate.prompt} {candidate.score:.2f}",
                color="white",
                bbox={"facecolor": "black", "alpha": 0.7},
            )
    axis.set_title(title)
    axis.axis("off")
    plt.show(block=False)
    plt.pause(0.001)


def collect_clicks(
    image: Image.Image, mask: Mask, initial_points: list[tuple[float, float, int]]
) -> list[tuple[float, float, int]] | None:
    import matplotlib.pyplot as plt
    from matplotlib.backend_bases import MouseButton

    plt.close("all")
    figure, axis = plt.subplots(figsize=(9, 8))
    axis.imshow(image)
    overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
    overlay[mask] = (0.0, 0.8, 0.3, 0.4)
    axis.imshow(overlay)
    axis.set_title(
        "Left click: tool; right click: background. Backspace: undo. Enter: apply. Esc: cancel."
    )
    points = list(initial_points)
    markers = [
        axis.scatter([x], [y], c="lime" if label else "red", marker="+", s=100)
        for x, y, label in points
    ]
    accepted = False

    def click(event: MouseEvent) -> None:
        if event.inaxes != axis or event.xdata is None or event.ydata is None:
            return
        if event.button not in {MouseButton.LEFT, MouseButton.RIGHT}:
            return
        label = int(event.button == MouseButton.LEFT)
        x = min(max(float(event.xdata), 0), image.width - 1)
        y = min(max(float(event.ydata), 0), image.height - 1)
        points.append((x, y, label))
        markers.append(
            axis.scatter([x], [y], c="lime" if label else "red", marker="+", s=100)
        )
        figure.canvas.draw_idle()

    def key(event: KeyEvent) -> None:
        nonlocal accepted
        if event.key == "backspace" and points:
            points.pop()
            markers.pop().remove()
            figure.canvas.draw_idle()
        elif event.key in {"enter", "escape"}:
            accepted = event.key == "enter"
            plt.close(figure)

    figure.canvas.mpl_connect("button_press_event", click)
    figure.canvas.mpl_connect("key_press_event", key)
    plt.show(block=True)
    return points if accepted else None


def select_region(image: Image.Image) -> tuple[int, int, int, int] | None:
    import matplotlib.pyplot as plt
    from matplotlib.backend_bases import MouseButton
    from matplotlib.widgets import RectangleSelector

    plt.close("all")
    figure, axis = plt.subplots(figsize=(9, 8))
    axis.imshow(image)
    axis.set_title("Drag a rectangle. Enter: select. Esc: cancel.")
    accepted = False

    def selected(press: MouseEvent, release: MouseEvent) -> None:
        pass

    selector = RectangleSelector(
        axis,
        selected,
        button=[MouseButton.LEFT],
        interactive=True,
        minspanx=1,
        minspany=1,
    )

    def key(event: KeyEvent) -> None:
        nonlocal accepted
        if event.key in {"enter", "escape"}:
            accepted = event.key == "enter"
            plt.close(figure)

    figure.canvas.mpl_connect("key_press_event", key)
    plt.show(block=True)
    if not accepted:
        return None
    left, right, top, bottom = selector.extents
    box = (
        max(0, math.floor(left)),
        max(0, math.floor(top)),
        min(image.width, math.ceil(right)),
        min(image.height, math.ceil(bottom)),
    )
    return box if box[0] < box[2] and box[1] < box[3] else None


def preview_frames(items: list[tuple[Image.Image, Mask, str]]) -> None:
    """Display a bounded page of frame overlays before range acceptance."""
    import matplotlib.pyplot as plt

    plt.close("all")
    figure, axes = plt.subplots(3, 4, figsize=(14, 10))
    for axis, (image, mask, title) in zip(axes.flat, items):
        axis.imshow(image)
        overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
        overlay[mask] = (0.0, 0.8, 0.3, 0.4)
        axis.imshow(overlay)
        axis.set_title(title)
        axis.axis("off")
    for axis in list(axes.flat)[len(items) :]:
        axis.axis("off")
    figure.suptitle(
        "Inspect every mask. Close this page to continue; acceptance is a separate command."
    )
    plt.tight_layout()
    plt.show(block=True)
