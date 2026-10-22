import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backend_bases import KeyEvent
from PIL import Image

from image_ui import ToolPreview, dispute_crops, show_overlap_review


def test_dispute_crops_cover_scattered_pixels_and_bound_large_regions():
    disputed = np.zeros((600, 800), dtype=np.bool_)
    disputed[0, 0] = True
    disputed[255:258, 254:258] = True
    disputed[300:580, 400:780] = True
    disputed[-1, -1] = True
    covered = np.zeros_like(disputed)
    crops = dispute_crops(disputed)
    assert len(crops) > 1
    for left, top, right, bottom in crops:
        assert 0 <= left < right <= 800
        assert 0 <= top < bottom <= 600
        assert 128 <= right - left <= 320
        assert 128 <= bottom - top <= 320
        covered[top:bottom, left:right] = True
    assert covered[disputed].all()
    assert dispute_crops(np.zeros_like(disputed)) == []


def test_dispute_crops_handle_images_smaller_than_minimum_crop():
    disputed = np.zeros((10, 12), dtype=np.bool_)
    disputed[-1, -1] = True
    assert dispute_crops(disputed) == [(0, 0, 12, 10)]


def test_inspection_preserves_pixels_ownership_and_linked_navigation(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda **_: None)
    monkeypatch.setattr(plt, "pause", lambda _: None)
    pixels = np.arange(300 * 600 * 3, dtype=np.uint8).reshape(300, 600, 3)
    top = np.zeros((300, 600), dtype=np.bool_)
    bottom = top.copy()
    disputed = top.copy()
    disputed[30:40, 30:40] = True
    disputed[260:270, 550:560] = True
    top[20:50, 20:50] = True
    top[250:280, 540:570] = True
    bottom[30:60, 30:60] = True
    bottom[260:290, 550:580] = True
    tools = [
        ToolPreview("12", "scissors", top, top, candidate_index=0, disputed=disputed),
        ToolPreview(
            "13", "forceps", bottom & ~top, bottom,
            candidate_index=1, disputed=disputed,
        ),
    ]
    crops = dispute_crops(disputed)
    try:
        show_overlap_review(Image.fromarray(pixels), tools, "image-1")
        figure = plt.gcf()
        original, before, after = figure.axes
        left, y, right, b = crops[0]
        np.testing.assert_array_equal(original.images[0].get_array(), pixels[y:b, left:right])
        assert all(axis.images[0].get_interpolation() == "nearest" for axis in figure.axes)
        # Tool 13 owns the disputed pixels before ordering, but not afterwards.
        before_alpha = np.asarray(before.images[2].get_array())[:, :, 3]
        after_alpha = np.asarray(after.images[2].get_array())[:, :, 3]
        assert before_alpha[30 - y, 30 - left] == 115
        assert after_alpha[30 - y, 30 - left] == 0
        before.set_xlim(left + 20, left + 40)
        assert original.get_xlim() == after.get_xlim() == before.get_xlim()
        figure.canvas.callbacks.process(
            "key_press_event", KeyEvent("key_press_event", figure.canvas, key="m")
        )
        assert not any(layer.get_visible() for axis in [before, after] for layer in axis.images[1:])
        figure.canvas.callbacks.process(
            "key_press_event", KeyEvent("key_press_event", figure.canvas, key="]")
        )
        left, y, right, b = crops[1]
        np.testing.assert_array_equal(original.images[0].get_array(), pixels[y:b, left:right])
        assert "region 2/" in figure._suptitle.get_text()
        assert not any(layer.get_visible() for axis in [before, after] for layer in axis.images[1:])
        figure.canvas.callbacks.process(
            "key_press_event", KeyEvent("key_press_event", figure.canvas, key="[")
        )
        assert "region 1/" in figure._suptitle.get_text()
        figure.canvas.callbacks.process(
            "key_press_event", KeyEvent("key_press_event", figure.canvas, key="escape")
        )
        assert not plt.fignum_exists(figure.number)
    finally:
        plt.close("all")
