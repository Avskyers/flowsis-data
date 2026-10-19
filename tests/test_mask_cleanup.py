"""Cleanup must preserve real topology unless an explicit operation changes it."""

from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

from annotate import review_candidates
from image_review import Candidate, review_image_instances
from mask_cleanup import clean_mask, parse_cleanup


def imperfect_mask():
    mask = np.zeros((40, 50), dtype=np.bool_)
    mask[5:30, 5:35] = True
    mask[10, 10] = False  # Small enclosed defect.
    mask[15:20, 15:20] = False  # Real opening larger than the default threshold.
    mask[0, 45] = True  # Detached speck.
    return mask


def test_small_cleanup_preserves_large_holes_and_separate_visible_portions():
    mask = imperfect_mask()
    mask[32:38, 40:47] = True
    original = mask.copy()
    result = clean_mask(mask, *parse_cleanup(""))
    assert result[10, 10]
    assert not result[0, 45]
    assert not result[15:20, 15:20].any()
    assert result[32:38, 40:47].all()
    np.testing.assert_array_equal(mask, original)


def test_hole_filling_does_not_close_edge_notches_or_exterior_background():
    mask = np.ones((20, 20), dtype=np.bool_)
    mask[:5, 10] = False  # Small but connected to the image boundary.
    mask[10, 10] = False
    result = clean_mask(mask, "holes", 16)
    assert result[10, 10]
    assert not result[:5, 10].any()
    assert not clean_mask(np.zeros_like(mask), "holes", 1000).any()


def test_opening_removes_long_thin_subtraction_slivers_that_area_filter_keeps():
    mask = np.zeros((50, 50), dtype=np.bool_)
    mask[5:30, 5:30] = True
    mask[5:45, 40] = True  # Too long to count as a small speck.
    assert clean_mask(mask, "specks", 16)[5:45, 40].all()
    result = clean_mask(mask, "open", 3)
    assert not result[:, 40].any()
    assert result[10:25, 10:25].all()


def test_closing_fills_edge_defects_and_keeps_objects_touching_image_boundary():
    mask = np.ones((30, 30), dtype=np.bool_)
    mask[0:3, 15] = False
    result = clean_mask(mask, "close", 3)
    assert result[1, 15]
    assert result[:, 0].all() and result[-1, :].all()
    np.testing.assert_array_equal(
        clean_mask(np.ones_like(mask), "close", 3), np.ones_like(mask)
    )


def test_protected_tool_pixels_and_ignore_regions_are_not_claimed():
    mask = imperfect_mask()
    protected = np.zeros_like(mask)
    protected[10, 10] = True
    result = clean_mask(mask, "holes", 16, protected=protected)
    assert not result[10, 10]
    # Cleanup does not silently resolve pre-existing overlaps either.
    protected[8, 8] = True
    assert clean_mask(mask, "holes", 16, protected=protected)[8, 8]


@pytest.mark.parametrize(
    "text", ["unknown", "open 2", "close 0", "holes -1", "small 1 extra"]
)
def test_invalid_cleanup_parameters(text):
    with pytest.raises(ValueError):
        parse_cleanup(text)


def review(monkeypatch, commands, masks, *, refine=None, show_tools=None):
    replies = iter(commands)
    monkeypatch.setattr("builtins.input", lambda _: next(replies))
    logs = []
    rows = [
        {
            "Image ID": "image",
            "Instance ID": f"internal-{index}",
            "Tool ID": str(index),
            "Tool Class": "pliers",
            "Tool Name": "rongeur",
            "Confidence": "1",
        }
        for index in range(len(masks))
    ]
    result = review_image_instances(
        Image.new("RGB", (50, 40)),
        [
            Candidate(mask, 0.9, "tools", f"candidate-{index}")
            for index, mask in enumerate(masks)
        ],
        rows,
        key="image",
        show=lambda *_: None,
        show_tools=show_tools,
        predict=lambda _: [],
        refine=refine,
        log=lambda *args, **kwargs: logs.append(kwargs),
    )
    return result, logs


def test_cleanup_is_optional_and_previewed_before_acceptance(monkeypatch):
    mask = imperfect_mask()
    unchanged, _ = review(monkeypatch, ["g 0 0", "a"], [mask])
    np.testing.assert_array_equal(unchanged.selected[0].mask, mask)
    previews = []
    result, logs = review(
        monkeypatch,
        ["g 0 0", "clean 0", "a"],
        [mask],
        show_tools=lambda image, candidates, key, tools: previews.append(tools),
    )
    np.testing.assert_array_equal(
        result.selected[0].mask, clean_mask(mask, "small", 16)
    )
    assert result.selected[0].identity == "internal-0"
    assert previews[-1][0].edit_note == "Cleaned mask — unclean 0 to undo"
    np.testing.assert_array_equal(previews[-1][0].mask, result.selected[0].mask)
    action = next(log for log in logs if log.get("action") == "cleanup")
    assert action["added_pixels"] == 1 and action["removed_pixels"] == 1


def test_undo_cleanup_preserves_upstream_crop_refinement(monkeypatch):
    mask = imperfect_mask()
    refined = mask.copy()
    refined[28:30, 5:35] = False
    result, _ = review(
        monkeypatch,
        ["g 0 0", "refine 0", "clean 0 holes 16", "clean 0 open 3", "unclean 0", "a"],
        [mask],
        refine=lambda seed, _: (replace(seed, mask=refined), "refined"),
    )
    np.testing.assert_array_equal(result.selected[0].mask, refined)


def test_cleanup_does_not_fill_another_tool_or_ignored_candidate(monkeypatch):
    first = imperfect_mask()
    other = np.zeros_like(first)
    other[10, 10] = True
    result, _ = review(
        monkeypatch, ["g 0 0", "g 1 1", "clean 0 holes 16", "a"], [first, other]
    )
    assert not (result.selected[0].mask & result.selected[1].mask).any()
    # An ignored candidate is protected even though it is not assigned to a tool.
    result, _ = review(
        monkeypatch,
        ["g 0 0", "v 1 absent", "i 1", "clean 0 holes 16", "a"],
        [first, other],
    )
    assert not result.selected[0].mask[10, 10]


def test_reassignment_clears_stale_cleanup(monkeypatch):
    mask = imperfect_mask()
    result, _ = review(monkeypatch, ["g 0 0", "clean 0", "g 0 0", "a"], [mask])
    np.testing.assert_array_equal(result.selected[0].mask, mask)


def test_empty_cleanup_retains_previous_mask(monkeypatch, capsys):
    mask = np.zeros((40, 50), dtype=np.bool_)
    mask[5:35, 5] = True
    result, _ = review(monkeypatch, ["g 0 0", "clean 0 open 3", "a"], [mask])
    np.testing.assert_array_equal(result.selected[0].mask, mask)
    assert "would erase the whole tool" in capsys.readouterr().out


def test_subtraction_uses_postprocessed_mask(monkeypatch):
    combined = np.ones((40, 50), dtype=np.bool_)
    part = imperfect_mask()
    result, _ = review(
        monkeypatch,
        ["g 1 1", "clean 1", "subtract 0 0 1", "x 0", "a"],
        [combined, part],
    )
    processed = clean_mask(part, "small", 16)
    np.testing.assert_array_equal(result.selected[0].mask, ~processed)
    np.testing.assert_array_equal(result.selected[1].mask, processed)


@pytest.mark.parametrize("undo", [False, True])
def test_single_tool_cleanup_and_undo(monkeypatch, undo):
    mask = imperfect_mask()
    candidate = Candidate(mask, 0.9, "pliers", "candidate-0")
    commands = ["clean holes 16", "clean specks 16"]
    if undo:
        commands.append("unclean")
    replies = iter([*commands, "a"])
    monkeypatch.setattr("builtins.input", lambda _: next(replies))
    monkeypatch.setattr("annotate.show_candidates", lambda *_args, **_kwargs: None)
    decision, selected, *_ = review_candidates(
        Image.new("RGB", (50, 40)), [candidate], key="image", auto_accept=False
    )
    assert decision == "accepted"
    np.testing.assert_array_equal(
        selected[0].mask, mask if undo else clean_mask(mask, "small", 16)
    )
    np.testing.assert_array_equal(candidate.mask, mask)


def test_cleanup_uses_union_instead_of_filtering_each_fragment(monkeypatch):
    first = np.zeros((40, 50), dtype=np.bool_)
    first[10:13, 10:14] = True
    second = np.zeros_like(first)
    second[10:13, 14:18] = True
    result, _ = review(
        monkeypatch, ["g 0 0,1", "v 1 absent", "clean 0", "a"], [first, second]
    )
    # Each fragment is <=16 pixels but their joined 24-pixel mask is real.
    np.testing.assert_array_equal(result.selected[0].mask, first | second)
