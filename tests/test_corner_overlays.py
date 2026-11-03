"""Overlay removal must preserve scientific pixels outside selected boxes."""
import numpy as np
import pytest

from qal.util import remove_corner_overlays


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
@pytest.mark.parametrize("replacement", ["zero", "mean", "median", 12, (1, 2, 3)])
def test_overlay_box_fill_preserves_input_alpha_and_center(dtype, replacement):
    maximum = np.iinfo(dtype).max if np.issubdtype(dtype, np.integer) else 1
    image = np.full((40, 60, 4), maximum * .1, dtype=dtype)
    image[..., 3] = maximum
    image[2:7, 3:8, :3] = [maximum, 0, 0]
    image[18:23, 28:33, :3] = [0, maximum, 0]  # central signal: never remove
    before = image.copy()
    result, boxes = remove_corner_overlays(image, corners="top_left", padding=0,
        min_component_area=4, high_intensity_threshold=None, replacement=replacement, return_bboxes=True)
    assert boxes == [{"corner": "top_left", "bbox": (2, 3, 7, 8)}]
    assert result.dtype == dtype
    np.testing.assert_array_equal(result[..., 3], image[..., 3])
    outside = np.ones(image.shape[:2], dtype=bool)
    outside[2:7, 3:8] = False
    np.testing.assert_array_equal(result[outside], image[outside])
    if replacement == "zero":
        expected = [0, 0, 0]
    elif isinstance(replacement, str):
        values = image[..., :3].astype(float).mean(axis=-1)[outside]
        expected = np.repeat(values.mean() if replacement == "mean" else np.median(values), 3)
        if np.issubdtype(dtype, np.integer):
            expected = np.rint(expected)
    else:
        expected = np.broadcast_to(replacement, (3,))
    np.testing.assert_allclose(result[2:7, 3:8, :3], np.broadcast_to(np.array(expected, dtype=dtype), (5, 5, 3)))
    np.testing.assert_array_equal(image, before)


def test_small_components_are_preserved_and_selected_channel_controls_detection():
    image = np.zeros((40, 40, 3), dtype=np.uint8)
    image[1:6, 2:7, 1] = 255
    image[8, 8, 1] = 255
    result, boxes = remove_corner_overlays(image, color_threshold=None,
        high_intensity_channels="green", min_component_area=4, padding=0, return_bboxes=True)
    assert len(boxes) == 1
    assert np.all(result[1:6, 2:7] == 0)
    assert result[8, 8, 1] == 255
    np.testing.assert_array_equal(remove_corner_overlays(image, color_threshold=None,
        high_intensity_channels="red", min_component_area=4), image)


@pytest.mark.parametrize("kwargs", [
    {"corners": "middle"}, {"corner_fraction": 0}, {"corner_fraction": .6},
    {"padding": -1}, {"color_threshold": 2}, {"min_component_area": 0},
    {"high_intensity_channels": "alpha"},
    {"color_threshold": None, "high_intensity_threshold": None},
])
def test_invalid_overlay_options_raise(kwargs):
    with pytest.raises(ValueError):
        remove_corner_overlays(np.zeros((20, 20, 3), dtype=np.uint8), **kwargs)
