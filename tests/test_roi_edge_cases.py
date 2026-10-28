"""Independent pixel-selection and statistical checks for public ROI helpers."""
import numpy as np
import pytest

from qal.util import circular_mask, raw_stats_in_circular_roi, mean_in_circular_roi, std_in_circular_roi


@pytest.mark.parametrize("center,radius", [((0, 0), 2), ((3.5, 2.25), 1.75), ((7, 5), 20)])
def test_clipped_and_subpixel_roi_matches_explicit_pixel_distances(center, radius):
    image = np.arange(48, dtype=np.uint16).reshape(6, 8)
    expected = np.array([image[y, x] for y in range(6) for x in range(8)
                         if (x - center[0])**2 + (y - center[1])**2 <= radius**2], dtype=float)
    stats = raw_stats_in_circular_roi(image, center, radius)
    assert stats.count == expected.size
    assert stats.mean == pytest.approx(expected.mean())
    assert stats.std == pytest.approx(expected.std())
    assert stats.std_sample == pytest.approx(expected.std(ddof=1))
    assert (stats.min_value, stats.max_value) == (expected.min(), expected.max())


def test_nonfinite_pixels_are_excluded_from_all_statistics():
    image = np.array([[np.nan, 1., np.inf], [3., 5., -np.inf], [7., 9., 11.]])
    stats = raw_stats_in_circular_roi(image, (1, 1), 5)
    assert stats.count == 6
    assert mean_in_circular_roi(image, (1, 1), 5) == 6
    assert std_in_circular_roi(image, (1, 1), 5) == pytest.approx(np.std([1, 3, 5, 7, 9, 11]))


def test_single_pixel_has_zero_sample_deviation():
    stats = raw_stats_in_circular_roi(np.array([[42]]), (0, 0), .1)
    assert stats.count == 1
    assert stats.mean == 42
    assert stats.std == stats.std_sample == 0


@pytest.mark.parametrize("image,center", [(np.zeros((3, 5)), (100, 100)), (np.full((3, 5), np.nan), (1, 1))])
def test_empty_or_nonfinite_roi_raises(image, center):
    with pytest.raises(ValueError, match="no finite"):
        raw_stats_in_circular_roi(image, center, 1)


@pytest.mark.parametrize("radius", [0, -1, np.nan, np.inf])
def test_invalid_radii_are_rejected(radius):
    with pytest.raises(ValueError, match="radius"):
        circular_mask((5, 7), (2, 2), radius)


def test_invalid_coordinate_order_is_rejected():
    with pytest.raises(ValueError, match="center_order"):
        circular_mask((5, 7), (2, 2), 1, center_order="yx")
