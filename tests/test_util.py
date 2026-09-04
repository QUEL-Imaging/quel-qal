import numpy as np
import pytest
from skimage import io

from qal.util import circular_mask, load_grayscale_image, mean_in_circular_roi


def test_circular_mask_supports_xy_and_row_column_centers():
    xy_mask = circular_mask((7, 9), (6, 2), 1, center_order="xy")
    rc_mask = circular_mask((7, 9), (2, 6), 1, center_order="rc")

    assert np.array_equal(xy_mask, rc_mask)
    assert xy_mask[2, 6]
    assert not xy_mask[6, 2]


def test_mean_in_circular_roi_uses_requested_center():
    image = np.zeros((9, 9), dtype=float)
    image[circular_mask(image.shape, (6, 2), 1)] = 7

    assert mean_in_circular_roi(image, (6, 2), 1) == 7


def test_load_grayscale_image_reads_first_rgb_channel(tmp_path):
    rgb = np.zeros((5, 6, 3), dtype=np.uint8)
    rgb[..., 0] = 11
    rgb[..., 1] = 22
    path = tmp_path / "rgb.tiff"
    io.imsave(path, rgb, check_contrast=False)

    loaded = load_grayscale_image(path)

    assert loaded.shape == (5, 6)
    assert np.all(loaded == 11)


def test_load_grayscale_image_rejects_non_image_array():
    with pytest.raises(ValueError, match="2-D"):
        load_grayscale_image(np.zeros((2, 3, 4, 5)))
