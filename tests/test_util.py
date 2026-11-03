import numpy as np
import pytest
from skimage import io

from qal.util import (
    circular_mask,
    describe_stored_values,
    load_grayscale_image,
    mean_in_circular_roi,
    raw_stats_in_circular_roi,
    to_log_scale,
)


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


def test_load_grayscale_image_keeps_8bit_samples_in_16bit_tiff(tmp_path):
    image = np.zeros((8, 9), dtype=np.uint16)
    image[2:6, 3:7] = np.arange(16, dtype=np.uint16).reshape(4, 4)
    path = tmp_path / "false_16bit.tiff"
    io.imsave(path, image, check_contrast=False)

    loaded = load_grayscale_image(path)
    info = describe_stored_values(loaded)

    assert loaded.dtype == np.uint16
    assert np.array_equal(loaded, image)
    assert loaded.max() == 15
    assert info.is_8bit_in_16bit
    assert info.used_bits == 4
    assert "not scaled to 0–65535" in info.summary()


def test_describe_stored_values_detects_true_16bit_range():
    image = np.array([[0, 40000]], dtype=np.uint16)
    info = describe_stored_values(image)

    assert not info.is_8bit_in_16bit
    assert info.container_bits == 16
    assert info.used_bits == 16


def test_raw_stats_in_circular_roi_use_stored_samples():
    image = np.zeros((11, 11), dtype=np.uint16)
    image[circular_mask(image.shape, (5, 5), 2)] = 200
    stats = raw_stats_in_circular_roi(image, (5, 5), 2)

    assert stats.mean == 200
    assert stats.min_value == 200
    assert stats.max_value == 200
    assert stats.std == 0
    assert stats.count > 1


def test_to_log_scale_keeps_dim_samples_visible():
    image = np.zeros((8, 8), dtype=np.uint16)
    image[1, 1] = 65535
    image[6, 6] = 1000

    log_image = to_log_scale(image)
    linear_dim = 1000 / 65535 * 255
    log_dim = log_image[6, 6] / 65535 * 255

    assert log_image.dtype == np.uint16
    assert log_image[1, 1] > log_image[6, 6] > 0
    assert log_dim > 10 * linear_dim
