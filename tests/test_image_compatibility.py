"""Image I/O and analysis contracts, using generated data only."""

import numpy as np
import pandas as pd
import pytest
from skimage import io
from skimage.color import rgb2gray

from qal import RetAnalyzer, WellAnalyzer
from qal.util import load_grayscale_image, to_grayscale, to_log_scale, describe_stored_values


@pytest.mark.parametrize("extension,dtype", [
    ("tiff", np.uint8), ("tiff", np.uint16),
    ("png", np.uint8), ("png", np.uint16), ("bmp", np.uint8),
])
def test_lossless_grayscale_files_preserve_samples_through_analysis(tmp_path, extension, dtype):
    image = (np.arange(121).reshape(11, 11) * (400 if dtype == np.uint16 else 2)).astype(dtype)
    path = tmp_path / f"samples.{extension}"
    io.imsave(path, image, check_contrast=False)
    loaded = load_grayscale_image(path)
    np.testing.assert_array_equal(loaded, image)
    assert loaded.dtype == dtype
    wells = pd.DataFrame([{"x": 5., "y": 5., "ROI Diameter": 4., "well": "signal"}])
    yy, xx = np.indices(image.shape)
    values = image[(xx - 5)**2 + (yy - 5)**2 <= 4].astype(float)
    stats = WellAnalyzer(loaded, wells.copy()).get_raw_stats(region_of_well_to_analyze=1)
    assert stats.loc[0, "mean intensity"] == pytest.approx(values.mean())
    assert stats.loc[0, "standard deviation"] == pytest.approx(values.std())
    ret = RetAnalyzer().analyze(loaded, wells.copy())
    assert ret.ok
    assert ret.stats_df.loc[0, "mean intensity"] == float(image[5, 5])


@pytest.mark.parametrize("extension", ["tiff", "png", "bmp"])
def test_rgb_file_loading_uses_first_channel_not_luminance(tmp_path, extension):
    rgb = np.empty((12, 15, 3), dtype=np.uint8)
    rgb[...] = [17, 83, 241]
    path = tmp_path / f"color.{extension}"
    io.imsave(path, rgb, check_contrast=False)
    loaded = load_grayscale_image(path)
    np.testing.assert_array_equal(loaded, rgb[..., 0])


def test_jpeg_is_accepted_without_claiming_lossless_measurements(tmp_path):
    rgb = np.full((24, 24, 3), [30, 100, 210], dtype=np.uint8)
    path = tmp_path / "preview.jpg"
    io.imsave(path, rgb, check_contrast=False)
    decoded = io.imread(path)
    np.testing.assert_array_equal(load_grayscale_image(path), decoded[..., 0])


def test_grayscale_stack_uses_first_page(tmp_path):
    import tifffile
    stack = np.arange(2 * 9 * 11, dtype=np.uint16).reshape(2, 9, 11)
    path = tmp_path / "stack.tiff"
    tifffile.imwrite(path, stack, photometric="minisblack")
    np.testing.assert_array_equal(load_grayscale_image(path), stack[0])


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32, np.float64])
@pytest.mark.parametrize("channels", [3, 4])
@pytest.mark.parametrize("method", ["average", "rgb2gray"])
def test_grayscale_conversion_matches_numeric_reference(dtype, channels, method):
    maximum = np.iinfo(dtype).max if np.issubdtype(dtype, np.integer) else 1
    rgb = (np.array([[[0.1, 0.4, 0.9], [0.8, 0.3, 0.2]]]) * maximum).astype(dtype)
    image = np.concatenate([rgb, np.zeros((1, 2, 1), dtype=dtype)], axis=-1) if channels == 4 else rgb
    original = image.copy()
    result = to_grayscale(image, method)
    expected = rgb.astype(float).mean(axis=-1) if method == "average" else rgb2gray(rgb)
    if np.issubdtype(dtype, np.integer):
        if method == "rgb2gray":
            expected *= maximum
        expected = np.rint(expected)
    assert result.dtype == dtype
    np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-7)
    np.testing.assert_array_equal(image, original)


def test_average_conversion_does_not_overflow_uint16():
    image = np.full((3, 5, 3), 65535, dtype=np.uint16)
    assert np.all(to_grayscale(image) == 65535)


def test_custom_converter_preserves_its_own_range_and_dtype():
    image = np.full((4, 6, 4), 100, dtype=np.uint16)
    result = to_grayscale(image, lambda x: x[..., 1].astype(float) / 10)
    assert result.dtype == np.float64
    assert np.all(result == 10)


@pytest.mark.parametrize("converter,error", [
    (lambda x: [], TypeError),
    (lambda x: x, ValueError),
    (lambda x: np.zeros((1, 1)), ValueError),
])
def test_custom_converter_rejects_invalid_output(converter, error):
    with pytest.raises(error):
        to_grayscale(np.zeros((4, 6, 3)), converter)


@pytest.mark.parametrize("image,error", [
    ([[1]], TypeError), (np.zeros((2, 3, 2)), ValueError),
    (np.zeros((2, 3, 4, 5)), ValueError),
])
def test_grayscale_conversion_rejects_invalid_inputs(image, error):
    with pytest.raises(error):
        to_grayscale(image)


def test_unknown_conversion_method_is_rejected():
    with pytest.raises(ValueError, match="method"):
        to_grayscale(np.zeros((3, 5, 3)), "unknown")


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
def test_log_display_is_monotonic_without_mutating_measurement_data(dtype):
    image = np.array([[0, 1, 10, 100]], dtype=dtype)
    before = image.copy()
    result = to_log_scale(image)
    assert result.dtype == np.uint16
    assert np.all(np.diff(result.astype(float)) > 0)
    np.testing.assert_array_equal(image, before)
    assert np.all(to_log_scale(np.full((4, 6), 13, dtype=dtype)) == 0)


def test_range_description_ignores_nonfinite_values():
    info = describe_stored_values(np.array([[np.nan, -4., 1000., np.inf]]))
    assert (info.vmin, info.vmax) == (-4, 1000)
    assert info.container_bits is None
    with pytest.raises(ValueError, match="finite"):
        describe_stored_values(np.array([np.nan, np.inf]))


def test_missing_file_raises_useful_io_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_grayscale_image(tmp_path / "absent.tiff")
