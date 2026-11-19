"""New feature contracts across image encodings and dependency versions."""
import subprocess
import sys

import cv2
import numpy as np
import pandas as pd
import pytest

from qal import RetAnalyzer, RrtAnalyzer, RrtROI, WellDetector
from qal.util import circular_mask


@pytest.mark.parametrize("dtype,scale", [(np.uint8, 1), (np.uint16, 1), (np.uint16, 200), (np.float32, .01)])
def test_local_well_refinement_is_independent_of_stored_range(dtype, scale):
    image = np.full((80, 100), 10 * scale, dtype=dtype)
    image[circular_mask(image.shape, (48, 36), 9)] = 150 * scale
    result = WellDetector(parallel_processing=False).refine_well_at_coordinate(
        image, (50, 35), search_radius=20)
    assert result["x"] == pytest.approx(48, abs=.5)
    assert result["y"] == pytest.approx(36, abs=.5)
    assert result["ROI Diameter"] == pytest.approx(18, abs=2)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
def test_ret_negative_baseline_does_not_wrap_unsigned_values(dtype):
    image = np.full((11, 11), 5, dtype=dtype)
    dark = (20 + np.indices(image.shape)[0] % 3).astype(dtype)
    well = pd.DataFrame([{"x": 5., "y": 5., "ROI Diameter": 8.}])
    stats = RetAnalyzer().analyze(image, well, dark_frame=dark).stats_df.iloc[0]
    assert stats["mean intensity baselined"] < 0
    assert stats["CNR"] < 0
    assert not stats["CNR pass"]


@pytest.mark.parametrize("dtype,scale", [(np.uint8, 1), (np.uint16, 1), (np.uint16, 257), (np.float32, 1/255)])
def test_rrt_quantized_extrema_and_contrast_are_scale_invariant(dtype, scale):
    bars = np.array([0, 0, 50, 200, 200, 200, 50, 0, 0, 50, 200, 200, 50, 0])
    profile = (bars * scale).astype(dtype)
    analyzer = RrtAnalyzer()
    peaks, troughs = analyzer.detect_peaks_and_troughs(profile)
    assert peaks == [4, 10]
    assert troughs == [7]
    # The contrast calculation expects floating profile samples, as its
    # profile extraction path supplies; casting avoids unsigned subtraction.
    assert analyzer.calculate_percentage_contrast(float(profile[4]), float(profile[7])) == 100


@pytest.mark.parametrize("name,binary", [("brisk", True), ("akaze", True), ("orb", True), ("sift", False)])
def test_installed_opencv_feature_factories_produce_expected_descriptor_type(name, binary):
    detector, norm = RrtROI.create_feature_detector(name)
    assert norm == (cv2.NORM_HAMMING if binary else cv2.NORM_L2)
    assert detector.descriptorType() == (cv2.CV_8U if binary else cv2.CV_32F)
    keypoints, descriptors = detector.detectAndCompute(np.zeros((64, 64), dtype=np.uint8), None)
    assert len(keypoints) == 0
    assert descriptors is None


@pytest.mark.parametrize("arguments,expected", [
    (["RCS-ICG-ST01-QUEL03"], "1000 nM"),
    (["ICG", "--family", "depth"], "0.5 mm"),
    (["RCS-ICG", "--all"], "icg_mini"),
])
def test_lookup_cli_entry_point(arguments, expected):
    result = subprocess.run([sys.executable, "-m", "qal.util", *arguments],
        capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert expected in result.stdout


def test_lookup_cli_reports_no_match():
    result = subprocess.run([sys.executable, "-m", "qal.util", "NO-SUCH-TARGET", "--all"],
        capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert "No product preset matches" in result.stderr
