"""Focused regressions for the open issues audited in October 2026."""
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from skimage.draw import ellipse

from qal import DepthAnalyzer, RrtROI, WellDetector, WellPlotter
from qal.util import circular_mask


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.mark.parametrize("location", ["top", "contrib", "contrib_class"])
def test_issue_43_opencv_factory_locations(monkeypatch, location):
    import qal.rta.roi_extraction.rrt_roi as module
    sentinel = object()
    create = lambda: sentinel
    namespace = SimpleNamespace(BRISK_create=create) if location != "contrib_class" else SimpleNamespace(BRISK=SimpleNamespace(create=create))
    fake = SimpleNamespace(NORM_HAMMING=6, NORM_L2=4, __version__="test")
    if location == "top":
        fake.BRISK_create = create
    else:
        fake.xfeatures2d = namespace
    monkeypatch.setattr(module, "cv2", fake)
    assert RrtROI.create_feature_detector("brisk") == (sentinel, 6)


def test_issue_43_legacy_harris_accepts_rgb_and_rgba():
    image = np.zeros((80, 90), dtype=np.uint16)
    image[15:60, 20:65] = 45000
    roi = RrtROI.__new__(RrtROI)
    points = [(20, 15), (65, 60)]
    expected = roi.detect_roi_corners(points, image)
    assert len(expected) == 2
    for channels in (3, 4):
        color = np.repeat(image[..., None], channels, axis=-1)
        np.testing.assert_array_equal(roi.detect_roi_corners(points, color), expected)


@pytest.mark.parametrize("detector", ["brisk", "akaze", "orb", "sift"])
def test_issue_43_real_template_feature_matching_accepts_uint16_template(monkeypatch, detector):
    from pathlib import Path
    import cv2
    from skimage import io
    import qal.rta.roi_extraction.rrt_roi as module

    template = io.imread(Path(__file__).resolve().parents[1] /
        "qal/data/USAF1951_template/res_source.png")
    assert template.dtype == np.uint16
    monkeypatch.setattr(module, "resolution_template", lambda: template)
    roi = RrtROI.__new__(RrtROI)
    base = roi.cvt_to_uint8(template)
    height, width = base.shape
    rotation = cv2.getRotationMatrix2D((width / 2, height / 2), 4, 1)
    source = cv2.warpAffine(base, rotation, (width, height))
    result = roi.get_resolution_target_cropped(source, detector=detector, return_debug=True)
    assert result["available"]
    assert result["n_good_matches"] >= 10
    assert result["n_inliers"] >= 4
    assert result["roi"] is not None
    assert np.isfinite(result["homography"]).all()
    # Repeated bars can produce a finite but incorrect homography. Verify
    # registration against the known inverse rotation at four interior points.
    points = np.float32([[[width * x, height * y]
                         for x, y in ((.25, .25), (.75, .25), (.75, .75), (.25, .75))]])
    projected = cv2.perspectiveTransform(points, result["homography"])
    expected = cv2.transform(points, cv2.invertAffineTransform(rotation))
    assert np.linalg.norm(projected - expected, axis=-1).max() < 2


def test_issue_36_region_filters_accept_small_elongated_well_when_requested():
    image = np.zeros((100, 100), dtype=np.uint16)
    rows, cols = ellipse(40, 45, 6, 11, shape=image.shape)
    image[rows, cols] = 500
    binary = image > 0
    detector = WellDetector(parallel_processing=False)
    assert detector.get_well_positions(image, binary).empty
    assert detector.get_well_positions(image, binary, min_area=100).empty
    accepted = detector.get_detected_features(image, [binary], min_area=100, min_eccentricity=.95)
    assert len(accepted) == 1
    assert accepted.iloc[0]["x"] == pytest.approx(45)
    assert accepted.iloc[0]["y"] == pytest.approx(40)


def test_issue_36_detect_wells_exposes_sensitivity_end_to_end():
    image = np.zeros((140, 140), dtype=np.uint16)
    centers = [(35, 35), (100, 35), (35, 100), (100, 100)]
    for point, intensity in zip(centers, [500, 1000, 2000, 4000]):
        image[circular_mask(image.shape, point, 10)] = intensity
    detector = WellDetector(parallel_processing=False)
    assert detector.detect_wells(image) is None
    selected = detector.detect_wells(image, min_area=100, min_eccentricity=.4)
    assert selected is not None and len(selected) == 4
    np.testing.assert_allclose(selected[["x", "y"]].to_numpy(), centers, atol=1)


@pytest.mark.parametrize("kwargs", [{"min_area": 0}, {"min_area": np.nan},
    {"min_eccentricity": -1}, {"min_eccentricity": 1.1}])
def test_issue_36_invalid_sensitivity_is_not_silently_swallowed(kwargs):
    with pytest.raises(ValueError):
        WellDetector(parallel_processing=False).detect_wells(np.zeros((20, 20)), **kwargs)


def test_issue_23_roi_radius_filter_preserves_inliers_and_rejects_outlier():
    radii = [10.] * 8 + [40.]
    features = pd.DataFrame({"x": np.arange(9) * 20., "y": np.arange(9) * 10.,
        "ROI Radius": radii, "ROI Diameter": np.array(radii) * 2})
    kept = WellDetector(parallel_processing=False).get_clusters_from_points(features)
    assert kept.index.tolist() == list(range(8))
    assert np.all(kept["ROI Radius"] == 10)


def test_issue_23_empty_roi_filter_is_handled():
    assert WellDetector(parallel_processing=False).get_clusters_from_points(pd.DataFrame()) is None


@pytest.mark.parametrize("use_cnr,expected", [(True, [100., 1000.]), (False, [1000., 100., 10.])])
def test_issue_20_concentration_trendline_can_include_low_cnr_wells(monkeypatch, use_cnr, expected):
    monkeypatch.setattr(plt, "show", lambda: None)
    data = pd.DataFrame({"well": ["1000 nM", "100 nM", "10 nM", "Control"],
        "value": [1000., 100., 10., 0.], "mean intensity": [100., 20., 2., 1.],
        "CNR": [10., 4., 1., 0.]})
    plotter = WellPlotter(data)
    captured = []
    monkeypatch.setattr(plotter, "add_trendline_and_annotation",
        lambda fit, x, y, xhat, **kw: captured.extend(x))
    plotter.plot(use_cnr=use_cnr)
    assert captured == expected


def test_issue_20_disabled_cnr_does_not_require_control_derived_column(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    data = pd.DataFrame({"well": ["100 nM", "10 nM"], "value": [100., 10.], "mean intensity": [20., 2.]})
    plotter = WellPlotter(data)
    plotter.plot(use_cnr=False)
    assert any(line.get_label() == "Trendline" for line in plotter.ax.lines)


def depth_analyzer(image):
    cropper = SimpleNamespace(img=image, borders={"top": 0, "bottom": image.shape[0],
        "left": 0, "right": image.shape[1]})
    return DepthAnalyzer(cropper)


def test_issue_39_explicit_control_row_is_not_replaced_by_brightest_row():
    image = np.full((60, 100), 5., dtype=float)
    image[15, :] += np.linspace(500, 0, 100)
    automatic = depth_analyzer(image)
    automatic.get_profiles()
    assert automatic.outputs["Peak max"] > 100
    control = depth_analyzer(image)
    control.get_profiles(channel_row=40, flip_horizontal=False)
    np.testing.assert_array_equal(control.outputs["Intensity profile"], image[40])
    assert control.outputs["Peak max"] == 5
    assert control.outputs["Vertical distance"][40] == 0


@pytest.mark.parametrize("row", [-1, 60, 2.5, True])
def test_issue_39_invalid_explicit_channel_is_rejected(row):
    with pytest.raises(ValueError, match="channel_row"):
        depth_analyzer(np.ones((60, 100))).get_profiles(channel_row=row)


def test_issue_39_explicit_orientation_does_not_depend_on_control_brightness():
    image = np.tile(np.arange(100), (60, 1)).astype(float)
    analyzer = depth_analyzer(image)
    analyzer.get_profiles(channel_row=20, flip_horizontal=False)
    np.testing.assert_array_equal(analyzer.outputs["Intensity profile"], image[20])
    analyzer.get_profiles(channel_row=20, flip_horizontal=True)
    np.testing.assert_array_equal(analyzer.outputs["Intensity profile"], image[20, ::-1])
