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


@pytest.mark.parametrize("bright_row", [10, 50])
@pytest.mark.parametrize("kwargs,channel_row", [
    ({}, 40),  # The default selects the center, even when another row is brighter.
    ({"channel_distance_from_top": .75}, 60),
])
def test_issue_39_control_row_is_not_replaced_by_brightest_row(bright_row, kwargs, channel_row):
    image = np.full((80, 120), 5.)
    image[bright_row] += np.linspace(500, 0, 120)
    image[channel_row] = np.linspace(8, 5, 120)
    control = depth_analyzer(image)
    control.get_profiles(**kwargs)
    np.testing.assert_array_equal(control.outputs["Intensity profile"], image[channel_row])
    assert control.outputs["Peak max"] == 8
    vertical = control.outputs["Vertical distance"]
    assert vertical[channel_row] == 0
    assert vertical[channel_row - 1] == pytest.approx(-30 / 80)
    assert vertical[channel_row + 1] == pytest.approx(30 / 80)


@pytest.mark.parametrize("fraction,row", [(0, 0), (.25, 15), (.5, 30), (.999, 60), (1, 60)])
def test_issue_39_channel_fraction_boundaries_and_rounding(fraction, row):
    # Odd height exercises row rounding; fractions near 1 must stay in bounds.
    image = np.repeat(np.arange(61, dtype=float)[:, None], 100, axis=1)
    analyzer = depth_analyzer(image)
    analyzer.get_profiles(channel_distance_from_top=fraction)
    np.testing.assert_array_equal(analyzer.outputs["Intensity profile"], image[row])
    assert analyzer.outputs["Vertical distance"][row] == 0


@pytest.mark.parametrize("fraction", [
    -.01, 1.01, np.nan, np.inf, -np.inf, True, False, np.bool_(True), None, "0.5", 1j,
])
def test_issue_39_invalid_channel_fraction_is_rejected(fraction):
    with pytest.raises(ValueError, match="channel_distance_from_top"):
        depth_analyzer(np.ones((60, 100))).get_profiles(channel_distance_from_top=fraction)


@pytest.mark.parametrize("rotate", [False, True])
@pytest.mark.parametrize("height,width,channel_row", [(80, 120, 60), (120, 80, 90)])
def test_issue_39_channel_selection_uses_cropped_and_explicitly_rotated_coordinates(
    rotate, height, width, channel_row
):
    expected = np.full((height, width), 5.)
    expected[10] += np.linspace(500, 0, width)
    expected[channel_row] = np.linspace(8, 5, width)
    source = expected.T if rotate else expected
    image = np.pad(source, ((8, 11), (12, 6)), constant_values=10000)
    original = image.copy()
    analyzer = depth_analyzer(image)
    analyzer.borders = {"top": 8, "bottom": 8 + source.shape[0],
                        "left": 12, "right": 12 + source.shape[1]}
    analyzer.get_profiles(rotate_image_90=rotate, channel_distance_from_top=.75)
    np.testing.assert_array_equal(analyzer.outputs["Intensity profile"], expected[channel_row])
    assert len(analyzer.outputs["Vertical distance"]) == height
    assert analyzer.outputs["Vertical distance"][channel_row] == 0
    np.testing.assert_array_equal(image, original)


@pytest.mark.parametrize("flip", [False, True])
def test_issue_39_explicit_orientation_does_not_depend_on_control_brightness(flip):
    image = np.tile(np.arange(100), (60, 1)).astype(float)
    # A brighter row points the other way; neither row can override the caller.
    image[5] += np.linspace(10000, 0, 100)
    original = image.copy()
    analyzer = depth_analyzer(image)
    analyzer.get_profiles(channel_distance_from_top=.5, flip_horizontal=flip)
    expected = np.fliplr(image) if flip else image
    np.testing.assert_array_equal(analyzer.outputs["Intensity profile"], expected[30])
    # Reorient the whole crop so transverse and longitudinal profiles agree.
    np.testing.assert_array_equal(
        analyzer.outputs["Spreads"]["1 mm depth"]["Spread profile"], expected[:, 16]
    )
    np.testing.assert_array_equal(image, original)


@pytest.mark.parametrize("reverse", [False, True])
def test_issue_39_automatic_horizontal_orientation_is_preserved(reverse):
    profile = np.linspace(500, 0, 100)
    image = np.tile(profile[::-1] if reverse else profile, (60, 1))
    analyzer = depth_analyzer(image)
    analyzer.get_profiles()
    np.testing.assert_array_equal(analyzer.outputs["Intensity profile"], profile)


@pytest.mark.parametrize("flip", [0, 1, "false", np.nan])
def test_issue_39_invalid_horizontal_flip_is_rejected(flip):
    with pytest.raises(ValueError, match="flip_horizontal"):
        depth_analyzer(np.ones((60, 100))).get_profiles(flip_horizontal=flip)
