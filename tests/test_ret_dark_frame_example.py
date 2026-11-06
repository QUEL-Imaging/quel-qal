"""Exercise the beginner RET workflow with generated frames and real analysis."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from PIL import Image
from skimage import io

from qal.examples.radiometric_emitter_ret import ret_dark_frame_example as example
from qal.rta.roi_extraction.well_detector_gui import WellSelectionSession
from qal.util import circular_mask


@pytest.fixture
def frames(tmp_path, monkeypatch):
    image = np.full((61, 61), 10, dtype=np.uint16)
    image[circular_mask(image.shape, (30, 28), 10)] = 110
    yy, xx = np.indices(image.shape)
    dark = (5 + (xx + 2 * yy) % 7).astype(np.uint16)
    light_path = tmp_path / "light frame.tiff"
    dark_path = tmp_path / "dark frame.tiff"
    io.imsave(light_path, image, check_contrast=False)
    io.imsave(dark_path, dark, check_contrast=False)
    for name, value in {
        "IMAGE_PATH": str(light_path), "DARK_FRAME_PATH": str(dark_path),
        "MODE": "manual_coordinates", "COORDINATES": [(30, 28)],
        "MANUAL_ROI_DIAMETER": 20, "SEARCH_RADIUS": 20,
        "SAVE_RESULTS": False, "REGION_OF_WELL_TO_ANALYZE": .5,
        "CNR_THRESHOLD": 3.,
    }.items():
        monkeypatch.setattr(example, name, value)
    monkeypatch.setattr(plt, "show", lambda: plt.close("all"))
    yield image, dark, light_path.parent / "light frame_results"
    plt.close("all")


def test_fixed_coordinates_measure_matching_dark_roi_and_save_before_show(frames, monkeypatch):
    image, dark, results_dir = frames
    monkeypatch.setattr(example, "SAVE_RESULTS", True)

    def must_not_open(*args, **kwargs):
        raise AssertionError("Fixed coordinates should bypass the window")

    monkeypatch.setattr(example, "select_wells_gui", must_not_open)
    result = example.main()

    yy, xx = np.indices(image.shape)
    values = dark[(xx - 30)**2 + (yy - 28)**2 <= 25].astype(float)
    row = result.stats_df.iloc[0]
    assert result.is_baselined
    assert row["mean intensity"] == 110
    assert row["dark frame mean intensity"] == pytest.approx(values.mean())
    assert row["CNR"] == pytest.approx((110 - values.mean()) / values.std())
    assert row["Analyzed ROI Diameter"] == 10
    stats = pd.read_csv(results_dir / "stats.csv")
    assert stats.loc[0, "CNR"] == pytest.approx(row["CNR"])
    wells = pd.read_csv(results_dir / "wells.csv")
    np.testing.assert_allclose(wells[["x", "y"]], [[30, 28]])
    with Image.open(results_dir / "roi.png") as plot:
        assert np.asarray(plot.convert("RGB")).std() > 5


def test_fixed_automatic_refinement_recenters_the_click(frames, monkeypatch):
    monkeypatch.setattr(example, "MODE", "automatic_refinement")
    monkeypatch.setattr(example, "COORDINATES", [(32, 29)])
    result = example.main()
    np.testing.assert_allclose(result.roi_df[["x", "y"]], [[30, 28]], atol=.5)
    assert result.stats_df.loc[0, "mean intensity"] == 110
    assert not frames[2].exists()


@pytest.mark.parametrize("mode", ["automatic_refinement", "manual_coordinates"])
def test_gui_selection_passes_through_to_real_dark_analysis(frames, monkeypatch, mode):
    monkeypatch.setattr(example, "MODE", mode)
    monkeypatch.setattr(example, "COORDINATES", None)
    calls = []

    def select(image, **kwargs):
        calls.append(kwargs)
        return example.select_wells_from_coordinates(
            image, [(30, 28)], well_ids=["RET"],
            coordinate_mode="manual", manual_roi_diameter=20,
        )

    monkeypatch.setattr(example, "select_wells_gui", select)
    result = example.main()
    assert result.is_baselined
    assert len(calls) == 1 and calls[0]["expected_count"] == 1
    assert calls[0]["coordinate_mode"] == ("detect" if mode == "automatic_refinement" else "manual")


def test_gui_cancellation_creates_no_output(frames, monkeypatch):
    monkeypatch.setattr(example, "COORDINATES", None)
    monkeypatch.setattr(example, "SAVE_RESULTS", True)
    monkeypatch.setattr(example, "select_wells_gui", lambda *args, **kwargs: None)
    assert example.main() is None
    assert not frames[2].exists()


def test_notebook_session_can_be_confirmed_then_analyzed(frames, monkeypatch):
    image, dark, results_dir = frames
    monkeypatch.setattr(example, "COORDINATES", None)
    monkeypatch.setattr(example, "SAVE_RESULTS", True)
    session = WellSelectionSession(image, expected_count=1, well_ids=["RET"],
        coordinate_mode="manual", manual_roi_diameter=20,
        resolve_in_background=False, close_on_confirm=False)
    monkeypatch.setattr(example, "select_wells_gui", lambda *args, **kwargs: session)
    assert example.main() is session
    assert not results_dir.exists()
    session.points.append((30., 28.))
    session.confirm()
    result = example.analyze_selection(session.image, dark, session.result)
    assert result.is_baselined and (results_dir / "stats.csv").is_file()


def test_automatic_detection_preserves_brightest_well_selection(frames, monkeypatch):
    monkeypatch.setattr(example, "MODE", "automatic_detection")
    monkeypatch.setattr(example, "COORDINATES", None)
    candidates = pd.DataFrame([
        {"x": 10., "y": 10., "ROI Diameter": 20.},
        {"x": 30., "y": 28., "ROI Diameter": 20.},
    ])
    monkeypatch.setattr(example.WellDetector, "detect_wells", lambda *args, **kwargs: candidates)
    result = example.main()
    assert result.stats_df.loc[0, "mean intensity"] == 110
    assert result.roi_df.iloc[0]["x"] == 30


def test_failed_automatic_detection_explains_interactive_alternatives(frames, monkeypatch):
    monkeypatch.setattr(example, "MODE", "automatic_detection")
    monkeypatch.setattr(example, "COORDINATES", None)
    monkeypatch.setattr(example.WellDetector, "detect_wells", lambda *args, **kwargs: None)
    with pytest.raises(RuntimeError, match="automatic_refinement"):
        example.main()


@pytest.mark.parametrize("setting", ["IMAGE_PATH", "DARK_FRAME_PATH"])
def test_missing_file_identifies_the_setting_to_edit(frames, monkeypatch, setting):
    monkeypatch.setattr(example, setting, "path/to/missing_frame.tiff")
    with pytest.raises(FileNotFoundError, match=setting):
        example.main()


def test_mismatched_frame_sizes_are_rejected_before_selection(frames, monkeypatch):
    dark_path = example.DARK_FRAME_PATH
    io.imsave(dark_path, np.ones((10, 10), dtype=np.uint16), check_contrast=False)
    with pytest.raises(ValueError, match="same dimensions"):
        example.main()


def test_invalid_mode_has_a_clear_error(frames, monkeypatch):
    monkeypatch.setattr(example, "MODE", "unknown")
    with pytest.raises(ValueError, match="Choose MODE"):
        example.main()


def test_automatic_detection_does_not_silently_ignore_fixed_coordinates(frames, monkeypatch):
    monkeypatch.setattr(example, "MODE", "automatic_detection")
    with pytest.raises(ValueError, match="COORDINATES = None"):
        example.main()
