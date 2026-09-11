import numpy as np
import pandas as pd
import pytest

from qal import WellDetector, select_concentration_grid
from qal.rta.roi_extraction.concentration_grid import (
    complete_concentration_grid,
    prepare_concentration_grid_selection,
)
from qal.util import circular_mask, lookup_concentration_preset


WELL_IDS = [
    "1000 nM",
    "300 nM",
    "100 nM",
    "60 nM",
    "30 nM",
    "10 nM",
    "3 nM",
    "1 nM",
    "Control",
]


def test_prepare_live_preview_uses_manual_coordinates_and_completes_grid():
    selection = prepare_concentration_grid_selection(
        WELL_IDS, mode="manual_live_preview", anchor_count=3
    )

    assert selection.coordinate_mode == "manual"
    assert selection.enable_live_grid_preview is True
    assert selection.allow_control_click is False
    assert selection.complete_grid is True
    assert selection.selection_well_ids == ("1000 nM", "300 nM", "100 nM")
    kwargs = selection.select_wells_gui_kwargs()
    assert kwargs["expected_count"] == 3
    assert kwargs["max_total_count"] is None
    assert "Press Enter/Return after all selections." in selection.instruction_lines()


def test_prepare_automatic_refinement_uses_detect_mode():
    selection = prepare_concentration_grid_selection(
        WELL_IDS, mode="automatic_refinement", anchor_count=4
    )

    assert selection.coordinate_mode == "detect"
    assert selection.enable_live_grid_preview is False
    assert selection.complete_grid is True
    assert selection.selection_well_ids == (
        "1000 nM",
        "300 nM",
        "100 nM",
        "60 nM",
    )


def test_prepare_manual_coordinates_preserves_clicks_and_allows_control():
    selection = prepare_concentration_grid_selection(
        WELL_IDS, mode="manual_coordinates", anchor_count=None
    )

    assert selection.expected_count is None
    assert selection.allow_control_click is True
    assert selection.assume_last_control is True
    assert selection.max_total_count == 9
    assert selection.complete_grid is False
    assert selection.selection_well_ids == tuple(WELL_IDS[:-1])


@pytest.mark.parametrize(
    ("mode", "anchor_count", "match"),
    [
        ("not_a_mode", 3, "Unknown MODE"),
        ("manual_live_preview", 2, "ANCHOR_COUNT must be between 3 and 9"),
        ("automatic_refinement", None, "ANCHOR_COUNT=None"),
    ],
)
def test_prepare_rejects_invalid_knobs(mode, anchor_count, match):
    with pytest.raises(ValueError, match=match):
        prepare_concentration_grid_selection(
            WELL_IDS, mode=mode, anchor_count=anchor_count
        )


def test_complete_manual_coordinates_returns_the_same_frame():
    anchors = pd.DataFrame(
        {
            "well": ["1000 nM", "300 nM", "Control"],
            "x": [10.0, 30.0, 12.0],
            "y": [10.0, 10.0, 40.0],
            "ROI Diameter": [8.0, 8.0, 8.0],
        }
    )

    wells = complete_concentration_grid(
        np.zeros((50, 50)),
        anchors,
        well_ids=WELL_IDS,
        mode="manual_coordinates",
    )

    assert wells is anchors


def test_complete_live_preview_infers_remaining_grid():
    centers = [(20, 20), (50, 20), (80, 20)]
    image = np.zeros((110, 110), dtype=np.float64)
    for x, y in centers:
        image[circular_mask(image.shape, (x, y), 8)] = 80
    anchors = pd.DataFrame(
        {
            "well": ["1000 nM", "300 nM", "100 nM"],
            "x": [20.0, 50.0, 80.0],
            "y": [20.0, 20.0, 20.0],
            "ROI Diameter": [16.0, 16.0, 16.0],
            "ROI Radius": [8.0, 8.0, 8.0],
        }
    )

    wells = complete_concentration_grid(
        image,
        anchors,
        well_ids=WELL_IDS,
        mode="manual_live_preview",
        detector=WellDetector(parallel_processing=False),
    )

    assert list(wells["well"]) == WELL_IDS
    assert len(wells) == 9
    assert np.allclose(
        wells.iloc[0][["x", "y"]].astype(float), (20, 20), atol=1.5
    )
    assert np.allclose(
        wells.iloc[2][["x", "y"]].astype(float), (80, 20), atol=1.5
    )


def test_lookup_concentration_preset_requires_well_ids():
    preset = lookup_concentration_preset("RCS-S-IC1-S1-A-000")

    assert preset["family_key"] == "concentration"
    assert preset["well_ids"][-1] == "Control"


def test_lookup_concentration_preset_rejects_other_families():
    with pytest.raises(ValueError, match="No product preset matches"):
        lookup_concentration_preset("RRT-70Q-ST01-QUEL01")


def test_select_concentration_grid_is_public():
    assert callable(select_concentration_grid)
