import numpy as np
import pandas as pd
import pytest

from qal import WellAnalyzer
from qal.util import circular_mask


def _wells(*rows):
    return pd.DataFrame(list(rows))


def test_get_raw_stats_uses_stored_8bit_values_in_uint16_image(capsys):
    image = np.zeros((21, 21), dtype=np.uint16)
    image[circular_mask(image.shape, (10, 10), 4)] = 180
    wells = _wells(
        {
            "x": 10.0,
            "y": 10.0,
            "ROI Diameter": 8.0,
            "well": "180 nM",
        }
    )

    analyzer = WellAnalyzer(image, wells)
    stats = analyzer.get_raw_stats(region_of_well_to_analyze=1.0)
    row = stats.iloc[0]

    assert analyzer.stored_values.is_8bit_in_16bit
    assert row["mean intensity"] == pytest.approx(180)
    assert row["min intensity"] == 180
    assert row["max intensity"] == 180
    assert "CNR" not in stats.columns
    assert "mean intensity baselined" not in stats.columns

    printed = analyzer.print_stats()
    assert printed is stats
    captured = capsys.readouterr()
    assert "Raw ROI values" in captured.out
    assert "CNR were skipped" in captured.out


def test_get_stats_adds_derived_metrics_only_with_control(capsys):
    image = np.zeros((21, 21), dtype=np.uint16)
    image[circular_mask(image.shape, (6, 6), 3)] = 40
    control_mask = circular_mask(image.shape, (14, 14), 3)
    image[control_mask] = 10
    image[14, 14] = 12
    wells = _wells(
        {"x": 6.0, "y": 6.0, "ROI Diameter": 6.0, "well": "40 nM"},
        {"x": 14.0, "y": 14.0, "ROI Diameter": 6.0, "well": "Control"},
    )

    analyzer = WellAnalyzer(image, wells)
    assert analyzer.stored_values.is_8bit_in_16bit
    stats = analyzer.get_stats(region_of_well_to_analyze=1.0)

    signal = stats.loc[stats["well"] == "40 nM"].iloc[0]
    control = stats.loc[stats["well"] == "Control"].iloc[0]
    assert signal["min intensity"] == 40
    assert signal["max intensity"] == 40
    assert signal["mean intensity baselined"] == pytest.approx(
        signal["mean intensity"] - control["mean intensity"]
    )
    assert signal["CNR"] == pytest.approx(
        signal["mean intensity baselined"] / control["standard deviation"]
    )

    analyzer.print_stats(mode="manual_coordinates")
    captured = capsys.readouterr()
    assert "Well statistics:" in captured.out
    assert "explicit Control was selected" in captured.out
