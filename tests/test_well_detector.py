import numpy as np
import pandas as pd
import pytest

from qal import WellDetector
from qal.util import circular_mask


def test_estimate_remaining_wells_3x3_keeps_supplied_row_major_order():
    """A tilted top row must not be re-sorted by y before the grid fit."""
    anchors = pd.DataFrame(
        {
            "x": [20.0, 50.0, 80.0],
            "y": [22.0, 20.0, 21.0],
            "ROI Diameter": 12.0,
            "ROI Radius": 6.0,
        }
    )
    image = np.zeros((120, 120), dtype=np.float64)
    for x, y in zip(anchors["x"], anchors["y"]):
        image[circular_mask(image.shape, (x, y), 6)] = 80

    completed = WellDetector(parallel_processing=False).estimate_remaining_wells_3x3(
        image,
        anchors,
        well_ids=["A", "B", "C", "D", "E", "F", "G", "H", "I"],
    )

    assert np.hypot(completed.iloc[0]["x"] - 20, completed.iloc[0]["y"] - 22) < 2
    assert np.hypot(completed.iloc[1]["x"] - 50, completed.iloc[1]["y"] - 20) < 2
    assert np.hypot(completed.iloc[2]["x"] - 80, completed.iloc[2]["y"] - 21) < 2
    assert completed["x"].max() - completed["x"].min() == pytest.approx(60, abs=3)
    assert len(completed) == 9
