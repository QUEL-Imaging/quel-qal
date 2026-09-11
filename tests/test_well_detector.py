import numpy as np
import pandas as pd
import pytest

from qal import WellDetector
from qal.util import circular_mask


def test_estimate_remaining_wells_3x3_fits_spatial_order_not_table_order():
    positions = [
        (100.0, 100.0), (200.0, 100.0), (300.0, 100.0),
        (100.0, 200.0), (200.0, 200.0), (300.0, 200.0),
        (100.0, 300.0), (200.0, 300.0), (300.0, 300.0),
    ]
    # Intensity-like table order that swaps the bottom-left and bottom-center wells.
    table_order = [0, 1, 2, 3, 4, 5, 7, 6, 8]
    wells = pd.DataFrame(
        {
            "x": [positions[index][0] for index in table_order],
            "y": [positions[index][1] for index in table_order],
            "ROI Diameter": 40.0,
            "ROI Radius": 20.0,
        }
    )
    image = np.zeros((400, 400), dtype=np.float64)
    for x, y in positions:
        image[circular_mask(image.shape, (x, y), 12)] = 80

    completed = WellDetector(parallel_processing=False).estimate_remaining_wells_3x3(
        image,
        wells,
        well_ids=["A", "B", "C", "D", "E", "F", "G", "H", "I"],
    )

    assert np.hypot(completed.iloc[0]["x"] - 100, completed.iloc[0]["y"] - 100) < 2
    assert np.hypot(completed.iloc[6]["x"] - 100, completed.iloc[6]["y"] - 300) < 2
    assert np.hypot(completed.iloc[7]["x"] - 200, completed.iloc[7]["y"] - 300) < 2
    assert np.hypot(completed.iloc[8]["x"] - 300, completed.iloc[8]["y"] - 300) < 2
    assert completed["x"].max() - completed["x"].min() == pytest.approx(200)
    assert completed["y"].max() - completed["y"].min() == pytest.approx(200)
