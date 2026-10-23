import numpy as np
import pandas as pd
import pytest
from unittest.mock import Mock, patch

from sklearn.cluster import AgglomerativeClustering

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


def _roi_candidates(radii):
    return pd.DataFrame({
        "x": np.arange(len(radii)) * 300.,
        "y": np.full(len(radii), 20.),
        "ROI Radius": radii,
        "ROI Diameter": np.asarray(radii) * 2,
    })


@pytest.mark.parametrize("radii,expected_indices", [
    ([10., 10., 10., 40.], [0, 1, 2]),  # Three survivors: keep fixed filter.
    ([10., 40., 70., 100.], [0, 1, 2, 3]),  # Two survivors: use std fallback.
    ([10., 100.], [0, 1]),  # No survivors: recover from original candidates.
])
def test_roi_filter_fallback_clusters_once(radii, expected_indices):
    candidates = _roi_candidates(radii)
    detector = WellDetector(parallel_processing=False)
    original_fit = AgglomerativeClustering.fit_predict
    with patch.object(AgglomerativeClustering, "fit_predict", autospec=True,
                      side_effect=original_fit) as fit:
        selected = detector.get_clusters_from_points(candidates, fallback_to_std=True)

    assert fit.call_count == 1
    assert selected.index.tolist() == expected_indices
    # Filtering reuses the original cluster labels and source coordinates.
    np.testing.assert_array_equal(selected["cluster"], candidates.loc[expected_indices, "cluster"])
    np.testing.assert_array_equal(selected[["x", "y"]], candidates.loc[expected_indices, ["x", "y"]])


def test_explicit_roi_filters_retain_their_previous_behavior():
    detector = WellDetector(parallel_processing=False)
    candidates = _roi_candidates([10., 40., 70., 100.])
    fixed = detector.get_clusters_from_points(candidates.copy())
    std = detector.get_clusters_from_points(candidates.copy(), use_std=True)

    assert fixed.index.tolist() == [1, 2]
    assert std.index.tolist() == [0, 1, 2, 3]


@pytest.mark.parametrize("radii,expected_indices", [
    ([10., 10., 10., 40.], [0, 1, 2]),
    ([10., 40., 70., 100.], [0, 1, 2, 3]),
])
def test_detect_wells_calls_roi_clustering_once(monkeypatch, radii, expected_indices):
    detector = WellDetector(parallel_processing=False)
    candidates = _roi_candidates(radii)
    # Feed deterministic threshold candidates into the real clustering/filter
    # stage, then capture exactly which candidates reach well assignment.
    monkeypatch.setattr(detector, "get_thresh_rolling_window_binary", lambda image: [])
    monkeypatch.setattr(detector, "get_detected_features", lambda *args, **kwargs: candidates)
    cluster = Mock(wraps=detector.get_clusters_from_points)
    monkeypatch.setattr(detector, "get_clusters_from_points", cluster)

    def assign_wells(image, selected, consistent, wells):
        assert selected.index.tolist() == expected_indices
        return selected.assign(mean_intensity=1.), 20.

    monkeypatch.setattr(detector, "get_cluster_id", assign_wells)
    selected = detector.detect_wells(np.arange(100).reshape(10, 10))

    assert selected is not None
    assert selected.index.tolist() == expected_indices
    assert cluster.call_count == 1
