import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle

from qal import WellPlotter


def test_visualize_roi_overlays_detected_and_analyzed_circles(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    image = np.zeros((40, 40), dtype=np.uint8)
    stats = pd.DataFrame(
        [
            {
                "well": "RET",
                "x": 20.0,
                "y": 18.0,
                "ROI Diameter": 16.0,
                "ROI Radius": 8.0,
                "Analyzed ROI Diameter": 8.0,
            }
        ]
    )

    WellPlotter(stats, image=image).visualize_roi()
    ax = plt.gca()
    circles = [patch for patch in ax.patches if isinstance(patch, Circle)]
    radii = sorted(circle.get_radius() for circle in circles)
    styles = {circle.get_linestyle() for circle in circles}

    assert radii == [4.0, 8.0]
    assert "--" in styles or "dashed" in styles
    assert "-" in styles or "solid" in styles
    edge = np.asarray(circles[0].get_edgecolor())[:3]
    assert edge.max() > 0.4
    labels = [handle.get_label() for handle in ax.get_legend().legend_handles]
    assert labels == ["Detected ROI", "Analyzed ROI"]
    plt.close("all")
