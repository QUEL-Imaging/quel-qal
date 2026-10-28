import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.patches import Circle

from qal import WellPlotter


@pytest.fixture
def roi_data():
    image = np.arange(64 * 64).reshape(64, 64).astype(np.uint16)
    wells = pd.DataFrame([
        {"well": "1000 nM", "x": 20., "y": 20., "ROI Diameter": 16.},
        {"well": "Control", "x": 44., "y": 44., "ROI Diameter": 16.},
    ])
    return image, wells


def assert_roi_png_has_content(path):
    # Inspect decoded RGB pixels: file size alone cannot distinguish a blank PNG.
    rgb = plt.imread(path)[..., :3]
    assert np.mean(np.max(rgb, axis=-1) < 0.8) > 0.05
    assert np.mean(np.min(rgb, axis=-1) > 0.95) > 0.05
    # Colored pixels prove that ROI overlays, not just the grayscale image, survived.
    assert np.count_nonzero(np.ptp(rgb, axis=-1) > 0.2) > 100


@pytest.mark.parametrize("close_on_show", [False, True])
def test_visualize_roi_saves_content_before_show(
    monkeypatch, tmp_path, roi_data, close_on_show
):
    image, wells = roi_data
    wells["Analyzed ROI Diameter"] = wells["ROI Diameter"] / 2
    path = tmp_path / "roi.png"

    def show():
        assert_roi_png_has_content(path)
        if close_on_show:
            # Reproduce closing the window after a blocking GUI show().
            plt.close("all")

    monkeypatch.setattr(plt, "show", show)
    try:
        fig = WellPlotter(wells, image=image).visualize_roi(save_plot=path)
        assert len(fig.axes[0].images) == 1
        assert len(fig.axes[0].patches) == 4
        # The returned figure also remains usable after the GUI window closes.
        later_path = tmp_path / "roi_after_close.png"
        fig.savefig(later_path, bbox_inches="tight")
        assert_roi_png_has_content(later_path)
        np.testing.assert_array_equal(plt.imread(path), plt.imread(later_path))
    finally:
        plt.close("all")


@pytest.mark.parametrize("example", ["gui", "batch", "notebook"])
def test_concentration_examples_save_nonblank_roi(
    monkeypatch, tmp_path, roi_data, example
):
    from qal.examples.concentration_rcs import concentration_gui_example as gui

    image, wells = roi_data
    monkeypatch.setattr(plt, "show", lambda: plt.close("all"))
    try:
        if example == "gui":
            monkeypatch.setattr(gui, "IMAGE_PATH", str(tmp_path / "sample.tiff"))
            monkeypatch.setattr(gui, "SAVE_RESULTS", True)
            monkeypatch.setattr(gui, "load_grayscale_image", lambda _: image)
            monkeypatch.setattr(gui, "select_concentration_grid", lambda *a, **kw: wells)
            gui.main()
            path = tmp_path / "sample_results" / "roi.png"
            stats_path = path.parent / "stats.csv"
        elif example == "batch":
            from qal.examples.concentration_rcs import concentration_batch_example as batch

            monkeypatch.setattr(batch, "SELECTION_PATH", None)
            monkeypatch.setattr(batch, "OUTPUT_DIR", tmp_path)
            monkeypatch.setattr(batch, "SAVE_PLOTS", True)
            monkeypatch.setattr(batch, "iter_images", lambda: iter([("sample", image)]))
            monkeypatch.setattr(batch, "select_concentration_grid", lambda *a, **kw: wells)
            batch.main()
            path = tmp_path / "sample_roi.png"
            stats_path = tmp_path / "sample_stats.csv"
        else:
            import json
            from pathlib import Path

            notebook = json.loads(Path(gui.__file__).with_suffix(".ipynb").read_text())
            cell = next(cell for cell in notebook["cells"] if any(
                "plotter.visualize_roi(" in line for line in cell.get("source", [])
            ))
            wells["Analyzed ROI Diameter"] = wells["ROI Diameter"] / 2
            exec("".join(cell["source"]), {
                "WellPlotter": WellPlotter, "stats": wells, "image": image,
                "results_dir": tmp_path, "SAVE_ROI_PLOT": "roi.png", "plt": plt,
            })
            path = tmp_path / "roi.png"
            stats_path = None

        assert_roi_png_has_content(path)
        if stats_path is not None:
            saved_stats = pd.read_csv(stats_path)
            assert len(saved_stats) == len(wells)
            assert np.isfinite(saved_stats["mean intensity"]).all()
            assert (saved_stats["standard deviation"] > 0).all()
    finally:
        plt.close("all")


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
