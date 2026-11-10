"""Run the RET notebook's cells with real analysis and a headless picker."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from skimage import io

from qal import data, select_wells_from_coordinates
from qal.util import circular_mask


NOTEBOOK = Path(__file__).resolve().parents[1] / (
    "qal/examples/jupyter-notebook/radiometric_emitter_ret_gui_example.ipynb"
)


@pytest.fixture
def notebook(tmp_path, monkeypatch):
    image = np.full((61, 61), 10, dtype=np.uint16)
    image[circular_mask(image.shape, (30, 28), 10)] = 110
    yy, xx = np.indices(image.shape)
    dark = (5 + (xx + 2 * yy) % 7).astype(np.uint16)
    monkeypatch.setattr(data, "ret_sample_1_lf", lambda: image)
    monkeypatch.setattr(data, "ret_sample_1_df", lambda: dark)
    monkeypatch.setattr(plt, "show", lambda: plt.close("all"))
    monkeypatch.chdir(tmp_path)
    cells = {
        cell["id"]: "".join(cell["source"])
        for cell in json.loads(NOTEBOOK.read_text())["cells"]
        if cell["cell_type"] == "code"
    }
    namespace = {}
    for cell_id in ("setup", "image-options", "saving-options", "analysis-options"):
        exec(compile(cells[cell_id], f"{NOTEBOOK.name}:{cell_id}", "exec"), namespace)

    backends = []
    namespace["use_native_window_backend"] = lambda: backends.append("native")
    namespace["use_inline_plots"] = lambda: backends.append("inline")

    def select(image, **kwargs):
        assert kwargs["expected_count"] == 1
        assert kwargs["enable_live_grid_preview"]
        return select_wells_from_coordinates(
            image, [(30, 28)], well_ids=kwargs["well_ids"],
            detector=kwargs["detector"], coordinate_mode=kwargs["coordinate_mode"],
            manual_roi_diameter=kwargs["manual_roi_diameter"], search_radius=20,
        )

    namespace["select_wells_gui"] = select
    namespace["MANUAL_ROI_DIAMETER"] = 20

    def run(*cell_ids):
        for cell_id in cell_ids:
            exec(compile(cells[cell_id], f"{NOTEBOOK.name}:{cell_id}", "exec"), namespace)

    yield namespace, run, backends
    plt.close("all")


@pytest.mark.parametrize("input_kind, with_dark", [
    ("sample", True), ("file", True), ("sample", False), ("file", False),
])
def test_notebook_analyzes_and_saves_raw_or_dark_frame_results(notebook, tmp_path, input_kind, with_dark):
    ns, run, backends = notebook
    if input_kind == "file":
        # Exercise custom image paths and manual coordinates as well as the sample.
        light_path = tmp_path / "light.tiff"
        dark_path = tmp_path / "dark.tiff"
        io.imsave(light_path, ns["IMAGE_PATH"], check_contrast=False)
        io.imsave(dark_path, ns["DARK_FRAME_PATH"], check_contrast=False)
        ns.update(IMAGE_PATH=light_path, DARK_FRAME_PATH=dark_path, MODE="manual_coordinates")
    if not with_dark:
        ns["DARK_FRAME_PATH"] = None
    loaded_inputs = []
    load = ns["load_grayscale_image"]

    def track_load(image_input):
        loaded_inputs.append(image_input)
        return load(image_input)

    ns["load_grayscale_image"] = track_load
    ns["SAVE_RESULTS"] = True
    run("load-image", "select-well", "analyze", "plot-roi", "save-csv")

    assert backends == ["native", "inline"]
    assert len(loaded_inputs) == (2 if with_dark else 1)
    assert ns["result"].is_baselined == with_dark
    assert ns["stats"].loc[0, "mean intensity"] == 110
    if with_dark:
        assert np.isfinite(ns["stats"].loc[0, "CNR"])
    else:
        assert ns["dark_frame"] is None
        assert "CNR" not in ns["stats"]
        assert "mean intensity baselined" not in ns["stats"]
    results_dir = tmp_path / ("light_results" if input_kind == "file" else "ret_sample_1_lf_results")
    saved = pd.read_csv(results_dir / "stats.csv")
    assert saved.loc[0, "mean intensity"] == 110
    assert ("CNR" in saved) == with_dark
    assert len(pd.read_csv(results_dir / "wells.csv")) == 1
    assert plt.imread(results_dir / "roi.png").std() > .01


def test_notebook_cancellation_restores_inline_and_writes_nothing(notebook, tmp_path):
    ns, run, backends = notebook
    ns["SAVE_RESULTS"] = True
    ns["select_wells_gui"] = lambda *args, **kwargs: None
    run("load-image", "select-well")
    with pytest.raises(RuntimeError, match="No RET well selected"):
        run("analyze")
    assert backends == ["native", "inline"]
    assert not (tmp_path / "ret_sample_1_lf_results").exists()


def test_notebook_rejects_mismatched_dark_frame_before_picker(notebook):
    ns, run, backends = notebook
    ns["DARK_FRAME_PATH"] = np.zeros((10, 10), dtype=np.uint16)
    with pytest.raises(ValueError, match="same dimensions"):
        run("load-image")
    assert backends == []
