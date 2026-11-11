"""Regression coverage for shared example output and malformed identifiers."""

import gc
import importlib
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backend_bases import KeyEvent
import numpy as np
import pandas as pd
import pytest
from skimage import io
import yaml

from qal.data import _fetchers
from qal.examples._output import save_sample_sources
from qal.util import describe_target_query, find_target_presets, lookup_target_preset


@pytest.mark.parametrize("query, recognized, unknown, candidate_families", [
    ("V3-ICG-ST03-RAS007", ["version=V3", "fluorophore_code=ICG", "form_code=ST03"],
     ["RAS007"], {"depth"}),
    ("V3-IRG-SR03-RAS007", ["version=V3"], ["IRG", "SR03", "RAS007"], set()),
])
def test_malformed_serial_reports_partial_fields_without_selecting_product(
    query, recognized, unknown, candidate_families
):
    diagnostic = describe_target_query(query)
    for field in recognized + unknown:
        assert field in diagnostic
    matches = find_target_presets(query)
    assert {p["family_key"] for p in matches} == candidate_families
    with pytest.raises(ValueError, match="Ambiguous target query") as exc:
        lookup_target_preset(query)
    assert diagnostic in str(exc.value)


def test_source_export_copies_original_bytes_and_ignores_user_inputs(tmp_path, monkeypatch):
    originals = tmp_path / "cache"
    originals.mkdir()
    arrays = []
    for name, offset in [("light.tiff", 100), ("dark.tiff", 5)]:
        path = originals / name
        io.imsave(path, (np.arange(64).reshape(8, 8) + offset).astype(np.uint16), check_contrast=False)
        monkeypatch.setattr(_fetchers, "_fetch", lambda _, p=path: str(p))
        arrays.append(_fetchers._load(name))
    before = {p.name: p.read_bytes() for p in originals.iterdir()}
    out = tmp_path / "results"
    paths = save_sample_sources(out, *arrays)
    assert {p.parent for p in paths} == {out / "source_images"}
    assert {p.name: p.read_bytes() for p in paths} == before
    assert {p.name: p.read_bytes() for p in originals.iterdir()} == before
    single = save_sample_sources(tmp_path / "single", arrays[0])
    assert single == [tmp_path / "single/light.tiff"]
    # Even an existing file is user input unless selected through a built-in loader.
    assert save_sample_sources(tmp_path / "user", originals / "light.tiff", arrays[0].copy()) == []
    assert not (tmp_path / "user").exists()
    key = id(arrays[1])
    arrays.pop()
    gc.collect()
    assert key not in _fetchers._SAMPLE_ARRAYS


def test_sample_directory_export_copies_only_registered_inputs(tmp_path, monkeypatch):
    sample_dir = tmp_path / "samples"
    sample_dir.mkdir()
    (sample_dir / "one.tiff").write_bytes(b"original one")
    (sample_dir / "two.tiff").write_bytes(b"original two")
    (sample_dir / "old_results.csv").write_text("not a source image")
    monkeypatch.setattr(_fetchers, "registry", {"sample/one.tiff": "", "sample/two.tiff": ""})
    monkeypatch.setattr(_fetchers, "_fetch", lambda name: str(sample_dir / Path(name).name))
    folder = _fetchers._fetch_directory("sample")
    paths = save_sample_sources(tmp_path / "results", folder)
    assert {p.name for p in paths} == {"one.tiff", "two.tiff"}
    assert all(p.parent.name == "source_images" for p in paths)


@pytest.mark.parametrize("family", ["concentration", "depth"])
def test_sku_examples_save_query_preset_and_analysis(tmp_path, monkeypatch, family):
    folder = "concentration_rcs" if family == "concentration" else "depth_rds"
    module = importlib.import_module(f"qal.examples.{folder}.{family}_sku_gui_example")
    image = (7000 - np.arange(80 * 80)).reshape(80, 80).astype(np.uint16)
    wells = pd.DataFrame([
        {"well": "1000 nM" if family == "concentration" else "0.5 mm", "x": 20., "y": 20., "ROI Diameter": 10.},
        {"well": "Control", "x": 60., "y": 60., "ROI Diameter": 10.},
    ])
    monkeypatch.setattr(module, "__file__", str(tmp_path / "example.py"))
    monkeypatch.setattr(module, "IMAGE_PATH", image)
    monkeypatch.setattr(module, "SAVE_RESULTS", True)
    monkeypatch.setattr(module, "select_concentration_grid", lambda *a, **kw: wells)
    monkeypatch.setattr(module.WellPlotter, "plot", lambda *a, **kw: None)
    monkeypatch.setattr(plt, "show", lambda: plt.close("all"))
    module.main()
    out = tmp_path / "example_results"
    metadata = yaml.safe_load((out / "target.yaml").read_text())
    assert metadata["query"] == module.TARGET_SERIAL
    assert metadata["preset"]["family_key"] == family
    assert len(pd.read_csv(out / "stats.csv")) == 2
    assert plt.imread(out / "roi.png").std() > .01


def test_depth_crop_confirmation_accepts_enter_and_keeps_other_keys_open(monkeypatch):
    from qal import PhantomCropper

    def show(**kwargs):
        fig = plt.gcf()
        fig.canvas.callbacks.process("key_press_event", KeyEvent("key_press_event", fig.canvas, key="a"))
        assert plt.fignum_exists(fig.number)
        fig.canvas.callbacks.process("key_press_event", KeyEvent("key_press_event", fig.canvas, key="enter"))
        assert not plt.fignum_exists(fig.number)

    monkeypatch.setattr(plt, "show", show)
    fig = plt.figure()
    PhantomCropper._confirm_view(fig)


def test_rrt_roi_export_uses_source_coordinates_without_opening_window(tmp_path, monkeypatch):
    from qal import RrtDataPlotter

    def no_window(*args, **kwargs):
        raise AssertionError("ROI-only export should not open a window")

    plotter = RrtDataPlotter()
    monkeypatch.setattr(plt, "show", no_window)
    monkeypatch.setattr(plt, "figure", no_window)
    path = tmp_path / "roi.png"
    coords = {0: {"coordinates": ((12., 16.), (25., 39.))}}
    fig = plotter.save_roi_image(np.arange(64 * 64).reshape(64, 64), coords, path)
    np.testing.assert_array_equal(fig.axes[0].lines[0].get_xydata(), [[12., 16.], [25., 39.]])
    assert plt.imread(path).std() > .01


@pytest.mark.parametrize("with_dark", [True, False])
def test_automatic_ret_entry_saves_real_analysis_without_gui(tmp_path, monkeypatch, with_dark):
    from qal.examples.radiometric_emitter_ret import ret_gui_example as gui
    from qal.examples.radiometric_emitter_ret import ret_light_frame_automatic_analysis_example as automatic
    from qal.util import circular_mask

    image = np.full((61, 61), 10, dtype=np.uint16)
    image[circular_mask(image.shape, (30, 28), 10)] = 110
    dark = (5 + np.indices(image.shape).sum(axis=0) % 7).astype(np.uint16)
    monkeypatch.setattr(automatic, "IMAGE_PATH", image)
    monkeypatch.setattr(automatic, "DARK_FRAME_PATH", dark if with_dark else None)
    monkeypatch.setattr(automatic, "__file__", str(tmp_path / "automatic.py"))
    monkeypatch.setattr(automatic, "SAVE_RESULTS", True)
    monkeypatch.setattr(gui.WellDetector, "detect_wells", lambda *a, **kw: pd.DataFrame([
        {"x": 30., "y": 28., "ROI Diameter": 20.},
    ]))
    monkeypatch.setattr(gui, "select_wells_gui", lambda *a, **kw: pytest.fail("Automatic analysis opened a picker"))
    monkeypatch.setattr(plt, "show", lambda: plt.close("all"))
    original_mode = gui.MODE
    result = automatic.main()
    assert gui.MODE == original_mode
    assert result.is_baselined == with_dark
    assert result.stats_df.loc[0, "mean intensity"] == 110
    saved = pd.read_csv(tmp_path / "automatic_results/stats.csv")
    assert ("CNR" in saved) == with_dark
    assert plt.imread(tmp_path / "automatic_results/roi.png").std() > .01


@pytest.mark.parametrize("query, expected_status, expected_families", [
    ("RCS-ICG-ST01-QUEL03", 0, {"concentration"}),
    ("V3-ICG-ST03-RAS007", 0, {"depth"}),
    ("V3-IRG-SR03-RAS007", 1, set()),
])
def test_lookup_example_saves_query_and_partial_diagnostic(tmp_path, query, expected_status, expected_families):
    import subprocess
    import sys

    run = subprocess.run([
        sys.executable, "-m", "qal.examples.general.target_preset_lookup_example",
        query, "--all", "--output-dir", str(tmp_path),
    ], capture_output=True, text=True)
    assert run.returncode == expected_status, run.stderr
    payload = yaml.safe_load((tmp_path / "target_lookup.yaml").read_text())
    assert payload["query"] == query
    assert {p["family_key"] for p in payload["matches"]} == expected_families
    if "RAS" in query:
        assert "version=V3" in payload["diagnostic"]
        assert payload["diagnostic"] in run.stdout
    else:
        assert payload["diagnostic"] is None
