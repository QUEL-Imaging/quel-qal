"""Run the saveable examples and check that they write real files.

Enable SAVE_RESULTS (or the example's save flag), assert every expected
output exists and is not empty / blank, then delete the results folder.

GUI pickers are skipped with stored or detected coordinates. Examples that
need mouse clicks (depth-resolution cropping) or do not save files are
omitted.

    pytest tests/test_example_saved_outputs.py -q
"""

from __future__ import annotations

import hashlib
import importlib
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from skimage import io

from qal import WellDetector
from qal.data import (
    cn_sample_3,
    depth_sample_1,
    get_concentration_target_preset,
    get_target_preset,
    res_sample_1,
    rud_example_1,
    rud_example_2,
    rud_example_3,
)
from qal.util import circular_mask


RESOLUTION_LINES = [
    ((64.1, 71.2), (64.1, 220.3)),
    ((199.0, 73.8), (199.0, 169.4)),
    ((131.9, 125.0), (131.9, 164.0)),
]


def _write_tiff(directory: Path, name: str, image: np.ndarray) -> Path:
    path = directory / f"{name}.tiff"
    io.imsave(path, image, check_contrast=False)
    return path


def _copy_tree(source: str | Path, destination: Path) -> Path:
    shutil.copytree(source, destination, dirs_exist_ok=True)
    return destination


def _grid_anchors(image: np.ndarray, family: str, count: int = 3):
    well_ids = (
        get_concentration_target_preset("icg")["well_ids"]
        if family == "concentration"
        else get_target_preset("depth", "icg")["well_ids"]
    )
    detector = WellDetector(parallel_processing=False)
    wells = detector.detect_wells(
        image,
        show_detected_wells=False,
        set_consistent_roi_region=True,
    )
    wells = detector.estimate_remaining_wells_3x3(
        image, wells, well_ids=well_ids
    )
    points = [
        (float(x), float(y))
        for x, y in wells[["x", "y"]].to_numpy()[:count]
    ]
    if len(points) < count:
        raise AssertionError(
            f"Expected {count} {family} anchors, found {len(points)}"
        )
    return points


def _assert_image_not_blank(path: Path) -> None:
    array = np.asarray(plt.imread(path), dtype=float)
    if array.ndim == 3:
        array = array[..., :3]
    assert array.size > 0, f"{path} is empty"
    assert np.isfinite(array).any(), f"{path} has no finite pixels"
    assert np.nanstd(array) > 1e-3, f"{path} is a blank image"


def _assert_table_has_data(path: Path) -> None:
    if path.suffix.lower() == ".csv":
        table = pd.read_csv(path)
    else:
        table = pd.read_excel(path)
    assert not table.empty, f"{path} has no rows"
    assert table.shape[1] > 0, f"{path} has no columns"


def _assert_saved_file(path: Path) -> None:
    assert path.is_file(), f"missing {path}"
    assert path.stat().st_size > 0, f"{path} is 0 bytes"
    suffix = path.suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".gif"}:
        _assert_image_not_blank(path)
    elif suffix in {".csv", ".xlsx", ".xls"}:
        _assert_table_has_data(path)
    elif suffix == ".json":
        payload = json.loads(path.read_text())
        assert payload, f"{path} is empty JSON"
    elif suffix in {".yaml", ".yml"}:
        assert path.read_text().strip(), f"{path} is empty"
    elif suffix in {".pkl", ".pickle"}:
        assert path.stat().st_size > 16, f"{path} is too small to hold data"


def _assert_results(results_dir: Path, expected_names: list[str]) -> None:
    assert results_dir.is_dir(), f"missing results folder {results_dir}"
    for name in expected_names:
        matches = [path for path in results_dir.glob(name) if path.is_file()]
        assert matches, f"{results_dir} is missing {name}"
        for path in matches:
            if path.suffix:
                _assert_saved_file(path)
            else:
                assert path.stat().st_size > 0, f"{path} is 0 bytes"
                try:
                    _assert_image_not_blank(path)
                except Exception:
                    pass


def _run_checked(module, results_dir: Path, expected: list[str]) -> None:
    if results_dir.exists():
        shutil.rmtree(results_dir)
    try:
        module.main()
        _assert_results(results_dir, expected)
    finally:
        plt.close("all")
        if results_dir.exists():
            shutil.rmtree(results_dir)


@pytest.fixture
def checkout_sample_data(monkeypatch):
    """Use verified Git LFS files from this checkout without HTTP downloads."""
    from qal.data import _fetchers

    data_dir = Path(_fetchers.__file__).resolve().parent

    def local_fetch(filename):
        path = data_dir / filename
        assert path.is_file(), f"Missing sample data: {path}; run git lfs pull"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == _fetchers.registry[filename], (
            f"Sample data hash mismatch: {path}; run git lfs pull"
        )
        return str(path)

    monkeypatch.setattr(_fetchers, "_fetch", local_fetch)


@pytest.fixture
def no_gui(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: plt.close("all"))
    yield
    plt.close("all")


def test_all_saveable_examples_write_nonblank_files(
    tmp_path, monkeypatch, no_gui, checkout_sample_data
):
    """One pass through every example that can save without a mouse click."""
    failures = []

    def check(module_name, results_dir, expected):
        module = importlib.import_module(module_name)
        try:
            _run_checked(module, results_dir, expected)
        except Exception as exc:
            failures.append(f"{module_name}: {exc}")

    cn_image = cn_sample_3()
    depth_image = depth_sample_1()
    res_image = res_sample_1()
    cn_path = _write_tiff(tmp_path, "cn_sample_3", cn_image)
    depth_path = _write_tiff(tmp_path, "depth_sample_1", depth_image)
    res_path = _write_tiff(tmp_path, "res_sample_1", res_image)
    cn_anchors = _grid_anchors(cn_image, "concentration")
    depth_anchors = _grid_anchors(depth_image, "depth")

    cn_auto = importlib.import_module(
        "qal.examples.concentration_rcs.concentration_automatic_example"
    )
    monkeypatch.setattr(cn_auto, "IMAGE_PATH", str(cn_path))
    monkeypatch.setattr(cn_auto, "SAVE_RESULTS", True)
    check(
        cn_auto.__name__,
        cn_path.parent / "cn_sample_3_results",
        ["roi.png", "concentration.png", "stats.csv"],
    )

    depth_auto = importlib.import_module(
        "qal.examples.depth_rds.depth_automatic_example"
    )
    monkeypatch.setattr(depth_auto, "IMAGE_PATH", str(depth_path))
    monkeypatch.setattr(depth_auto, "SAVE_RESULTS", True)
    check(
        depth_auto.__name__,
        depth_path.parent / "depth_sample_1_results",
        ["roi.png", "depth.png", "stats.csv"],
    )

    cn_gui = importlib.import_module(
        "qal.examples.concentration_rcs.concentration_gui_example"
    )
    monkeypatch.setattr(cn_gui, "IMAGE_PATH", str(cn_path))
    monkeypatch.setattr(cn_gui, "SAVE_RESULTS", True)
    monkeypatch.setattr(cn_gui, "MODE", "automatic_refinement")
    monkeypatch.setattr(cn_gui, "COORDINATES", cn_anchors)
    check(
        cn_gui.__name__,
        cn_path.parent / "cn_sample_3_results",
        ["roi.png", "concentration.png", "stats.csv", "selection.json"],
    )

    depth_gui = importlib.import_module(
        "qal.examples.depth_rds.depth_gui_example"
    )
    monkeypatch.setattr(depth_gui, "IMAGE_PATH", str(depth_path))
    monkeypatch.setattr(depth_gui, "SAVE_RESULTS", True)
    monkeypatch.setattr(depth_gui, "MODE", "automatic_refinement")
    monkeypatch.setattr(depth_gui, "COORDINATES", depth_anchors)
    check(
        depth_gui.__name__,
        depth_path.parent / "depth_sample_1_results",
        ["roi.png", "depth.png", "stats.csv", "selection.json"],
    )

    rrt = importlib.import_module(
        "qal.examples.resolution_rrt.resolution_gui_example"
    )
    monkeypatch.setattr(rrt, "IMAGE_PATH", str(res_path))
    monkeypatch.setattr(rrt, "SAVE_RESULTS", True)
    monkeypatch.setattr(rrt, "MODE", "manual_line_profiles")
    monkeypatch.setattr(rrt, "AUTO_CROP", True)
    monkeypatch.setattr(rrt, "REGISTRATION_POINTS", None)
    monkeypatch.setattr(rrt, "LINE_COORDINATES", RESOLUTION_LINES)
    check(
        rrt.__name__,
        res_path.parent / "res_sample_1_results",
        [
            "line_profiles.png",
            "percentage_contrast.png",
            "percentage_contrast.csv",
            "selection.yaml",
        ],
    )

    ret = importlib.import_module(
        "qal.examples.radiometric_emitter_ret.ret_gui_example"
    )
    light = np.full((61, 61), 10, dtype=np.uint16)
    light[circular_mask(light.shape, (30, 28), 10)] = 110
    yy, xx = np.indices(light.shape)
    dark = (5 + (xx + 2 * yy) % 7).astype(np.uint16)
    light_path = _write_tiff(tmp_path, "ret_light", light)
    dark_path = _write_tiff(tmp_path, "ret_dark", dark)
    monkeypatch.setattr(ret, "IMAGE_PATH", str(light_path))
    monkeypatch.setattr(ret, "DARK_FRAME_PATH", str(dark_path))
    monkeypatch.setattr(ret, "SAVE_RESULTS", True)
    monkeypatch.setattr(ret, "MODE", "manual_coordinates")
    monkeypatch.setattr(ret, "COORDINATES", [(30, 28)])
    monkeypatch.setattr(ret, "MANUAL_ROI_DIAMETER", 20, raising=False)
    monkeypatch.setattr(ret, "SEARCH_RADIUS", 20)
    check(
        ret.__name__,
        light_path.parent / "ret_light_results",
        ["roi.png", "stats.csv"],
    )

    def run_rud(module_name, fetcher, subdir, expected, enable):
        source = Path(fetcher())
        copied = _copy_tree(source, tmp_path / f"{source.name}_{module_name.rsplit('.', 1)[-1]}")
        module = importlib.import_module(module_name)
        monkeypatch.setattr(module, fetcher.__name__, lambda: str(copied))
        enable(module)
        check(module_name, copied / subdir, expected)

    def enable_uniformity(module):
        # Exercise GIF export without rendering an entire 120-frame rotation.
        monkeypatch.setattr(module.UniformityVisualizer, "ANI_FRAMES", 3)
        original = module.UniformityAnalyzer.update_params

        def update(self, params):
            patched = dict(params)
            patched["Save output"] = True
            return original(self, patched)

        monkeypatch.setattr(module.UniformityAnalyzer, "update_params", update)
        original_vis = module.UniformityVisualizer.update_params

        def vis_update(self, params):
            patched = dict(params)
            patched["Show figures"] = False
            return original_vis(self, patched)

        monkeypatch.setattr(module.UniformityVisualizer, "update_params", vis_update)

    def enable_distortion(module):
        original = module.DistortionVisualizer.visualize_distortion

        def visualize(self, outputs, save=False, lens_maker=False):
            return original(self, outputs, save=True, lens_maker=lens_maker)

        monkeypatch.setattr(
            module.DistortionVisualizer, "visualize_distortion", visualize
        )

    run_rud(
        "qal.examples.uniformity_distortion_rud.uniformity_good_system_example",
        rud_example_1,
        "Surface Representation",
        ["output_dict.pkl", "*"],
        enable_uniformity,
    )
    run_rud(
        "qal.examples.uniformity_distortion_rud.uniformity_poor_system_example",
        rud_example_2,
        "Surface Representation",
        ["output_dict.pkl", "*"],
        enable_uniformity,
    )
    run_rud(
        "qal.examples.uniformity_distortion_rud.distortion_full_fov_example",
        rud_example_1,
        "Distortion Figures",
        ["*"],
        enable_distortion,
    )
    run_rud(
        "qal.examples.uniformity_distortion_rud.distortion_sub_fov_example",
        rud_example_3,
        "Distortion Figures",
        ["*"],
        enable_distortion,
    )

    assert not failures, "Example save checks failed:\n" + "\n".join(failures)
