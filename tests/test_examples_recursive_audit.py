"""Exercise every runnable example outside the excluded RUD script directory."""

import ast
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from skimage import io

from qal import LungPhantom
from qal.data import _fetchers, get_target_preset
from qal.util import circular_mask

ROOT = Path(__file__).resolve().parents[1] / "qal/examples"
FILES = sorted(p for p in ROOT.rglob("*")
               if p.suffix in {".py", ".ipynb"}
               and "uniformity_distortion_rud" not in p.parts)
RUNNERS = [p for p in FILES if p.suffix == ".ipynb" or "def main(" in p.read_text()]
GRID_SCRIPTS = [p for p in RUNNERS if p.suffix == ".py"
                and p.parent.name in {"concentration_rcs", "depth_rds"}]


def code_cells(path):
    if path.suffix == ".py":
        return [path.read_text()]
    return ["".join(c["source"]) for c in json.loads(path.read_text())["cells"]
            if c["cell_type"] == "code"]


@pytest.mark.parametrize("path", FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_recursive_example_contract(path):
    trees = [ast.parse(s, filename=str(path)) for s in code_cells(path)]
    assert path in RUNNERS, "Shared implementation belongs in qal/util, outside the examples tree."
    settings = {target.id: node.value for tree in trees for node in tree.body
                if isinstance(node, ast.Assign) for target in node.targets
                if isinstance(target, ast.Name)}
    assert ast.literal_eval(settings["SAVE_RESULTS"]) is True
    if "target_preset_lookup" not in path.name:
        assert ast.literal_eval(settings["SAVE_SOURCE_IMAGES"]) is False


@pytest.fixture
def outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda *a, **kw: plt.close("all"))
    monkeypatch.chdir(tmp_path)
    cache = tmp_path / "sample_cache"
    cache.mkdir()

    def sample(name, image):
        path = cache / name
        io.imsave(path, image, check_contrast=False)
        # Exercise the same provenance tracking as the public built-in loaders.
        with monkeypatch.context() as context:
            context.setattr(_fetchers, "_fetch", lambda _: str(path))
            return _fetchers._load(name), path

    yield sample
    plt.close("all")


def grid_image(family):
    image = np.full((260, 260), 10, dtype=np.uint16)
    wells = []
    yy, xx = np.indices(image.shape)
    for i, label in enumerate(get_target_preset(family, "icg")["well_ids"]):
        center = (50 + (i % 3) * 80, 50 + (i // 3) * 80)
        mask = circular_mask(image.shape, center, 20)
        image[mask] = (900 - i * 100 + (xx + yy) % 3)[mask]
        wells.append({"well": label, "value": float(label.split()[0]) if label != "Control" else 0., "x": center[0], "y": center[1],
                      "ROI Diameter": 40., "ROI Radius": 20.})
    return image, pd.DataFrame(wells)


def configure(module, image, source, kind, tmp_path, monkeypatch):
    monkeypatch.setattr(module, "__file__", str(tmp_path / "scripts" / Path(module.__file__).name))
    monkeypatch.setattr(module, "IMAGE_PATH", source if kind == "user" else image)
    monkeypatch.setattr(module, "SAVE_RESULTS", kind != "disabled")
    monkeypatch.setattr(module, "SAVE_SOURCE_IMAGES", True)
    base = source if kind == "user" else Path(module.__file__)
    return base.parent / f"{base.stem}_results"


def assert_sources(out, source, kind, expected):
    if kind == "disabled":
        assert not out.exists()
        return
    for filename in expected:
        path = out / filename
        assert path.is_file() and path.stat().st_size > 0, filename
        if path.suffix == ".png":
            assert plt.imread(path).std() > .01
    if kind == "user":
        assert not list(out.rglob("*.tiff"))
    else:
        assert (out / source.name).read_bytes() == source.read_bytes()


@pytest.mark.parametrize("path", GRID_SCRIPTS, ids=lambda p: p.stem)
@pytest.mark.parametrize("kind", ["sample", "user", "disabled"])
def test_each_grid_script_exports_analysis_and_optional_original(path, kind, outputs, tmp_path, monkeypatch):
    module = importlib.import_module("qal.examples." + ".".join(path.relative_to(ROOT).with_suffix("").parts))
    family = "concentration" if path.parent.name == "concentration_rcs" else "depth"
    image, wells = grid_image(family)
    image, source = outputs("grid.tiff", image)
    out = configure(module, image, source, kind, tmp_path, monkeypatch)
    if "automatic_example" in path.name:
        monkeypatch.setattr(module, "WellDetector", lambda: SimpleNamespace(
            detect_wells=lambda *a, **kw: wells,
            estimate_remaining_wells_3x3=lambda *a, **kw: wells))
    else:
        monkeypatch.setattr(module, "select_concentration_grid", lambda *a, **kw: wells)
    module.main()
    assert_sources(out, source, kind, ["stats.csv", "roi.png", f"{family}.png", "target.yaml"])
    if kind != "disabled":
        assert len(pd.read_csv(out / "stats.csv")) == 9


@pytest.mark.parametrize("name", ["ret_gui_example", "ret_light_frame_automatic_analysis_example"])
@pytest.mark.parametrize("kind", ["sample", "user", "disabled"])
def test_each_ret_script_exports_original_pair(name, kind, outputs, tmp_path, monkeypatch):
    module = importlib.import_module(f"qal.examples.radiometric_emitter_ret.{name}")
    from qal.examples.radiometric_emitter_ret import ret_gui_example as workflow
    image = np.full((61, 61), 10, dtype=np.uint16)
    image[circular_mask(image.shape, (30, 28), 10)] = 110
    dark = (5 + np.indices(image.shape).sum(axis=0) % 7).astype(np.uint16)
    image, light_source = outputs("light.tiff", image)
    dark, dark_source = outputs("dark.tiff", dark)
    out = configure(module, image, light_source, kind, tmp_path, monkeypatch)
    monkeypatch.setattr(module, "DARK_FRAME_PATH", dark_source if kind == "user" else dark)
    if name == "ret_gui_example":
        monkeypatch.setattr(module, "MODE", "manual_coordinates")
        monkeypatch.setattr(module, "COORDINATES", [(30, 28)])
        monkeypatch.setattr(module, "MANUAL_ROI_DIAMETER", 20., raising=False)
    else:
        monkeypatch.setattr(workflow.WellDetector, "detect_wells", lambda *a, **kw: pd.DataFrame([
            {"x": 30., "y": 28., "ROI Diameter": 20.}]))
    module.main()
    if kind == "disabled":
        assert not out.exists()
    else:
        assert len(pd.read_csv(out / "stats.csv")) == 1
        assert (out / "roi.png").is_file()
        assert not (out / "wells.csv").exists()
        if kind == "sample":
            for source in (light_source, dark_source):
                assert (out / "source_images" / source.name).read_bytes() == source.read_bytes()
        else:
            assert not (out / "source_images").exists()


def lung_inputs():
    yy, xx = np.indices((240, 240))
    body = np.zeros((240, 240), dtype=np.uint16)
    body[(xx - 120)**2 + (yy - 120)**2 < 110**2] = (1000 + (xx + yy) % 20)[(xx - 120)**2 + (yy - 120)**2 < 110**2]
    inclusions = np.zeros_like(body)
    for i, (center, radius) in enumerate(zip([(70, 70), (170, 70), (70, 170), (170, 170), (120, 120)], [8, 12, 16, 20, 24])):
        mask = circular_mask(body.shape, center, radius)
        inclusions[mask] = 500
        body[mask] += 200 + i * 50
    return body, inclusions


@pytest.mark.parametrize("kind", ["sample", "user", "disabled"])
def test_lung_reference_exports_results_and_original_pair(kind, outputs, tmp_path, monkeypatch):
    from qal import data
    monkeypatch.setattr(data, "lung_reference", lung_inputs)
    from qal.examples.lung_phantom import create_lung_reference as module
    lung_module = importlib.import_module("qal.vpa.quel_lung_phantom")
    monkeypatch.setattr(lung_module, "get_cache_dir", lambda: str(tmp_path / "cache"))
    body, inclusions = lung_inputs()
    body, body_source = outputs("body.tiff", body)
    inclusions, inclusion_source = outputs("inclusions.tiff", inclusions)
    out = configure(module, body, body_source, kind, tmp_path, monkeypatch)
    monkeypatch.setattr(module, "INCLUSIONS_IMAGE_PATH", inclusion_source if kind == "user" else inclusions)
    reference = module.main()
    assert len(np.unique(reference.mask_reference)) == 3
    if kind == "disabled":
        assert not out.exists()
    else:
        assert (out / "Lung_reference_image.tiff").is_file()
        assert (out / "Lung_reference_mask.png").is_file()
        assert len(pd.read_pickle(out / "Inclusions_info.pkl")) == 5
        if kind == "sample":
            for source in (body_source, inclusion_source):
                assert (out / "source_images" / source.name).read_bytes() == source.read_bytes()
        else:
            assert not (out / "source_images").exists()


@pytest.mark.parametrize("kind", ["sample", "user", "disabled"])
def test_lung_analysis_exports_real_excel_metrics(kind, outputs, tmp_path, monkeypatch):
    from qal import data
    monkeypatch.setattr(data, "lung_test_image", lambda: lung_inputs()[0])
    from qal.examples.lung_phantom import lung_phantom_example as module
    lung_module = importlib.import_module("qal.vpa.quel_lung_phantom")
    monkeypatch.setattr(lung_module, "get_cache_dir", lambda: str(tmp_path / "cache"))
    body, inclusions = lung_inputs()
    reference = LungPhantom()
    reference_dir = tmp_path / "reference"
    reference_dir.mkdir()
    reference.ref_dir = str(reference_dir)
    reference.create_reference_mask(body, inclusions)
    # Keep real mask warping and metrics; make registration deterministic and fast.
    monkeypatch.setattr(reference, "_reg_ref_to_target", lambda _: np.array([1., 0., 0., 0.]))
    monkeypatch.setattr(module, "LungPhantom", lambda: reference)
    image, source = outputs("lung.tiff", body)
    out = configure(module, image, source, kind, tmp_path, monkeypatch)
    monkeypatch.setattr(module, "SHOW_PROGRESS", False)
    metrics = module.main()
    assert len(metrics[0]) == 5
    assert_sources(out, source, kind, ["Phantom_metrics.xlsx", "Phantom_summary_metrics.xlsx"])
    if kind != "disabled":
        assert len(pd.read_excel(out / "Phantom_metrics.xlsx")) == 5


@pytest.mark.parametrize("kind", ["sample", "user", "disabled"])
def test_depth_resolution_exports_profiles_and_original(kind, outputs, tmp_path, monkeypatch):
    from qal.examples.depth_resolution_dr import depth_resolution_example as module
    yy, xx = np.indices((160, 240))
    image = (20 + 1000 * np.exp(-xx / 80) * np.exp(-((yy - 80) / (5 + xx / 40))**2)).astype(np.uint16)
    image, source = outputs("channel.tiff", image)
    out = configure(module, image, source, kind, tmp_path, monkeypatch)
    monkeypatch.setattr(module, "use_native_window_backend", lambda: None)

    def crop(self, image):
        self.img = image
        self.borders = {"top": 0, "bottom": image.shape[0], "left": 0, "right": image.shape[1]}

    monkeypatch.setattr(module.PhantomCropper, "crop_image", crop)
    module.main()
    assert_sources(out, source, kind, ["Intensity vs channel length.png", "Normalized intensity vs depth.png",
                                       "Intensity spread profiles.png", "FWHM vs depth.png"])


RRT_LINES = [((64.1, 71.2), (64.1, 220.3)),
             ((199., 73.8), (199., 169.4)),
             ((131.9, 125.), (131.9, 164.))]


@pytest.mark.parametrize("kind", ["sample", "user", "disabled"])
def test_resolution_script_exports_overlay_and_original(kind, outputs, tmp_path, monkeypatch):
    from qal.examples.resolution_rrt import resolution_gui_example as module
    from qal.data import res_sample_1
    image, source = outputs("resolution.tiff", res_sample_1())
    out = configure(module, image, source, kind, tmp_path, monkeypatch)
    monkeypatch.setattr(module, "MODE", "manual_line_profiles")
    monkeypatch.setattr(module, "LINE_COORDINATES", RRT_LINES)
    module.main()
    assert_sources(out, source, kind, ["roi.png", "line_profiles.png", "percentage_contrast.png",
                                      "percentage_contrast.csv", "selection.yaml"])


NOTEBOOKS = [p for p in RUNNERS if p.suffix == ".ipynb" and "lookup" not in p.name]


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.stem)
@pytest.mark.parametrize("kind", ["sample", "user", "disabled"])
def test_each_image_notebook_exports_results_and_optional_sources(path, kind, outputs, tmp_path, monkeypatch):
    from qal import select_wells_from_coordinates
    from qal.data import res_sample_1

    name = path.name
    dark = dark_source = None
    if name.startswith("concentration") or name.startswith("depth"):
        family = "concentration" if name.startswith("concentration") else "depth"
        image, wells = grid_image(family)
        expected = ["roi.png", f"{family}.png", "stats.csv", "target.yaml"]
    elif name.startswith("radiometric"):
        image = np.full((61, 61), 10, dtype=np.uint16)
        image[circular_mask(image.shape, (30, 28), 10)] = 110
        dark, dark_source = outputs("dark.tiff", (5 + np.indices(image.shape).sum(axis=0) % 7).astype(np.uint16))
        expected = ["roi.png", "stats.csv"]
    elif name.startswith("resolution"):
        image = res_sample_1()
        expected = ["roi.png", "line_profiles.png", "percentage_contrast.png", "percentage_contrast.csv", "selection.yaml"]
    else:
        image = np.arange(16 * 16).reshape(16, 16).astype(np.uint16)
        expected = ["detected_wells.png", "distortion_vs_image_height.png", "distortion_map.png", "detected_wells.csv", "distortion.csv"]
    image, source = outputs("image.tiff", image)
    sources = [source]
    if dark_source is not None:
        sources.append(dark_source)

    ns = {}
    for cell in code_cells(path):
        if "if IMAGE_PATH is None:" in cell or "if IMAGE_DIR is None:" in cell:
            ns.update(IMAGE_PATH=source if kind == "user" else image,
                      SAVE_RESULTS=kind != "disabled", SAVE_SOURCE_IMAGES=True)
            if name.startswith("concentration") or name.startswith("depth"):
                ns["select_concentration_grid"] = lambda *a, **kw: wells
            elif name.startswith("radiometric"):
                ns.update(DARK_FRAME_PATH=dark_source if kind == "user" else dark,
                          MODE="manual_coordinates", MANUAL_ROI_DIAMETER=20,
                          use_native_window_backend=lambda: None, use_inline_plots=lambda: None,
                          select_wells_gui=lambda image, **kw: select_wells_from_coordinates(
                              image, [(30, 28)], well_ids=["RET"], coordinate_mode="manual", manual_roi_diameter=20))
            elif name.startswith("resolution"):
                ns.update(MODE="manual_line_profiles", LINE_COORDINATES=RRT_LINES)
            else:
                second, second_source = outputs("second.tiff", image + 1)
                sources.append(second_source)
                folder = source.parent
                with monkeypatch.context() as context:
                    context.setattr(_fetchers, "registry", {f"rud/{p.name}": "" for p in sources})
                    context.setattr(_fetchers, "_fetch", lambda name: str(folder / Path(name).name))
                    _fetchers._fetch_directory("rud")
                if kind == "user":
                    folder = tmp_path / "user_images"
                    folder.mkdir()
                    for p in sources:
                        (folder / p.name).write_bytes(p.read_bytes())
                ns["IMAGE_DIR"] = str(folder)
                dots = pd.DataFrame({"x": [4., 10.], "y": [5., 11.]})
                data = {"Directory": str(folder), "Actual distances": np.linspace(1, 10, 20),
                        "Distortion": np.linspace(.01, .5, 20)**2, "FOV": (16, 16),
                        "xq": np.arange(16), "yq": np.arange(16),
                        "Distortion map": np.indices((16, 16)).sum(axis=0) / 30}
                ns["RudDetector"] = lambda: SimpleNamespace(
                    detect_dots_distortion=lambda _: None, output=data,
                    dots=[dots, dots], images=[image, second])
                ns["DistortionAnalyzer"] = lambda: SimpleNamespace(
                    compute_distortion=lambda *a, **kw: None, output=data)
        exec(compile(cell, str(path), "exec"), ns)

    out = ns["results_dir"]
    if kind == "disabled":
        assert out is None
        assert not list(tmp_path.rglob("*.csv"))
        assert not list(tmp_path.rglob("*.png"))
        return
    assert out is not None
    for filename in expected:
        saved = out / filename
        assert saved.is_file() and saved.stat().st_size > 0, filename
        if saved.suffix == ".png":
            assert plt.imread(saved).std() > .01
    if kind == "sample":
        folder = out / "source_images" if len(sources) > 1 else out
        for p in sources:
            assert (folder / p.name).read_bytes() == p.read_bytes()
        assert not out.is_relative_to(source.parent)
    else:
        assert not list(out.rglob("*.tiff"))
    assert not (out / "wells.csv").exists()


@pytest.mark.parametrize("save", [True, False])
def test_lookup_notebook_exports_yaml_only_when_enabled(save, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ns = {}
    for cell in code_cells(ROOT / "jupyter-notebook/target_preset_lookup_example.ipynb"):
        exec(cell, ns)
        ns["SAVE_RESULTS"] = save
    out = tmp_path / "target_preset_lookup_results/target_lookup.yaml"
    assert out.is_file() == save
    if save:
        import yaml
        payload = yaml.safe_load(out.read_text())
        assert payload["query"] == "ICG"
        assert payload["matches"]
