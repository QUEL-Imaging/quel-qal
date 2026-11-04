"""Build real wheels and sdists, then resolve their provenance without Git."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("ref, override", [("dev", None), ("main", None),
                                           ("feature/test-data", None), ("dev", "v1.2")])
def test_wheel_and_sdist_keep_source_ref(tmp_path, ref, override):
    pytest.importorskip("build")
    from build import ProjectBuilder
    project = tmp_path / "project"
    project.mkdir()
    for name in ("pyproject.toml", "setup.py", "MANIFEST.in", "README.md", "LICENSE"):
        if (ROOT / name).exists():
            shutil.copy2(ROOT / name, project / name)
    # Use actual package code but leave large datasets outside build fixtures.
    shutil.copytree(ROOT / "qal", project / "qal", ignore=lambda folder, names: [
        name for name in names if name == "__pycache__" or
        (Path(folder) / name).is_file() and Path(name).suffix not in (".py", ".pyi", ".yaml", ".md")
    ])
    def git(*args):
        subprocess.run(["git", "-C", str(project), *args], check=True, capture_output=True)
    git("init", "-b", ref)
    git("add", ".")
    git("-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-m", "fixture")
    git("remote", "add", "origin", "https://github.com/testing/build-repo.git")
    env = os.environ.copy()
    for name in ("QAL_DATA_REF", "QAL_DATA_REPOSITORY"):
        env.pop(name, None)
    if override:
        env["QAL_DATA_REF"] = override
    env["QAL_REQUIRE_DATA_PROVENANCE"] = "1"
    def runner(cmd, cwd, extra_environ):
        subprocess.run(cmd, cwd=cwd, env=env | extra_environ, check=True, capture_output=True)
    builder = ProjectBuilder(str(project), runner=runner)
    direct = builder.build("wheel", str(tmp_path / "direct"))
    sdist = builder.build("sdist", str(tmp_path / "sdist"))
    unpacked = tmp_path / "unpacked"
    with tarfile.open(sdist) as archive:
        archive.extractall(unpacked, filter="data")
    sdist_root = next(unpacked.iterdir())
    roundtrip = ProjectBuilder(str(sdist_root), runner=runner).build("wheel", str(tmp_path / "roundtrip"))
    for index, wheel in enumerate((direct, roundtrip)):
        installed = tmp_path / f"installed-{index}"
        # Installing pure-Python wheel contents is sufficient to exercise the
        # runtime without dependencies or access to the original checkout.
        with zipfile.ZipFile(wheel) as archive:
            archive.extractall(installed)
        runtime = env | {"PATH": ""}
        runtime.pop("QAL_DATA_REF", None)
        code = ("import sys, json; "
                f"sys.path.insert(0, {str(installed)!r}); "
                "from qal.data._source import resolve_source; "
                "from qal.data._fetchers import _create_image_fetcher; "
                "s=resolve_source(); "
                "assert s.ref.replace('/', '%2F') in _create_image_fetcher().get_url('concentration_targets/cn_sample_1.tiff'); "
                "print(json.dumps([s.ref,s.repository,s.checkout]))")
        result = subprocess.check_output([sys.executable, "-I", "-c", code], cwd=tmp_path,
                                         env=runtime, text=True)
        assert json.loads(result) == [override or ref, "testing/build-repo", None]
    assert not (project / "qal/data/_data_source.json").exists()


def test_build_missing_provenance_is_actionable(tmp_path):
    source = (ROOT / "setup.py").read_text()
    # Execute definitions without invoking setuptools; no package or Git in fixture.
    scope = {"__file__": str(ROOT / "setup.py")}
    exec(source[:source.index('setup(cmdclass=')], scope)
    scope["source"]["checkout_source"] = lambda: None
    scope["source"]["recorded_source"] = lambda: None
    from unittest.mock import patch
    with patch.dict(os.environ, {"QAL_REQUIRE_DATA_PROVENANCE": "1"}, clear=True):
        with pytest.raises(RuntimeError, match="provenance is missing"):
            scope["write_provenance"](tmp_path / "data")
    with patch.dict(os.environ, {}, clear=True):
        with pytest.warns(UserWarning, match="provenance is missing"):
            scope["write_provenance"](tmp_path / "data")
