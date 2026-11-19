"""Record data provenance in build outputs without changing the source checkout."""
import json
import os
from pathlib import Path
import runpy
import warnings

from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.command.sdist import sdist

ROOT = Path(__file__).resolve().parent
source = runpy.run_path(str(ROOT / "qal/data/_source.py"))


def write_provenance(data_dir):
    resolved = source["resolve_source"]()
    if not (os.environ.get("QAL_DATA_REF") or source["checkout_source"]()
            or source["recorded_source"]()):
        message = ("QAL data build provenance is missing; using main. Build from a Git "
                   "checkout or a provenance-bearing sdist, or set QAL_DATA_REF.")
        if os.environ.get("QAL_REQUIRE_DATA_PROVENANCE") == "1":
            raise RuntimeError(message)
        warnings.warn(message, stacklevel=2)
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "_data_source.json").write_text(json.dumps({
        "ref": resolved.ref, "repository": resolved.repository,
    }) + "\n", encoding="utf-8")


class BuildPy(build_py):
    def run(self):
        super().run()
        if not self.editable_mode:
            write_provenance(Path(self.build_lib) / "qal/data")


class Sdist(sdist):
    def make_release_tree(self, base_dir, files):
        super().make_release_tree(base_dir, files)
        write_provenance(Path(base_dir) / "qal/data")


setup(cmdclass={"build_py": BuildPy, "sdist": Sdist})
