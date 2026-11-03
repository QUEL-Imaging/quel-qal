"""Prevent metadata and the reproducible pip environment from drifting apart."""
from pathlib import Path
import tomllib

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


ROOT = Path(__file__).resolve().parents[1]


def requirements(filename):
    return {
        canonicalize_name(requirement.name): requirement
        for raw in (ROOT / filename).read_text().splitlines()
        for line in [raw.strip()]
        if line and not line.startswith(("#", "-"))
        for requirement in [Requirement(line)]
    }


def test_runtime_pins_satisfy_every_declared_package_dependency():
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())
    dependencies = {canonicalize_name(req.name): req for value in metadata["project"]["dependencies"]
                    for req in [Requirement(value)]}
    pins = requirements("requirements.txt")
    assert pins.keys() == dependencies.keys()
    for name, pin in pins.items():
        versions = list(pin.specifier)
        assert len(versions) == 1 and versions[0].operator == "==", name
        assert versions[0].version in dependencies[name].specifier, name


def test_transitive_constraints_match_runtime_and_test_pins():
    constraints = requirements("requirements-constraints.txt")
    pins = requirements("requirements.txt") | requirements("requirements-test.txt")
    for name, requirement in pins.items():
        assert str(constraints[name].specifier) == str(requirement.specifier), name
    # Installing multiple distributions supplying cv2 corrupts the namespace.
    assert {name for name in constraints if name.startswith("opencv-")} == {"opencv-contrib-python"}


def test_requirements_apply_the_shared_constraints():
    assert "-c requirements-constraints.txt" in (ROOT / "requirements.txt").read_text().splitlines()
    assert "-r requirements.txt" in (ROOT / "requirements-test.txt").read_text().splitlines()
