"""YAML selection persistence with support for existing JSON files."""

import json
from pathlib import Path
from typing import Union

import yaml


def save_selection(path: Union[str, Path], data: dict) -> Path:
    """Save YAML by default; an explicit .json suffix requests JSON."""
    path = Path(path)
    if not path.suffix:
        path = path.with_suffix(".yaml")
    path.parent.mkdir(parents=True, exist_ok=True)
    contents = (
        json.dumps(data, indent=2)
        if path.suffix.lower() == ".json"
        else yaml.safe_dump(data, sort_keys=False)
    )
    path.write_text(contents, encoding="utf-8")
    return path


def load_selection(path: Union[str, Path]) -> dict:
    """Read YAML or a legacy JSON selection."""
    path = Path(path)
    contents = path.read_text(encoding="utf-8")
    return (
        json.loads(contents)
        if path.suffix.lower() == ".json"
        else yaml.safe_load(contents)
    )
