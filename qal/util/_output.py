"""Utilities for analysis output folders, sample images, and target metadata."""

from pathlib import Path
import shutil

import numpy as np
import yaml


def results_directory(image_input, fallback):
    """Use the input file's location, or the supplied script/notebook fallback."""
    base = Path(fallback if isinstance(image_input, np.ndarray) else image_input)
    base = base.expanduser().resolve()
    return base.parent / f"{base.stem}_results"


def save_sample_sources(results_dir, *image_inputs):
    """Copy original built-in samples; ignore user paths and untracked arrays.

    This copies resolved source files byte-for-byte without moving the cache
    or re-encoding quantitative image data. Multiple images share a folder.
    """
    from qal.data._fetchers import _sample_source_paths

    sources = list(dict.fromkeys(
        path for value in image_inputs for path in _sample_source_paths(value)
    ))
    if not sources:
        return []
    destination = Path(results_dir)
    if len(sources) > 1:
        destination /= "source_images"
    destination.mkdir(parents=True, exist_ok=True)
    saved = []
    for source in sources:
        target = destination / source.name
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
        saved.append(target)
    return saved


def save_target_metadata(results_dir, preset, *, query=None):
    """Save the actual query and complete resolved preset with analysis output."""
    path = Path(results_dir) / "target.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(
        {"query": query, "preset": dict(preset)}, sort_keys=False
    ), encoding="utf-8")
    return path
