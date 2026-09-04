"""Image loading helpers shared across QAL workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
from skimage import io

ImageInput = Union[str, Path, np.ndarray]


def load_grayscale_image(image: ImageInput) -> np.ndarray:
    """Return a two-dimensional image from a path or array.

    RGB/RGBA images are reduced to their first channel. For multi-page image
    stacks, the first page is used.
    """
    im = image if isinstance(image, np.ndarray) else io.imread(str(image))
    im = np.asarray(im)

    if im.ndim == 3:
        im = im[..., 0] if im.shape[-1] in (3, 4) else im[0]
    if im.ndim != 2:
        raise ValueError(f"Expected a 2-D grayscale image, got shape {im.shape}")

    return im
