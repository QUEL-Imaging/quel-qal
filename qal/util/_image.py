"""Image loading helpers shared across QAL workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
from skimage import io

ImageInput = Union[str, Path, np.ndarray]


@dataclass(frozen=True)
class StoredValueInfo:
    """Container dtype versus the range actually occupied by stored samples.

    A 16-bit TIFF whose samples only occupy 0–255 is still ``uint16``;
    it is not scaled to 0–65535.
    """

    dtype: np.dtype
    vmin: float
    vmax: float
    container_bits: int | None
    used_bits: int | None
    is_8bit_in_16bit: bool

    def summary(self) -> str:
        dtype_name = np.dtype(self.dtype).name
        range_text = f"stored values {self.vmin:g}–{self.vmax:g}"
        if self.is_8bit_in_16bit:
            return (
                f"{dtype_name} container, {range_text} "
                "(8-bit samples in a 16-bit file; not scaled to 0–65535)"
            )
        if (
            self.container_bits is not None
            and self.used_bits is not None
            and self.used_bits < self.container_bits
        ):
            return (
                f"{dtype_name} container ({self.container_bits}-bit), "
                f"{range_text} (effective {self.used_bits}-bit samples)"
            )
        return f"{dtype_name}, {range_text}"


def describe_stored_values(image: np.ndarray) -> StoredValueInfo:
    """Describe the stored sample range without rescaling the image."""
    array = np.asarray(image)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        raise ValueError("Image contains no finite samples")

    vmin = float(np.min(finite))
    vmax = float(np.max(finite))
    dtype = array.dtype
    container_bits = None
    used_bits = None
    if np.issubdtype(dtype, np.integer):
        container_bits = int(np.iinfo(dtype).bits)
        peak = max(abs(vmin), abs(vmax))
        used_bits = int(np.ceil(np.log2(peak + 1))) if peak > 0 else 0

    is_8bit_in_16bit = (
        np.issubdtype(dtype, np.unsignedinteger)
        and container_bits == 16
        and vmin >= 0
        and vmax <= 255
    )
    return StoredValueInfo(
        dtype=dtype,
        vmin=vmin,
        vmax=vmax,
        container_bits=container_bits,
        used_bits=used_bits,
        is_8bit_in_16bit=is_8bit_in_16bit,
    )


def to_log_scale(image: np.ndarray) -> np.ndarray:
    """Return a 16-bit log-compressed copy for display and detection.

    This is the stretch used by well detection, so ROI overlays on the
    log image line up with ``WellDetector.plot_detected_wells``.
    """
    uint16_max = 65535
    scaled = np.asarray(image, dtype=float) - np.nanmin(image)
    peak = np.nanmax(scaled)
    if not np.isfinite(peak) or peak <= 0:
        return np.zeros(np.asarray(image).shape[:2], dtype=np.uint16)
    scaled = uint16_max * (scaled / peak)
    c = uint16_max / np.log(1 + np.max(scaled))
    return np.array(c * np.log(scaled + 1.0), dtype=np.uint16)


def load_grayscale_image(image: ImageInput) -> np.ndarray:
    """Return stored grayscale samples from a path or array.

    Values are read with :func:`skimage.io.imread` and are not rescaled by
    container bit depth. A 16-bit file whose samples only occupy 0–255 is
    returned as ``uint16`` with those values.

    RGB/RGBA images are reduced to their first channel. For multi-page image
    stacks, the first page is used.
    """
    im = image if isinstance(image, np.ndarray) else io.imread(str(image))
    im = np.squeeze(np.asarray(im))

    if im.ndim == 3:
        im = im[..., 0] if im.shape[-1] in (3, 4) else im[0]
    if im.ndim != 2:
        raise ValueError(f"Expected a 2-D grayscale image, got shape {im.shape}")

    return im
