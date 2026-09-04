"""Region-of-interest helpers shared across QAL workflows."""

from __future__ import annotations

from typing import Literal, Sequence

import numpy as np

CenterOrder = Literal["xy", "rc"]


def circular_mask(
    shape: Sequence[int],
    center: Sequence[float],
    radius: float,
    *,
    center_order: CenterOrder = "xy",
) -> np.ndarray:
    """Return a boolean circular mask for a two-dimensional image shape.

    ``center_order="xy"`` interprets the center as ``(column, row)`` while
    ``center_order="rc"`` interprets it as ``(row, column)``.
    """
    if len(shape) < 2:
        raise ValueError(f"Expected an image shape with two dimensions, got {shape}")
    if len(center) != 2:
        raise ValueError(f"Expected a two-coordinate center, got {center}")
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError(f"radius must be positive and finite, got {radius!r}")

    height, width = int(shape[0]), int(shape[1])
    if center_order == "xy":
        x, y = float(center[0]), float(center[1])
    elif center_order == "rc":
        y, x = float(center[0]), float(center[1])
    else:
        raise ValueError(f"center_order must be 'xy' or 'rc', got {center_order!r}")

    yy, xx = np.ogrid[:height, :width]
    return (xx - x) ** 2 + (yy - y) ** 2 <= float(radius) ** 2


def mean_in_circular_roi(
    image: np.ndarray,
    center: Sequence[float],
    radius: float,
    *,
    center_order: CenterOrder = "xy",
) -> float:
    """Return the finite-pixel mean inside a circular ROI."""
    mask = circular_mask(image.shape, center, radius, center_order=center_order)
    values = np.asarray(image, dtype=float)[mask]
    if values.size == 0 or not np.isfinite(values).any():
        raise ValueError("Circular ROI contains no finite pixels")
    return float(np.nanmean(values))


def std_in_circular_roi(
    image: np.ndarray,
    center: Sequence[float],
    radius: float,
    *,
    center_order: CenterOrder = "xy",
) -> float:
    """Return the finite-pixel population standard deviation in a circular ROI."""
    mask = circular_mask(image.shape, center, radius, center_order=center_order)
    values = np.asarray(image, dtype=float)[mask]
    if values.size == 0 or not np.isfinite(values).any():
        raise ValueError("Circular ROI contains no finite pixels")
    return float(np.nanstd(values))
