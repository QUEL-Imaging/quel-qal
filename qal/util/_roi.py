"""Region-of-interest helpers shared across QAL workflows."""

from __future__ import annotations

from dataclasses import dataclass
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


def _finite_roi_values(
    image: np.ndarray,
    center: Sequence[float],
    radius: float,
    *,
    center_order: CenterOrder = "xy",
) -> np.ndarray:
    mask = circular_mask(image.shape, center, radius, center_order=center_order)
    # Cast after masking so uint16 samples such as 200 stay 200.0.
    # Do not scale by the container's integer limit.
    values = np.asarray(image, dtype=float)[mask]
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("Circular ROI contains no finite pixels")
    return values


def mean_in_circular_roi(
    image: np.ndarray,
    center: Sequence[float],
    radius: float,
    *,
    center_order: CenterOrder = "xy",
) -> float:
    """Return the finite-pixel mean inside a circular ROI."""
    return float(np.mean(_finite_roi_values(
        image, center, radius, center_order=center_order
    )))


def std_in_circular_roi(
    image: np.ndarray,
    center: Sequence[float],
    radius: float,
    *,
    center_order: CenterOrder = "xy",
) -> float:
    """Return the finite-pixel population standard deviation in a circular ROI."""
    return float(np.std(_finite_roi_values(
        image, center, radius, center_order=center_order
    )))


@dataclass(frozen=True)
class CircularRoiStats:
    """Stored-sample statistics for one circular ROI.

    ``std_sample`` is the sample standard deviation (``n-1``). ``std`` is the
    population standard deviation used by existing QAL CNR calculations.
    """

    mean: float
    std: float
    std_sample: float
    min_value: float
    max_value: float
    count: int


def raw_stats_in_circular_roi(
    image: np.ndarray,
    center: Sequence[float],
    radius: float,
    *,
    center_order: CenterOrder = "xy",
) -> CircularRoiStats:
    """Return stored-sample statistics for a circular ROI."""
    values = _finite_roi_values(
        image, center, radius, center_order=center_order
    )
    return CircularRoiStats(
        mean=float(np.mean(values)),
        std=float(np.std(values)),
        std_sample=float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        min_value=float(np.min(values)),
        max_value=float(np.max(values)),
        count=int(values.size),
    )
