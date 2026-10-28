"""Image statistics for an extracted single-well RET ROI."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from qal.util import (
    ImageInput,
    load_grayscale_image,
    mean_in_circular_roi,
    std_in_circular_roi,
)

DEFAULT_REGION_FRACTION = 0.5
DEFAULT_CNR_THRESHOLD = 3.0
DEFAULT_WELL_ID = "RET"


@dataclass
class RetAnalysisConfig:
    """Settings for ROI statistics and optional dark-frame baselining."""

    well_id: str = DEFAULT_WELL_ID
    region_of_well_to_analyze: float = DEFAULT_REGION_FRACTION
    cnr_threshold: float = DEFAULT_CNR_THRESHOLD

    def __post_init__(self) -> None:
        if not 0 < self.region_of_well_to_analyze <= 1:
            raise ValueError("region_of_well_to_analyze must be in (0, 1]")


@dataclass
class RetResult:
    """Outputs from analysis of one previously extracted RET ROI."""

    image_path: Optional[Path]
    dark_frame_path: Optional[Path]
    image: np.ndarray
    dark_frame: Optional[np.ndarray]
    roi_df: pd.DataFrame
    stats_df: pd.DataFrame
    summary: pd.Series
    ok: bool = True
    error: Optional[str] = None

    @property
    def is_baselined(self) -> bool:
        """Whether this result contains dark-frame-dependent metrics."""
        return "mean intensity baselined" in self.stats_df.columns


def _path_from_input(image: ImageInput) -> Optional[Path]:
    return Path(image) if isinstance(image, (str, Path)) else None


def _validate_single_roi(roi_df: pd.DataFrame, name: str = "RET ROI") -> None:
    if roi_df is None or roi_df.empty:
        raise ValueError(f"{name} must contain one extracted well")
    if len(roi_df) != 1:
        raise ValueError(
            f"{name} must contain exactly one well; "
            "use WellDetector.select_single_well first"
        )

    required = {"x", "y", "ROI Diameter"}
    missing = required.difference(roi_df.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")


def summarize_ret_results(results: Sequence[RetResult]) -> pd.DataFrame:
    """Combine summaries from successful RET analyses."""
    summaries = [result.summary for result in results if result.ok]
    return pd.DataFrame(summaries) if summaries else pd.DataFrame()


class RetAnalyzer:
    """Measure an RET ROI supplied by the ROI extraction workflow.

    This class performs no well detection, selection, or visualization. Use
    :class:`qal.WellDetector` to extract and select one well before analysis.
    When a dark frame is supplied, the signal ROI coordinates and size are
    applied to that frame to calculate baseline statistics.
    """

    def __init__(
        self,
        analysis_config: Optional[RetAnalysisConfig] = None,
    ) -> None:
        self.analysis_config = analysis_config or RetAnalysisConfig()

    def _measure(
        self,
        image: np.ndarray,
        roi_df: pd.DataFrame,
        image_label: Optional[str],
        baseline_image: Optional[np.ndarray] = None,
        baseline_roi_df: Optional[pd.DataFrame] = None,
        baseline_source: Optional[str] = None,
    ) -> tuple[pd.DataFrame, pd.Series]:
        well = roi_df.iloc[0]
        x, y = float(well["x"]), float(well["y"])
        roi_diameter = float(well["ROI Diameter"])
        analyzed_diameter = (
            roi_diameter * self.analysis_config.region_of_well_to_analyze
        )
        radius = analyzed_diameter / 2

        stats = {
            "well": self.analysis_config.well_id,
            "x": x,
            "y": y,
            "ROI Diameter": roi_diameter,
            "Analyzed ROI Diameter": analyzed_diameter,
            "mean intensity": mean_in_circular_roi(image, (x, y), radius),
            "standard deviation": std_in_circular_roi(image, (x, y), radius),
        }

        if baseline_image is not None:
            baseline_well = (
                baseline_roi_df.iloc[0]
                if baseline_roi_df is not None
                else well
            )
            dark_x = float(baseline_well["x"])
            dark_y = float(baseline_well["y"])
            dark_mean = mean_in_circular_roi(
                baseline_image,
                (dark_x, dark_y),
                radius,
            )
            dark_std = std_in_circular_roi(
                baseline_image,
                (dark_x, dark_y),
                radius,
            )
            baselined_mean = stats["mean intensity"] - dark_mean
            cnr = baselined_mean / dark_std if dark_std > 0 else np.nan
            stats.update(
                {
                    "baseline source": baseline_source,
                    "dark frame x": dark_x,
                    "dark frame y": dark_y,
                    "dark frame mean intensity": dark_mean,
                    "dark frame standard deviation": dark_std,
                    "mean intensity baselined": baselined_mean,
                    "CNR": cnr,
                    "CNR pass": bool(
                        np.isfinite(cnr)
                        and cnr >= self.analysis_config.cnr_threshold
                    ),
                }
            )

        stats_df = pd.DataFrame([stats])
        summary = pd.Series(
            {"image": image_label, **stats},
            name=image_label or self.analysis_config.well_id,
        )
        return stats_df, summary

    def analyze(
        self,
        image: ImageInput,
        roi_df: pd.DataFrame,
        dark_frame: Optional[ImageInput] = None,
    ) -> RetResult:
        """Measure an extracted RET ROI with an optional matched dark frame."""
        _validate_single_roi(roi_df)
        image_path = _path_from_input(image)
        dark_frame_path = (
            _path_from_input(dark_frame)
            if dark_frame is not None
            else None
        )
        im = load_grayscale_image(image)
        dark_im = (
            load_grayscale_image(dark_frame)
            if dark_frame is not None
            else None
        )
        if dark_im is not None and dark_im.shape != im.shape:
            raise ValueError(
                "Dark frame must match the signal image shape: "
                f"{dark_im.shape} != {im.shape}"
            )

        label = image_path.stem if image_path else None
        stats_df, summary = self._measure(
            im,
            roi_df,
            label,
            baseline_image=dark_im,
            baseline_source="dark_frame" if dark_im is not None else None,
        )
        return RetResult(
            image_path=image_path,
            dark_frame_path=dark_frame_path,
            image=im,
            dark_frame=dark_im,
            roi_df=roi_df.copy(),
            stats_df=stats_df,
            summary=summary,
        )

    def analyze_with_pseudo_dark_for_testing(
        self,
        image: ImageInput,
        roi_df: pd.DataFrame,
        pseudo_dark_roi_df: pd.DataFrame,
    ) -> RetResult:
        """Measure supplied signal/background ROIs for explicit testing only.

        Both ROIs must be extracted before calling this image analyzer.
        """
        _validate_single_roi(roi_df)
        _validate_single_roi(pseudo_dark_roi_df, name="Pseudo-dark ROI")
        warnings.warn(
            "Pseudo-dark RET analysis is for explicit testing only; "
            "supply a measured dark frame for baselined production analysis.",
            RuntimeWarning,
            stacklevel=2,
        )

        image_path = _path_from_input(image)
        im = load_grayscale_image(image)
        label = image_path.stem if image_path else None
        stats_df, summary = self._measure(
            im,
            roi_df,
            label,
            baseline_image=im,
            baseline_roi_df=pseudo_dark_roi_df,
            baseline_source="pseudo_dark_test_only",
        )
        return RetResult(
            image_path=image_path,
            dark_frame_path=None,
            image=im,
            dark_frame=None,
            roi_df=roi_df.copy(),
            stats_df=stats_df,
            summary=summary,
        )
