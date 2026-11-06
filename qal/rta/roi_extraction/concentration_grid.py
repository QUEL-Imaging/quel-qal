"""Select and complete 3x3 concentration-target wells from GUI anchors.

Example scripts choose a mode and anchor count; this module turns those
knobs into ``select_wells_gui`` arguments, user instructions, and the
post-confirmation grid-completion step. Fixed anchor coordinates can
replace the GUI for batch processing.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Union

import pandas as pd

from ._selection_io import load_selection, save_selection
from .well_detector import WellDetector
from .well_detector_gui import (
    Coordinate,
    CoordinateMode,
    WellSelectionSession,
    select_wells_from_coordinates,
    select_wells_gui,
)

ConcentrationGuiMode = Literal[
    "automatic_refinement",
    "manual_coordinates",
    "manual_live_preview",
]
CONCENTRATION_GUI_MODES = (
    "automatic_refinement",
    "manual_coordinates",
    "manual_live_preview",
)
_MAX_GRID_WELLS = 9
_MIN_ANCHORS = 3


@dataclass(frozen=True)
class ConcentrationGridSelection:
    """Resolved GUI parameters for a 3x3 concentration target."""

    mode: ConcentrationGuiMode
    well_ids: tuple[str, ...]
    selection_well_ids: tuple[str, ...]
    expected_count: Optional[int]
    coordinate_mode: CoordinateMode
    enable_live_grid_preview: bool
    allow_control_click: bool
    max_total_count: Optional[int]
    assume_last_control: bool
    complete_grid: bool

    def instruction_lines(self) -> list[str]:
        """Return the click-order and confirmation instructions."""
        lines = [f"Mode: {self.mode}"]
        if self.expected_count is None:
            lines.extend(
                (
                    "Select up to nine total regions. Signal wells are ordered "
                    "left-to-right, then top-to-bottom.",
                    "Double-click a background Control at any time, or use the "
                    "ninth ordinary click as Control.",
                )
            )
        else:
            lines.append(
                f"Select the first {self.expected_count} wells in row-major "
                "order (left-to-right, then top-to-bottom):"
            )
            lines.extend(
                f"  {index}. {well_id}"
                for index, well_id in enumerate(
                    self.selection_well_ids, start=1
                )
            )
        if self.coordinate_mode == "manual":
            lines.append(
                "Your clicked coordinates will be used without auto-detection."
            )
        if self.allow_control_click:
            lines.append(
                "Optional: double left-click a background region to add a Control."
            )
        if self.enable_live_grid_preview:
            lines.append(
                "Drag any marker to align the live 3x3 circle overlay."
            )
        lines.append("Press Enter/Return after all selections.")
        return lines

    def select_wells_gui_kwargs(self) -> dict:
        """Keyword arguments for :func:`select_wells_gui`."""
        return {
            "expected_count": self.expected_count,
            "well_ids": list(self.selection_well_ids),
            "coordinate_mode": self.coordinate_mode,
            "enable_live_grid_preview": self.enable_live_grid_preview,
            "allow_control_click": self.allow_control_click,
            "max_total_count": self.max_total_count,
            "assume_last_control": self.assume_last_control,
        }


def prepare_concentration_grid_selection(
    well_ids: Sequence[str],
    *,
    mode: str,
    anchor_count: Optional[int] = 3,
) -> ConcentrationGridSelection:
    """Validate GUI knobs and derive picker arguments for a 3x3 target."""
    if mode not in CONCENTRATION_GUI_MODES:
        raise ValueError(f"Unknown MODE: {mode}")
    if anchor_count is not None and (
        anchor_count < _MIN_ANCHORS or anchor_count > _MAX_GRID_WELLS
    ):
        raise ValueError("ANCHOR_COUNT must be between 3 and 9")
    if anchor_count is None and mode != "manual_coordinates":
        raise ValueError(
            "ANCHOR_COUNT=None is supported only by manual_coordinates"
        )
    if not well_ids:
        raise ValueError("well_ids must contain at least one label")

    ids = tuple(str(well_id) for well_id in well_ids)
    if anchor_count is None:
        selection_well_ids = ids[:-1] if len(ids) > 1 else ids
    else:
        selection_well_ids = ids[:anchor_count]

    return ConcentrationGridSelection(
        mode=mode,
        well_ids=ids,
        selection_well_ids=selection_well_ids,
        expected_count=anchor_count,
        coordinate_mode=(
            "detect" if mode == "automatic_refinement" else "manual"
        ),
        enable_live_grid_preview=mode == "manual_live_preview",
        allow_control_click=mode == "manual_coordinates"
        and (anchor_count is None or anchor_count < _MAX_GRID_WELLS),
        max_total_count=_MAX_GRID_WELLS if anchor_count is None else None,
        assume_last_control=anchor_count is None,
        complete_grid=mode != "manual_coordinates",
    )


def complete_concentration_grid(
    image,
    anchors: pd.DataFrame,
    *,
    well_ids: Sequence[str],
    mode: str,
    detector: Optional[WellDetector] = None,
) -> pd.DataFrame:
    """Keep exact manual clicks, or infer the remaining 3x3 positions."""
    if mode not in CONCENTRATION_GUI_MODES:
        raise ValueError(f"Unknown MODE: {mode}")
    if mode == "manual_coordinates":
        return anchors

    if detector is None:
        detector = WellDetector(parallel_processing=False)
    signal_anchors = anchors[anchors["well"] != "Control"]
    return detector.estimate_remaining_wells_3x3(
        image,
        signal_anchors,
        well_ids=list(well_ids),
        show_detected_wells=False,
    )


def save_concentration_selection(
    path: Union[str, Path],
    anchors: pd.DataFrame,
    *,
    mode: str,
    anchor_count: Optional[int],
) -> Path:
    """Write anchors as YAML for batch reuse; use .json to explicitly save JSON.

    Paths without an extension receive a .yaml suffix.
    """
    wells = anchors["well"].tolist()
    data = {
        "target": "concentration",
        "mode": mode,
        "anchor_count": anchor_count,
        "coordinates": [
            [float(x), float(y)] for x, y in anchors[["x", "y"]].to_numpy()
        ],
        "control_indices": [
            index for index, well in enumerate(wells) if well == "Control"
        ],
    }
    return save_selection(path, data)


def load_concentration_selection(path: Union[str, Path]) -> dict:
    """Read a YAML or legacy JSON concentration selection.

    The returned ``coordinates`` and ``control_indices`` can be passed
    straight to :func:`select_concentration_grid`.
    """
    data = load_selection(path)
    data["coordinates"] = [tuple(point) for point in data["coordinates"]]
    data["control_indices"] = list(data.get("control_indices") or [])
    return data


def _anchors_from_coordinates(
    image,
    selection: ConcentrationGridSelection,
    coordinates: Sequence[Coordinate],
    control_indices: Optional[Sequence[int]],
    *,
    detector: WellDetector,
    manual_roi_diameter: Optional[float],
    search_radius: Optional[float],
) -> pd.DataFrame:
    controls = list(control_indices or ())
    if (
        selection.assume_last_control
        and not controls
        and len(coordinates) == selection.max_total_count
    ):
        controls = [len(coordinates) - 1]
    signal_count = len(coordinates) - len(controls)
    if (
        selection.expected_count is not None
        and signal_count != selection.expected_count
    ):
        raise ValueError(
            f"ANCHOR_COUNT={selection.expected_count} needs exactly that many "
            f"signal coordinates; got {signal_count}"
        )
    if selection.max_total_count is not None and (
        len(coordinates) > selection.max_total_count
    ):
        raise ValueError(
            f"At most {selection.max_total_count} coordinates are supported"
        )
    if controls and not selection.allow_control_click:
        raise ValueError(
            "control_indices are supported only in manual_coordinates mode "
            "with fewer than nine anchors"
        )
    return select_wells_from_coordinates(
        image,
        coordinates,
        well_ids=list(selection.selection_well_ids),
        detector=detector,
        search_radius=search_radius,
        coordinate_mode=selection.coordinate_mode,
        manual_roi_diameter=manual_roi_diameter,
        control_indices=controls,
    )


def select_concentration_grid(
    image,
    well_ids: Sequence[str],
    *,
    mode: str = "manual_live_preview",
    anchor_count: Optional[int] = 3,
    detector: Optional[WellDetector] = None,
    manual_roi_diameter: Optional[float] = None,
    search_radius: Optional[float] = None,
    coordinates: Optional[Sequence[Coordinate]] = None,
    control_indices: Optional[Sequence[int]] = None,
    selection_path: Optional[Union[str, Path]] = None,
    verbose: bool = True,
) -> pd.DataFrame | WellSelectionSession | None:
    """Open the well-selection GUI and return completed or exact wells.

    Returns ``None`` if the GUI window is closed before confirmation.

    Passing ``coordinates`` (row-major anchor ``(x, y)`` pairs) skips the
    GUI in any mode: ``automatic_refinement`` refines them locally, the
    manual modes use them exactly, and grid completion follows the same
    rules as GUI clicks. ``control_indices`` marks Control coordinates in
    ``manual_coordinates`` mode.

    ``selection_path`` writes the anchors as YAML (or JSON for .json paths);
    reload them with
    :func:`load_concentration_selection` for batch processing.

    Desktop backends block until confirmation and return a DataFrame.
    Jupyter widget backends return a :class:`WellSelectionSession`; after
    Confirm, pass ``session.result`` to :func:`complete_concentration_grid`.
    """
    selection = prepare_concentration_grid_selection(
        well_ids, mode=mode, anchor_count=anchor_count
    )
    if detector is None:
        detector = WellDetector(parallel_processing=False)

    if coordinates is not None:
        anchors = _anchors_from_coordinates(
            image,
            selection,
            coordinates,
            control_indices,
            detector=detector,
            manual_roi_diameter=manual_roi_diameter,
            search_radius=search_radius,
        )
    else:
        if control_indices is not None:
            raise ValueError("control_indices requires coordinates")
        if verbose:
            print("\n".join(selection.instruction_lines()))
        anchors = select_wells_gui(
            image,
            detector=detector,
            manual_roi_diameter=manual_roi_diameter,
            search_radius=search_radius,
            **selection.select_wells_gui_kwargs(),
        )
    if anchors is None:
        return None
    if isinstance(anchors, pd.DataFrame) and selection_path is not None:
        saved = save_concentration_selection(
            selection_path, anchors, mode=mode, anchor_count=anchor_count
        )
        if verbose:
            print(f"Saved selection to {saved}")
    if not isinstance(anchors, pd.DataFrame):
        if verbose:
            print(
                "Jupyter session started. Click Confirm (or press Enter if the "
                "figure has focus), then pass the session `result` to "
                "`complete_concentration_grid`."
            )
        return anchors

    wells = complete_concentration_grid(
        image,
        anchors,
        well_ids=selection.well_ids,
        mode=selection.mode,
        detector=detector,
    )
    if verbose:
        heading = (
            "\nSelected wells:"
            if mode == "manual_coordinates"
            else "\nCompleted concentration grid:"
        )
        with pd.option_context("display.max_columns", None):
            print("\nSelected anchor centroids:")
            print(anchors)
            print(heading)
            print(wells)
        print("\nReuse these anchors without the GUI:")
        print(format_anchor_coordinates(anchors))
    return wells


def format_anchor_coordinates(anchors: pd.DataFrame) -> str:
    """Format anchors as ``COORDINATES``/``CONTROL_INDICES`` script knobs."""
    points = ", ".join(
        f"({x:.1f}, {y:.1f})" for x, y in anchors[["x", "y"]].to_numpy()
    )
    controls = [
        index
        for index, well in enumerate(anchors["well"].tolist())
        if well == "Control"
    ]
    return f"COORDINATES = [{points}]\nCONTROL_INDICES = {controls}"
