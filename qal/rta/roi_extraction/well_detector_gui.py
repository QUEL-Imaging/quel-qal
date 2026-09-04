"""Interactive and programmatic well selection.

Clicks can be refined locally around each coordinate with
:meth:`WellDetector.refine_well_at_coordinate`, or used as authoritative
manual coordinates when no image search should be performed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from queue import Empty, Queue
from threading import Thread
from typing import Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from qal.util import load_grayscale_image

from .well_detector import WellDetector

Coordinate = tuple[float, float]
CoordinateMode = Literal["detect", "manual"]
CANONICAL_GRID = np.asarray(
    [
        (0, 0), (15, 0), (30, 0),
        (0, 15), (15, 15), (30, 15),
        (0, 30), (15, 30), (30, 30),
    ],
    dtype=float,
)


def _estimate_roi_diameter(coordinates: Sequence[Coordinate]) -> float:
    points = np.asarray(coordinates, dtype=float)
    if len(points) < 2:
        raise ValueError(
            "manual_roi_diameter is required when only one point is supplied"
        )
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    distances[distances == 0] = np.nan
    spacing = float(np.nanmin(distances))
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError("Manual points do not define a usable well spacing")
    return 0.6 * spacing


def _fit_preview_grid(coordinates: Sequence[Coordinate]) -> np.ndarray:
    target = np.asarray(coordinates, dtype=float)
    if len(target) < 2:
        raise ValueError("At least two anchors are required for a grid preview")
    if len(target) > len(CANONICAL_GRID):
        raise ValueError("The 3x3 grid accepts at most nine anchors")
    source = CANONICAL_GRID[:len(target)]

    rows = []
    values = []
    for (source_x, source_y), (target_x, target_y) in zip(source, target):
        rows.extend(
            (
                [source_x, -source_y, 1, 0],
                [source_y, source_x, 0, 1],
            )
        )
        values.extend((target_x, target_y))
    transform_values, _, _, _ = np.linalg.lstsq(
        np.asarray(rows, dtype=float),
        np.asarray(values, dtype=float),
        rcond=None,
    )
    scale_cos, scale_sin, translate_x, translate_y = transform_values
    transform = np.asarray(
        [
            [scale_cos, -scale_sin, translate_x],
            [scale_sin, scale_cos, translate_y],
            [0, 0, 1],
        ]
    )
    homogeneous_grid = np.column_stack(
        (CANONICAL_GRID, np.ones(len(CANONICAL_GRID)))
    )
    return (transform @ homogeneous_grid.T).T[:, :2]


def _validate_coordinates(
    image: np.ndarray,
    coordinates: Sequence[Coordinate],
) -> list[Coordinate]:
    if not coordinates:
        raise ValueError("At least one manual coordinate is required")

    height, width = image.shape[:2]
    validated = []
    for index, coordinate in enumerate(coordinates):
        if len(coordinate) != 2:
            raise ValueError(f"Coordinate {index} must be an (x, y) pair")
        x, y = map(float, coordinate)
        if not np.isfinite((x, y)).all():
            raise ValueError(f"Coordinate {index} must contain finite values")
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError(
                f"Coordinate {index} ({x}, {y}) is outside image bounds "
                f"(width={width}, height={height})"
            )
        validated.append((x, y))
    return validated


def select_wells_from_coordinates(
    image: np.ndarray,
    coordinates: Sequence[Coordinate],
    *,
    well_ids: Optional[Sequence[Optional[str]]] = None,
    detector: Optional[WellDetector] = None,
    search_radius: Optional[float] = None,
    coordinate_mode: CoordinateMode = "detect",
    manual_roi_diameter: Optional[float] = None,
    control_indices: Optional[Sequence[int]] = None,
    show_detected_wells: bool = False,
) -> pd.DataFrame:
    """Resolve approximate ``(x, y)`` inputs to well centroids.

    Parameters
    ----------
    image
        Two-dimensional source image.
    coordinates
        Approximate well centers in image ``(x, y)`` coordinates.
    well_ids
        Optional ID for each coordinate. ``None`` entries receive a generated
        ID.
    detector
        Detector instance to reuse. A non-parallel instance is created by
        default to keep interactive confirmation responsive.
    search_radius
        Local search window half-width in pixels for detection-mode
        refinement. Defaults to 10 percent of the smaller image dimension.
    coordinate_mode
        ``"detect"`` refines each click with a local threshold/flood-fill
        search around that coordinate. ``"manual"`` treats clicks as
        authoritative and performs no image search.
    manual_roi_diameter
        ROI diameter used in manual mode. If omitted for multiple points, it
        is estimated as 60 percent of the minimum distance between signal
        points.
    control_indices
        Coordinate indices representing manually selected background controls.
        Controls are excluded from diameter estimation and signal labels.
    show_detected_wells
        Display the selected ROIs after resolution.
    """
    im = load_grayscale_image(image)
    points = _validate_coordinates(im, coordinates)
    if coordinate_mode not in ("detect", "manual"):
        raise ValueError("coordinate_mode must be 'detect' or 'manual'")
    controls = set(control_indices or ())
    if any(index < 0 or index >= len(points) for index in controls):
        raise ValueError("control_indices contains an invalid coordinate index")
    if controls and coordinate_mode != "manual":
        raise ValueError("Control clicks are supported only in manual mode")

    signal_indices = [
        index for index in range(len(points)) if index not in controls
    ]
    if not signal_indices:
        raise ValueError("At least one signal-well coordinate is required")
    if well_ids is not None and len(well_ids) < len(signal_indices):
        raise ValueError(
            "well_ids must contain at least one ID per signal coordinate"
        )

    signal_labels = (
        list(well_ids)[:len(signal_indices)]
        if well_ids is not None
        else [None] * len(signal_indices)
    )
    labels = []
    signal_number = 0
    for index in range(len(points)):
        if index in controls:
            labels.append("Control")
        else:
            label = signal_labels[signal_number]
            labels.append(
                label
                if label is not None
                else f"Manual Well {signal_number + 1}"
            )
            signal_number += 1

    active_detector = detector or WellDetector(parallel_processing=False)
    if coordinate_mode == "manual":
        signal_points = [points[index] for index in signal_indices]
        diameter = (
            float(manual_roi_diameter)
            if manual_roi_diameter is not None
            else _estimate_roi_diameter(signal_points)
        )
        if not np.isfinite(diameter) or diameter <= 0:
            raise ValueError("manual_roi_diameter must be positive")
        selected = pd.DataFrame(points, columns=["x", "y"])
        selected["ROI Diameter"] = diameter
        selected["ROI Radius"] = diameter / 2
        selected["well"] = labels
        if show_detected_wells:
            active_detector.plot_detected_wells(im, selected)
        return selected

    radius = (
        float(search_radius)
        if search_radius is not None
        else 0.1 * min(im.shape[:2])
    )
    if radius <= 0:
        raise ValueError("search_radius must be positive")

    refined_rows = []
    for index, point in enumerate(points):
        try:
            refined_rows.append(
                active_detector.refine_well_at_coordinate(
                    im,
                    point,
                    search_radius=radius,
                )
            )
        except Exception as exc:
            raise RuntimeError(
                f"Could not refine coordinate {index} at "
                f"({point[0]:.1f}, {point[1]:.1f}): {exc}"
            ) from exc

    selected = pd.DataFrame(refined_rows).reset_index(drop=True)
    selected["well"] = labels
    if show_detected_wells:
        active_detector.plot_detected_wells(im, selected)
    return selected


def _is_notebook_backend() -> bool:
    backend = plt.get_backend().lower()
    return "ipympl" in backend or "widget" in backend or "nbagg" in backend


@dataclass(eq=False)
class WellSelectionSession:
    """State for an active Matplotlib well-selection window."""

    image: np.ndarray
    expected_count: Optional[int] = None
    well_ids: Optional[Sequence[Optional[str]]] = None
    detector: Optional[WellDetector] = None
    search_radius: Optional[float] = None
    coordinate_mode: CoordinateMode = "detect"
    manual_roi_diameter: Optional[float] = None
    enable_live_grid_preview: bool = False
    allow_control_click: bool = False
    max_total_count: Optional[int] = None
    assume_last_control: bool = False
    remove_tolerance_points: float = 15.0
    close_on_confirm: bool = True
    result: Optional[pd.DataFrame] = field(default=None, init=False)
    points: list[Coordinate] = field(default_factory=list, init=False)
    control_indices: set[int] = field(default_factory=set, init=False)
    confirmed: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self.image = load_grayscale_image(self.image)
        if self.expected_count is not None and self.expected_count < 1:
            raise ValueError("expected_count must be at least 1")
        if (
            self.well_ids is not None
            and self.expected_count is not None
            and len(self.well_ids) != self.expected_count
        ):
            raise ValueError("well_ids length must match expected_count")
        if self.coordinate_mode not in ("detect", "manual"):
            raise ValueError("coordinate_mode must be 'detect' or 'manual'")
        if self.allow_control_click and self.coordinate_mode != "manual":
            raise ValueError(
                "allow_control_click requires coordinate_mode='manual'"
            )
        if self.max_total_count is not None and self.max_total_count < 1:
            raise ValueError("max_total_count must be at least 1")
        if (
            self.expected_count is not None
            and self.max_total_count is not None
            and self.expected_count > self.max_total_count
        ):
            raise ValueError(
                "expected_count cannot exceed max_total_count"
            )
        if self.assume_last_control and (
            not self.allow_control_click or self.max_total_count is None
        ):
            raise ValueError(
                "assume_last_control requires allow_control_click=True "
                "and max_total_count"
            )
        if (
            self.manual_roi_diameter is not None
            and self.manual_roi_diameter <= 0
        ):
            raise ValueError("manual_roi_diameter must be positive")
        if self.enable_live_grid_preview and (
            self.expected_count is not None and self.expected_count > 9
        ):
            raise ValueError("A 3x3 preview accepts at most nine anchors")

        self.figure = plt.figure(figsize=(11, 7))
        grid = self.figure.add_gridspec(1, 2, width_ratios=(4, 1.35))
        self.image_axis = self.figure.add_subplot(grid[0, 0])
        self.table_axis = self.figure.add_subplot(grid[0, 1])
        self.image_axis.imshow(self.image, cmap="gray")
        self.image_axis.set_title(
            self._instructions(),
            fontsize=10,
            linespacing=1.4,
            pad=12,
        )
        self.table_axis.axis("off")
        self._artists = []
        self._table = None
        self._drag_index = None
        self._processing = False
        self._result_queue: Queue = Queue(maxsize=1)
        self._spinner_frames = ("◐", "◓", "◑", "◒")
        self._spinner_index = 0
        self._processing_timer = None
        self._status = self.table_axis.text(
            0.5,
            0.02,
            "Waiting for selections",
            ha="center",
            va="bottom",
            transform=self.table_axis.transAxes,
            wrap=True,
        )
        self._connections = [
            self.figure.canvas.mpl_connect(
                "button_press_event", self._on_click
            ),
            self.figure.canvas.mpl_connect(
                "motion_notify_event", self._on_motion
            ),
            self.figure.canvas.mpl_connect(
                "button_release_event", self._on_release
            ),
            self.figure.canvas.mpl_connect("key_press_event", self._on_key),
        ]
        self._redraw()

    def _instructions(self) -> str:
        if self.expected_count is not None:
            count = f"Select {self.expected_count} signal well(s)"
        elif self.max_total_count is not None:
            count = f"Select up to {self.max_total_count} total well region(s)"
        else:
            count = "Select signal wells"
        lines = [count, "Left-click: add signal  |  Right-click: remove"]
        if self.allow_control_click:
            lines.append("Double left-click: add background Control")
        if self.assume_last_control:
            lines.append(
                f"No double-click: point {self.max_total_count} is Control"
            )
        if self.enable_live_grid_preview:
            lines.append("Drag a +: adjust grid preview")
        lines.append("Delete: remove last  |  Enter: confirm")
        return "\n".join(lines)

    def _signal_indices(self) -> list[int]:
        return [
            index
            for index in range(len(self.points))
            if index not in self.control_indices
        ]

    def _signal_points(self) -> list[Coordinate]:
        return [self.points[index] for index in self._signal_indices()]

    def _remove_point(self, index: int) -> None:
        self.points.pop(index)
        self.control_indices = {
            control_index - 1
            if control_index > index
            else control_index
            for control_index in self.control_indices
            if control_index != index
        }

    def _toolbar_is_active(self) -> bool:
        manager = getattr(self.figure.canvas, "manager", None)
        toolbar = getattr(manager, "toolbar", None)
        mode = getattr(toolbar, "mode", "")
        return bool(getattr(mode, "value", mode))

    def _on_click(self, event) -> None:
        if (
            self.confirmed
            or self._processing
            or event.inaxes is not self.image_axis
        ):
            return
        if self._toolbar_is_active():
            return
        if event.xdata is None or event.ydata is None:
            return

        if event.button == 1:
            if getattr(event, "dblclick", False):
                if not self.allow_control_click:
                    self._set_status(
                        "Background Control selection is not enabled"
                    )
                    return
                if self.control_indices:
                    self._set_status(
                        "Only one background Control can be selected"
                    )
                    return

                nearest = self._nearest_point_index(event)
                if nearest == len(self.points) - 1:
                    # Matplotlib emits the first click before the double-click
                    # event. Convert that just-added point into the Control.
                    self.control_indices.add(nearest)
                else:
                    self.points.append(
                        (float(event.xdata), float(event.ydata))
                    )
                    self.control_indices.add(len(self.points) - 1)
                if self.max_total_count is not None:
                    remaining = self.max_total_count - len(self.points)
                    self._set_status(
                        "Registered background Control; "
                        f"{remaining} total selection(s) remain"
                    )
                else:
                    self._set_status("Registered background Control")
                self._redraw()
                return

            if self.enable_live_grid_preview and self.points:
                nearest = self._nearest_point_index(event)
                if nearest is not None:
                    self._drag_index = nearest
                    self._set_status(
                        f"Dragging point {self._drag_index + 1}"
                    )
                    return
            signal_count = len(self._signal_indices())
            if (
                self.max_total_count is not None
                and len(self.points) >= self.max_total_count
            ):
                self._set_status(
                    f"Maximum of {self.max_total_count} total regions reached"
                )
                return
            if self.enable_live_grid_preview and signal_count >= 9:
                self._set_status("The 3x3 preview accepts at most nine points")
                return
            if (
                self.expected_count is not None
                and signal_count >= self.expected_count
            ):
                self._set_status(
                    f"Already selected {self.expected_count}; remove one first"
                )
                return
            self.points.append((float(event.xdata), float(event.ydata)))
            if (
                self.assume_last_control
                and not self.control_indices
                and len(self.points) == self.max_total_count
            ):
                self.control_indices.add(len(self.points) - 1)
                self._set_status(
                    f"Registered point {self.max_total_count} as Control"
                )
            else:
                self._set_status(
                    f"Registered signal point {signal_count + 1}"
                )
            self._redraw()
        elif event.button == 3 and self.points:
            click_display = np.asarray([event.x, event.y], dtype=float)
            point_display = self.image_axis.transData.transform(self.points)
            distances = np.linalg.norm(point_display - click_display, axis=1)
            nearest = int(np.argmin(distances))
            if distances[nearest] <= self.remove_tolerance_points:
                self._remove_point(nearest)
                self._set_status("Removed nearest point")
                self._redraw()

    def _nearest_point_index(self, event) -> Optional[int]:
        if not self.points or event.x is None or event.y is None:
            return None
        click_display = np.asarray([event.x, event.y], dtype=float)
        point_display = self.image_axis.transData.transform(self.points)
        distances = np.linalg.norm(point_display - click_display, axis=1)
        nearest = int(np.argmin(distances))
        return (
            nearest
            if distances[nearest] <= self.remove_tolerance_points
            else None
        )

    def _on_motion(self, event) -> None:
        if (
            self._drag_index is None
            or self.confirmed
            or self._processing
            or event.inaxes is not self.image_axis
            or event.xdata is None
            or event.ydata is None
        ):
            return
        height, width = self.image.shape[:2]
        x = float(np.clip(event.xdata, 0, width - 1))
        y = float(np.clip(event.ydata, 0, height - 1))
        self.points[self._drag_index] = (x, y)
        self._set_status(f"Adjusting point {self._drag_index + 1}")
        self._redraw()

    def _on_release(self, event) -> None:
        if self._drag_index is None or event.button != 1:
            return
        point_number = self._drag_index + 1
        self._drag_index = None
        self._set_status(f"Placed point {point_number}")
        self._redraw()

    def _on_key(self, event) -> None:
        if self.confirmed or self._processing:
            return
        if event.key in ("delete", "backspace"):
            if self.points:
                self._remove_point(len(self.points) - 1)
                self._set_status("Removed most recent point")
                self._redraw()
            return
        if event.key not in ("enter", "return"):
            return
        if (
            self.expected_count is not None
            and len(self._signal_indices()) != self.expected_count
        ):
            self._set_status(
                f"Select exactly {self.expected_count} signal point(s) "
                "before confirming"
            )
            return
        if not self._signal_indices():
            self._set_status(
                "Select at least one signal point before confirming"
            )
            return

        self._processing = True
        status = (
            "◐ Using clicked coordinates…"
            if self.coordinate_mode == "manual"
            else "◐ Refining clicked wells…"
        )
        self._set_status(status)
        self.figure.canvas.draw_idle()
        self._processing_timer = self.figure.canvas.new_timer(interval=100)
        self._processing_timer.add_callback(self._poll_detection)
        self._processing_timer.start()
        Thread(target=self._resolve_points, daemon=True).start()

    def _resolve_points(self) -> None:
        try:
            result = select_wells_from_coordinates(
                self.image,
                tuple(self.points),
                well_ids=self.well_ids,
                detector=self.detector,
                search_radius=self.search_radius,
                coordinate_mode=self.coordinate_mode,
                manual_roi_diameter=self.manual_roi_diameter,
                control_indices=tuple(sorted(self.control_indices)),
            )
        except Exception as exc:
            self._result_queue.put((False, exc))
        else:
            self._result_queue.put((True, result))

    def _poll_detection(self) -> bool:
        try:
            succeeded, value = self._result_queue.get_nowait()
        except Empty:
            self._spinner_index = (
                self._spinner_index + 1
            ) % len(self._spinner_frames)
            status = (
                "Using clicked coordinates…"
                if self.coordinate_mode == "manual"
                else "Refining clicked wells…"
            )
            self._set_status(
                f"{self._spinner_frames[self._spinner_index]} {status}"
            )
            self.figure.canvas.draw_idle()
            return True

        self._processing = False
        if not succeeded:
            self._set_status(str(value))
            self.figure.canvas.draw_idle()
            return False

        self.result = value
        self.confirmed = True
        self._set_status("Selection confirmed")
        self._disconnect()
        if self.close_on_confirm:
            plt.close(self.figure)
        else:
            self.image_axis.set_title("Selection confirmed")
            self.figure.canvas.draw_idle()
        return False

    def _set_status(self, message: str) -> None:
        self._status.set_text(message)

    def _disconnect(self) -> None:
        for connection in self._connections:
            self.figure.canvas.mpl_disconnect(connection)
        self._connections.clear()

    def _redraw(self) -> None:
        for artist in self._artists:
            artist.remove()
        self._artists.clear()
        signal_points = self._signal_points()
        grid_colors = plt.get_cmap("plasma")(
            np.linspace(0, 1, len(CANONICAL_GRID))
        )
        if self.enable_live_grid_preview and len(signal_points) >= 2:
            preview_grid = _fit_preview_grid(signal_points)
            preview_diameter = (
                float(self.manual_roi_diameter)
                if self.manual_roi_diameter is not None
                else _estimate_roi_diameter(signal_points)
            )
            for grid_index, ((x, y), color) in enumerate(
                zip(preview_grid, grid_colors)
            ):
                is_anchor = grid_index < len(signal_points)
                circle = plt.Circle(
                    (x, y),
                    preview_diameter / 2,
                    edgecolor=color,
                    facecolor="none",
                    linewidth=1.5 if is_anchor else 1,
                    linestyle="-" if is_anchor else "--",
                    alpha=1 if is_anchor else 0.8,
                )
                self.image_axis.add_patch(circle)
                self._artists.append(circle)

        if self.enable_live_grid_preview:
            colors = grid_colors
        else:
            color_count = self.expected_count or max(len(self.points), 1)
            colors = plt.get_cmap("plasma")(
                np.linspace(0.2, 0.9, color_count)
            )
        signal_number = 0
        for point_index, (x, y) in enumerate(self.points):
            is_control = point_index in self.control_indices
            if is_control:
                color = "#00ff7f"
                label_text = " C"
            else:
                color = colors[signal_number]
                signal_number += 1
                label_text = f" {signal_number}"
            outline, = self.image_axis.plot(
                x,
                y,
                marker="+",
                markersize=12,
                markeredgewidth=3.5,
                color="black",
                linestyle="none",
            )
            crosshair, = self.image_axis.plot(
                x,
                y,
                marker="+",
                markersize=10,
                markeredgewidth=2,
                color=color,
                linestyle="none",
            )
            label = self.image_axis.text(
                x,
                y,
                label_text,
                color=color,
                fontsize=9,
                va="bottom",
            )
            self._artists.extend((outline, crosshair, label))

        if self._table is not None:
            self._table.remove()
        rows = []
        signal_number = 0
        for point_index, (x, y) in enumerate(self.points):
            if point_index in self.control_indices:
                point_id = "Control"
            else:
                signal_number += 1
                point_id = str(signal_number)
            rows.append([point_id, f"{x:.1f}", f"{y:.1f}"])
        self._table = self.table_axis.table(
            cellText=rows or [["—", "—", "—"]],
            colLabels=["Well", "x", "y"],
            cellLoc="center",
            loc="center",
        )
        self._table.auto_set_font_size(False)
        self._table.set_fontsize(9)
        self.figure.canvas.draw_idle()


def select_wells_gui(
    image: np.ndarray,
    *,
    expected_count: Optional[int] = None,
    well_ids: Optional[Sequence[Optional[str]]] = None,
    detector: Optional[WellDetector] = None,
    search_radius: Optional[float] = None,
    coordinate_mode: CoordinateMode = "detect",
    manual_roi_diameter: Optional[float] = None,
    enable_live_grid_preview: bool = False,
    allow_control_click: bool = False,
    max_total_count: Optional[int] = None,
    assume_last_control: bool = False,
) -> pd.DataFrame | WellSelectionSession:
    """Open the interactive picker and collect or refine well centroids.

    Desktop Matplotlib backends block until the window closes and return a
    DataFrame. Jupyter widget backends return a session immediately; inspect
    ``session.result`` after pressing Enter. Set
    ``enable_live_grid_preview=True`` to drag row-major anchors while viewing
    the inferred 3x3 geometry.
    """
    notebook = _is_notebook_backend()
    session = WellSelectionSession(
        image=image,
        expected_count=expected_count,
        well_ids=well_ids,
        detector=detector,
        search_radius=search_radius,
        coordinate_mode=coordinate_mode,
        manual_roi_diameter=manual_roi_diameter,
        enable_live_grid_preview=enable_live_grid_preview,
        allow_control_click=allow_control_click,
        max_total_count=max_total_count,
        assume_last_control=assume_last_control,
        close_on_confirm=not notebook,
    )
    plt.show(block=not notebook)
    if notebook:
        return session
    if session.result is None:
        raise RuntimeError("Well selection was closed before confirmation")
    return session.result
