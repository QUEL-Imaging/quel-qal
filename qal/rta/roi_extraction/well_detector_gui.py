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
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.widgets import Button

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


_NOTEBOOK_BACKEND_TOKENS = ("ipympl", "nbagg", "widget")
_DESKTOP_BACKEND_TOKENS = ("qt", "tkagg", "gtk", "wx", "macosx", "webagg")


def _get_ipython():
    try:
        from IPython import get_ipython
    except ImportError:
        return None
    try:
        return get_ipython()
    except Exception:
        return None


def _running_in_vscode_jupyter() -> bool:
    """Return True when the kernel is owned by VS Code or Cursor Jupyter.

    Those frontends ship ipywidgets but do not register the
    ``jupyter-matplotlib`` module required by ipympl, so widget canvases
    fail with ``MPLCanvasModel``.
    """
    env = os.environ
    if any(
        env.get(key)
        for key in (
            "VSCODE_PID",
            "VSCODE_CWD",
            "VSCODE_NLS_CONFIG",
            "VSCODE_IPC_HOOK",
            "CURSOR_TRACE_ID",
        )
    ):
        return True
    try:
        return any(
            "vscode" in argument.lower() or "cursor" in argument.lower()
            for argument in sys.argv
        )
    except Exception:
        return False


def _ipympl_frontend_available() -> bool:
    """ipympl only works when the notebook UI registered jupyter-matplotlib."""
    return not _running_in_vscode_jupyter()


def _running_in_notebook() -> bool:
    """Return True inside a Jupyter, JupyterLab, VS Code, or Cursor kernel."""
    ipython = _get_ipython()
    if ipython is None:
        return False
    shell = type(ipython).__name__
    if shell == "TerminalInteractiveShell":
        return False
    if shell == "ZMQInteractiveShell":
        return True
    config = getattr(ipython, "config", None)
    try:
        return bool(config) and "IPKernelApp" in config
    except Exception:
        return False


def _is_notebook_widget_backend(backend: Optional[str] = None) -> bool:
    name = (backend or plt.get_backend()).lower()
    return any(token in name for token in _NOTEBOOK_BACKEND_TOKENS)


def _is_desktop_gui_backend(backend: Optional[str] = None) -> bool:
    name = (backend or plt.get_backend())
    lowered = name.lower()
    if _is_notebook_widget_backend(lowered):
        return False
    try:
        from matplotlib.rcsetup import interactive_bk

        if name in interactive_bk:
            return True
    except Exception:
        pass
    return any(token in lowered for token in _DESKTOP_BACKEND_TOKENS)


def _run_matplotlib_magic(name: str) -> bool:
    ipython = _get_ipython()
    if ipython is None or not hasattr(ipython, "run_line_magic"):
        return False
    try:
        ipython.run_line_magic("matplotlib", name)
        return True
    except Exception:
        return False


def _switch_backend(name: str) -> bool:
    try:
        plt.switch_backend(name)
        return True
    except Exception:
        return False


def _activate_desktop_gui_backend() -> bool:
    magics = []
    backends = []
    if sys.platform == "darwin":
        magics.append("osx")
        backends.append("MacOSX")
    magics.extend(("qt", "qt5", "tk"))
    backends.extend(("QtAgg", "Qt5Agg", "TkAgg"))
    for magic in magics:
        _run_matplotlib_magic(magic)
        if _is_desktop_gui_backend():
            return True
    for name in backends:
        if _switch_backend(name) and _is_desktop_gui_backend():
            return True
    return False


def _prepare_gui_backend() -> bool:
    """Make the Matplotlib backend interactive for well selection.

    Returns True when the GUI should run as a non-blocking Jupyter widget
    session. Desktop GUI backends keep the blocking window behavior, even if
    the caller is a notebook kernel.

    VS Code and Cursor cannot load ipympl's ``MPLCanvasModel``, so those
    kernels are switched to a native window instead.
    """
    vscode = _running_in_vscode_jupyter()
    if vscode and (
        _is_notebook_widget_backend()
        or (
            _running_in_notebook()
            and not _is_desktop_gui_backend()
        )
    ):
        if _is_desktop_gui_backend() or _activate_desktop_gui_backend():
            return False
        raise RuntimeError(
            "Interactive well selection needs a native Matplotlib window in "
            "VS Code/Cursor. The notebook widget renderer does not include "
            "jupyter-matplotlib, so ipympl cannot be used. Install a GUI "
            "backend such as macOS or Tk, or run the example as a script."
        )

    if _is_notebook_widget_backend():
        return True
    if _is_desktop_gui_backend():
        return False
    if not _running_in_notebook():
        return False

    if _ipympl_frontend_available():
        for magic in ("widget", "ipympl"):
            _run_matplotlib_magic(magic)
            if _is_notebook_widget_backend():
                return True
        if _switch_backend("module://ipympl.backend_nbagg") and (
            _is_notebook_widget_backend()
        ):
            return True

    if _activate_desktop_gui_backend():
        return False

    raise RuntimeError(
        "Interactive well selection requires an interactive Matplotlib "
        "backend. In JupyterLab install ipympl, or use a native GUI "
        "backend such as Qt, Tk, or macOS."
    )


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
    notebook_controls: bool = False
    resolve_in_background: bool = True
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
        if self.notebook_controls:
            grid = self.figure.add_gridspec(
                3,
                2,
                height_ratios=(10, 1, 1),
                width_ratios=(4, 1.35),
                hspace=0.18,
                wspace=0.25,
            )
            self.image_axis = self.figure.add_subplot(grid[0:3, 0])
            self.table_axis = self.figure.add_subplot(grid[0, 1])
            remove_axis = self.figure.add_subplot(grid[1, 1])
            confirm_axis = self.figure.add_subplot(grid[2, 1])
            self._remove_button = Button(remove_axis, "Remove last")
            self._confirm_button = Button(confirm_axis, "Confirm")
            self._remove_button.on_clicked(
                lambda _event: self._remove_last_point()
            )
            self._confirm_button.on_clicked(lambda _event: self.confirm())
        else:
            grid = self.figure.add_gridspec(1, 2, width_ratios=(4, 1.35))
            self.image_axis = self.figure.add_subplot(grid[0, 0])
            self.table_axis = self.figure.add_subplot(grid[0, 1])
            self._remove_button = None
            self._confirm_button = None
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
        lines = [count, "Left-click: add point  |  Right-click: remove"]
        if self.allow_control_click:
            lines.append("Double left-click: add background Control")
        if self.assume_last_control:
            lines.append(
                f"No double-click: point {self.max_total_count} is Control"
            )
        if self.enable_live_grid_preview:
            if self.expected_count == 1:
                lines.append("Left-click and drag marker: reposition the selected point")
            else:
                lines.append("Left-click and drag marker: adjust grid preview")
        if self.notebook_controls:
            lines.append("Remove last / Confirm: use the buttons")
            lines.append("Delete also removes last if the figure has focus")
        else:
            lines.append("Delete: remove last marker |  Enter: confirm")
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
        if hasattr(mode, "value"):
            mode = mode.value
        return str(mode).strip().lower() not in {"", "none"}

    def _event_in_image_axis(self, event) -> bool:
        if event.xdata is None or event.ydata is None:
            return False
        if event.inaxes is self.image_axis:
            return True
        if event.inaxes is not None:
            return False
        contains = getattr(self.image_axis, "contains", None)
        if not callable(contains):
            return False
        try:
            inside, _ = self.image_axis.contains(event)
            return bool(inside)
        except Exception:
            return False

    def _on_click(self, event) -> None:
        if (
            self.confirmed
            or self._processing
            or not self._event_in_image_axis(event)
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
            or             self.confirmed
            or self._processing
            or not self._event_in_image_axis(event)
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
        key = str(event.key or "").lower()
        if key in {"delete", "backspace"}:
            self._remove_last_point()
            return
        if key in {"enter", "return"}:
            self.confirm()

    def _remove_last_point(self) -> None:
        if self.confirmed or self._processing or not self.points:
            return
        self._remove_point(len(self.points) - 1)
        self._set_status("Removed most recent point")
        self._redraw()

    def confirm(self) -> Optional[pd.DataFrame]:
        """Confirm the current points and resolve well coordinates.

        Notebook widget backends resolve immediately and return the DataFrame.
        Desktop backends refine in the background and populate ``result``
        when detection finishes.
        """
        if self.confirmed:
            return self.result
        if self._processing:
            return None
        if (
            self.expected_count is not None
            and len(self._signal_indices()) != self.expected_count
        ):
            self._set_status(
                f"Select exactly {self.expected_count} signal point(s) "
                "before confirming"
            )
            self.figure.canvas.draw_idle()
            return None
        if not self._signal_indices():
            self._set_status(
                "Select at least one signal point before confirming"
            )
            self.figure.canvas.draw_idle()
            return None

        status = (
            "Using clicked coordinates…"
            if self.coordinate_mode == "manual"
            else "Refining clicked wells…"
        )
        if not self.resolve_in_background:
            self._set_status(status)
            self.figure.canvas.draw_idle()
            try:
                result = self._selection_from_points()
            except Exception as exc:
                self._set_status(str(exc))
                self.figure.canvas.draw_idle()
                return None
            self._apply_result(result)
            return self.result

        self._processing = True
        self._set_status(f"◐ {status}")
        self.figure.canvas.draw_idle()
        self._processing_timer = self.figure.canvas.new_timer(interval=100)
        self._processing_timer.add_callback(self._poll_detection)
        self._processing_timer.start()
        Thread(target=self._resolve_points, daemon=True).start()
        return None

    def _selection_from_points(self) -> pd.DataFrame:
        return select_wells_from_coordinates(
            self.image,
            tuple(self.points),
            well_ids=self.well_ids,
            detector=self.detector,
            search_radius=self.search_radius,
            coordinate_mode=self.coordinate_mode,
            manual_roi_diameter=self.manual_roi_diameter,
            control_indices=tuple(sorted(self.control_indices)),
        )

    def _resolve_points(self) -> None:
        try:
            result = self._selection_from_points()
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

        self._apply_result(value)
        return False

    def _apply_result(self, result: pd.DataFrame) -> None:
        self.result = result
        self.confirmed = True
        self._processing = False
        self._set_status("Selection confirmed")
        self._disconnect()
        if self.close_on_confirm:
            plt.close(self.figure)
        else:
            self.image_axis.set_title("Selection confirmed")
            self.figure.canvas.draw_idle()

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
        elif (
            self.enable_live_grid_preview
            and self.manual_roi_diameter is not None
            and signal_points
        ):
            preview_radius = float(self.manual_roi_diameter) / 2
            signal_number = 0
            for point_index, (x, y) in enumerate(self.points):
                if point_index in self.control_indices:
                    color = "#00ff7f"
                else:
                    color = grid_colors[signal_number]
                    signal_number += 1
                circle = plt.Circle(
                    (x, y),
                    preview_radius,
                    edgecolor=color,
                    facecolor="none",
                    linewidth=1.5,
                    linestyle="-",
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
    DataFrame. JupyterLab can use ``ipympl`` with Confirm/Remove buttons and
    returns a session immediately. VS Code and Cursor do not register
    ``jupyter-matplotlib``, so those kernels open a native window instead.
    Set ``enable_live_grid_preview=True`` to drag placed ``+`` markers.
    With two or more row-major anchors this also shows the inferred 3x3
    geometry. A single point with ``manual_roi_diameter`` shows that ROI
    circle.
    """
    notebook = _prepare_gui_backend()
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
        notebook_controls=notebook,
        resolve_in_background=not notebook,
    )
    if not notebook and _running_in_notebook():
        print(
            "Opening a native Matplotlib window. Click the wells, then "
            "press Enter. VS Code/Cursor cannot load ipympl's "
            "jupyter-matplotlib widget."
        )
    plt.show(block=not notebook)
    if notebook:
        return session
    if session.result is None:
        raise RuntimeError("Well selection was closed before confirmation")
    return session.result
