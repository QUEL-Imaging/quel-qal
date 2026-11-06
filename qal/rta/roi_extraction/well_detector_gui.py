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
from matplotlib._pylab_helpers import Gcf
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


def is_confirm_key(key) -> bool:
    """True for Enter or Return, including the keypad, on every backend.

    Matplotlib reports ``enter`` on some builds and ``return`` on others.
    A modifier prefix such as ``ctrl+enter`` still counts.
    """
    token = str(key or "").lower().replace(" ", "")
    token = token.split("+")[-1]
    return token in {"enter", "return", "kp_enter", "\r", "\n"}


def _window_is_shown(figure) -> bool:
    """True when a native window is on screen on macOS, Windows, or Linux."""
    canvas = getattr(figure, "canvas", None)
    manager = getattr(canvas, "manager", None)
    if manager is None:
        return False
    if getattr(manager, "_shown", False):
        return True
    window = getattr(manager, "window", None)
    if window is None:
        return False
    for name in ("isVisible", "winfo_viewable", "winfo_ismapped"):
        method = getattr(window, name, None)
        if not callable(method):
            continue
        try:
            if bool(method()):
                return True
        except Exception:
            continue
    return False


def _stop_figure_timers(figure) -> None:
    """Cancel Matplotlib and QAL timers that can resurrect a closed window."""
    canvas = getattr(figure, "canvas", None)
    for timer in list(getattr(figure, "_qal_timers", []) or []):
        stop = getattr(timer, "stop", None)
        if callable(stop):
            try:
                stop()
            except Exception:
                pass
    try:
        figure._qal_timers = []
    except Exception:
        pass
    close_timer = getattr(figure, "_qal_close_timer", None)
    if close_timer is not None:
        try:
            close_timer.stop()
        except Exception:
            pass
        try:
            del figure._qal_close_timer
        except Exception:
            pass
    timers = getattr(canvas, "_timers", None)
    if timers:
        while timers:
            timer = timers.pop()
            stop = getattr(timer, "stop", None)
            if callable(stop):
                try:
                    stop()
                except Exception:
                    pass


def _destroy_figure(figure) -> None:
    """Close ``figure`` the same way the window's close button does."""
    figure._qal_closed = True
    _stop_figure_timers(figure)
    canvas = getattr(figure, "canvas", None)
    manager = getattr(canvas, "manager", None)
    number = getattr(figure, "number", None)
    try:
        if number is not None and plt.fignum_exists(number):
            if manager is not None:
                Gcf.destroy(manager)
            else:
                plt.close(figure)
        else:
            destroy = getattr(manager, "destroy", None)
            if callable(destroy):
                destroy()
    except Exception:
        try:
            if number is not None and plt.fignum_exists(number):
                plt.close(figure)
        except Exception:
            pass
    flush = getattr(canvas, "flush_events", None)
    if callable(flush):
        try:
            flush()
        except Exception:
            pass


def _is_interactive_gui_canvas(canvas) -> bool:
    module = getattr(type(canvas), "__module__", "")
    return any(
        token in module
        for token in (
            "backend_macosx",
            "backend_tk",
            "backend_qt",
            "backend_wx",
            "backend_gtk",
        )
    )


def _stop_shown_macos_loop(figure) -> bool:
    """Return from ``plt.show`` on the macOS backend.

    Destroying a macOS figure inside a key, click, or timer callback makes
    the process exit instead of returning from ``plt.show``. ``_macosx.stop``
    ends the event loop and leaves the figure open so the caller can close
    it after ``show`` returns. Prefer :func:`run_selection_window` so
    notebooks do not need this path.
    """
    canvas = getattr(figure, "canvas", None)
    module = getattr(type(canvas), "__module__", "")
    if "backend_macosx" not in module:
        return False
    macosx = sys.modules.get("matplotlib.backends._macosx")
    stop = getattr(macosx, "stop", None)
    if not callable(stop):
        return False
    try:
        stop(None)
    except TypeError:
        try:
            stop()
        except Exception:
            return False
    except Exception:
        return False
    return True


def run_selection_window(figure, *, block: bool = True) -> None:
    """Show the selection figure and wait until Enter or the close button.

    Scripts and notebooks use the same wait: map the window, then run the
    canvas event loop. Enter calls :meth:`~.FigureCanvasBase.stop_event_loop`
    through :func:`close_figure_after_callback`. That avoids ``plt.show``'s
    macOS ``NSApp`` loop, which conflicts with Jupyter's input hook.

    Destroying the native window happens here after the loop returns, not
    inside the Enter key callback. That keeps Jupyter notebooks and desktop
    scripts on the same close path without aborting the macOS backend.
    On macOS, bounded waits also handle close events consumed by Cocoa's
    event flushing and unrelated events that return from the loop early.
    """
    if not block:
        plt.show(block=False)
        return

    canvas = getattr(figure, "canvas", None)
    if canvas is not None and _is_interactive_gui_canvas(canvas):
        manager = getattr(canvas, "manager", None)
        if manager is not None:
            try:
                manager.show()
            except Exception:
                pass
        focus_selection_canvas(figure)
        start = getattr(canvas, "start_event_loop", None)
        if callable(start):
            figure._qal_using_event_loop = True
            figure._qal_close_after_loop = False
            loop_started = False
            try:
                try:
                    if "backend_macosx" in type(canvas).__module__:
                        # Cocoa returns on *any* application-defined event,
                        # including startup events. Conversely, the close
                        # button's flush_events() can consume the event that
                        # would wake this loop. Use bounded waits and check
                        # the figure itself instead of trusting either event.
                        while (
                            Gcf.get_fig_manager(figure.number) is manager
                            and not getattr(figure, "_qal_close_after_loop", False)
                        ):
                            start(0.1)
                    else:
                        start(0)
                    loop_started = True
                except TypeError:
                    start(timeout=0)
                    loop_started = True
            except Exception:
                loop_started = False
            finally:
                figure._qal_using_event_loop = False
                if loop_started and getattr(figure, "_qal_close_after_loop", False):
                    _destroy_figure(figure)
            if loop_started:
                return

    figure._qal_close_after_loop = False
    plt.show(block=True)
    if getattr(figure, "_qal_close_after_loop", False):
        _destroy_figure(figure)


def close_figure_after_callback(figure) -> None:
    """Close ``figure`` without destroying it inside a mouse or key callback.

    When the picker is waiting in :func:`run_selection_window`, stop that
    canvas event loop and mark the figure so the window is destroyed after
    the loop returns. Otherwise a visible non-macOS window is closed on the
    next timer turn, and a shown macOS ``plt.show`` loop is stopped without
    destroying the window from inside the key callback.

    The timer has to stay referenced. Matplotlib cancels a timer when its
    Python object is collected, so a timer that exists only as a local
    variable never fires and the selection window stays open after Enter.
    """
    disarm_confirm_window(figure)
    canvas = getattr(figure, "canvas", None)

    def _close() -> None:
        _destroy_figure(figure)

    if canvas is None or not _window_is_shown(figure):
        _close()
        return

    if getattr(figure, "_qal_using_event_loop", False):
        figure._qal_close_after_loop = True
        stop_loop = getattr(canvas, "stop_event_loop", None)
        if callable(stop_loop):
            try:
                stop_loop()
                return
            except Exception:
                pass

    if _stop_shown_macos_loop(figure):
        figure._qal_close_after_loop = True
        return
    if getattr(figure, "_qal_close_timer", None) is not None:
        return
    try:
        timer = canvas.new_timer(interval=0)
        timer.single_shot = True

        def _fire() -> None:
            if getattr(figure, "_qal_close_timer", None) is timer:
                try:
                    del figure._qal_close_timer
                except Exception:
                    pass
            _close()
            timer.callbacks.clear()

        timer.add_callback(_fire)
        # Keep the timer alive after this function returns.
        figure._qal_close_timer = timer
        timer.start()
    except Exception:
        _close()


def focus_selection_canvas(figure) -> None:
    """Give the image window keyboard focus so Enter reaches the picker.

    Notebook hosts (Cursor, VS Code, JupyterLab) often steal keyboard focus
    back after a mouse click. Call this again after clicks so Enter still
    confirms there the same way it does in a plain script.
    """
    if getattr(figure, "_qal_closed", False):
        return
    canvas = getattr(figure, "canvas", None)
    if canvas is None:
        return
    manager = getattr(canvas, "manager", None)
    module = getattr(type(canvas), "__module__", "")
    if "backend_macosx" in module:
        # orderFront alone leaves another app key, so Return never reaches
        # the figure. makeKeyAndOrderFront is FigureManager._show.
        show = getattr(manager, "_show", None)
        if callable(show):
            try:
                show()
            except Exception:
                pass
        raise_window = getattr(manager, "_raise", None)
        if callable(raise_window):
            try:
                raise_window()
            except Exception:
                pass
        return
    window = getattr(manager, "window", None)
    getter = getattr(canvas, "get_tk_widget", None)
    if callable(getter):
        try:
            widget = getter()
            widget.focus_force()
            top = widget.winfo_toplevel()
            top.lift()
            top.focus_force()
        except Exception:
            pass
    for owner in (canvas, window):
        if owner is None:
            continue
        for name in ("setFocus", "activateWindow", "raise_", "requestActivate"):
            method = getattr(owner, name, None)
            if not callable(method):
                continue
            try:
                method()
            except Exception:
                pass


def install_confirm_shortcuts(figure, confirm) -> None:
    """Bind Enter on the native window, not only on the Matplotlib canvas.

    Tk on Windows and Linux delivers Return to the toplevel even when the
    canvas does not have focus. Qt and macOS use the canvas key event once
    :func:`focus_selection_canvas` has run.
    """
    canvas = getattr(figure, "canvas", None)
    getter = getattr(canvas, "get_tk_widget", None)
    if not callable(getter):
        return
    try:
        top = getter().winfo_toplevel()
    except Exception:
        return

    def _on_return(_event=None):
        confirm()
        return "break"

    for sequence in ("<Return>", "<KP_Enter>"):
        try:
            top.bind(sequence, _on_return)
        except Exception:
            pass


def arm_confirm_window(figure, confirm) -> None:
    """Focus the picker and bind Enter before ``plt.show`` blocks.

    Re-focus after the window maps and after every click or mouse enter so
    notebooks and scripts share the same Enter / close-window behavior.
    """
    install_confirm_shortcuts(figure, confirm)
    focus_selection_canvas(figure)
    canvas = getattr(figure, "canvas", None)
    if canvas is None:
        return

    held = getattr(figure, "_qal_timers", None)
    if not isinstance(held, list):
        held = []
        figure._qal_timers = held

    def _refocus(_event=None) -> None:
        if getattr(figure, "_qal_closed", False):
            return
        focus_selection_canvas(figure)

    if not getattr(figure, "_qal_focus_hooks", False):
        cids = list(getattr(figure, "_qal_focus_cids", []) or [])
        for name in (
            "button_press_event",
            "button_release_event",
            "figure_enter_event",
        ):
            try:
                cids.append(canvas.mpl_connect(name, _refocus))
            except Exception:
                pass
        figure._qal_focus_cids = cids
        figure._qal_focus_hooks = True

    new_timer = getattr(canvas, "new_timer", None)
    if not callable(new_timer):
        return
    try:
        for delay in (50, 200, 500):
            timer = new_timer(interval=delay)
            timer.single_shot = True
            timer.add_callback(_refocus)
            # A local timer is cancelled as soon as this function returns.
            held.append(timer)
            timer.start()
    except Exception:
        return


def disarm_confirm_window(figure) -> None:
    """Stop refocus timers and hooks so they cannot reopen a closed window.

    ``arm_confirm_window`` schedules ``makeKeyAndOrderFront`` on macOS. If
    those timers fire after Enter, Jupyter's macOS input hook keeps a
    Matplotlib window alive even after the notebook cell has finished.
    """
    figure._qal_closed = True
    _stop_figure_timers(figure)
    canvas = getattr(figure, "canvas", None)
    if canvas is None:
        return
    for connection in list(getattr(figure, "_qal_focus_cids", []) or []):
        try:
            canvas.mpl_disconnect(connection)
        except Exception:
            pass
    figure._qal_focus_cids = []
    figure._qal_focus_hooks = False


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


def _is_inline_backend(backend: Optional[str] = None) -> bool:
    name = (backend or plt.get_backend()).lower()
    return "inline" in name


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


def _release_unloaded_gui_selection() -> None:
    """Forget a GUI name that never became the active Matplotlib backend.

    IPython stores the first requested GUI before that backend loads. A failed
    macOS or Qt request is then retried by every later ``%matplotlib`` call,
    including Tk, until the kernel is restarted.
    """
    ipython = _get_ipython()
    if ipython is None:
        return
    selected = getattr(ipython, "pylab_gui_select", None)
    if not selected or selected in {"inline", "agg"}:
        return
    if _is_desktop_gui_backend() or _is_notebook_widget_backend():
        return
    try:
        ipython.pylab_gui_select = None
    except Exception:
        return


def _disable_ipython_gui() -> None:
    """Drop IPython's macOS/Qt input hook after the picker window closes.

    ``%matplotlib osx`` leaves ``enable_gui('osx')`` active. If a native
    window is still mapped, that hook keeps ``NSApp`` running in the
    background after the notebook cell has already finished.
    """
    ipython = _get_ipython()
    if ipython is None:
        return
    enable_gui = getattr(ipython, "enable_gui", None)
    if not callable(enable_gui):
        return
    try:
        enable_gui(None)
    except Exception:
        pass


def _run_matplotlib_magic(name: str) -> bool:
    ipython = _get_ipython()
    if ipython is None or not hasattr(ipython, "run_line_magic"):
        return False
    _release_unloaded_gui_selection()
    try:
        ipython.run_line_magic("matplotlib", name)
        return True
    except Exception:
        _release_unloaded_gui_selection()
        return False


def _switch_backend(name: str) -> bool:
    try:
        plt.switch_backend(name)
        return True
    except Exception:
        return False


def _gui_candidates() -> list[tuple[str, str]]:
    """Native window backends for this OS, in the order they should be tried.

    Windows and Linux use Tk. macOS uses the macOS backend first because many
    Homebrew and framework Pythons have no Tk. Qt is intentionally absent:
    trying it after a failure makes IPython reuse that failed name. The
    desktop activator tries Qt only after these candidates fail.
    """
    if sys.platform == "darwin":
        return [("osx", "MacOSX"), ("tk", "TkAgg")]
    return [("tk", "TkAgg")]


def _candidate_is_active(magic: str) -> bool:
    name = plt.get_backend().lower()
    if magic == "tk":
        return "tk" in name
    if magic == "osx":
        return "mac" in name
    if magic.startswith("qt"):
        return "qt" in name
    return False


def _try_magic_backend(magic: str, backend_name: str) -> bool:
    """Activate one GUI, and forget the name if it does not load."""

    def ready() -> bool:
        if not _candidate_is_active(magic):
            return False
        if magic == "osx":
            # The macOS backend forces a Dock icon. Put it back before the
            # next window, after an earlier confirmation hid it.
            _present_macos_app()
        return True

    if ready():
        return True
    if _is_desktop_gui_backend():
        return False
    _release_unloaded_gui_selection()
    _run_matplotlib_magic(magic)
    if ready():
        return True
    _release_unloaded_gui_selection()
    if _is_desktop_gui_backend():
        return False
    switched = _switch_backend(backend_name) and ready()
    if not switched:
        _release_unloaded_gui_selection()
    return bool(switched)


def _try_native_window_backend() -> bool:
    if any(_candidate_is_active(magic) for magic, _name in _gui_candidates()):
        return True
    for magic, backend_name in _gui_candidates():
        if _try_magic_backend(magic, backend_name):
            return True
    return False


def _objc():
    """Load the system Objective-C runtime used to hide the Dock icon.

    On Apple Silicon, ``objc_msgSend`` has to be looked up by name. Calling
    the ctypes wrapper object crashes the interpreter. Windows and Linux
    have no such library; callers must keep this off those platforms.
    """
    if sys.platform != "darwin":
        raise OSError("The macOS Dock helper is only used on macOS.")
    import ctypes

    lib = ctypes.cdll.LoadLibrary("/usr/lib/libobjc.dylib")
    lib.objc_getClass.restype = ctypes.c_void_p
    lib.objc_getClass.argtypes = [ctypes.c_char_p]
    lib.sel_registerName.restype = ctypes.c_void_p
    lib.sel_registerName.argtypes = [ctypes.c_char_p]
    return lib


def _objc_msg(lib, restype, *argtypes):
    import ctypes

    return ctypes.CFUNCTYPE(restype, *argtypes)(("objc_msgSend", lib))


def _ns_application():
    """Return the already-created macOS app, without starting a new one."""
    if sys.platform != "darwin":
        return None
    if "matplotlib.backends._macosx" not in sys.modules:
        return None
    try:
        import ctypes

        lib = _objc()
        app_class = lib.objc_getClass(b"NSApplication")
        if not app_class:
            return None
        return _objc_msg(lib, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)(
            app_class, lib.sel_registerName(b"sharedApplication")
        )
    except Exception:
        return None


def _set_macos_activation_policy(policy: int) -> None:
    """Show or hide the macOS Matplotlib dock icon without quitting Python.

    Quitting that icon sends terminate to the kernel. Policy 0 is a normal
    app, which is what the click window needs. Policy 1 hides the Dock
    icon and still allows the next window. Policy 2 removes the icon by
    prohibiting the app, and macOS then quits the process.
    """
    app = _ns_application()
    if not app:
        return
    try:
        import ctypes

        lib = _objc()
        _objc_msg(
            lib,
            ctypes.c_bool,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_long,
        )(app, lib.sel_registerName(b"setActivationPolicy:"), policy)
    except Exception:
        return


_MACOS_HOST_PID: Optional[int] = None


def _macos_frontmost_pid() -> Optional[int]:
    """Read the active app without starting a Cocoa application."""
    if sys.platform != "darwin" or "matplotlib.backends._macosx" not in sys.modules:
        return None
    try:
        import ctypes

        lib = _objc()
        workspace_class = lib.objc_getClass(b"NSWorkspace")
        if not workspace_class:
            return None
        send = _objc_msg(lib, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)
        workspace = send(workspace_class, lib.sel_registerName(b"sharedWorkspace"))
        frontmost = send(workspace, lib.sel_registerName(b"frontmostApplication"))
        if not frontmost:
            return None
        return _objc_msg(lib, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p)(
            frontmost, lib.sel_registerName(b"processIdentifier")
        )
    except Exception:
        return None


def _activate_macos_app(pid: int) -> None:
    """Request activation of the app that owned focus before the picker."""
    if sys.platform != "darwin":
        return
    try:
        import ctypes

        lib = _objc()
        app_class = lib.objc_getClass(b"NSRunningApplication")
        if not app_class:
            return
        app = _objc_msg(
            lib, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int
        )(app_class, lib.sel_registerName(b"runningApplicationWithProcessIdentifier:"), pid)
        if app:
            _objc_msg(
                lib, ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong
            )(app, lib.sel_registerName(b"activateWithOptions:"), 0)
    except Exception:
        return


def _restore_macos_host_focus(canvas=None) -> None:
    global _MACOS_HOST_PID
    pid, _MACOS_HOST_PID = _MACOS_HOST_PID, None
    # Never interrupt someone who switched to another app during selection.
    if pid is not None and _macos_frontmost_pid() == os.getpid():
        _activate_macos_app(pid)
        # Activation is asynchronous. Deliver the Cocoa handoff before
        # Jupyter disables its GUI input hook and stops servicing NSApp.
        if canvas is not None:
            try:
                for _ in range(20):
                    if _macos_frontmost_pid() != os.getpid():
                        break
                    canvas.start_event_loop(0.01)
            except Exception:
                pass


def _dismiss_macos_app(canvas=None) -> None:
    """Hide the macOS plot app and release focus after its windows are gone.

    ``plt.close`` removes the window and leaves the Matplotlib dock icon.
    Closing that icon quits the Jupyter kernel. Hide the icon and do not
    terminate the process.

    The window close already stops the event loop. ``stop`` is not called
    here: after the first window it makes the next ``show`` return
    immediately or quit the interpreter. ``event_loop_is_running`` also
    stays true for the rest of the process, so it cannot be used as a
    signal to stop.
    """
    if sys.platform != "darwin":
        return
    if "matplotlib.backends._macosx" not in sys.modules:
        return
    if plt.get_fignums():
        return
    # Accessory (1), not Prohibited (2). Prohibited hides the icon and
    # then macOS quits the interpreter the next time a window closes.
    _set_macos_activation_policy(1)
    # Accessory policy removes the Dock icon but can leave this windowless
    # process frontmost. Its stopped Cocoa loop then shows a spinning cursor
    # until another app gains focus, even though the kernel is already idle.
    _restore_macos_host_focus(canvas)


def _present_macos_app() -> None:
    """Remember the host app and let the next plot window appear in the Dock."""
    global _MACOS_HOST_PID
    if sys.platform != "darwin":
        return
    if "mac" not in plt.get_backend().lower():
        return
    pid = _macos_frontmost_pid()
    if pid is not None and pid != os.getpid():
        _MACOS_HOST_PID = pid
    _set_macos_activation_policy(0)


def use_inline_plots() -> bool:
    """Show later figures in the notebook.

    Scripts are left alone. A notebook that is already inline stays inline.
    Otherwise this asks IPython for the inline backend, then tries Matplotlib's
    inline backend directly if that request does not stick. On macOS the
    native plot app is hidden first so switching backends does not leave a
    Dock icon that can quit the kernel.
    """
    if not _running_in_notebook():
        return False
    _dismiss_macos_app()
    switched = False
    if _is_inline_backend():
        switched = True
    elif _run_matplotlib_magic("inline") and _is_inline_backend():
        switched = True
    elif _switch_backend("module://matplotlib_inline.backend_inline") and (
        _is_inline_backend()
    ):
        switched = True
    _disable_ipython_gui()
    _dismiss_macos_app()
    return switched


def use_native_window_backend() -> None:
    """Open later figures in a native window.

    Windows and Linux request Tk. macOS requests the macOS backend, then Tk.
    This function does not try Qt. If a different GUI toolkit is already
    running, IPython cannot switch until the kernel is restarted.
    """
    if _try_native_window_backend():
        return
    backend = plt.get_backend()
    if _is_desktop_gui_backend(backend):
        raise RuntimeError(
            f"Matplotlib is already using {backend}. Restart the kernel, "
            "then run this notebook from the top so the picker can open "
            "a native window."
        )
    raise RuntimeError(
        "Could not open a native window. Windows and Linux need tkinter. "
        "Restart the kernel and run this notebook from the top."
    )


def _activate_desktop_gui_backend() -> bool:
    """Switch to a native window.

    The OS-native backend is tried before Qt. A failed candidate is forgotten
    before the next one is tried; otherwise IPython keeps replaying the first
    name.
    """
    if _try_native_window_backend():
        return True
    for magic, backend_name in (("qt", "QtAgg"), ("qt5", "Qt5Agg")):
        if _try_magic_backend(magic, backend_name):
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
            "jupyter-matplotlib, so ipympl cannot be used. Use Tk on "
            "Windows or Linux, or the macOS backend on a Mac, or run the "
            "example as a script."
        )

    if _is_notebook_widget_backend():
        return True
    if _is_desktop_gui_backend():
        return False
    if not _running_in_notebook():
        return False

    # A native window lets the cell wait for clicks. ipympl is the fallback
    # when this Python cannot open Tk, the macOS backend, or Qt.
    if _activate_desktop_gui_backend():
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

    raise RuntimeError(
        "Interactive well selection requires an interactive Matplotlib "
        "backend. In JupyterLab install ipympl, or use Tk on Windows "
        "or Linux, or the macOS backend on a Mac."
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
    closed: bool = field(default=False, init=False)

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
        self.image_axis.grid(False)
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
            self.figure.canvas.mpl_connect("close_event", self._on_close),
        ]
        self._key_cid = self.figure.canvas.mpl_connect(
            "key_press_event", self._on_key
        )
        self._connections.append(self._key_cid)
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
            lines.append(
                "Delete: remove last marker |  Enter: confirm |  "
                "Close the window: confirm"
            )
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
        if is_confirm_key(getattr(event, "key", None)):
            if self.confirmed:
                if self.close_on_confirm and not self.closed:
                    close_figure_after_callback(self.figure)
                return
            if not self._processing:
                self.confirm()
            return
        if self.confirmed or self._processing:
            return
        key = str(event.key or "").lower()
        if key in {"delete", "backspace"}:
            self._remove_last_point()
            return

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
        if self.confirmed or self.closed:
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
        if not self.resolve_in_background or getattr(self, "_handling_close", False):
            # The close button is already destroying the window. Record the
            # points and do not draw on a canvas that is going away.
            if not getattr(self, "_handling_close", False):
                self._set_status(status)
                try:
                    self.figure.canvas.draw_idle()
                except Exception:
                    pass
            try:
                result = self._selection_from_points()
            except Exception as exc:
                if not getattr(self, "_handling_close", False):
                    self._set_status(str(exc))
                    try:
                        self.figure.canvas.draw_idle()
                    except Exception:
                        pass
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
        if self.closed:
            self._processing = False
            return False
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
        try:
            self._set_status("Selection confirmed")
        except Exception:
            pass
        if getattr(self, "_handling_close", False):
            self._disconnect()
            return
        # Leave the Enter handler connected so a second press can close the
        # window if the deferred close has not run yet.
        self._disconnect(keep_confirm_key=True)
        if self.close_on_confirm:
            close_figure_after_callback(self.figure)
        else:
            try:
                self.image_axis.set_title("Selection confirmed")
                self.figure.canvas.draw_idle()
            except Exception:
                pass

    def _set_status(self, message: str) -> None:
        self._status.set_text(message)

    def _selection_ready(self) -> bool:
        """True when Enter would accept the points currently placed."""
        signal_count = len(self._signal_indices())
        if self.expected_count is not None:
            return signal_count == self.expected_count
        return signal_count > 0

    def _on_close(self, _event=None) -> None:
        if getattr(self, "_handling_close", False):
            return
        self._handling_close = True
        try:
            # The window is already going away. A finished selection confirms
            # the same way Enter does; an unfinished one is discarded.
            if (
                not self.confirmed
                and not self._processing
                and self._selection_ready()
            ):
                self.close_on_confirm = False
                self.resolve_in_background = False
                self.confirm()
        finally:
            self.closed = True
            self._drag_index = None
            if self._processing_timer is not None:
                self._processing_timer.stop()
            self._processing = False
            self._disconnect()

    def _disconnect(self, *, keep_confirm_key: bool = False) -> None:
        kept = []
        for connection in self._connections:
            if keep_confirm_key and connection == getattr(self, "_key_cid", None):
                kept.append(connection)
                continue
            try:
                self.figure.canvas.mpl_disconnect(connection)
            except Exception:
                pass
        self._connections = kept

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
) -> pd.DataFrame | WellSelectionSession | None:
    """Open the interactive picker and collect or refine well centroids.

    Desktop Matplotlib backends block until the window closes and return a
    DataFrame, or ``None`` if it is closed before the required points are
    placed. Closing a finished selection confirms it, the same as Enter.
    JupyterLab can use ``ipympl`` with Confirm/Remove buttons and
    returns a session immediately. VS Code and Cursor do not register
    ``jupyter-matplotlib``, so those kernels open a native window instead.
    Set ``enable_live_grid_preview=True`` to drag placed ``+`` markers.
    With two or more row-major anchors this also shows the inferred 3x3
    geometry. A single point with ``manual_roi_diameter`` shows that ROI
    circle.
    """
    notebook = _prepare_gui_backend()
    # Cursor/VS Code notebooks open a native window (notebook=False). Resolve
    # synchronously there so Enter can stop the event loop immediately; a
    # background timer often never fires under Jupyter's input hook, which
    # leaves the concentration/depth picker open after confirmation.
    native_notebook = (not notebook) and _running_in_notebook()
    _present_macos_app()
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
        resolve_in_background=not notebook and not native_notebook,
    )
    if native_notebook:
        print("Opening a window. Click the wells, then press Enter or close the window.")
    arm_confirm_window(session.figure, session.confirm)
    try:
        run_selection_window(session.figure, block=not notebook)
    except KeyboardInterrupt:
        session._on_close()
        disarm_confirm_window(session.figure)
        _destroy_figure(session.figure)
    finally:
        if not notebook:
            disarm_confirm_window(session.figure)
            _destroy_figure(session.figure)
            _dismiss_macos_app(session.figure.canvas)
            use_inline_plots()
    if notebook:
        return session
    if session.result is None:
        _destroy_figure(session.figure)
        print(
            "Well selection was closed before confirmation; nothing was "
            "selected."
        )
        return None
    return session.result
