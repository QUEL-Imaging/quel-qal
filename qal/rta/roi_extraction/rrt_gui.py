"""Interactive and programmatic line-profile selection for USAF 1951 targets.

Line ``i`` always measures Group ``i``. Even groups span elements 2-6 (their
element 1 sits apart at the lower right); odd groups span elements 1-6. The
standard measurement uses three vertical lines: Group 0 E2-6, Group 1 E1-6,
and Group 2 E2-6.

Example scripts choose one of four modes:

- ``automatic_refinement``: click two registration corners; each snaps to the
  nearest Harris corner and the standard three lines are derived from them.
- ``manual_live_preview``: drag two registration crosshairs; the standard
  three lines and their profiles update live from the exact positions.
- ``manual_coordinates``: no GUI. Supply registration points or explicit line
  endpoints, e.g. for batches imaged in a fixed jig.
- ``manual_line_profiles``: click start/end pairs for ``line_count`` lines.

Passing ``registration_points`` or ``line_coordinates`` to
:func:`select_rrt_lines` skips the GUI in every mode.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import cv2
from matplotlib.widgets import Button
from skimage.feature import corner_harris, corner_peaks

from qal.util import load_grayscale_image

from ..image_analyzer.rrt_analyzer import RrtAnalyzer
from ._selection_io import load_selection, save_selection
from .well_detector_gui import _prepare_gui_backend, _running_in_notebook

Coordinate = tuple[float, float]
Line = tuple[Coordinate, Coordinate]
RrtGuiMode = Literal[
    "automatic_refinement",
    "manual_live_preview",
    "manual_coordinates",
    "manual_line_profiles",
]
RRT_GUI_MODES = (
    "automatic_refinement",
    "manual_live_preview",
    "manual_coordinates",
    "manual_line_profiles",
)
STANDARD_LINE_COUNT = 3
MAX_LINE_COUNT = 6

# Line geometry as fractions of the registration box, whose corners are the
# top-left of Group 0 Element 2's top bar and the bottom-right of Group 0
# Element 1's bottom bar. Per group: (x through the centre of element 6's
# horizontal bars, top of the first element's bars, bottom of element 6's
# bars, margin added beyond each end). Assumes an undistorted image.
REGISTRATION_LINE_RATIOS: dict[int, tuple[float, float, float, float]] = {
    0: (0.058, 0.000, 0.998, 0.035),
    1: (0.972, 0.003, 0.648, 0.020),
    2: (0.5175, 0.365, 0.6145, 0.015),
}
# Groups 2-3 repeat Groups 0-1 at quarter scale, anchored at this box
# fraction (left of Group 2's bar column, top of Element 2); Groups 4-5
# repeat that again inside the centre.
NESTED_GROUP_ORIGIN = (0.503, 0.365)
NESTED_GROUP_SCALE = 0.25

# Pixel-centre corner positions in qal/data/USAF1951_template/res_source.png.
TEMPLATE_REGISTRATION_POINTS: tuple[Coordinate, Coordinate] = (
    (54.9, 76.6),
    (202.8, 215.4),
)

_REGISTRATION_LABELS = (
    "top-left corner of the top bar, Group 0 Element 2",
    "bottom-right corner of the bottom bar, Group 0 Element 1",
)


def group_elements(group: int) -> range:
    """Element numbers crossed by a vertical line through ``group``."""
    if not 0 <= group < MAX_LINE_COUNT:
        raise ValueError(f"group must be between 0 and {MAX_LINE_COUNT - 1}")
    return range(1, 7) if group % 2 else range(2, 7)


def validate_line_count(line_count: int) -> int:
    if isinstance(line_count, bool) or int(line_count) != line_count:
        raise ValueError("LINE_COUNT must be an integer")
    line_count = int(line_count)
    if not 1 <= line_count <= MAX_LINE_COUNT:
        raise ValueError(f"LINE_COUNT must be between 1 and {MAX_LINE_COUNT}")
    return line_count


def lines_from_registration(
    top_left: Coordinate,
    bottom_right: Coordinate,
) -> list[Line]:
    """Return the standard three lines from the two registration corners.

    ``top_left`` is the top-left corner of the top bar of Group 0 Element 2;
    ``bottom_right`` is the bottom-right corner of the bottom bar of Group 0
    Element 1. Both are image ``(x, y)`` coordinates.
    """
    left, top = map(float, top_left)
    right, bottom = map(float, bottom_right)
    width, height = right - left, bottom - top
    lines = []
    for group in range(STANDARD_LINE_COUNT):
        x_ratio, top_ratio, bottom_ratio, margin = (
            REGISTRATION_LINE_RATIOS[group]
        )
        x = left + x_ratio * width
        lines.append(
            (
                (x, top + (top_ratio - margin) * height),
                (x, top + (bottom_ratio + margin) * height),
            )
        )
    return lines


def group_line_ratios(group: int) -> tuple[float, float, float]:
    """``(x, top, bottom)`` box fractions spanning ``group``'s bars.

    Groups 2-5 follow the nested quarter-scale layout; only Groups 0-2 are
    used for registration-derived measurements.
    """
    if group in REGISTRATION_LINE_RATIOS:
        return REGISTRATION_LINE_RATIOS[group][:3]
    level, parity = divmod(group, 2)
    x_ratio, top_ratio, bottom_ratio, _ = REGISTRATION_LINE_RATIOS[parity]
    origin_x = origin_y = 0.0
    for depth in range(level):
        origin_x += NESTED_GROUP_ORIGIN[0] * NESTED_GROUP_SCALE ** depth
        origin_y += NESTED_GROUP_ORIGIN[1] * NESTED_GROUP_SCALE ** depth
    scale = NESTED_GROUP_SCALE ** level
    return (
        origin_x + x_ratio * scale,
        origin_y + top_ratio * scale,
        origin_y + bottom_ratio * scale,
    )


def template_guide_line(group: int) -> Line:
    """Bar span of ``group`` in the packaged USAF1951 template image."""
    (left, top), (right, bottom) = TEMPLATE_REGISTRATION_POINTS
    x_ratio, top_ratio, bottom_ratio = group_line_ratios(group)
    x = left + x_ratio * (right - left)
    return (
        (x, top + top_ratio * (bottom - top)),
        (x, top + bottom_ratio * (bottom - top)),
    )


def group_coordinates_from_lines(lines: Sequence[Line]) -> dict:
    """Build the ``RrtAnalyzer.load_and_process_groups`` input from lines."""
    validate_line_count(len(lines))
    return {
        group: {
            "coordinates": (
                (float(start[0]), float(start[1])),
                (float(end[0]), float(end[1])),
            ),
            "elements": group_elements(group),
        }
        for group, (start, end) in enumerate(lines)
    }


def detect_harris_corners(image: np.ndarray) -> np.ndarray:
    """Return Harris corner positions as an ``(N, 2)`` array of ``(x, y)``."""
    response = corner_harris(np.asarray(image, dtype=float), k=0.04)
    peaks = corner_peaks(response, min_distance=1, threshold_rel=0.02)
    return peaks[:, ::-1].astype(float) if len(peaks) else np.empty((0, 2))


def snap_registration_points(
    image: Optional[np.ndarray],
    points: Sequence[Coordinate],
    *,
    corners: Optional[np.ndarray] = None,
) -> list[Coordinate]:
    """Snap each point to the nearest Harris corner, then refine it.

    The search radius is 10 percent of the horizontal distance between the
    two points. A point with no corner inside that radius is kept as given.
    When ``image`` is supplied, snapped points are refined with
    ``cv2.cornerSubPix`` so they sit on the bar-edge corner that
    :data:`REGISTRATION_LINE_RATIOS` is measured from; Harris peaks alone
    land about a pixel inside the bar.
    """
    if len(points) != 2:
        raise ValueError("Exactly two registration points are required")
    if corners is None:
        corners = detect_harris_corners(image)
    spacing = abs(points[1][0] - points[0][0])
    radius = max(0.1 * spacing, 1.0)
    snapped = []
    for point in points:
        point = (float(point[0]), float(point[1]))
        if len(corners):
            distances = np.linalg.norm(corners - np.asarray(point), axis=1)
            nearest = int(np.argmin(distances))
            if distances[nearest] <= radius:
                point = (float(corners[nearest, 0]), float(corners[nearest, 1]))
                if image is not None:
                    point = _refine_corner(image, point, spacing, radius)
        snapped.append(point)
    return snapped


def _refine_corner(
    image: np.ndarray,
    point: Coordinate,
    spacing: float,
    max_shift: float,
) -> Coordinate:
    # The window must stay narrower than the gap between neighbouring bars
    # (about 4 percent of the registration-box width).
    half_window = max(2, int(round(0.02 * spacing)))
    pixels = np.asarray(image, dtype=np.float32)
    refined = np.asarray([[point]], dtype=np.float32)
    try:
        cv2.cornerSubPix(
            pixels,
            refined,
            (half_window, half_window),
            (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 50, 0.001),
        )
    except cv2.error:
        return point
    x, y = map(float, refined[0, 0])
    if not np.isfinite((x, y)).all() or np.hypot(x - point[0], y - point[1]) > max_shift:
        return point
    return (x, y)


def _validate_points(
    image: np.ndarray,
    points: Sequence[Sequence[float]],
    name: str,
) -> list[Coordinate]:
    height, width = image.shape[:2]
    validated = []
    for index, point in enumerate(points):
        if len(point) != 2:
            raise ValueError(f"{name}[{index}] must be an (x, y) pair")
        x, y = map(float, point)
        if not np.isfinite((x, y)).all():
            raise ValueError(f"{name}[{index}] must contain finite values")
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError(
                f"{name}[{index}] ({x}, {y}) is outside image bounds "
                f"(width={width}, height={height})"
            )
        validated.append((x, y))
    return validated


@dataclass(frozen=True)
class RrtLineSelection:
    """Final line-profile placement for a resolution target image."""

    mode: str
    lines: tuple[Line, ...]
    registration_points: Optional[tuple[Coordinate, Coordinate]] = None

    @property
    def group_coordinates(self) -> dict:
        """Input for ``RrtAnalyzer`` and ``RrtDataPlotter``."""
        return group_coordinates_from_lines(self.lines)

    def to_dict(self) -> dict:
        return {
            "target": "resolution",
            "mode": self.mode,
            "registration_points": (
                [list(point) for point in self.registration_points]
                if self.registration_points is not None
                else None
            ),
            "line_coordinates": [
                [list(start), list(end)] for start, end in self.lines
            ],
        }

    def save(self, path: Union[str, Path]) -> Path:
        """Write YAML and return the path; use .json to explicitly save JSON.

        Paths without an extension receive a .yaml suffix.
        """
        return save_selection(path, self.to_dict())

    @classmethod
    def from_dict(cls, data: dict) -> "RrtLineSelection":
        registration = data.get("registration_points")
        return cls(
            mode=data.get("mode", "manual_coordinates"),
            lines=tuple(
                (tuple(map(float, start)), tuple(map(float, end)))
                for start, end in data["line_coordinates"]
            ),
            registration_points=(
                tuple(tuple(map(float, point)) for point in registration)
                if registration is not None
                else None
            ),
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "RrtLineSelection":
        """Read a YAML selection or a legacy JSON selection."""
        return cls.from_dict(load_selection(path))

    def describe(self) -> str:
        """Coordinates formatted for pasting into a batch script."""
        lines = []
        if self.registration_points is not None:
            points = ", ".join(
                f"({x:.1f}, {y:.1f})" for x, y in self.registration_points
            )
            lines.append(f"REGISTRATION_POINTS = [{points}]")
        line_text = ",\n    ".join(
            f"(({x1:.1f}, {y1:.1f}), ({x2:.1f}, {y2:.1f}))"
            for (x1, y1), (x2, y2) in self.lines
        )
        lines.append(f"LINE_COORDINATES = [\n    {line_text},\n]")
        return "\n".join(lines)


@dataclass(frozen=True, eq=False)
class ResolutionTargetCrop:
    """Auto-cropped chart plus the mapping back to the original image.

    ``image`` is perspective-warped into the USAF1951 template frame and
    bilinearly resampled, so its pixel values are interpolated. It is meant
    for locating the chart; profiles should be sampled from ``source``.
    ``homography`` maps ``source`` ``(x, y)`` to the warped frame, and
    ``offset`` is the crop's top-left corner in that frame.
    """

    image: np.ndarray
    source: np.ndarray
    homography: np.ndarray
    offset: Coordinate

    def to_source(self, points: Sequence[Coordinate]) -> list[Coordinate]:
        """Map cropped-image ``(x, y)`` points to original-image ``(x, y)``."""
        points = np.asarray(points, dtype=float).reshape(-1, 2)
        warped = points + np.asarray(self.offset, dtype=float)
        return _apply_homography(np.linalg.inv(self.homography), warped)

    def to_crop(self, points: Sequence[Coordinate]) -> list[Coordinate]:
        """Map original-image ``(x, y)`` points to cropped-image ``(x, y)``."""
        warped = np.asarray(
            _apply_homography(self.homography, points), dtype=float
        )
        return [
            (float(x), float(y))
            for x, y in warped - np.asarray(self.offset, dtype=float)
        ]

    def line_to_source(self, line: Line) -> Line:
        start, end = self.to_source(line)
        return (start, end)

    def selection_to_source(self, selection: "RrtLineSelection") -> "RrtLineSelection":
        """Express a selection made on ``image`` in ``source`` coordinates.

        Straight lines stay straight under the homography, so mapping the
        endpoints maps the whole line exactly.
        """
        registration = selection.registration_points
        return RrtLineSelection(
            mode=selection.mode,
            lines=tuple(self.line_to_source(line) for line in selection.lines),
            registration_points=(
                tuple(self.to_source(registration))
                if registration is not None
                else None
            ),
        )


def _apply_homography(
    matrix: np.ndarray,
    points: Sequence[Coordinate],
) -> list[Coordinate]:
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    homogeneous = np.column_stack((points, np.ones(len(points)))) @ matrix.T
    mapped = homogeneous[:, :2] / homogeneous[:, 2:3]
    return [(float(x), float(y)) for x, y in mapped]


def select_rrt_lines_from_coordinates(
    image: np.ndarray,
    *,
    registration_points: Optional[Sequence[Coordinate]] = None,
    line_coordinates: Optional[Sequence[Line]] = None,
    snap_to_corners: bool = False,
    mode: str = "manual_coordinates",
) -> RrtLineSelection:
    """Build a selection from fixed coordinates without opening a GUI.

    Supply exactly one of ``registration_points`` (two corners; the standard
    three lines are derived) or ``line_coordinates`` (one ``(start, end)``
    pair per line, Group 0 first, up to six). ``snap_to_corners`` refines
    registration points with Harris corner detection.
    """
    if (registration_points is None) == (line_coordinates is None):
        raise ValueError(
            "Provide exactly one of registration_points or line_coordinates"
        )
    im = load_grayscale_image(image)
    if registration_points is not None:
        points = _validate_points(im, registration_points, "registration_points")
        if len(points) != 2:
            raise ValueError("registration_points must contain two (x, y) pairs")
        if snap_to_corners:
            points = snap_registration_points(im, points)
        return RrtLineSelection(
            mode=mode,
            lines=tuple(lines_from_registration(*points)),
            registration_points=(points[0], points[1]),
        )

    if snap_to_corners:
        raise ValueError("snap_to_corners applies only to registration_points")
    validate_line_count(len(line_coordinates))
    lines = []
    for index, line in enumerate(line_coordinates):
        if len(line) != 2:
            raise ValueError(f"line_coordinates[{index}] must be (start, end)")
        start, end = _validate_points(im, line, f"line_coordinates[{index}]")
        lines.append((start, end))
    return RrtLineSelection(mode=mode, lines=tuple(lines))


@dataclass(eq=False)
class RrtSelectionSession:
    """State for an active Matplotlib resolution-target selection window.

    Points occupy fixed slots: two registration corners, or a start and end
    slot per line. Clicking fills the first empty slot, so clearing a point
    and clicking again replaces exactly that point.
    """

    image: np.ndarray
    mode: str = "manual_live_preview"
    line_count: int = STANDARD_LINE_COUNT
    template: Optional[np.ndarray] = None
    crop: Optional[ResolutionTargetCrop] = None
    remove_tolerance_points: float = 15.0
    close_on_confirm: bool = True
    notebook_controls: bool = False
    result: Optional[RrtLineSelection] = field(default=None, init=False)
    confirmed: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if self.mode not in RRT_GUI_MODES or self.mode == "manual_coordinates":
            raise ValueError(
                "RrtSelectionSession mode must be automatic_refinement, "
                "manual_live_preview, or manual_line_profiles"
            )
        self.image = load_grayscale_image(self.image)
        self._image_float = self.image.astype(np.float32)
        self._profile_image = (
            load_grayscale_image(self.crop.source).astype(np.float32)
            if self.crop is not None
            else self._image_float
        )
        self.line_mode = self.mode == "manual_line_profiles"
        self.snap = self.mode == "automatic_refinement"
        if self.line_mode:
            self.line_count = validate_line_count(self.line_count)
        else:
            self.line_count = STANDARD_LINE_COUNT
        slot_count = 2 * self.line_count if self.line_mode else 2
        self.slots: list[Optional[Coordinate]] = [None] * slot_count
        self._history: list[int] = []
        self._drag_index: Optional[int] = None
        self._corners = detect_harris_corners(self.image) if self.snap else None
        self._analyzer = RrtAnalyzer()
        # Matches the default color cycle used by RrtDataPlotter's plots.
        self._colors = [f"C{group}" for group in range(MAX_LINE_COUNT)]
        self._point_colors = ("#ff4d4d", "#00e5ff")
        self.closed = False

        self.figure = plt.figure(figsize=(13, 8.5))
        manager = getattr(self.figure.canvas, "manager", None)
        if manager is not None and hasattr(manager, "set_window_title"):
            manager.set_window_title("Resolution target selection")
        self.guide_axis = self.figure.add_axes((0.01, 0.73, 0.2, 0.26))
        self.figure.text(
            0.23, 0.975, self._title(),
            fontsize=13, fontweight="bold", va="top",
        )
        self.figure.text(
            0.23, 0.935, self._instructions(),
            fontsize=9, va="top", linespacing=1.5,
        )
        grid = self.figure.add_gridspec(
            self.line_count,
            2,
            left=0.04,
            right=0.98,
            bottom=0.09,
            top=0.7,
            width_ratios=(1.25, 1),
            hspace=0.9 if self.line_count > 3 else 0.6,
            wspace=0.2,
        )
        self.image_axis = self.figure.add_subplot(grid[:, 0])
        self.profile_axes = [
            self.figure.add_subplot(grid[row, 1])
            for row in range(self.line_count)
        ]
        self.image_axis.imshow(self.image, cmap="gray")
        self.image_axis.grid(False)
        self._status = self.figure.text(0.04, 0.02, "", fontsize=10)
        if self.notebook_controls:
            remove_axis = self.figure.add_axes((0.68, 0.01, 0.13, 0.045))
            confirm_axis = self.figure.add_axes((0.83, 0.01, 0.13, 0.045))
            self._remove_button = Button(remove_axis, "Remove last")
            self._confirm_button = Button(confirm_axis, "Confirm")
            self._remove_button.on_clicked(
                lambda _event: self._remove_last_point()
            )
            self._confirm_button.on_clicked(lambda _event: self.confirm())
        self._draw_guide()
        self._init_profile_axes()
        self._artists: list = []
        self._connections = [
            self.figure.canvas.mpl_connect("button_press_event", self._on_click),
            self.figure.canvas.mpl_connect("motion_notify_event", self._on_motion),
            self.figure.canvas.mpl_connect(
                "button_release_event", self._on_release
            ),
            self.figure.canvas.mpl_connect("key_press_event", self._on_key),
            self.figure.canvas.mpl_connect("close_event", self._on_close),
        ]
        self._set_status(self._next_step_text())
        self._redraw()

    def _title(self) -> str:
        if self.line_mode:
            return (
                f"Resolution target: click {self.line_count} line profile(s), "
                "start then end"
            )
        return "Resolution target: place the 2 registration corners shown"

    def _instructions(self) -> str:
        if self.line_mode:
            spans = ", ".join(
                f"G{group} E{group_elements(group)[0]}-6"
                for group in range(self.line_count)
            )
            lines = [
                "Clicks 1-2 = line 1 (Group 0), clicks 3-4 = line 2 "
                "(Group 1), and so on, following the guide.",
                "Start each line above its largest element and end below "
                f"element 6: {spans}.",
                "Drag a +: move  |  Shift+drag: keep the line vertical",
            ]
        else:
            lines = [
                f"1 \u2192 {_REGISTRATION_LABELS[0]}",
                f"2 \u2192 {_REGISTRATION_LABELS[1]}",
                "Drag a +: move  |  lines and profiles update live",
            ]
            if self.snap:
                lines.append(
                    "Each + snaps to the nearest Harris corner (circle)"
                )
        if self.notebook_controls:
            lines.append(
                "Right-click a +: clear it  |  Remove last / Confirm: buttons"
            )
        else:
            lines.append(
                "Right-click a +: clear it  |  Delete: undo last  |  "
                "Enter: confirm"
            )
        return "\n".join(lines)

    def _draw_guide(self) -> None:
        axis = self.guide_axis
        template = self.template
        if template is None:
            try:
                from qal.data import resolution_template

                template = resolution_template()
            except Exception:
                template = None
        axis.axis("off")
        if template is None:
            axis.text(0.5, 0.5, "USAF 1951 guide unavailable",
                      ha="center", va="center", fontsize=8)
            return
        axis.imshow(template, cmap="gray")
        arrow = dict(arrowstyle="-|>", lw=1.4, shrinkA=0, shrinkB=2)
        if not self.line_mode:
            for group, (start, end) in enumerate(
                lines_from_registration(*TEMPLATE_REGISTRATION_POINTS)
            ):
                axis.plot(
                    [start[0], end[0]], [start[1], end[1]],
                    color=self._colors[group], lw=1, ls="--", alpha=0.8,
                )
            offsets = ((-30, -42), (30, 32))
            for number, (point, offset, color) in enumerate(
                zip(TEMPLATE_REGISTRATION_POINTS, offsets, self._point_colors),
                start=1,
            ):
                axis.plot(*point, marker="+", color=color, ms=9, mew=2)
                axis.annotate(
                    str(number),
                    xy=point,
                    xytext=(point[0] + offset[0], point[1] + offset[1]),
                    color=color,
                    fontsize=12,
                    fontweight="bold",
                    ha="center",
                    va="center",
                    arrowprops={**arrow, "color": color},
                    annotation_clip=False,
                )
            return

        for group in range(self.line_count):
            start, end = template_guide_line(group)
            color = self._colors[group]
            axis.annotate(
                "", xy=end, xytext=start,
                arrowprops={**arrow, "color": color},
            )
            if group < 3:
                for number, (x, y), dy in (
                    (2 * group + 1, start, -6),
                    (2 * group + 2, end, 12),
                ):
                    axis.text(
                        x + 5, y + dy, str(number), color=color,
                        fontsize=9, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.1", fc="black",
                                  ec="none", alpha=0.6),
                    )
        if self.line_count > 3:
            axis.text(
                250, 245, "lines 4+ continue\ninto the centre",
                color="white", fontsize=6, ha="right", va="bottom",
            )

    def _init_profile_axes(self) -> None:
        self._profile_artists = []
        for group, axis in enumerate(self.profile_axes):
            color = self._colors[group]
            profile, = axis.plot([], [], color=color, lw=1)
            peaks, = axis.plot([], [], "x", color="red", ms=5)
            troughs, = axis.plot([], [], "o", color="green", ms=3, mfc="none")
            waiting = axis.text(
                0.5, 0.5, f"Waiting for line {group + 1}",
                transform=axis.transAxes, ha="center", va="center",
                color="gray", fontsize=8,
            )
            axis.set_title(
                f"Group {group}  (E{group_elements(group)[0]}-6)",
                fontsize=9, loc="left",
            )
            axis.tick_params(labelsize=7)
            contrast = axis.text(
                1.0, 1.02, "", transform=axis.transAxes,
                ha="right", va="bottom", fontsize=7,
            )
            self._profile_artists.append(
                (profile, peaks, troughs, waiting, contrast)
            )
        self.profile_axes[-1].set_xlabel(
            "Pixel position (original image)"
            if self.crop is not None
            else "Pixel position",
            fontsize=8,
        )

    def _placed_count(self) -> int:
        return sum(slot is not None for slot in self.slots)

    def _next_step_text(self) -> str:
        try:
            index = self.slots.index(None)
        except ValueError:
            return "All points placed. Drag to adjust, then confirm."
        if not self.line_mode:
            return (
                f"Next: click point {index + 1}, the "
                f"{_REGISTRATION_LABELS[index]}"
            )
        group, end = divmod(index, 2)
        part = "end (below element 6)" if end else "start (above the largest element)"
        return (
            f"Next: click {index + 1} = {part} of line {group + 1} "
            f"(Group {group})"
        )

    def registration_points(self) -> Optional[list[Coordinate]]:
        """Current registration points, snapped when refinement is on."""
        if self.line_mode or any(slot is None for slot in self.slots):
            return None
        points = list(self.slots)
        if self.snap:
            points = snap_registration_points(
                self._image_float, points, corners=self._corners
            )
        return points

    def current_lines(self) -> list[Optional[Line]]:
        """Lines from the current points; ``None`` where incomplete."""
        if not self.line_mode:
            points = self.registration_points()
            if points is None:
                return [None] * self.line_count
            return list(lines_from_registration(*points))
        return [
            (self.slots[2 * group], self.slots[2 * group + 1])
            if self.slots[2 * group] is not None
            and self.slots[2 * group + 1] is not None
            else None
            for group in range(self.line_count)
        ]

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
        return event.inaxes is self.image_axis

    def _nearest_slot(self, event) -> Optional[int]:
        placed = [
            (index, slot) for index, slot in enumerate(self.slots)
            if slot is not None
        ]
        if not placed or event.x is None or event.y is None:
            return None
        display = self.image_axis.transData.transform(
            [slot for _, slot in placed]
        )
        distances = np.linalg.norm(
            display - np.asarray([event.x, event.y], dtype=float), axis=1
        )
        nearest = int(np.argmin(distances))
        if distances[nearest] > self.remove_tolerance_points:
            return None
        return placed[nearest][0]

    def _clip(self, x: float, y: float) -> Coordinate:
        height, width = self.image.shape[:2]
        return (
            float(np.clip(x, 0, width - 1)),
            float(np.clip(y, 0, height - 1)),
        )

    def _on_click(self, event) -> None:
        if self.confirmed or not self._event_in_image_axis(event):
            return
        if self._toolbar_is_active():
            return
        if event.button == 1:
            nearest = self._nearest_slot(event)
            if nearest is not None:
                self._drag_index = nearest
                self._set_status(f"Dragging point {nearest + 1}")
                return
            try:
                index = self.slots.index(None)
            except ValueError:
                self._set_status(
                    "All points placed. Drag a + to adjust, right-click to "
                    "clear one, or confirm."
                )
                self.figure.canvas.draw_idle()
                return
            self.slots[index] = self._clip(event.xdata, event.ydata)
            self._history.append(index)
            self._set_status(self._next_step_text())
            self._redraw()
        elif event.button == 3:
            nearest = self._nearest_slot(event)
            if nearest is not None:
                self._clear_slot(nearest)
                self._set_status(
                    f"Cleared point {nearest + 1}. {self._next_step_text()}"
                )
                self._redraw()

    def _clear_slot(self, index: int) -> None:
        self.slots[index] = None
        self._history = [entry for entry in self._history if entry != index]

    def _on_motion(self, event) -> None:
        if (
            self._drag_index is None
            or self.confirmed
            or not self._event_in_image_axis(event)
        ):
            return
        x, y = self._clip(event.xdata, event.ydata)
        if self.line_mode and "shift" in str(event.key or "").lower():
            partner = self.slots[self._drag_index ^ 1]
            if partner is not None:
                x = partner[0]
        self.slots[self._drag_index] = (x, y)
        self._set_status(f"Adjusting point {self._drag_index + 1}")
        self._redraw()

    def _on_release(self, event) -> None:
        if self._drag_index is None or event.button != 1:
            return
        number = self._drag_index + 1
        self._drag_index = None
        self._set_status(f"Placed point {number}. {self._next_step_text()}")
        self._redraw()

    def _on_key(self, event) -> None:
        if self.confirmed:
            return
        key = str(event.key or "").lower()
        if key in {"delete", "backspace"}:
            self._remove_last_point()
        elif key in {"enter", "return"}:
            self.confirm()

    def _on_close(self, _event=None) -> None:
        self.closed = True
        self._drag_index = None
        for connection in self._connections:
            self.figure.canvas.mpl_disconnect(connection)
        self._connections.clear()

    def _remove_last_point(self) -> None:
        if self.confirmed or not self._history:
            return
        index = self._history[-1]
        self._clear_slot(index)
        self._set_status(f"Removed point {index + 1}. {self._next_step_text()}")
        self._redraw()

    def confirm(self) -> Optional[RrtLineSelection]:
        """Confirm the placed points and store an :class:`RrtLineSelection`."""
        if self.confirmed or self.closed:
            return self.result
        missing = len(self.slots) - self._placed_count()
        if missing:
            self._set_status(
                f"{missing} point(s) still needed. {self._next_step_text()}"
            )
            self.figure.canvas.draw_idle()
            return None
        registration = self.registration_points()
        self.result = RrtLineSelection(
            mode=self.mode,
            lines=tuple(self.current_lines()),
            registration_points=(
                tuple(registration) if registration is not None else None
            ),
        )
        self.confirmed = True
        self._set_status("Selection confirmed")
        for connection in self._connections:
            self.figure.canvas.mpl_disconnect(connection)
        self._connections.clear()
        if self.close_on_confirm:
            plt.close(self.figure)
        else:
            self.figure.canvas.draw_idle()
        return self.result

    def _set_status(self, message: str) -> None:
        self._status.set_text(message)

    def _redraw(self) -> None:
        for artist in self._artists:
            artist.remove()
        self._artists.clear()
        axis = self.image_axis

        lines = self.current_lines()
        for group, line in enumerate(lines):
            if line is None:
                continue
            (x1, y1), (x2, y2) = line
            color = self._colors[group]
            drawn, = axis.plot([x1, x2], [y1, y2], color=color, lw=1.8)
            label = axis.text(
                x1, y1, f" G{group}", color=color, fontsize=8,
                va="bottom", fontweight="bold",
            )
            self._artists.extend((drawn, label))

        snapped = self.registration_points() if self.snap else None
        for index, slot in enumerate(self.slots):
            if slot is None:
                continue
            if self.line_mode:
                color = self._colors[index // 2]
            else:
                color = self._point_colors[index]
            outline, = axis.plot(
                *slot, marker="+", ms=13, mew=3.5, color="black", ls="none"
            )
            cross, = axis.plot(
                *slot, marker="+", ms=11, mew=2, color=color, ls="none"
            )
            label = axis.text(
                slot[0], slot[1], f" {index + 1}", color=color, fontsize=9,
                va="top", fontweight="bold",
            )
            self._artists.extend((outline, cross, label))
            if snapped is not None:
                corner = snapped[index]
                ring, = axis.plot(
                    *corner, marker="o", ms=9, mfc="none", mew=1.5,
                    color=color, ls="none",
                )
                self._artists.append(ring)

        self._update_profiles(lines)
        self.figure.canvas.draw_idle()

    def _update_profiles(self, lines: Sequence[Optional[Line]]) -> None:
        for group, (axis, line) in enumerate(zip(self.profile_axes, lines)):
            profile_line, peaks_line, troughs_line, waiting, contrast = (
                self._profile_artists[group]
            )
            axis.xaxis.set_visible(line is not None)
            axis.yaxis.set_visible(line is not None)
            if line is None:
                for artist in (profile_line, peaks_line, troughs_line):
                    artist.set_data([], [])
                waiting.set_visible(True)
                contrast.set_text("")
                continue
            waiting.set_visible(False)
            if self.crop is not None:
                line = self.crop.line_to_source(line)
            (x1, y1), (x2, y2) = line
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    profile = self._analyzer.extract_line_profiles(
                        self._profile_image, x1, y1, x2, y2, offset=1
                    )["center"]
                    stats = self._analyzer.process_group(
                        self._profile_image, x1, y1, x2, y2, group,
                        group_elements(group), window_length=1, polyorder=0,
                    )
            except Exception:
                profile, stats = np.asarray([]), None
            positions = np.arange(len(profile))
            profile_line.set_data(positions, profile)
            peak_indices, trough_indices, labels = [], [], []
            if stats is not None:
                for _, row in stats.iterrows():
                    peak_indices.extend(row["Peak Indices"])
                    trough_indices.extend(row["Trough Indices"])
                    value = row["Percentage Contrast"]
                    text = "--" if np.isnan(value) else f"{value:.0f}%"
                    labels.append(f"E{row['Element']} {text}")
            peaks_line.set_data(peak_indices, profile[peak_indices])
            troughs_line.set_data(trough_indices, profile[trough_indices])
            contrast.set_text("  ".join(labels))
            if len(profile):
                axis.set_xlim(0, max(len(profile) - 1, 1))
                low, high = float(np.min(profile)), float(np.max(profile))
                pad = 0.05 * (high - low) or 1.0
                axis.set_ylim(low - pad, high + pad)


def select_rrt_lines(
    image: np.ndarray,
    *,
    mode: str = "manual_live_preview",
    line_count: int = STANDARD_LINE_COUNT,
    registration_points: Optional[Sequence[Coordinate]] = None,
    line_coordinates: Optional[Sequence[Line]] = None,
    selection_path: Optional[Union[str, Path]] = None,
    crop: Optional[ResolutionTargetCrop] = None,
    verbose: bool = True,
) -> Optional[Union[RrtLineSelection, RrtSelectionSession]]:
    """Choose resolution-target line profiles with a GUI or fixed coordinates.

    Coordinates in and out are in ``image``'s frame. When ``image`` is
    ``crop.image``, pass ``crop`` so the live profiles sample the original
    pixels, then call ``crop.selection_to_source`` before analysis.

    Passing ``registration_points`` or ``line_coordinates`` skips the GUI in
    any mode; ``automatic_refinement`` then snaps registration points to
    Harris corners and the manual modes use them exactly.
    ``manual_coordinates`` requires one of them.

    ``line_count`` (1-6, default 3) applies to ``manual_line_profiles``.
    Registration modes always produce the standard three lines.

    Desktop backends block until confirmation and return an
    :class:`RrtLineSelection`, or ``None`` if the window is closed before
    confirmation. Jupyter widget backends return an
    :class:`RrtSelectionSession`; after Confirm, use ``session.result``.
    ``selection_path`` writes the confirmed selection as YAML for reuse
    (an explicit .json suffix saves JSON).
    """
    if mode not in RRT_GUI_MODES:
        raise ValueError(f"Unknown MODE: {mode}")
    line_count = validate_line_count(line_count)

    if registration_points is not None or line_coordinates is not None:
        selection = select_rrt_lines_from_coordinates(
            image,
            registration_points=registration_points,
            line_coordinates=line_coordinates,
            snap_to_corners=(
                mode == "automatic_refinement"
                and registration_points is not None
            ),
            mode=mode,
        )
        return _finish(selection, selection_path, verbose)
    if mode == "manual_coordinates":
        raise ValueError(
            "manual_coordinates requires registration_points or "
            "line_coordinates"
        )

    notebook = _prepare_gui_backend()
    session = RrtSelectionSession(
        image=image,
        mode=mode,
        line_count=line_count,
        crop=crop,
        close_on_confirm=not notebook,
        notebook_controls=notebook,
    )
    if verbose:
        print(f"Mode: {mode}")
        print(session._instructions())
        if not notebook and _running_in_notebook():
            print("Opening a native Matplotlib window.")
    try:
        plt.show(block=not notebook)
    except KeyboardInterrupt:
        session._on_close()
        plt.close(session.figure)
    if notebook:
        if verbose:
            print(
                "Jupyter session started. Click Confirm, then use "
                "`session.result`."
            )
        return session
    if session.result is None:
        plt.close(session.figure)
        if verbose:
            print(
                "Resolution target selection was closed before confirmation; "
                "nothing was selected."
            )
        return None
    return _finish(session.result, selection_path, verbose)


def _finish(
    selection: RrtLineSelection,
    selection_path: Optional[Union[str, Path]],
    verbose: bool,
) -> RrtLineSelection:
    if verbose:
        print("\nSelected coordinates:")
        print(selection.describe())
    if selection_path is not None:
        path = selection.save(selection_path)
        if verbose:
            print(f"Saved selection to {path}")
    return selection
