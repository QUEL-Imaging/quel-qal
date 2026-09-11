from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest

from qal import WellDetector
from qal.rta.roi_extraction.well_detector_gui import (
    WellSelectionSession,
    select_wells_from_coordinates,
)
from qal.util import circular_mask


def test_manual_coordinates_bypass_detection_and_estimate_diameter():
    class DetectorThatMustNotRun:
        def detect_wells(self, *args, **kwargs):
            raise AssertionError("automatic detection was called")

        def refine_well_at_coordinate(self, *args, **kwargs):
            raise AssertionError("local refinement was called")

    points = [(10, 10), (30, 10), (50, 10), (10, 30)]
    selected = select_wells_from_coordinates(
        np.zeros((80, 80)),
        points,
        detector=DetectorThatMustNotRun(),
        coordinate_mode="manual",
    )

    assert np.allclose(selected[["x", "y"]], points)
    assert np.allclose(selected["ROI Diameter"], 12)
    assert np.allclose(selected["ROI Radius"], 6)


def test_manual_control_is_exact_and_excluded_from_diameter_estimate():
    signal_points = [(10, 20), (30, 20), (80, 20), (160, 20)]
    control = (11, 20)
    points = [*signal_points, control]

    selected = select_wells_from_coordinates(
        np.zeros((100, 220)),
        points,
        well_ids=["A", "B", "C", "D"],
        coordinate_mode="manual",
        control_indices=[4],
    )

    assert np.allclose(selected[["x", "y"]], points)
    assert selected["well"].tolist() == ["A", "B", "C", "D", "Control"]
    # The minimum signal-to-signal distance is 20 px. The 1 px distance to
    # the Control must not influence the 60% diameter estimate.
    assert np.allclose(selected["ROI Diameter"], 12)


def test_double_click_converts_latest_point_to_control():
    session = WellSelectionSession(
        np.zeros((100, 100)),
        expected_count=2,
        well_ids=["A", "B"],
        coordinate_mode="manual",
        manual_roi_diameter=10,
        allow_control_click=True,
        close_on_confirm=False,
    )
    session.figure.canvas.draw()

    def click(x, y, *, double=False):
        display_x, display_y = session.image_axis.transData.transform((x, y))
        session._on_click(
            SimpleNamespace(
                inaxes=session.image_axis,
                xdata=x,
                ydata=y,
                x=display_x,
                y=display_y,
                button=1,
                dblclick=double,
            )
        )

    click(20, 20)
    click(80, 80)
    click(80, 80, double=True)

    assert session.points == [(20.0, 20.0), (80.0, 80.0)]
    assert session.control_indices == {1}
    assert session._signal_points() == [(20.0, 20.0)]
    plt.close(session.figure)


def test_open_ended_selection_makes_ninth_click_control():
    session = WellSelectionSession(
        np.zeros((120, 220)),
        expected_count=None,
        well_ids=[f"Well {index}" for index in range(1, 9)],
        coordinate_mode="manual",
        manual_roi_diameter=10,
        allow_control_click=True,
        max_total_count=9,
        assume_last_control=True,
        close_on_confirm=False,
    )
    session.figure.canvas.draw()

    for index in range(9):
        x, y = 10 + 20 * index, 50
        display_x, display_y = session.image_axis.transData.transform((x, y))
        session._on_click(
            SimpleNamespace(
                inaxes=session.image_axis,
                xdata=x,
                ydata=y,
                x=display_x,
                y=display_y,
                button=1,
                dblclick=False,
            )
        )

    assert len(session.points) == 9
    assert session.control_indices == {8}
    assert len(session._signal_points()) == 8

    selected = select_wells_from_coordinates(
        session.image,
        session.points,
        well_ids=session.well_ids,
        coordinate_mode="manual",
        manual_roi_diameter=10,
        control_indices=session.control_indices,
    )
    assert selected["well"].tolist() == [
        *[f"Well {index}" for index in range(1, 9)],
        "Control",
    ]
    plt.close(session.figure)


def test_early_control_limits_open_ended_selection_to_eight_signals():
    session = WellSelectionSession(
        np.zeros((160, 240)),
        expected_count=None,
        well_ids=[f"Well {index}" for index in range(1, 9)],
        coordinate_mode="manual",
        manual_roi_diameter=10,
        allow_control_click=True,
        max_total_count=9,
        assume_last_control=True,
        close_on_confirm=False,
    )
    session.figure.canvas.draw()

    def click(x, y, *, double=False):
        display_x, display_y = session.image_axis.transData.transform((x, y))
        session._on_click(
            SimpleNamespace(
                inaxes=session.image_axis,
                xdata=x,
                ydata=y,
                x=display_x,
                y=display_y,
                button=1,
                dblclick=double,
            )
        )

    click(20, 20)
    click(200, 130)
    click(200, 130, double=True)
    for index in range(7):
        click(40 + 20 * index, 60)
    click(200, 60)

    assert len(session.points) == 9
    assert len(session.control_indices) == 1
    assert len(session._signal_points()) == 8
    assert (200.0, 60.0) not in session.points
    plt.close(session.figure)


def test_similarity_grid_fit_uses_more_than_three_anchors():
    canonical = np.asarray(
        [
            (0, 0), (15, 0), (30, 0),
            (0, 15), (15, 15), (30, 15),
            (0, 30), (15, 30), (30, 30),
        ],
        dtype=float,
    )
    angle = np.deg2rad(8)
    transform = np.asarray(
        [
            [4 * np.cos(angle), -4 * np.sin(angle)],
            [4 * np.sin(angle), 4 * np.cos(angle)],
        ]
    )
    expected = canonical @ transform.T + (100, 50)

    result = WellDetector(
        parallel_processing=False
    ).compute_transformed_points(canonical[:4], expected[:4])

    assert np.allclose(result, expected)


def test_single_point_live_preview_can_be_dragged():
    session = WellSelectionSession(
        np.zeros((80, 80)),
        expected_count=1,
        well_ids=["RET"],
        coordinate_mode="manual",
        manual_roi_diameter=20,
        enable_live_grid_preview=True,
        close_on_confirm=False,
    )
    session.figure.canvas.draw()
    session.points.append((30, 28))
    session._redraw()

    display_x, display_y = session.image_axis.transData.transform((30, 28))
    session._on_click(
        SimpleNamespace(
            inaxes=session.image_axis,
            xdata=30,
            ydata=28,
            x=display_x,
            y=display_y,
            button=1,
            dblclick=False,
        )
    )
    assert session._drag_index == 0
    assert session.points == [(30, 28)]

    session._on_motion(
        SimpleNamespace(inaxes=session.image_axis, xdata=36, ydata=33)
    )
    session._on_release(SimpleNamespace(button=1))

    assert session.points[0] == (36, 33)
    assert session._drag_index is None
    circles = [
        artist
        for artist in session._artists
        if type(artist).__name__ == "Circle"
    ]
    assert len(circles) == 1
    assert circles[0].center == (36, 33)
    assert circles[0].radius == 10
    plt.close(session.figure)


def test_live_preview_anchor_can_be_dragged():
    session = WellSelectionSession(
        np.zeros((100, 100)),
        expected_count=3,
        coordinate_mode="manual",
        enable_live_grid_preview=True,
        close_on_confirm=False,
    )
    session.points.extend([(20, 20), (40, 20), (60, 20)])
    session._drag_index = 0

    session._on_motion(
        SimpleNamespace(
            inaxes=session.image_axis,
            xdata=22,
            ydata=24,
        )
    )
    session._on_release(SimpleNamespace(button=1))

    assert session.points[0] == (22, 24)
    assert session._drag_index is None
    assert sum(
        type(artist).__name__ == "Circle"
        for artist in session._artists
    ) == 9
    for index in range(3):
        circle_color = session._artists[index].get_edgecolor()
        crosshair_color = session._artists[10 + 3 * index].get_color()
        label_color = session._artists[11 + 3 * index].get_color()
        assert np.allclose(circle_color, crosshair_color)
        assert np.allclose(circle_color, label_color)
    plt.close(session.figure)


def test_notebook_detection_helpers_recognize_widget_backends():
    from qal.rta.roi_extraction import well_detector_gui as gui

    assert gui._running_in_notebook() is False
    assert gui._is_notebook_widget_backend("module://ipympl.backend_nbagg")
    assert gui._is_notebook_widget_backend("nbAgg")
    assert gui._is_notebook_widget_backend("widget")
    assert not gui._is_notebook_widget_backend(
        "module://matplotlib_inline.backend_inline"
    )
    assert not gui._is_notebook_widget_backend("Agg")
    assert gui._prepare_gui_backend() is False


def test_prepare_gui_backend_switches_notebook_inline(monkeypatch):
    from qal.rta.roi_extraction import well_detector_gui as gui

    monkeypatch.setattr(gui, "_running_in_notebook", lambda: True)
    monkeypatch.setattr(gui, "_running_in_vscode_jupyter", lambda: False)
    monkeypatch.setattr(gui, "_is_desktop_gui_backend", lambda backend=None: False)
    state = {"widget": False}

    def is_widget(backend=None):
        return state["widget"]

    def magic(name):
        if name in {"widget", "ipympl"}:
            state["widget"] = True
            return True
        return False

    monkeypatch.setattr(gui, "_is_notebook_widget_backend", is_widget)
    monkeypatch.setattr(gui, "_run_matplotlib_magic", magic)
    assert gui._prepare_gui_backend() is True


def test_prepare_gui_backend_errors_when_notebook_cannot_go_interactive(
    monkeypatch,
):
    from qal.rta.roi_extraction import well_detector_gui as gui

    monkeypatch.setattr(gui, "_running_in_notebook", lambda: True)
    monkeypatch.setattr(gui, "_running_in_vscode_jupyter", lambda: False)
    monkeypatch.setattr(gui, "_ipympl_frontend_available", lambda: True)
    monkeypatch.setattr(gui, "_is_notebook_widget_backend", lambda backend=None: False)
    monkeypatch.setattr(gui, "_is_desktop_gui_backend", lambda backend=None: False)
    monkeypatch.setattr(gui, "_run_matplotlib_magic", lambda name: False)
    monkeypatch.setattr(gui, "_switch_backend", lambda name: False)
    monkeypatch.setattr(gui, "_activate_desktop_gui_backend", lambda: False)
    with pytest.raises(RuntimeError, match="interactive Matplotlib"):
        gui._prepare_gui_backend()


def test_prepare_gui_backend_skips_ipympl_in_vscode(monkeypatch):
    from qal.rta.roi_extraction import well_detector_gui as gui

    monkeypatch.setattr(gui, "_running_in_notebook", lambda: True)
    monkeypatch.setattr(gui, "_running_in_vscode_jupyter", lambda: True)
    monkeypatch.setattr(gui, "_is_notebook_widget_backend", lambda backend=None: True)
    monkeypatch.setattr(gui, "_is_desktop_gui_backend", lambda backend=None: False)
    activated = []
    monkeypatch.setattr(
        gui,
        "_activate_desktop_gui_backend",
        lambda: activated.append(True) or True,
    )
    assert gui._prepare_gui_backend() is False
    assert activated


def test_notebook_controls_confirm_resolves_manual_points():
    session = WellSelectionSession(
        np.zeros((40, 40)),
        expected_count=3,
        well_ids=["A", "B", "C"],
        coordinate_mode="manual",
        manual_roi_diameter=10,
        notebook_controls=True,
        resolve_in_background=False,
        close_on_confirm=False,
    )
    assert session._confirm_button is not None
    assert session._remove_button is not None
    session.points.extend([(5, 5), (20, 5), (35, 5)])
    result = session.confirm()
    assert result is not None
    assert list(result["well"]) == ["A", "B", "C"]
    assert session.confirmed
    plt.close(session.figure)


def _synthetic_wells(centers, radius=12, intensities=None, shape=(120, 180)):
    image = np.zeros(shape, dtype=np.float64)
    intensities = intensities or [100] * len(centers)
    for (x, y), intensity in zip(centers, intensities):
        image[circular_mask(shape, (x, y), radius)] = intensity
    return image


def test_local_refinement_recovers_dim_and_bright_wells():
    centers = [(40, 40), (90, 40), (140, 40), (40, 90)]
    image = _synthetic_wells(
        centers,
        radius=14,
        intensities=[120, 40, 12, 8],
    )
    detector = WellDetector(parallel_processing=False)

    refined = select_wells_from_coordinates(
        image,
        [(42, 38), (88, 43), (143, 37), (37, 92)],
        detector=detector,
        coordinate_mode="detect",
        search_radius=30,
    )

    assert len(refined) == 4
    for expected, (_, row) in zip(centers, refined.iterrows()):
        assert np.hypot(row["x"] - expected[0], row["y"] - expected[1]) < 2.5
        assert 20 < row["ROI Diameter"] < 35


def test_local_refinement_rejects_empty_background_click():
    image = _synthetic_wells([(60, 60)], radius=12, intensities=[80])
    detector = WellDetector(parallel_processing=False)

    with pytest.raises(RuntimeError, match="Could not refine"):
        select_wells_from_coordinates(
            image,
            [(150, 100)],
            detector=detector,
            coordinate_mode="detect",
            search_radius=20,
        )
