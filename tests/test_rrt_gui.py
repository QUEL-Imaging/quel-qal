from pathlib import Path
from types import SimpleNamespace

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytest
from skimage import io

from qal import (
    RrtLineSelection,
    RrtROI,
    RrtSelectionSession,
    select_rrt_lines,
    select_rrt_lines_from_coordinates,
)
from qal.rta.roi_extraction.rrt_gui import (
    NESTED_GROUP_ORIGIN,
    NESTED_GROUP_SCALE,
    REGISTRATION_LINE_RATIOS,
    TEMPLATE_REGISTRATION_POINTS,
    group_elements,
    group_line_ratios,
    lines_from_registration,
    snap_registration_points,
)


IMAGE = np.zeros((120, 160), dtype=np.float32)
TEMPLATE = np.zeros((20, 20))


def _event(session, x, y, *, button=1, key=None):
    session.figure.canvas.draw()
    display_x, display_y = session.image_axis.transData.transform((x, y))
    return SimpleNamespace(
        inaxes=session.image_axis,
        xdata=x,
        ydata=y,
        x=display_x,
        y=display_y,
        button=button,
        key=key,
    )


def _click(session, x, y, **kwargs):
    event = _event(session, x, y, **kwargs)
    session._on_click(event)
    session._on_release(event)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_group_elements_follow_usaf_layout():
    assert list(group_elements(0)) == [2, 3, 4, 5, 6]
    assert list(group_elements(1)) == [1, 2, 3, 4, 5, 6]
    assert list(group_elements(4)) == [2, 3, 4, 5, 6]
    with pytest.raises(ValueError):
        group_elements(6)


def test_registration_lines_match_legacy_rrt_roi():
    roi = RrtROI.__new__(RrtROI)
    legacy = roi.process_resolution_target([(78, 56), (215, 202)])
    lines = lines_from_registration((56, 78), (202, 215))

    for group, line in enumerate(lines):
        assert np.allclose(legacy[group]["coordinates"], line)
    assert list(legacy[1]["elements"]) == [1, 2, 3, 4, 5, 6]


def test_legacy_process_rejects_missing_corner():
    roi = RrtROI.__new__(RrtROI)
    with pytest.raises(ValueError, match="Two registration corners"):
        roi.process_resolution_target([(78, 56)])


def test_snap_moves_to_nearby_corner_and_keeps_distant_points():
    corners = np.asarray([[11.0, 21.0], [140.0, 100.0]])
    snapped = snap_registration_points(
        None, [(10, 20), (100, 90)], corners=corners
    )

    assert snapped[0] == (11.0, 21.0)
    assert snapped[1] == (100.0, 90.0)


TEMPLATE_PATH = (
    Path(__file__).resolve().parents[1]
    / "qal" / "data" / "USAF1951_template" / "res_source.png"
)


def _bar_centre(row_profile, near):
    """Half-max centre of the bright run containing index ``near``."""
    profile = np.asarray(row_profile, dtype=float)
    half = (np.percentile(profile, 5) + profile[near]) / 2
    start = end = near
    while profile[start - 1] >= half:
        start -= 1
    while profile[end + 1] >= half:
        end += 1
    left = start - 1 + (half - profile[start - 1]) / (profile[start] - profile[start - 1])
    right = end + (profile[end] - half) / (profile[end] - profile[end + 1])
    return (left + right) / 2


@pytest.mark.parametrize(
    ("group", "element_6_row"),
    [(0, 213), (1, 165), (2, 160)],
)
def test_registration_lines_centre_on_element_6_in_template(group, element_6_row):
    template = io.imread(TEMPLATE_PATH).astype(float)
    (x, _), _ = lines_from_registration(*TEMPLATE_REGISTRATION_POINTS)[group]

    centre = _bar_centre(template[element_6_row], int(round(x)))

    assert abs(x - centre) < 0.5


def test_nested_layout_reproduces_group_2_ratios():
    origin_x, origin_y = NESTED_GROUP_ORIGIN
    x0, top0, bottom0, _ = REGISTRATION_LINE_RATIOS[0]
    x2, top2, bottom2, _ = REGISTRATION_LINE_RATIOS[2]

    assert x2 == pytest.approx(origin_x + x0 * NESTED_GROUP_SCALE, abs=1e-3)
    assert top2 == pytest.approx(origin_y + top0 * NESTED_GROUP_SCALE, abs=1e-3)
    assert bottom2 == pytest.approx(origin_y + bottom0 * NESTED_GROUP_SCALE, abs=1e-3)
    assert group_line_ratios(3)[0] == pytest.approx(0.746, abs=1e-3)


def test_snapping_refines_harris_corner_to_bar_edge():
    image = np.zeros((80, 120), dtype=np.float32)
    image[20:30, 20:60] = 1.0
    image[50:60, 70:100] = 1.0
    image = cv2.GaussianBlur(image, (5, 5), 1.0)

    snapped = snap_registration_points(image, [(21, 21), (98, 58)])

    assert np.allclose(snapped[0], (19.5, 19.5), atol=0.35)
    assert np.allclose(snapped[1], (99.5, 59.5), atol=0.35)


def test_coordinates_build_lines_without_gui():
    lines = [((10, 5), (10, 80)), ((50, 5), (50, 60))]
    selection = select_rrt_lines(
        IMAGE, mode="manual_coordinates", line_coordinates=lines, verbose=False
    )

    groups = selection.group_coordinates
    assert list(groups) == [0, 1]
    assert groups[1]["coordinates"] == ((50.0, 5.0), (50.0, 60.0))
    assert list(groups[1]["elements"]) == [1, 2, 3, 4, 5, 6]


@pytest.mark.parametrize("count", [0, 7])
def test_line_coordinates_count_is_limited(count):
    lines = [((10, 5), (10, 80))] * count
    with pytest.raises(ValueError, match="between 1 and 6"):
        select_rrt_lines_from_coordinates(IMAGE, line_coordinates=lines)


def test_coordinates_must_be_inside_image():
    with pytest.raises(ValueError, match="outside image bounds"):
        select_rrt_lines_from_coordinates(
            IMAGE, registration_points=[(10, 10), (500, 10)]
        )


def test_manual_coordinates_requires_coordinates():
    with pytest.raises(ValueError, match="requires registration_points"):
        select_rrt_lines(IMAGE, mode="manual_coordinates", verbose=False)


def test_line_count_knob_is_validated():
    with pytest.raises(ValueError, match="LINE_COUNT"):
        select_rrt_lines(IMAGE, mode="manual_line_profiles", line_count=7)


def test_selection_round_trips_through_json(tmp_path):
    selection = select_rrt_lines_from_coordinates(
        IMAGE, registration_points=[(20, 20), (140, 110)]
    )
    path = selection.save(tmp_path / "selection.json")

    loaded = RrtLineSelection.load(path)
    assert loaded.registration_points == ((20.0, 20.0), (140.0, 110.0))
    assert np.allclose(loaded.lines, selection.lines)


def test_line_session_clicks_fill_lines_in_order_and_confirm():
    session = RrtSelectionSession(
        IMAGE, mode="manual_line_profiles", line_count=2,
        template=TEMPLATE, close_on_confirm=False,
    )
    for point in [(10, 5), (10, 80), (60, 5), (60, 70)]:
        _click(session, *point)

    result = session.confirm()
    assert result.lines == (((10.0, 5.0), (10.0, 80.0)), ((60.0, 5.0), (60.0, 70.0)))
    assert result.registration_points is None


def test_cleared_point_is_refilled_by_next_click():
    session = RrtSelectionSession(
        IMAGE, mode="manual_line_profiles", line_count=2,
        template=TEMPLATE, close_on_confirm=False,
    )
    for point in [(10, 5), (10, 80), (60, 5)]:
        _click(session, *point)
    _click(session, 10, 80, button=3)
    _click(session, 12, 90)

    assert session.slots[1] == (12.0, 90.0)
    assert session.slots[3] is None
    assert session.confirm() is None


def test_shift_drag_keeps_line_vertical():
    session = RrtSelectionSession(
        IMAGE, mode="manual_line_profiles", line_count=1,
        template=TEMPLATE, close_on_confirm=False,
    )
    _click(session, 30, 10)
    _click(session, 30, 90)
    session._on_click(_event(session, 30, 90))
    session._on_motion(_event(session, 45, 100, key="shift"))
    session._on_release(_event(session, 45, 100))

    assert session.slots[1] == (30.0, 100.0)


def test_closing_window_before_confirm_returns_none(monkeypatch, capsys):
    monkeypatch.setattr(
        "qal.rta.roi_extraction.rrt_gui._prepare_gui_backend", lambda: False
    )
    monkeypatch.setattr(plt, "show", lambda **kwargs: plt.close("all"))

    result = select_rrt_lines(IMAGE, mode="manual_live_preview")

    assert result is None
    assert "closed before confirmation" in capsys.readouterr().out
    assert not plt.get_fignums()


def test_closed_session_ignores_confirm():
    session = RrtSelectionSession(
        IMAGE, mode="manual_live_preview", template=TEMPLATE,
        close_on_confirm=False,
    )
    _click(session, 20, 20)
    _click(session, 140, 110)
    session._on_close()

    assert session.confirm() is None
    assert session.closed


def test_line_colors_match_contrast_plot_cycle():
    session = RrtSelectionSession(
        IMAGE, mode="manual_line_profiles", line_count=3, template=TEMPLATE,
    )
    assert session._colors[:3] == ["C0", "C1", "C2"]


def test_live_preview_drag_updates_derived_lines():
    session = RrtSelectionSession(
        IMAGE, mode="manual_live_preview", template=TEMPLATE,
        close_on_confirm=False,
    )
    _click(session, 20, 20)
    _click(session, 140, 110)
    before = session.current_lines()[0]
    session._on_click(_event(session, 20, 20))
    session._on_motion(_event(session, 30, 25))
    session._on_release(_event(session, 30, 25))

    assert session.current_lines()[0] != before
    result = session.confirm()
    assert result.registration_points == ((30.0, 25.0), (140.0, 110.0))
    assert len(result.lines) == 3


def test_crop_maps_lines_back_to_source_pixels():
    from skimage import transform as sk_transform

    from qal import ResolutionTargetCrop

    source = np.zeros((200, 300), dtype=np.float32)
    source[40:160, 100:110] = 1.0
    homography = np.array([[0.5, 0.05, -20.0], [0.0, 0.5, -10.0], [0.0, 0.0, 1.0]])
    warped = sk_transform.warp(
        source,
        sk_transform.ProjectiveTransform(homography).inverse,
        output_shape=source.shape,
        preserve_range=True,
    )
    crop = ResolutionTargetCrop(
        image=warped[5:, 10:], source=source, homography=homography,
        offset=(10.0, 5.0),
    )

    source_point = (104.5, 100.0)
    crop_point = crop.to_crop([source_point])[0]
    assert np.allclose(crop.to_source([crop_point])[0], source_point)
    assert crop.image[int(round(crop_point[1])), int(round(crop_point[0]))] > 0.5

    selection = RrtLineSelection(
        mode="manual_line_profiles",
        lines=(tuple(crop.to_crop([(104.5, 30.0), (104.5, 170.0)])),),
    )
    (start, end), = crop.selection_to_source(selection).lines
    assert np.allclose(start, (104.5, 30.0))
    assert np.allclose(end, (104.5, 170.0))
