import numpy as np

from qal import RrtAnalyzer


def test_flat_topped_bars_are_detected_once_at_their_centre():
    profile = np.array(
        [0, 0, 8, 16, 16, 16, 8, 0, 0, 8, 16, 16, 8, 0, 0, 8, 16, 16, 16, 16, 8, 0],
        dtype=float,
    )

    peaks, troughs = RrtAnalyzer().detect_peaks_and_troughs(profile)

    assert peaks == [4, 10, 17]
    assert troughs == [7, 13]


def test_shallow_wiggle_on_a_bar_top_is_merged():
    profile = np.array(
        [0, 100, 200, 196, 200, 100, 0, 100, 200, 100, 0], dtype=float
    )

    peaks, troughs = RrtAnalyzer().detect_peaks_and_troughs(profile)

    assert peaks == [2, 8]
    assert troughs == [6]


def test_strict_extrema_match_previous_behaviour():
    rng = np.random.default_rng(0)
    profile = np.sin(np.linspace(0, 12 * np.pi, 300)) + 0.01 * rng.standard_normal(300)

    peaks, troughs = RrtAnalyzer().detect_peaks_and_troughs(profile, min_swing=0)

    gradient = np.diff(profile)
    old_peaks = [i for i in range(1, len(gradient)) if gradient[i] < 0 < gradient[i - 1]]
    old_troughs = [i for i in range(1, len(gradient)) if gradient[i - 1] < 0 < gradient[i]]
    assert peaks == old_peaks
    assert troughs == old_troughs


def test_line_profile_plot_marks_every_detected_extremum():
    import matplotlib.pyplot as plt
    import pandas as pd

    from qal import RrtDataPlotter

    image = np.zeros((40, 60), dtype=np.float32)
    image[:, 10:60:6] = 200.0
    group_coordinates = {0: {"coordinates": ((5, 20), (55, 20)), "elements": [2]}}
    df = pd.DataFrame({
        "Group": [0, 0],
        "Element": [5, 6],
        "Peak Indices": [[10, 16, 22], [28]],
        "Trough Indices": [[13, 19, 25], []],
    })

    RrtDataPlotter().plot_line_profiles(image, group_coordinates, df)
    axis = plt.gcf().axes[1]
    marked = sorted(
        int(x) for collection in axis.collections for x, _ in collection.get_offsets()
    )
    plt.close("all")

    assert marked == sorted([10, 16, 22, 28, 13, 19, 25])


def test_print_smallest_resolved_reports_the_last_bar_above_the_threshold(capsys):
    import pandas as pd

    from qal import RrtROI

    contrast = pd.DataFrame({
        "Group": [0, 0, 1],
        "Element": [5, 6, 1],
        "Resolution (lp/mm)": [1.59, 1.78, 2.0],
        "Percentage Contrast": [40.0, 26.0, 10.0],
    })

    RrtROI.print_smallest_resolved(contrast, 26)

    assert "Group 0, Element 6, 1.78 lp/mm" in capsys.readouterr().out
