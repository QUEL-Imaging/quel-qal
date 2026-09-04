"""Select concentration-grid anchors with the Matplotlib GUI.

With no image argument, QAL's ``cn_sample_3`` example image is used.

Usage:
    python -m qal.examples.concentration_gui_example
    python -m qal.examples.concentration_gui_example path/to/image.tiff
"""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
from skimage import io

from qal import WellDetector, WellAnalyzer, WellPlotter
from qal.data import cn_sample_3
from qal.rta.roi_extraction.well_detector_gui import select_wells_gui


WELL_IDS = [
    "1000 nM",
    "300 nM",
    "100 nM",
    "60 nM",
    "30 nM",
    "10 nM",
    "3 nM",
    "1 nM",
    "Control",
]

# Choose one variation by uncommenting it and commenting out the active one.
# automatic_refinement: local flood-fill around each click (not full-image detect)
# MODE = "automatic_refinement"
# MODE = "manual_live_preview"
MODE = "manual_coordinates"

# Set an exact signal count, or use None in manual_coordinates mode to accept
# up to nine total regions. With None, a double-clicked Control leaves room
# for eight signal wells; otherwise the ninth ordinary click becomes Control.
ANCHOR_COUNT = 5

# In either manual mode, None estimates the diameter as 60% of target grid spacing.
# Set a pixel value (for example 100.0 for a small BMP) to override it.
MANUAL_ROI_DIAMETER = None
REGION_OF_WELL_TO_ANALYZE = 0.5


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "image",
        nargs="?",
        type=Path,
        help="Optional concentration-target image; defaults to cn_sample_3",
    )
    args = parser.parse_args()

    image = (
        io.imread(str(args.image))
        if args.image is not None
        else cn_sample_3()
    )
    detector = WellDetector(parallel_processing=False)

    if MODE not in {
        "automatic_refinement",
        "manual_coordinates",
        "manual_live_preview",
    }:
        raise ValueError(f"Unknown MODE: {MODE}")
    if ANCHOR_COUNT is not None and (
        ANCHOR_COUNT < 3 or ANCHOR_COUNT > 9
    ):
        raise ValueError("ANCHOR_COUNT must be between 3 and 9")
    if ANCHOR_COUNT is None and MODE != "manual_coordinates":
        raise ValueError(
            "ANCHOR_COUNT=None is supported only by manual_coordinates"
        )

    coordinate_mode = (
        "detect" if MODE == "automatic_refinement" else "manual"
    )
    live_preview = MODE == "manual_live_preview"

    print(f"Mode: {MODE}")
    if ANCHOR_COUNT is None:
        print(
            "Select up to nine total regions. Signal wells are ordered "
            "left-to-right, then top-to-bottom."
        )
        print(
            "Double-click a background Control at any time, or use the "
            "ninth ordinary click as Control."
        )
        selection_well_ids = WELL_IDS[:-1]
    else:
        print(
            f"Select the first {ANCHOR_COUNT} wells in row-major order "
            "(left-to-right, then top-to-bottom):"
        )
        for index, well_id in enumerate(
            WELL_IDS[:ANCHOR_COUNT], start=1
        ):
            print(f"  {index}. {well_id}")
        selection_well_ids = WELL_IDS[:ANCHOR_COUNT]
    if coordinate_mode == "manual":
        print("Your clicked coordinates will be used without auto-detection.")
    allow_control = MODE == "manual_coordinates" and (
        ANCHOR_COUNT is None or ANCHOR_COUNT < 9
    )
    if allow_control:
        print(
            "Optional: double left-click a background region to add a Control."
        )
    if live_preview:
        print("Drag any + marker to align the live 3x3 circle overlay.")
    print("Press Enter/Return after all selections.")
    anchors = select_wells_gui(
        image,
        expected_count=ANCHOR_COUNT,
        well_ids=selection_well_ids,
        detector=detector,
        coordinate_mode=coordinate_mode,
        manual_roi_diameter=MANUAL_ROI_DIAMETER,
        enable_live_grid_preview=live_preview,
        allow_control_click=allow_control,
        max_total_count=9 if ANCHOR_COUNT is None else None,
        assume_last_control=ANCHOR_COUNT is None,
    )

    if not isinstance(anchors, pd.DataFrame):
        print(
            "Jupyter session started. After pressing Enter, pass "
            "`anchors.result` to `detector.estimate_remaining_wells_3x3`."
        )
        return

    if MODE == "manual_coordinates":
        # Preserve exactly the selected coordinates. No missing grid positions
        # are inferred in this mode.
        wells = anchors
    else:
        signal_anchors = anchors[anchors["well"] != "Control"]
        wells = detector.estimate_remaining_wells_3x3(
            image,
            signal_anchors,
            well_ids=WELL_IDS,
            show_detected_wells=False,
        )

    with pd.option_context("display.max_columns", None):
        print("\nSelected anchor centroids:")
        print(anchors)
        heading = (
            "\nSelected wells:"
            if MODE == "manual_coordinates"
            else "\nCompleted concentration grid:"
        )
        print(heading)
        print(wells)

    control_rows = wells[wells["well"] == "Control"]
    if MODE == "manual_coordinates" and control_rows.empty:
        print(
            "\nNo Control was selected; concentration baselining and "
            "control-dependent analysis were skipped."
        )
        return

    if control_rows.empty:
        print("\nNo Control well is available; analysis was skipped.")
        return

    if MODE == "manual_coordinates" and len(wells) < 9:
        print(
            "\nAn explicit Control was selected, so baselining and "
            "control-dependent analysis will use the selected subset."
        )

    analyzer = WellAnalyzer(image, wells)
    stats = analyzer.get_stats(
        region_of_well_to_analyze=REGION_OF_WELL_TO_ANALYZE
    )
    plotter = WellPlotter(stats, image=image)
    plotter.visualize_roi()

    with pd.option_context("display.max_columns", None):
        print(stats)


if __name__ == "__main__":
    main()
