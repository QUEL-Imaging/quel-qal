"""Select and analyze one RET well with the Matplotlib GUI.

Usage:
    python -m qal.examples.ret_gui_example
    python -m qal.examples.ret_gui_example path/to/image.tiff
"""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
from skimage import io

from qal import RetAnalyzer, WellDetector
from qal.rta.roi_extraction.well_detector_gui import select_wells_gui
from qal.util import (
    mean_in_circular_roi,
    std_in_circular_roi,
)


DEFAULT_IMAGE = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "RET_images"
    / "690_nm_LP.tiff"
)

# Choose one variation.
MODE = "automatic_refinement"
# MODE = "manual_coordinates"

# A single manual RET point has no neighboring signal well from which to
# estimate diameter, so manual mode requires an explicit pixel diameter.
MANUAL_ROI_DIAMETER = 250.0


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "image",
        nargs="?",
        type=Path,
        default=DEFAULT_IMAGE,
        help=f"RET image (default: {DEFAULT_IMAGE})",
    )
    args = parser.parse_args()

    image = io.imread(str(args.image))
    detector = WellDetector(parallel_processing=False)
    if MODE not in {"automatic_refinement", "manual_coordinates"}:
        raise ValueError(f"Unknown MODE: {MODE}")
    coordinate_mode = (
        "detect" if MODE == "automatic_refinement" else "manual"
    )

    print(f"Mode: {MODE}")
    print("Left-click the RET signal well.")
    if coordinate_mode == "manual":
        print(
            "Optional: double left-click a background region to add a Control."
        )
    print("Right-click near the crosshair to remove an accidental selection.")
    print("Press Enter/Return to confirm.")
    selections = select_wells_gui(
        image,
        expected_count=1,
        well_ids=["RET"],
        detector=detector,
        search_radius=300,
        coordinate_mode=coordinate_mode,
        manual_roi_diameter=(
            MANUAL_ROI_DIAMETER
            if coordinate_mode == "manual"
            else None
        ),
        allow_control_click=coordinate_mode == "manual",
    )

    # Desktop backends return a DataFrame. In a Jupyter widget backend the
    # function returns a session immediately; run analysis after Enter with:
    # result = RetAnalyzer().analyze(image, session.result)
    if not isinstance(selections, pd.DataFrame):
        print(
            "Jupyter session started. After pressing Enter, use "
            "`RetAnalyzer().analyze(image, selections.result)`."
        )
        return

    ret_roi = selections[selections["well"] == "RET"]
    result = RetAnalyzer().analyze(image, ret_roi)
    with pd.option_context("display.max_columns", None):
        print("\nResolved RET ROI:")
        print(ret_roi)
        print("\nRaw RET statistics:")
        print(result.stats_df)

    control_rows = selections[selections["well"] == "Control"]
    if not control_rows.empty:
        control = control_rows.iloc[0]
        center = (control["x"], control["y"])
        radius = control["ROI Radius"]
        print(
            "\nSelected background estimate: "
            f"mean={mean_in_circular_roi(image, center, radius):.3f}, "
            f"std={std_in_circular_roi(image, center, radius):.3f}"
        )
        print(
            "This background is reported separately and is not used to "
            "baseline the RET result."
        )

    detector.plot_detected_wells(image, selections)


if __name__ == "__main__":
    main()
