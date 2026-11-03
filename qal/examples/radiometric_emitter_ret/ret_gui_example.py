"""Select and analyze one RET well with the Matplotlib GUI.

Edit the path below, then run:

    python -m qal.examples.radiometric_emitter_ret.ret_gui_example
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from qal import RetAnalyzer, WellDetector, WellPlotter
from qal.rta.roi_extraction.well_detector_gui import (
    select_wells_from_coordinates,
    select_wells_gui,
)
from qal.util import (
    describe_stored_values,
    load_grayscale_image,
    mean_in_circular_roi,
    std_in_circular_roi,
)


# Replace with your light-frame file. Relative paths start where Python is run.
IMAGE_PATH = Path("path/to/light_frame.tiff").expanduser()

# Choose one variation.
MODE = "automatic_refinement"
# MODE = "manual_coordinates"

# A single manual RET point has no neighboring signal well from which to
# estimate diameter, so manual mode requires an explicit pixel diameter.
MANUAL_ROI_DIAMETER = 250.0

# Fixed (x, y) for the RET well skips the GUI (e.g. a jig-mounted emitter).
# automatic_refinement re-centers it on the well; manual uses it exactly.
COORDINATES = None  # [(x, y)]


def main() -> None:
    image = load_grayscale_image(IMAGE_PATH)
    print(describe_stored_values(image).summary())
    detector = WellDetector(parallel_processing=False)
    if MODE not in {"automatic_refinement", "manual_coordinates"}:
        raise ValueError(f"Unknown MODE: {MODE}")
    coordinate_mode = (
        "detect" if MODE == "automatic_refinement" else "manual"
    )

    print(f"Mode: {MODE}")
    manual_roi_diameter = (
        MANUAL_ROI_DIAMETER if coordinate_mode == "manual" else None
    )
    if COORDINATES is not None:
        selections = select_wells_from_coordinates(
            image,
            COORDINATES,
            well_ids=["RET"],
            detector=detector,
            search_radius=300,
            coordinate_mode=coordinate_mode,
            manual_roi_diameter=manual_roi_diameter,
        )
    else:
        print("Left-click the RET signal well.")
        print("Drag the + marker to reposition the selected point.")
        if coordinate_mode == "manual":
            print(
                "Optional: double left-click a background region to add a "
                "Control."
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
            manual_roi_diameter=manual_roi_diameter,
            enable_live_grid_preview=True,
            allow_control_click=coordinate_mode == "manual",
        )

    # Desktop backends return a DataFrame. In a Jupyter widget backend the
    # function returns a session immediately; run analysis after Confirm with:
    # result = RetAnalyzer().analyze(image, session.result)
    if selections is None:
        return
    if not isinstance(selections, pd.DataFrame):
        print(
            "Jupyter session started. Click Confirm (or press Enter if the "
            "figure has focus), then use "
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

    WellPlotter(result.stats_df, image=image).visualize_roi()


if __name__ == "__main__":
    main()
