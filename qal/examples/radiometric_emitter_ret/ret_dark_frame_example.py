"""Select one RET well and measure it against a matching dark frame.

Edit the settings below, then run from the repository root:

    python -m qal.examples.radiometric_emitter_ret.ret_dark_frame_example

Click the light-frame well, drag the + if needed, and press Enter to confirm.
SAVE_RESULTS writes wells.csv, stats.csv and roi.png beside the light frame.
There is no bundled RET image pair; supply your own light and dark frames.
"""

from pathlib import Path

import pandas as pd

from qal import (
    RetAnalysisConfig,
    RetAnalyzer,
    WellDetector,
    WellPlotter,
    select_wells_from_coordinates,
    select_wells_gui,
)
from qal.util import describe_stored_values, load_grayscale_image


# --- Settings ---------------------------------------------------------------

# Replace these placeholders with the paths to YOUR files.
# Relative paths start from the folder where you launch Python.
# Absolute paths work too; on Windows, forward slashes work: C:/path/to/file.tiff.
IMAGE_PATH = "path/to/light_frame.tiff"
DARK_FRAME_PATH = "path/to/dark_frame.tiff"

# True saves to <image_stem>_results in the same folder as IMAGE_PATH.
SAVE_RESULTS = False

# Pick one mode. Leave the others commented out.
MODE = "automatic_refinement"  # click the well; refine its center and size
# MODE = "manual_coordinates"  # use the exact click and the diameter below
# MODE = "automatic_detection"  # no picker; detect and choose the brightest well

# Leave as None to click in the window. One (x, y) pair skips the window.
# Used by automatic_refinement and manual_coordinates, not automatic_detection.
COORDINATES = None
# COORDINATES = [(120, 100)]

# Manual selection needs an explicit well diameter in pixels.
MANUAL_ROI_DIAMETER = 250.0
# Search distance in pixels around the click in automatic_refinement mode.
SEARCH_RADIUS = 300.0

# Fraction of the well DIAMETER to measure: 0.5 measures the inner half.
REGION_OF_WELL_TO_ANALYZE = 0.5
CNR_THRESHOLD = 3.0


def analyze_selection(image, dark_frame, wells):
    """Measure confirmed wells; also usable after Confirm in JupyterLab."""
    if wells is None or wells.empty:
        raise ValueError("Confirm one RET well before running analysis.")

    # Apply the same location and inner ROI size to the measured dark frame.
    analyzer = RetAnalyzer(
        analysis_config=RetAnalysisConfig(
            region_of_well_to_analyze=REGION_OF_WELL_TO_ANALYZE,
            cnr_threshold=CNR_THRESHOLD,
        )
    )
    result = analyzer.analyze(image, wells, dark_frame=dark_frame)
    print("\nSelected RET well:")
    print(wells.to_string(index=False))
    print("\nRET statistics (measured dark-frame baseline):")
    print(result.stats_df.to_string(index=False))
    row = wells.iloc[0]
    print(f"\nCOORDINATES = [({float(row.x):.3f}, {float(row.y):.3f})]")

    # Save before showing the plot, so closing a window cannot erase the file.
    results_dir = None
    roi_path = None
    if SAVE_RESULTS:
        image_path = Path(IMAGE_PATH).expanduser().resolve()
        results_dir = image_path.parent / f"{image_path.stem}_results"
        results_dir.mkdir(parents=True, exist_ok=True)
        wells.to_csv(results_dir / "wells.csv", index=False)
        result.stats_df.to_csv(results_dir / "stats.csv", index=False)
        roi_path = results_dir / "roi.png"

    WellPlotter(result.stats_df, image=image).visualize_roi(save_plot=roi_path)
    if results_dir is not None:
        print(f"\nSaved wells.csv, stats.csv and roi.png to {results_dir}")
    return result


def main():
    # 1. Check the settings before opening a window or creating output files.
    if MODE not in {"automatic_refinement", "manual_coordinates", "automatic_detection"}:
        raise ValueError(
            "Choose MODE = 'automatic_refinement', 'manual_coordinates', "
            "or 'automatic_detection'."
        )
    if MODE == "automatic_detection" and COORDINATES is not None:
        raise ValueError("Set COORDINATES = None when using automatic_detection.")

    image_path = Path(IMAGE_PATH).expanduser().resolve()
    dark_path = Path(DARK_FRAME_PATH).expanduser().resolve()
    for setting, path in (("IMAGE_PATH", image_path), ("DARK_FRAME_PATH", dark_path)):
        if not path.is_file():
            raise FileNotFoundError(f"Set {setting} to an existing image file. File not found: {path}")

    # 2. Load the frames. Their pixels must describe the same image geometry.
    image = load_grayscale_image(image_path)
    dark_frame = load_grayscale_image(dark_path)
    if image.shape != dark_frame.shape:
        raise ValueError(
            "The light frame and dark frame must have the same dimensions; "
            f"got {image.shape} and {dark_frame.shape}."
        )
    print(f"Light frame: {describe_stored_values(image).summary()}")
    print(f"Dark frame:  {describe_stored_values(dark_frame).summary()}")
    print(f"Mode: {MODE}")

    # 3. Select the well in the LIGHT frame. Do not select a separate dark ROI.
    detector = WellDetector(parallel_processing=False)
    if MODE == "automatic_detection":
        detections = detector.detect_wells(image, well_ids=["RET"])
        wells = detector.select_single_well(image, detections)
        if wells.empty:
            raise RuntimeError(
                "No circular RET well was found. Try MODE = 'automatic_refinement' "
                "to click the well, or 'manual_coordinates' with a pixel diameter."
            )
    else:
        coordinate_mode = "detect" if MODE == "automatic_refinement" else "manual"
        diameter = MANUAL_ROI_DIAMETER if coordinate_mode == "manual" else None
        if COORDINATES is not None:
            wells = select_wells_from_coordinates(
                image, COORDINATES, well_ids=["RET"], detector=detector,
                coordinate_mode=coordinate_mode, manual_roi_diameter=diameter,
                search_radius=SEARCH_RADIUS,
            )
        else:
            print("Click the RET well. Drag the + to adjust it; right-click to remove it.")
            print("Press Enter to confirm, or close the window to cancel.")
            wells = select_wells_gui(
                image, expected_count=1, well_ids=["RET"], detector=detector,
                coordinate_mode=coordinate_mode, manual_roi_diameter=diameter,
                search_radius=SEARCH_RADIUS, enable_live_grid_preview=True,
            )

    if wells is None:
        print("Well selection cancelled.")
        return None
    if not isinstance(wells, pd.DataFrame):
        print(
            "JupyterLab session started. Run `session = example.main()` to keep it. "
            "After Confirm, run:\n"
            "example.analyze_selection(session.image, "
            "load_grayscale_image(Path(example.DARK_FRAME_PATH).expanduser()), "
            "session.result)\n"
            "See the RET section of well_detector_gui_README.md for the imports."
        )
        return wells

    # 4. Measure, print the statistics, optionally save, then show the ROI plot.
    return analyze_selection(image, dark_frame, wells)


if __name__ == "__main__":
    main()
