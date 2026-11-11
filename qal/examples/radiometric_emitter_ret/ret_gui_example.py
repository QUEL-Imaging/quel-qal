"""Select one RET well, with an optional matching dark-frame baseline.

Edit the settings below, then run from the repository root:

    python -m qal.examples.radiometric_emitter_ret.ret_gui_example

Click the light-frame well, drag the + if needed, and press Enter to confirm.
SAVE_RESULTS writes stats.csv and roi.png beside your light frame, or beside
this script for built-in samples. Set DARK_FRAME_PATH = None for light-only analysis.
The default inputs are the matching ret_sample_1_lf and ret_sample_1_df frames.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from qal.examples._output import results_directory, save_sample_sources
from qal import (
    RetAnalysisConfig,
    RetAnalyzer,
    WellDetector,
    WellPlotter,
    select_wells_from_coordinates,
    select_wells_gui,
)
from qal.data import ret_sample_1_df, ret_sample_1_lf
from qal.util import (describe_stored_values, load_grayscale_image,
                      mean_in_circular_roi, std_in_circular_roi)


# --- Settings ---------------------------------------------------------------

# Use the built-in pair, or replace both inputs with your own matching file paths.
# Relative paths start from the folder where you launch Python.
# Absolute paths work too; on Windows, forward slashes work: C:/path/to/file.tiff.
# Images are loaded and converted to grayscale with load_grayscale_image.
# Already-grayscale images remain unchanged: stored values and dtype are preserved.
# RGB uses scikit-image rgb2gray weights across all channels (floating-point output).
IMAGE_PATH = ret_sample_1_lf()
DARK_FRAME_PATH = ret_sample_1_df()
# IMAGE_PATH = "path/to/light_frame.tiff"
# DARK_FRAME_PATH = "path/to/dark_frame.tiff"
# DARK_FRAME_PATH = None  # Light frame only: no baseline or CNR.

# True saves beside your input file, or beside this script for built-in samples.
SAVE_RESULTS = True
# Copy original built-in frames only; user-supplied images are not copied.
SAVE_SOURCE_IMAGES = False

# Pick one mode. Leave the others commented out.
MODE = "automatic_refinement"  # click the well; refine its center and size
# MODE = "manual_coordinates"  # use the exact click and the diameter below
# MODE = "automatic_detection"  # no picker; detect and choose the brightest well

# Leave as None to click in the window. One (x, y) pair skips the window.
# Used by automatic_refinement and manual_coordinates, not automatic_detection.
COORDINATES = None
# COORDINATES = [(120, 100)]

# Manual selection needs an explicit well diameter in pixels.
# MANUAL_ROI_DIAMETER = 250.0
# Search distance in pixels around the click in automatic_refinement mode.
SEARCH_RADIUS = 300.0

# Fraction of the well DIAMETER to measure: 0.5 measures the inner half.
REGION_OF_WELL_TO_ANALYZE = 0.5
CNR_THRESHOLD = 3.0


def analyze_selection(image, dark_frame, wells, *, image_input=None,
                      output_fallback=None, save_results=None, save_sources=None,
                      region=None, cnr_threshold=None, dark_input=None):
    """Measure confirmed wells; also usable after Confirm in JupyterLab."""
    if wells is None or wells.empty:
        raise ValueError("Confirm one RET well before running analysis.")

    image_input = IMAGE_PATH if image_input is None else image_input
    output_fallback = __file__ if output_fallback is None else output_fallback
    save_results = SAVE_RESULTS if save_results is None else save_results
    save_sources = SAVE_SOURCE_IMAGES if save_sources is None else save_sources
    region = REGION_OF_WELL_TO_ANALYZE if region is None else region
    cnr_threshold = CNR_THRESHOLD if cnr_threshold is None else cnr_threshold
    if dark_input is None and dark_frame is not None:
        dark_input = DARK_FRAME_PATH

    # Apply the same location and inner ROI size to the measured dark frame.
    analyzer = RetAnalyzer(
        analysis_config=RetAnalysisConfig(
            region_of_well_to_analyze=region,
            cnr_threshold=cnr_threshold,
        )
    )
    ret_wells = wells[wells["well"] == "RET"]
    result = analyzer.analyze(image, ret_wells, dark_frame=dark_frame)
    controls = wells[wells["well"] == "Control"]
    if not controls.empty:
        control = controls.iloc[0]
        center = (control.x, control.y)
        radius = control["ROI Radius"]
        print(f"Selected background estimate: mean={mean_in_circular_roi(image, center, radius):.3f}, "
              f"std={std_in_circular_roi(image, center, radius):.3f}")
        print("Background is reported separately; only a dark frame baselines the result.")
    print("\nSelected RET well:")
    print(wells.to_string(index=False))
    print("\nRET statistics:")
    print(result.stats_df.to_string(index=False))
    if not result.is_baselined:
        print("No dark frame supplied; baseline and CNR were not calculated.")
    row = ret_wells.iloc[0]
    print(f"\nCOORDINATES = [({float(row.x):.3f}, {float(row.y):.3f})]")

    # Save before showing the plot, so closing a window cannot erase the file.
    results_dir = None
    roi_path = None
    if save_results:
        results_dir = results_directory(image_input, output_fallback)
        results_dir.mkdir(parents=True, exist_ok=True)
        result.stats_df.to_csv(results_dir / "stats.csv", index=False)
        roi_path = results_dir / "roi.png"
        if save_sources:
            save_sample_sources(results_dir, image_input, dark_input)

    WellPlotter(result.stats_df, image=image).visualize_roi(save_plot=roi_path)
    if results_dir is not None:
        print(f"\nSaved stats.csv and roi.png to {results_dir}")
    return result


def run_ret_example(image_input, dark_input, *, mode="automatic_refinement",
                    coordinates=None, manual_roi_diameter=None, search_radius=300.,
                    save_results=True, save_sources=False, output_fallback=None,
                    region=0.5, cnr_threshold=3.):
    """Shared GUI/automatic workflow; configuration is local to each run."""
    # 1. Check the settings before opening a window or creating output files.
    if mode not in {"automatic_refinement", "manual_coordinates", "automatic_detection"}:
        raise ValueError(
            "Choose MODE = 'automatic_refinement', 'manual_coordinates', "
            "or 'automatic_detection'."
        )
    if mode == "automatic_detection" and coordinates is not None:
        raise ValueError("Set COORDINATES = None when using automatic_detection.")

    if image_input is None:
        raise ValueError("Set IMAGE_PATH to a light-frame file or built-in sample.")

    image_path = (
        None if isinstance(image_input, np.ndarray)
        else Path(image_input).expanduser().resolve()
    )
    dark_path = (
        None if dark_input is None or isinstance(dark_input, np.ndarray)
        else Path(dark_input).expanduser().resolve()
    )
    for setting, path in (("IMAGE_PATH", image_path), ("DARK_FRAME_PATH", dark_path)):
        if path is not None and not path.is_file():
            raise FileNotFoundError(f"Set {setting} to an existing image file. File not found: {path}")

    # 2. Load the frames. Their pixels must describe the same image geometry.
    image = load_grayscale_image(image_input if image_path is None else image_path)
    dark_frame = (None if dark_input is None else
                  load_grayscale_image(dark_input if dark_path is None else dark_path))
    if dark_frame is not None and image.shape != dark_frame.shape:
        raise ValueError(
            "The light frame and dark frame must have the same dimensions; "
            f"got {image.shape} and {dark_frame.shape}."
        )
    print(f"Light frame: {describe_stored_values(image).summary()}")
    if dark_frame is not None:
        print(f"Dark frame:  {describe_stored_values(dark_frame).summary()}")
    print(f"Mode: {mode}")

    # 3. Select the well in the LIGHT frame. Do not select a separate dark ROI.
    detector = WellDetector(parallel_processing=False)
    if mode == "automatic_detection":
        detections = detector.detect_wells(image, well_ids=["RET"])
        wells = detector.select_single_well(image, detections)
        if wells.empty:
            raise RuntimeError(
                "No circular RET well was found. Try mode = 'automatic_refinement' "
                "to click the well, or 'manual_coordinates' with a pixel diameter."
            )
        wells = wells.copy()
        wells["well"] = "RET"
    else:
        coordinate_mode = "detect" if mode == "automatic_refinement" else "manual"
        diameter = manual_roi_diameter if coordinate_mode == "manual" else None
        if coordinate_mode == "manual" and diameter is None:
            raise ValueError("Uncomment MANUAL_ROI_DIAMETER and set a pixel diameter for manual_coordinates.")
        if coordinates is not None:
            wells = select_wells_from_coordinates(
                image, coordinates, well_ids=["RET"], detector=detector,
                coordinate_mode=coordinate_mode, manual_roi_diameter=diameter,
                search_radius=search_radius,
            )
        else:
            print("Click the RET well. Drag the + to adjust it; right-click to remove it.")
            print("Press Enter to confirm, or close the window to cancel.")
            wells = select_wells_gui(
                image, expected_count=1, well_ids=["RET"], detector=detector,
                coordinate_mode=coordinate_mode, manual_roi_diameter=diameter,
                search_radius=search_radius, enable_live_grid_preview=True,
                allow_control_click=coordinate_mode == "manual",
            )

    if wells is None:
        print("Well selection cancelled.")
        return None
    if not isinstance(wells, pd.DataFrame):
        print(
            "JupyterLab session started. Run `session = example.main()` to keep it. "
            "After Confirm, run:\n"
            "example.analyze_selection(session.image, "
            "None if example.DARK_FRAME_PATH is None else load_grayscale_image(example.DARK_FRAME_PATH), "
            "session.result)\n"
            "See the RET section of doc/well_detector_gui_README.md for the imports."
        )
        return wells

    # 4. Measure, print the statistics, optionally save, then show the ROI plot.
    return analyze_selection(
        image, dark_frame, wells, image_input=image_input, dark_input=dark_input,
        output_fallback=output_fallback, save_results=save_results,
        save_sources=save_sources, region=region, cnr_threshold=cnr_threshold,
    )



def main():
    return run_ret_example(
        IMAGE_PATH, DARK_FRAME_PATH, mode=MODE, coordinates=COORDINATES,
        manual_roi_diameter=globals().get("MANUAL_ROI_DIAMETER"),
        search_radius=SEARCH_RADIUS, save_results=SAVE_RESULTS,
        save_sources=SAVE_SOURCE_IMAGES, output_fallback=__file__,
        region=REGION_OF_WELL_TO_ANALYZE, cnr_threshold=CNR_THRESHOLD,
    )


if __name__ == "__main__":
    main()
