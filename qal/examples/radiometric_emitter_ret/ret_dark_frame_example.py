"""Run single-well RET analysis with a measured dark frame.

Edit the paths below, then run:

    python -m qal.examples.radiometric_emitter_ret.ret_dark_frame_example
"""

from pathlib import Path

import pandas as pd

from qal import RetAnalyzer, WellDetector, WellPlotter
from qal.util import load_grayscale_image


# Set the RET signal image (light frame) and matching dark frame.
IMAGE_PATH = Path("../../data/RET_images/690_nm_LP.tiff")
DARK_FRAME_PATH = Path("../../data/RET_images/690_nm_LP_dark_frame.tiff")


def main() -> None:
    image = load_grayscale_image(IMAGE_PATH)
    dark_frame = load_grayscale_image(DARK_FRAME_PATH)

    # Step 1: Detect the RET well.
    detector = WellDetector(parallel_processing=False)
    detections = detector.detect_wells(image, well_ids=["RET"])
    ret_roi = detector.select_single_well(image, detections)
    if ret_roi.empty:
        raise RuntimeError("WellDetector did not find a circular RET well")

    with pd.option_context("display.max_columns", None):
        print("RET ROI:")
        print(ret_roi)

    # Step 2: Analyze the extracted ROI with the dark frame.
    result = RetAnalyzer().analyze(image, ret_roi, dark_frame=dark_frame)
    with pd.option_context("display.max_columns", None):
        print("\nRET statistics:")
        print(result.stats_df)

    # Step 3: Visualize the analyzed ROI.
    WellPlotter(result.stats_df, image=image).visualize_roi()


if __name__ == "__main__":
    main()
