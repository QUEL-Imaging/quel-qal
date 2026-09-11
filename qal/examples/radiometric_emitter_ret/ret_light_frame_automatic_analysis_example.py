"""Run single-well RET analysis without a dark frame.

Edit the path below, then run:

    python -m qal.examples.radiometric_emitter_ret.ret_light_frame_automatic_analysis_example
"""

from pathlib import Path

import pandas as pd

from qal import RetAnalyzer, WellDetector
from qal.util import describe_stored_values, load_grayscale_image


# Set the RET signal image (light frame).
IMAGE_PATH = Path("../../data/RET_images/690_nm_LP.tiff")


def main() -> None:
    image = load_grayscale_image(IMAGE_PATH)
    print(describe_stored_values(image).summary())

    detector = WellDetector(parallel_processing=False)
    detections = detector.detect_wells(image, well_ids=["RET"])
    ret_roi = detector.select_single_well(image, detections)
    if ret_roi.empty:
        raise RuntimeError("WellDetector did not find a circular RET well")

    result = RetAnalyzer().analyze(image, ret_roi)

    with pd.option_context("display.max_columns", None):
        print(result.stats_df)
    print("\nNo dark frame supplied; baseline and CNR were not calculated.")


if __name__ == "__main__":
    main()
