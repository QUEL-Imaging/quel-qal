"""Run single-well RET analysis without a dark frame.

Edit the path below, then run:

    python -m qal.examples.radiometric_emitter_ret.ret_light_frame_automatic_analysis_example
"""

from pathlib import Path

import pandas as pd

from qal import RetAnalyzer, WellDetector
from qal.util import describe_stored_values, load_grayscale_image


# Replace with your light-frame file. Relative paths start where Python is run.
IMAGE_PATH = Path("path/to/light_frame.tiff").expanduser()


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
