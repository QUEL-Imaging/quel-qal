"""Run single-well RET analysis without a dark frame.

Uses ret_sample_1_lf by default. To use your own image, edit IMAGE_PATH, then run:

    python -m qal.examples.radiometric_emitter_ret.ret_light_frame_automatic_analysis_example
"""

from pathlib import Path

import pandas as pd

from qal import RetAnalyzer, WellDetector
from qal.data import ret_sample_1_lf
from qal.util import describe_stored_values, load_grayscale_image


# Use the built-in sample, or replace with your light-frame file path.
# Relative paths start where Python is run.
# Images are loaded and converted to grayscale with load_grayscale_image.
# Already-grayscale images remain unchanged: stored values and dtype are preserved.
# RGB uses scikit-image rgb2gray weights across all channels (floating-point output).
IMAGE_PATH = ret_sample_1_lf()
# IMAGE_PATH = Path("path/to/light_frame.tiff").expanduser()


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
