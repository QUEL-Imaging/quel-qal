"""Run single-well RET analysis with a measured dark frame.

Usage:
    python -m qal.examples.ret_dark_frame_example sample.tiff dark.tiff
"""

from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
from skimage import io

from qal import RetAnalyzer, WellDetector


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="RET signal image")
    parser.add_argument("dark_frame", type=Path, help="Matching dark frame")
    parser.add_argument(
        "--show-detection",
        action="store_true",
        help="Display the detected RET ROI",
    )
    args = parser.parse_args()

    image = io.imread(str(args.image))
    dark_frame = io.imread(str(args.dark_frame))

    detector = WellDetector(parallel_processing=False)
    detections = detector.detect_wells(image, well_ids=["RET"])
    ret_roi = detector.select_single_well(image, detections)
    if ret_roi.empty:
        raise RuntimeError("WellDetector did not find a circular RET well")

    result = RetAnalyzer().analyze(
        image,
        ret_roi,
        dark_frame=dark_frame,
    )

    with pd.option_context("display.max_columns", None):
        print(result.stats_df)

    if args.show_detection:
        detector.plot_detected_wells(image, ret_roi)


if __name__ == "__main__":
    main()
