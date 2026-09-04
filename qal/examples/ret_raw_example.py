"""Run raw single-well RET analysis without baselining.

Usage:
    python -m qal.examples.ret_raw_example sample.tiff
"""

from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
from skimage import io

from qal import RetAnalyzer, WellDetector


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="RET signal image")
    args = parser.parse_args()

    image = io.imread(str(args.image))

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
