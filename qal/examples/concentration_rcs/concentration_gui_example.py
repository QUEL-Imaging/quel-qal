"""Select concentration-grid anchors with the Matplotlib GUI.

Edit the knobs below, then run:

    python -m qal.examples.concentration_rcs.concentration_gui_example

For a notebook that processes one local image and can save results
beside it, open ``concentration_gui_example.ipynb`` in this folder.
Set ``IMAGE_PATH`` there, or leave it as ``None`` for the example image.

For SKU or unit-serial lookup, use ``concentration_sku_gui_example``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from qal import WellAnalyzer, WellPlotter, select_concentration_grid
from qal.data import cn_sample_3, get_concentration_target_preset
from qal.util import describe_stored_values, load_grayscale_image


# Select a packaged QUEL product preset: "icg", "q800", or "q700".
# Use "custom" to activate CUSTOM_WELL_IDS.
TARGET_TAG = "icg"

CUSTOM_WELL_IDS = [
    "1000 nM",
    "300 nM",
    "100 nM",
    "60 nM",
    "30 nM",
    "10 nM",
    "3 nM",
    "1 nM",
    "Control",
]

# None uses QAL's cn_sample_3 example image.
IMAGE_PATH = None  # "path/to/image.tiff"
# Writes plots and CSVs to <image_stem>_results next to IMAGE_PATH.
SAVE_RESULTS = False

# Choose one variation by uncommenting it and commenting out the active one.
# MODE = "automatic_refinement"
MODE = "manual_live_preview"
# MODE = "manual_coordinates"

# 3–9 row-major anchors, or None in manual_coordinates for up to nine regions.
ANCHOR_COUNT = 3
# None estimates diameter as 60% of grid spacing; set a pixel value to override.
MANUAL_ROI_DIAMETER = None
REGION_OF_WELL_TO_ANALYZE = 0.5


def main() -> None:
    image_path = (
        Path(IMAGE_PATH).expanduser().resolve()
        if IMAGE_PATH is not None
        else None
    )
    image = (
        load_grayscale_image(image_path)
        if image_path is not None
        else cn_sample_3()
    )
    print(describe_stored_values(image).summary())
    target = get_concentration_target_preset(
        TARGET_TAG,
        custom_well_ids=(
            CUSTOM_WELL_IDS if TARGET_TAG == "custom" else None
        ),
    )
    print(f"Target: {target['display_name']} ({target['tag']})")

    results_dir = None
    if SAVE_RESULTS:
        if image_path is None:
            raise ValueError(
                "SAVE_RESULTS requires IMAGE_PATH so results can be written "
                "next to the image."
            )
        results_dir = image_path.parent / f"{image_path.stem}_results"
        results_dir.mkdir(parents=True, exist_ok=True)
        print(f"Results folder: {results_dir}")

    wells = select_concentration_grid(
        image,
        target["well_ids"],
        mode=MODE,
        anchor_count=ANCHOR_COUNT,
        manual_roi_diameter=MANUAL_ROI_DIAMETER,
    )
    if not isinstance(wells, pd.DataFrame):
        return

    analyzer = WellAnalyzer(image, wells)
    stats = analyzer.get_stats(
        region_of_well_to_analyze=REGION_OF_WELL_TO_ANALYZE
    )
    analyzer.print_stats(mode=MODE)
    plotter = WellPlotter(stats, image=image)
    plotter.visualize_roi()
    if results_dir is not None:
        roi_path = results_dir / "roi.png"
        plt.savefig(roi_path, bbox_inches="tight")
        wells_path = results_dir / "wells.csv"
        stats_path = results_dir / "stats.csv"
        wells.to_csv(wells_path, index=False)
        stats.to_csv(stats_path, index=False)
        print(f"Saved {roi_path}")
        print(f"Saved {wells_path}")
        print(f"Saved {stats_path}")


if __name__ == "__main__":
    main()
