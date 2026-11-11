"""Select concentration-grid anchors using a catalog SKU or unit serial.

Edit the knobs below, then run:

    python -m qal.examples.concentration_rcs.concentration_sku_gui_example

For a notebook that processes one local image and can save results
beside it, open ``qal/examples/jupyter-notebook/concentration_rcs_gui_example.ipynb``.
Set ``IMAGE_PATH`` there to your image file or a built-in example call.

"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from qal.examples._output import results_directory, save_sample_sources, save_target_metadata
from qal import WellAnalyzer, WellPlotter, select_concentration_grid
from qal.data import cn_sample_1
from qal.util import describe_stored_values, load_grayscale_image, lookup_concentration_preset


# Catalog SKU, unit serial, or unique fragment, scoped to concentration.
TARGET_SERIAL = "RCS-S-IC1-S1-A-000"

# Images are loaded and converted to grayscale with load_grayscale_image.
# Already-grayscale images remain unchanged: stored values and dtype are preserved.
# RGB uses scikit-image rgb2gray weights across all channels (floating-point output).
# Choose one option; leave the others commented out.
# Uncomment the next line and put the path to your image:
# IMAGE_PATH = "/path/to/your/image.tiff"

# Example using a file that ships with QAL:
# Run from this example's directory in a GitHub checkout or download.
# IMAGE_PATH = "../../data/concentration_targets/cn_sample_1.tiff"

# Selecting the built-in example image:
IMAGE_PATH = cn_sample_1()

# Writes plots and CSVs beside your input file, or beside this script for samples.
SAVE_RESULTS = True
# Optional: copy built-in source images; user-supplied files are never copied.
SAVE_SOURCE_IMAGES = False

# Choose one variation by uncommenting it and commenting out the active one.
# MODE = "automatic_refinement"
MODE = "manual_live_preview"
# MODE = "manual_coordinates"

# 3–9 row-major anchors, or None in manual_coordinates for up to nine regions.
ANCHOR_COUNT = 3
# Fixed anchors skip the GUI in any mode (e.g. a jig-mounted target).
# The GUI prints these after each selection; SAVE_RESULTS also writes them
# to selection.json so the coordinates can be reused.
COORDINATES = None  # [(x1, y1), (x2, y2), (x3, y3)]
CONTROL_INDICES = None  # manual_coordinates only, e.g. [3]
# None estimates diameter as 60% of grid spacing; set a pixel value to override.
MANUAL_ROI_DIAMETER = None
REGION_OF_WELL_TO_ANALYZE = 0.5


def main() -> None:
    if IMAGE_PATH is None:
        raise ValueError(
            "No image has been supplied. Set IMAGE_PATH to an image file path "
            "or a built-in example call."
        )

    image_path = (
        Path(IMAGE_PATH).expanduser().resolve()
        if not isinstance(IMAGE_PATH, np.ndarray)
        else None
    )
    image = load_grayscale_image(
        image_path if image_path is not None else IMAGE_PATH
    )
    print(describe_stored_values(image).summary())
    target = lookup_concentration_preset(TARGET_SERIAL)
    print(f"Target: {target['display_name']} ({target['tag']})")

    results_dir = None
    if SAVE_RESULTS:
        results_dir = results_directory(IMAGE_PATH, __file__)
        results_dir.mkdir(parents=True, exist_ok=True)
        print(f"Results folder: {results_dir}")

    wells = select_concentration_grid(
        image,
        target["well_ids"],
        mode=MODE,
        anchor_count=ANCHOR_COUNT,
        manual_roi_diameter=MANUAL_ROI_DIAMETER,
        coordinates=COORDINATES,
        control_indices=CONTROL_INDICES,
        selection_path=(
            results_dir / "selection.json" if results_dir is not None else None
        ),
    )
    if not isinstance(wells, pd.DataFrame):
        return

    analyzer = WellAnalyzer(image, wells)
    stats = analyzer.get_stats(
        region_of_well_to_analyze=REGION_OF_WELL_TO_ANALYZE
    )
    analyzer.print_stats(mode=MODE)
    plotter = WellPlotter(stats, image=image)
    roi_path = results_dir / "roi.png" if results_dir is not None else None
    plot_path = (
        results_dir / "concentration.png" if results_dir is not None else None
    )
    plotter.visualize_roi(save_plot=roi_path)
    plotter.plot(
        graph_type="concentration",
        col_to_plot="mean intensity normalized",
        fluorophore_label=target.get("fluorophore_label", "ICG-eq"),
        trendline_lib="statsmodels",
        save_plot=plot_path,
    )
    if results_dir is not None:
        stats_path = results_dir / "stats.csv"
        stats.to_csv(stats_path, index=False)
        save_target_metadata(results_dir, target, query=TARGET_SERIAL)
        if SAVE_SOURCE_IMAGES:
            save_sample_sources(results_dir, IMAGE_PATH)
        print(f"Saved {roi_path}")
        print(f"Saved {plot_path}")
        print(f"Saved {stats_path}")


if __name__ == "__main__":
    main()
