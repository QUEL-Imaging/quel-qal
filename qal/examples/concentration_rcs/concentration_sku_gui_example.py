"""Select concentration-grid anchors from a catalog SKU or unit serial.

Looks up the matching RCS product preset, then opens the Matplotlib
well-selection GUI. Edit the knobs below, then run:

    python -m qal.examples.concentration_rcs.concentration_sku_gui_example
"""

from __future__ import annotations

import pandas as pd

from qal import WellAnalyzer, WellPlotter, select_concentration_grid
from qal.data import cn_sample_3
from qal.util import (
    describe_stored_values,
    load_grayscale_image,
    lookup_concentration_preset,
)


# Catalog SKU, unit serial, unique fragment, or family-scoped tag.
# Examples: "RCS-ICG-ST01-QUEL03", "RCS-S-IC1-S1-A-000", "V3-ICG-ST01-RCS000".
TARGET_SERIAL = "RCS-S-IC1-S1-A-000"

# None uses QAL's cn_sample_3 example image.
IMAGE_PATH = None  # "path/to/image.tiff"

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
    image = (
        load_grayscale_image(IMAGE_PATH)
        if IMAGE_PATH is not None
        else cn_sample_3()
    )
    print(describe_stored_values(image).summary())
    target = lookup_concentration_preset(TARGET_SERIAL)

    print(f"Query: {TARGET_SERIAL}")
    print(f"Target: {target['display_name']} ({target['tag']})")
    if target.get("old_sku"):
        print(f"Old SKU: {target['old_sku']}")
    if target.get("current_sku"):
        print(f"Current SKU: {target['current_sku']}")

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
    WellPlotter(stats, image=image).visualize_roi()


if __name__ == "__main__":
    main()
