"""Find concentration-target wells automatically, then analyze them.

Edit the knobs below, then run:

    python -m qal.examples.concentration_rcs.concentration_automatic_example

The example uses ``IMAGE_PATH = cn_sample_1()`` to load a built-in image.
To use your own file, set ``IMAGE_PATH`` to a TIFF, BMP, PNG, or JPEG.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from qal import WellAnalyzer, WellDetector, WellPlotter
from qal.data import cn_sample_1, get_concentration_target_preset
from qal.util import describe_stored_values, load_grayscale_image


# Select a packaged QUEL product preset: "icg", "q800", or "q700".
TARGET_TAG = "icg"

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

# Writes plots and CSVs to <image_stem>_results next to IMAGE_PATH.
SAVE_RESULTS = False

# Fraction of each well used for intensity.
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

    target = get_concentration_target_preset(TARGET_TAG)
    print(f"Target: {target['display_name']}")

    results_dir = None
    if SAVE_RESULTS:
        if image_path is None:
            raise ValueError(
                "SAVE_RESULTS requires a file path in IMAGE_PATH so results can be written "
                "next to the image."
            )
        results_dir = image_path.parent / f"{image_path.stem}_results"
        results_dir.mkdir(parents=True, exist_ok=True)
        print(f"Results folder: {results_dir}")

    # 1. Find the wells. This script does not open an interactive GUI window.
    detector = WellDetector()
    wells = detector.detect_wells(
        image,
        show_detected_wells=True,
        set_consistent_roi_region=True,
    )
    wells = detector.estimate_remaining_wells_3x3(
        image,
        wells,
        well_ids=target["well_ids"],
    )
    print("\nDetected wells:")
    print(wells)

    # 2. Measure each well.
    analyzer = WellAnalyzer(image, wells)
    stats = analyzer.get_stats(
        region_of_well_to_analyze=REGION_OF_WELL_TO_ANALYZE
    )
    print("\nWell statistics:")
    print(stats)

    # 3. Show the wells and the concentration curve.
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
        plot_error_bars=False,
        save_plot=plot_path,
    )
    if results_dir is not None:
        stats_path = results_dir / "stats.csv"
        stats.to_csv(stats_path, index=False)
        print(f"Saved {roi_path}")
        print(f"Saved {plot_path}")
        print(f"Saved {stats_path}")


if __name__ == "__main__":
    main()
