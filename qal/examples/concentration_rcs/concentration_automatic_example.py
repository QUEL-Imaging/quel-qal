"""Find concentration-target wells automatically, then analyze them.

Edit the knobs below, then run:

    python -m qal.examples.concentration_rcs.concentration_automatic_example

Leave ``IMAGE_PATH`` as ``None`` to use QAL's built-in example image.
To use your own file, set ``IMAGE_PATH`` to a TIFF, BMP, PNG, or JPEG.
"""

from __future__ import annotations

from pathlib import Path

from qal import WellAnalyzer, WellDetector, WellPlotter
from qal.data import cn_sample_3, get_concentration_target_preset
from qal.util import describe_stored_values, load_grayscale_image


# Select a packaged QUEL product preset: "icg", "q800", or "q700".
TARGET_TAG = "icg"

# None uses QAL's cn_sample_3 example image.
IMAGE_PATH = None  # "path/to/image.tiff"
# Writes plots and CSVs to <image_stem>_results next to IMAGE_PATH.
SAVE_RESULTS = False

# Fraction of each well used for intensity.
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

    target = get_concentration_target_preset(TARGET_TAG)
    print(f"Target: {target['display_name']}")

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
        wells_path = results_dir / "wells.csv"
        stats_path = results_dir / "stats.csv"
        wells.to_csv(wells_path, index=False)
        stats.to_csv(stats_path, index=False)
        print(f"Saved {roi_path}")
        print(f"Saved {plot_path}")
        print(f"Saved {wells_path}")
        print(f"Saved {stats_path}")


if __name__ == "__main__":
    main()
