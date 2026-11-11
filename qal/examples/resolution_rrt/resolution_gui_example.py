"""Place USAF 1951 line profiles with the Matplotlib GUI and measure contrast.

Edit the settings below, then run:

    python -m qal.examples.resolution_rrt.resolution_gui_example

The window prints the coordinates you chose. With SAVE_RESULTS on, it also
writes plots, a CSV, and selection.yaml next to your image, or next to this
script when using the built-in example. Paste those coordinates into
rrt_batch_example to measure more images without the window.

The matching notebook is ``qal/examples/jupyter-notebook/resolution_rrt_gui_example.ipynb``.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from qal.examples._output import results_directory, save_sample_sources
from qal import RrtAnalyzer, RrtDataPlotter, RrtLineSelection, RrtROI, select_rrt_lines
from qal.data import res_sample_1
from qal.util import load_grayscale_image


# --- Settings ---------------------------------------------------------------

# Images are loaded and converted to grayscale with load_grayscale_image.
# Already-grayscale images remain unchanged: stored values and dtype are preserved.
# RGB uses scikit-image rgb2gray weights across all channels (floating-point output).
# Choose one option; leave the others commented out.
# Uncomment the next line and put the path to your image:
# IMAGE_PATH = "/path/to/your/image.tiff"

# Example using a file that ships with QAL:
# Run from this example's directory in a GitHub checkout or download.
# IMAGE_PATH = "../../data/resolution_targets/res_sample_1.tiff"

# Selecting the built-in example image:
IMAGE_PATH = res_sample_1()


# True saves plots, a CSV, and selection.yaml.
# With a file path, files go next to that image in <image_name>_results/.
# With built-in images, files go next to this script in resolution_gui_example_results/.
SAVE_RESULTS = True
# Optional copying of the original built-in image only.
SAVE_SOURCE_IMAGES = False

# Pick one mode. Leave the others commented out.
# MODE = "automatic_refinement"  # click two corners; they snap onto the chart
MODE = "manual_live_preview"  # drag two crosshairs; the profiles update as you drag
# MODE = "manual_line_profiles"  # draw each measurement line yourself
# MODE = "manual_coordinates"  # no window; use the coordinates below

# True shows a straightened crop of the chart, which makes clicking easier.
# The contrast numbers are still measured on the original image.
AUTO_CROP = True

# Used only by manual_line_profiles. How many lines to draw, from 1 to 6.
# Line 1 measures Group 0, line 2 measures Group 1, and so on.
LINE_COUNT = 3

# Leave both as None to choose the points in the window.
# To skip the window, fill in one of them with coordinates from a previous run.
REGISTRATION_POINTS = None
# REGISTRATION_POINTS = [(54.9, 76.6), (202.8, 215.4)]

LINE_COORDINATES = None
# LINE_COORDINATES = [
#     ((63.5, 71.7), (63.5, 220.0)),
#     ((198.7, 74.2), (198.7, 169.3)),
#     ((131.4, 125.2), (131.4, 164.0)),
# ]

# A pattern counts as resolved when its contrast is at least this percent.
RESOLUTION_THRESHOLD = 26


def main():
    # 1. Load the supplied image.
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

    # 2. Make the results folder.
    results_dir = None
    selection_path = None
    if SAVE_RESULTS:
        results_dir = results_directory(IMAGE_PATH, __file__)
        results_dir.mkdir(parents=True, exist_ok=True)
        selection_path = results_dir / "selection.yaml"
        print(f"Results folder: {results_dir}")

    # 3. Straighten the chart for the clicking window.
    #    gui_image is what you click on. image stays the original photo.
    crop = None
    if AUTO_CROP:
        crop = RrtROI().crop_resolution_target(image)
    if AUTO_CROP and not crop:
        raise RuntimeError(
            "Could not find the resolution chart. Set AUTO_CROP = False."
        )

    gui_image = image
    if crop:
        gui_image = crop.image

    # 4. Pick the lines, either by clicking or from the coordinates above.
    selection = select_rrt_lines(
        gui_image,
        crop=crop,
        mode=MODE,
        line_count=LINE_COUNT,
        registration_points=REGISTRATION_POINTS,
        line_coordinates=LINE_COORDINATES,
        selection_path=selection_path,
    )

    # Nothing to measure if the window was closed, or if this is a Jupyter
    # session that is still waiting for Confirm.
    if not isinstance(selection, RrtLineSelection):
        return

    # 5. Move lines drawn on the crop back onto the original photo.
    if crop:
        selection = crop.selection_to_source(selection)
        print("\nLines on the original image:")
        print(selection.describe())

    # 6. Measure contrast and show the plots.
    group_coordinates = selection.group_coordinates
    contrast = RrtAnalyzer().load_and_process_groups(image, group_coordinates)
    print("\nPercentage contrast:")
    print(contrast.drop(columns=["Peak Indices", "Trough Indices"]).to_string())
    RrtROI.print_smallest_resolved(contrast, RESOLUTION_THRESHOLD)

    # ion() keeps each plot open, so it can be saved and stays visible at the end.
    plt.ion()
    plotter = RrtDataPlotter()
    line_profile_path = (
        None if results_dir is None else results_dir / "line_profiles.png"
    )
    contrast_plot_path = (
        None if results_dir is None else results_dir / "percentage_contrast.png"
    )

    plotter.plot_line_profiles(
        image, group_coordinates, contrast, save_path=line_profile_path
    )
    plotter.plot_percentage_contrast(
        contrast,
        resolution_threshold=RESOLUTION_THRESHOLD,
        save_path=contrast_plot_path,
    )
    if results_dir:
        contrast.to_csv(results_dir / "percentage_contrast.csv", index=False)
        plotter.save_roi_image(image, group_coordinates, results_dir / "roi.png")
        if SAVE_SOURCE_IMAGES:
            save_sample_sources(results_dir, IMAGE_PATH)
        print(f"Saved results to {results_dir}")

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
