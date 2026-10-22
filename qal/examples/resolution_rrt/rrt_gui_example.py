"""Place USAF 1951 line profiles with the Matplotlib GUI and measure contrast.

Edit the settings below, then run:

    python -m qal.examples.resolution_rrt.rrt_gui_example

The window prints the coordinates you chose. With SAVE_RESULTS on, it also
writes plots, a CSV, and selection.json next to your image. Paste those
coordinates into rrt_batch_example to measure more images without the window.
"""

from pathlib import Path

import matplotlib.pyplot as plt

from qal import RrtAnalyzer, RrtDataPlotter, RrtLineSelection, RrtROI, select_rrt_lines
from qal.data import res_sample_1
from qal.util import load_grayscale_image


# --- Settings ---------------------------------------------------------------

# Leave this as None to use QAL's built-in example image.
# Or set your own file, for example "path/to/image.tiff".
IMAGE_PATH = None

# True saves plots, a CSV, and selection.json in a folder next to the image.
# This needs IMAGE_PATH, because the built-in example has no folder of its own.
SAVE_RESULTS = False

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
    # Saving needs a real file path, so the results can sit next to the image.
    if SAVE_RESULTS and not IMAGE_PATH:
        raise ValueError("Set IMAGE_PATH before turning SAVE_RESULTS on.")

    # 1. Load the image. The built-in example is used when IMAGE_PATH is empty.
    image = res_sample_1()
    image_path = None
    if IMAGE_PATH:
        image_path = Path(IMAGE_PATH).expanduser().resolve()
        image = load_grayscale_image(image_path)

    # 2. Make the results folder.
    results_dir = None
    selection_path = None
    if SAVE_RESULTS:
        results_dir = image_path.parent / f"{image_path.stem}_results"
        results_dir.mkdir(parents=True, exist_ok=True)
        selection_path = results_dir / "selection.json"
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

    plotter.plot_line_profiles(image, group_coordinates, contrast)
    if results_dir:
        plt.savefig(results_dir / "line_profiles.png", bbox_inches="tight")

    plotter.plot_percentage_contrast(contrast, resolution_threshold=RESOLUTION_THRESHOLD)
    if results_dir:
        plt.savefig(results_dir / "percentage_contrast.png", bbox_inches="tight")
        contrast.to_csv(results_dir / "percentage_contrast.csv", index=False)
        print(f"Saved results to {results_dir}")

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
