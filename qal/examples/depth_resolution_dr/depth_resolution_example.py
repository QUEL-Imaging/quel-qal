"""Crop and analyze one depth-resolution sample.

Choose one sample below; each run asks for four phantom edges. Sample 2
shows the additional 2% intensity line. Sample 3 demonstrates custom
physical dimensions. See doc/depth_resolution_examples.md for details.
"""

import numpy as np

from qal import PhantomCropper, DepthAnalyzer, DepthDataPlotter
from qal.data import dr_sample1, dr_sample2, dr_sample3
from qal.examples._output import results_directory, save_sample_sources
from qal.rta.roi_extraction.well_detector_gui import use_native_window_backend
from qal.util import load_grayscale_image

# Choose one sample; leave the alternatives commented out.
IMAGE_PATH = dr_sample1()
# IMAGE_PATH = dr_sample2()
# IMAGE_PATH = dr_sample3()
# IMAGE_PATH = "path/to/your/image.tiff"

SAVE_RESULTS = True
SAVE_SOURCE_IMAGES = False

# Sample 3 uses different physical dimensions; enable these with dr_sample3().
CUSTOM_DIMENSIONS = False
ROTATE_IMAGE_90 = False


def main():
    if IMAGE_PATH is None:
        raise ValueError("Set IMAGE_PATH to an image file or built-in sample.")
    image = load_grayscale_image(IMAGE_PATH)
    # Reuse the platform-aware backend selection rather than forcing Tk on macOS.
    use_native_window_backend()
    cropper = PhantomCropper()
    cropper.crop_image(image)
    analyzer = DepthAnalyzer(cropper)
    if CUSTOM_DIMENSIONS:
        analyzer.depth_start_end = [1.3, 7.3]
        analyzer.descent_start_end = [6, 44]
        analyzer.phantom_dimensions = [35, 50]
        analyzer.get_profiles(
            depths=np.linspace(1.3, 7.3, 10),
            rotate_image_90=ROTATE_IMAGE_90,
            channel_distance_from_top=0.5,
        )
    else:
        analyzer.get_profiles(depths=np.linspace(1, 6, 10))
    results_dir = None
    if SAVE_RESULTS:
        results_dir = results_directory(IMAGE_PATH, __file__)
        results_dir.mkdir(parents=True, exist_ok=True)
        if SAVE_SOURCE_IMAGES:
            save_sample_sources(results_dir, IMAGE_PATH)
    DepthDataPlotter(analyzer.outputs).plot_data(
        graph_type="All", plot_smoothed=True, save_dir=results_dir
    )


if __name__ == "__main__":
    main()
