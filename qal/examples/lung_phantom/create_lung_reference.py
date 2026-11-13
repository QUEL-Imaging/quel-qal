"""Create the lung reference image, mask, and inclusion metadata.

Run ``python -m qal.examples.lung_phantom.create_lung_reference``.
Results save beside this script for built-in samples, or beside IMAGE_PATH
for your own files. Source-image copying is optional and off by default.
"""

from qal.data import lung_reference
from qal import LungPhantom
from qal.util import results_directory, save_sample_sources
from qal.util import load_grayscale_image


# Keep the built-in pair, or replace both inputs with matching file paths.
IMAGE_PATH, INCLUSIONS_IMAGE_PATH = lung_reference()
# IMAGE_PATH = "path/to/phantom_body.tiff"
# INCLUSIONS_IMAGE_PATH = "path/to/phantom_inclusions.tiff"

SAVE_RESULTS = True
# Copy only original built-in samples; user-supplied files are never copied.
SAVE_SOURCE_IMAGES = False


def main():
    if IMAGE_PATH is None or INCLUSIONS_IMAGE_PATH is None:
        raise ValueError("Set IMAGE_PATH and INCLUSIONS_IMAGE_PATH to a matching image pair.")
    im_phantom = load_grayscale_image(IMAGE_PATH)
    im_inclusions = load_grayscale_image(INCLUSIONS_IMAGE_PATH)
    lung_phantom = LungPhantom()
    results_dir = None
    if SAVE_RESULTS:
        results_dir = results_directory(IMAGE_PATH, __file__)
        results_dir.mkdir(parents=True, exist_ok=True)
        # Keep generated references in the output folder, outside the sample cache.
        lung_phantom.ref_dir = str(results_dir)
        print(f"Results folder: {results_dir}")
    lung_phantom.create_reference_mask(im_phantom, im_inclusions, save=SAVE_RESULTS)
    if results_dir is not None and SAVE_SOURCE_IMAGES:
        save_sample_sources(results_dir, IMAGE_PATH, INCLUSIONS_IMAGE_PATH)
    return lung_phantom


if __name__ == "__main__":
    main()
