"""Register a lung-phantom image and measure its inclusions.

Run ``python -m qal.examples.lung_phantom.lung_phantom_example``.
Results save beside this script for built-in samples, or beside your image.
"""

from qal.data import lung_test_image
from qal import LungPhantom
from qal.util import results_directory, save_sample_sources
from qal.util import load_grayscale_image


IMAGE_PATH = lung_test_image()
# IMAGE_PATH = "path/to/your/lung_phantom.tiff"

SAVE_RESULTS = True
# Optional copying of the original built-in sample only.
SAVE_SOURCE_IMAGES = False
SHOW_PROGRESS = True


def main():

    if IMAGE_PATH is None:
        raise ValueError("Set IMAGE_PATH to an image file or built-in sample.")
    im = load_grayscale_image(IMAGE_PATH)
    save_dir = None
    if SAVE_RESULTS:
        save_dir = results_directory(IMAGE_PATH, __file__)
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"Results folder: {save_dir}")

    # Because the parameter space for finding the right registration parameters is so broad, it is easy for the
    # optimization search to land in local minima. Hence, the initial guesses are very important. Initial parameters
    # were chosen by visually comparing the reference image and the test image.
    lung_phantom = LungPhantom()
    lung_phantom.update_params({
        "Initial registration parameters": [0.75, 0.5, 25, 5],      # Scale, rotation (in radians), x and y translation
    })
    # Registration minimizes negative mutual information using scipy.optimize.fmin
    # (Nelder-Mead), rather than stopping at a fixed target cost. It stops when
    # both the parameter spread is within xtol=0.005 and the cost spread is within
    # SciPy's default ftol=0.0001, or when the default iteration/evaluation limit
    # is reached (4 parameters * 200 = 800 for each limit). The optimizer prints
    # its termination status. Runtime depends on the image, initial guess, and
    # plotting overhead; there is no fixed duration or function-call count.
    metrics = lung_phantom.get_inclusion_stats(
        im, show_progress=SHOW_PROGRESS, save_dir=save_dir
    )
    if save_dir is not None and SAVE_SOURCE_IMAGES:
        save_sample_sources(save_dir, IMAGE_PATH)
    print(f"\nSummary metrics:\n{metrics[0]}")
    return metrics


if __name__ == "__main__":
    main()
