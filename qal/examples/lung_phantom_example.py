"""
This script shows example use of the LungPhantom class.
"""

from qal.data import lung_test_image
from qal import LungPhantom


def main():

    # The "data" folder contains a sample image of the lung phantom that has been scaled and rotated. This image will
    # be used in this example. To obtain the Excel files containing the metrics output from the program, change
    # SAVE_DIR to the location of the directory you want to save to.
    im = lung_test_image()
    save_dir = None

    # Because the parameter space for finding the right registration parameters is so broad, it is easy for the
    # optimization search to land in local minima. Hence, the initial guesses are very important. Initial parameters
    # were chosen by visually comparing the reference image and the test image.
    lung_phantom = LungPhantom()
    lung_phantom.update_params({
        "Initial registration parameters": [0.75, 0.5, 25, 5],      # Scale, rotation (in radians), x and y translation
    })
    metrics = lung_phantom.get_inclusion_stats(im, save_dir=save_dir)
    print(f"\nSummary metrics:\n{metrics[0]}")


if __name__ == "__main__":
    main()
