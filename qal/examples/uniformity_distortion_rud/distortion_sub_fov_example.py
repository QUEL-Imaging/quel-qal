"""
This script shows an example of distortion analysis using only one image that does not span the field of view of the
imaging system. Results will only be provided for the portion of the field of view in which wells were detected.
"""

from qal.data import rud_example_3
from qal import RudDetector, DistortionAnalyzer, DistortionVisualizer


def main():

    # Location of the input image.
    image_dir = rud_example_3()
    print(f"Images for this example have been downloaded and are located in {image_dir}.")

    # First, detect the wells. Due to intensity fall-off in this image, two thresholding passes were needed to detect
    # all wells. Minimum area for detected wells, as well as the kernel size for filtering, also needed to be changed.
    detector = RudDetector()
    detector.detect_dots_distortion(image_dir)

    # Next, analyze the data for distortion. Since this input image only covers a portion of the field of view, the
    # analyzer will be instructed to not extrapolate outside the data range when producing the distortion map.
    analyzer = DistortionAnalyzer()
    analyzer.compute_distortion(detector.output, ignore_extra=True)

    # Finally, visualize the results. Default operation is to save the figures generated. Here, they will only be
    # displayed.
    visualizer = DistortionVisualizer()
    visualizer.visualize_distortion(analyzer.output, save=False)


if __name__ == "__main__":
    main()
