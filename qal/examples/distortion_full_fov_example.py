"""
This script shows the basic process of analyzing images of QUEL Imaging's reference uniformity and distortion target to
quantify local geometric distortion. The images used in this example span the field of view of the imaging system.
"""

from qal.data import rud_example_1
from qal import RudDetector, DistortionAnalyzer, DistortionVisualizer


def main():

    # Location of the input image(s). All images in the folder should have the same dimensions.
    image_dir = rud_example_1()
    print(f"Images for this example have been downloaded and are located in {image_dir}.")

    # First, detect the target's wells in all images in IMAGE_DIR.
    detector = RudDetector()
    detector.detect_dots_distortion(image_dir)
    detector_output = detector.output

    # Next, analyze the data for distortion.
    analyzer = DistortionAnalyzer()
    analyzer.compute_distortion(detector_output)
    analyzer_output = analyzer.output

    # Finally, visualize the results. Default operation is to save the figures generated. Here, they will only be
    # displayed.
    visualizer = DistortionVisualizer()
    visualizer.visualize_distortion(analyzer_output, save=False)


if __name__ == "__main__":
    main()
