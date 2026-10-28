"""
This example shows the basic process for analyzing images of QUEL Imaging's reference uniformity and distortion (RUD)
target. The input images are from an imaging system that is fairly uniform in terms of fluorescence collection across
the field of view. As such, the default parameters for processing should work well for these images.
"""

from qal.data import rud_example_1
from qal import RudDetector, UniformityAnalyzer, UniformityVisualizer


def main():

    # Location of the input image(s). All images in the folder should have the same dimensions.
    image_dir = rud_example_1()
    print(f"Images for this example have been downloaded and are located in {image_dir}.")

    # First, detect the RUD target wells in all images in IMAGE_DIR.
    detector = RudDetector()
    detector.detect_dots_uniformity(image_dir)
    rud_dots = detector.output

    # Next, generate the fluorescence uniformity map from the extracted data. In this example, the outputs will not be
    # saved. Change this if obtaining the output files (pickle file and PNGs) is desired - the outputs will be saved to
    # a subdirectory named "Surface Representation" within the input image directory.
    analyzer = UniformityAnalyzer()
    analyzer.update_params({"Save output": False})
    analyzer.generate_surf_rep(rud_dots)
    analyzer_output = analyzer.output

    # Finally, generate figures from the fit. Default operation is to not display the images but save them to file.
    # Since this example is not saving outputs, we will change this to display the images.
    visualizer = UniformityVisualizer()
    visualizer.update_params({"Show figures": True})
    visualizer.visualize_fluorescence_profiles(analyzer_output)


if __name__ == "__main__":
    main()
