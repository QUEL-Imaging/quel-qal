"""
This example analyzes images of QUEL Imaging's reference uniformity and distortion (RUD) target to produce a
fluorescence uniformity map. The images were taken on an imaging system with a very non-uniform fluorescence collection
profile. As a result, parameters need to be adjusted in order to obtain the fluorescence profile of the imaging system.
"""

from qal.data import rud_example_2
from qal import RudDetector, UniformityAnalyzer, UniformityVisualizer


def main():

    # Location of the input image(s). All images in the folder should have the same dimensions.
    image_dir = rud_example_2()
    print(f"Images for this example have been downloaded and are located in {image_dir}.")

    # First, detect the RUD target wells in all images in IMAGE_DIR. Through trial and error, it was determined that
    # four thresholding passes are needed to identify enough wells to cover about half the field of view. It is
    # difficult to identify more wells because this imaging system also has a lot of spherical aberrations, hence wells
    # are not well-defined further out from the image center. To visualize intermediate steps in the well-finding
    # process, add -"Show images": True- to the update_params call.
    detector = RudDetector()
    detector.update_params({
        "Number of thresholding passes": 4,
        "Threshold multipliers": [1, 3.5, 3, 2.8],
        "Maximum eccentricity": 1,
        "ROI deletion extra fraction": 1
    })
    detector.detect_dots_uniformity(image_dir)
    rud_dots = detector.output

    # Next, generate the fluorescence uniformity map from the extracted data. Because the data in this case is confined
    # to the central portion of the field of view (due to undetectable wells further out from the center), we will
    # update the default parameters to tell the program to ignore regions outside the data range when fitting.
    # Additionally, in this example, the outputs will not be saved. Change this if obtaining the output files (pickle
    # file and PNGs) is desired - the outputs will be saved to a subdirectory named "Surface Representation" within the
    # input image directory.
    analyzer = UniformityAnalyzer()
    analyzer.update_params({
        "Zero outside data range": True,
        "Save output": False
    })
    analyzer.generate_surf_rep(rud_dots)
    analyzer_output = analyzer.output

    # Finally, generate figures from the fit. Default operation is to not display the images but save them to file.
    # Since this example is not saving outputs, we will change this to display the images.
    visualizer = UniformityVisualizer()
    visualizer.update_params({"Show figures": True})
    visualizer.visualize_fluorescence_profiles(analyzer_output)


if __name__ == "__main__":
    main()
