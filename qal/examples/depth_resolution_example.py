import numpy as np
from qal.data._fetchers import dr_sample1, dr_sample2
from qal.rta.roi_extraction.crop_depth_phantom import PhantomCropper
from qal.rta.image_analyzer.depth_resolution_analyzer import DepthAnalyzer
from qal.rta.data_visualizer.depth_resolution_plotter import DepthDataPlotter


def main():
    # Load two images of the depth phantom
    im1 = dr_sample1()
    im2 = dr_sample2()

    # Directories to save plots to if desired (change from None)
    save_dir1 = None
    save_dir2 = None

    # FIRST IMAGE
    # ------------------------------------------------------------------------------------------------------------------
    # Crop the image
    cropper = PhantomCropper()
    cropper.crop_image(im1)

    # Analyze CROPPER for relevant information
    analyzer = DepthAnalyzer(cropper)
    analyzer.get_profiles(depths=np.linspace(1, 6, 10))

    # Plot data in ANALYZER
    depth_data_plotter = DepthDataPlotter(analyzer.outputs)
    depth_data_plotter.plot_data(graph_type='All', plot_smoothed=True, save_dir=save_dir1)

    # FOR THE SECOND IMAGE, INTENSITY ALONG THE CHANNEL DROPS BELOW 2% SO AN ADDITIONAL LINE INDICATING THIS IS ADDED TO
    # THE FHWM PLOT
    # ------------------------------------------------------------------------------------------------------------------
    # Crop the image
    cropper = PhantomCropper()
    cropper.crop_image(im2)

    # Analyze CROPPER for relevant information
    analyzer = DepthAnalyzer(cropper)
    analyzer.get_profiles()

    # Plot data in ANALYZER
    depth_data_plotter = DepthDataPlotter(analyzer.outputs)
    depth_data_plotter.plot_data(graph_type='All', plot_smoothed=True, save_dir=save_dir2)


if __name__ == "__main__":
    main()
