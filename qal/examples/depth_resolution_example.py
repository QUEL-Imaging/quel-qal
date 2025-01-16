import numpy as np
from qal.data import dr_sample1, dr_sample2
from qal import PhantomCropper, DepthAnalyzer, DepthDataPlotter


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
