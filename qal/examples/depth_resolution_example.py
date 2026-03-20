import platform
import matplotlib

if platform.system() == "Darwin":
    try:
        matplotlib.use("TkAgg", force=True)
    except Exception:
        pass

import numpy as np
from qal.data import dr_sample1, dr_sample2, dr_sample3
from qal import PhantomCropper, DepthAnalyzer, DepthDataPlotter
from skimage import io
import matplotlib.pyplot as plt

def main():

    # EXAMPLE 1
    # ------------------------------------------------------------------------------------------------------------------
    # Load the example image of the depth resolution phantom
    im1 = dr_sample1()

    # Directories to save plots to if desired (change from None)
    save_dir1 = None

    # Crop the image
    cropper = PhantomCropper()
    cropper.crop_image(im1)

    # Analyze CROPPER for relevant information
    analyzer = DepthAnalyzer(cropper)
    analyzer.get_profiles(depths=np.linspace(1, 6, 10))

    # Plot data in ANALYZER
    depth_data_plotter = DepthDataPlotter(analyzer.outputs)
    depth_data_plotter.plot_data(graph_type='All', plot_smoothed=True, save_dir=save_dir1)


    # EXAMPLE 2
    # ------------------------------------------------------------------------------------------------------------------
    # FOR THE SECOND IMAGE, INTENSITY ALONG THE CHANNEL DROPS BELOW 2% SO AN ADDITIONAL LINE INDICATING THIS IS ADDED TO
    # THE FHWM PLOT

    # Load the example image of the depth resolution phantom
    im2 = dr_sample2()

    # Directory to save plots to if desired (change from None)
    save_dir2 = None

    # Crop the image
    cropper = PhantomCropper()
    cropper.crop_image(im2)

    # Analyze CROPPER for relevant information
    analyzer = DepthAnalyzer(cropper)
    analyzer.get_profiles()

    # Plot data in ANALYZER
    depth_data_plotter = DepthDataPlotter(analyzer.outputs)
    depth_data_plotter.plot_data(graph_type='All', plot_smoothed=True, save_dir=save_dir2)

    # EXAMPLE 3: CUSTOM DIMENSIONS AND PLOTTING
    # ------------------------------------------------------------------------------------------------------------------
    # Load example image 3 or custom image
    # im3 = io.imread('***replace-with-path-to-your-image***')
    im3 = dr_sample3()

    # Directories to save plots to if desired (change from None)
    save_dir3 = None

    # Rotate the phantom image 90 degrees if needed to ensure channels are horizontal
    rotate_image_90 = False

    # Crop the image
    cropper = PhantomCropper()
    cropper.crop_image(im3)

    # Plot the cropped image with inferno colormap and unit dimensions
    img = cropper.img
    top = cropper.borders["top"]
    bottom = cropper.borders["bottom"]
    left = cropper.borders["left"]
    right = cropper.borders["right"]

    cropped = img[int(top):int(bottom), int(left):int(right)]
    if rotate_image_90 == True:
            cropped = cropped.T

    fig, ax = plt.subplots()
    ax.imshow(cropped, extent=[0, 50, 0 , 35], cmap='inferno') # change extent to x and y dimensions (mm)
    ax.set_xlabel('X-axis (mm)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Y-axis (mm)', fontsize=16, fontweight='bold')
    ax.set_title('Cropped Fluorescence Image', fontsize=16)
    plt.show(block=False)
    plt.pause(0.01)


    # Set dimensions and analyze CROPPER for relevant information
    analyzer = DepthAnalyzer(cropper)
    analyzer.depth_start_end = [1.3, 7.3] # z-depths (mm)
    analyzer.descent_start_end = [6, 44] # x-positions (mm)
    analyzer.phantom_dimensions = [35, 50] # y, x dimensions
    analyzer.get_profiles(rotate_image_90, channel_distance_from_top = 0.5, depths=np.linspace(1.3, 7.3, 10)) # channel_distance_from_top = percent (as fraction) from the top where the channel is (0.5 for 50%), depths = same as depth_start_end, 10 points in between

    # Plot data in ANALYZER
    depth_data_plotter = DepthDataPlotter(analyzer.outputs)
    depth_data_plotter.plot_data(graph_type='All', plot_smoothed=True, save_dir=save_dir3)

if __name__ == "__main__":
    main()
