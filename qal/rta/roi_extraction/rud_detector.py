import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io
from scipy.ndimage import label
from skimage.measure import regionprops
from matplotlib.patches import Circle, Rectangle
from numba import njit


class RudDetector:

    # Some parameters for optional figures produced
    FIG_SIZE = (8, 6)
    IM_COLORMAP = 'inferno'
    THRESH_COLORMAP = 'gray'
    BBOX_COLOR = 'red'
    DOT_COLOR = 'green'
    DOT_ALPHA = 0.5

    def __init__(self, params=None):
        """
        Initialize RudDetector with optional parameters for class methods.

        :param params:      Parameters for class methods (optional)
        """

        self.environment = self.check_environment()

        # Default parameters for class methods
        if params is None:
            self.params = {
                "Threshold multipliers": [1],
                "Number of thresholding passes": 1,
                "Kernel size (Opening)": 5,
                "Minimum dot area": 30,
                "Maximum dot area": None,
                "Maximum eccentricity": 0.7,
                "ROI deletion extra fraction": 0.6,
                "Minimum center dots": 9,
                "Crop images": False,
                "Show images": False
            }
        else:
            self.params = params

        # Initialize class attributes from PARAMS
        self.th_mult = self.params["Threshold multipliers"]
        self.n_pass = self.params["Number of thresholding passes"]
        self.k_opening = self.params["Kernel size (Opening)"]
        self.min_area = self.params["Minimum dot area"]
        self.max_area = self.params["Maximum dot area"]
        self.max_eccentricity = self.params["Maximum eccentricity"]
        self.bbox_extra_fraction = self.params["ROI deletion extra fraction"]
        self.min_dots_center = self.params["Minimum center dots"]
        self.crop_im = self.params["Crop images"]
        self.show_images = self.params["Show images"]

        # Make sure length of threshold multipliers is compatible with number of passes
        if len(self.th_mult) > 1:
            assert len(self.th_mult) == self.n_pass, "Length of TH_MULT should be same as N_PASS if greater than 1"

        # Initialize other attributes needed for uniformity analysis
        self.dots = None
        self.images = None
        self.im_names = None
        self.cropped_images = None
        self.fov = None
        self.output = None

        # Initialize other attributes needed for distortion analysis
        self.image_center = None

    def detect_dots_uniformity(self, image_dir):
        """
        Find wells of the reference uniformity and distortion (RUD) target from images in the folder IMAGE_DIR. The
        images in the folder should all have the same shape.

        :param image_dir:       Directory containing image(s) of the RUD target
        """

        # Load images in image directory
        self.images, self.im_names = self.load_images(image_dir, return_filenames=True)

        # Maybe crop out wells in the RUD target before analysis
        if self.crop_im:
            self.cropped_images = self.crop_images()
        else:
            self.cropped_images = self.images

        # Find wells in each image of the RUD target and compile locations and intensities
        print("\nGENERATING SURFACE REPRESENTATION")
        dots = []
        for itr, (im, im_name) in enumerate(zip(self.cropped_images, self.im_names)):
            print(f"Extracting data from image {itr + 1} of {len(self.cropped_images)}...")
            im_dots = self.find_dots(im, im_name, show_images=self.show_images)
            if im_dots is None:
                print(f"\nNo valid region proposals found. Check image or parameters, or consider reducing number of "
                      f"thresholding passes.")
                return
            elif len(im_dots) < 500:
                print("  WARNING: Less than 500 dots found. May affect results")
            dots.extend(im_dots)
        print("Data extraction complete")

        self.dots = pd.DataFrame(dots)

        # Update output
        self.output = {
            "dots_df": self.dots,
            "fov": self.fov,
            "image_dir": image_dir
        }

    def detect_dots_distortion(self, image_dir):
        """
        Find wells of the distortion target from images in the folder IMAGE_DIR. The images in the folder should all
        have the same shape.

        :param image_dir:       Directory containing image(s) of the distortion target
        """

        # Load images in image directory
        self.images, self.im_names = self.load_images(image_dir, return_filenames=True)

        # Maybe crop out wells in the RUD target before analysis
        if self.crop_im:
            self.cropped_images = self.crop_images()
        else:
            self.cropped_images = self.images

        # Find wells in each image of the RUD target and compile locations
        print("\nFINDING WELLS")
        dots_dfs = []
        for itr, (im, im_name) in enumerate(zip(self.cropped_images, self.im_names)):
            print(f"Extracting data from image {itr + 1} of {len(self.cropped_images)}...")
            im_dots = self.find_dots(im, im_name, show_images=self.show_images)
            if im_dots is None:
                print(f"\nNo valid region proposals found. Check image or parameters, or consider reducing number of "
                      f"thresholding passes.")
                return
            elif len(im_dots) < 500:
                print("  WARNING: Less than 500 dots found. May affect results")
            dots = pd.DataFrame(im_dots)
            dots_dfs.append(dots)
        print("Data extraction complete")

        self.dots = dots_dfs

        # Remove data from images without sufficient dots around the image center
        self.remove_insufficient()
        if len(self.dots) == 0:
            print(f"\nInput images were not suitable for further analysis")
            return

        # Update output
        self.output = {
            "dots": self.dots,
            "fov": self.fov,
            "image center": self.image_center,
            "image_dir": image_dir
        }

    def load_images(self, im_dir, return_filenames=True):
        """
        Read in images from im_dir and return as a list

        :param im_dir:              Path to input images
        :param return_filenames:    Whether to return the names of files loaded
        :return:                    List containing images; optionally, names of files loaded
        """

        im_array = []
        im_names = []

        # Look through contents of directory, attempt to read image files and add them to list
        for content in os.listdir(im_dir):
            f = os.path.join(im_dir, content)
            if os.path.isfile(f):
                try:
                    io.imread(f)
                except OSError:
                    print(f"Cannot read {content} as image. Skipping.")
                else:
                    im = io.imread(f)
                    if len(im) == 0:
                        print(f"Skipping {content}.")
                    else:
                        im_array.append(im)
                        im_names.append(os.path.splitext(content)[0])

        # Images should be the same shape
        im_shape = im_array[0].shape
        for im in im_array:
            assert im.shape == im_shape, "Images should be the same shape."
        self.fov = im_shape

        # Check image data type and maybe convert to 16-bit
        if im_array[0].dtype != 'uint16':
            im_array = self.cvt_to_uint16(im_array)

        if return_filenames:
            return im_array, im_names
        else:
            return im_array

    def crop_images(self):
        """
        Prompt user to select region of each image from which to locate wells. Zero all pixels outside this region.

        :return:        Images same size as input, but with zeros outside the "cropped" regions
        """

        if self.environment == "Standard Python":
            from roipoly import RoiPoly
            cropped_images = []
            for im in self.images:
                fig, ax = plt.subplots(figsize=self.FIG_SIZE)
                ax.imshow(im, cmap=self.IM_COLORMAP)
                ax.set_axis_off()
                ax.set_title('Crop desired region. Right-click when done')
                roi = RoiPoly(color='r')
                mask = roi.get_mask(im)
                cropped_images.append((mask * im).astype(np.uint16))
            return cropped_images
        else:
            print(f"Image cropping currently not supported for {self.environment}")
            return self.images

    def find_dots(self, im, im_name, show_images=False):
        """
        Take image of RUD target as input and compute locations and mean intensities of dots

        :param im:                  Image of the RUD target
        :param im_name:             Name of the image file
        :param show_images:         Whether to display intermediate images leading to final results
        :return:                    List of dictionaries representing x and y positions, and intensities of dots
        """

        # Get parameters
        th_mult = self.th_mult              # Threshold multipliers
        n_pass = self.n_pass                # Number of passes for thresholding
        k = self.k_opening                  # Kernel size for Opening transformation (erosion followed by dilation)
        min_area = self.min_area            # Minimum area for region proposals
        max_area = self.max_area            # If not None, region proposals with area larger than this are excluded
        max_ecc = self.max_eccentricity     # Maximum eccentricity for region proposals

        im_original = im.copy()
        thresh_ims = []
        cleaned_thresh_ims = []
        regions = []
        av_radius = None
        dots = []

        for itr in range(n_pass):

            print(f"  Finding wells, pass {itr + 1} of {n_pass}...")

            # Threshold image and multiply by TH_MULT
            if len(th_mult) == 1:
                mult = th_mult[0]
            else:
                mult = th_mult[itr]
            thresh, _ = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            if thresh == 0:
                thresh = np.mean(im)
                print(f"    Otsu threshold failed. Using image mean")
            thresh_im = (im > mult * thresh).astype(np.uint8)
            thresh_ims.append(thresh_im)

            # Remove noise by performing Opening operation
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            cleaned_thresh_im = cv2.morphologyEx(thresh_im, cv2.MORPH_OPEN, kernel)
            cleaned_thresh_ims.append(cleaned_thresh_im)

            # Obtain region proposals for dots
            labeled_im, _ = label(cleaned_thresh_im)
            props = regionprops(labeled_im, intensity_image=cleaned_thresh_im)

            # Filter out regions based on size and eccentricity
            valid_props = self.find_valid_props(props, min_area, max_ecc, max_area)
            regions.append(valid_props)

            # inform user if no valid region proposals found
            if not valid_props:

                print(f"    No valid region proposals found for {im_name} on thresholding pass {itr + 1} of {n_pass}.")

            else:

                # Find average radius of proposed regions on the first pass
                if itr == 0:
                    av_radius = self.find_average_radius(valid_props, scale_by=0.5)

                # Compile dot centroid locations and mean intensities
                itr_dots = self.get_dots_info(im_original, valid_props, av_radius)
                dots.extend(itr_dots)

                # Zero out regions found in this pass
                im = self.zero_regions(im, valid_props, extra_fraction=self.bbox_extra_fraction)

        # Optionally display intermediate images and final identified dots
        if show_images:
            self.display_images(im_original, thresh_ims, cleaned_thresh_ims, regions, dots, av_radius)

        return dots

    def get_dots_info(self, im, props, radius):
        """
        Compile information on centroid locations and mean intensities of the input region proposals.

        :param im:              Image of the RUD target
        :param props:           Region proposals
        :param radius:          Radius from centroids within which to calculate mean intensities
        :return:                List of dictionaries with centroid locations and mean intensities
        """
        dots = []
        for p in props:
            y_c, x_c = p.centroid
            mean_intensity = self.compute_mean_intensity(im, y_c, x_c, radius)
            dots.append({
                "x": x_c,
                "y": y_c,
                "dot_intensity": mean_intensity
            })
        return dots

    def zero_regions(self, im, props, extra_fraction=0.1):
        """
        Zero out the regions in the input image defined by PROPS.

        :param im:                  Input image
        :param props:               List of regions to zero out
        :param extra_fraction:      Extra fraction of the region bounding box to zero
        :return:                    Output image
        """

        y_ext, x_ext = self.fov

        for p in props:
            height = p.bbox[2] - p.bbox[0]
            width = p.bbox[3] - p.bbox[1]
            top = int(p.bbox[0] - (extra_fraction * height))
            bottom = int(p.bbox[2] + (extra_fraction * height))
            left = int(p.bbox[1] - (extra_fraction * width))
            right = int(p.bbox[3] + (extra_fraction * width))

            if top < 0:
                top = 0
            if bottom > y_ext:
                bottom = y_ext
            if left < 0:
                left = 0
            if right > x_ext:
                right = x_ext

            im[top:bottom, left:right] = 0

        return im

    def display_images(self, im_original, thresh_ims, cleaned_thresh_ims, regions, dots, av_radius):
        """
        Display intermediate images from dot finding process and final image of identified dots.

        :param im_original:             Original image
        :param thresh_ims:              Thresholded image(s)
        :param cleaned_thresh_ims:      Cleaned thresholded image(s)
        :param regions:                 List of region proposals
        :param dots:                    List of identified dots
        :param av_radius:               Radius of dots to overlay on original image
        """

        plt.rcParams.update({'font.size': 12})

        for itr, (thresh_im, cleaned_thresh_im, props) in enumerate(zip(thresh_ims, cleaned_thresh_ims, regions)):

            fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(14, 6))
            plt.subplots_adjust(left=0.05, bottom=0.01, right=0.95, top=0.92, wspace=0.05, hspace=0.1)

            # Thresholded images
            ax1.imshow(thresh_im, cmap=self.THRESH_COLORMAP)
            ax1.set_title(f"Iteration {itr + 1} thresholded image")
            ax1.set_axis_off()

            # Cleaned thresholded images
            ax2.imshow(cleaned_thresh_im, cmap=self.THRESH_COLORMAP)
            ax2.set_title(f"Iteration {itr + 1} cleaned thresholded image")
            ax2.set_axis_off()

            # Region proposals
            ax3.imshow(im_original, cmap=self.IM_COLORMAP)
            for p in props:
                height = p.bbox[2] - p.bbox[0]
                width = p.bbox[3] - p.bbox[1]
                ax3.add_patch(Rectangle((p.bbox[1], p.bbox[0]), width, height, color=self.BBOX_COLOR, fill=False))
            ax3.set_title(f"Iteration {itr + 1} valid region proposals")
            ax3.set_axis_off()

            plt.draw()

        # Identified dots
        fig, ax = plt.subplots(figsize=self.FIG_SIZE)
        ax.imshow(im_original, cmap=self.IM_COLORMAP)
        for dot in dots:
            x, y = dot["x"], dot["y"]
            ax.add_patch(Circle((x, y), radius=av_radius, color=self.DOT_COLOR, alpha=self.DOT_ALPHA))
        ax.set_title("Identified dots")
        ax.set_axis_off()
        plt.show()

    def remove_insufficient(self):
        """
        Remove data from images without a sufficient number of dots (specified by SELF.MIN_DOTS_CENTER) around the image
        center. Update SELF.DOTS.
        """

        center = np.array(self.images[0].shape) / 2
        self.image_center = center
        dots_to_keep = []

        for dots_df, im_name in zip(self.dots, self.im_names):

            # Find the closest dot to image center
            x_locs = dots_df["x"].to_numpy()
            y_locs = dots_df["y"].to_numpy()
            distances_to_center = self.dist_between(center[1], center[0], x_locs, y_locs)
            closest_idx = np.argmin(distances_to_center)
            closest_dot = (x_locs[closest_idx], y_locs[closest_idx])

            # Find the closest two points to that point and measure the average distance
            distances_to_closest = self.dist_between(closest_dot[0], closest_dot[1], x_locs, y_locs)
            closest_two_dists = np.sort(distances_to_closest)[1:2]
            av_dist = np.mean(closest_two_dists)

            # There should be at least SELF.MIN_DOTS_CENTER dots within a radius of 2 x AV_DIST of the center
            num_dots_center = np.sum(distances_to_center <= 2 * av_dist)
            if num_dots_center < self.min_dots_center:
                print(f"\nRemoving {im_name} due to insufficient number of dots around image center")
            else:
                dots_to_keep.append(dots_df)

        # Update SELF.DOTS
        self.dots = dots_to_keep

    def update_params(self, new_params):
        """
        Update some or all parameters.

        :param new_params:      Dictionary with parameter(s) to update
        """

        # Check that the input parameter(s) has/have the correct name(s), then update the parameter list
        for new_param in new_params:
            assert new_param in self.params.keys(), f"'{new_param}' is not a recognized parameter"
            self.params[new_param] = new_params[new_param]

        # Update class attributes
        self.th_mult = self.params["Threshold multipliers"]
        self.n_pass = self.params["Number of thresholding passes"]
        self.k_opening = self.params["Kernel size (Opening)"]
        self.min_area = self.params["Minimum dot area"]
        self.max_area = self.params["Maximum dot area"]
        self.max_eccentricity = self.params["Maximum eccentricity"]
        self.bbox_extra_fraction = self.params["ROI deletion extra fraction"]
        self.min_dots_center = self.params["Minimum center dots"]
        self.crop_im = self.params["Crop images"]
        self.show_images = self.params["Show images"]

        # Make sure length of threshold multipliers is compatible with number of passes
        if len(self.th_mult) > 1:
            assert len(self.th_mult) == self.n_pass, "Length of TH_MULT should be same as N_PASS if greater than 1"

    @staticmethod
    def check_environment():
        """
        Return the Python environment type.
        """
        try:
            from IPython import get_ipython
            get_ipython()
            if 'IPKernelApp' in get_ipython().config:
                get_ipython().magic('matplotlib widget')
                get_ipython().run_line_magic('matplotlib', 'ipympl')
                return "Jupyter Notebook"
            else:
                get_ipython().run_line_magic('matplotlib', 'ipympl')
                return "JupyterLab"
        except AttributeError:
            return "Standard Python"

    @staticmethod
    def cvt_to_uint16(im_array):
        """
        Convert the input image(s) to 16-bit.

        :param im_array:        List of input image(s)
        :return:                Image(s) converted to 16-bit
        """

        im_array_uint16 = []
        im_type = im_array[0].dtype

        for im in im_array:
            if im_type == 'uint8':
                im16 = ((im.astype(np.float64) / 255) * 65535).astype(np.uint16)
                im_array_uint16.append(im16)
            else:
                im[im < 0] = 0
                im16 = ((im.astype(np.float64) / np.max(im.astype(np.float64))) * 65535).astype(np.uint16)
                im_array_uint16.append(im16)

        return im_array_uint16

    @staticmethod
    def find_valid_props(props, min_area, max_eccentricity, max_area):
        """
        Filter out region proposals based on input parameters.

        :param props:               Original region proposals
        :param min_area:            Minimum area for valid region proposals
        :param max_eccentricity:    Maximum eccentricity for valid region proposals
        :param max_area:            Maximum area for valid region proposals
        :return:                    Valid region proposals
        """

        valid_props = [p for p in props if p.area >= min_area]
        valid_props = [p for p in valid_props if p.eccentricity <= max_eccentricity]
        if max_area is not None:
            valid_props = [p for p in valid_props if p.area < max_area]

        return valid_props

    @staticmethod
    def find_average_radius(props, scale_by=0.5):
        """
        Find the average radius of blobs in the input list of region proposals.

        :param props:       List of region proposals
        :param scale_by:    Scale the average radius by this value
        :return:            Average radius of regions
        """

        radii = []
        for p in props:
            bbox = p.bbox
            height = bbox[2] - bbox[0]
            width = bbox[3] - bbox[1]
            radii.append(np.mean([height, width]) / 2)
        av_radius = np.mean(np.array(radii))
        use_radius = av_radius * scale_by

        return use_radius

    @staticmethod
    @njit
    def compute_mean_intensity(im, y_c, x_c, radius):
        sum_intensity = 0.0
        count = 0
        y_min = max(0, int(y_c - radius))
        y_max = min(im.shape[0], int(y_c + radius) + 1)
        x_min = max(0, int(x_c - radius))
        x_max = min(im.shape[1], int(x_c + radius) + 1)
        for y in range(y_min, y_max):
            for x in range(x_min, x_max):
                dy = y - y_c
                dx = x - x_c
                if dx * dx + dy * dy <= radius * radius:
                    sum_intensity += im[y, x]
                    count += 1
        return sum_intensity / count if count > 0 else 0

    @staticmethod
    def dist_between(x, y, xx, yy):
        """
        Calculate the Euclidean distance between the point (x, y) and the point(s) (xx, yy).

        :param x:           x-location of point. Must be single value
        :param y:           y-location of point. Must be single value
        :param xx:          x-location(s) of point(s) to compare. Can be an array
        :param yy:          x-location(s) of point(s) to compare. Can be an array
        :return:            Distance(s) between the points
        """

        dist = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
        return dist
