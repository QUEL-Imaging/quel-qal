import os
import cv2
import time
import pickle
import warnings
import matplotlib
import numpy as np
import pandas as pd
import scipy.optimize as optim
import matplotlib.pyplot as plt
import skimage.transform as transform
from IPython.core.pylabtools import figsize
from skimage import io
from scipy.ndimage import label
from skimage.measure import regionprops
from matplotlib.patches import Rectangle
from qal.data import get_cache_dir, lung_reference_image, lung_reference_mask, load_lung_info


class LungPhantom:

    # Some attributes for figures created
    FIG_SIZE = (7, 7)
    IM_CMAP = 'inferno'
    HIST_CMAP = 'gray'
    MASK_CMAP = 'hsv'
    TITLE_FS = 14
    OVERLAY_ALPHA = 0.6

    def __init__(self, params=None):
        """
        Initialize LungPhantom with optional parameters for class methods.

        :param params:      Parameters for class methods (optional)
        """

        # Default parameters
        if params is None:
            self.params = {
                "Image shape for registration": [256, 256],             # Should be square
                "Initial registration parameters": [1, 0, 0, 0],        # scale, rotation, x and y translation
                "Number of histogram bins": 64,
                "fmin delta x parameter": 0.005
            }
        else:
            self.params = params

        # Initialize attributes from parameters
        self.reg_shape = self.params["Image shape for registration"]
        assert self.reg_shape[0] == self.reg_shape[1], "Registration image shape should be square"
        self.init_reg_params = self.params["Initial registration parameters"]
        self.n_bins = self.params["Number of histogram bins"]
        self.x_tol = self.params["fmin delta x parameter"]

        # Load the reference image and mask (if they exist)
        self.ref_dir = "../vpa/reference"
        try:
            self.im_reference = io.imread(os.path.join(self.ref_dir, "Lung_reference_image.tiff"))
            self.mask_reference = io.imread(os.path.join(self.ref_dir, "Lung_reference_mask.png"))
        except OSError:
            self.im_reference = None
            self.mask_reference = None
            if not os.path.exists(self.ref_dir):
                self.ref_dir = os.path.join(get_cache_dir(), "lung_reference")
                os.makedirs(self.ref_dir, exist_ok=True)

        # Remaining needed attribute initializations
        self.environment = self.check_environment()
        self.opt_reg_params = None
        self.show_progress = None
        self.progress_figure = None
        self.ax1, self.ax2, self.ax3, self.ax4 = None, None, None, None
        self.warping_artist, self.hist_artist = None, None
        self.progress_bg = None
        self.verbose = None
        self.print_every = None
        self.n_calls = 0

    def update_params(self, new_params):
        """
        Update parameters specified by NEW_PARAMS.

        :param new_params:      Dictionary with parameter(s) to update
        """

        # Check that the input parameter(s) has/have the correct name(s), then update the parameter list
        for new_param in new_params:
            assert new_param in self.params.keys(), f"'{new_param}' is not a recognized parameter"
            self.params[new_param] = new_params[new_param]

        # Update parameter attributes
        self.reg_shape = self.params["Image shape for registration"]
        assert self.reg_shape[0] == self.reg_shape[1], "Registration image shape should be square"
        self.init_reg_params = self.params["Initial registration parameters"]
        self.n_bins = self.params["Number of histogram bins"]
        self.x_tol = self.params["fmin delta x parameter"]

    def create_reference_mask(self, phantom_image, inclusions_image, k=5, min_area=100, inclusion_rad=0.7,
                              void_frac=1.5, save=True):
        """
        Create a reference mask for the lung phantom from the input images. Optionally, save the images in the
        Reference folder.

        :param phantom_image:           Image of the lung phantom with the top on
        :param inclusions_image:        Image of the lung phantom with the top off, showing the inclusions
        :param k:                       Kernel size for Opening operation to remove edge noise
        :param min_area:                Minimum area for inclusions identified by REGIONPROPS
        :param inclusion_rad:           Fraction of the true radii of inclusions for which to draw the mask
        :param void_frac:               Fraction of the radius of each inclusion to be excluded from background mask
        :param save:                    If True, save the reference image and mask to the Reference folder
        """

        # Pad the images to make square
        phantom_image, inclusions_image = self.make_square([phantom_image, inclusions_image])

        reference_mask = np.zeros(phantom_image.shape, dtype=np.uint8)

        # Convert images to 16-bit for CV2 thresholding
        phantom_image, inclusions_image = self.cvt_to_uint16([phantom_image, inclusions_image])

        # Obtain the outline of the phantom by thresholding PHANTOM_IMAGE, erode and dilate to get rid of edge noise
        print("\nObtaining phantom outline...")
        _, phantom_mask = cv2.threshold(phantom_image, 0, 1, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        phantom_mask = cv2.morphologyEx(phantom_mask, cv2.MORPH_OPEN, kernel, iterations=5)
        reference_mask[phantom_mask.astype(bool)] = 128

        # Obtain the outline of the inclusions by thresholding INCLUSIONS_IMAGE, then create circular masks
        print("Making mask from image of inclusions...")
        _, inclusions_mask = cv2.threshold(inclusions_image, 0, 1, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        labeled_im, _ = label(inclusions_mask)
        inclusions = regionprops(labeled_im, intensity_image=inclusions_mask)
        inclusions = [p for p in inclusions if p.area >= min_area]
        diameters = []
        for inclusion in inclusions:
            bbox = inclusion.bbox
            height = bbox[2] - bbox[0]
            width = bbox[3] - bbox[1]
            rad = np.mean([height, width]) / 2
            diameters.append(rad * 2)
            void_mask = self.circular_mask(inclusions_image.shape, inclusion.centroid, rad * void_frac)
            inclusion_mask = self.circular_mask(inclusions_image.shape, inclusion.centroid, rad * inclusion_rad)
            reference_mask[void_mask] = 0
            reference_mask[inclusion_mask] = 255

        # Compile properties of the identified regions
        labels = np.arange(1, 6)
        assert len(labels) == len(inclusions), "There should be 5 inclusions for the lung phantom. More/less were found"
        sorted_idx = np.argsort(np.array(diameters))
        sorted_diameters = np.array(diameters)[sorted_idx]
        sorted_centroids = [inc.centroid for inc in np.array(inclusions)[sorted_idx]]
        sorted_eccentricity = [inc.eccentricity for inc in np.array(inclusions)[sorted_idx]]
        inclusions_dict = {
            "Label": labels,
            "x coord": [centroid[1] for centroid in sorted_centroids],
            "y coord": [centroid[0] for centroid in sorted_centroids],
            "Eccentricity": sorted_eccentricity,
            "Equivalent diameter (pixels)": sorted_diameters,
            "Actual diameter (mm)": [4, 8, 10, 15, 20],
            "Depth (mm)": [2.5, 14, 5, 17, 10]
        }
        inclusions_df = pd.DataFrame(inclusions_dict)

        # Assign the reference image and mask to class attributes and optionally save to image files
        self.im_reference = phantom_image
        self.mask_reference = reference_mask
        fig, (ax1, ax2) = plt.subplots(figsize=(10, 5), ncols=2, layout='tight')
        ax1.imshow(phantom_image, cmap='inferno')
        ax1.set_title("Lung Reference Image", fontweight='bold')
        ax1.set_axis_off()
        ax2.imshow(reference_mask, cmap='gray')
        ax2.add_patch(Rectangle((-3, -3), width=1, height=1, color=[1, 1, 1], label='Inclusions'))
        ax2.add_patch(Rectangle((-3, -3), width=1, height=1, color=[0.5, 0.5, 0.5], label='Background'))
        ax2.legend(loc='lower left', fontsize=self.TITLE_FS)
        ax2.set_title("Lung Reference Mask", fontweight='bold')
        ax2.set_axis_off()
        if save:
            io.imsave(os.path.join(self.ref_dir, "Lung_reference_image.tiff"), phantom_image)
            io.imsave(os.path.join(self.ref_dir, "Lung_reference_mask.png"), reference_mask)
            with open(os.path.join(self.ref_dir, "Inclusions_info.pkl"), 'wb') as f:
                pickle.dump(inclusions_df, f)
            print(f"Lung phantom reference image and mask successfully created and saved to "
                  f"{os.path.abspath(self.ref_dir)}")
        plt.show()

    def get_inclusion_stats(self, im_target, show_progress=True, verbose=True, print_every=10, save_dir=None):
        """
        Get statistics on the inclusions in the input image of the lung phantom. First register the reference image to
        the input target image, then use the registration parameters to warp the reference mask. Obtain statistics from
        within the marked regions of the transformed mask.

        :param im_target:           Input image of lung phantom
        :param show_progress:       If True, display the registration progress
        :param verbose:             If True, print the current iteration along with the cost value at that iteration
        :param print_every:         Number of iterations between prints
        :param save_dir:            If provided, directory in which to save results
        :return:                    Two dataframes (summary and full) containing statistics on inclusions
        """

        if self.im_reference is None:
            try:
                self.im_reference = io.imread(os.path.join(self.ref_dir, "Lung_reference_image.tiff"))
                self.mask_reference = io.imread(os.path.join(self.ref_dir, "Lung_reference_mask.png"))
            except OSError:
                self.im_reference = lung_reference_image()
                self.mask_reference = lung_reference_mask()
                load_lung_info()

        assert self.im_reference is not None and self.mask_reference is not None, "Missing reference image and/or mask"

        self.show_progress = show_progress
        self.verbose = verbose
        self.print_every = print_every

        # Pad image to make square and preserve aspect ratio. Get original image boundaries in relation to padded image
        if im_target.shape[1] > im_target.shape[0]:
            left = 0
            right = im_target.shape[1]
            extra = im_target.shape[1] - im_target.shape[0]
            top = extra // 2
            bottom = extra // 2 + im_target.shape[0]
        else:
            top = 0
            bottom = im_target.shape[0]
            extra = im_target.shape[0] - im_target.shape[1]
            left = extra // 2
            right = extra // 2 + im_target.shape[1]
        im_target = self.make_square([im_target])[0]

        # Get optimal registration parameters
        start_time = time.time()
        self.opt_reg_params = self._reg_ref_to_target(im_target)
        print(f"         Time taken: {time.time() - start_time} s")
        print(f"         Optimal registration parameters: {self.opt_reg_params}")
        scale = self.opt_reg_params[0]
        rotation = self.opt_reg_params[1]
        t_scale = im_target.shape[0] / self.reg_shape[0]
        translation = [self.opt_reg_params[2] * t_scale, self.opt_reg_params[3] * t_scale]

        # Warp reference mask to the target image
        shift_y, shift_x = np.array(self.im_reference.shape) / 2
        tform_rotation = transform.SimilarityTransform(rotation=rotation)
        tform_shift = transform.SimilarityTransform(translation=[-shift_x, -shift_y])
        tform_shift_inv = transform.SimilarityTransform(translation=[shift_x, shift_y])
        tform_scale_trans = transform.SimilarityTransform(scale=scale, translation=translation)
        tform = (tform_shift + (tform_rotation + tform_shift_inv)) + tform_scale_trans
        mask_registered = transform.warp(self.mask_reference, tform.inverse)
        inc_mask = (mask_registered == 1).astype(np.int64)
        bkg_mask = (mask_registered > 0).astype(np.int64)
        bkg_mask = bkg_mask - inc_mask

        # Display the registered mask overlaid on the target image
        c_inc = 0.18    # 0.18 to get yellow with hsv colormap
        c_bkg = 0.5     # 0.5 to get cyan with hsv colormap
        fig, ax = plt.subplots(figsize=self.FIG_SIZE)
        ax.imshow(im_target, cmap=self.IM_CMAP)
        alpha_inc = inc_mask * self.OVERLAY_ALPHA
        alpha_bkg = bkg_mask * self.OVERLAY_ALPHA
        ax.imshow(c_inc * inc_mask, cmap=self.MASK_CMAP, vmin=0, vmax=1, alpha=alpha_inc)
        ax.imshow(c_bkg * bkg_mask, cmap=self.MASK_CMAP, vmin=0, vmax=1, alpha=alpha_bkg)
        cmap = matplotlib.colormaps[self.MASK_CMAP]
        ax.add_patch(Rectangle((-3, -3), width=1, height=1, color=cmap(int(c_inc * 255)), label='Inclusions'))
        ax.add_patch(Rectangle((-3, -3), width=1, height=1, color=cmap(int(c_bkg * 255)), label='Background'))
        ax.legend(loc='lower left', fontsize=self.TITLE_FS)
        ax.set_title("Registered mask overlaid on target image", fontsize=self.TITLE_FS)
        ax.set_xlim([left, right])
        ax.set_ylim([bottom, top])
        ax.set_axis_off()
        plt.show()

        # Get metrics from the target image using the registered mask
        metrics = self._get_metrics(im_target, inc_mask.astype(bool), bkg_mask.astype(bool), tform, save_dir)

        return metrics

    def initialize_progress_figure(self):
        """
        Initialize the interactive figure (for Jupyter notebooks).
        """
        if "Jupyter" in self.environment:
            self.progress_figure, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(nrows=2, ncols=2,
                                                                                              figsize=self.FIG_SIZE)
        else:
            print("Figure initialization not needed when running in standard Python environment")

    def _reg_ref_to_target(self, im_target):
        """
        Register the reference image to the input target image and return the transformation parameters that achieve
        this. This function is called by SELF.GET_INCLUSION_STATS.

        :param im_target:
        :return:
        """

        # Convert images to uint8, then resize reference and target images to the same shape
        im_target, im_reference = self.cvt_to_uint8([im_target, self.im_reference])
        im_target = transform.resize(im_target, self.reg_shape, order=3, preserve_range=True)
        im_reference = transform.resize(im_reference, self.reg_shape, order=3, preserve_range=True)

        # Maybe create progress figure
        if self.show_progress:
            if "Jupyter" not in self.environment:
                self.progress_figure, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(nrows=2, ncols=2,
                                                                                                  figsize=self.FIG_SIZE)

            self.ax1.imshow(im_target, cmap=self.IM_CMAP)
            self.ax1.set_axis_off()
            self.ax1.set_title("Target image", fontsize=self.TITLE_FS)

            self.ax2.imshow(im_reference, cmap=self.IM_CMAP)
            self.ax2.set_axis_off()
            self.ax2.set_title("Reference image", fontsize=self.TITLE_FS)

            self.ax3.set_axis_off()
            self.ax3.set_title("Warping reference image...", fontsize=self.TITLE_FS)

            self.ax4.set_axis_off()
            self.ax4.set_title("Joint histogram", fontsize=self.TITLE_FS)

            if "Jupyter" not in self.environment:
                plt.show(block=False)
                plt.pause(0.1)
            else:
                self.progress_figure.canvas.draw()
                time.sleep(0.1)
            self.progress_bg = self.progress_figure.canvas.copy_from_bbox(self.progress_figure.bbox)
            self.progress_figure.canvas.blit(self.progress_figure.bbox)

        # Find optimal parameters to warp reference image onto target image
        if self.verbose:
            print("\nFINDING REGISTRATION PARAMETERS...")
        self.n_calls = 0
        p_opt = optim.fmin(self._reg_cost_func, self.init_reg_params, args=(im_target, im_reference), xtol=self.x_tol)

        # Close figure when done
        if self.show_progress:
            plt.close(self.progress_figure)

        return p_opt

    def _reg_cost_func(self, reg_params, im_target, im_reference):
        """
        Cost function to be minimized in order to register the reference image to the target image. The reference image
        is transformed by scaling, rotating and translating, then mutual information is calculated. This function is
        called by SELF._REG_REF_TO_TARGET.

        :param reg_params:          Current registration parameters
        :param im_target:           Target image
        :param im_reference:        Reference image to be warped
        :return:                    Cost to be minimized
        """

        self.n_calls += 1

        # Unpack registration parameters and scale to help optimization search
        scale = reg_params[0]
        rotation = reg_params[1]
        translation = [reg_params[2], reg_params[3]]

        # Generate transform based on input parameters and warp reference image
        shift_y, shift_x = np.array(im_reference.shape) / 2
        tform_rotation = transform.SimilarityTransform(rotation=rotation)
        tform_shift = transform.SimilarityTransform(translation=[-shift_x, -shift_y])
        tform_shift_inv = transform.SimilarityTransform(translation=[shift_x, shift_y])
        tform_scale_trans = transform.SimilarityTransform(scale=scale, translation=translation)
        tform = (tform_shift + (tform_rotation + tform_shift_inv)) + tform_scale_trans
        im_warping = transform.warp(im_reference, tform.inverse, order=3)

        # Maybe show the warped image
        if self.show_progress:
            if self.n_calls == 1:
                self.warping_artist = self.ax3.imshow(im_warping, cmap=self.IM_CMAP, animated=True)
                self.ax3.draw_artist(self.warping_artist)
            else:
                self.progress_figure.canvas.restore_region(self.progress_bg)
                self.warping_artist.set_data(im_warping)
                self.ax3.draw_artist(self.warping_artist)
            self.progress_figure.canvas.blit(self.progress_figure.bbox)
            self.progress_figure.canvas.flush_events()
            self.progress_bg = self.progress_figure.canvas.copy_from_bbox(self.progress_figure.bbox)

        # Calculate cost as mutual information between warped reference image and target image
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero encountered in log")
            cost = self._mutual_information(im_target, im_warping)

        # Maybe print current cost value
        if self.verbose and (self.n_calls - 1) % self.print_every == 0:
            print(f"  Function call {self.n_calls}. Cost: {cost:1.4e}")

        return cost

    def _mutual_information(self, im_target, im_warping):
        """
        Computes similarity between the target image and the image being registered to the target using the
        Kullback-Leibler measure: https://en.wikipedia.org/wiki/Mutual_information
        This function is called by SELF._REG_COST_FUNC.

        :param im_target:       Target image
        :param im_warping:      Image being registered to target
        :return:                Mutual information
        """

        bins = np.linspace(0, 255, self.n_bins)
        hx = np.histogram(im_target, bins=bins)[0]
        hy = np.histogram(im_warping, bins=bins)[0]
        px = hx / np.sum(hx)
        py = hy / np.sum(hy)
        log_px = np.log(px)
        log_py = np.log(py)
        log_px[log_px == -np.inf] = 0
        log_py[log_py == -np.inf] = 0
        hhx = np.sum(px * log_px)
        hhy = np.sum(py * log_py)

        # Compute joint histogram
        h_joint = np.zeros((len(hy)+1, len(hx)+1))
        for i in range(im_target.shape[0]):
            for j in range(im_target.shape[1]):
                x = im_warping[i, j]
                y = im_target[i, j]

                xi = np.argmin(np.abs(bins - x))
                if bins[xi] <= x:
                    xi += 1
                yi = np.argmin(np.abs(bins - y))
                if bins[yi] <= y:
                    yi += 1

                h_joint[yi - 1, xi - 1] += 1

        # Maybe show the joint histogram
        if self.show_progress:
            if self.n_calls == 1:
                v_max = 0.01 * np.max(h_joint)
                self.hist_artist = self.ax4.imshow(h_joint, vmax=v_max, cmap=self.HIST_CMAP, animated=True)
                self.ax4.draw_artist(self.hist_artist)
            else:
                self.progress_figure.canvas.restore_region(self.progress_bg)
                self.hist_artist.set_data(h_joint)
                self.ax4.draw_artist(self.hist_artist)
            self.progress_figure.canvas.blit(self.progress_figure.bbox)
            self.progress_figure.canvas.flush_events()
            self.progress_bg = self.progress_figure.canvas.copy_from_bbox(self.progress_figure.bbox)

        # Calculate mutual information
        pxy = h_joint / np.sum(h_joint)
        log_pxy = np.log(pxy)
        log_pxy[log_pxy == -np.inf] = 0
        hxy = np.sum(pxy * log_pxy)
        mi = hhx + hhy - hxy

        return mi

    def _get_metrics(self, im_target, inc_mask, bkg_mask, tform, save_dir):
        """
        Obtain metrics from the target image using the registered masks, and optionally save the results to an Excel
        file in SAVE_DIR. This function is called by SELF.GET_INCLUSION_STATS.

        :param im_target:       The target image
        :param inc_mask:        Registered mask giving locations of inclusions
        :param bkg_mask:        Registered mask of phantom background
        :param tform:           Transform that registered the reference to the target
        :param save_dir:        Directory in which to save results
        :return:                Dataframe containing metrics
        """

        # Get reference mask inclusion properties
        with open(os.path.join(self.ref_dir, "Inclusions_info.pkl"), 'rb') as f:
            inclusions_df = pickle.load(f)

        # Get the center locations of the registered inclusions
        x = inclusions_df["x coord"].to_numpy()
        y = inclusions_df["y coord"].to_numpy()
        centers = [np.array([x[i], y[i], 1]) for i in range(len(x))]
        reg_centers = [np.matmul(tform, center) for center in centers]
        x_reg = [reg_center[0] for reg_center in reg_centers]
        y_reg = [reg_center[1] for reg_center in reg_centers]

        # Label inclusions mask and sort by area
        labeled_im, _ = label(inc_mask)
        inclusions = regionprops(labeled_im, intensity_image=inc_mask)
        areas = [inclusion.area for inclusion in inclusions]
        sorted_idx = np.argsort(np.array(areas))
        inc_masks = []
        for inclusion in np.array(inclusions)[sorted_idx]:
            rough_diameter = inclusion.bbox[2] - inclusion.bbox[0]
            selection_mask = self.circular_mask(inc_mask.shape, inclusion.centroid, (rough_diameter * 1.5) / 2)
            inc_masks.append(inc_mask * selection_mask)

        # Calculate metrics
        cnt = [np.sum(mask) for mask in inc_masks]
        avg = [np.mean(im_target[mask]) for mask in inc_masks]
        std = [np.std(im_target[mask]) for mask in inc_masks]
        med = [np.median(im_target[mask]) for mask in inc_masks]
        p90 = [np.percentile(im_target[mask], q=90) for mask in inc_masks]
        mxm = [np.max(im_target[mask]) for mask in inc_masks]
        mnm = [np.min(im_target[mask]) for mask in inc_masks]
        tbr = [np.mean(im_target[mask]) / np.mean(im_target[bkg_mask]) for mask in inc_masks]
        cnr = [(np.mean(im_target[mask]) - np.mean(im_target[bkg_mask])) / np.std(im_target[bkg_mask])
               for mask in inc_masks]
        cvr = [(np.mean(im_target[mask]) - np.mean(im_target[bkg_mask])) /
               np.sqrt(np.var(im_target[mask]) + np.var(im_target[bkg_mask])) for mask in inc_masks]

        # Write metrics to dataframe and optionally save to file
        metrics_dict = {
            "Label": inclusions_df["Label"].to_numpy(),
            "Inclusion diameter (mm)": inclusions_df["Actual diameter (mm)"].to_numpy(),
            "Inclusion depth (mm)": inclusions_df["Depth (mm)"].to_numpy(),
            "TBR": tbr,
            "CNR": cnr,
            "CVR": cvr,
            "Pixel count": cnt,
            "Mean": avg,
            "Std": std,
            "Median": med,
            "P90": p90,
            "Max": mxm,
            "Min": mnm,
            "X-coord": x_reg,
            "Y-coord": y_reg,
            "Eccentricity": inclusions_df["Eccentricity"].to_numpy(),
            "Equivalent diameter (pixels)": inclusions_df["Equivalent diameter (pixels)"].to_numpy()
        }
        metrics_df_full = pd.DataFrame(metrics_dict)
        metrics_df_full["TBR"] = metrics_df_full["TBR"].map("{:,.3f}".format).astype(np.float64)
        metrics_df_full["CNR"] = metrics_df_full["CNR"].map("{:,.3f}".format).astype(np.float64)
        metrics_df_full["CVR"] = metrics_df_full["CVR"].map("{:,.3f}".format).astype(np.float64)
        metrics_df_summary = metrics_df_full[
            ["Label", "Inclusion diameter (mm)", "Inclusion depth (mm)", "TBR", "CNR", "CVR"]
        ]
        if save_dir is not None:
            with pd.ExcelWriter(os.path.join(save_dir, "Phantom_metrics.xlsx")) as writer:
                metrics_df_full.to_excel(writer, index=False)
            with pd.ExcelWriter(os.path.join(save_dir, "Phantom_summary_metrics.xlsx")) as writer:
                metrics_df_summary.to_excel(writer, index=False)

        return metrics_df_summary, metrics_df_full

    @staticmethod
    def check_environment():
        """
        Return the Python environment type.
        """
        try:
            from IPython import get_ipython
            get_ipython()
            if 'IPKernelApp' in get_ipython().config:
                # print("Running in Jupyter Notebook")
                get_ipython().magic('matplotlib widget')
                get_ipython().run_line_magic('matplotlib', 'ipympl')
                return "Jupyter Notebook"
            else:
                # print("Running in JupyterLab")
                get_ipython().run_line_magic('matplotlib', 'ipympl')
                return "JupyterLab"
        except AttributeError:
            # print("Running in a standard Python environment")
            return "Standard Python"

    @staticmethod
    def make_square(im_array):
        """
        Pad the input image(s) in whichever dimension is required to make a square.

        :param im_array:        List of input image(s)
        :return:                Image(s) padded to make a square
        """

        im_array_square = []

        for im in im_array:
            bkg = im[:int(0.1 * im.shape[0]), :int(0.1 * im.shape[1])]
            bkg_avg = np.mean(bkg)
            bkg_std = np.std(bkg)
            if im.shape[1] > im.shape[0]:       # Pad rows
                im_square = np.zeros(shape=(im.shape[1], im.shape[1]), dtype=im.dtype)
                im_square += (bkg_avg * np.ones(im_square.shape)).astype(im.dtype) + \
                    (bkg_std * np.random.randn(*im_square.shape)).astype(im.dtype)
                extra = im.shape[1] - im.shape[0]
                im_square[extra // 2:extra // 2 + im.shape[0], :] = im
            elif im.shape[0] > im.shape[1]:     # Pad columns
                im_square = np.zeros(shape=(im.shape[0], im.shape[0]), dtype=im.dtype)
                im_square += (bkg_avg * np.ones(im_square.shape)).astype(im.dtype) + \
                    (bkg_std * np.random.randn(*im_square.shape)).astype(im.dtype)
                extra = im.shape[0] - im.shape[1]
                im_square[:, extra // 2:extra // 2 + im.shape[1]] = im
            else:       # Do nothing, already square
                im_square = im
            im_array_square.append(im_square)

        return im_array_square

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
            if im_type == 'uint16':
                im_array_uint16.append(im)
            elif im_type == 'uint8':
                im16 = ((im.astype(np.float64) / 255) * 65535).astype(np.uint16)
                im_array_uint16.append(im16)
            else:
                im[im < 0] = 0
                im16 = ((im.astype(np.float64) / np.max(im.astype(np.float64))) * 65535).astype(np.uint16)
                im_array_uint16.append(im16)

        return im_array_uint16

    @staticmethod
    def cvt_to_uint8(im_array):
        """
        Convert the input image(s) to 8-bit.

        :param im_array:        List of input image(s)
        :return:                Image(s) converted to 16-bit
        """

        im_array_uint8 = []
        im_type = im_array[0].dtype

        for im in im_array:
            if im_type == 'uint8':
                im_array_uint8.append(im)
            elif im_type == 'uint16':
                im8 = ((im.astype(np.float64) / 65535) * 255).astype(np.uint8)
                im_array_uint8.append(im8)
            else:
                im[im < 0] = 0
                im8 = ((im.astype(np.float64) / np.max(im.astype(np.float64))) * 255).astype(np.uint8)
                im_array_uint8.append(im8)

        return im_array_uint8

    @staticmethod
    def circular_mask(shape, center, radius):
        """
        Create a circular mask of radius RADIUS, centered on CENTER.

        :param shape:           Shape of the image/background
        :param center:          Center of the circle
        :param radius:          Radius of the mask to produce
        :return:                Mask
        """

        yy, xx = np.ogrid[:shape[0], :shape[1]]
        distance_from_center = np.sqrt((xx - center[1]) ** 2 + (yy - center[0]) ** 2)
        mask = distance_from_center <= radius

        return mask
