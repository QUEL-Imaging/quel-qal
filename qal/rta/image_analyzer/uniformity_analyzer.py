import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable


class UniformityAnalyzer:

    # Some parameters for optional figures produced
    FIG_SIZE = (8, 6)
    IM_COLORMAP = 'inferno'
    BBOX_COLOR = 'red'
    DOT_COLOR = 'green'
    DOT_ALPHA = 0.5
    SURF_ALPHA = 0.5

    def __init__(self, params=None):
        """
        Initialize UniformityAnalyzer with optional parameters for class methods.

        :param params:      Parameters for class methods (optional)
        """

        # Default parameters for class methods
        if params is None:
            self.params = {
                "Number of x query points": 500,
                "Number of y query points": 500,
                "Fit method": 'b-spline',
                "Spline degree": 5,
                "RBF smoothing": 300,
                "No negatives in fit": False,
                "Zero outside data range": False,
                "Show fit": False,
                "Save output": True
            }
        else:
            self.params = params

        # Initialize class attributes from PARAMS
        self.n_xq = self.params["Number of x query points"]
        self.n_yq = self.params["Number of y query points"]
        self.method = self.params["Fit method"]
        self.k_spline = self.params["Spline degree"]
        self.rbf_smooth = self.params["RBF smoothing"]
        self.no_neg = self.params["No negatives in fit"]
        self.zero_extra = self.params["Zero outside data range"]
        self.show_fit = self.params["Show fit"]
        self.save_output = self.params["Save output"]

        # Initialize other needed attributes
        self.dots = None
        self.fov = None
        self.xq = None
        self.yq = None
        self.surf_rep = None
        self.output = None

    def generate_surf_rep(self, rud_dots, method=None):
        """
        Generate a surface representation of fluorescence uniformity across the field of view using information about
        the spatial location of the uniformity and distortion target wells and their respective intensities.

        :param rud_dots:        A dataframe of RUD target well locations and average intensities
        :param method:          Method to use in generating the surface representation. Options are 'b-spline' and
                                'rbf'. If None, defaults to SELF.METHOD
        """

        if method is not None:
            assert method == 'b-spline' or method == 'rbf', f"'{method}' is not an accepted method"
        else:
            method = self.method

        # Parse the input data
        self.dots = rud_dots["dots_df"]
        self.fov = rud_dots["fov"]
        image_dir = rud_dots["image_dir"]
        x = self.dots["x"].to_numpy()
        y = self.dots["y"].to_numpy()
        z = self.dots["dot_intensity"].to_numpy()
        data = (x, y, z)

        # Generate surface representation
        if method == 'b-spline':
            self.surf_rep, rss, r_sq = self.fit_data(data, show_fit=self.show_fit)
            self.output = {
                "surf_rep": self.surf_rep,
                "xq": self.xq,
                "yq": self.yq,
                "rss": rss,
                "r_sq": r_sq,
                "dots": self.dots,
                "fov": self.fov,
                "save_output": self.save_output,
                "image_dir": image_dir
            }
        elif method == 'rbf':
            self.surf_rep = self.interp_data(data, show_fit=self.show_fit)
            self.output = {
                "surf_rep": self.surf_rep,
                "xq": self.xq,
                "yq": self.yq,
                "dots": self.dots,
                "fov": self.fov,
                "save_output": self.save_output,
                "image_dir": image_dir
            }
        print("  Done")

        # Optionally save the output
        if self.save_output:
            save_dir = os.path.join(image_dir, "Surface Representation")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            with open(os.path.join(save_dir, "output_dict.pkl"), 'wb') as f:
                pickle.dump(self.output, f)

    def fit_data(self, data, distortion_call=False, show_fit=True):
        """
        Find a 2D B-spline representation of the input data.

        :param data:                Tuple of x, y, and z data
        :param distortion_call:     Whether this function was called by DISTORTION_ANALYZER
        :param show_fit:            Whether to display the fit
        :return:                    Surface representation
        """

        weights = np.ones(len(data[0]))
        if distortion_call:
            extra_val = np.nan
            weights[0] = 50
        else:
            extra_val = 0

        x, y, z = data
        z /= np.max(z)      # Normalize to help fitting

        # Get parameters
        k = self.k_spline
        no_neg = self.no_neg
        zero_extra = self.zero_extra

        # Create query points
        if zero_extra:
            xq_min = np.min(x)
            xq_max = np.max(x)
            yq_min = np.min(y)
            yq_max = np.max(y)
        else:
            xq_min = 0
            xq_max = self.fov[1]
            yq_min = 0
            yq_max = self.fov[0]
        self.xq = np.linspace(xq_min, xq_max, self.n_xq)
        self.yq = np.linspace(yq_min, yq_max, self.n_yq)

        # Fit to B-spline
        print("\nPERFORMING B-SPLINE FITTING...")
        tck, rss, _, _ = interpolate.bisplrep(x, y, z, xb=xq_min, xe=xq_max, yb=yq_min, ye=yq_max, kx=k, ky=k,
                                              w=weights, full_output=True)
        fit = interpolate.bisplev(self.xq, self.yq, tck).T
        if no_neg:
            fit[fit < 0] = 0

        # Calculate total sum of squares, then pseudo R^2
        tss = np.sum(np.square(fit - np.mean(fit)))
        r_sq = 1 - rss / tss
        if show_fit:
            print(f"  Average squared residual of fit: {rss / len(z):.5f}\n  R^2: {r_sq:.3f}")

        # Maybe zero some points
        if zero_extra:
            fit = self.remove_extra(fit, x, y, extra_val)

        # Optionally display the fit
        if show_fit:
            self.display_fit(fit, x, y, z, rss, r_sq)

        return fit, rss, r_sq

    def interp_data(self, data, show_fit=True):
        """
        Interpolate the input data using RBF interpolation.

        :param data:            Tuple of x, y, and z data
        :param show_fit:        Whether to display the interpolation results
        :return:                Surface representation
        """

        x, y, z = data
        z /= np.max(z)
        rbf_y = np.vstack((x, y)).T
        rbf_d = z         # Normalize to help with fitting

        # Create query points
        if self.zero_extra:
            xq_min = np.min(x)
            xq_max = np.max(x)
            yq_min = np.min(y)
            yq_max = np.max(y)
        else:
            xq_min = 0
            xq_max = self.fov[1]
            yq_min = 0
            yq_max = self.fov[0]
        self.xq = np.linspace(xq_min, xq_max, self.n_xq)
        self.yq = np.linspace(yq_min, yq_max, self.n_yq)

        # Interpolate the input data and generate surface representation
        print("\nPERFORMING RBF INTERPOLATION...")
        print("  Finding function representing input data...")
        rbf = interpolate.RBFInterpolator(rbf_y, rbf_d, smoothing=self.rbf_smooth, kernel='linear', neighbors=100)
        print("  Generating surface representation...")
        xx, yy = np.meshgrid(self.xq, self.yq)
        query_pts = np.vstack((xx.flatten(), yy.flatten())).T
        fit = rbf(query_pts).reshape((len(self.yq), len(self.xq)))
        if self.no_neg:
            fit[fit < 0] = 0

        # Maybe zero some points
        if self.zero_extra:
            fit = self.remove_extra(fit, x, y, extra_val=0)

        # Optionally display the interpolation results
        if show_fit:
            self.display_fit(fit, x, y, rbf_d)

        return fit

    def remove_extra(self, fit, x, y, extra_val, sc=1.1):
        """
        Further zero out locations of FIT that are more than a specified distance away from any other data point.

        :param fit:         Surface representation fit
        :param x:           x-value of input data
        :param y:           y-value of input data
        :param extra_val:   Value with which to replace points outside the data range
        :param sc:          Scaling factor for inter-dot distance; zero points further than this from any data point
        :return:            Fit with some locations maybe set to zero
        """

        # Estimate smallest distance between data points in the first image (there should be 500 points at least)
        inter_dot = np.sort(self.dist_between(x[0], y[0], x[1:501], y[1:501]))[1]

        # Zero points in fit too far away
        xqm, yqm = np.meshgrid(self.xq, self.yq)
        for i in range(fit.shape[0]):
            for j in range(fit.shape[1]):
                dists_to_data = self.dist_between(xqm[i, j], yqm[i, j], x, y)
                if np.all(dists_to_data > sc * inter_dot):
                    fit[i, j] = extra_val

        return fit

    def display_fit(self, fit, x, y, z, rss=None, r_sq=None):
        """
        Display the resulting fit from 2D B-spline interpolation.

        :param fit:     Fit from 2D B-spline interpolation
        :param x:       x-values from data
        :param y:       y-values from data
        :param z:       z-values from data (normalized to max)
        :param rss:     Sum of squared residuals of fit
        :param r_sq:    A pseudo R^2 for the fit
        """

        plt.rcParams.update({'font.size': 12})

        xq_min = np.min(self.xq)
        xq_max = np.max(self.xq)
        yq_min = np.min(self.yq)
        yq_max = np.max(self.yq)

        x_min = np.min(x)
        x_max = np.max(x)
        y_min = np.min(y)
        y_max = np.max(y)
        x_range = x_max - x_min
        y_range = y_max - y_min

        xqm, yqm = np.meshgrid(self.xq, self.yq)

        fig1, ax1 = plt.subplots(figsize=self.FIG_SIZE)
        if self.zero_extra:
            xq_spacing = self.xq[1] - self.xq[0]
            yq_spacing = self.yq[1] - self.yq[0]
            x_fov = np.arange(0, self.fov[1], xq_spacing)
            y_fov = np.arange(0, self.fov[0], yq_spacing)
            fit_in_fov = np.zeros((len(y_fov), len(x_fov)))
            xq_start = np.argmin(np.abs(x_fov - xq_min))
            xq_end = np.argmin(np.abs(x_fov - xq_max))
            yq_start = np.argmin(np.abs(y_fov - yq_min))
            yq_end = np.argmin(np.abs(y_fov - yq_max))
            fit_in_fov[yq_start:yq_end + 1, xq_start:xq_end + 1] = fit
        else:
            fit_in_fov = fit
        im = ax1.imshow(fit_in_fov, cmap=self.IM_COLORMAP, extent=(0, self.fov[1], self.fov[0], 0))
        ax1.add_patch(Rectangle((x_min, y_min), x_range, y_range, color=self.BBOX_COLOR, linestyle='--',
                                linewidth=1.5, fill=False, label='Data bounding box'))
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("right", size="3%", pad=0.1)
        cb = plt.colorbar(im, cax=cax)
        cb.set_label("Normalized Intensity")
        ax1.set_axis_off()
        ax1.legend()

        fig2, ax2 = plt.subplots(figsize=self.FIG_SIZE, subplot_kw={"projection": "3d"})
        ax2.plot_surface(xqm, yqm, fit, cmap=self.IM_COLORMAP, linewidth=0, alpha=self.SURF_ALPHA)
        ax2.scatter(x, y, z, color=self.DOT_COLOR, label='Data')
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.set_zlabel("Normalized Intensity")
        if rss is not None:
            ax2.set_title(f"Average squared residual of b-spline fit: {rss / len(z):.5f}\n$R^2: {r_sq:.3f}$")
        else:
            ax2.set_title("RBF interpolation results")
        ax2.legend()

        plt.show()

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
        self.n_xq = self.params["Number of x query points"]
        self.n_yq = self.params["Number of y query points"]
        self.method = self.params["Fit method"]
        self.k_spline = self.params["Spline degree"]
        self.rbf_smooth = self.params["RBF smoothing"]
        self.no_neg = self.params["No negatives in fit"]
        self.zero_extra = self.params["Zero outside data range"]
        self.show_fit = self.params["Show fit"]
        self.save_output = self.params["Save output"]

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
