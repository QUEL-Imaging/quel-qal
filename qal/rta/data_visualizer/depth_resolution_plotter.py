import os
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score


class DepthDataPlotter:

    GRAPH_PARAMS = {
        'figure.figsize': [8, 6],
        'figure.autolayout': True,
        'axes.grid': True,
        'axes.labelsize': 22,
        'axes.labelweight': 700,
        'xtick.labelsize': 'x-large',
        'ytick.labelsize': 'x-large',
        'font.weight': 'bold',
        'font.size': 8,
        'legend.edgecolor': '#454545',
        'patch.linewidth': 3,
        'legend.fontsize': 'x-large'
    }
    ANNOTATION_FS = 16
    INTENSITY_MS = 4
    SPREAD_MS = 200
    SPREAD_PROFILE_CMAP = 'jet'

    def __init__(self, analyzer_outputs):
        """
        Initialize DepthDataPlotter with the outputs of a DepthAnalyzer object.

        :param analyzer_outputs:        Outputs of a DepthAnalyzer object
        """

        self.data = analyzer_outputs                # Data to be plotted
        self.plot_intensity_tf = True               # Whether to plot intensity profiles
        self.plot_spreads_tf = True                 # Whether to plot spread information
        self.plot_fidelity_line = False             # If True, plot line where intensity falls below SELF.FIDELITY_RATIO
        self.fidelity_line_idx = None               # Index where intensity falls below SELF.FIDELITY_RATIO
        self.fidelity_ratio = 0.02                  # Peak intensity ratio below which FWHM and AUC data is questionable
        plt.rcParams.update(self.GRAPH_PARAMS)      # Update graph parameters

    def plot_data(self, graph_type='All', plot_smoothed=True, save_dir=None, fidelity_ratio=None):
        """
        Plot the data and optionally save.

        :param graph_type:      Determines which types of graphs to produce. Options are 'Intensity Profile', 'Spread',
                                'All'
        :param plot_smoothed:   If True, plot the smoothed data. Otherwise, plot the raw data
        :param save_dir:        If provided, directory to which to save plots
        :param fidelity_ratio:  Peak intensity ratio below which FWHM and AUC data is questionable
        :return:
        """

        if graph_type == 'Intensity Profile':
            self.plot_spreads_tf = False
        elif graph_type == 'Spread':
            self.plot_intensity_tf = False

        if fidelity_ratio is not None:
            self.fidelity_ratio = fidelity_ratio

        if self.plot_intensity_tf:
            self.plot_intensity(plot_smoothed=plot_smoothed, save_dir=save_dir)

        if self.plot_spreads_tf:
            self.plot_spreads(plot_smoothed=plot_smoothed, save_dir=save_dir)

        plt.rcdefaults()        # Restore original matplotlib graph parameters before returning

    def plot_intensity(self, plot_smoothed, save_dir):
        """
        Plot the intensity profile along the depth channel

        :param plot_smoothed:       Boolean. If Tru, plot the smoothed intensity profile
        :param save_dir:            Directory to save plots to
        :return:
        """

        # Get x-axis data
        full_span = self.data["Distance in mm"]
        depth = self.data["Descent depth"]

        # Get y-axis data
        if plot_smoothed:
            full_profile = self.data["Smoothed intensity profile"]
            descent_profile = self.data["Smoothed intensities along descent"]
        else:
            full_profile = self.data["Intensity profile"]
            descent_profile = self.data["Intensities along descent"]

        # Plot intensity against distance along the channel
        fig1, ax1 = plt.subplots()
        ax1.plot(full_span, full_profile, color='#1B3D87', marker='D', ms=self.INTENSITY_MS, linestyle='')
        ax1.set_xlabel("Distance (mm)")
        ax1.set_ylabel("Intensity (a.u.)")
        ax1.set_title("Intensity vs distance along channel", fontsize=self.ANNOTATION_FS)

        # Optionally save the figure
        if save_dir is not None:
            fig_name = "Intensity vs channel length"
            plt.savefig(os.path.join(save_dir, fig_name))

        # Normalize the intensities along the descent and fit an exponential trend line
        def exponential(x, a, b):
            return a * np.exp(b * x)
        descent_profile = ((descent_profile - self.data["Background"]) /
                           (np.max(descent_profile) - self.data["Background"]))
        fit_start = np.argmax(descent_profile) + int(0.01 * len(descent_profile))
        to_fit = descent_profile[fit_start:]
        depth_to_fit = depth[fit_start:]
        params = curve_fit(exponential, depth_to_fit, to_fit)[0]
        fitted = exponential(depth_to_fit, *params)
        r2 = r2_score(to_fit, fitted)

        # Check if and where intensity falls below acceptable limit for calculating FHWM and AUC
        if np.any(descent_profile < self.fidelity_ratio):
            self.plot_fidelity_line = True
            self.fidelity_line_idx = np.argmin(np.abs(descent_profile - self.fidelity_ratio))

        # Plot the normalized intensity along the descent and fitted trend line
        fig2, ax2 = plt.subplots()
        ax2.plot(depth, descent_profile, color='#1B3D87', marker='D', ms=self.INTENSITY_MS, linestyle='', label='Data')
        ax2.plot(depth_to_fit, fitted, color='g', linestyle='--', linewidth=3, label='Trendline')
        ax2.set_xlabel("Depth (mm)")
        ax2.set_ylabel("Normalized intensity (a.u.)")
        ax2.set_title("Normalized intensity vs channel depth", fontsize=self.ANNOTATION_FS)
        equation = f"$y = {params[0]:.3f}e^{{{params[1]:.3f}x}}$"
        r2_str = f"$r^2 = {r2:.2f}$"
        annotation = f"{equation}\n{r2_str}"
        ax2.annotate(annotation, xy=(0.6, 0.8), xycoords='axes fraction', color='g', fontsize=self.ANNOTATION_FS)
        ax2.legend(loc='lower left')

        # Optionally save the figure
        if save_dir is not None:
            fig_name = "Normalized intensity vs depth"
            plt.savefig(os.path.join(save_dir, fig_name))

        # Display the figures or wait to display if also plotting spreads
        if self.plot_spreads_tf:
            plt.draw()
        else:
            plt.show()

    def plot_spreads(self, plot_smoothed, save_dir):
        """
        Plot the intensity spreads, full-width-half-max, and area under the curve at the depth values in SELF.DATA

        :param plot_smoothed:       Boolean. If Tru, plot the smoothed spread profiles
        :param save_dir:            Directory to save plots to
        :return:
        """

        # Get x-axis data
        dist = self.data["Vertical distance"]   # For spread profiles
        eval_depths = self.data["Depths"]       # For FWHM
        depth = self.data["Descent depth"]

        # Get y-axis data
        spreads = self.data["Spreads"]
        fwhm = np.zeros(len(spreads))
        auc = np.zeros(len(spreads))
        spread_profiles = []
        for i in range(len(spreads)):
            if plot_smoothed:
                spread_profiles.append(spreads[f"{eval_depths[i]} mm depth"]["Smoothed spread profile"])
                fwhm[i] = spreads[f"{eval_depths[i]} mm depth"]["FWHM (smoothed)"]
                auc[i] = spreads[f"{eval_depths[i]} mm depth"]["AUC (smoothed)"]
            else:
                spread_profiles.append(spreads[f"{eval_depths[i]} mm depth"]["Spread profile"])
                fwhm[i] = spreads[f"{eval_depths[i]} mm depth"]["FWHM"]
                auc[i] = spreads[f"{eval_depths[i]} mm depth"]["AUC"]

        # Plot the spread profiles
        fig1, ax1 = plt.subplots()
        cmap = matplotlib.colormaps[self.SPREAD_PROFILE_CMAP]
        cmap_idx = (eval_depths - np.min(eval_depths)) / (np.max(eval_depths) - np.min(eval_depths))
        cmap_idx = (cmap_idx * 255).astype(np.int64)[::-1]
        for i, cmap_id in enumerate(cmap_idx):
            ax1.plot(dist, spread_profiles[i], linewidth=3, color=cmap(cmap_id), label=f"{eval_depths[i]:.2f} mm depth")
        ax1.set_xlabel("Distance from channel center (mm)")
        ax1.set_ylabel("Intensity (a.u.)")
        ax1.set_title("Intensity spread profiles", fontsize=self.ANNOTATION_FS)
        ax1.legend()

        # Optionally save the figure
        if save_dir is not None:
            fig_name = "Intensity spread profiles"
            plt.savefig(os.path.join(save_dir, fig_name))

        # Plot the full-width-half-max against depth
        fig2, ax2 = plt.subplots()
        colors = [cmap(idx) for idx in cmap_idx]
        ax2.scatter(eval_depths, fwhm, color=colors, marker='D', s=self.SPREAD_MS)
        if self.plot_fidelity_line:
            y_lim = ax2.get_ylim()
            idx = self.fidelity_line_idx
            ax2.plot([depth[idx], depth[idx]], [y_lim[0], y_lim[1]], color='#800000', linestyle='--', linewidth=1.5)
            dx = 0.2 * (np.max(depth) - depth[idx])
            hdl = 0.05 * (np.max(depth) - depth[idx])
            hdw = 0.02 * (y_lim[1] - y_lim[0])
            ax2.arrow(depth[idx], np.mean(np.array(y_lim)), dx, 0, color='#800000', linewidth=2, head_width=hdw,
                      head_length=hdl)
            annotation = f"Intensities below\n{self.fidelity_ratio * 100:.1f} % of peak"
            ax2.annotate(annotation, xy=(0.5, 0.6), xycoords='axes fraction', color='#800000',
                         fontsize=self.ANNOTATION_FS)
            ax2.set_ylim(y_lim)
        ax2.set_xlabel("Depth (mm)")
        ax2.set_ylabel("FWHM (mm)")
        ax2.set_title("Fluorescence full-width-half-max vs channel depth", fontsize=self.ANNOTATION_FS)

        # Optionally save the figure
        if save_dir is not None:
            fig_name = "FWHM vs depth"
            plt.savefig(os.path.join(save_dir, fig_name))

        plt.show()
