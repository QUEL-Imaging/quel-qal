import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable


class DistortionVisualizer:

    # Fixed attributes for figures generated
    COLORMAP = 'seismic'
    DOT_COLOR = '#1B3D87'
    DOT_BORDER = 'cyan'
    DOT_SIZE = 8
    FIT_COLOR = '#454545'
    ANNOTATION_FS = 11
    LEGEND_FS = 14
    GRAPH_PARAMS = {
        'figure.figsize': [9, 6],
        'font.size': 10,
        'axes.labelsize': 16,
        'axes.labelweight': 500,
    }

    def __init__(self):
        self.save_figures = False
        self.im_dir = None
        self.save_dir = None
        self.lens_maker = None

        plt.rcParams.update(self.GRAPH_PARAMS)

    def visualize_distortion(self, distortion_analyzer_outputs, save=True, lens_maker=False):
        """
        Generate figures displaying distortion from the input data.

        :param distortion_analyzer_outputs:     Distortion data
        :param save:                            If True, save figures
        :param lens_maker:                      If True, generate lens manufacturer style distortion plot
        """

        self.lens_maker = lens_maker

        print("\nGENERATING FIGURES...")

        im_dir = distortion_analyzer_outputs["Directory"]
        self.save_figures = save
        if self.save_figures:
            self.save_dir = os.path.join(im_dir, "Distortion Figures")
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

        distances = distortion_analyzer_outputs["Actual distances"]
        distortion = distortion_analyzer_outputs["Distortion"]
        fov = distortion_analyzer_outputs["FOV"]
        xq = distortion_analyzer_outputs["xq"]
        yq = distortion_analyzer_outputs["yq"]
        distortion_map = distortion_analyzer_outputs["Distortion map"] * np.max(distortion)

        # Plot image height against distortion
        self.plot_distortion(distances, distortion, fov)

        # Display map of distortion across the field of view
        self.map_distortion(xq, yq, distortion_map, fov)

        # Show images, then return matplotlib parameters to default
        plt.show()
        plt.rcdefaults()

    def plot_distortion(self, distances, distortion, fov):
        """
        Generate figure for image height versus distortion. Fit distortion data to 3rd degree polynomial.

        :param distances:       Actual image heights of grid dots
        :param distortion:      Calculated local geometric distortion at each grid point
        :param fov:             Field of view of the imaging system (x pixels, y pixels)
        """

        distances_max = np.sqrt(fov[0]**2 + fov[1]**2) / 2
        image_height = (distances / distances_max) * 100

        # Fit data to 3rd degree polynomial that intersects the origin (no distortion at image center)
        def poly3_no_constant(x, a, b, c):
            return a * np.power(x, 3) + b * (x**2) + c * x

        p = curve_fit(poly3_no_constant, image_height, distortion)[0]
        interp_height = np.linspace(0, np.max(image_height), 500)
        fitted_interp = poly3_no_constant(interp_height, *p)
        fitted_distortion = poly3_no_constant(image_height, *p)
        r2 = r2_score(distortion, fitted_distortion)
        sign1 = "+" if p[1] >= 0 else "-"
        sign2 = "+" if p[2] >= 0 else "-"
        max_distortion = (np.min(fitted_interp) if np.abs(np.min(fitted_interp)) > np.abs(np.max(fitted_interp))
                          else np.max(fitted_interp))
        ann_loc = (0.5, 0.85) if max_distortion < 0 else (0.1, 0.85)
        leg_loc = 'lower left' if max_distortion < 0 else 'lower right'

        fig1, ax1 = plt.subplots()
        ax1.axhline(y=0, xmin=0, color='#D3D3D3', linewidth=1)
        ax1.axvline(x=0, color='#D3D3D3', linewidth=1)
        ax1.scatter(image_height, distortion, s=self.DOT_SIZE, edgecolors=self.DOT_BORDER, facecolors=self.DOT_COLOR,
                    label="Data")
        ax1.plot(interp_height, fitted_interp, linewidth=3, linestyle='--', color=self.FIT_COLOR, label="Fit")
        ax1.set_xlabel("Image Height (%)")
        ax1.set_ylabel("Distortion (%)")
        ax1.set_title("Distortion vs image height - ISO 17850-aligned\n", fontweight='bold')
        equation = f"$y = {p[0]:.2g} x^{3} {sign1} {np.abs(p[1]):.2g} x^{2} {sign2} {np.abs(p[2]):.2g} x$"
        r2_str = f"$r^2 = {r2:.2f}$"
        max_str = f"Max distortion = {max_distortion:.2f} %"
        annotation = f"{equation}\n{r2_str}\n{max_str}"
        ax1.annotate(annotation, xy=ann_loc, xycoords='axes fraction', color=self.FIT_COLOR,
                     fontsize=self.ANNOTATION_FS,
                     bbox=dict(facecolor='white', edgecolor='none', alpha=0.6, boxstyle='round'))
        ax1.legend(loc=leg_loc, fontsize=self.LEGEND_FS)

        # Optionally save the figure
        if self.save_figures:
            fig_name = "Fitted distortion vs image height"
            plt.savefig(os.path.join(self.save_dir, fig_name))

        # Create figure similar to lens manufacturers' distortion plots (flip axes)
        if self.lens_maker:
            fig2, ax2 = plt.subplots()
            ax2.axhline(y=0, color='#D3D3D3', linewidth=1)
            ax2.axvline(x=0, color='#D3D3D3', linewidth=1)
            ax2.plot(fitted_interp, interp_height, linewidth=3, color=self.DOT_COLOR)
            ax2.set_xlabel("Distortion (%)")
            ax2.set_ylabel("Image Height (%)")
            ax2.set_title("Image height vs distortion - lens designer plot\n", fontweight='bold')

            # Optionally save the figure
            if self.save_figures:
                fig_name = "Fitted image height vs distortion"
                plt.savefig(os.path.join(self.save_dir, fig_name))

    def map_distortion(self, xq, yq, distortion_map, fov):
        """
        Generate figure mapping distortion across the field of view.

        :param xq:                  x-axis points for which DISTORTION_MAP was generated
        :param yq:                  y-axis points for which DISTORTION_MAP was generated
        :param distortion_map:      Surface representation of distortion
        :param fov:                 Field of view of the imaging system (x pixels, y pixels)
        """

        xq_min = np.min(xq)
        xq_max = np.max(xq)
        yq_min = np.min(yq)
        yq_max = np.max(yq)
        max_distortion = (np.nanmin(distortion_map) if np.abs(np.nanmin(distortion_map)) >
                          np.abs(np.nanmax(distortion_map)) else np.nanmax(distortion_map))

        if xq[0] > 0:
            xq_spacing = xq[1] - xq[0]
            yq_spacing = yq[1] - yq[0]
            x_fov = np.arange(0, fov[1], xq_spacing)
            y_fov = np.arange(0, fov[0], yq_spacing)
            fit_in_fov = np.nan * np.ones((len(y_fov), len(x_fov)))
            xq_start = np.argmin(np.abs(x_fov - xq_min))
            xq_end = np.argmin(np.abs(x_fov - xq_max))
            yq_start = np.argmin(np.abs(y_fov - yq_min))
            yq_end = np.argmin(np.abs(y_fov - yq_max))
            fit_in_fov[yq_start:yq_end + 1, xq_start:xq_end + 1] = distortion_map
        else:
            fit_in_fov = distortion_map.copy()

        fig1, ax1 = plt.subplots()
        v_min = max_distortion if max_distortion < 0 else -max_distortion
        v_max = -v_min
        im = ax1.imshow(fit_in_fov, vmin=v_min, vmax=v_max, cmap=self.COLORMAP, extent=(0, fov[1], fov[0], 0))
        ax1.add_patch(Rectangle((0, 0), fov[1]-1, fov[0]-1, color='k', linestyle=':', linewidth=1.5, fill=False))
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("right", size="3%", pad=0.1)
        cb = plt.colorbar(im, cax=cax)
        cb.set_label("Distortion (%)")
        ax1.set_title("Local geometric distortion across FOV\n", fontweight='bold')
        ax1.set_axis_off()

        # Optionally save the figure
        if self.save_figures:
            fig_name = "Distortion map"
            plt.savefig(os.path.join(self.save_dir, fig_name))
