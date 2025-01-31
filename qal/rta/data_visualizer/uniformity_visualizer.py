import os
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable


class UniformityVisualizer:

    # Fixed attributes for figures generated
    IM_COLORMAP = 'inferno'
    BKG_COLORMAP = 'gray'
    ISO_COLORMAP = 'rainbow'
    LINE_COLORS_MAP = 'jet'
    LINE_STYLES = ['-.', '--', ':']
    DOT_COLOR = 'green'
    SURF_ALPHA = 0.5
    ANI_FRAMES = 120
    ANI_INTERVAL = 50
    ANI_FPS = 5
    ANI_BITRATE = 1800
    TITLE_FONTSIZE = 14
    GRAPH_PARAMS = {
        'figure.figsize': [9, 6],
        'font.size': 10,
        'axes.labelsize': 18,
        'axes.labelweight': 500,
    }

    def __init__(self, params=None):
        """
        Initialize UniformityVisualizer with optional parameters for class methods.

        :param params:      Parameters for class methods (optional)
        """

        self.environment = self.check_environment()

        # Default parameters
        if params is None:
            self.params = {
                "n horizontal profiles": 3,
                "n vertical profiles": 3,
                "Iso-levels": [0.6, 0.8, 0.9, 0.95],
                "Show figures": False
            }
            self.params["Alpha scaling"] = 0.8 * np.ones(len(self.params["Iso-levels"]))
        else:
            self.params = params

        # Initialize class attributes
        self.n_horizontal = self.params["n horizontal profiles"]
        self.n_vertical = self.params["n vertical profiles"]
        self.levels = self.params["Iso-levels"]
        self.alpha_mults = self.params["Alpha scaling"]
        self.show_figs = self.params["Show figures"]
        assert len(self.levels) == len(self.alpha_mults), "Number of iso-levels and alphas should be the same"

        self.analyzer_outputs = None
        self.save_figures = False
        self.im_dir = None
        self.save_dir = None

        plt.rcParams.update(self.GRAPH_PARAMS)

    def visualize_fluorescence_profiles(self, uniformity_analyzer_outputs):
        """
        Generate figures displaying fluorescence uniformity from the input data.

        :param uniformity_analyzer_outputs:      Fluorescence uniformity profile data
        """

        self.analyzer_outputs = uniformity_analyzer_outputs
        if uniformity_analyzer_outputs is None:
            return

        print("\nGENERATING FIGURES...")

        surf_rep = uniformity_analyzer_outputs["surf_rep"]
        norm_surf_rep = surf_rep / np.max(surf_rep)
        xq = uniformity_analyzer_outputs["xq"]
        yq = uniformity_analyzer_outputs["yq"]
        dots = uniformity_analyzer_outputs["dots"]
        fov = uniformity_analyzer_outputs["fov"]
        im_dir = uniformity_analyzer_outputs["image_dir"]
        self.save_figures = uniformity_analyzer_outputs["save_output"]
        if self.save_figures:
            self.save_dir = os.path.join(im_dir, "Surface Representation")

        # Display the surface fit
        fit_in_fov = self.display_surf_rep(surf_rep, xq, yq, dots, fov)

        # Display line profiles
        self.display_line_profiles(fit_in_fov, norm_surf_rep, xq, yq, fov)

        # Display iso-maps
        self.display_iso_maps(fit_in_fov, fov)

        # Restore default matplotlib graph settings
        plt.rcdefaults()
        print("  Done")

    def display_surf_rep(self, surf_rep, xq, yq, dots, fov):
        """
        Show the surface fit in 2D and 3D form.

        :param surf_rep:        Surface fit
        :param xq:              x-axis points for which SURF_REP was generated
        :param yq:              y-axis points for which SURF_REP was generated
        :param dots:            Original x, y, z data from RUD target images
        :param fov:             Camera field of view
        :return:                The surface fit shown within the field of view
        """

        xq_min = np.min(xq)
        xq_max = np.max(xq)
        yq_min = np.min(yq)
        yq_max = np.max(yq)

        x = dots["x"].to_numpy()
        y = dots["y"].to_numpy()
        z = dots["dot_intensity"].to_numpy()

        xqm, yqm = np.meshgrid(xq, yq)

        if xq[0] > 0:
            xq_spacing = xq[1] - xq[0]
            yq_spacing = yq[1] - yq[0]
            x_fov = np.arange(0, fov[1], xq_spacing)
            y_fov = np.arange(0, fov[0], yq_spacing)
            fit_in_fov = np.zeros((len(y_fov), len(x_fov)))
            xq_start = np.argmin(np.abs(x_fov - xq_min))
            xq_end = np.argmin(np.abs(x_fov - xq_max))
            yq_start = np.argmin(np.abs(y_fov - yq_min))
            yq_end = np.argmin(np.abs(y_fov - yq_max))
            fit_in_fov[yq_start:yq_end + 1, xq_start:xq_end + 1] = surf_rep
        else:
            fit_in_fov = surf_rep.copy()
        fit_in_fov /= np.max(fit_in_fov)

        if self.environment == "Standard Python":
            anim = self.make_animation(self.analyzer_outputs)

        fig1, ax1 = plt.subplots()
        im = ax1.imshow(fit_in_fov, cmap=self.IM_COLORMAP, vmin=0, vmax=1, extent=(0, fov[1], fov[0], 0))
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("right", size="3%", pad=0.1)
        cb = plt.colorbar(im, cax=cax)
        cb.set_label("Normalized Intensity")
        ax1.set_title("Calculated fluorescence uniformity\n", fontsize=self.TITLE_FONTSIZE, fontweight='bold')
        ax1.set_axis_off()

        if self.save_figures:
            fig_name = "Fluorescence uniformity"
            plt.savefig(os.path.join(self.save_dir, fig_name))

        if self.show_figs:
            plt.show()

        return fit_in_fov

    def make_animation(self, uniformity_analyzer_outputs):
        """
        Create animated 3D plot of input data points and fitted surface.

        :param uniformity_analyzer_outputs:      Fluorescence uniformity profile data
        """

        if "Jupyter" in self.environment:
            print("\nGENERATING ANIMATION...")

        surf_rep = uniformity_analyzer_outputs["surf_rep"]
        xq = uniformity_analyzer_outputs["xq"]
        yq = uniformity_analyzer_outputs["yq"]
        dots = uniformity_analyzer_outputs["dots"]
        im_dir = uniformity_analyzer_outputs["image_dir"]
        self.save_figures = uniformity_analyzer_outputs["save_output"]
        if self.save_figures:
            self.save_dir = os.path.join(im_dir, "Surface Representation")
        x = dots["x"].to_numpy()
        y = dots["y"].to_numpy()
        z = dots["dot_intensity"].to_numpy()
        xqm, yqm = np.meshgrid(xq, yq)

        fig1, ax1 = plt.subplots(subplot_kw={"projection": "3d"})
        ax1.plot_surface(xqm, yqm, surf_rep, cmap=self.IM_COLORMAP, linewidth=0, alpha=self.SURF_ALPHA)
        ax1.scatter(x, y, z, color=self.DOT_COLOR, label='Data')
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_zlabel("Normalized Intensity")
        ax1.legend()

        def animate(i):
            angle = i * 3
            ax1.view_init(10, angle)
            return ax1,

        ani = animation.FuncAnimation(fig1, animate, repeat=True, frames=self.ANI_FRAMES, interval=self.ANI_INTERVAL)
        writer = animation.PillowWriter(fps=self.ANI_FPS, metadata=dict(artist='Me'), bitrate=self.ANI_BITRATE)

        if self.save_figures:
            fig_name = "Fit_animation.gif"
            ani.save(os.path.join(self.save_dir, fig_name), writer=writer)

        if self.show_figs and "Jupyter" in self.environment:
            plt.show()

        return ani

    def display_line_profiles(self, fit_in_fov, surf_rep, xq, yq, fov):
        """
        Display line profiles across the fluorescence uniformity map.

        :param fit_in_fov:      The surface fit shown within the field of view
        :param surf_rep:        The surface fit
        :param xq:              x-axis points for which SURF_REP was generated
        :param yq:              y-axis points for which SURF_REP was generated
        :param fov:             Camera field of view
        """

        vert_lines = np.linspace(0, 1, self.n_vertical + 2)[1:-1] * (xq[-1] - xq[0]) + xq[0]
        hor_lines = np.linspace(0, 1, self.n_horizontal + 2)[1:-1] * (yq[-1] - yq[0]) + yq[0]

        vert_line_i = [np.argmin(np.abs(xq - vert_line)) for vert_line in vert_lines]
        hor_line_i = [np.argmin(np.abs(yq - hor_line)) for hor_line in hor_lines]

        vert_profiles = [surf_rep[:, v_line_i] for v_line_i in vert_line_i]
        hor_profiles = [surf_rep[h_line_i, :] for h_line_i in hor_line_i]
        for vert_profile in vert_profiles:
            vert_profile[vert_profile == 0] = np.nan
        for hor_profile in hor_profiles:
            hor_profile[hor_profile == 0] = np.nan

        vcl = np.linspace(0, 255, self.n_vertical).astype(np.int64)
        hcl = np.linspace(0, 255, self.n_horizontal).astype(np.int64)
        v_colors = self.get_colors(vcl, self.LINE_COLORS_MAP)
        h_colors = self.get_colors(hcl, self.LINE_COLORS_MAP)

        fig1, ax1 = plt.subplots()
        im = ax1.imshow(fit_in_fov, cmap=self.IM_COLORMAP, vmin=0, vmax=1, extent=(0, fov[1], fov[0], 0))
        for n, x_loc in enumerate(vert_lines):
            l_style = self.LINE_STYLES[n % len(vert_lines)]
            ax1.plot([x_loc, x_loc], [0, fov[0]], color=v_colors[n], linestyle=l_style, linewidth=2)
        for n, y_loc in enumerate(hor_lines):
            l_style = self.LINE_STYLES[n % len(hor_lines)]
            ax1.plot([0, fov[1]], [y_loc, y_loc], color=h_colors[n], linestyle=l_style, linewidth=2)
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("right", size="3%", pad=0.1)
        cb = plt.colorbar(im, cax=cax)
        cb.set_label("Normalized Intensity")
        ax1.set_title("Calculated fluorescence uniformity\n", fontsize=self.TITLE_FONTSIZE, fontweight='bold')
        ax1.set_axis_off()
        if self.show_figs and "Jupyter" in self.environment:
            plt.show()

        if self.save_figures:
            fig_name = "Fluorescence uniformity - line profiles"
            plt.savefig(os.path.join(self.save_dir, fig_name))

        fig2, ax2 = plt.subplots()
        for n in range(len(hor_profiles)):
            l_style = self.LINE_STYLES[n % len(hor_lines)]
            ax2.plot(xq, hor_profiles[n], color=h_colors[n], linestyle=l_style, linewidth=3)
        ax2.set_ylim([0, 1.03])
        ax2.set_xlabel('Horizontal pixel count')
        ax2.set_ylabel('Normalized intensity')
        ax2.set_title('Horizontal Profiles\n', fontsize=self.TITLE_FONTSIZE, fontweight='bold')
        if self.show_figs and "Jupyter" in self.environment:
            plt.show()

        if self.save_figures:
            fig_name = "Horizontal profiles"
            plt.savefig(os.path.join(self.save_dir, fig_name))

        fig3, ax3 = plt.subplots()
        for n in range(len(vert_profiles)):
            l_style = self.LINE_STYLES[n % len(vert_lines)]
            ax3.plot(yq, vert_profiles[n], color=v_colors[n], linestyle=l_style, linewidth=3)
        ax3.set_ylim([0, 1.03])
        ax3.set_xlabel('Vertical pixel count')
        ax3.set_ylabel('Normalized intensity')
        ax3.set_title('Vertical Profiles\n', fontsize=self.TITLE_FONTSIZE, fontweight='bold')

        if self.save_figures:
            fig_name = "Vertical profiles"
            plt.savefig(os.path.join(self.save_dir, fig_name))

        if self.show_figs:
            plt.show()

    def display_iso_maps(self, fit_in_fov, fov):
        """
        Display regions within specified percentages of the surface representation maximum.

        :param fit_in_fov:      The surface fit shown within the field of view
        :param fov:             Camera field of view
        """

        masks = []
        alphas = []
        for n, (level, a_mult) in enumerate(zip(self.levels, self.alpha_mults)):
            masks.append(np.array(fit_in_fov >= level))
            alphas.append(masks[n] * a_mult)

        color_levels = np.linspace(0, 255, len(self.levels)).astype(np.int64)
        colors = self.get_colors(color_levels, self.ISO_COLORMAP)

        fig1, ax1 = plt.subplots()
        ax1.imshow(fit_in_fov, cmap=self.BKG_COLORMAP, vmin=0, vmax=1, extent=(0, fov[1], fov[0], 0))
        for n, mask in enumerate(masks):
            mask = mask * color_levels[n] / 255
            ax1.imshow(mask, cmap=self.ISO_COLORMAP, vmin=0, vmax=1, alpha=alphas[n], extent=(0, fov[1], fov[0], 0))
            ax1.add_patch(Rectangle((-3-n, -3), width=1, height=1, color=colors[n],
                                    label=f"{self.levels[n] * 100}% of max"))
        ax1.legend(loc='lower left', fontsize=16)
        ax1.set_axis_off()
        ax1.set_title("Fluorescence uniformity regions\n", fontsize=self.TITLE_FONTSIZE, fontweight='bold')
        if self.show_figs and "Jupyter" in self.environment:
            plt.show()

        if self.save_figures:
            fig_name = "Iso-maps"
            plt.savefig(os.path.join(self.save_dir, fig_name))

        # Find fraction of the field of view that is covered by each iso-map, then save that data in a table
        fov_frac = []
        for mask in masks:
            fov_frac.append(np.sum(mask) / np.prod(fit_in_fov.shape))

        iso_dict = {
            "Percentage of max intensity (%)": np.array(self.levels) * 100,
            "Fraction of field of view": np.array(fov_frac)
        }
        iso_df = pd.DataFrame(iso_dict)
        iso_df["Fraction of field of view"] = iso_df["Fraction of field of view"].map('{:,.3f}'.format)

        fig2, ax2 = plt.subplots(figsize=(7, 3))
        ax2.table(cellText=iso_df.values, colLabels=iso_df.keys(), loc='center', cellLoc='center')
        ax2.axis('off')

        if self.save_figures:
            fig_name = "Iso-maps table"
            plt.savefig(os.path.join(self.save_dir, fig_name))

        if self.show_figs:
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
        self.n_horizontal = self.params["n horizontal profiles"]
        self.n_vertical = self.params["n vertical profiles"]
        self.levels = self.params["Iso-levels"]
        self.alpha_mults = self.params["Alpha scaling"]
        self.show_figs = self.params["Show figures"]
        assert len(self.levels) == len(self.alpha_mults), "Number of iso-levels and alphas should be the same"

    @staticmethod
    def check_environment():
        """
        Return the Python environment type.
        """
        try:
            from IPython import get_ipython
            get_ipython()
            if 'IPKernelApp' in get_ipython().config:
                get_ipython().run_line_magic('matplotlib', 'ipympl')
                return "Jupyter Notebook"
            else:
                get_ipython().run_line_magic('matplotlib', 'ipympl')
                return "JupyterLab"
        except AttributeError:
            return "Standard Python"

    @staticmethod
    def get_colors(cmap_levels, cmap):
        """
        Obtain rgba color values for the input color map at the level(s) requested.

        :param cmap_levels:     List/ndarray of levels (0 - 255) at which to obtain colors
        :param cmap:            Colormap to obtain colors from
        :return:                Color(s) in rgba format
        """

        cmap = matplotlib.colormaps[cmap]

        colors = []
        for level in cmap_levels:
            colors.append(cmap(level))

        if len(colors) == 1:
            colors = colors[0]

        return colors
