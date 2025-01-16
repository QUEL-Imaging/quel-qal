import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import statsmodels.api as sm
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit
from skimage.draw import disk
from matplotlib.patches import Patch


class WellPlotter:
    # Class-level constants defining default graph settings
    GRAPH_PARAMS = {
        'figure.figsize': [6,6],
        'figure.autolayout': True,
        'axes.grid': True,
        'axes.labelsize': 24,
        'axes.labelweight': 700,
        'xtick.labelsize': 'x-large',
        'ytick.labelsize':'x-large',
        'font.weight': 'bold',
        'font.size': 8,
        'legend.edgecolor': '#454545',
        'patch.linewidth': 3,
        'legend.fontsize': 'x-large'
    }
    FONT_SIZE = 20
    DIMENSIONS = (100, 100)
    MARKER_SIZE = 200

    def __init__(self, df, image=None):
        """
        Initialize the WellPlotter with a dataframe and optionally an image.
        
        :param df: DataFrame containing the data to be plotted.
        :param image: Optional input image for visualizing ROIs.
        """
        self.set_graph_params()
        self.df = df  # DataFrame to be plotted
        self.image = image  # Input image for ROI visualization

    def set_graph_params(self):
        """Update the global graph settings using the defined parameters."""
        plt.rcParams.update(self.GRAPH_PARAMS)

    def set_axis_scales(self, graph_type='concentration'):
        """Set both x and y axes to log scale if graph_type is 'concentration'."""
        if graph_type == 'concentration':
            self.ax.set_xscale('log')
            self.ax.set_yscale('log')
        else:
            self.ax.set_xscale('linear')
            self.ax.set_yscale('linear')

    def custom_formatter(self, x, _):
        if x < 0.01:
            return f"{x:.3f}"  # Display three decimal places for numbers < 0.01
        else:
            return f"{x:.2f}"  # Display two decimal places for all other numbers

    def set_formatter(self, x_set_scientific=True, y_set_scientific=False):
        """
        Configure the number format for both axes.
        
        :param x_set_scientific: Boolean indicating whether to use scientific notation for x-axis.
        :param y_set_scientific: Boolean indicating whether to use scientific notation for y-axis.
        """
        get_formatter = lambda sci: ticker.ScalarFormatter(useMathText=True) if sci else ticker.FuncFormatter(self.custom_formatter)
        
        self.ax.xaxis.set_major_formatter(get_formatter(x_set_scientific))
        self.ax.yaxis.set_major_formatter(get_formatter(y_set_scientific))
        
    def add_annotation(self):
        """Add a sample annotation to the graph."""
        self.ax.annotate('Test annotation', 
                         xy=(0.05, 0.95),  # Fractional coordinates for the axes
                         xycoords='axes fraction', 
                         ha='left', 
                         va='top')

    def rescale_y_axis(self, col_to_plot, graph_type='concentration'):
        """
        Rescale the y-axis based on the data and a scaling percentage.
        
        :param col_to_plot: Column name in the dataframe used for y-axis data.
        :param graph_type: Type of graph, either 'concentration' or 'depth'.
        """
        scale_percentage = 0.5 if graph_type == 'concentration' else 0.1
        self.ax.set_ylim(self.df[col_to_plot].nsmallest(2).iloc[-1]*(1-scale_percentage), 
                         self.df[col_to_plot].max() * (1+scale_percentage))


    def apply_plot_settings(self, graph_type, set_scientific=False):
        """
        Apply all predefined settings to the graph.

        :param graph_type: Type of graph, either 'concentration' or 'depth'.
        :param set_scientific: Boolean indicating whether to use scientific notation.
        """
        self.set_axis_scales(graph_type)
        self.set_formatter(y_set_scientific=False)

        if graph_type == 'depth':
            legend = self.ax.legend(loc='upper left', fontsize='x-large', bbox_to_anchor=(0.45, 0.98), markerscale=0.5)
        else:
            legend = self.ax.legend(loc='lower right', fontsize='x-large', markerscale=0.5)

    def power_law(self, x, a, b):
        return a * np.power(x, b)
    
    def exponential(self, x, a, b):
        return a * np.exp(b * x)

    def fit_with_scipy(self, fit_type, x, y, x_hat):
        if fit_type == "power":
            params, _ = curve_fit(self.power_law, x, y)
            fitted = self.power_law(x, *params)
            y_hat = self.power_law(x_hat, *params)
            equation = f"$y = {params[0]:.3f}x^{{{params[1]:.3f}}}$"
        elif fit_type == "exponential":
            params, _ = curve_fit(self.exponential, x, y)
            fitted = self.exponential(x, *params)
            y_hat = self.exponential(x_hat, *params)
            equation = f"$y = {params[0]:.3f}e^{{{params[1]:.3f}x}}$"
        r_squared = r2_score(y, fitted)
        return y_hat, equation, r_squared

    def fit_with_statsmodels(self, fit_type, x, y, x_hat):
        if fit_type == "power":
            x_with_const = sm.add_constant(np.log(x))
            model = sm.OLS(np.log(y), x_with_const).fit()
            a, b = np.exp(model.params[0]), model.params[1]
            fitted = self.power_law(x, a, b)
            y_hat = self.power_law(x_hat, a, b)
            equation = f"$y = {a:.3f}x^{{{b:.3f}}}$"  # <-- Fixed here
        elif fit_type == "exponential":
            x_with_const = sm.add_constant(x)
            model = sm.OLS(np.log(y), x_with_const).fit()
            a, b = np.exp(model.params[0]), model.params[1]
            fitted = self.exponential(x, a, b)
            y_hat = self.exponential(x_hat, a, b)
            equation = f"$y = {a:.3f}e^{{{b:.3f}x}}$"  # <-- Fixed here
        r_squared = model.rsquared
        return y_hat, equation, r_squared

    def fit_data(self, fit_type, x, y, x_hat, library="statsmodels"):
        if library == "scipy":
            return self.fit_with_scipy(fit_type, x, y, x_hat)
        elif library == "statsmodels":
            return self.fit_with_statsmodels(fit_type, x, y, x_hat)
        else:
            raise ValueError("Unknown library")

    def scatter_plot_data(self, df_plot, col_to_plot, fluorophore_label):
        """Plot scatter points for the data."""
        self.ax.scatter(df_plot['value'], df_plot[col_to_plot], color='#1B3D87', marker='D', s=self.MARKER_SIZE, label=f"{fluorophore_label} wells")

    def plot_error_bars(self, df_plot, col_to_plot):
        """Plot error bars for the data."""
        acceptable_cols = ['mean_intensity', 'mean intensity', 'mean intensity baselined', 'mean intensity normalized']
        if col_to_plot in acceptable_cols:
            y_errors = df_plot['standard deviation']
            if col_to_plot == 'mean intensity normalized':
                y_errors = df_plot['standard deviation normalized']
            self.ax.errorbar(df_plot['value'], df_plot[col_to_plot], yerr=y_errors, fmt='none', color='#1B3D87',
                             ecolor='red', capsize=5, label=f"Error Bars", markersize=10)
        else:
            print(f"Error bars unavailable for column '{col_to_plot}'")

    def add_trendline_and_annotation(self, fit_type, x_data, y_data, x_hat, library, graph_type):
        """Fit data, plot trendline, and add annotations."""
        fitted, equation, r_squared = self.fit_data(fit_type, x_data, y_data, x_hat, library=library)
        self.ax.plot(x_hat, fitted, 'g--', label="Trendline")
        
        if graph_type == 'depth':
            annotation_text = f"{equation}"
            annotation_xy = (0.2, 0.7)
        else:
            annotation_text = f"{equation}\n$R^2 = {r_squared:.2f}$"
            annotation_xy = (0.05, 0.84)

        self.ax.annotate(annotation_text, 
                        xy=annotation_xy,  
                        xycoords='axes fraction',
                        fontsize=self.FONT_SIZE,
                        color="black")

    def add_depth_detection_limit(self, fit_type, x_data, y_data, x_hat, library):
        fitted, equation, r_squared = self.fit_data(fit_type, x_data, y_data, x_hat, library=library)
        depth_lim_idx = np.argmin(np.abs(fitted - 3))  # Index where CNR = 3
        depth_lim = x_hat[depth_lim_idx]
        if (y_data > 3).all():
            print("\nCNR is above 3 for all data.")
        elif depth_lim == x_data[-1]:
            print("\nCNR is above 3 for all points on the fitted curve.")
        else:
            print(f"\nCNR falls below 3 at {depth_lim:.1f} mm on the fitted curve.")
            if depth_lim > 0.75 * np.max(x_data):
                annotation_x_loc = (depth_lim / np.max(x_data)) - 0.25
            else:
                annotation_x_loc = (depth_lim / np.max(x_data)) - 0.02
            y_lim = self.ax.get_ylim()
            self.ax.axvline(x=depth_lim, ymin=y_lim[0], ymax=y_lim[1], color='#454545', linewidth=2, linestyle=':')
            self.ax.annotate("(CNR = 3)",
                             xy=(annotation_x_loc, 0.2),
                             xycoords='axes fraction',
                             fontsize=12,
                             color='#454545')

    def set_axis_titles(self, x_axis_title, y_axis_title='Intensity (a.u.)'):
        """Set titles for x and y axes."""
        self.ax.set_xlabel(x_axis_title)
        self.ax.set_ylabel(y_axis_title)

    def plot(self, graph_type='concentration', col_to_plot='mean intensity', fluorophore_label='ICG-eq', plot_error_bars=False, trendline_lib='statsmodels', save_plot=None):
        """
        Plot the graph using the provided or default column and labels.
        
        :param graph_type: Type of graph, either 'concentration' or 'depth'.
        :param col_to_plot: Column name in the dataframe used for y-axis data.
        :param fluorophore_label: Label used for the scatter plot data.
        """
        self.fig, self.ax = plt.subplots(figsize=(6, 6))  # Create a new figure and axes

        if graph_type == 'concentration':
            x_axis_title = 'Concentration (nM)'
            fit_type = "power"
        elif graph_type == 'depth':
            x_axis_title = 'Depth (mm)'
            fit_type = "exponential"
        else:
            raise ValueError("Unknown graph_type. Please select either 'concentration' or 'depth'.")

        if col_to_plot == 'mean intensity':
            y_axis_title = 'Intensity (a.u.)'
        elif col_to_plot == 'mean intensity baselined':
            y_axis_title = 'Baselined Intensity (a.u.)'
        elif col_to_plot == 'mean intensity normalized':
            y_axis_title = 'Normalized Intensity (a.u.)'
        else:
            y_axis_title = f"{col_to_plot}"

        self.set_axis_titles(x_axis_title, y_axis_title)

        df_plot = self.df[self.df['well'] != 'Control']
        self.scatter_plot_data(df_plot, col_to_plot, fluorophore_label)

        if plot_error_bars:
            self.plot_error_bars(df_plot, col_to_plot)

        x_data = df_plot['value'].values
        y_data = df_plot[col_to_plot].values
        cnr_data = df_plot['CNR'].values
        if graph_type == 'concentration':
            linear_min_idx = np.argmin(np.abs(cnr_data - 3))
            if cnr_data[linear_min_idx] < 3:
                linear_min_idx -= 1
            x_hat = np.geomspace(x_data[linear_min_idx], np.max(x_data), 100)
        else:
            x_hat = np.linspace(np.min(x_data), np.max(x_data), 100)
        self.add_trendline_and_annotation(fit_type, x_data, y_data, x_hat, library=trendline_lib, graph_type=graph_type)
        if graph_type == 'depth':
            self.add_depth_detection_limit(fit_type, x_data, cnr_data, x_hat, library=trendline_lib)

        self.apply_plot_settings(graph_type)

        if save_plot is not None:
            plt.savefig(os.path.abspath(save_plot))
        plt.show()

    def visualize_roi(self, colormap='plasma'):
        """
        Visualize the ROI of the detected wells on the input image with distinct colors and a legend.
        
        :param colormap: The colormap to use for visualizing ROIs (default is 'plasma').
        """
        if self.image is None:
            raise ValueError("No image provided for ROI visualization.")
        
        # Normalize the image to 8-bit for visualization
        if self.image.dtype == np.uint8:
            vis_image = self.image  # 8-bit images don't need conversion
        elif self.image.dtype == np.uint16:
            vis_image = self.image.astype(np.float32)
            vis_image = (vis_image - vis_image.min()) / (vis_image.max() - vis_image.min())  # Normalize to [0, 1]
            vis_image = (vis_image * 255).astype(np.uint8)  # Convert to 8-bit
        else:
            # General fallback for non-standard image types
            vis_image = self.image.astype(np.float32)
            vis_image = (vis_image - np.nanmin(vis_image)) / (np.nanmax(vis_image) - np.nanmin(vis_image))  # Normalize to [0, 1]
            vis_image = (vis_image * 255).astype(np.uint8)  # Convert to 8-bit

        # Assign distinct colors to each well
        unique_wells = self.df['well'].unique()
        cmap = plt.get_cmap(colormap)  # Dynamically get the colormap
        colors = cmap(np.linspace(0, 1, len(unique_wells)))
        color_map = {well: color for well, color in zip(unique_wells, colors)}

        # Overlay ROIs with different colors and add crosshairs
        fig, ax = plt.subplots(figsize=(8, 8))  # Explicit figure and axes
        ax.imshow(vis_image, cmap='gray')
        legend_patches = []
        for _, row in self.df.iterrows():
            rr, cc = disk(
                (int(row['y']), int(row['x'])), int(row['Analyzed ROI Diameter'] / 2), shape=vis_image.shape
            )
            color = color_map[row['well']]
            
            # Draw circle
            ax.add_patch(
                plt.Circle((row['x'], row['y']), row['Analyzed ROI Diameter'] / 2, edgecolor=color, facecolor='none', lw=2)
            )
            
            # Draw crosshairs
            ax.plot(
                [row['x'], row['x']], [row['y'] - 10, row['y'] + 10],  # Vertical line
                color=color, lw=1
            )
            ax.plot(
                [row['x'] - 10, row['x'] + 10], [row['y'], row['y']],  # Horizontal line
                color=color, lw=1
            )

            # Add legend if not already present
            if row['well'] not in [patch.get_label() for patch in legend_patches]:
                legend_patches.append(Patch(color=color, label=row['well']))

        # Add legend outside the image
        ax.legend(
            handles=legend_patches,
            loc='upper left',  # Position relative to bbox_to_anchor
            bbox_to_anchor=(1.05, 1),  # Position outside the plot area
            title="Well IDs",
            fontsize=12
        )
        ax.axis("off")
        plt.title(f"ROI Visualization", fontsize=18, fontweight='bold')  # Larger, bold title
        plt.tight_layout()  # Adjust layout to prevent clipping
        plt.show()
