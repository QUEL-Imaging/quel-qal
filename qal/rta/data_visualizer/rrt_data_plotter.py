import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import profile_line

class RrtDataPlotter:
    def __init__(self):
        self.line_profile_fig_size = (12, 6)
        self.check_environment()

    def check_environment(self):
        try:
            from IPython import get_ipython
            get_ipython()
            if 'IPKernelApp' in get_ipython().config:
                # print("Running in Jupyter Notebook")
                get_ipython().magic('matplotlib widget')
                self.line_profile_fig_size = (10, 5)
                return "Jupyter Notebook"
            else:
                # print("Running in JupyterLab")
                self.line_profile_fig_size = (10, 5)
                return "JupyterLab"
        except AttributeError:
            # print("Running in a standard Python environment")
            return "Standard Python"

    def plot_percentage_contrast(self, df, resolution_threshold=26):
        """
        Plot the percentage contrast values for each element.
        
        Parameters:
        - df: DataFrame containing the percentage contrast data.
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.canvas.manager.set_window_title('Percentage Contrast Plot')
        fig.suptitle('Percentage Contrast Plot', fontsize=16)
        
        groups = df['Group'].unique()
        
        for group in groups:
            group_data = df[df['Group'] == group]
            x = group_data['Resolution (lp/mm)']
            y = group_data['Percentage Contrast']
            yerr = group_data['Percentage Contrast Std']

            # Replace NaN values with 0 in yerr
            yerr = np.nan_to_num(yerr, nan=0)

            ax.errorbar(x, y, yerr=yerr, fmt='o', capsize=4, capthick=1, ecolor='black', label=f'Group {group}')

            for i, txt in enumerate(group_data['Element']):
                ax.annotate(txt, (x.iloc[i], y.iloc[i]), xytext=(5, 5), textcoords='offset points')

        # Add a horizontal line at 26%
        ax.axhline(y=resolution_threshold, color='r', linestyle='--', linewidth=1)

        ax.set_xlabel('Resolution (lp/mm)')
        ax.set_ylabel('Percentage Contrast')
        ax.set_title('Percentage Contrast vs Resolution')
        ax.grid(True)
        ax.legend()
        
        plt.tight_layout()
        plt.show()
    
    def extract_line_profiles(self, im, x1, y1, x2, y2, offset, linewidth=4, order=0, mode="constant"):
        """
        Extracts line profiles from an image using the profile_line function
        from skimage.measure, given two points (x1, y1) and (x2, y2).
        """
        center_profile = profile_line(im, (y1, x1), (y2, x2), linewidth=linewidth, order=order, mode=mode)
        left_profile = profile_line(im, (y1, x1 - offset), (y2, x2 - offset), linewidth=linewidth, order=order, mode=mode)
        right_profile = profile_line(im, (y1, x1 + offset), (y2, x2 + offset), linewidth=linewidth, order=order, mode=mode)
        
        profiles = {
            'left': left_profile,
            'center': center_profile,
            'right': right_profile,
        }
        
        return profiles

    def plot_line_profiles(self, im, group_coordinates, percentage_contrast_df):
        num_groups = len(group_coordinates)
        fig, axs = plt.subplots(1, num_groups + 1, figsize=self.line_profile_fig_size)
        fig.canvas.manager.set_window_title('Line Profile Plots')
        fig.suptitle('Line Profile Plots', fontsize=16)

        im = im.astype(np.float32)

        axs[0].imshow(im, cmap='gray')
        axs[0].set_title('Input Image')
        axs[0].set_axis_off()

        for i, group_number in enumerate(group_coordinates.keys()):
            group_data = group_coordinates[group_number]
            (x1, y1), (x2, y2) = group_data['coordinates']  # Correctly unpacked

            elements = group_data['elements']
            profiles = self.extract_line_profiles(im, x1, y1, x2, y2, offset=4 if group_number < 2 else 1)

            group_df = percentage_contrast_df[percentage_contrast_df['Group'] == group_number]

            axs[0].plot([x1, x2], [y1, y2], linewidth=2, label=f'Group {group_number}')

            axs[i + 1].plot(profiles['center'], linewidth=1, color='blue', label='Profile')

            for _, row in group_df.iterrows():
                element_number = row['Element']
                peak_indices = row['Peak Indices']
                trough_indices = row['Trough Indices']

                if len(peak_indices) > 0 and len(trough_indices) > 0:
                    axs[i + 1].scatter(peak_indices, [profiles['center'][index] for index in peak_indices], color='red', label=f'Element {element_number} Peaks', marker='x')
                    axs[i + 1].scatter(trough_indices, [profiles['center'][index] for index in trough_indices], color='green', label=f'Element {element_number} Troughs', marker='o')

            axs[i + 1].set_title(f'Group {group_number}')
            axs[i + 1].set_xlabel('Pixel Position')
            axs[i + 1].set_ylabel('Intensity (a.u.)')
            axs[i + 1].set_xlim(0, len(profiles['center']) - 1)

        axs[0].legend()
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

