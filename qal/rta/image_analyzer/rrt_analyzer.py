import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from skimage.measure import profile_line


class RrtAnalyzer:
    def __init__(self):
        pass

    def calculate_percentage_contrast(self, peak_height, trough_height):
        """Calculate the percentage contrast given the peak and trough heights."""
        return (peak_height - trough_height) / (peak_height + trough_height) * 100

    def calculate_resolution(self, group_number, element_number):
        """
        Calculates the resolution in line pairs per millimeter (lp/mm) given the group and element numbers,
        using the formula: 2^(Group Number + (Element Number - 1) / 6).
        
        Parameters:
        - group_number: The group number in the resolution chart.
        - element_number: The element number within the group.
        
        Returns:
        - Corrected resolution in line pairs per millimeter (lp/mm).
        """
        resolution = 2 ** (group_number + (element_number - 1) / 6)
        return resolution

    def calculate_line_width_microns(self, group_number, element_number):
        # Updated mapping for groups 0 to 5 based on the given table
        line_widths = {
            0: [500.00, 445.45, 396.85, 353.55, 314.98, 280.62],
            1: [250.00, 222.72, 198.43, 176.78, 157.49, 140.31],
            2: [125.00, 111.36, 99.21, 88.39, 78.75, 70.15],
            3: [62.50, 55.68, 49.61, 44.19, 39.37, 35.08],
            4: [31.25, 27.84, 24.80, 22.10, 19.69, 17.54],
            5: [15.63, 13.92, 12.40, 11.05, 9.84, 8.77],
        }

        # Validate group and element numbers
        if group_number in line_widths and 1 <= element_number <= 6:
            return line_widths[group_number][element_number - 1]
        else:
            raise ValueError("Invalid group number or element number.")

    def detect_peaks_and_troughs(self, data):
        gradient = np.diff(data)
        peaks = []
        troughs = []
        for i in range(1, len(gradient)):
            if gradient[i] < 0 and gradient[i-1] > 0:
                peaks.append(i)
            elif gradient[i] > 0 and gradient[i-1] < 0:
                troughs.append(i)
        return peaks, troughs

    def filter_peaks_and_troughs(self, peaks, troughs, profile):
        """Filter peaks and troughs based on their heights."""
        peaks_heights = np.array([profile[index] for index in peaks])
        troughs_heights = np.array([profile[index] for index in troughs])
        
        flag_peaks, flag_troughs = self.flag_bad_peak_trough(peaks_heights, troughs_heights)
        
        # Filter out flagged peaks and troughs
        filtered_peaks = np.array(peaks)[~flag_peaks]
        filtered_troughs = np.array(troughs)[~flag_troughs]
        
        return filtered_peaks, filtered_troughs

    def flag_bad_peak_trough(self, peaks, troughs):
        """
        Identify peaks and troughs that are unusually low or high based on median values.
        
        Parameters:
        - peaks: An array of peak heights.
        - troughs: An array of trough heights.
        
        Returns:
        - A tuple of two arrays:
          - flag_peaks: Boolean array where True indicates a peak is considered unusually low.
          - flag_troughs: Boolean array where True indicates a trough is considered unusually high.
        """
        # Calculate the median heights of peaks and troughs
        median_peaks, median_troughs = np.nanmedian(peaks), np.nanmedian(troughs)
        
        # Determine the difference between the median heights
        diff_median = median_peaks - median_troughs
        
        # Flag peaks and troughs based on their deviation from the median values
        flag_peaks = peaks < (median_peaks - 0.5 * diff_median)
        flag_troughs = troughs > (median_troughs + 0.5 * diff_median)
        
        return flag_peaks, flag_troughs

    def extract_line_profiles(self, image, x1, y1, x2, y2, offset, linewidth=4, order=0, mode="constant"):
        """
        Extracts line profiles from an image using the profile_line function
        from skimage.measure, given two points (x1, y1) and (x2, y2).
        """
        center_profile = profile_line(image, (y1, x1), (y2, x2), linewidth=linewidth, order=order, mode=mode)
        left_profile = profile_line(image, (y1, x1 - offset), (y2, x2 - offset), linewidth=linewidth, order=order, mode=mode)
        right_profile = profile_line(image, (y1, x1 + offset), (y2, x2 + offset), linewidth=linewidth, order=order, mode=mode)
        
        profiles = {
            'left': left_profile,
            'center': center_profile,
            'right': right_profile,
            'left_original': left_profile,
            'center_original': center_profile,
            'right_original': right_profile,
        }
        
        return profiles

    def process_group(self, image, x1, y1, x2, y2, group_number, elements, window_length=5, polyorder=3):
        offset = 1 if group_number >= 2 else 1
        profiles = self.extract_line_profiles(image, x1, y1, x2, y2, offset)
        peaks_troughs = {}

        for position in ['left', 'center', 'right']:
            profiles[position] = savgol_filter(profiles[position], window_length=window_length, polyorder=polyorder)
            peaks, troughs = self.detect_peaks_and_troughs(profiles[position])
            filtered_peaks, filtered_troughs = self.filter_peaks_and_troughs(peaks, troughs, profiles[position])
            peaks_troughs[position] = {'peaks': filtered_peaks.tolist(), 'troughs': filtered_troughs.tolist()}

        percentage_contrasts = []
        resolutions = []
        
        peak_indices = []
        trough_indices = []
        
        for element_number in elements:
            element_peaks = peaks_troughs['center']['peaks'][(element_number - elements[0]) * 3 : (element_number - elements[0] + 1) * 3]
            element_troughs = peaks_troughs['center']['troughs'][(element_number - elements[0]) * 3 : (element_number - elements[0] + 1) * 3]
            
            peak_indices.append(element_peaks)
            trough_indices.append(element_troughs)

            element_percentage_contrasts = []
            for peak, trough in zip(element_peaks, element_troughs):
                peak_height = np.average([profiles['left_original'][peak], profiles['center_original'][peak], profiles['right_original'][peak]])
                trough_height = np.average([profiles['left_original'][trough], profiles['center_original'][trough], profiles['right_original'][trough]])
                percentage_contrast = self.calculate_percentage_contrast(peak_height, trough_height)
                element_percentage_contrasts.append(percentage_contrast)
            
            percentage_contrasts.append(element_percentage_contrasts)
            resolution = self.calculate_resolution(group_number, element_number)
            resolutions.append(resolution)
        
        # Create a DataFrame to store the percentage contrast data
        data = {
            'Group': [group_number] * len(elements),
            'Element': [element_number for element_number in elements],
            'Resolution (lp/mm)': resolutions,
            'Line Width (microns)': [self.calculate_line_width_microns(group_number, element_number) for element_number in elements],
            'Percentage Contrast': [np.mean(contrasts) if len(contrasts) > 0 else np.nan for contrasts in percentage_contrasts],
            'Percentage Contrast Std': [np.std(contrasts) if len(contrasts) > 0 else np.nan for contrasts in percentage_contrasts],
            'Peak Indices': peak_indices,
            'Trough Indices': trough_indices
        }
        df = pd.DataFrame(data)
        
        return df

    def load_and_process_groups(self, im, group_coordinates):
        percentage_contrast_df = pd.DataFrame()

        im = im.astype(np.float32)

        for group_number, group_data in group_coordinates.items():
            (x1, y1), (x2, y2) = group_data['coordinates']  # Unpack the four coordinates
            elements = group_data['elements']
            group_df = self.process_group(im, x1, y1, x2, y2, group_number, elements, window_length=1, polyorder=0)
            percentage_contrast_df = pd.concat([percentage_contrast_df, group_df])

        percentage_contrast_df.reset_index(drop=True, inplace=True)
        return percentage_contrast_df
