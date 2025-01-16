import numpy as np
from scipy.signal import find_peaks, peak_widths, savgol_filter


class DepthAnalyzer:

    def __init__(self, cropper):
        """
        Initialize DepthAnalyzer with a PhantomCropper object.

        :param cropper:     Instance of PhantomCropper containing data to be analyzed
        """

        self.img = cropper.img
        self.borders = cropper.borders
        self.outputs = {}
        self.depth_start_end = [1, 6]
        self.descent_start_end = [7.5, 37.5]
        self.phantom_dimensions = [30, 45]
        self.smoothing_params = {
            "Window length": 30,
            "Polynomial order": 3
        }

    def get_profiles(self, depths=None):
        """
        Analyze the image from CROPPER to produce an intensity profile and spread profile.

        :param depths:              Depths for which to evaluate intensity spread
        :return:
        """

        if depths is None:
            depths = np.arange(self.depth_start_end[0], self.depth_start_end[1] + 1)    # Default depths
        else:
            depths = np.array(depths)
            if np.min(depths) < self.depth_start_end[0]:
                print(f"Invalid entry {np.min(depths)} mm in DEPTHS")
                return
            elif np.max(depths) > self.depth_start_end[1]:
                print(f"Invalid entry {np.max(depths)} mm in DEPTHS")
                return

        img = self.img
        top = self.borders["top"]
        bottom = self.borders["bottom"]
        left = self.borders["left"]
        right = self.borders["right"]

        # Obtain region of image to be analyzed
        cropped = img[int(top):int(bottom), int(left):int(right)]
        if (bottom - top) > (right - left):     # Check whether phantom orientation is vertical
            cropped = cropped.T

        # Check if phantom is oriented such that it has higher intensity on the right. If so, flip
        average_row_intensity = np.mean(cropped, axis=0)
        if np.argmax(average_row_intensity) > (cropped.shape[1] - np.argmax(average_row_intensity)):
            cropped = np.fliplr(cropped)

        # Estimate the background intensity of the phantom
        self.outputs["Background"] = np.mean(
            cropped[int(0.05 * cropped.shape[0]):int(0.1 * cropped.shape[0]),
                    int(0.9 * cropped.shape[1]):int(0.95 * cropped.shape[1])]
        )

        # Get the intensity profile and peak intensity along the channel
        average_col_intensity = np.mean(cropped, axis=1)
        max_row_index = np.argmax(average_col_intensity)
        intensity_profile = cropped[max_row_index, :]
        peak_max = np.max(intensity_profile)

        # Smooth the intensity profile
        intensity_profile_smoothed = self.smooth_data(intensity_profile)

        # Calculate distance in mm along channel and isolate region where depth goes from 1 to 6 mm. Also calculate
        # distance in axis perpendicular to channel
        distance_mm = (np.arange(len(intensity_profile)) / (len(intensity_profile) - 1)) * self.phantom_dimensions[1]
        mask = (distance_mm >= self.descent_start_end[0]) & (distance_mm <= self.descent_start_end[1])
        descent_distance = distance_mm[mask]
        descent_depth = np.linspace(self.depth_start_end[0], self.depth_start_end[1], len(descent_distance))
        descent_intensities = intensity_profile[mask].astype(np.float64)
        descent_intensities_smoothed = intensity_profile_smoothed[mask]
        vert_distance = ((np.arange(cropped.shape[0]) - max_row_index) / cropped.shape[0]) * self.phantom_dimensions[0]

        # Store some outputs
        self.outputs["Depths"] = depths
        self.outputs["Peak max"] = peak_max
        self.outputs["Intensity profile"] = intensity_profile
        self.outputs["Smoothed intensity profile"] = intensity_profile_smoothed
        self.outputs["Distance in mm"] = distance_mm
        self.outputs["Descent distance"] = descent_distance
        self.outputs["Descent depth"] = descent_depth
        self.outputs["Intensities along descent"] = descent_intensities
        self.outputs["Smoothed intensities along descent"] = descent_intensities_smoothed
        self.outputs["Vertical distance"] = vert_distance

        # Evaluate the intensity spread and store in SELF.OUTPUTS
        self.outputs["Spreads"] = self.get_spread_at_depths(cropped, depths)

    def get_spread_at_depths(self, cropped, depths):
        """
        Calculate the intensity spread in the direction perpendicular to the channel.

        :param cropped:         Cropped image to use in evaluating spread
        :param depths:          Depths in mm at which to evaluate intensity spread
        :return:                Dictionary containing spread profiles, FWHM and AUC
        """

        # Find indices corresponding to requested depths
        dist_at_depth = (((depths - self.depth_start_end[0]) / (self.depth_start_end[1] - self.depth_start_end[0])) *
                         (self.descent_start_end[1] - self.descent_start_end[0]) + self.descent_start_end[0])
        depths_idx = ((dist_at_depth / self.phantom_dimensions[1]) * cropped.shape[1]).astype(np.int64)

        # Extract the spread profiles, calculate full-width-half-max and area under the curve
        spreads = {}
        for i, depth in enumerate(depths):
            label = f"{depth} mm depth"
            info = {
                "Spread profile":           cropped[:, depths_idx[i]],
                "Smoothed spread profile":  self.smooth_data(cropped[:, depths_idx[i]]),
                "FWHM (smoothed)":          self.calculate_fwhm(cropped[:, depths_idx[i]], smooth=True),
                "AUC (smoothed)":           self.calculate_auc(cropped[:, depths_idx[i]], smooth=True),
                "FWHM":                     self.calculate_fwhm(cropped[:, depths_idx[i]], smooth=False),
                "AUC":                      self.calculate_auc(cropped[:, depths_idx[i]], smooth=False)
            }
            spreads[label] = info

        return spreads

    def calculate_fwhm(self, curve, smooth=True):
        """
        Calculate the full-width-half-max for the input 1D profile.

        :param curve:       Input for which to calculate full-width-half-max
        :param smooth:      Whether to first smooth the input data
        :return:            FWHM
        """

        # Optionally smooth the input data
        if smooth:
            curve = self.smooth_data(curve)

        # Find the largest peak and calculate FWHM
        peaks, _ = find_peaks(curve)
        if peaks.size == 0:
            return np.nan
        peak = peaks[np.argmax(curve[peaks])]
        fwhm_in_pixels = peak_widths(curve, [peak], rel_height=0.5)[0][0]
        fwhm = (fwhm_in_pixels / len(curve)) * self.phantom_dimensions[0]

        return fwhm

    def calculate_auc(self, curve, smooth=True):
        """
        Calculate the area under the input curve.

        :param curve:       Input for which to calculate area under the curve
        :param smooth:      Whether to first smooth the input data
        :return:            AUC
        """

        # Optionally smooth the input data, the calculate AUC
        if smooth:
            curve = self.smooth_data(curve)
        auc = np.trapz(curve)

        return auc

    def smooth_data(self, data_2_smooth):
        """
        Smooth the input data using Savitsky-Golay filtering.

        :param data_2_smooth:       Input 1D data
        :return:                    Smoothed data
        """

        wl = self.smoothing_params["Window length"]
        po = self.smoothing_params["Polynomial order"]
        smoothed = savgol_filter(data_2_smooth, window_length=wl, polyorder=po)

        return smoothed
