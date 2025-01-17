import os
import pickle
import warnings
import numpy as np
import matplotlib.pyplot as plt
from qal import UniformityAnalyzer


class DistortionAnalyzer:

    def __init__(self):
        """
        DistortionAnalyzer analyzes radial distortion in an image of the distortion target. It is only meant to analyze
        radial distortion - presence of keystone distortion may produce inaccurate results.
        """

        self.dots = None                            # Centroid locations of dots in distortion target image
        self.image_center = None                    # Pixel indices of center of image(s)
        self.fov = None                             # Camera field of view
        self.show_dots = None                       # If True, display the actual and predicted dot locations
        self.av_dist = None                         # Average distance between dots around the image center
        self.output = None                          # Outputs to be passed to visualizer

    def compute_distortion(self, detector_output, show_dots=False, ignore_extra=False, save_output=False):
        """
        Analyze the input data to produce a radial distortion curve.

        :param detector_output:     Output of a RudDetector object containing data to be analyzed
        :param show_dots:           If True, display the actual and predicted dot locations
        :param ignore_extra:        If True, do not extrapolate distortion map for values outside data range
        :param save_output:         If True, save the outputs to a pickle file
        """

        self.dots = detector_output["dots"]
        self.image_center = detector_output["image center"]
        self.fov = detector_output["fov"]
        self.show_dots = show_dots

        # Estimate average distance between dots around the image center
        self.av_dist = self.get_average_distance()

        actual_distances = []
        predicted_distances = []
        x = []
        y = []

        for dots_df in self.dots:

            # Calculate actual distances between dots and the image center
            x_locs = dots_df["x"].to_numpy()
            y_locs = dots_df["y"].to_numpy()
            distances_to_center = self.dist_between(self.image_center[1], self.image_center[0], x_locs, y_locs)
            sorted_idx = np.argsort(distances_to_center)
            actual_distances.append(distances_to_center[sorted_idx])
            x.append(x_locs[sorted_idx])
            y.append(y_locs[sorted_idx])

            # Find distances between each dot and the dot closest to the image center
            closest_idx = np.argmin(distances_to_center)
            closest_dot = (x_locs[closest_idx], y_locs[closest_idx])
            distances_to_closest = self.dist_between(closest_dot[0], closest_dot[1], x_locs, y_locs)

            # Calculate rotation angle of dots relative to dot closest to center
            sorted_distances_to_closest = np.argsort(distances_to_closest)
            neighbor_idxs = sorted_distances_to_closest[1:5]
            neighbor_angles = []
            for itr, neighbor_idx in enumerate(neighbor_idxs):
                neighbor_dot = (x_locs[neighbor_idx], y_locs[neighbor_idx])
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=RuntimeWarning,
                                            message="divide by zero encountered in scalar divide")
                    angle = np.arctan((closest_dot[1] - neighbor_dot[1]) / (closest_dot[0] - neighbor_dot[0]))
                    if itr == 0:
                        neighbor_angles.append(angle)
                    else:
                        if np.abs(neighbor_angles[0] - angle) < np.pi / 4:
                            neighbor_angles.append(angle)
                        elif np.abs(np.abs(neighbor_angles[0]) - np.abs(angle)) < np.pi / 4:
                            neighbor_angles.append(-angle)
                        elif neighbor_angles[0] > angle:
                            neighbor_angles.append(angle + np.pi / 2)
                        else:
                            neighbor_angles.append(angle - np.pi / 2)

            grid_angle = np.mean(neighbor_angles)

            # Calculated the predicted distances between dots and the image center
            pred_dist = self.get_predicted_distances(dots_df, closest_dot, grid_angle)
            predicted_distances.append(pred_dist)

        actual_distances = np.concatenate(actual_distances)
        predicted_distances = np.concatenate(predicted_distances)

        distortion = 100 * (actual_distances - predicted_distances) / predicted_distances

        # Generate a spatial map of the distortion
        x = np.insert(np.concatenate(x), 0, self.image_center[1])
        y = np.insert(np.concatenate(y), 0, self.image_center[0])
        z = np.insert(distortion.copy(), 0, 0)
        mapper = UniformityAnalyzer()
        mapper.fov = self.fov
        mapper.k_spline = 3
        mapper.zero_extra = ignore_extra
        distortion_map, rss, r_sq = mapper.fit_data(data=(x, y, z), distortion_call=True, show_fit=False)

        # Collect outputs to be used by visualizer and optionally save
        self.output = {
            "Actual distances": actual_distances,
            "Distortion": distortion,
            "FOV": self.fov,
            "xq": mapper.xq,
            "yq": mapper.yq,
            "Distortion map": distortion_map,
            "Directory": detector_output["image_dir"],
            "dots": self.dots
        }
        if save_output:
            save_dir = os.path.join(detector_output["image_dir"], "Distortion Figures")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            with open(os.path.join(save_dir, "output_dict.pkl"), 'wb') as f:
                pickle.dump(self.output, f)

    def get_average_distance(self):
        """
        Calculate the average distance between dots around the image center and use that as the estimate for the grid
        spacing of expected positions.

        :return:                Estimated average distance between grid points
        """

        av_dist = 0

        for dots_df in self.dots:

            # Calculate actual distances between dots and the image center
            x_locs = dots_df["x"].to_numpy()
            y_locs = dots_df["y"].to_numpy()
            distances_to_center = self.dist_between(self.image_center[1], self.image_center[0], x_locs, y_locs)

            # Estimate average distance between dots close to the image center
            closest_idx = np.argmin(distances_to_center)
            closest_dot = (x_locs[closest_idx], y_locs[closest_idx])
            distances_to_closest = self.dist_between(closest_dot[0], closest_dot[1], x_locs, y_locs)
            closest_four_dists = np.sort(distances_to_closest)[1:5]
            av_dist += np.mean(closest_four_dists)

        av_dist /= len(self.dots)

        return av_dist

    def get_predicted_distances(self, dots_df, closest_dot, angle):
        """
        Calculate the predicted distances of the dots in DOTS_DF from image center. Predicted distances are calculated
        by assuming that dots around the image center are undistorted and so can be used to create a grid of undistorted
        dots. Parameters are passed from SELF.COMPUTE_DISTORTION.

        :param dots_df:         Dataframe containing x-y locations of dots
        :param closest_dot:     Closest dot to the image center
        :param angle:           Rotation angle of points with respect to the closest point to image center
        :return:                Predicted distances of the dots from image center
        """

        # Estimate number of grid points in each dimension
        n = int(np.floor(np.sqrt(len(dots_df))))

        # Make grid centered on (0, 0) and spaced by AV_DIST
        start = -n * self.av_dist
        stop = n * self.av_dist + 1e-10
        v = np.arange(start, stop, self.av_dist)
        xx, yy = np.meshgrid(v, v)

        # Rotate the grid about the origin by ANGLE
        xx_rot, yy_rot = self.rotate_grid(xx, yy, angle)

        # Translate the grid to be centered on CLOSEST_DOT
        xx_rot += closest_dot[0]
        yy_rot += closest_dot[1]

        # Match grid points with actual dots
        xx_pred, yy_pred = self.get_predicted_points(xx_rot, yy_rot, dots_df)

        # Calculate predicted distances from image center
        pred_dist = self.dist_between(self.image_center[1], self.image_center[0], xx_pred, yy_pred)

        return pred_dist

    def get_predicted_points(self, xx, yy, dots_df):
        """
        Match the locations of points in the predicted grid to their corresponding actual points. Parameters are passed
        from SELF.GET_PREDICTED_DISTANCES.

        :param xx:          x-locations of grid points
        :param yy:          y-locations of grid points
        :param dots_df:     Dataframe containing x-y locations of dots
        :return:            Locations of predicted dots
        """

        xx = xx.flatten()
        yy = yy.flatten()
        x_actual = dots_df["x"].to_numpy()
        y_actual = dots_df["y"].to_numpy()

        # Sort dots by distance to the image center
        distances_to_center = self.dist_between(self.image_center[1], self.image_center[0], x_actual, y_actual)
        sorted_idx = np.argsort(distances_to_center)

        # Find the closest grid point to each actual dot, then assign it and remove from the search list
        xx_keep = []
        yy_keep = []
        for px, py in zip(x_actual[sorted_idx], y_actual[sorted_idx]):
            distances_to_dot = self.dist_between(px, py, xx, yy)
            closest = np.argmin(distances_to_dot)
            xx_keep.append(xx[closest])
            yy_keep.append(yy[closest])
            xx = np.delete(xx, closest)
            yy = np.delete(yy, closest)

        if self.show_dots:
            fig, ax = plt.subplots()
            ax.scatter(x_actual, y_actual, s=8, color='k')
            ax.scatter(xx_keep, yy_keep, s=16, color='r', facecolors='None')
            ax.invert_yaxis()
            plt.draw()

        return np.array(xx_keep), np.array(yy_keep)

    @staticmethod
    def rotate_grid(xx, yy, angle):
        """
        Rotate the input grid around the origin by the input angle

        :param xx:          x-coordinates of grid
        :param yy:          y-coordinates of grid
        :param angle:       Rotation angle
        :return:            Rotated grid
        """

        rot_mat = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
        xy = np.array([xx.flatten(), yy.flatten()]).T
        xy_rot = np.matmul(xy, rot_mat)
        xx_rot = xy_rot[:, 0].reshape(xx.shape)
        yy_rot = xy_rot[:, 1].reshape(yy.shape)
        return xx_rot, yy_rot

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
