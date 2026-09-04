import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io
from scipy.ndimage import label
from skimage.transform import rescale
from skimage.measure import regionprops
import pandas as pd
import cv2
from sklearn.cluster import AgglomerativeClustering
import re
from joblib import Parallel, delayed

from qal.util import mean_in_circular_roi

class WellDetector:
    def __init__(self, parallel_processing=True):
        self.use_parallel_processing = parallel_processing
        self.environment = self.check_environment()
        self.scale_factor = 1

    def get_colormap(self, unique_ids, cmap='gist_rainbow'):
        plt_cmap = plt.get_cmap(cmap)
        color_space = np.linspace(0, 1, len(unique_ids))**0.5
        return {id_val: plt_cmap(color) for id_val, color in zip(unique_ids, color_space)}

    def display_images(self, im_array, file_names=[]):
        num_images = len(im_array)
        # Determine grid size based on the number of images
        grid_size = int(np.ceil(np.sqrt(num_images)))

        # Create a new figure
        fig, axes = plt.subplots(grid_size, grid_size, figarea=(15, 15))

        # Hide any extra subplots
        for i in range(num_images, grid_size * grid_size):
            fig.delaxes(axes.flat[i])

        # Display each image in the grid
        for i, ax in enumerate(axes.flat[:num_images]):
            ax.imshow(im_array[i], cmap='gray')  # You can specify a different colormap if needed
            if len(file_names) > 0:
                ax.set_title(file_names[i], fontsize=12)  # Set title with the file name
            ax.axis('off')  # Hide axes

        plt.show()

    def read_raw_tiff(self, im, remap_16bit=False):
        im_array = io.imread(im)
        if remap_16bit:
            im_array_raw = im_array.copy()
            im_array = (im_array - np.min(im_array)) / (np.max(im_array) - np.min(im_array))*(pow(2,16)-1)
            im_array = im_array.astype(int)
            im_array = im_array.astype(np.uint16)
        return im_array

    def cvt_to_log_scale(self, img):
        """
        Input: 16 bit tiff image
        Output: 16 bit log scale image
        """
        uint16_max = 65535
        image = img.astype(float) - np.nanmin(img)
        image = uint16_max * (image.astype(float) / np.max(image))
        
        c = uint16_max / np.log(1 + np.max(image))
        log_image = c * (np.log(image + 1.0))
        log_image = np.array(log_image, dtype=np.uint16)
        
        return log_image

    def cvt_to_uint8(self, img):
        image = img.astype(float) - np.nanmin(img)
        image = 255 * (image.astype(float) / np.max(image))
        image = image.astype(np.uint8)
        return image

    def get_thresh_rolling_window_binary(self, img):
        thresh_step = 5
        thresholds = np.arange(255 - thresh_step, 0, -thresh_step)

        def process_thresh(thresh):
            ret, thresh_img = cv2.threshold(img, thresh-thresh_step, thresh+thresh_step, cv2.THRESH_BINARY)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
            thresh_img = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, kernel)
            return thresh_img

        if self.use_parallel_processing:
            images = Parallel(n_jobs=2)(delayed(process_thresh)(thresh) for thresh in thresholds)
        else:
            images = [process_thresh(thresh) for thresh in thresholds]

        return images

    def compute_well_positions(self, im_src, binary_image, labeled_image, min_area=500, min_eccentricity=0.2):
        try:
            props = regionprops(labeled_image, intensity_image=binary_image)

            # Filter regions
            valid_props = [p for p in props if p.area >= min_area and p.eccentricity <= min_eccentricity]
            
            # if not valid_props:
            #     return pd.DataFrame()

            pts = []
            for idx, prop in enumerate(valid_props):
                well_pos = prop.centroid
                bounding_box = prop.bbox
                height = bounding_box[2] - bounding_box[0]
                width = bounding_box[3] - bounding_box[1]
                estimated_radius = min(height, width) / 2 # takes min of bounding box of detected circle and divides by 2 to get radius
                
                mask = self.circular_mask(im_src.shape, well_pos, estimated_radius)
                im_copy = im_src.copy()
                im_copy[~mask] = 0
                non_zero_pixels = im_copy[im_copy > 0]

                if non_zero_pixels.size == 0:
                    average_intensity = np.nan
                else:
                    average_intensity = np.mean(non_zero_pixels)

                pts.append({
                    'x': well_pos[1] / self.scale_factor,
                    'y': well_pos[0] / self.scale_factor,
                    'area': prop.area / (self.scale_factor ** 2),
                    'ROI Diameter': estimated_radius * 2 / self.scale_factor,
                    'ROI Radius': estimated_radius / self.scale_factor,
                    'mean_intensity': float(average_intensity)
                })
            
            df = pd.DataFrame(pts)
            return df

        except Exception as e:
            print(f"Error in compute_well_positions: {e}")
            return pd.DataFrame()


    def apply_threshold(self, image, threshold):
        return image > threshold

    def get_well_positions(self, im_src, binary_image):
        labeled_image, num_features = self.connected_components(binary_image)
        
        if num_features == 0:
            return pd.DataFrame()
        
        df = self.compute_well_positions(im_src, binary_image, labeled_image)
        return df

    def get_clusters_from_points(self, df, threshold=15, use_std=False):
        if df is None or df.empty:
            print("Error: Empty DataFrame passed to clustering.")
            return None

        try:
            df['point'] = df.apply(lambda row: [row['x'], row['y']], axis=1)
            points = df[['point']].to_numpy()
            points = [pt[0] for pt in points]

            clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=df['ROI Diameter'].mean())
            clusters = clustering.fit_predict(points)
            df['cluster'] = clusters

            # Calculate overall mean and std dev of the radius column
            mean_radius = df['ROI Radius'].mean()
            std_dev_radius = df['ROI Radius'].std()
            # Any value more than 2 times the std dev from the mean
            thresh = 0.1 * threshold * std_dev_radius
            # Determine rows where the radius is significantly different from the mean
            if use_std:
                mask = abs(df['ROI Radius'] - mean_radius) > thresh
            else:
                mask = abs(df['ROI Radius'] - mean_radius) > threshold
            # Remove those rows
            df_filtered = df[~mask]

            return df_filtered

        except Exception as e:
            print(f"Error in clustering: {e}")
            return None

    def get_well_mean_intensity(self, im, coord, radius):
        mask = self.circular_mask(im.shape, coord, radius)
        masked_im = im[mask]
        
        if masked_im.size == 0:
            return np.nan

        mean_intensity = np.nanmean(masked_im.astype(float))
        return mean_intensity

    def get_well_intensities(self, im, df, sort_by_intensity=True):
        # Compute mean_intensity for each circle in well list
        def compute_intensity(row):
            intensity = self.get_well_mean_intensity(im, (row['x'], row['y']), row['ROI Diameter'])
            if np.isnan(intensity):
                intensity = 0  # or any default value you want to assign when intensity is NaN
            return intensity

        if self.use_parallel_processing:
            intensities = Parallel(n_jobs=-1)(
                delayed(compute_intensity)(row) for idx, row in df.iterrows()
            )
        else:
            intensities = [compute_intensity(row) for idx, row in df.iterrows()]

        df['mean_intensity'] = np.array(intensities, dtype=np.float64)  # Ensure float64

        if sort_by_intensity:
            df = df.sort_values(by='mean_intensity', ascending=False) 
        return df

    def select_single_well(self, im, df, selection='brightest'):
        """Reduce detected wells to the first or brightest circular ROI.

        Parameters
        ----------
        im : numpy.ndarray
            Source image used when selecting by intensity.
        df : pandas.DataFrame
            Detection table containing ``x``, ``y``, and ``ROI Diameter``.
        selection : {'brightest', 'first'}
            Selection strategy when more than one well was detected.
        """
        if df is None or df.empty:
            return pd.DataFrame()
        if selection not in ('brightest', 'first'):
            raise ValueError("selection must be 'brightest' or 'first'")

        required = {'x', 'y', 'ROI Diameter'}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"Detection table is missing columns: {sorted(missing)}")

        wells = df.reset_index(drop=True).copy()
        if len(wells) == 1 or selection == 'first':
            return wells.iloc[[0]].copy()

        radii = (
            wells['ROI Radius']
            if 'ROI Radius' in wells
            else wells['ROI Diameter'] / 2
        )
        wells['_selection_mean'] = [
            mean_in_circular_roi(im, (row.x, row.y), radius)
            for (_, row), radius in zip(wells.iterrows(), radii)
        ]
        selected = wells.nlargest(1, '_selection_mean')
        return selected.drop(columns='_selection_mean').copy()

    def refine_well_at_coordinate(
        self,
        im,
        coordinate,
        search_radius=None,
        min_area=None,
        max_eccentricity=0.85,
        intensity_fractions=(0.25, 0.15, 0.08, 0.04),
        seed_neighborhood=3,
    ):
        """Refine one approximate click to a local well centroid and diameter.

        Full-image ``detect_wells`` can miss low-contrast wells because its
        rolling global thresholds keep only the most persistent regions. This
        method instead thresholds a window around the click relative to the
        local background, keeps the connected component containing the seed,
        and returns that region's centroid.

        Parameters
        ----------
        im : numpy.ndarray
            Two-dimensional source image.
        coordinate : sequence of float
            Approximate ``(x, y)`` well center in image coordinates.
        search_radius : float, optional
            Half-width of the local search window in pixels. Defaults to 10% of
            the smaller image dimension, with a minimum of 32 pixels.
        min_area : float, optional
            Minimum connected-component area in pixels. Defaults to a fraction
            of the search window.
        max_eccentricity : float
            Reject elongated components above this eccentricity.
        intensity_fractions : sequence of float
            Candidate thresholds of the form
            ``background + fraction * (seed - background)``, tried in order.
        seed_neighborhood : int
            Radius used to pick the brightest nearby seed when the click is
            slightly off-center.
        """
        if im.ndim != 2:
            raise ValueError(f"Expected a 2-D image, got shape {im.shape}")
        if len(coordinate) != 2:
            raise ValueError("coordinate must be an (x, y) pair")

        height, width = im.shape
        x, y = map(float, coordinate)
        if not np.isfinite((x, y)).all():
            raise ValueError("coordinate must contain finite values")
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError(
                f"coordinate ({x}, {y}) is outside image bounds "
                f"(width={width}, height={height})"
            )

        radius = (
            float(search_radius)
            if search_radius is not None
            else max(0.1 * min(height, width), 32.0)
        )
        if radius <= 0:
            raise ValueError("search_radius must be positive")

        x0 = max(0, int(np.floor(x - radius)))
        x1 = min(width, int(np.ceil(x + radius)) + 1)
        y0 = max(0, int(np.floor(y - radius)))
        y1 = min(height, int(np.ceil(y + radius)) + 1)
        crop = np.asarray(im[y0:y1, x0:x1], dtype=float)
        if crop.size == 0:
            raise RuntimeError("Local search window is empty")

        seed_x = int(np.clip(round(x) - x0, 0, crop.shape[1] - 1))
        seed_y = int(np.clip(round(y) - y0, 0, crop.shape[0] - 1))
        neighborhood = max(int(seed_neighborhood), 0)
        y_lo = max(0, seed_y - neighborhood)
        y_hi = min(crop.shape[0], seed_y + neighborhood + 1)
        x_lo = max(0, seed_x - neighborhood)
        x_hi = min(crop.shape[1], seed_x + neighborhood + 1)
        local = crop[y_lo:y_hi, x_lo:x_hi]
        local_offset = np.unravel_index(np.argmax(local), local.shape)
        seed_y = y_lo + int(local_offset[0])
        seed_x = x_lo + int(local_offset[1])
        seed = float(crop[seed_y, seed_x])

        border = np.concatenate(
            (crop[0, :], crop[-1, :], crop[:, 0], crop[:, -1])
        )
        background = float(np.median(border))
        dynamic_range = seed - background
        if not np.isfinite(dynamic_range) or dynamic_range <= 0:
            raise RuntimeError(
                f"No brighter well signal than the local background near "
                f"({x:.1f}, {y:.1f})"
            )

        area_floor = (
            float(min_area)
            if min_area is not None
            else max(0.02 * crop.size, 25.0)
        )
        area_ceiling = 0.85 * crop.size
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        last_error = None

        for fraction in intensity_fractions:
            threshold = background + float(fraction) * dynamic_range
            binary = (crop >= threshold).astype(np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            labeled_image, num_features = label(binary)
            if num_features == 0:
                last_error = "No connected region survived local thresholding"
                continue

            component_id = int(labeled_image[seed_y, seed_x])
            if component_id == 0:
                ys, xs = np.where(labeled_image > 0)
                distances = (xs - seed_x) ** 2 + (ys - seed_y) ** 2
                nearest = int(np.argmin(distances))
                if distances[nearest] > (0.35 * radius) ** 2:
                    last_error = (
                        "No connected region is close enough to the click"
                    )
                    continue
                component_id = int(labeled_image[ys[nearest], xs[nearest]])

            component = labeled_image == component_id
            props = regionprops(component.astype(np.int32))
            if not props:
                last_error = "Connected region could not be measured"
                continue
            prop = props[0]
            if prop.area < area_floor or prop.area > area_ceiling:
                last_error = (
                    f"Connected region area {prop.area:.0f} is outside "
                    f"[{area_floor:.0f}, {area_ceiling:.0f}]"
                )
                continue
            if prop.eccentricity > max_eccentricity:
                last_error = (
                    f"Connected region eccentricity {prop.eccentricity:.3f} "
                    f"exceeds {max_eccentricity}"
                )
                continue

            cy, cx = prop.centroid
            bbox_height = prop.bbox[2] - prop.bbox[0]
            bbox_width = prop.bbox[3] - prop.bbox[1]
            estimated_radius = min(bbox_height, bbox_width) / 2.0
            if estimated_radius <= 0:
                last_error = "Connected region has a non-positive radius"
                continue

            well_x = float(cx + x0)
            well_y = float(cy + y0)
            if np.hypot(well_x - x, well_y - y) > radius:
                last_error = (
                    "Refined centroid left the local search window"
                )
                continue

            mean_intensity = mean_in_circular_roi(
                im,
                (well_x, well_y),
                estimated_radius,
            )
            return pd.Series(
                {
                    "x": well_x,
                    "y": well_y,
                    "area": float(prop.area),
                    "ROI Diameter": float(2.0 * estimated_radius),
                    "ROI Radius": float(estimated_radius),
                    "mean_intensity": float(mean_intensity),
                }
            )

        raise RuntimeError(
            last_error
            or f"Could not refine a well near ({x:.1f}, {y:.1f})"
        )

    def get_pseudo_dark_roi_for_testing(
        self,
        im,
        well_df,
        region_of_well_to_analyze=0.5,
    ):
        """Place a background ROI for explicit testing only.

        This helper must not be used to produce a production analysis. A
        measured dark frame is the appropriate baseline for RET analysis.
        """
        selected = self.select_single_well(im, well_df, selection='first')
        if selected.empty:
            raise ValueError("A detected signal well is required")

        well = selected.iloc[0]
        analyzed_radius = (
            float(well['ROI Diameter']) * region_of_well_to_analyze / 2
        )
        height, width = im.shape[:2]
        margin = max(int(np.ceil(2.5 * analyzed_radius)), 1)
        candidates = (
            (margin, margin),
            (width - margin - 1, margin),
            (width - margin - 1, height - margin - 1),
            (margin, height - margin - 1),
            (margin, height / 2),
            (width - margin - 1, height / 2),
            (width / 2, margin),
            (width / 2, height - margin - 1),
        )

        signal_x, signal_y = float(well['x']), float(well['y'])
        valid = []
        for x, y in candidates:
            inside = (
                analyzed_radius <= x < width - analyzed_radius
                and analyzed_radius <= y < height - analyzed_radius
            )
            separated = (
                np.hypot(x - signal_x, y - signal_y)
                >= 4 * analyzed_radius
            )
            if inside and separated:
                mean = mean_in_circular_roi(
                    im,
                    (x, y),
                    analyzed_radius,
                )
                valid.append((mean, float(x), float(y)))

        if not valid:
            raise ValueError(
                "No valid pseudo-dark ROI exists outside the signal well"
            )

        _, x, y = min(valid)
        pseudo_dark = selected.copy()
        pseudo_dark.loc[:, 'x'] = x
        pseudo_dark.loc[:, 'y'] = y
        pseudo_dark.loc[:, 'well'] = 'Pseudo Dark (Testing Only)'
        return pseudo_dark

    def set_consistent_roi_region(self, df, df_source=None):
        # Set the radius for all rows to the largest radius from the input df
        if df_source is None:
            df['ROI Diameter'] = df['ROI Diameter'].max()
            df['ROI Radius'] = df['ROI Radius'].max()
            return df
        
        df['ROI Diameter'] = df_source['ROI Diameter'].max()
        df['ROI Radius'] = df_source['ROI Radius'].max()
        return df

    def get_cluster_id(self, im, df, set_consistent_roi_region, wells=None):
        avg_centers = df.groupby(['cluster'])[['x', 'y', 'ROI Diameter', 'ROI Radius']].mean()
        roi_size = int(avg_centers['ROI Radius'].max())

        if set_consistent_roi_region:
            avg_centers = self.set_consistent_roi_region(avg_centers)
        avg_centers = self.get_well_intensities(im, avg_centers)
        
        # If labels are provided, assign them; otherwise, assign numeric labels
        if wells is not None:
            for idx, (index, row) in enumerate(avg_centers.iterrows()):
                if idx < len(wells):
                    avg_centers.loc[index, 'well'] = wells[idx]
                else:
                    avg_centers.loc[index, 'well'] = f"Cluster {idx + 1 - len(wells)}"
        else:
            for idx, (index, row) in enumerate(avg_centers.iterrows()):
                avg_centers.loc[index, 'well'] = f"Cluster {idx + 1}"

        return avg_centers, roi_size

    def get_detected_features(self, im_src, thresh_im_array):
        try:
            def process_thresh_im(thresh_im):
                wells = self.get_well_positions(im_src, thresh_im)
                return wells

            if self.scale_factor != 1:
                im_src = rescale(im_src, self.scale_factor, anti_aliasing=True, preserve_range=True)

            if self.use_parallel_processing:
                wells_list = Parallel(n_jobs=-1)(delayed(process_thresh_im)(thresh_im) for thresh_im in thresh_im_array)
            else:
                wells_list = [process_thresh_im(thresh_im) for thresh_im in thresh_im_array]

            df = pd.DataFrame()
            for wells in wells_list:
                if wells is not None and not wells.empty:
                    df = pd.concat([df, wells], ignore_index=True)

            if df.empty:
                print("Warning: No features detected in any threshold image.")
                return None

            sorted_df = df.sort_values(by='ROI Radius', ascending=False)
            return sorted_df

        except Exception as e:
            print(f"Error in get_detected_features: {e}")
            return None

    def circular_mask(self, shape, center, radius):
        Y, X = np.ogrid[:shape[0], :shape[1]]
        distance_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
        mask = distance_from_center <= radius
        return mask

    def connected_components(self, binary_image):
        """Computes the connected components of the given binary image."""
        labeled_image, num_features = label(binary_image)
        return labeled_image, num_features

    def debug_detected_wells(self, im, df):
        fig, (ax) = plt.subplots(1, 1, figsize=(20, 10))

        # Display the image with blue circles on the left subplot
        ax.imshow(im, cmap='gray')
        # ax.set_title(f"{tif_files[idx]} - Blue Circles")

        for index, row in df.iterrows():
            circle = patches.Circle((row['x'], row['y']), row['ROI Diameter'], edgecolor='b', facecolor='none')
            ax.add_patch(circle)
        ax.axis('on')
        plt.show()

    def plot_detected_wells(self, im, df, im_cmap='gray', colormap='plasma', save_path=None):
        """
        Plot detected wells with colored circles, crosshairs, and a legend using a specified colormap.

        :param im: Input image to plot.
        :param df: DataFrame containing well detection data with 'x', 'y', and 'ROI Diameter' columns.
        :param im_cmap: Colormap for the image background.
        :param colormap: Colormap to use for well ROI visualization (default is 'plasma').
        :param save_path: Path to save the output figure (optional).
        """
        # Create the figure and axis
        fig, ax = plt.subplots(figsize=(8, 8))

        # Display the image
        ax.imshow(im, cmap=im_cmap)

        # Generate distinct colors for each unique well ID using the specified colormap
        cmap = plt.get_cmap(colormap)  # Get the colormap
        unique_ids = df.index.unique()
        colors = cmap(np.linspace(0, 1, len(unique_ids)))  # Generate distinct colors
        id_colors = {id_val: color for id_val, color in zip(unique_ids, colors)}

        legend_patches = []  # Custom legend items

        # Loop through each row in the DataFrame to draw circles and crosshairs
        for id_val, row in df.iterrows():
            # Draw the circle
            ax.add_patch(
                plt.Circle(
                    (row['x'], row['y']),
                    row['ROI Radius'],
                    edgecolor=id_colors[id_val],
                    facecolor='none',
                    lw=2
                )
            )
            # Draw crosshairs for detected well centers
            ax.plot(
                [row['x'], row['x']], [row['y'] - 10, row['y'] + 10],  # Vertical line
                color=id_colors[id_val], lw=1
            )
            ax.plot(
                [row['x'] - 10, row['x'] + 10], [row['y'], row['y']],  # Horizontal line
                color=id_colors[id_val], lw=1
            )

            # Add to legend if not already present
            if id_val not in [patch.get_label() for patch in legend_patches]:
                legend_patches.append(patches.Patch(color=id_colors[id_val], label=id_val))

        # Add legend outside the plot
        ax.legend(
            handles=legend_patches,
            loc='upper left',
            bbox_to_anchor=(1.05, 1),
            title="Well IDs",
            fontsize=12
        )
        ax.axis('off')  # Turn off axis
        plt.title("Detected Wells (Log Scale Image)", fontsize=18, fontweight='bold')  # Add title

        # Save the figure if a save path is provided
        if save_path:
            plt.tight_layout()
            plt.savefig(save_path, bbox_inches='tight')
            plt.close(fig)  # Close the figure after saving
        else:
            plt.tight_layout()
            plt.show()

    def detect_wells(self, im, well_ids=None, show_detected_wells=False, debug=False, set_consistent_roi_region=False,
                     downscale=1):
        self.scale_factor = downscale
        try:
            im_copy = im.copy()

            # Get log scale of the original image using WellDetector's method
            log_im = self.cvt_to_log_scale(im_copy)

            # Convert 16 bit log scale image to 8-bit using WellDetector's method
            log_im_uint8 = self.cvt_to_uint8(log_im)
            if self.scale_factor != 1:
                log_im_uint8 = rescale(log_im_uint8, self.scale_factor, anti_aliasing=True, preserve_range=True).astype(
                    np.uint8)

            # Using WellDetector's methods for further processing
            thresh_im_array = self.get_thresh_rolling_window_binary(log_im_uint8)
            df_features = self.get_detected_features(im, thresh_im_array)

            if df_features is not None:
                df_feature_clusters = self.get_clusters_from_points(df_features)
                if df_feature_clusters.empty or len(df_feature_clusters) < 3:
                    df_feature_clusters = self.get_clusters_from_points(df_features, use_std=True)
                df_clusters, roi_size = self.get_cluster_id(im, df_feature_clusters, set_consistent_roi_region, wells=well_ids)

                # Replace the cluster index with the id column
                # df_clusters.set_index('well', inplace=True)
                if debug:
                    self.debug_detected_wells(log_im, df_features)

                if show_detected_wells:
                    self.plot_detected_wells(log_im, df_clusters)
            
            self.df = df_clusters
            self.df.drop('mean_intensity', axis=1, inplace=True)

            return df_clusters
        except Exception as e:
            # print(f"An error occurred: {e}")
            return None

    def compute_transformed_points(self, source_points, target_points):
        """Map the canonical 3x3 grid through a fitted similarity transform."""

        def compute_transformation_matrix(source_points, target_points):
            source = np.asarray(source_points, dtype=float)
            target = np.asarray(target_points, dtype=float)
            if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 2:
                raise ValueError(
                    "source_points and target_points must be matching (n, 2) arrays"
                )
            if len(source) < 2:
                raise ValueError("At least two corresponding points are required")

            rows = []
            values = []
            for (source_x, source_y), (target_x, target_y) in zip(source, target):
                rows.extend(
                    (
                        [source_x, -source_y, 1, 0],
                        [source_y, source_x, 0, 1],
                    )
                )
                values.extend((target_x, target_y))
            A = np.asarray(rows, dtype=float)
            B = np.asarray(values, dtype=float)
            X, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
            s_cos_theta, s_sin_theta, tx, ty = X
            theta = np.arctan2(s_sin_theta, s_cos_theta)
            s = np.sqrt(s_cos_theta**2 + s_sin_theta**2)
            if not np.isfinite(s) or s <= 0:
                raise ValueError("Anchor points do not define a valid grid scale")
            T = np.array([
                [s * np.cos(theta), -s * np.sin(theta), tx],
                [s * np.sin(theta), s * np.cos(theta), ty],
                [0, 0, 1]
            ])
            return T
        
        # Function to apply the transformation to a point
        def transform_point(point, T):
            x, y = point
            transformed_point = np.dot(T, np.array([x, y, 1]))
            return tuple(transformed_point[:2])
        
        # Compute the transformation matrix
        T = compute_transformation_matrix(source_points, target_points)
        
        # Original grid points
        grid_points = [
            [(0, 0), (15, 0), (30, 0)],
            [(0, 15), (15, 15), (30, 15)],
            [(0, 30), (15, 30), (30, 30)]
        ]
        
        # Transform all the points in the grid
        transformed_grid_points = [transform_point(point, T) for row in grid_points for point in row]
        
        return transformed_grid_points

    def extract_value(self, s):
        # Check if the string is 'Control' and return 0.0
        if s.lower() == 'control':
            return 0.0

        # Use regular expression to find numbers in the string
        numbers = re.findall(r'\d+\.?\d*', s)

        # Convert the first found number to float and return it
        # If no number is found, return 0.0 as a default
        return float(numbers[0]) if numbers else 0.0

    def apply_well_ids(self, df, well_ids=None):
        if well_ids is not None:
            for idx, (index, row) in enumerate(df.iterrows()):
                if idx < len(well_ids):
                    df.loc[index, 'well'] = well_ids[idx]
                else:
                    df.loc[index, 'well'] = f"Cluster {idx + 1 - len(well_ids)}"
        else:
            for idx, (index, row) in enumerate(df.iterrows()):
                df.loc[index, 'well'] = f"Cluster {idx + 1}"
        
        df['value'] = df['well'].apply(self.extract_value)
        # df.set_index('well', inplace=True)

    def estimate_remaining_wells_3x3(self, im, df, well_ids=None, show_detected_wells=False):
        try:
            if df is None or len(df) < 3:
                raise ValueError(
                    "At least three row-major anchor wells are required"
                )
            canonical_grid = [
                (0, 0), (15, 0), (30, 0),
                (0, 15), (15, 15), (30, 15),
                (0, 30), (15, 30), (30, 30),
            ]
            anchor_count = min(len(df), len(canonical_grid))
            source_points = canonical_grid[:anchor_count]
            target_points = (
                df.iloc[:anchor_count][['x', 'y']]
                .apply(tuple, axis=1)
                .tolist()
            )
            
            transformed_grid = self.compute_transformed_points(
                source_points,
                target_points,
            )
            
            # Create a new DataFrame from the transformed points
            transformed_df = pd.DataFrame(transformed_grid, columns=['x', 'y'])
            transformed_df = self.set_consistent_roi_region(transformed_df, df_source=df)

            transformed_df = self.get_well_intensities(im, transformed_df, sort_by_intensity=False)
            if show_detected_wells:
                self.plot_detected_wells(self.cvt_to_log_scale(im), transformed_df)

            self.apply_well_ids(transformed_df, well_ids)
            transformed_df.drop('mean_intensity', axis=1, inplace=True)

            return transformed_df
        
        except Exception as e:
            print(e)

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