# rrt_roi.py

import matplotlib.pyplot as plt
from skimage import io
from skimage.feature import corner_harris, corner_peaks
from skimage.draw import circle_perimeter
import numpy as np
from skimage import io, img_as_float, img_as_ubyte, transform as sk_transform
from skimage import io, segmentation
from qal.data import resolution_template
import cv2
import os

class RrtROI:
    def __init__(self):
        self.fig = None
        self.ax = None
        self.points = []
        self.image = None
        self.roi_corners = None
        self.group_coordinates = None

        self.environment = self.check_environment()

    def check_environment(self):
        try:
            from IPython import get_ipython
            get_ipython()
            if 'IPKernelApp' in get_ipython().config:
                # print("Running in Jupyter Notebook")
                get_ipython().magic('matplotlib widget')
                get_ipython().run_line_magic('matplotlib', 'ipympl')
                return "Jupyter Notebook"
            else:
                # print("Running in JupyterLab")
                get_ipython().run_line_magic('matplotlib', 'ipympl')
                return "JupyterLab"
        except AttributeError:
            # print("Running in a standard Python environment")
            return "Standard Python"

    def on_click(self, event):
        if len(self.points) < 2:
            ix, iy = event.xdata, event.ydata
            print(f'x = {ix}, y = {iy}')
            self.points.append((ix, iy))
            self.ax.plot(ix, iy, marker='+', color='r', markersize=10)
            self.fig.canvas.draw_idle()

        if len(self.points) >= 2:
            self.fig.canvas.mpl_disconnect(self.cid)
            self.roi_corners = self.detect_roi_corners(self.points, self.image)
            self.visualize_corners(self.roi_corners)
            self.fig.canvas.draw_idle()
            self.process_resolution_target(self.roi_corners)

    def detect_roi_corners(self, points, image):
        # Extract the coordinates of the selected points
        x1, y1 = points[0]
        x2, y2 = points[1]

        # Calculate the width between the selected points
        width = abs(x2 - x1)

        # Calculate the radius as 10% of the width
        radius = int(0.1 * width)

        # Perform Harris corner detection
        harris_response = corner_harris(image, k=0.04)
        corners = corner_peaks(harris_response, min_distance=1, threshold_rel=0.02)

        # Find the closest corner to each selected point
        roi_corners = []
        for selected_point in [(x1, y1), (x2, y2)]:
            min_distance = float('inf')
            closest_corner = None
            for corner in corners:
                distance = np.sqrt((corner[0] - selected_point[1])**2 + (corner[1] - selected_point[0])**2)
                if distance <= radius and distance < min_distance:
                    min_distance = distance
                    closest_corner = corner
            if closest_corner is not None:
                roi_corners.append(closest_corner)

        return roi_corners

    def visualize_corners(self, corners):
        # Draw circles around the closest corners
        for corner in corners:
            rr, cc = circle_perimeter(corner[0], corner[1], radius=5)
            self.ax.plot(cc, rr, 'bo', markersize=0.5)

    def select_points(self, im):
        if "Jupyter" in self.environment:
            self.select_points_jupyter(im)
        else:
            self.select_points_standard(im)

    def select_points_jupyter(self, im):
        self.image = im
        self.fig, self.ax = plt.subplots()
        self.fig.suptitle("Keypoint Selection", fontsize=16)
        self.fig.text(0.5, 0.91, "(Select points)", ha='center', fontsize=10, color='gray')
        self.fig.subplots_adjust(top=0.85)  # Add padding between title/message and plot
        self.ax.imshow(self.image)
        self.points = []
        self.roi_corners = None
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        plt.show()

    def select_points_standard(self, im):
        self.image = im
        self.fig, self.ax = plt.subplots()
        self.fig.canvas.manager.set_window_title('Keypoint Selection')
        self.fig.suptitle("Keypoint Selection", fontsize=16)
        self.fig.text(0.5, 0.91, "(Select points, then close window to continue)", 
                    ha='center', fontsize=10, color='gray')
        self.fig.subplots_adjust(top=0.85)  # Add padding between title/message and plot
        self.ax.imshow(self.image, cmap='gray')
        plt.ion()
        plt.show()
        print("Please make the following selections. \n1: Upper left corner of Group 0 Element 2 \n2: Bottom right corner of Group 0 Element 1")
        self.points = plt.ginput(2)
        self.roi_corners = self.detect_roi_corners(self.points, self.image)
        self.visualize_corners(self.roi_corners)
        plt.ioff()
        plt.show()
        self.process_resolution_target(self.roi_corners)

    def calculate_group_0_line(self, top_left, bottom_right, lp_x_per, lp_y_per):
        """
        Calculate the coordinates for the group 0 line.
        
        Args:
            top_left (tuple): (x, y) coordinates of the top-left corner of the ROI.
            bottom_right (tuple): (x, y) coordinates of the bottom-right corner of the ROI.
            lp_x_per (float): Percentage of the line's x-coordinate relative to the ROI width.
            lp_y_per (float): Percentage of the line's y-coordinate relative to the ROI height.
            
        Returns:
            tuple: A tuple containing the (x, y) coordinates of the start and end points of the line.
        """
        lp_roi_width, lp_roi_height = bottom_right[0] - top_left[0], bottom_right[1] - top_left[1]
        x = top_left[0] + (lp_roi_width * lp_x_per)
        y1 = top_left[1] - (lp_roi_height * lp_y_per)
        y2 = bottom_right[1] + (lp_roi_height * lp_y_per)
        return ((x, y1), (x, y2))

    def calculate_group_1_line(self, top_left, bottom_right, lp_x_per, lp_y_per, lp_length_per):
        """
        Calculate the coordinates for the group 1 line.
        
        Args:
            top_left (tuple): (x, y) coordinates of the top-left corner of the ROI.
            bottom_right (tuple): (x, y) coordinates of the bottom-right corner of the ROI.
            lp_x_per (float): Percentage of the line's x-coordinate relative to the ROI width.
            lp_y_per (float): Percentage of the line's y-coordinate relative to the ROI height.
            lp_length_per (float): Percentage of the line's length relative to the ROI height.
            
        Returns:
            tuple: A tuple containing the (x, y) coordinates of the start and end points of the line.
        """
        lp_roi_width, lp_roi_height = bottom_right[0] - top_left[0], bottom_right[1] - top_left[1]
        x = top_left[0] + (lp_roi_width * lp_x_per)
        y1 = top_left[1] - (lp_roi_height * lp_y_per)
        y2 = top_left[1] + (lp_roi_height * lp_length_per) + (lp_roi_height * lp_y_per)
        return ((x, y1), (x, y2))

    def calculate_group_2_line(self, top_left, bottom_right, lp_x_per, lp_y_per, lp_length_per, lp_y_start_offset):
        """
        Calculate the coordinates for the group 2 line.
        
        Args:
            top_left (tuple): (x, y) coordinates of the top-left corner of the ROI.
            bottom_right (tuple): (x, y) coordinates of the bottom-right corner of the ROI.
            lp_x_per (float): Percentage of the line's x-coordinate relative to the ROI width.
            lp_y_per (float): Percentage of the line's y-coordinate relative to the ROI height.
            lp_length_per (float): Percentage of the line's length relative to the ROI height.
            
        Returns:
            tuple: A tuple containing the (x, y) coordinates of the start and end points of the line.
        """
        lp_roi_width, lp_roi_height = bottom_right[0] - top_left[0], bottom_right[1] - top_left[1]
        x = top_left[0] + (lp_roi_width * lp_x_per)
        y1 = top_left[1] + (lp_roi_height * lp_y_start_offset) - (lp_roi_height * lp_y_per)
        y2 = top_left[1] + (lp_roi_height * lp_y_start_offset) + (lp_roi_height * lp_length_per) + (lp_roi_height * lp_y_per)
        return ((x, y1), (x, y2))

    def process_resolution_target(self, roi_corners):
        top_left = [roi_corners[0][1], roi_corners[0][0]]
        bottom_right = [roi_corners[1][1], roi_corners[1][0]]

        group_0_lp_x_per, group_0_lp_y_per = 0.06, 0.035
        group_0_line = self.calculate_group_0_line(top_left, bottom_right, group_0_lp_x_per, group_0_lp_y_per)

        group_1_lp_x_per, group_1_lp_y_per = 0.98, 0.02
        group_1_lp_length_per = 0.65
        group_1_line = self.calculate_group_1_line(top_left, bottom_right, group_1_lp_x_per, group_1_lp_y_per, group_1_lp_length_per)

        group_2_lp_x_per, group_2_lp_y_per = 0.517, 0.015
        group_2_lp_y_start_offset = 0.36
        group_2_lp_length_per = 0.252
        group_2_line = self.calculate_group_2_line(top_left, bottom_right, group_2_lp_x_per, group_2_lp_y_per, group_2_lp_length_per, group_2_lp_y_start_offset)

        self.group_coordinates = {
            0: {'coordinates': group_0_line, 'elements': range(2, 7)},
            1: {'coordinates': group_1_line, 'elements': range(1, 7)},
            2: {'coordinates': group_2_line, 'elements': range(2, 7)}
        }

        return self.group_coordinates

    def cvt_to_uint8(self, img):
        image = img.astype(float) - np.nanmin(img)
        image = 255 * (image.astype(float) / np.max(image))
        image = image.astype(np.uint8)
        return image

    def read_raw_tiff(self, img):
        im_array = io.imread(img)
        if im_array.dtype == 'float16' or im_array.dtype == 'float32':
            im_array = (im_array - np.min(im_array)) / (np.max(im_array) - np.min(im_array))*(pow(2,16)-1)
            im_array = im_array.astype(int)
            im_array = im_array.astype(np.uint16)
        return im_array

    def get_resolution_target_cropped(self, im_src, show_kp=False, min_good_matches=10, min_kp_dist_threshold=0.45, save_cropped_im=None):
        """
        Detects and extracts a resolution target (USAF 1951) from the input image using template matching, 
        keypoints detection, and homography transformation. Optionally saves the cropped image.

        Parameters:
        - im_src (numpy.ndarray): 
            The source image from which the resolution target will be extracted. Can be RGB or grayscale.

        - show_kp (bool, optional): 
            If True, visualizes the detected keypoints and matching results.

        - min_good_matches (int, optional): 
            The minimum number of "good" matches required to detect the resolution target reliably.

        - min_kp_dist_threshold (float, optional): 
            A ratio threshold used to filter keypoint matches based on their distance. 
            Values closer to 0 filter more matches; default is 0.45.

        - save_cropped_im (str, optional): 
            If provided, the path (with or without an extension) to save the cropped resolution target image.
            Defaults to saving as a TIFF file if no extension is specified.

        Returns:
        - roi (numpy.ndarray or None): 
            The cropped region of interest (ROI) containing the resolution target.
            Returns None if the resolution target is not detected due to insufficient matches.
        """
        # template_path = "USAF1951_template/res_source.png"
        template_path = '../rta/roi_extraction/USAF1951_template/res_source.png'
        t_pad, b_pad, l_pad, r_pad = 20, 30, 25, 30

        # Load the template image
        if os.path.exists(template_path):
            template = cv2.imread(template_path, 0)
        else:
            template = resolution_template()

        # Convert the im_src image to uint8 if necessary
        numpy_image = self.cvt_to_uint8(im_src)

        # Convert both the template and the image to grayscale for feature detection
        img_gray = cv2.cvtColor(numpy_image, cv2.COLOR_BGR2GRAY) if len(numpy_image.shape) >= 3 else numpy_image
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY) if len(template.shape) >= 3 else template

        # Detect keypoints and descriptors in both the template and the image
        brisk = cv2.BRISK_create()
        kp1, des1 = brisk.detectAndCompute(img_gray, None)
        kp2, des2 = brisk.detectAndCompute(template_gray, None)

        # Match descriptors
        bf = cv2.BFMatcher()
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        # Apply ratio test to filter good matches
        good_matches = []
        for m in matches:
            if m.distance < min_kp_dist_threshold * matches[-1].distance:
                good_matches.append(m)

        # Confidence level check
        if len(good_matches) < min_good_matches:
            # print("Resolution target not detected. Insufficient good matches.")
            return None

        # Extract matching points
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Compute homography matrix
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # Convert OpenCV Homography matrix to scikit-image format
        tform = sk_transform.ProjectiveTransform(M)

        # Warp the source image using scikit-image
        im_src_warped = sk_transform.warp(im_src, tform.inverse, output_shape=(im_src.shape[0], im_src.shape[1]), preserve_range=True)
        im_src_warped = im_src_warped.astype(im_src.dtype)

        # Crop the ROI using keypoints
        roi = self.crop_using_keypoints(im_src_warped, dst_pts, padding_percentage=0.2)

        # Save the cropped image if save_cropped_im is provided
        if save_cropped_im is not None:
            # Extract the file extension
            base, ext = os.path.splitext(save_cropped_im)
            
            # Default to '.tiff' if no valid extension is provided
            if ext == "":
                ext = ".tiff"

            # Define the save path
            save_path = base + ext

            # Save the image
            io.imsave(save_path, roi)
            print(f"Saved cropped image at: {save_path}")
        
        if show_kp:
            # Draw keypoints and matches for visualization
            img_keypoints = cv2.drawKeypoints(img_gray, kp1, None, color=(255, 0, 0))
            template_keypoints = cv2.drawKeypoints(template_gray, kp2, None, color=(255, 0, 0))
            img_matches = cv2.drawMatches(img_gray, kp1, template_gray, kp2, good_matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 3, 1)
            plt.title("Image Keypoints")
            plt.imshow(img_keypoints, cmap='gray')
            plt.subplot(1, 3, 2)
            plt.title("Template Keypoints")
            plt.imshow(template_keypoints, cmap='gray')
            plt.subplot(1, 3, 3)
            plt.title("Matches")
            plt.imshow(img_matches)
            plt.show()

        return roi

    def crop_using_keypoints(self, im, dst_pts, padding_percentage=0.2):
        """
        Automatically crop the region of interest using keypoints.
        
        Parameters:
        - im: The source image.
        - dst_pts: The destination points (keypoints) in the image.
        - padding: Optional padding to apply around the bounding box.
        - margin_scale: Optional scale factor to increase the bounding box size.
        
        Returns:
        - Cropped region of interest.
        """

        # Compute bounding box from dst_pts
        x_coords, y_coords = dst_pts[:, :, 0], dst_pts[:, :, 1]
        min_x, max_x = np.min(x_coords), np.max(x_coords)
        min_y, max_y = np.min(y_coords), np.max(y_coords)

        # Calculate the center coordinates of the bounding box
        center_x = (min_x + max_x) // 2
        center_y = (min_y + max_y) // 2

        # Calculate the width and height of the bounding box
        bbox_width = max_x - min_x
        bbox_height = max_y - min_y

        # Calculate the maximum dimension of the bounding box
        max_dimension = max(bbox_width, bbox_height)
        padding = int(max_dimension * padding_percentage)


        # Apply the margin and padding evenly in all directions from the center
        min_x = max(center_x - (max_dimension // 2) - padding, 0)
        max_x = min(center_x + (max_dimension // 2) + padding, im.shape[1])
        min_y = max(center_y - (max_dimension // 2) - padding, 0)
        max_y = min(center_y + (max_dimension // 2) + padding, im.shape[0])

        # Crop the ROI based on the updated bounding box coordinates
        roi = im[int(min_y):int(max_y), int(min_x):int(max_x)]

        return roi

    def preprocess_roi(self, roi):
        # Convert the cropped ROI to float32
        roi_float32 = roi.astype(np.float32)

        # Normalize the cropped ROI to the range [0, 1]
        roi_normalized = (roi_float32 - np.min(roi_float32)) / (np.max(roi_float32) - np.min(roi_float32))

        # Scale the normalized ROI to the range [0, 255] and convert to uint8
        roi_uint8 = (roi_normalized * 255).astype(np.uint8)

        # Invert the cropped ROI
        roi_inverted = cv2.bitwise_not(roi_uint8)

        return roi_inverted

    def preprocess_template(self, template_path):
        # Read the template image
        template = io.imread(template_path)

        # Normalize the template to 8-bit range (0-255)
        template_normalized = img_as_ubyte(img_as_float(template))

        # Invert the template image to handle white features on a black background
        template_inverted = cv2.bitwise_not(template_normalized)

        return template_inverted

    def perform_template_matching(self, roi_inverted, template_inverted):
        # Define the scale factors for multi-scale template matching
        scales = np.linspace(0.1, 1.0, 20)[::-1]

        # Perform multi-scale template matching
        found = None
        for scale in scales:
            new_width = int(roi_inverted.shape[1] * scale)
            new_height = int(roi_inverted.shape[0] * scale)

            if new_width < template_inverted.shape[1] or new_height < template_inverted.shape[0]:
                continue

            resized = cv2.resize(roi_inverted, (new_width, new_height))
            r = roi_inverted.shape[1] / float(resized.shape[1])

            # Convert resized and template_inverted to the same data type (CV_8U)
            resized = resized.astype(np.uint8)
            template_inverted = template_inverted.astype(np.uint8)

            result = cv2.matchTemplate(resized, template_inverted, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            if found is None or max_val > found[0]:
                found = (max_val, max_loc, r)

        return found

    def extract_matched_region(self, roi_inverted, found, template_inverted):
        if found is not None:
            _, max_loc, r = found
            x, y = max_loc
            w, h = int(template_inverted.shape[1] * r), int(template_inverted.shape[0] * r)

            x, y = int(x * r), int(y * r)

            # Extract the matched region from the cropped ROI
            matched_region = roi_inverted[y:y+h, x:x+w]

            return matched_region, x, y
        else:
            print("Template not found in the image.")
            return None, None, None

    def visualize_matched_region(self, roi_inverted, matched_region, x, y, template_inverted):
        w, h = template_inverted.shape[1], template_inverted.shape[0]
        # Draw the bounding box on the cropped ROI for visualization
        visualization = cv2.cvtColor(roi_inverted, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(visualization, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the cropped ROI with the bounding box using Matplotlib
        plt.figure(figsize=(8, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(visualization, cv2.COLOR_BGR2RGB), cmap='gray')
        plt.title('Bounding Box')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(matched_region, cmap='gray')
        plt.title('Matched Region')
        plt.axis('off')

        plt.tight_layout()
        plt.show()