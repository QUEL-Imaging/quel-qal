# rrt_roi.py

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import io, transform as sk_transform

from qal.data import resolution_template


class RrtROI:
    # Trial order for comparing USAF-target feature matchers. BRISK/AKAZE are
    # the closest classical pair, but they are not in opencv-python 5.x.
    FEATURE_DETECTOR_TRIAL_ORDER = ("brisk", "akaze", "orb", "sift", "affine_sift")

    @staticmethod
    def create_feature_detector(name):
        """Create a Feature2D detector and the BFMatcher norm it expects.

        Parameters:
        - name (str): One of ``brisk``, ``akaze``, ``orb``, ``sift``,
          ``affine_sift``.

        Returns:
        - tuple: ``(detector, norm)`` where ``norm`` is ``cv2.NORM_HAMMING``
          for binary descriptors and ``cv2.NORM_L2`` for float descriptors.

        Raises:
        - ValueError: If ``name`` is not a supported detector.
        - AttributeError: If the detector is not in this OpenCV build.
        """
        key = name.lower().replace("-", "_")

        def factory(attr_name):
            for namespace in (cv2, getattr(cv2, "xfeatures2d", None)):
                if namespace is None:
                    continue
                if hasattr(namespace, attr_name):
                    return getattr(namespace, attr_name)
                feature_class = getattr(namespace, attr_name.removesuffix("_create"), None)
                if feature_class is not None and hasattr(feature_class, "create"):
                    return feature_class.create
            raise AttributeError(attr_name)

        creators = {
            "brisk": lambda: (factory("BRISK_create")(), cv2.NORM_HAMMING),
            "akaze": lambda: (factory("AKAZE_create")(), cv2.NORM_HAMMING),
            "orb": lambda: (factory("ORB_create")(nfeatures=4000), cv2.NORM_HAMMING),
            "sift": lambda: (factory("SIFT_create")(), cv2.NORM_L2),
            "affine_sift": lambda: (
                factory("AffineFeature_create")(factory("SIFT_create")()),
                cv2.NORM_L2,
            ),
        }
        if key not in creators:
            supported = ", ".join(RrtROI.FEATURE_DETECTOR_TRIAL_ORDER)
            raise ValueError(f"Unknown feature detector {name!r}. Supported: {supported}")
        try:
            return creators[key]()
        except AttributeError as exc:
            raise AttributeError(
                f"{key} is not available in this OpenCV build "
                f"(cv2 {cv2.__version__}). On OpenCV 5, BRISK and AKAZE live in "
                "opencv-contrib-python as cv2.xfeatures2d."
            ) from exc

    def cvt_to_uint8(self, img):
        image = img.astype(float) - np.nanmin(img)
        image = 255 * (image.astype(float) / np.max(image))
        image = image.astype(np.uint8)
        return image

    def get_resolution_target_cropped(
        self,
        im_src,
        show_kp=False,
        min_good_matches=10,
        min_kp_dist_threshold=0.45,
        save_cropped_im=None,
        detector="brisk",
        return_debug=False,
    ):
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

        - detector (str, optional):
            Feature detector/descriptor used for matching. One of ``brisk``,
            ``akaze``, ``orb``, ``sift``, ``affine_sift``. Default is ``brisk``.

        - return_debug (bool, optional):
            If True, return a dict of match statistics and visualization
            images instead of only the cropped ROI.

        Returns:
        - roi (numpy.ndarray or None): 
            The cropped region of interest (ROI) containing the resolution target.
            Returns None if the resolution target is not detected due to insufficient matches.
            If ``return_debug`` is True, returns a dict that includes ``roi``.
        """
        detector_name = detector.lower().replace("-", "_")
        debug = {
            "detector": detector_name,
            "available": False,
            "error": None,
            "n_keypoints_src": 0,
            "n_keypoints_template": 0,
            "n_matches": 0,
            "n_good_matches": 0,
            "n_inliers": 0,
            "roi": None,
            "img_keypoints": None,
            "template_keypoints": None,
            "img_matches": None,
            "homography": None,
            "crop_offset": None,
        }

        # template_path = "USAF1951_template/res_source.png"
        template_path = '../rta/roi_extraction/USAF1951_template/res_source.png'

        # Load the template image
        if os.path.exists(template_path):
            template = cv2.imread(template_path, 0)
        else:
            template = resolution_template()

        # The stored-value loader preserves uint16 templates. OpenCV feature
        # descriptors need uint8 for both inputs, not just the source image.
        template = self.cvt_to_uint8(template)

        # Convert the im_src image to uint8 if necessary
        numpy_image = self.cvt_to_uint8(im_src)

        # Convert both the template and the image to grayscale for feature detection
        img_gray = cv2.cvtColor(numpy_image, cv2.COLOR_BGR2GRAY) if len(numpy_image.shape) >= 3 else numpy_image
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY) if len(template.shape) >= 3 else template

        try:
            feature_detector, norm = self.create_feature_detector(detector_name)
        except (AttributeError, ValueError) as exc:
            debug["error"] = str(exc)
            if return_debug:
                return debug
            raise

        debug["available"] = True

        empty_kp = []
        kp1, des1 = feature_detector.detectAndCompute(img_gray, None)
        kp2, des2 = feature_detector.detectAndCompute(template_gray, None)
        kp1 = empty_kp if kp1 is None else kp1
        kp2 = empty_kp if kp2 is None else kp2
        debug["n_keypoints_src"] = len(kp1)
        debug["n_keypoints_template"] = len(kp2)

        matches = []
        if des1 is not None and des2 is not None and len(des1) and len(des2):
            bf = cv2.BFMatcher(norm)
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)
        debug["n_matches"] = len(matches)

        # Apply ratio test to filter good matches
        good_matches = []
        if matches:
            worst_distance = matches[-1].distance
            for m in matches:
                if m.distance < min_kp_dist_threshold * worst_distance:
                    good_matches.append(m)
        debug["n_good_matches"] = len(good_matches)

        img_keypoints = cv2.drawKeypoints(img_gray, kp1, None, color=(255, 0, 0))
        template_keypoints = cv2.drawKeypoints(template_gray, kp2, None, color=(255, 0, 0))
        img_matches = cv2.drawMatches(
            img_gray,
            kp1,
            template_gray,
            kp2,
            good_matches[:10],
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        debug["img_keypoints"] = img_keypoints
        debug["template_keypoints"] = template_keypoints
        debug["img_matches"] = img_matches

        roi = None
        if len(good_matches) >= min_good_matches:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            debug["n_inliers"] = 0 if mask is None else int(mask.sum())

            if M is not None:
                tform = sk_transform.ProjectiveTransform(M)
                im_src_warped = sk_transform.warp(
                    im_src,
                    tform.inverse,
                    output_shape=(im_src.shape[0], im_src.shape[1]),
                    preserve_range=True,
                )
                im_src_warped = im_src_warped.astype(im_src.dtype)
                roi = self.crop_using_keypoints(im_src_warped, dst_pts, padding_percentage=0.2)
                min_x, _, min_y, _ = self._keypoint_crop_box(
                    im_src.shape, dst_pts, padding_percentage=0.2
                )
                debug["homography"] = M
                debug["crop_offset"] = (min_x, min_y)

        debug["roi"] = roi

        if save_cropped_im is not None and roi is not None:
            base, ext = os.path.splitext(save_cropped_im)
            if ext == "":
                ext = ".tiff"
            save_path = base + ext
            io.imsave(save_path, roi)
            print(f"Saved cropped image at: {save_path}")

        if show_kp and img_matches is not None:
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

        if return_debug:
            return debug
        return roi

    def crop_resolution_target(self, im_src, **kwargs):
        """Crop the chart for display and keep the mapping to ``im_src``.

        The crop is a perspective-warped, resampled image; use it to place
        lines, then map them back with
        :meth:`ResolutionTargetCrop.selection_to_source` so profiles are
        measured on the original pixels. Keyword arguments go to
        :meth:`get_resolution_target_cropped`.

        Returns ``None`` when the chart is not found.
        """
        from .rrt_gui import ResolutionTargetCrop

        debug = self.get_resolution_target_cropped(
            im_src, return_debug=True, **kwargs
        )
        if debug.get("roi") is None:
            return None
        return ResolutionTargetCrop(
            image=debug["roi"],
            source=im_src,
            homography=np.asarray(debug["homography"], dtype=float),
            offset=tuple(float(value) for value in debug["crop_offset"]),
        )

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

        min_x, max_x, min_y, max_y = self._keypoint_crop_box(
            im.shape, dst_pts, padding_percentage
        )
        return im[min_y:max_y, min_x:max_x]

    @staticmethod
    def _keypoint_crop_box(shape, dst_pts, padding_percentage=0.2):
        """Integer ``(min_x, max_x, min_y, max_y)`` used by the keypoint crop."""
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
        max_x = min(center_x + (max_dimension // 2) + padding, shape[1])
        min_y = max(center_y - (max_dimension // 2) - padding, 0)
        max_y = min(center_y + (max_dimension // 2) + padding, shape[0])
        return int(min_x), int(max_x), int(min_y), int(max_y)

    @staticmethod
    def print_smallest_resolved(contrast, threshold):
        """Print the finest bar pattern still at or above ``threshold`` percent.

        ``contrast`` is the table from ``RrtAnalyzer.load_and_process_groups``.
        Rows are read in order, so the last row at or above the threshold is
        the finest resolved feature.
        """
        resolved = contrast[contrast["Percentage Contrast"] >= threshold]
        if resolved.empty:
            print(f"\nAll elements are below {threshold}% contrast.")
            return

        finest = resolved.iloc[-1]
        print(
            f"\nSmallest feature above {threshold}% contrast: "
            f"Group {int(finest['Group'])}, Element {int(finest['Element'])}, "
            f"{finest['Resolution (lp/mm)']:.2f} lp/mm"
        )
