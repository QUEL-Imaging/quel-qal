"""Image loading and conversion helpers shared across QAL workflows."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Union

import numpy as np
from scipy.ndimage import label
from skimage import io
from skimage.color import rgb2gray

ImageInput = Union[str, Path, np.ndarray]
GrayscaleMethod = Union[Literal["average", "rgb2gray"], Callable[[np.ndarray], np.ndarray]]


@dataclass(frozen=True)
class StoredValueInfo:
    """Container dtype versus the range actually occupied by stored samples.

    A 16-bit TIFF whose samples only occupy 0–255 is still ``uint16``;
    it is not scaled to 0–65535.
    """

    dtype: np.dtype
    vmin: float
    vmax: float
    container_bits: int | None
    used_bits: int | None
    is_8bit_in_16bit: bool

    def summary(self) -> str:
        dtype_name = np.dtype(self.dtype).name
        range_text = f"stored values {self.vmin:g}–{self.vmax:g}"
        if self.is_8bit_in_16bit:
            return (
                f"{dtype_name} container, {range_text} "
                "(8-bit samples in a 16-bit file; not scaled to 0–65535)"
            )
        if (
            self.container_bits is not None
            and self.used_bits is not None
            and self.used_bits < self.container_bits
        ):
            return (
                f"{dtype_name} container ({self.container_bits}-bit), "
                f"{range_text} (effective {self.used_bits}-bit samples)"
            )
        return f"{dtype_name}, {range_text}"


def describe_stored_values(image: np.ndarray) -> StoredValueInfo:
    """Describe the stored sample range without rescaling the image."""
    array = np.asarray(image)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        raise ValueError("Image contains no finite samples")

    vmin = float(np.min(finite))
    vmax = float(np.max(finite))
    dtype = array.dtype
    container_bits = None
    used_bits = None
    if np.issubdtype(dtype, np.integer):
        container_bits = int(np.iinfo(dtype).bits)
        peak = max(abs(vmin), abs(vmax))
        used_bits = int(np.ceil(np.log2(peak + 1))) if peak > 0 else 0

    is_8bit_in_16bit = (
        np.issubdtype(dtype, np.unsignedinteger)
        and container_bits == 16
        and vmin >= 0
        and vmax <= 255
    )
    return StoredValueInfo(
        dtype=dtype,
        vmin=vmin,
        vmax=vmax,
        container_bits=container_bits,
        used_bits=used_bits,
        is_8bit_in_16bit=is_8bit_in_16bit,
    )


def to_log_scale(image: np.ndarray) -> np.ndarray:
    """Return a 16-bit log-compressed copy for display and detection.

    This is the stretch used by well detection, so ROI overlays on the
    log image line up with ``WellDetector.plot_detected_wells``.
    """
    uint16_max = 65535
    scaled = np.asarray(image, dtype=float) - np.nanmin(image)
    peak = np.nanmax(scaled)
    if not np.isfinite(peak) or peak <= 0:
        return np.zeros(np.asarray(image).shape[:2], dtype=np.uint16)
    scaled = uint16_max * (scaled / peak)
    c = uint16_max / np.log(1 + np.max(scaled))
    return np.array(c * np.log(scaled + 1.0), dtype=np.uint16)


def load_grayscale_image(image: ImageInput) -> np.ndarray:
    """Return stored grayscale samples from a path or array.

    Values are read with :func:`skimage.io.imread` and are not rescaled by
    container bit depth. A 16-bit file whose samples only occupy 0–255 is
    returned as ``uint16`` with those values.

    RGB/RGBA images are reduced to their first channel. For multi-page image
    stacks, the first page is used.
    """
    im = image if isinstance(image, np.ndarray) else io.imread(str(image))
    im = np.squeeze(np.asarray(im))

    if im.ndim == 3:
        im = im[..., 0] if im.shape[-1] in (3, 4) else im[0]
    if im.ndim != 2:
        raise ValueError(f"Expected a 2-D grayscale image, got shape {im.shape}")

    return im


def to_grayscale(image: np.ndarray, method: GrayscaleMethod = "average") -> np.ndarray:
    """Convert an RGB or RGBA image to grayscale using the selected method.

    Two-dimensional grayscale images are returned unchanged. For the built-in
    methods, an alpha channel is ignored and the input dtype and intensity range
    are preserved. A custom callable receives the original RGB or RGBA image and
    must return a two-dimensional array.

    Parameters
    ----------
    image : numpy.ndarray
        A two-dimensional grayscale image or an RGB/RGBA image with channels in
        the final dimension.
    method : {"average", "rgb2gray"} or callable, optional
        ``"average"`` computes the unweighted mean of the RGB channels.
        ``"rgb2gray"`` uses :func:`skimage.color.rgb2gray`. A callable can be
        supplied for a custom conversion. The default is ``"average"``.

    Returns
    -------
    numpy.ndarray
        A two-dimensional grayscale image.

    Raises
    ------
    TypeError
        If ``image`` is not a NumPy array or a custom converter does not return
        one.
    ValueError
        If the image shape or conversion method is unsupported, or if a custom
        converter does not return a two-dimensional image of the expected size.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a NumPy array")

    if image.ndim == 2:
        return image

    if image.ndim != 3 or image.shape[-1] not in (3, 4):
        raise ValueError(
            "expected a 2D grayscale image or an RGB/RGBA image with shape "
            f"(height, width, 3 or 4); received {image.shape}"
        )

    if callable(method):
        grayscale = method(image)
        if not isinstance(grayscale, np.ndarray):
            raise TypeError("custom grayscale converter must return a NumPy array")
    else:
        rgb = image[..., :3]
        if method == "average":
            grayscale = np.mean(rgb.astype(np.float64), axis=-1)
            grayscale = _restore_average_dtype(grayscale, image.dtype)
        elif method == "rgb2gray":
            grayscale = rgb2gray(rgb)
            grayscale = _restore_rgb2gray_dtype(grayscale, image.dtype)
        else:
            raise ValueError(
                "grayscale conversion method must be 'average', 'rgb2gray', or a callable"
            )

    if grayscale.ndim != 2 or grayscale.shape != image.shape[:2]:
        raise ValueError(
            "grayscale converter must return an array with shape "
            f"{image.shape[:2]}; received {grayscale.shape}"
        )

    return grayscale


def remove_corner_overlays(
    image: np.ndarray,
    *,
    corners=("top_left", "top_right", "bottom_left", "bottom_right"),
    corner_fraction=0.25,
    color_threshold=0.15,
    high_intensity_threshold=0.65,
    high_intensity_channels=(0, 1, 2),
    min_component_area=0.0001,
    padding=5,
    replacement="zero",
    return_bboxes=False,
):
    """Detect and replace likely graphical overlays in color-image corners.

    Overlay candidates are pixels with either strong RGB channel differences
    (color) or high intensity in selected channels. Connected components smaller
    than ``min_component_area`` are ignored, which helps preserve small target
    features. Qualifying components in each corner are enclosed by one bounding
    box and replaced.

    Parameters
    ----------
    image : numpy.ndarray
        RGB or RGBA image with channels in the final dimension.
    corners : iterable of str, optional
        Corners to inspect. Accepted values are ``"top_left"``,
        ``"top_right"``, ``"bottom_left"``, and ``"bottom_right"``.
    corner_fraction : float or tuple of float, optional
        Fraction of image height and width searched in each corner. A scalar
        applies to both dimensions. Each value must be in ``(0, 0.5]``.
    color_threshold : float or None, optional
        Minimum normalized difference between the brightest and darkest RGB
        channels. Set to ``None`` to disable color-based detection.
    high_intensity_threshold : float or None, optional
        Minimum normalized intensity in any selected channel. Set to ``None``
        to disable intensity-based detection.
    high_intensity_channels : iterable of int or str, optional
        RGB channels used for high-intensity detection. Channels may be indices
        0-2 or the names ``"red"``, ``"green"``, and ``"blue"``.
    min_component_area : int or float, optional
        Minimum candidate component area. An integer is interpreted as pixels;
        a float in ``(0, 1)`` is interpreted as a fraction of total image area.
        A minimum floor of 16 pixels is used for fractional values.
    padding : int, optional
        Pixels added around each detected corner bounding box.
    replacement : {"zero", "mean", "median"}, number, or RGB sequence, optional
        Replacement for detected boxes. Mean and median are calculated from the
        grayscale values outside all detected boxes. Numeric scalar values are
        applied equally to RGB channels; a three-value sequence supplies RGB.
    return_bboxes : bool, optional
        If True, return ``(cleaned_image, bounding_boxes)``. Each bounding-box
        record contains its corner and ``(top, left, bottom, right)`` coordinates,
        where bottom and right are exclusive.

    Returns
    -------
    numpy.ndarray or tuple
        Cleaned image, optionally accompanied by detected bounding boxes.
    """
    _validate_color_image(image)
    search_height_fraction, search_width_fraction = _normalize_corner_fraction(corner_fraction)
    selected_corners = _validate_corners(corners)
    selected_channels = (
        _normalize_rgb_channels(high_intensity_channels)
        if high_intensity_threshold is not None
        else None
    )
    component_area = _resolve_component_area(min_component_area, image.shape[:2])

    if color_threshold is None and high_intensity_threshold is None:
        raise ValueError("at least one overlay detection threshold must be enabled")
    _validate_normalized_threshold(color_threshold, "color_threshold")
    _validate_normalized_threshold(high_intensity_threshold, "high_intensity_threshold")
    if not isinstance(padding, (int, np.integer)) or padding < 0:
        raise ValueError("padding must be a non-negative integer")

    rgb_normalized = _normalize_rgb_for_detection(image[..., :3])
    image_height, image_width = image.shape[:2]
    search_height = max(1, int(np.ceil(image_height * search_height_fraction)))
    search_width = max(1, int(np.ceil(image_width * search_width_fraction)))

    boxes = []
    for corner in selected_corners:
        top, left, bottom, right = _corner_region(
            corner, image_height, image_width, search_height, search_width
        )
        region = rgb_normalized[top:bottom, left:right]
        candidate_mask = np.zeros(region.shape[:2], dtype=bool)

        if color_threshold is not None:
            channel_range = np.max(region, axis=-1) - np.min(region, axis=-1)
            candidate_mask |= channel_range >= color_threshold
        if high_intensity_threshold is not None:
            candidate_mask |= np.any(
                region[..., selected_channels] >= high_intensity_threshold,
                axis=-1,
            )

        labeled, component_count = label(candidate_mask)
        component_boxes = []
        for component_id in range(1, component_count + 1):
            component_rows, component_cols = np.nonzero(labeled == component_id)
            if len(component_rows) < component_area:
                continue
            component_boxes.append((
                top + int(component_rows.min()),
                left + int(component_cols.min()),
                top + int(component_rows.max()) + 1,
                left + int(component_cols.max()) + 1,
            ))

        if component_boxes:
            box_top = max(0, min(box[0] for box in component_boxes) - padding)
            box_left = max(0, min(box[1] for box in component_boxes) - padding)
            box_bottom = min(image_height, max(box[2] for box in component_boxes) + padding)
            box_right = min(image_width, max(box[3] for box in component_boxes) + padding)
            boxes.append({
                "corner": corner,
                "bbox": (box_top, box_left, box_bottom, box_right),
            })

    cleaned = image.copy()
    if boxes:
        fill_value = _resolve_overlay_replacement(image, boxes, replacement)
        for box in boxes:
            top, left, bottom, right = box["bbox"]
            cleaned[top:bottom, left:right, :3] = fill_value

    if return_bboxes:
        return cleaned, boxes
    return cleaned


def _restore_average_dtype(image: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """Restore the source dtype after an averaging conversion."""
    if np.issubdtype(dtype, np.bool_):
        return image >= 0.5
    if np.issubdtype(dtype, np.integer):
        limits = np.iinfo(dtype)
        return np.rint(np.clip(image, limits.min, limits.max)).astype(dtype)
    if np.issubdtype(dtype, np.floating):
        return image.astype(dtype)
    raise TypeError(f"unsupported image dtype: {dtype}")


def _restore_rgb2gray_dtype(image: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """Restore the source dtype and intensity range after ``rgb2gray``."""
    if np.issubdtype(dtype, np.bool_):
        return image >= 0.5
    if np.issubdtype(dtype, np.unsignedinteger):
        maximum = np.iinfo(dtype).max
        return np.rint(np.clip(image, 0, 1) * maximum).astype(dtype)
    if np.issubdtype(dtype, np.signedinteger):
        limits = np.iinfo(dtype)
        scaled = np.where(image >= 0, image * limits.max, image * -limits.min)
        return np.rint(np.clip(scaled, limits.min, limits.max)).astype(dtype)
    if np.issubdtype(dtype, np.floating):
        return image.astype(dtype)
    raise TypeError(f"unsupported image dtype: {dtype}")


def _validate_color_image(image: np.ndarray) -> None:
    """Require a non-empty RGB or RGBA array with a numeric dtype."""
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a NumPy array")
    if image.ndim != 3 or image.shape[-1] not in (3, 4):
        raise ValueError(
            "expected an RGB or RGBA image with shape (height, width, 3 or 4); "
            f"received {image.shape}"
        )
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("image dimensions must be non-zero")
    if not (
        np.issubdtype(image.dtype, np.bool_)
        or np.issubdtype(image.dtype, np.integer)
        or np.issubdtype(image.dtype, np.floating)
    ):
        raise TypeError(f"unsupported image dtype: {image.dtype}")


def _normalize_corner_fraction(corner_fraction):
    """Return ``(height_fraction, width_fraction)`` in ``(0, 0.5]``."""
    if np.isscalar(corner_fraction):
        fractions = (float(corner_fraction), float(corner_fraction))
    else:
        try:
            fractions = tuple(float(value) for value in corner_fraction)
        except TypeError as exc:
            raise TypeError("corner_fraction must be a number or a two-value iterable") from exc
        if len(fractions) != 2:
            raise ValueError("corner_fraction must contain height and width fractions")
    if any(value <= 0 or value > 0.5 for value in fractions):
        raise ValueError("corner_fraction values must be in (0, 0.5]")
    return fractions


def _validate_corners(corners):
    """Return unique accepted corner names in input order."""
    accepted = {"top_left", "top_right", "bottom_left", "bottom_right"}
    if isinstance(corners, str):
        corners = (corners,)
    corners = tuple(corners)
    if not corners:
        raise ValueError("at least one corner must be selected")
    invalid = set(corners) - accepted
    if invalid:
        raise ValueError(f"unsupported corners: {sorted(invalid)}")
    return tuple(dict.fromkeys(corners))


def _normalize_rgb_channels(channels):
    """Return unique RGB channel indices from indices or color names."""
    channel_names = {"red": 0, "green": 1, "blue": 2}
    if isinstance(channels, (str, int, np.integer)):
        channels = (channels,)
    normalized = []
    for channel in channels:
        if isinstance(channel, str):
            try:
                channel = channel_names[channel.lower()]
            except KeyError as exc:
                raise ValueError(f"unsupported RGB channel: {channel}") from exc
        if not isinstance(channel, (int, np.integer)) or channel not in (0, 1, 2):
            raise ValueError("RGB channels must be 0, 1, 2, red, green, or blue")
        normalized.append(int(channel))
    if not normalized:
        raise ValueError("at least one high-intensity channel must be selected")
    return tuple(dict.fromkeys(normalized))


def _resolve_component_area(min_component_area, image_shape):
    """Convert a pixel count or image-area fraction into a pixel threshold."""
    if isinstance(min_component_area, (int, np.integer)):
        if min_component_area <= 0:
            raise ValueError("min_component_area must be greater than zero")
        return int(min_component_area)
    if isinstance(min_component_area, (float, np.floating)):
        if min_component_area <= 0 or min_component_area >= 1:
            raise ValueError("fractional min_component_area must be in (0, 1)")
        return max(16, int(np.ceil(min_component_area * np.prod(image_shape))))
    raise TypeError("min_component_area must be a positive pixel count or image-area fraction")


def _validate_normalized_threshold(value, name):
    """Require ``None`` or a scalar threshold in ``[0, 1]``."""
    if value is not None and (not np.isscalar(value) or value < 0 or value > 1):
        raise ValueError(f"{name} must be None or a value in [0, 1]")


def _normalize_rgb_for_detection(rgb):
    """Scale RGB samples to ``[0, 1]`` for overlay detection."""
    source_dtype = rgb.dtype
    rgb = rgb.astype(np.float64)
    if np.issubdtype(source_dtype, np.bool_):
        return rgb

    if np.issubdtype(source_dtype, np.unsignedinteger):
        return rgb / np.iinfo(source_dtype).max
    if np.issubdtype(source_dtype, np.signedinteger):
        limits = np.iinfo(source_dtype)
        return (rgb - limits.min) / (limits.max - limits.min)

    finite = rgb[np.isfinite(rgb)]
    if finite.size == 0:
        raise ValueError("image contains no finite RGB values")
    minimum = float(np.min(finite))
    maximum = float(np.max(finite))
    if minimum >= 0 and maximum <= 1:
        return np.nan_to_num(rgb, nan=0.0, posinf=1.0, neginf=0.0)
    if maximum == minimum:
        return np.zeros_like(rgb)
    normalized = (rgb - minimum) / (maximum - minimum)
    return np.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=0.0)


def _corner_region(corner, image_height, image_width, search_height, search_width):
    """Return the exclusive ``(top, left, bottom, right)`` search window."""
    if corner.startswith("top"):
        top, bottom = 0, search_height
    else:
        top, bottom = image_height - search_height, image_height
    if corner.endswith("left"):
        left, right = 0, search_width
    else:
        left, right = image_width - search_width, image_width
    return top, left, bottom, right


def _resolve_overlay_replacement(image, boxes, replacement):
    """Return an RGB fill matching the source image dtype."""
    if isinstance(replacement, str):
        if replacement == "zero":
            value = 0.0
        elif replacement in ("mean", "median"):
            excluded = np.zeros(image.shape[:2], dtype=bool)
            for box in boxes:
                top, left, bottom, right = box["bbox"]
                excluded[top:bottom, left:right] = True
            grayscale = np.mean(image[..., :3].astype(np.float64), axis=-1)
            background = grayscale[~excluded & np.isfinite(grayscale)]
            if background.size == 0:
                raise ValueError("cannot calculate replacement from an entirely excluded image")
            value = np.mean(background) if replacement == "mean" else np.median(background)
        else:
            raise ValueError("replacement must be 'zero', 'mean', 'median', a number, or an RGB sequence")
    else:
        value = replacement

    values = np.asarray(value)
    if values.ndim == 0:
        values = np.repeat(values, 3)
    elif values.shape != (3,):
        raise ValueError("a replacement sequence must contain exactly three RGB values")

    if not np.issubdtype(values.dtype, np.number):
        raise TypeError("replacement values must be numeric")
    values = values.astype(np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("replacement values must be finite")

    if np.issubdtype(image.dtype, np.bool_):
        return values >= 0.5
    if np.issubdtype(image.dtype, np.integer):
        limits = np.iinfo(image.dtype)
        return np.rint(np.clip(values, limits.min, limits.max)).astype(image.dtype)
    return values.astype(image.dtype)
