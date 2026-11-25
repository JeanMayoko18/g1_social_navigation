"""
Extract upper/lower appearance measurements from a source RGB image.

Receives:
    RGB uint8 image matching the source detection.
    Relative body-region definitions.
    Histogram configuration.
    Explicit external reliability for each region.
    Optional person-specific foreground mask.

Calculates:
    Region coordinates relative to the original detection box.
    Pixel support after image clipping and optional masking.
    Normalized chromatic and achromatic histogram bins.

Quality policy:
    support = selected_pixel_count / intended_region_pixel_count
    quality = support * external_reliability

Returns:
    Upper and lower Measurement[Histogram] objects.

Boundary:
    No identity association or image synchronization occurs here.
    No automatic occlusion or segmentation model is implemented.
    Without a foreground mask, background pixels may enter the descriptor.
"""


import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .types import (
    Histogram,
    Measurement,
    SourceDetection,
)
from .validation import (
    integer,
    positive,
    unit_interval,
)


# ============================================================================
# EXTRACTION CONFIGURATION
# ============================================================================


@dataclass(frozen=True)
class RelativeRegion:
    """
    Region boundaries as fractions of the original detection box.

    Fields:
        left, top, right, bottom in [0, 1].

    Equations:
        region_left = bbox_left + left * bbox_width
        region_top = bbox_top + top * bbox_height

    Right and bottom follow the same construction.

    Define the region before clipping to the image so a truncated
    detection does not silently redefine the selected body region.
    """

    left: float
    top: float
    right: float
    bottom: float

    def __post_init__(self):
        for name in ("left", "top", "right", "bottom"):
            unit_interval(name, getattr(self, name))

        if self.left >= self.right or self.top >= self.bottom:
            raise ValueError("Region must have positive width and height")


@dataclass(frozen=True)
class AppearancePolicy:
    """
    Configuration defining descriptor semantics and eligibility.

    Chromatic pixels:
        Quantized into hue/saturation/value bins.

    Achromatic pixels:
        Low-saturation or dark pixels, represented by value bins.
        Hue is not treated as reliable for these pixels.

    minimum_pixels:
        Minimum selected pixels needed to form a descriptor.

    minimum_support:
        Minimum fraction of the intended region supported by selected pixels.

    Region definitions and binning parameters are included in schema IDs.
    Eligibility thresholds do not change histogram bin semantics.
    """

    upper: RelativeRegion
    lower: RelativeRegion

    hue_bins: int
    saturation_bins: int
    value_bins: int
    achromatic_bins: int

    minimum_saturation: float
    minimum_value: float

    minimum_pixels: int
    minimum_support: float

    def __post_init__(self):
        if not isinstance(self.upper, RelativeRegion):
            raise ValueError("upper must be RelativeRegion")

        if not isinstance(self.lower, RelativeRegion):
            raise ValueError("lower must be RelativeRegion")

        for name in (
            "hue_bins",
            "saturation_bins",
            "value_bins",
            "achromatic_bins",
            "minimum_pixels",
        ):
            integer(name, getattr(self, name), minimum=1)

        unit_interval("minimum_saturation", self.minimum_saturation)
        unit_interval("minimum_value", self.minimum_value)
        unit_interval("minimum_support", self.minimum_support)
        positive("minimum_support", self.minimum_support)


# ============================================================================
# REGION GEOMETRY AND SCHEMA
# ============================================================================


def _region_bounds(source, region):
    """
    Convert a relative region to integer image bounds.

    Returns:
        Left/top inclusive and right/bottom exclusive coordinates.

    Floor/ceil include the pixels intersecting the continuous region.
    These original bounds define the intended support area.
    """
    bbox = source.bbox

    box_left = bbox.center_u - bbox.width / 2.0
    box_top = bbox.center_v - bbox.height / 2.0

    return (
        math.floor(box_left + region.left * bbox.width),
        math.floor(box_top + region.top * bbox.height),
        math.ceil(box_left + region.right * bbox.width),
        math.ceil(box_top + region.bottom * bbox.height),
    )


def _schema_id(policy, region, region_name):
    """
    Identify the extraction definition used by histogram comparison.

    Upper and lower descriptors have different schema IDs.
    Changing region boundaries or bin semantics changes the identifier.
    """
    parameters = (
        region.left,
        region.top,
        region.right,
        region.bottom,
        policy.hue_bins,
        policy.saturation_bins,
        policy.value_bins,
        policy.achromatic_bins,
        policy.minimum_saturation,
        policy.minimum_value,
    )

    return (
        "rgb_hsv_chromatic_achromatic_v1:"
        + region_name
        + ":"
        + ":".join(str(value) for value in parameters)
    )


# ============================================================================
# NORMALIZED HISTOGRAM
# ============================================================================


def _histogram_from_rgb(pixels, policy, schema_id):
    """
    Convert selected RGB pixels into a normalized histogram.

    Processing:
        RGB [0, 255] -> HSV components in [0, 1].
        Each pixel contributes to exactly one bin:
            chromatic HSV bin, or
            achromatic value bin.

    Equation:
        histogram[b] = count[b] / selected_pixel_count

    All bins share one normalization.
    Therefore the descriptor retains the relative chromatic/achromatic mass.
    """
    rgb = pixels.astype(np.float64) / 255.0

    red = rgb[:, 0]
    green = rgb[:, 1]
    blue = rgb[:, 2]

    maximum = rgb.max(axis=1)
    minimum = rgb.min(axis=1)
    delta = maximum - minimum

    value = maximum
    saturation = np.zeros_like(value)

    np.divide(
        delta,
        maximum,
        out=saturation,
        where=maximum > 0.0,
    )

    hue = np.zeros_like(value)
    varying = delta > 0.0

    red_max = varying & (maximum == red)
    green_max = varying & ~red_max & (maximum == green)
    blue_max = varying & ~red_max & ~green_max

    hue[red_max] = (
        (green[red_max] - blue[red_max]) / delta[red_max]
    ) % 6.0

    hue[green_max] = (
        (blue[green_max] - red[green_max]) / delta[green_max]
        + 2.0
    )

    hue[blue_max] = (
        (red[blue_max] - green[blue_max]) / delta[blue_max]
        + 4.0
    )

    hue /= 6.0

    chromatic = (
        varying
        & (saturation >= policy.minimum_saturation)
        & (value >= policy.minimum_value)
    )

    chromatic_bin_count = (
        policy.hue_bins
        * policy.saturation_bins
        * policy.value_bins
    )

    # Start with achromatic value bins.
    indices = chromatic_bin_count + np.minimum(
        (value * policy.achromatic_bins).astype(np.int64),
        policy.achromatic_bins - 1,
    )

    hue_indices = np.minimum(
        (hue[chromatic] * policy.hue_bins).astype(np.int64),
        policy.hue_bins - 1,
    )

    saturation_indices = np.minimum(
        (
            saturation[chromatic] * policy.saturation_bins
        ).astype(np.int64),
        policy.saturation_bins - 1,
    )

    value_indices = np.minimum(
        (value[chromatic] * policy.value_bins).astype(np.int64),
        policy.value_bins - 1,
    )

    indices[chromatic] = (
        (
            hue_indices * policy.saturation_bins
            + saturation_indices
        )
        * policy.value_bins
        + value_indices
    )

    counts = np.bincount(
        indices,
        minlength=chromatic_bin_count + policy.achromatic_bins,
    )

    bins = counts.astype(np.float64) / pixels.shape[0]

    return Histogram(
        bins=tuple(float(value) for value in bins),
        schema_id=schema_id,
    )


# ============================================================================
# REGION MEASUREMENT
# ============================================================================


def _extract_region(
    image_rgb,
    source,
    policy,
    region,
    region_name,
    external_reliability,
    foreground_mask,
):
    """
    Extract one descriptor and preserve its actual observation provenance.

    Quality:
        q_region = pixel_support * external_reliability

    External reliability represents separately assessed limitations,
    such as occlusion or image quality. It is not inferred here.

    Clipping and masking reduce support; they do not move the body region.
    """
    if external_reliability == 0.0:
        return Measurement.unavailable(
            region_name + " region has zero external reliability"
        )

    left, top, right, bottom = _region_bounds(source, region)
    intended_area = (right - left) * (bottom - top)

    image_height, image_width = image_rgb.shape[:2]

    clipped_left = max(0, min(image_width, left))
    clipped_right = max(0, min(image_width, right))
    clipped_top = max(0, min(image_height, top))
    clipped_bottom = max(0, min(image_height, bottom))

    if (
        intended_area <= 0
        or clipped_left >= clipped_right
        or clipped_top >= clipped_bottom
    ):
        return Measurement.unavailable(
            region_name + " region has no image support"
        )

    patch = image_rgb[
        clipped_top:clipped_bottom,
        clipped_left:clipped_right,
    ]

    if foreground_mask is None:
        pixels = patch.reshape(-1, 3)
    else:
        selected = foreground_mask[
            clipped_top:clipped_bottom,
            clipped_left:clipped_right,
        ]
        pixels = patch[selected]

    selected_count = pixels.shape[0]
    support = selected_count / intended_area

    if selected_count < policy.minimum_pixels:
        return Measurement.unavailable(
            region_name + " region has insufficient selected pixels"
        )

    if support < policy.minimum_support:
        return Measurement.unavailable(
            region_name + " region has insufficient pixel support"
        )

    histogram = _histogram_from_rgb(
        pixels=pixels,
        policy=policy,
        schema_id=_schema_id(policy, region, region_name),
    )

    return Measurement(
        value=histogram,
        quality=support * external_reliability,
        stamp_ns=source.stamp_ns,
        frame_id=source.image_frame,
        reason=(
            region_name
            + " region extracted; pixel_support="
            + format(support, ".3f")
        ),
    )


# ============================================================================
# PUBLIC EXTRACTION ENTRY POINT
# ============================================================================


def extract_appearance(
    image_rgb: np.ndarray,
    source: SourceDetection,
    policy: AppearancePolicy,
    upper_reliability: float,
    lower_reliability: float,
    foreground_mask: Optional[np.ndarray] = None,
) -> Tuple[Measurement[Histogram], Measurement[Histogram]]:
    """
    Extract upper and lower appearance for one actual source detection.

    Receives:
        H-by-W-by-3 uint8 RGB image.
        Source detection referring to that image.
        Explicit region and histogram policy.
        External reliability in [0, 1] for each region.
        Optional H-by-W boolean mask specific to this person.

    Returns:
        (appearance_upper, appearance_lower)

    Missing or insufficient regional evidence produces an unavailable
    Measurement rather than an artificial all-zero histogram.

    Caller responsibility:
        Verify image timestamp and frame against SourceDetection.
        Convert BGR transport images to RGB before calling.
        Ensure any foreground mask belongs to this candidate and image.

    Setting external reliability to one declares that no additional
    reliability reduction is being applied; it does not prove no occlusion.
    """
    if not isinstance(source, SourceDetection):
        raise ValueError("source must be SourceDetection")

    if not isinstance(policy, AppearancePolicy):
        raise ValueError("policy must be AppearancePolicy")

    if (
        not isinstance(image_rgb, np.ndarray)
        or image_rgb.dtype != np.uint8
        or image_rgb.ndim != 3
        or image_rgb.shape[2] != 3
        or image_rgb.shape[0] == 0
        or image_rgb.shape[1] == 0
    ):
        raise ValueError("image_rgb must be a nonempty uint8 RGB image")

    unit_interval("upper_reliability", upper_reliability)
    unit_interval("lower_reliability", lower_reliability)

    if foreground_mask is not None:
        if (
            not isinstance(foreground_mask, np.ndarray)
            or foreground_mask.dtype != np.bool_
            or foreground_mask.shape != image_rgb.shape[:2]
        ):
            raise ValueError(
                "foreground_mask must be boolean and match the image shape"
            )

    upper = _extract_region(
        image_rgb,
        source,
        policy,
        policy.upper,
        "upper",
        upper_reliability,
        foreground_mask,
    )

    lower = _extract_region(
        image_rgb,
        source,
        policy,
        policy.lower,
        "lower",
        lower_reliability,
        foreground_mask,
    )

    return upper, lower


# ============================================================================
# DETAILED APPEARANCE EXTRACTION FOR ROS MESSAGE SERIALIZATION
# ============================================================================

from dataclasses import dataclass
from types import SimpleNamespace


@dataclass(frozen=True)
class RegionExtraction:
    """One extracted region with its actual pixel evidence."""

    measurement: Measurement
    schema_id: str
    relative_region: tuple
    intended_pixel_count: int
    selected_pixel_count: int
    support_fraction: float
    foreground_mask_used: bool


def _extract_region_detailed(
    image_rgb,
    bbox,
    stamp_ns,
    image_frame,
    policy,
    region,
    region_name,
    external_reliability,
    foreground_mask,
):
    """Extract one histogram and preserve its pixel statistics."""

    schema = _schema_id(policy, region, region_name)

    relative = (
        region.left,
        region.top,
        region.right,
        region.bottom,
    )

    # Reuse the existing geometric definition.
    source = SimpleNamespace(bbox=bbox)

    left, top, right, bottom = _region_bounds(
        source, region
    )

    intended = max(
        0, (right - left) * (bottom - top)
    )

    height, width = image_rgb.shape[:2]

    x0 = max(0, min(width, left))
    x1 = max(0, min(width, right))
    y0 = max(0, min(height, top))
    y1 = max(0, min(height, bottom))

    if x0 < x1 and y0 < y1:
        patch = image_rgb[y0:y1, x0:x1]

        if foreground_mask is None:
            pixels = patch.reshape(-1, 3)
        else:
            pixels = patch[
                foreground_mask[y0:y1, x0:x1]
            ]
    else:
        pixels = np.empty((0, 3), dtype=np.uint8)

    selected = int(pixels.shape[0])

    support = (
        selected / intended
        if intended > 0
        else 0.0
    )

    if external_reliability == 0.0:
        reason = "ZERO_EXTERNAL_RELIABILITY"

    elif selected < policy.minimum_pixels:
        reason = "INSUFFICIENT_REGION_PIXELS"

    elif support < policy.minimum_support:
        reason = "INSUFFICIENT_REGION_SUPPORT"

    else:
        reason = ""

    if reason:
        measurement = Measurement.unavailable(reason)

    else:
        histogram = _histogram_from_rgb(
            pixels, policy, schema
        )

        measurement = Measurement(
            value=histogram,
            quality=support * external_reliability,
            stamp_ns=stamp_ns,
            frame_id=image_frame,
            reason="VALID_REGION_HISTOGRAM",
        )

    return RegionExtraction(
        measurement=measurement,
        schema_id=schema,
        relative_region=relative,
        intended_pixel_count=intended,
        selected_pixel_count=selected,
        support_fraction=support,
        foreground_mask_used=foreground_mask is not None,
    )


def extract_appearance_detailed(
    image_rgb,
    bbox,
    stamp_ns,
    image_frame,
    policy,
    upper_reliability=1.0,
    lower_reliability=1.0,
    foreground_mask=None,
):
    """
    Extract two region descriptors with real pixel statistics.

    Image synchronization must be established by the ROS node.
    """

    from .types import BoundingBox2D

    if not isinstance(bbox, BoundingBox2D):
        raise ValueError("INVALID_BOUNDING_BOX")

    if not isinstance(policy, AppearancePolicy):
        raise ValueError("INVALID_APPEARANCE_POLICY")

    if (
        not isinstance(image_rgb, np.ndarray)
        or image_rgb.dtype != np.uint8
        or image_rgb.ndim != 3
        or image_rgb.shape[2] != 3
        or image_rgb.shape[0] == 0
        or image_rgb.shape[1] == 0
    ):
        raise ValueError("INVALID_RGB_IMAGE")

    integer("stamp_ns", stamp_ns, minimum=1)

    if not isinstance(image_frame, str) or not image_frame.strip():
        raise ValueError("INVALID_IMAGE_FRAME")

    unit_interval("upper_reliability", upper_reliability)
    unit_interval("lower_reliability", lower_reliability)

    if foreground_mask is not None:
        if (
            not isinstance(foreground_mask, np.ndarray)
            or foreground_mask.dtype != np.bool_
            or foreground_mask.shape != image_rgb.shape[:2]
        ):
            raise ValueError("INVALID_FOREGROUND_MASK")

    upper = _extract_region_detailed(
        image_rgb, bbox, stamp_ns, image_frame,
        policy, policy.upper, "upper",
        upper_reliability, foreground_mask,
    )

    lower = _extract_region_detailed(
        image_rgb, bbox, stamp_ns, image_frame,
        policy, policy.lower, "lower",
        lower_reliability, foreground_mask,
    )

    return upper, lower

