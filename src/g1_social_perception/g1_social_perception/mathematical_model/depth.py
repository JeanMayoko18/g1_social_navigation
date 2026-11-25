"""
Depth statistics and explicit reliability policy.

Receives:
    Aligned depth image in metres.
    Actual image bounding box.
    Depth range and central-region scale.

Calculates:
    Median depth Z.
    Median absolute deviation MAD.
    Valid-pixel support.
    Reliability from confidence, support, dispersion and synchronization.

Boundary:
    No ROS messages, image decoding, buffering or TF lookup.
    A reliable central depth does not prove anatomical body dimensions.
"""


import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .types import BoundingBox2D
from .validation import positive, unit_interval, finite


@dataclass(frozen=True)
class DepthStatistics:
    """Depth support and robust statistics for one clipped central ROI."""

    roi_pixels: int
    valid_pixels: int
    median_m: Optional[float]
    mad_m: Optional[float]


def central_depth_statistics(
    depth_m: np.ndarray,
    box: BoundingBox2D,
    roi_scale: float,
    minimum_m: float,
    maximum_m: float,
) -> DepthStatistics:
    """
    Estimate depth from the central fraction of the observed box.

    Equations:
        Z = median(valid_depths)
        MAD = median(abs(valid_depths - Z))

    Valid samples are finite and inside the configured metric range.
    A box centre outside the image produces unavailable statistics.
    """
    positive("roi_scale", roi_scale)
    unit_interval("roi_scale", roi_scale)
    positive("minimum_m", minimum_m)
    positive("maximum_m", maximum_m)

    if minimum_m >= maximum_m:
        raise ValueError("minimum_m must be smaller than maximum_m")

    if not isinstance(depth_m, np.ndarray) or depth_m.ndim != 2:
        raise ValueError("depth_m must be a two-dimensional NumPy array")

    height, width = depth_m.shape

    if not (
        0 <= box.center_u < width
        and 0 <= box.center_v < height
    ):
        return DepthStatistics(0, 0, None, None)

    half_w = box.width * roi_scale / 2.0
    half_h = box.height * roi_scale / 2.0

    left = max(0, min(width, math.floor(box.center_u - half_w)))
    right = max(0, min(width, math.ceil(box.center_u + half_w)))
    top = max(0, min(height, math.floor(box.center_v - half_h)))
    bottom = max(0, min(height, math.ceil(box.center_v + half_h)))

    roi = depth_m[top:bottom, left:right]

    valid = roi[
        np.isfinite(roi)
        & (roi >= minimum_m)
        & (roi <= maximum_m)
    ]

    if valid.size == 0:
        return DepthStatistics(int(roi.size), 0, None, None)

    median = float(np.median(valid))
    mad = float(np.median(np.abs(valid - median)))

    return DepthStatistics(
        roi_pixels=int(roi.size),
        valid_pixels=int(valid.size),
        median_m=median,
        mad_m=mad,
    )


def depth_reliability(
    stats: DepthStatistics,
    confidence: float,
    offset_sec: float,
    dispersion_scale_m: float,
    sync_scale_sec: float,
) -> float:
    """
    Calculate the configured depth-evidence reliability heuristic.

    Equation:
        q = confidence * valid_fraction
            * exp(-MAD / dispersion_scale_m)
            * exp(-abs(offset_sec) / sync_scale_sec)

    Missing depth contributes zero quality.
    Hard sample-count and synchronization gates belong to the caller.
    """
    unit_interval("confidence", confidence)
    finite("offset_sec", offset_sec)
    positive("dispersion_scale_m", dispersion_scale_m)
    positive("sync_scale_sec", sync_scale_sec)

    if stats.roi_pixels == 0 or stats.valid_pixels == 0:
        return 0.0

    return (
        confidence
        * (stats.valid_pixels / stats.roi_pixels)
        * math.exp(-stats.mad_m / dispersion_scale_m)
        * math.exp(-abs(offset_sec) / sync_scale_sec)
    )