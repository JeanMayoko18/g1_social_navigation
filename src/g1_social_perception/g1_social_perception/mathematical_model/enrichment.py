"""
Build an enriched candidate from one actual source observation.

Receives:
    Source YOLO detection.
    Odom position and metric extent produced by depth/TF processing.
    Matching RGB image when available.
    Appearance extraction policy and regional reliability.

Processing:
    Validate RGB provenance.
    Extract upper/lower appearance.
    Wrap the actual bbox dimensions as apparent-size evidence.
    Assemble all features into EnrichedDetection.

Returns:
    One candidate ready for comparison with retained tracks.

Boundary:
    This function performs no ROS subscription, depth synchronization,
    TF lookup, identity assignment or track-memory update.
"""


from typing import Optional

import numpy as np

from .appearance import AppearancePolicy, extract_appearance
from .types import (
    EnrichedDetection,
    Extent2D,
    Measurement,
    SourceDetection,
    Vector3,
)
from .validation import integer, nonempty, unit_interval


def enrich_detection(
    source: SourceDetection,
    position_odom: Measurement[Vector3],
    metric_extent: Measurement[Extent2D],
    appearance_policy: AppearancePolicy,
    upper_reliability: float,
    lower_reliability: float,
    image_rgb: Optional[np.ndarray] = None,
    image_stamp_ns: Optional[int] = None,
    image_frame_id: Optional[str] = None,
    foreground_mask: Optional[np.ndarray] = None,
) -> EnrichedDetection:
    """
    Assemble the enriched observation D_tilde[k, j].

    Feature correspondence:
        D_YOLO     -> source
        p_odom     -> supplied depth/TF position
        a_upper    -> extracted upper-region histogram
        a_lower    -> extracted lower-region histogram
        e_apparent -> observed bbox width/height in pixels
        e_metric   -> supplied visible dimensions in metres

    Quality:
        Each feature carries its own reliability and source provenance.
        Apparent-size quality uses detector confidence as an explicit policy.
        Appearance quality is calculated by extract_appearance.

    Missing RGB:
        Preserve spatial and size evidence.
        Return explicitly unavailable appearance measurements.

    Mismatched RGB:
        Reject construction instead of extracting from another observation.

    Source stamps are reference timestamps for the same observation.
    Depth synchronization must already have been checked upstream.
    """
    if not isinstance(source, SourceDetection):
        raise ValueError("source must be SourceDetection")

    if not isinstance(appearance_policy, AppearancePolicy):
        raise ValueError("appearance_policy must be AppearancePolicy")

    unit_interval("upper_reliability", upper_reliability)
    unit_interval("lower_reliability", lower_reliability)

    if image_rgb is None:
        if (
            image_stamp_ns is not None
            or image_frame_id is not None
            or foreground_mask is not None
        ):
            raise ValueError(
                "Missing RGB image must not carry image metadata or a mask"
            )

        appearance_upper = Measurement.unavailable(
            "Matching source RGB image unavailable"
        )
        appearance_lower = Measurement.unavailable(
            "Matching source RGB image unavailable"
        )

    else:
        integer("image_stamp_ns", image_stamp_ns, minimum=1)
        nonempty("image_frame_id", image_frame_id)

        if (
            image_stamp_ns != source.stamp_ns
            or image_frame_id != source.image_frame
        ):
            raise ValueError(
                "RGB image must match the detection source stamp and frame"
            )

        # Actual extraction call: descriptors become candidate features below.
        appearance_upper, appearance_lower = extract_appearance(
            image_rgb=image_rgb,
            source=source,
            policy=appearance_policy,
            upper_reliability=upper_reliability,
            lower_reliability=lower_reliability,
            foreground_mask=foreground_mask,
        )

    # Apparent size comes from the actual detection, never from a prediction.
    apparent_extent = Measurement(
        value=Extent2D(
            width=source.bbox.width,
            height=source.bbox.height,
        ),
        quality=source.confidence,
        stamp_ns=source.stamp_ns,
        frame_id=source.image_frame,
        reason="Observed bbox dimensions; quality uses detector confidence",
    )

    # EnrichedDetection validates feature types, timestamps and frames.
    return EnrichedDetection(
        source=source,
        position_odom=position_odom,
        appearance_upper=appearance_upper,
        appearance_lower=appearance_lower,
        apparent_extent=apparent_extent,
        metric_extent=metric_extent,
    )