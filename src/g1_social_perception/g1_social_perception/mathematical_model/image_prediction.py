
"""Projection d'une piste odom vers une boîte dans l'image courante.

Approximation frontoparallèle de l'étendue visible.
Aucune recherche TF ni mutation de piste dans ce module.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

from .geometry import CameraIntrinsics, project_point
from .prediction import PositionPrediction
from .types import BoundingBox2D, Extent2D, Measurement, Vector3

Matrix3 = Tuple[Tuple[float, float, float], ...]


def _stamp(value):
    if type(value) is not int or value <= 0:
        raise ValueError("Invalid source timestamp")


def _quality(value):
    if (
        isinstance(value, bool)
        or not isinstance(value, (float, int))
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise ValueError("Invalid geometry quality")


def _rotation(matrix):
    if len(matrix) != 3 or any(len(row) != 3 for row in matrix):
        raise ValueError("Expected 3x3 rotation")

    if not all(math.isfinite(x) for row in matrix for x in row):
        raise ValueError("Nonfinite rotation")

    for i in range(3):
        for j in range(3):
            dot = sum(matrix[i][k] * matrix[j][k] for k in range(3))
            if abs(dot - (1.0 if i == j else 0.0)) > 1e-5:
                raise ValueError("Nonorthonormal rotation")

    a, b, c = matrix
    determinant = (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )
    if abs(determinant - 1.0) > 1e-5:
        raise ValueError("Improper rotation")


@dataclass(frozen=True)
class ReferenceView:
    stamp_ns: int
    camera_frame: str
    rotation_camera_to_odom: Matrix3
    quality: float

    def __post_init__(self):
        _stamp(self.stamp_ns)
        if not self.camera_frame:
            raise ValueError("Missing reference optical frame")
        _rotation(self.rotation_camera_to_odom)
        _quality(self.quality)


@dataclass(frozen=True)
class CurrentView:
    stamp_ns: int
    camera_frame: str
    rotation_odom_to_camera: Matrix3
    translation_odom_to_camera: Vector3
    intrinsics: CameraIntrinsics
    width_px: int
    height_px: int
    quality: float

    def __post_init__(self):
        _stamp(self.stamp_ns)
        if not self.camera_frame:
            raise ValueError("Missing current optical frame")
        _rotation(self.rotation_odom_to_camera)
        if not isinstance(self.translation_odom_to_camera, Vector3):
            raise ValueError("Invalid camera translation")
        if not isinstance(self.intrinsics, CameraIntrinsics):
            raise ValueError("Invalid camera intrinsics")
        if (
            type(self.width_px) is not int or self.width_px <= 0
            or type(self.height_px) is not int or self.height_px <= 0
        ):
            raise ValueError("Invalid image dimensions")
        _quality(self.quality)


def reference_from_current(view: CurrentView) -> ReferenceView:
    """Save the camera pose of an actual metric-extent observation."""
    r = view.rotation_odom_to_camera
    return ReferenceView(
        stamp_ns=view.stamp_ns,
        camera_frame=view.camera_frame,
        rotation_camera_to_odom=tuple(
            tuple(r[j][i] for j in range(3))
            for i in range(3)
        ),
        quality=view.quality,
    )


@dataclass(frozen=True)
class ImageBoxPrediction:
    bbox: Optional[BoundingBox2D]
    quality: float
    target_stamp_ns: int
    reason: str
    orientation_difference_rad: Optional[float] = None

    def __post_init__(self):
        _stamp(self.target_stamp_ns)
        _quality(self.quality)
        if self.bbox is None and self.quality != 0:
            raise ValueError("Unavailable bbox must have zero quality")

    @property
    def usable(self):
        return self.bbox is not None and self.quality > 0

    @classmethod
    def unavailable(cls, stamp_ns, reason, angle=None):
        return cls(None, 0.0, stamp_ns, reason, angle)


def _relative_angle(reference, current):
    """Full relative rotation, including roll around the optical axis."""
    r = reference.rotation_camera_to_odom
    c = current.rotation_odom_to_camera
    trace = sum(
        sum(c[i][k] * r[k][i] for k in range(3))
        for i in range(3)
    )
    cosine = max(-1.0, min(1.0, (trace - 1.0) / 2.0))
    return math.acos(cosine)


def predict_image_box(
    *,
    position_prediction: PositionPrediction,
    metric_extent: Measurement,
    reference_view: Optional[ReferenceView],
    current_view: Optional[CurrentView],
    maximum_extent_age_sec: float,
    maximum_view_angle_rad: float,
    minimum_depth_m: float,
    maximum_depth_m: float,
) -> ImageBoxPrediction:
    """Predict a current-image bounding box, or explain its exclusion."""

    target_ns = position_prediction.target_stamp_ns

    for name, value in (
        ("maximum_extent_age_sec", maximum_extent_age_sec),
        ("minimum_depth_m", minimum_depth_m),
        ("maximum_depth_m", maximum_depth_m),
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError("Invalid " + name)

    if maximum_depth_m <= minimum_depth_m:
        raise ValueError("Invalid depth interval")

    if (
        not math.isfinite(maximum_view_angle_rad)
        or not 0 < maximum_view_angle_rad <= math.pi
    ):
        raise ValueError("Invalid camera angle limit")

    if current_view is None:
        return ImageBoxPrediction.unavailable(
            target_ns, "CURRENT_CAMERA_GEOMETRY_UNAVAILABLE"
        )

    if current_view.stamp_ns != target_ns:
        return ImageBoxPrediction.unavailable(
            target_ns, "CAMERA_TIMESTAMP_MISMATCH"
        )

    if (
        position_prediction.position is None
        or position_prediction.quality <= 0
    ):
        return ImageBoxPrediction.unavailable(
            target_ns, "POSITION_PREDICTION_UNAVAILABLE"
        )

    if reference_view is None:
        return ImageBoxPrediction.unavailable(
            target_ns, "REFERENCE_CAMERA_GEOMETRY_UNAVAILABLE"
        )

    if (
        not isinstance(metric_extent, Measurement)
        or not metric_extent.usable
        or not isinstance(metric_extent.value, Extent2D)
    ):
        return ImageBoxPrediction.unavailable(
            target_ns, "METRIC_EXTENT_UNAVAILABLE"
        )

    if (
        metric_extent.stamp_ns != reference_view.stamp_ns
        or metric_extent.frame_id != reference_view.camera_frame
        or current_view.camera_frame != reference_view.camera_frame
    ):
        return ImageBoxPrediction.unavailable(
            target_ns, "CAMERA_OR_EXTENT_PROVENANCE_MISMATCH"
        )

    age_sec = (target_ns - metric_extent.stamp_ns) / 1e9
    if not 0 <= age_sec <= maximum_extent_age_sec:
        return ImageBoxPrediction.unavailable(
            target_ns, "METRIC_EXTENT_TOO_OLD_OR_FROM_FUTURE"
        )

    angle = _relative_angle(reference_view, current_view)

    if angle > maximum_view_angle_rad:
        return ImageBoxPrediction.unavailable(
            target_ns, "CAMERA_ORIENTATION_CHANGED", angle
        )

    p = position_prediction.position
    r = current_view.rotation_odom_to_camera
    t = current_view.translation_odom_to_camera
    coordinates = (p.x, p.y, p.z)
    translation = (t.x, t.y, t.z)

    camera_xyz = Vector3(
        *(
            sum(r[i][k] * coordinates[k] for k in range(3))
            + translation[i]
            for i in range(3)
        )
    )

    if not minimum_depth_m <= camera_xyz.z <= maximum_depth_m:
        return ImageBoxPrediction.unavailable(
            target_ns, "DEPTH_OUT_OF_RANGE", angle
        )

    try:
        u, v = project_point(camera_xyz, current_view.intrinsics)

        width = (
            current_view.intrinsics.fx
            * metric_extent.value.width
            / camera_xyz.z
        )
        height = (
            current_view.intrinsics.fy
            * metric_extent.value.height
            / camera_xyz.z
        )

        box = BoundingBox2D(u, v, width, height)

    except (ValueError, ZeroDivisionError, OverflowError):
        return ImageBoxPrediction.unavailable(
            target_ns, "INVALID_IMAGE_PROJECTION", angle
        )

    if (
        u + width / 2 <= 0
        or u - width / 2 >= current_view.width_px
        or v + height / 2 <= 0
        or v - height / 2 >= current_view.height_px
    ):
        return ImageBoxPrediction.unavailable(
            target_ns, "OUTSIDE_CURRENT_IMAGE", angle
        )

    quality = min(
        position_prediction.quality,
        metric_extent.quality,
        reference_view.quality,
        current_view.quality,
    )

    if quality <= 0:
        return ImageBoxPrediction.unavailable(
            target_ns, "ZERO_GEOMETRY_QUALITY", angle
        )

    return ImageBoxPrediction(
        bbox=box,
        quality=quality,
        target_stamp_ns=target_ns,
        reason="FRONTOPARALLEL_VISIBLE_EXTENT",
        orientation_difference_rad=angle,
    )
