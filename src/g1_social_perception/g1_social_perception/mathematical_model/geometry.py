"""
Shared geometry for detection enrichment and association.

Equations:
    Camera deprojection:
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        camera_position = (X, Y, Z)

    Camera projection:
        u = fx * X / Z + cx
        v = fy * Y / Z + cy

    Approximate metric extent:
        W = w * Z / fx
        H = h * Z / fy

    Rigid transformation:
        target_position = R * source_position + translation
        target_velocity = R * source_velocity

    Bounding-box overlap:
        IoU = intersection_area / union_area

    Euclidean distance:
        d = sqrt(dx**2 + dy**2 + dz**2)

Boundary:
    This module calculates geometry only.
    Nodes provide synchronized inputs and timestamp-correct transforms.
    Quality, freshness and association decisions belong to other modules.
"""


import math
from dataclasses import dataclass
from typing import Tuple

from .types import BoundingBox2D, Extent2D, Vector3
from .validation import finite, positive


# ============================================================================
# CAMERA MODEL
# ============================================================================


@dataclass(frozen=True)
class CameraIntrinsics:
    """
    Pinhole intrinsics for the image containing the detection.

    Fields:
        fx, fy: focal lengths in pixels.
        cx, cy: principal point in pixels.

    Assumption:
        Image coordinates are rectified or otherwise compatible with
        this pinhole model. Distortion correction is handled upstream.
    """

    fx: float
    fy: float
    cx: float
    cy: float

    def __post_init__(self):
        positive("fx", self.fx)
        positive("fy", self.fy)
        finite("cx", self.cx)
        finite("cy", self.cy)


def deproject_pixel(
    u: float,
    v: float,
    depth_m: float,
    intrinsics: CameraIntrinsics,
) -> Vector3:
    """
    Convert an image location and optical-axis depth into a 3D point.

    Receives:
        Pixel coordinates (u, v).
        Positive depth Z in metres.
        Intrinsics matching the source image.

    Equation:
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy

    Returns:
        Camera optical coordinates in metres:
        +X right, +Y down, +Z forward.

    Depth must represent optical-axis Z, not Euclidean camera range.
    Selecting a reliable depth sample belongs to depth fusion.
    """
    finite("u", u)
    finite("v", v)
    positive("depth_m", depth_m)

    return Vector3(
        x=(u - intrinsics.cx) * depth_m / intrinsics.fx,
        y=(v - intrinsics.cy) * depth_m / intrinsics.fy,
        z=depth_m,
    )


def project_point(
    camera_position: Vector3,
    intrinsics: CameraIntrinsics,
) -> Tuple[float, float]:
    """
    Project a camera-frame point into image coordinates.

    Equation:
        u = fx * X / Z + cx
        v = fy * Y / Z + cy

    Returns:
        Pixel coordinates (u, v), without clipping to image boundaries.

    Points on or behind the camera plane cannot be projected here.
    A projected centre alone does not define a predicted bounding box.
    """
    positive("camera_position.z", camera_position.z)

    u = (
        intrinsics.fx * camera_position.x / camera_position.z
        + intrinsics.cx
    )
    v = (
        intrinsics.fy * camera_position.y / camera_position.z
        + intrinsics.cy
    )

    finite("projected u", u)
    finite("projected v", v)

    return u, v


def metric_extent_from_bbox(
    bbox: BoundingBox2D,
    depth_m: float,
    intrinsics: CameraIntrinsics,
) -> Extent2D:
    """
    Estimate visible metric dimensions from image size and depth.

    Equation:
        W = bbox.width * Z / fx
        H = bbox.height * Z / fy

    Returns:
        Width and height in metres.

    Assumption:
        Both dimensions are approximated at the same representative depth.
        Occlusion, posture and depth variation affect their reliability.

    These are visible box extents, not anatomical measurements.
    The quality module determines their contribution to size matching.
    """
    positive("depth_m", depth_m)

    return Extent2D(
        width=bbox.width * depth_m / intrinsics.fx,
        height=bbox.height * depth_m / intrinsics.fy,
    )


# ============================================================================
# RIGID FRAME TRANSFORMATIONS
# ============================================================================


def rotation_from_quaternion(
    x: float,
    y: float,
    z: float,
    w: float,
) -> Tuple[Tuple[float, float, float], ...]:
    """
    Convert a quaternion in ROS component order (x, y, z, w) to R.

    Processing:
        Validate finite components and normalize a nonzero quaternion.

    Returns:
        A 3-by-3 rotation matrix.

    The caller supplies the source-to-target quaternion.
    This function does not choose frames or look up TF.
    """
    for name, value in (
        ("quaternion.x", x),
        ("quaternion.y", y),
        ("quaternion.z", z),
        ("quaternion.w", w),
    ):
        finite(name, value)

    # Scale first to avoid overflow when calculating the norm.
    scale = max(abs(x), abs(y), abs(z), abs(w))

    if scale == 0.0:
        raise ValueError("Quaternion must have a nonzero norm")

    x, y, z, w = (
        x / scale,
        y / scale,
        z / scale,
        w / scale,
    )

    norm = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm

    return (
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
        ),
        (
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
        ),
        (
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ),
    )


def rotate_vector(
    vector: Vector3,
    quaternion_xyzw: Tuple[float, float, float, float],
) -> Vector3:
    """
    Express a geometric vector in another frame using v_target = R * v.

    Receives:
        A vector and the source-to-target rotation.

    Returns:
        Rotated components with the original physical units.

    Translation does not act on a vector.

    For velocity, this only changes the expression frame of the same
    velocity. It does not calculate velocity relative to a moving robot.
    """
    rotation = rotation_from_quaternion(*quaternion_xyzw)
    components = (vector.x, vector.y, vector.z)

    result = tuple(
        math.fsum(
            coefficient * component
            for coefficient, component in zip(row, components)
        )
        for row in rotation
    )

    return Vector3(*result)


def transform_position(
    position: Vector3,
    translation: Vector3,
    quaternion_xyzw: Tuple[float, float, float, float],
) -> Vector3:
    """
    Transform a position using p_target = R * p_source + translation.

    Receives:
        Source position in metres.
        Source-origin position expressed in the target frame, in metres.
        Source-to-target rotation.

    Returns:
        Target-frame position in metres.

    Example:
        Camera position -> odom position using odom <- camera TF.

    The node must provide the transform at the observation timestamp.
    No timestamp or frame identifier is changed by this calculation.
    """
    rotated = rotate_vector(position, quaternion_xyzw)

    return Vector3(
        x=rotated.x + translation.x,
        y=rotated.y + translation.y,
        z=rotated.z + translation.z,
    )


# ============================================================================
# IMAGE-SPACE BOX GEOMETRY
# ============================================================================


def bbox_edges(
    bbox: BoundingBox2D,
) -> Tuple[float, float, float, float]:
    """
    Convert centre-size representation into continuous box edges.

    Equation:
        left = u - w / 2
        top = v - h / 2
        right = u + w / 2
        bottom = v + h / 2

    Returns:
        (left, top, right, bottom), in pixels.

    No image clipping or integer rounding is applied.
    """
    return (
        bbox.center_u - bbox.width / 2.0,
        bbox.center_v - bbox.height / 2.0,
        bbox.center_u + bbox.width / 2.0,
        bbox.center_v + bbox.height / 2.0,
    )


def intersection_over_union(
    first: BoundingBox2D,
    second: BoundingBox2D,
) -> float:
    """
    Calculate box overlap IoU in [0, 1].

    Equation:
        intersection = overlapping_width * overlapping_height
        union = first_area + second_area - intersection
        IoU = intersection / union

    Returns:
        Zero for disjoint or touching boxes.
        One for identical boxes.

    Both boxes must describe the same image coordinate system.
    For association, the caller supplies the predicted track box
    and the current candidate box.
    """
    left_a, top_a, right_a, bottom_a = bbox_edges(first)
    left_b, top_b, right_b, bottom_b = bbox_edges(second)

    overlap_width = max(
        0.0,
        min(right_a, right_b) - max(left_a, left_b),
    )
    overlap_height = max(
        0.0,
        min(bottom_a, bottom_b) - max(top_a, top_b),
    )

    intersection = overlap_width * overlap_height
    union = (
        first.width * first.height
        + second.width * second.height
        - intersection
    )

    finite("intersection area", intersection)
    positive("union area", union)

    return min(1.0, max(0.0, intersection / union))


# ============================================================================
# DISTANCES FOR ASSOCIATION
# ============================================================================


def distance_3d(
    first: Vector3,
    second: Vector3,
) -> float:
    """
    Calculate Euclidean distance between positions in the same frame.

    Equation:
        d = sqrt(
            (first.x - second.x)**2
            + (first.y - second.y)**2
            + (first.z - second.z)**2
        )

    Returns:
        Distance in metres when input positions are in metres.

    Association compares the candidate position with the predicted
    track position. Squared differences already remove sign dependence;
    an additional absolute-value operation is unnecessary.
    """
    distance = math.hypot(
        first.x - second.x,
        first.y - second.y,
        first.z - second.z,
    )

    finite("3D distance", distance)
    return distance


def bbox_center_distance_px(
    first: BoundingBox2D,
    second: BoundingBox2D,
) -> float:
    """
    Calculate image-plane distance between box centres.

    Equation:
        d_px = sqrt((u_first - u_second)**2 + (v_first - v_second)**2)

    Returns:
        Distance in pixels.

    This may support an explicitly qualified image-space fallback.
    It must not be compared with a distance threshold expressed in metres.
    """
    distance = math.hypot(
        first.center_u - second.center_u,
        first.center_v - second.center_v,
    )

    finite("bbox centre distance", distance)
    return distance