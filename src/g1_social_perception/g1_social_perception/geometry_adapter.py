
#!/usr/bin/env python3
"""
geometry_adapter.py

ROS adapter for time-consistent image projection in G1 tracking.

RESPONSIBILITIES
    - Cache compatible CameraInfo messages.
    - Resolve camera pose in odom at the RGB observation stamp.
    - Construct mathematical_model.image_prediction.CurrentView.
    - Validate per-candidate RGB/depth metric provenance.
    - Report explicit reasons when geometry is unavailable.

BOUNDARIES
    - No identity association or track mutation.
    - No image processing or depth fusion.
    - No modification of source timestamps.
    - No substitution of latest TF for historical TF.
    - No independent ROS node or executor.

The parent person_tracking_node owns this adapter.
"""

import math
import threading

from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple

from rclpy.duration import Duration
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time

from sensor_msgs.msg import CameraInfo

from tf2_ros import (
    Buffer,
    TransformException,
    TransformListener,
)

from g1_social_msgs.msg import (
    DepthEvidence,
    ExtentFeature,
    FeatureMetadata,
    PersonDetectionArray,
)

from g1_social_perception.mathematical_model.geometry import (
    CameraIntrinsics,
    rotation_from_quaternion,
)
from g1_social_perception.mathematical_model.image_prediction import (
    CurrentView,
)
from g1_social_perception.mathematical_model.types import Vector3


# ============================================================================
# DATA / CONFIGURATION
# ============================================================================


def _finite_nonnegative(name, value):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0.0
    ):
        raise ValueError(f"INVALID_{name.upper()}")


def _finite_positive(name, value):
    _finite_nonnegative(name, value)

    if value == 0.0:
        raise ValueError(f"INVALID_{name.upper()}")


def _stamp_ns(stamp, *, allow_zero=False):
    seconds = int(stamp.sec)
    nanoseconds = int(stamp.nanosec)

    if seconds < 0 or not 0 <= nanoseconds < 1_000_000_000:
        raise ValueError("INVALID_ROS_TIMESTAMP")

    result = seconds * 1_000_000_000 + nanoseconds

    if result == 0 and not allow_zero:
        raise ValueError("ZERO_ROS_TIMESTAMP")

    return result


@dataclass(frozen=True)
class GeometryAdapterConfig:
    camera_info_topic: str
    odom_frame: str = "odom"

    tf_cache_sec: float = 10.0
    camera_info_buffer_size: int = 32

    # Maximum distance between a stamped calibration and RGB.
    maximum_camera_info_age_sec: float = 2.0

    # A zero-stamped CameraInfo can be used as a persistent
    # calibration only if explicitly permitted by configuration.
    allow_static_camera_info: bool = False

    # Must be consistent with upstream RGB/depth matching policy.
    maximum_depth_rgb_offset_sec: float = 0.10

    # Numerical calibration cross-check tolerance.
    calibration_relative_tolerance: float = 1e-6
    calibration_absolute_tolerance: float = 1e-6

    def __post_init__(self):
        if not self.camera_info_topic:
            raise ValueError("EMPTY_CAMERA_INFO_TOPIC")

        if not self.odom_frame:
            raise ValueError("EMPTY_ODOM_FRAME")

        _finite_positive("tf_cache_sec", self.tf_cache_sec)
        _finite_nonnegative(
            "maximum_camera_info_age_sec",
            self.maximum_camera_info_age_sec,
        )
        _finite_nonnegative(
            "maximum_depth_rgb_offset_sec",
            self.maximum_depth_rgb_offset_sec,
        )
        _finite_nonnegative(
            "calibration_relative_tolerance",
            self.calibration_relative_tolerance,
        )
        _finite_nonnegative(
            "calibration_absolute_tolerance",
            self.calibration_absolute_tolerance,
        )

        if (
            type(self.camera_info_buffer_size) is not int
            or self.camera_info_buffer_size < 1
        ):
            raise ValueError("INVALID_CAMERA_INFO_BUFFER_SIZE")

        if type(self.allow_static_camera_info) is not bool:
            raise ValueError("INVALID_STATIC_CALIBRATION_POLICY")


@dataclass(frozen=True)
class GeometryResult:
    view: Optional[CurrentView]
    reason: str
    observation_stamp_ns: int

    @property
    def usable(self):
        return self.view is not None and self.view.quality > 0.0


@dataclass(frozen=True)
class MetricProvenanceResult:
    usable: bool
    reason: str

    rgb_stamp_ns: Optional[int]
    depth_stamp_ns: Optional[int]
    depth_offset_ns: Optional[int]


# ============================================================================
# DATA / MATRIX OPERATIONS
# ============================================================================


def _transpose(matrix):
    return tuple(
        tuple(matrix[j][i] for j in range(3))
        for i in range(3)
    )


def _inverse_translation(rotation_camera_to_odom, translation):
    """
    Inverse of:

        p_odom = R_odom_camera * p_camera + t_odom_camera

    Gives:

        p_camera = R_camera_odom * p_odom + t_camera_odom
    """
    inverse_rotation = _transpose(rotation_camera_to_odom)

    values = (translation.x, translation.y, translation.z)

    inverse_translation = Vector3(
        *(
            -sum(
                inverse_rotation[i][j] * values[j]
                for j in range(3)
            )
            for i in range(3)
        )
    )

    return inverse_rotation, inverse_translation


# ============================================================================
# PROCESSING / CAMERA CALIBRATION
# ============================================================================


def _intrinsics_from_camera_info(info):
    """
    Support the same zero-distortion K pinhole convention
    already used by person_depth_fusion_node.py.

    Raw images with nonzero distortion need rectification
    or an explicit distortion-aware model.
    """

    if not isinstance(info, CameraInfo):
        raise ValueError("INVALID_CAMERA_INFO_TYPE")

    if not info.header.frame_id:
        raise ValueError("EMPTY_CAMERA_INFO_FRAME")

    if info.width <= 0 or info.height <= 0:
        raise ValueError("INVALID_CALIBRATION_IMAGE_SIZE")

    if len(info.k) != 9:
        raise ValueError("INVALID_CAMERA_MATRIX_LENGTH")

    if not all(math.isfinite(float(v)) for v in info.k):
        raise ValueError("NONFINITE_CAMERA_MATRIX")

    if any(
        not math.isfinite(float(value)) or float(value) != 0.0
        for value in info.d
    ):
        raise ValueError("UNSUPPORTED_CAMERA_DISTORTION")

    if (
        info.binning_x not in (0, 1)
        or info.binning_y not in (0, 1)
        or info.roi.x_offset != 0
        or info.roi.y_offset != 0
        or info.roi.width not in (0, info.width)
        or info.roi.height not in (0, info.height)
    ):
        raise ValueError("UNSUPPORTED_CAMERA_ROI_OR_BINNING")

    if (
        info.k[1] != 0.0
        or info.k[3] != 0.0
        or info.k[6] != 0.0
        or info.k[7] != 0.0
        or info.k[8] != 1.0
    ):
        raise ValueError("UNSUPPORTED_PINHOLE_CAMERA_MATRIX")

    intrinsics = CameraIntrinsics(
        fx=float(info.k[0]),
        fy=float(info.k[4]),
        cx=float(info.k[2]),
        cy=float(info.k[5]),
    )

    return intrinsics


def _calibration_matches_depth_evidence(
    candidate,
    intrinsics,
    config,
):
    """
    Confirm that the intrinsics used for current projection
    match the calibration retained by the upstream depth stage.

    This check is meaningful for an AVAILABLE depth estimate.
    """

    evidence = candidate.depth_evidence

    if evidence.status != DepthEvidence.AVAILABLE:
        return True

    if not evidence.calibration_available:
        return False

    for recorded, current in (
        (evidence.fx, intrinsics.fx),
        (evidence.fy, intrinsics.fy),
        (evidence.cx, intrinsics.cx),
        (evidence.cy, intrinsics.cy),
    ):
        if not math.isclose(
            float(recorded),
            float(current),
            rel_tol=config.calibration_relative_tolerance,
            abs_tol=config.calibration_absolute_tolerance,
        ):
            return False

    return True


# ============================================================================
# ROS ADAPTER
# ============================================================================


class GeometryAdapter:
    """
    Auxiliary object created inside person_tracking_node.

    The parent ROS node owns observation delivery, identity
    processing and publication.

    This adapter owns only CameraInfo reception and its TF buffer.
    """

    def __init__(self, node, config: GeometryAdapterConfig):
        self.node = node
        self.config = config

        self._lock = threading.RLock()

        self._camera_info = deque(
            maxlen=config.camera_info_buffer_size
        )

        self.tf_buffer = Buffer(
            cache_time=Duration(seconds=config.tf_cache_sec)
        )

        self.tf_listener = TransformListener(
            self.tf_buffer,
            node,
            spin_thread=False,
        )

        self.camera_info_subscription = node.create_subscription(
            CameraInfo,
            config.camera_info_topic,
            self._camera_info_callback,
            qos_profile_sensor_data,
        )

    # ========================================================================
    # INPUT / CALIBRATION CACHE
    # ========================================================================

    def _camera_info_callback(self, info):
        """
        Accept only calibrations compatible with our projection model.

        No source observation is created by this callback.
        """

        try:
            _stamp_ns(info.header.stamp, allow_zero=True)
            _intrinsics_from_camera_info(info)

        except (
            ValueError,
            TypeError,
            AttributeError,
            OverflowError,
        ) as exc:
            self.node.get_logger().warning(
                "[GEOMETRY] Ignoring CameraInfo: "
                + str(exc),
                throttle_duration_sec=2.0,
            )
            return

        with self._lock:
            self._camera_info.append(info)

    def _select_camera_info(self, frame_id, rgb_stamp_ns):
        """
        Select the newest compatible past calibration.

        Never silently select a calibration from a future time.

        A zero-stamped calibration is allowed only by explicit
        static-calibration policy.
        """

        with self._lock:
            cached = tuple(self._camera_info)

        candidates = []
        static_candidates = []

        for info in cached:
            if info.header.frame_id != frame_id:
                continue

            stamp_ns = _stamp_ns(
                info.header.stamp,
                allow_zero=True,
            )

            if stamp_ns == 0:
                if self.config.allow_static_camera_info:
                    static_candidates.append(info)
                continue

            if stamp_ns > rgb_stamp_ns:
                continue

            age_sec = (rgb_stamp_ns - stamp_ns) / 1e9

            if age_sec > self.config.maximum_camera_info_age_sec:
                continue

            candidates.append((stamp_ns, info))

        if candidates:
            return max(
                candidates,
                key=lambda item: item[0],
            )[1]

        if static_candidates:
            # Only allow an unambiguous static calibration.
            # Distinct configurations cannot be ordered by time.
            first = static_candidates[0]

            for other in static_candidates[1:]:
                if (
                    other.k != first.k
                    or other.d != first.d
                    or other.width != first.width
                    or other.height != first.height
                ):
                    return None

            return first

        return None

    # ========================================================================
    # INPUT / OBSERVATION CONTRACT
    # ========================================================================

    def _validate_observation(self, msg):
        if not isinstance(msg, PersonDetectionArray):
            raise ValueError("INVALID_OBSERVATION_TYPE")

        if msg.header.frame_id != self.config.odom_frame:
            raise ValueError("OBSERVATION_NOT_IN_ODOM")

        image_frame = msg.image_header.frame_id

        if not image_frame:
            raise ValueError("MISSING_RGB_OPTICAL_FRAME")

        rgb_ns = _stamp_ns(msg.image_header.stamp)
        source_ns = _stamp_ns(msg.header.stamp)

        if rgb_ns != source_ns:
            raise ValueError("SOURCE_RGB_TIMESTAMP_MISMATCH")

        if msg.processing_stage != PersonDetectionArray.APPEARANCE:
            raise ValueError("EXPECTED_APPEARANCE_STAGE")

        if msg.identity_evaluated or msg.tracking_session_id:
            raise ValueError("IDENTITY_ALREADY_EVALUATED")

        return rgb_ns, image_frame

    # ========================================================================
    # PROCESSING / TIMESTAMP-CORRECT TF
    # ========================================================================

    def _lookup_camera_pose(self, frame_id, rgb_stamp):
        """
        Look up odom <- camera at the image acquisition time.

        Using Time.from_msg(image stamp), not Time() and not
        latest available TF.
        """

        transform = self.tf_buffer.lookup_transform(
            self.config.odom_frame,
            frame_id,
            Time.from_msg(rgb_stamp),
            timeout=Duration(seconds=0.0),
        )

        if (
            transform.header.frame_id != self.config.odom_frame
            or transform.child_frame_id != frame_id
        ):
            raise ValueError("TF_FRAME_MISMATCH")

        quaternion = transform.transform.rotation

        rotation = rotation_from_quaternion(
            float(quaternion.x),
            float(quaternion.y),
            float(quaternion.z),
            float(quaternion.w),
        )

        translation_msg = transform.transform.translation

        translation = Vector3(
            float(translation_msg.x),
            float(translation_msg.y),
            float(translation_msg.z),
        )

        return _inverse_translation(
            rotation,
            translation,
        )

    # ========================================================================
    # PROCESSING / CAMERA VIEW
    # ========================================================================

    def prepare(self, msg) -> GeometryResult:
        """
        Build one view for the entire RGB observation.

        No exception for ordinary missing TF or calibration.
        Invalid observation contracts remain explicit failures.
        """

        rgb_ns, image_frame = self._validate_observation(msg)

        info = self._select_camera_info(
            image_frame,
            rgb_ns,
        )

        if info is None:
            return GeometryResult(
                None,
                "CAMERA_INFO_UNAVAILABLE_FOR_RGB",
                rgb_ns,
            )

        try:
            intrinsics = _intrinsics_from_camera_info(info)

        except (ValueError, TypeError, OverflowError):
            return GeometryResult(
                None,
                "CAMERA_CALIBRATION_INVALID",
                rgb_ns,
            )

        # Reject an incompatible calibration rather than mixing
        # the RGB projection with the upstream depth model.
        for candidate in msg.candidates:
            if not _calibration_matches_depth_evidence(
                candidate,
                intrinsics,
                self.config,
            ):
                return GeometryResult(
                    None,
                    "CAMERA_INTRINSICS_DIFFER_FROM_DEPTH_FUSION",
                    rgb_ns,
                )

        try:
            rotation, translation = self._lookup_camera_pose(
                image_frame,
                msg.image_header.stamp,
            )

        except TransformException:
            return GeometryResult(
                None,
                "TF_UNAVAILABLE_AT_RGB_TIMESTAMP",
                rgb_ns,
            )

        except (
            ValueError,
            TypeError,
            OverflowError,
        ):
            return GeometryResult(
                None,
                "INVALID_CAMERA_TF",
                rgb_ns,
            )

        view = CurrentView(
            stamp_ns=rgb_ns,
            camera_frame=image_frame,
            rotation_odom_to_camera=rotation,
            translation_odom_to_camera=translation,
            intrinsics=intrinsics,
            width_px=int(info.width),
            height_px=int(info.height),
            quality=1.0,
        )

        return GeometryResult(
            view=view,
            reason="VALID_CAMERA_VIEW",
            observation_stamp_ns=rgb_ns,
        )

    # ========================================================================
    # PROCESSING / PER-CANDIDATE DEPTH PROVENANCE
    # ========================================================================

    def validate_metric_provenance(
        self,
        candidate,
        *,
        rgb_stamp_ns,
        camera_frame,
    ) -> MetricProvenanceResult:
        """
        Validate one metric extent against its original RGB/depth
        timestamps and optical camera frame.

        This function DOES NOT modify the candidate or its quality.
        It must be checked before decoding metric evidence.

        Distinguishes:
            RGB observation time,
            depth acquisition time,
            fused metric-feature reference time.
        """

        extent = candidate.metric_extent
        evidence = candidate.depth_evidence

        if (
            extent.metadata.status != FeatureMetadata.AVAILABLE
            or extent.metadata.quality <= 0.0
            or extent.unit != ExtentFeature.METRES
        ):
            return MetricProvenanceResult(
                False,
                "METRIC_EXTENT_UNAVAILABLE",
                rgb_stamp_ns,
                None,
                None,
            )

        try:
            extent_ns = _stamp_ns(extent.metadata.header.stamp)

        except ValueError:
            return MetricProvenanceResult(
                False,
                "INVALID_METRIC_TIMESTAMP",
                rgb_stamp_ns,
                None,
                None,
            )

        if (
            extent_ns != rgb_stamp_ns
            or extent.metadata.header.frame_id != camera_frame
        ):
            return MetricProvenanceResult(
                False,
                "METRIC_RGB_PROVENANCE_MISMATCH",
                rgb_stamp_ns,
                None,
                None,
            )

        if (
            evidence.status != DepthEvidence.AVAILABLE
            or not evidence.image_selected
            or not evidence.calibration_available
        ):
            return MetricProvenanceResult(
                False,
                "DEPTH_PROVENANCE_UNAVAILABLE",
                rgb_stamp_ns,
                None,
                None,
            )

        try:
            depth_ns = _stamp_ns(evidence.image_header.stamp)

        except ValueError:
            return MetricProvenanceResult(
                False,
                "INVALID_DEPTH_TIMESTAMP",
                rgb_stamp_ns,
                None,
                None,
            )

        actual_offset_ns = depth_ns - rgb_stamp_ns

        if actual_offset_ns != int(evidence.image_offset_ns):
            return MetricProvenanceResult(
                False,
                "DEPTH_OFFSET_INCONSISTENT",
                rgb_stamp_ns,
                depth_ns,
                actual_offset_ns,
            )

        if (
            abs(actual_offset_ns) / 1e9
            > self.config.maximum_depth_rgb_offset_sec
        ):
            return MetricProvenanceResult(
                False,
                "DEPTH_RGB_SYNC_LIMIT_EXCEEDED",
                rgb_stamp_ns,
                depth_ns,
                actual_offset_ns,
            )

        if not (
            math.isfinite(extent.width)
            and math.isfinite(extent.height)
            and extent.width > 0.0
            and extent.height > 0.0
        ):
            return MetricProvenanceResult(
                False,
                "INVALID_METRIC_DIMENSIONS",
                rgb_stamp_ns,
                depth_ns,
                actual_offset_ns,
            )

        return MetricProvenanceResult(
            usable=True,
            reason="VALID_METRIC_RGB_DEPTH_PROVENANCE",
            rgb_stamp_ns=rgb_stamp_ns,
            depth_stamp_ns=depth_ns,
            depth_offset_ns=actual_offset_ns,
        )
