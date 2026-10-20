"""
Maintain spatial state for already identified persons.

Responsibility:
    Consume 3D person detections expressed in base_footprint and maintain
    persistent spatial state for each person ID.

Flow:
    INPUT
        /yolo/detections_3d_base
            ↓
    VALIDATION
            ↓
    TRACK STATE UPDATE
            ↓
    VELOCITY ESTIMATION
            ↓
    TRACK STATE CLASSIFICATION
            ↓
    RESULT OBJECTS
            ↓
    OUTPUT
        /social_perception/tracked_persons

This node does not perform:
    - YOLO detection,
    - 2D identity association,
    - RGB-D fusion,
    - TF transformation.

Those responsibilities belong to earlier pipeline stages.
"""

import math
from dataclasses import dataclass
from typing import Dict, Optional
import time
import rclpy

from rclpy.node import Node
from rclpy.time import Time

from yolo_msgs.msg import DetectionArray

from g1_social_msgs.msg import (
    TrackedPerson,
    TrackedPersonArray,
)

from g1_qos_manager.diagnostics import log_qos_resolution
from g1_qos_manager.profiles import reliable_qos
from g1_qos_manager.resolver import resolve_subscription_qos


# =============================================================================
# RESULT OBJECTS / INTERNAL STATE
# =============================================================================


@dataclass
class SpatialTrackState:
    """
    Store spatial history for one persistent person ID.

    Attributes:
        person_id:
            Persistent identifier inherited from person_tracking_node.

        class_id:
            Semantic detector class ID.

        class_name:
            Semantic detector class name.

        confidence:
            Latest detector confidence.

        x, y, z:
            Latest position in target frame.

        last_observation_stamp_sec:
            ROS timestamp carried by the latest spatial observation.
            Used only for velocity and motion estimation.

        last_update_ros_sec:
            ROS node time at which the latest observation was processed.
            Used only for TRACKED / LOST / DELETE lifecycle management.

        vx, vy, vz:
            Current filtered velocity estimate.

        observation_count:
            Number of successful spatial observations.
    """

    person_id: str

    class_id: int
    class_name: str
    confidence: float

    x: float
    y: float
    z: float

    # Timestamp carried by the spatial observation.
    # Used only for motion/velocity estimation.
    last_observation_stamp_sec: float

    # ROS time at which this track was last received by this node.
    # Used only for lifecycle management.
    last_update_ros_sec: float

    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0

    observation_count: int = 1

# =============================================================================
# NODE
# =============================================================================


class PersonSpatialTrackingNode(Node):
    """
    Maintain spatial motion state for person IDs.

    Input:
        /yolo/detections_3d_base

    Output:
        /social_perception/tracked_persons
    """

    def __init__(self):
        super().__init__(
            "person_spatial_tracking"
        )

        # =====================================================================
        # INPUT / CONFIGURATION
        # =====================================================================

        self.declare_parameter(
            "input_topic",
            "/yolo/detections_3d_base",
        )

        self.declare_parameter(
            "output_topic",
            "/social_perception/tracked_persons",
        )

        self.declare_parameter(
            "expected_frame",
            "base_footprint",
        )

        self.declare_parameter(
            "person_class_id",
            0,
        )

        self.declare_parameter(
            "velocity_filter_alpha",
            0.35,
        )

        self.declare_parameter(
            "minimum_velocity_dt_sec",
            0.02,
        )

        self.declare_parameter(
            "maximum_velocity_dt_sec",
            1.00,
        )

        self.declare_parameter(
            "minimum_tracking_observations",
            3,
        )

        self.declare_parameter(
            "lost_timeout_sec",
            3.0,
        )

        self.declare_parameter(
            "delete_timeout_sec",
            5.0,
        )

        self.declare_parameter(
            "maximum_speed_mps",
            4.0,
        )

        self.declare_parameter(
            "publish_frequency_hz",
            15.0,
        )

        self.input_topic = (
            self.get_parameter("input_topic")
            .get_parameter_value()
            .string_value
        )

        self.output_topic = (
            self.get_parameter("output_topic")
            .get_parameter_value()
            .string_value
        )

        self.expected_frame = (
            self.get_parameter("expected_frame")
            .get_parameter_value()
            .string_value
        )

        self.person_class_id = (
            self.get_parameter("person_class_id")
            .get_parameter_value()
            .integer_value
        )

        self.velocity_filter_alpha = (
            self.get_parameter("velocity_filter_alpha")
            .get_parameter_value()
            .double_value
        )

        self.minimum_velocity_dt_sec = (
            self.get_parameter("minimum_velocity_dt_sec")
            .get_parameter_value()
            .double_value
        )

        self.maximum_velocity_dt_sec = (
            self.get_parameter("maximum_velocity_dt_sec")
            .get_parameter_value()
            .double_value
        )

        self.minimum_tracking_observations = (
            self.get_parameter("minimum_tracking_observations")
            .get_parameter_value()
            .integer_value
        )

        self.lost_timeout_sec = (
            self.get_parameter("lost_timeout_sec")
            .get_parameter_value()
            .double_value
        )

        self.delete_timeout_sec = (
            self.get_parameter("delete_timeout_sec")
            .get_parameter_value()
            .double_value
        )

        self.maximum_speed_mps = (
            self.get_parameter("maximum_speed_mps")
            .get_parameter_value()
            .double_value
        )

        self.publish_frequency_hz = (
            self.get_parameter("publish_frequency_hz")
            .get_parameter_value()
            .double_value
        )

        # =====================================================================
        # VALIDATION
        # =====================================================================

        self.velocity_filter_alpha = max(
            0.0,
            min(
                1.0,
                self.velocity_filter_alpha,
            ),
        )

        if self.publish_frequency_hz <= 0.0:
            self.publish_frequency_hz = 15.0

        # =====================================================================
        # INTERNAL STATE
        # =====================================================================

        self.tracks: Dict[
            str,
            SpatialTrackState,
        ] = {}

        # ================================================================
        # OBSERVATION PROVENANCE
        # ================================================================

        self.last_input_stamp = None
        self.last_input_stamp_ns = None
        self.last_input_received_monotonic = None

        self.new_observation_pending = False

        self.accepted_observation_arrays = 0
        self.empty_observation_arrays = 0
        self.rejected_observation_timestamps = 0
        self.stale_observation_cycles = 0

        self.declare_parameter(
            "observation_freshness_timeout_sec",
            1.0,
        )

        self.observation_freshness_timeout_sec = float(
            self.get_parameter(
                "observation_freshness_timeout_sec"
            ).value
        )

        if (
            not math.isfinite(self.observation_freshness_timeout_sec)
            or self.observation_freshness_timeout_sec <= 0.0
        ):
            raise ValueError(
                "observation_freshness_timeout_sec must be positive and finite"
            )

        self.arrays_received = 0
        self.persons_received = 0
        self.valid_updates = 0

        self.invalid_id = 0
        self.invalid_position = 0
        self.invalid_frame = 0
        self.rejected_velocity = 0

        # =====================================================================
        # QOS
        # =====================================================================

        input_qos = resolve_subscription_qos(
            self,
            self.input_topic,
            fallback_depth=10,
            prefer_sensor_data=False,
        )

        log_qos_resolution(
            self,
            self.input_topic,
            input_qos,
        )

        output_qos = reliable_qos(
            depth=10,
        )

        # =====================================================================
        # ROS INTERFACES
        # =====================================================================

        self.subscription = self.create_subscription(
            DetectionArray,
            self.input_topic,
            self.detections_callback,
            input_qos,
        )

        self.publisher = self.create_publisher(
            TrackedPersonArray,
            self.output_topic,
            output_qos,
        )

        self.publish_timer = self.create_timer(
            1.0 / self.publish_frequency_hz,
            self.publish_tracks,
        )

        self.diagnostics_timer = self.create_timer(
            5.0,
            self.log_diagnostics,
        )

        # =====================================================================
        # OUTPUT / REPORTING
        # =====================================================================

        self.get_logger().info("")
        self.get_logger().info("=" * 82)
        self.get_logger().info(
            "  PERSON SPATIAL TRACKING - INITIALIZATION"
        )
        self.get_logger().info("=" * 82)

        self.get_logger().info(
            f"  Input topic                  : {self.input_topic}"
        )

        self.get_logger().info(
            f"  Output topic                 : {self.output_topic}"
        )

        self.get_logger().info(
            f"  Expected frame               : {self.expected_frame}"
        )

        self.get_logger().info(
            f"  Velocity filter alpha        : "
            f"{self.velocity_filter_alpha:.2f}"
        )

        self.get_logger().info(
            f"  Minimum observations         : "
            f"{self.minimum_tracking_observations}"
        )

        self.get_logger().info(
            f"  Lost timeout                 : "
            f"{self.lost_timeout_sec:.2f} s"
        )

        self.get_logger().info(
            f"  Delete timeout               : "
            f"{self.delete_timeout_sec:.2f} s"
        )

        self.get_logger().info(
            f"  Maximum plausible speed      : "
            f"{self.maximum_speed_mps:.2f} m/s"
        )

        self.get_logger().info("-" * 82)

    # =========================================================================
    # COLLECTION
    # =========================================================================

    def detections_callback(
        self,
        msg: DetectionArray,
    ) -> None:
        """
        Accept only new and valid spatial observations.

        An accepted empty DetectionArray is an observation without
        transformed persons, not proof that the entire scene is clear.
        """

        self.arrays_received += 1

        if msg.header.frame_id != self.expected_frame:
            self.invalid_frame += 1

            self.get_logger().warning(
                "[SPATIAL TRACKING] Unexpected frame "
                f"{msg.header.frame_id!r}; "
                f"expected {self.expected_frame!r}",
                throttle_duration_sec=2.0,
            )
            return

        stamp = msg.header.stamp
        stamp_sec = int(stamp.sec)
        stamp_nanosec = int(stamp.nanosec)

        if (
            stamp_sec < 0
            or stamp_nanosec < 0
            or stamp_nanosec >= 1_000_000_000
        ):
            self.rejected_observation_timestamps += 1
            return

        stamp_ns = (
            stamp_sec * 1_000_000_000
            + stamp_nanosec
        )

        if (
            stamp_ns <= 0
            or (
                self.last_input_stamp_ns is not None
                and stamp_ns <= self.last_input_stamp_ns
            )
        ):
            self.rejected_observation_timestamps += 1

            self.get_logger().warning(
                "[SPATIAL TRACKING] Duplicate, invalid "
                "or out-of-order observation timestamp",
                throttle_duration_sec=2.0,
            )
            return

        accepted = []
        seen_ids = set()

        for detection in msg.detections:
            if detection.class_id != self.person_class_id:
                continue

            self.persons_received += 1

            person_id = str(detection.id).strip()

            if not person_id or person_id in seen_ids:
                self.invalid_id += 1
                self.get_logger().warning(
                    "[SPATIAL TRACKING] Invalid or duplicate person ID",
                    throttle_duration_sec=2.0,
                )
                return

            position = detection.bbox3d.center.position

            if not self._is_valid_position(
                position.x,
                position.y,
                position.z,
            ):
                self.invalid_position += 1
                self.get_logger().warning(
                    "[SPATIAL TRACKING] Invalid 3D position",
                    throttle_duration_sec=2.0,
                )
                return

            if not math.isfinite(float(detection.score)):
                self.invalid_position += 1
                return

            seen_ids.add(person_id)
            accepted.append((detection, person_id, position))

        # Commit only after the complete array passes validation.
        self.last_input_stamp = (
            stamp_sec,
            stamp_nanosec,
        )
        self.last_input_stamp_ns = stamp_ns
        self.last_input_received_monotonic = time.monotonic()

        self.accepted_observation_arrays += 1

        if not accepted:
            self.empty_observation_arrays += 1

        observation_stamp_sec = stamp_ns / 1e9

        for detection, person_id, position in accepted:
            self._update_spatial_track(
                person_id=person_id,
                class_id=detection.class_id,
                class_name=detection.class_name,
                confidence=float(detection.score),
                x=float(position.x),
                y=float(position.y),
                z=float(position.z),
                stamp_sec=observation_stamp_sec,
            )

            self.valid_updates += 1

        self.new_observation_pending = True

    # =========================================================================
    # PROCESSING / TRACK UPDATE
    # =========================================================================

    def _update_spatial_track(
        self,
        person_id: str,
        class_id: int,
        class_name: str,
        confidence: float,
        x: float,
        y: float,
        z: float,
        stamp_sec: float,
    ) -> None:
        """
        Create or update one spatial person track.

        Inputs:
            person_id:
                Persistent identity inherited from the 2D tracker.

            class_id, class_name, confidence:
                Latest semantic detection information.

            x, y, z:
                Current spatial observation in expected_frame.

            stamp_sec:
                ROS timestamp carried by the spatial observation.

        Processing:
            - Use observation timestamps for velocity estimation.
            - Use node ROS time for lifecycle management.
            - Reject physically implausible velocity spikes.
            - Apply exponential velocity smoothing.

        Output:
            None.

        Side effects:
            Creates or updates one SpatialTrackState.
        """

        now_ros_sec = (
            self.get_clock()
            .now()
            .nanoseconds
            / 1e9
        )

        track = self.tracks.get(
            person_id
        )

        # -------------------------------------------------------------------------
        # NEW TRACK
        # -------------------------------------------------------------------------

        if track is None:
            self.tracks[
                person_id
            ] = SpatialTrackState(
                person_id=person_id,
                class_id=class_id,
                class_name=class_name,
                confidence=confidence,
                x=x,
                y=y,
                z=z,
                last_observation_stamp_sec=stamp_sec,
                last_update_ros_sec=now_ros_sec,
            )

            return

        # -------------------------------------------------------------------------
        # OBSERVATION TIME DELTA
        # -------------------------------------------------------------------------

        dt = (
            stamp_sec
            - track.last_observation_stamp_sec
        )

        filtered_vx = track.vx
        filtered_vy = track.vy
        filtered_vz = track.vz

        # -------------------------------------------------------------------------
        # VELOCITY ESTIMATION
        # -------------------------------------------------------------------------

        if (
            self.minimum_velocity_dt_sec
            <= dt
            <= self.maximum_velocity_dt_sec
        ):
            raw_vx = (
                x - track.x
            ) / dt

            raw_vy = (
                y - track.y
            ) / dt

            raw_vz = (
                z - track.z
            ) / dt

            raw_speed = math.hypot(
                raw_vx,
                raw_vy,
            )

            if (
                raw_speed
                <= self.maximum_speed_mps
            ):
                alpha = (
                    self.velocity_filter_alpha
                )

                filtered_vx = (
                    alpha * raw_vx
                    + (1.0 - alpha) * track.vx
                )

                filtered_vy = (
                    alpha * raw_vy
                    + (1.0 - alpha) * track.vy
                )

                filtered_vz = (
                    alpha * raw_vz
                    + (1.0 - alpha) * track.vz
                )

            else:
                self.rejected_velocity += 1

        # -------------------------------------------------------------------------
        # TRACK UPDATE
        # -------------------------------------------------------------------------

        track.class_id = class_id
        track.class_name = class_name
        track.confidence = confidence

        track.x = x
        track.y = y
        track.z = z

        track.vx = filtered_vx
        track.vy = filtered_vy
        track.vz = filtered_vz

        # Observation time:
        # only used for motion estimation.
        track.last_observation_stamp_sec = (
            stamp_sec
        )

        # Node ROS time:
        # only used for track lifecycle management.
        track.last_update_ros_sec = (
            now_ros_sec
        )

        track.observation_count += 1

    # =========================================================================
    # PROCESSING / TRACK STATE
    # =========================================================================

    def _get_tracking_state(
        self,
        track: SpatialTrackState,
        now_sec: float,
    ) -> str:
        """
        Classify current track lifecycle state.

        Inputs:
            track:
                Current spatial track.

            now_sec:
                Current ROS node time in seconds.

        Output:
            TENTATIVE:
                Track has not yet accumulated enough observations.

            TRACKED:
                Track is confirmed and was recently updated.

            LOST:
                Track has temporarily stopped receiving observations but
                remains retained until delete_timeout_sec.
        """

        age = max(
            0.0,
            now_sec
            - track.last_update_ros_sec,
        )

        if (
            age
            >= self.lost_timeout_sec
        ):
            return "LOST"

        if (
            track.observation_count
            < self.minimum_tracking_observations
        ):
            return "TENTATIVE"

        return "TRACKED"



    def _remove_expired_tracks(
        self,
        now_sec: float,
    ) -> None:
        """
        Remove tracks that have not received a new observation within
        delete_timeout_sec.

        Inputs:
            now_sec:
                Current ROS node time in seconds.

        Side effects:
            Deletes expired tracks from self.tracks.
        """

        expired_ids = [
            person_id
            for person_id, track
            in self.tracks.items()
            if (
                now_sec
                - track.last_update_ros_sec
            ) > self.delete_timeout_sec
        ]

        for person_id in expired_ids:
            del self.tracks[
                person_id
            ]

    # =========================================================================
    # OUTPUT / RESULT OBJECTS
    # =========================================================================

    def publish_tracks(self) -> None:
        """
        Publish the tracking snapshot only after a new observation.

        The timer maintains track expiration, but does not create
        artificial camera observations.
        """

        now_sec = (
            self.get_clock().now().nanoseconds / 1e9
        )

        self._remove_expired_tracks(now_sec)

        if not self.new_observation_pending:
            self.stale_observation_cycles += 1
            return

        if (
            self.last_input_stamp is None
            or self.last_input_received_monotonic is None
        ):
            self.new_observation_pending = False
            return

        observation_age = (
            time.monotonic()
            - self.last_input_received_monotonic
        )

        if observation_age > self.observation_freshness_timeout_sec:
            self.new_observation_pending = False
            self.stale_observation_cycles += 1
            return

        output = TrackedPersonArray()

        # Keep the exact accepted upstream observation timestamp.
        output.header.stamp.sec = self.last_input_stamp[0]
        output.header.stamp.nanosec = self.last_input_stamp[1]
        output.header.frame_id = self.expected_frame

        for person_id, track in sorted(self.tracks.items()):
            person = TrackedPerson()

            person.id = person_id
            person.class_id = track.class_id
            person.class_name = track.class_name
            person.confidence = track.confidence

            person.position.x = track.x
            person.position.y = track.y
            person.position.z = track.z

            person.velocity.x = track.vx
            person.velocity.y = track.vy
            person.velocity.z = track.vz

            person.speed = float(
                math.hypot(track.vx, track.vy)
            )

            if person.speed > 1e-4:
                person.motion_heading = float(
                    math.atan2(track.vy, track.vx)
                )
            else:
                person.motion_heading = 0.0

            person.time_since_last_seen = float(
                max(
                    0.0,
                    now_sec - track.last_update_ros_sec,
                )
            )

            person.observation_count = int(
                track.observation_count
            )

            person.tracking_state = self._get_tracking_state(
                track,
                now_sec,
            )

            output.persons.append(person)

        self.publisher.publish(output)

        self.new_observation_pending = False

    # =========================================================================
    # VALIDATION / UTILITY
    # =========================================================================

    @staticmethod
    def _is_valid_position(
        x: float,
        y: float,
        z: float,
    ) -> bool:
        """Return True when all spatial coordinates are finite."""

        return (
            math.isfinite(x)
            and math.isfinite(y)
            and math.isfinite(z)
        )

    @staticmethod
    def _stamp_to_seconds(
        stamp,
    ) -> float:
        """Convert builtin_interfaces/Time to floating-point seconds."""

        return (
            float(stamp.sec)
            + float(stamp.nanosec) * 1e-9
        )

    # =========================================================================
    # OUTPUT / REPORTING
    # =========================================================================
    
    def log_diagnostics(self) -> None:
        """Report spatial tracking and observation freshness."""

        now_sec = (
            self.get_clock().now().nanoseconds / 1e9
        )

        tracked_count = 0
        tentative_count = 0
        lost_count = 0

        for track in self.tracks.values():
            state = self._get_tracking_state(
                track,
                now_sec,
            )

            if state == "TRACKED":
                tracked_count += 1
            elif state == "TENTATIVE":
                tentative_count += 1
            elif state == "LOST":
                lost_count += 1

        if self.last_input_received_monotonic is None:
            observation_age = float("inf")
        else:
            observation_age = max(
                0.0,
                time.monotonic()
                - self.last_input_received_monotonic,
            )

        self.get_logger().info(
            "[SPATIAL TRACKING] "
            f"ARRAYS={self.arrays_received} | "
            f"PERSONS={self.persons_received} | "
            f"UPDATES={self.valid_updates} | "
            f"ACTIVE={len(self.tracks)} | "
            f"TRACKED={tracked_count} | "
            f"TENTATIVE={tentative_count} | "
            f"LOST={lost_count} | "
            f"INVALID_ID={self.invalid_id} | "
            f"INVALID_POS={self.invalid_position} | "
            f"INVALID_FRAME={self.invalid_frame} | "
            f"VEL_REJECTED={self.rejected_velocity} | "
            f"OBS_ACCEPTED={self.accepted_observation_arrays} | "
            f"OBS_EMPTY={self.empty_observation_arrays} | "
            f"OBS_REJECTED_STAMPS="
            f"{self.rejected_observation_timestamps} | "
            f"STALE_CYCLES={self.stale_observation_cycles} | "
            f"OBS_AGE={observation_age:.2f}s"
        )


# =============================================================================
# ORCHESTRATION
# =============================================================================


def main(args=None):
    """Run the spatial person tracking node."""

    rclpy.init(
        args=args
    )

    node = PersonSpatialTrackingNode()

    try:
        rclpy.spin(
            node
        )

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
