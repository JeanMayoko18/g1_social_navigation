"""
Generate social zones from spatially tracked persons.

Responsibility:
    Convert tracked person states into social-space representations suitable
    for visualization and later Nav2 costmap integration.

Flow:
    INPUT
        /social_perception/tracked_persons
            ↓
    TRACK VALIDATION
            ↓
    STATIC / DYNAMIC CLASSIFICATION
            ↓
    SOCIAL ZONE GEOMETRY
            ↓
    RESULT OBJECTS
            ↓
    OUTPUT
        /social_perception/social_zones
"""

import math

import rclpy
from rclpy.node import Node

from g1_social_msgs.msg import (
    SocialZone,
    SocialZoneArray,
    TrackedPersonArray,
)
from g1_social_perception.social_zone_geometry import (
    build_social_zone,
)
import json

from std_msgs.msg import String
from g1_qos_manager.diagnostics import log_qos_resolution
from g1_qos_manager.profiles import reliable_qos
from g1_qos_manager.resolver import resolve_subscription_qos


class SocialZoneGeneratorNode(Node):
    """Generate static or directional social zones around tracked persons."""

    def __init__(self):
        super().__init__("social_zone_generator")

        # =====================================================================
        # INPUT / CONFIGURATION
        # =====================================================================

        self.declare_parameter(
            "input_topic",
            "/social_perception/tracked_persons",
        )

        self.declare_parameter(
            "output_topic",
            "/social_perception/social_zones",
        )

        self.declare_parameter(
            "expected_frame",
            "base_footprint",
        )

        self.declare_parameter(
            "static_radius_m",
            0.80,
        )

        self.declare_parameter(
            "dynamic_speed_threshold_mps",
            0.15,
        )

        self.declare_parameter(
            "dynamic_base_longitudinal_radius_m",
            0.90,
        )

        self.declare_parameter(
            "dynamic_base_lateral_radius_m",
            0.70,
        )

        self.declare_parameter(
            "dynamic_speed_gain",
            0.60,
        )

        self.declare_parameter(
            "maximum_longitudinal_radius_m",
            2.00,
        )

        self.declare_parameter(
            "include_tentative_tracks",
            False,
        )

        self.declare_parameter(
            "include_lost_tracks",
            False,
        )

        self.declare_parameter(
            "forward_only",
            True,
        )

        self.declare_parameter(
            "minimum_forward_x_m",
            0.10,
        )

        self.declare_parameter(
            "maximum_distance_m",
            5.00,
        )

        self.declare_parameter(
            "forward_field_of_view_deg",
            120.0,
        )

        # =====================================================================
        # READ
        # =====================================================================

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

        self.static_radius_m = (
            self.get_parameter("static_radius_m")
            .get_parameter_value()
            .double_value
        )

        self.dynamic_speed_threshold_mps = (
            self.get_parameter("dynamic_speed_threshold_mps")
            .get_parameter_value()
            .double_value
        )

        self.dynamic_base_longitudinal_radius_m = (
            self.get_parameter(
                "dynamic_base_longitudinal_radius_m"
            )
            .get_parameter_value()
            .double_value
        )

        self.dynamic_base_lateral_radius_m = (
            self.get_parameter(
                "dynamic_base_lateral_radius_m"
            )
            .get_parameter_value()
            .double_value
        )

        self.dynamic_speed_gain = (
            self.get_parameter("dynamic_speed_gain")
            .get_parameter_value()
            .double_value
        )

        self.maximum_longitudinal_radius_m = (
            self.get_parameter(
                "maximum_longitudinal_radius_m"
            )
            .get_parameter_value()
            .double_value
        )

        self.include_tentative_tracks = (
            self.get_parameter("include_tentative_tracks")
            .get_parameter_value()
            .bool_value
        )

        self.include_lost_tracks = (
            self.get_parameter("include_lost_tracks")
            .get_parameter_value()
            .bool_value
        )

        self.forward_only = (
            self.get_parameter("forward_only")
            .get_parameter_value()
            .bool_value
        )

        self.minimum_forward_x_m = (
            self.get_parameter("minimum_forward_x_m")
            .get_parameter_value()
            .double_value
        )

        self.maximum_distance_m = (
            self.get_parameter("maximum_distance_m")
            .get_parameter_value()
            .double_value
        )

        self.forward_field_of_view_deg = (
            self.get_parameter("forward_field_of_view_deg")
            .get_parameter_value()
            .double_value
        )

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
            TrackedPersonArray,
            self.input_topic,
            self.tracked_persons_callback,
            input_qos,
        )

        self.publisher = self.create_publisher(
            SocialZoneArray,
            self.output_topic,
            output_qos,
        )
        # Report individual generation outcomes without changing SocialZoneArray.
        self.individual_results_publisher = self.create_publisher(
            String,
            "/social_perception/individual_zone_results",
            reliable_qos(depth=10),
        )

        # =====================================================================
        # DIAGNOSTICS
        # =====================================================================

        self.arrays_received = 0
        self.persons_received = 0
        self.zones_generated = 0
        self.static_zones = 0
        self.dynamic_zones = 0
        self.filtered_tracks = 0

        # ================================================================
        # OBSERVATION VALIDATION
        # ================================================================

        self.last_input_stamp_ns = None
        
        self.tracking_session_id = ""
        self.retired_tracking_sessions = set()

        self.invalid_timestamps = 0
        self.rejected_arrays = 0
        self.ambiguous_empty_arrays = 0
        self.confirmed_input_empty_arrays = 0
        self.published_arrays = 0

        self.diagnostics_timer = self.create_timer(
            5.0,
            self.log_diagnostics,
        )

        self.get_logger().info("")
        self.get_logger().info("=" * 82)
        self.get_logger().info(
            "  SOCIAL ZONE GENERATOR - INITIALIZATION"
        )
        self.get_logger().info("=" * 82)

        self.get_logger().info(
            f"  Input topic             : {self.input_topic}"
        )

        self.get_logger().info(
            f"  Output topic            : {self.output_topic}"
        )

        self.get_logger().info(
            f"  Expected frame          : {self.expected_frame}"
        )

        self.get_logger().info(
            f"  Static radius           : {self.static_radius_m:.2f} m"
        )

        self.get_logger().info(
            f"  Dynamic threshold       : "
            f"{self.dynamic_speed_threshold_mps:.2f} m/s"
        )

        self.get_logger().info("-" * 82)

    # =========================================================================
    # COLLECTION
    # =========================================================================

    def _is_in_forward_sector(
        self,
        person,
    ) -> bool:
        """
        Return True when the person's center is inside the robot's
        forward observation sector.

        The position must be expressed in base_footprint:
            +x: front
            +y: left
            -y: right
        """

        if not self.forward_only:
            return True

        x = float(person.position.x)
        y = float(person.position.y)

        if not math.isfinite(x) or not math.isfinite(y):
            return False

        # Reject persons behind or too close to the robot's transverse axis.
        if x < self.minimum_forward_x_m:
            return False

        distance = math.hypot(x, y)

        if distance > self.maximum_distance_m:
            return False

        angle = abs(
            math.atan2(y, x)
        )

        half_fov_rad = math.radians(
            self.forward_field_of_view_deg / 2.0
        )

        return angle <= half_fov_rad



    def tracked_persons_callback(
        self,
        msg: TrackedPersonArray,
    ) -> None:
        """
        Generate zones from current, confirmed observations.

        Filtered persons remain represented in the source count.
        Missing zones never imply missing persons or clearance.

        A derived array is complete only if:
          - upstream person coverage is complete;
          - every source person is currently observed and TRACKED;
          - every source person has a generated social zone.
        """
        self.arrays_received += 1

        if msg.header.frame_id != self.expected_frame:
            self.rejected_arrays += 1
            return

        session = str(msg.tracking_session_id).strip()

        if (
            not session
            or session in self.retired_tracking_sessions
        ):
            self.rejected_arrays += 1
            return

        sec = int(msg.header.stamp.sec)
        nanosec = int(msg.header.stamp.nanosec)

        if (
            sec < 0
            or not 0 <= nanosec < 1_000_000_000
        ):
            self.invalid_timestamps += 1
            return

        stamp_ns = sec * 1_000_000_000 + nanosec

        if (
            stamp_ns <= 0
            or (
                session == self.tracking_session_id
                and self.last_input_stamp_ns is not None
                and stamp_ns <= self.last_input_stamp_ns
            )
        ):
            self.invalid_timestamps += 1
            return

        source_count = int(msg.source_detection_count)

        if source_count < 0:
            self.rejected_arrays += 1
            return

        output = SocialZoneArray()
        output.header = msg.header
        output.tracking_session_id = session

        seen_ids = set()
        observed_count = 0
        all_current_tracked = True

        staged_zones = []
        staged_filtered = 0
        staged_persons = 0
        individual_results = []

        for person in msg.persons:
            staged_persons += 1

            person_id = str(person.id).strip()
            identity_generation = person.identity_generation

            source_sec = int(
                person.last_observation_stamp.sec
            )
            source_nanosec = int(
                person.last_observation_stamp.nanosec
            )

            source_ns = (
                source_sec * 1_000_000_000
                + source_nanosec
            )

            observed = bool(
                person.observed_in_current_array
            )

            tracking_state = str(
                person.tracking_state
            ).strip()

            if (
                not person_id
                or person_id in seen_ids
                or person.tracking_session_id != session
                or source_sec < 0
                or not 0 <= source_nanosec < 1_000_000_000
                or source_ns <= 0
                or type(identity_generation) is not int
                or identity_generation <= 0
                or source_ns > stamp_ns
                or observed != (source_ns == stamp_ns)
                or tracking_state not in {
                    "TENTATIVE",
                    "TRACKED",
                    "LOST",
                }
                or (
                    observed
                    and tracking_state == "LOST"
                )
            ):
                self.rejected_arrays += 1
                self.get_logger().warning(
                    "[SOCIAL ZONE] Invalid person provenance.",
                    throttle_duration_sec=2.0,
                )
                return

            seen_ids.add(person_id)

            def record_individual_result(status, reason, direction_basis="UNKNOWN"):
                individual_results.append({
                    "tracking_session_id": session,
                    "person_id": person_id,
                    "identity_generation": identity_generation,
                    "tracking_state": tracking_state,
                    "observed_in_current_array": observed,
                    "observation_stamp_ns": source_ns,
                    "status": status,
                    "reason": reason,
                    "direction_basis": direction_basis,
                })

            if not observed:
                all_current_tracked = False
                staged_filtered += 1

                record_individual_result(
                    "UNAVAILABLE",
                    "NO_CURRENT_OBSERVATION",
                )
                continue

            if tracking_state != "TRACKED":
                all_current_tracked = False
                staged_filtered += 1

                record_individual_result(
                    "UNAVAILABLE",
                    "TRACK_NOT_CONFIRMED",
                )
                continue

            # The navigation/social-memory consumer requires
            # TRACKED persons only. Parameters allowing
            # tentative/lost visualizations do not override
            # this authoritative output contract.
            values = (
                float(person.position.x),
                float(person.position.y),
                float(person.position.z),
                float(person.speed),
            )

            if person.speed >= self.dynamic_speed_threshold_mps:
                values += (float(person.motion_heading),)

            if (
                not all(
                    math.isfinite(value)
                    for value in values
                )
                or person.speed < 0.0
            ):
                self.rejected_arrays += 1
                self.get_logger().warning(
                    "[SOCIAL ZONE] Invalid current geometry.",
                    throttle_duration_sec=2.0,
                )
                return

            if not self._is_in_forward_sector(person):
                staged_filtered += 1

                record_individual_result(
                    "FILTERED",
                    "OUTSIDE_CONFIGURED_FORWARD_SECTOR",
                )
                continue

            zone = self._build_zone(person)
            staged_zones.append(zone)

            record_individual_result(
                "GENERATED",
                "VALID_CURRENT_OBSERVATION",
                (
                    "MOTION_DIRECTION"
                    if zone.zone_type == "DYNAMIC"
                    else "UNKNOWN"
                ),
            )

        if observed_count > source_count:
            self.rejected_arrays += 1
            self.get_logger().warning(
                "[SOCIAL ZONE] Observations exceed source count.",
                throttle_duration_sec=2.0,
            )
            return

        # The original source may contain candidates that
        # cannot yet be spatially tracked.
        source_complete = bool(
            msg.source_observation_complete
        )

        output.zones = staged_zones
        output.source_person_count = len(msg.persons)

        output.source_observation_complete = bool(
            source_complete
            and observed_count == source_count
            and all_current_tracked
            and len(staged_zones) == len(msg.persons)
        )

        if (
            self.tracking_session_id
            and session != self.tracking_session_id
        ):
            self.retired_tracking_sessions.add(
                self.tracking_session_id
            )

        self.tracking_session_id = session
        self.last_input_stamp_ns = stamp_ns

        if msg.persons and not staged_zones:
            self.ambiguous_empty_arrays += 1

        if (
            not msg.persons
            and source_count == 0
            and source_complete
        ):
            self.confirmed_input_empty_arrays += 1

        self.publisher.publish(output)

        # Publish individual results for the same validated source frame.
        self.individual_results_publisher.publish(
            String(
                data=json.dumps(
                    {
                        "version": 1,
                        "producer": "social_zone_generator",
                        "tracking_session_id": session,
                        "source_stamp_ns": stamp_ns,
                        "source_frame": msg.header.frame_id,
                        "source_observation_complete": bool(
                            output.source_observation_complete
                        ),
                        "persons": individual_results,
                    },
                    separators=(",", ":"),
                    allow_nan=False,
                )
            )
        )

        self.persons_received += staged_persons
        self.filtered_tracks += staged_filtered

        self.static_zones += sum(
            zone.zone_type == "STATIC"
            for zone in staged_zones
        )

        self.dynamic_zones += sum(
            zone.zone_type == "DYNAMIC"
            for zone in staged_zones
        )

        self.published_arrays += 1
        self.zones_generated += len(staged_zones)

    # =========================================================================
    # PROCESSING / VALIDATION
    # =========================================================================

    def _should_use_track(
        self,
        tracking_state: str,
    ) -> bool:
        """Return whether a tracked person should generate a social zone."""

        if tracking_state == "TRACKED":
            return True

        if (
            tracking_state == "TENTATIVE"
            and self.include_tentative_tracks
        ):
            return True

        if (
            tracking_state == "LOST"
            and self.include_lost_tracks
        ):
            return True

        return False

    # =========================================================================
    # PROCESSING / SOCIAL ZONE GEOMETRY
    # =========================================================================

    def _build_zone(self, person) -> SocialZone:
        """Build a social zone using the shared geometry model."""
        return build_social_zone(
            person,
            static_radius_m=self.static_radius_m,
            dynamic_speed_threshold_mps=(
                self.dynamic_speed_threshold_mps
            ),
            dynamic_base_longitudinal_radius_m=(
                self.dynamic_base_longitudinal_radius_m
            ),
            dynamic_base_lateral_radius_m=(
                self.dynamic_base_lateral_radius_m
            ),
            dynamic_speed_gain=self.dynamic_speed_gain,
            maximum_longitudinal_radius_m=(
                self.maximum_longitudinal_radius_m
            ),
        )

    # =========================================================================
    # OUTPUT / REPORTING
    # =========================================================================

    def log_diagnostics(self) -> None:
        """Report social-zone generation and observation quality."""

        self.get_logger().info(
            "[SOCIAL ZONE] "
            f"ARRAYS_RX={self.arrays_received} | "
            f"ARRAYS_TX={self.published_arrays} | "
            f"PERSONS={self.persons_received} | "
            f"ZONES={self.zones_generated} | "
            f"STATIC={self.static_zones} | "
            f"DYNAMIC={self.dynamic_zones} | "
            f"FILTERED={self.filtered_tracks} | "
            f"INVALID_STAMPS={self.invalid_timestamps} | "
            f"REJECTED_ARRAYS={self.rejected_arrays} | "
            f"AMBIGUOUS_EMPTY={self.ambiguous_empty_arrays} | "
            f"INPUT_EMPTY={self.confirmed_input_empty_arrays} | "
            f"FORWARD_ONLY={self.forward_only} | "
            f"MIN_X={self.minimum_forward_x_m:.2f}m | "
            f"MAX_DISTANCE={self.maximum_distance_m:.2f}m | "
            f"FOV={self.forward_field_of_view_deg:.1f}deg"
        )


# =============================================================================
# ORCHESTRATION
# =============================================================================


def main(args=None):
    """Run the social zone generator node."""

    rclpy.init(
        args=args
    )

    node = SocialZoneGeneratorNode()

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
