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
        Generate social zones from valid tracked persons.

        Preserve observation timestamps and existing safety filters.
        Report why an observed person did not generate a zone.
        """

        self.arrays_received += 1

        # -----------------------------------------------------
        # 1. Validate frame
        # -----------------------------------------------------
        if msg.header.frame_id != self.expected_frame:
            self.rejected_arrays += 1
            self.get_logger().warning(
                "[SOCIAL ZONE] Rejected frame | "
                f"FRAME={msg.header.frame_id!r} | "
                f"EXPECTED={self.expected_frame!r}",
                throttle_duration_sec=2.0,
            )
            return

        # -----------------------------------------------------
        # 2. Validate timestamp
        # -----------------------------------------------------
        stamp = msg.header.stamp
        sec = int(stamp.sec)
        nanosec = int(stamp.nanosec)

        if sec < 0 or not (0 <= nanosec < 1_000_000_000):
            self.invalid_timestamps += 1
            return

        stamp_ns = sec * 1_000_000_000 + nanosec

        if (
            stamp_ns <= 0
            or (
                self.last_input_stamp_ns is not None
                and stamp_ns <= self.last_input_stamp_ns
            )
        ):
            self.invalid_timestamps += 1
            self.get_logger().warning(
                "[SOCIAL ZONE] Invalid or repeated timestamp",
                throttle_duration_sec=2.0,
            )
            return

        # -----------------------------------------------------
        # 3. Initialize output
        # -----------------------------------------------------
        output = SocialZoneArray()
        output.header = msg.header

        rejected_geometry = False
        filtered_in_this_array = 0

        # -----------------------------------------------------
        # 4. Validate each tracked person
        # -----------------------------------------------------
        for person in msg.persons:
            self.persons_received += 1

            person_id = str(person.id).strip()
            tracking_state = str(person.tracking_state).strip()

            x = float(person.position.x)
            y = float(person.position.y)
            z = float(person.position.z)
            speed = float(person.speed)

            if not self._should_use_track(tracking_state):
                self.filtered_tracks += 1
                filtered_in_this_array += 1

                self.get_logger().warning(
                    "[SOCIAL ZONE FILTER] "
                    f"REASON=TRACK_STATE | "
                    f"ID={person_id} | "
                    f"STATE={tracking_state} | "
                    f"OBS={person.observation_count}",
                    throttle_duration_sec=2.0,
                )
                continue

            if not (
                person_id
                and math.isfinite(x)
                and math.isfinite(y)
                and math.isfinite(z)
                and math.isfinite(speed)
                and speed >= 0.0
            ):
                rejected_geometry = True
                self.filtered_tracks += 1

                self.get_logger().warning(
                    "[SOCIAL ZONE FILTER] "
                    f"REASON=INVALID_GEOMETRY | "
                    f"ID={person_id}",
                    throttle_duration_sec=2.0,
                )
                continue

            if speed >= self.dynamic_speed_threshold_mps:
                heading = float(person.motion_heading)

                if not math.isfinite(heading):
                    rejected_geometry = True
                    self.filtered_tracks += 1

                    self.get_logger().warning(
                        "[SOCIAL ZONE FILTER] "
                        f"REASON=INVALID_HEADING | "
                        f"ID={person_id}",
                        throttle_duration_sec=2.0,
                    )
                    continue

            if not self._is_in_forward_sector(person):
                self.filtered_tracks += 1
                filtered_in_this_array += 1

                distance = math.hypot(x, y)
                angle_deg = math.degrees(math.atan2(y, x))

                self.get_logger().warning(
                    "[SOCIAL ZONE FILTER] "
                    f"REASON=OUTSIDE_FORWARD_SECTOR | "
                    f"ID={person_id} | "
                    f"X={x:.3f} | "
                    f"Y={y:.3f} | "
                    f"Z={z:.3f} | "
                    f"DIST={distance:.3f} | "
                    f"ANGLE={angle_deg:.1f} | "
                    f"MIN_X={self.minimum_forward_x_m:.2f} | "
                    f"MAX_DIST={self.maximum_distance_m:.2f} | "
                    f"FOV={self.forward_field_of_view_deg:.1f}",
                    throttle_duration_sec=2.0,
                )
                continue

            # -------------------------------------------------
            # 5. Generate zone
            # -------------------------------------------------
            zone = self._build_zone(person)
            output.zones.append(zone)

            self.get_logger().info(
                "[SOCIAL ZONE GENERATED] "
                f"ID={person_id} | "
                f"STATE={tracking_state} | "
                f"TYPE={zone.zone_type} | "
                f"X={x:.3f} | Y={y:.3f} | "
                f"DIST={math.hypot(x, y):.3f}",
                throttle_duration_sec=2.0,
            )

        # -----------------------------------------------------
        # 6. Reject incomplete geometry
        # -----------------------------------------------------
        if rejected_geometry:
            self.rejected_arrays += 1
            self.get_logger().warning(
                "[SOCIAL ZONE] Rejected incomplete array",
                throttle_duration_sec=2.0,
            )
            return

        # Do not turn filtered persons into an empty scene.
        if msg.persons and not output.zones:
            self.ambiguous_empty_arrays += 1

            self.get_logger().warning(
                "[SOCIAL ZONE] All tracks filtered | "
                f"PERSONS={len(msg.persons)} | "
                f"FILTERED={filtered_in_this_array}",
                throttle_duration_sec=2.0,
            )
            return

        if not msg.persons:
            self.confirmed_input_empty_arrays += 1

        # -----------------------------------------------------
        # 7. Publish valid observation
        # -----------------------------------------------------
        self.publisher.publish(output)

        self.static_zones += sum(
            zone.zone_type == "STATIC"
            for zone in output.zones
        )

        self.dynamic_zones += sum(
            zone.zone_type == "DYNAMIC"
            for zone in output.zones
        )

        self.last_input_stamp_ns = stamp_ns
        self.published_arrays += 1
        self.zones_generated += len(output.zones)

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

    def _build_zone(
        self,
        person,
    ) -> SocialZone:
        """
        Create one social zone from one tracked person.

        Static:
            Circular zone.

        Dynamic:
            Directional ellipse whose longitudinal radius grows with speed.
        """

        zone = SocialZone()

        zone.person_id = person.id
        zone.center = person.position

        zone.person_speed = float(
            person.speed
        )

        zone.tracking_state = (
            person.tracking_state
        )

        if (
            person.speed
            < self.dynamic_speed_threshold_mps
        ):
            zone.zone_type = "STATIC"

            zone.heading = 0.0

            zone.radius_longitudinal = float(
                self.static_radius_m
            )

            zone.radius_lateral = float(
                self.static_radius_m
            )

            return zone

        zone.zone_type = "DYNAMIC"

        zone.heading = float(
            person.motion_heading
        )

        longitudinal_radius = (
            self.dynamic_base_longitudinal_radius_m
            + self.dynamic_speed_gain
            * float(person.speed)
        )

        zone.radius_longitudinal = float(
            min(
                longitudinal_radius,
                self.maximum_longitudinal_radius_m,
            )
        )

        zone.radius_lateral = float(
            self.dynamic_base_lateral_radius_m
        )

        return zone

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
