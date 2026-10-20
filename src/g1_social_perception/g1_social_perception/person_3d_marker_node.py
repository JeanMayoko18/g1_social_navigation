"""
Convert fused 3D person detections into RViz MarkerArray messages.


Responsibility:
    Subscribe to /yolo/detections_3d and publish visualization markers
    representing detected persons in the camera coordinate frame.

Flow:
    INPUT
        DetectionArray
            ↓
    FILTER / VALIDATE PERSON DETECTIONS
            ↓
    MARKER GENERATION
            ↓
    MarkerArray
            ↓
    OUTPUT
        /person_3d_markers
"""

import math
import rclpy
from rclpy.node import Node
import math
import time
from collections import OrderedDict

from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

from yolo_msgs.msg import DetectionArray

from g1_qos_manager.profiles import reliable_qos
from g1_qos_manager.resolver import resolve_subscription_qos
from g1_qos_manager.diagnostics import log_qos_resolution
from g1_social_msgs.msg import SocialZoneArray

class Person3DMarkerNode(Node):
    """Publish RViz markers for fused 3D person detections."""

    def __init__(self):
        super().__init__("person_3d_marker")

        # ---------------------------------------------------------------------
        # PARAMETERS
        # ---------------------------------------------------------------------

        self.declare_parameter(
            "input_topic",
            "/yolo/detections_3d",
        )

        self.declare_parameter(
            "output_topic",
            "/person_3d_markers",
        )

        self.declare_parameter(
            "person_class_id",
            0,
        )

        self.declare_parameter(
            "marker_scale",
            0.25,
        )

        self.declare_parameter(
            "marker_type",
            "sphere",
        )

        self.declare_parameter(
            "sphere_diameter",
            0.25,
        )

        self.declare_parameter(
            "cylinder_height",
            1.70,
        )

        self.declare_parameter(
            "cylinder_min_diameter",
            0.30,
        )

        self.declare_parameter(
            "social_zones_topic",
            "/social_perception/social_zones",
        )

        self.declare_parameter(
            "visualization_ground_z",
            -0.78,
        )

        self.declare_parameter(
            "show_social_zones",
            True,
        )

        self.declare_parameter(
            "social_zone_height",
            0.02,
        )

        self.declare_parameter(
            "social_zone_timeout_sec",
            1.0,
        )
        self.marker_id_ttl_sec = float(
            self.declare_parameter(
                "marker_id_ttl_sec",
                10.0,
            ).value
        )

        self.max_cached_marker_ids = max(
            1,
            int(
                self.declare_parameter(
                    "max_cached_marker_ids",
                    256,
                ).value
            ),
        )


        self.social_zone_timeout_sec = float(
            self.get_parameter(
                "social_zone_timeout_sec"
            ).value
        )

        self.social_zones_topic = (
            self.get_parameter("social_zones_topic")
            .get_parameter_value()
            .string_value
        )

        self.show_social_zones = (
            self.get_parameter("show_social_zones")
            .get_parameter_value()
            .bool_value
        )

        self.social_zone_height = (
            self.get_parameter("social_zone_height")
            .get_parameter_value()
            .double_value
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

        self.person_class_id = (
            self.get_parameter("person_class_id")
            .get_parameter_value()
            .integer_value
        )

        self.marker_scale = (
            self.get_parameter("marker_scale")
            .get_parameter_value()
            .double_value
        )

        self.marker_type = (
            self.get_parameter("marker_type")
            .get_parameter_value()
            .string_value
            .lower()
        )

        self.sphere_diameter = (
            self.get_parameter("sphere_diameter")
            .get_parameter_value()
            .double_value
        )

        self.cylinder_height = (
            self.get_parameter("cylinder_height")
            .get_parameter_value()
            .double_value
        )

        self.cylinder_min_diameter = (
            self.get_parameter("cylinder_min_diameter")
            .get_parameter_value()
            .double_value
        )

        valid_marker_types = (
            "sphere",
            "cylinder_mz",
            "cylinder_mx",
        )

        self.visualization_ground_z = (
            self.get_parameter("visualization_ground_z")
            .get_parameter_value()
            .double_value
        )

        if self.marker_type not in valid_marker_types:
            self.get_logger().warn(
                f"[PERSON MARKERS] Unsupported marker_type="
                f"'{self.marker_type}'. Falling back to 'sphere'."
            )
            self.marker_type = "sphere"
        # ---------------------------------------------------------------------
        # QOS
        # ---------------------------------------------------------------------

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

        social_zones_qos = resolve_subscription_qos(
            self,
            self.social_zones_topic,
            fallback_depth=10,
            prefer_sensor_data=False,
        )

        log_qos_resolution(
            self,
            self.social_zones_topic,
            social_zones_qos,
        )

        # ---------------------------------------------------------------------
        # ROS INTERFACES
        # ---------------------------------------------------------------------

        self.subscription = self.create_subscription(
            DetectionArray,
            self.input_topic,
            self.detections_callback,
            input_qos,
        )
        
        self.social_zones_sub = self.create_subscription(
            SocialZoneArray,
            self.social_zones_topic,
            self.social_zones_callback,
            social_zones_qos,
        )


        self.publisher = self.create_publisher(
            MarkerArray,
            self.output_topic,
            output_qos,
        )

        self.previous_marker_count = 0

        # ---------------------------------------------------------------------
        # RUNTIME STATE
        # ---------------------------------------------------------------------

        # Latest social-zone message received from the social-zone generator.
        # Zones are cached so that person markers, labels, and social zones can
        # be published together in one coherent MarkerArray.
        self.latest_social_zones = None
        self.latest_social_zones_received_at = None

        # Track previously published social-zone marker IDs so stale zones can
        # explicitly be removed from RViz.
        self.previous_social_zone_ids = set()

        # Stable RViz marker identity:
        # person_id -> marker_id
        #self.social_zone_marker_ids = {}
        # person_id -> (marker_id, last_seen_monotonic)
        # Least recently used entries are stored first.
        self.social_zone_marker_ids = OrderedDict()

        self.next_social_zone_marker_id = 1000
        self.latest_social_zones_received_at = None


        self.get_logger().info(
            f"[PERSON MARKERS] INPUT={self.input_topic} | "
            f"OUTPUT={self.output_topic} | "
            f"PERSON_CLASS_ID={self.person_class_id} | "
            f"MARKER_TYPE={self.marker_type} | "
            f"CYLINDER_HEIGHT={self.cylinder_height:.2f} | "
            f"CYLINDER_MIN_DIAMETER={self.cylinder_min_diameter:.2f}"
        )

    def detections_callback(
        self,
        msg: DetectionArray,
    ) -> None:
        """Convert fused person detections into RViz markers."""

        marker_array = MarkerArray()

        current_marker_id = 0

        for detection in msg.detections:
            if detection.class_id != self.person_class_id:
                continue

            position = detection.bbox3d.center.position

            if not self._is_valid_position(
                position.x,
                position.y,
                position.z,
            ):
                continue

            sphere_marker = self._create_person_marker(
                msg,
                detection,
                current_marker_id,
            )

            marker_array.markers.append(
                sphere_marker
            )

            current_marker_id += 1

            text_marker = self._create_text_marker(
                msg,
                detection,
                current_marker_id,
            )

            marker_array.markers.append(
                text_marker
            )

            current_marker_id += 1

        self._append_delete_markers(
            marker_array,
            msg,
            current_marker_id,
        )

        self.previous_marker_count = current_marker_id


        # Add the latest social-zone visualization to the same MarkerArray.
        self._append_social_zone_markers(
            marker_array
        )

        self.publisher.publish(
            marker_array
        )

    def social_zones_callback(
        self,
        msg: SocialZoneArray,
    ) -> None:
        """
        Cache the latest social zones for RViz visualization.

        Preserve the original observation timestamp and separately record
        local receive time for freshness monitoring.
        """

        self.latest_social_zones = msg

        self.latest_social_zones_received_at = (
            time.monotonic()
        )

    def _get_social_marker_id(
        self,
        person_id: str,
        now: float,
    ) -> int:

        cache = self.social_zone_marker_ids

        # Remove expired associations.
        expired = [
            key
            for key, (_, last_seen) in cache.items()
            if now - last_seen > self.marker_id_ttl_sec
        ]

        for key in expired:
            del cache[key]

        # Existing person: preserve stable marker ID.
        if person_id in cache:
            marker_id, _ = cache.pop(person_id)
            cache[person_id] = (marker_id, now)
            return marker_id

        # Hard memory limit: evict the oldest association.
        if len(cache) >= self.max_cached_marker_ids:
            cache.popitem(last=False)

        # Use a new ID to avoid accidental aliasing.
        marker_id = self.next_social_zone_marker_id
        self.next_social_zone_marker_id += 1

        cache[person_id] = (marker_id, now)

        return marker_id

    def _create_person_marker(
        self,
        msg,
        detection,
        marker_id: int,
    ) -> Marker:
        """
        Create the geometric RViz marker representing a detected person.

        Inputs:
            msg:
                Source DetectionArray containing the shared header.

            detection:
                Fused person detection containing its 3D center and estimated
                transverse dimensions.

            marker_id:
                Unique RViz marker identifier.

        Processing:
            Creates either:
                - a SPHERE centered on the detected person, or
                - a CYLINDER whose diameter is estimated from bbox3d.size.

            The marker type is selected through the marker_type ROS parameter.

        Output:
            Configured visualization_msgs/Marker.

        Called by:
            detections_callback().

        Side effects:
            None.
        """

        marker = Marker()

        marker.header = msg.header
        marker.ns = "persons"
        marker.id = marker_id
        marker.action = Marker.ADD
        marker.lifetime.sec = 2
        marker.lifetime.nanosec = 0

        x = float(
            detection.bbox3d.center.position.x
        )
        y = float(
            detection.bbox3d.center.position.y
        )
        z = float(
            detection.bbox3d.center.position.z
        )

        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = z

        marker.pose.orientation.w = 1.0

        if self.marker_type == "cylinder_mz":
            marker.type = Marker.CYLINDER

            estimated_diameter = max(
                float(detection.bbox3d.size.x),
                self.cylinder_min_diameter,
            )

            marker.scale.x = estimated_diameter
            marker.scale.y = estimated_diameter
            marker.scale.z = self.cylinder_height

            marker.pose.position.x = x
            marker.pose.position.y = y
            marker.pose.position.z = (
                self.visualization_ground_z
                + self.cylinder_height / 2.0
            )

            # Cylinder aligned with marker Z axis.
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0

        elif self.marker_type == "cylinder_mx":
            marker.type = Marker.CYLINDER

            estimated_diameter = max(
                float(detection.bbox3d.size.x),
                self.cylinder_min_diameter,
            )

            marker.scale.x = estimated_diameter
            marker.scale.y = estimated_diameter
            marker.scale.z = self.cylinder_height

            # Cylinder local Z rotated around X.
            marker.pose.orientation.x = -0.7071068
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 0.7071068

        else:
            marker.type = Marker.SPHERE

            marker.scale.x = self.sphere_diameter
            marker.scale.y = self.sphere_diameter
            marker.scale.z = self.sphere_diameter

            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0

        marker.color.r = 0.1
        marker.color.g = 1.0
        marker.color.b = 0.1
        marker.color.a = 0.45

        return marker

    def _create_text_marker(
        self,
        msg,
        detection,
        marker_id: int,
    ) -> Marker:
        """Create the text label displayed above a detected person."""

        marker = Marker()

        marker.header = msg.header
        marker.ns = "person_labels"
        marker.id = marker_id

        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.lifetime.sec = 2
        marker.lifetime.nanosec = 0

        x = detection.bbox3d.center.position.x
        y = detection.bbox3d.center.position.y
        z = detection.bbox3d.center.position.z

        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = (
            self.visualization_ground_z
            + self.cylinder_height
            + 0.15
        )

        marker.pose.orientation.w = 1.0

        marker.scale.z = 0.18

        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 1.0

        distance = math.sqrt(
            x * x
            + y * y
            + z * z
        )

        marker.text = (
            f"person detected at | {distance:.2f} m\n"
            f"x={x:.2f} y={y:.2f} z={z:.2f}"
        )

        return marker

    def _append_delete_markers(
        self,
        marker_array: MarkerArray,
        msg,
        current_marker_count: int,
    ) -> None:
        """
        Delete stale person and label markers left from previous callbacks.

        RViz identifies markers using the pair (namespace, id), therefore the
        DELETE marker must use the same namespace as the original ADD marker.
        """

        for marker_id in range(
            current_marker_count,
            self.previous_marker_count,
        ):
            marker = Marker()

            marker.header = msg.header
            marker.id = marker_id
            marker.action = Marker.DELETE

            if marker_id % 2 == 0:
                marker.ns = "persons"
            else:
                marker.ns = "person_labels"

            marker_array.markers.append(
                marker
            )

    def _create_social_zone_marker(
        self,
        msg: SocialZoneArray,
        zone,
        marker_id: int,
    ) -> Marker:
        """
        Create an RViz marker representing one person's social zone.

        Inputs:
            msg:
                SocialZoneArray providing the shared frame and timestamp.

            zone:
                Social zone associated with one tracked person.

            marker_id:
                Stable visualization marker identifier.

        Processing:
            Creates a flattened cylinder in the navigation XY plane.

            STATIC:
                Circular zone.

            DYNAMIC:
                Elliptical zone oriented along the estimated motion heading.

        Output:
            Configured visualization_msgs/Marker.

        Called by:
            _append_social_zone_markers().

        Side effects:
            None.
        """

        marker = Marker()

        marker.header = msg.header
        marker.ns = "social_zones"
        marker.id = marker_id

        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        marker.lifetime.sec = 2
        marker.lifetime.nanosec = 0

        # -----------------------------------------------------------------
        # POSITION
        # -----------------------------------------------------------------

        marker.pose.position.x = float(
            zone.center.x
        )

        marker.pose.position.y = float(
            zone.center.y
        )

        # Social zones belong to the navigation plane, not to the person's
        # measured camera-space height.
        marker.pose.position.z = (
            self.visualization_ground_z
            + self.social_zone_height / 2.0
        )

        # -----------------------------------------------------------------
        # ORIENTATION
        # -----------------------------------------------------------------

        half_heading = (
            float(zone.heading) / 2.0
        )

        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = math.sin(
            half_heading
        )
        marker.pose.orientation.w = math.cos(
            half_heading
        )

        # -----------------------------------------------------------------
        # GEOMETRY
        # -----------------------------------------------------------------

        # SocialZone stores semi-axis radii.
        # RViz Marker scale expects full dimensions.
        marker.scale.x = (
            2.0
            * float(zone.radius_longitudinal)
        )

        marker.scale.y = (
            2.0
            * float(zone.radius_lateral)
        )

        marker.scale.z = (
            self.social_zone_height
        )

        # -----------------------------------------------------------------
        # COLOR
        # -----------------------------------------------------------------

        if zone.zone_type == "DYNAMIC":
            marker.color.r = 1.0
            marker.color.g = 0.45
            marker.color.b = 0.0
            marker.color.a = 0.25

        else:
            marker.color.r = 0.2
            marker.color.g = 0.6
            marker.color.b = 1.0
            marker.color.a = 0.20

        return marker

    def _append_social_zone_markers(
        self,
        marker_array: MarkerArray,
    ) -> None:
        """
        Append fresh cached social-zone markers using stable person identities.

        Marker IDs are assigned once per person_id and are independent of the
        ordering of SocialZoneArray.zones.

        Therefore:

            person A -> marker 1000
            person B -> marker 1001

        remains true even if a later SocialZoneArray contains:

            [person B, person A]

        Stale cached social zones are never re-published.

        Zones that disappear from the latest observation are explicitly deleted
        from RViz while their person_id -> marker_id association is retained in
        case that same track reappears later.
        """

        if not self.show_social_zones:
            return

        # ================================================================
        # CACHE FRESHNESS
        # ================================================================

        zones_are_fresh = (
            self.latest_social_zones is not None
            and self.latest_social_zones_received_at
            is not None
            and (
                time.monotonic()
                - self.latest_social_zones_received_at
            )
            <= self.social_zone_timeout_sec
        )

        # ================================================================
        # STALE / UNAVAILABLE SOCIAL OBSERVATION
        # ================================================================

        if not zones_are_fresh:

            # Explicitly remove every social-zone marker still visible in RViz.
            for marker_id in (
                self.previous_social_zone_ids
            ):

                marker = Marker()

                if self.latest_social_zones is not None:
                    marker.header = (
                        self.latest_social_zones.header
                    )

                marker.ns = "social_zones"
                marker.id = marker_id
                marker.action = Marker.DELETE

                marker_array.markers.append(
                    marker
                )

            self.previous_social_zone_ids = set()

            return

        # ================================================================
        # CURRENT FRESH SOCIAL ZONES
        # ================================================================

        current_zone_ids = set()

        for zone in (
            self.latest_social_zones.zones
        ):

            person_id = str(
                zone.person_id
            ).strip()

            # A social-zone marker must have a stable owner.
            #
            # An empty person_id cannot provide stable marker identity, so do
            # not publish such a visualization object.
            if not person_id:

                self.get_logger().warn(
                    "[PERSON MARKERS] "
                    "Ignoring social zone without person_id.",
                    throttle_duration_sec=2.0,
                )

                continue

            # ------------------------------------------------------------
            # STABLE PERSON -> MARKER ID
            # ------------------------------------------------------------

            #marker_id = (
            #    self.social_zone_marker_ids.get(
            #        person_id
            #    )
            #)
            marker_id = self._get_social_marker_id(
                person_id,
                time.monotonic(),
            )

            if marker_id is None:

                marker_id = (
                    self.next_social_zone_marker_id
                )

                self.next_social_zone_marker_id += 1

                self.social_zone_marker_ids[
                    person_id
                ] = marker_id

                self.get_logger().debug(
                    "[PERSON MARKERS] "
                    "Assigned stable social-zone marker | "
                    f"PERSON={person_id!r} | "
                    f"MARKER_ID={marker_id}"
                )

            # ------------------------------------------------------------
            # CREATE MARKER
            # ------------------------------------------------------------

            marker = (
                self._create_social_zone_marker(
                    self.latest_social_zones,
                    zone,
                    marker_id,
                )
            )

            marker_array.markers.append(
                marker
            )

            current_zone_ids.add(
                marker_id
            )

        # ================================================================
        # DELETE ZONES THAT DISAPPEARED
        # ================================================================

        stale_ids = (
            self.previous_social_zone_ids
            - current_zone_ids
        )

        for marker_id in stale_ids:

            marker = Marker()

            marker.header = (
                self.latest_social_zones.header
            )

            marker.ns = "social_zones"
            marker.id = marker_id
            marker.action = Marker.DELETE

            marker_array.markers.append(
                marker
            )

        # ================================================================
        # UPDATE DISPLAYED STATE
        # ================================================================

        self.previous_social_zone_ids = (
            current_zone_ids
        )
        

    @staticmethod
    def _is_valid_position(
        x: float,
        y: float,
        z: float,
    ) -> bool:
        """Return True when a 3D position contains finite coordinates."""

        return (
            math.isfinite(x)
            and math.isfinite(y)
            and math.isfinite(z)
            #and z > 0.0
        )



def main(args=None):
    """Run the person 3D marker node."""

    rclpy.init(
        args=args
    )

    node = Person3DMarkerNode()

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
