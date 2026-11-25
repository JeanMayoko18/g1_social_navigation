
#!/usr/bin/env python3
"""Local person-relative directional keepout mask for ROS 2 Humble Nav2.

LEFT/RIGHT are interpreted in the tracked person's heading frame.
Only a bounded neighborhood of the person is restricted. Nav2 owns
planning, avoidance and replanning; this node does not inspect paths.
"""

import copy
import hashlib
import math
import time

import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener
import json
from types import SimpleNamespace

from std_msgs.msg import String, Header
from g1_social_msgs.msg import (
    DirectionalConstraintRequest,
    DirectionalConstraintStatus,
    SocialZoneArray,
)


def _nanoseconds(stamp):
    """Return a nonnegative ROS time or duration in nanoseconds."""
    sec = int(stamp.sec)
    nanosec = int(stamp.nanosec)

    if sec < 0 or not 0 <= nanosec < 1_000_000_000:
        return 0

    return sec * 1_000_000_000 + nanosec


def _yaw(quaternion):
    """Get the planar yaw from a valid quaternion."""
    values = (
        float(quaternion.x),
        float(quaternion.y),
        float(quaternion.z),
        float(quaternion.w),
    )

    if not all(math.isfinite(value) for value in values):
        raise ValueError("Non-finite quaternion")

    norm = math.sqrt(
        sum(value * value for value in values)
    )

    if norm < 1e-8:
        raise ValueError("Invalid quaternion")

    x, y, z, w = (
        value / norm for value in values
    )

    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


class DirectionalKeepoutMaskNode(Node):
    """Publish a bounded person-relative mask and track its lease."""

    def __init__(self):
        super().__init__("directional_keepout_mask_node")

        defaults = {
            "request_topic":
                "/social_navigation/directional_constraint_request",
            "status_topic":
                "/social_navigation/directional_constraint_status",
            "zones_topic":
                "/social_perception/social_zones",
            "map_topic":
                "/map",
            "mask_topic":
                "/social_navigation/directional_keepout_mask",
            "qualified_geometry_topic":
                "/social_perception/social_geometry_status",
            "qualified_geometry_max_delay_sec": 1.0,
            "memory_geometry_max_age_sec": 2.0,

            "zone_max_age_sec": 1.5,
            "publish_frequency": 5.0,
            "robot_radius_m": 0.30,
            "person_clearance_m": 0.35,
            "local_longitudinal_extension_m": 1.0,
            "local_lateral_extension_m": 1.0,
            "max_map_cells": 2_000_000,
            "max_lease_sec": 120.0,
            "directional_robot_frame": "base_footprint",
        }

        for name, value in defaults.items():
            self.declare_parameter(name, value)

        self.params = {
            name: self.get_parameter(name).value
            for name in defaults
        }

        self.directional_robot_frame = str(
            self.params["directional_robot_frame"]
        ).strip()

        for name in (
            "zone_max_age_sec",
            "publish_frequency",
            "robot_radius_m",
            "person_clearance_m",
            "local_longitudinal_extension_m",
            "local_lateral_extension_m",
            "max_lease_sec",
        ):
            value = float(self.params[name])

            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"{name} must be finite and positive"
                )

        if int(self.params["max_map_cells"]) <= 0:
            raise ValueError(
                "max_map_cells must be positive"
            )

        # --------------------------------------------------
        # OBSERVATION STATE
        # --------------------------------------------------

        self.map_msg = None
        self.zones_msg = None
        self.zones_received_at = 0.0

        self.last_zone_stamp_ns = 0
        self.last_tracking_session = ""
        self.retired_sessions = set()

        # Historical geometry is distinct from current observation evidence.
        self._zone_geometry_cache = {}

        # Maximum duration for the stationary-person hypothesis.
        self.cached_zone_max_age_sec = 2.0

        # Conservative growth of spatial uncertainty during observation loss.
        self.cached_zone_uncertainty_m_per_sec = 0.50

        # --------------------------------------------------
        # DIRECTIONAL REQUEST STATE
        # --------------------------------------------------

        self.active_request = None
        self.request_expires_at = 0.0
        # Freeze the robot-relative direction at mask installation.
        self.direction_reference_heading_map = None

        self.state = "INACTIVE"
        self.reason = "NO_DIRECTIONAL_REQUEST"
        self.mask_id = ""

        self.last_mask_digest = ""
        self.last_mask_published_at = 0.0

        # --------------------------------------------------
        # TF
        # --------------------------------------------------

        self.tf_buffer = Buffer()

        self.tf_listener = TransformListener(
            self.tf_buffer,
            self,
            spin_thread=False,
        )

        # --------------------------------------------------
        # ROS INTERFACES
        # --------------------------------------------------

        latched_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.mask_pub = self.create_publisher(
            OccupancyGrid,
            str(self.params["mask_topic"]),
            latched_qos,
        )

        self.status_pub = self.create_publisher(
            DirectionalConstraintStatus,
            str(self.params["status_topic"]),
            latched_qos,
        )

        self.map_sub = self.create_subscription(
            OccupancyGrid,
            str(self.params["map_topic"]),
            self._map_callback,
            latched_qos,
        )

        self.zones_sub = self.create_subscription(
            SocialZoneArray,
            str(self.params["zones_topic"]),
            self._zones_callback,
            10,
        )

        self.qualified_geometry_msg = None
        self.qualified_geometry_received_at = 0.0

        self.qualified_geometry_sub = self.create_subscription(
            String,
            str(self.params["qualified_geometry_topic"]),
            self._qualified_geometry_callback,
            10,
        )

        self._last_geometry_evidence = None

        self.request_sub = self.create_subscription(
            DirectionalConstraintRequest,
            str(self.params["request_topic"]),
            self._request_callback,
            10,
        )

        self.timer = self.create_timer(
            1.0 / float(
                self.params["publish_frequency"]
            ),
            self._tick,
        )

        self.get_logger().info(
            "[DIRECTIONAL KEEPOUT] Ready | "
            "geometry=LOCAL | "
            "renewal=ENABLED | "
            "planning_owner=NAV2"
        )

    # --------------------------------------------------
    # REQUEST IDENTITY
    # --------------------------------------------------

    @staticmethod
    def _request_identity(request):
        """Bind renewals and revocations to one authorization."""
        return (
            str(request.request_id),
            str(request.navigation_goal_id),
            str(request.interaction_id),
            str(request.person_id),
            str(request.tracking_session_id),
            int(request.identity_generation),
            str(
                request.directional_constraint
            ).strip().upper(),
        )

    def _same_request(self, first, second):
        """Compare the exact authorization identities."""
        return (
            self._request_identity(first)
            == self._request_identity(second)
        )

    def _select_current_zone_for_install(self, request):
        """
        Fallback to an actual, validated SocialZoneArray.

        Never use memory, a different identity or an
        expired observation for initial mask installation.
        """
        snapshot = self.zones_msg

        if snapshot is None or self.map_msg is None:
            return None

        source_ns = _nanoseconds(snapshot.header.stamp)
        now_ns = self.get_clock().now().nanoseconds

        max_age = float(self.params["zone_max_age_sec"])

        if not (
            source_ns > 0
            and -0.1 <= (now_ns - source_ns) / 1e9 <= max_age
            and 0.0 <= (
                time.monotonic() - self.zones_received_at
            ) <= max_age
            and str(snapshot.tracking_session_id)
                == str(request.tracking_session_id)
        ):
            return None

        matches = [
            zone
            for zone in snapshot.zones
            if (
                str(zone.person_id) == str(request.person_id)
                and str(zone.tracking_session_id)
                    == str(request.tracking_session_id)
                and int(zone.identity_generation)
                    == int(request.identity_generation)
                and str(zone.tracking_state) == "TRACKED"
                and _nanoseconds(zone.observation_stamp)
                    == source_ns
            )
        ]

        if len(matches) != 1:
            return None

        return (
            copy.deepcopy(matches[0]),
            copy.deepcopy(snapshot.header),
            "OBSERVED",
        )
    
    def _request_lifetime(self, request):
        """Check a directional authorization lease."""
        duration_ns = _nanoseconds(
            request.validity_duration
        )

        duration = duration_ns / 1e9

        if not (
            0.0 < duration
            <= float(self.params["max_lease_sec"])
        ):
            return None

        return duration

    def _qualified_geometry_callback(self, msg):
        """Receive qualified geometry without renewing source timestamps."""
        try:
            payload = json.loads(msg.data)

            if not isinstance(payload, dict):
                return

            if (
                type(payload.get("version")) is not int
                or payload["version"] != 2
                or payload.get("producer")
                    != "social_navigation_decision"
                or payload.get("authoritative_for_navigation") is not False
                or not isinstance(payload.get("persons"), list)
                or not str(payload.get("map_frame", "")).strip()
            ):
                return

            evaluation_ns = int(payload["evaluation_stamp_ns"])
            age = (
                self.get_clock().now().nanoseconds - evaluation_ns
            ) / 1e9

            maximum = float(
                self.params["qualified_geometry_max_delay_sec"]
            )

            if not (-0.1 <= age <= maximum):
                return

            previous = self.qualified_geometry_msg

            if (
                previous is not None
                and evaluation_ns
                    <= int(previous["evaluation_stamp_ns"])
            ):
                return

            self.qualified_geometry_msg = payload
            self.qualified_geometry_received_at = time.monotonic()

        except (
            ValueError,
            TypeError,
            KeyError,
            OverflowError,
        ):
            return
        
    def _select_directional_geometry(self, request):
        """
        Select one qualified social geometry for one exact D request.

        OBSERVED:
            May establish the first directional installation.

        MEMORY:
            May maintain only an already verified installation
            belonging to this exact request and person identity.

        Returns (zone, header, provenance), or None.
        """
        payload = self.qualified_geometry_msg

        if payload is None or self.map_msg is None:
            return None

        now_ns = self.get_clock().now().nanoseconds
        now_mono = time.monotonic()

        target_frame = str(self.map_msg.header.frame_id)

        if payload.get("map_frame") != target_frame:
            return None

        evaluation_ns = int(
            payload.get("evaluation_stamp_ns", 0)
        )

        delivery_age = (now_ns - evaluation_ns) / 1e9
        receipt_age = (
            now_mono - self.qualified_geometry_received_at
        )

        maximum_delay = float(
            self.params["qualified_geometry_max_delay_sec"]
        )

        if not (
            -0.1 <= delivery_age <= maximum_delay
            and 0.0 <= receipt_age <= maximum_delay
        ):
            return None

        request_key = self._request_identity(request)

        # Match the complete tracking identity, not only person_id.
        matches = []

        for row in payload.get("persons", []):
            if not isinstance(row, dict):
                continue

            try:
                identity_matches = (
                    str(row.get("tracking_session_id", ""))
                        == str(request.tracking_session_id)
                    and str(row.get("person_id", ""))
                        == str(request.person_id)
                    and int(row.get("identity_generation", 0))
                        == int(request.identity_generation)
                )
            except (ValueError, TypeError):
                continue

            if identity_matches:
                matches.append(row)

        if len(matches) != 1:
            return None

        row = matches[0]

        provenance = str(
            row.get("status", "")
        ).strip().upper()

        if provenance not in {"OBSERVED", "MEMORY"}:
            return None

        geometry = row.get("geometry")

        if not isinstance(geometry, dict):
            return None

        if geometry.get("frame_id") != target_frame:
            return None

        try:
            source_ns = int(
                geometry["source_stamp_ns"]
            )

            if source_ns <= 0:
                return None

            source_age = (now_ns - source_ns) / 1e9

            uncertainty = 0.0

            if provenance == "OBSERVED":
                if not (
                    row.get("observed_now") is True
                    and payload.get("source_status") == "MATCHED"
                    and -0.1 <= source_age
                        <= float(self.params["zone_max_age_sec"])
                ):
                    return None

            else:
                # Memory cannot authorize a new installation.
                # This exact request must first have been
                # verified with current OBSERVED geometry.
                if (
                    self._verified_directional_request
                    != request_key
                ):
                    return None

                quality = row.get("historical_quality")

                if not isinstance(quality, dict):
                    return None

                if (
                    quality.get("memory_within_age_horizon")
                    is not True
                    or quality.get("uncertainty_status")
                        != "ASSUMPTION_BOUND_WITHIN_HORIZON"
                ):
                    return None

                if not (
                    0.0 <= source_age
                    <= float(
                        self.params["memory_geometry_max_age_sec"]
                    )
                ):
                    return None

                uncertainty = float(
                    quality["position_uncertainty_m"]
                )

                if (
                    not math.isfinite(uncertainty)
                    or uncertainty < 0.0
                ):
                    return None

            position = [
                float(value)
                for value in geometry["position"]
            ]

            heading = float(
                geometry["heading"]
            )

            longitudinal = float(
                geometry["radius_longitudinal"]
            )

            lateral = float(
                geometry["radius_lateral"]
            )

            if (
                len(position) != 3
                or not all(
                    math.isfinite(value)
                    for value in position
                )
                or not all(
                    math.isfinite(value)
                    for value in (
                        heading,
                        longitudinal,
                        lateral,
                    )
                )
                or longitudinal <= 0.0
                or lateral <= 0.0
            ):
                return None

        except (
            ValueError,
            TypeError,
            KeyError,
            OverflowError,
        ):
            return None

        # Historical uncertainty enlarges the social zone.
        # The stored person position is not moved or refreshed.
        zone = SimpleNamespace(
            center=SimpleNamespace(
                x=position[0],
                y=position[1],
                z=position[2],
            ),
            heading=heading,
            radius_longitudinal=(
                longitudinal + uncertainty
            ),
            radius_lateral=(
                lateral + uncertainty
            ),
        )

        header = Header()
        header.frame_id = target_frame
        header.stamp = Time(
            nanoseconds=source_ns
        ).to_msg()

        return zone, header, provenance
    
    def _valid_install_request(self, request):
        """Validate an installation request."""
        identifiers = (
            request.request_id,
            request.navigation_goal_id,
            request.interaction_id,
            request.person_id,
            request.tracking_session_id,
        )

        return bool(
            str(request.action).strip().upper()
                == "INSTALL"
            and str(
                request.directional_constraint
            ).strip().upper() in {"LEFT", "RIGHT"}
            and all(
                str(value).strip()
                for value in identifiers
            )
            and int(request.identity_generation) > 0
            and self._request_lifetime(request)
                is not None
        )


    # --------------------------------------------------
    # MAP CALLBACK
    # --------------------------------------------------

    def _map_callback(self, msg):
        """Cache map geometry without creating global restrictions."""
        try:
            info = msg.info

            width = int(info.width)
            height = int(info.height)
            count = width * height

            if (
                not str(msg.header.frame_id).strip()
                or width <= 0
                or height <= 0
                or count > int(
                    self.params["max_map_cells"]
                )
                or len(msg.data) != count
                or not math.isfinite(
                    float(info.resolution)
                )
                or float(info.resolution) <= 0.0
                or not math.isfinite(
                    float(info.origin.position.x)
                )
                or not math.isfinite(
                    float(info.origin.position.y)
                )
            ):
                raise ValueError(
                    "Invalid map geometry"
                )

            _yaw(info.origin.orientation)

            self.map_msg = copy.deepcopy(msg)

        except (
            ValueError,
            TypeError,
            AttributeError,
            OverflowError,
        ) as exc:
            self.map_msg = None

            if self.active_request is not None:
                self._fail(
                    f"INVALID_MAP: {exc}"
                )

    # --------------------------------------------------
    # SOCIAL ZONE CALLBACK
    # --------------------------------------------------

    def _zones_callback(self, msg):
        """Accept ordered and fresh tracking observations."""
        now_ns = self.get_clock().now().nanoseconds

        source_ns = _nanoseconds(
            msg.header.stamp
        )

        session = str(
            msg.tracking_session_id
        ).strip()

        max_age_ns = int(
            float(self.params["zone_max_age_sec"])
            * 1e9
        )

        if session in self.retired_sessions:
            return

        if (
            self.last_tracking_session
            and session != self.last_tracking_session
        ):
            self.retired_sessions.add(
                self.last_tracking_session
            )

            self.last_tracking_session = session
            self.last_zone_stamp_ns = 0
            self.zones_msg = None

            current_request = self.active_request

            if (
                current_request is not None
                and not (
                    self.state == "PUBLISHED"
                    and self._verified_directional_request
                        == self._request_identity(current_request)
                    and all(self._costmap_confirmed.values())
                )
            ):
                self._fail("TRACKING_SESSION_CHANGED")

        elif (
            not self.last_tracking_session
            and session
        ):
            self.last_tracking_session = session


        source_person_count = int(msg.source_person_count)
        zone_count = len(msg.zones)
        coverage_complete = bool(
            msg.source_observation_complete
        )

        valid = bool(
            session
            and str(msg.header.frame_id).strip()
            and source_ns > self.last_zone_stamp_ns
            and source_ns > 0
            and -100_000_000
                <= now_ns - source_ns
                <= max_age_ns
            and source_person_count >= zone_count
            and (
                not coverage_complete
                or source_person_count == zone_count
            )
        )


        if valid:
            seen = set()

            for zone in msg.zones:
                person_id = str(
                    zone.person_id
                ).strip()

                values = (
                    zone.center.x,
                    zone.center.y,
                    zone.center.z,
                    zone.heading,
                    zone.radius_lateral,
                    zone.radius_longitudinal,
                )

                if (
                    not person_id
                    or person_id in seen
                    or str(zone.tracking_session_id)
                        != session
                    or str(zone.tracking_state)
                        != "TRACKED"
                    or _nanoseconds(
                        zone.observation_stamp
                    ) != source_ns
                    or not all(
                        math.isfinite(float(value))
                        for value in values
                    )
                    or float(
                        zone.radius_lateral
                    ) <= 0.0
                    or float(
                        zone.radius_longitudinal
                    ) <= 0.0
                    or int(
                        zone.identity_generation
                    ) <= 0
                ):
                    valid = False
                    break

                seen.add(person_id)

        if not valid:
            self.get_logger().warning(
                "[DIRECTIONAL MASK] Invalid zone array ignored | "
                f"session={session} | "
                f"zones={len(msg.zones)} | "
                f"source_person_count={msg.source_person_count} | "
                f"coverage_complete={msg.source_observation_complete}",
                throttle_duration_sec=2.0,
            )
            return

        self.last_zone_stamp_ns = source_ns

        self.zones_msg = copy.deepcopy(msg)
        self.zones_received_at = time.monotonic()

    # --------------------------------------------------
    # REQUEST CALLBACK
    # --------------------------------------------------

    def _request_callback(self, msg):
        """Install, renew or revoke an exact authorization."""
        action = str(
            msg.action
        ).strip().upper()

        if action == "REVOKE":
            self._revoke_request(msg)
            return

        if action != "INSTALL":
            return

        if not self._valid_install_request(msg):
            self.get_logger().warning(
                "[DIRECTIONAL KEEPOUT] Invalid INSTALL"
            )

            if self.active_request is None:
                self.state = "FAILED"
                self.reason = "INVALID_INSTALL_REQUEST"
                self.mask_id = ""
                self._publish_status(msg)

            return

        current = self.active_request

        if current is None:
            self._install_request(msg)

        elif self._same_request(current, msg):
            self._renew_request(msg)

        else:
            self.get_logger().warning(
                "[DIRECTIONAL KEEPOUT] "
                "Replacement requires REVOKE"
            )

    # --------------------------------------------------
    # INSTALL
    # --------------------------------------------------

    def _install_request(self, request):
        """Install one exact directional authorization."""
        lifetime = self._request_lifetime(request)

        if lifetime is None:
            return

        self.active_request = copy.deepcopy(request)

        # Never inherit a mask verification from a
        # previous request or tracking identity.
        self._verified_directional_request = None
        self._last_geometry_evidence = None

        self.direction_reference_heading_map = None

        self.request_expires_at = (
            time.monotonic() + lifetime
        )

        self.state = "INSTALLING"
        self.reason = "WAITING_FOR_LOCAL_MASK"
        self.mask_id = ""

        self.last_mask_digest = ""
        self.last_mask_published_at = 0.0

        self._observed_mask_id = ""

        self._initialize_costmap_observation()

        self._costmap_confirmed = {
            "global": False,
            "local": False,
        }

        self._costmap_probe_points = {
            "global": [],
            "local": [],
        }

        self._publish_status(
            self.active_request
        )

    # --------------------------------------------------
    # RENEWAL
    # --------------------------------------------------

    def _renew_request(self, request):
        """Renew only an active, non-revoking constraint."""
        if self.active_request is None:
            return

        if self.state == "REVOKING":
            return

        if self.state == "FAILED":
            self._publish_status(self.active_request)
            return

        if time.monotonic() >= self.request_expires_at:
            self._fail("DIRECTIONAL_REQUEST_EXPIRED")
            return

        lifetime = self._request_lifetime(request)

        if lifetime is None:
            return

        self.request_expires_at = time.monotonic() + lifetime
        self._publish_status(self.active_request)

    # --------------------------------------------------
    # REVOKE
    # --------------------------------------------------

    def _revoke_request(self, request):
        """
        Revoke one exact directional request.

        If this request never published a directional mask,
        complete revocation without waiting for occupied-cell
        probes that do not exist.

        If a mask was published, retain the normal neutral-mask
        confirmation protocol for both costmaps.
        """
        self._initialize_costmap_observation()

        current = self.active_request

        if current is None:
            return

        if not self._same_request(current, request):
            return

        if self.state == "REVOKING":
            self._publish_status(current)
            return

        had_mask = self.last_mask_published_at > 0.0

        self.state = "REVOKING"
        self.reason = (
            "WAITING_FOR_NEUTRAL_MASK_COSTMAP_UPDATE"
        )
        self.mask_id = ""

        self._verified_directional_request = None

        self._costmap_confirmed = {
            "global": False,
            "local": False,
        }

        self._revoke_stamp_ns = (
            self.get_clock().now().nanoseconds
        )

        if not self._publish_clear_mask():
            self._fail("NEUTRAL_MASK_PUBLICATION_FAILED")
            return

        # This exact request never published occupied cells.
        if not had_mask:
            self.state = "REVOKED"
            self.reason = (
                "NO_DIRECTIONAL_MASK_WAS_PUBLISHED"
            )

            self._publish_status(current)

            self._last_revoked_request = copy.deepcopy(
                current
            )
            self._last_revoked_at = time.monotonic()

            self.active_request = None
            self.direction_reference_heading_map = None
            self.request_expires_at = 0.0

            self.last_mask_digest = ""
            self._observed_mask_id = ""
            self._last_geometry_evidence = None

            return

        # A directional grid was published. The existing
        # costmap callback must confirm neutralization.
        self._publish_status(current)

    # --------------------------------------------------
    # STATUS
    # --------------------------------------------------
    def _initialize_costmap_observation(self):
        """Initialize subscribers once, without changing Nav2 filters."""
        if getattr(self, "_costmap_observation_initialized", False):
            return

        self._costmap_observation_initialized = True
        self._costmap_grids = {
            "global": None,
            "local": None,
        }
        self._costmap_confirmed = {
            "global": False,
            "local": False,
        }
        self._costmap_probe_points = {
            "global": [],
            "local": [],
        }
        self._observed_mask_id = ""
        self._mask_observation_stamp_ns = 0
        self._revoke_stamp_ns = 0
        self._last_revoked_request = None
        self._last_revoked_at = 0.0

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        self._global_costmap_sub = self.create_subscription(
            OccupancyGrid,
            "/global_costmap/costmap",
            lambda msg: self._costmap_observation_callback(
                "global", msg
            ),
            qos,
        )
        self._local_costmap_sub = self.create_subscription(
            OccupancyGrid,
            "/local_costmap/costmap",
            lambda msg: self._costmap_observation_callback(
                "local", msg
            ),
            qos,
        )

    def _costmap_cell_value(self, grid, x, y, source_frame):
        """Read an observed cost at a map-frame world position."""
        try:
            target = str(grid.header.frame_id).strip()
            source = str(source_frame).strip()

            if not target or not source:
                return None

            if target != source:
                transform = self.tf_buffer.lookup_transform(
                    target,
                    source,
                    Time(),
                    timeout=Duration(seconds=0.0),
                )

                angle = _yaw(transform.transform.rotation)
                cosine = math.cos(angle)
                sine = math.sin(angle)

                tx = transform.transform.translation.x
                ty = transform.transform.translation.y

                x, y = (
                    cosine * x - sine * y + tx,
                    sine * x + cosine * y + ty,
                )

            info = grid.info
            ox = info.origin.position.x
            oy = info.origin.position.y
            angle = _yaw(info.origin.orientation)

            dx = x - ox
            dy = y - oy

            cosine = math.cos(angle)
            sine = math.sin(angle)

            lx = cosine * dx + sine * dy
            ly = -sine * dx + cosine * dy

            resolution = float(info.resolution)

            if resolution <= 0:
                return None

            ix = math.floor(lx / resolution)
            iy = math.floor(ly / resolution)

            if (
                ix < 0
                or iy < 0
                or ix >= int(info.width)
                or iy >= int(info.height)
            ):
                return None

            index = iy * int(info.width) + ix
            return int(grid.data[index])

        except (
            TransformException,
            ValueError,
            TypeError,
            AttributeError,
            IndexError,
            OverflowError,
        ):
            return None

    def _costmap_observation_callback(self, name, msg):
        """Verify actual mask effects in each Nav2 costmap."""
        self._costmap_grids[name] = msg

        request = self.active_request

        if request is None:
            return

        if self.state not in {"PUBLISHED", "REVOKING"}:
            return

        stamp_ns = _nanoseconds(msg.header.stamp)

        threshold = (
            self._revoke_stamp_ns
            if self.state == "REVOKING"
            else self._mask_observation_stamp_ns
        )

        if threshold <= 0 or stamp_ns <= threshold:
            return

        probes = self._costmap_probe_points[name]

        if len(probes) < 3 or self.map_msg is None:
            return

        reference_frame = self.map_msg.header.frame_id
        matched = 0

        for x, y in probes:
            value = self._costmap_cell_value(
                msg,
                x,
                y,
                reference_frame,
            )

            if value is None:
                continue

            if self.state == "PUBLISHED" and value >= 99:
                matched += 1

            elif (
                self.state == "REVOKING"
                and 0 <= value < 90
            ):
                matched += 1

        if matched < 3:
            return

        if self._costmap_confirmed[name]:
            return

        self._costmap_confirmed[name] = True

        both_confirmed = all(
            self._costmap_confirmed.values()
        )

        # --------------------------------------------------
        # REVOKE: existing exact neutral-mask verification
        # --------------------------------------------------

        if self.state == "REVOKING" and both_confirmed:
            previous = self.active_request

            self.state = "REVOKED"
            self.reason = (
                "NEUTRAL_MASK_OBSERVED_IN_BOTH_COSTMAPS"
            )
            self.mask_id = ""

            self._verified_directional_request = None

            self._publish_status(previous)

            self._last_revoked_request = copy.deepcopy(previous)
            self._last_revoked_at = time.monotonic()

            self.active_request = None
            self.direction_reference_heading_map = None
            self.request_expires_at = 0.0
            self.last_mask_digest = ""

            return

        # --------------------------------------------------
        # INSTALL: latch first verified OBSERVED installation
        # --------------------------------------------------

        if self.state == "PUBLISHED" and both_confirmed:
            if (
                self._last_geometry_evidence == "OBSERVED"
                and str(self.mask_id).strip()
            ):
                self._verified_directional_request = (
                    self._request_identity(request)
                )

                self.get_logger().info(
                    "[DIRECTIONAL MASK] "
                    "Initial D installation verified | "
                    f"goal={request.navigation_goal_id} | "
                    f"person={request.person_id} | "
                    f"direction={request.directional_constraint} | "
                    "global=True | local=True | "
                    "memory_maintenance_eligible=True"
                )

            if (
                self._last_geometry_evidence == "MEMORY"
                and self._verified_directional_request
                    == self._request_identity(request)
            ):
                self.reason = (
                    "DIRECTIONAL_MASK_CONFIRMED_FROM_MEMORY"
                )

            else:
                self.reason = (
                    "DIRECTIONAL_MASK_OBSERVED_IN_COSTMAPS"
                )

        self._publish_status(request)

    def _prepare_directional_mask_probes(self, mask):
        """
        Prepare pre-mask free-space probes for the new grid.

        A different grid invalidates previous costmap evidence.
        """
        info = mask.info

        width = int(info.width)
        height = int(info.height)
        resolution = float(info.resolution)

        ox = float(info.origin.position.x)
        oy = float(info.origin.position.y)

        angle = _yaw(info.origin.orientation)

        cosine = math.cos(angle)
        sine = math.sin(angle)

        occupied = [
            index
            for index, value in enumerate(mask.data)
            if int(value) == 100
        ]

        stride = max(
            1,
            len(occupied) // 128,
        )

        points = []

        for index in occupied[::stride]:
            ix = index % width
            iy = index // width

            if iy >= height:
                continue

            lx = (ix + 0.5) * resolution
            ly = (iy + 0.5) * resolution

            x = (
                ox
                + cosine * lx
                - sine * ly
            )

            y = (
                oy
                + sine * lx
                + cosine * ly
            )

            points.append((x, y))

        self._costmap_confirmed = {
            "global": False,
            "local": False,
        }

        self._costmap_probe_points = {
            "global": [],
            "local": [],
        }

        for name, grid in self._costmap_grids.items():
            if grid is None:
                continue

            for x, y in points:
                value = self._costmap_cell_value(
                    grid,
                    x,
                    y,
                    mask.header.frame_id,
                )

                if value is not None and 0 <= value < 90:
                    self._costmap_probe_points[name].append(
                        (x, y)
                    )

        self._mask_observation_stamp_ns = (
            self.get_clock().now().nanoseconds
        )

    def _publish_status(self, request=None):
        """Publish observed costmap evidence for the exact request."""
        self._initialize_costmap_observation()

        status = DirectionalConstraintStatus()
        status.header.stamp = self.get_clock().now().to_msg()
        status.header.frame_id = (
            self.map_msg.header.frame_id
            if self.map_msg is not None
            else "map"
        )

        if request is not None:
            for field in (
                "request_id",
                "navigation_goal_id",
                "interaction_id",
                "person_id",
                "tracking_session_id",
                "identity_generation",
                "directional_constraint",
            ):
                setattr(status, field, getattr(request, field))

        status.state = self.state
        status.reason = self.reason
        status.mask_id = self.mask_id

        observed = self._costmap_confirmed

        status.global_costmap_confirmed = bool(
            observed["global"]
            and self.state in {
                "PUBLISHED",
                "REVOKED",
            }
        )

        status.local_costmap_confirmed = bool(
            observed["local"]
            and self.state in {
                "PUBLISHED",
                "REVOKED",
            }
        )

        self.status_pub.publish(status)

    # --------------------------------------------------
    # FAILURE HANDLING
    # --------------------------------------------------

    def _fail(self, reason):
        """Neutralize a failed optional mask without authorizing motion."""
        if self.active_request is None:
            return

        reason = str(reason or "UNKNOWN_MASK_FAILURE")
        changed = (
            self.state != "FAILED"
            or self.reason != reason
        )

        self._verified_directional_request = None
        self.state = "FAILED"
        self.reason = reason
        self.mask_id = ""

        cleared = self._publish_clear_mask()

        if changed:
            self.get_logger().warning(
                "[DIRECTIONAL MASK] Optional constraint failed | "
                f"reason={reason} | "
                f"neutral_mask_published={cleared} | "
                "costmap_removal_verified=False | "
                "navigation_authorized=False"
            )

        self._publish_status(self.active_request)

    # --------------------------------------------------
    # MASK CONSTRUCTION
    # --------------------------------------------------

    def _new_mask(self, data):
        """Create an OccupancyGrid using source-map geometry."""
        if self.map_msg is None:
            return None

        mask = OccupancyGrid()

        mask.header.frame_id = (
            self.map_msg.header.frame_id
        )

        mask.header.stamp = (
            self.get_clock().now().to_msg()
        )

        mask.info = copy.deepcopy(
            self.map_msg.info
        )

        mask.info.map_load_time = (
            self.map_msg.info.map_load_time
        )

        mask.data = data

        return mask

    def _publish_clear_mask(self):
        """Publish an unrestricted directional layer after REVOKE."""
        if self.map_msg is None:
            return False

        count = (
            int(self.map_msg.info.width)
            * int(self.map_msg.info.height)
        )

        mask = self._new_mask(
            [0] * count
        )

        if mask is None:
            return False

        self.mask_pub.publish(mask)
        return True

    # --------------------------------------------------
    # PERSON COORDINATES
    # --------------------------------------------------

    def _person_geometry_in_map(
        self,
        zone,
        header,
    ):
        """Transform person position and heading into map coordinates."""
        x = float(zone.center.x)
        y = float(zone.center.y)
        heading = float(zone.heading)

        target = self.map_msg.header.frame_id
        source = str(header.frame_id)

        if source == target:
            return x, y, heading

        transform = self.tf_buffer.lookup_transform(
            target,
            source,
            Time.from_msg(header.stamp),
            timeout=Duration(seconds=0.0),
        )

        theta = _yaw(
            transform.transform.rotation
        )

        cosine = math.cos(theta)
        sine = math.sin(theta)

        translation = (
            transform.transform.translation
        )

        return (
            cosine * x - sine * y
                + float(translation.x),
            sine * x + cosine * y
                + float(translation.y),
            heading + theta,
        )
    
    def _request_direction_heading_in_map(self, request):
        """Resolve the robot heading at the original request time."""
        source = str(request.header.frame_id).strip()
        target = str(self.map_msg.header.frame_id).strip()

        if source != self.directional_robot_frame:
            raise ValueError(
                f"Unexpected directional reference frame: {source}"
            )
        stamp_ns = _nanoseconds(request.header.stamp)

        if stamp_ns <= 0:
            raise ValueError("INVALID_DIRECTION_REFERENCE_STAMP")

        if source == target:
            return 0.0

        transform = self.tf_buffer.lookup_transform(
            target,
            source,
            Time.from_msg(request.header.stamp),
            timeout=Duration(seconds=0.0),
        )

        return _yaw(transform.transform.rotation)

    # --------------------------------------------------
    # LOCAL DIRECTIONAL GEOMETRY
    # --------------------------------------------------

    def _render_directional_mask(
        self,
        zone,
        header,
        direction,
        direction_heading,
    ):
        """Forbid only the unauthorized person-relative neighborhood.

        The personal-clearance ellipse and the directional exclusion
        ellipse are both spatially bounded.

        All cells outside these local shapes remain zero.
        """
        person_x, person_y, heading = (
            self._person_geometry_in_map(
                zone,
                header,
            )
        )

        info = self.map_msg.info

        width = int(info.width)
        height = int(info.height)
        resolution = float(info.resolution)

        origin = info.origin.position

        origin_yaw = _yaw(
            info.origin.orientation
        )

        origin_cosine = math.cos(origin_yaw)
        origin_sine = math.sin(origin_yaw)

        # The social ellipse keeps its original geometric orientation.
        zone_forward_x = math.cos(heading)
        zone_forward_y = math.sin(heading)

        zone_left_x = -zone_forward_y
        zone_left_y = zone_forward_x

        # LEFT and RIGHT use the frozen robot-relative reference.
        direction_forward_x = math.cos(direction_heading)
        direction_forward_y = math.sin(direction_heading)

        direction_left_x = -direction_forward_y
        direction_left_y = direction_forward_x

        if direction not in {"LEFT", "RIGHT"}:
            raise ValueError(
                "Invalid directional constraint"
            )

        # LEFT permits passage on the person's left.
        # RIGHT permits passage on the person's right.
        sign = (
            1.0 if direction == "LEFT" else -1.0
        )

        margin = (
            float(self.params["robot_radius_m"])
            + float(self.params["person_clearance_m"])
        )

        # Person clearance is based on the observed zone.
        personal_lateral_radius = (
            float(zone.radius_lateral)
            + margin
        )

        personal_longitudinal_radius = (
            float(zone.radius_longitudinal)
            + margin
        )

        # The forbidden directional region is strictly local.
        local_lateral_radius = (
            personal_lateral_radius
            + float(
                self.params[
                    "local_lateral_extension_m"
                ]
            )
        )

        local_longitudinal_radius = (
            personal_longitudinal_radius
            + float(
                self.params[
                    "local_longitudinal_extension_m"
                ]
            )
        )

        # All map cells are unrestricted by default.
        data = [0] * (width * height)

        for iy in range(height):
            local_y = (
                (iy + 0.5) * resolution
            )

            for ix in range(width):
                local_x = (
                    (ix + 0.5) * resolution
                )

                world_x = (
                    float(origin.x)
                    + origin_cosine * local_x
                    - origin_sine * local_y
                )

                world_y = (
                    float(origin.y)
                    + origin_sine * local_x
                    + origin_cosine * local_y
                )

                dx = world_x - person_x
                dy = world_y - person_y

                # Person-relative coordinates.
                # Coordinates for the original personal-clearance ellipse.
                person_longitudinal = (
                    dx * zone_forward_x
                    + dy * zone_forward_y
                )

                person_lateral = (
                    dx * zone_left_x
                    + dy * zone_left_y
                )

                # Coordinates for the robot-relative directional restriction.
                direction_longitudinal = (
                    dx * direction_forward_x
                    + dy * direction_forward_y
                )

                direction_lateral = (
                    dx * direction_left_x
                    + dy * direction_left_y
                )

                # Independent personal-clearance ellipse.
                inside_person_clearance = (
                    (
                        person_longitudinal
                        / personal_longitudinal_radius
                    ) ** 2
                    +
                    (
                        person_lateral
                        / personal_lateral_radius
                    ) ** 2
                    <= 1.0
                )

                inside_local_region = (
                    (
                        direction_longitudinal
                        / local_longitudinal_radius
                    ) ** 2
                    +
                    (
                        direction_lateral
                        / local_lateral_radius
                    ) ** 2
                    <= 1.0
                )

                # Forbid the side opposite to the confirmed robot-relative direction.
                forbidden_side = (
                    inside_local_region
                    and sign * direction_lateral < 0.0
                )

                if (
                    inside_person_clearance
                    or forbidden_side
                ):
                    data[iy * width + ix] = 100

        return self._new_mask(data)

    # --------------------------------------------------
    # PERIODIC SUPERVISION
    # --------------------------------------------------

    def _tick(self):
        """
        Install a directional mask once per exact D request.

        After publication, preserve the same mask geometry.
        Observation aging cannot restart installation or erase
        the costmap confirmations.

        Request expiration and explicit revocation remain
        independent of perception freshness.
        """
        self._initialize_costmap_observation()

        now = time.monotonic()
        request = self.active_request

        # No active authorization.
        if request is None:
            if (
                self.state == "REVOKED"
                and self._last_revoked_request is not None
                and now - self._last_revoked_at < 5.0
            ):
                self._publish_status(
                    self._last_revoked_request
                )
            return

        # Explicit revocation. Keep publishing the neutral
        # mask while waiting for costmap confirmation.
        if self.state == "REVOKING":
            if (
                now - getattr(
                    self,
                    "_last_clear_publication_at",
                    0.0,
                ) >= 1.0
            ):
                self._last_clear_publication_at = now
                self._publish_clear_mask()

            self._publish_status(request)
            return

        # A failed INSTALL must not silently recover under
        # the same request_id. Recovery belongs to the
        # manager: REVOKE, then a new INSTALL.
        if self.state == "FAILED":
            self._publish_status(request)
            return

        # The INSTALL authorization has a finite lease.
        # If it expires after a mask was published,
        # invalidate authorization without silently clearing
        # the directional constraint during navigation.
        if now >= self.request_expires_at:
            if self.last_mask_published_at > 0.0:
                self.state = "FAILED"
                self.reason = "DIRECTIONAL_REQUEST_EXPIRED"
                self._verified_directional_request = None

                self._costmap_confirmed = {
                    "global": False,
                    "local": False,
                }

                self._publish_status(request)
            else:
                self._fail("DIRECTIONAL_REQUEST_EXPIRED")

            return

        # IMPORTANT:
        # The first grid has already been published.
        # Do not reconstruct it from changing observations.
        #
        # The two costmap callbacks will independently
        # confirm the published cells.
        if self.state == "PUBLISHED" and self.mask_id:
            self._publish_status(request)
            return

        # First installation requires a valid map.
        if self.map_msg is None:
            self.state = "INSTALLING"
            self.reason = "WAITING_FOR_MAP"
            self._publish_status(request)
            return

        # First installation requires qualified OBSERVED

        selected = self._select_directional_geometry(request)

        if selected is None or selected[2] != "OBSERVED":
            selected = self._select_current_zone_for_install(request)

        if (
            selected is None
            or selected[2] != "OBSERVED"
        ):
            self.state = "INSTALLING"
            self.reason = (
                "WAITING_FOR_CURRENT_QUALIFIED_GEOMETRY"
            )
            self._publish_status(request)
            return

        zone, header, _ = selected

        try:
            direction = str(
                request.directional_constraint
            ).strip().upper()

            if direction not in {"LEFT", "RIGHT"}:
                raise ValueError(
                    "INVALID_DIRECTIONAL_CONSTRAINT"
                )

            # Freeze the robot's directional reference
            # for the exact INSTALL transaction.
            if self.direction_reference_heading_map is None:
                self.direction_reference_heading_map = (
                    self._request_direction_heading_in_map(
                        request
                    )
                )

            mask = self._render_directional_mask(
                zone,
                header,
                direction,
                self.direction_reference_heading_map,
            )

            if mask is None:
                raise ValueError(
                    "DIRECTIONAL_MASK_GENERATION_RETURNED_NONE"
                )

            digest = hashlib.blake2b(
                bytes(mask.data),
                digest_size=16,
            ).hexdigest()

        except (
            TransformException,
            ValueError,
            TypeError,
            OverflowError,
        ) as exc:
            self._fail(
                f"MASK_GENERATION_FAILED: {exc}"
            )
            return

        mask_id = f"{request.request_id}:{digest}"

        # Prepare fresh costmap evidence for exactly
        # these mask cells, only once.
        self._prepare_directional_mask_probes(mask)

        try:
            self.mask_pub.publish(mask)

        except Exception as exc:
            self._fail(
                f"DIRECTIONAL_MASK_PUBLICATION_FAILED: {exc}"
            )
            return

        self._observed_mask_id = mask_id
        self.mask_id = mask_id
        self.last_mask_digest = digest
        self.last_mask_published_at = now

        self._last_geometry_evidence = "OBSERVED"

        self.state = "PUBLISHED"
        self.reason = "WAITING_FOR_COSTMAP_OBSERVATION"

        self._publish_status(request)

def main(args=None):
    rclpy.init(args=args)

    node = DirectionalKeepoutMaskNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
