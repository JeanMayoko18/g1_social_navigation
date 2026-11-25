
"""
Generate deterministic social-navigation events from observed social zones.

Responsibilities:
    - Detect confirmed social intrusions.
    - Maintain unresolved social episodes.
    - Detect loss and reacquisition of observations.
    - Confirm geometric clearance with temporal hysteresis.

This node reports factual events only.

Navigation decisions such as WAIT, REPLAN and RESUME
belong to CollaborationManager.

Observation loss is never interpreted as clearance.
"""

import math
import time
import uuid
import json
from typing import Optional, Tuple
from dataclasses import asdict
from copy import deepcopy

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)

from std_msgs.msg import String, Header
from g1_social_msgs.msg import (
    NavigationState,
    SocialNavigationDecision,
    SocialZone,
    SocialZoneArray,
    TrackedPersonArray,
    SocialParticipant,
)
from g1_social_perception.social_context_memory import (
    SocialContextMemory,
)

from g1_social_perception.social_source_buffer import (
    SocialSourceBuffer,
)
from g1_social_perception.individual_social_evaluator import (
    IndividualSocialEvaluator,
    ObservedPerson,
)

from g1_social_perception.social_situation_evaluator import (
    SocialSituationEvaluator,
)
from rclpy.time import Time
from rclpy.duration import Duration

from tf2_ros import (
    Buffer,
    TransformListener,
    TransformException,
)
from g1_qos_manager.diagnostics import log_qos_resolution
from g1_qos_manager.profiles import reliable_qos
from g1_qos_manager.resolver import resolve_subscription_qos

from g1_social_common.social_navigation_events import (
    SOCIAL_EVENT_INTRUSION_CONFIRMED,
    SOCIAL_EVENT_CLEAR,
    SOCIAL_EVENT_OBSERVATION_LOST,
    SOCIAL_EVENT_OBSERVATION_REACQUIRED,
)

from g1_social_common.social_episode_states import (
    EPISODE_IDLE,
    EPISODE_CONFLICT,
    EPISODE_OBSERVATION_LOST,
    EPISODE_CLEARED,
)


# ================================================================
# CONSTANTS
# ================================================================

ZONE_NONE = "NONE"
TRACKING_NONE = "NONE"


# ================================================================
# NODE
# ================================================================

class SocialNavigationDecisionNode(Node):

    def __init__(self):

        super().__init__("social_navigation_decision")

        # ========================================================
        # PARAMETERS
        # ========================================================

        self.declare_parameter(
            "input_topic",
            "/social_perception/social_zones",
        )

        self.declare_parameter(
            "output_topic",
            "/social_navigation/decision",
        )

        self.declare_parameter(
            "navigation_state_topic",
            "/social_navigation/navigation_state",
        )

        self.declare_parameter(
            "episode_state_topic",
            "/social_navigation/episode_state",
        )

        self.declare_parameter(
            "expected_frame",
            "base_footprint",
        )

        self.declare_parameter(
            "intrusion_margin_m",
            0.05,
        )

        self.declare_parameter(
            "release_margin_m",
            0.15,
        )

        self.declare_parameter(
            "minimum_radius_m",
            0.05,
        )

        self.declare_parameter(
            "conflict_confirmation_sec",
            0.20,
        )

        self.declare_parameter(
            "clear_confirmation_sec",
            1.00,
        )

        self.declare_parameter(
            "intrusion_distance_m",
            1.50,
        )

        self.declare_parameter(
            "social_observation_timeout_sec",
            1.5,
        )
        # Bound retries to the expected TF history window.
        self.declare_parameter(
            "social_geometry_tf_retry_max_age_sec",
            5.0,
        )
        # Zero disables the corresponding uncertainty assumption.
        self.declare_parameter(
            "memory_assumed_max_person_speed_mps",
            0.0,
        )

        self.declare_parameter(
            "memory_initial_position_uncertainty_m",
            0.0,
        )

        self.declare_parameter(
            "memory_geometry_max_age_sec",
            2.0,
        )


        # ========================================================
        # PARAMETER RESOLUTION
        # ========================================================

        self.input_topic = self.get_parameter(
            "input_topic"
        ).value

        self.output_topic = self.get_parameter(
            "output_topic"
        ).value

        self.navigation_state_topic = self.get_parameter(
            "navigation_state_topic"
        ).value

        self.episode_state_topic = self.get_parameter(
            "episode_state_topic"
        ).value

        self.expected_frame = self.get_parameter(
            "expected_frame"
        ).value

        self.intrusion_margin_m = float(
            self.get_parameter("intrusion_margin_m").value
        )

        self.release_margin_m = float(
            self.get_parameter("release_margin_m").value
        )

        self.minimum_radius_m = float(
            self.get_parameter("minimum_radius_m").value
        )

        self.conflict_confirmation_sec = float(
            self.get_parameter("conflict_confirmation_sec").value
        )

        self.clear_confirmation_sec = float(
            self.get_parameter("clear_confirmation_sec").value
        )

        self.intrusion_distance_m = float(
            self.get_parameter("intrusion_distance_m").value
        )

        self.social_observation_timeout_sec = float(
            self.get_parameter(
                "social_observation_timeout_sec"
            ).value
        )

        self.social_geometry_tf_retry_max_age_sec = float(
            self.get_parameter(
                "social_geometry_tf_retry_max_age_sec"
            ).value
        )
        self.memory_assumed_max_person_speed_mps = float(
            self.get_parameter(
                "memory_assumed_max_person_speed_mps"
            ).value
        )

        self.memory_initial_position_uncertainty_m = float(
            self.get_parameter(
                "memory_initial_position_uncertainty_m"
            ).value
        )

        self.memory_geometry_max_age_sec = float(
            self.get_parameter(
                "memory_geometry_max_age_sec"
            ).value
        )

        # ========================================================
        # PARAMETER VALIDATION
        # ========================================================

        if (
            not math.isfinite(self.intrusion_distance_m)
            or self.intrusion_distance_m <= 0.0
        ):
            raise ValueError(
                "intrusion_distance_m must be finite and positive."
            )

        if (
            not math.isfinite(self.social_geometry_tf_retry_max_age_sec)
            or self.social_geometry_tf_retry_max_age_sec <= 0.0
        ):
            raise ValueError(
                "social_geometry_tf_retry_max_age_sec must be positive"
            )
        if (
            not math.isfinite(
                self.social_observation_timeout_sec
            )
            or self.social_observation_timeout_sec <= 0.0
        ):
            raise ValueError(
                "social_observation_timeout_sec must be "
                "finite and positive."
            )
        

        if (
            not all(
                math.isfinite(value)
                for value in (
                    self.memory_assumed_max_person_speed_mps,
                    self.memory_initial_position_uncertainty_m,
                    self.memory_geometry_max_age_sec,
                )
            )
            or self.memory_assumed_max_person_speed_mps < 0.0
            or self.memory_initial_position_uncertainty_m < 0.0
            or self.memory_geometry_max_age_sec <= 0.0
        ):
            raise ValueError(
                "INVALID_SOCIAL_MEMORY_UNCERTAINTY_PARAMETERS"
            )

        self.minimum_radius_m = max(
            0.001,
            self.minimum_radius_m,
        )

        self.entry_distance = self.intrusion_distance_m

        self.release_distance = (
            self.entry_distance
            + self.release_margin_m
        )

        # ========================================================
        # RUNTIME STATE
        # ========================================================

        self.current_person_id = ""

        # State refresh is bounded to 2 Hz, except for transitions.
        self.state_refresh_period_sec = 0.50
        self.last_state_publication_at = None
        self.last_published_state_event = ""

        # Stable identity for one continuous social episode.
        self.social_episode_id = ""

        # Retain the participants needed to reconfirm clearance.
        self.cleared_person_ids = set()
        self.cleared_person_id = ""

        self.social_conflict_active = False

        self.episode_state = EPISODE_IDLE

        self.conflict_candidate_since = None
        self.clear_candidate_since = None

        # Monotonic timestamp of the first loss in this episode.
        self.observation_lost_since = None
        
        self.episode_retirement_notified = False

        self.retired_episode_id = ""
        self.retired_new_person_id = ""
        self.retired_new_person_since = None
        self.retired_new_person_last_seen = None
        self.retired_new_person_samples = 0
        self.unresolved_retired_episodes = {}

        # Retire the producer episode after the manager's 15 s timeout.
        # The extra margin lets CollaborationManager handle STOP first.
        self.lost_episode_retirement_sec = 20.0

        self.last_social_observation_at = None
        self.last_social_stamp_ns = None

        self.conflict_person_ids = set()
        self.person_last_seen = {}

        self.latest_navigation_state = None

        # Decision IDs remain unique across node restarts.

        self.decision_session_id = uuid.uuid4().hex[:12]
        self.decision_sequence = 0

        self.map_frame = "map"


        self._last_social_geometry_tf_retry_at = 0.0

        self.tf_buffer = Buffer()

        self.tf_listener = TransformListener(
            self.tf_buffer,
            self,
            spin_thread=False,
        )

        # ========================================================
        # DIAGNOSTICS
        # ========================================================

        self.arrays_received = 0
        self.zones_evaluated = 0
        self.conflicts_detected = 0
        self.clear_events = 0

        self.invalid_frame = 0
        self.invalid_zone = 0

        self.invalid_observation_stamps = 0
        self.rejected_observation_arrays = 0
        self.empty_observation_arrays = 0

        # ========================================================
        # QOS
        # ========================================================

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

        output_qos = reliable_qos(depth=10)

        # ========================================================
        # ROS SUBSCRIPTIONS
        # ========================================================

        self.navigation_state_subscription = (
            self.create_subscription(
                NavigationState,
                self.navigation_state_topic,
                self._navigation_state_callback,
                reliable_qos(depth=10),
            )
        )

        self.subscription = self.create_subscription(
            SocialZoneArray,
            self.input_topic,
            self.social_zones_callback,
            input_qos,
        )

        # ========================================================
        # ROS PUBLISHERS
        # ========================================================

        self.publisher = self.create_publisher(
            SocialNavigationDecision,
            self.output_topic,
            output_qos,
        )

        self.episode_state_publisher = (
            self.create_publisher(
                String,
                self.episode_state_topic,
                QoSProfile(
                    history=HistoryPolicy.KEEP_LAST,
                    depth=1,
                    reliability=ReliabilityPolicy.RELIABLE,
                    durability=DurabilityPolicy.TRANSIENT_LOCAL,
                ),
            )
        )

        # ========================================================
        # TIMERS
        # ========================================================

        self.diagnostics_timer = self.create_timer(
            5.0,
            self.log_diagnostics,
        )

        self.context_timer = self.create_timer(
            0.20,
            self.context_timer_callback,
        )

        # Publish initial episode state.

        self.episode_state_publisher.publish(
            String(data=self.episode_state)
        )

        # Exact retirement notifications.
        # Separate from the canonical episode-state topic.
        self.episode_retirement_publisher = self.create_publisher(
            String,
            "/social_navigation/episode_retirement",
            QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            ),
        )

        self.observation_health_publisher = self.create_publisher(
            String,
            "/social_navigation/observation_health",
            QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=10,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
            ),
        )

        # ========================================================
        # INITIALIZATION LOG
        # ========================================================

        self.get_logger().info("=" * 84)
        self.get_logger().info(
            "SOCIAL NAVIGATION DECISION - INITIALIZATION"
        )
        self.get_logger().info("=" * 84)

        self.get_logger().info(
            f"Input topic: {self.input_topic}"
        )

        self.get_logger().info(
            f"Output topic: {self.output_topic}"
        )

        self.get_logger().info(
            f"Expected frame: {self.expected_frame}"
        )

        self.get_logger().info(
            f"Entry distance: {self.entry_distance:.2f} m"
        )

        self.get_logger().info(
            f"Release distance: {self.release_distance:.2f} m"
        )

        
        self.social_memory = SocialContextMemory(
            expected_frame=self.expected_frame,
            freshness_sec=self.social_observation_timeout_sec,
        )

        self.declare_parameter(
            "tracked_persons_topic",
            "/social_perception/tracked_persons",
        )

        tracked_topic = str(
            self.get_parameter("tracked_persons_topic").value
        )

        tracked_qos = resolve_subscription_qos(
            self,
            tracked_topic,
            fallback_depth=10,
            prefer_sensor_data=False,
        )

        self.memory_subscription = self.create_subscription(
            TrackedPersonArray,
            tracked_topic,
            self._tracked_persons_memory_callback,
            tracked_qos,
        )

        self._initialize_social_reassessment()

        self._initialize_social_evaluator()

        self.individual_social_evaluator = IndividualSocialEvaluator(
            entry_distance=self.entry_distance,
            release_distance=self.release_distance,
            entry_duration=self.conflict_confirmation_sec,
            clear_duration=self.clear_confirmation_sec,
            missing_grace=self.effective_observation_loss_grace_sec,
            maximum_gap=self.social_observation_timeout_sec,
            reacquisition_duration=0.30,
            persistence_duration=2.0,
            retirement_duration=self.lost_episode_retirement_sec,
        )

        self._individual_last_source = None
        self._individual_last_publication = 0.0

        self.individual_situation_publisher = self.create_publisher(
            String,
            "/social_perception/individual_social_situation",
            reliable_qos(depth=10),
        )

        self.latest_memory_situation = None
        self._last_memory_situation_revision = None
        self._last_memory_situation_publication = 0.0

        self.memory_situation_publisher = self.create_publisher(
            String,
            "/social_perception/social_situation",
            reliable_qos(depth=10),
        )

        # Publish qualified geometry without authorizing its use for motion.
        self.social_geometry_publisher = self.create_publisher(
            String,
            "/social_perception/social_geometry_status",
            reliable_qos(depth=10),
        )

        # Buffer individual reports by exact source identity.
        # This is transport synchronization, not social observation memory.
        self._individual_zone_reports = {}
        self._individual_zone_reports_max_pending = 32

        self.individual_zone_results_subscription = (
            self.create_subscription(
                String,
                "/social_perception/individual_zone_results",
                self._individual_zone_results_callback,
                reliable_qos(depth=10),
            )
        )

        self._last_social_geometry_publication = 0.0

        self.memory_evaluation_timer = self.create_timer(
            0.10,
            self._consume_ready_source_pairs,
        )

        # Adapter state. Episode identity belongs to the evaluator.
        self._memory_message = None
        self._memory_current = {}
        self._memory_zones = {}
        self._memory_participants = set()
        self._memory_primary = None
        self._memory_event_key = None
        self._initialize_social_source_buffer()

    # ============================================================
    # NAVIGATION CONTEXT
    # ============================================================

    def _navigation_state_callback(
        self,
        msg: NavigationState,
    ) -> None:

        self.latest_navigation_state = msg

    # ============================================================
    # OBSERVATION VALIDATION
    # ============================================================
    
    def _memory_clear_is_current(self):
        """Require complete current evidence for the historical resolution."""
        situation = self.latest_memory_situation
        current = self._memory_current

        return bool(
            self._paired_evidence_is_fresh()
            and self._source_coverage_complete
            and situation is not None
            and not situation.unresolved
            and situation.clearance_status == "CONFIRMED"
            and situation.participants
            and all(
                key in current
                for key in situation.participants
            )
            and all(
                math.hypot(*person.latest.position[:2])
                > self.release_distance
                for person in current.values()
            )
            and self._memory_message is not None
        )
    
    def _initialize_social_source_buffer(self):
        """Initialize bounded source pairing and evidence bookkeeping."""
        self.declare_parameter(
            "source_pairing_timeout_sec",
            0.20,
        )
        self.declare_parameter(
            "source_pairing_max_pending",
            32,
        )

        self.social_source_buffer = SocialSourceBuffer(
            expected_frame=self.expected_frame,
            freshness_sec=self.social_observation_timeout_sec,
            pairing_timeout_sec=float(
                self.get_parameter(
                    "source_pairing_timeout_sec"
                ).value
            ),
            max_pending=self.get_parameter(
                "source_pairing_max_pending"
            ).value,
        )

        self._source_pair = None
        self._source_pair_status = "UNAVAILABLE"
        self._source_missing_since = None
        self._source_coverage_complete = False

        # After overflow, queued older evidence must not restore
        # certainty. Recovery requires a newly received complete pair.
        self._source_recovery_after_monotonic = None

        self._last_clear_source = None
        self._source_pair_counts = {}

    def _project_zone_to_map(self, zone, source_header):
        """
        Transform a validated zone using its original observation time.

        Never use the current transform to reinterpret old coordinates.
        """
        source_frame = str(source_header.frame_id).strip()

        x = float(zone.center.x)
        y = float(zone.center.y)
        z = float(zone.center.z)
        heading = float(zone.heading)

        if source_frame == self.map_frame:
            return (x, y, z), heading

        transform = self.tf_buffer.lookup_transform(
            self.map_frame,
            source_frame,
            Time.from_msg(zone.observation_stamp),
            timeout=Duration(seconds=0.0),
        )

        q = transform.transform.rotation

        norm = math.sqrt(
            q.x * q.x
            + q.y * q.y
            + q.z * q.z
            + q.w * q.w
        )

        if not math.isfinite(norm) or norm < 1e-8:
            raise ValueError("INVALID_TF_ROTATION")

        qx = q.x / norm
        qy = q.y / norm
        qz = q.z / norm
        qw = q.w / norm

        theta = math.atan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy * qy + qz * qz),
        )

        c = math.cos(theta)
        s = math.sin(theta)
        t = transform.transform.translation

        # Planar social-navigation geometry.
        map_position = (
            c * x - s * y + float(t.x),
            s * x + c * y + float(t.y),
            z + float(t.z),
        )

        map_heading = math.atan2(
            math.sin(heading + theta),
            math.cos(heading + theta),
        )

        if not all(
            math.isfinite(value)
            for value in (*map_position, map_heading)
        ):
            raise ValueError("INVALID_MAP_GEOMETRY")

        return map_position, map_heading

    def _retry_unprojected_social_zones(self):
        """
        Retry map projection for validated historical social zones.

        Preserve the original observation timestamp.
        Never turn a historical zone into a fresh observation.
        """
        now = time.monotonic()

        # Limit TF requests to 2 Hz.
        if (
            now - self._last_social_geometry_tf_retry_at
            < 0.50
        ):
            return

        self._last_social_geometry_tf_retry_at = now

        now_ros_ns = self.get_clock().now().nanoseconds

        snapshots = self.social_memory.snapshot(
            now_ros_ns=now_ros_ns,
            now_monotonic=now,
        )

        for person in snapshots:
            source = person.last_social_zone
            projected = person.last_social_zone_map

            if source is None:
                continue

            # A projection is complete only if it belongs
            # to the current stored source observation.
            if (
                projected is not None
                and projected.stamp_ns == source.stamp_ns
                and projected.frame_id == self.map_frame
            ):
                continue

            age = (
                now_ros_ns - source.stamp_ns
            ) / 1e9

            if not (
                0.0 <= age
                <= self.social_geometry_tf_retry_max_age_sec
            ):
                continue

            zone = SocialZone()

            zone.center.x = float(source.position[0])
            zone.center.y = float(source.position[1])
            zone.center.z = float(source.position[2])
            zone.heading = float(source.heading)

            zone.observation_stamp = Time(
                nanoseconds=int(source.stamp_ns)
            ).to_msg()

            # The source frame and original timestamp must
            # be used together for the historical transform.
            header = Header()
            header.frame_id = source.frame_id
            header.stamp = zone.observation_stamp

            try:
                position, heading = self._project_zone_to_map(
                    zone,
                    header,
                )

            except (
                TransformException,
                ValueError,
                TypeError,
                OverflowError,
            ):
                continue

            stored = self.social_memory.remember_projected_zone(
                session_id=person.session_id,
                person_id=person.person_id,
                identity_generation=person.identity_generation,
                source_stamp_ns=source.stamp_ns,
                target_frame=self.map_frame,
                position=position,
                heading=heading,
            )

            if stored:
                self.get_logger().info(
                    "[SOCIAL MEMORY] Historical TF projection recovered | "
                    f"person={person.person_id} | "
                    f"generation={person.identity_generation} | "
                    f"source_stamp_ns={source.stamp_ns} | "
                    f"age={age:.2f}s"
                )

    def _individual_zone_results_callback(self, msg):
        """
        Buffer individual generation reports by their exact source key.

        Reports remain advisory and never modify authoritative
        social observations or episode state.
        """
        try:
            data = json.loads(msg.data)

            if (
                not isinstance(data, dict)
                or type(data.get("version")) is not int
                or data["version"] != 1
                or data.get("producer") != "social_zone_generator"
                or not isinstance(data.get("persons"), list)
            ):
                return

            key = (
                str(data["tracking_session_id"]).strip(),
                int(data["source_stamp_ns"]),
                str(data["source_frame"]).strip(),
            )

            if (
                not key[0]
                or key[1] <= 0
                or key[2] != self.expected_frame
            ):
                return

            now = time.monotonic()

            # Drop reports that have exceeded the pairing horizon.
            max_age = max(
                1.0,
                2.0 * self.social_observation_timeout_sec,
            )

            for stored_key, (_, received_at) in list(
                self._individual_zone_reports.items()
            ):
                if now - received_at > max_age:
                    del self._individual_zone_reports[stored_key]

            # Keep the first accepted report for a source frame.
            # Duplicates cannot overwrite its contents.
            if key in self._individual_zone_reports:
                return

            self._individual_zone_reports[key] = (
                data,
                now,
            )

            # Bound memory usage independently of source frequency.
            while (
                len(self._individual_zone_reports)
                > self._individual_zone_reports_max_pending
            ):
                oldest_key = min(
                    self._individual_zone_reports,
                    key=lambda item: (
                        self._individual_zone_reports[item][1]
                    ),
                )

                del self._individual_zone_reports[oldest_key]

        except (
            ValueError,
            TypeError,
            KeyError,
            AttributeError,
            OverflowError,
        ):
            return

    def _validated_individual_zone_results(self, pair):
        """
        Match a generation report to one validated source pair.

        Return individual results only when the report is
        complete and consistent with the source persons and zones.
        """
        if pair is None:
            return {}

        stored = self._individual_zone_reports.get(pair.key)

        if stored is None:
            return {}

        report, received_at = stored

        # Never associate an expired auxiliary report with current evidence.
        if (
            time.monotonic() - received_at
            > max(1.0, 2.0 * self.social_observation_timeout_sec)
        ):
            return {}

        try:
            report_key = (
                str(report["tracking_session_id"]).strip(),
                int(report["source_stamp_ns"]),
                str(report["source_frame"]).strip(),
            )

            if report_key != pair.key:
                return {}

            source_persons = {
                (
                    str(person.tracking_session_id).strip(),
                    str(person.id).strip(),
                    int(person.identity_generation),
                ): person
                for person in pair.persons.persons
            }

            source_zones = {
                (
                    str(zone.tracking_session_id).strip(),
                    str(zone.person_id).strip(),
                    int(zone.identity_generation),
                )
                for zone in pair.zones.zones
            }

            rows = report["persons"]

            if len(rows) != len(source_persons):
                return {}

            matched = {}

            for row in rows:
                if not isinstance(row, dict):
                    return {}

                identity = (
                    str(row["tracking_session_id"]).strip(),
                    str(row["person_id"]).strip(),
                    int(row["identity_generation"]),
                )

                if (
                    identity in matched
                    or identity not in source_persons
                ):
                    return {}

                person = source_persons[identity]

                if (
                    str(row["tracking_state"])
                    != str(person.tracking_state)
                    or type(row["observed_in_current_array"]) is not bool
                    or row["observed_in_current_array"] != bool(
                        person.observed_in_current_array
                    )
                    or int(row["observation_stamp_ns"])
                    != SocialContextMemory._stamp_ns(
                        person.last_observation_stamp
                    )
                ):
                    return {}

                status = str(row["status"])
                reason = str(row["reason"])

                if not reason.strip():
                    return {}

                if status == "GENERATED" and reason != "VALID_CURRENT_OBSERVATION":
                    return {}

                if status == "FILTERED" and reason != "OUTSIDE_CONFIGURED_FORWARD_SECTOR":
                    return {}

                if status == "UNAVAILABLE" and reason not in {
                    "NO_CURRENT_OBSERVATION",
                    "TRACK_NOT_CONFIRMED",
                }:
                    return {}

                direction_basis = str(
                    row.get("direction_basis", "UNKNOWN")
                )

                has_zone = identity in source_zones

                if status == "GENERATED":
                    if not has_zone or not (
                        person.observed_in_current_array
                        and person.tracking_state == "TRACKED"
                    ):
                        return {}

                elif status == "FILTERED":
                    if has_zone or not (
                        person.observed_in_current_array
                        and person.tracking_state == "TRACKED"
                    ):
                        return {}

                elif status == "UNAVAILABLE":
                    if has_zone or (
                        person.observed_in_current_array
                        and person.tracking_state == "TRACKED"
                    ):
                        return {}

                else:
                    return {}

                matched[identity] = {
                    "status": status,
                    "reason": reason,
                    "direction_basis": direction_basis,
                }

            return matched

        except (
            ValueError,
            TypeError,
            KeyError,
            AttributeError,
            OverflowError,
        ):
            return {}


    def _qualify_historical_geometry(
        self,
        *,
        source_stamp_ns,
        now_ros_ns,
        projection_available,
    ):
        """
        Qualify one historical map-frame geometry estimate.
        
        u(t)=u_0+v_{max}t
        
        u_0: initiale uncertainty of the position, v_{max})
        
        The uncertainty is conditional on an explicitly configured
        maximum person speed and initial localization error.

        This method does not certify safety or authorize motion.
        """
        if (
            not projection_available
            or source_stamp_ns <= 0
            or now_ros_ns < source_stamp_ns
        ):
            return {
                "memory_age_sec": None,
                "memory_within_age_horizon": False,
                "position_uncertainty_m": None,
                "uncertainty_status": "UNAVAILABLE",
                "memory_motion_eligible": False,
            }

        age_sec = (
            now_ros_ns - source_stamp_ns
        ) / 1e9

        within_horizon = bool(
            age_sec <= self.memory_geometry_max_age_sec
        )

        speed_bound = (
            self.memory_assumed_max_person_speed_mps
        )

        initial_uncertainty = (
            self.memory_initial_position_uncertainty_m
        )

        assumptions_configured = bool(
            speed_bound > 0.0
            and initial_uncertainty > 0.0
        )

        if not assumptions_configured:
            return {
                "memory_age_sec": age_sec,
                "memory_within_age_horizon": within_horizon,
                "position_uncertainty_m": None,
                "uncertainty_status": "UNBOUNDED",
                "memory_motion_eligible": False,
            }

        uncertainty_m = (
            initial_uncertainty
            + speed_bound * age_sec
        )

        if not math.isfinite(uncertainty_m):
            return {
                "memory_age_sec": age_sec,
                "memory_within_age_horizon": False,
                "position_uncertainty_m": None,
                "uncertainty_status": "UNAVAILABLE",
                "memory_motion_eligible": False,
            }

        return {
            "memory_age_sec": age_sec,
            "memory_within_age_horizon": within_horizon,
            "position_uncertainty_m": uncertainty_m,
            "uncertainty_status": (
                "ASSUMPTION_BOUND_WITHIN_HORIZON"
                if within_horizon
                else "AGE_HORIZON_EXCEEDED"
            ),
            "memory_motion_eligible": False,
        }

    def _build_directional_geometry_candidate(
        self,
        *,
        person,
        status,
        source,
        projected,
        projection_valid,
        historical_quality,
        observed,
    ):
        """
        Build a per-person directional geometry candidate.

        Geometry availability, directional reference and human
        authorization are independent requirements.

        This method never authorizes mask installation.
        """
        identity = {
            "tracking_session_id": person.session_id,
            "person_id": person.person_id,
            "identity_generation": int(
                person.identity_generation
            ),
        }

        candidate = {
            **identity,
            "geometry_status": status,
            "geometry_available": False,
            "geometry_source_stamp_ns": (
                int(source.stamp_ns)
                if source is not None
                else None
            ),
            "geometry_frame": "",
            "geometry": None,
            "observed_now": bool(observed),
            "position_uncertainty_m": (
                historical_quality.get(
                    "position_uncertainty_m"
                )
            ),
            "uncertainty_status": (
                historical_quality.get(
                    "uncertainty_status",
                    "UNAVAILABLE",
                )
            ),
            "direction_reference": "UNRESOLVED",
            "direction_reference_valid": False,
            "human_authorization_required": True,
            "mask_eligible": False,
            "reason": "GEOMETRY_UNAVAILABLE",
        }

        if (
            not projection_valid
            or source is None
            or projected is None
        ):
            return candidate

        if (
            projected.frame_id != self.map_frame
            or projected.stamp_ns != source.stamp_ns
        ):
            candidate["reason"] = "SOURCE_PROJECTION_MISMATCH"
            return candidate

        values = (
            *projected.position,
            projected.heading,
            projected.radius_longitudinal,
            projected.radius_lateral,
        )

        if (
            not all(math.isfinite(float(v)) for v in values)
            or projected.radius_longitudinal <= 0.0
            or projected.radius_lateral <= 0.0
        ):
            candidate["reason"] = "INVALID_MAP_GEOMETRY"
            return candidate

        candidate["geometry_available"] = True
        candidate["geometry_frame"] = self.map_frame

        candidate["geometry"] = {
            "position": [
                float(v)
                for v in projected.position
            ],
            "heading": float(projected.heading),
            "radius_longitudinal": float(
                projected.radius_longitudinal
            ),
            "radius_lateral": float(
                projected.radius_lateral
            ),
            "zone_type": projected.zone_type,
        }

        if projected.zone_type == "DYNAMIC":
            candidate["direction_reference"] = (
                "MOTION_DIRECTION"
            )
        elif projected.zone_type == "STATIC":
            candidate["direction_reference"] = (
                "UNKNOWN_BODY_ORIENTATION"
            )

        # Motion heading does not establish body orientation.
        # The interaction contract must define the reference
        # used to interpret LEFT and RIGHT.
        candidate["direction_reference_valid"] = False

        if status == "OBSERVED":
            candidate["reason"] = (
                "CURRENT_GEOMETRY_DIRECTION_UNRESOLVED"
            )

        elif status == "MEMORY":
            within_horizon = bool(
                historical_quality.get(
                    "memory_within_age_horizon",
                    False,
                )
            )

            uncertainty_status = historical_quality.get(
                "uncertainty_status"
            )

            if not within_horizon:
                candidate["reason"] = (
                    "MEMORY_AGE_HORIZON_EXCEEDED"
                )
            elif uncertainty_status == "UNBOUNDED":
                candidate["reason"] = (
                    "MEMORY_UNCERTAINTY_UNBOUNDED"
                )
            else:
                candidate["reason"] = (
                    "HISTORICAL_GEOMETRY_NOT_AUTHORIZED"
                )

        else:
            candidate["reason"] = (
                "GEOMETRY_PROVENANCE_UNAVAILABLE"
            )

        return candidate

    def _publish_qualified_social_geometry(self):
        """
        Publish the provenance of stored map-frame social geometry.

        Current observations and historical geometry remain distinct.
        This output does not authorize navigation or mask installation.
        """
        now_monotonic = time.monotonic()

        if (
            now_monotonic
            - self._last_social_geometry_publication
            < 0.50
        ):
            return

        now_ros_ns = self.get_clock().now().nanoseconds

        persons = self.social_memory.snapshot(
            now_ros_ns=now_ros_ns,
            now_monotonic=now_monotonic,
        )

        pair = self._source_pair
        matched = self._paired_evidence_is_fresh()

        individual_results = (
            self._validated_individual_zone_results(pair)
            if matched
            else {}
        )

        current_zones = {}

        if matched and pair is not None:
            current_zones = {
                (
                    str(zone.tracking_session_id),
                    str(zone.person_id),
                    int(zone.identity_generation),
                ): zone
                for zone in pair.zones.zones
            }

        results = []

        for person in persons:
            identity = (
                person.session_id,
                person.person_id,
                person.identity_generation,
            )

            generation_result = individual_results.get(identity)

            generation_status = (
                generation_result["status"]
                if generation_result is not None
                else "NOT_AVAILABLE"
            )

            generation_reason = (
                generation_result["reason"]
                if generation_result is not None
                else "NO_MATCHED_INDIVIDUAL_REPORT"
            )

            source = person.last_social_zone
            projected = person.last_social_zone_map

            projection_valid = bool(
                source is not None
                and projected is not None
                and projected.frame_id == self.map_frame
                and projected.stamp_ns == source.stamp_ns
            )

            historical_quality = self._qualify_historical_geometry(
                source_stamp_ns=(
                    source.stamp_ns if source is not None else 0
                ),
                now_ros_ns=now_ros_ns,
                projection_available=projection_valid,
            )

            current_zone = current_zones.get(identity)

            observed = bool(
                matched
                and person.geometry_fresh
                and person.observed_in_current_array
                and person.tracking_state == "TRACKED"
                and source is not None
                and current_zone is not None
                and source.stamp_ns
                == SocialContextMemory._stamp_ns(
                    current_zone.observation_stamp
                )
                and source.stamp_ns == person.latest.stamp_ns
            )

            if not projection_valid:
                status = "UNAVAILABLE"
            elif observed:
                status = "OBSERVED"
            else:
                status = "MEMORY"

            age_sec = (
                max(
                    0.0,
                    (now_ros_ns - source.stamp_ns) / 1e9,
                )
                if source is not None
                else None
            )

            geometry = None

            if projection_valid:
                geometry = {
                    "frame_id": projected.frame_id,
                    "position": list(projected.position),
                    "heading": projected.heading,
                    "radius_longitudinal": (
                        projected.radius_longitudinal
                    ),
                    "radius_lateral": (
                        projected.radius_lateral
                    ),
                    "zone_type": projected.zone_type,
                    "source_stamp_ns": projected.stamp_ns,
                }

            direction_basis = "UNAVAILABLE"

            if source is not None:
                direction_basis = (
                    "MOTION_HEADING"
                    if source.zone_type == "DYNAMIC"
                    else "UNRESOLVED_STATIC_ORIENTATION"
                )
            
            # Build a candidate without granting directional authorization.
            directional_candidate = (
                self._build_directional_geometry_candidate(
                    person=person,
                    status=status,
                    source=source,
                    projected=projected,
                    projection_valid=projection_valid,
                    historical_quality=historical_quality,
                    observed=observed,
                )
            )

            results.append({
                "tracking_session_id": person.session_id,
                "person_id": person.person_id,
                "identity_generation": (
                    person.identity_generation
                ),
                
                "status": status,
                "observed_now": observed,
                "source_age_sec": age_sec,
                "source_frame": (
                    source.frame_id if source else ""
                ),
                
                "map_projection_available": projection_valid,
                "direction_basis": direction_basis,
                "geometry": geometry,
                "mask_eligible": False,
                "historical_quality": historical_quality,
                "directional_candidate": directional_candidate,


                "generation_status": generation_status,
                "generation_reason": generation_reason,
            })

        payload = {
            "version": 1,
            "producer": "social_navigation_decision",
            "evaluation_stamp_ns": now_ros_ns,
            "map_frame": self.map_frame,
            "source_status": (
                "MATCHED"
                if matched
                else self._source_pair_status
            ),
            "authoritative_for_navigation": False,
            "persons": results,
        }

        self.social_geometry_publisher.publish(
            String(
                data=json.dumps(
                    payload,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            )
        )

        self._last_social_geometry_publication = (
            now_monotonic
        )

    def _publish_memory_event(
        self,
        event,
        source=None,
        zone=None,
        force=False,
        explain=False,
    ):
        """Publish a factual event for its actual participant."""

        if event == SOCIAL_EVENT_OBSERVATION_LOST:
            target_person_id = self.current_person_id

        elif event == SOCIAL_EVENT_CLEAR:
            target_person_id = self.cleared_person_id

        else:
            if source is None or zone is None:
                return

            target_person_id = str(zone.person_id).strip()

            if not target_person_id:
                return

        key = (
            self.social_episode_id,
            event,
            target_person_id,
        )

        now = time.monotonic()

        if (
            not force
            and key == self._memory_event_key
            and self.last_state_publication_at is not None
            and (
                now - self.last_state_publication_at
            ) < self.state_refresh_period_sec
        ):
            return

        if event == SOCIAL_EVENT_OBSERVATION_LOST:
            decision = SocialNavigationDecision()

            decision.header.frame_id = self.expected_frame
            decision.header.stamp = (
                self.get_clock().now().to_msg()
            )
            decision.decision_id = self._next_decision_id()
            decision.event = event
            decision.person_id = target_person_id
            decision.social_conflict_active = True
            decision.zone_type = ZONE_NONE
            decision.tracking_state = TRACKING_NONE

            self._attach_navigation_snapshot(decision)

        elif event == SOCIAL_EVENT_CLEAR:
            if source is None:
                return

            decision = self._build_clear_event(
                source,
                target_person_id,
            )
            decision.header.stamp = source.header.stamp

        else:
            distance = math.hypot(
                zone.center.x,
                zone.center.y,
            )
            radius = self._effective_zone_radius(zone)

            decision = self._build_event(
                source,
                zone,
                event,
                distance,
                radius,
                radius / max(distance, 1e-6),
            )

            decision.header.stamp = zone.observation_stamp

        decision.requires_explanation = bool(explain)

        self._attach_episode_participants(decision)
        self.publisher.publish(decision)

        self._memory_event_key = key
        self.last_state_publication_at = now
        self.last_published_state_event = event

    def _validate_social_observation(self, msg):
        """Validate zone-array provenance and geometry without committing it."""
        try:
            session = str(msg.tracking_session_id).strip()

            stamp_ns = SocialContextMemory._stamp_ns(
                msg.header.stamp
            )

            age = (
                self.get_clock().now().nanoseconds
                - stamp_ns
            ) / 1e9

            if (
                not session
                or msg.header.frame_id != self.expected_frame
                or not (
                    -0.10
                    <= age
                    <= self.social_observation_timeout_sec
                )
            ):
                return False

            seen = set()

            for zone in msg.zones:
                person_id = str(zone.person_id).strip()

                if (
                    not person_id
                    or person_id in seen
                    or zone.tracking_session_id != session
                    or SocialContextMemory._stamp_ns(
                        zone.observation_stamp
                    ) != stamp_ns
                    or zone.tracking_state != "TRACKED"
                    or zone.zone_type not in {"STATIC", "DYNAMIC"}
                    or not self._is_valid_zone(zone)
                    or not math.isfinite(float(zone.center.z))
                    or zone.person_speed < 0.0
                ):
                    return False

                seen.add(person_id)

            return True

        except (
            ValueError,
            TypeError,
            AttributeError,
            OverflowError,
        ):
            return False

    def _validate_source_pair(self, pair):
        """
        Validate a synchronized persons/zones observation.

        Return:
            (staged_social_memory, coverage_complete)
            when the evidence is coherent.

            None only for invalid or contradictory evidence.

        Important:
            A valid subset of zones is not an invalid pair.
            Missing zones reduce coverage, never prove clearance.
        """
        try:
            if (
                pair.status != "MATCHED"
                or pair.persons is None
                or pair.zones is None
            ):
                return None

            for msg in (pair.persons, pair.zones):
                key = (
                    str(msg.tracking_session_id).strip(),
                    SocialContextMemory._stamp_ns(
                        msg.header.stamp
                    ),
                    msg.header.frame_id,
                )

                if key != pair.key:
                    return None

            now_monotonic = time.monotonic()
            now_ros_ns = (
                self.get_clock().now().nanoseconds
            )

            if not all(
                received is not None
                and math.isfinite(received)
                and (
                    0.0
                    <= now_monotonic - received
                    <= self.social_observation_timeout_sec
                )
                for received in (
                    pair.persons_received_monotonic,
                    pair.zones_received_monotonic,
                )
            ):
                return None

            if not self._validate_social_observation(
                pair.zones
            ):
                return None

            source_count = int(
                pair.persons.source_detection_count
            )

            if source_count < 0:
                return None

            # The zone producer must describe the same
            # number of tracked-person records that it
            # actually received.
            if (
                int(pair.zones.source_person_count)
                != len(pair.persons.persons)
            ):
                return None

            # Validate and stage social memory atomically.
            # No state is committed before this entire
            # person/zone relationship is accepted.
            candidate_memory = deepcopy(
                self.social_memory
            )

            if not candidate_memory.update(
                msg=pair.persons,
                now_ros_ns=now_ros_ns,
                now_monotonic=(
                    pair.persons_received_monotonic
                ),
            ):
                return None

            persons = {
                str(person.id).strip(): person
                for person in pair.persons.persons
            }

            if len(persons) != len(
                pair.persons.persons
            ):
                return None

            observed_ids = {
                person_id
                for person_id, person in persons.items()
                if person.observed_in_current_array
            }

            # Retained tracks are historical records,
            # not new source observations.
            if len(observed_ids) > source_count:
                return None

            eligible_ids = {
                person_id
                for person_id, person in persons.items()
                if (
                    person.observed_in_current_array
                    and person.tracking_state == "TRACKED"
                )
            }

            zone_ids = set()

            for zone in pair.zones.zones:
                person_id = str(zone.person_id).strip()
                person = persons.get(person_id)

                if (
                    person is None
                    or person_id in zone_ids
                ):
                    return None

                # A zone must correspond to an actual,
                # current, confirmed person observation.
                if person_id not in eligible_ids:
                    return None

                if (
                    person.tracking_session_id
                    != zone.tracking_session_id
                ):
                    return None
                if (
                    int(person.identity_generation)
                    != int(zone.identity_generation)
                ):
                    return None
                if (
                    SocialContextMemory._stamp_ns(
                        person.last_observation_stamp
                    )
                    != SocialContextMemory._stamp_ns(
                        zone.observation_stamp
                    )
                ):
                    return None

                if (
                    person.position != zone.center
                    or person.speed != zone.person_speed
                ):
                    return None

                zone_ids.add(person_id)

            # Zones must be a subset of currently eligible
            # tracked persons. Missing zones are permitted
            # because the generator may have filtered them.
            #
            # They are NOT evidence that a person vanished.
            if not zone_ids.issubset(eligible_ids):
                return None

            all_observed_tracked = all(
                person.observed_in_current_array
                and person.tracking_state == "TRACKED"
                for person in persons.values()
            )

            derived_complete = bool(
                pair.persons.source_observation_complete
                and len(observed_ids) == source_count
                and all_observed_tracked
                and zone_ids == eligible_ids
                and len(zone_ids) == len(persons)
            )

            # The zone generator must not claim complete
            # coverage when the independently checked
            # evidence says otherwise.
            if (
                pair.zones.source_observation_complete
                and not derived_complete
            ):
                return None

            coverage_complete = bool(
                derived_complete
                and pair.zones.source_observation_complete
            )

            if not candidate_memory.remember_validated_zones(
                pair.zones
            ):
                return None
            
            # Project validated zones into map using source-time TF.
            # Missing TF must not invalidate the source observation.
            for zone in pair.zones.zones:
                try:
                    position, heading = self._project_zone_to_map(
                        zone,
                        pair.zones.header,
                    )

                    stored = candidate_memory.remember_projected_zone(
                        session_id=zone.tracking_session_id,
                        person_id=zone.person_id,
                        identity_generation=int(
                            zone.identity_generation
                        ),
                        source_stamp_ns=(
                            SocialContextMemory._stamp_ns(
                                zone.observation_stamp
                            )
                        ),
                        target_frame=self.map_frame,
                        position=position,
                        heading=heading,
                    )

                    if not stored:
                        self.get_logger().warning(
                            "[SOCIAL MEMORY] Projection not stored | "
                            f"person={zone.person_id}"
                        )

                except (
                    TransformException,
                    ValueError,
                    TypeError,
                    OverflowError,
                ) as exc:
                    self.get_logger().warning(
                        "[SOCIAL MEMORY] Map projection unavailable | "
                        f"person={zone.person_id} | "
                        f"reason={exc}",
                        throttle_duration_sec=2.0,
                    )

            return (
                candidate_memory,
                coverage_complete,
            )

        except (
            ValueError,
            TypeError,
            AttributeError,
            OverflowError,
        ):
            return None

    def _evaluate_social_memory(self):
        """Evaluate factual evidence while preserving episode identity."""
        now = time.monotonic()
        ros_now = self.get_clock().now().nanoseconds

        persons = self.social_memory.snapshot(
            now_ros_ns=ros_now,
            now_monotonic=now,
        )

        pair = self._source_pair
        matched = self._paired_evidence_is_fresh()
        status = self._source_pair_status

        self._evaluate_individual_situation(
            pair=pair,
            matched=matched,
            now=now,
            ros_now=ros_now,
        )

        if pair is not None and not matched:
            status = "STALE"
            self._memory_message = None
            self._sr_observation = None

        stamp = (
            pair.key[1]
            if pair is not None
            else self.social_memory.last_array_stamp_ns
        )

        current = {
            (person.session_id, person.person_id): person
            for person in persons
            if (
                matched
                and person.geometry_fresh
                and person.latest.stamp_ns == stamp
                and person.session_id == self.social_memory.session_id
            )
        }

        zones = (
            {
                (zone.tracking_session_id, zone.person_id): zone
                for zone in pair.zones.zones
            }
            if matched
            else {}
        )

        situation = self.social_situation_evaluator.evaluate(
            persons=persons,
            source_stamp_ns=stamp,
            now_monotonic=now,
            eligible_keys=set(zones) if matched else None,
            evidence_status="MATCHED" if matched else status,
            source_session_id=self.social_memory.session_id,
            evidence_received_monotonic=(
                pair.persons_received_monotonic
                if matched
                else None
            ),
            unavailable_since_monotonic=self._source_missing_since,
            coverage_complete=(
                matched and self._source_coverage_complete
            ),
        )

        self.latest_memory_situation = situation
        self._memory_current = current
        self._memory_zones = zones

        previous_participants = set(self._memory_participants)
        participants = set(situation.participants)

        new_episode = bool(
            situation.episode_id
            and situation.episode_id != self.social_episode_id
        )

        if new_episode:
            self.social_episode_id = situation.episode_id
            self._memory_primary = None
            self._memory_event_key = None
            self._last_clear_source = None

            self.observation_lost_since = None
            self.episode_retirement_notified = False
            self.retired_episode_id = ""

            self.cleared_person_ids.clear()
            self.cleared_person_id = ""

            self.conflicts_detected += 1

        self._memory_participants = participants
        self.conflict_person_ids = {
            key[1] for key in participants
        }
        self.social_conflict_active = situation.unresolved

        # Only current persons WITH current zones provide
        # authoritative social geometry.
        candidates = (
            participants
            & set(current)
            & set(zones)
        )

        # Newly confirmed episode members, not merely tracks
        # that reappeared after temporary observation loss.
        additions = (
            participants - previous_participants
            if not new_episode
            else set()
        )

        # ---------------------------------------------------------
        # Stable primary identity.
        #
        # Never replace the primary solely because it is missing
        # from the latest camera frame.
        # ---------------------------------------------------------

        if self._memory_primary not in participants:
            self._memory_primary = None

        if self._memory_primary is None and candidates:
            self._memory_primary = min(
                candidates,
                key=lambda key: (
                    math.hypot(
                        *current[key].latest.position[:2]
                    ),
                    key,
                ),
            )

        if self._memory_primary is not None:
            self.current_person_id = self._memory_primary[1]
        elif not participants:
            self.current_person_id = ""

        if situation.transition == "CLEAR":
            self.clear_events += 1
            self.cleared_person_ids = set(
                self.conflict_person_ids
            )
            self.cleared_person_id = self.current_person_id

        retired = self.social_situation_evaluator.retired

        # ---------------------------------------------------------
        # Report newly confirmed participants independently.
        #
        # Each event uses that person's actual current zone.
        # The episode's primary identity remains unchanged.
        # ---------------------------------------------------------

        if situation.unresolved and not retired:
            for key in sorted(additions & candidates):
                self._publish_memory_event(
                    SOCIAL_EVENT_INTRUSION_CONFIRMED,
                    source=pair.zones,
                    zone=zones[key],
                    force=True,
                    explain=True,
                )

        # ---------------------------------------------------------
        # Canonical episode state.
        # ---------------------------------------------------------

        if situation.unresolved:
            if situation.observation_status == "LOST":
                self._set_episode_state(
                    EPISODE_OBSERVATION_LOST
                )

                self._publish_memory_event(
                    SOCIAL_EVENT_OBSERVATION_LOST
                )

            elif (
                situation.observation_status == "OBSERVED"
                and not retired
            ):
                was_lost = (
                    self.episode_state
                    == EPISODE_OBSERVATION_LOST
                )

                self._set_episode_state(
                    EPISODE_CONFLICT
                )

                if self._memory_primary in candidates:
                    event = (
                        SOCIAL_EVENT_OBSERVATION_REACQUIRED
                        if was_lost and not new_episode
                        else SOCIAL_EVENT_INTRUSION_CONFIRMED
                    )

                    # New participants have already received
                    # individual intrusion events above.
                    # Do not generate a redundant forced
                    # primary intrusion just for that change.
                    if (
                        not additions
                        or new_episode
                        or was_lost
                    ):
                        self._publish_memory_event(
                            event,
                            source=pair.zones,
                            zone=zones[self._memory_primary],
                            force=(
                                new_episode
                                or was_lost
                            ),
                            explain=new_episode,
                        )

            # TEMPORARILY_MISSING deliberately preserves
            # the episode without claiming clearance.

        elif situation.episode_id:
            self._set_episode_state(
                EPISODE_CLEARED
            )

            if self._memory_clear_is_current():
                clear_source = (
                    situation.episode_id,
                    pair.key,
                )

                if clear_source != self._last_clear_source:
                    self._publish_memory_event(
                        SOCIAL_EVENT_CLEAR,
                        source=pair.persons,
                        force=(
                            situation.transition == "CLEAR"
                        ),
                    )

                    self._last_clear_source = clear_source

        # ---------------------------------------------------------
        # Publish factual situation diagnostics.
        # ---------------------------------------------------------

        changed = (
            situation.revision
            != self._last_memory_situation_revision
        )

        if (
            changed
            or situation.transition
            or now - self._last_memory_situation_publication >= 0.50
        ):
            payload = asdict(situation)

            observed_conflicts = [
                key
                for key in sorted(candidates)
                if (
                    math.hypot(
                        *current[key].latest.position[:2]
                    ) <= self.release_distance
                )
            ]

            payload.update(
                version=2,
                producer="social_situation_evaluator",
                evaluation_stamp_ns=ros_now,
                authoritative_social_state=True,
                authoritative_for_navigation=False,
                retired=retired,
                source_status=(
                    "MATCHED" if matched else status
                ),
                source_pair_counts=dict(
                    self._source_pair_counts
                ),
                source_buffer_rejected_count=(
                    self.social_source_buffer.rejected_count
                ),
                observed_conflicts=observed_conflicts,
                temporal_continuity_enabled=(
                    self.temporal_continuity_enabled
                ),
                effective_observation_loss_grace_sec=(
                    self.effective_observation_loss_grace_sec
                ),
            )

            self.memory_situation_publisher.publish(
                String(data=json.dumps(payload))
            )

            self._last_memory_situation_revision = (
                situation.revision
            )
            self._last_memory_situation_publication = now

    def _evaluate_individual_situation(
        self,
        *,
        pair,
        matched,
        now,
        ros_now,
    ):
        """
        Evaluate individual observations from a validated source pair.

        Publish individual states without promoting partial
        source coverage to complete coverage.
        """
        evaluator = self.individual_social_evaluator

        new_pair = bool(
            matched
            and pair is not None
            and (
                pair.key[0],
                pair.key[1],
            ) != self._individual_last_source
        )

        if new_pair:
            try:
                persons_by_id = {
                    str(person.id).strip(): person
                    for person in pair.persons.persons
                }

                observations = []

                for zone in pair.zones.zones:
                    person = persons_by_id[
                        str(zone.person_id).strip()
                    ]

                    generation = person.identity_generation

                    if (
                        type(generation) is not int
                        or generation < 0
                    ):
                        raise ValueError(
                            "Invalid identity_generation"
                        )

                    observations.append(
                        ObservedPerson(
                            key=(
                                str(pair.key[0]),
                                str(person.id).strip(),
                                generation,
                            ),
                            distance=math.hypot(
                                float(zone.center.x),
                                float(zone.center.y),
                            ),
                        )
                    )

                result = evaluator.update(
                    session=str(pair.key[0]),
                    stamp_ns=int(pair.key[1]),
                    received_monotonic=(
                        pair.persons_received_monotonic
                    ),
                    now_ros_ns=int(ros_now),
                    now_monotonic=now,
                    persons=observations,
                    coverage_complete=bool(
                        self._source_coverage_complete
                    ),
                )

            except (
                ValueError,
                TypeError,
                KeyError,
                AttributeError,
                OverflowError,
            ) as exc:
                self.get_logger().error(
                    "[INDIVIDUAL SOCIAL] "
                    f"Invalid source: {exc}",
                    throttle_duration_sec=2.0,
                )

                result = evaluator.unavailable(
                    now_ros_ns=int(ros_now),
                    now_monotonic=now,
                )

            else:
                self._individual_last_source = (
                    str(pair.key[0]),
                    int(pair.key[1]),
                )

        elif matched:
            result = evaluator.tick(
                now_ros_ns=int(ros_now),
                now_monotonic=now,
            )

        else:
            result = evaluator.unavailable(
                now_ros_ns=int(ros_now),
                now_monotonic=now,
            )

        # Publish every new pair.
        # Limit publications without a new pair to 5 Hz.
        if (
            not new_pair
            and now - self._individual_last_publication < 0.20
        ):
            return

        payload = {
            "version": 1,
            "producer": "individual_social_evaluator",
            "source_session": result.source_session,
            "source_stamp_ns": result.source_stamp_ns,
            "source_fresh": result.source_fresh,
            "source_coverage_complete": (
                result.source_coverage_complete
            ),
            "evaluation_stamp_ns": int(ros_now),
            "participants": [
                asdict(row)
                for row in result.participants
            ],
        }

        self.individual_situation_publisher.publish(
            String(
                data=json.dumps(
                    payload,
                    separators=(",", ":"),
                )
            )
        )

        self._individual_last_publication = now
        
    def _social_memory_snapshot(self):
        """Provide a consistent immutable view to the future evaluator."""
        return self.social_memory.snapshot(
            now_ros_ns=self.get_clock().now().nanoseconds,
            now_monotonic=time.monotonic(),
        )
    
    def _initialize_social_reassessment(self):
        """Initialize the factual reassessment request channel."""
        self._sr_pending = None
        self._sr_observation = None

        self._sr_result_pub = self.create_publisher(
            String,
            "/social_navigation/reassessment_result",
            reliable_qos(depth=10),
        )

        self._sr_request_sub = self.create_subscription(
            String,
            "/social_navigation/reassessment_request",
            self._social_reassessment_request_callback,
            reliable_qos(depth=10),
        )

        # Uses the same default callback group as social_zones_callback.
        # The timer therefore reads the observation after its callback
        # has completed the normal episode evaluation.
        self._sr_timer = self.create_timer(
            0.10,
            self._social_reassessment_tick,
        )

    def _initialize_social_evaluator(self):
        """Select the startup-only temporal-continuity ablation condition."""
        self.declare_parameter(
            "temporal_continuity_enabled",
            True,
        )

        self.declare_parameter(
            "observation_loss_grace_sec",
            0.50,
        )

        enabled = self.get_parameter(
            "temporal_continuity_enabled"
        ).value

        if not isinstance(enabled, bool):
            raise ValueError(
                "temporal_continuity_enabled must be a boolean."
            )

        configured_grace = float(
            self.get_parameter(
                "observation_loss_grace_sec"
            ).value
        )

        if (
            not math.isfinite(configured_grace)
            or configured_grace <= 0.0
        ):
            raise ValueError(
                "observation_loss_grace_sec must be "
                "finite and positive."
            )

        self.temporal_continuity_enabled = enabled

        self.effective_observation_loss_grace_sec = (
            configured_grace
            if enabled
            else 0.0
        )

        self.social_situation_evaluator = SocialSituationEvaluator(
            entry_distance=self.entry_distance,
            release_distance=self.release_distance,
            entry_duration=self.conflict_confirmation_sec,
            clear_duration=self.clear_confirmation_sec,
            missing_grace=(
                self.effective_observation_loss_grace_sec
            ),
            maximum_gap=self.social_observation_timeout_sec,
        )

        condition = (
            "TEMPORAL"
            if enabled
            else "REACTIVE"
        )

        self.get_logger().info(
            "[SOCIAL ABLATION] "
            f"condition={condition} | "
            "missing_grace_sec="
            f"{self.effective_observation_loss_grace_sec:.3f} | "
            "episode_memory=RETAINED | "
            "historical_geometry_for_motion=DISABLED"
        )

    def _social_reassessment_request_callback(self, msg):
        """Accept one recent request without resetting the episode."""
        try:
            data = json.loads(msg.data)

            if (
                not isinstance(data, dict)
                or data.get("version") != 1
            ):
                return

            request_id = str(data["request_id"]).strip()
            goal_id = str(data["goal_id"]).strip()
            requested_ns = int(data["requested_ns"])
            session = str(data["session"])
            episode = str(data["episode"])

        except (ValueError, TypeError, KeyError):
            return

        age = (
            self.get_clock().now().nanoseconds - requested_ns
        ) / 1e9

        if (
            not request_id
            or not goal_id
            or not 0.0 <= age <= 2.0
        ):
            return

        # Duplicates and competing requests cannot extend the deadline.
        if self._sr_pending is not None:
            return

        self._sr_pending = {
            "request_id": request_id,
            "goal_id": goal_id,
            "requested_ns": requested_ns,
            "session": session,
            "episode": episode,
            "deadline": time.monotonic() + 3.0,
        }

    def _social_reassessment_tick(self):
        """Answer using newer evidence and the authoritative social episode."""
        pending = self._sr_pending

        if pending is None:
            return

        self._evaluate_social_memory()

        identity = (
            self.decision_session_id,
            self.social_episode_id,
        )

        expected = (
            pending["session"],
            pending["episode"],
        )

        status = None
        stamp = 0

        if any(expected) and identity != expected:
            status = "CONTEXT_CHANGED"

        else:
            msg = self._memory_message

            if msg is not None:
                source = (
                    msg.header.stamp.sec * 1_000_000_000
                    + msg.header.stamp.nanosec
                )

                age = (
                    self.get_clock().now().nanoseconds
                    - source
                ) / 1e9

                fresh_after_request = (
                    source > pending["requested_ns"]
                    and (
                        0.0
                        <= age
                        <= self.social_observation_timeout_sec
                    )
                )

                if (
                    fresh_after_request
                    and self._paired_evidence_is_fresh()
                ):
                    situation = self.latest_memory_situation

                    if situation is not None:
                        if self._memory_clear_is_current():
                            status = "CLEAR"

                        elif (
                            situation.unresolved
                            and situation.observation_status == "OBSERVED"
                            and not self.social_situation_evaluator.retired
                            and self._memory_primary in self._memory_zones
                            and any(
                                math.hypot(*person.latest.position[:2])
                                <= self.release_distance
                                for person in self._memory_current.values()
                            )
                        ):
                            # Positive observed conflict remains reportable
                            # even when other source candidates are unresolved.
                            status = "CONFLICT"

                        elif (
                            not msg.persons
                            and situation.unresolved
                        ):
                            # Empty current tracks cannot clear an episode.
                            status = "EMPTY_UNRESOLVED"

                        elif (
                            not situation.unresolved
                            and self._source_coverage_complete
                            and (
                                len(self._memory_current)
                                == len(msg.persons)
                            )
                            and all(
                                math.hypot(*person.latest.position[:2])
                                > self.entry_distance
                                for person in self._memory_current.values()
                            )
                        ):
                            # Includes a genuine complete empty observation.
                            status = "NO_CURRENT_INTRUSION"

                        if status is not None:
                            stamp = source

        if status is None:
            if time.monotonic() < pending["deadline"]:
                return

            status = "UNKNOWN"

        response = {
            "version": 1,
            "request_id": pending["request_id"],
            "goal_id": pending["goal_id"],
            "session": identity[0],
            "episode": identity[1],
            "sequence": self.decision_sequence,
            "status": status,
            "observation_stamp_ns": stamp,
        }

        self._sr_pending = None

        self._sr_result_pub.publish(
            String(data=json.dumps(response))
        )

    def _enqueue_social_source(self, kind, msg):
        """Drain expired work, queue one input, then consume ready pairs."""
        self._consume_ready_source_pairs()

        now = time.monotonic()

        accepted = self.social_source_buffer.push(
            kind=kind,
            msg=msg,
            now_ros_ns=self.get_clock().now().nanoseconds,
            now_monotonic=now,
        )

        if not accepted:
            reason = self.social_source_buffer.last_rejection

            self.get_logger().warning(
                "[SOCIAL SOURCE] Input rejected | "
                f"kind={kind} | reason={reason}",
                throttle_duration_sec=2.0,
            )

            if reason == "BUFFER_FULL":
                self._source_recovery_after_monotonic = now

                self._invalidate_source_evidence(
                    status="OVERFLOW",
                    since=now,
                )

        self._consume_ready_source_pairs()

    def _invalidate_source_evidence(self, *, status, since):
        """Remove current evidence without deleting observational memory."""
        self._source_pair = None
        self._source_pair_status = status
        self._source_coverage_complete = False

        self._memory_message = None
        self._sr_observation = None

        if self._source_missing_since is None:
            self._source_missing_since = since
        else:
            self._source_missing_since = min(
                self._source_missing_since,
                since,
            )

    def _paired_evidence_is_fresh(self):
        """Check both source age and original reception ages."""
        pair = self._source_pair

        if (
            pair is None
            or self._source_pair_status != "MATCHED"
        ):
            return False

        if pair.key != (
            self.social_memory.session_id,
            self.social_memory.last_array_stamp_ns,
            self.expected_frame,
        ):
            return False

        now = time.monotonic()

        source_age = (
            self.get_clock().now().nanoseconds
            - pair.key[1]
        ) / 1e9

        return bool(
            -0.10
            <= source_age
            <= self.social_observation_timeout_sec
            and all(
                received is not None
                and (
                    0.0
                    <= now - received
                    <= self.social_observation_timeout_sec
                )
                for received in (
                    pair.persons_received_monotonic,
                    pair.zones_received_monotonic,
                )
            )
        )

    def _consume_ready_source_pairs(self):
        """Validate, commit and evaluate each ready source frame in order."""
        ready = self.social_source_buffer.take_ready(
            now_ros_ns=self.get_clock().now().nanoseconds,
            now_monotonic=time.monotonic(),
        )

        for pair in ready:
            candidate_memory = None
            coverage_complete = False
            status = pair.status

            recovery_after = (
                self._source_recovery_after_monotonic
            )

            if (
                status == "MATCHED"
                and recovery_after is not None
                and (
                    pair.persons_received_monotonic <= recovery_after
                    or pair.zones_received_monotonic <= recovery_after
                )
            ):
                status = "PRE_OVERFLOW_EVIDENCE"

            if status == "MATCHED":
                validated = self._validate_source_pair(pair)

                if validated is None:
                    status = "INVALID"
                else:
                    candidate_memory, coverage_complete = validated

            if candidate_memory is None:
                # INCOMPLETE, STALE and INVALID never update memory.
                self._invalidate_source_evidence(
                    status=status,
                    since=pair.first_received_monotonic,
                )

            else:
                # Commit only after validating the complete relation:
                # source persons, derived zones and their provenance.
                self.social_memory = candidate_memory

                self._source_pair = pair
                self._source_pair_status = "MATCHED"
                self._source_coverage_complete = coverage_complete
                self._source_missing_since = None
                self._source_recovery_after_monotonic = None

                self._memory_message = pair.persons
                self._sr_observation = pair.zones

                self.last_social_stamp_ns = pair.key[1]
                self.last_social_observation_at = (
                    pair.persons_received_monotonic
                )

                self.zones_evaluated += len(pair.zones.zones)

                # Count genuine empty source observations, not missing zones.
                if (
                    pair.persons.source_observation_complete
                    and pair.persons.source_detection_count == 0
                    and not any(
                        person.observed_in_current_array
                        for person in pair.persons.persons
                    )
                ):
                    self.empty_observation_arrays += 1

            self._source_pair_counts[status] = (
                self._source_pair_counts.get(status, 0) + 1
            )

            # Evaluate every consumed frame, not only the last frame
            # in a batch. This preserves confirmation sample counts.
            self._evaluate_social_memory()

            if (
                self._paired_evidence_is_fresh()
                and self._source_coverage_complete
            ):
                health = (
                    "OCCUPIED"
                    if pair.persons.persons
                    else "EMPTY"
                )

                stamp = pair.persons.header.stamp

                self.observation_health_publisher.publish(
                    String(
                        data=(
                            f"{health}:"
                            f"{self.decision_session_id}:"
                            f"{self.decision_sequence}:"
                            f"{stamp.sec}:{stamp.nanosec}"
                        )
                    )
                )

            else:
                # Do not renew a positive health report using partial evidence.
                # Consumers must age earlier reports using their source stamp.
                self.get_logger().warning(
                    "[SOCIAL SOURCE] Current coverage unavailable | "
                    f"pair_status={self._source_pair_status} | "
                    f"coverage_complete={self._source_coverage_complete}",
                    throttle_duration_sec=2.0,
                )

        if not ready:
            # No new sample: only age the existing evidence.
            self._evaluate_social_memory()
        
        # Retry missing map projections independently of new observations.
        self._retry_unprojected_social_zones()
        # Publish qualified geometry after projection retries.
        self._publish_qualified_social_geometry()

    # ============================================================
    # SOCIAL OBSERVATION CALLBACK
    # ============================================================

    def _tracked_persons_memory_callback(self, msg):
        """Queue tracked persons without independently updating memory."""
        self._enqueue_social_source("persons", msg)

    def social_zones_callback(self, msg):
        """Queue derived zones for exact source pairing."""
        self.arrays_received += 1
        self._enqueue_social_source("zones", msg)

    # ============================================================
    # EPISODE STATE
    # ============================================================

    def _set_episode_state(self, state: str) -> None:
        """
        Maintain episode state and the current loss interval.

        Reacquisition or factual clearance ends the loss interval.
        Repeated LOST updates preserve its original start time.
        """

        if state == self.episode_state:
            return

        previous = self.episode_state

        if state == EPISODE_OBSERVATION_LOST:
            if self.observation_lost_since is None:
                self.observation_lost_since = time.monotonic()

            # A new loss interval may issue one retirement notice.
            self.episode_retirement_notified = False

        else:
            self.observation_lost_since = None
            self.episode_retirement_notified = False

        self.episode_state = state

        self.episode_state_publisher.publish(
            String(data=state)
        )

        self.get_logger().info(
            f"[SOCIAL EPISODE] {previous} -> {state}"
        )

    def _attach_episode_participants(self, decision):
        """Attach factual evidence for every episode participant."""

        situation = self.latest_memory_situation

        decision.participants = []
        decision.observation_complete = False
        decision.tracking_session_id = ""

        if situation is None:
            return

        matched = self._paired_evidence_is_fresh()
        current = self._memory_current

        for key in situation.participants:
            session_id, person_id = key

            participant = SocialParticipant()
            participant.tracking_session_id = session_id
            participant.person_id = person_id

            participant.observed = bool(
                matched
                and key in current
                and key in self._memory_zones
            )

            participant.conflict = bool(
                participant.observed
                and math.hypot(
                    *current[key].latest.position[:2]
                ) <= self.release_distance
            )

            decision.participants.append(participant)

            if person_id == decision.person_id:
                decision.tracking_session_id = session_id

        # Completeness concerns the full current source,
        # never just the people listed in an episode.
        decision.observation_complete = bool(
            matched
            and self._source_coverage_complete
            and decision.participants
            and all(
                participant.observed
                for participant in decision.participants
            )
        )


    # ============================================================
    # SOCIAL GEOMETRY
    # ============================================================

    def _effective_zone_radius(
        self,
        zone: SocialZone,
    ) -> float:
        """
        Calculate the radius of a social ellipse toward the robot.
        """

        a = max(
            float(zone.radius_longitudinal),
            self.minimum_radius_m,
        )

        b = max(
            float(zone.radius_lateral),
            self.minimum_radius_m,
        )

        robot_direction = math.atan2(
            -float(zone.center.y),
            -float(zone.center.x),
        )

        relative_angle = (
            robot_direction
            - float(zone.heading)
        )

        cos_angle = math.cos(relative_angle)
        sin_angle = math.sin(relative_angle)

        denominator = math.sqrt(
            (b * cos_angle) ** 2
            + (a * sin_angle) ** 2
        )

        if denominator <= 1e-9:
            return max(a, b)

        return a * b / denominator
 
    # ============================================================
    # OBSERVATION TIMEOUT
    # ============================================================

    def context_timer_callback(self):
        """Retire prolonged loss without independently classifying perception."""
        if (
            self.social_conflict_active
            and self.episode_state == EPISODE_OBSERVATION_LOST
            and self.observation_lost_since is not None
            and (
                time.monotonic() - self.observation_lost_since
            ) >= self.lost_episode_retirement_sec
        ):
            self._retire_unobservable_episode()

    def _retire_unobservable_episode(self):
        """End one interaction lifecycle without reporting geometric clearance."""
        if (
            not self.social_conflict_active
            or self.episode_state != EPISODE_OBSERVATION_LOST
            or not self.social_episode_id
            or self.episode_retirement_notified
        ):
            return

        self.social_situation_evaluator.retire()

        self.episode_retirement_notified = True
        self.retired_episode_id = self.social_episode_id

        self.episode_retirement_publisher.publish(
            String(
                data=(
                    f"RETIRED:"
                    f"{self.decision_session_id}:"
                    f"{self.social_episode_id}"
                )
            )
        )

        self.get_logger().warning(
            "[SOCIAL RETIREMENT] "
            "Episode retired without clearance | "
            f"episode={self.social_episode_id}"
        )

    # ============================================================
    # NAVIGATION SNAPSHOT
    # ============================================================

    def _navigation_snapshot(
        self,
    ) -> Tuple[str, str, bool]:

        msg = self.latest_navigation_state

        if msg is None:
            return "", "", False

        return (
            str(msg.navigation_state or "").strip(),
            str(msg.navigation_goal_id or "").strip(),
            bool(msg.navigation_active),
        )

    def _attach_navigation_snapshot(
        self,
        decision: SocialNavigationDecision,
    ) -> None:

        (
            state,
            goal_id,
            navigation_active,
        ) = self._navigation_snapshot()

        decision.navigation_state = state
        decision.navigation_goal_id = goal_id
        decision.navigation_active = navigation_active

    # ============================================================
    # BUILD SOCIAL EVENT
    # ============================================================

    def _build_event(
        self,
        msg: SocialZoneArray,
        zone: SocialZone,
        event: str,
        person_distance: float,
        social_radius: float,
        occupancy_ratio: float,
    ) -> SocialNavigationDecision:

        decision = SocialNavigationDecision()

        decision.header.frame_id = msg.header.frame_id
        decision.header.stamp = self.get_clock().now().to_msg()
        decision.decision_id = self._next_decision_id()

        decision.event = event
        decision.social_conflict_active = True

        decision.person_id = zone.person_id

        decision.zone_type = zone.zone_type
        decision.tracking_state = zone.tracking_state

        decision.person_position = zone.center

        decision.person_distance = float(
            person_distance
        )

        decision.social_radius = float(
            social_radius
        )

        decision.person_speed = float(
            zone.person_speed
        )

        decision.occupancy_ratio = float(
            occupancy_ratio
        )

        self._attach_navigation_snapshot(decision)

        decision.requires_explanation = True

        return decision

    # ============================================================
    # BUILD CLEAR EVENT
    # ============================================================

    def _build_clear_event(
        self,
        msg: SocialZoneArray,
        previous_person_id: str,
    ) -> SocialNavigationDecision:

        decision = SocialNavigationDecision()

        decision.header.frame_id = msg.header.frame_id
        decision.header.stamp = self.get_clock().now().to_msg()
        decision.decision_id = self._next_decision_id()

        decision.event = SOCIAL_EVENT_CLEAR
        decision.social_conflict_active = False

        decision.person_id = previous_person_id

        decision.zone_type = ZONE_NONE
        decision.tracking_state = TRACKING_NONE

        decision.person_distance = 0.0
        decision.social_radius = 0.0
        decision.person_speed = 0.0
        decision.occupancy_ratio = 0.0

        self._attach_navigation_snapshot(decision)

        decision.requires_explanation = False

        return decision

    # ============================================================
    # ZONE VALIDATION
    # ============================================================

    def _is_valid_zone(
        self,
        zone: SocialZone,
    ) -> bool:
        """Validate observed zone geometry."""

        values = (
            zone.center.x,
            zone.center.y,
            zone.radius_longitudinal,
            zone.radius_lateral,
            zone.person_speed,
            zone.heading,
        )

        if not all(
            math.isfinite(float(value))
            for value in values
        ):
            return False

        if zone.radius_longitudinal <= 0.0:
            return False

        if zone.radius_lateral <= 0.0:
            return False

        return True

    # ============================================================
    # DECISION IDENTIFIERS
    # ============================================================

    def _next_decision_id(self) -> str:
        """
        Generate IDs unique across process restarts.
        """

        self.decision_sequence += 1

        return (
            f"{self.decision_session_id}:"
            f"{self.social_episode_id}:"
            f"{self.decision_sequence}"
        )
    
    # ============================================================
    # DIAGNOSTICS
    # ============================================================

    def log_diagnostics(self) -> None:
        """
        Report social event and observation health.
        """

        now = time.monotonic()

        if self.last_social_observation_at is None:
            observation_age = float("inf")
        else:
            observation_age = max(
                0.0,
                now - self.last_social_observation_at,
            )

        self.get_logger().info(
            "[SOCIAL NAVIGATION] "
            f"ARRAYS={self.arrays_received} | "
            f"ZONES={self.zones_evaluated} | "
            f"CONFLICTS={self.conflicts_detected} | "
            f"EPISODE_STATE={self.episode_state} | "
            f"CONFLICT_ACTIVE={self.social_conflict_active} | "
            f"CLEAR_EVENTS={self.clear_events} | "
            f"CURRENT_PERSON="
            f"{self.current_person_id or 'NONE'} | "
            f"INVALID_FRAME={self.invalid_frame} | "
            f"INVALID_ZONE={self.invalid_zone} | "
            f"INVALID_STAMPS={self.invalid_observation_stamps} | "
            f"REJECTED_ARRAYS={self.rejected_observation_arrays} | "
            f"EMPTY_ARRAYS={self.empty_observation_arrays} | "
            f"OBS_AGE={observation_age:.2f}s | "
            f"SOURCE_STATUS={self._source_pair_status} | "
            f"COVERAGE_COMPLETE={self._source_coverage_complete} | "
            f"MEMORY_PERSONS={len(self._social_memory_snapshot())}"
        )

# ================================================================
# MAIN
# ================================================================

def main(args=None):

    rclpy.init(args=args)

    node = SocialNavigationDecisionNode()

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
