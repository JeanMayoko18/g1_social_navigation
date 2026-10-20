
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

from typing import Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)

from std_msgs.msg import String

from g1_social_msgs.msg import (
    NavigationState,
    SocialNavigationDecision,
    SocialZone,
    SocialZoneArray,
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
            not math.isfinite(
                self.social_observation_timeout_sec
            )
            or self.social_observation_timeout_sec <= 0.0
        ):
            raise ValueError(
                "social_observation_timeout_sec must be "
                "finite and positive."
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

    def _validate_social_observation(self, msg):
        """Validate observation identity, geometry and source time."""
        if msg.header.frame_id != self.expected_frame:
            self.invalid_frame += 1
            self.rejected_observation_arrays += 1
            return False

        stamp = msg.header.stamp
        sec = int(stamp.sec)
        nanosec = int(stamp.nanosec)

        stamp_ns = (
            sec * 1_000_000_000
            + nanosec
        )

        age = (
            self.get_clock().now().nanoseconds
            - stamp_ns
        ) / 1e9

        if (
            sec < 0
            or nanosec < 0
            or nanosec >= 1_000_000_000
            or stamp_ns <= 0
            or not (
                -0.10
                <= age
                <= self.social_observation_timeout_sec
            )
            or (
                self.last_social_stamp_ns is not None
                and stamp_ns <= self.last_social_stamp_ns
            )
        ):
            self.invalid_observation_stamps += 1
            self.rejected_observation_arrays += 1

            self.get_logger().warning(
                "[SOCIAL OBSERVATION REJECTED] "
                f"age={age:.3f}s | "
                f"stamp_ns={stamp_ns}",
                throttle_duration_sec=2.0,
            )
            return False

        seen = set()

        for zone in msg.zones:
            person_id = str(zone.person_id).strip()

            if (
                not person_id
                or person_id in seen
                or zone.tracking_state != "TRACKED"
                or not self._is_valid_zone(zone)
                or not math.isfinite(float(zone.center.z))
            ):
                self.invalid_zone += 1
                self.rejected_observation_arrays += 1
                return False

            seen.add(person_id)

        self.last_social_stamp_ns = stamp_ns

        if not msg.zones:
            self.empty_observation_arrays += 1

        return True
    # ============================================================
    # SOCIAL OBSERVATION CALLBACK
    # ============================================================

    def social_zones_callback(self, msg):
        self.arrays_received += 1

        previous_stamp = self.last_social_stamp_ns

        if not self._validate_social_observation(msg):
            return

        # Only validated, current source observations
        # may establish observation health.
        #
        # EMPTY means no zones in this validated array.
        # It is NOT geometric CLEAR of an old episode.
        health = "EMPTY" if not msg.zones else "OCCUPIED"

        self.observation_health_publisher.publish(
            String(
                data=(
                    f"{health}:"
                    f"{self.decision_session_id}:"
                    f"{self.decision_sequence}:"
                    f"{int(msg.header.stamp.sec)}:"
                    f"{int(msg.header.stamp.nanosec)}"
                )
            )
        )

        now = time.monotonic()

        gap = (
            (
                self.last_social_observation_at is not None
                and (
                    now - self.last_social_observation_at
                ) > self.social_observation_timeout_sec
            )
            or (
                previous_stamp is not None
                and (
                    self.last_social_stamp_ns - previous_stamp
                ) / 1e9 > self.social_observation_timeout_sec
            )
        )

        if gap:
            self.conflict_candidate_since = None
            self.clear_candidate_since = None

            if self.social_conflict_active:
                self._mark_observation_lost()

        self.last_social_observation_at = now

        observed = {
            zone.person_id: zone
            for zone in msg.zones
        }

        conflict = self._find_most_constraining_zone(
            msg
        )

        if conflict is not None:
            zone, distance, radius, occupancy = conflict

            new_person_after_retirement = (
                self.social_conflict_active
                and bool(self.social_episode_id)
                and self.social_episode_id == self.retired_episode_id
                and zone.person_id not in self.conflict_person_ids
                and self.conflict_person_ids.isdisjoint(observed.keys())
            )

            if new_person_after_retirement:
                continuous = (
                    self.retired_new_person_id == zone.person_id
                    and self.retired_new_person_last_seen is not None
                    and now - self.retired_new_person_last_seen
                    <= self.social_observation_timeout_sec
                )

                if not continuous:
                    self.retired_new_person_id = zone.person_id
                    self.retired_new_person_since = now
                    self.retired_new_person_last_seen = now
                    self.retired_new_person_samples = 1
                    return

                self.retired_new_person_samples += 1
                self.retired_new_person_last_seen = now

                if (
                    self.retired_new_person_samples < 2
                    or now - self.retired_new_person_since
                    < self.conflict_confirmation_sec
                ):
                    return

                old_episode = self.social_episode_id

                # Retain the unresolved old participants.
                self.unresolved_retired_episodes[old_episode] = tuple(
                    sorted(self.conflict_person_ids)
                )

                if len(self.unresolved_retired_episodes) > 32:
                    oldest = next(
                        iter(self.unresolved_retired_episodes)
                    )
                    self.unresolved_retired_episodes.pop(oldest)

                # The normal new_episode block below creates a fresh
                # episode ID and publishes INTRUSION_CONFIRMED.
                # No geometric CLEAR is fabricated.
                self.social_conflict_active = False

                # Confirmation has already occurred over two fresh
                # samples. The margin avoids floating-point rounding.
                self.conflict_candidate_since = (
                    now - self.conflict_confirmation_sec - 0.001
                )

                self.get_logger().warning(
                    "[SOCIAL EPISODE REARM] "
                    f"retired={old_episode} | "
                    f"new_person={zone.person_id} | "
                    "previous_clear=False"
                )

            self.retired_new_person_id = ""
            self.retired_new_person_since = None
            self.retired_new_person_last_seen = None
            self.retired_new_person_samples = 0

            self.conflicts_detected += 1
            self.clear_candidate_since = None

            new_episode = not self.social_conflict_active

            if new_episode:
                if self.conflict_candidate_since is None:
                    self.conflict_candidate_since = now

                if (
                    now - self.conflict_candidate_since
                    < self.conflict_confirmation_sec
                ):
                    return

                self.social_episode_id = uuid.uuid4().hex
                self.social_conflict_active = True

                self.conflict_person_ids.clear()
                self.cleared_person_ids.clear()

            for person_id, observed_zone in observed.items():
                observed_distance = math.hypot(
                    float(observed_zone.center.x),
                    float(observed_zone.center.y),
                )

                if observed_distance <= self.release_distance:
                    self.conflict_person_ids.add(
                        person_id
                    )

            self.current_person_id = zone.person_id
            self.conflict_candidate_since = None

            # A missing participant remains unresolved even if
            # another participant is still visible.
            if not self.conflict_person_ids.issubset(observed):
                self._mark_observation_lost()
                return

            changed = (
                new_episode
                or self.episode_state != EPISODE_CONFLICT
            )

            self._set_episode_state(
                EPISODE_CONFLICT
            )

            self._publish_current_social_state(
                msg=msg,
                conflict=conflict,
                force=changed,
            )
            return

        self.conflict_candidate_since = None

        if not self.social_conflict_active:
            # Repeat CLEAR only when new geometry confirms it again.
            if (
                self.episode_state == EPISODE_CLEARED
                and self._observed_episode_is_clear(observed)
            ):
                self._publish_current_social_state(
                    msg=msg
                )

            return

        if not self._observed_episode_is_clear(observed):
            self.clear_candidate_since = None
            self._mark_observation_lost()
            return

        if self.clear_candidate_since is None:
            self.clear_candidate_since = now

        if (
            now - self.clear_candidate_since
            < self.clear_confirmation_sec
        ):
            return

        self._handle_episode_clear(msg)
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
    # ============================================================
    # OBSERVATION LOST
    # ============================================================

    def _mark_observation_lost(self):
        """
        Report observation loss after a short continuity grace
        period, rather than after one missing observation.

        Missing perception remains UNKNOWN, never CLEAR.
        """
        if not self.social_conflict_active:
            return

        if self.episode_state == EPISODE_OBSERVATION_LOST:
            self._publish_current_social_state(
                force=False
            )
            return

        now = time.monotonic()

        last_publication = (
            self.last_state_publication_at
        )

        # The normal conflict refresh is up to 2 Hz.
        # Short missed detections are not sufficient to
        # declare loss of the tracked social episode.
        grace_sec = min(
            1.0,
            self.social_observation_timeout_sec,
        )

        if (
            self.episode_state == EPISODE_CONFLICT
            and last_publication is not None
            and now - last_publication < grace_sec
        ):
            return

        self.conflict_candidate_since = None
        self.clear_candidate_since = None

        self._set_episode_state(
            EPISODE_OBSERVATION_LOST
        )

        self._publish_current_social_state(
            force=True
        )
        
    
    def _retire_unobservable_episode(self) -> None:
        """
        Notify retirement once while retaining unresolved evidence.

        Retirement ends the interaction lifecycle. It does not erase
        the participant identities needed for later reacquisition
        or factual geometric clearance.
        """

        if (
            not self.social_conflict_active
            or self.episode_state != EPISODE_OBSERVATION_LOST
            or not self.social_episode_id
        ):
            return

        if getattr(self, "episode_retirement_notified", False):
            return

        # A previous retirement remains known after reacquisition.
        if self.retired_episode_id == self.social_episode_id:
            self.episode_retirement_notified = True
            return

        self.retired_episode_id = self.social_episode_id
        self.episode_retirement_notified = True

        # Preserve:
        # - social_episode_id
        # - current_person_id
        # - conflict_person_ids
        # - social_conflict_active
        # - EPISODE_OBSERVATION_LOST
        #
        # A later observation must still be able to resolve
        # this exact episode.

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
            "[SOCIAL EPISODE RETIRED] "
            f"EPISODE={self.social_episode_id} | "
            "REASON=OBSERVATION_LOST_TIMEOUT | "
            "PARTICIPANT_MEMORY=RETAINED | "
            "GEOMETRIC_CLEAR=False | "
            "NAVIGATION_AUTHORIZED=False"
        )

    # ============================================================
    # SOCIAL GEOMETRY
    # ============================================================

    def _find_most_constraining_zone(
        self,
        msg: SocialZoneArray,
    ) -> Optional[
        Tuple[
            SocialZone,
            float,
            float,
            float,
        ]
    ]:
        """
        Find the zone producing the strongest social conflict.
        """

        strongest_conflict = None
        strongest_ratio = 0.0

        for zone in msg.zones:

            self.zones_evaluated += 1

            if not self._is_valid_zone(zone):

                self.invalid_zone += 1
                continue

            person_distance = math.hypot(
                float(zone.center.x),
                float(zone.center.y),
            )

            effective_social_radius = (
                self._effective_zone_radius(zone)
            )

            if self.social_conflict_active:
                threshold_distance = self.release_distance
            else:
                threshold_distance = self.entry_distance

            if person_distance > threshold_distance:
                continue

            denominator = max(
                person_distance,
                1e-6,
            )

            occupancy_ratio = (
                effective_social_radius
                / denominator
            )

            if occupancy_ratio > strongest_ratio:

                strongest_ratio = occupancy_ratio

                strongest_conflict = (
                    zone,
                    person_distance,
                    effective_social_radius,
                    occupancy_ratio,
                )

        return strongest_conflict

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
    # CLEAR EVENT
    # ============================================================

    def _handle_episode_clear(self, msg):
        if not self.social_conflict_active:
            return

        # Retain identities for subsequent fresh clearance updates.
        self.cleared_person_ids = set(
            self.conflict_person_ids
        )
        self.cleared_person_id = self.current_person_id

        self.social_conflict_active = False
        self.current_person_id = ""

        self.conflict_person_ids.clear()
        self.person_last_seen.clear()

        self.conflict_candidate_since = None
        self.clear_candidate_since = None

        self._set_episode_state(
            EPISODE_CLEARED
        )

        self._publish_current_social_state(
            msg=msg,
            force=True,
        )

        self.clear_events += 1
    # ============================================================
    # EVENT PUBLICATION
    # ============================================================

    def _publish_intrusion_confirmed(
        self,
        msg: SocialZoneArray,
        zone: SocialZone,
        distance: float,
        radius: float,
        occupancy: float,
    ) -> None:

        decision = self._build_event(
            msg=msg,
            zone=zone,
            event=SOCIAL_EVENT_INTRUSION_CONFIRMED,
            person_distance=distance,
            social_radius=radius,
            occupancy_ratio=occupancy,
        )

        self.publisher.publish(decision)

        self.get_logger().info(
            "[SOCIAL EVENT] "
            f"EVENT={SOCIAL_EVENT_INTRUSION_CONFIRMED} | "
            f"PERSON={zone.person_id} | "
            f"DISTANCE={distance:.3f}"
        )

    def _publish_observation_reacquired(
        self,
        msg: SocialZoneArray,
        zone: SocialZone,
        distance: float,
        radius: float,
        occupancy: float,
    ) -> None:
        """
        Report observation restoration during an unresolved episode.
        """

        decision = self._build_event(
            msg=msg,
            zone=zone,
            event=SOCIAL_EVENT_OBSERVATION_REACQUIRED,
            person_distance=distance,
            social_radius=radius,
            occupancy_ratio=occupancy,
        )

        decision.requires_explanation = False

        self.publisher.publish(decision)

        self.get_logger().info(
            "[SOCIAL EVENT] "
            f"EVENT={SOCIAL_EVENT_OBSERVATION_REACQUIRED} | "
            f"PERSON={zone.person_id} | "
            f"DISTANCE={distance:.3f}"
        )
    
    def _publish_current_social_state(
        self,
        msg=None,
        conflict=None,
        force=False,
    ):
        """Publish current evidence without replaying stale geometry."""
        now = time.monotonic()

        if self.episode_state == EPISODE_OBSERVATION_LOST:
            event = SOCIAL_EVENT_OBSERVATION_LOST

        elif (
            self.episode_state == EPISODE_CONFLICT
            and msg is not None
            and conflict is not None
        ):
            event = SOCIAL_EVENT_INTRUSION_CONFIRMED

        elif (
            self.episode_state == EPISODE_CLEARED
            and msg is not None
        ):
            event = SOCIAL_EVENT_CLEAR

        else:
            return

        if (
            not force
            and event == self.last_published_state_event
            and self.last_state_publication_at is not None
            and (
                now - self.last_state_publication_at
            ) < self.state_refresh_period_sec
        ):
            return

        if event == SOCIAL_EVENT_INTRUSION_CONFIRMED:
            zone, distance, radius, occupancy = conflict

            decision = self._build_event(
                msg=msg,
                zone=zone,
                event=event,
                person_distance=distance,
                social_radius=radius,
                occupancy_ratio=occupancy,
            )

            # Preserve the source observation timestamp.
            decision.header.stamp = msg.header.stamp

        elif event == SOCIAL_EVENT_CLEAR:
            decision = self._build_clear_event(
                msg=msg,
                previous_person_id=self.cleared_person_id,
            )

            # Clearance is supported by this fresh observation.
            decision.header.stamp = msg.header.stamp

        else:
            # This timestamp describes a current lack of observation.
            # No historical person geometry is attached.
            decision = SocialNavigationDecision()

            decision.header.stamp = (
                self.get_clock().now().to_msg()
            )
            decision.header.frame_id = self.expected_frame

            decision.decision_id = self._next_decision_id()
            decision.event = event
            decision.person_id = self.current_person_id

            decision.zone_type = ZONE_NONE
            decision.tracking_state = TRACKING_NONE
            decision.social_conflict_active = True

            self._attach_navigation_snapshot(decision)

        decision.requires_explanation = bool(
            force
            and event == SOCIAL_EVENT_INTRUSION_CONFIRMED
        )

        self.publisher.publish(decision)

        self.last_state_publication_at = now
        self.last_published_state_event = event
        
    # ============================================================
    # CLEARANCE VALIDATION
    # ============================================================

    def _observed_episode_is_clear(self, observed_zones):
        """
        Confirm clearance of the current episode participants.

        Never treat missing observations as geometric clearance.
        """

        required = (
            self.conflict_person_ids
            if self.social_conflict_active
            else self.cleared_person_ids
        )

        if not required:
            return False

        if not required.issubset(observed_zones.keys()):
            return False

        for person_id in required:
            zone = observed_zones[person_id]

            distance = math.hypot(
                float(zone.center.x),
                float(zone.center.y),
            )

            if (
                not math.isfinite(distance)
                or distance <= self.release_distance
            ):
                return False

        return True

    # ============================================================
    # OBSERVATION TIMEOUT
    # ============================================================


    def context_timer_callback(self):
        """
        Monitor observation freshness and retire episodes
        that remain unobservable for too long.

        Loss is never converted into geometric clearance.
        """

        if not self.social_conflict_active:
            return

        now = time.monotonic()

        source_age = (
            float("inf")
            if self.last_social_stamp_ns is None
            else (
                self.get_clock().now().nanoseconds
                - self.last_social_stamp_ns
            ) / 1e9
        )

        stale = (
            self.last_social_observation_at is None
            or (
                now - self.last_social_observation_at
            ) > self.social_observation_timeout_sec
            or not (
                -0.10
                <= source_age
                <= self.social_observation_timeout_sec
            )
        )

        if stale:
            self._mark_observation_lost()

        elif self.episode_state == EPISODE_OBSERVATION_LOST:
            # Keep reporting uncertainty while the episode
            # remains unresolved.
            self._publish_current_social_state()

        # A fresh array does not necessarily mean the missing
        # person has returned. Only a transition back to
        # EPISODE_CONFLICT cancels the loss timer.
        if self.episode_state != EPISODE_OBSERVATION_LOST:
            return

        if self.observation_lost_since is None:
            self.observation_lost_since = now
            return

        elapsed = now - self.observation_lost_since

        if elapsed < self.lost_episode_retirement_sec:
            return

        self._retire_unobservable_episode()

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
            f"OBS_AGE={observation_age:.2f}s"
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
