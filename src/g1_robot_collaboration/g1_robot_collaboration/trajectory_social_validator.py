#!/usr/bin/env python3

"""
Validate candidate navigation trajectories against current social zones.

Responsibility:
    Determine whether one exact CandidateTrajectory is socially admissible
    with respect to the latest usable SocialZoneArray.

Flow:
    CandidateTrajectory
            +
    SocialZoneArray
            ↓
    INPUT / FRESHNESS VALIDATION
            ↓
    TF2 FRAME ALIGNMENT
            ↓
    PATH SEGMENT SAMPLING
            ↓
    ORIENTED SOCIAL-ZONE GEOMETRY
            ↓
    TrajectorySocialValidation
        VALID
        CONFLICT
        UNKNOWN

Important:
    This node does not authorize navigation.

    It does not decide WAIT, REPLAN or RESUME.

    A VALID result applies only to the exact path_id contained in the
    CandidateTrajectory that was evaluated.

    Missing, stale, invalid or non-transformable social observations produce
    UNKNOWN, never VALID.

    The candidate path is transformed into the SocialZoneArray frame before
    geometric validation when the two frames differ.
"""

import math
import time
from typing import Optional, Tuple

import rclpy

from geometry_msgs.msg import PoseStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
import copy
from tf2_ros import (
    Buffer,
    TransformException,
    TransformListener,
)
import json
from std_msgs.msg import String
from g1_social_msgs.msg import (
    CandidateTrajectory,
    SocialZone,
    SocialZoneArray,
    TrajectorySocialValidation,
)

from g1_social_common.trajectory_validation import (
    TRAJECTORY_VALIDATION_UNKNOWN,
    TRAJECTORY_VALIDATION_VALID,
    TRAJECTORY_VALIDATION_CONFLICT,
)

from g1_qos_manager.diagnostics import log_qos_resolution
from g1_qos_manager.profiles import reliable_qos
from g1_qos_manager.resolver import resolve_subscription_qos


class TrajectorySocialValidator(Node):
    """
    Validate exact candidate paths against fresh social-zone observations.

    The geometric comparison is always performed in the SocialZoneArray frame.

    If the candidate path uses another frame, TF2 is used to transform every
    path pose before validation.
    """

    def __init__(self):
        super().__init__("trajectory_social_validator")

        # =====================================================================
        # PARAMETERS
        # =====================================================================

        self.declare_parameter(
            "candidate_trajectory_topic",
            "/social_navigation/candidate_trajectory",
        )

        self.declare_parameter(
            "social_zones_topic",
            "/social_perception/social_zones",
        )

        self.declare_parameter(
            "validation_topic",
            "/social_navigation/trajectory_social_validation",
        )

        # Maximum accepted age since the SocialZoneArray was received.
        self.declare_parameter(
            "social_observation_timeout_sec",
            1.0,
        )

        self.declare_parameter(
            "social_observation_ingress_max_age_sec",
            1.5,
        )

        # Distance between samples along each path segment.
        self.declare_parameter(
            "path_sampling_step_m",
            0.05,
        )

        self.declare_parameter(
            "minimum_radius_m",
            0.05,
        )

        # Maximum time allowed to resolve the required TF transformation.
        self.declare_parameter(
            "tf_lookup_timeout_sec",
            0.20,
        )

        # ---------------------------------------------------------------------
        # Resolve parameters
        # ---------------------------------------------------------------------

        self.candidate_trajectory_topic = str(
            self.get_parameter(
                "candidate_trajectory_topic"
            ).value
        ).strip()

        self.social_zones_topic = str(
            self.get_parameter(
                "social_zones_topic"
            ).value
        ).strip()

        self.validation_topic = str(
            self.get_parameter(
                "validation_topic"
            ).value
        ).strip()

        self.social_observation_timeout_sec = float(
            self.get_parameter(
                "social_observation_timeout_sec"
            ).value
        )

        self.path_sampling_step_m = float(
            self.get_parameter(
                "path_sampling_step_m"
            ).value
        )

        self.minimum_radius_m = float(
            self.get_parameter(
                "minimum_radius_m"
            ).value
        )

        self.tf_lookup_timeout_sec = float(
            self.get_parameter(
                "tf_lookup_timeout_sec"
            ).value
        )

        # =====================================================================
        # PARAMETER VALIDATION
        # =====================================================================

        if not self.candidate_trajectory_topic:
            raise ValueError(
                "candidate_trajectory_topic cannot be empty."
            )

        if not self.social_zones_topic:
            raise ValueError(
                "social_zones_topic cannot be empty."
            )

        if not self.validation_topic:
            raise ValueError(
                "validation_topic cannot be empty."
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
            not math.isfinite(
                self.path_sampling_step_m
            )
            or self.path_sampling_step_m <= 0.0
        ):
            raise ValueError(
                "path_sampling_step_m must be finite and positive."
            )

        if (
            not math.isfinite(
                self.minimum_radius_m
            )
            or self.minimum_radius_m <= 0.0
        ):
            raise ValueError(
                "minimum_radius_m must be finite and positive."
            )

        if (
            not math.isfinite(
                self.tf_lookup_timeout_sec
            )
            or self.tf_lookup_timeout_sec <= 0.0
        ):
            raise ValueError(
                "tf_lookup_timeout_sec must be finite and positive."
            )

        self.social_observation_ingress_max_age_sec = float(
            self.get_parameter(
                "social_observation_ingress_max_age_sec"
            ).value
        )

        if (
            not math.isfinite(self.social_observation_ingress_max_age_sec)
            or self.social_observation_ingress_max_age_sec <= 0.0
        ):
            raise ValueError(
                "social_observation_ingress_max_age_sec must be "
                "finite and positive."
            )

        self.minimum_radius_m = max(
            0.001,
            self.minimum_radius_m,
        )

        # =====================================================================
        # RUNTIME STATE
        # =====================================================================

        self.latest_social_zones: Optional[
            SocialZoneArray
        ] = None

        # Monotonic receive time is deliberately stored separately from the
        # ROS message timestamp. This protects against a stalled input stream.
        self.latest_social_zones_received_at: Optional[
            float
        ] = None

        # Diagnostics.
        self.candidates_received = 0
        self.valid_results = 0
        self.conflict_results = 0
        self.unknown_results = 0

        self.tf_transforms_requested = 0
        self.tf_transforms_succeeded = 0
        self.tf_transforms_failed = 0
        self.last_social_source_ns = None
        self.latest_social_coverage_complete = False

        self.individual_source_session = ""
        self.individual_source_stamp_ns = 0

        self.individual_source_fresh = False
        self.individual_source_complete = False

        self.individual_received_at = None
        self.individual_records = []

        self.retired_individual_sessions = set()
        
        self.social_source_session = ""
        self.retired_social_source_sessions = set()

        # =====================================================================
        # TF2
        # =====================================================================

        self.tf_buffer = Buffer()

        self.tf_listener = TransformListener(
            self.tf_buffer,
            self,
            spin_thread=False,
        )

        # =====================================================================
        # QOS
        # =====================================================================

        candidate_qos = resolve_subscription_qos(
            self,
            self.candidate_trajectory_topic,
            fallback_depth=10,
            prefer_sensor_data=False,
        )

        social_zones_qos = resolve_subscription_qos(
            self,
            self.social_zones_topic,
            fallback_depth=10,
            prefer_sensor_data=False,
        )

        output_qos = reliable_qos(
            depth=10,
        )

        log_qos_resolution(
            self,
            self.candidate_trajectory_topic,
            candidate_qos,
        )

        log_qos_resolution(
            self,
            self.social_zones_topic,
            social_zones_qos,
        )

        # =====================================================================
        # ROS INTERFACES
        # =====================================================================

        self.social_zones_subscription = (
            self.create_subscription(
                SocialZoneArray,
                self.social_zones_topic,
                self._social_zones_callback,
                social_zones_qos,
            )
        )
        self.individual_situation_subscription = (
            self.create_subscription(
                String,
                "/social_perception/individual_social_situation",
                self._individual_situation_callback,
                social_zones_qos,
            )
        )

        self.candidate_subscription = (
            self.create_subscription(
                CandidateTrajectory,
                self.candidate_trajectory_topic,
                self._candidate_trajectory_callback,
                candidate_qos,
            )
        )

        self.revalidation_subscription = self.create_subscription(
            CandidateTrajectory,
            "/social_navigation/trajectory_revalidation_request",
            self._revalidation_request_callback,
            candidate_qos,
        )

        self.validation_publisher = (
            self.create_publisher(
                TrajectorySocialValidation,
                self.validation_topic,
                output_qos,
            )
        )

        self.diagnostics_timer = self.create_timer(
            5.0,
            self._log_diagnostics,
        )

        # =====================================================================
        # INITIALIZATION REPORT
        # =====================================================================

        self.get_logger().info("")
        self.get_logger().info("=" * 84)

        self.get_logger().info(
            "  TRAJECTORY SOCIAL VALIDATOR - INITIALIZATION"
        )

        self.get_logger().info("=" * 84)

        self.get_logger().info(
            "  Candidate topic     : "
            f"{self.candidate_trajectory_topic}"
        )

        self.get_logger().info(
            "  Social zones topic  : "
            f"{self.social_zones_topic}"
        )

        self.get_logger().info(
            "  Validation topic    : "
            f"{self.validation_topic}"
        )

        self.get_logger().info(
            "  Observation timeout : "
            f"{self.social_observation_timeout_sec:.2f} s"
        )

        self.get_logger().info(
            "  Path sampling step  : "
            f"{self.path_sampling_step_m:.3f} m"
        )

        self.get_logger().info(
            "  TF lookup timeout   : "
            f"{self.tf_lookup_timeout_sec:.3f} s"
        )

        self.get_logger().info("=" * 84)

    # =========================================================================
    # COLLECTION
    # =========================================================================

    def _social_zones_callback(self, msg):
        """
        Conserver les zones valides de l'observation courante.

        Une couverture partielle permet de détecter un conflit,
        mais ne permet jamais de produire VALID.
        """
        try:
            session = str(msg.tracking_session_id).strip()
            stamp = msg.header.stamp

            source_ns = (
                int(stamp.sec) * 1_000_000_000
                + int(stamp.nanosec)
            )

            age = (
                self.get_clock().now().nanoseconds
                - source_ns
            ) / 1e9

            valid_header = bool(
                session
                and session not in self.retired_social_source_sessions
                and str(msg.header.frame_id).strip()
                and stamp.sec >= 0
                and 0 <= stamp.nanosec < 1_000_000_000
                and source_ns > 0
                and (
                    -0.10
                    <= age
                    <= self.social_observation_ingress_max_age_sec
                )
            )

            if not valid_header:
                return

            if (
                session == self.social_source_session
                and self.last_social_source_ns is not None
                and source_ns <= self.last_social_source_ns
            ):
                return

            # Une identité dupliquée est ambiguë :
            # aucune de ses occurrences ne sera conservée.
            identity_counts = {}

            for zone in msg.zones:
                person_id = str(zone.person_id).strip()

                identity_counts[person_id] = (
                    identity_counts.get(person_id, 0) + 1
                )

            valid_zones = []

            for zone in msg.zones:
                try:
                    person_id = str(zone.person_id).strip()
                    zone_stamp = zone.observation_stamp

                    zone_ns = (
                        int(zone_stamp.sec) * 1_000_000_000
                        + int(zone_stamp.nanosec)
                    )

                    valid = bool(
                        person_id
                        and identity_counts[person_id] == 1
                        and zone.tracking_session_id == session
                        and zone_stamp.sec >= 0
                        and (
                            0 <= zone_stamp.nanosec
                            < 1_000_000_000
                        )
                        and zone_ns == source_ns
                        and zone.tracking_state == "TRACKED"
                        and self._is_valid_zone(zone)
                        and math.isfinite(float(zone.center.z))
                        and math.isfinite(float(zone.person_speed))
                        and zone.person_speed >= 0.0
                    )

                except (
                    ValueError,
                    TypeError,
                    AttributeError,
                    OverflowError,
                ):
                    valid = False

                if valid:
                    valid_zones.append(zone)

            complete = bool(
                msg.source_observation_complete
                and msg.source_person_count >= 0
                and msg.source_person_count == len(msg.zones)
                and len(valid_zones) == len(msg.zones)
            )

            # Ne pas modifier le message reçu.
            current = copy.deepcopy(msg)
            current.zones = copy.deepcopy(valid_zones)

            # Le filtrage ne transforme jamais une source
            # incomplète en observation complète.
            current.source_observation_complete = complete

        except (
            ValueError,
            TypeError,
            AttributeError,
            OverflowError,
        ):
            return

        if (
            self.social_source_session
            and session != self.social_source_session
        ):
            self.retired_social_source_sessions.add(
                self.social_source_session
            )

        self.social_source_session = session
        self.last_social_source_ns = source_ns

        self.latest_social_zones = current
        self.latest_social_zones_received_at = time.monotonic()
        self.latest_social_coverage_complete = complete

        if not complete:
            self.get_logger().warning(
                "[SOCIAL VALIDATION] Partial source | "
                f"session={session} | "
                f"source_ns={source_ns} | "
                f"usable_zones={len(valid_zones)} | "
                "valid_result_allowed=False",
                throttle_duration_sec=2.0,
            )

    # =========================================================================
    # CANDIDATE VALIDATION
    # =========================================================================

    def _revalidation_request_callback(self, msg):
        """
        Revalidate an existing immutable candidate path.

        This uses the exact same conservative social-geometry
        validation as ordinary candidate generation.

        The request does not start Nav2 planning.
        It never authorizes navigation.
        """

        path_id = str(msg.path_id or "").strip()
        request_id = str(msg.request_id or "").strip()
        goal_id = str(msg.navigation_goal_id or "").strip()

        if not all((path_id, request_id, goal_id)):
            self.get_logger().warning(
                "[PATH REVALIDATION] Missing candidate identity."
            )
            return

        if (
            not msg.path.header.frame_id
            or not msg.path.poses
            or not self._is_valid_path(msg.path)
        ):
            self._publish_validation(
                path_id,
                TRAJECTORY_VALIDATION_UNKNOWN,
                "",
                "INVALID_REVALIDATION_CANDIDATE",
            )
            return

        self.get_logger().info(
            "[PATH REVALIDATION] "
            f"goal={goal_id} | "
            f"request={request_id} | "
            f"path={path_id} | "
            "using_current_social_observation=True"
        )

        # Do not duplicate or weaken the existing geometry checks.
        self._candidate_trajectory_callback(msg)

    def _individual_situation_callback(self, msg):
        """Accepter un instantané individuel horodaté."""

        try:
            data = json.loads(msg.data)

            if (
                not isinstance(data, dict)
                or data.get("version") != 1
            ):
                return

            session = str(data["source_session"]).strip()
            stamp_ns = int(data["source_stamp_ns"])

            fresh = data["source_fresh"]
            complete = data["source_coverage_complete"]
            rows = data["participants"]

            if (
                not session
                or stamp_ns <= 0
                or type(fresh) is not bool
                or type(complete) is not bool
                or not isinstance(rows, list)
                or len(rows) > 512
            ):
                return

            records = []
            keys = set()

            for row in rows:
                key = row["key"]

                if (
                    not isinstance(key, list)
                    or len(key) != 3
                    or not isinstance(key[0], str)
                    or not key[0].strip()
                    or not isinstance(key[1], str)
                    or not key[1].strip()
                    or type(key[2]) is not int
                    or key[2] < 0
                    or type(row["unresolved"]) is not bool
                    or type(row["observed_now"]) is not bool
                    or type(row["retired"]) is not bool
                ):
                    return

                identity = tuple(key)

                if identity in keys:
                    return

                keys.add(identity)

                records.append(
                    (
                        identity,
                        row["unresolved"],
                        row["observed_now"],
                        str(row["observation_status"]),
                        row["retired"],
                    )
                )

        except (
            ValueError,
            TypeError,
            KeyError,
            AttributeError,
        ):
            return

        if session in self.retired_individual_sessions:
            return

        if session == self.individual_source_session:
            if stamp_ns < self.individual_source_stamp_ns:
                return

        elif self.individual_source_session:
            self.retired_individual_sessions.add(
                self.individual_source_session
            )

        self.individual_source_session = session
        self.individual_source_stamp_ns = stamp_ns

        self.individual_source_fresh = fresh
        self.individual_source_complete = complete

        self.individual_records = records
        self.individual_received_at = time.monotonic()


    def _individual_validation_blocker(self, zones):
        """
        Retourner la raison qui interdit une validation positive.

        Cette méthode n'autorise aucun mouvement.
        """

        if self.individual_received_at is None:
            return "NO_INDIVIDUAL_EPISODE_SNAPSHOT"

        source = zones.header.stamp

        source_ns = (
            int(source.sec) * 1_000_000_000
            + int(source.nanosec)
        )

        if (
            not self.individual_source_fresh
            or not self.individual_source_complete
            or self.individual_source_session
            != str(zones.tracking_session_id)
            or self.individual_source_stamp_ns != source_ns
        ):
            return "INDIVIDUAL_SNAPSHOT_SOURCE_MISMATCH"

        age = (
            self.get_clock().now().nanoseconds - source_ns
        ) / 1e9

        receipt_age = (
            time.monotonic() - self.individual_received_at
        )

        if (
            not -0.10 <= age
            <= self.social_observation_timeout_sec
            or not 0.0 <= receipt_age
            <= self.social_observation_timeout_sec
        ):
            return "INDIVIDUAL_SNAPSHOT_STALE"

        current_zone_ids = {
            str(zone.person_id)
            for zone in zones.zones
        }

        for (
            key,
            unresolved,
            observed,
            status,
            retired,
        ) in self.individual_records:

            if not unresolved:
                continue

            if (
                retired
                or not observed
                or status != "OBSERVED"
            ):
                return "UNRESOLVED_UNOBSERVED_PERSON"

            if (
                key[0] != self.individual_source_session
                or key[1] not in current_zone_ids
            ):
                return "UNRESOLVED_PERSON_ZONE_MISSING"

        return ""

    def _candidate_trajectory_callback(self, msg):
        """
        Valider une trajectoire exacte contre les zones courantes.

        CONFLICT est possible avec une couverture partielle.
        VALID exige une couverture complète et fraîche.
        """
        self.candidates_received += 1

        path_id = str(msg.path_id or "").strip()

        def publish(result, reason, person_id=""):
            self._publish_validation(
                path_id,
                result,
                person_id,
                reason,
            )

        if not path_id:
            publish(
                TRAJECTORY_VALIDATION_UNKNOWN,
                "MISSING_PATH_ID",
            )
            return

        path = msg.path

        if (
            not path.header.frame_id.strip()
            or not path.poses
            or not self._is_valid_path(path)
        ):
            publish(
                TRAJECTORY_VALIDATION_UNKNOWN,
                "INVALID_PATH",
            )
            return

        zones = self.latest_social_zones
        received = self.latest_social_zones_received_at

        if zones is None or received is None:
            publish(
                TRAJECTORY_VALIDATION_UNKNOWN,
                "NO_SOCIAL_OBSERVATION",
            )
            return

        stamp = zones.header.stamp

        source_ns = (
            stamp.sec * 1_000_000_000
            + stamp.nanosec
        )

        source_age = (
            self.get_clock().now().nanoseconds
            - source_ns
        ) / 1e9

        receipt_age = time.monotonic() - received

        if (
            source_ns <= 0
            or not 0 <= stamp.nanosec < 1_000_000_000
            or not (
                -0.1
                <= source_age
                <= self.social_observation_timeout_sec
            )
            or not (
                0.0
                <= receipt_age
                <= self.social_observation_timeout_sec
            )
            or not zones.header.frame_id.strip()
        ):
            self.get_logger().warning(
                "[TRAJECTORY VALIDATION FRESHNESS] "
                f"path={path_id} | "
                f"source_age={source_age:.3f}s | "
                f"receipt_age={receipt_age:.3f}s | "
                f"frame={zones.header.frame_id!r} | "
                f"timeout={self.social_observation_timeout_sec:.3f}s"
            )

            publish(
                TRAJECTORY_VALIDATION_UNKNOWN,
                "INVALID_OR_STALE_SOCIAL_OBSERVATION",
            )
            return

        seen = set()

        for zone in zones.zones:
            person_id = str(zone.person_id).strip()

            if (
                not person_id
                or person_id in seen
                or zone.tracking_state != "TRACKED"
                or not self._is_valid_zone(zone)
                or not math.isfinite(float(zone.center.z))
            ):
                publish(
                    TRAJECTORY_VALIDATION_UNKNOWN,
                    "INVALID_SOCIAL_ZONE",
                )
                return

            seen.add(person_id)

        aligned = path

        if path.header.frame_id != zones.header.frame_id:
            aligned = self._transform_path_to_frame(
                path=path,
                target_frame=zones.header.frame_id,
            )

            if aligned is None:
                publish(
                    TRAJECTORY_VALIDATION_UNKNOWN,
                    "TF_UNAVAILABLE",
                )
                return

        try:
            conflict = self._find_path_conflict(
                aligned,
                zones.zones,
            )

        except (ArithmeticError, ValueError):
            publish(
                TRAJECTORY_VALIDATION_UNKNOWN,
                "NUMERICAL_GEOMETRY_FAILURE",
            )
            return

        # Une observation ne doit pas devenir une preuve fraîche
        # simplement parce que le calcul vient de se terminer.
        if (
            time.monotonic() - received
            > self.social_observation_timeout_sec
            or (
                self.get_clock().now().nanoseconds
                - source_ns
            ) / 1e9
            > self.social_observation_timeout_sec
        ):
            publish(
                TRAJECTORY_VALIDATION_UNKNOWN,
                "OBSERVATION_EXPIRED_DURING_VALIDATION",
            )
            return

        blocker = self._individual_validation_blocker(zones)

        if conflict is not None:
            zone, _, _ = conflict

            publish(
                TRAJECTORY_VALIDATION_CONFLICT,
                "PATH_INTERSECTS_SOCIAL_ZONE",
                zone.person_id,
            )

        elif not self.latest_social_coverage_complete:
            publish(
                TRAJECTORY_VALIDATION_UNKNOWN,
                "INCOMPLETE_SOCIAL_COVERAGE",
            )

        elif blocker:
            publish(
                TRAJECTORY_VALIDATION_UNKNOWN,
                blocker,
            )

        else:
            publish(
                TRAJECTORY_VALIDATION_VALID,
                (
                    "PATH_CLEAR_OF_SOCIAL_ZONES"
                    if zones.zones
                    else "NO_TRACKED_SOCIAL_ZONE"
                ),
            )

    # =========================================================================
    # TF2
    # =========================================================================

    def _transform_path_to_frame(
        self,
        *,
        path,
        target_frame: str
    ):
        """
        Transform one candidate path into target_frame
        for social validation.

        The transform is resolved at the timestamp of the
        latest accepted social observation so path geometry
        and SocialZone geometry refer to the same
        temporal snapshot.

        The input CandidateTrajectory is never
        modified in place.

        Failure to obtain or apply the required
        transform returns None.

        The caller converts that result into UNKNOWN.
        """
        source_frame = str(
            path.header.frame_id
        ).strip()

        target_frame = str(
            target_frame
        ).strip()

        if not source_frame or not target_frame:
            return None

        if source_frame == target_frame:
            return copy.deepcopy(path)

        if self.latest_social_zones is None:
            return None

        self.tf_transforms_requested += 1

        social_stamp = (
            self.latest_social_zones.header.stamp
        )

        try:
            if (
                int(social_stamp.sec) == 0
                and int(social_stamp.nanosec) == 0
            ):
                self.tf_transforms_failed += 1

                self.get_logger().warning(
                    "[TRAJECTORY VALIDATION] "
                    "Social observation has zero timestamp | "
                    f"SOURCE={source_frame} | "
                    f"TARGET={target_frame}"
                )
                return None

            transform_time = Time.from_msg(
                social_stamp
            )

            transform = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                transform_time,
                timeout=Duration(seconds=0.0)
            )

        except TransformException as exc:
            self.tf_transforms_failed += 1

            self.get_logger().warning(
                "[TRAJECTORY VALIDATION] TF lookup failed | "
                f"SOURCE={source_frame} | "
                f"TARGET={target_frame} | "
                f"STAMP={social_stamp.sec}."
                f"{social_stamp.nanosec:09d} | "
                f"ERROR={exc}"
            )
            return None

        except Exception as exc:
            self.tf_transforms_failed += 1

            self.get_logger().warning(
                "[TRAJECTORY VALIDATION] "
                "Unexpected TF lookup failure | "
                f"SOURCE={source_frame} | "
                f"TARGET={target_frame} | "
                f"ERROR={exc}"
            )
            return None

        transformed_path = type(path)()

        transformed_path.header = copy.deepcopy(
            path.header
        )

        transformed_path.header.frame_id = (
            target_frame
        )

        transformed_path.header.stamp = copy.deepcopy(
            social_stamp
        )

        transformed_path.poses = []

        for pose_stamped in path.poses:
            transformed_pose = (
                self._transform_pose_with_transform(
                    pose_stamped=pose_stamped,
                    transform=transform,
                    source_frame=source_frame,
                    target_frame=target_frame,
                    output_stamp=social_stamp
                )
            )

            if transformed_pose is None:
                self.tf_transforms_failed += 1
                return None

            transformed_path.poses.append(
                transformed_pose
            )

        if not transformed_path.poses:
            self.tf_transforms_failed += 1
            return None

        if not self._is_valid_path(transformed_path):
            self.tf_transforms_failed += 1

            self.get_logger().warning(
                "[TRAJECTORY VALIDATION] "
                "TF produced invalid path geometry | "
                f"SOURCE={source_frame} | "
                f"TARGET={target_frame}"
            )
            return None

        self.tf_transforms_succeeded += 1

        self.get_logger().debug(
            "[TRAJECTORY VALIDATION] "
            "Path transformed for social validation | "
            f"SOURCE={source_frame} | "
            f"TARGET={target_frame} | "
            f"STAMP={social_stamp.sec}."
            f"{social_stamp.nanosec:09d} | "
            f"POSES={len(transformed_path.poses)}"
        )

        return transformed_path
    def _transform_pose_with_transform(
        self,
        *,
        pose_stamped: PoseStamped,
        transform,
        source_frame: str,
        target_frame: str,
        output_stamp,
    ) -> Optional[PoseStamped]:
        """
        Apply one already-resolved rigid transform to one PoseStamped.

        The input pose is never modified in place.

        output_stamp is the common temporal reference used for the social
        observation and the path transformation.
        """

        try:

            translation = (
                transform.transform.translation
            )

            rotation = (
                transform.transform.rotation
            )

            position = (
                pose_stamped.pose.position
            )

            orientation = (
                pose_stamped.pose.orientation
            )

            # =============================================================
            # ROTATE POSITION
            # =============================================================

            rx, ry, rz = (
                self._rotate_vector_by_quaternion(
                    x=float(position.x),
                    y=float(position.y),
                    z=float(position.z),
                    qx=float(rotation.x),
                    qy=float(rotation.y),
                    qz=float(rotation.z),
                    qw=float(rotation.w),
                )
            )

            output = PoseStamped()

            # Detached header: never mutate the candidate path.
            output.header = copy.deepcopy(
                pose_stamped.header
            )

            output.header.frame_id = (
                target_frame
            )

            output.header.stamp = copy.deepcopy(
                output_stamp
            )

            output.pose.position.x = (
                rx + float(translation.x)
            )

            output.pose.position.y = (
                ry + float(translation.y)
            )

            output.pose.position.z = (
                rz + float(translation.z)
            )

            # =============================================================
            # ORIENTATION
            # =============================================================

            (
                output.pose.orientation.x,
                output.pose.orientation.y,
                output.pose.orientation.z,
                output.pose.orientation.w,
            ) = self._quaternion_multiply(
                (
                    float(rotation.x),
                    float(rotation.y),
                    float(rotation.z),
                    float(rotation.w),
                ),
                (
                    float(orientation.x),
                    float(orientation.y),
                    float(orientation.z),
                    float(orientation.w),
                ),
            )

            values = (
                output.pose.position.x,
                output.pose.position.y,
                output.pose.position.z,
                output.pose.orientation.x,
                output.pose.orientation.y,
                output.pose.orientation.z,
                output.pose.orientation.w,
            )

            if not all(
                math.isfinite(
                    float(value)
                )
                for value in values
            ):

                self.get_logger().warning(
                    "[TRAJECTORY VALIDATION] "
                    "TF produced non-finite pose | "
                    f"SOURCE={source_frame} | "
                    f"TARGET={target_frame}"
                )

                return None

            return output

        except Exception as exc:

            self.get_logger().warning(
                "[TRAJECTORY VALIDATION] "
                "Unable to transform path pose | "
                f"SOURCE={source_frame} | "
                f"TARGET={target_frame} | "
                f"ERROR={exc}"
            )

            return None
        
    @staticmethod
    def _rotate_vector_by_quaternion(
        *,
        x: float,
        y: float,
        z: float,
        qx: float,
        qy: float,
        qz: float,
        qw: float,
    ) -> Tuple[float, float, float]:
        """
        Rotate a 3-D vector by quaternion q.

        Uses the equivalent rotation-matrix formulation.
        """

        norm = math.sqrt(
            qx * qx
            + qy * qy
            + qz * qz
            + qw * qw
        )

        if (
            not math.isfinite(norm)
            or norm <= 1e-12
        ):
            raise ValueError(
                "Invalid TF quaternion."
            )

        qx /= norm
        qy /= norm
        qz /= norm
        qw /= norm

        r00 = (
            1.0
            - 2.0 * (
                qy * qy
                + qz * qz
            )
        )

        r01 = (
            2.0 * (
                qx * qy
                - qz * qw
            )
        )

        r02 = (
            2.0 * (
                qx * qz
                + qy * qw
            )
        )

        r10 = (
            2.0 * (
                qx * qy
                + qz * qw
            )
        )

        r11 = (
            1.0
            - 2.0 * (
                qx * qx
                + qz * qz
            )
        )

        r12 = (
            2.0 * (
                qy * qz
                - qx * qw
            )
        )

        r20 = (
            2.0 * (
                qx * qz
                - qy * qw
            )
        )

        r21 = (
            2.0 * (
                qy * qz
                + qx * qw
            )
        )

        r22 = (
            1.0
            - 2.0 * (
                qx * qx
                + qy * qy
            )
        )

        return (
            r00 * x + r01 * y + r02 * z,
            r10 * x + r11 * y + r12 * z,
            r20 * x + r21 * y + r22 * z,
        )

    @staticmethod
    def _quaternion_multiply(
        q1,
        q2,
    ) -> Tuple[
        float,
        float,
        float,
        float,
    ]:
        """Return quaternion product q1 * q2."""

        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2

        x = (
            w1 * x2
            + x1 * w2
            + y1 * z2
            - z1 * y2
        )

        y = (
            w1 * y2
            - x1 * z2
            + y1 * w2
            + z1 * x2
        )

        z = (
            w1 * z2
            + x1 * y2
            - y1 * x2
            + z1 * w2
        )

        w = (
            w1 * w2
            - x1 * x2
            - y1 * y2
            - z1 * z2
        )

        norm = math.sqrt(
            x * x
            + y * y
            + z * z
            + w * w
        )

        if (
            not math.isfinite(norm)
            or norm <= 1e-12
        ):
            raise ValueError(
                "Invalid transformed orientation quaternion."
            )

        return (
            x / norm,
            y / norm,
            z / norm,
            w / norm,
        )

    # =========================================================================
    # PATH / SOCIAL GEOMETRY
    # =========================================================================

    def _find_path_conflict(self, path, zones):
        """Intersect whole planar line segments with oriented ellipses."""
        points = [
            (
                float(p.pose.position.x),
                float(p.pose.position.y)
            )
            for p in path.poses
        ]

        segments = (
            list(zip(points, points[1:]))
            if len(points) > 1
            else [(points[0], points[0])]
        )

        for (x0, y0), (x1, y1) in segments:
            for zone in zones:
                a = max(
                    float(zone.radius_longitudinal),
                    self.minimum_radius_m
                )

                b = max(
                    float(zone.radius_lateral),
                    self.minimum_radius_m
                )

                c, s = (
                    math.cos(zone.heading),
                    math.sin(zone.heading)
                )

                dx = x0 - zone.center.x
                dy = y0 - zone.center.y

                u = (
                    c * dx + s * dy
                ) / a

                v = (
                    -s * dx + c * dy
                ) / b

                ex = x1 - x0
                ey = y1 - y0

                du = (
                    c * ex + s * ey
                ) / a

                dv = (
                    -s * ex + c * ey
                ) / b

                denominator = (
                    du * du + dv * dv
                )

                if not all(
                    math.isfinite(t)
                    for t in (
                        u,
                        v,
                        du,
                        dv,
                        denominator
                    )
                ):
                    raise ValueError(
                        "Non-finite ellipse intersection"
                    )

                t = (
                    0.0
                    if denominator == 0
                    else max(
                        0.0,
                        min(
                            1.0,
                            -(u * du + v * dv)
                            / denominator
                        )
                    )
                )

                if (
                    math.hypot(
                        u + t * du,
                        v + t * dv
                    )
                    <= 1.0 + 1e-12
                ):
                    return (
                        zone,
                        x0 + t * ex,
                        y0 + t * ey
                    )

        return None
    def _first_containing_zone(
        self,
        x: float,
        y: float,
        zones,
    ) -> Optional[SocialZone]:
        """Return the first social zone containing the given path point."""

        for zone in zones:

            if self._point_inside_social_zone(
                x=x,
                y=y,
                zone=zone,
            ):
                return zone

        return None

    def _point_inside_social_zone(
        self,
        x: float,
        y: float,
        zone: SocialZone,
    ) -> bool:
        """
        Test one point against an oriented social ellipse.

        SocialZone convention:

            a = radius_longitudinal
            b = radius_lateral
            theta = heading

        A point is inside the social zone when:

            (local_x / a)^2 + (local_y / b)^2 <= 1
        """

        center_x = float(
            zone.center.x
        )

        center_y = float(
            zone.center.y
        )

        theta = float(
            zone.heading
        )

        a = max(
            float(
                zone.radius_longitudinal
            ),
            self.minimum_radius_m,
        )

        b = max(
            float(
                zone.radius_lateral
            ),
            self.minimum_radius_m,
        )

        dx = (
            x - center_x
        )

        dy = (
            y - center_y
        )

        cos_theta = math.cos(
            theta
        )

        sin_theta = math.sin(
            theta
        )

        # Rotate world displacement by -theta into the person's local frame.
        local_x = (
            cos_theta * dx
            + sin_theta * dy
        )

        local_y = (
            -sin_theta * dx
            + cos_theta * dy
        )

        normalized_distance = (
            (local_x / a) ** 2
            + (local_y / b) ** 2
        )

        return (
            normalized_distance <= 1.0
        )

    # =========================================================================
    # INPUT VALIDATION
    # =========================================================================

    @staticmethod
    def _is_valid_path(path):
        if (
            not path.poses
            or not path.header.frame_id.strip()
        ):
            return False

        for pose in path.poses:
            if pose.header.frame_id.strip() not in {
                "",
                path.header.frame_id
            }:
                return False

            p = pose.pose.position
            q = pose.pose.orientation

            if not all(
                math.isfinite(float(v))
                for v in (
                    p.x,
                    p.y,
                    p.z,
                    q.x,
                    q.y,
                    q.z,
                    q.w
                )
            ):
                return False

            if math.hypot(
                q.x,
                q.y,
                q.z,
                q.w
            ) <= 1e-12:
                return False

        return True
    
    @staticmethod
    def _is_valid_zone(
        zone: SocialZone,
    ) -> bool:
        """Validate social-zone geometry."""

        values = (
            zone.center.x,
            zone.center.y,
            zone.radius_longitudinal,
            zone.radius_lateral,
            zone.person_speed,
            zone.heading,
        )

        if not all(
            math.isfinite(
                float(value)
            )
            for value in values
        ):
            return False

        if float(
            zone.radius_longitudinal
        ) <= 0.0:
            return False

        if float(
            zone.radius_lateral
        ) <= 0.0:
            return False

        return True

    # =========================================================================
    # RESULT
    # =========================================================================

    def _publish_validation(
        self,
        path_id: str,
        validation: str,
        person_id: str,
        reason: str
    ) -> None:
        """Publish validation for one exact path identifier."""
        result = TrajectorySocialValidation()

        result.header.stamp = (
            self.get_clock().now().to_msg()
        )

        if (
            validation in {
                TRAJECTORY_VALIDATION_VALID,
                TRAJECTORY_VALIDATION_CONFLICT
            }
            and self.latest_social_zones is not None
        ):
            result.header.stamp = copy.deepcopy(
                self.latest_social_zones.header.stamp
            )

        if self.latest_social_zones is not None:
            result.header.frame_id = str(
                self.latest_social_zones.header.frame_id
            )

        result.path_id = path_id
        result.validation = validation
        result.person_id = person_id
        result.reason = reason

        self.validation_publisher.publish(result)

        if validation == TRAJECTORY_VALIDATION_VALID:
            self.valid_results += 1

        elif validation == TRAJECTORY_VALIDATION_CONFLICT:
            self.conflict_results += 1

        else:
            self.unknown_results += 1

        self.get_logger().info(
            "[TRAJECTORY VALIDATION] "
            f"PATH={path_id or 'UNKNOWN'} | "
            f"RESULT={validation} | "
            f"PERSON={person_id or 'NONE'} | "
            f"REASON={reason}"
        )
    # =========================================================================
    # DIAGNOSTICS
    # =========================================================================

    def _log_diagnostics(
        self,
    ) -> None:
        """Report trajectory-validator health."""

        if (
            self.latest_social_zones_received_at
            is None
        ):
            observation_age = "NONE"

        else:
            observation_age = (
                f"{time.monotonic() - self.latest_social_zones_received_at:.2f}"
            )

        self.get_logger().info(
            "[TRAJECTORY SOCIAL VALIDATOR] "
            f"CANDIDATES={self.candidates_received} | "
            f"VALID={self.valid_results} | "
            f"CONFLICT={self.conflict_results} | "
            f"UNKNOWN={self.unknown_results} | "
            f"TF_REQUESTED={self.tf_transforms_requested} | "
            f"TF_OK={self.tf_transforms_succeeded} | "
            f"TF_FAILED={self.tf_transforms_failed} | "
            f"SOCIAL_OBSERVATION_AGE={observation_age}"
        )


def main(args=None):
    """Run the trajectory social validator."""

    rclpy.init(
        args=args
    )

    node = (
        TrajectorySocialValidator()
    )

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