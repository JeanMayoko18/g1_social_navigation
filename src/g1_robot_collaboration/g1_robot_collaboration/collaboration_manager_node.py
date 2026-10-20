#!/usr/bin/env python3

"""
Central collaboration FSM for socially aware navigation.

Responsibilities:
    - Supervise social episodes and navigation holds.
    - Coordinate explanation delivery and human dialogue.
    - Request candidate trajectories through NavigationGoalGate.
    - Authorize only the exact, freshly validated candidate.

Important:
    - Observation loss is never geometric clearance.
    - Episode retirement never authorizes movement.
    - CLEAR requires a fresh trajectory before RESUME.
    - Human permission requires semantic confirmation.
    - CandidateTrajectory owns candidate identity.
    - Validation may arrive before the candidate and is buffered.
    - WAIT retains the public goal; STOP cancels it.
"""

import math
import time
import uuid
import json
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
    CandidateTrajectory,
    CollaborativeDecision,
    DialogueRequest,
    ExplanationContext,
    ExplanationText,
    HumanIntent,
    NavigationGoalContext,
    NavigationState,
    RobotMotionState,
    SocialNavigationDecision,
    SpeechCancelRequest,
    SpeechStatus,
    TrajectoryGenerationRequest,
    TrajectorySocialValidation,
)

from g1_social_common.social_navigation_events import (
    SOCIAL_EVENT_CLEAR,
    SOCIAL_EVENT_INTRUSION_CONFIRMED,
    SOCIAL_EVENT_OBSERVATION_LOST,
    SOCIAL_EVENT_OBSERVATION_REACQUIRED,
    normalize_social_navigation_event,
)

from g1_social_common.collaboration_states import (
    STATE_ANNOUNCING_DETECTION,
    STATE_ASKING,
    STATE_CONFIRMING_HUMAN_INTENT,
    STATE_NAVIGATING,
    STATE_OBSERVING,
    STATE_PROPOSING_ALTERNATIVE,
    STATE_REPLANNING,
    STATE_RESUMING,
    STATE_SOCIAL_RESOLUTION,
    STATE_STOPPING_FOR_SOCIAL_CONFLICT,
    STATE_VALIDATING_TRAJECTORY,
    STATE_WAITING,
    STATE_WAITING_FOR_CONFIRMATION,
    STATE_WAITING_FOR_HUMAN,
)

from g1_social_common.navigation_states import (
    NAVIGATION_NAVIGATING,
    normalize_navigation_state,
)

from g1_social_common.dialogue_texts import (
    get_dialogue_text,
    normalize_dialogue_language,
)

from g1_social_common.trajectory_validation import (
    TRAJECTORY_VALIDATION_CONFLICT,
    TRAJECTORY_VALIDATION_UNKNOWN,
    TRAJECTORY_VALIDATION_VALID,
    is_trajectory_validation_result,
    normalize_trajectory_validation,
)


class CollaborationManager(Node):
    """Coordinate social perception, communication and navigation."""

    VALID_DIRECTIONS = {"NONE", "LEFT", "RIGHT"}
    VALID_INTERACTIONS = {"none", "dialogue"}

    def __init__(self):
        super().__init__("collaboration_manager")

        # ------------------------------------------------------------
        # Parameters
        # ------------------------------------------------------------

        self.declare_parameter("interaction", "none")
        self.declare_parameter("dialogue_response_timeout_sec", 120.0)
        self.declare_parameter("dialogue_max_retries", 1)
        self.declare_parameter("dialogue_speech_max_retries", 1)
        self.declare_parameter("social_decision_max_age_sec", 1.5)
        self.declare_parameter("trajectory_validation_max_age_sec", 1.0)
        self.declare_parameter("language", "en")
        self.declare_parameter("conflict_persistence_sec", 2.0)
        self.declare_parameter("initial_speech_start_timeout_sec", 15.0)
        self.declare_parameter("speech_completion_timeout_sec", 30.0)

        self.declare_parameter(
            "navigation_goal_context_topic",
            "/social_navigation/navigation_goal_context",
        )
        self.declare_parameter(
            "social_decision_topic",
            "/social_navigation/decision",
        )
        self.declare_parameter(
            "human_intent_topic",
            "/hri/human_intent",
        )
        self.declare_parameter(
            "navigation_state_topic",
            "/social_navigation/navigation_state",
        )
        self.declare_parameter(
            "dialogue_request_topic",
            "/hri/dialogue_request",
        )
        self.declare_parameter(
            "collaborative_decision_topic",
            "/social_navigation/collaborative_decision",
        )
        self.declare_parameter(
            "robot_motion_state_topic",
            "/social_navigation/robot_motion_state",
        )
        self.declare_parameter(
            "trajectory_validation_topic",
            "/social_navigation/trajectory_social_validation",
        )
        self.declare_parameter(
            "candidate_trajectory_topic",
            "/social_navigation/candidate_trajectory",
        )
        self.declare_parameter(
            "trajectory_generation_request_topic",
            "/social_navigation/trajectory_generation_request",
        )
        self.declare_parameter(
            "explanation_context_topic",
            "/explainability/context",
        )
        self.declare_parameter(
            "explanation_text_topic",
            "/explainability/text",
        )
        self.declare_parameter(
            "speech_status_topic",
            "/explainability/speech_status",
        )

        self.declare_parameter(
            "human_initiative_timeout_sec",
            8.0,
        )
        self.declare_parameter(
            "route_search_max_attempts",
            3,
        )
        self.declare_parameter(
            "social_resolution_timeout_sec",
            180.0,
        )

        self.human_initiative_timeout_sec = float(
            self.get_parameter(
                "human_initiative_timeout_sec"
            ).value
        )
        self.route_search_max_attempts = int(
            self.get_parameter(
                "route_search_max_attempts"
            ).value
        )
        self.social_resolution_timeout_sec = float(
            self.get_parameter(
                "social_resolution_timeout_sec"
            ).value
        )

        for name, value in {
            "human_initiative_timeout_sec":
                self.human_initiative_timeout_sec,
            "social_resolution_timeout_sec":
                self.social_resolution_timeout_sec,
        }.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"{name} must be finite and positive."
                )

        if self.route_search_max_attempts < 1:
            raise ValueError(
                "route_search_max_attempts must be >= 1."
            )

        self._reset_route_search()

        self.interaction = str(
            self.get_parameter("interaction").value
        ).strip().lower()

        self.dialogue_response_timeout_sec = float(
            self.get_parameter("dialogue_response_timeout_sec").value
        )
        self.dialogue_max_retries = int(
            self.get_parameter("dialogue_max_retries").value
        )
        self.dialogue_speech_max_retries = int(
            self.get_parameter("dialogue_speech_max_retries").value
        )
        self.social_decision_max_age_sec = float(
            self.get_parameter("social_decision_max_age_sec").value
        )
        self.trajectory_validation_max_age_sec = float(
            self.get_parameter("trajectory_validation_max_age_sec").value
        )
        self.conflict_persistence_sec = float(
            self.get_parameter("conflict_persistence_sec").value
        )
        self.initial_speech_start_timeout_sec = float(
            self.get_parameter("initial_speech_start_timeout_sec").value
        )
        self.speech_completion_timeout_sec = float(
            self.get_parameter("speech_completion_timeout_sec").value
        )

        self.navigation_goal_context_topic = str(
            self.get_parameter("navigation_goal_context_topic").value
        )
        self.social_decision_topic = str(
            self.get_parameter("social_decision_topic").value
        )
        self.human_intent_topic = str(
            self.get_parameter("human_intent_topic").value
        )
        self.navigation_state_topic = str(
            self.get_parameter("navigation_state_topic").value
        )
        self.dialogue_request_topic = str(
            self.get_parameter("dialogue_request_topic").value
        )
        self.collaborative_decision_topic = str(
            self.get_parameter("collaborative_decision_topic").value
        )
        self.robot_motion_state_topic = str(
            self.get_parameter("robot_motion_state_topic").value
        )
        self.trajectory_validation_topic = str(
            self.get_parameter("trajectory_validation_topic").value
        )
        self.candidate_trajectory_topic = str(
            self.get_parameter("candidate_trajectory_topic").value
        )
        self.trajectory_generation_request_topic = str(
            self.get_parameter("trajectory_generation_request_topic").value
        )
        self.explanation_context_topic = str(
            self.get_parameter("explanation_context_topic").value
        ).strip()
        self.explanation_text_topic = str(
            self.get_parameter("explanation_text_topic").value
        ).strip()
        self.speech_status_topic = str(
            self.get_parameter("speech_status_topic").value
        ).strip()

        configured_language = str(
            self.get_parameter("language").value or "en"
        ).strip().lower()

        self.language = normalize_dialogue_language(configured_language)

        if self.language != configured_language:
            self.get_logger().warning(
                f"Unsupported language {configured_language!r}; "
                f"falling back to {self.language!r}."
            )

        if self.dialogue_max_retries < 0:
            raise ValueError("dialogue_max_retries must be >= 0.")

        if self.dialogue_speech_max_retries < 0:
            raise ValueError("dialogue_speech_max_retries must be >= 0.")

        self._validate_parameters()

        # ------------------------------------------------------------
        # FSM and social episode
        # ------------------------------------------------------------

        self.fsm_state = STATE_NAVIGATING
        self.state_entered_at = time.monotonic()

        self.latest_social_decision = None
        self.social_state_session = ""
        self.social_state_episode = ""
        self.social_state_sequence = 0
        self.retired_social_sessions = set()
        #self.retired_social_episode_ids = set()

        self.active_decision_id = ""
        self.active_person_id = ""

        self.latest_observation_health = "UNKNOWN"
        self.latest_observation_health_at = None
        self.empty_observation_since = None
        self.health_session = ""
        self.health_stamp_ns = None

        self.retired_recovery_goal_id = ""
        self.retired_recovery_path_id = ""

        self.retired_episode_retired_at = None
        self.retired_recovery_attempt_count = 0
        self.retired_recovery_last_attempt_at = 0.0
        self.retired_recovery_retry_delay_sec = 3.0
        self.retired_recovery_exhaustion_logged = False

        self.initial_start_authorized_goal_id = ""

        self.human_proximity_active = False
        self.social_observation_available = True
        self.waiting_for_observation_reacquisition = False
        self.waiting_for_human_initiative = False
        self.observation_loss_held_goal_id = ""

        self.social_loss_timeout_sec = 15.0
        self.social_loss_started_at = None
        self.social_loss_expired = False
        self.pending_retired_episode = None
        self.social_episode_retired = False

        # Identity of the navigation goal at the opening of
        # the current social episode. Empty means pre-navigation.
        self.social_episode_origin_goal_id = ""

        self.conflict_observed_since = None
        self.persistent_conflict_handled = False

        # ------------------------------------------------------------
        # Navigation and physical stop
        # ------------------------------------------------------------

        self.latest_navigation_state = None
        self.navigation_state = "UNKNOWN"
        self.navigation_active = False
        self.active_navigation_goal_id = ""
        self.active_navigation_goal_pose = None

        # Independent deadline for an initial start or navigation hold.
        self.resolution_watch_goal_id = ""
        self.resolution_watch_started_at = None

        # Source identity and timing of bounded reassessments.
        self.reassessment_last_at = 0.0
        self.reassessment_last_stamp_ns = None

        self.nav2_cancel_requested = False
        self.nav2_cancel_confirmed = False
        self.pending_stop_hold_request_id = ""
        self.navigation_was_active_at_intrusion = False
        self.last_navigation_interruption_key = None

        self.stable_stop_confirmed = False
        self.stable_stop_source = ""
        self.motion_state_received_at = None
        self.motion_state_stamp_sec = None
        self.stop_request_stamp_sec = None

        # ------------------------------------------------------------
        # Trajectory pipeline
        # ------------------------------------------------------------

        self.candidate_path_id = ""
        self.candidate_navigation_goal_id = ""
        self.trajectory_validation = TRAJECTORY_VALIDATION_UNKNOWN
        self.validated_path_id = ""
        self.validation_received_at = None
        self.validation_source_stamp_sec = None

        self.pending_generation_request_id = ""
        self.pending_replan_directional_constraint = "NONE"
        self.generation_requested_at = None

        # Results received before the authoritative candidate.
        self.early_validations = {}

        # ------------------------------------------------------------
        # Human interaction
        # ------------------------------------------------------------

        self.active_interaction_id = ""
        self.active_dialogue_request_type = ""
        self.active_alternative_direction = "NONE"

        self.pending_human_intent = "UNKNOWN"
        self.pending_human_direction = "NONE"
        self.confirmed_human_intent = "UNKNOWN"
        self.confirmed_human_direction = "NONE"

        self.dialogue_started_at = None
        self.dialogue_retry_count = 0
        self.dialogue_speech_retry_count = 0
        self.dialogue_audio_recovery_pending = False
        self.audio_recovery_interaction_id = ""
        self.audio_recovery_request_id = ""

        

        # ------------------------------------------------------------
        # Announcement and speech transaction
        # ------------------------------------------------------------

        self.announced_observe = False
        self.finished_announcement_attempts = set()
        self.retired_social_episode_ids = {}
        self.announcement_context_key = None
        self.deferred_social_goal_id = ""

        self.announced_navigation_goal_id = ""
        self.current_announcement_action = ""
        self.announcement_decision_id = ""
        self.announcement_social_decision = None
        self.clear_pending_after_announcement = False
        self.uncertain_generation_request_id = ""

        self.pending_speech_decision_id = ""
        self.pending_explanation_ready = False
        self.pending_speech_started = False
        self.pending_speech_started_at = None
        self.pending_speech_requested_at = None
        self.pending_speech_completion_action = ""
        self.pending_autonomous_readaptation_reason = ""

        self.initial_speech_retry_count = 0
        self.max_initial_speech_retries = 1


        self.gate_ready_goal_id = ""
        self.pending_gate_command = None
        self.latest_start_request_id = ""
        self.initial_start_attempt_count = 0
        self.initial_start_max_attempts = 3
        self.initial_start_next_attempt_at = 0.0

        # Clôture d'un dialogue HRI avant toute reprise dépendante.
        self.pending_hri_close = None
        self.hri_close_fault_reported = False
        self.deferred_clear_replan_goal_id = ""

        # Terminaison publique supervisée.
        self.pending_terminal_stop = None
        self.terminal_stop_fault_reported = False

        # Identité conservée lors d'une récupération audio.
        self.dialogue_audio_recovery_pending = False
        self.audio_recovery_interaction_id = ""
        self.audio_recovery_request_id = ""

        # Récupération de l'épisode social retiré.
        self.retired_episode_retired_at = None
        self.retired_recovery_attempt_count = 0
        self.retired_recovery_last_attempt_at = 0.0
        self.retired_recovery_retry_delay_sec = 3.0
        self.retired_recovery_exhaustion_logged = False

        # ------------------------------------------------------------
        # Subscriptions
        # ------------------------------------------------------------

        self.hri_dialogue_control_pub = self.create_publisher(
            String,
            "/hri/dialogue_control",
            20,
        )

        self.hri_dialogue_control_result_sub = self.create_subscription(
            String,
            "/hri/dialogue_control_result",
            self._hri_dialogue_control_result_callback,
            20,
        )
        self.social_decision_sub = self.create_subscription(
            SocialNavigationDecision,
            self.social_decision_topic,
            self._social_decision_callback,
            10,
        )

        self.gate_command_result_sub = self.create_subscription(
            String,
            "/social_navigation/gate_command_result",
            self._gate_command_result_callback,
            20,
        )

        self.social_episode_retirement_sub = self.create_subscription(
            String,
            "/social_navigation/episode_retirement",
            self._social_episode_retirement_callback,
            QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            ),
        )
        self.navigation_goal_context_sub = self.create_subscription(
            NavigationGoalContext,
            self.navigation_goal_context_topic,
            self._navigation_goal_context_callback,
            10,
        )
        self.human_intent_sub = self.create_subscription(
            HumanIntent,
            self.human_intent_topic,
            self._human_intent_callback,
            10,
        )
        self.navigation_state_sub = self.create_subscription(
            NavigationState,
            self.navigation_state_topic,
            self._navigation_state_callback,
            10,
        )
        self.robot_motion_state_sub = self.create_subscription(
            RobotMotionState,
            self.robot_motion_state_topic,
            self._robot_motion_state_callback,
            10,
        )
        self.candidate_trajectory_sub = self.create_subscription(
            CandidateTrajectory,
            self.candidate_trajectory_topic,
            self._candidate_trajectory_callback,
            10,
        )
        self.trajectory_validation_sub = self.create_subscription(
            TrajectorySocialValidation,
            self.trajectory_validation_topic,
            self._trajectory_validation_callback,
            10,
        )
        self.explanation_text_sub = self.create_subscription(
            ExplanationText,
            self.explanation_text_topic,
            self._explanation_text_callback,
            10,
        )
        self.speech_status_sub = self.create_subscription(
            SpeechStatus,
            self.speech_status_topic,
            self._speech_status_callback,
            10,
        )

        self.trajectory_generation_result_sub = self.create_subscription(
            String,
            "/social_navigation/trajectory_generation_result",
            self._trajectory_generation_result_callback,
            10,
        )

        self.observation_health_sub = self.create_subscription(
            String,
            "/social_navigation/observation_health",
            self._observation_health_callback,
            QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=10,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
            ),
        )

        # ------------------------------------------------------------
        # Publishers and timer
        # ------------------------------------------------------------

        self.speech_cancel_pub = self.create_publisher(
            SpeechCancelRequest,
            "/explainability/speech_cancel_request",
            20,
        )
        self.dialogue_request_pub = self.create_publisher(
            DialogueRequest,
            self.dialogue_request_topic,
            10,
        )
        self.collaborative_decision_pub = self.create_publisher(
            CollaborativeDecision,
            self.collaborative_decision_topic,
            10,
        )
        self.trajectory_generation_request_pub = self.create_publisher(
            TrajectoryGenerationRequest,
            self.trajectory_generation_request_topic,
            10,
        )
        self.explanation_context_pub = self.create_publisher(
            ExplanationContext,
            self.explanation_context_topic,
            10,
        )

        self.timer = self.create_timer(0.1, self._timer_callback)

        self.get_logger().info(
            "[COLLABORATION] Ready | "
            f"interaction={self.interaction} | "
            f"language={self.language} | "
            "speech_completion=REQUIRED"
        )

    # ================================================================
    # Parameters and common state
    # ================================================================

    def _validate_parameters(self):
        if self.interaction not in self.VALID_INTERACTIONS:
            raise ValueError(
                "interaction must be 'none' or 'dialogue', "
                f"got {self.interaction!r}"
            )

        positive_parameters = {
            "dialogue_response_timeout_sec":
                self.dialogue_response_timeout_sec,
            "social_decision_max_age_sec":
                self.social_decision_max_age_sec,
            "trajectory_validation_max_age_sec":
                self.trajectory_validation_max_age_sec,
            "conflict_persistence_sec":
                self.conflict_persistence_sec,
            "initial_speech_start_timeout_sec":
                self.initial_speech_start_timeout_sec,
            "speech_completion_timeout_sec":
                self.speech_completion_timeout_sec,
        }

        for name, value in positive_parameters.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")

    def _request_passage_assistance(self):
        """Request optional assistance once after multiple geometric failures."""

        scope = self._social_scope_key()
        nav = self.latest_navigation_state

        if (
            not all(scope)
            or not self.human_proximity_active
            or self.conflict_evidence_scope != scope
            or len(self.conflicting_candidate_ids) < 2
            or self.assistance_requested_scope == scope
            or self.social_resolution_aborted
            or self.navigation_active
            or nav is None
            or str(nav.navigation_goal_id or "").strip()
                != self.active_navigation_goal_id
            or not self._message_is_recent(nav, 1.0)
            or self.navigation_state not in {
                "WAITING",
                "SOCIAL_HOLD",
                "REPLANNING",
            }
            or not self._route_observation_is_fresh()
            or self.pending_terminal_stop is not None
            or self.pending_hri_close is not None
            or self.pending_gate_command is not None
            or self.pending_speech_decision_id
            or self.pending_dialogue_request_id
            or self.dialogue_audio_recovery_pending
            or self.pending_generation_request_id
            or self.uncertain_generation_request_id
            or self.announcement_context_key is not None
            or not self.stable_stop_confirmed
            or self.motion_state_received_at is None
            or (
                time.monotonic()
                - self.motion_state_received_at
            ) > 0.5
            or self.motion_state_stamp_sec is None
        ):
            return False

        motion_age = (
            self.get_clock().now().nanoseconds / 1e9
            - self.motion_state_stamp_sec
        )

        if not -0.1 <= motion_age <= 0.5:
            return False

        if (
            self.interaction == "dialogue"
            and self.dialogue_outcome not in {
                "ALLOW_CONFIRMED",
                "DENY_CONFIRMED",
                "NO_RESPONSE",
                "NOT_UNDERSTOOD",
            }
        ):
            return False

        context = ExplanationContext()
        context.header.stamp = self.get_clock().now().to_msg()

        context.decision_id = (
            f"PASSAGE_ASSISTANCE_{uuid.uuid4().hex}"
        )
        context.navigation_goal_id = (
            self.active_navigation_goal_id
        )
        context.navigation_state = self.navigation_state
        context.navigation_active = False
        context.social_conflict_active = True

        context.person_id = self.active_person_id
        context.language = self.language

        context.action = "WAIT"
        context.reason_code = "PASSAGE_ASSISTANCE"
        context.requires_replan = False

        context.situation_summary = (
            "Several checked candidate trajectories conflicted "
            "with observed social zones."
        )

        context.robot_intention = (
            "Remain stopped, ask for optional assistance, "
            "then reassess using fresh observations."
        )

        context.explanation_goal = (
            "Politely ask whether the person could move slightly "
            "if convenient. Do not blame the person or claim "
            "that every possible route is blocked."
        )

        # Reserve before publication to prevent duplicate requests.
        self.assistance_requested_scope = scope

        self._invalidate_trajectory_validation(
            reason="PASSAGE_ASSISTANCE"
        )
        self._publish_wait(
            reason="PASSAGE_ASSISTANCE"
        )

        self.route_waiting_for_change = False

        self._clear_pending_speech_transaction()

        self.pending_speech_decision_id = context.decision_id
        self.pending_speech_completion_action = (
            "PASSAGE_ASSISTANCE"
        )
        self.pending_speech_requested_at = time.monotonic()

        self._transition_to(
            STATE_ANNOUNCING_DETECTION,
            reason="PASSAGE_ASSISTANCE",
        )

        self.explanation_context_pub.publish(context)
        return True
    
    def _transition_to(self, new_state, *, reason):
        if new_state == self.fsm_state:
            return

        previous = self.fsm_state
        self.fsm_state = new_state
        self.state_entered_at = time.monotonic()

        self.get_logger().info(
            f"[COLLABORATION FSM] {previous} -> {new_state} | "
            f"reason={reason}"
        )

    def _reset_route_search(self):
        """Reset policy and semantic memory for a new context."""

        self.route_attempt_count = 0
        self.route_tried_directions = set()
        self.route_generation_reason = "REPLAN"

        self.route_waiting_for_change = False
        self.route_blocked_signature = None

        self.initial_empty_retry_goal_id = ""
        self.initial_empty_retry_after_health_ns = None
        self.initial_empty_retry_not_before = 0.0

        self.dialogue_outcome = "NONE"
        self.dialogue_resolution_complete = False
        self.forbidden_human_direction = "NONE"

        self.pending_dialogue_request_id = ""
        self.pending_dialogue_scope = None

        # Semantic permission for one exact social context.
        self.authorized_episode_key = None

        # Context already handled by actual validated execution.
        # This is not a reusable execution authorization.
        self.executed_social_scope = None

        self.conflict_evidence_scope = None
        self.conflicting_candidate_ids = set()
        self.assistance_requested_scope = None

        self.social_resolution_started_at = None
        self.social_resolution_aborted = False

    def _route_observation_is_fresh(self):
        if (not self.social_observation_available or self.social_loss_expired
                or self.social_episode_retired or self.waiting_for_observation_reacquisition):
            return False

        if not self.human_proximity_active:
            # CLEAR can be supported by a visible person outside the release zone.
            # OCCUPIED does not imply that the trajectory intersects a social zone.
            return self._fresh_observation_health_available()

        return (
            self.latest_social_decision is not None
            and self._message_is_recent(
                self.latest_social_decision,
                self.social_decision_max_age_sec
            )
        )
    
    def _route_scene_signature(self):
        """
        Detect a meaningful change in the reported social context.

        This only triggers reassessment. It never validates a path.
        """
        decision = self.latest_social_decision

        if decision is None:
            return None

        position = decision.person_position

        values = (
            float(position.x),
            float(position.y),
            float(decision.social_radius),
        )

        if not all(math.isfinite(value) for value in values):
            return None

        return (
            str(decision.person_id),
            bool(self.human_proximity_active),
            round(values[0] / 0.20),
            round(values[1] / 0.20),
            round(values[2] / 0.10),
        )

    def _hold_route_search(self, reason, *, reconsider=None):
        """
        Hold route execution and schedule reassessment.

        Repeated identical holds must not keep creating new
        Gate transactions while the robot is already held.

        A new hazard or an active execution still requires
        the normal WAIT command.
        """
        reason = str(
            reason or "ROUTE_SEARCH_HELD"
        ).strip()

        recoverable_reasons = {
            "ALTERNATIVE_REQUIRES_FRESH_OBSERVATION",
            "REPLAN_REQUIRES_FRESH_SOCIAL_EVIDENCE",
            "SOCIAL_ROUTE_ATTEMPTS_EXHAUSTED",
            "NO_PERMITTED_SOCIAL_ALTERNATIVE",
            "STALE_TRAJECTORY_VALIDATION",
            "TRAJECTORY_VALIDATION_UNKNOWN",
            "SOCIAL_CLEARANCE_NOT_CONFIRMED",
            "SOCIAL_RECOVERY_EVIDENCE_REQUIRED",
            "SOCIAL_OBSERVATION_LOST",
            "DIRECTIONAL_WAYPOINT_UNAVAILABLE",
        }

        if reconsider is None:
            reconsider = reason in recoverable_reasons

        goal_id = str(
            self.active_navigation_goal_id or ""
        ).strip()

        nav = self.latest_navigation_state

        gate_already_held = bool(
            goal_id
            and nav is not None
            and str(nav.navigation_goal_id or "").strip()
                == goal_id
            and self._message_is_recent(nav, 1.0)
            and not self.navigation_active
            and self.navigation_state in {
                "WAITING",
                "SOCIAL_HOLD",
                "REPLANNING",
            }
        )

        already_waiting = (
            self.fsm_state == STATE_WAITING
        )

        previously_scheduled = bool(
            self.route_waiting_for_change
        )

        previous_reason = getattr(
            self,
            "_last_route_hold_reason",
            None,
        )

        previous_goal = getattr(
            self,
            "_last_route_hold_goal_id",
            None,
        )

        same_hold = bool(
            gate_already_held
            and already_waiting
            and previous_reason == reason
            and previous_goal == goal_id
        )

        self.route_waiting_for_change = bool(reconsider)

        if not same_hold:
            self.route_blocked_signature = (
                self._route_scene_signature()
            )

        if reconsider and not previously_scheduled:
            self.reassessment_last_at = time.monotonic()
            self.reassessment_last_stamp_ns = (
                self.health_stamp_ns
            )

        # Invalidate old trajectory authorization regardless
        # of whether an additional Gate command is needed.
        self._invalidate_trajectory_validation(
            reason=reason
        )

        self._last_route_hold_reason = reason
        self._last_route_hold_goal_id = goal_id

        if not same_hold:
            self._publish_wait(reason=reason)

            self.get_logger().info(
                "[ROUTE WAIT] "
                f"reason={reason} | "
                f"reassessment_enabled={bool(reconsider)} | "
                f"goal={goal_id}"
            )
        else:
            self.get_logger().debug(
                "[ROUTE WAIT] "
                f"Duplicate hold suppressed | reason={reason}"
            )

        self._transition_to(
            STATE_WAITING,
            reason=reason,
        )

    def _request_next_social_alternative(self):
        """Search permitted alternatives, then optionally request assistance."""

        if (
            not self.active_navigation_goal_id
            or self.social_resolution_aborted
        ):
            return

        if not self._route_observation_is_fresh():
            self._hold_route_search(
                reason="ALTERNATIVE_REQUIRES_FRESH_OBSERVATION",
                reconsider=True,
            )
            return

        if (
            self.interaction == "dialogue"
            and self.human_proximity_active
        ):
            if self.dialogue_outcome == "NONE":
                self._start_human_initiative_window()
                return

            if self.dialogue_outcome == "PENDING":
                return

            if self.dialogue_outcome not in {
                "ALLOW_CONFIRMED",
                "DENY_CONFIRMED",
                "NO_RESPONSE",
                "NOT_UNDERSTOOD",
            }:
                self._hold_route_search(
                    reason="DIALOGUE_DELIVERY_RECOVERY_REQUIRED",
                    reconsider=False,
                )
                return

        permitted = ("LEFT", "RIGHT")

        if (
            self.interaction == "dialogue"
            and self.dialogue_outcome == "ALLOW_CONFIRMED"
            and self.confirmed_human_direction in {
                "LEFT",
                "RIGHT",
            }
        ):
            # Preserve the confirmed side.
            # Do not repeatedly negotiate its opposite.
            permitted = (self.confirmed_human_direction,)

        available = [
            side
            for side in permitted
            if (
                side not in self.route_tried_directions
                and side != self.forbidden_human_direction
            )
        ]

        exhausted = (
            self.route_attempt_count
            >= self.route_search_max_attempts
        )

        if exhausted or not available:
            if self._request_passage_assistance():
                return

            self._hold_route_search(
                reason=(
                    "SOCIAL_ROUTE_ATTEMPTS_EXHAUSTED"
                    if exhausted
                    else "NO_PERMITTED_SOCIAL_ALTERNATIVE"
                ),
                reconsider=True,
            )
            return

        self._enter_replanning(
            reason="SOCIAL_ALTERNATIVE",
            directional_constraint=available[0],
        )

    def _clear_pending_speech_transaction(self):
        self.pending_speech_decision_id = ""
        self.pending_speech_completion_action = ""
        self.pending_explanation_ready = False
        self.pending_speech_started = False
        self.pending_speech_requested_at = None
        self.pending_speech_started_at = None
        self.pending_autonomous_readaptation_reason = ""

    def _hold_after_speech_uncertainty(
        self,
        *,
        decision_id,
        reason,
        request_cancel=False,
    ):
        decision_id = str(decision_id or "").strip()
        reason = str(reason or "SPEECH_UNCONFIRMED").strip()

        self._clear_pending_speech_transaction()

        if request_cancel and decision_id:
            cancel_msg = SpeechCancelRequest()
            cancel_msg.decision_id = decision_id

            try:
                self.speech_cancel_pub.publish(cancel_msg)
                self.get_logger().warning(
                    "[SPEECH BARRIER] Cancellation requested | "
                    f"decision_id={decision_id}"
                )
            except Exception as exc:
                self.get_logger().error(
                    "[SPEECH BARRIER] Cancellation publish failed | "
                    f"decision_id={decision_id} | error={exc}"
                )

        self._invalidate_trajectory_validation(reason=reason)
        self.dialogue_started_at = None
        self.waiting_for_human_initiative = False

        self._publish_wait(reason=reason)
        self._transition_to(STATE_WAITING, reason=reason)

    # ================================================================
    # Observation health
    # ================================================================

    def _observation_health_callback(self, msg):
        parts = str(msg.data or "").split(":")

        if len(parts) != 5:
            return

        status, session, _, sec, nanosec = parts

        if status not in {"EMPTY", "OCCUPIED"} or not session:
            return

        if session in self.retired_social_sessions:
            return

        if self.social_state_session and session != self.social_state_session:
            return

        try:
            sec = int(sec)
            nanosec = int(nanosec)
        except ValueError:
            return

        stamp_ns = sec * 1_000_000_000 + nanosec
        age = (self.get_clock().now().nanoseconds - stamp_ns) / 1e9

        if (
            sec < 0
            or not 0 <= nanosec < 1_000_000_000
            or stamp_ns <= 0
        ):
            return

        if not -0.1 <= age <= self.social_decision_max_age_sec:
            return

        now = time.monotonic()

        if session != self.health_session:
            self.health_session = session
            self.health_stamp_ns = None
            self.empty_observation_since = None

        if (
            self.health_stamp_ns is not None
            and stamp_ns <= self.health_stamp_ns
        ):
            return

        interrupted = (
            self.latest_observation_health_at is None
            or (
                now - self.latest_observation_health_at
                > self.social_decision_max_age_sec
            )
            or (
                self.health_stamp_ns is not None
                and (
                    (stamp_ns - self.health_stamp_ns) / 1e9
                    > self.social_decision_max_age_sec
                )
            )
        )

        if status == "EMPTY":
            if (
                interrupted
                or self.latest_observation_health != "EMPTY"
                or self.empty_observation_since is None
            ):
                self.empty_observation_since = now
        else:
            self.empty_observation_since = None

        self.latest_observation_health = status
        self.latest_observation_health_at = now
        self.health_stamp_ns = stamp_ns

    def _fresh_empty_observation_available(self, *, minimum_duration_sec):
        """EMPTY describes the current stream, not clearance of a lost person."""
        now = time.monotonic()

        if self.latest_observation_health != "EMPTY":
            return False

        if (
            self.latest_observation_health_at is None
            or self.empty_observation_since is None
            or self.health_stamp_ns is None
        ):
            return False

        if (
            self.social_state_session
            and self.health_session != self.social_state_session
        ):
            return False

        source_age = (
            self.get_clock().now().nanoseconds - self.health_stamp_ns
        ) / 1e9

        return (
            -0.1 <= source_age <= self.social_decision_max_age_sec
            and (
                now - self.latest_observation_health_at
                <= self.social_decision_max_age_sec
            )
            and now - self.empty_observation_since >= minimum_duration_sec
        )

    def _resolution_budget_available(self, now):
        """Bound each interval from goal reception or navigation hold to execution."""
        goal_id = str(
            self.active_navigation_goal_id or ""
        ).strip()

        if not goal_id:
            self.resolution_watch_goal_id = ""
            self.resolution_watch_started_at = None
            return False

        if goal_id != self.resolution_watch_goal_id:
            self.resolution_watch_goal_id = goal_id
            self.resolution_watch_started_at = now
            self.reassessment_last_at = 0.0
            self.reassessment_last_stamp_ns = None

        if (
            self.pending_terminal_stop is not None
            or self.social_resolution_aborted
        ):
            return False

        state = self.latest_navigation_state

        execution_confirmed = (
            self.fsm_state == STATE_NAVIGATING
            and self.navigation_active
            and self.navigation_state == NAVIGATION_NAVIGATING
            and state is not None
            and str(state.navigation_goal_id or "").strip() == goal_id
            and self._message_is_recent(state, 1.0)
        )

        if execution_confirmed:
            # The next navigation hold starts a new resolution interval.
            self.resolution_watch_started_at = None
            return True

        if self.resolution_watch_started_at is None:
            self.resolution_watch_started_at = now

        elapsed = now - self.resolution_watch_started_at

        if elapsed >= self.social_resolution_timeout_sec:
            self.get_logger().warning(
                "[SOCIAL RESOLUTION] Resolution remains pending | "
                f"goal={goal_id} | "
                f"elapsed={elapsed:.1f}s | "
                "public_goal_retained=True | "
                "reassessment_enabled=True | "
                "navigation_authorized=False"
            )

            # Restart the diagnostic interval only.
            # Do not reset dialogue, safety barriers or route attempts.
            self.resolution_watch_started_at = now

        return True

    def _initial_social_start_allowed(self):
        """
        Authorize an initial goal without requiring a social
        perception message.

        Never bypass an active or unresolved social episode.
        """
        if (
            self.human_proximity_active
            or self.social_episode_retired
            or self.social_loss_expired
            or self.waiting_for_observation_reacquisition
            or not self.social_observation_available
        ):
            return False

        if (
            self.nav2_cancel_requested
            or self.pending_stop_hold_request_id
            or self.social_resolution_aborted
        ):
            return False

        return True

    def _initial_social_start_allowed_old(self):
        """Allow initial assessment of a fresh scene without a known conflict."""
        if (
            self.human_proximity_active
            or self.social_episode_retired
            or self.social_loss_expired
            or self.waiting_for_observation_reacquisition
            or not self.social_observation_available
        ):
            return False

        #return self._fresh_observation_health_available()
        return True

    def _evaluate_initial_goal_start_old(self):
        goal_id = str(self.active_navigation_goal_id or "").strip()

        if not goal_id or self.active_navigation_goal_pose is None:
            return

        if (
            self.gate_ready_goal_id != goal_id
            or self.navigation_active
        ):
            return

        if (
            self.social_resolution_aborted
            or self.pending_terminal_stop is not None
            or self.pending_hri_close is not None
            or self.pending_gate_command is not None
            or self.pending_speech_decision_id
            or self.dialogue_audio_recovery_pending
        ):
            return

        if (
            self.fsm_state != STATE_NAVIGATING
            or not self._initial_social_start_allowed()
        ):
            return

        self._enter_replanning(
            reason="INITIAL_EXACT_PATH_ASSESSMENT",
            directional_constraint="NONE"
        )
    
    def _evaluate_initial_goal_start(self):
        goal_id = str(
            self.active_navigation_goal_id or ""
        ).strip()

        if not goal_id or self.active_navigation_goal_pose is None:
            return

        if (
            self.gate_ready_goal_id != goal_id
            or self.navigation_active
            or self.initial_start_authorized_goal_id == goal_id
            or self.initial_start_attempt_count >=
                self.initial_start_max_attempts
            or time.monotonic() <
                self.initial_start_next_attempt_at
        ):
            return

        if (
            self.social_resolution_aborted
            or self.pending_terminal_stop is not None
            or self.pending_hri_close is not None
            or self.pending_gate_command is not None
            or self.pending_speech_decision_id
            or self.dialogue_audio_recovery_pending
        ):
            return

        if (
            self.fsm_state != STATE_NAVIGATING
            or not self._initial_social_start_allowed()
        ):
            return


        request_id = self._publish_collaborative_decision(
            action="START",
            outcome="INITIAL_NAVIGATION",
            human_intent="UNKNOWN",
            directional_constraint="NONE",
            path_id="",
            reason="INITIAL_NAVIGATION_NO_SOCIAL_CONFLICT",
        )

        if not request_id:
            return

        self.latest_start_request_id = request_id
        self.initial_start_attempt_count += 1

        self.pending_gate_command = {
            "action": "START",
            "goal_id": goal_id,
            "request_id": request_id,
            "path_id": "",
            "sent_at": time.monotonic(),
            "status": "SENT",
        }

        self.get_logger().info(
            "[INITIAL START] "
            f"goal={goal_id} | "
            f"request={request_id} | "
            "social_detection_required=False"
        )
        
    # ================================================================
    # Episode retirement
    # ================================================================

    def _social_episode_retirement_callback(self, msg):
        parts = str(msg.data or "").strip().split(":")

        if len(parts) != 3 or parts[0] != "RETIRED":
            self.get_logger().warning(
                "[SOCIAL SYNC] Invalid retirement notification."
            )
            return

        _, session, episode = parts

        if not session or not episode:
            return

        if (
            session != self.social_state_session
            or episode != self.social_state_episode
        ):
            return

        self.pending_retired_episode = (session, episode)
        self._apply_pending_social_retirement()

    def _apply_pending_social_retirement(self):
        identity = (
            self.social_state_session,
            self.social_state_episode,
        )

        if self.pending_retired_episode != identity:
            return

        if (
            not self.human_proximity_active
            or self.social_observation_available
            or not self.social_loss_expired
        ):
            return

        if self.social_episode_retired:
            return

        self.social_episode_retired = True

        self.retired_episode_retired_at = time.monotonic()
        self.retired_recovery_attempt_count = 0
        self.retired_recovery_last_attempt_at = 0.0
        self.retired_recovery_exhaustion_logged = False

        # The old geometry no longer defines the new search.
        # Do not reset the overall social-resolution deadline.
        self.route_attempt_count = 0
        self.route_tried_directions.clear()
        self.route_waiting_for_change = False
        self.route_blocked_signature = None

        self.pending_retired_episode = None
        self.retired_recovery_goal_id = ""
        self.retired_recovery_path_id = ""

        self._invalidate_trajectory_validation(
            reason="PRODUCER_EPISODE_RETIRED"
        )

        self.get_logger().warning(
            "[SOCIAL SYNC] "
            f"RETIRED_EPISODE={identity[0]}:{identity[1]} | "
            "GEOMETRIC_CLEAR=False | NAVIGATION_AUTHORIZED=False"
        )

    def _evaluate_idle_episode_retirement(self):

        # Preserve ownership of an unfinished HRI/audio transaction.
        if (
            self.pending_hri_close is not None
            or self.dialogue_audio_recovery_pending
        ):
            return
        # Never retire an idle episode while a public navigation
        # goal is being held.
        if self.active_navigation_goal_id:
            return

        # No automatic retirement during an unfinished real stop.
        if self.fsm_state == STATE_STOPPING_FOR_SOCIAL_CONFLICT:
            return

        # The producer must explicitly retire the old episode.
        if not self.social_episode_retired:
            return

        if not self.social_loss_expired:
            return

        # Fresh empty observation is required to conclude the
        # previous idle interaction locally.
        if not self._fresh_empty_observation_available(
            minimum_duration_sec=2.0
        ):
            return

        if self.fsm_state == STATE_ANNOUNCING_DETECTION:
            return

        self.get_logger().info(
            "[SOCIAL EPISODE] Retiring idle interaction | "
            f"session={self.social_state_session!r} | "
            f"episode={self.social_state_episode!r}"
        )

        # Cancel the old logical interaction state.
        self._reset_episode()

        self._transition_to(
            STATE_NAVIGATING,
            reason="IDLE_SOCIAL_EPISODE_RETIRED",
        )

    def _evaluate_route_resolution(self, now):
        """Supervise initial waiting and bounded reassessment after fresh input."""
        if not self._resolution_budget_available(now):
            return

        # Initial waiting can use NAVIGATING without actual execution.
        # Its deadline has already been checked above.
        if self.fsm_state == STATE_NAVIGATING:
            return

        if self.fsm_state in {
            STATE_REPLANNING,
            STATE_VALIDATING_TRAJECTORY,
        }:
            started = self.generation_requested_at

            if started is None:
                started = self.state_entered_at

            if now - started >= 45.0:
                if self.pending_generation_request_id:
                    self.uncertain_generation_request_id = (
                        self.pending_generation_request_id
                    )

                self._hold_route_search(
                    reason="TRAJECTORY_PIPELINE_TIMEOUT",
                    reconsider=False,
                )

                self.get_logger().error(
                    "[TRAJECTORY PIPELINE] Timeout. "
                    "Public goal retained; execution inhibited. "
                    "Outstanding planning termination must "
                    "be reconciled before another request."
                )

            return

        if (
            self.fsm_state != STATE_WAITING
            or not self.route_waiting_for_change
            or self.navigation_active
            or self.pending_hri_close is not None
            or self.pending_gate_command is not None
            or self.pending_speech_decision_id
            or self.dialogue_audio_recovery_pending
            or self.pending_generation_request_id
            or self.uncertain_generation_request_id
            or self.announcement_context_key is not None
        ):
            return

        # Lost and retired episodes retain their separate recovery policy.
        if (
            self.social_episode_retired
            or self.social_loss_expired
            or self.waiting_for_observation_reacquisition
        ):
            return

        if not self._route_observation_is_fresh():
            return

        if not self._fresh_observation_health_available():
            return

        stamp_ns = self.health_stamp_ns

        if stamp_ns is None:
            return

        if (
            self.reassessment_last_stamp_ns is not None
            and stamp_ns <= self.reassessment_last_stamp_ns
        ):
            return

        if now - self.reassessment_last_at < 3.0:
            return

        # These existing fields also apply to fresh OCCUPIED observations.
        # A retry after UNKNOWN must use a newer source observation.
        if (
            self.initial_empty_retry_goal_id
            == self.active_navigation_goal_id
        ):
            if now < self.initial_empty_retry_not_before:
                return

            previous_stamp = self.initial_empty_retry_after_health_ns

            if (
                previous_stamp is not None
                and stamp_ns <= previous_stamp
            ):
                return

        if self.route_attempt_count >= self.route_search_max_attempts:
            # Freshness and the waiting interval were checked above.
            self.route_attempt_count = 0
            self.route_tried_directions.clear()

            self.get_logger().info(
                "[ROUTE SEARCH] Starting another bounded search cycle | "
                "human_preferences_preserved=True | "
                "navigation_authorized=False"
            )

        self.reassessment_last_at = now
        self.reassessment_last_stamp_ns = stamp_ns

        self.initial_empty_retry_goal_id = ""
        self.initial_empty_retry_after_health_ns = None
        self.route_waiting_for_change = False

        self.get_logger().info(
            "[SOCIAL REASSESSMENT] Fresh input received | "
            f"health={self.latest_observation_health} | "
            f"source_ns={stamp_ns} | "
            "navigation_authorized=False"
        )

        if self.confirmed_human_intent == "DENY_PASSAGE":
            self._request_next_social_alternative()
            return

        # Existing mode-D guards require the applicable dialogue outcome.
        self._enter_replanning(
            reason="FRESH_SOCIAL_REASSESSMENT",
            directional_constraint=self.confirmed_human_direction,
        )

    def _evaluate_retired_episode_recovery(self):
        """Schedule repeated, evidence-based recovery assessments."""
        goal_id = str(
            self.active_navigation_goal_id or ""
        ).strip()

        if (
            not goal_id
            or self.active_navigation_goal_pose is None
            or self.social_resolution_aborted
        ):
            return

        if self.fsm_state != STATE_WAITING:
            return

        # Do not overlap unresolved navigation, speech,
        # dialogue or trajectory-generation transactions.
        if (
            self.pending_terminal_stop is not None
            or self.pending_hri_close is not None
            or self.pending_gate_command is not None
            or self.pending_speech_decision_id
            or self.announcement_context_key is not None
            or self.dialogue_audio_recovery_pending
            or self.pending_generation_request_id
            or self.uncertain_generation_request_id
            or self.navigation_active
        ):
            return

        if not (
            self.social_episode_retired
            and self.social_loss_expired
            and self.human_proximity_active
        ):
            return

        # The Gate must have completed goal reception.
        if self.gate_ready_goal_id != goal_id:
            return

        # Preserve the existing recovery evidence requirements.
        # Retirement and elapsed time are not clearance.
        if not self._retired_recovery_evidence_available(
            require_selected_goal=False
        ):
            return

        now = time.monotonic()

        if (
            now - self.retired_recovery_last_attempt_at
            < self.retired_recovery_retry_delay_sec
        ):
            return

        stamp_ns = self.health_stamp_ns

        if stamp_ns is None:
            return

        # Repeated delivery of the same observation cannot
        # trigger another recovery assessment.
        if (
            self.reassessment_last_stamp_ns is not None
            and stamp_ns <= self.reassessment_last_stamp_ns
        ):
            return

        if now - self.reassessment_last_at < 3.0:
            return

        # Renew only the search budget.
        # Preserve the episode, dialogue outcome and constraints.
        if (
            self.retired_recovery_attempt_count
            >= self.route_search_max_attempts
            or self.route_attempt_count
            >= self.route_search_max_attempts
        ):
            self.retired_recovery_attempt_count = 0
            self.route_attempt_count = 0
            self.route_tried_directions.clear()
            self.retired_recovery_exhaustion_logged = False

        self.reassessment_last_at = now
        self.reassessment_last_stamp_ns = stamp_ns

        self.retired_recovery_attempt_count += 1
        self.retired_recovery_last_attempt_at = now
        self.retired_recovery_goal_id = goal_id
        self.retired_recovery_path_id = ""

        self.get_logger().info(
            "[SOCIAL RECOVERY] Fresh assessment requested | "
            f"goal={goal_id} | "
            f"assessment={self.retired_recovery_attempt_count}/"
            f"{self.route_search_max_attempts} | "
            f"source_ns={stamp_ns} | "
            "direction=NONE | "
            "navigation_authorized=False"
        )

        self._enter_replanning(
            reason="RETIRED_EPISODE_FRESH_REASSESSMENT",
            directional_constraint="NONE",
        )
    
    # ================================================================
    # Ordered social events
    # ================================================================

    def _social_decision_callback(self, msg):
        """
        Process ordered social events.

        A retired episode may be superseded by a new,
        freshly confirmed intrusion even while navigation
        is active.

        Old events cannot reactivate retired episodes.
        """
        if not self._is_fresh_social_decision(msg):
            return

        event = normalize_social_navigation_event(msg.event)

        supported_events = {
            SOCIAL_EVENT_INTRUSION_CONFIRMED,
            SOCIAL_EVENT_OBSERVATION_REACQUIRED,
            SOCIAL_EVENT_OBSERVATION_LOST,
            SOCIAL_EVENT_CLEAR,
        }

        if event not in supported_events:
            return

        try:
            session, episode, sequence_text = (
                str(msg.decision_id).split(":")
            )
            sequence = int(sequence_text)

            if not session or not episode or sequence <= 0:
                raise ValueError("Invalid social identity.")

        except (ValueError, TypeError):
            self.get_logger().warning(
                "[SOCIAL STATE] Invalid decision identity.",
                throttle_duration_sec=2.0,
            )
            return

        if session in self.retired_social_sessions:
            return

        incoming_identity = (session, episode)

        if incoming_identity in self.retired_social_episode_ids:
            return

        current_identity = (
            self.social_state_session,
            self.social_state_episode,
        )

        new_episode = (
            incoming_identity != current_identity
        )

        if not new_episode:
            if sequence <= self.social_state_sequence:
                return

        else:
            # Only a confirmed intrusion may open an episode.
            if event != SOCIAL_EVENT_INTRUSION_CONFIRMED:
                return

            try:
                position = msg.person_position

                valid_geometry = (
                    bool(str(msg.person_id).strip())
                    and str(msg.tracking_state).upper()
                        == "TRACKED"
                    and bool(str(msg.header.frame_id).strip())
                    and all(
                        math.isfinite(float(value))
                        for value in (
                            msg.person_distance,
                            msg.social_radius,
                            position.x,
                            position.y,
                            position.z,
                        )
                    )
                    and float(msg.person_distance) >= 0.0
                    and float(msg.social_radius) > 0.0
                )

            except (ValueError, TypeError, AttributeError):
                valid_geometry = False

            if not valid_geometry:
                self.get_logger().warning(
                    "[SOCIAL STATE] New intrusion rejected: "
                    "invalid tracking or geometry."
                )
                return

            # Decision sequence is monotonic within a producer session.
            # OBSERVATION_LOST uses publication time, whereas
            # INTRUSION_CONFIRMED uses camera observation time.
            # Comparing these timestamps rejects valid new episodes.
            # Freshness is already checked at callback entry.
            if (
                session == self.social_state_session
                and sequence <= self.social_state_sequence
            ):
                self.get_logger().warning(
                    "[SOCIAL STATE] Out-of-order new episode rejected | "
                    f"session={session} | "
                    f"received_sequence={sequence} | "
                    f"last_sequence={self.social_state_sequence}"
                )
                return

            if self.human_proximity_active:
                previous_retired = bool(
                    self.social_episode_retired
                )

                previous_lost_and_expired = bool(
                    self.social_loss_expired
                    and not self.social_observation_available
                    and self.fsm_state == STATE_WAITING
                    and not self.navigation_active
                )

                if not (
                    previous_retired
                    or previous_lost_and_expired
                ):
                    self.get_logger().warning(
                        "[SOCIAL STATE] New episode rejected: "
                        "previous episode is not retired or resolved.",
                        throttle_duration_sec=2.0,
                    )
                    return

                # Never reset an unfinished stop or audio
                # transaction. The new episode must be
                # observed again after the barrier completes.
                if (
                    self.fsm_state
                        == STATE_STOPPING_FOR_SOCIAL_CONFLICT
                    or self.pending_hri_close is not None
                    or self.pending_terminal_stop is not None
                    or self.pending_speech_decision_id
                ):
                    if self.active_navigation_goal_id:
                        self._publish_wait(
                            reason="NEW_SOCIAL_EPISODE_BARRIER_PENDING"
                        )
                    return

                self.get_logger().warning(
                    "[SOCIAL STATE] New confirmed intrusion "
                    "supersedes retired episode | "
                    f"old={current_identity} | "
                    f"new={incoming_identity}"
                )

            # Preserve the identity of the replaced episode.
            # Dictionary insertion order is deterministic.
            if current_identity[0] and current_identity[1]:
                self.retired_social_episode_ids[
                    current_identity
                ] = True

            # Bound retained history without unordered set slicing.
            while len(self.retired_social_episode_ids) > 256:
                oldest_identity = next(
                    iter(self.retired_social_episode_ids)
                )

                self.retired_social_episode_ids.pop(
                    oldest_identity,
                    None,
                )

            if (
                self.social_state_session
                and session != self.social_state_session
            ):
                self.retired_social_sessions.add(
                    self.social_state_session
                )

            episode_origin_goal_id = str(
                self.active_navigation_goal_id or ""
            ).strip()

            self._reset_episode()

            self.social_state_session = session
            self.social_state_episode = episode
            self.social_state_sequence = 0

            self.social_episode_origin_goal_id = (
                episode_origin_goal_id
            )

            self._transition_to(
                STATE_NAVIGATING,
                reason="NEW_CONFIRMED_SOCIAL_EPISODE",
            )

        self.social_state_sequence = sequence
        self.latest_social_decision = msg

        if event in {
            SOCIAL_EVENT_INTRUSION_CONFIRMED,
            SOCIAL_EVENT_OBSERVATION_REACQUIRED,
        }:
            if not self.human_proximity_active:
                self._handle_intrusion_confirmed(msg)

            elif (
                not self.social_observation_available
                or self.waiting_for_observation_reacquisition
            ):
                self._handle_observation_reacquired(msg)

            else:
                self._interrupt_navigation_for_existing_social_episode()

            return

        if event == SOCIAL_EVENT_OBSERVATION_LOST:
            if (
                self.human_proximity_active
                and self.social_observation_available
            ):
                self._handle_observation_lost(msg)
            return

        if event == SOCIAL_EVENT_CLEAR:
            if self.human_proximity_active:
                self._handle_clear(msg)

    def _handle_intrusion_confirmed(self, msg):
        self.active_decision_id = msg.decision_id
        self.active_person_id = msg.person_id
        self.human_proximity_active = True
        self.social_observation_available = True

        if self.fsm_state == STATE_NAVIGATING:
            self.conflict_observed_since = None
            self.persistent_conflict_handled = False

        if self.fsm_state != STATE_NAVIGATING:
            self._interrupt_navigation_for_existing_social_episode()
            return

        self.announcement_decision_id = str(msg.decision_id or "").strip()
        self.announcement_social_decision = msg
        self.clear_pending_after_announcement = False

        self._invalidate_trajectory_validation(reason="NEW_SOCIAL_INTRUSION")
        self._ensure_interaction_id()

        self.pending_human_intent = "UNKNOWN"
        self.pending_human_direction = "NONE"
        self.confirmed_human_intent = "UNKNOWN"
        self.confirmed_human_direction = "NONE"

        if self.active_navigation_goal_id:
            self._interrupt_navigation_for_existing_social_episode()
            return

        # No public navigation goal exists to interrupt.
        self.navigation_was_active_at_intrusion = False
        self.nav2_cancel_requested = False
        self.nav2_cancel_confirmed = False
        self._enter_announcement()

    def _watch_hri_close(self, now):
        pending = self.pending_hri_close

        if pending is None:
            return

        if (
            pending.get("send_count", 1) < 3
            and now - pending.get(
                "last_sent_at",
                pending["requested_at"]
            ) >= 2.0
        ):
            self._send_pending_hri_close()
            pending["last_sent_at"] = now
            pending["send_count"] = (
                pending.get("send_count", 1) + 1
            )

        if now - pending["requested_at"] < 10.0:
            return

        if not self.hri_close_fault_reported:
            self.hri_close_fault_reported = True
            self.get_logger().error(
                "[HRI CLOSE FAULT] No terminal response | "
                f"close={pending['close_id']}"
            )

        # Retain ownership for a late CLOSED,
        # but terminate the public transaction.
        if self.active_navigation_goal_id:
            self._request_terminal_stop(
                reason="HRI_CLOSE_ACK_TIMEOUT"
            )
    def _hri_dialogue_control_result_callback(self, msg):
        try:
            data = json.loads(msg.data)
        except (TypeError, ValueError):
            return

        if not isinstance(data, dict):
            return

        if (
            data.get("version") != 1
            or data.get("producer") != "hri_manager"
        ):
            return

        pending = self.pending_hri_close

        if pending is None:
            return

        if (
            str(data.get("close_id") or "")
            != pending["close_id"]
            or str(data.get("interaction_id") or "")
            != pending["interaction_id"]
        ):
            return

        status = str(
            data.get("status") or ""
        ).upper()

        if status == "TERMINATION_UNCONFIRMED":
            self.hri_close_fault_reported = True

            self._publish_wait(
                reason="HRI_AUDIO_TERMINATION_UNCONFIRMED"
            )

            self.get_logger().error(
                "[HRI CLOSE] Audio termination unconfirmed | "
                f"close={pending['close_id']}"
            )

            # Keep ownership and accept a late terminal response.
            return

        if status != "CLOSED":
            self.get_logger().error(
                "[HRI CLOSE] Unexpected result | "
                f"status={status} | close={pending['close_id']}"
            )
            return

        self.get_logger().info(
            "[HRI CLOSE] Confirmed | "
            f"close={pending['close_id']}"
        )

        closed_interaction_id = pending["interaction_id"]

        self.pending_hri_close = None
        self.hri_close_fault_reported = False
        self.dialogue_audio_recovery_pending = False
        self.audio_recovery_interaction_id = ""
        self.audio_recovery_request_id = ""

        if self.active_interaction_id == closed_interaction_id:
            self.active_interaction_id = ""
            self.active_dialogue_request_type = ""
            self.pending_dialogue_request_id = ""
            self.dialogue_started_at = None
            self.waiting_for_human_initiative = False

        # Resume policy evaluation after the exact HRI transaction ended.
        if (
            self.active_navigation_goal_id
            and not self.social_resolution_aborted
            and self.pending_terminal_stop is None
            and not self.pending_speech_decision_id
            and self.fsm_state == STATE_WAITING
            and self._route_observation_is_fresh()
        ):
            if not self.human_proximity_active:
                self.deferred_clear_replan_goal_id = (
                    self.active_navigation_goal_id
                )

            elif (
                "WAIT",
                self.active_navigation_goal_id
            ) in self.finished_announcement_attempts:
                if self.interaction == "dialogue":
                    if self.dialogue_outcome == "ALLOW_CONFIRMED":
                        self._enter_replanning(
                            reason="HRI_CLOSED_PERMISSION_CONFIRMED",
                            directional_constraint=(
                                self.confirmed_human_direction
                            ),
                        )
                    elif self.dialogue_outcome == "NONE":
                        self._start_human_initiative_window()
                else:
                    self.conflict_observed_since = (
                        time.monotonic()
                    )
                    self.persistent_conflict_handled = False
                    self._transition_to(
                        STATE_OBSERVING,
                        reason="HRI_CLOSED_OBSERVATION_RESTORED"
                    )

    def _request_hri_close(self, reason):
        """Close only a dialogue turn actually submitted to HRI."""
        if self.pending_hri_close is not None:
            return True

        if self.interaction != "dialogue":
            return False

        interaction_id = str(
            self.audio_recovery_interaction_id
            or self.active_interaction_id
            or ""
        ).strip()

        request_id = str(
            self.audio_recovery_request_id
            or self.pending_dialogue_request_id
            or ""
        ).strip()

        # An OBSERVE/WAIT explanation can own an interaction_id
        # without any HRI turn.
        if not interaction_id or not request_id:
            return False

        now = time.monotonic()

        self.pending_hri_close = {
            "close_id": f"HRI_CLOSE_{uuid.uuid4().hex}",
            "interaction_id": interaction_id,
            "request_id": request_id,
            "reason": str(reason),
            "requested_at": now,
            "last_sent_at": now,
            "send_count": 1,
        }

        self.hri_close_fault_reported = False
        self._send_pending_hri_close()

        return True
    
    def _handle_clear(self, msg):
        """
        Process factual social clearance.

        Do not interrupt an already executing, validated
        social recovery merely because the person has moved
        outside the social conflict zone.

        CLEAR never creates execution authorization.
        """

        if self.pending_terminal_stop is not None:
            self.get_logger().info(
                "[CLEAR] Public goal termination already requested."
            )
            return

        goal_id = str(
            self.active_navigation_goal_id or ""
        ).strip()

        person_id = str(
            msg.person_id or ""
        ).strip()

        # Capture the previous scope before modifying the
        # active social context.
        resolved_scope = self._same_resolved_social_scope(
            person_id
        )

        # Nav2 execution must already have been confirmed.
        # A pending RESUME is not sufficient.
        execution_confirmed = bool(
            goal_id
            and resolved_scope
            and self.fsm_state == STATE_NAVIGATING
            and self.navigation_active
            and self.navigation_state == NAVIGATION_NAVIGATING
            and self.pending_gate_command is None
            and self.pending_terminal_stop is None
        )

        self.active_decision_id = msg.decision_id

        if person_id:
            self.active_person_id = person_id

        self.human_proximity_active = False
        self.social_observation_available = True
        self.waiting_for_observation_reacquisition = False

        self.social_loss_started_at = None
        self.social_loss_expired = False
        self.pending_retired_episode = None
        self.social_episode_retired = False

        self.retired_recovery_goal_id = ""
        self.retired_recovery_path_id = ""
        self.observation_loss_held_goal_id = ""

        self.conflict_observed_since = None
        self.persistent_conflict_handled = False

        if execution_confirmed:
            # Do not reset route-search memory, cancel Nav2,
            # send WAIT or start another planning transaction.
            #
            # The existing validated execution continues
            # under Nav2 controller/costmap supervision.
            self.route_waiting_for_change = False
            self.route_blocked_signature = None

            self.get_logger().info(
                "[SOCIAL CLEAR] "
                "Resolved social episode cleared | "
                f"goal={goal_id} | "
                "validated_navigation_continues=True | "
                "additional_replanning=False"
            )
            return

        # In every other case, retain the original
        # conservative clearance procedure.
        hri_closing = self._request_hri_close(
            "SOCIAL_CLEAR"
        )

        self._reset_route_search()

        if self.fsm_state in {
            STATE_STOPPING_FOR_SOCIAL_CONFLICT,
            STATE_ANNOUNCING_DETECTION,
        }:
            self.clear_pending_after_announcement = True

            self.get_logger().info(
                "[SOCIAL POLICY] "
                "CLEAR received during stop or speech | "
                "replanning deferred"
            )
            return

        if hri_closing:
            self.deferred_clear_replan_goal_id = goal_id

            self._publish_wait(
                reason="SOCIAL_CLEAR_WAITING_FOR_HRI_CLOSE"
            )

            self._transition_to(
                STATE_WAITING,
                reason="SOCIAL_CLEAR_HRI_CLOSE_PENDING",
            )
            return

        if not goal_id:
            self._reset_episode()

            self._transition_to(
                STATE_NAVIGATING,
                reason="SOCIAL_CLEAR_WITHOUT_NAVIGATION",
            )
            return

        self._invalidate_trajectory_validation(
            reason="SOCIAL_CLEAR_REQUIRES_REVALIDATION"
        )

        self._enter_replanning(
            reason="SOCIAL_CLEAR",
            directional_constraint="NONE",
        )

    def _social_scope_key(self, person_id=None):
        """Return the goal/session/episode/person identity."""

        person = (
            self.active_person_id
            if person_id is None
            else person_id
        )

        return (
            str(self.active_navigation_goal_id or "").strip(),
            str(self.social_state_session or "").strip(),
            str(self.social_state_episode or "").strip(),
            str(person or "").strip(),
        )

    def _same_resolved_social_scope(self, person_id=None):
        """Identify repeated presence after resolution in mode C or D."""

        key = self._social_scope_key(person_id)

        return bool(
            all(key)
            and self.human_proximity_active
            and not self.social_resolution_aborted
            and self.pending_terminal_stop is None
            and (
                self._permission_applies_to_current_episode(
                    person_id
                )
                or key == self.executed_social_scope
            )
        )

    def _schedule_social_reassessment(self):
        """Arm reassessment without restarting its delay on every event."""

        if not self.route_waiting_for_change:
            self.reassessment_last_at = time.monotonic()
            self.reassessment_last_stamp_ns = self.health_stamp_ns

        self.route_waiting_for_change = True

    def _permission_applies_to_current_episode(
        self,
        person_id=None,
    ):
        """Check semantic consent, never execution authorization."""

        key = self._social_scope_key(person_id)

        return bool(
            all(key)
            and key == self.authorized_episode_key
            and self.confirmed_human_intent == "ALLOW_PASSAGE"
            and self.dialogue_outcome == "ALLOW_CONFIRMED"
        )
    
    def _handle_observation_lost(self, msg):
        """
        Preserve social interaction and planning ownership
        during observation loss.

        Never equate missing perception with geometric CLEAR.
        Never validate a candidate from stale observations.
        """
        if not self.human_proximity_active:
            return

        self.active_decision_id = msg.decision_id

        if msg.person_id:
            self.active_person_id = msg.person_id

        if self.social_observation_available:
            self.social_loss_started_at = time.monotonic()
            self.social_loss_expired = False
            self.social_episode_retired = False

        self.social_observation_available = False
        self.waiting_for_observation_reacquisition = True

        self.conflict_observed_since = None
        self.persistent_conflict_handled = False

        dialogue_active = bool(
            self.interaction == "dialogue"
            and self.active_interaction_id
            and self.pending_dialogue_request_id
            and self.fsm_state in {
                STATE_WAITING_FOR_HUMAN,
                STATE_WAITING_FOR_CONFIRMATION,
                STATE_ASKING,
                STATE_CONFIRMING_HUMAN_INTENT,
            }
        )

        # A running trajectory computation is not a motion
        # authorization. Retain its request identity so that
        # its eventual candidate/failure can be correlated.
        if (
            self.fsm_state in {
                STATE_REPLANNING,
                STATE_VALIDATING_TRAJECTORY,
            }
            and not self.navigation_active
        ):
            self.trajectory_validation = (
                TRAJECTORY_VALIDATION_UNKNOWN
            )
            self.validated_path_id = ""
            self.validation_received_at = None
            self.validation_source_stamp_sec = None

            self.get_logger().debug(
                "[SOCIAL LOSS] "
                "Planning transaction retained; "
                "motion validation invalidated."
            )
            return

        if dialogue_active or self.fsm_state in {
            STATE_STOPPING_FOR_SOCIAL_CONFLICT,
            STATE_ANNOUNCING_DETECTION,
        }:
            self._invalidate_trajectory_validation(
                reason="SOCIAL_LOSS_DURING_TRANSACTION"
            )
            return

        # There is no need to repeatedly send WAIT
        # if the Gate has already confirmed the hold.
        if (
            self.fsm_state == STATE_WAITING
            and not self.navigation_active
        ):
            self._schedule_social_reassessment()
            return

        self._hold_route_search(
            reason="SOCIAL_OBSERVATION_LOST",
            reconsider=True,
        )

        self.observation_loss_held_goal_id = (
            self.active_navigation_goal_id
        )
        
    def _handle_observation_reacquired(self, msg):
        """Restore perception without restarting a resolved dialogue."""

        same_scope = self._same_resolved_social_scope(
            msg.person_id
        )
        was_waiting = self.waiting_for_observation_reacquisition

        self.active_decision_id = msg.decision_id

        if msg.person_id:
            self.active_person_id = msg.person_id

        self.human_proximity_active = True
        self.social_observation_available = True
        self.waiting_for_observation_reacquisition = False
        self.observation_loss_held_goal_id = ""

        self.social_loss_started_at = None
        self.social_loss_expired = False
        self.pending_retired_episode = None
        self.social_episode_retired = False

        self.conflict_observed_since = None
        self.persistent_conflict_handled = False

        if same_scope:
            if self.fsm_state == STATE_WAITING:
                self._schedule_social_reassessment()

            # Preserve REPLANNING, VALIDATING_TRAJECTORY,
            # RESUMING and NAVIGATING transactions.
            return

        if (
            self.interaction == "dialogue"
            and self.active_interaction_id
            and self.pending_dialogue_request_id
            and self.fsm_state in {
                STATE_WAITING_FOR_HUMAN,
                STATE_WAITING_FOR_CONFIRMATION,
                STATE_ASKING,
                STATE_CONFIRMING_HUMAN_INTENT,
            }
        ):
            return

        if self._interrupt_navigation_for_existing_social_episode():
            return

        if (
            self.fsm_state != STATE_WAITING
            or not was_waiting
        ):
            return

        if not self.active_navigation_goal_id:
            if self.announced_observe:
                self._transition_to(
                    STATE_OBSERVING,
                    reason=(
                        "SOCIAL_OBSERVATION_REACQUIRED_WITHOUT_GOAL"
                    ),
                )

            elif (
                msg.tracking_state == "TRACKED"
                and msg.person_id
                and msg.header.frame_id
                and math.isfinite(float(msg.person_distance))
                and msg.person_distance >= 0.0
                and math.isfinite(float(msg.social_radius))
                and msg.social_radius > 0.0
            ):
                self.announcement_social_decision = msg
                self.announcement_decision_id = str(
                    msg.decision_id or ""
                ).strip()

                self._ensure_interaction_id()
                self._enter_announcement()

            return

        if (
            "WAIT",
            self.active_navigation_goal_id,
        ) not in self.finished_announcement_attempts:
            return

        if self.interaction == "none":
            self._transition_to(
                STATE_OBSERVING,
                reason="SOCIAL_OBSERVATION_REACQUIRED",
            )
            self.conflict_observed_since = time.monotonic()
            return

        if self.dialogue_outcome in {
            "ALLOW_CONFIRMED",
            "DENY_CONFIRMED",
            "NO_RESPONSE",
            "NOT_UNDERSTOOD",
        }:
            self._schedule_social_reassessment()
            return

        if self.dialogue_outcome in {
            "PENDING",
            "DELIVERY_FAILED",
        }:
            return

        self.dialogue_retry_count = 0
        self.dialogue_speech_retry_count = 0
        self.waiting_for_human_initiative = False

        self._transition_to(
            STATE_ASKING,
            reason="SOCIAL_OBSERVATION_REACQUIRED",
        )
        self._request_passage_clarification()

    def _evaluate_conflict_persistence(self, now):
        if self.interaction == "dialogue":
            return

        if self.fsm_state != STATE_OBSERVING:
            return

        if not self.human_proximity_active:
            return

        if not self.social_observation_available:
            return

        if self.persistent_conflict_handled:
            return

        if self.conflict_observed_since is None:
            self.conflict_observed_since = now
            return

        if not self.active_navigation_goal_id:
            return

        elapsed = now - self.conflict_observed_since

        if elapsed < self.conflict_persistence_sec:
            return

        self.persistent_conflict_handled = True
        self.get_logger().info(
            "[SOCIAL POLICY] Persistent conflict | "
            f"elapsed={elapsed:.2f}s"
        )
        self._enter_replanning(reason="SOCIAL_CONFLICT_PERSISTENT")

    # ================================================================
    # Navigation context and interruption
    # ================================================================

    def _close_obsolete_pre_navigation_episode(self):
        """
        Close a pre-navigation social episode only after its
        observation has been lost.

        Never close a currently confirmed intrusion, an
        active navigation conflict, or a pending transaction.
        """

        if not self.human_proximity_active:
            return True

        # Episode belonged to an existing navigation goal.
        if self.social_episode_origin_goal_id:
            return False

        # Current positive observation: preserve protection.
        if self.social_observation_available:
            return False

        # Never reset a navigation stop or an active HRI turn.
        if (
            self.navigation_active
            or self.nav2_cancel_requested
            or self.pending_terminal_stop is not None
            or self.pending_hri_close is not None
            or self.pending_speech_decision_id
            or self.pending_dialogue_request_id
            or self.dialogue_audio_recovery_pending
        ):
            return False

        old_identity = (
            self.social_state_session,
            self.social_state_episode,
        )

        if old_identity[0] and old_identity[1]:
            self.retired_social_episode_ids[
                old_identity
            ] = True

            while len(self.retired_social_episode_ids) > 256:
                oldest_identity = next(
                    iter(self.retired_social_episode_ids)
                )

                self.retired_social_episode_ids.pop(
                    oldest_identity,
                    None,
                )

        self.get_logger().info(
            "[PRE-NAVIGATION SOCIAL CLOSE] "
            f"episode={old_identity} | "
            "reason=LOST_PRE_NAVIGATION_EPISODE"
        )

        self._reset_episode()

        # The identity is retired, not forgotten.
        # Later messages from the same episode must not
        # reactivate its previous transaction.
        self.latest_social_decision = None

        return True

    def _navigation_goal_context_callback(self, msg):
        goal_id = str(msg.navigation_goal_id or "").strip()

        if not goal_id:
            self.get_logger().warning(
                "Ignoring NavigationGoalContext without navigation_goal_id."
            )
            return

        if not msg.goal.header.frame_id:
            self.get_logger().warning(
                "Ignoring NavigationGoalContext without goal frame_id."
            )
            return

        if (
            self.active_navigation_goal_id
            and goal_id != self.active_navigation_goal_id
        ):
            self._invalidate_trajectory_validation(
                reason="NAVIGATION_GOAL_CHANGED"
            )
            self._reset_route_search()

        if goal_id != self.active_navigation_goal_id:
            
            if self.pending_terminal_stop is not None:
                self.get_logger().error(
                    "[GOAL] Previous public termination is still pending."
                )
                return

            if (
                self.human_proximity_active
                and not self.social_episode_origin_goal_id
                and not self.social_observation_available
            ):
                self._close_obsolete_pre_navigation_episode()
                
            self._reset_route_search()
            self._invalidate_trajectory_validation(
                "NEW_PUBLIC_GOAL"
            )

            if (
                not self.pending_speech_decision_id
                and self.pending_hri_close is None
            ):
                self._transition_to(
                    STATE_NAVIGATING,
                    reason="NEW_PUBLIC_GOAL"
                )

            self.observation_loss_held_goal_id = ""
            self.gate_ready_goal_id = ""
            self.pending_gate_command = None
            self.latest_start_request_id = ""
            self.initial_start_authorized_goal_id = ""
            self.initial_start_attempt_count = 0
            self.initial_start_next_attempt_at = 0.0

        self.active_navigation_goal_id = goal_id
        self.active_navigation_goal_pose = msg.goal

        self.get_logger().info(
            "[NAVIGATION GOAL CONTEXT] "
            f"goal_id={goal_id!r} | "
            f"frame={msg.goal.header.frame_id!r}"
        )

        self.get_logger().info(
            "[GOAL] Context received; waiting for matching Gate state | "
            f"goal={goal_id}"
        )

    def _interrupt_navigation_for_existing_social_episode(self):
        """
        Interrupt new conflicts.

        Repeated messages from a resolved social context do not
        restart the stop/announcement/dialogue sequence.
        """
        goal_id = str(
            self.active_navigation_goal_id or ""
        ).strip()

        if not goal_id or not self.human_proximity_active:
            return False

        if self.pending_terminal_stop is not None:
            return True

        decision = self.latest_social_decision

        person_id = (
            decision.person_id
            if decision is not None
            else self.active_person_id
        )

        same_scope = self._same_resolved_social_scope(
            person_id
        )

        if same_scope:
            if (
                self.social_observation_available
                and not self.social_loss_expired
                and not self.social_episode_retired
                and self._route_observation_is_fresh()
            ):
                return False

            if self.fsm_state in {
                STATE_REPLANNING,
                STATE_VALIDATING_TRAJECTORY,
            }:
                # Computation may finish, but execution still
                # requires compatible current observations.
                return False

            if self.fsm_state in {
                STATE_NAVIGATING,
                STATE_RESUMING,
            }:
                self._hold_route_search(
                    reason="RESOLVED_EPISODE_OBSERVATION_UNAVAILABLE",
                    reconsider=True,
                )
                self.waiting_for_observation_reacquisition = True
                return True

            if self.fsm_state == STATE_WAITING:
                self._schedule_social_reassessment()
                return True

            return False

        nav = self.latest_navigation_state

        if (
            nav is None
            or str(nav.navigation_goal_id or "").strip() != goal_id
            or not self._message_is_recent(nav, 1.0)
        ):
            return False

        if self.fsm_state == STATE_ANNOUNCING_DETECTION:
            if self.announcement_context_key != ("WAIT", goal_id):
                if self.deferred_social_goal_id != goal_id:
                    self.deferred_social_goal_id = goal_id

                    self._invalidate_trajectory_validation(
                        reason="PUBLIC_GOAL_DURING_SOCIAL_SPEECH"
                    )
                    self._publish_wait(
                        reason="PUBLIC_GOAL_DURING_SOCIAL_SPEECH"
                    )

                return True

            return False

        if self.fsm_state not in {
            STATE_NAVIGATING,
            STATE_OBSERVING,
            STATE_WAITING,
        }:
            return False

        if (
            self.social_episode_retired
            or self.social_loss_expired
            or not self.social_observation_available
        ):
            if self.observation_loss_held_goal_id != goal_id:
                self.observation_loss_held_goal_id = goal_id

                self._hold_route_search(
                    reason=(
                        "SOCIAL_OBSERVATION_UNAVAILABLE_FOR_PUBLIC_GOAL"
                    ),
                    reconsider=True,
                )

            self.waiting_for_observation_reacquisition = True

            self._transition_to(
                STATE_WAITING,
                reason="SOCIAL_OBSERVATION_UNAVAILABLE_FOR_PUBLIC_GOAL",
            )
            return True

        if decision is None:
            self._hold_route_search(
                reason="SOCIAL_INTRUSION_CONTEXT_MISSING",
                reconsider=True,
            )
            return True

        event = normalize_social_navigation_event(
            decision.event
        )

        if event not in {
            SOCIAL_EVENT_INTRUSION_CONFIRMED,
            SOCIAL_EVENT_OBSERVATION_REACQUIRED,
        }:
            return False

        if not self._message_is_recent(
            decision,
            self.social_decision_max_age_sec,
        ):
            if (
                not self.navigation_active
                and self.navigation_state in {
                    "WAITING",
                    "SOCIAL_HOLD",
                }
            ):
                self._schedule_social_reassessment()

                self._transition_to(
                    STATE_WAITING,
                    reason="SOCIAL_INTRUSION_CONTEXT_STALE",
                )
                return False

            self._hold_route_search(
                reason="SOCIAL_INTRUSION_CONTEXT_STALE",
                reconsider=True,
            )
            return True

        self.announcement_social_decision = decision
        self.announcement_decision_id = str(
            decision.decision_id or ""
        ).strip()
        self.active_person_id = str(
            decision.person_id or ""
        ).strip()

        if not self.announcement_decision_id:
            self._hold_route_search(
                reason="SOCIAL_DECISION_ID_MISSING",
                reconsider=False,
            )
            return True

        interruption_key = (
            goal_id,
            self.social_state_session,
            self.social_state_episode,
            self.active_person_id,
        )

        running = bool(
            self.navigation_active
            and self.navigation_state == NAVIGATION_NAVIGATING
        )

        announcement_finished = (
            ("WAIT", goal_id)
            in self.finished_announcement_attempts
        )

        stopped = bool(
            not self.navigation_active
            and self.navigation_state in {
                "WAITING",
                "SOCIAL_HOLD",
                "REPLANNING",
            }
            and self.nav2_cancel_requested
            and self.nav2_cancel_confirmed
            and self.stable_stop_confirmed
            and self.motion_state_received_at is not None
            and (
                time.monotonic()
                - self.motion_state_received_at
            ) <= 0.5
            and self.motion_state_stamp_sec is not None
            and self.stop_request_stamp_sec is not None
            and (
                self.motion_state_stamp_sec
                >= self.stop_request_stamp_sec
            )
        )

        if stopped:
            if not announcement_finished:
                self._ensure_interaction_id()
                self.last_navigation_interruption_key = (
                    interruption_key
                )
                self._enter_announcement()
                return True

            return False

        if announcement_finished and not running:
            return False

        prestart = bool(
            self.initial_start_authorized_goal_id != goal_id
            and self.pending_gate_command is None
            and self.initial_start_attempt_count == 0
            and self.navigation_state in {
                "WAITING_FOR_SPEECH",
                "WAITING",
            }
            and not self.navigation_active
            and not self.nav2_cancel_requested
            and not self.nav2_cancel_confirmed
        )

        self._invalidate_trajectory_validation(
            reason="SOCIAL_INTRUSION_INTERRUPTS_GOAL"
        )
        self._ensure_interaction_id()

        self.pending_human_intent = "UNKNOWN"
        self.pending_human_direction = "NONE"
        self.confirmed_human_intent = "UNKNOWN"
        self.confirmed_human_direction = "NONE"

        self.authorized_episode_key = None
        self.executed_social_scope = None
        self.pending_dialogue_scope = None
        self.dialogue_outcome = "NONE"

        self.last_navigation_interruption_key = interruption_key
        self.navigation_was_active_at_intrusion = running

        self.nav2_cancel_requested = not prestart
        self.nav2_cancel_confirmed = False

        self.stable_stop_confirmed = False
        self.stable_stop_source = ""
        self.motion_state_received_at = None
        self.motion_state_stamp_sec = None

        if prestart:
            self.stop_request_stamp_sec = None

            self._publish_wait(
                reason="SOCIAL_CONFLICT_BEFORE_NAVIGATION_START"
            )
            self._enter_announcement()
            return True

        self.stop_request_stamp_sec = (
            self.get_clock().now().nanoseconds / 1e9
        )

        self._transition_to(
            STATE_STOPPING_FOR_SOCIAL_CONFLICT,
            reason="ACTIVE_NAVIGATION_SOCIAL_INTRUSION",
        )
        self._publish_wait(
            reason="ACTIVE_NAVIGATION_SOCIAL_INTRUSION"
        )
        return True
    
    def _gate_command_result_callback(self, msg):
        try:
            data = json.loads(msg.data)
        except (TypeError, ValueError):
            return

        if not isinstance(data, dict):
            return

        if (
            data.get("version") != 1
            or data.get("producer") != "navigation_goal_gate"
        ):
            return

        goal_id = str(
            data.get("navigation_goal_id") or ""
        ).strip()

        if not goal_id or goal_id != self.active_navigation_goal_id:
            return

        action = str(data.get("action") or "").upper()
        request_id = str(data.get("request_id") or "").strip()
        status = str(data.get("status") or "").upper()
        reason = str(data.get("reason") or "")
        path_id = str(data.get("path_id") or "").strip()

        if action == "GOAL_READY" and status == "READY":
            self.gate_ready_goal_id = goal_id

            self.get_logger().info(
                f"[GATE] Goal ready after speech | goal={goal_id}"
            )

            self._close_obsolete_pre_navigation_episode()

            self._evaluate_initial_goal_start()
            return

        if (
            action == "WAIT"
            and status == "HOLD_CONFIRMED"
            and self.fsm_state
                == STATE_STOPPING_FOR_SOCIAL_CONFLICT
            and self.nav2_cancel_requested
        ):
            if (
                not self.pending_stop_hold_request_id
                or request_id
                    != self.pending_stop_hold_request_id
            ):
                self.get_logger().warning(
                    "[STOP BARRIER] Obsolete WAIT acknowledgement | "
                    f"received={request_id} | "
                    f"expected={self.pending_stop_hold_request_id}"
                )
                return

            self.nav2_cancel_confirmed = True

            self.get_logger().info(
                "[STOP BARRIER] Matching Gate hold confirmed | "
                f"goal={goal_id} | request={request_id}"
            )

            self._evaluate_stop_barrier()
            return

        if (
            action == "REPLAN"
            and status == "HOLD_CONFIRMED"
            and self.fsm_state == STATE_REPLANNING
            and not self.pending_generation_request_id
            and not self.navigation_active
        ):
            self._request_current_trajectory_validation(
                generation_reason="REPLAN",
                directional_constraint=(
                    self.pending_replan_directional_constraint
                ),
            )
            return

        if action == "START":
            if request_id != self.latest_start_request_id:
                return

            if status == "ACCEPTED":
                self.initial_start_authorized_goal_id = goal_id
                self.pending_gate_command = None

                self.get_logger().info(
                    "[START] Gate accepted command | "
                    f"goal={goal_id} | request={request_id}"
                )
                return

            if status in {"REJECTED", "EXPIRED"}:
                self.initial_start_authorized_goal_id = ""

                pending = self.pending_gate_command
                if (
                    pending is not None
                    and pending["request_id"] == request_id
                ):
                    self.pending_gate_command = None

                # Do not retry immediately: give the Gate and
                # observation stream time to settle.
                self.initial_start_next_attempt_at = (
                    time.monotonic() + 0.5
                )

                self.get_logger().warning(
                    "[START] Gate did not retain authorization | "
                    f"status={status} | reason={reason}"
                )
                return

        if action == "RESUME":
            pending = self.pending_gate_command

            if (
                pending is None
                or pending["action"] != "RESUME"
                or pending["goal_id"] != goal_id
                or pending["request_id"] != request_id
                or pending["path_id"] != path_id
            ):
                return

            if status in {
                "ACCEPTED",
                "ACCEPTED_PENDING_CANDIDATE",
                "ACCEPTED_PENDING_TERMINATION",
            }:
                pending["status"] = status
                pending["accepted_at"] = time.monotonic()

                self.get_logger().info(
                    "[RESUME] Gate response | "
                    f"status={status} | path={path_id}"
                )
                return

            if status in {"REJECTED", "EXPIRED"}:
                self.pending_gate_command = None

                self._invalidate_trajectory_validation(
                    reason=f"GATE_RESUME_{status}"
                )

                self._publish_wait(
                    reason=f"GATE_RESUME_{status}"
                )

                self._transition_to(
                    STATE_WAITING,
                    reason=f"GATE_RESUME_{status}",
                )

                self.get_logger().warning(
                    "[RESUME] Authorization lost | "
                    f"status={status} | reason={reason}"
                )
                return

        if (
            action == "REPLAN"
            and status == "REJECTED"
            and self.fsm_state == STATE_REPLANNING
        ):
            self._hold_route_search(
                reason="GATE_REPLAN_REJECTED",
                reconsider=True,
            )

    def _navigation_state_callback(self, msg):
        self.latest_navigation_state = msg

        goal_id = str(
            msg.navigation_goal_id or ""
        ).strip()

        if (
            not goal_id
            or goal_id != self.active_navigation_goal_id
        ):
            return

        if not self._message_is_recent(msg, 1.0):
            return

        self.navigation_state = normalize_navigation_state(
            msg.navigation_state
        )
        self.navigation_active = bool(msg.navigation_active)

        self.get_logger().info(
            "[NAVIGATION STATE] "
            f"goal_id={goal_id!r} | "
            f"state={self.navigation_state!r} | "
            f"active={self.navigation_active}"
        )

        if self.navigation_state in {
            "SUCCEEDED",
            "ABORTED",
            "CANCELED"
        }:
            if (
                self.pending_terminal_stop is not None
                and self.pending_terminal_stop["goal_id"]
                == goal_id
            ):
                self.pending_terminal_stop = None
                self.terminal_stop_fault_reported = False

            self.pending_gate_command = None
            self.active_navigation_goal_id = ""
            self.active_navigation_goal_pose = None
            self.pending_replan_directional_constraint = "NONE"

            self._invalidate_trajectory_validation(
                reason="NAVIGATION_GOAL_TERMINATED"
            )

            if (
                self.fsm_state
                == STATE_STOPPING_FOR_SOCIAL_CONFLICT
            ):
                self._transition_to(
                    STATE_WAITING,
                    reason="PUBLIC_GOAL_TERMINATED_DURING_STOP",
                )

            self._evaluate_stop_barrier()
            return

        if (
            self.fsm_state == STATE_REPLANNING
            and self.navigation_state == "REPLANNING"
            and not self.navigation_active
            and not self.pending_generation_request_id
        ):
            direction = (
                self.pending_replan_directional_constraint
            )

            self.get_logger().info(
                "[REPLAN] Gate acknowledged REPLANNING | "
                f"goal={goal_id!r} | direction={direction}"
            )

            self._request_current_trajectory_validation(
                generation_reason="REPLAN",
                directional_constraint=direction,
            )

            self._evaluate_stop_barrier()
            return

        if (
            self.navigation_state == NAVIGATION_NAVIGATING
            and self.navigation_active
            and self.fsm_state == STATE_WAITING
            and not self.human_proximity_active
        ):
            self._transition_to(
                STATE_NAVIGATING,
                reason="NAVIGATION_STARTED",
            )

        self._interrupt_navigation_for_existing_social_episode()
        self._evaluate_stop_barrier()

        if (
            self.fsm_state == STATE_RESUMING
            and self.navigation_state == NAVIGATION_NAVIGATING
            and self.navigation_active
        ):
            self._finish_episode_after_resume()
    # ================================================================
    # Physical-stop barrier
    # ================================================================

    def _message_is_recent(self, msg, max_age_sec):
        stamp_sec = self._stamp_to_seconds(msg.header.stamp)
        age = self.get_clock().now().nanoseconds / 1e9 - stamp_sec

        return (
            stamp_sec > 0.0
            and math.isfinite(age)
            and -0.1 <= age <= max_age_sec
        )

    def _robot_motion_state_callback(self, msg):
        fresh = self._message_is_recent(msg, 0.5)

        self.stable_stop_confirmed = (
            fresh and bool(msg.stable_stop_confirmed)
        )
        self.stable_stop_source = str(msg.source or "")
        self.motion_state_received_at = (
            time.monotonic() if fresh else None
        )
        self.motion_state_stamp_sec = (
            self._stamp_to_seconds(msg.header.stamp) if fresh else None
        )

        self._evaluate_stop_barrier()

    def _evaluate_stop_barrier(self):
        """
        Complete a social interruption only after:

          1. Nav2 hold/cancellation confirmed by the Gate.
          2. A fresh stable-stop observation.
          3. Stop observation newer than the stop request.

        Observation loss never bypasses these conditions.
        """
        if self.fsm_state != STATE_STOPPING_FOR_SOCIAL_CONFLICT:
            return

        self.get_logger().info(
            "[STOP BARRIER] "
            f"cancel_requested={self.nav2_cancel_requested} | "
            f"cancel_confirmed={self.nav2_cancel_confirmed} | "
            f"stable_stop={self.stable_stop_confirmed} | "
            f"observation={self.social_observation_available}",
            throttle_duration_sec=1.0,
        )

        if not self.nav2_cancel_requested:
            return

        if not self.nav2_cancel_confirmed:
            return

        if not self.stable_stop_confirmed:
            return

        if (
            self.motion_state_received_at is None
            or self.motion_state_stamp_sec is None
            or self.stop_request_stamp_sec is None
        ):
            return

        now_monotonic = time.monotonic()

        now_ros = (
            self.get_clock().now().nanoseconds / 1e9
        )

        receipt_age = (
            now_monotonic - self.motion_state_received_at
        )

        source_age = (
            now_ros - self.motion_state_stamp_sec
        )

        if (
            not math.isfinite(receipt_age)
            or not 0.0 <= receipt_age <= 0.5
            or not math.isfinite(source_age)
            or not -0.1 <= source_age <= 0.5
        ):
            self.stable_stop_confirmed = False
            return

        if (
            self.motion_state_stamp_sec
            < self.stop_request_stamp_sec
        ):
            self.stable_stop_confirmed = False
            return

        self.get_logger().info(
            "[STOP BARRIER] Physical stop confirmed | "
            f"goal={self.active_navigation_goal_id} | "
            f"source={self.stable_stop_source}"
        )

        # The physical stop is now established.
        # Whether social perception is available determines
        # the next interaction state, not whether cancellation
        # was successful.
        if not self.social_observation_available:
            self.waiting_for_observation_reacquisition = True

            self._invalidate_trajectory_validation(
                reason="SOCIAL_OBSERVATION_LOST_AFTER_STOP"
            )

            # The robot remains held by the Goal Gate.
            # Observation loss does not invalidate the
            # already confirmed intrusion snapshot.
            if (
                self.announcement_social_decision is not None
                and self.announcement_decision_id
            ):
                self.get_logger().warning(
                    "[STOP BARRIER] Physical stop confirmed "
                    "with observation unavailable. "
                    "Preserving the confirmed intrusion snapshot "
                    "and starting the announcement. "
                    "Navigation remains unauthorized."
                )

                self._transition_to(
                    STATE_WAITING,
                    reason="STOP_CONFIRMED_WITH_INTRUSION_SNAPSHOT",
                )

                self._enter_announcement()
                return

            # No trustworthy intrusion snapshot exists.
            # Remain stopped and wait for valid context.
            self._publish_wait(
                reason="SOCIAL_CONTEXT_MISSING_AFTER_STOP"
            )

            self._transition_to(
                STATE_WAITING,
                reason="STOP_CONFIRMED_CONTEXT_UNAVAILABLE",
            )
            return
        
        if (
            self.social_loss_expired
            or self.social_episode_retired
        ):
            self._publish_wait(
                reason="UNRESOLVED_SOCIAL_EPISODE_AFTER_STOP"
            )

            self._transition_to(
                STATE_WAITING,
                reason="STOP_CONFIRMED_SOCIAL_REASSESSMENT_REQUIRED",
            )
            return

        if not self.human_proximity_active:
            self._publish_wait(
                reason="SOCIAL_CLEAR_REVALIDATION_REQUIRED"
            )

            self._transition_to(
                STATE_WAITING,
                reason="STOP_CONFIRMED_SOCIAL_CLEAR",
            )

            if self.active_navigation_goal_id:
                self._enter_replanning(
                    reason="SOCIAL_CLEAR_AFTER_STOP",
                    directional_constraint="NONE",
                )
            return

        if (
            self.announcement_social_decision is None
            or not self.announcement_decision_id
        ):
            self._publish_wait(
                reason="SOCIAL_CONTEXT_MISSING_AFTER_STOP"
            )

            self._transition_to(
                STATE_WAITING,
                reason="STOP_CONFIRMED_CONTEXT_UNAVAILABLE",
            )
            return

        self._transition_to(
            STATE_WAITING,
            reason="NAVIGATION_CANCELLED_AND_STABLE_STOP_CONFIRMED",
        )

        self._enter_announcement()

    # ================================================================
    # Social announcements
    # ================================================================

    def _enter_announcement(self):
        goal_id = str(self.active_navigation_goal_id or "").strip()
        action = "WAIT" if goal_id else "OBSERVE"
        key = (action, goal_id)

        self.announcement_context_key = key
        self.current_announcement_action = action

        if key in self.finished_announcement_attempts:
            self._enter_social_resolution()
            return

        self._transition_to(
            STATE_ANNOUNCING_DETECTION,
            reason="INITIAL_SOCIAL_ANNOUNCEMENT",
        )
        self.pending_explanation_ready = False
        self.pending_speech_started = False
        self.pending_speech_started_at = None
        self.initial_speech_retry_count = 0

        self._request_initial_explanation()

    def _request_initial_explanation(self):
        if self.fsm_state != STATE_ANNOUNCING_DETECTION:
            return

        decision = self.announcement_social_decision
        source_id = str(self.announcement_decision_id or "").strip()

        if not source_id:
            self._publish_wait(reason="EXPLANATION_DECISION_ID_MISSING")
            self._transition_to(
                STATE_WAITING,
                reason="EXPLANATION_DECISION_ID_MISSING",
            )
            return

        if decision is None:
            self._publish_wait(reason="EXPLANATION_SOCIAL_CONTEXT_MISSING")
            self._transition_to(
                STATE_WAITING,
                reason="EXPLANATION_SOCIAL_CONTEXT_MISSING",
            )
            return

        action, speech_goal_id = self.announcement_context_key
        has_navigation_context = action == "WAIT"

        clearance_confirmed = (
            self.clear_pending_after_announcement
            and not self.human_proximity_active
            and self.social_observation_available
        )

        context = ExplanationContext()
        context.header.stamp = self.get_clock().now().to_msg()
        context.navigation_state = self.navigation_state
        context.navigation_goal_id = speech_goal_id
        context.navigation_active = bool(
            has_navigation_context
            and speech_goal_id == self.active_navigation_goal_id
            and self.navigation_active
        )
        context.social_conflict_active = bool(self.human_proximity_active)
        context.language = self.language
        context.action = action
        context.person_id = self.active_person_id

        # Preserve the immutable intrusion snapshot.
        context.zone_type = str(
            getattr(decision, "zone_type", "NONE") or "NONE"
        )
        context.tracking_state = str(
            getattr(decision, "tracking_state", "") or ""
        )

        if hasattr(decision, "person_position"):
            context.person_position = decision.person_position

        context.person_distance = float(
            getattr(decision, "person_distance", 0.0)
        )
        context.social_radius = float(
            getattr(decision, "social_radius", 0.0)
        )
        context.person_speed = float(
            getattr(decision, "person_speed", 0.0)
        )
        context.occupancy_ratio = float(
            getattr(decision, "occupancy_ratio", 0.0)
        )
        context.requires_replan = False

        if clearance_confirmed:
            context.reason_code = "SOCIAL_CLEAR_BEFORE_ANNOUNCEMENT"
            context.situation_summary = (
                "The navigation or social interaction was paused because "
                "of a detected proximity conflict. Perception has since "
                "confirmed that the proximity conflict has cleared."
            )

            if has_navigation_context:
                context.requires_replan = True
                context.robot_intention = (
                    "Remain stopped, then generate and validate a fresh "
                    "trajectory before considering navigation resumption."
                )
                context.explanation_goal = (
                    "Briefly explain that the earlier proximity conflict "
                    "has cleared and that a fresh route will be checked "
                    "before continuing."
                )
            else:
                context.robot_intention = (
                    "Remain idle and acknowledge that the proximity "
                    "episode has cleared."
                )
                context.explanation_goal = (
                    "Briefly acknowledge the earlier nearby presence "
                    "and its confirmed clearance."
                )

        elif not self.social_observation_available:
            context.reason_code = "SOCIAL_OBSERVATION_LOST"
            context.situation_summary = (
                "A person was previously detected nearby, but current "
                "visual observation is unavailable. The social episode "
                "remains unresolved."
            )
            context.robot_intention = (
                "Keep navigation paused."
                if has_navigation_context
                else "Remain idle."
            )
            context.explanation_goal = (
                "Explain that the person's current position cannot be "
                "confirmed and that the social situation remains unresolved."
            )

        else:
            context.reason_code = "SOCIAL_ZONE_INTRUSION"

            if has_navigation_context:
                context.situation_summary = (
                    "A confirmed social-zone intrusion affected the "
                    "current navigation execution."
                )
                context.robot_intention = (
                    "Remain stopped while the social situation is resolved."
                )
                context.explanation_goal = (
                    "Briefly explain why navigation was paused because "
                    "of the detected social situation. Leave room for "
                    "spontaneous human initiative and do not ask the "
                    "passage question in this announcement."
                )
            else:
                context.situation_summary = (
                    "A person has been detected within the social "
                    "interaction zone while no navigation goal is active."
                )
                context.robot_intention = (
                    "Acknowledge the person's presence while remaining idle."
                )
                context.explanation_goal = (
                    "Briefly acknowledge the detected person's presence "
                    "without claiming that navigation was interrupted."
                )

        navigation_context_id = (
            speech_goal_id if has_navigation_context else "NO_GOAL"
        )
        decision_id = (
            f"{source_id}:{action}:{navigation_context_id}:"
            f"{context.reason_code}"
        )

        if self.initial_speech_retry_count > 0:
            decision_id += f":RETRY_{self.initial_speech_retry_count}"

        context.decision_id = decision_id
        self.pending_speech_decision_id = decision_id
        self.pending_explanation_ready = False
        self.pending_speech_started = False
        self.pending_speech_started_at = None
        self.pending_speech_requested_at = time.monotonic()
        self.pending_speech_completion_action = "INITIAL_SOCIAL_RESOLUTION"
        self.pending_autonomous_readaptation_reason = ""

        self.explanation_context_pub.publish(context)
        self.get_logger().info(
            "[EXPLAINABILITY] Context published | "
            f"decision_id={decision_id!r} | action={action} | "
            f"reason={context.reason_code}"
        )

    def _enter_social_resolution(self, *, speech_completed=False):
        key = self.announcement_context_key
        action, speech_goal_id = key or (
            self.current_announcement_action,
            "",
        )

        if key is not None:
            self.finished_announcement_attempts.add(key)

        if speech_completed:
            if action == "OBSERVE":
                self.announced_observe = True
            elif action == "WAIT":
                self.announced_navigation_goal_id = speech_goal_id

        self.current_announcement_action = ""
        self.announcement_context_key = None
        self.deferred_social_goal_id = ""

        if not self.social_observation_available:
            self.waiting_for_observation_reacquisition = True
            self._publish_wait(
                reason="SOCIAL_OBSERVATION_UNAVAILABLE_AFTER_SPEECH"
            )
            self._transition_to(
                STATE_WAITING,
                reason="SOCIAL_OBSERVATION_UNAVAILABLE_AFTER_SPEECH",
            )
            return

        if (
            self.clear_pending_after_announcement
            and not self.human_proximity_active
        ):
            self.clear_pending_after_announcement = False

            if not self.active_navigation_goal_id:
                self._reset_episode()
                self._transition_to(
                    STATE_NAVIGATING,
                    reason="DEFERRED_SOCIAL_CLEAR_WITHOUT_NAVIGATION",
                )
            else:
                if self.pending_hri_close is not None:
                    self.deferred_clear_replan_goal_id = (
                        self.active_navigation_goal_id
                    )

                    self._publish_wait(
                        reason="DEFERRED_CLEAR_HRI_CLOSE_PENDING"
                    )

                    self._transition_to(
                        STATE_WAITING,
                        reason="DEFERRED_CLEAR_HRI_CLOSE_PENDING",
                    )
                    return
                self._enter_replanning(
                    reason="DEFERRED_SOCIAL_CLEAR",
                    directional_constraint="NONE",
                )
            return

        if (
            not self.active_navigation_goal_id
            or action == "OBSERVE"
            or speech_goal_id != self.active_navigation_goal_id
        ):
            self.waiting_for_human_initiative = False
            self.dialogue_started_at = None
            self._transition_to(
                STATE_OBSERVING,
                reason="PRESENCE_OR_PREVIOUS_GOAL_ANNOUNCEMENT_FINISHED",
            )
            self.conflict_observed_since = time.monotonic()
            self.persistent_conflict_handled = False
            self._interrupt_navigation_for_existing_social_episode()
            return

        self._transition_to(
            STATE_SOCIAL_RESOLUTION,
            reason="INITIAL_ANNOUNCEMENT_ATTEMPT_FINISHED",
        )

        if self.interaction == "none":
            self._transition_to(
                STATE_OBSERVING,
                reason="DETERMINISTIC_RESOLUTION_MODE",
            )
            self.conflict_observed_since = time.monotonic()
            self.persistent_conflict_handled = False

        elif self.interaction == "dialogue":
            self._start_human_initiative_window()

        else:
            self._publish_wait(reason="UNKNOWN_COMMUNICATION_MODE")
            self._transition_to(
                STATE_WAITING,
                reason="UNKNOWN_COMMUNICATION_MODE",
            )

    # ================================================================
    # Autonomous readaptation explanation
    # ================================================================

    def _begin_autonomous_readaptation_announcement(self, *, reason):
        self.dialogue_started_at = None
        self.waiting_for_human_initiative = False
        self._clear_pending_human_preference()

        reason = str(reason or "").strip().upper()

        self._transition_to(
            STATE_ANNOUNCING_DETECTION,
            reason="ANNOUNCING_AUTONOMOUS_READAPTATION",
        )
        self.get_logger().warning(
            "[SOCIAL POLICY] Announcing autonomous readaptation | "
            f"reason={reason}"
        )
        self._request_autonomous_readaptation_explanation(reason=reason)

    def _request_autonomous_readaptation_explanation(self, *, reason):
        self._clear_pending_human_preference()
        reason = str(reason or "").strip().upper()

        supported = {
            "NO_HUMAN_RESPONSE",
            "HUMAN_RESPONSE_NOT_UNDERSTOOD",
            "DIALOGUE_SPEECH_UNAVAILABLE",
        }

        if reason not in supported:
            self.get_logger().error(
                f"Unsupported readaptation reason: {reason!r}"
            )
            reason = "DIALOGUE_SPEECH_UNAVAILABLE"

        decision_id = f"{self.active_interaction_id}:{reason}"

        context = ExplanationContext()
        context.header.stamp = self.get_clock().now().to_msg()
        context.decision_id = decision_id
        context.navigation_state = self.navigation_state
        context.navigation_goal_id = self.active_navigation_goal_id
        context.navigation_active = bool(self.navigation_active)
        context.social_conflict_active = bool(self.human_proximity_active)
        context.language = self.language
        context.action = "WAIT"
        context.reason_code = reason
        context.person_id = self.active_person_id
        context.requires_replan = True

        if reason == "NO_HUMAN_RESPONSE":
            context.situation_summary = (
                "No human response was received after the dialogue "
                "request and its retry."
            )
            context.robot_intention = (
                "Remain stopped, explain that no response was received, "
                "then look for another safe navigation solution."
            )
            context.explanation_goal = (
                "Explain briefly that no response was received and that "
                "another safe way to continue will be considered."
            )

        elif reason == "HUMAN_RESPONSE_NOT_UNDERSTOOD":
            context.situation_summary = (
                "Human responses were received, but their intended meaning "
                "could not be determined reliably after clarification."
            )
            context.robot_intention = (
                "Remain stopped, explain that the response could not be "
                "understood reliably, then look for another safe "
                "navigation solution."
            )
            context.explanation_goal = (
                "Explain briefly that the response could not be understood "
                "well enough to rely on it, and that another safe route "
                "will be considered."
            )

        else:
            context.situation_summary = (
                "The spoken dialogue could not be delivered reliably."
            )
            context.robot_intention = (
                "Remain stopped and look for another safe navigation "
                "solution without inferring human silence."
            )
            context.explanation_goal = (
                "Explain briefly that the spoken exchange cannot continue "
                "and that another safe way to continue will be considered."
            )

        self.pending_speech_decision_id = decision_id
        self.pending_speech_completion_action = (
            "REPLAN_AUTONOMOUS_READAPTATION"
        )
        self.pending_autonomous_readaptation_reason = reason
        self.pending_explanation_ready = False
        self.pending_speech_started = False
        self.pending_speech_started_at = None
        self.pending_speech_requested_at = time.monotonic()

        self.explanation_context_pub.publish(context)
        self.get_logger().info(
            "[EXPLAINABILITY] Readaptation explanation requested | "
            f"decision_id={decision_id!r} | reason={reason}"
        )

    # ================================================================
    # Speech feedback
    # ================================================================

    def _explanation_text_callback(self, msg):
        if self.fsm_state != STATE_ANNOUNCING_DETECTION:
            return

        decision_id = str(msg.decision_id or "").strip()

        if not decision_id or decision_id != self.pending_speech_decision_id:
            return

        if not str(msg.text or "").strip():
            self.get_logger().warning(
                "[EXPLAINABILITY] Matching ExplanationText is empty."
            )
            return

        self.pending_explanation_ready = True
        self.get_logger().info(
            "[EXPLAINABILITY] Text ready | "
            f"decision_id={decision_id!r} | provider={msg.provider!r}"
        )

    def _speech_status_callback(self, msg):
        """Complete the exact speech transaction without inventing consent."""

        if self.fsm_state != STATE_ANNOUNCING_DETECTION:
            return

        decision_id = str(msg.decision_id or "").strip()

        if (
            not decision_id
            or decision_id != self.pending_speech_decision_id
        ):
            return

        state = str(msg.state or "").strip().upper()
        action = self.pending_speech_completion_action

        if state == "STARTED":
            if not self.pending_speech_started:
                self.pending_speech_started = True
                self.pending_speech_started_at = time.monotonic()
            return

        if state == "FAILED":
            detail = str(msg.detail or "").strip()
            retryable = detail.startswith("AUDIO_RETRYABLE|")

            self.get_logger().error(
                "[SPEECH FAILURE] "
                f"decision_id={decision_id} | "
                f"retryable={retryable} | "
                f"detail={detail}"
            )

            if not retryable:
                self._hold_after_speech_uncertainty(
                    decision_id=decision_id,
                    reason="SPEECH_AUDIO_STATE_UNCONFIRMED",
                    request_cancel=False,
                )
                return

        if action == "PASSAGE_ASSISTANCE":
            if state in {
                "COMPLETED",
                "CANCELED",
                "FAILED",
            }:
                # FAILED reaches here only with a retryable audio state.
                # Do not repeat the assistance announcement.
                self._clear_pending_speech_transaction()
                self.clear_pending_after_announcement = False

                self._hold_route_search(
                    reason="PASSAGE_ASSISTANCE_FINISHED",
                    reconsider=True,
                )

            return

        if state == "CANCELED":
            self._hold_after_speech_uncertainty(
                decision_id=decision_id,
                reason="SPEECH_CANCELED",
                request_cancel=False,
            )
            return

        if state not in {"COMPLETED", "FAILED"}:
            return

        autonomous_reason = str(
            self.pending_autonomous_readaptation_reason or ""
        ).strip().upper()

        self._clear_pending_speech_transaction()

        if action == "INITIAL_SOCIAL_RESOLUTION":
            if state == "COMPLETED":
                self._enter_social_resolution(
                    speech_completed=True
                )

            elif (
                self.initial_speech_retry_count
                < self.max_initial_speech_retries
            ):
                self.initial_speech_retry_count += 1
                self._request_initial_explanation()

            else:
                self._enter_social_resolution()

            return

        if action == "REPLAN_AUTONOMOUS_READAPTATION":
            reason = (
                autonomous_reason
                or "AUTONOMOUS_READAPTATION"
            )

            if state == "FAILED":
                reason += "_SPEECH_FAILED"

            self._enter_replanning(reason=reason)
            return

        self._publish_wait(
            reason="SPEECH_COMPLETION_ACTION_MISSING"
        )
        self._transition_to(
            STATE_WAITING,
            reason="SPEECH_COMPLETION_ACTION_MISSING",
        )
        
    # ================================================================
    # Human dialogue
    # ================================================================

    def _new_dialogue_request(self):
        """Create a dialogue turn bound to its exact social context."""

        self._ensure_interaction_id()

        request = DialogueRequest()
        request.header.stamp = self.get_clock().now().to_msg()

        request.interaction_id = self.active_interaction_id
        request.request_id = f"HRI_TURN_{uuid.uuid4().hex}"

        request.person_id = self.active_person_id
        request.response_timeout = float(
            self.dialogue_response_timeout_sec
        )
        request.language = self.language

        self.pending_dialogue_request_id = request.request_id
        self.pending_dialogue_scope = self._social_scope_key()

        return request
    
    def _start_human_initiative_window(self):
        """Give the person a bounded opportunity to speak first."""
        if (
            self.pending_hri_close is not None
            or self.pending_terminal_stop is not None
            or self.dialogue_audio_recovery_pending
        ):
            self._hold_route_search(
                "HUMAN_INITIATIVE_WAITING_FOR_PREVIOUS_TERMINATION"
            )
            return

        if (
            not self.active_navigation_goal_id
            or not self._route_observation_is_fresh()
        ):
            self._hold_route_search(
                "HUMAN_INITIATIVE_CONTEXT_UNAVAILABLE"
            )
            return

        self._ensure_interaction_id()
        self.dialogue_outcome = "PENDING"

        request = self._new_dialogue_request()

        request.request_type = "HUMAN_INITIATIVE_WINDOW"
        request.context = "SOCIAL_CONFLICT"
        request.question = ""
        request.intent_to_confirm = ""
        request.direction_to_confirm = "NONE"
        request.robot_initiated = False

        request.response_timeout = float(
            self.human_initiative_timeout_sec
        )

        self.active_dialogue_request_type = (
            "HUMAN_INITIATIVE_WINDOW"
        )

        self.waiting_for_human_initiative = True
        self.dialogue_started_at = time.monotonic()

        self.dialogue_request_pub.publish(request)

        self._transition_to(
            STATE_WAITING_FOR_HUMAN,
            reason="WAITING_FOR_HUMAN_INITIATIVE",
        )
    
    def _request_passage_clarification(self):
        self._ensure_interaction_id()
        request = self._new_dialogue_request()

        request.request_type = "PASSAGE_CLARIFICATION"
        self.active_dialogue_request_type = "PASSAGE_CLARIFICATION"
        request.context = "SOCIAL_CONFLICT"
        request.question = get_dialogue_text("passage", self.language)
        request.intent_to_confirm = ""
        request.direction_to_confirm = "NONE"
        request.robot_initiated = True

        self.waiting_for_human_initiative = False
        self.dialogue_outcome = "PENDING"
        self.dialogue_request_pub.publish(request)
        self.dialogue_started_at = time.monotonic()

        self._transition_to(
            STATE_WAITING_FOR_HUMAN,
            reason="PASSAGE_CLARIFICATION_REQUESTED",
        )

    def _request_terminal_stop(self, *, reason):
        if self.pending_terminal_stop is not None:
            return

        goal_id = str(
            self.active_navigation_goal_id or ""
        ).strip()

        if not goal_id:
            return

        self.social_resolution_aborted = True
        self.route_waiting_for_change = False
        self.deferred_clear_replan_goal_id = ""

        self._invalidate_trajectory_validation(
            reason=reason
        )

        request_id = self._publish_collaborative_decision(
            action="STOP",
            outcome="SOCIAL_RESOLUTION_FAILED",
            human_intent=self.confirmed_human_intent,
            directional_constraint="NONE",
            path_id="",
            reason=reason,
        )

        self.pending_terminal_stop = {
            "goal_id": goal_id,
            "request_id": request_id,
            "requested_at": time.monotonic(),
            "reason": reason,
        }

        self.terminal_stop_fault_reported = False

        self._transition_to(
            STATE_WAITING,
            reason=f"TERMINAL_STOP_{reason}",
        )

        self.get_logger().warning(
            "[TERMINAL STOP] Requested | "
            f"goal={goal_id} | request={request_id} | "
            f"reason={reason}"
        )


    def _human_intent_callback(self, msg):
        interaction_id = str(msg.interaction_id or "").strip()
        request_id = str(msg.request_id or "").strip()

        source = str(msg.source or "").strip().upper()

        if source in {
            "AUDIO_TERMINATED",
            "AUDIO_TERMINATION_UNCONFIRMED",
        }:
            if not self.dialogue_audio_recovery_pending:
                return

            if (
                interaction_id != self.audio_recovery_interaction_id
                or request_id != self.audio_recovery_request_id
            ):
                return

            if source == "AUDIO_TERMINATION_UNCONFIRMED":
                self.dialogue_outcome = "DELIVERY_FAILED"
                self._publish_wait(
                    reason="AUDIO_TERMINATION_UNCONFIRMED"
                )
                self._transition_to(
                    STATE_WAITING,
                    reason="AUDIO_RECOVERY_UNCONFIRMED",
                )
                # Preserve the transaction for a later terminal result.
                return

            # Exact audio termination has been confirmed.
            self.dialogue_audio_recovery_pending = False
            self.audio_recovery_interaction_id = ""
            self.audio_recovery_request_id = ""

            # The former person or dialogue may already have
            # disappeared. Never reopen that dialogue blindly.
            if (
                self.social_loss_expired
                or self.social_episode_retired
                or not self.social_observation_available
            ):
                self.get_logger().info(
                    "[DIALOGUE AUDIO] Old audio reconciled; "
                    "social recovery will use fresh observations."
                )
                return

            if (
                self.active_interaction_id != interaction_id
                or self.fsm_state != STATE_WAITING
                or self.social_resolution_aborted
                or not self._route_observation_is_fresh()
            ):
                return

            if (
                self.dialogue_speech_retry_count
                >= self.dialogue_speech_max_retries
            ):
                self.dialogue_outcome = "DELIVERY_FAILED"
                self._publish_wait(
                    reason="DIALOGUE_AUDIO_RETRIES_EXHAUSTED"
                )
                return

            self.dialogue_speech_retry_count += 1
            self.dialogue_outcome = "PENDING"

            self._transition_to(
                STATE_ASKING,
                reason="DIALOGUE_AUDIO_RECOVERED",
            )

            self._retry_active_dialogue_request()
            return

        if not self.active_interaction_id:
            return

        if interaction_id != self.active_interaction_id:
            return

        expected_request_id = str(
            self.pending_dialogue_request_id or ""
        ).strip()

        if not expected_request_id or request_id != expected_request_id:
            self.get_logger().warning(
                "[HRI] Ignoring result from non-current turn | "
                f"received={request_id!r} | "
                f"expected={expected_request_id!r}"
            )
            return

        source = str(msg.source or "").strip().upper()

        # Technical rejection is not a human answer.
        if source == "DIALOGUE_REQUEST_REJECTED":
            reason = str(
                msg.utterance or "UNKNOWN"
            ).strip().upper()

            self.dialogue_started_at = None
            self.dialogue_outcome = "DELIVERY_FAILED"

            self._publish_wait(
                reason=f"HRI_REQUEST_REJECTED_{reason}"
            )

            self._transition_to(
                STATE_WAITING,
                reason="HRI_REQUEST_REJECTED",
            )
            return


        if self.fsm_state not in {
            STATE_WAITING_FOR_HUMAN,
            STATE_WAITING_FOR_CONFIRMATION,
        }:
            return

        if source == "TIMEOUT":
            self._publish_wait(
                reason="AMBIGUOUS_HRI_TIMEOUT"
            )
            return

        if source in {
            "ROBOT_SPEECH_FAILED",
            "ROBOT_SPEECH_TIMEOUT",
        }:
            self._handle_dialogue_delivery_failure(
                source=source
            )
            return


        if source == "HUMAN_RESPONSE_TIMEOUT":
            self._handle_dialogue_timeout()
            return

        if not self.social_observation_available:
            self.get_logger().warning(
                "[HRI] Matching human response received while "
                "social observation is unavailable. "
                "Semantic processing continues, but navigation "
                "remains subject to fresh trajectory validation."
            )
        
        if self.pending_hri_close is not None:
            return
        
        intent = str(msg.intent or "").strip().upper()
        direction = self._normalize_direction(msg.direction)

        if self.fsm_state == STATE_WAITING_FOR_HUMAN:
            self._handle_initial_human_intent(
                intent,
                direction,
            )
            return

        if self.fsm_state == STATE_WAITING_FOR_CONFIRMATION:
            self._handle_confirmation_response(intent)

    def _handle_initial_human_intent(self, intent, direction):
        if intent == "REQUEST_STOP":
            self.confirmed_human_intent = "REQUEST_STOP"
            self._request_terminal_stop(
                reason="HUMAN_REQUESTED_STOP"
            )
            return

        if intent == "REJECT":
            self._handle_rejected_human_response()
            return

        if intent == "DENY_PASSAGE":
            self.pending_human_intent = "DENY_PASSAGE"
            self.pending_human_direction = direction

            self.dialogue_retry_count = 0
            self.dialogue_speech_retry_count = 0

            self._transition_to(
                STATE_CONFIRMING_HUMAN_INTENT,
                reason="DENY_PASSAGE_REQUIRES_SEMANTIC_CONFIRMATION",
            )

            self._request_intent_confirmation()
            return

        if intent == "CONFIRM":
            if self.waiting_for_human_initiative:
                self.get_logger().warning(
                    "[DIALOGUE] CONFIRM received without a pending question."
                )

                self._publish_wait(
                    reason="AMBIGUOUS_CONFIRM_WITHOUT_PENDING_QUESTION"
                )
                return

            self.pending_human_intent = "ALLOW_PASSAGE"
            self.pending_human_direction = direction

            self.dialogue_retry_count = 0
            self.dialogue_speech_retry_count = 0

            self._transition_to(
                STATE_CONFIRMING_HUMAN_INTENT,
                reason="PASSAGE_PERMISSION_REQUIRES_SEMANTIC_CONFIRMATION",
            )

            self._request_intent_confirmation()
            return

        if intent == "ALLOW_PASSAGE":
            self.pending_human_intent = "ALLOW_PASSAGE"
            self.pending_human_direction = direction

            self.dialogue_retry_count = 0
            self.dialogue_speech_retry_count = 0

            self._transition_to(
                STATE_CONFIRMING_HUMAN_INTENT,
                reason="ALLOW_PASSAGE_REQUIRES_SEMANTIC_CONFIRMATION",
            )

            self._request_intent_confirmation()
            return

        if intent == "REQUEST_FOLLOW":
            self._publish_wait(
                reason="HUMAN_REQUESTED_FOLLOW"
            )
            self._transition_to(
                STATE_WAITING,
                reason="HUMAN_REQUESTED_FOLLOW",
            )
            return

        self._publish_wait(
            reason="HUMAN_INTENT_UNKNOWN"
        )

        self._transition_to(
            STATE_WAITING,
            reason="HUMAN_INTENT_UNKNOWN",
        )
    def _handle_rejected_human_response(self):
        self.dialogue_started_at = None
        self._clear_pending_human_preference()

        if self.dialogue_retry_count < self.dialogue_max_retries:
            self.dialogue_retry_count += 1
            self.dialogue_speech_retry_count = 0

            self.get_logger().warning(
                "[DIALOGUE] Interpretation rejected; asking again | "
                f"retry={self.dialogue_retry_count}/"
                f"{self.dialogue_max_retries}"
            )
            self._transition_to(
                STATE_ASKING,
                reason="HUMAN_RESPONSE_REJECTED_RETRY",
            )
            self._request_passage_clarification()
            return

        self.dialogue_retry_count = 0
        self.dialogue_speech_retry_count = 0
        self.dialogue_outcome = "NOT_UNDERSTOOD"
        self._begin_autonomous_readaptation_announcement(
            reason="HUMAN_RESPONSE_NOT_UNDERSTOOD"
        )

    def _request_intent_confirmation(self):
        self._ensure_interaction_id()

        intent = self.pending_human_intent
        direction = self.pending_human_direction

        if intent == "DENY_PASSAGE":
            question_key = "confirm_deny"
        elif direction == "LEFT":
            question_key = "confirm_allow_left"
        elif direction == "RIGHT":
            question_key = "confirm_allow_right"
        else:
            question_key = "confirm_allow"

        request = self._new_dialogue_request()
        request.request_type = "INTENT_CONFIRMATION"
        self.active_dialogue_request_type = "INTENT_CONFIRMATION"
        request.context = "INTENT_REQUIRES_CONFIRMATION"
        request.question = get_dialogue_text(question_key, self.language)
        request.intent_to_confirm = self.pending_human_intent
        request.direction_to_confirm = self.pending_human_direction
        request.robot_initiated = True
        self.dialogue_outcome = "PENDING"
        self.dialogue_request_pub.publish(request)
        self.dialogue_started_at = time.monotonic()

        self._transition_to(
            STATE_WAITING_FOR_CONFIRMATION,
            reason="SEMANTIC_CONFIRMATION_REQUESTED",
        )

    def _handle_confirmation_response(self, intent):
        """Apply semantic confirmation without authorizing motion."""

        intent = str(intent or "").strip().upper()

        if intent == "REQUEST_STOP":
            self.authorized_episode_key = None
            self.executed_social_scope = None
            self.confirmed_human_intent = "REQUEST_STOP"

            self._request_terminal_stop(
                reason="HUMAN_REQUESTED_STOP"
            )
            return

        if intent == "REJECT":
            self._handle_rejected_human_response()
            return

        if intent != "CONFIRM":
            self._hold_route_search(
                reason="CONFIRMATION_NOT_ESTABLISHED",
                reconsider=False,
            )
            return

        confirmed_intent = self.pending_human_intent
        direction = self._normalize_direction(
            self.pending_human_direction
        )
        scope = self.pending_dialogue_scope

        if (
            confirmed_intent not in {
                "ALLOW_PASSAGE",
                "DENY_PASSAGE",
            }
            or scope is None
            or not all(scope)
            or scope != self._social_scope_key()
        ):
            self._hold_route_search(
                reason="CONFIRMATION_CONTEXT_INVALID",
                reconsider=False,
            )
            return

        self.confirmed_human_intent = confirmed_intent
        self.confirmed_human_direction = direction

        self.executed_social_scope = None
        self.authorized_episode_key = (
            scope
            if confirmed_intent == "ALLOW_PASSAGE"
            else None
        )

        self._clear_pending_human_preference()

        self.pending_dialogue_request_id = ""
        self.pending_dialogue_scope = None

        self.dialogue_started_at = None
        self.waiting_for_human_initiative = False
        self.dialogue_retry_count = 0
        self.dialogue_speech_retry_count = 0

        if confirmed_intent == "DENY_PASSAGE":
            self.dialogue_outcome = "DENY_CONFIRMED"

            if direction in {"LEFT", "RIGHT"}:
                self.forbidden_human_direction = direction

            self.confirmed_human_direction = "NONE"

            self._request_next_social_alternative()
            return

        self.dialogue_outcome = "ALLOW_CONFIRMED"

        if (
            self.active_navigation_goal_id
            and self.pending_terminal_stop is None
            and self.pending_hri_close is None
        ):
            self.social_resolution_aborted = False
            self.resolution_watch_goal_id = (
                self.active_navigation_goal_id
            )
            self.resolution_watch_started_at = time.monotonic()

        self.get_logger().info(
            "[DIALOGUE] Permission retained | "
            f"scope={scope} | "
            f"direction={direction} | "
            "navigation_authorized=False"
        )

        self._enter_replanning(
            reason="HUMAN_PERMISSION_CONFIRMED",
            directional_constraint=direction,
        )

    def propose_alternative_direction(self, direction):
        direction = self._normalize_direction(direction)

        if direction not in {"LEFT", "RIGHT"}:
            self.get_logger().error(
                "Alternative direction must be LEFT or RIGHT."
            )
            return

        self.pending_human_intent = "ALLOW_PASSAGE"
        self.pending_human_direction = direction
        self.active_dialogue_request_type = "ALTERNATIVE_PROPOSAL"
        self.active_alternative_direction = direction

        self._transition_to(
            STATE_PROPOSING_ALTERNATIVE,
            reason="ALTERNATIVE_ROUTE_AVAILABLE",
        )

        question_key = (
            "alternative_left" if direction == "LEFT" else "alternative_right"
        )
        request = self._new_dialogue_request()
        request.request_type = "ALTERNATIVE_PROPOSAL"
        request.context = "REQUESTED_ROUTE_UNAVAILABLE"
        request.question = get_dialogue_text(question_key, self.language)
        request.intent_to_confirm = "ALLOW_PASSAGE"
        request.direction_to_confirm = direction
        request.robot_initiated = True
        self.dialogue_outcome = "PENDING"
        self.dialogue_request_pub.publish(request)
        self.dialogue_started_at = time.monotonic()

        self._transition_to(
            STATE_WAITING_FOR_CONFIRMATION,
            reason="ALTERNATIVE_PROPOSAL_SENT",
        )

    def _retry_active_dialogue_request(self):
        request_type = self.active_dialogue_request_type

        if request_type == "PASSAGE_CLARIFICATION":
            self._request_passage_clarification()
            return

        if request_type == "INTENT_CONFIRMATION":
            self._request_intent_confirmation()
            return

        if request_type == "ALTERNATIVE_PROPOSAL":
            self.propose_alternative_direction(
                self.active_alternative_direction
            )
            return

        self.get_logger().error(
            "[DIALOGUE] Unknown retry context | "
            f"type={request_type!r}"
        )
        self._publish_wait(reason="DIALOGUE_RETRY_CONTEXT_MISSING")


    def _handle_dialogue_delivery_failure(self, *, source):
        source = str(source or "").strip().upper()
        self.dialogue_started_at = None

        if source in {
            "ROBOT_SPEECH_TIMEOUT",
            "ROBOT_SPEECH_FAILED",
        }:
            # Do not assume audio playback has stopped.
            # The HRI/Speech Manager must reconcile this
            # exact transaction before any new speech.
            self.dialogue_audio_recovery_pending = True
            self.audio_recovery_interaction_id = (
                self.active_interaction_id
            )
            self.audio_recovery_request_id = (
                self.pending_dialogue_request_id
            )
            self.dialogue_outcome = "PENDING"

            self._publish_wait(
                reason="DIALOGUE_AUDIO_TERMINATION_PENDING"
            )

            self._transition_to(
                STATE_WAITING,
                reason="DIALOGUE_AUDIO_RECOVERY",
            )

            self.get_logger().warning(
                "[DIALOGUE AUDIO] Awaiting termination | "
                f"source={source} | "
                f"request={self.audio_recovery_request_id}"
            )
            return

        self.dialogue_outcome = "DELIVERY_FAILED"

        self._publish_wait(
            reason="DIALOGUE_DELIVERY_FAILURE"
        )

        self._transition_to(
            STATE_WAITING,
            reason="DIALOGUE_DELIVERY_FAILURE",
        )

    def _handle_dialogue_timeout(self):
        self.dialogue_started_at = None

        if self.waiting_for_human_initiative:
            self.waiting_for_human_initiative = False
            self.dialogue_retry_count = 0
            self.dialogue_speech_retry_count = 0
            self._transition_to(
                STATE_ASKING,
                reason="HUMAN_INITIATIVE_TIMEOUT",
            )
            self._request_passage_clarification()
            return

        if self.dialogue_retry_count < self.dialogue_max_retries:
            self.dialogue_retry_count += 1
            self.dialogue_speech_retry_count = 0
            self.get_logger().warning(
                "[DIALOGUE] No response after delivered question | "
                f"retry={self.dialogue_retry_count}/"
                f"{self.dialogue_max_retries}"
            )
            self._transition_to(
                STATE_ASKING,
                reason="DIALOGUE_RETRY_AFTER_HUMAN_TIMEOUT",
            )
            self._retry_active_dialogue_request()
            return

        self.dialogue_retry_count = 0
        self.dialogue_speech_retry_count = 0
        self._clear_pending_human_preference()
        self.dialogue_outcome = "NO_RESPONSE"
        self._begin_autonomous_readaptation_announcement(
            reason="NO_HUMAN_RESPONSE"
        )

    # ================================================================
    # Candidate generation
    # ================================================================

    def _request_current_trajectory_validation(
        self,
        *,
        generation_reason,
        directional_constraint,
    ):
        """
        Submit a bounded trajectory generation request.

        A missing human response does not prevent planning.
        A generated candidate is never automatically authorized
        for execution.
        """

        if self.social_resolution_aborted:
            return

        if not self.active_navigation_goal_id:
            return

        if self.active_navigation_goal_pose is None:
            return

        # Do not overlap uncertain planner transactions.
        if self.uncertain_generation_request_id:
            self._hold_route_search(
                "PREVIOUS_PLANNING_TERMINATION_UNCONFIRMED"
            )
            return

        if self.route_attempt_count >= self.route_search_max_attempts:
            self._hold_route_search(
                reason="SOCIAL_ROUTE_ATTEMPTS_EXHAUSTED",
                reconsider=False,
            )
            return

        direction = self._normalize_direction(
            directional_constraint
        )

        retired_recovery = (
            self.retired_recovery_goal_id
            == self.active_navigation_goal_id
            and self._retired_recovery_evidence_available()
        )

        ordinary_no_conflict = (
            not self.human_proximity_active
            and self.social_observation_available
            and not self.social_episode_retired
            and not self.social_loss_expired
            and not self.waiting_for_observation_reacquisition
        )

        confirmed_passage_search = (
            self.interaction == "dialogue"
            and self.dialogue_outcome == "ALLOW_CONFIRMED"
            and self.confirmed_human_intent == "ALLOW_PASSAGE"
            and self.active_navigation_goal_id
            == self.social_episode_origin_goal_id
        )

        # Permit planning after an unanswered or
        # unrecognized dialogue, not movement.
        autonomous_readaptation_search = (
            self.interaction == "dialogue"
            and self.dialogue_outcome in {
                "NO_RESPONSE",
                "NOT_UNDERSTOOD",
            }
            and self.pending_hri_close is None
            and self.pending_terminal_stop is None
            and not self.dialogue_audio_recovery_pending
            and not self.pending_speech_decision_id
        )

        if not (
            retired_recovery
            or ordinary_no_conflict
            or confirmed_passage_search
            or autonomous_readaptation_search
            or self._route_observation_is_fresh()
        ):
            self._hold_route_search(
                "PLANNING_OBSERVATION_EXPIRED"
            )
            return

        if retired_recovery and direction != "NONE":
            self._hold_route_search(
                "RETIRED_RECOVERY_REQUIRES_STANDARD_PATH"
            )
            return

        # Remove authorization associated with the old path.
        self._invalidate_trajectory_validation(
            reason="FRESH_TRAJECTORY_REQUIRED"
        )

        request = self._build_trajectory_generation_request(
            generation_reason=(
                self.route_generation_reason
                or generation_reason
            ),
            directional_constraint=direction,
        )

        if request is None:
            self._hold_route_search(
                "TRAJECTORY_GENERATION_CONTEXT_UNAVAILABLE"
            )
            return

        if retired_recovery:
            request.person_id = ""

        self.route_attempt_count += 1
        self.route_tried_directions.add(direction)

        self.pending_generation_request_id = request.request_id
        self.generation_requested_at = time.monotonic()

        self.trajectory_generation_request_pub.publish(request)

        self.get_logger().info(
            "[TRAJECTORY SEARCH] "
            f"attempt={self.route_attempt_count}/"
            f"{self.route_search_max_attempts} | "
            f"request={request.request_id} | "
            f"reason={request.generation_reason} | "
            f"direction={direction} | "
            f"retired_recovery={retired_recovery} | "
            f"autonomous_readaptation={autonomous_readaptation_search}"
        )
    def _build_trajectory_generation_request(
        self,
        *,
        generation_reason,
        directional_constraint,
    ):
        if not self.active_navigation_goal_id:
            return None

        if self.active_navigation_goal_pose is None:
            return None

        request = TrajectoryGenerationRequest()

        request.header.stamp = self.get_clock().now().to_msg()
        request.request_id = f"TRAJ_REQ_{uuid.uuid4().hex}"
        request.navigation_goal_id = self.active_navigation_goal_id
        request.goal = self.active_navigation_goal_pose
        request.use_start = False
        request.planner_id = ""

        request.generation_reason = str(
            generation_reason or "REPLAN"
        ).strip().upper()

        request.directional_constraint = self._normalize_direction(
            directional_constraint
        )

        request.person_id = self.active_person_id

        return request
    def _candidate_trajectory_callback(self, msg):
        if self.fsm_state not in {
            STATE_REPLANNING,
            STATE_VALIDATING_TRAJECTORY,
        }:
            return

        if not self.active_navigation_goal_id:
            return

        request_id = str(msg.request_id or "").strip()

        if (
            not self.pending_generation_request_id
            or not request_id
            or request_id != self.pending_generation_request_id
        ):
            return

        path_id = str(msg.path_id or "").strip()
        goal_id = str(msg.navigation_goal_id or "").strip()

        if not path_id:
            return

        if not goal_id or goal_id != self.active_navigation_goal_id:
            return

        if path_id != self.candidate_path_id:
            self.trajectory_validation = TRAJECTORY_VALIDATION_UNKNOWN
            self.validated_path_id = ""
            self.validation_received_at = None

        # Only this callback establishes the candidate identity.
        self.candidate_path_id = path_id
        self.candidate_navigation_goal_id = goal_id
        self.pending_generation_request_id = ""

        self.get_logger().info(
            "[CANDIDATE TRAJECTORY] "
            f"path_id={path_id} | goal={goal_id!r}"
        )
        self._transition_to(
            STATE_VALIDATING_TRAJECTORY,
            reason="CANDIDATE_TRAJECTORY_RECEIVED",
        )

        self.generation_requested_at = time.monotonic()

        early = self.early_validations.pop(path_id, None)
        self.early_validations.clear()

        if early is not None:
            self._trajectory_validation_callback(early)

    # ================================================================
    # Exact trajectory validation
    # ================================================================

    def _trajectory_generation_result_callback(self, msg):
        """Handle planner results without confusing them with social conflicts."""
        try:
            data = json.loads(msg.data)
        except (TypeError, ValueError):
            return

        if not isinstance(data, dict):
            return

        if (
            data.get("version") != 1
            or data.get("producer") != "trajectory_manager"
        ):
            return

        goal_id = str(
            data.get("navigation_goal_id") or ""
        ).strip()

        request_id = str(
            data.get("request_id") or ""
        ).strip()

        status = str(
            data.get("status") or ""
        ).strip().upper()

        reason = str(
            data.get("reason") or "UNSPECIFIED"
        ).strip()

        if (
            not goal_id
            or goal_id != self.active_navigation_goal_id
            or not request_id
        ):
            return

        retired_recovery = (
            self.social_episode_retired
            and self.retired_recovery_goal_id == goal_id
        )

        if status == "TERMINATED":
            if request_id != self.uncertain_generation_request_id:
                return

            self.uncertain_generation_request_id = ""

            self.get_logger().info(
                "[TRAJECTORY RECOVERY] Previous generation terminated | "
                f"request={request_id}"
            )

            if self.social_resolution_aborted:
                return

            # The recovery timer owns retries for retired episodes.
            if retired_recovery:
                return

            if (
                self.fsm_state == STATE_WAITING
                and self.human_proximity_active
                and self._route_observation_is_fresh()
            ):
                self._request_next_social_alternative()

            return

        if request_id != self.pending_generation_request_id:
            return

        if self.fsm_state not in {
            STATE_REPLANNING,
            STATE_VALIDATING_TRAJECTORY,
        }:
            return

        if status not in {
            "FAILED",
            "TERMINATION_UNCONFIRMED",
        }:
            return

        self.get_logger().error(
            "[TRAJECTORY RESULT] "
            f"request={request_id} | "
            f"status={status} | "
            f"reason={reason}"
        )

        if status == "TERMINATION_UNCONFIRMED":
            self.uncertain_generation_request_id = request_id

            self._hold_route_search(
                "PLANNING_TERMINATION_UNCONFIRMED"
            )
            return

        # FAILED is a planning failure, not a social-validation result.
        self._hold_route_search(
            reason=f"NAV2_PLANNING_FAILED:{reason}",
            reconsider=not retired_recovery,
        )

        if self.social_resolution_aborted:
            return

        if retired_recovery:
            self.get_logger().warning(
                "[TRAJECTORY RECOVERY] No path produced by Nav2. "
                "Waiting for the bounded standard-path retry; "
                "no directional bypass requested."
            )
            return

        # Directional alternatives require a currently observed person.
        # The existing alternative policy also enforces dialogue in mode D.
        if (
            self.human_proximity_active
            and self._route_observation_is_fresh()
        ):
            self._request_next_social_alternative()


    def _trajectory_validation_callback(self, msg):
        """Accept an exact result and retain actual geometric conflicts."""

        if self.fsm_state not in {
            STATE_REPLANNING,
            STATE_VALIDATING_TRAJECTORY,
        }:
            return

        goal_id = str(
            self.active_navigation_goal_id or ""
        ).strip()

        if not goal_id:
            return

        result = normalize_trajectory_validation(msg.validation)

        if not is_trajectory_validation_result(result):
            return

        path_id = str(msg.path_id or "").strip()

        fresh = self._message_is_recent(
            msg,
            self.trajectory_validation_max_age_sec,
        )

        # A result may arrive before its authoritative candidate.
        if not self.candidate_path_id:
            if (
                path_id
                and self.pending_generation_request_id
                and fresh
            ):
                if len(self.early_validations) >= 16:
                    self.early_validations.pop(
                        next(iter(self.early_validations))
                    )

                self.early_validations[path_id] = msg

            return

        if (
            not path_id
            or path_id != self.candidate_path_id
            or self.candidate_navigation_goal_id != goal_id
        ):
            return

        if not fresh:
            self._hold_route_search(
                reason="STALE_TRAJECTORY_VALIDATION",
                reconsider=True,
            )
            return

        reason = str(msg.reason or "").strip()
        normalized_reason = reason.upper()

        self.trajectory_validation = result
        self.validated_path_id = ""
        self.validation_received_at = time.monotonic()
        self.validation_source_stamp_sec = self._stamp_to_seconds(
            msg.header.stamp
        )

        if result == TRAJECTORY_VALIDATION_CONFLICT:
            scope = self._social_scope_key()

            if self.conflict_evidence_scope != scope:
                self.conflict_evidence_scope = scope
                self.conflicting_candidate_ids = set()

            # Duplicate results for one path count only once.
            self.conflicting_candidate_ids.add(path_id)

            self._invalidate_trajectory_validation(
                reason="CANDIDATE_SOCIAL_CONFLICT"
            )

            self._request_next_social_alternative()
            return

        if result == TRAJECTORY_VALIDATION_UNKNOWN:
            if (
                not self.human_proximity_active
                and not self.navigation_active
                and self.social_observation_available
                and not self.social_episode_retired
                and not self.social_loss_expired
                and not self.waiting_for_observation_reacquisition
                and normalized_reason in {
                    "NO_SOCIAL_OBSERVATION",
                    "INVALID_OR_STALE_SOCIAL_OBSERVATION",
                    "OBSERVATION_EXPIRED_DURING_VALIDATION",
                }
            ):
                self.initial_empty_retry_goal_id = goal_id
                self.initial_empty_retry_after_health_ns = (
                    self.health_stamp_ns
                )
                self.initial_empty_retry_not_before = (
                    time.monotonic() + 0.25
                )
            else:
                self.initial_empty_retry_goal_id = ""
                self.initial_empty_retry_after_health_ns = None

            self._hold_route_search(
                reason=reason or "TRAJECTORY_VALIDATION_UNKNOWN",
                reconsider=True,
            )
            return

        retired_recovery = bool(
            normalized_reason == "NO_TRACKED_SOCIAL_ZONE"
            and self._retired_recovery_evidence_available()
        )

        if (
            self.human_proximity_active
            and normalized_reason == "NO_TRACKED_SOCIAL_ZONE"
            and not retired_recovery
        ):
            hold_reason = "SOCIAL_CLEARANCE_NOT_CONFIRMED"

        elif (
            not self.social_observation_available
            or self.social_loss_expired
            or self.social_episode_retired
        ) and not retired_recovery:
            hold_reason = "SOCIAL_RECOVERY_EVIDENCE_REQUIRED"

        else:
            hold_reason = ""

        if hold_reason:
            self._hold_route_search(
                reason=hold_reason,
                reconsider=True,
            )
            return

        self.validated_path_id = path_id

        if retired_recovery:
            self.retired_recovery_path_id = path_id

        self._enter_resuming(path_id)

    def _invalidate_trajectory_validation(self, reason):
        self.trajectory_validation = TRAJECTORY_VALIDATION_UNKNOWN
        self.validated_path_id = ""
        self.validation_received_at = None
        self.validation_source_stamp_sec = None
        self.candidate_path_id = ""
        self.candidate_navigation_goal_id = ""
        self.pending_generation_request_id = ""
        self.generation_requested_at = None
        self.early_validations = {}

        self.get_logger().debug(
            f"Trajectory validation invalidated: {reason}"
        )

    def _trajectory_validation_is_fresh(self):
        if (
            self.validation_received_at is None
            or self.validation_source_stamp_sec is None
        ):
            return False

        receipt_age = time.monotonic() - self.validation_received_at
        source_age = (
            self.get_clock().now().nanoseconds / 1e9
            - self.validation_source_stamp_sec
        )

        return (
            0.0 <= receipt_age <= self.trajectory_validation_max_age_sec
            and -0.1 <= source_age <= self.trajectory_validation_max_age_sec
        )

    def _retired_recovery_evidence_available(
        self,
        *,
        require_selected_goal=True,
    ):
        goal_id = str(self.active_navigation_goal_id or "").strip()

        if not goal_id or self.active_navigation_goal_pose is None:
            return False

        if self.social_resolution_aborted:
            return False
        
        if (
            self.pending_hri_close is not None
            or self.hri_close_fault_reported
        ):
            return False

        if (
            self.interaction == "dialogue"
            and getattr(self, "dialogue_outcome", "NONE")
                == "DENY_CONFIRMED"
        ):
            return False

        if not (
            self.social_episode_retired
            and self.social_loss_expired
            and self.human_proximity_active
        ):
            return False

        if (
            require_selected_goal
            and self.retired_recovery_goal_id != goal_id
        ):
            return False

        retired_at = self.retired_episode_retired_at

        if retired_at is None:
            return False

        now = time.monotonic()

        # The empty observation must be confirmed AFTER retirement.
        if now - retired_at < 2.0:
            return False

        if (
            self.latest_observation_health_at is None
            or self.latest_observation_health_at <= retired_at
        ):
            return False

        # Uses the existing source-stamp, reception-stamp,
        # session and continuity checks.
        if not self._fresh_empty_observation_available(
            minimum_duration_sec=2.0
        ):
            return False

        if self.navigation_active:
            return False

        # UNKNOWN and FORWARDING are not evidence of inactivity.
        if self.navigation_state not in {
            "WAITING",
            "REPLANNING",
        }:
            return False

        # An unfinished Nav2 cancellation remains a hard barrier.
        if self.nav2_cancel_requested:
            if not (
                self.nav2_cancel_confirmed
                and self.stable_stop_confirmed
            ):
                return False

            if (
                self.motion_state_received_at is None
                or now - self.motion_state_received_at > 0.5
                or self.motion_state_stamp_sec is None
                or self.stop_request_stamp_sec is None
                or self.motion_state_stamp_sec
                    < self.stop_request_stamp_sec
            ):
                return False

        # Do not bypass an unresolved speech transaction.
        if self.pending_speech_decision_id:
            return False

        if getattr(
            self,
            "dialogue_audio_recovery_pending",
            False,
        ):
            return False

        return True

    def _trajectory_execution_allowed(self, path_id):
        goal_id = str(self.active_navigation_goal_id or "").strip()
        path_id = str(path_id or "").strip()

        if (
            self.pending_hri_close is not None
            or self.pending_terminal_stop is not None
        ):
            return False

        if not goal_id or not path_id:
            return False

        if self.active_navigation_goal_pose is None:
            return False

        if self.candidate_navigation_goal_id != goal_id:
            return False

        if self.trajectory_validation != TRAJECTORY_VALIDATION_VALID:
            return False

        if not (
            path_id
            == self.validated_path_id
            == self.candidate_path_id
        ):
            return False

        if not self._trajectory_validation_is_fresh():
            return False

        # An ordinary recovery requires available observation
        # and no outstanding loss or retirement state.
        ordinary_recovery = (
            self.social_observation_available
            and not self.social_loss_expired
            and not self.social_episode_retired
            and not self.waiting_for_observation_reacquisition
        )

        # A retired episode has a separate, explicit recovery
        # path. It must correspond to the exact validated path.
        retired_recovery = (
            self.retired_recovery_path_id == path_id
            and self._retired_recovery_evidence_available()
        )

        return bool(ordinary_recovery or retired_recovery)


    # ================================================================
    # Replanning and resuming
    # ================================================================


    def _enter_replanning(
        self,
        reason,
        directional_constraint=None,
    ):
        """
        Start trajectory replanning.

        Human silence or an unrecognized response may trigger
        autonomous trajectory search.

        Planning is not authorization to move.
        Execution still requires exact, fresh validation.
        """
        if (
            self.pending_hri_close is not None
            or self.pending_terminal_stop is not None
        ):
            return

        if self.social_resolution_aborted:
            return

        if (
            not self.active_navigation_goal_id
            or self.active_navigation_goal_pose is None
        ):
            self._hold_route_search(
                "REPLAN_WITHOUT_PUBLIC_GOAL"
            )
            return

        retired_recovery = (
            self.retired_recovery_goal_id
            == self.active_navigation_goal_id
            and self._retired_recovery_evidence_available()
        )

        confirmed_passage_search = (
            self.interaction == "dialogue"
            and self.dialogue_outcome == "ALLOW_CONFIRMED"
            and self.confirmed_human_intent == "ALLOW_PASSAGE"
            and self.active_navigation_goal_id
            == self.social_episode_origin_goal_id
        )

        # Human silence does not authorize passage,
        # but it permits an autonomous planning attempt.
        autonomous_readaptation_search = (
            self.interaction == "dialogue"
            and self.dialogue_outcome in {
                "NO_RESPONSE",
                "NOT_UNDERSTOOD",
            }
            and self.pending_hri_close is None
            and self.pending_terminal_stop is None
            and not self.dialogue_audio_recovery_pending
            and not self.pending_speech_decision_id
        )

        # Fresh social evidence is not mandatory merely
        # to calculate a candidate trajectory.
        if not (
            retired_recovery
            or confirmed_passage_search
            or autonomous_readaptation_search
            or self._route_observation_is_fresh()
        ):
            self._hold_route_search(
                "REPLAN_REQUIRES_FRESH_SOCIAL_EVIDENCE"
            )
            return

        dialogue_outcome = self.dialogue_outcome

        dialogue_ready = dialogue_outcome in {
            "ALLOW_CONFIRMED",
            "DENY_CONFIRMED",
            "NO_RESPONSE",
            "NOT_UNDERSTOOD",
        }

        if (
            self.interaction == "dialogue"
            and self.human_proximity_active
            and not retired_recovery
        ):
            if not dialogue_ready:

                if dialogue_outcome == "NONE":
                    self._start_human_initiative_window()
                    return

                if dialogue_outcome == "PENDING":
                    return

                if dialogue_outcome == "DELIVERY_FAILED":
                    self._hold_route_search(
                        "DIALOGUE_TECHNICAL_RECOVERY_REQUIRED"
                    )
                    return

                self._hold_route_search(
                    "DIALOGUE_OUTCOME_NOT_RESOLVED"
                )
                return

        if self.route_attempt_count >= self.route_search_max_attempts:
            self._hold_route_search(
                "SOCIAL_ROUTE_ATTEMPTS_EXHAUSTED",
                reconsider=True,
            )
            return

        direction = self._normalize_direction(
            self.confirmed_human_direction
            if directional_constraint is None
            else directional_constraint
        )

        if (
            direction in {"LEFT", "RIGHT"}
            and direction == self.forbidden_human_direction
        ):
            self._hold_route_search(
                "HUMAN_FORBIDDEN_DIRECTION"
            )
            return

        self.route_generation_reason = (
            "ALTERNATIVE"
            if reason == "SOCIAL_ALTERNATIVE"
            else "REPLAN"
        )

        self.route_waiting_for_change = False

        self._invalidate_trajectory_validation(
            reason="REPLANNING_STARTED"
        )

        self.pending_replan_directional_constraint = direction

        self._transition_to(
            STATE_REPLANNING,
            reason=reason,
        )

        self._publish_collaborative_decision(
            action="REPLAN",
            outcome="TRAJECTORY_SEARCH",
            human_intent=self.confirmed_human_intent,
            directional_constraint=direction,
            path_id="",
            reason=reason,
        )

    def _enter_resuming(self, path_id):
        """Request exact-path execution and retain a recovery path."""

        goal_id = str(
            self.active_navigation_goal_id or ""
        ).strip()
        path_id = str(path_id or "").strip()

        if not self._trajectory_execution_allowed(path_id):
            self._hold_route_search(
                reason="RESUME_VALIDATION_BARRIER_FAILED",
                reconsider=True,
            )
            return

        # Used only if the FSM later returns to WAITING.
        self._schedule_social_reassessment()

        self._transition_to(
            STATE_RESUMING,
            reason="EXACT_TRAJECTORY_VALIDATED",
        )

        request_id = self._publish_collaborative_decision(
            action="RESUME",
            outcome="TRAJECTORY_VALIDATED",
            human_intent=self.confirmed_human_intent,
            directional_constraint=self.confirmed_human_direction,
            path_id=path_id,
            reason="EXACT_TRAJECTORY_VALID_AND_FRESH",
        )

        self.pending_gate_command = {
            "goal_id": goal_id,
            "action": "RESUME",
            "request_id": request_id,
            "path_id": path_id,
            "sent_at": time.monotonic(),
            "status": "SENT",
        }

    def _finish_episode_after_resume(self):
        """Remember resolution after actual validated execution starts."""

        goal_id = str(
            self.active_navigation_goal_id or ""
        ).strip()

        if not (
            goal_id
            and self.fsm_state == STATE_RESUMING
            and self.navigation_active
            and self.navigation_state == NAVIGATION_NAVIGATING
        ):
            return

        recovered_retired = bool(
            self.social_episode_retired
            and self.retired_recovery_goal_id == goal_id
            and self.retired_recovery_path_id
        )

        scope = self._social_scope_key()

        self.pending_gate_command = None
        self.initial_start_authorized_goal_id = goal_id
        self.route_waiting_for_change = False

        if recovered_retired:
            self._reset_episode()

        else:
            if self.human_proximity_active and all(scope):
                self.executed_social_scope = scope

            self.nav2_cancel_requested = False
            self.nav2_cancel_confirmed = False
            self.navigation_was_active_at_intrusion = False

            self._clear_pending_speech_transaction()

            self.dialogue_started_at = None
            self.waiting_for_human_initiative = False
            self.dialogue_retry_count = 0
            self.dialogue_speech_retry_count = 0

            # Consumed validation is not reusable for another execution.
            self._invalidate_trajectory_validation(
                reason="VALIDATED_TRAJECTORY_EXECUTION_STARTED"
            )

        self._transition_to(
            STATE_NAVIGATING,
            reason="NAVIGATION_RESUMED",
        )

    # ================================================================
    # Collaborative decisions
    # ================================================================

    def _publish_wait(self, reason):
        """
        Publish a non-terminal WAIT and retain its request identity.

        During a physical stop transaction, only the latest
        matching Gate HOLD_CONFIRMED may confirm cancellation.
        """
        if self.pending_gate_command is not None:
            if self.pending_gate_command.get("action") in {
                "START", "RESUME"
            }:
                self.pending_gate_command = None

        if not self.active_navigation_goal_id:
            self.get_logger().debug(
                "[COLLABORATION] Local WAIT only | "
                f"reason={reason}"
            )
            return ""

        request_id = self._publish_collaborative_decision(
            action="WAIT",
            outcome="WAITING",
            human_intent=self.confirmed_human_intent,
            directional_constraint=self.confirmed_human_direction,
            path_id="",
            reason=reason,
        )

        if (
            self.fsm_state
            == STATE_STOPPING_FOR_SOCIAL_CONFLICT
            and self.nav2_cancel_requested
        ):
            self.pending_stop_hold_request_id = request_id

            # A newer WAIT supersedes an older acknowledgement.
            self.nav2_cancel_confirmed = False

        return request_id
    
    def _publish_collaborative_decision(
        self,
        *,
        action,
        outcome,
        human_intent,
        directional_constraint,
        path_id,
        reason,
    ):
        action = str(
            action or ""
        ).strip().upper()

        path_id = str(
            path_id or ""
        ).strip()

        if action not in {
            "START",
            "STOP",
            "WAIT",
            "REPLAN",
            "RESUME"
        }:
            raise ValueError(
                f"Unsupported collaborative action: {action}"
            )

        if (
            self.pending_terminal_stop is not None
            and action not in {"STOP", "WAIT"}
        ):
            return ""

        navigation_authorized = action in {"START", "RESUME"}

        if action == "RESUME":
            if not path_id:
                raise RuntimeError(
                    "RESUME requires a path_id."
                )

            if not self._trajectory_execution_allowed(path_id):
                raise RuntimeError(
                    "RESUME rejected: exact-path validation barrier failed."
                )

        elif path_id:
            raise RuntimeError(
                f"{action} must not contain a path_id."
            )

        msg = CollaborativeDecision()

        msg.header.stamp = self.get_clock().now().to_msg()
        msg.interaction_id = self.active_interaction_id
        msg.navigation_goal_id = self.active_navigation_goal_id
        msg.request_id = f"REQ_{uuid.uuid4().hex}"
        msg.person_id = self.active_person_id

        msg.action = action
        msg.outcome = outcome
        msg.human_intent = human_intent
        msg.directional_constraint = directional_constraint
        msg.path_id = path_id
        msg.reason = reason

        msg.navigation_authorized = navigation_authorized
        msg.requires_replan = action == "REPLAN"

        self.collaborative_decision_pub.publish(msg)

        self.get_logger().info(
            "[COLLABORATIVE DECISION] "
            f"action={action} | "
            f"authorized={navigation_authorized} | "
            f"replan={msg.requires_replan} | "
            f"goal={self.active_navigation_goal_id!r} | "
            f"path_id={path_id!r} | "
            f"intent={human_intent} | "
            f"direction={directional_constraint} | "
            f"reason={reason}"
        )

        return msg.request_id

    def _watch_terminal_stop(self, now):
        pending = self.pending_terminal_stop

        if (
            pending is None
            or pending["goal_id"] != self.active_navigation_goal_id
        ):
            return

        elapsed = now - pending["requested_at"]

        if elapsed < 10.0:
            return

        if not self.terminal_stop_fault_reported:
            self.terminal_stop_fault_reported = True

            self.get_logger().error(
                "[STOP FAULT] Public termination unconfirmed; "
                "execution remains inhibited. "
                "Operator check required."
            )

        if (
            now - pending.get(
                "last_retry_at",
                pending["requested_at"]
            ) < 5.0
        ):
            return

        request_id = self._publish_collaborative_decision(
            action="STOP",
            outcome="TERMINATION_PENDING",
            human_intent=self.confirmed_human_intent,
            directional_constraint="NONE",
            path_id="",
            reason="PUBLIC_TERMINATION_UNCONFIRMED",
        )

        pending["request_id"] = request_id
        pending["last_retry_at"] = now

    def _send_pending_hri_close(self):
        pending = self.pending_hri_close

        if pending is None:
            return

        payload = {
            key: pending[key]
            for key in (
                "close_id",
                "interaction_id",
                "request_id",
                "reason"
            )
        }

        payload.update(
            version=1,
            command="CLOSE"
        )

        self.hri_dialogue_control_pub.publish(
            String(data=json.dumps(payload))
        )

    def _fresh_observation_health_available(self):
        if self.latest_observation_health not in {
            "EMPTY",
            "OCCUPIED"
        }:
            return False

        if (
            self.latest_observation_health_at is None
            or self.health_stamp_ns is None
        ):
            return False

        receipt_age = (
            time.monotonic()
            - self.latest_observation_health_at
        )

        source_age = (
            self.get_clock().now().nanoseconds
            - self.health_stamp_ns
        ) / 1e9

        return (
            0 <= receipt_age <= self.social_decision_max_age_sec
            and -0.1 <= source_age <= self.social_decision_max_age_sec
        )



    # ================================================================
    # Timers
    # ================================================================

    def _evaluate_pending_gate_command(self, now):
        pending = self.pending_gate_command

        if pending is None:
            return

        goal_id = pending["goal_id"]
        action = pending["action"]
        request_id = pending["request_id"]

        if goal_id != self.active_navigation_goal_id:
            self.pending_gate_command = None
            return

        elapsed = now - pending["sent_at"]

        if action == "START":
            timeout = 2.0

        elif action == "RESUME":
            # Includes deferred acceptance and actual start.
            timeout = 10.0

        else:
            timeout = 5.0

        if elapsed < timeout:
            return

        self.pending_gate_command = None

        self.get_logger().error(
            "[GATE WATCHDOG] Transaction timeout | "
            f"action={action} | goal={goal_id} | "
            f"request={request_id} | elapsed={elapsed:.1f}s"
        )

        if action == "START":
            # The Gate may still have received the request.
            # Never assume this timeout means Nav2 is idle.
            self.initial_start_authorized_goal_id = ""

            self.initial_start_next_attempt_at = now + 0.5

            # An explicit WAIT revokes any unconsumed authorization.
            self._publish_wait(
                reason="START_ACK_TIMEOUT"
            )
            return

        if action == "RESUME":
            self._publish_wait(
                reason="RESUME_START_TIMEOUT"
            )

            self._invalidate_trajectory_validation(
                reason="RESUME_START_TIMEOUT"
            )

            self._transition_to(
                STATE_WAITING,
                reason="RESUME_START_TIMEOUT",
            )

    def _timer_callback(self):
        """
        Periodic supervision of collaboration transactions.

        Responsibilities:
            - Verify the physical stop barrier.
            - Monitor HRI dialogue closure.
            - Supervise social observation loss.
            - Process episode retirement.
            - Monitor speech completion.
            - Supervise Gate transactions.
            - Release deferred CLEAR recovery.
            - Supervise bounded trajectory resolution.
            - Evaluate initial navigation authorization.

        Timeouts never authorize motion by themselves.
        """
        now = time.monotonic()

        # 1. PHYSICAL STOP BARRIER
        self._evaluate_stop_barrier()

        # 2. HRI CLOSE SUPERVISION
        self._watch_hri_close(now)
        self._watch_terminal_stop(now)

        # 3. SOCIAL OBSERVATION SUPERVISION
        self._evaluate_social_loss_timeout(now)
        self._apply_pending_social_retirement()

        # 4. SPEECH AND IDLE EPISODE SUPERVISION
        self._evaluate_initial_announcement_timeout(now)
        self._evaluate_idle_episode_retirement()

        # 5. NAVIGATION GATE TRANSACTION SUPERVISION
        self._evaluate_pending_gate_command(now)

        # 6. DEFERRED CLEAR RECOVERY
        self._evaluate_deferred_clear_replan()

        # 7. BOUNDED SOCIAL ROUTE RESOLUTION
        self._evaluate_route_resolution(now)

        # 8. NEW MOTION-RELATED WORK
        if not self.social_resolution_aborted:
            self._evaluate_retired_episode_recovery()
            self._evaluate_conflict_persistence(now)
            self._evaluate_initial_goal_start()

        # 9. FSM DIAGNOSTIC WATCHDOG
        monitored_states = {
            STATE_STOPPING_FOR_SOCIAL_CONFLICT,
            STATE_WAITING,
            STATE_RESUMING,
            STATE_REPLANNING,
            STATE_VALIDATING_TRAJECTORY,
        }

        if self.fsm_state not in monitored_states:
            return

        elapsed = now - self.state_entered_at

        if elapsed < 30.0:
            return

        self.get_logger().warning(
            "[FSM WATCHDOG] Prolonged state | "
            f"state={self.fsm_state} | "
            f"elapsed={elapsed:.1f}s | "
            f"goal={self.active_navigation_goal_id!r} | "
            f"nav_active={self.navigation_active} | "
            f"social_retired={self.social_episode_retired} | "
            f"hri_close_pending="
            f"{self.pending_hri_close is not None} | "
            f"generation_uncertain="
            f"{bool(self.uncertain_generation_request_id)}",
            throttle_duration_sec=10.0,
        )
    def _evaluate_deferred_clear_replan(self):
        goal_id = self.deferred_clear_replan_goal_id

        if not goal_id:
            return

        if goal_id != self.active_navigation_goal_id:
            self.deferred_clear_replan_goal_id = ""
            return

        if (
            self.social_resolution_aborted
            or self.pending_terminal_stop is not None
            or self.pending_hri_close is not None
            or self.pending_speech_decision_id
            or self.fsm_state != STATE_WAITING
            or self.human_proximity_active
            or not self._route_observation_is_fresh()
        ):
            return

        if (
            self.nav2_cancel_requested
            and not (
                self.nav2_cancel_confirmed
                and self.stable_stop_confirmed
            )
        ):
            return

        self.deferred_clear_replan_goal_id = ""

        self._enter_replanning(
            reason="SOCIAL_CLEAR_HRI_CLOSED",
            directional_constraint="NONE"
        )
    def _evaluate_social_loss_timeout(self, now):
        if not self.human_proximity_active:
            return

        if self.social_observation_available:
            return

        if self.social_loss_expired:
            return

        if self.social_loss_started_at is None:
            self.social_loss_started_at = now
            return

        elapsed = now - self.social_loss_started_at

        if elapsed < self.social_loss_timeout_sec:
            return

        if self.fsm_state == STATE_ANNOUNCING_DETECTION:
            return

        self.social_loss_expired = True
        self._apply_pending_social_retirement()
        self._invalidate_trajectory_validation(reason="SOCIAL_LOSS_TIMEOUT")
        if (
            self.interaction == "dialogue"
            and self.fsm_state in {
                STATE_WAITING_FOR_HUMAN,
                STATE_WAITING_FOR_CONFIRMATION,
                STATE_ASKING,
                STATE_CONFIRMING_HUMAN_INTENT,
            }
            and self.active_interaction_id
            and self.pending_dialogue_request_id
        ):
            # Social observation has expired, but the HRI
            # request retains ownership of its own timeout.
            #
            # Do not reset dialogue_started_at.
            # Do not clear pending_dialogue_request_id.
            # Do not interpret observation loss as silence.
            #
            # The trajectory remains invalidated.
            self.get_logger().warning(
                "[SOCIAL LOSS] Observation timeout reached; "
                "active HRI transaction retained | "
                f"interaction={self.active_interaction_id} | "
                f"request={self.pending_dialogue_request_id}"
            )
            return

        self._clear_pending_human_preference()

        #self.confirmed_human_intent = "UNKNOWN"
        #self.confirmed_human_direction = "NONE"
        #self.active_interaction_id = ""
        #self.active_dialogue_request_type = ""
        self.active_alternative_direction = "NONE"

        self.dialogue_started_at = None
        self.waiting_for_human_initiative = False
        self.dialogue_retry_count = 0
        self.dialogue_speech_retry_count = 0
        self.pending_replan_directional_constraint = "NONE"
        self.waiting_for_observation_reacquisition = True

        self.get_logger().warning(
            "[SOCIAL LOSS] Interaction expired; public goal retained | "
            f"goal={self.active_navigation_goal_id!r} | "
            f"elapsed={elapsed:.2f}s"
        )

        if self.active_navigation_goal_id:
            self._publish_wait(reason="SOCIAL_LOSS_TIMEOUT_HOLD")

        self._transition_to(
            STATE_WAITING,
            reason="SOCIAL_LOSS_EXPIRED_WAITING_FOR_OBSERVATION",
        )

    def _evaluate_initial_announcement_timeout(self, now):
        if self.fsm_state != STATE_ANNOUNCING_DETECTION:
            return

        decision_id = str(self.pending_speech_decision_id or "").strip()

        if not decision_id:
            return

        if not self.pending_speech_started:
            started_at = self.pending_speech_requested_at

            if started_at is None:
                return

            elapsed = now - started_at
            timeout = self.initial_speech_start_timeout_sec
            reason = "SPEECH_START_TIMEOUT_UNCONFIRMED"

        else:
            started_at = self.pending_speech_started_at

            if started_at is None:
                return

            elapsed = now - started_at
            timeout = self.speech_completion_timeout_sec
            reason = "SPEECH_COMPLETION_TIMEOUT_UNCONFIRMED"

        if elapsed < timeout:
            return

        self.get_logger().error(
            "[SPEECH BARRIER] Timeout | "
            f"decision_id={decision_id} | elapsed={elapsed:.2f}s | "
            f"limit={timeout:.2f}s | reason={reason}"
        )

        self._hold_after_speech_uncertainty(
            decision_id=decision_id,
            reason=reason,
            request_cancel=True,
        )

    # ================================================================
    # Episode reset
    # ================================================================

    def _reset_episode(self):
        # Preserve public goal identity and transport ordering.
        self.active_interaction_id = ""
        self.active_decision_id = ""
        self.active_person_id = ""
        self.announcement_decision_id = ""

        self.human_proximity_active = False
        self.social_observation_available = True
        self.observation_loss_held_goal_id = ""
        self.waiting_for_observation_reacquisition = False
        self.waiting_for_human_initiative = False
        self.conflict_observed_since = None
        self.persistent_conflict_handled = False

        self.announcement_social_decision = None
        self.clear_pending_after_announcement = False

        self.nav2_cancel_requested = False
        self.nav2_cancel_confirmed = False
        self.navigation_was_active_at_intrusion = False

        self.stable_stop_confirmed = False
        self.stable_stop_source = ""
        self.motion_state_received_at = None
        self.motion_state_stamp_sec = None
        self.stop_request_stamp_sec = None

        self.announced_observe = False
        self.finished_announcement_attempts = set()
        self.announcement_context_key = None
        self.deferred_social_goal_id = ""
        self.announced_navigation_goal_id = ""
        self.current_announcement_action = ""

        self.dialogue_started_at = None
        self.dialogue_retry_count = 0
        self.dialogue_speech_retry_count = 0
        self.active_dialogue_request_type = ""
        self.active_alternative_direction = "NONE"
        self.initial_speech_retry_count = 0

        self.pending_human_intent = "UNKNOWN"
        self.pending_human_direction = "NONE"
        self.confirmed_human_intent = "UNKNOWN"
        self.confirmed_human_direction = "NONE"

        self.pending_replan_directional_constraint = "NONE"
        self.social_loss_started_at = None
        self.social_loss_expired = False
        self.pending_retired_episode = None
        self.social_episode_retired = False
        self.retired_recovery_goal_id = ""
        self.retired_recovery_path_id = ""
        
        self.retired_episode_retired_at = None
        self.retired_recovery_attempt_count = 0
        self.retired_recovery_last_attempt_at = 0.0
        self.retired_recovery_retry_delay_sec = 3.0
        self.retired_recovery_exhaustion_logged = False

        self._clear_pending_speech_transaction()
        self._invalidate_trajectory_validation(
            reason="COLLABORATION_EPISODE_RESET"
        )
        self._reset_route_search()
        self.social_episode_origin_goal_id = ""

    # ================================================================
    # Utilities
    # ================================================================

    def _ensure_interaction_id(self):
        if not self.active_interaction_id:
            self.active_interaction_id = f"COLLAB_{uuid.uuid4().hex[:12]}"

    def _clear_pending_human_preference(self):
        self.pending_human_intent = "UNKNOWN"
        self.pending_human_direction = "NONE"

    def _normalize_direction(self, direction):
        normalized = str(direction or "").strip().upper()

        if not normalized:
            return "NONE"

        if normalized not in self.VALID_DIRECTIONS:
            self.get_logger().warning(
                f"Unsupported human direction {direction!r}; using NONE."
            )
            return "NONE"

        return normalized

    def _is_fresh_social_decision(self, msg):
        if self._message_is_recent(msg, self.social_decision_max_age_sec):
            return True

        stamp_sec = self._stamp_to_seconds(msg.header.stamp)
        age = self.get_clock().now().nanoseconds / 1e9 - stamp_sec

        self.get_logger().warning(
            "[SOCIAL DECISION REJECTED] "
            f"id={msg.decision_id!r} | event={msg.event!r} | "
            f"age={age:.3f}s | stamp={stamp_sec:.9f}",
            throttle_duration_sec=2.0,
        )
        return False

    @staticmethod
    def _stamp_to_seconds(stamp):
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def main(args=None):
    rclpy.init(args=args)
    node = CollaborationManager()

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