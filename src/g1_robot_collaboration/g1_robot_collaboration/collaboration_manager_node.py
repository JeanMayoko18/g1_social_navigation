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
import copy
import os
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
    TrackedPersonArray,
    TrajectoryGenerationRequest,
    TrajectorySocialValidation,
)


from g1_social_msgs.msg import (
    DirectionalConstraintRequest,
    DirectionalConstraintStatus,
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

        self.declare_parameter(
            "planning_reconciliation_timeout_sec",
            120.0,
        )

        self.planning_reconciliation_timeout_sec = float(
            self.get_parameter(
                "planning_reconciliation_timeout_sec"
            ).value
        )

        if (
            not math.isfinite(
                self.planning_reconciliation_timeout_sec
            )
            or self.planning_reconciliation_timeout_sec <= 0.0
        ):
            raise ValueError(
                "planning_reconciliation_timeout_sec "
                "must be finite and positive."
            )

        self.planning_reconciliation = None

        self.generation_status_request_pub = self.create_publisher(
            String,
            "/social_navigation/trajectory_generation_status_request",
            10,
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

        self.individual_participants = {}

        self.individual_source_session = ""
        self.individual_source_stamp_ns = 0
        self.individual_source_fresh = False
        self.individual_source_complete = False
        self.individual_received_at = None

        self.individual_retired_source_sessions = set()

        #self.retired_social_episode_ids = set()

        self.active_decision_id = ""
        self.active_person_id = ""

        self.latest_observation_health = "UNKNOWN"
        self.latest_observation_health_at = None
        self.empty_observation_since = None
        self.health_session = ""
        self.route_geometry_person_id = ""
        self.health_stamp_ns = None

        self.retired_recovery_goal_id = ""
        self.retired_recovery_path_id = ""

        self.retired_episode_retired_at = None
        self.retired_recovery_attempt_count = 0
        self.retired_recovery_last_attempt_at = 0.0
        self.retired_recovery_retry_delay_sec = 3.0
        self.retired_recovery_exhaustion_logged = False
        self._handled_persistent_person_episodes = set()
        self.reactive_next_search_at = 0.0
        self.reactive_search_signature = None

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

        self._active_actor_scope = None
        self._actor_semantics = {}
        self.social_observation_complete = True
        
        # 
        self.mode_d_cases = {}

        # 
        self.mode_d_hri_owner = None

        # (interaction_id, request_id) -> identité individuelle.
        self.mode_d_hri_turns = {}

        # 
        self.mode_d_hri_closures = {}

        # 
        self.mode_d_anticipation_queues = {}


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


        self.pending_hri_close = None
        self.hri_close_fault_reported = False
        self.deferred_clear_replan_goal_id = ""

        # Terminaison publique supervisée.
        self.pending_terminal_stop = None
        self.terminal_stop_fault_reported = False


        self.dialogue_audio_recovery_pending = False
        self.audio_recovery_interaction_id = ""
        self.audio_recovery_request_id = ""


        self.retired_episode_retired_at = None
        self.retired_recovery_attempt_count = 0
        self.retired_recovery_last_attempt_at = 0.0
        self.retired_recovery_retry_delay_sec = 3.0
        self.retired_recovery_exhaustion_logged = False
        #self.mode_c_cached_candidate = None
        self.reactive_cached_candidate = None
        self.mode_c_last_checked_source_ns = 0
        self.mode_c_last_revalidation_at = 0.0


        # ------------------------------------------------------------
        # Subscriptions
        # ------------------------------------------------------------

        self.hri_dialogue_control_pub = self.create_publisher(
            String,
            "/hri/dialogue_control",
            20,
        )
        
        self.mode_c_revalidation_pub = self.create_publisher(
            CandidateTrajectory,
            "/social_navigation/trajectory_revalidation_request",
            10,
        )

        self.hri_dialogue_control_result_sub = self.create_subscription(
            String,
            "/hri/dialogue_control_result",
            self._hri_dialogue_control_result_callback,
            20,
        )

        self.individual_situation_sub = self.create_subscription(
            String,
            "/social_perception/individual_social_situation",
            self._individual_situation_callback,
            10,
        )
        self.social_decision_sub = self.create_subscription(
            SocialNavigationDecision,
            self.social_decision_topic,
            self._social_decision_callback,
            10,
        )
        # Subscribe to spatial identity observations for Mode D.
        self.directional_identity_snapshot = None
        self.directional_identity_received_at = None

        self.directional_identity_sub = self.create_subscription(
            TrackedPersonArray,
            "/social_perception/tracked_persons",
            self._directional_identity_callback,
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

        self._initialize_social_reassessment()

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

    def _individual_situation_callback(self, msg):
        """Mémoriser les épisodes individuels sans réinitialiser le HRI."""

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

            if not session or stamp_ns <= 0:
                return

            if (
                not isinstance(fresh, bool)
                or not isinstance(complete, bool)
            ):
                return

            age = (
                self.get_clock().now().nanoseconds - stamp_ns
            ) / 1e9

            if not (
                -0.10 <= age <= self.social_decision_max_age_sec
            ):
                return

            rows = data["participants"]

            if (
                not isinstance(rows, list)
                or len(rows) > 512
            ):
                return

            participants = {}

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
                    or type(row["persistent"]) is not bool
                ):
                    return

                identity = (key[0], key[1], key[2])

                if identity in participants:
                    return

                participants[identity] = {
                    "episode_id": str(row["episode_id"]),
                    "unresolved": row["unresolved"],
                    "observed_now": row["observed_now"],
                    "persistent": row["persistent"],
                    "observation_status": str(
                        row["observation_status"]
                    ),
                    "conflict_status": str(
                        row["conflict_status"]
                    ),
                    "retired": row["retired"],
                    "transition": str(row["transition"]),
                }

        except (
            ValueError,
            TypeError,
            KeyError,
            AttributeError,
        ):
            return

        if session in self.individual_retired_source_sessions:
            return

        if session == self.individual_source_session:
            if stamp_ns < self.individual_source_stamp_ns:
                return

        elif self.individual_source_session:
            self.individual_retired_source_sessions.add(
                self.individual_source_session
            )

        previous = self.individual_participants

        new_conflicts = [
            identity
            for identity, state in participants.items()
            if (
                state["unresolved"]
                and state["observed_now"]
                and state["conflict_status"] in {
                    "CONFLICT",
                    "UNRESOLVED",
                }
                and (
                    identity not in previous
                    or not previous[identity]["unresolved"]
                )
            )
        ]

        self.individual_participants = participants
        self.individual_source_session = session
        self.individual_source_stamp_ns = stamp_ns
        self.individual_source_fresh = fresh
        self.individual_source_complete = complete
        self.individual_received_at = time.monotonic()

        if (
            new_conflicts
            and self.active_navigation_goal_id
            and self.fsm_state in {
                STATE_NAVIGATING,
                STATE_RESUMING,
            }
            and (
                self.navigation_active
                or self.fsm_state == STATE_RESUMING
            )
            and self.pending_terminal_stop is None
        ):
            self.get_logger().warning(
                "[INDIVIDUAL SOCIAL] New conflict during navigation | "
                f"identities={new_conflicts!r}"
            )

            self._invalidate_trajectory_validation(
                reason="NEW_INDIVIDUAL_SOCIAL_CONFLICT"
            )

            self.nav2_cancel_requested = True
            self.nav2_cancel_confirmed = False
            self.stable_stop_confirmed = False

            self.stop_request_stamp_sec = (
                self.get_clock().now().nanoseconds / 1e9
            )

            self._transition_to(
                STATE_STOPPING_FOR_SOCIAL_CONFLICT,
                reason="NEW_INDIVIDUAL_SOCIAL_CONFLICT",
            )

            self._publish_wait(
                reason="NEW_INDIVIDUAL_SOCIAL_CONFLICT"
            )

            self._schedule_social_reassessment()
            
        for identity, row in participants.items():
            old = previous.get(identity)

            if (
                row["transition"]
                or (
                    old is not None
                    and (
                        old["unresolved"],
                        old["observation_status"],
                    ) != (
                        row["unresolved"],
                        row["observation_status"],
                    )
                )
            ):
                self.get_logger().info(
                    "[INDIVIDUAL SOCIAL] "
                    f"person={identity!r} | "
                    f"episode={row['episode_id']} | "
                    f"transition="
                    f"{row['transition'] or 'STATE_CHANGE'} | "
                    f"unresolved={row['unresolved']} | "
                    f"observation={row['observation_status']}"
                )

    def _individual_scene_current(self):
        if (
            not self.individual_source_session
            or not self.individual_source_fresh
            or self.individual_received_at is None
        ):
            return False

        age = (
            self.get_clock().now().nanoseconds
            - self.individual_source_stamp_ns
        ) / 1e9

        receipt_age = (
            time.monotonic() - self.individual_received_at
        )

        return bool(
            -0.10 <= age <= self.social_decision_max_age_sec
            and 0.0 <= receipt_age
            <= self.social_decision_max_age_sec
        )

    def _individual_scene_blocks_motion(self):
        # L'absence ou la péremption des preuves bloque
        # l'autorisation, jamais le simple calcul de trajectoire.
        if not self._individual_scene_current():
            return True

        return any(
            state["unresolved"]
            and (
                not state["observed_now"]
                or state["observation_status"] != "OBSERVED"
                or state["retired"]
            )
            for state in self.individual_participants.values()
        )

    def _individual_scene_all_clear(self):
        # Un CLEAR collectif exige que tous les épisodes
        # individuels soient effectivement résolus.
        return bool(
            self._individual_scene_current()
            and self.individual_source_complete
            and not any(
                state["unresolved"]
                for state in self.individual_participants.values()
            )
        )

    def _reactive_planning_ready(self):
        """Allow planning only after a verified navigation hold."""

        goal_id = str(self.active_navigation_goal_id or "").strip()
        nav = self.latest_navigation_state
        now = time.monotonic()

        if (
            not goal_id
            or self.active_navigation_goal_pose is None
            or self.navigation_active
            or nav is None
            or str(nav.navigation_goal_id or "").strip() != goal_id
            or not self._message_is_recent(nav, 1.0)
            or self.navigation_state not in {
                "WAITING",
                "SOCIAL_HOLD",
                "REPLANNING",
            }
            or not self.stable_stop_confirmed
            or self.motion_state_received_at is None
            or self.motion_state_stamp_sec is None
            or now - self.motion_state_received_at > 0.5
            or self.pending_terminal_stop is not None
            or self.pending_gate_command is not None
            or self.pending_hri_close is not None
            or self.pending_speech_decision_id
            or self.pending_dialogue_request_id
            or self.dialogue_audio_recovery_pending
            or self.pending_generation_request_id
            or self.uncertain_generation_request_id
            or self.social_resolution_aborted
        ):
            return False

        motion_age = (
            self.get_clock().now().nanoseconds / 1e9
            - self.motion_state_stamp_sec
        )

        if not -0.1 <= motion_age <= 0.5:
            return False

        if self.nav2_cancel_requested and not (
            self.nav2_cancel_confirmed
            and self.stop_request_stamp_sec is not None
            and self.motion_state_stamp_sec
            >= self.stop_request_stamp_sec
        ):
            return False

        return True

    def _individual_search_signature(self):
        """Represent social changes without using camera frame counters."""

        return (
            self.active_navigation_goal_id,
            self.individual_source_session,
            tuple(
                sorted(
                    (
                        key,
                        row["episode_id"],
                        row["unresolved"],
                        row["observation_status"],
                        row["conflict_status"],
                        row["retired"],
                        row["persistent"],
                    )
                    for key, row in self.individual_participants.items()
                )
            ),
        )

    def _reactive_search_tick(self, now):
        """Continue deferred HRI fallback and preserve confirmed route semantics."""
        deferred = getattr(
            self,
            "_readaptation_after_hri_close",
            None,
        )

        if deferred is not None:
            goal_id, scope, reason = deferred

            if (
                goal_id != self.active_navigation_goal_id
                or scope != self._social_scope_key()
            ):
                self._readaptation_after_hri_close = None
                return

            if (
                self.fsm_state == STATE_WAITING
                and self.pending_hri_close is None
                and not self.hri_close_fault_reported
                and not self.pending_dialogue_request_id
                and not self.dialogue_audio_recovery_pending
                and not self.pending_speech_decision_id
                and self.pending_terminal_stop is None
                and not self.social_resolution_aborted
            ):
                self._begin_autonomous_readaptation_announcement(
                    reason=reason
                )

            return

        if (
            self.fsm_state != STATE_WAITING
            or not self.route_waiting_for_change
            or not self._reactive_planning_ready()
            or now < self.reactive_next_search_at
            or self.anticipatory_pending_request_id
            or self.anticipatory_uncertain_request_id
            or self.announcement_context_key is not None
            or self.current_announcement_action
        ):
            return

        if self.interaction == "dialogue":
            if self.dialogue_outcome == "NONE":
                if self._route_observation_is_fresh():
                    self._start_human_initiative_window()
                return

            if self.dialogue_outcome not in {
                "ALLOW_CONFIRMED",
                "DENY_CONFIRMED",
                "NO_RESPONSE",
                "NOT_UNDERSTOOD",
            }:
                return

            if (
                self.dialogue_outcome == "ALLOW_CONFIRMED"
                and not self._permission_applies_to_current_episode()
            ):
                return

        signature = self._individual_search_signature()

        if signature != self.reactive_search_signature:
            self.reactive_search_signature = signature
            self.route_attempt_count = 0
            self.route_tried_directions.clear()

        if self.route_attempt_count >= self.route_search_max_attempts:
            self._request_passage_assistance()
            return

        self.reactive_next_search_at = now + 2.0

        if self.dialogue_outcome == "DENY_CONFIRMED":
            self._request_next_social_alternative()
            return

        direction = (
            self.confirmed_human_direction
            if self.dialogue_outcome == "ALLOW_CONFIRMED"
            else "NONE"
        )

        self._enter_replanning(
            reason="INDIVIDUAL_REACTIVE_ROUTE_SEARCH",
            directional_constraint=direction,
        )

    def _mode_c_reactive_generation_allowed(self):
        """
        Allow mode-C planning after verifying the physical hold
        and the absence of conflicting transactions.

        This method never authorizes movement.
        """
        return bool(
            self.interaction == "none"
            and self.human_proximity_active
            and self.announcement_context_key is None
            and not self.current_announcement_action
            and not self.anticipatory_pending_request_id
            and not self.anticipatory_uncertain_request_id
            and self._reactive_planning_ready()
        )
    
    def _mode_c_reactive_tick(self):
        """Relancer une recherche après perte de perception."""

        if (
            self.fsm_state != STATE_WAITING
            or self.route_attempt_count != 0
            or self.pending_generation_request_id
            or self.uncertain_generation_request_id
            or not self._mode_c_reactive_generation_allowed()
        ):
            return

        self._enter_replanning(
            reason="MODE_C_REACTIVE_AFTER_OBSERVATION_LOSS",
            directional_constraint="NONE",
        )
        
    def _route_observation_is_fresh(self):
        """
        Require current factual evidence for route resolution.

        Semantic consent and historical episode state never
        establish current geometric observation.
        """
        if (
            self.individual_source_session
            and self._individual_scene_blocks_motion()
        ):
            return False

        if (
            not self.social_observation_available
            or self.social_loss_expired
            or self.social_episode_retired
            or self.waiting_for_observation_reacquisition
            or not self.social_observation_complete
        ):
            return False

        if not self._fresh_observation_health_available():
            return False

        # Observation health and canonical events must belong
        # to the same DecisionNode producer session.
        if (
            self.social_state_session
            and self.health_session != self.social_state_session
        ):
            return False

        if not self.human_proximity_active:
            # OCCUPIED is compatible with observed clearance.
            # A person may remain visible outside the release boundary.
            return True

        decision = self.latest_social_decision

        if decision is None:
            return False

        if not self._message_is_recent(
            decision,
            self.social_decision_max_age_sec,
        ):
            return False

        event = normalize_social_navigation_event(
            decision.event
        )

        if event not in {
            SOCIAL_EVENT_INTRUSION_CONFIRMED,
            SOCIAL_EVENT_OBSERVATION_REACQUIRED,
        }:
            return False

        try:
            session, episode, sequence_text = (
                str(decision.decision_id).split(":")
            )
            sequence = int(sequence_text)

            position = decision.person_position

            valid_geometry = (
                bool(str(decision.person_id).strip())
                and bool(str(decision.header.frame_id).strip())
                and decision.tracking_state == "TRACKED"
                and bool(decision.social_conflict_active)
                and all(
                    math.isfinite(float(value))
                    for value in (
                        position.x,
                        position.y,
                        position.z,
                        decision.person_distance,
                        decision.social_radius,
                    )
                )
                and float(decision.person_distance) >= 0.0
                and float(decision.social_radius) > 0.0
            )

        except (
            ValueError,
            TypeError,
            AttributeError,
            OverflowError,
        ):
            return False

        return bool(
            valid_geometry
            and session == self.social_state_session
            and episode == self.social_state_episode
            and sequence == self.social_state_sequence
            and self.announcement_context_key is None
            and not self.current_announcement_action
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

        if not (
            self._route_observation_is_fresh()
            or self._reactive_planning_ready()
        ):
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

    def _watch_planning_reconciliation(self, now):
        """Reconcile the exact request against its original public goal."""
        if (
            self.fsm_state == STATE_REPLANNING
            and getattr(self, "_replan_hold_ready", None)
                == self.active_navigation_goal_id
            and not self.pending_generation_request_id
            and not self.uncertain_generation_request_id
        ):
            self._request_current_trajectory_validation(
                generation_reason="REPLAN",
                directional_constraint=(
                    self.pending_replan_directional_constraint
                ),
            )

        request_id = self.uncertain_generation_request_id

        if (
            not request_id
            and self.pending_generation_request_id
            and self.fsm_state not in {
                STATE_REPLANNING,
                STATE_VALIDATING_TRAJECTORY,
            }
        ):
            request_id = self.pending_generation_request_id
            self.uncertain_generation_request_id = request_id
            self.pending_generation_request_id = ""

        if not request_id:
            self.planning_reconciliation = None
            return

        goal_id = getattr(
            self,
            "_generation_owners",
            {},
        ).get(request_id)

        if not goal_id:
            self.get_logger().error(
                "[PLANNING] Missing request owner; "
                "new generation remains blocked.",
                throttle_duration_sec=5.0,
            )
            return

        scope = (goal_id, request_id)
        watch = self.planning_reconciliation

        if watch is None or watch["scope"] != scope:
            watch = {
                "scope": scope,
                "started_at": now,
                "next_query_at": now,
                "queries": 0,
            }
            self.planning_reconciliation = watch

        if (
            now - watch["started_at"]
            >= self.planning_reconciliation_timeout_sec
        ):
            if (
                goal_id == self.active_navigation_goal_id
                and self.pending_terminal_stop is None
            ):
                self._request_terminal_stop(
                    reason="PLANNING_RECONCILIATION_EXHAUSTED"
                )

        # Continue querying terminal evidence even after STOP.
        if now < watch["next_query_at"]:
            return

        self.generation_status_request_pub.publish(
            String(
                data=json.dumps({
                    "version": 1,
                    "navigation_goal_id": goal_id,
                    "request_id": request_id,
                })
            )
        )

        watch["queries"] += 1
        watch["next_query_at"] = now + 3.0

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
            self.get_logger().error(
                "[SOCIAL RESOLUTION] Deadline exceeded | "
                f"goal={goal_id} | "
                f"elapsed={elapsed:.1f}s | "
                "requesting terminal controlled stop"
            )

            self._request_terminal_stop(
                reason="SOCIAL_RESOLUTION_DEADLINE_EXCEEDED"
            )
            return False

        return True

    def _initial_social_start_allowed(self):
        """
        Allow the initial navigation start for a fresh public goal.

        Historical social states cannot block START. A new active
        intrusion or an unfinished physical-stop transaction
        must still prevent navigation authorization.
        """
        goal_id = str(self.active_navigation_goal_id or "").strip()

        if not goal_id or self.active_navigation_goal_pose is None:
            return False

        if (
            self.nav2_cancel_requested
            or self.pending_stop_hold_request_id
            or self.pending_terminal_stop is not None
            or self.social_resolution_aborted
        ):
            return False

        # A newly confirmed intrusion belonging to this goal
        # remains an active safety barrier.
        if (
            self.human_proximity_active
            and self.social_episode_origin_goal_id == goal_id
        ):
            return False

        # Only fresh, actually observed conflicts may block a start.
        if self._individual_scene_current():
            for state in self.individual_participants.values():
                if (
                    state["unresolved"]
                    and state["observed_now"]
                    and state["conflict_status"] in {
                        "CONFLICT",
                        "UNRESOLVED",
                    }
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
     
    def _try_cached_path_revalidation(self, now):
        """
        Select a reconciled Mode-D candidate and request fresh
        social validation without starting another Nav2 plan.

        No cache entry authorizes execution by itself.
        """

        if not (
            self.interaction == "dialogue"
            and self.anticipatory_planning_enabled
            and self.fsm_state == STATE_WAITING
            and self.dialogue_outcome == "ALLOW_CONFIRMED"
            and self.confirmed_human_intent == "ALLOW_PASSAGE"
            and self._permission_applies_to_current_episode()
        ):
            return False

        scope = self._anticipation_scope()

        if (
            scope is None
            or getattr(self, "_closed_hri_scope", None) != scope
        ):
            return False

        if (
            self.pending_hri_close is not None
            or self.hri_close_fault_reported
            or self.pending_dialogue_request_id
            or self.dialogue_audio_recovery_pending
            or self.pending_speech_decision_id
            or self.pending_terminal_stop is not None
            or self.pending_gate_command is not None
            or self.pending_generation_request_id
            or self.uncertain_generation_request_id
            or self.anticipatory_pending_request_id
            or getattr(
                self, "anticipatory_uncertain_request_id", ""
            )
            or self.navigation_active
            or self.social_resolution_aborted
            or not self._route_observation_is_fresh()
            or not self._fresh_observation_health_available()
        ):
            return False

        nav = self.latest_navigation_state

        if (
            nav is None
            or str(nav.navigation_goal_id or "").strip() != scope[0]
            or not self._message_is_recent(nav, 1.0)
            or bool(nav.navigation_active)
            or self.navigation_state not in {
                "WAITING",
                "SOCIAL_HOLD",
                "REPLANNING",
            }
        ):
            return False

        # Preserve the physical hold barrier.
        if (
            not self.stable_stop_confirmed
            or self.motion_state_received_at is None
            or self.motion_state_stamp_sec is None
            or now - self.motion_state_received_at > 0.5
        ):
            return False

        source_age = (
            self.get_clock().now().nanoseconds / 1e9
            - self.motion_state_stamp_sec
        )

        if not -0.1 <= source_age <= 0.5:
            return False

        if self.nav2_cancel_requested and not (
            self.nav2_cancel_confirmed
            and self.stop_request_stamp_sec is not None
            and self.motion_state_stamp_sec
                >= self.stop_request_stamp_sec
        ):
            return False

        candidate = self._select_anticipatory_candidate()

        if candidate is None:
            return False

        request_id = str(candidate.request_id).strip()
        path_id = str(candidate.path_id).strip()

        entry = self.anticipatory_candidates.get(request_id)

        if entry is None or entry.get("scope") != scope:
            return False

        # Do not retry the same candidate against the same source
        # frame, or spin the FSM in a tight validation loop.
        stamp_ns = self.health_stamp_ns

        if stamp_ns is None:
            return False

        last_stamp = entry.get("last_revalidation_stamp_ns")
        last_attempt = entry.get("last_revalidation_at", 0.0)

        if (
            last_stamp is not None
            and stamp_ns <= last_stamp
        ):
            return False

        if now - last_attempt < 1.0:
            return False

        # This request retains exactly the previously generated path.
        # The new message goes ONLY to the validator, not to Nav2
        # and not to GoalGate's candidate ingestion channel.
        if not hasattr(self, "_anticipatory_revalidation_pub"):
            self._anticipatory_revalidation_pub = (
                self.create_publisher(
                    CandidateTrajectory,
                    "/social_navigation/trajectory_revalidation_request",
                    10,
                )
            )

        entry["last_revalidation_stamp_ns"] = stamp_ns
        entry["last_revalidation_at"] = now

        # Establish the exact selected candidate before the
        # asynchronous validation response can be processed.
        self._invalidate_trajectory_validation(
            reason="CACHED_PATH_REVALIDATION_STARTED"
        )

        self.candidate_path_id = path_id
        self.candidate_navigation_goal_id = scope[0]
        self.generation_requested_at = now
        self.anticipatory_selected_request_id = request_id

        self._transition_to(
            STATE_VALIDATING_TRAJECTORY,
            reason="CACHED_PATH_FRESH_REVALIDATION",
        )

        self._anticipatory_revalidation_pub.publish(candidate)

        self.get_logger().info(
            "[MODE D REVALIDATION] "
            f"goal={scope[0]} | "
            f"person={scope[4]} | "
            f"path={path_id} | "
            "nav2_generation_reused=True | "
            "movement_authorized=False"
        )

        return True
    
    # ================================================================
    # Episode retirement
    # ================================================================

    def _initialize_social_reassessment(self):
        """
        Initialize factual reassessment and optional anticipatory planning.

        Anticipatory planning is strictly isolated from mode C.
        No anticipatory work starts during initialization.
        """

        self._sr_scope = None
        self._sr_pending = None
        self._sr_result = None
        self._sr_next_at = time.monotonic() + 3.0

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        self._sr_request_pub = self.create_publisher(
            String,
            "/social_navigation/reassessment_request",
            qos,
        )

        self._sr_result_sub = self.create_subscription(
            String,
            "/social_navigation/reassessment_result",
            self._social_reassessment_result_callback,
            qos,
        )

        # ---------------------------------------------------------
        # Optional Mode-D anticipation
        # ---------------------------------------------------------

        self.declare_parameter(
            "anticipatory_planning_enabled",
            False,
        )

        configured = self.get_parameter(
            "anticipatory_planning_enabled"
        ).value

        if not isinstance(configured, bool):
            raise ValueError(
                "anticipatory_planning_enabled must be boolean"
            )

        self.anticipatory_planning_enabled = bool(
            configured and self.interaction == "dialogue"
        )

        # Immutable candidate metadata, indexed by request ID.
        self.anticipatory_requests = {}

        # Successfully generated candidate messages, indexed by
        # the same request IDs. These do not authorize execution.
        self.anticipatory_candidates = {}

        # Current anticipatory generation, if any.
        self.anticipatory_pending_request_id = ""
        self.anticipatory_uncertain_request_id = ""

        # Scope for which the cache was constructed.
        self.anticipatory_scope = None

        # Directions awaiting generation.
        self.anticipatory_queue = []

        # A candidate may be ready but require revalidation.
        self.anticipatory_selected_request_id = ""

        self.get_logger().info(
            "[ANTICIPATORY PLANNING] "
            f"interaction={self.interaction} | "
            f"enabled={self.anticipatory_planning_enabled} | "
            "execution_authorization=False"
        )
        
    def _social_reassessment_scope(self):
        """Identify the public goal and current social episode."""
        return (
            self.active_navigation_goal_id,
            self.social_state_session or self.health_session,
            self.social_state_episode,
        )

    def _social_reassessment_tick(self, now):
        """
        Request fresh factual assessment while waiting.

        This method never publishes REPLAN or RESUME.
        Technical transaction uncertainty does not prevent
        perception reassessment.
        """
        scope = self._social_reassessment_scope()

        if scope != self._sr_scope:
            self._sr_scope = scope
            self._sr_pending = None
            self._sr_result = None
            self._sr_next_at = now + 3.0

        if (
            not scope[0]
            or self.fsm_state != STATE_WAITING
            or self.pending_terminal_stop is not None
            or self.social_resolution_aborted
        ):
            self._sr_pending = None
            self._sr_result = None
            self._sr_next_at = now + 3.0
            return

        pending = self._sr_pending

        if pending is not None:
            if now >= pending["deadline"]:
                self.get_logger().warning(
                    "[SOCIAL REASSESSMENT] "
                    "Response timeout; motion remains held."
                )

                self._sr_pending = None
                self._sr_result = None
                self._sr_next_at = now + 3.0

            return

        if now < self._sr_next_at:
            return

        request_id = (
            f"SOCIAL_REASSESS_{uuid.uuid4().hex}"
        )

        requested_ns = self.get_clock().now().nanoseconds

        self._sr_pending = {
            "request_id": request_id,
            "requested_ns": requested_ns,
            "deadline": now + 4.0,
            "scope": scope,
        }

        self._sr_result = None

        self._sr_request_pub.publish(
            String(
                data=json.dumps({
                    "version": 1,
                    "request_id": request_id,
                    "goal_id": scope[0],
                    "session": scope[1],
                    "episode": scope[2],
                    "requested_ns": requested_ns,
                })
            )
        )

    def _social_reassessment_result_callback(self, msg):
        """
        Store a matching fresh result.

        Canonical social events remain responsible for updating
        episode state. This callback never authorizes motion.
        """
        pending = self._sr_pending

        if pending is None:
            return

        try:
            data = json.loads(msg.data)

            if (
                not isinstance(data, dict)
                or data.get("version") != 1
            ):
                return

            if (
                data.get("request_id") != pending["request_id"]
                or data.get("goal_id") != pending["scope"][0]
            ):
                return

            status = str(data["status"])
            stamp_ns = int(data["observation_stamp_ns"])
            sequence = int(data["sequence"])

            identity = (
                str(data["session"]),
                str(data["episode"]),
            )

        except (ValueError, TypeError, KeyError):
            return

        now = time.monotonic()

        self._sr_pending = None
        self._sr_result = None
        self._sr_next_at = now + 3.0

        if (
            now >= pending["deadline"]
            or self.fsm_state != STATE_WAITING
            or self._social_reassessment_scope() != pending["scope"]
            or identity != pending["scope"][1:]
        ):
            return

        if status not in {
            "CONFLICT",
            "CLEAR",
            "NO_CURRENT_INTRUSION",
            "EMPTY_UNRESOLVED",
        }:
            self.get_logger().info(
                "[SOCIAL REASSESSMENT] "
                f"result={status}; motion remains held."
            )
            return

        age = (
            self.get_clock().now().nanoseconds - stamp_ns
        ) / 1e9

        if (
            stamp_ns <= pending["requested_ns"]
            or not 0.0 <= age <= self.social_decision_max_age_sec
        ):
            return

        self._sr_result = {
            "scope": pending["scope"],
            "status": status,
            "stamp_ns": stamp_ns,
            "sequence": sequence,
        }

    def _social_reassessment_ready(self, allowed_statuses):
        """
        Require a fresh response consistent with canonical episode state.

        This method authorizes policy reconsideration only.
        It never authorizes navigation execution.
        """
        result = self._sr_result

        if result is None:
            return False

        if (
            result["scope"] != self._social_reassessment_scope()
            or result["status"] not in allowed_statuses
        ):
            return False

        age = (
            self.get_clock().now().nanoseconds
            - result["stamp_ns"]
        ) / 1e9

        if not 0.0 <= age <= self.social_decision_max_age_sec:
            self._sr_result = None
            return False

        if result["sequence"] < 0:
            self._sr_result = None
            return False

        # Canonical events and reassessment responses use
        # separate topics and may arrive in either order.
        if (
            self.social_state_episode
            and self.social_state_sequence < result["sequence"]
        ):
            return False

        status = result["status"]

        if status == "CONFLICT":
            return bool(
                self.human_proximity_active
                and self._route_observation_is_fresh()
            )

        if status == "CLEAR":
            decision = self.latest_social_decision

            if (
                decision is None
                or normalize_social_navigation_event(
                    decision.event
                ) != SOCIAL_EVENT_CLEAR
            ):
                return False

            return bool(
                not self.human_proximity_active
                and not self.social_episode_retired
                and not self.social_loss_expired
                and not self.waiting_for_observation_reacquisition
                and self._message_is_recent(
                    decision,
                    self.social_decision_max_age_sec,
                )
                and self._route_observation_is_fresh()
            )

        if status == "NO_CURRENT_INTRUSION":
            # This response cannot silently resolve an episode
            # that remains unresolved in canonical state.
            return bool(
                not self.human_proximity_active
                and not self.social_episode_retired
                and not self.social_loss_expired
                and not self.waiting_for_observation_reacquisition
                and self._route_observation_is_fresh()
            )

        if status == "EMPTY_UNRESOLVED":
            # Empty observations permit the dedicated recovery
            # policy to reconsider the situation.
            # They do not establish CLEAR or permission to move.
            return bool(
                self.human_proximity_active
                and self.latest_observation_health == "EMPTY"
                and self._fresh_observation_health_available()
                and (
                    not self.social_state_session
                    or self.health_session == self.social_state_session
                )
            )

        return False
    
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
        """Reassess first, then apply the existing collaboration policy."""
        self._social_reassessment_tick(now)

        if not self._resolution_budget_available(now):
            return
        
        # Mode D: the cached-path policy owns route selection
        # after confirmed human permission.
        if (
            self.interaction == "dialogue"
            and self.anticipatory_planning_enabled
            and self.dialogue_outcome == "ALLOW_CONFIRMED"
            and self._permission_applies_to_current_episode()
            and self.pending_terminal_stop is None
            and self.fsm_state == STATE_WAITING
            and (
                self.anticipatory_candidates
                or self.anticipatory_pending_request_id
            )
        ):
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
                    reconsider=True,
                )

            return

        if (
            self.fsm_state != STATE_WAITING
            or not self.route_waiting_for_change
            or self.navigation_active
            or self.pending_terminal_stop is not None
            or self.pending_hri_close is not None
            or self.pending_gate_command is not None
            or self.pending_speech_decision_id
            or self.dialogue_audio_recovery_pending
            or self.pending_generation_request_id
            or self.uncertain_generation_request_id
            or self.announcement_context_key is not None
        ):
            return

        # Retired episodes retain their explicit recovery policy.
        if (
            self.social_episode_retired
            or self.social_loss_expired
            or self.waiting_for_observation_reacquisition
        ):
            return

        if not self._social_reassessment_ready({
            "CONFLICT",
            "CLEAR",
            "NO_CURRENT_INTRUSION",
        }):
            return

        if not self._route_observation_is_fresh():
            return

        if not self._fresh_observation_health_available():
            return

        nav = self.latest_navigation_state

        if (
            nav is None
            or nav.navigation_goal_id != self.active_navigation_goal_id
            or not self._message_is_recent(nav, 1.0)
            or bool(nav.navigation_active)
            or self.navigation_state not in {
                "WAITING",
                "SOCIAL_HOLD",
                "REPLANNING",
            }
        ):
            return

        if now - self.reassessment_last_at < 3.0:
            return

        stamp_ns = self.health_stamp_ns

        if stamp_ns is None:
            return

        if (
            self.reassessment_last_stamp_ns is not None
            and stamp_ns <= self.reassessment_last_stamp_ns
        ):
            return

        if (
            self.initial_empty_retry_goal_id
            == self.active_navigation_goal_id
        ):
            if now < self.initial_empty_retry_not_before:
                return

            previous = self.initial_empty_retry_after_health_ns

            if previous is not None and stamp_ns <= previous:
                return

        if self.route_attempt_count >= self.route_search_max_attempts:
            self.route_attempt_count = 0
            self.route_tried_directions.clear()

        # Consume this assessment for one policy continuation.
        self._sr_result = None

        self.reassessment_last_at = now
        self.reassessment_last_stamp_ns = stamp_ns

        self.initial_empty_retry_goal_id = ""
        self.initial_empty_retry_after_health_ns = None
        self.route_waiting_for_change = False

        if self.confirmed_human_intent == "DENY_PASSAGE":
            self._request_next_social_alternative()
            return

        # Mode-D dialogue guards remain inside _enter_replanning().
        self._enter_replanning(
            reason="SOCIAL_REASSESSMENT_COMPLETED",
            directional_constraint=self.confirmed_human_direction,
        )

    def _evaluate_retired_episode_recovery(self):
        """Require explicit reassessment before retired-episode recovery."""
        goal_id = str(
            self.active_navigation_goal_id or ""
        ).strip()

        if (
            not goal_id
            or self.active_navigation_goal_pose is None
            or self.social_resolution_aborted
            or self.fsm_state != STATE_WAITING
        ):
            return

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

        if self.gate_ready_goal_id != goal_id:
            return

        if not self._social_reassessment_ready({
            "EMPTY_UNRESOLVED",
        }):
            return

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

        if now - self.reassessment_last_at < 3.0:
            return

        stamp_ns = self.health_stamp_ns

        if (
            stamp_ns is None
            or (
                self.reassessment_last_stamp_ns is not None
                and stamp_ns <= self.reassessment_last_stamp_ns
            )
        ):
            return

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

        self._sr_result = None

        self.reassessment_last_at = now
        self.reassessment_last_stamp_ns = stamp_ns

        self.retired_recovery_attempt_count += 1
        self.retired_recovery_last_attempt_at = now
        self.retired_recovery_goal_id = goal_id
        self.retired_recovery_path_id = ""

        self._enter_replanning(
            reason="RETIRED_EPISODE_REASSESSED",
            directional_constraint="NONE",
        )
        
    # ================================================================
    # Ordered social events
    # ================================================================


    def _social_decision_callback(self, msg):
        """
        Process ordered social events.

        Preserve confirmed human permission across a new
        social episode only for the same public goal,
        person ID, and tracking session.

        Preserve directional semantics separately from
        physical motion authorization.
        """
        import copy

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

        if not self._validate_participant_context(msg, event):
            self.get_logger().warning(
                "[SOCIAL STATE] Invalid participant context.",
                throttle_duration_sec=2.0,
            )
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

        new_episode = incoming_identity != current_identity

        if not new_episode:
            if sequence <= self.social_state_sequence:
                return

        else:
            if event != SOCIAL_EVENT_INTRUSION_CONFIRMED:
                return

            try:
                position = msg.person_position

                valid_geometry = (
                    bool(str(msg.person_id).strip())
                    and str(msg.tracking_state).upper() == "TRACKED"
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

            if current_identity[0] and current_identity[1]:
                self.retired_social_episode_ids[
                    current_identity
                ] = True

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

            previous_scope = self.authorized_episode_key

            incoming_person_id = str(
                msg.person_id or ""
            ).strip()

            incoming_tracking_session = str(
                msg.tracking_session_id or ""
            ).strip()

            same_person_consent = bool(
                self.interaction == "dialogue"
                and previous_scope is not None
                and len(previous_scope) == 5
                and self.confirmed_human_intent == "ALLOW_PASSAGE"
                and self.dialogue_outcome == "ALLOW_CONFIRMED"
                and previous_scope[0] == episode_origin_goal_id
                and previous_scope[3] == incoming_tracking_session
                and previous_scope[4] == incoming_person_id
                and incoming_person_id
                and incoming_tracking_session
            )

            retained_direction = (
                str(
                    self.confirmed_human_direction or "NONE"
                ).strip().upper()
                if same_person_consent
                else "NONE"
            )

            retained_interaction_id = (
                str(
                    getattr(
                        self,
                        "_directional_confirmed_interaction_id",
                        "",
                    ) or ""
                ).strip()
                if same_person_consent
                else ""
            )

            retained_finalization = None

            if (
                same_person_consent
                and retained_direction in {"LEFT", "RIGHT"}
            ):
                old_finalization = getattr(
                    self,
                    "_directional_finalization",
                    None,
                )

                if isinstance(old_finalization, dict):
                    candidate = old_finalization.get("candidate")

                    if (
                        old_finalization.get("phase") == "ANNOUNCED"
                        and old_finalization.get("scope")
                            == previous_scope
                        and isinstance(candidate, dict)
                        and candidate.get("scope") == previous_scope
                        and candidate.get(
                            "final_confirmation_received"
                        ) is True
                        and candidate.get("direction")
                            == retained_direction
                        and candidate.get("interaction_id")
                            == retained_interaction_id
                        and candidate.get(
                            "final_confirmation_request_id"
                        ) == old_finalization.get("request_id")
                        and candidate.get(
                            "resolved_robot_direction"
                        ) in {"LEFT", "RIGHT"}
                        and getattr(
                            self,
                            "_directional_announcement_complete",
                            False,
                        )
                        and getattr(
                            self,
                            "_direction_semantic_ready",
                            False,
                        )
                    ):
                        retained_finalization = copy.deepcopy(
                            old_finalization
                        )

                # Directional consent is not reusable unless
                # the original confirmation can be preserved.
                if retained_finalization is None:
                    same_person_consent = False

            self._reset_episode()

            self.social_state_session = session
            self.social_state_episode = episode
            self.social_state_sequence = 0

            self._retained_same_person_consent = (
                {
                    "scope": (
                        episode_origin_goal_id,
                        session,
                        episode,
                        incoming_tracking_session,
                        incoming_person_id,
                    ),
                    "direction": retained_direction,
                    "interaction_id": retained_interaction_id,
                    "finalization": retained_finalization,
                }
                if same_person_consent
                else None
            )

            self.social_episode_origin_goal_id = (
                episode_origin_goal_id
            )

            if same_person_consent:
                self.get_logger().info(
                    "[SOCIAL CONSENT] Permission retained | "
                    f"goal={episode_origin_goal_id} | "
                    f"new_episode={episode} | "
                    f"person={incoming_person_id} | "
                    f"tracking_session={incoming_tracking_session} | "
                    f"human_direction={retained_direction} | "
                    "dialogue_restart_required=False"
                )

            self._transition_to(
                STATE_NAVIGATING,
                reason="NEW_CONFIRMED_SOCIAL_EPISODE",
            )

        self.social_state_sequence = sequence
        self.latest_social_decision = msg

        if not self._adopt_participant_context(msg):
            return

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
            if not self._individual_scene_all_clear():
                self.get_logger().warning(
                    "[INDIVIDUAL SOCIAL] Scene CLEAR held: "
                    "an independent episode remains unresolved."
                )
                return

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

        # An HRI close timeout is not a request to abandon
        # the public navigation goal.
        # Remain stopped, retain the pending close transaction,
        # and accept a matching late CLOSED acknowledgement.
        #
        # Never authorize motion while closure is uncertain.
        if self.active_navigation_goal_id:
            self._hold_route_search(
                reason="HRI_CLOSE_ACK_TIMEOUT",
                reconsider=False,
            )

    def _hri_dialogue_control_result_callback(self, msg):
        """Accept exact HRI closure without bypassing navigation safety."""

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
        ).strip().upper()

        if status == "TERMINATION_UNCONFIRMED":
            self.hri_close_fault_reported = True

            self._publish_wait(
                reason="HRI_AUDIO_TERMINATION_UNCONFIRMED"
            )

            self.get_logger().error(
                "[HRI CLOSE] Termination unconfirmed | "
                f"close={pending['close_id']}"
            )
            return

        if status != "CLOSED":
            return

        closed_interaction_id = pending["interaction_id"]

        turn_key = (
            pending["interaction_id"],
            pending["request_id"],
        )

        owner_scope = self.mode_d_hri_turns.get(turn_key)

        if owner_scope is not None:
            case = self.mode_d_cases.get(owner_scope)

            if case is not None:
                case["hri_closed"] = True

            self.mode_d_hri_closures[turn_key] = {
                "close_id": pending["close_id"],
                "scope": owner_scope,
            }

            if self.mode_d_hri_owner == owner_scope:
                self.mode_d_hri_owner = None

            for key in list(self.mode_d_hri_turns):
                if key[0] == closed_interaction_id:
                    self.mode_d_hri_turns.pop(key, None)

            while len(self.mode_d_hri_closures) > 256:
                oldest_key = next(iter(self.mode_d_hri_closures))
                self.mode_d_hri_closures.pop(oldest_key, None)
                
        self.pending_hri_close = None
        self.hri_close_fault_reported = False
        self.dialogue_audio_recovery_pending = False
        self.audio_recovery_interaction_id = ""
        self.audio_recovery_request_id = ""

        if (
            self.interaction == "dialogue"
            and self.active_interaction_id == closed_interaction_id
            and self.dialogue_outcome == "ALLOW_CONFIRMED"
            and self._permission_applies_to_current_episode()
        ):
            self._closed_hri_scope = self._anticipation_scope()
            
        if self.active_interaction_id == closed_interaction_id:
            self.active_interaction_id = ""
            self.active_dialogue_request_type = ""
            self.pending_dialogue_request_id = ""
            self.pending_dialogue_scope = None
            self.dialogue_started_at = None
            self.waiting_for_human_initiative = False

        self.get_logger().info(
            "[HRI CLOSE] Exact closure confirmed | "
            f"interaction={closed_interaction_id} | "
            f"outcome={self.dialogue_outcome}"
        )

        if (
            not self.active_navigation_goal_id
            or self.social_resolution_aborted
            or self.pending_terminal_stop is not None
            or self.pending_speech_decision_id
            or self.pending_gate_command is not None
            or self.fsm_state != STATE_WAITING
        ):
            return

        if not self.human_proximity_active:
            self.deferred_clear_replan_goal_id = (
                self.active_navigation_goal_id
            )
            return

        if (
            self.interaction == "dialogue"
            and self.dialogue_outcome == "ALLOW_CONFIRMED"
            and self._permission_applies_to_current_episode()
        ):
            if self.anticipatory_planning_enabled:
                # No second Nav2 generation here.
                # A separate selector will reuse a candidate
                # only after exact terminal reconciliation.
                self._schedule_social_reassessment()

                self.get_logger().info(
                    "[MODE D] HRI closed; awaiting cached-path "
                    "selection and fresh validation."
                )
                return

            # Existing non-anticipatory mode-D fallback.
            self._enter_replanning(
                reason="HRI_CLOSED_PERMISSION_CONFIRMED",
                directional_constraint=(
                    self.confirmed_human_direction
                ),
            )
            return

        if self.dialogue_outcome == "DENY_CONFIRMED":
            if self._route_observation_is_fresh():
                self._request_next_social_alternative()
            else:
                self._schedule_social_reassessment()
            return

        if (
            self.dialogue_outcome == "NONE"
            and self._route_observation_is_fresh()
            and not self.dialogue_audio_recovery_pending
        ):
            self._start_human_initiative_window()
            return

        self._schedule_social_reassessment()
        
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
        Process factual clearance without confusing it with
        dialogue ownership or semantic permission.

        Mode C: preserve the existing clearance policy.
        Mode D: CLEAR never revokes a confirmed permission
        and never closes an unfinished dialogue by itself.
        """

        if self.pending_terminal_stop is not None:
            return

        goal_id = str(
            self.active_navigation_goal_id or ""
        ).strip()

        person_id = str(
            msg.person_id or ""
        ).strip()

        resolved_scope = self._same_resolved_social_scope(
            person_id
        )

        execution_confirmed = bool(
            goal_id
            and resolved_scope
            and self.fsm_state == STATE_NAVIGATING
            and self.navigation_active
            and self.navigation_state == NAVIGATION_NAVIGATING
            and self.pending_gate_command is None
            and self.pending_terminal_stop is None
        )

        # Preserve the canonical actor while a Mode-D dialogue
        # owns it. CLEAR may concern another participant.
        if self.interaction != "dialogue" and person_id:
            self.active_person_id = person_id

        self.active_decision_id = msg.decision_id

        self.human_proximity_active = False
        self.social_observation_available = True
        self.social_observation_complete = bool(
            msg.observation_complete
        )

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
            self.route_waiting_for_change = False
            self.route_blocked_signature = None

            self.get_logger().info(
                "[SOCIAL CLEAR] "
                "Already validated navigation continues | "
                f"goal={goal_id}"
            )
            return

        # =====================================================
        # MODE D: independent dialogue and clearance policies
        # =====================================================

        if self.interaction == "dialogue":

            permission_confirmed = (
                self.dialogue_outcome == "ALLOW_CONFIRMED"
                and self.confirmed_human_intent == "ALLOW_PASSAGE"
                and self._permission_applies_to_current_episode()
            )

            dialogue_unfinished = bool(
                self.pending_hri_close is not None
                or self.pending_dialogue_request_id
                or self.dialogue_audio_recovery_pending
            )

            if permission_confirmed:

                # Semantic authorization remains valid for this
                # exact person and episode. Geometry must be
                # checked again before any movement.
                self._invalidate_trajectory_validation(
                    reason="CLEAR_REQUIRES_FRESH_PATH_VALIDATION"
                )

                self._schedule_social_reassessment()

                if dialogue_unfinished:
                    if goal_id:
                        self._publish_wait(
                            reason="CLEAR_AWAITING_EXISTING_HRI_TRANSACTION"
                        )

                    # Keep the already-owned HRI transaction.
                    # Do not request another close.
                    return

                if not goal_id:
                    return

                if self.anticipatory_planning_enabled:
                    # The cached-path selector will run through
                    # the existing timer and require new validation.
                    self._publish_wait(
                        reason="CLEAR_REVALIDATE_CACHED_PATH"
                    )

                    self._transition_to(
                        STATE_WAITING,
                        reason="CLEAR_REVALIDATE_CACHED_PATH",
                    )
                    return

                self._enter_replanning(
                    reason="CLEAR_AFTER_CONFIRMED_PERMISSION",
                    directional_constraint=(
                        self.confirmed_human_direction
                    ),
                )
                return

            if dialogue_unfinished:

                # CLEAR does not own or terminate a dialogue.
                # The HRI response/close protocol remains in charge.
                self._invalidate_trajectory_validation(
                    reason="CLEAR_DURING_ACTIVE_DIALOGUE"
                )

                if goal_id:
                    self._publish_wait(
                        reason="CLEAR_DIALOGUE_STILL_ACTIVE"
                    )

                self._schedule_social_reassessment()

                self.get_logger().info(
                    "[MODE D CLEAR] "
                    "Social conflict cleared; "
                    "ongoing dialogue preserved."
                )
                return

            # No confirmed permission and no active HRI turn.
            # Existing conservative clearance recovery applies.
            self._reset_route_search()

            if self.fsm_state in {
                STATE_STOPPING_FOR_SOCIAL_CONFLICT,
                STATE_ANNOUNCING_DETECTION,
            }:
                self.clear_pending_after_announcement = True
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
            return

        # =====================================================
        # MODE C: original behavior, unchanged
        # =====================================================

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

    def _validate_participant_context(self, msg, event):
        """Validate typed participant evidence before adopting an event."""
        try:
            seen = set()
            primary = (
                str(msg.tracking_session_id).strip(),
                str(msg.person_id).strip(),
            )

            if not msg.participants:
                return False

            for participant in msg.participants:
                key = (
                    str(participant.tracking_session_id).strip(),
                    str(participant.person_id).strip(),
                )

                if (
                    not all(key)
                    or key in seen
                    or (
                        participant.conflict
                        and not participant.observed
                    )
                ):
                    return False

                seen.add(key)

            if primary not in seen:
                return False

            if (
                msg.observation_complete
                and not all(
                    participant.observed
                    for participant in msg.participants
                )
            ):
                return False

            if event == SOCIAL_EVENT_CLEAR:
                return bool(
                    msg.observation_complete
                    and not msg.social_conflict_active
                    and not any(
                        participant.conflict
                        for participant in msg.participants
                    )
                )

            if not msg.social_conflict_active:
                return False

            if event in {
                SOCIAL_EVENT_INTRUSION_CONFIRMED,
                SOCIAL_EVENT_OBSERVATION_REACQUIRED,
            }:
                return any(
                    (
                        participant.tracking_session_id,
                        participant.person_id,
                    ) == primary
                    and participant.observed
                    for participant in msg.participants
                )

            return event == SOCIAL_EVENT_OBSERVATION_LOST

        except (TypeError, AttributeError, ValueError):
            return False

    def _adopt_participant_context(self, msg):
        """
        Adopt canonical participant context without transferring consent.

        Mode D protects ownership of an unfinished HRI transaction.
        Mode C retains the existing participant-switch procedure.
        """

        scope = self._social_scope_key(msg.person_id)
        previous = self._active_actor_scope

        # =====================================================
        # MODE D: protect active dialogue ownership
        # =====================================================

        if (
            self.interaction == "dialogue"
            and previous is not None
            and scope != previous
        ):

            hri_owned_by_previous = bool(
                self.pending_hri_close is not None
                or self.pending_dialogue_request_id
                or self.dialogue_audio_recovery_pending
            )

            if hri_owned_by_previous:

                # An event concerning B cannot close A's HRI
                # transaction or overwrite A's semantic state.
                #
                # Nevertheless, B remains relevant to collision
                # avoidance and must not be ignored geometrically.
                self.social_observation_complete = bool(
                    msg.observation_complete
                )

                self._invalidate_trajectory_validation(
                    reason="SECONDARY_PARTICIPANT_DURING_HRI"
                )

                if self.active_navigation_goal_id:
                    self._publish_wait(
                        reason="SECONDARY_PARTICIPANT_DURING_HRI"
                    )

                if not msg.observation_complete:
                    self.social_observation_available = False
                    self.waiting_for_observation_reacquisition = True

                self._schedule_social_reassessment()

                self.get_logger().info(
                    "[SOCIAL PARTICIPANTS] "
                    "Secondary participant observed while "
                    "existing HRI transaction owns the dialogue | "
                    f"dialogue_person={previous[4]} | "
                    f"event_person={scope[4]} | "
                    "dialogue_preserved=True | "
                    "execution_authorized=False"
                )

                return False

        # =====================================================
        # Existing canonical actor switch
        # =====================================================

        if scope != previous:

            if previous is not None:
                self._actor_semantics[previous] = {
                    "intent": self.confirmed_human_intent,
                    "direction": self.confirmed_human_direction,
                    "outcome": self.dialogue_outcome,
                    "authorized": (
                        self.authorized_episode_key == previous
                    ),
                    "executed": (
                        self.executed_social_scope == previous
                    ),
                    "forbidden_direction": (
                        self.forbidden_human_direction
                    ),
                }

            while len(self._actor_semantics) > 128:
                self._actor_semantics.pop(
                    next(iter(self._actor_semantics))
                )

            changed_active_actor = bool(
                previous is not None
                and self.human_proximity_active
            )

            if changed_active_actor:

                # A change of actor invalidates any outstanding
                # trajectory ownership, not the previous actor's
                # stored permission.
                if self.pending_generation_request_id:
                    self.uncertain_generation_request_id = (
                        self.pending_generation_request_id
                    )

                # Only request a close if there really is an
                # outstanding HRI turn. Never invent a closure.
                if (
                    self.pending_dialogue_request_id
                    or self.dialogue_audio_recovery_pending
                    or self.pending_hri_close is not None
                ):
                    self._request_hri_close(
                        reason="SOCIAL_PARTICIPANT_CHANGED"
                    )

                self.pending_dialogue_request_id = ""
                self.pending_dialogue_scope = None
                self.dialogue_started_at = None
                self.waiting_for_human_initiative = False

                self._clear_pending_human_preference()

            saved = self._actor_semantics.get(scope, {})

            self.confirmed_human_intent = saved.get(
                "intent",
                "UNKNOWN",
            )

            self.confirmed_human_direction = saved.get(
                "direction",
                "NONE",
            )

            self.dialogue_outcome = saved.get(
                "outcome",
                "NONE",
            )

            self.authorized_episode_key = (
                scope
                if saved.get("authorized", False)
                else None
            )

            self.executed_social_scope = (
                scope
                if saved.get("executed", False)
                else None
            )

            self.forbidden_human_direction = saved.get(
                "forbidden_direction",
                "NONE",
            )

            self.dialogue_resolution_complete = (
                self.dialogue_outcome in {
                    "ALLOW_CONFIRMED",
                    "DENY_CONFIRMED",
                    "NO_RESPONSE",
                    "NOT_UNDERSTOOD",
                }
            )

            self._active_actor_scope = scope
            self.active_person_id = msg.person_id

            if self.interaction == "dialogue":
                # A closure for A must never be accepted as the
                # completed closure of B's dialogue.
                self._closed_hri_scope = None

                # Cached paths remain bound to the exact actor
                # scope and cannot be reused for another person.
                self.anticipatory_selected_request_id = ""

            if changed_active_actor:
                if self.fsm_state in {
                    STATE_STOPPING_FOR_SOCIAL_CONFLICT,
                    STATE_ANNOUNCING_DETECTION,
                }:
                    self._invalidate_trajectory_validation(
                        reason="SOCIAL_PARTICIPANT_CHANGED"
                    )

                    self._publish_wait(
                        reason="SOCIAL_PARTICIPANT_CHANGED"
                    )

                    self._schedule_social_reassessment()

                else:
                    self._hold_route_search(
                        reason="SOCIAL_PARTICIPANT_CHANGED",
                        reconsider=True,
                    )

        # =====================================================
        # Observation completeness remains authoritative
        # =====================================================

        self.social_observation_complete = bool(
            msg.observation_complete
        )

        if msg.observation_complete:
            return True

        if not self.human_proximity_active:
            self.human_proximity_active = True
            self.active_person_id = msg.person_id
            self.active_decision_id = msg.decision_id

            self._ensure_interaction_id()

            self.announcement_social_decision = msg
            self.announcement_decision_id = msg.decision_id

        self._handle_observation_lost(msg)

        if self.fsm_state in {
            STATE_NAVIGATING,
            STATE_OBSERVING,
            STATE_WAITING,
        }:
            self._hold_route_search(
                reason="SOCIAL_PARTICIPANTS_PARTIALLY_OBSERVED",
                reconsider=True,
            )

        return False

    def _social_scope_key(self, person_id=None):
        """
        Return the exact canonical social scope for one person.

        Mode C retains the original event-based behavior.

        In mode D, another participant's event must not change
        the tracking session of the active dialogue participant.
        """

        person = str(
            self.active_person_id
            if person_id is None
            else person_id
        ).strip()

        decision = self.latest_social_decision

        goal_id = str(
            self.active_navigation_goal_id or ""
        ).strip()

        producer_session = str(
            self.social_state_session or ""
        ).strip()

        episode = str(
            self.social_state_episode or ""
        ).strip()

        tracking_session = ""

        if self.interaction == "dialogue":
            if decision is not None:
                matches = {
                    str(p.tracking_session_id or "").strip()
                    for p in decision.participants
                    if str(p.person_id or "").strip() == person
                }

                matches.discard("")

                if len(matches) == 1:
                    tracking_session = next(iter(matches))

            if not tracking_session:
                previous = self._active_actor_scope

                if (
                    previous is not None
                    and len(previous) == 5
                    and previous[0] == goal_id
                    and previous[1] == producer_session
                    and previous[2] == episode
                    and previous[4] == person
                ):
                    tracking_session = previous[3]

        else:
            # Preserve the original Mode-C behavior.
            if decision is not None:
                tracking_session = str(
                    decision.tracking_session_id or ""
                ).strip()

        return (
            goal_id,
            producer_session,
            episode,
            tracking_session,
            person,
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
        Preserve social and semantic context during observation loss.

        Planning may continue while the robot remains held.
        A pending motion authorization must be revoked.

        Loss never establishes clearance and never erases
        confirmed human permission for the current episode.
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

        # Computing a candidate is not permission to execute it.
        # Preserve the outstanding planning transaction.
        if self.fsm_state in {
            STATE_REPLANNING,
            STATE_VALIDATING_TRAJECTORY,
        }:
            return

        # RESUME may already have reached the Gate.
        # Explicitly revoke it instead of waiting for the
        # prolonged observation-loss timeout.
        if self.fsm_state == STATE_RESUMING:
            self._hold_route_search(
                reason="SOCIAL_OBSERVATION_LOST_DURING_RESUME",
                reconsider=True,
            )

            self.observation_loss_held_goal_id = (
                self.active_navigation_goal_id
            )
            return

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

        # Preserve an already established dialogue or speech
        # transaction. Motion remains subject to its own barriers.
        if dialogue_active or self.fsm_state in {
            STATE_STOPPING_FOR_SOCIAL_CONFLICT,
            STATE_ANNOUNCING_DETECTION,
        }:
            return

        if (
            self.fsm_state == STATE_WAITING
            and not self.navigation_active
        ):
            self._schedule_social_reassessment()
            return

        if (
            self.fsm_state == STATE_NAVIGATING
            and self.navigation_active
        ):
            self._hold_route_search(
                reason="SOCIAL_OBSERVATION_LOST",
                reconsider=True,
            )

            self.observation_loss_held_goal_id = (
                self.active_navigation_goal_id
            )
            return

        if not self.navigation_active:
            self._schedule_social_reassessment()

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
        """
        Trigger mode-C policy for a currently observed person
        whose individual conflict is persistent.

        Persistence memory belongs to the evaluator.
        """
        if self.fsm_state not in {
            STATE_OBSERVING,
            STATE_WAITING,
        }:
            return

        if not self._mode_c_reactive_generation_allowed():
            return

        if not self._individual_scene_current():
            return

        for identity, state in sorted(
            self.individual_participants.items()
        ):
            if (
                identity[0] != self.individual_source_session
                or not state["unresolved"]
                or not state["persistent"]
                or state["retired"]
                or not state["observed_now"]
                or state["observation_status"] != "OBSERVED"
                or state["conflict_status"] != "CONFLICT"
            ):
                continue

            episode = str(
                state["episode_id"]
            ).strip()

            if not episode:
                continue

            token = (
                self.active_navigation_goal_id,
                identity,
                episode,
            )

            if token in self._handled_persistent_person_episodes:
                continue

            self._enter_replanning(
                reason="SOCIAL_CONFLICT_PERSISTENT",
                directional_constraint="NONE",
            )

            # Consume the trigger only after the FSM has
            # actually entered the replanning phase.
            if self.fsm_state == STATE_REPLANNING:
                self._handled_persistent_person_episodes.add(
                    token
                )

                self.get_logger().info(
                    "[INDIVIDUAL PERSISTENCE] "
                    f"goal={self.active_navigation_goal_id} | "
                    f"identity={identity!r} | "
                    f"episode={episode} | "
                    "reactive_replanning_started=True"
                )

            return
    
    def _anticipation_scope(self):
        """Return the exact canonical context eligible for preplanning."""

        if not (
            self.interaction == "dialogue"
            and self.anticipatory_planning_enabled
        ):
            return None

        actor_scope = self._active_actor_scope

        if actor_scope is None or len(actor_scope) != 5:
            return None

        scope = tuple(str(value or "").strip() for value in actor_scope)

        if not all(scope):
            return None

        if (
            scope[0] != self.active_navigation_goal_id
            or scope[1] != self.social_state_session
            or scope[2] != self.social_state_episode
            or scope[4] != self.active_person_id
        ):
            return None

        return scope

    def _select_anticipatory_candidate(self):
        """
        Find an already generated path for the confirmed Mode-D
        social context.

        Selection is not path validation and never authorizes motion.
        """

        if not (
            self.interaction == "dialogue"
            and self.anticipatory_planning_enabled
            and self.dialogue_outcome == "ALLOW_CONFIRMED"
            and self.confirmed_human_intent == "ALLOW_PASSAGE"
            and self._permission_applies_to_current_episode()
        ):
            return None

        # HRI must be conclusively closed.
        if (
            self.pending_hri_close is not None
            or self.hri_close_fault_reported
            or self.dialogue_audio_recovery_pending
            or self.pending_speech_decision_id
            or self.pending_dialogue_request_id
            or self.pending_terminal_stop is not None
            or self.social_resolution_aborted
        ):
            return None

        scope = self._anticipation_scope()

        if (
            scope is None
            or scope != self.anticipatory_scope
        ):
            return None

        direction = self._normalize_direction(
            self.confirmed_human_direction
        )

        if (
            direction in {"LEFT", "RIGHT"}
            and direction == self.forbidden_human_direction
        ):
            return None

        # Confirmed LEFT never silently becomes RIGHT, or vice versa.
        # An unconstrained NONE candidate is not a substitute for an
        # explicitly confirmed person-relative direction.
        candidates = []

        for request_id, entry in self.anticipatory_candidates.items():
            metadata = self.anticipatory_requests.get(request_id)

            if metadata is None:
                continue

            if (
                entry.get("scope") != scope
                or metadata.get("scope") != scope
                or entry.get("direction") != direction
                or metadata.get("direction") != direction
                or entry.get("status") != "READY"
            ):
                continue

            # The Nav2 generation must be terminally reconciled.
            if (
                #metadata.get("status") != "READY"
                metadata.get("status") != "TERMINATED"
                or request_id == self.anticipatory_pending_request_id
                or request_id == getattr(
                    self, "anticipatory_uncertain_request_id", ""
                )
            ):
                continue

            candidate = entry.get("candidate")

            if candidate is None:
                continue

            path_id = str(candidate.path_id or "").strip()

            if (
                not path_id
                or candidate.request_id != request_id
                or candidate.navigation_goal_id != scope[0]
                or path_id != entry.get("path_id")
                or not candidate.path.header.frame_id
                or not candidate.path.poses
            ):
                continue

            candidates.append((
                float(entry.get("received_at", 0.0)),
                request_id,
                candidate,
            ))

        if not candidates:
            return None

        # Prefer the latest candidate within the same exact scope.
        _, request_id, candidate = max(
            candidates,
            key=lambda item: (item[0], item[1]),
        )

        self.anticipatory_selected_request_id = request_id

        self.get_logger().info(
            "[ANTICIPATORY SELECTION] "
            f"goal={scope[0]} | "
            f"person={scope[4]} | "
            f"direction={direction} | "
            f"path={candidate.path_id} | "
            "fresh_validation_required=True | "
            "navigation_authorized=False"
        )

        return candidate

    def _anticipation_tick(self, now):
        """
        Run Mode-D preplanning without overlapping Nav2 generations.

        HRI and planning advance independently, but Nav2 planning
        requests are serialized for the same public goal.
        """

        if not (
            self.interaction == "dialogue"
            and self.anticipatory_planning_enabled
        ):
            return

        pending_id = self.anticipatory_pending_request_id

        # Reconcile the current speculative generation before
        # considering another one, even if the episode changed.
        if pending_id:
            metadata = self.anticipatory_requests.get(pending_id)

            if metadata is None:
                self.anticipatory_uncertain_request_id = pending_id
                return

            status = metadata.get("status", "GENERATING")
            started = metadata.get("requested_at", now)
            last_query = metadata.get("last_query_at", 0.0)

            if (
                status == "READY"
                or status == "TERMINATION_UNCONFIRMED"
                or now - started >= 15.0
            ):
                if now - last_query >= 1.0:
                    metadata["last_query_at"] = now

                    self.generation_status_request_pub.publish(
                        String(
                            data=json.dumps({
                                "version": 1,
                                "navigation_goal_id": metadata["scope"][0],
                                "request_id": pending_id,
                            })
                        )
                    )

                if status != "READY":
                    self.anticipatory_uncertain_request_id = pending_id

            return

        if (
            self.uncertain_generation_request_id
            or getattr(
                self, "anticipatory_uncertain_request_id", ""
            )
            or self.pending_generation_request_id
        ):
            return

        scope = self._anticipation_scope()

        if scope is None:
            return

        # The cache belongs to one exact social context.
        if self.anticipatory_scope != scope:
            self.anticipatory_scope = scope

            self.anticipatory_queue = (
                self.mode_d_anticipation_queues.setdefault(
                    scope,
                    ["NONE", "LEFT", "RIGHT"],
                )
            )

            self.anticipatory_selected_request_id = ""

            # Ne pas effacer les candidats des autres personnes.
            # Chaque entrée conserve son propre scope.

        if not self.anticipatory_queue:
            return

        # Keep the existing execution transaction independent.
        if self.fsm_state in {
            STATE_REPLANNING,
            STATE_VALIDATING_TRAJECTORY,
            STATE_RESUMING,
        }:
            return

        if self.fsm_state not in {
            STATE_ANNOUNCING_DETECTION,
            STATE_SOCIAL_RESOLUTION,
            STATE_WAITING_FOR_HUMAN,
            STATE_WAITING_FOR_CONFIRMATION,
            STATE_ASKING,
            STATE_CONFIRMING_HUMAN_INTENT,
            STATE_WAITING,
        }:
            return

        if self.dialogue_outcome not in {
            "NONE",
            "PENDING",
        }:
            return

        if (
            not self.active_navigation_goal_id
            or self.active_navigation_goal_pose is None
            or self.navigation_active
            or self.social_resolution_aborted
            or self.pending_terminal_stop is not None
            or self.pending_gate_command is not None
            or self.social_episode_retired
            or self.social_loss_expired
            or self.waiting_for_observation_reacquisition
            or not self.human_proximity_active
            or not self._route_observation_is_fresh()
        ):
            return

        nav = self.latest_navigation_state

        if (
            nav is None
            or str(nav.navigation_goal_id or "").strip()
                != self.active_navigation_goal_id
            or not self._message_is_recent(nav, 1.0)
            or bool(nav.navigation_active)
            or self.navigation_state not in {
                "WAITING",
                "SOCIAL_HOLD",
                "REPLANNING",
            }
        ):
            return

        # Planning may start only after a verified physical hold.
        if (
            not self.stable_stop_confirmed
            or self.motion_state_received_at is None
            or self.motion_state_stamp_sec is None
            or now - self.motion_state_received_at > 0.5
        ):
            return

        motion_age = (
            self.get_clock().now().nanoseconds / 1e9
            - self.motion_state_stamp_sec
        )

        if not -0.1 <= motion_age <= 0.5:
            return

        if self.nav2_cancel_requested:
            if not (
                self.nav2_cancel_confirmed
                and self.stop_request_stamp_sec is not None
                and self.motion_state_stamp_sec
                    >= self.stop_request_stamp_sec
            ):
                return

        direction = self.anticipatory_queue[0]

        request = self._build_trajectory_generation_request(
            generation_reason=(
                "REPLAN"
                if direction == "NONE"
                else "ALTERNATIVE"
            ),
            directional_constraint=direction,
        )

        if request is None:
            return

        request.person_id = scope[4]

        request_id = request.request_id

        self.anticipatory_requests[request_id] = {
            "scope": scope,
            "direction": direction,
            "status": "GENERATING",
            "requested_at": now,
            "last_query_at": 0.0,
        }

        self.anticipatory_pending_request_id = request_id
        self.anticipatory_queue.pop(0)

        self.trajectory_generation_request_pub.publish(request)

        self.get_logger().info(
            "[ANTICIPATORY PLANNING] "
            f"request={request_id} | "
            f"direction={direction} | "
            f"remaining={len(self.anticipatory_queue)} | "
            "navigation_authorized=False"
        )


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

    def _reset_social_context_for_new_goal(self, goal_id):
        """
        Start a fresh social transaction for a new public navigation goal.

        Discard all goal-scoped social history while preserving the
        producer's sequence watermark for ordered incoming messages.
        """
        self._reset_episode()

        # Discard social observations belonging to the previous context.
        self.latest_social_decision = None
        self._active_actor_scope = None
        self._actor_semantics.clear()
        self.social_observation_complete = True

        # Treat every participant as newly observed for this goal.
        self.individual_participants.clear()
        self.individual_source_session = ""
        self.individual_source_stamp_ns = 0
        self.individual_source_fresh = False
        self.individual_source_complete = False
        self.individual_received_at = None
        self.individual_retired_source_sessions.clear()

        # Previous social decisions cannot suppress new goal observations.
        self.retired_social_episode_ids.clear()
        self.retired_social_sessions.clear()
        self._handled_persistent_person_episodes.clear()
        self.last_navigation_interruption_key = None

        # Reset observation-health evidence from before goal reception.
        self.latest_observation_health = "UNKNOWN"
        self.latest_observation_health_at = None
        self.empty_observation_since = None
        self.health_session = ""
        self.health_stamp_ns = None
        self.route_geometry_person_id = ""

        # Remove previous dialogue and participant ownership.
        self.mode_d_cases.clear()
        self.mode_d_hri_owner = None
        self.mode_d_hri_turns.clear()
        self.mode_d_hri_closures.clear()
        self.mode_d_anticipation_queues.clear()

        # Clear obsolete trajectory transactions.
        self.pending_generation_request_id = ""
        self.uncertain_generation_request_id = ""
        self.generation_requested_at = None
        self.candidate_path_id = ""
        self.candidate_navigation_goal_id = ""

        # Discard previous navigation-state evidence.
        self.latest_navigation_state = None
        self.navigation_state = "UNKNOWN"

        # Restart goal-scoped supervision.
        self.resolution_watch_goal_id = ""
        self.resolution_watch_started_at = None
        self.reassessment_last_at = 0.0
        self.reassessment_last_stamp_ns = None

        # Keep producer session and sequence ordering intact so that a
        # fresh observation of the same upstream episode is accepted.
        self.social_episode_origin_goal_id = goal_id

        self._transition_to(
            STATE_NAVIGATING,
            reason="NEW_PUBLIC_GOAL_FRESH_SOCIAL_CONTEXT",
        )

        self.get_logger().info(
            "[GOAL SOCIAL RESET] "
            f"goal={goal_id} | "
            "previous_social_state_discarded=True | "
            "same_person_can_be_observed_again=True"
        )
        
    def _navigation_goal_context_callback(self, msg):
        """
        Establish a new public goal independently of pre-goal social history.

        The first fresh observation after goal reception belongs to
        the new goal, even if the physical person was seen previously.
        """
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

        if goal_id == self.active_navigation_goal_id:
            self.active_navigation_goal_pose = msg.goal
            return

        # Never discard an unfinished physical stop or termination.
        if self.pending_terminal_stop is not None:
            self.get_logger().error(
                "[GOAL] Previous public termination is still pending."
            )
            return

        if (
            self.active_navigation_goal_id
            and (
                self.navigation_active
                or self.nav2_cancel_requested
                or self.pending_stop_hold_request_id
                or self.pending_gate_command is not None
                or self.pending_hri_close is not None
            )
        ):
            self.get_logger().warning(
                "[GOAL] New context deferred because the previous "
                "navigation or control transaction is still active."
            )
            return

        # Invalidate trajectory evidence from the previous goal.
        self._invalidate_trajectory_validation(
            reason="NEW_PUBLIC_GOAL"
        )

        # Reset all social history before adopting the new goal.
        self._reset_social_context_for_new_goal(goal_id)

        self.active_navigation_goal_id = goal_id
        self.active_navigation_goal_pose = msg.goal

        # Reset exact Gate START transaction ownership.
        self.observation_loss_held_goal_id = ""
        self.gate_ready_goal_id = ""
        self.pending_gate_command = None
        self.latest_start_request_id = ""

        self.initial_start_authorized_goal_id = ""
        self.initial_start_attempt_count = 0
        self.initial_start_next_attempt_at = 0.0

        self.get_logger().info(
            "[NAVIGATION GOAL CONTEXT] "
            f"goal_id={goal_id!r} | "
            f"frame={msg.goal.header.frame_id!r} | "
            "fresh_social_transaction=True"
        )

        self.get_logger().info(
            "[GOAL] Awaiting matching Goal Gate readiness | "
            f"goal={goal_id}"
        )
        
    def _interrupt_navigation_for_existing_social_episode(self):
        """
        Interrupt only an actionable social intrusion.

        A resolved episode cannot interrupt its own RESUME
        transaction through repeated social observations.

        A genuinely new episode is still actionable.
        Observation loss is handled by its dedicated policy.
        """
        goal_id = str(
            self.active_navigation_goal_id or ""
        ).strip()

        if not goal_id or not self.human_proximity_active:
            return False

        if self.pending_terminal_stop is not None:
            return True

        decision = self.latest_social_decision

        person_id = str(
            (
                decision.person_id
                if decision is not None
                else self.active_person_id
            ) or ""
        ).strip()

        scope = self._social_scope_key(person_id)

        same_scope = self._same_resolved_social_scope(
            person_id
        )

        # A resolved episode cannot issue another interruption
        # just because visibility changed or Nav2 reported
        # RESUMING with navigation_active=False.
        #
        # This is NOT a general immunity against new intrusions.
        # A new episode receives a different scope and therefore
        # does not satisfy this condition.
        if same_scope:
            if self.fsm_state == STATE_WAITING:
                self._schedule_social_reassessment()

            return False

        # An unconfirmed event must never acquire authority
        # over an ongoing Gate transaction.
        if decision is None:
            return False

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
            return False

        nav = self.latest_navigation_state

        if (
            nav is None
            or str(nav.navigation_goal_id or "").strip()
                != goal_id
            or not self._message_is_recent(nav, 1.0)
        ):
            return False

        # Preserve an existing announcement transaction.
        if self.fsm_state == STATE_ANNOUNCING_DETECTION:
            if self.announcement_context_key != (
                "WAIT",
                goal_id,
            ):
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

        # IMPORTANT:
        # A newly confirmed episode is allowed to interrupt
        # RESUMING even before Nav2 reports active=True.
        if self.fsm_state not in {
            STATE_NAVIGATING,
            STATE_RESUMING,
            STATE_OBSERVING,
            STATE_WAITING,
        }:
            return False

        interruption_key = (
            goal_id,
            self.social_state_session,
            self.social_state_episode,
            person_id,
        )

        if (
            self.last_navigation_interruption_key
            == interruption_key
        ):
            return False

        # Only a positive, current social observation can
        # initiate a new conflict stop.
        if (
            not self.social_observation_available
            or self.social_loss_expired
            or self.social_episode_retired
        ):
            return False

        self.announcement_social_decision = decision
        self.announcement_decision_id = str(
            decision.decision_id or ""
        ).strip()

        self.active_person_id = person_id

        if not self.announcement_decision_id:
            return False

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

        # The initial public goal may have been accepted by
        # the Gate without beginning physical movement.
        prestart = bool(
            self.fsm_state != STATE_RESUMING
            and self.initial_start_authorized_goal_id
                != goal_id
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

        # Reserve the exact interruption identity BEFORE
        # publishing WAIT to avoid duplicate callbacks.
        self.last_navigation_interruption_key = (
            interruption_key
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
        """Match Gate acknowledgements to their exact command transaction."""
        try:
            data = json.loads(msg.data)

            if (
                not isinstance(data, dict)
                or data.get("version") != 1
                or data.get("producer") != "navigation_goal_gate"
            ):
                return

            goal_id = str(
                data.get("navigation_goal_id") or ""
            ).strip()

            action = str(
                data.get("action") or ""
            ).strip().upper()

            request_id = str(
                data.get("request_id") or ""
            ).strip()

            status = str(
                data.get("status") or ""
            ).strip().upper()

            path_id = str(
                data.get("path_id") or ""
            ).strip()

        except (ValueError, TypeError):
            return

        if (
            not goal_id
            or goal_id != self.active_navigation_goal_id
        ):
            return

        if action == "GOAL_READY" and status == "READY":
            self.gate_ready_goal_id = goal_id
            self._close_obsolete_pre_navigation_episode()
            self._evaluate_initial_goal_start()
            return

        if action == "WAIT" and status == "HOLD_CONFIRMED":
            if (
                self.fsm_state == STATE_STOPPING_FOR_SOCIAL_CONFLICT
                and self.nav2_cancel_requested
                and request_id
                and request_id == self.pending_stop_hold_request_id
            ):
                self.nav2_cancel_confirmed = True
                self._evaluate_stop_barrier()

            return

        pending = self.pending_gate_command

        if not pending or (
            pending["goal_id"] != goal_id
            or pending["action"] != action
            or pending["request_id"] != request_id
            or (
                action == "RESUME"
                and pending["path_id"] != path_id
            )
        ):
            return

        if action == "REPLAN" and status == "HOLD_CONFIRMED":
            if self.fsm_state != STATE_REPLANNING:
                return

            self.pending_gate_command = None
            self._replan_hold_ready = goal_id

            self._request_current_trajectory_validation(
                generation_reason="REPLAN",
                directional_constraint=(
                    self.pending_replan_directional_constraint
                ),
            )
            return

        if status in {"REJECTED", "EXPIRED"}:
            self.pending_gate_command = None
            self._replan_hold_ready = None

            if action == "START":
                self.initial_start_authorized_goal_id = ""
                self.initial_start_next_attempt_at = (
                    time.monotonic() + 0.5
                )

            self._hold_route_search(
                reason=f"GATE_{action}_{status}",
                reconsider=True,
            )
            return

        if action == "START" and status == "ACCEPTED":
            self.initial_start_authorized_goal_id = goal_id
            self.pending_gate_command = None

        elif action == "RESUME" and status in {
            "ACCEPTED",
            "ACCEPTED_PENDING_CANDIDATE",
            "ACCEPTED_PENDING_TERMINATION",
        }:
            pending["status"] = status
            pending["accepted_at"] = time.monotonic()

    def _navigation_state_callback(self, msg):
        self.latest_navigation_state = msg

        goal_id = str(
            msg.navigation_goal_id or ""
        ).strip()

        if (
            not goal_id
            or goal_id != self.active_navigation_goal_id
            or not self._message_is_recent(msg, 1.0)
        ):
            return

        self.navigation_state = normalize_navigation_state(
            msg.navigation_state
        )
        self.navigation_active = bool(msg.navigation_active)

        self.get_logger().debug(
            "[NAVIGATION STATE] "
            f"goal={goal_id} | "
            f"state={self.navigation_state} | "
            f"active={self.navigation_active}"
        )

        # Public goal termination is authoritative.
        if self.navigation_state in {
            "SUCCEEDED",
            "ABORTED",
            "CANCELED",
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

        # Preserve the existing REPLAN acknowledgement path.
        if (
            self.fsm_state == STATE_REPLANNING
            and self.navigation_state == "REPLANNING"
            and not self.navigation_active
            and not self.pending_generation_request_id
        ):
            direction = (
                self.pending_replan_directional_constraint
            )

            self._request_current_trajectory_validation(
                generation_reason="REPLAN",
                directional_constraint=direction,
            )

            self._evaluate_stop_barrier()
            return

        # CRITICAL ORDER:
        # Confirm actual execution before evaluating repeated
        # events from the social episode that was just resolved.
        if self.fsm_state == STATE_RESUMING:
            if (
                self.navigation_state == NAVIGATION_NAVIGATING
                and self.navigation_active
            ):
                self._finish_episode_after_resume()

                # A new independently confirmed episode will
                # be processed through the social-event callback.
                return

            # RESUMING with active=False is expected while
            # the Gate submits the FollowPath action.
            # It must not be interpreted as a fresh conflict.
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

            if self._mode_c_reactive_generation_allowed():
                self._enter_replanning(
                    reason="MODE_C_REACTIVE_AFTER_OBSERVATION_LOSS",
                    directional_constraint="NONE",
                )

            else:
                self._hold_route_search(
                    reason="SOCIAL_OBSERVATION_UNAVAILABLE_AFTER_SPEECH",
                    reconsider=True,
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

    def _begin_autonomous_readaptation_announcement(
        self,
        *,
        reason,
    ):
        """Close the HRI turn before announcing an autonomous fallback."""
        self.dialogue_started_at = None
        self.waiting_for_human_initiative = False
        self._clear_pending_human_preference()

        reason = str(reason or "").strip().upper()

        if (
            self.pending_terminal_stop is not None
            or self.social_resolution_aborted
        ):
            return

        if (
            self.pending_hri_close is not None
            or self.pending_dialogue_request_id
            or self.dialogue_audio_recovery_pending
        ):
            self._readaptation_after_hri_close = (
                self.active_navigation_goal_id,
                self._social_scope_key(),
                reason,
            )

            if self.pending_hri_close is None:
                if not self._request_hri_close(
                    reason="AUTONOMOUS_READAPTATION"
                ):
                    self._hold_route_search(
                        reason="HRI_CLOSE_TRANSACTION_UNAVAILABLE",
                        reconsider=False,
                    )
                    return

            self._hold_route_search(
                reason="READAPTATION_AWAITING_HRI_CLOSE",
                reconsider=True,
            )
            return

        self._readaptation_after_hri_close = None
        self._ensure_interaction_id()

        self._transition_to(
            STATE_ANNOUNCING_DETECTION,
            reason="ANNOUNCING_AUTONOMOUS_READAPTATION",
        )

        self._request_autonomous_readaptation_explanation(
            reason=reason
        )

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

        # L'identité doit être renseignée AVANT la réservation.
        request.person_id = str(
            self.active_person_id or ""
        ).strip()

        request.response_timeout = float(
            self.dialogue_response_timeout_sec
        )
        request.language = self.language

        if self.interaction == "dialogue":
            if not self._mode_d_bind_hri_turn(request):
                self.get_logger().warning(
                    "[HRI OWNER] Turn not created | "
                    f"person={request.person_id!r} | "
                    "reason=PERSON_SCOPE_UNAVAILABLE_OR_BUSY"
                )
                return None

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

        if request is None:
            self._hold_route_search(
                reason="HRI_PERSON_SCOPE_UNAVAILABLE",
                reconsider=True,
            )
            return

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

        if request is None:
            self._hold_route_search(
                reason="HRI_PERSON_SCOPE_UNAVAILABLE",
                reconsider=True,
            )
            return

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
        if request is None:
            self._hold_route_search(
                reason="HRI_PERSON_SCOPE_UNAVAILABLE",
                reconsider=True,
            )
            return
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


    def _directional_identity_callback(self, msg):
        """Retain the latest spatial identity observation."""
        import copy

        self.directional_identity_snapshot = copy.deepcopy(msg)
        self.directional_identity_received_at = time.monotonic()

    def _request_directional_constraint_installation(self, scope):
        """Request a person-relative keepout mask after confirmed consent.

        This method never authorizes navigation execution.
        """
        import math
        import uuid

        if (
            self.interaction != "dialogue"
            or self.confirmed_human_intent != "ALLOW_PASSAGE"
            or self.confirmed_human_direction not in {"LEFT", "RIGHT"}
            or not isinstance(scope, tuple)
            or len(scope) != 5
            or not all(scope)
            or scope != self._social_scope_key()
        ):
            return False

        goal_id, _, _, tracking_session, person_id = scope

        if (
            goal_id != self.active_navigation_goal_id
            or person_id != self.active_person_id
            or self.pending_terminal_stop is not None
            or self.social_resolution_aborted
        ):
            return False

        snapshot = self.directional_identity_snapshot
        received_at = self.directional_identity_received_at

        if snapshot is None or received_at is None:
            self.get_logger().warning(
                "[DIRECTIONAL MASK] Identity observation unavailable"
            )
            return False

        if time.monotonic() - received_at > 0.5:
            self.get_logger().warning(
                "[DIRECTIONAL MASK] Identity observation expired"
            )
            return False

        if (
            str(snapshot.tracking_session_id).strip()
            != tracking_session
        ):
            return False

        if not bool(snapshot.source_observation_complete):
            return False

        source_stamp = snapshot.header.stamp
        source_ns = (
            int(source_stamp.sec) * 1_000_000_000
            + int(source_stamp.nanosec)
        )

        source_age = (
            self.get_clock().now().nanoseconds - source_ns
        ) / 1e9

        if not math.isfinite(source_age) or not -0.1 <= source_age <= 0.5:
            return False

        persons = [
            person
            for person in snapshot.persons
            if (
                str(person.id).strip() == person_id
                and str(person.tracking_session_id).strip()
                == tracking_session
                and str(person.tracking_state).strip() == "TRACKED"
                and bool(person.observed_in_current_array)
                and int(person.identity_generation) > 0
            )
        ]

        if len(persons) != 1:
            self.get_logger().warning(
                "[DIRECTIONAL MASK] Person identity not uniquely verified"
            )
            return False

        # One installation transaction per confirmed social scope.
        if self.directional_constraint_scope == scope:
            return bool(self.directional_constraint_request_id)

        request = DirectionalConstraintRequest()

        request.header.stamp = self.get_clock().now().to_msg()
        request.header.frame_id = str(snapshot.header.frame_id)

        request.request_id = f"DIR_{uuid.uuid4().hex}"
        request.navigation_goal_id = goal_id
        request.interaction_id = self.active_interaction_id
        request.person_id = person_id
        request.tracking_session_id = tracking_session
        request.identity_generation = int(persons[0].identity_generation)

        request.action = "INSTALL"
        request.directional_constraint = self.confirmed_human_direction

        # Installation lease; observation freshness remains checked
        # independently by the mask producer.
        request.validity_duration.sec = 30
        request.validity_duration.nanosec = 0

        if not request.interaction_id:
            return False

        self.directional_constraint_pub.publish(request)

        self.directional_constraint_request_id = request.request_id
        self.directional_constraint_scope = scope

        self.get_logger().info(
            "[DIRECTIONAL MASK] INSTALL requested | "
            f"goal={goal_id} | "
            f"person={person_id} | "
            f"direction={request.directional_constraint} | "
            f"request={request.request_id}"
        )

        return True


    def _handle_confirmation_response(self, intent):
        """Commit confirmed intent, then close the exact HRI turn."""

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
        
        person_scope = self.mode_d_hri_turns.get(
            (
                self.active_interaction_id,
                self.pending_dialogue_request_id,
            )
        )

        if (
            self.interaction == "dialogue"
            and (
                person_scope is None
                or person_scope != self.mode_d_hri_owner
            )
        ):
            self._hold_route_search(
                reason="HRI_PERSON_TRANSACTION_MISMATCH",
                reconsider=False,
            )
            return

        confirmed_intent = str(
            self.pending_human_intent or ""
        ).strip().upper()

        direction = self._normalize_direction(
            self.pending_human_direction
        )

        if self.interaction == "dialogue":
            case = self._mode_d_case(person_scope)

            case["confirmed_intent"] = confirmed_intent
            case["confirmed_direction"] = (
                direction
                if confirmed_intent == "ALLOW_PASSAGE"
                else "NONE"
            )

            case["dialogue_outcome"] = (
                "ALLOW_CONFIRMED"
                if confirmed_intent == "ALLOW_PASSAGE"
                else "DENY_CONFIRMED"
            )

            case["hri_closed"] = False

            if (
                confirmed_intent == "DENY_PASSAGE"
                and direction in {"LEFT", "RIGHT"}
            ):
                case["forbidden_direction"] = direction

        scope = self.pending_dialogue_scope

        if (
            confirmed_intent not in {
                "ALLOW_PASSAGE",
                "DENY_PASSAGE",
            }
            or scope is None
            or not all(scope)
            or scope != self._social_scope_key()
            or not self.pending_dialogue_request_id
            or not self.active_interaction_id
        ):
            self._hold_route_search(
                reason="CONFIRMATION_CONTEXT_INVALID",
                reconsider=False,
            )
            return

        # Bind the semantic decision to this exact actor,
        # episode and public navigation goal.
        self.confirmed_human_intent = confirmed_intent
        self.confirmed_human_direction = direction

        self.executed_social_scope = None

        if confirmed_intent == "ALLOW_PASSAGE":
            self.authorized_episode_key = scope
            self.dialogue_outcome = "ALLOW_CONFIRMED"

        else:
            self.authorized_episode_key = None
            self.dialogue_outcome = "DENY_CONFIRMED"

            if direction in {"LEFT", "RIGHT"}:
                self.forbidden_human_direction = direction

            self.confirmed_human_direction = "NONE"

        self._clear_pending_human_preference()

        self.dialogue_started_at = None
        self.waiting_for_human_initiative = False
        self.dialogue_retry_count = 0
        self.dialogue_speech_retry_count = 0

        # Simple C/D: confirmed consent is independent of HRI closure.
        if hasattr(self, "_begin_direct_nav2_handoff"):

            if confirmed_intent == "ALLOW_PASSAGE":
                # Preserve the exact consent identity before HRI cleanup.
                self._directional_confirmed_scope = scope
                self._directional_confirmed_interaction_id = str(
                    self.active_interaction_id or ""
                ).strip()

            # Request HRI closure without waiting for its acknowledgment.
            self._request_hri_close(
                reason="SEMANTIC_INTENT_CONFIRMED"
            )

            # Detach HRI ownership from the navigation transaction.
            self._detach_hri_turn()

            self._transition_to(
                STATE_WAITING,
                reason="HUMAN_INTENT_CONFIRMED",
            )

            if confirmed_intent == "ALLOW_PASSAGE":
                self._release_navigation(direction)
            else:
                self._hold_after_denial()

            return

        # Important: request close BEFORE clearing HRI IDs.
        close_requested = self._request_hri_close(
            reason="SEMANTIC_INTENT_CONFIRMED"
        )

        if not close_requested:
            self.get_logger().error(
                "[DIALOGUE] Exact HRI closure unavailable; "
                "navigation remains inhibited."
            )

            self._publish_wait(
                reason="HRI_CLOSE_TRANSACTION_UNAVAILABLE"
            )

            self._transition_to(
                STATE_WAITING,
                reason="HRI_CLOSE_TRANSACTION_UNAVAILABLE",
            )
            return

        if (
            confirmed_intent == "ALLOW_PASSAGE"
            and direction in {"LEFT", "RIGHT"}
        ):
            installation_requested = (
                self._request_directional_constraint_installation(scope)
            )

            if not installation_requested:
                self._publish_wait(
                    reason="DIRECTIONAL_MASK_INSTALLATION_UNAVAILABLE"
                )

                self._transition_to(
                    STATE_WAITING,
                    reason="DIRECTIONAL_MASK_INSTALLATION_UNAVAILABLE",
                )
                return
            
        # The close transaction now owns the original IDs.
        self.pending_dialogue_request_id = ""
        self.pending_dialogue_scope = None

        if (
            self.active_navigation_goal_id
            and self.pending_terminal_stop is None
            and not self.social_resolution_aborted
        ):
            self.resolution_watch_goal_id = (
                self.active_navigation_goal_id
            )
            self.resolution_watch_started_at = time.monotonic()

        self.get_logger().info(
            "[DIALOGUE] Confirmed intent retained | "
            f"scope={scope} | "
            f"intent={confirmed_intent} | "
            f"direction={direction} | "
            "hri_close_pending=True | "
            "navigation_authorized=False"
        )

        # Confirmed permission does not renew the Gate WAIT.
        # Explicit human denial preserves the stop policy.
        if confirmed_intent == "DENY_PASSAGE":
            self._publish_wait(
                reason="HUMAN_DENIED_PASSAGE"
            )

        self._transition_to(
            STATE_WAITING,
            reason="HUMAN_INTENT_CONFIRMED",
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
        if request is None:
            self._hold_route_search(
                reason="HRI_PERSON_SCOPE_UNAVAILABLE",
                reconsider=True,
            )
            return
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
        """Generate once, after the matching REPLAN hold acknowledgement."""
        if (
            getattr(self, "_replan_hold_ready", None)
            != self.active_navigation_goal_id
        ):
            return

        if (
            self.fsm_state != STATE_REPLANNING
            or not self._reactive_planning_ready()
            or self.anticipatory_pending_request_id
            or self.anticipatory_uncertain_request_id
            or self.announcement_context_key is not None
            or self.current_announcement_action
        ):
            return

        if self.route_attempt_count >= self.route_search_max_attempts:
            self._hold_route_search(
                reason="SOCIAL_ROUTE_ATTEMPTS_EXHAUSTED",
                reconsider=True,
            )
            return

        direction = self._normalize_direction(
            directional_constraint
        )

        if (
            direction in {"LEFT", "RIGHT"}
            and direction == self.forbidden_human_direction
        ):
            self._hold_route_search(
                reason="HUMAN_FORBIDDEN_DIRECTION",
                reconsider=False,
            )
            return

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
                reason="TRAJECTORY_GENERATION_CONTEXT_UNAVAILABLE",
                reconsider=True,
            )
            return

        self._replan_hold_ready = None

        self.route_attempt_count += 1
        self.route_tried_directions.add(direction)

        self.pending_generation_request_id = request.request_id
        self.generation_requested_at = time.monotonic()

        self.trajectory_generation_request_pub.publish(request)

    def _build_trajectory_generation_request(
        self,
        *,
        generation_reason,
        directional_constraint,
    ):
        """Build a request and retain its immutable public-goal owner."""
        if (
            not self.active_navigation_goal_id
            or self.active_navigation_goal_pose is None
        ):
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

        request.person_id = (
            self.route_geometry_person_id
            if (
                self.interaction == "none"
                and request.directional_constraint in {"LEFT", "RIGHT"}
            )
            else ""
            if self.interaction == "none"
            else self.active_person_id
        )

        if not hasattr(self, "_generation_owners"):
            self._generation_owners = {}

        self._generation_owners[request.request_id] = (
            request.navigation_goal_id
        )

        return request
    
    def _mode_d_person_scope(self, person_id):
        """Identifier une incarnation et son épisode individuel."""

        person_id = str(person_id or "").strip()
        goal_id = str(
            self.active_navigation_goal_id or ""
        ).strip()

        if not goal_id or not person_id:
            return None

        matches = [
            (identity, state)
            for identity, state
            in self.individual_participants.items()
            if (
                identity[1] == person_id
                and state["episode_id"]
                and not state["retired"]
            )
        ]

        # Une identité ambiguë ne peut pas recevoir une
        # nouvelle autorisation sémantique.
        if len(matches) != 1:
            return None

        identity, state = matches[0]

        return (
            goal_id,
            identity[0],
            identity[1],
            identity[2],
            state["episode_id"],
        )

    def _mode_d_case(self, scope):
        """Obtenir l'état propre à une personne et un épisode."""

        if scope is None:
            return None

        if scope not in self.mode_d_cases:
            self.mode_d_cases[scope] = {
                "confirmed_intent": "UNKNOWN",
                "confirmed_direction": "NONE",
                "dialogue_outcome": "NONE",
                "hri_closed": False,
                "forbidden_direction": "NONE",
                "candidate_request_ids": set(),
            }

        return self.mode_d_cases[scope]

    def _mode_d_bind_hri_turn(self, request):
        """Réserver le propriétaire exact d'un tour HRI."""

        scope = self._mode_d_person_scope(
            request.person_id
        )

        if scope is None:
            return False

        if self.mode_d_hri_owner not in {
            None,
            scope,
        }:
            return False

        identity = (
            request.interaction_id,
            request.request_id,
        )

        if identity in self.mode_d_hri_turns:
            return False

        self.mode_d_hri_owner = scope
        self.mode_d_hri_turns[identity] = scope

        self._mode_d_case(scope)

        return True

    def _candidate_trajectory_callback(self, msg):
        """Accept candidates independently of cross-topic terminal-result order."""
        request_id = str(msg.request_id or "").strip()
        path_id = str(msg.path_id or "").strip()
        goal_id = str(msg.navigation_goal_id or "").strip()

        if (
            not all((request_id, path_id, goal_id))
            or not msg.path.header.frame_id
            or not msg.path.poses
        ):
            return

        metadata = self.anticipatory_requests.get(request_id)

        if metadata is not None:
            scope = metadata["scope"]

            if (
                scope != self._anticipation_scope()
                or goal_id != scope[0]
                or metadata.get("status") not in {
                    "GENERATING",
                    "READY",
                    "TERMINATED",
                }
            ):
                return

            existing = self.anticipatory_candidates.get(request_id)

            if existing is not None:
                return

            self.anticipatory_candidates[request_id] = {
                "scope": scope,
                "direction": metadata["direction"],
                "path_id": path_id,
                "candidate": copy.deepcopy(msg),
                "received_at": time.monotonic(),
                "status": "READY",
            }

            if metadata["status"] != "TERMINATED":
                metadata["status"] = "READY"

            return

        if (
            self.fsm_state not in {
                STATE_REPLANNING,
                STATE_VALIDATING_TRAJECTORY,
            }
            or goal_id != self.active_navigation_goal_id
            or request_id != self.pending_generation_request_id
            or self.uncertain_generation_request_id
        ):
            return

        self.pending_generation_request_id = ""

        getattr(
            self,
            "_generation_owners",
            {},
        ).pop(request_id, None)

        self.trajectory_validation = TRAJECTORY_VALIDATION_UNKNOWN
        self.validated_path_id = ""
        self.validation_received_at = None
        self.validation_source_stamp_sec = None

        self.candidate_path_id = path_id
        self.candidate_navigation_goal_id = goal_id

        self.reactive_cached_candidate = copy.deepcopy(msg)
        self._reactive_cached_scope = self._social_scope_key()
        self._reactive_cached_direction = (
            self.pending_replan_directional_constraint
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
        """Accept terminal evidence for the exact recorded request owner."""
        try:
            data = json.loads(msg.data)

            if (
                not isinstance(data, dict)
                or data.get("version") != 1
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
            )

        except (ValueError, TypeError):
            return

        if (
            not goal_id
            or not request_id
            or status not in {
                "FAILED",
                "TERMINATED",
                "TERMINATION_UNCONFIRMED",
            }
        ):
            return

        metadata = self.anticipatory_requests.get(request_id)
        owners = getattr(self, "_generation_owners", {})

        owner = (
            metadata["scope"][0]
            if metadata is not None
            else owners.get(request_id)
        )

        if goal_id != owner:
            return

        if metadata is not None:
            if status == "TERMINATION_UNCONFIRMED":
                if metadata.get("status") in {
                    "TERMINATED",
                    "FAILED",
                }:
                    return

                metadata["status"] = status
                self.anticipatory_uncertain_request_id = request_id
                return

            metadata["status"] = status

            if self.anticipatory_pending_request_id == request_id:
                self.anticipatory_pending_request_id = ""

            if self.anticipatory_uncertain_request_id == request_id:
                self.anticipatory_uncertain_request_id = ""

            if status == "FAILED":
                self.anticipatory_candidates.pop(request_id, None)

            owners.pop(request_id, None)
            return

        if request_id not in {
            self.pending_generation_request_id,
            self.uncertain_generation_request_id,
        }:
            return

        if status == "TERMINATION_UNCONFIRMED":
            self.uncertain_generation_request_id = request_id

            if request_id == self.pending_generation_request_id:
                self.pending_generation_request_id = ""

            if goal_id == self.active_navigation_goal_id:
                self._hold_route_search(
                    reason="PLANNING_TERMINATION_UNCONFIRMED",
                    reconsider=True,
                )

            return

        if self.pending_generation_request_id == request_id:
            self.pending_generation_request_id = ""

        if self.uncertain_generation_request_id == request_id:
            self.uncertain_generation_request_id = ""

        owners.pop(request_id, None)
        self.planning_reconciliation = None

        if (
            self.social_resolution_aborted
            or self.pending_terminal_stop is not None
        ):
            return

        if goal_id == self.active_navigation_goal_id:
            self._hold_route_search(
                reason=f"PLANNING_{status}:{reason}",
                reconsider=True,
            )
     
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
            
            self.route_geometry_person_id = str(
                msg.person_id or ""
            ).strip()
                        

            self.reactive_cached_candidate = None
            scope = self._social_scope_key()

            if self.conflict_evidence_scope != scope:
                self.conflict_evidence_scope = scope
                self.conflicting_candidate_ids = set()

            # Duplicate results for one path count only once.
            self.conflicting_candidate_ids.add(path_id)

            self._invalidate_trajectory_validation(
                reason="CANDIDATE_SOCIAL_CONFLICT"
            )

            if (
                self.interaction == "dialogue"
                and self.anticipatory_planning_enabled
                and self.dialogue_outcome == "ALLOW_CONFIRMED"
                and self._permission_applies_to_current_episode()
                and self.anticipatory_selected_request_id
            ):
                # The physical conflict remains real.
                # Keep human permission, wait for a new observation,
                # and revalidate the cached path without new Nav2 planning.
                self._hold_route_search(
                    reason="CACHED_PATH_SOCIAL_CONFLICT",
                    reconsider=True,
                )

                self._request_passage_assistance()
                return

            self._request_next_social_alternative()
            return

        if result == TRAJECTORY_VALIDATION_UNKNOWN:
            
            self.mode_c_last_checked_source_ns = max(
                self.mode_c_last_checked_source_ns,
                int(self.health_stamp_ns or 0),
            )
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

        if (
            not hold_reason
            and self._individual_scene_blocks_motion()
        ):
            hold_reason = "INDIVIDUAL_EPISODE_UNRESOLVED"
            
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
        """Revoke path authorization without forgetting outstanding planning."""
        pending = self.pending_generation_request_id

        if pending:
            if self.uncertain_generation_request_id not in {
                "",
                pending,
            }:
                self.get_logger().error(
                    "Concurrent ordinary planning ownership; "
                    "retaining both IDs."
                )
            else:
                self.uncertain_generation_request_id = pending
                self.pending_generation_request_id = ""

        self.trajectory_validation = TRAJECTORY_VALIDATION_UNKNOWN
        self.validated_path_id = ""
        self.validation_received_at = None
        self.validation_source_stamp_sec = None

        self.candidate_path_id = ""
        self.candidate_navigation_goal_id = ""
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
        """
        Retirement and empty observations never resolve a missing participant.

        Recovery requires new factual evidence through the normal
        observation and exact-path validation pipeline.
        """
        return False
    
    def _trajectory_execution_allowed(self, path_id):
        """
        Authorize only an exact freshly validated trajectory.

        Human consent is semantic context.
        It never replaces current observation or path validation.
        """
        goal_id = str(
            self.active_navigation_goal_id or ""
        ).strip()

        path_id = str(path_id or "").strip()

        if (
            self.social_resolution_aborted
            or self.pending_hri_close is not None
            or self.pending_terminal_stop is not None
            or self.hri_close_fault_reported
            or self.pending_speech_decision_id
            or self.dialogue_audio_recovery_pending
        ):
            return False

        if not goal_id or not path_id:
            return False

        if self.active_navigation_goal_pose is None:
            return False

        if self.candidate_navigation_goal_id != goal_id:
            return False

        if (
            self.trajectory_validation
            != TRAJECTORY_VALIDATION_VALID
        ):
            return False

        if not (
            path_id
            == self.validated_path_id
            == self.candidate_path_id
        ):
            return False

        if not self._trajectory_validation_is_fresh():
            return False
        
        if self._individual_scene_blocks_motion():
            return False

        ordinary_recovery = bool(
            self._route_observation_is_fresh()
        )

        # Retired-episode recovery has its own evidence policy.
        # It must remain bound to this exact validated path.
        retired_recovery = bool(
            self.retired_recovery_path_id == path_id
            and self._retired_recovery_evidence_available()
        )

        return bool(
            ordinary_recovery
            or retired_recovery
        )

    # ================================================================
    # Replanning and resuming
    # ================================================================

    def _unknown_candidate_revalidation_tick(self, now):
        """Revalidate a retained candidate without replacing another transaction."""
        if (
            getattr(self, "_readaptation_after_hri_close", None)
                is not None
            or self.fsm_state != STATE_WAITING
            or not self.route_waiting_for_change
            or not self._reactive_planning_ready()
            or self.announcement_context_key is not None
            or self.current_announcement_action
            or self.anticipatory_pending_request_id
            or self.anticipatory_uncertain_request_id
        ):
            return

        if (
            self.interaction == "dialogue"
            and (
                self.dialogue_outcome not in {
                    "ALLOW_CONFIRMED",
                    "DENY_CONFIRMED",
                    "NO_RESPONSE",
                    "NOT_UNDERSTOOD",
                }
                or self.mode_d_hri_owner is not None
            )
        ):
            return

        candidate = self.reactive_cached_candidate

        if candidate is None:
            return

        if (
            getattr(self, "_reactive_cached_scope", None)
            != self._social_scope_key()
        ):
            self.reactive_cached_candidate = None
            return

        if (
            self.dialogue_outcome == "ALLOW_CONFIRMED"
            and self.confirmed_human_direction in {"LEFT", "RIGHT"}
        ):
            if (
                getattr(self, "_reactive_cached_direction", None)
                != self.confirmed_human_direction
            ):
                self.reactive_cached_candidate = None
                return

        goal_id = str(
            self.active_navigation_goal_id or ""
        ).strip()

        path_id = str(
            candidate.path_id or ""
        ).strip()

        if (
            not goal_id
            or not path_id
            or not str(candidate.request_id or "").strip()
            or candidate.navigation_goal_id != goal_id
            or not candidate.path.header.frame_id
            or not candidate.path.poses
            or (
                self.candidate_path_id
                and self.candidate_path_id != path_id
            )
        ):
            return

        if (
            not self._fresh_observation_health_available()
            or not self._individual_scene_current()
        ):
            return

        stamp_ns = self.health_stamp_ns

        if (
            stamp_ns is None
            or stamp_ns <= self.mode_c_last_checked_source_ns
            or self.individual_source_stamp_ns != stamp_ns
            or now - self.mode_c_last_revalidation_at < 1.0
        ):
            return

        self.mode_c_last_checked_source_ns = stamp_ns
        self.mode_c_last_revalidation_at = now

        self._invalidate_trajectory_validation(
            reason="MODE_C_FRESH_SOURCE_REVALIDATION"
        )

        self.candidate_path_id = path_id
        self.candidate_navigation_goal_id = goal_id
        self.generation_requested_at = now

        self._transition_to(
            STATE_VALIDATING_TRAJECTORY,
            reason="MODE_C_FRESH_SOURCE_REVALIDATION",
        )

        self.mode_c_revalidation_pub.publish(
            copy.deepcopy(candidate)
        )

        self.get_logger().info(
            "[CANDIDATE REVALIDATION] "
            f"goal={goal_id} | "
            f"path={path_id} | "
            f"source={stamp_ns} | "
            "nav2_generation=False | "
            "navigation_authorized=False"
        )

    def _enter_replanning(
        self,
        reason,
        directional_constraint=None,
    ):
        """Request one exact Gate hold before starting ordinary planning."""
        if (
            getattr(self, "_readaptation_after_hri_close", None)
                is not None
            or self.social_resolution_aborted
            or self.pending_terminal_stop is not None
            or self.pending_hri_close is not None
            or self.pending_gate_command is not None
            or self.pending_generation_request_id
            or self.uncertain_generation_request_id
            or self.anticipatory_pending_request_id
            or self.anticipatory_uncertain_request_id
            or self.pending_speech_decision_id
            or self.pending_dialogue_request_id
            or self.dialogue_audio_recovery_pending
        ):
            return

        if (
            not self.active_navigation_goal_id
            or self.active_navigation_goal_pose is None
        ):
            return

        if not self._reactive_planning_ready():
            self._hold_route_search(
                reason="REPLAN_REQUIRES_CONFIRMED_HOLD",
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

            if self.dialogue_outcome not in {
                "ALLOW_CONFIRMED",
                "DENY_CONFIRMED",
                "NO_RESPONSE",
                "NOT_UNDERSTOOD",
            }:
                return

            if (
                self.dialogue_outcome == "ALLOW_CONFIRMED"
                and not self._permission_applies_to_current_episode()
            ):
                return

        if self.route_attempt_count >= self.route_search_max_attempts:
            if not self._request_passage_assistance():
                self._hold_route_search(
                    reason="SOCIAL_ROUTE_ATTEMPTS_EXHAUSTED",
                    reconsider=True,
                )
            return

        direction = self._normalize_direction(
            self.confirmed_human_direction
            if directional_constraint is None
            else directional_constraint
        )

        if (
            self.dialogue_outcome == "ALLOW_CONFIRMED"
            and self.confirmed_human_direction in {"LEFT", "RIGHT"}
        ):
            direction = self.confirmed_human_direction

        if (
            direction in {"LEFT", "RIGHT"}
            and direction == self.forbidden_human_direction
        ):
            self._hold_route_search(
                reason="HUMAN_FORBIDDEN_DIRECTION",
                reconsider=False,
            )
            return

        self.route_generation_reason = (
            "ALTERNATIVE"
            if direction in {"LEFT", "RIGHT"}
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

        request_id = self._publish_collaborative_decision(
            action="REPLAN",
            outcome="TRAJECTORY_SEARCH",
            human_intent=self.confirmed_human_intent,
            directional_constraint=direction,
            path_id="",
            reason=reason,
        )

        if not request_id:
            self._hold_route_search(
                reason="REPLAN_PUBLICATION_FAILED",
                reconsider=True,
            )
            return

        self.pending_gate_command = {
            "goal_id": self.active_navigation_goal_id,
            "action": "REPLAN",
            "request_id": request_id,
            "path_id": "",
            "sent_at": time.monotonic(),
            "status": "SENT",
        }

        self._replan_hold_ready = None

    def _enter_resuming(self, path_id):
        """
        Submit one exact-path execution authorization.

        Keep reassessment armed for rejection, expiry or timeout.
        The reassessment evaluator cannot run during RESUMING
        or while a Gate transaction remains pending.

        Actual execution confirmation disarms reassessment.
        """
        goal_id = str(
            self.active_navigation_goal_id or ""
        ).strip()

        path_id = str(path_id or "").strip()

        if self.pending_terminal_stop is not None:
            return

        # Never overwrite an outstanding Gate transaction.
        if self.pending_gate_command is not None:
            self.get_logger().warning(
                "[RESUME] Another Gate transaction is pending."
            )
            return

        if not self._trajectory_execution_allowed(path_id):
            self._hold_route_search(
                reason="RESUME_VALIDATION_BARRIER_FAILED",
                reconsider=True,
            )
            return

        # This is a recovery fallback, not an immediate retry.
        # STATE_RESUMING prevents the evaluator from using it.
        self._schedule_social_reassessment()
        self.route_blocked_signature = None

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

        if not request_id:
            self._hold_route_search(
                reason="RESUME_REQUEST_PUBLICATION_FAILED",
                reconsider=True,
            )
            return

        self.pending_gate_command = {
            "goal_id": goal_id,
            "action": "RESUME",
            "request_id": request_id,
            "path_id": path_id,
            "sent_at": time.monotonic(),
            "status": "SENT",
        }

        self.get_logger().info(
            "[RESUME TRANSACTION] "
            f"goal={goal_id} | "
            f"path={path_id} | "
            f"request={request_id} | "
            "recovery_armed=True"
        )

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
        """Retain one WAIT identity throughout the same physical-stop transaction."""
        self.pending_gate_command = None
        self._replan_hold_ready = None

        goal_id = self.active_navigation_goal_id

        if not goal_id:
            return ""

        stopping = bool(
            self.fsm_state == STATE_STOPPING_FOR_SOCIAL_CONFLICT
            and self.nav2_cancel_requested
        )

        scope = (
            goal_id,
            self.stop_request_stamp_sec,
        )

        saved = getattr(self, "_physical_wait", None)
        now = time.monotonic()

        if (
            stopping
            and saved is not None
            and saved["scope"] == scope
        ):
            self.pending_stop_hold_request_id = (
                saved["message"].request_id
            )

            if (
                not self.nav2_cancel_confirmed
                and now - saved["sent_at"] >= 2.0
            ):
                msg = copy.deepcopy(saved["message"])
                msg.header.stamp = self.get_clock().now().to_msg()

                self.collaborative_decision_pub.publish(msg)
                saved["sent_at"] = now

            return self.pending_stop_hold_request_id

        msg = CollaborativeDecision()
        msg.header.stamp = self.get_clock().now().to_msg()

        msg.interaction_id = self.active_interaction_id
        msg.navigation_goal_id = goal_id
        msg.request_id = f"REQ_{uuid.uuid4().hex}"
        msg.person_id = self.active_person_id

        msg.action = "WAIT"
        msg.outcome = "WAITING"
        msg.human_intent = self.confirmed_human_intent
        msg.directional_constraint = self.confirmed_human_direction
        msg.path_id = ""
        msg.reason = reason

        msg.navigation_authorized = False
        msg.requires_replan = False

        if stopping:
            self.pending_stop_hold_request_id = msg.request_id
            self.nav2_cancel_confirmed = False

            self._physical_wait = {
                "scope": scope,
                "message": copy.deepcopy(msg),
                "sent_at": now,
            }

        self.collaborative_decision_pub.publish(msg)
        return msg.request_id
    
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
        """Publish a collaborative command under the execution authorization policy."""
        action = str(action or "").strip().upper()
        path_id = str(path_id or "").strip()
        reason = str(reason or "").strip().upper()

        if action not in {"START", "STOP", "WAIT", "REPLAN", "RESUME"}:
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
                raise RuntimeError("RESUME requires a path_id.")

            if not self._trajectory_execution_allowed(path_id):
                raise RuntimeError(
                    "RESUME rejected: exact trajectory authorization failed."
                )

        elif action == "START" and path_id:
            mode_c_restart = bool(
                reason == "MODE_C_VALIDATED_NAV2_RESTART"
                and str(
                    getattr(self, "_episode_mode", "")
                ).strip().upper() == "C"
                and str(
                    directional_constraint or ""
                ).strip().upper() == "NONE"
                and self._released
                and not self._denied_scope
                and self.dialogue_outcome != "DENY_CONFIRMED"
                and self.pending_terminal_stop is None
                and self._stopped()
                and self._trajectory_execution_allowed(path_id)
            )

            if not mode_c_restart:
                raise RuntimeError(
                    "START with path_id requires validated Mode C recovery."
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
            f"goal={self.active_navigation_goal_id!r} | "
            f"path={path_id!r} | "
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
        """
        Supervise Gate acknowledgement deadlines.

        A timeout does not prove that execution never started.
        Revoke the authorization through WAIT before recovery.

        Recovery requires fresh evidence and the normal Gate
        planning barrier. It never reuses the expired path.
        """
        pending = self.pending_gate_command

        if pending is None:
            return

        goal_id = pending["goal_id"]
        action = pending["action"]
        request_id = pending["request_id"]

        if goal_id != self.active_navigation_goal_id:
            self.pending_gate_command = None
            return

        timeout = {
            "START": 2.0,
            "RESUME": 10.0,
        }.get(action, 5.0)

        elapsed = now - pending["sent_at"]

        if elapsed < timeout:
            return

        self.pending_gate_command = None

        self.get_logger().error(
            "[GATE WATCHDOG] Transaction timeout | "
            f"action={action} | "
            f"goal={goal_id} | "
            f"request={request_id} | "
            f"elapsed={elapsed:.1f}s"
        )

        if action == "START":
            self.initial_start_authorized_goal_id = ""
            self.initial_start_next_attempt_at = now + 0.5

            # Revoke a possibly received but unacknowledged START.
            self._publish_wait(
                reason="START_ACK_TIMEOUT"
            )
            return

        if action == "RESUME":
            # This helper invalidates the old candidate and validation,
            # publishes WAIT and enables observation-driven reassessment.
            # It preserves the confirmed human preference.
            self._hold_route_search(
                reason="RESUME_START_TIMEOUT",
                reconsider=True,
            )
            return

        self._hold_route_search(
            reason=f"GATE_{action}_ACK_TIMEOUT",
            reconsider=True,
        )

    def _timer_callback(self):
        """
        Supervise collaboration and optional Mode-D preplanning.

        Mode C uses its existing control path unchanged.
        Anticipation never publishes execution authorization.
        """

        now = time.monotonic()

        # 1. Physical stop barrier.
        self._evaluate_stop_barrier()

        # 2. HRI close and terminal stop supervision.
        self._watch_hri_close(now)
        self._watch_terminal_stop(now)

        # 3. Observation supervision.
        self._evaluate_social_loss_timeout(now)
        self._apply_pending_social_retirement()

        # 4. Speech and episode supervision.
        self._evaluate_initial_announcement_timeout(now)
        self._evaluate_idle_episode_retirement()

        # 5. Gate transactions.
        self._evaluate_pending_gate_command(now)

        # 6. Deferred clearance.
        self._evaluate_deferred_clear_replan()
        if self.interaction == "dialogue":
            self._try_cached_path_revalidation(now)
            
        # 7. Original route-resolution procedure.
        self._evaluate_route_resolution(now)
        self._unknown_candidate_revalidation_tick(now)
        self._reactive_search_tick(now)
        self._watch_planning_reconciliation(now)

        # 8. Optional speculative generation for mode D.
        # This branch does not exist in mode C.
        if (
            self.interaction == "dialogue"
            and self.anticipatory_planning_enabled
        ):
            self._anticipation_tick(now)

        # 9. Original goal and social supervision.
        if not self.social_resolution_aborted:
            #self._mode_c_reactive_tick()
            self._evaluate_retired_episode_recovery()
            self._evaluate_conflict_persistence(now)
            self._evaluate_initial_goal_start()

        # 10. Existing diagnostic watchdog.
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
        """
        Supervise prolonged loss of social observation.

        Keep semantic consent and its direction.
        Never erase an outstanding planner request.
        Never treat timeout as geometric clearance.
        """
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
        self.waiting_for_observation_reacquisition = True

        self._apply_pending_social_retirement()

        # A planning request belongs to TrajectoryManager.
        # Do not erase its identity while awaiting an
        # authoritative candidate or terminal outcome.
        generation_outstanding = bool(
            self.pending_generation_request_id
            or self.uncertain_generation_request_id
        )

        if not generation_outstanding:
            self._invalidate_trajectory_validation(
                reason="SOCIAL_LOSS_TIMEOUT"
            )

        dialogue_active = bool(
            self.interaction == "dialogue"
            and self.fsm_state in {
                STATE_WAITING_FOR_HUMAN,
                STATE_WAITING_FOR_CONFIRMATION,
                STATE_ASKING,
                STATE_CONFIRMING_HUMAN_INTENT,
            }
            and self.active_interaction_id
            and self.pending_dialogue_request_id
        )

        if dialogue_active:
            self.get_logger().warning(
                "[SOCIAL LOSS] Prolonged loss during dialogue | "
                "HRI transaction preserved | "
                f"interaction={self.active_interaction_id}"
            )
            return

        # Do not overwrite:
        # - confirmed_human_intent
        # - confirmed_human_direction
        # - authorized_episode_key
        # - dialogue_outcome
        #
        # Consent is semantic memory, not permission to move.

        self.pending_human_intent = "UNKNOWN"
        self.pending_human_direction = "NONE"

        self.active_alternative_direction = "NONE"
        self.dialogue_started_at = None
        self.waiting_for_human_initiative = False

        # Do not reset the actual generation direction
        # or a still-running planning transaction.
        self.dialogue_retry_count = 0
        self.dialogue_speech_retry_count = 0

        self.get_logger().warning(
            "[SOCIAL LOSS] Observation timeout | "
            f"goal={self.active_navigation_goal_id} | "
            f"elapsed={elapsed:.2f}s | "
            f"generation_outstanding={generation_outstanding} | "
            f"confirmed_direction={self.confirmed_human_direction}"
        )

        # Planning or validation may finish, but cannot
        # authorize execution using stale observations.
        if self.fsm_state in {
            STATE_REPLANNING,
            STATE_VALIDATING_TRAJECTORY,
        }:
            return

        if (
            self.fsm_state == STATE_WAITING
            and not self.navigation_active
        ):
            self._schedule_social_reassessment()
            return

        # A pending RESUME must not be abandoned silently.
        # The Gate owns the pending command; a prolonged
        # loss prevents treating stale evidence as clearance.
        if self.fsm_state == STATE_RESUMING:
            self._publish_wait(
                reason="SOCIAL_LOSS_TIMEOUT_DURING_RESUME"
            )
            self._transition_to(
                STATE_WAITING,
                reason="SOCIAL_LOSS_TIMEOUT_DURING_RESUME",
            )
            return

        if self.active_navigation_goal_id:
            self._hold_route_search(
                reason="SOCIAL_LOSS_TIMEOUT_HOLD",
                reconsider=True,
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
        # Clear retained consent before opening another context.
        self._retained_same_person_consent = None
        self._directional_retry_pending = False

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

    selector = None
    node = None

    try:
        # Read ROS parameters before selecting the manager class.
        selector = rclpy.create_node(
            "collaboration_manager"
        )

        selector.declare_parameter(
            "collaboration_manager_type",
            "original",
        )

        manager_type = str(
            selector.get_parameter(
                "collaboration_manager_type"
            ).value
        ).strip().lower()

        # Release the temporary node before creating the manager.
        selector.destroy_node()
        selector = None

        if manager_type == "simple_cd":
            from .simple_collaboration_manager import (
                SimpleCDCollaborationManager,
            )

            manager_class = SimpleCDCollaborationManager

        elif manager_type == "original":
            manager_class = CollaborationManager

        else:
            raise ValueError(
                f"Unsupported collaboration manager: {manager_type}"
            )

        node = manager_class()

        node.get_logger().info(
            "[COLLABORATION IMPLEMENTATION] "
            f"selected={manager_type} | "
            f"class={type(node).__name__}"
        )

        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        if selector is not None:
            selector.destroy_node()

        if node is not None:
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()