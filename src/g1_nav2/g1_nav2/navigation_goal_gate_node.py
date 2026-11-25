#!/usr/bin/env python3

"""
Navigation Goal Gate.

The gate is the single execution boundary between social navigation
decisions and Nav2.

Flow:

    User / RViz
        |
        v
    /navigate_to_pose
        |
        v
    NavigationGoalGate <--- /social_navigation/collaborative_decision
        |
        v
    /navigate_to_pose_internal
        |
        v
       Nav2

Final decision contract:

    WAIT
        Cancel the current internal attempt while keeping the public
        navigation goal alive.

    RESUME
        Execute the exact CandidateTrajectory identified by path_id through
        FollowPath. navigation_authorized must be true.

    STOP
        Cancel the internal attempt and terminate the public goal.

    REPLAN
        Place execution in SOCIAL_HOLD, cancel the current internal attempt,
        keep the public goal alive, and wait for a future RESUME(path_id).

The gate does not consume HumanIntent, SocialNavigationDecision or
NavigationControl.

The public navigation goal identity remains stable across WAIT/RESUME
and REPLAN operations.

Speech remains fail-open by default.
"""

import threading
import time
import uuid
from typing import Dict
import copy
import rclpy
import math

import json
from std_msgs.msg import String

from rclpy.time import Time
from tf2_ros import Buffer, TransformListener

from action_msgs.msg import GoalStatus, GoalStatusArray
from nav2_msgs.action import FollowPath, NavigateToPose
from unique_identifier_msgs.msg import UUID

from rclpy.action import (
    ActionClient,
    ActionServer,
    CancelResponse,
    GoalResponse,
)
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)

from g1_social_msgs.msg import (
    CandidateTrajectory,
    CollaborativeDecision,
    ExplanationText,
    NavigationGoalContext,
    NavigationState,
    SpeechStatus,
    TrajectorySocialValidation,
    DirectionalConstraintRequest,
    DirectionalConstraintStatus,
    
)


from g1_qos_manager.profiles import reliable_qos


class NavigationGoalGateNode(Node):
    """
    Gate public NavigateToPose goals before and during Nav2 execution.

    One public goal may own multiple internal Nav2 attempts.

    Internal attempts receive distinct UUIDs.

    Public goal identity remains stable for the whole lifetime of the
    navigation request.
    """

    CONTROL_RUNNING = "RUNNING"
    CONTROL_WAITING = "WAITING"
    CONTROL_SOCIAL_HOLD = "SOCIAL_HOLD"
    CONTROL_REPLANNING = "REPLANNING"
    CONTROL_STOPPING = "STOPPING"

    PHASE_WAITING_FOR_SPEECH = "WAITING_FOR_SPEECH"
    PHASE_WAITING = "WAITING"
    PHASE_SOCIAL_HOLD = "SOCIAL_HOLD"
    PHASE_RESUMING = "RESUMING"
    PHASE_REPLANNING = "REPLANNING"
    PHASE_FORWARDING = "FORWARDING"
    PHASE_NAVIGATING = "NAVIGATING"
    PHASE_STOPPING = "STOPPING"

    def __init__(self) -> None:
        super().__init__("navigation_goal_gate")

        self.callback_group = ReentrantCallbackGroup()

        # ================================================================
        # PARAMETERS
        # ================================================================

        self.declare_parameter("enabled", True)

        self.declare_parameter(
            "public_action_name",
            "/navigate_to_pose",
        )

        self.declare_parameter(
            "internal_action_name",
            "/navigate_to_pose_internal",
        )

        self.declare_parameter(
            "speech_topic",
            "/explainability/text",
        )

        self.declare_parameter(
            "speech_status_topic",
            "/explainability/speech_status",
        )


        self.declare_parameter(
            "collaborative_decision_topic",
            "/social_navigation/collaborative_decision",
        )

        self.declare_parameter(
            "navigation_state_topic",
            "/social_navigation/navigation_state",
        )

        self.declare_parameter(
            "navigation_goal_context_topic",
            "/social_navigation/navigation_goal_context",
        )

        self.declare_parameter(
            "candidate_trajectory_topic",
            "/social_navigation/candidate_trajectory",
        )

        self.declare_parameter(
            "follow_path_action_name",
            "/follow_path",
        )

        self.declare_parameter(
            "language",
            "en",
        )

        self.declare_parameter(
            "speech_timeout_sec",
            35.0,
        )
        self.declare_parameter(
            "goal_success_position_tolerance_m",
            0.25,
        )

        self.declare_parameter(
            "tts_discovery_timeout_sec",
            2.0,
        )

        self.declare_parameter(
            "navigation_server_timeout_sec",
            10.0,
        )

        self.declare_parameter(
            "navigation_cancel_timeout_sec",
            5.0,
        )

        self.declare_parameter(
            "fail_open",
            True,
        )

        self.declare_parameter(
            "assume_idle_on_start",
            False,
        )

        self.declare_parameter(
            "resume_candidate_wait_timeout_sec",
            1.0,
        )

        self.declare_parameter(
            "execution_authorization_max_age_sec",
            2.0,
        )

        # ================================================================
        # RESOLVE PARAMETERS
        # ================================================================

        self.enabled = bool(
            self.get_parameter("enabled").value
        )

        self.public_action_name = str(
            self.get_parameter(
                "public_action_name"
            ).value
        ).strip()

        self.internal_action_name = str(
            self.get_parameter(
                "internal_action_name"
            ).value
        ).strip()

        self.speech_topic = str(
            self.get_parameter(
                "speech_topic"
            ).value
        ).strip()

        self.speech_status_topic = str(
            self.get_parameter(
                "speech_status_topic"
            ).value
        ).strip()

        self.collaborative_decision_topic = str(
            self.get_parameter(
                "collaborative_decision_topic"
            ).value
        ).strip()

        self.navigation_state_topic = str(
            self.get_parameter(
                "navigation_state_topic"
            ).value
        ).strip()

        self.navigation_goal_context_topic = str(
            self.get_parameter(
                "navigation_goal_context_topic"
            ).value
        ).strip()

        self.candidate_trajectory_topic = str(
            self.get_parameter(
                "candidate_trajectory_topic"
            ).value
        ).strip()

        self.follow_path_action_name = str(
            self.get_parameter(
                "follow_path_action_name"
            ).value
        ).strip()

        self.language = str(
            self.get_parameter(
                "language"
            ).value
        ).strip().lower()

        self.speech_timeout_sec = max(
            0.0,
            float(
                self.get_parameter(
                    "speech_timeout_sec"
                ).value
            ),
        )

        self.tts_discovery_timeout_sec = max(
            0.0,
            float(
                self.get_parameter(
                    "tts_discovery_timeout_sec"
                ).value
            ),
        )

        self.navigation_server_timeout_sec = max(
            0.1,
            float(
                self.get_parameter(
                    "navigation_server_timeout_sec"
                ).value
            ),
        )

        self.navigation_cancel_timeout_sec = max(
            0.1,
            float(
                self.get_parameter(
                    "navigation_cancel_timeout_sec"
                ).value
            ),
        )

        self.fail_open = bool(
            self.get_parameter(
                "fail_open"
            ).value
        )

        self.assume_idle_on_start = bool(
            self.get_parameter(
                "assume_idle_on_start"
            ).value
        )

        self.goal_success_position_tolerance_m = float(
            self.get_parameter(
                "goal_success_position_tolerance_m"
            ).value
        )

        self.resume_candidate_wait_timeout_sec = float(
            self.get_parameter(
                "resume_candidate_wait_timeout_sec"
            ).value
        )

        self.execution_authorization_max_age_sec = float(
            self.get_parameter(
                "execution_authorization_max_age_sec"
            ).value
        )

        for name, value in (
            (
                "resume_candidate_wait_timeout_sec",
                self.resume_candidate_wait_timeout_sec,
            ),
            (
                "execution_authorization_max_age_sec",
                self.execution_authorization_max_age_sec,
            ),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"{name} must be finite and strictly positive"
                )

        if (
            not math.isfinite(
                self.goal_success_position_tolerance_m
            )
            or self.goal_success_position_tolerance_m <= 0.0
        ):
            raise ValueError(
                "goal_success_position_tolerance_m "
                "must be finite and strictly positive."
            )

        # ================================================================
        # VALIDATION
        # ================================================================

        if self.language not in {"en", "es", "fr"}:
            self.get_logger().warning(
                "[GOAL GATE] Unsupported language="
                f"'{self.language}'. Using English."
            )
            self.language = "en"

        if not self.public_action_name:
            raise RuntimeError(
                "public_action_name cannot be empty."
            )

        if not self.internal_action_name:
            raise RuntimeError(
                "internal_action_name cannot be empty."
            )

        if (
            self.public_action_name
            == self.internal_action_name
        ):
            raise RuntimeError(
                "public_action_name and internal_action_name "
                "must be different."
            )

        if not self.collaborative_decision_topic:
            raise RuntimeError(
                "collaborative_decision_topic cannot be empty."
            )

        if not self.navigation_state_topic:
            raise RuntimeError(
                "navigation_state_topic cannot be empty."
            )

        if not self.navigation_goal_context_topic:
            raise RuntimeError(
                "navigation_goal_context_topic cannot be empty."
            )

        if not self.candidate_trajectory_topic:
            raise RuntimeError(
                "candidate_trajectory_topic cannot be empty."
            )

        if not self.follow_path_action_name:
            raise RuntimeError(
                "follow_path_action_name cannot be empty."
            )

        # ================================================================
        # RUNTIME STATE
        # ================================================================

        self.speech_condition = threading.Condition()

        self.speech_states: Dict[str, str] = {}

        self.internal_goal_handles: Dict[str, object] = {}
        self.internal_goal_lock = threading.Lock()

        # Exact candidate paths generated for social recovery.
        self.trajectory_lock = threading.Lock()
        self.candidate_trajectories: Dict[str, CandidateTrajectory] = {}

        # Path authorized and ready for the NEXT internal attempt.
        self.authorized_resume_path_ids: Dict[str, str] = {}

        # RESUME(path_id) received while the CURRENT internal
        # attempt is still being canceled.
        #
        # It must not become executable until that previous
        # internal attempt is confirmed terminal with CANCELED.
        self.pending_resume_path_ids: Dict[str, str] = {}

        # Public goals whose current internal attempt has an
        # outstanding cancellation request.
        self.internal_cancel_pending = set()

        # A previously declared missing candidate.
        # goal_id -> (path_id, expiration_monotonic)
        self.resume_waiting_for_candidate = {}

        # Authorization expiry, including deferred authorizations.
        # goal_id -> expiration_monotonic
        self.execution_authorization_deadlines = {}

        # Initialized before subscriptions and action callbacks.
        self.internal_send_pending = set()
        
        # goal_id -> request_id that authorized the pending send.
        self.internal_send_command_ids = {}


        self.goals_received = 0
        self.goals_forwarded = 0
        self.goals_succeeded = 0
        self.goals_aborted = 0
        self.goals_canceled = 0

        self.speech_completed = 0
        self.speech_failed = 0
        self.speech_timed_out = 0
        self.speech_unavailable = 0

        # ================================================================
        # PUBLIC NAVIGATION STATE
        # ================================================================

        self.navigation_state_lock = threading.Lock()

        # Public goal UUID -> action goal handle
        self.public_navigation_goals = {}
        self.public_goal_reserved = False

        # Public goal UUID -> current gate phase
        self.public_navigation_phases = {}

        # Public goal UUID -> external control state
        self.navigation_control_states = {}

        # Public goal UUID -> latest control request ID
        self.navigation_control_requests = {}
        
        # Most recent WAIT/REPLAN request awaiting confirmation
        # that no internal Nav2 attempt remains outstanding.
        #
        # goal_id -> (action, request_id)
        self.pending_hold_requests = {}

        # goal_id -> most recently acknowledged hold request_id
        self.confirmed_hold_requests = {}
        # Reject out-of-order control commands for each public goal.
        self.last_control_stamp_ns = {}

        # A goal that encountered a social hold cannot
        # subsequently use initial START authorization.
        self.goals_with_social_hold = set()

        # ================================================================
        # INTERNAL NAV2 OBSERVATION
        # ================================================================

        self.last_navigation_goal_id = ""
        self.last_navigation_state = "UNKNOWN"

        self.observed_internal_statuses = None

        self.idle_start_allowed = (
            self.assume_idle_on_start
        )

        self.internal_server_was_ready = False

        self.authorization_actions = {}
        self.goals_ever_sent = set()

        # ================================================================
        # ROS INTERFACES
        # ================================================================

        self.speech_publisher = self.create_publisher(
            ExplanationText,
            self.speech_topic,
            reliable_qos(depth=10),
        )

        self.speech_status_subscription = (
            self.create_subscription(
                SpeechStatus,
                self.speech_status_topic,
                self.speech_status_callback,
                reliable_qos(depth=20),
                callback_group=self.callback_group,
            )
        )

        self.collaborative_decision_subscription = (
            self.create_subscription(
                CollaborativeDecision,
                self.collaborative_decision_topic,
                self._collaborative_decision_callback,
                reliable_qos(depth=20),
                callback_group=self.callback_group,
            )
        )

        self.internal_action_client = ActionClient(
            self,
            NavigateToPose,
            self.internal_action_name,
            callback_group=self.callback_group,
        )

        self.follow_path_action_client = ActionClient(
            self,
            FollowPath,
            self.follow_path_action_name,
            callback_group=self.callback_group,
        )

        self.navigation_goal_context_publisher = self.create_publisher(
            NavigationGoalContext,
            self.navigation_goal_context_topic,
            reliable_qos(depth=10),
        )

        self.candidate_trajectory_subscription = self.create_subscription(
            CandidateTrajectory,
            self.candidate_trajectory_topic,
            self._candidate_trajectory_callback,
            reliable_qos(depth=20),
            callback_group=self.callback_group,
        )

        state_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.navigation_state_publisher = (
            self.create_publisher(
                NavigationState,
                self.navigation_state_topic,
                state_qos,
            )
        )

        self.internal_status_subscription = (
            self.create_subscription(
                GoalStatusArray,
                (
                    self.internal_action_name.rstrip("/")
                    + "/_action/status"
                ),
                self._internal_status_callback,
                state_qos,
                callback_group=self.callback_group,
            )
        )

        self.gate_command_result_pub = self.create_publisher(
            String,
            "/social_navigation/gate_command_result",
            reliable_qos(depth=20),
        )


        # ================================================================
        # DIRECTIONAL NAVIGATION INITIALIZATION
        # ================================================================

        self._initialize_directional_gate()

        # Mode D Behavior Tree, provided by navigation.launch.py
        self.declare_parameter(
            "mode_d_behavior_tree_xml",
            "",
        )

        # ================================================================
        # SOCIAL VALIDATION
        # ================================================================

        self.social_validation_cache = {}
        self.authorization_sources = {}
        self.validation_max_age_sec = 2.0

        self.validation_subscription = self.create_subscription(
            TrajectorySocialValidation,
            "/social_navigation/trajectory_social_validation",
            self._social_validation_callback,
            10,
            callback_group=self.callback_group,
        )

        # ================================================================
        # TIMERS
        # ================================================================

        self.navigation_state_timer = self.create_timer(
            0.20,
            self._publish_navigation_state,
            callback_group=self.callback_group,
        )

        self.authorization_watchdog_timer = self.create_timer(
            0.20,
            self._watch_authorization_deadlines,
            callback_group=self.callback_group,
        )

        # ================================================================
        # PUBLIC ACTION SERVER
        # ================================================================

        self.public_action_server = None

        if self.enabled:
            self.public_action_server = ActionServer(
                self,
                NavigateToPose,
                self.public_action_name,
                execute_callback=self.execute_callback,
                goal_callback=self.goal_callback,
                cancel_callback=self.cancel_callback,
                callback_group=self.callback_group,
            )

        self.diagnostics_timer = self.create_timer(
            5.0,
            self.log_diagnostics,
            callback_group=self.callback_group,
        )


    # ====================================================================
    # NAVIGATION GOAL / CANDIDATE TRAJECTORY CONTEXT
    # ====================================================================
    
    def _mode_c_candidate_valid_locked(self, goal_id, path_id):
        """Verify exact, fresh, unconstrained planning evidence for one public goal."""
        handle = self.public_navigation_goals.get(goal_id)

        if (
            handle is None
            or not handle.is_active
            or handle.is_cancel_requested
            or not path_id
        ):
            return False

        with self.trajectory_lock:
            candidate = self.candidate_trajectories.get(path_id)

        validation = self.social_validation_cache.get(path_id)

        if candidate is None or validation is None:
            return False

        if (
            str(candidate.navigation_goal_id or "").strip() != goal_id
            or str(
                candidate.generation_reason or ""
            ).strip().upper() != "REPLAN"
            or not candidate.path.poses
            or str(
                validation.validation or ""
            ).strip().upper() != "VALID"
            or not (
                -0.1 <= self._source_age(validation.header.stamp)
                < self.validation_max_age_sec
            )
            or not (
                -0.1 <= self._source_age(candidate.header.stamp)
                < self.validation_max_age_sec
            )
        ):
            return False

        destination = handle.request.pose
        final_pose = candidate.path.poses[-1]

        frame = str(destination.header.frame_id or "").strip()

        if (
            not frame
            or candidate.path.header.frame_id != frame
            or final_pose.header.frame_id != frame
        ):
            return False

        dx = (
            final_pose.pose.position.x
            - destination.pose.position.x
        )

        dy = (
            final_pose.pose.position.y
            - destination.pose.position.y
        )

        return bool(
            math.isfinite(dx)
            and math.isfinite(dy)
            and math.hypot(dx, dy)
                <= self.goal_success_position_tolerance_m
        )

    def _source_age(self, stamp):
        return (
            self.get_clock().now().nanoseconds / 1e9
            - (
                stamp.sec
                + stamp.nanosec / 1e9
            )
        )

    def _social_validation_callback(self, msg):
        path_id = str(
            msg.path_id or ""
        ).strip()

        if not path_id:
            return

        with self.navigation_state_lock:
            previous = self.social_validation_cache.get(
                path_id
            )

            if previous is not None:
                old_ns = (
                    previous.header.stamp.sec * 1000000000
                    + previous.header.stamp.nanosec
                )

                new_ns = (
                    msg.header.stamp.sec * 1000000000
                    + msg.header.stamp.nanosec
                )

                if new_ns < old_ns:
                    return

                if (
                    new_ns == old_ns
                    and str(previous.validation).upper()
                    != "VALID"
                ):
                    return

            self.social_validation_cache[path_id] = msg

            while len(self.social_validation_cache) > 128:
                self.social_validation_cache.pop(
                    next(iter(self.social_validation_cache))
                )

    def _publish_navigation_goal_context(
        self,
        goal_id,
        goal_pose,
    ) -> None:
        """
        Publish the authoritative destination of the active public goal.

        The goal geometry and frame are preserved, but the context and embedded
        PoseStamped share one fresh timestamp.

        This message describes the current authoritative navigation destination;
        it must not propagate the historical RViz/client timestamp into the
        social trajectory pipeline.
        """

        context_stamp = (
            self.get_clock()
            .now()
            .to_msg()
        )

        context_goal = copy.deepcopy(
            goal_pose
        )

        context_goal.header.stamp = (
            context_stamp
        )

        msg = NavigationGoalContext()

        msg.header.stamp = (
            context_stamp
        )

        msg.header.frame_id = (
            context_goal.header.frame_id
        )

        msg.navigation_goal_id = (
            goal_id
        )

        msg.goal = (
            context_goal
        )

        self.navigation_goal_context_publisher.publish(
            msg
        )

        self.get_logger().info(
            "[GOAL GATE] "
            f"NAVIGATION_GOAL_CONTEXT={goal_id} | "
            f"FRAME={context_goal.header.frame_id!r} | "
            "STAMP="
            f"{context_stamp.sec}."
            f"{context_stamp.nanosec:09d}"
        )

    def _promote_resume_locked(self, goal_id, path_id):
        """
        Promote a RESUME authorization for an exact candidate.

        Caller must hold navigation_state_lock.

        Return:
            MISSING
            PENDING_INTERNAL
            READY
            REJECTED
        """

        public_handle = self.public_navigation_goals.get(goal_id)

        if (
            public_handle is None
            or not public_handle.is_active
            or public_handle.is_cancel_requested
            or self.navigation_control_states.get(goal_id)
            == self.CONTROL_STOPPING
        ):
            self._clear_resume_authorization_locked(goal_id)
            return "REJECTED"

        if not path_id:
            return "REJECTED"

        with self.trajectory_lock:
            candidate = self.candidate_trajectories.get(path_id)

        if candidate is None:
            return "MISSING"

        if (
            str(candidate.navigation_goal_id or "").strip()
            != goal_id
            or not candidate.path.poses
        ):
            self._clear_resume_authorization_locked(goal_id)
            return "REJECTED"

        with self.internal_goal_lock:
            has_internal_handle = (
                goal_id in self.internal_goal_handles
            )

        internal_busy = (
            has_internal_handle
            or goal_id in self.internal_send_pending
            or goal_id in self.internal_cancel_pending
        )

        with self.trajectory_lock:
            if internal_busy:
                self.pending_resume_path_ids[goal_id] = path_id
                self.authorized_resume_path_ids.pop(goal_id, None)
            else:
                self.authorized_resume_path_ids[goal_id] = path_id
                self.pending_resume_path_ids.pop(goal_id, None)

        self.resume_waiting_for_candidate.pop(goal_id, None)

        if internal_busy:
            self.navigation_control_states[goal_id] = (
                self.CONTROL_SOCIAL_HOLD
            )
            self.public_navigation_phases[goal_id] = (
                self.PHASE_SOCIAL_HOLD
            )
            return "PENDING_INTERNAL"

        self.navigation_control_states[goal_id] = (
            self.CONTROL_RUNNING
        )
        self.public_navigation_phases[goal_id] = (
            self.PHASE_RESUMING
        )

        return "READY"

    @staticmethod
    def _directional_revoke_complete(status):
        if status is None or status.state != "REVOKED":
            return False

        if status.reason == "NO_DIRECTIONAL_MASK_WAS_PUBLISHED":
            return True

        return bool(
            status.global_costmap_confirmed
            and status.local_costmap_confirmed
        )
    
    # Initialize directional-mask subscriptions and state.
    def _initialize_directional_gate(self):
        """Initialize exact directional-mask evidence and subscriptions."""

        self.declare_parameter(
            "negotiated_adaptation_enabled",
            False,
        )
        self.negotiated_adaptation_enabled = bool(
            self.get_parameter(
                "negotiated_adaptation_enabled"
            ).value
        )
        self.directional_lock = threading.Lock()
        self.directional_requests = {}
        self.directional_statuses = {}
        self.directional_received_at = {}
        self.directional_active_goals = {}
        self.directional_pending_installs = {}

        status_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.directional_request_subscription = (
            self.create_subscription(
                DirectionalConstraintRequest,
                "/social_navigation/directional_constraint_request",
                self._directional_request_callback,
                reliable_qos(depth=20),
                callback_group=self.callback_group,
            )
        )

        self.directional_status_subscription = (
            self.create_subscription(
                DirectionalConstraintStatus,
                "/social_navigation/directional_constraint_status",
                self._directional_status_callback,
                status_qos,
                callback_group=self.callback_group,
            )
        )

    @staticmethod
    def _directional_identity(msg):
        """Identify one exact directional permission transaction."""
        return (
            str(msg.request_id).strip(),
            str(msg.navigation_goal_id).strip(),
            str(msg.interaction_id).strip(),
            str(msg.person_id).strip(),
            str(msg.tracking_session_id).strip(),
            int(msg.identity_generation),
            str(msg.directional_constraint).strip().upper(),
        )

    def _directional_request_callback(self, msg):
        """
        Handle INSTALL, renewal and REVOKE without losing
        verified costmap evidence.

        A new INSTALL may be deferred until exact revocation.
        """
        goal_id = str(
            msg.navigation_goal_id or ""
        ).strip()

        action = str(
            msg.action or ""
        ).strip().upper()

        if (
            not goal_id
            or not str(msg.request_id).strip()
            or action not in {"INSTALL", "REVOKE"}
        ):
            return

        with self.directional_lock:
            previous = self.directional_requests.get(goal_id)
            old_status = self.directional_statuses.get(goal_id)

            if action == "REVOKE":
                if (
                    previous is None
                    or self._directional_identity(previous)
                    != self._directional_identity(msg)
                ):
                    return

                self.directional_requests[goal_id] = (
                    copy.deepcopy(msg)
                )

                self.directional_received_at[goal_id] = (
                    time.monotonic()
                )

                # It is possible to receive the final status
                # before this REVOKE callback, because ROS
                # topics have independent delivery ordering.
                if self._directional_revoke_complete(
                    old_status
                ):
                    pending = (
                        self.directional_pending_installs.pop(
                            goal_id, None
                        )
                    )

                    if (
                        pending is not None
                        and self._directional_identity(pending)
                            != self._directional_identity(msg)
                    ):
                        self.directional_requests[goal_id] = (
                            pending
                        )
                        self.directional_statuses.pop(
                            goal_id, None
                        )
                        self.directional_received_at[goal_id] = (
                            time.monotonic()
                        )

                return

            if previous is not None:
                same = (
                    self._directional_identity(previous)
                    == self._directional_identity(msg)
                )

                previous_action = str(
                    previous.action
                ).strip().upper()

                if previous_action == "INSTALL":
                    if same:
                        # RENEWAL: keep the verified status.
                        self.directional_requests[goal_id] = (
                            copy.deepcopy(msg)
                        )
                        self.directional_received_at[goal_id] = (
                            time.monotonic()
                        )
                        return

                    # Replacement requires an exact REVOKE.
                    self.directional_pending_installs[goal_id] = (
                        copy.deepcopy(msg)
                    )
                    return

                if previous_action == "REVOKE":
                    if same:
                        # Never revive a request under revocation
                        # using a late renewal message.
                        return

                    if not self._directional_revoke_complete(
                        old_status
                    ):
                        self.directional_pending_installs[goal_id] = (
                            copy.deepcopy(msg)
                        )
                        return

            # First installation, or confirmed replacement.
            self.directional_pending_installs.pop(goal_id, None)

            self.directional_requests[goal_id] = (
                copy.deepcopy(msg)
            )

            self.directional_statuses.pop(goal_id, None)

            self.directional_received_at[goal_id] = (
                time.monotonic()
            )

    def _directional_status_callback(self, msg):
        """Accept only current producer status for the exact request."""
        goal_id = str(msg.navigation_goal_id).strip()

        if not goal_id:
            return

        with self.directional_lock:
            request = self.directional_requests.get(goal_id)

            if request is None:
                return

            if (
                self._directional_identity(request)
                != self._directional_identity(msg)
            ):
                return

            previous = self.directional_statuses.get(goal_id)

            new_stamp = (
                int(msg.header.stamp.sec) * 1_000_000_000
                + int(msg.header.stamp.nanosec)
            )

            if previous is not None:
                old_stamp = (
                    int(previous.header.stamp.sec) * 1_000_000_000
                    + int(previous.header.stamp.nanosec)
                )

                if new_stamp < old_stamp:
                    return

            self.directional_statuses[goal_id] = copy.deepcopy(msg)

            if (
                str(request.action).upper() == "REVOKE"
                and self._directional_revoke_complete(msg)
            ):
                pending = self.directional_pending_installs.pop(
                    goal_id, None
                )

                if pending is not None:
                    self.directional_requests[goal_id] = pending
                    self.directional_statuses.pop(goal_id, None)
                    self.directional_received_at[goal_id] = (
                        time.monotonic()
                    )

    def _retain_failed_directional_attempt(
        self,
        goal_id,
        request_id,
        status,
    ):
        """
        Handle confirmed internal Nav2 ABORTED.

        D:
            No automatic D retry.
            Retain G and negotiate Mode C.
            Do not consume the Mode-C retry budget.

        C / non-directional:
            Preserve one same-goal retry.
            End the public action if that retry fails.

        Navigation Interaction owns terminal speech.
        """
        if status != GoalStatus.STATUS_ABORTED:
            return False

        failure_reason = ""
        is_directional = False

        with self.navigation_state_lock:
            handle = self.public_navigation_goals.get(
                goal_id
            )

            if (
                handle is None
                or not handle.is_active
                or handle.is_cancel_requested
                or self.navigation_control_states.get(
                    goal_id
                ) != self.CONTROL_RUNNING
                or self.navigation_control_requests.get(
                    goal_id
                ) != request_id
            ):
                return False

            with self.internal_goal_lock:
                has_internal_handle = (
                    goal_id in self.internal_goal_handles
                )

            internal_idle = (
                not has_internal_handle
                and goal_id not in self.internal_send_pending
                and goal_id not in self.internal_cancel_pending
            )

            if not internal_idle:
                return False

            with self.directional_lock:
                is_directional = (
                    str(
                        self.directional_active_goals.get(
                            goal_id,
                            "",
                        )
                    ).strip().upper() in {"LEFT", "RIGHT"}
                )

            if is_directional:
                failure_reason = (
                    "NAV2_D_NO_RECOVERY_ABORT"
                )

            else:
                used_retries = getattr(
                    self,
                    "_nav2_goal_retries_used",
                    None,
                )

                if used_retries is None:
                    self._nav2_goal_retries_used = set()
                    used_retries = (
                        self._nav2_goal_retries_used
                    )

                if goal_id in used_retries:
                    self.get_logger().error(
                        "[NAV2 GOAL RECOVERY] "
                        "Non-directional retry exhausted | "
                        f"goal={goal_id} | "
                        "next=PUBLIC_GOAL_ABORTED"
                    )
                    return False

                used_retries.add(goal_id)

                failure_reason = (
                    "NAV2_ABORT_RETRY_AVAILABLE"
                )

            self._clear_resume_authorization_locked(
                goal_id
            )

            self.authorization_actions.pop(
                goal_id, None
            )

            self.authorization_sources.pop(
                goal_id, None
            )

            self.goals_with_social_hold.add(goal_id)

            self.navigation_control_states[
                goal_id
            ] = self.CONTROL_WAITING

            self.public_navigation_phases[
                goal_id
            ] = self.PHASE_WAITING

        if is_directional:
            self.get_logger().warning(
                "[NAV2 D] No-recovery attempt aborted | "
                f"goal={goal_id} | "
                "public_goal_retained=True | "
                "next=NEGOTIATE_MODE_C | "
                "mode_c_retry_budget_unchanged=True"
            )

        else:
            self.get_logger().warning(
                "[NAV2 C] First non-directional abort | "
                f"goal={goal_id} | "
                "public_goal_retained=True | "
                "one_fresh_START_retry_available=True"
            )

        self._publish_gate_command_result(
            goal_id=goal_id,
            action="START",
            request_id=request_id,
            status="EXECUTION_FAILED",
            reason=failure_reason,
        )

        self._publish_navigation_state()
        return True
    
    def _directional_ready_for_start(
        self,
        goal_id,
        direction,
        person_id="",
    ):
        """Require exact authorization and real costmap confirmations.

        PUBLISHED alone is insufficient. This method never checks
        route geometry or path feasibility.
        """
        with self.directional_lock:
            request = self.directional_requests.get(goal_id)
            status = self.directional_statuses.get(goal_id)
            received_at = self.directional_received_at.get(goal_id)

            if (
                request is None
                or status is None
                or received_at is None
            ):
                return False

            if (
                self._directional_identity(request)
                != self._directional_identity(status)
            ):
                return False

            if (
                str(request.action).strip().upper() != "INSTALL"
                or str(
                    request.directional_constraint
                ).strip().upper() != direction
                or (
                    person_id
                    and str(request.person_id).strip() != person_id
                )
                or str(status.state).strip().upper() not in (
                    {"PUBLISHED", "ACTIVE"}
                    if self.negotiated_adaptation_enabled
                    else {"PUBLISHED"}
                )
                or not str(status.mask_id).strip()
            ):
                return False

            lease = (
                int(request.validity_duration.sec)
                + int(request.validity_duration.nanosec) / 1e9
            )

            if (
                not math.isfinite(lease)
                or lease <= 0.0
                or time.monotonic() - received_at >= lease
            ):
                return False

            status_age = self._source_age(
                status.header.stamp
            )

            if not (-0.1 <= status_age < 1.0):
                return False

            # These must be populated from real Nav2 evidence.
            # The mask producer currently keeps both False.
            return bool(
                status.global_costmap_confirmed
                and status.local_costmap_confirmed
            )

    def _directional_execution_ready(self, goal_id):
        """Revalidate a directional authorization during execution."""
        with self.directional_lock:
            direction = self.directional_active_goals.get(
                goal_id
            )

        if direction is None:
            return True

        return self._directional_ready_for_start(
            goal_id,
            direction,
        )

    def _decision_expiration(self, msg):
        """Validate START and reject C until old D is removed."""
        action = str(msg.action or "").strip().upper()
        goal_id = str(msg.navigation_goal_id or "").strip()
        path_id = str(msg.path_id or "").strip()

        if (
            action != "START"
            or not bool(msg.navigation_authorized)
            or not goal_id
            or path_id
        ):
            return None

        age = self._source_age(msg.header.stamp)

        if not (
            -0.1 <= age
            < self.execution_authorization_max_age_sec
        ):
            return None

        reason = str(msg.reason or "").strip().upper()
        direction = str(
            msg.directional_constraint or ""
        ).strip().upper()
        intent = str(
            msg.human_intent or ""
        ).strip().upper()
        outcome = str(
            msg.outcome or ""
        ).strip().upper()

        if direction not in {"NONE", "LEFT", "RIGHT"}:
            return None

        mode_c = reason == "AUTONOMOUS_MODE_C_NAV2_HANDOFF"
        mode_d = reason == "CONFIRMED_PASSAGE_NAV2_HANDOFF"

        if mode_c:
            if (
                outcome != "NAVIGATION_TRANSACTION"
                or bool(msg.requires_replan)
                or direction != "NONE"
                or intent != "UNKNOWN"
            ):
                return None

            with self.directional_lock:
                previous = self.directional_requests.get(goal_id)
                status = self.directional_statuses.get(goal_id)

                if previous is not None:
                    if not (
                        str(previous.action).upper() == "REVOKE"
                        and status is not None
                        and self._directional_identity(previous)
                            == self._directional_identity(status)
                        and self._directional_revoke_complete(status)
                    ):
                        return None

        elif mode_d:
            if (
                outcome != "NAVIGATION_TRANSACTION"
                or bool(msg.requires_replan)
                or intent not in {"UNKNOWN", "ALLOW_PASSAGE"}
            ):
                return None

            if direction in {"LEFT", "RIGHT"}:
                if intent != "ALLOW_PASSAGE":
                    return None

                if not self._directional_ready_for_start(
                    goal_id,
                    direction,
                    str(msg.person_id or "").strip(),
                ):
                    return None

        else:
            if (
                direction != "NONE"
                or goal_id in self.goals_with_social_hold
                or goal_id in self.goals_ever_sent
            ):
                return None

        remaining = (
            self.execution_authorization_max_age_sec
            - max(age, 0.0)
        )

        if remaining <= 0.0:
            return None

        return time.monotonic() + remaining
    
    def _resume_authorization_fresh_locked(self, goal_id):
        """Validate an unconsumed, pathless START authorization."""
        handle = self.public_navigation_goals.get(goal_id)

        if (
            handle is None
            or not handle.is_active
            or handle.is_cancel_requested
            or self.navigation_control_states.get(goal_id)
                == self.CONTROL_STOPPING
        ):
            return False

        deadline = self.execution_authorization_deadlines.get(
            goal_id
        )

        source = self.authorization_sources.get(goal_id)

        if (
            deadline is None
            or source is None
            or not math.isfinite(deadline)
            or time.monotonic() >= deadline
        ):
            return False

        stamp, path_id = source

        if (
            path_id
            or self.authorization_actions.get(goal_id) != "START"
            or not self.navigation_control_requests.get(goal_id)
        ):
            return False

        if not (
            -0.1 <= self._source_age(stamp)
            < self.execution_authorization_max_age_sec
        ):
            return False

        if goal_id in self.goals_with_social_hold:
            return bool(
                self.confirmed_hold_requests.get(goal_id)
            )

        return goal_id not in self.goals_ever_sent

    def _collaborative_decision_callback(self, msg):
        """Apply exact START/WAIT/REPLAN/STOP transactions."""
        goal_id = str(msg.navigation_goal_id or "").strip()
        request_id = str(msg.request_id or "").strip()
        action = str(msg.action or "").strip().upper()
        path_id = str(msg.path_id or "").strip()
        reason_code = str(msg.reason or "").strip().upper()

        if not goal_id or not request_id:
            return

        if action not in {"START", "WAIT", "REPLAN", "STOP"}:
            self._publish_gate_command_result(
                goal_id=goal_id,
                action=action,
                request_id=request_id,
                status="REJECTED",
                reason=(
                    "FOLLOW_PATH_DISABLED_FOR_AUTONOMOUS_NAV2"
                    if action == "RESUME"
                    else "UNSUPPORTED_ACTION"
                ),
                path_id=path_id,
            )
            return

        stamp = msg.header.stamp
        stamp_ns = (
            int(stamp.sec) * 1_000_000_000
            + int(stamp.nanosec)
        )

        valid_stamp = (
            int(stamp.sec) >= 0
            and 0 <= int(stamp.nanosec) < 1_000_000_000
            and stamp_ns > 0
        )

        result = "REJECTED"
        result_reason = ""
        publish_state = False

        with self.navigation_state_lock:
            handle = self.public_navigation_goals.get(goal_id)

            if handle is None or not handle.is_active:
                result_reason = "PUBLIC_GOAL_NOT_ACTIVE"

            elif handle.is_cancel_requested and action != "STOP":
                result_reason = "PUBLIC_GOAL_CANCEL_REQUESTED"

            elif (
                self.navigation_control_requests.get(goal_id)
                == request_id
            ):
                result = "DUPLICATE"
                result_reason = "REQUEST_ALREADY_PROCESSED"

            elif not valid_stamp:
                result_reason = "INVALID_CONTROL_TIMESTAMP"

            elif (
                action != "STOP"
                and stamp_ns <= self.last_control_stamp_ns.get(
                    goal_id, 0
                )
            ):
                result_reason = "OUT_OF_ORDER_CONTROL_COMMAND"

            elif path_id:
                result_reason = "UNEXPECTED_PATH_ID"

            else:
                current_state = self.navigation_control_states.get(
                    goal_id,
                    self.CONTROL_WAITING,
                )

                if (
                    current_state == self.CONTROL_STOPPING
                    and action != "STOP"
                ):
                    result_reason = "PUBLIC_GOAL_STOPPING"

                elif action in {"WAIT", "REPLAN", "STOP"}:
                    self._clear_resume_authorization_locked(goal_id)
                    self.authorization_actions.pop(goal_id, None)
                    self.authorization_sources.pop(goal_id, None)

                    self.goals_with_social_hold.add(goal_id)
                    self.navigation_control_requests[goal_id] = (
                        request_id
                    )

                    if action == "WAIT":
                        self.navigation_control_states[goal_id] = (
                            self.CONTROL_WAITING
                        )

                        with self.internal_goal_lock:
                            has_handle = (
                                goal_id in self.internal_goal_handles
                            )

                        internal_busy = (
                            has_handle
                            or goal_id in self.internal_send_pending
                            or goal_id in self.internal_cancel_pending
                        )

                        self.public_navigation_phases[goal_id] = (
                            self.PHASE_SOCIAL_HOLD
                            if internal_busy
                            else self.PHASE_WAITING
                        )

                        self.pending_hold_requests[goal_id] = (
                            "WAIT",
                            request_id,
                        )

                    elif action == "REPLAN":
                        # REPLAN does not authorize FollowPath.
                        self.navigation_control_states[goal_id] = (
                            self.CONTROL_SOCIAL_HOLD
                        )

                        self.public_navigation_phases[goal_id] = (
                            self.PHASE_SOCIAL_HOLD
                        )

                        self.pending_hold_requests[goal_id] = (
                            "REPLAN",
                            request_id,
                        )

                    else:
                        self.navigation_control_states[goal_id] = (
                            self.CONTROL_STOPPING
                        )

                        self.public_navigation_phases[goal_id] = (
                            self.PHASE_STOPPING
                        )

                        self.pending_hold_requests.pop(
                            goal_id, None
                        )

                    result = "ACCEPTED"
                    result_reason = "CONTROL_APPLIED"
                    publish_state = True

                elif action == "START":
                    expiration = self._decision_expiration(msg)

                    with self.internal_goal_lock:
                        has_handle = (
                            goal_id in self.internal_goal_handles
                        )

                    with self.trajectory_lock:
                        has_legacy_path = (
                            goal_id in self.authorized_resume_path_ids
                            or goal_id in self.pending_resume_path_ids
                        )

                    internal_idle = (
                        not has_handle
                        and not has_legacy_path
                        and goal_id not in self.internal_send_pending
                        and goal_id not in self.internal_cancel_pending
                    )

                    phase = self.public_navigation_phases.get(
                        goal_id
                    )

                    direct = reason_code in {
                        "CONFIRMED_PASSAGE_NAV2_HANDOFF",
                        "AUTONOMOUS_MODE_C_NAV2_HANDOFF",
                    }

                    pending = self.pending_hold_requests.get(
                        goal_id
                    )

                    hold_id = self.confirmed_hold_requests.get(
                        goal_id
                    )

                    confirmed_wait = bool(
                        pending is not None
                        and pending[0] == "WAIT"
                        and pending[1] == hold_id
                        and hold_id
                    )

                    initial_start = (
                        not direct
                        and current_state == self.CONTROL_WAITING
                        and goal_id not in self.goals_with_social_hold
                        and goal_id not in self.goals_ever_sent
                        and phase in {
                            self.PHASE_WAITING_FOR_SPEECH,
                            self.PHASE_WAITING,
                        }
                    )

                    direct_start = (
                        direct
                        and current_state == self.CONTROL_WAITING
                        and confirmed_wait
                        and phase == self.PHASE_WAITING
                    )

                    allowed = (
                        expiration is not None
                        and internal_idle
                        and (initial_start or direct_start)
                    )

                    if not allowed:
                        result_reason = "START_PRECONDITIONS_NOT_MET"

                    else:
                        self.execution_authorization_deadlines[
                            goal_id
                        ] = expiration

                        self.authorization_sources[goal_id] = (
                            msg.header.stamp,
                            "",
                        )

                        self.authorization_actions[goal_id] = "START"

                        direction = str(
                            msg.directional_constraint or "NONE"
                        ).strip().upper()

                        with self.directional_lock:
                            if direction in {"LEFT", "RIGHT"}:
                                self.directional_active_goals[
                                    goal_id
                                ] = direction
                            else:
                                self.directional_active_goals.pop(
                                    goal_id, None
                                )

                        self.navigation_control_states[goal_id] = (
                            self.CONTROL_RUNNING
                        )

                        self.navigation_control_requests[goal_id] = (
                            request_id
                        )

                        self.pending_hold_requests.pop(
                            goal_id, None
                        )

                        result = "ACCEPTED"
                        result_reason = "START_AUTHORIZED"
                        publish_state = True

                if result != "REJECTED":
                    self.last_control_stamp_ns[goal_id] = max(
                        stamp_ns,
                        self.last_control_stamp_ns.get(goal_id, 0),
                    )

        self._publish_gate_command_result(
            goal_id=goal_id,
            action=action,
            request_id=request_id,
            status=result,
            reason=result_reason,
            path_id=path_id,
        )

        if publish_state:
            self._publish_navigation_state()

    def _send_internal_goal_if_running(
        self,
        public_goal_handle,
        public_goal_id,
        client,
        goal,
        internal_uuid,
        resume_path_id="",
        feedback_callback=None,
    ):
        """
        Submit an exact START under Gate ownership.

        Mode D:
            Select the configured no-motor-recovery BT.
            Never silently fall back to the default Nav2 BT.

        Mode C:
            Preserve the original NavigateToPose BT.

        Missing D BT configuration:
            Keep public G in WAIT and report failure.
        """
        if (
            resume_path_id
            or client is not self.internal_action_client
        ):
            return None

        configuration_error = ""
        command_id = ""
        mode_d = False
        bt_xml = ""

        with self.navigation_state_lock:
            if (
                not public_goal_handle.is_active
                or public_goal_handle.is_cancel_requested
                or self.navigation_control_states.get(
                    public_goal_id
                ) != self.CONTROL_RUNNING
            ):
                return None

            if not self._resume_authorization_fresh_locked(
                public_goal_id
            ):
                self._expire_resume_authorization_locked(
                    public_goal_id
                )
                return None

            if self.authorization_actions.get(
                public_goal_id
            ) != "START":
                return None

            if not self._directional_execution_ready(
                public_goal_id
            ):
                self._expire_resume_authorization_locked(
                    public_goal_id
                )
                return None

            command_id = self.navigation_control_requests.get(
                public_goal_id,
                "",
            )

            if not command_id:
                return None

            if (
                public_goal_id in self.internal_send_pending
                or public_goal_id in self.internal_cancel_pending
            ):
                return None

            with self.internal_goal_lock:
                if public_goal_id in self.internal_goal_handles:
                    return None

            with self.trajectory_lock:
                if (
                    self.authorized_resume_path_ids.get(
                        public_goal_id, ""
                    )
                    or self.pending_resume_path_ids.get(
                        public_goal_id, ""
                    )
                ):
                    return None

            with self.directional_lock:
                mode_d = (
                    str(
                        self.directional_active_goals.get(
                            public_goal_id,
                            "",
                        )
                    ).strip().upper() in {"LEFT", "RIGHT"}
                )

            if mode_d:
                bt_xml = str(
                    self.get_parameter(
                        "mode_d_behavior_tree_xml"
                    ).value or ""
                ).strip()

                if (
                    not bt_xml
                    or not bt_xml.startswith("/")
                ):
                    configuration_error = (
                        "D_BT_XML_NOT_CONFIGURED"
                    )

                elif not hasattr(goal, "behavior_tree"):
                    configuration_error = (
                        "D_BT_FIELD_UNAVAILABLE"
                    )

            if configuration_error:
                # No internal Nav2 action was submitted.
                self._clear_resume_authorization_locked(
                    public_goal_id
                )

                self.authorization_actions.pop(
                    public_goal_id, None
                )

                self.authorization_sources.pop(
                    public_goal_id, None
                )

                self.goals_with_social_hold.add(
                    public_goal_id
                )

                self.navigation_control_states[
                    public_goal_id
                ] = self.CONTROL_WAITING

                self.public_navigation_phases[
                    public_goal_id
                ] = self.PHASE_WAITING

            else:
                if mode_d:
                    # goal is a copy of the public request.
                    goal.behavior_tree = bt_xml

                self.goals_ever_sent.add(public_goal_id)
                self.internal_send_pending.add(public_goal_id)

                self.internal_send_command_ids[
                    public_goal_id
                ] = command_id

                self.public_navigation_phases[
                    public_goal_id
                ] = self.PHASE_FORWARDING

                kwargs = {"goal_uuid": internal_uuid}

                if feedback_callback is not None:
                    kwargs["feedback_callback"] = (
                        feedback_callback
                    )

                if mode_d:
                    self.get_logger().info(
                        "[GOAL GATE] Sending no-recovery D | "
                        f"goal={public_goal_id} | "
                        f"bt={bt_xml} | "
                        "recoveries=NONE"
                    )

                return client.send_goal_async(
                    goal,
                    **kwargs,
                )

        self.get_logger().error(
            "[GOAL GATE] Mode D tree unavailable | "
            f"goal={public_goal_id} | "
            f"reason={configuration_error} | "
            "internal_goal_sent=False | "
            "public_G_retained=True"
        )

        self._publish_gate_command_result(
            goal_id=public_goal_id,
            action="START",
            request_id=command_id,
            status="EXECUTION_FAILED",
            reason=configuration_error,
        )

        self._publish_navigation_state()
        return None
    
    def _watch_authorization_deadlines(self):
        """Expire unconsumed START and supervise active mask evidence."""
        now = time.monotonic()
        expired_goals = []
        direction_lost = []

        with self.navigation_state_lock:
            for goal_id, deadline in list(
                self.execution_authorization_deadlines.items()
            ):
                if goal_id in self.internal_send_pending:
                    continue

                if (
                    not math.isfinite(deadline)
                    or now >= deadline
                ):
                    expired_goals.append(goal_id)

            for goal_id in expired_goals:
                self._expire_resume_authorization_locked(
                    goal_id
                )

            for goal_id, control in list(
                self.navigation_control_states.items()
            ):
                if control != self.CONTROL_RUNNING:
                    continue

                with self.directional_lock:
                    directional = (
                        goal_id in self.directional_active_goals
                    )

                if not directional:
                    continue

                if not self._directional_execution_ready(
                    goal_id
                ):
                    self.navigation_control_states[goal_id] = (
                        self.CONTROL_SOCIAL_HOLD
                    )

                    self.public_navigation_phases[goal_id] = (
                        self.PHASE_SOCIAL_HOLD
                    )

                    self.goals_with_social_hold.add(goal_id)
                    direction_lost.append(goal_id)

        if expired_goals or direction_lost:
            self._publish_navigation_state()

        for goal_id in direction_lost:
            self.get_logger().error(
                "[DIRECTIONAL GATE] Constraint evidence lost | "
                f"goal={goal_id} | "
                "internal Nav2 cancellation required"
            )

    def _decision_expiration_old(self, msg):
        """Reject directional execution until Nav2 enforces the authorized side."""
        action = str(msg.action or "").strip().upper()
        goal_id = str(msg.navigation_goal_id or "").strip()
        path_id = str(msg.path_id or "").strip()

        if (
            action not in {"START", "RESUME"}
            or not bool(msg.navigation_authorized)
            or not goal_id
        ):
            return None

        age = self._source_age(msg.header.stamp)

        if not (
            -0.1 <= age < self.execution_authorization_max_age_sec
        ):
            return None

        if action == "START":
            if path_id:
                return None

            reason = str(msg.reason or "").strip().upper()
            direction = str(
                msg.directional_constraint or ""
            ).strip().upper()
            intent = str(
                msg.human_intent or ""
            ).strip().upper()
            outcome = str(
                msg.outcome or ""
            ).strip().upper()

            if reason == "CONFIRMED_PASSAGE_NAV2_HANDOFF":
                if (
                    outcome != "NAVIGATION_TRANSACTION"
                    or bool(msg.requires_replan)
                    or direction not in {"NONE", "LEFT", "RIGHT"}
                ):
                    return None

                if direction in {"LEFT", "RIGHT"}:
                    # An HRI direction is not a Nav2 spatial constraint.
                    # Foxy must have a verified enforcement mechanism
                    # before directional execution can be enabled.
                    self.get_logger().error(
                        "[DIRECTIONAL GATE] START rejected | "
                        f"goal={goal_id} | "
                        f"direction={direction} | "
                        "reason=NAV2_DIRECTION_ENFORCEMENT_NOT_INSTALLED"
                    )
                    return None

                if intent not in {"UNKNOWN", "ALLOW_PASSAGE"}:
                    return None

            else:
                # Ordinary START is valid only before the first
                # internal execution and before any social hold.
                if (
                    direction != "NONE"
                    or goal_id in self.goals_with_social_hold
                    or goal_id in self.goals_ever_sent
                ):
                    return None

            remaining = (
                self.execution_authorization_max_age_sec
                - max(age, 0.0)
            )

            return (
                time.monotonic() + remaining
                if remaining > 0.0
                else None
            )

        # Disable the legacy FollowPath execution path.
        # It bypasses autonomous NavigateToPose replanning.
        self.get_logger().warning(
            "[GOAL GATE] Legacy RESUME rejected | "
            f"goal={goal_id} | "
            "reason=FOLLOW_PATH_DISABLED_FOR_AUTONOMOUS_NAV2"
        )

        return None

    def _clear_resume_authorization_locked(self, goal_id):
        """Revoke all unconsumed path and Mode C restart authorizations."""
        self.resume_waiting_for_candidate.pop(goal_id, None)

        restart = getattr(
            self,
            "_mode_c_nav2_restart",
            None,
        )

        if restart is not None and restart[0] == goal_id:
            self._mode_c_nav2_restart = None

        self.execution_authorization_deadlines.pop(
            goal_id,
            None,
        )

        if hasattr(self, "resume_authorization_deadlines"):
            self.resume_authorization_deadlines.pop(
                goal_id,
                None,
            )

        with self.trajectory_lock:
            self.pending_resume_path_ids.pop(
                goal_id,
                None,
            )

            self.authorized_resume_path_ids.pop(
                goal_id,
                None,
            )

    def _resume_authorization_fresh_locked_old(self, goal_id):
        """Check that the accepted execution transaction is still fresh."""
        handle = self.public_navigation_goals.get(goal_id)

        if (
            handle is None
            or not handle.is_active
            or handle.is_cancel_requested
            or self.navigation_control_states.get(goal_id)
                == self.CONTROL_STOPPING
        ):
            return False

        deadline = self.execution_authorization_deadlines.get(
            goal_id
        )

        source = self.authorization_sources.get(
            goal_id
        )

        if (
            deadline is None
            or source is None
            or not math.isfinite(deadline)
            or time.monotonic() >= deadline
        ):
            return False

        stamp, path_id = source

        if not (
            -0.1 <= self._source_age(stamp)
            < self.execution_authorization_max_age_sec
        ):
            return False

        action = self.authorization_actions.get(goal_id)

        if action == "START":
            if path_id:
                return False

            if not self.navigation_control_requests.get(goal_id):
                return False

            if goal_id in self.goals_with_social_hold:
                # The callback checked the exact latest WAIT
                # acknowledgement before accepting this START.
                return bool(
                    self.confirmed_hold_requests.get(goal_id)
                )

            return goal_id not in self.goals_ever_sent

        if action != "RESUME" or not path_id:
            return False

        validation = self.social_validation_cache.get(path_id)

        return bool(
            validation is not None
            and str(
                validation.validation or ""
            ).strip().upper() == "VALID"
            and -0.1 <= self._source_age(
                validation.header.stamp
            ) < self.validation_max_age_sec
        )

    def _expire_resume_authorization_locked(self, goal_id):
        """
        Expire an unconsumed START/RESUME authorization.

        Caller holds navigation_state_lock.
        Expiration never means that an accepted Nav2 action
        has terminated.
        """
        had_authorization = (
            goal_id in self.execution_authorization_deadlines
            or goal_id in self.resume_waiting_for_candidate
        )

        if not had_authorization:
            return False

        action = self.authorization_actions.get(goal_id, "RESUME")
        request_id = self.navigation_control_requests.get(goal_id, "")

        self._clear_resume_authorization_locked(goal_id)
        self.authorization_actions.pop(goal_id, None)

        if (
            self.navigation_control_states.get(goal_id)
            != self.CONTROL_STOPPING
        ):
            self.navigation_control_states[goal_id] = (
                self.CONTROL_WAITING
            )
            self.public_navigation_phases[goal_id] = (
                self.PHASE_WAITING
            )

        self._publish_gate_command_result(
            goal_id=goal_id,
            action=action,
            request_id=request_id,
            status="EXPIRED",
            reason="AUTHORIZATION_NOT_CONSUMED_BEFORE_DEADLINE",
        )

        return True

    def _candidate_trajectory_callback(
        self,
        msg: CandidateTrajectory,
    ) -> None:
        """
        Cache the exact candidate for its public goal.

        A candidate can resolve a pending RESUME, but it
        never creates a navigation authorization itself.
        """

        path_id = str(msg.path_id or "").strip()
        goal_id = str(msg.navigation_goal_id or "").strip()

        if not path_id or not goal_id:
            return

        if not msg.path.poses:
            return

        promotion = "CACHED"
        changed = False

        with self.navigation_state_lock:

            handle = self.public_navigation_goals.get(goal_id)

            if (
                handle is None
                or not handle.is_active
                or handle.is_cancel_requested
            ):
                return

            if self.navigation_control_states.get(
                goal_id
            ) == self.CONTROL_STOPPING:
                return

            with self.trajectory_lock:

                existing = self.candidate_trajectories.get(
                    path_id
                )

                if existing is not None:
                    if existing != msg:
                        self.get_logger().error(
                            "[GOAL GATE] PATH_ID_COLLISION | "
                            f"goal={goal_id} | path={path_id}"
                        )
                        return

                else:
                    self.candidate_trajectories[path_id] = msg

            waiting = self.resume_waiting_for_candidate.get(
                goal_id
            )

            if waiting is not None:

                if isinstance(waiting, tuple):
                    expected_path = str(waiting[0]).strip()
                else:
                    expected_path = str(waiting).strip()

                if expected_path == path_id:

                    candidate_deadline = (
                        waiting[1]
                        if isinstance(waiting, tuple)
                        else None
                    )

                    if (
                        not self._resume_authorization_fresh_locked(goal_id)
                        or (
                            candidate_deadline is not None
                            and time.monotonic() >= candidate_deadline
                        )
                    ):
                        self._expire_resume_authorization_locked(goal_id)
                        promotion = "REVOKED"
                        changed = True

                    else:
                        promotion = self._promote_resume_locked(
                            goal_id,
                            path_id,
                        )
                        changed = True

        self.get_logger().info(
            "[GOAL GATE] CANDIDATE_CACHED | "
            f"goal={goal_id} | "
            f"path={path_id} | "
            f"promotion={promotion}"
        )

        if changed:
            self._publish_navigation_state()
            
    # ====================================================================
    # NAVIGATION CONTROL
    # ====================================================================
    
    def _watch_authorization_deadlines_old(self):
        now = time.monotonic()
        expired_goals = []

        with self.navigation_state_lock:
            for goal_id, deadline in list(
                self.execution_authorization_deadlines.items()
            ):
                # Do not expire authorization while the submitted
                # internal action awaits Nav2 acceptance.
                if goal_id in self.internal_send_pending:
                    continue

                waiting = self.resume_waiting_for_candidate.get(
                    goal_id
                )

                candidate_deadline = (
                    waiting[1]
                    if isinstance(waiting, tuple)
                    else None
                )

                if (
                    not math.isfinite(deadline)
                    or now >= deadline
                    or (
                        candidate_deadline is not None
                        and now >= candidate_deadline
                    )
                ):
                    expired_goals.append(goal_id)

            for goal_id in expired_goals:
                self._expire_resume_authorization_locked(goal_id)

        if expired_goals:
            self._publish_navigation_state()


    def _publish_gate_command_result(
        self,
        *,
        goal_id,
        action,
        request_id,
        status,
        reason="",
        path_id="",
    ):
        payload = {
            "version": 1,
            "producer": "navigation_goal_gate",
            "navigation_goal_id": str(goal_id or ""),
            "action": str(action or "").upper(),
            "request_id": str(request_id or ""),
            "status": str(status or "").upper(),
            "reason": str(reason or ""),
            "path_id": str(path_id or ""),
        }

        msg = String()
        msg.data = json.dumps(
            payload,
            separators=(",", ":"),
        )
        self.gate_command_result_pub.publish(msg)

        self.get_logger().info(
            "[GATE RESULT] "
            f"goal={payload['navigation_goal_id']} | "
            f"action={payload['action']} | "
            f"request={payload['request_id']} | "
            f"status={payload['status']} | "
            f"reason={payload['reason']}"
        )

    def _collaborative_decision_callback_old(self, msg):
        """Process collaborative commands with exact goal and recovery ownership."""
        goal_id = str(msg.navigation_goal_id or "").strip()
        request_id = str(msg.request_id or "").strip()
        action = str(msg.action or "").strip().upper()
        path_id = str(msg.path_id or "").strip()
        command_reason = str(msg.reason or "").strip().upper()

        if not goal_id or not request_id:
            return

        if action not in {
            "START", "WAIT", "REPLAN", "RESUME", "STOP"
        }:
            self._publish_gate_command_result(
                goal_id=goal_id,
                action=action,
                request_id=request_id,
                status="REJECTED",
                reason="UNSUPPORTED_ACTION",
                path_id=path_id,
            )
            return

        stamp = msg.header.stamp
        stamp_ns = (
            int(stamp.sec) * 1_000_000_000
            + int(stamp.nanosec)
        )

        valid_stamp = bool(
            int(stamp.sec) >= 0
            and 0 <= int(stamp.nanosec) < 1_000_000_000
            and stamp_ns > 0
        )

        result = "REJECTED"
        reason = ""
        publish_state = False

        with self.navigation_state_lock:
            handle = self.public_navigation_goals.get(goal_id)

            if handle is None or not handle.is_active:
                reason = "PUBLIC_GOAL_NOT_ACTIVE"

            elif handle.is_cancel_requested and action != "STOP":
                reason = "PUBLIC_GOAL_CANCEL_REQUESTED"

            elif (
                self.navigation_control_requests.get(goal_id)
                == request_id
            ):
                result = "DUPLICATE"
                reason = "REQUEST_ALREADY_PROCESSED"

            elif not valid_stamp:
                reason = "INVALID_CONTROL_TIMESTAMP"

            elif (
                action != "STOP"
                and stamp_ns <= self.last_control_stamp_ns.get(
                    goal_id,
                    0,
                )
            ):
                reason = "OUT_OF_ORDER_CONTROL_COMMAND"

            elif (
                action != "RESUME"
                and path_id
                and not (
                    action == "START"
                    and command_reason
                        == "MODE_C_VALIDATED_NAV2_RESTART"
                )
            ):
                reason = "UNEXPECTED_PATH_ID"

            else:
                current_state = self.navigation_control_states.get(
                    goal_id,
                    self.CONTROL_WAITING,
                )

                if (
                    current_state == self.CONTROL_STOPPING
                    and action != "STOP"
                ):
                    reason = "PUBLIC_GOAL_STOPPING"

                elif action in {"WAIT", "REPLAN", "STOP"}:
                    self._clear_resume_authorization_locked(goal_id)
                    self.authorization_actions.pop(goal_id, None)
                    self.authorization_sources.pop(goal_id, None)

                    # Every explicit hold prevents ordinary START reuse.
                    self.goals_with_social_hold.add(goal_id)

                    self.navigation_control_requests[goal_id] = (
                        request_id
                    )

                    if action == "WAIT":
                        self.navigation_control_states[goal_id] = (
                            self.CONTROL_WAITING
                        )

                        with self.internal_goal_lock:
                            has_handle = (
                                goal_id in self.internal_goal_handles
                            )

                        internal_busy = bool(
                            has_handle
                            or goal_id in self.internal_send_pending
                            or goal_id in self.internal_cancel_pending
                        )

                        self.public_navigation_phases[goal_id] = (
                            self.PHASE_SOCIAL_HOLD
                            if internal_busy
                            else self.PHASE_WAITING
                        )

                        self.pending_hold_requests[goal_id] = (
                            "WAIT",
                            request_id,
                        )

                    elif action == "REPLAN":
                        self.navigation_control_states[goal_id] = (
                            self.CONTROL_SOCIAL_HOLD
                        )

                        self.public_navigation_phases[goal_id] = (
                            self.PHASE_SOCIAL_HOLD
                        )

                        self.pending_hold_requests[goal_id] = (
                            "REPLAN",
                            request_id,
                        )

                    else:
                        self.navigation_control_states[goal_id] = (
                            self.CONTROL_STOPPING
                        )

                        self.public_navigation_phases[goal_id] = (
                            self.PHASE_STOPPING
                        )

                        self.pending_hold_requests.pop(goal_id, None)

                    result = "ACCEPTED"
                    reason = "CONTROL_APPLIED"
                    publish_state = True

                elif action == "START":
                    expiration = self._decision_expiration(msg)

                    with self.internal_goal_lock:
                        has_internal_handle = (
                            goal_id in self.internal_goal_handles
                        )

                    with self.trajectory_lock:
                        has_resume = bool(
                            goal_id in self.authorized_resume_path_ids
                            or goal_id in self.pending_resume_path_ids
                        )

                    phase = self.public_navigation_phases.get(goal_id)

                    mode_c_restart = (
                        command_reason
                        == "MODE_C_VALIDATED_NAV2_RESTART"
                    )

                    direct_handoff = (
                        command_reason
                        == "CONFIRMED_PASSAGE_NAV2_HANDOFF"
                    )

                    pending_hold = self.pending_hold_requests.get(
                        goal_id
                    )

                    confirmed_hold_id = self.confirmed_hold_requests.get(
                        goal_id
                    )

                    confirmed_wait = bool(
                        pending_hold is not None
                        and pending_hold[0] == "WAIT"
                        and pending_hold[1] == confirmed_hold_id
                        and confirmed_hold_id
                    )

                    confirmed_replan = bool(
                        pending_hold is not None
                        and pending_hold[0] == "REPLAN"
                        and pending_hold[1] == confirmed_hold_id
                        and confirmed_hold_id
                    )

                    internal_idle = bool(
                        not has_internal_handle
                        and not has_resume
                        and goal_id not in self.internal_send_pending
                        and goal_id not in self.internal_cancel_pending
                    )

                    common_ready = bool(
                        msg.navigation_authorized
                        and expiration is not None
                        and internal_idle
                    )

                    initial_start = bool(
                        not mode_c_restart
                        and not direct_handoff
                        and not path_id
                        and current_state == self.CONTROL_WAITING
                        and goal_id not in self.goals_with_social_hold
                        and goal_id not in self.goals_ever_sent
                        and phase in {
                            self.PHASE_WAITING_FOR_SPEECH,
                            self.PHASE_WAITING,
                        }
                    )

                    reauthorized_start = bool(
                        not mode_c_restart
                        and direct_handoff
                        and not path_id
                        and current_state == self.CONTROL_WAITING
                        and confirmed_wait
                        and phase == self.PHASE_WAITING
                    )

                    validated_mode_c_restart = bool(
                        mode_c_restart
                        and path_id
                        and confirmed_replan
                        and current_state in {
                            self.CONTROL_SOCIAL_HOLD,
                            self.CONTROL_REPLANNING,
                        }
                        and phase in {
                            self.PHASE_SOCIAL_HOLD,
                            self.PHASE_REPLANNING,
                        }
                        and goal_id in self.goals_with_social_hold
                        and self._mode_c_candidate_valid_locked(
                            goal_id,
                            path_id,
                        )
                    )

                    allowed = bool(
                        common_ready
                        and (
                            initial_start
                            or reauthorized_start
                            or validated_mode_c_restart
                        )
                    )

                    if not allowed:
                        reason = "START_PRECONDITIONS_NOT_MET"

                    else:
                        self.execution_authorization_deadlines[
                            goal_id
                        ] = expiration

                        self.authorization_sources[goal_id] = (
                            msg.header.stamp,
                            path_id if mode_c_restart else "",
                        )

                        self.authorization_actions[goal_id] = "START"

                        if mode_c_restart:
                            self._mode_c_nav2_restart = (
                                goal_id,
                                request_id,
                                path_id,
                            )

                        else:
                            restart = getattr(
                                self,
                                "_mode_c_nav2_restart",
                                None,
                            )

                            if (
                                restart is not None
                                and restart[0] == goal_id
                            ):
                                self._mode_c_nav2_restart = None

                        self.navigation_control_states[goal_id] = (
                            self.CONTROL_RUNNING
                        )

                        self.navigation_control_requests[goal_id] = (
                            request_id
                        )

                        self.pending_hold_requests.pop(goal_id, None)

                        result = "ACCEPTED"
                        reason = "START_AUTHORIZED"
                        publish_state = True

                elif action == "RESUME":
                    expiration = self._decision_expiration(msg)

                    if (
                        not msg.navigation_authorized
                        or not path_id
                        or expiration is None
                    ):
                        reason = "INVALID_OR_STALE_RESUME"

                    else:
                        self._clear_resume_authorization_locked(goal_id)

                        self.execution_authorization_deadlines[
                            goal_id
                        ] = expiration

                        self.authorization_sources[goal_id] = (
                            msg.header.stamp,
                            path_id,
                        )

                        self.authorization_actions[goal_id] = "RESUME"

                        promotion = self._promote_resume_locked(
                            goal_id,
                            path_id,
                        )

                        if promotion == "MISSING":
                            deadline = min(
                                expiration,
                                time.monotonic()
                                + self.resume_candidate_wait_timeout_sec,
                            )

                            self.resume_waiting_for_candidate[goal_id] = (
                                path_id,
                                deadline,
                            )

                            self.navigation_control_states[goal_id] = (
                                self.CONTROL_SOCIAL_HOLD
                            )

                            self.public_navigation_phases[goal_id] = (
                                self.PHASE_SOCIAL_HOLD
                            )

                            result = "ACCEPTED_PENDING_CANDIDATE"
                            reason = "CANDIDATE_NOT_YET_RECEIVED"

                        elif promotion == "PENDING_INTERNAL":
                            result = "ACCEPTED_PENDING_TERMINATION"
                            reason = "OLD_INTERNAL_ATTEMPT_PENDING"

                        elif promotion == "READY":
                            result = "ACCEPTED"
                            reason = "RESUME_AUTHORIZED"

                        else:
                            self._clear_resume_authorization_locked(
                                goal_id
                            )

                            self.authorization_actions.pop(
                                goal_id,
                                None,
                            )

                            self.authorization_sources.pop(
                                goal_id,
                                None,
                            )

                            if (
                                self.navigation_control_states.get(goal_id)
                                != self.CONTROL_STOPPING
                            ):
                                self.navigation_control_states[goal_id] = (
                                    self.CONTROL_WAITING
                                )

                            reason = "CANDIDATE_PROMOTION_REJECTED"

                        if result != "REJECTED":
                            self.navigation_control_requests[goal_id] = (
                                request_id
                            )

                            self.pending_hold_requests.pop(goal_id, None)
                            self.goals_with_social_hold.add(goal_id)
                            publish_state = True

                if result != "REJECTED":
                    self.last_control_stamp_ns[goal_id] = max(
                        stamp_ns,
                        self.last_control_stamp_ns.get(goal_id, 0),
                    )

        self._publish_gate_command_result(
            goal_id=goal_id,
            action=action,
            request_id=request_id,
            status=result,
            reason=reason,
            path_id=path_id,
        )

        if publish_state:
            self._publish_navigation_state()
                
    def _get_navigation_control_state(
        self,
        goal_id: str,
    ) -> str:
        """Return the current external control state."""

        with self.navigation_state_lock:
            return self.navigation_control_states.get(
                goal_id,
                self.CONTROL_STOPPING,
            )

    def _wait_for_control_release(
        self,
        public_goal_handle,
        public_goal_id: str,
    ) -> str:
        """
        Wait for START/RESUME while supervising hold ownership.

        HOLD_CONFIRMED is emitted only when no internal
        goal handle, pending send, or pending cancellation
        remains.

        Each WAIT/REPLAN request may receive its own
        acknowledgement, even if the phase is unchanged.
        """
        while rclpy.ok():
            changed = False
            acknowledgement = None

            with self.navigation_state_lock:

                if public_goal_handle.is_cancel_requested:
                    return "CANCELED"

                if not public_goal_handle.is_active:
                    return "STOPPING"

                control = self.navigation_control_states.get(
                    public_goal_id,
                    self.CONTROL_STOPPING,
                )

                if control == self.CONTROL_STOPPING:
                    return "STOPPING"

                if control == self.CONTROL_RUNNING:
                    return "RUNNING"

                with self.internal_goal_lock:
                    has_internal_handle = (
                        public_goal_id
                        in self.internal_goal_handles
                    )

                internal_busy = (
                    has_internal_handle
                    or public_goal_id in self.internal_send_pending
                    or public_goal_id in self.internal_cancel_pending
                )

                # Never acknowledge a hold while an
                # internal attempt has uncertain ownership.
                if not internal_busy:

                    if control == self.CONTROL_WAITING:
                        phase = self.PHASE_WAITING

                    elif control in {
                        self.CONTROL_SOCIAL_HOLD,
                        self.CONTROL_REPLANNING,
                    }:
                        phase = self.PHASE_REPLANNING

                        self.navigation_control_states[
                            public_goal_id
                        ] = self.CONTROL_REPLANNING

                    else:
                        phase = None

                    if (
                        phase is not None
                        and self.public_navigation_phases.get(
                            public_goal_id
                        ) != phase
                    ):
                        self.public_navigation_phases[
                            public_goal_id
                        ] = phase

                        changed = True

                    pending = self.pending_hold_requests.get(
                        public_goal_id
                    )

                    if pending is not None:
                        action, request_id = pending

                        expected_action = (
                            "WAIT"
                            if phase == self.PHASE_WAITING
                            else (
                                "REPLAN"
                                if phase == self.PHASE_REPLANNING
                                else ""
                            )
                        )

                        if (
                            action == expected_action
                            and request_id
                            and self.confirmed_hold_requests.get(
                                public_goal_id
                            ) != request_id
                        ):
                            self.confirmed_hold_requests[
                                public_goal_id
                            ] = request_id

                            acknowledgement = (
                                action,
                                request_id,
                            )

            if changed:
                self._publish_navigation_state()

            if acknowledgement is not None:
                action, request_id = acknowledgement

                # Do not acknowledge an obsolete command
                # superseded while leaving the state lock.
                with self.navigation_state_lock:
                    still_current = (
                        self.pending_hold_requests.get(
                            public_goal_id
                        ) == acknowledgement
                        and self.navigation_control_states.get(
                            public_goal_id
                        ) != self.CONTROL_STOPPING
                        and self.navigation_control_states.get(
                            public_goal_id
                        ) != self.CONTROL_RUNNING
                    )

                if still_current:
                    self.get_logger().info(
                        "[GOAL GATE] HOLD_CONFIRMED | "
                        f"goal={public_goal_id} | "
                        f"action={action} | "
                        f"request={request_id}"
                    )

                    self._publish_gate_command_result(
                        goal_id=public_goal_id,
                        action=action,
                        request_id=request_id,
                        status="HOLD_CONFIRMED",
                        reason="NO_INTERNAL_ATTEMPT_OUTSTANDING",
                    )

            time.sleep(0.05)

        return "STOPPING"
    
    def _activate_pending_resume(
        self,
        public_goal_id: str,
    ) -> bool:
        """
        Activate a pending resume after confirmed termination
        of the previous internal attempt.
        """

        result = "REJECTED"
        path_id = ""

        with self.navigation_state_lock:

            handle = self.public_navigation_goals.get(
                public_goal_id
            )

            if (
                handle is None
                or not handle.is_active
                or handle.is_cancel_requested
            ):
                return False

            if (
                self.navigation_control_states.get(
                    public_goal_id
                ) == self.CONTROL_STOPPING
            ):
                return False

            with self.internal_goal_lock:
                has_internal_handle = (
                    public_goal_id in self.internal_goal_handles
                )

            if (
                has_internal_handle
                or public_goal_id in self.internal_send_pending
                or public_goal_id in self.internal_cancel_pending
            ):
                return False

            with self.trajectory_lock:
                path_id = self.pending_resume_path_ids.get(
                    public_goal_id,
                    "",
                )

            if not path_id:
                return False

            if not self._resume_authorization_fresh_locked(
                public_goal_id
            ):
                self._expire_resume_authorization_locked(
                    public_goal_id
                )
                return False

            result = self._promote_resume_locked(
                public_goal_id,
                path_id,
            )

        if result != "READY":
            return False

        self.get_logger().info(
            "[GOAL GATE] PENDING_RESUME_ACTIVATED | "
            f"goal={public_goal_id} | path={path_id}"
        )

        self._publish_navigation_state()
        return True

    # ====================================================================
    # PUBLIC ACTION SERVER
    # ====================================================================

    def goal_callback(
        self,
        goal_request: NavigateToPose.Goal,
    ) -> GoalResponse:
        """
        Accept at most one outstanding public navigation goal.

        Reserve the execution slot before execute_callback()
        registers the actual goal handle.
        """
        if not self.enabled:
            self.get_logger().warning(
                "[GOAL GATE] Public goal rejected: gate disabled."
            )
            return GoalResponse.REJECT

        with self.navigation_state_lock:
            occupied = (
                self.public_goal_reserved
                or bool(self.public_navigation_goals)
            )

            if not occupied:
                self.public_goal_reserved = True

        if occupied:
            self.get_logger().warning(
                "[GOAL GATE] Public goal rejected: "
                "another public goal is already owned."
            )
            return GoalResponse.REJECT

        self.goals_received += 1

        self.get_logger().info(
            "[GOAL GATE] Public navigation goal accepted."
        )

        return GoalResponse.ACCEPT
    
    def cancel_callback(
        self,
        goal_handle,
    ) -> CancelResponse:
        """Accept a public cancellation request."""

        goal_id = self._goal_id_hex(goal_handle)

        with self.navigation_state_lock:

            self.navigation_control_states[
                goal_id
            ] = self.CONTROL_STOPPING

            self.navigation_control_requests[
                goal_id
            ] = "USER_CANCEL"

            self.public_navigation_phases[
                goal_id
            ] = self.PHASE_STOPPING

        with self.speech_condition:
            self.speech_condition.notify_all()

        self.get_logger().warning(
            "[GOAL GATE] "
            f"USER_CANCEL_REQUESTED={goal_id}"
        )

        self._publish_navigation_state()

        return CancelResponse.ACCEPT

    # ====================================================================
    # PUBLIC GOAL EXECUTION
    # ====================================================================

    def _execute_goal(
        self,
        goal_handle,
    ) -> NavigateToPose.Result:
        """
        Announce the public goal once, then wait for an
        explicit collaborative execution authorization.

        Initial WAITING is a normal authorization barrier,
        not evidence of a social intrusion.

        GOAL_READY is published exactly once.
        """
        goal_id = self._goal_id_hex(goal_handle)
        decision_id = f"navigation_goal_{goal_id}"

        self.get_logger().info(
            "[GOAL GATE] "
            f"GOAL_RECEIVED={goal_id}"
        )

        # ====================================================
        # 1. ANNOUNCE GOAL RECEPTION
        # ====================================================

        speech_result = self._announce_goal(
            goal_handle=goal_handle,
            decision_id=decision_id,
        )

        if (
            goal_handle.is_cancel_requested
            or speech_result == "CANCELED"
        ):
            return self._finish_navigation(
                goal_handle,
                goal_id,
                GoalStatus.STATUS_CANCELED,
            )

        if (
            speech_result != "COMPLETED"
            and not self.fail_open
        ):
            return self._finish_navigation(
                goal_handle,
                goal_id,
                GoalStatus.STATUS_ABORTED,
                termination_reason="GOAL_SPEECH_NOT_COMPLETED",
            )

        if speech_result != "COMPLETED":
            self.get_logger().warning(
                "[GOAL GATE] Reception speech not completed | "
                f"goal={goal_id} | state={speech_result}"
            )

        # ====================================================
        # 2. CHECK PUBLIC GOAL OWNERSHIP
        # ====================================================

        with self.navigation_state_lock:
            public_handle = self.public_navigation_goals.get(
                goal_id
            )

            owned = (
                public_handle is goal_handle
                and goal_handle.is_active
            )

            control = self.navigation_control_states.get(
                goal_id,
                self.CONTROL_STOPPING,
            )

        if not owned:
            return self._finish_navigation(
                goal_handle,
                goal_id,
                GoalStatus.STATUS_ABORTED,
                termination_reason="PUBLIC_GOAL_OWNERSHIP_LOST",
            )

        if control == self.CONTROL_STOPPING:
            return self._finish_navigation(
                goal_handle,
                goal_id,
                GoalStatus.STATUS_CANCELED,
            )

        # ====================================================
        # 3. INFORM COLLABORATION MANAGER
        # ====================================================

        # This means that the reception announcement has
        # finished. It does NOT authorize navigation.

        self._publish_gate_command_result(
            goal_id=goal_id,
            action="GOAL_READY",
            request_id="",
            status="READY",
            reason="GOAL_RECEPTION_ANNOUNCEMENT_FINISHED",
        )

        self.get_logger().info(
            "[GOAL GATE] "
            f"GOAL_READY={goal_id} | "
            "WAITING_FOR_COLLABORATIVE_AUTHORIZATION"
        )

        # ====================================================
        # 4. WAIT FOR START OR RESUME
        # ====================================================

        release = self._wait_for_control_release(
            goal_handle,
            goal_id,
        )

        if release in {"STOPPING", "CANCELED"}:
            return self._finish_navigation(
                goal_handle,
                goal_id,
                GoalStatus.STATUS_CANCELED,
            )

        if release != "RUNNING":
            return self._finish_navigation(
                goal_handle,
                goal_id,
                GoalStatus.STATUS_ABORTED,
                termination_reason="CONTROL_RELEASE_UNAVAILABLE",
            )

        # ====================================================
        # 5. FORWARD ONLY UNDER AUTHORIZATION
        # ====================================================

        return self._forward_to_nav2(
            public_goal_handle=goal_handle,
            public_goal_id=goal_id,
        )
    
    def execute_callback(
        self,
        goal_handle,
    ) -> NavigateToPose.Result:
        """
        Own the lifetime of exactly one public navigation goal.

        Register its initial control state, publish its
        authoritative context, and release all local
        bookkeeping after public termination.
        """
        goal_id = self._goal_id_hex(goal_handle)

        with self.navigation_state_lock:
            self.public_goal_reserved = False

            if not hasattr(self, "resume_authorization_deadlines"):
                self.resume_authorization_deadlines = {}

            if not hasattr(self, "tf_buffer"):
                self.tf_buffer = Buffer()

                self.tf_listener = TransformListener(
                    self.tf_buffer,
                    self,
                )

            self.public_navigation_goals[goal_id] = goal_handle

            canceled_before_execution = (
                goal_handle.is_cancel_requested
            )

            if canceled_before_execution:
                self.public_navigation_phases[goal_id] = (
                    self.PHASE_STOPPING
                )

                self.navigation_control_states[goal_id] = (
                    self.CONTROL_STOPPING
                )

            else:
                self.public_navigation_phases[goal_id] = (
                    self.PHASE_WAITING_FOR_SPEECH
                )

                self.navigation_control_states[goal_id] = (
                    self.CONTROL_WAITING
                )

            self.navigation_control_requests[goal_id] = ""

        try:
            self._publish_navigation_goal_context(
                goal_id,
                goal_handle.request.pose,
            )

            self._publish_navigation_state()

            return self._execute_goal(goal_handle)

        finally:
            terminal_states = {
                GoalStatus.STATUS_SUCCEEDED: "SUCCEEDED",
                GoalStatus.STATUS_ABORTED: "ABORTED",
                GoalStatus.STATUS_CANCELED: "CANCELED",
            }

            terminal_state = terminal_states.get(
                goal_handle.status,
                "UNKNOWN",
            )

            with self.navigation_state_lock:
                self.public_navigation_goals.pop(
                    goal_id, None
                )

                self.public_navigation_phases.pop(
                    goal_id, None
                )

                self.navigation_control_states.pop(
                    goal_id, None
                )

                self.navigation_control_requests.pop(
                    goal_id, None
                )

                self.pending_hold_requests.pop(
                    goal_id, None
                )

                self.confirmed_hold_requests.pop(
                    goal_id, None
                )

                self.last_control_stamp_ns.pop(
                    goal_id, None
                )

                self.internal_send_command_ids.pop(
                    goal_id, None
                )

                self.goals_with_social_hold.discard(goal_id)
                self.goals_ever_sent.discard(goal_id)

                self.authorization_actions.pop(
                    goal_id, None
                )

                self.execution_authorization_deadlines.pop(
                    goal_id, None
                )

                self.resume_authorization_deadlines.pop(
                    goal_id, None
                )

                self.resume_waiting_for_candidate.pop(
                    goal_id, None
                )

                with self.trajectory_lock:
                    self.authorized_resume_path_ids.pop(
                        goal_id, None
                    )

                    self.pending_resume_path_ids.pop(
                        goal_id, None
                    )

                    stale_paths = [
                        path_id
                        for path_id, candidate
                        in self.candidate_trajectories.items()
                        if str(
                            candidate.navigation_goal_id or ""
                        ).strip() == goal_id
                    ]

                    for path_id in stale_paths:
                        self.candidate_trajectories.pop(
                            path_id, None
                        )

                self.internal_cancel_pending.discard(
                    goal_id
                )

                self.internal_send_pending.discard(
                    goal_id
                )

                self.last_navigation_goal_id = goal_id
                self.last_navigation_state = terminal_state

            self._publish_navigation_state()

    def _set_navigation_phase(
        self,
        goal_id: str,
        phase: str,
    ) -> None:
        """Update the phase of an active public goal."""

        with self.navigation_state_lock:

            if goal_id not in self.public_navigation_goals:
                return

            self.public_navigation_phases[
                goal_id
            ] = phase

        self._publish_navigation_state()

    # ====================================================================
    # INTERNAL NAV2 STATUS
    # ====================================================================

    def _internal_status_callback(
        self,
        msg: GoalStatusArray,
    ) -> None:
        """Observe goals directly from the internal Nav2 action."""

        statuses = {}

        for entry in msg.status_list:

            uuid_bytes = bytes(
                entry.goal_info.goal_id.uuid
            )

            if (
                len(uuid_bytes) != 16
                or not any(uuid_bytes)
            ):
                continue

            statuses[
                uuid_bytes.hex()
            ] = entry.status

        with self.navigation_state_lock:

            self.observed_internal_statuses = statuses
            self.idle_start_allowed = False

        self._publish_navigation_state()

    def _resolve_unowned_navigation_state(self):
        """
        Resolve navigation state when no public goal is owned.

        Caller must hold navigation_state_lock.
        """

        statuses = self.observed_internal_statuses

        if statuses is None:

            if self.idle_start_allowed:
                return "", "IDLE"

            return "", "UNKNOWN"

        active_states = {
            GoalStatus.STATUS_ACCEPTED: "FORWARDING",
            GoalStatus.STATUS_EXECUTING: "NAVIGATING",
            GoalStatus.STATUS_CANCELING: "CANCELING",
        }

        active_goals = [
            (goal_id, active_states[status])
            for goal_id, status in statuses.items()
            if status in active_states
        ]

        if len(active_goals) > 1:
            return "", "UNKNOWN"

        if len(active_goals) == 1:
            return active_goals[0]

        terminal_states = {
            GoalStatus.STATUS_SUCCEEDED,
            GoalStatus.STATUS_ABORTED,
            GoalStatus.STATUS_CANCELED,
        }

        if any(
            status not in terminal_states
            for status in statuses.values()
        ):
            return "", "UNKNOWN"

        if self.last_navigation_state in {
            "SUCCEEDED",
            "ABORTED",
            "CANCELED",
        }:
            return (
                self.last_navigation_goal_id,
                self.last_navigation_state,
            )

        return "", "IDLE"

    def _publish_navigation_state(self) -> None:
        """
        Publish one consistent PUBLIC navigation-context snapshot.

        Internal Nav2 attempts may terminate, be canceled and be replaced
        while the same public navigation goal remains alive.

        Therefore an internal terminal status must never directly become
        a public terminal navigation state.
        """

        with self.navigation_state_lock:

            server_ready = self.internal_action_client.server_is_ready()

            if (
                self.internal_server_was_ready
                and not server_ready
            ):
                self.observed_internal_statuses = None
                self.idle_start_allowed = False

            self.internal_server_was_ready = server_ready

            goals = list(
                self.public_navigation_goals.items()
            )

            goal_id = ""
            state = "UNKNOWN"

            if len(goals) > 1:

                state = "UNKNOWN"

            elif len(goals) == 1:

                goal_id, public_handle = goals[0]

                phase = self.public_navigation_phases.get(
                    goal_id,
                    "UNKNOWN",
                )

                # Public social/gate phases always have priority over
                # the status of the current internal Nav2 attempt.
                if phase in {
                    self.PHASE_WAITING_FOR_SPEECH,
                    self.PHASE_WAITING,
                    self.PHASE_SOCIAL_HOLD,
                    self.PHASE_RESUMING,
                    self.PHASE_REPLANNING,
                    self.PHASE_STOPPING,
                }:

                    state = phase

                elif not server_ready:

                    state = "UNKNOWN"

                else:

                    state = phase

                    with self.internal_goal_lock:
                        internal_handle = (
                            self.internal_goal_handles.get(
                                goal_id
                            )
                        )

                    internal_status = None

                    if internal_handle is not None:

                        internal_status = internal_handle.status

                        if (
                            self.observed_internal_statuses
                            is not None
                        ):

                            internal_goal_id = bytes(
                                internal_handle.goal_id.uuid
                            ).hex()

                            internal_status = (
                                self.observed_internal_statuses.get(
                                    internal_goal_id,
                                    internal_status,
                                )
                            )

                    if (
                        internal_status
                        == GoalStatus.STATUS_EXECUTING
                    ):

                        state = "NAVIGATING"

                    elif (
                        internal_status
                        == GoalStatus.STATUS_ACCEPTED
                    ):

                        state = "FORWARDING"

                    elif (
                        internal_status
                        == GoalStatus.STATUS_CANCELING
                    ):

                        # This is cancellation of an INTERNAL attempt.
                        # Keep the current public phase.
                        state = phase

                    elif internal_status in {
                        GoalStatus.STATUS_SUCCEEDED,
                        GoalStatus.STATUS_ABORTED,
                        GoalStatus.STATUS_CANCELED,
                    }:

                        # The internal attempt has terminated, but the
                        # public action is finalized only by
                        # _finish_navigation().
                        state = phase

            elif server_ready:

                goal_id, state = (
                    self._resolve_unowned_navigation_state()
                )

            output = NavigationState()

            output.header.stamp = (
                self.get_clock()
                .now()
                .to_msg()
            )

            output.navigation_goal_id = goal_id
            output.navigation_state = state

            output.navigation_active = (
                state == "NAVIGATING"
                and bool(goal_id)
            )

            self.navigation_state_publisher.publish(
                output
            )

    # ====================================================================
    # SPEECH GATE
    # ====================================================================
    
    def _wait_for_speech_terminal(
        self,
        *,
        goal_handle,
        decision_id,
        goal_id="",
        pause_announcement=False,
    ):
        """
        Bounded speech transaction wait.

        Return:
            COMPLETED
            FAILED
            AUDIO_CANCELED
            START_TIMEOUT
            COMPLETION_TIMEOUT
            CANCELED
            SUPERSEDED
            ROS_SHUTDOWN
        """

        start_deadline = (
            time.monotonic()
            + self.speech_timeout_sec
        )

        completion_deadline = None

        while rclpy.ok():

            if goal_handle.is_cancel_requested:
                return "CANCELED"

            with self.speech_condition:

                state = self.speech_states.get(
                    decision_id
                )

                now = time.monotonic()

                if state == "COMPLETED":

                    self.speech_states.pop(
                        decision_id, None
                    )

                    self.speech_completed += 1
                    return "COMPLETED"

                if state == "FAILED":

                    self.speech_states.pop(
                        decision_id, None
                    )

                    self.speech_failed += 1
                    return "FAILED"

                if state == "CANCELED":

                    self.speech_states.pop(
                        decision_id, None
                    )

                    return "AUDIO_CANCELED"

                if state == "STARTED":
                    if completion_deadline is None:
                        completion_deadline = now + 45.0

                if completion_deadline is None:

                    if now >= start_deadline:
                        self.speech_timed_out += 1
                        return "START_TIMEOUT"

                elif now >= completion_deadline:

                    self.speech_timed_out += 1
                    return "COMPLETION_TIMEOUT"

                self.speech_condition.wait(
                    timeout=0.1
                )

            # A pause announcement can become obsolete
            # before playback starts.

            if (
                pause_announcement
                and completion_deadline is None
                and goal_id
            ):
                if self._get_navigation_control_state(
                    goal_id
                ) != self.CONTROL_WAITING:
                    return "SUPERSEDED"

        return "ROS_SHUTDOWN"

    def _send_internal_goal_if_running_old(
        self,
        public_goal_handle,
        public_goal_id,
        client,
        goal,
        internal_uuid,
        resume_path_id="",
        feedback_callback=None,
    ):
        """
        Submit an internal goal with atomic authorization ownership.

        Record the exact collaborative request_id under which
        the send occurred.

        An exception during send does not prove that Nav2
        never received the action.
        """
        with self.navigation_state_lock:

            if (
                not public_goal_handle.is_active
                or public_goal_handle.is_cancel_requested
            ):
                return None

            if self.navigation_control_states.get(
                public_goal_id
            ) != self.CONTROL_RUNNING:
                return None

            if not self._resume_authorization_fresh_locked(
                public_goal_id
            ):
                self._expire_resume_authorization_locked(
                    public_goal_id
                )
                return None

            expected_action = (
                "RESUME" if resume_path_id else "START"
            )

            if self.authorization_actions.get(
                public_goal_id
            ) != expected_action:
                return None

            command_id = self.navigation_control_requests.get(
                public_goal_id,
                "",
            )

            if not command_id:
                return None

            if (
                public_goal_id in self.internal_send_pending
                or public_goal_id in self.internal_cancel_pending
            ):
                return None

            with self.internal_goal_lock:
                if public_goal_id in self.internal_goal_handles:
                    return None

            with self.trajectory_lock:
                authorized_path = (
                    self.authorized_resume_path_ids.get(
                        public_goal_id,
                        "",
                    )
                )

                if authorized_path != resume_path_id:
                    return None

                if resume_path_id:
                    candidate = self.candidate_trajectories.get(
                        resume_path_id
                    )

                    if candidate is None:
                        return None

                    if (
                        str(
                            candidate.navigation_goal_id or ""
                        ).strip()
                        != public_goal_id
                    ):
                        return None

                    if not candidate.path.poses:
                        return None

                    if (
                        goal.path.header.frame_id
                        != candidate.path.header.frame_id
                    ):
                        return None

            self.goals_ever_sent.add(public_goal_id)

            self.internal_send_pending.add(public_goal_id)

            self.internal_send_command_ids[
                public_goal_id
            ] = command_id

            self.public_navigation_phases[
                public_goal_id
            ] = self.PHASE_FORWARDING

            kwargs = {
                "goal_uuid": internal_uuid,
            }

            if feedback_callback is not None:
                kwargs["feedback_callback"] = (
                    feedback_callback
                )

            # Both the authorization check and submission
            # occur while owning navigation_state_lock.
            #
            # If send_goal_async raises, keep the pending
            # marker. The caller must quarantine the attempt.
            return client.send_goal_async(
                goal,
                **kwargs,
            )
    def _announce_goal(
        self,
        *,
        goal_handle,
        decision_id: str,
    ) -> str:
        """Announce receipt without claiming navigation has begun."""

        if not self._wait_for_tts_discovery(goal_handle):
            if goal_handle.is_cancel_requested:
                return "CANCELED"

            self.speech_unavailable += 1
            return "UNAVAILABLE"

        request = ExplanationText()
        stamp = self.get_clock().now().to_msg()

        request.header.stamp = stamp
        request.header.frame_id = "map"
        request.decision_id = decision_id
        request.language = self.language

        texts = {
            "en": "I received the navigation destination.",
            "fr": "J'ai reçu la destination de navigation.",
            "es": "He recibido el destino de navegación.",
        }

        request.text = texts.get(
            self.language,
            texts["en"],
        )

        request.provider = "deterministic"
        request.model = "navigation_goal_gate_v3"
        request.used_fallback = False
        request.used_cache = False
        request.speech_kind = "NAVIGATION_GOAL"
        request.context_stamp = stamp

        with self.speech_condition:
            self.speech_states[decision_id] = "PENDING"

        self.speech_publisher.publish(request)

        self.get_logger().info(
            "[GOAL GATE] SPEECH_REQUESTED | "
            f"decision={decision_id}"
        )

        try:
            return self._wait_for_speech_terminal(
                goal_handle=goal_handle,
                decision_id=decision_id,
            )
        finally:
            with self.speech_condition:
                self.speech_states.pop(decision_id, None)
                
    def _announce_navigation_paused(
        self,
        *,
        goal_handle,
        goal_id: str,
    ) -> str:
        """Announce a social hold without changing authorization."""

        decision_id = (
            f"navigation_goal_paused_{goal_id}"
        )

        if self._get_navigation_control_state(
            goal_id
        ) != self.CONTROL_WAITING:
            return "SUPERSEDED"

        if not self._wait_for_tts_discovery(goal_handle):
            if goal_handle.is_cancel_requested:
                return "CANCELED"

            self.speech_unavailable += 1
            return "UNAVAILABLE"

        request = ExplanationText()
        stamp = self.get_clock().now().to_msg()

        request.header.stamp = stamp
        request.header.frame_id = "map"
        request.decision_id = decision_id
        request.language = self.language

        texts = {
            "en": (
                "Navigation is currently paused because "
                "the social situation cannot be confirmed."
            ),
            "fr": (
                "La navigation est actuellement en pause "
                "car la situation sociale ne peut pas être confirmée."
            ),
            "es": (
                "La navegación está actualmente en pausa "
                "porque no se puede confirmar la situación social."
            ),
        }

        request.text = texts.get(
            self.language,
            texts["en"],
        )

        request.provider = "deterministic"
        request.model = "navigation_goal_gate_v3"
        request.used_fallback = False
        request.used_cache = False
        request.speech_kind = "NAVIGATION_STATUS"
        request.context_stamp = stamp

        with self.speech_condition:
            self.speech_states[decision_id] = "PENDING"

        # Recheck immediately before publication.
        if self._get_navigation_control_state(
            goal_id
        ) != self.CONTROL_WAITING:
            with self.speech_condition:
                self.speech_states.pop(decision_id, None)
            return "SUPERSEDED"

        self.speech_publisher.publish(request)

        try:
            return self._wait_for_speech_terminal(
                goal_handle=goal_handle,
                decision_id=decision_id,
                goal_id=goal_id,
                pause_announcement=True,
            )
        finally:
            with self.speech_condition:
                self.speech_states.pop(decision_id, None)
                
    def _wait_for_tts_discovery(
        self,
        goal_handle,
    ) -> bool:
        """Wait briefly for TTS request and status endpoints."""

        deadline = (
            time.monotonic()
            + self.tts_discovery_timeout_sec
        )

        while rclpy.ok():

            if goal_handle.is_cancel_requested:
                return False

            request_subscribers = (
                self.count_subscribers(
                    self.speech_topic
                )
            )

            status_publishers = (
                self.count_publishers(
                    self.speech_status_topic
                )
            )

            if (
                request_subscribers > 0
                and status_publishers > 0
            ):
                return True

            if time.monotonic() >= deadline:
                return False

            time.sleep(0.10)

        return False

    def speech_status_callback(
        self,
        msg: SpeechStatus,
    ) -> None:
        """
        Store exact speech status and wake waiters.

        Terminal states cannot be overwritten by
        delayed STARTED messages.
        """

        decision_id = str(
            msg.decision_id or ""
        ).strip()

        state = str(
            msg.state or ""
        ).strip().upper()

        if not decision_id:
            return

        if state not in {
            "STARTED",
            "COMPLETED",
            "FAILED",
            "CANCELED",
        }:
            return

        with self.speech_condition:

            # Only active speech requests should be
            # represented in this state table.
            if decision_id not in self.speech_states:
                return

            previous = self.speech_states.get(
                decision_id
            )

            if previous in {
                "COMPLETED",
                "FAILED",
                "CANCELED",
            }:
                return

            self.speech_states[decision_id] = state

            self.speech_condition.notify_all()

        self.get_logger().info(
            "[GOAL GATE] SPEECH_STATUS | "
            f"decision={decision_id} | state={state}"
        )
        
    # ====================================================================
    # INTERNAL NAV2 ACTION CLIENT
    # ====================================================================

    def _forward_to_nav2(
        self,
        *,
        public_goal_handle,
        public_goal_id: str,
    ) -> NavigateToPose.Result:
        """
        Supervise sequential internal Nav2 attempts.

        Invariants:
          - One outstanding internal attempt per public goal.
          - Every send belongs to an exact authorization request.
          - WAIT/REPLAN retain the public goal.
          - STOP requests cancellation and waits for termination.
          - A late acceptance is never considered safe by default.
          - No new attempt starts before the previous terminal result.
        """
        attempt = 0

        terminal_statuses = {
            GoalStatus.STATUS_SUCCEEDED,
            GoalStatus.STATUS_ABORTED,
            GoalStatus.STATUS_CANCELED,
        }

        def finish(status, result=None, reason=None):
            return self._finish_navigation(
                public_goal_handle,
                public_goal_id,
                status,
                result,
                termination_reason=reason,
            )

        def quarantine(reason):
            """
            Keep uncertain internal action ownership.

            This is a fault containment state, not a
            confirmation that the robot has stopped.
            """
            with self.navigation_state_lock:
                current = self.navigation_control_states.get(
                    public_goal_id
                )

                if current != self.CONTROL_STOPPING:
                    self.navigation_control_states[
                        public_goal_id
                    ] = self.CONTROL_SOCIAL_HOLD

                    self.public_navigation_phases[
                        public_goal_id
                    ] = self.PHASE_SOCIAL_HOLD

                self._clear_resume_authorization_locked(
                    public_goal_id
                )

                self.authorization_actions.pop(
                    public_goal_id,
                    None,
                )

                self.goals_with_social_hold.add(
                    public_goal_id
                )

                # Prevent another internal attempt.
                self.internal_send_pending.add(
                    public_goal_id
                )

            self._publish_navigation_state()

            while rclpy.ok():
                self.get_logger().error(
                    "[GOAL GATE] INTERNAL_OWNERSHIP_UNCONFIRMED | "
                    f"goal={public_goal_id} | "
                    f"reason={reason} | "
                    "new_execution_inhibited=True",
                    throttle_duration_sec=10.0,
                )
                time.sleep(0.2)

            return finish(
                GoalStatus.STATUS_ABORTED,
                reason=(
                    "ROS_SHUTDOWN_WITH_UNCONFIRMED_INTERNAL_ACTION"
                ),
            )

        while rclpy.ok():

            control = self._get_navigation_control_state(
                public_goal_id
            )

            if (
                public_goal_handle.is_cancel_requested
                or control == self.CONTROL_STOPPING
            ):
                # We reach this branch only between
                # internal attempts.
                return finish(
                    GoalStatus.STATUS_CANCELED
                )

            if control != self.CONTROL_RUNNING:
                release = self._wait_for_control_release(
                    public_goal_handle,
                    public_goal_id,
                )

                if release in {"STOPPING", "CANCELED"}:
                    return finish(
                        GoalStatus.STATUS_CANCELED
                    )

                if release != "RUNNING":
                    return finish(
                        GoalStatus.STATUS_ABORTED,
                        reason="CONTROL_RELEASE_UNAVAILABLE",
                    )

                continue

            # ====================================================
            # 1. SNAPSHOT CURRENT AUTHORIZATION
            # ====================================================

            with self.navigation_state_lock:
                if not self._resume_authorization_fresh_locked(
                    public_goal_id
                ):
                    self._expire_resume_authorization_locked(
                        public_goal_id
                    )
                    continue

                with self.trajectory_lock:
                    resume_path_id = (
                        self.authorized_resume_path_ids.get(
                            public_goal_id,
                            "",
                        )
                    )

                    candidate = (
                        self.candidate_trajectories.get(
                            resume_path_id
                        )
                        if resume_path_id
                        else None
                    )

            if resume_path_id and candidate is None:
                with self.navigation_state_lock:
                    self._expire_resume_authorization_locked(
                        public_goal_id
                    )
                continue

            use_follow_path = bool(resume_path_id)

            client = (
                self.follow_path_action_client
                if use_follow_path
                else self.internal_action_client
            )

            # ====================================================
            # 2. CHECK NAV2 ACTION SERVER
            # ====================================================

            try:
                server_available = client.wait_for_server(
                    timeout_sec=self.navigation_server_timeout_sec
                )

            except Exception as exc:
                self.get_logger().error(
                    "[GOAL GATE] Action-server discovery failed | "
                    f"goal={public_goal_id} | error={exc}"
                )
                server_available = False

            if not server_available:
                if self._get_navigation_control_state(
                    public_goal_id
                ) != self.CONTROL_RUNNING:
                    continue

                return finish(
                    GoalStatus.STATUS_ABORTED,
                    reason="NAV2_ACTION_SERVER_UNAVAILABLE",
                )

            stamp = self.get_clock().now().to_msg()

            internal_uuid = UUID(
                uuid=list(uuid.uuid4().bytes)
            )

            if use_follow_path:
                goal = FollowPath.Goal()

                goal.path = copy.deepcopy(
                    candidate.path
                )

                goal.path.header.stamp = stamp

                for pose in goal.path.poses:
                    pose.header.stamp = stamp

                feedback_callback = None

            else:
                goal = copy.deepcopy(
                    public_goal_handle.request
                )

                goal.pose.header.stamp = stamp

                feedback_callback = (
                    lambda feedback:
                    self._forward_feedback(
                        public_goal_handle,
                        feedback,
                    )
                )

            # ====================================================
            # 3. SEND UNDER ATOMIC AUTHORIZATION
            # ====================================================

            submission_started_at = time.monotonic()

            try:
                send_future = (
                    self._send_internal_goal_if_running(
                        public_goal_handle,
                        public_goal_id,
                        client,
                        goal,
                        internal_uuid,
                        resume_path_id=resume_path_id,
                        feedback_callback=feedback_callback,
                    )
                )

            except Exception as exc:
                self.get_logger().error(
                    "[GOAL GATE] Submission uncertain | "
                    f"goal={public_goal_id} | error={exc}"
                )

                return quarantine(
                    "SEND_GOAL_EXCEPTION"
                )

            if send_future is None:
                with self.navigation_state_lock:
                    send_outstanding = (
                        public_goal_id in self.internal_send_pending
                    )

                if send_outstanding:
                    return quarantine(
                        "SEND_GOAL_FUTURE_MISSING"
                    )

                continue

            attempt += 1

            with self.navigation_state_lock:
                sending_command_id = (
                    self.internal_send_command_ids.get(
                        public_goal_id,
                        "",
                    )
                )

            if not sending_command_id:
                return quarantine(
                    "SENDING_AUTHORIZATION_ID_MISSING"
                )

            # ====================================================
            # 4. WAIT FOR ACCEPTANCE WITHOUT LOSING OWNERSHIP
            # ====================================================

            acceptance_deadline = (
                time.monotonic()
                + self.navigation_server_timeout_sec
            )

            acceptance_timeout_reported = False

            while rclpy.ok() and not send_future.done():

                if (
                    not acceptance_timeout_reported
                    and time.monotonic() >= acceptance_deadline
                ):
                    acceptance_timeout_reported = True

                    with self.navigation_state_lock:
                        current = self.navigation_control_states.get(
                            public_goal_id
                        )

                        if current == self.CONTROL_RUNNING:
                            self.navigation_control_states[
                                public_goal_id
                            ] = self.CONTROL_SOCIAL_HOLD

                            self.public_navigation_phases[
                                public_goal_id
                            ] = self.PHASE_SOCIAL_HOLD

                            self._clear_resume_authorization_locked(
                                public_goal_id
                            )

                            self.authorization_actions.pop(
                                public_goal_id,
                                None,
                            )

                            self.goals_with_social_hold.add(
                                public_goal_id
                            )

                    self._publish_navigation_state()

                    self.get_logger().error(
                        "[GOAL GATE] ACCEPTANCE_UNCONFIRMED | "
                        f"goal={public_goal_id} | "
                        "public_goal_retained=True | "
                        "waiting_for_late_response=True"
                    )

                time.sleep(0.05)

            if not rclpy.ok():
                return finish(
                    GoalStatus.STATUS_ABORTED,
                    reason="ROS_SHUTDOWN_DURING_ACCEPTANCE",
                )

            try:
                internal_handle = send_future.result()

            except Exception as exc:
                self.get_logger().error(
                    "[GOAL GATE] Acceptance response exception | "
                    f"goal={public_goal_id} | error={exc}"
                )

                return quarantine(
                    "GOAL_ACCEPTANCE_RESPONSE_UNCONFIRMED"
                )

            if internal_handle is None:
                return quarantine(
                    "GOAL_ACCEPTANCE_HANDLE_MISSING"
                )

            if not internal_handle.accepted:
                with self.navigation_state_lock:
                    self.internal_send_pending.discard(
                        public_goal_id
                    )

                    self.internal_send_command_ids.pop(
                        public_goal_id,
                        None,
                    )

                if self._get_navigation_control_state(
                    public_goal_id
                ) != self.CONTROL_RUNNING:
                    continue

                return finish(
                    GoalStatus.STATUS_ABORTED,
                    reason="NAV2_GOAL_REJECTED",
                )

            # ====================================================
            # 5. REGISTER ACCEPTED INTERNAL ACTION
            # ====================================================

            with self.navigation_state_lock:
                current_control = (
                    self.navigation_control_states.get(
                        public_goal_id
                    )
                )

                current_command_id = (
                    self.navigation_control_requests.get(
                        public_goal_id,
                        "",
                    )
                )

                with self.trajectory_lock:
                    current_authorized_path = (
                        self.authorized_resume_path_ids.get(
                            public_goal_id,
                            "",
                        )
                    )

                accepted_under_valid_control = (
                    public_goal_handle.is_active
                    and not public_goal_handle.is_cancel_requested
                    and current_control == self.CONTROL_RUNNING
                    and current_command_id == sending_command_id
                    and public_goal_id in self.internal_send_pending
                    and self.internal_send_command_ids.get(
                        public_goal_id
                    ) == sending_command_id
                    and self.authorization_actions.get(
                        public_goal_id
                    ) == ("RESUME" if use_follow_path else "START")
                    and current_authorized_path == resume_path_id
                    #and time.monotonic() - submission_started_at
                    #< self.execution_authorization_max_age_sec
                )

                with self.internal_goal_lock:
                    self.internal_goal_handles[
                        public_goal_id
                    ] = internal_handle

                self.internal_send_pending.discard(
                    public_goal_id
                )

                self.internal_send_command_ids.pop(
                    public_goal_id,
                    None,
                )

                if accepted_under_valid_control:
                    # The authorization has been consumed.
                    self.execution_authorization_deadlines.pop(
                        public_goal_id,
                        None,
                    )

                    self.authorization_actions.pop(
                        public_goal_id,
                        None,
                    )

                    if use_follow_path:
                        with self.trajectory_lock:
                            self.authorized_resume_path_ids.pop(
                                public_goal_id,
                                None,
                            )

                elif current_command_id == sending_command_id:
                    # Expired or revoked without a newer
                    # authorization. Do not preserve it.
                    self._clear_resume_authorization_locked(
                        public_goal_id
                    )

                    self.authorization_actions.pop(
                        public_goal_id,
                        None,
                    )

                    if current_control != self.CONTROL_STOPPING:
                        self.navigation_control_states[
                            public_goal_id
                        ] = self.CONTROL_SOCIAL_HOLD

                        self.public_navigation_phases[
                            public_goal_id
                        ] = self.PHASE_SOCIAL_HOLD

                # If the command ID changed, preserve the newer
                # RESUME transaction. This accepted goal still
                # belongs to the OLD command and must be canceled.

            self.goals_forwarded += 1

            self.get_logger().info(
                "[GOAL GATE] INTERNAL_ACCEPTED | "
                f"attempt={attempt} | "
                f"goal={public_goal_id} | "
                f"request={sending_command_id} | "
                f"action={'FollowPath' if use_follow_path else 'NavigateToPose'} | "
                f"path={resume_path_id or 'INITIAL'} | "
                f"authorization_valid={accepted_under_valid_control}"
            )

            # ====================================================
            # 6. OBTAIN TERMINAL RESULT FUTURE
            # ====================================================

            try:
                result_future = internal_handle.get_result_async()

                if result_future is None:
                    raise RuntimeError(
                        "Nav2 result future missing."
                    )

            except Exception as exc:
                try:
                    internal_handle.cancel_goal_async()
                except Exception:
                    pass

                self.get_logger().error(
                    "[GOAL GATE] Result monitoring unavailable | "
                    f"goal={public_goal_id} | error={exc}"
                )

                return quarantine(
                    "RESULT_MONITORING_UNCONFIRMED"
                )

            # ====================================================
            # 7. MONITOR TERMINATION AND SOCIAL CANCELLATION
            # ====================================================

            cancel_future = None
            cancel_ack_checked = False
            next_cancel_retry_at = 0.0

            cancel_started_at = None
            cancel_timeout_reported = False

            while rclpy.ok() and not result_future.done():
                now = time.monotonic()

                control = self._get_navigation_control_state(
                    public_goal_id
                )

                if public_goal_handle.is_cancel_requested:
                    control = self.CONTROL_STOPPING

                must_cancel = (
                    not accepted_under_valid_control
                    or control != self.CONTROL_RUNNING
                )

                if (
                    must_cancel
                    and cancel_future is None
                    and now >= next_cancel_retry_at
                ):
                    with self.navigation_state_lock:
                        self.internal_cancel_pending.add(
                            public_goal_id
                        )

                    try:
                        cancel_future = (
                            internal_handle.cancel_goal_async()
                        )

                        if cancel_future is None:
                            raise RuntimeError(
                                "Cancel future missing."
                            )

                        cancel_ack_checked = False

                        if cancel_started_at is None:
                            cancel_started_at = now

                    except Exception as exc:
                        cancel_future = None
                        next_cancel_retry_at = now + 1.0

                        self.get_logger().error(
                            "[GOAL GATE] Cancel submission failed | "
                            f"goal={public_goal_id} | error={exc}",
                            throttle_duration_sec=2.0,
                        )

                if (
                    cancel_future is not None
                    and cancel_future.done()
                    and not cancel_ack_checked
                ):
                    cancel_ack_checked = True

                    try:
                        response = cancel_future.result()

                        target_uuid = bytes(
                            internal_handle.goal_id.uuid
                        )

                        acknowledged = (
                            response is not None
                            and any(
                                bytes(info.goal_id.uuid)
                                == target_uuid
                                for info in response.goals_canceling
                            )
                        )

                    except Exception:
                        acknowledged = False

                    if not acknowledged:
                        self.get_logger().warning(
                            "[GOAL GATE] Cancellation not acknowledged | "
                            f"goal={public_goal_id} | "
                            "waiting_for_terminal_result=True",
                            throttle_duration_sec=2.0,
                        )

                        cancel_future = None
                        next_cancel_retry_at = (
                            time.monotonic() + 1.0
                        )

                if (
                    cancel_started_at is not None
                    and not cancel_timeout_reported
                    and now - cancel_started_at
                    >= self.navigation_cancel_timeout_sec
                ):
                    cancel_timeout_reported = True

                    self.get_logger().error(
                        "[GOAL GATE] CANCEL_TERMINAL_TIMEOUT | "
                        f"goal={public_goal_id} | "
                        "internal_goal_still_owned=True"
                    )

                time.sleep(0.05)

            if not rclpy.ok():
                return finish(
                    GoalStatus.STATUS_ABORTED,
                    reason="ROS_SHUTDOWN_WITH_ACTIVE_INTERNAL_ACTION",
                )

            # ====================================================
            # 8. REQUIRE A CONFIRMED TERMINAL STATUS
            # ====================================================

            try:
                wrapped = result_future.result()

            except Exception as exc:
                self.get_logger().error(
                    "[GOAL GATE] Cannot read terminal result | "
                    f"goal={public_goal_id} | error={exc}"
                )

                return quarantine(
                    "INTERNAL_RESULT_READ_UNCONFIRMED"
                )

            if wrapped is None:
                return quarantine(
                    "INTERNAL_RESULT_MISSING"
                )

            status = int(wrapped.status)

            if status not in terminal_statuses:
                return quarantine(
                    f"INTERNAL_STATUS_NOT_TERMINAL_{status}"
                )

            # ====================================================
            # 9. RELEASE CONFIRMED TERMINAL OWNERSHIP
            # ====================================================

            with self.navigation_state_lock:
                with self.internal_goal_lock:
                    self.internal_goal_handles.pop(
                        public_goal_id,
                        None,
                    )

                self.internal_cancel_pending.discard(
                    public_goal_id
                )

                self.internal_send_pending.discard(
                    public_goal_id
                )

                self.internal_send_command_ids.pop(
                    public_goal_id,
                    None,
                )

            public_result = (
                None
                if use_follow_path
                else wrapped.result
            )

            control = self._get_navigation_control_state(
                public_goal_id
            )

            if (
                public_goal_handle.is_cancel_requested
                or control == self.CONTROL_STOPPING
            ):
                return finish(
                    GoalStatus.STATUS_CANCELED,
                    public_result,
                )

            # A new RESUME may have been accepted while
            # the previous attempt was terminating.
            #
            # Any confirmed terminal status releases the
            # previous attempt, not only STATUS_CANCELED.
            if self._activate_pending_resume(
                public_goal_id
            ):
                continue

            if not accepted_under_valid_control:
                continue

            if control != self.CONTROL_RUNNING:
                continue

            # ====================================================
            # 10. PROPAGATE THE ACTUAL NAV2 TERMINAL RESULT
            # ====================================================
            if self._retain_failed_directional_attempt(
                public_goal_id,
                sending_command_id,
                status,
            ):
                continue

            return finish(
                status,
                public_result,
                reason=(
                    "NAV2_ABORTED"
                    if status == GoalStatus.STATUS_ABORTED
                    else None
                ),
            )

        return finish(
            GoalStatus.STATUS_ABORTED,
            reason="ROS_SHUTDOWN",
        )
           
    def _verify_goal_reached(
        self,
        public_goal_handle,
        public_goal_id: str,
    ):
        """
        Corroborate a genuine internal Nav2 SUCCEEDED result.

        Nav2 remains responsible for its goal checker,
        including orientation.

        The Gate independently checks the final XY position
        with a small verification margin to accommodate
        measurement differences.

        Returns:
            (True, distance): arrival corroborated
            (False, distance): position outside tolerance
            (None, None): position cannot be verified
        """
        goal_pose = public_goal_handle.request.pose

        goal_frame = str(
            goal_pose.header.frame_id or ""
        ).strip()

        if not goal_frame:
            self.get_logger().error(
                "[GOAL GATE] "
                f"SUCCESS_VERIFICATION_FAILED={public_goal_id} | "
                "REASON=GOAL_FRAME_EMPTY"
            )
            return None, None

        try:
            transform = self.tf_buffer.lookup_transform(
                goal_frame,
                "base_footprint",
                Time(),
            )

        except Exception as exc:
            self.get_logger().error(
                "[GOAL GATE] "
                f"SUCCESS_VERIFICATION_FAILED={public_goal_id} | "
                f"TF={goal_frame}->base_footprint | "
                f"ERROR={exc}"
            )
            return None, None

        robot_x = float(
            transform.transform.translation.x
        )
        robot_y = float(
            transform.transform.translation.y
        )

        goal_x = float(
            goal_pose.pose.position.x
        )
        goal_y = float(
            goal_pose.pose.position.y
        )

        values = (
            robot_x,
            robot_y,
            goal_x,
            goal_y,
        )

        if not all(
            math.isfinite(value)
            for value in values
        ):
            self.get_logger().error(
                "[GOAL GATE] "
                f"SUCCESS_VERIFICATION_FAILED={public_goal_id} | "
                "REASON=NON_FINITE_POSITION"
            )
            return None, None

        distance = math.hypot(
            goal_x - robot_x,
            goal_y - robot_y,
        )

        # The configured value remains the nominal target.
        nominal_tolerance = float(
            self.goal_success_position_tolerance_m
        )

        # Additional corroboration margin, applied only
        # after Nav2 has returned an actual SUCCEEDED.
        verification_margin_m = 0.05

        effective_tolerance = (
            nominal_tolerance
            + verification_margin_m
        )

        reached = (
            distance <= effective_tolerance
        )

        self.get_logger().info(
            "[GOAL GATE] "
            f"SUCCESS_VERIFICATION={public_goal_id} | "
            f"ROBOT=({robot_x:.3f},{robot_y:.3f}) | "
            f"GOAL=({goal_x:.3f},{goal_y:.3f}) | "
            f"DISTANCE={distance:.3f} | "
            f"NOMINAL_TOLERANCE={nominal_tolerance:.3f} | "
            f"VERIFICATION_MARGIN={verification_margin_m:.3f} | "
            f"TOLERANCE={effective_tolerance:.3f} | "
            f"REACHED={reached}"
        )

        return reached, distance
               
    def _finish_navigation(
        self,
        public_goal_handle,
        public_goal_id: str,
        status: int,
        result=None,
        *,
        termination_reason=None,
    ) -> NavigateToPose.Result:
        """
        Complete the public NavigateToPose action.

        WAIT never calls this method while the public goal remains active.

        STOP maps to CANCELED.

        Nav2 ABORTED maps to ABORTED.

        Nav2 SUCCEEDED is accepted publicly only after the current physical
        robot position has independently corroborated arrival at the requested
        destination.
        """

        if result is None:
            result = NavigateToPose.Result()

        # ================================================================
        # PUBLIC USER CANCELLATION
        # ================================================================

        if (
            public_goal_handle.is_cancel_requested
            and status != GoalStatus.STATUS_ABORTED
        ):

            public_goal_handle.canceled()
            self.goals_canceled += 1

            self.get_logger().info(
                "[GOAL GATE] "
                f"PUBLIC_CANCELED={public_goal_id}"
            )

            return result

        # ================================================================
        # NAV2 SUCCESS
        #
        # Nav2's result is necessary but no longer sufficient.
        #
        # Corroborate it using the robot's current map-frame position before
        # publishing a terminal public SUCCEEDED state.
        # ================================================================

        if status == GoalStatus.STATUS_SUCCEEDED:

            reached, distance = (
                self._verify_goal_reached(
                    public_goal_handle,
                    public_goal_id,
                )
            )

            # ------------------------------------------------------------
            # Verified physical arrival.
            # ------------------------------------------------------------

            if reached is True:

                public_goal_handle.succeed()
                self.goals_succeeded += 1

                self.get_logger().info(
                    "[GOAL GATE] "
                    f"PUBLIC_SUCCEEDED={public_goal_id} | "
                    f"VERIFIED_DISTANCE={distance:.3f}"
                )

                return result

            # ------------------------------------------------------------
            # TF/current pose unavailable.
            #
            # Never turn an unverifiable Nav2 success into public success.
            # ------------------------------------------------------------

            if reached is None:

                public_goal_handle.abort()
                self.goals_aborted += 1

                self.get_logger().error(
                    "[GOAL GATE] "
                    f"PUBLIC_ABORTED={public_goal_id} | "
                    "REASON=NAV2_SUCCESS_UNVERIFIABLE"
                )

                return result

            # ------------------------------------------------------------
            # Nav2 says SUCCEEDED, but the actual robot remains outside
            # the goal tolerance.
            #
            # This is the false-success condition we observed previously.
            # ------------------------------------------------------------

            public_goal_handle.abort()
            self.goals_aborted += 1

            self.get_logger().error(
                "[GOAL GATE] "
                f"PUBLIC_ABORTED={public_goal_id} | "
                "REASON=NAV2_FALSE_SUCCESS | "
                f"DISTANCE={distance:.3f} | "
                "TOLERANCE="
                f"{self.goal_success_position_tolerance_m:.3f}"
            )

            return result

        # ================================================================
        # INTENTIONAL CANCELLATION
        # ================================================================
        """
        Complete the public NavigateToPose action.

        WAIT and REPLAN retain the public goal.

        A user-requested cancellation maps to CANCELED.

        A collaborative terminal STOP maps to ABORTED.

        Nav2 SUCCEEDED requires physical goal verification.
        """
        if status == GoalStatus.STATUS_CANCELED:

            if public_goal_handle.is_cancel_requested:
                public_goal_handle.canceled()
                self.goals_canceled += 1

                self.get_logger().info(
                    "[GOAL GATE] "
                    f"PUBLIC_CANCELED={public_goal_id}"
                )

            else:
                public_goal_handle.abort()
                self.goals_aborted += 1

                self.get_logger().warning(
                    "[GOAL GATE] "
                    f"PUBLIC_ABORTED={public_goal_id} | "
                    "REASON=COLLABORATIVE_TERMINAL_STOP"
                )

            return result

        # ================================================================
        # ABORT
        # ================================================================

        if status == GoalStatus.STATUS_ABORTED:

            public_goal_handle.abort()
            self.goals_aborted += 1

            reason = (
                termination_reason
                if termination_reason
                else "NAV2_ABORTED"
            )

            self.get_logger().error(
                "[GOAL GATE] "
                f"PUBLIC_ABORTED={public_goal_id} | "
                f"REASON={reason}"
            )

            return result

        # ================================================================
        # UNKNOWN
        # ================================================================

        public_goal_handle.abort()
        self.goals_aborted += 1

        self.get_logger().error(
            "[GOAL GATE] "
            f"PUBLIC_ABORTED={public_goal_id} | "
            f"REASON=UNEXPECTED_STATUS_{status}"
        )

        return result

    def _cancel_late_accepted_goal(
        self,
        future,
    ) -> None:
        """Cancel a goal accepted after the forwarding wait expired."""

        try:

            handle = future.result()

            if (
                handle is not None
                and handle.accepted
            ):

                handle.cancel_goal_async()

                self.get_logger().error(
                    "[GOAL GATE] "
                    "Late internal acceptance: "
                    "cancellation requested."
                )

        except Exception as exc:

            self.get_logger().error(
                "[GOAL GATE] "
                f"Late-goal cleanup failed: {exc}"
            )

    def _forward_feedback(
        self,
        public_goal_handle,
        feedback_message,
    ) -> None:
        """Forward Nav2 feedback to the public action client."""

        if not public_goal_handle.is_active:
            return

        try:

            public_goal_handle.publish_feedback(
                feedback_message.feedback
            )

        except Exception as exc:

            self.get_logger().warning(
                "[GOAL GATE] "
                f"Unable to forward feedback: {exc}"
            )

    # ====================================================================
    # HELPERS
    # ====================================================================

    @staticmethod
    def _goal_id_hex(
        goal_handle,
    ) -> str:
        """Return the public action goal UUID as hexadecimal text."""

        return bytes(
            goal_handle.goal_id.uuid
        ).hex()

    def log_diagnostics(self) -> None:
        """Report goal-gate runtime state."""

        with self.navigation_state_lock:

            active_goals = len(
                self.public_navigation_goals
            )

            control_states = dict(
                self.navigation_control_states
            )

        self.get_logger().info(
            "[GOAL GATE] "
            f"GOALS_RX={self.goals_received} | "
            f"FORWARDED={self.goals_forwarded} | "
            f"SUCCEEDED={self.goals_succeeded} | "
            f"ABORTED={self.goals_aborted} | "
            f"CANCELED={self.goals_canceled} | "
            f"ACTIVE={active_goals} | "
            f"CONTROL={control_states} | "
            f"SPEECH_COMPLETED={self.speech_completed} | "
            f"SPEECH_FAILED={self.speech_failed} | "
            f"SPEECH_TIMEOUT={self.speech_timed_out} | "
            f"SPEECH_UNAVAILABLE={self.speech_unavailable}"
        )

    def destroy_node(self) -> None:
        """Destroy action interfaces before destroying the node."""

        if self.public_action_server is not None:
            self.public_action_server.destroy()

        self.internal_action_client.destroy()
        self.follow_path_action_client.destroy()

        super().destroy_node()


def main(args=None) -> None:
    """Run the navigation goal gate."""

    rclpy.init(args=args)

    node = NavigationGoalGateNode()

    executor = MultiThreadedExecutor(
        num_threads=4
    )

    executor.add_node(node)

    try:
        executor.spin()

    except KeyboardInterrupt:
        pass

    finally:
        executor.shutdown()
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()