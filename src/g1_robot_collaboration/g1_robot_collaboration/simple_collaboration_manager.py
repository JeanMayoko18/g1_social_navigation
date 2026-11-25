#!/usr/bin/env python3
"""ROS 2 entry points and event routing for the C/D policy."""

import json
import time

from std_msgs.msg import String

from g1_social_msgs.msg import (
    DirectionalConstraintRequest,
    DirectionalConstraintStatus,
    SpeechCancelRequest,
    ExplanationText,
)

from .collaboration_manager_node import CollaborationManager
from .simple_cd.negotiated_adaptation import NegotiatedAdaptationMixin
from .simple_cd.goal_context import GoalContextMixin
from .simple_cd.social_events import SocialEventsMixin
from .simple_cd.interaction import InteractionMixin
from .simple_cd.navigation_pipeline import NavigationPipelineMixin
from .simple_cd.announcements import AnnouncementsMixin
from .simple_cd.supervision import SupervisionMixin


class SimpleCDCollaborationManager(
    NegotiatedAdaptationMixin,
    GoalContextMixin,
    SocialEventsMixin,
    InteractionMixin,
    NavigationPipelineMixin,
    AnnouncementsMixin,
    SupervisionMixin,
    CollaborationManager,
):
    """Route ROS 2 callbacks through the ordered policy mixins."""

    def __init__(self):
        self._scope = None
        self._handled = set()
        self._watermarks = {}
        self._goal_epoch = 0
        self._ready_goals = set()

        self._released = False

        self._control = None
        self._generation = None
        self._candidate = None
        self._validation = None
        self._early_validations = {}

        self._nav_phase = "IDLE"
        self._attempts = 0
        self._next_attempt = 0.0
        self._direction = "NONE"

        self._assisted = False
        self._assistance_id = ""
        self._idle_seen = set()

        self._denied_scope = None
        self._silence_pending = False
        self._silence_announcement_id = ""
        self._silence_announcement_started_at = None
        self._silence_announcement_fault = False
        
        # Directional semantics are scoped to one exact human dialogue turn.
        self._direction_semantic_candidate = None
        self._direction_semantic_ready = False

        # Exact closure and announcement state for a confirmed direction.
        self._directional_finalization = None
        self._directional_announcement_complete = False
        # Global social waiting interval, independent of HRI turns.
        self._social_wait_started_at = None
        self._social_wait_goal_id = ""
        self._social_wait_timeout_triggered = False
        
        self._refusal_restart_started_at = None
        self._refusal_restart_goal = ""
        # Unique refusal transaction, owned by this final node.
        self._refusal_transaction = None
        self._d_wait_started_at = None
        self._d_wait_goal = ""
        self._d_wait_recovery_last_at = 0.0
        self._d_wait_scope = None

        super().__init__()

        self.declare_parameter(
            "directional_robot_frame",
            "base_footprint",
        )

        self.directional_robot_frame = str(
            self.get_parameter(
                "directional_robot_frame"
            ).value
        ).strip()

        if not self.directional_robot_frame:
            raise ValueError(
                "directional_robot_frame must not be empty"
            )

        self.declare_parameter(
            "demo_robot_only_single_yes",
            False,
        )

        self.demo_robot_only_single_yes = bool(
            self.get_parameter(
                "demo_robot_only_single_yes"
            ).value
        )

        self.get_logger().info(
            "[HRI DEMO] "
            f"robot_only_single_yes="
            f"{self.demo_robot_only_single_yes}"
        )
        # The parent constructor has loaded the ROS interaction parameter.
        self._episode_mode = (
            "D" if self.interaction == "dialogue" else "C"
        )

        # Directional constraint interfaces for dialogue-based navigation.
        self._directional_request = None
        self._directional_status = None

        self.directional_constraint_pub = self.create_publisher(
            DirectionalConstraintRequest,
            "/social_navigation/directional_constraint_request",
            10,
        )
        self.directional_announcement_pub = self.create_publisher(
            ExplanationText,
            "/explainability/text",
            10,
        )

        self.directional_constraint_status_sub = self.create_subscription(
            DirectionalConstraintStatus,
            "/social_navigation/directional_constraint_status",
            self._directional_constraint_status_callback,
            10,
        )

    # ------------------------------------------------------
    # INITIALIZATION AND DIRECTIONAL STATUS
    # ------------------------------------------------------
    
    def _watch_mode_d_wait_180(self, now):
        """
        Observe all actual Mode-D waiting situations.

        The clock belongs to (public goal, social scope).
        Repeated WAIT, INSTALL and REVOKE messages never
        reset it.

        At 180 seconds, request recovery without inventing
        human consent or bypassing physical stop checks.

        The directional retry FSM remains the only owner
        of INSTALL and CLEAR transactions.
        """
        goal = str(
            self.active_navigation_goal_id or ""
        ).strip()

        scope = self._scope

        def reset_clock():
            self._d_wait_started_at = None
            self._d_wait_goal = ""
            self._d_wait_scope = None
            self._d_wait_recovery_last_at = 0.0

        # --------------------------------------------------
        # 1. CHECK CURRENT GOAL AND MODE
        # --------------------------------------------------

        if (
            not goal
            or not scope
            or self.pending_terminal_stop is not None
            or str(self._episode_mode).upper() != "D"
        ):
            reset_clock()
            return False

        navigation_running = bool(
            self.navigation_active
            and str(self.navigation_state).upper()
                == "NAVIGATING"
        )

        if navigation_running:
            reset_clock()
            return False

        # --------------------------------------------------
        # 2. IDENTIFY ACTUAL WAIT
        # --------------------------------------------------

        phase = str(
            self._nav_phase or ""
        ).strip().upper()

        control = self._control

        pending_wait = bool(
            isinstance(control, dict)
            and str(control.get("action") or "").upper()
                == "WAIT"
        )

        existing_social_wait = bool(
            self._social_wait_started_at is not None
            and self._social_wait_goal_id == goal
        )

        refusal = self._refusal_transaction

        active_refusal = bool(
            isinstance(refusal, dict)
            and refusal.get("goal") == goal
            and refusal.get("scope") == scope
        )

        recovery_phase = bool(
            phase in {
                "HELD",
                "DIRECT_NAV2_BLOCKED",
                "DIRECT_NAV2_WAIT_STOP",
                "DIRECT_NAV2_WAIT_MASK",
                "D_FINAL_WAIT",
                "SOCIAL_DENIED",
            }
            or phase.startswith("ADAPTATION_")
        )

        gate_waiting = (
            str(self.navigation_state or "").upper()
            in {"WAITING", "SOCIAL_HOLD"}
        )

        waiting = bool(
            existing_social_wait
            or active_refusal
            or pending_wait
            or recovery_phase
            or gate_waiting
        )

        if not waiting:
            reset_clock()
            return False

        # --------------------------------------------------
        # 3. START OR PRESERVE THE GLOBAL WAIT CLOCK
        # --------------------------------------------------

        if (
            self._d_wait_started_at is None
            or self._d_wait_goal != goal
            or self._d_wait_scope != scope
        ):
            started = now

            if existing_social_wait:
                started = min(
                    started,
                    self._social_wait_started_at,
                )

            if active_refusal:
                started = min(
                    started,
                    refusal["started_at"],
                )

            self._d_wait_started_at = started
            self._d_wait_goal = goal
            self._d_wait_scope = scope
            self._d_wait_recovery_last_at = 0.0

        elapsed = now - self._d_wait_started_at

        if elapsed < 180.0:
            return False

        # Avoid repeatedly triggering recovery every timer tick.
        if now - self._d_wait_recovery_last_at < 5.0:
            return False

        self._d_wait_recovery_last_at = now

        self.get_logger().warning(
            "[MODE D WAIT 180] Deadline reached | "
            f"goal={goal} | "
            f"elapsed={elapsed:.1f}s | "
            f"phase={phase} | "
            f"refusal={active_refusal}"
        )

        # --------------------------------------------------
        # 4. EXPLICIT REFUSAL
        # --------------------------------------------------

        # An existing refusal transaction owns its dialogue.
        # Never create another dialogue in parallel.
        if active_refusal:
            return self._watch_refusal_transaction(now)

        # --------------------------------------------------
        # 5. PHYSICAL STOP AND GATE BARRIER
        # --------------------------------------------------

        # Never authorize motion when the robot has not
        # been confirmed stopped or a Gate command is pending.
        if (
            not self.nav2_cancel_confirmed
            or not self._stopped()
            or self._control is not None
        ):
            self.get_logger().error(
                "[MODE D WAIT 180] Recovery blocked | "
                "physical_stop_or_gate_ack_unconfirmed=True"
            )
            return False

        # --------------------------------------------------
        # 6. ACTIVE C/D NEGOTIATED ADAPTATION
        # --------------------------------------------------

        recovery = self._adaptation

        if recovery is not None:
            adaptation_phase = str(
                recovery.get("phase") or ""
            ).upper()

            if adaptation_phase == "DENIED":
                if self._restart_adaptation_dialogue():
                    reset_clock()
                    return True

            elif adaptation_phase == "QUESTION":
                self._close_adaptation(
                    "TECHNICAL_FAILURE"
                )
                return True

            elif adaptation_phase == "STOP":
                if not recovery.get("held", False):
                    self._adaptation_wait()
                return True

            # Preserve the currently owned negotiation.
            # Do not turn confirmed NO into YES.
            # Do not override a C announcement or HRI closure.
            return False

        # --------------------------------------------------
        # 7. FINAL D FAILURE
        # --------------------------------------------------

        # A final failure cannot automatically start
        # another directional navigation attempt.
        if phase == "D_FINAL_WAIT":
            if self._restart_denied_dialogue():
                reset_clock()
                return True

            return False

        # --------------------------------------------------
        # 8. DIRECTIONAL INSTALL / CLEAR TRANSACTION
        # --------------------------------------------------

        tx = self._directional_attempt_state

        if isinstance(tx, dict):
            tx_phase = str(
                tx.get("phase") or ""
            ).upper()

            if tx_phase in {"INSTALL", "CLEAR"}:
                self.get_logger().warning(
                    "[MODE D WAIT 180] "
                    "Directional transaction still pending | "
                    f"phase={tx_phase} | "
                    f"attempt={tx.get('count')} | "
                    f"goal={goal} | "
                    "owner=DIRECTIONAL_RETRY_FSM | "
                    "goal_preserved=True"
                )

                # IMPORTANT CORRECTION:
                #
                # Do not invoke _drive_directional_attempts()
                # from the global watchdog.
                #
                # _on_timer() already owns the ordinary
                # directional retry supervision.
                #
                # Returning False allows _timer_callback()
                # to continue to _on_timer().
                #
                # Preserve:
                #   - the current INSTALL attempt;
                #   - the exact REVOKE transaction;
                #   - the existing attempt counter;
                #   - the same public navigation goal G.
                return False

            if tx_phase == "AWAIT_C_DECISION":
                if self._adaptation is None:
                    self._start_adaptation(
                        "TWO_DIRECTIONAL_INSTALLATIONS_FAILED"
                    )
                return True

            if tx_phase == "FINAL_WAIT":
                self._nav_phase = "D_FINAL_WAIT"
                return True

        # --------------------------------------------------
        # 9. EXISTING VALID D MASK
        # --------------------------------------------------

        # If D is still authorized and the exact mask is
        # ready, retry the normal handoff for the same G.
        if (
            self._direction in {"LEFT", "RIGHT"}
            and self._direct_directional_ready()
            and self._permission_applies_to_current_episode()
        ):
            self._released = True
            self._nav_phase = "DIRECT_NAV2_WAIT_MASK"

            self._try_direct_nav2_handoff()
            return True

        # --------------------------------------------------
        # 10. AUTHORIZED D BUT MASK NOT READY
        # --------------------------------------------------

        if self._direction in {"LEFT", "RIGHT"}:
            self._start_adaptation(
                "D_WAIT_EXCEEDED_180_SECONDS"
            )
            return True

        # --------------------------------------------------
        # 11. NO CONFIRMED DIRECTION
        # --------------------------------------------------

        # Preserve existing unresolved-HRI policy.
        # This is not an implicit human permission.
        if not getattr(
            self,
            "_social_wait_timeout_triggered",
            False,
        ):
            self._social_wait_timeout_triggered = True

            self._begin_autonomous_readaptation_announcement(
                "AUTONOMOUS_RESUME_AFTER_WAIT_TIMEOUT"
            )

            return True

        return False
    
    def _directional_constraint_status_callback(self, msg):
        """Retain status only for the currently owned constraint request."""
        current = self._directional_request

        if current is None:
            return

        if (
            str(msg.request_id) != str(current.request_id)
            or str(msg.navigation_goal_id)
                != str(current.navigation_goal_id)
            or str(msg.interaction_id)
                != str(current.interaction_id)
            or str(msg.person_id)
                != str(current.person_id)
            or str(msg.tracking_session_id)
                != str(current.tracking_session_id)
            or int(msg.identity_generation)
                != int(current.identity_generation)
            or str(msg.directional_constraint).upper()
                != str(current.directional_constraint).upper()
        ):
            return

        self._directional_status = msg

        if msg.state == "FAILED":
            self.get_logger().error(
                "[DIRECTIONAL KEEPOUT] Constraint failed | "
                f"goal={msg.navigation_goal_id} | "
                f"direction={msg.directional_constraint} | "
                f"reason={msg.reason}"
            )

    def _initialize_social_reassessment(self):
        """Initialize inherited reassessment without anticipatory planning."""
        super()._initialize_social_reassessment()
        self.anticipatory_planning_enabled = False

    # ------------------------------------------------------
    # SOCIAL CALLBACKS
    # ------------------------------------------------------

    def _individual_situation_callback(self, msg):
        """Prevent individual observations from opening social episodes."""
        return

    def _social_episode_retirement_callback(self, msg):
        """Prevent retirement events from reopening social episodes."""
        return

    def _social_decision_callback(self, msg):
        """Route social decisions to the social event policy."""
        return self._on_social_decision(msg)

    # ------------------------------------------------------
    # NAVIGATION CALLBACKS
    # ------------------------------------------------------

    def _navigation_goal_context_callback(self, msg):
        """Route public goal context to the goal context policy."""
        return self._on_navigation_goal_context(msg)

    def _navigation_state_callback(self, msg):
        """Route navigation state updates to the goal context policy."""
        return self._on_navigation_state(msg)

    def _gate_command_result_callback(self, msg):
        """Route Gate results through the ordered policy mixins."""
        return self._on_gate_command_result(msg)

    # ------------------------------------------------------
    # TRAJECTORY CALLBACKS
    # ------------------------------------------------------

    def _candidate_trajectory_callback(self, msg):
        """Route candidate trajectories to the navigation pipeline."""
        return self._on_candidate_trajectory(msg)

    def _trajectory_generation_result_callback(self, msg):
        """Route trajectory generation results to the navigation pipeline."""
        return self._on_trajectory_generation_result(msg)

    def _trajectory_validation_callback(self, msg):
        """Route trajectory validation results to the navigation pipeline."""
        return self._on_trajectory_validation(msg)

    # ------------------------------------------------------
    # HRI AND AUDIO CALLBACKS
    # ------------------------------------------------------

    def _hri_dialogue_control_result_callback(self, msg):
        """Dispatch closure results through the ordered mixin chain."""
        return super()._hri_dialogue_control_result_callback(msg)

    def _speech_status_callback(self, msg):
        """Dispatch speech completion to the owning transaction."""
        decision_id = str(
            msg.decision_id or ""
        ).strip()

        state = str(
            msg.state or ""
        ).strip().upper()

        if decision_id.startswith("PRE_GOAL_GREETING_"):
            return

        # Confirmed directional announcement.
        # This handler performs the exact confirmation handoff.
        if self._complete_directional_final_speech(msg):
            return

        # Autonomous recovery announcement.
        resume_id = str(
            getattr(
                self,
                "_silence_announcement_id",
                "",
            ) or ""
        ).strip()

        if resume_id and decision_id == resume_id:
            if state == "COMPLETED":
                self._complete_silence_announcement()

            elif state in {"FAILED", "CANCELED"}:
                self._silence_announcement_fault = True
                self._complete_silence_announcement()

            # Intermediate states are part of this transaction.
            return

        # Other speech transactions, including negotiated adaptation.
        return self._on_speech_status(msg)

    # ------------------------------------------------------
    # TIMER
    # ------------------------------------------------------
    def _on_gate_command_result(self, msg):
        """
        Handle the first genuine Nav2 ABORTED for an initial
        autonomous goal.

        All ordinary C/D results remain owned by the existing
        NegotiatedAdaptationMixin and NavigationPipelineMixin.
        """
        data = self._json(
            msg,
            "navigation_goal_gate",
        )

        if data is None:
            return super()._on_gate_command_result(msg)

        goal_id = str(
            data.get("navigation_goal_id") or ""
        ).strip()

        request_id = str(
            data.get("request_id") or ""
        ).strip()

        last_start = getattr(
            self,
            "_nav2_last_start",
            None,
        )

        initial_abort = bool(
            str(data.get("action") or "").upper() == "START"
            and str(data.get("status") or "").upper()
                == "EXECUTION_FAILED"
            and str(data.get("reason") or "").upper()
                == "NAV2_ABORT_RETRY_AVAILABLE"
            and isinstance(last_start, dict)
            and last_start.get("mode") == "INITIAL"
            and last_start.get("goal") == goal_id
            and last_start.get("request_id") == request_id
            and last_start.get("scope") is None
            and self._scope is None
            and goal_id
            and goal_id == self.active_navigation_goal_id
            and self.active_navigation_goal_pose is not None
            and self.pending_terminal_stop is None
        )

        if not initial_abort:
            return super()._on_gate_command_result(msg)

        used = getattr(
            self,
            "_nav2_retried_goals",
            None,
        )

        if used is None:
            self._nav2_retried_goals = set()
            used = self._nav2_retried_goals

        if goal_id in used:
            return

        if getattr(self, "_nav2_same_goal_retry", None) is not None:
            return

        used.add(goal_id)

        now = time.monotonic()

        self._nav2_same_goal_retry = {
            "goal": goal_id,
            "scope": None,
            "mode": "INITIAL",
            "human_direction": "NONE",
            "robot_direction": "NONE",
            "wait_id": "",
            "held": False,
            "last_sent_at": now,
            "started_at": now,
        }

        self._control = None

        hold_id = self._command(
            "WAIT",
            "NAV2_SAME_GOAL_RETRY_HOLD",
        )

        self._nav2_same_goal_retry["wait_id"] = (
            hold_id or ""
        )

        self.get_logger().warning(
            "[NAV2 INITIAL RETRY] First internal abort | "
            f"goal={goal_id} | "
            "public_goal_retained=True | "
            "new_dialogue=False | "
            "speech=False"
        )

    def _on_timer(self):
        """
        Retry the initial autonomous goal after a confirmed hold.

        Never create a second public goal and never bypass
        the Gate's physical-stop or WAIT acknowledgment.
        """
        retry = getattr(
            self,
            "_nav2_same_goal_retry",
            None,
        )

        if (
            retry is None
            or retry.get("mode") != "INITIAL"
        ):
            return super()._on_timer()

        now = time.monotonic()
        goal_id = retry["goal"]

        if (
            goal_id != self.active_navigation_goal_id
            or self._scope is not None
            or self.pending_terminal_stop is not None
            or self.active_navigation_goal_pose is None
        ):
            self._nav2_same_goal_retry = None
            return

        if not retry["held"]:
            if self._control is not None:
                return

            if now - retry["last_sent_at"] >= 5.0:
                retry["wait_id"] = (
                    self._command(
                        "WAIT",
                        "NAV2_SAME_GOAL_RETRY_HOLD",
                    ) or ""
                )

                retry["last_sent_at"] = now

            return

        if (
            not self.nav2_cancel_confirmed
            or not self._stopped()
            or self._control is not None
            or self.gate_ready_goal_id != goal_id
        ):
            return

        # The Gate expects a fresh autonomous START after
        # the confirmed WAIT. This is an internal retry
        # of the same public destination.
        request_id = self._publish_collaborative_decision(
            action="START",
            outcome="NAVIGATION_TRANSACTION",
            human_intent="UNKNOWN",
            directional_constraint="NONE",
            path_id="",
            reason="AUTONOMOUS_MODE_C_NAV2_HANDOFF",
        )

        if not request_id:
            return

        self._control = {
            "action": "START",
            "request_id": request_id,
            "goal": goal_id,
            "path": "",
            "sent": now,
        }

        self._nav2_last_start = {
            "goal": goal_id,
            "request_id": request_id,
            "mode": "C",
            "scope": None,
            "robot_direction": "NONE",
            "human_direction": "NONE",
        }

        self._adaptation_start_id = request_id
        self._nav_phase = "DIRECT_NAV2_START_PENDING"
        self._execution_sent_at = now
        self._nav2_same_goal_retry = None

        self.get_logger().info(
            "[NAV2 INITIAL RETRY] Fresh internal START requested | "
            f"goal={goal_id} | "
            "public_goal_unchanged=True | "
            "human_dialogue=False"
        )
    
    def _watch_refusal_transaction(self, now):
        """
        Own the complete refusal WAIT transaction.

        Return True while this watchdog owns the timer,
        so no other mixin may start navigation or launch
        a competing dialogue.

        Once the configured deadline expires:
        - preserve public G;
        - verify physical hold;
        - wait for exact HRI closure;
        - clear only the previous dialogue permission;
        - open a fresh human initiative window.

        Never turn the elapsed time into consent.
        """
        tx = self._refusal_transaction

        if tx is None:
            return False

        goal = str(
            self.active_navigation_goal_id or ""
        ).strip()

        if (
            not goal
            or goal != tx["goal"]
            or self.pending_terminal_stop is not None
        ):
            self.get_logger().info(
                "[REFUSAL] Transaction ended | "
                "reason=GOAL_CHANGED_OR_TERMINATED"
            )

            self._refusal_transaction = None
            return False

        # Keep physical and HRI close supervision alive.
        self._evaluate_stop_barrier()
        self._watch_hri_close(now)
        self._watch_terminal_stop(now)

        if now < tx["deadline"]:
            return True

        elapsed = now - tx["started_at"]

        # Physical hold must remain confirmed.
        if (
            not self.nav2_cancel_confirmed
            or not self._stopped()
            or self._control is not None
        ):
            self.get_logger().warning(
                "[REFUSAL] RESTART_BLOCKED_STOP | "
                f"goal={goal} | elapsed={elapsed:.1f}s",
                throttle_duration_sec=10.0,
            )
            return True

        # Do not open a new HRI turn until the previous
        # one has been closed through its actual protocol.
        if (
            self.pending_hri_close is not None
            or self.pending_dialogue_request_id
            or getattr(
                self,
                "_hri_detached_close",
                None,
            ) is not None
            or self.dialogue_audio_recovery_pending
        ):
            self.get_logger().warning(
                "[REFUSAL] RESTART_BLOCKED_HRI | "
                f"goal={goal} | "
                f"pending_close={self.pending_hri_close is not None} | "
                f"pending_turn={bool(self.pending_dialogue_request_id)} | "
                f"elapsed={elapsed:.1f}s",
                throttle_duration_sec=10.0,
            )
            return True

        # Preserve any exact D-mask revocation contract.
        if not self._revoke_optional_directional_mask(
            "REFUSAL_DIALOGUE_RESTART"
        ):
            self.get_logger().warning(
                "[REFUSAL] RESTART_BLOCKED_MASK | "
                f"goal={goal} | elapsed={elapsed:.1f}s",
                throttle_duration_sec=10.0,
            )
            return True

        if not self._scope:
            self.get_logger().warning(
                "[REFUSAL] RESTART_BLOCKED_SOCIAL_CONTEXT | "
                f"goal={goal} | elapsed={elapsed:.1f}s",
                throttle_duration_sec=10.0,
            )
            return True

        # A previous adaptation may still own the
        # denied transaction. Do not leave two owners.
        recovery = getattr(
            self,
            "_adaptation",
            None,
        )

        if recovery is not None:
            if recovery.get("phase") != "DENIED":
                self.get_logger().warning(
                    "[REFUSAL] RESTART_BLOCKED_ADAPTATION | "
                    f"phase={recovery.get('phase')}",
                    throttle_duration_sec=10.0,
                )
                return True

            self._adaptation = None

        # The old refusal is finished. It cannot be
        # carried forward as consent or as a new refusal.
        self._denied_scope = None

        self._handled.discard(
            tx["scope"]
        )

        self.authorized_episode_key = None
        self._retained_same_person_consent = None

        self.confirmed_human_intent = "UNKNOWN"
        self.confirmed_human_direction = "NONE"
        self.pending_human_intent = "UNKNOWN"
        self.pending_human_direction = "NONE"

        self.dialogue_outcome = "NONE"
        self.dialogue_resolution_complete = False
        self.dialogue_retry_count = 0
        self.dialogue_speech_retry_count = 0
        self.dialogue_started_at = None

        self._direction_semantic_candidate = None
        self._direction_semantic_ready = False
        self._directional_finalization = None
        self._directional_announcement_complete = False

        self._hri_retry_after_close = None
        self._hri_retry_notice = "MISSED"
        self._hri_nav_fallback_active = False

        self._episode_mode = "D"
        self._direction = "NONE"
        self._released = False
        self._nav_phase = "HELD"

        # Prevent the old social wait timer from
        # firing a second unrelated recovery.
        self._social_wait_started_at = None
        self._social_wait_goal_id = ""
        self._social_wait_timeout_triggered = False

        # Consume the old deadline exactly once.
        self._refusal_transaction = None

        self.get_logger().info(
            "[REFUSAL] WAIT_COMPLETED | "
            f"goal={goal} | "
            f"elapsed={elapsed:.1f}s | "
            "next=FRESH_HUMAN_INITIATIVE | "
            "nav2_start=False"
        )

        self._start_human_initiative_window()

        return True
    
    def _announce_refusal_wait(self):
        """
        Acknowledge the NO and explain the configured WAIT.

        The result of speech synthesis is deliberately
        NOT an authorization gate and does not affect
        the refusal deadline.
        """
        import uuid

        tx = self._refusal_transaction

        if tx is None or tx["announced"]:
            return

        # Reserve before publishing to avoid duplicate speech.
        tx["announced"] = True

        duration = tx["duration"]

        if duration >= 60.0:
            minutes = duration / 60.0

            if minutes.is_integer():
                en_duration = (
                    f"{int(minutes)} minutes"
                )
                fr_duration = (
                    f"{int(minutes)} minutes"
                )
                es_duration = (
                    f"{int(minutes)} minutos"
                )
            else:
                seconds = int(round(duration))
                en_duration = f"{seconds} seconds"
                fr_duration = f"{seconds} secondes"
                es_duration = f"{seconds} segundos"
        else:
            seconds = int(round(duration))
            en_duration = f"{seconds} seconds"
            fr_duration = f"{seconds} secondes"
            es_duration = f"{seconds} segundos"

        texts = {
            "en": (
                "I understand your instruction. "
                f"I will remain stopped for {en_duration}, "
                "then ask you again before continuing."
            ),
            "fr": (
                "J'ai compris votre instruction. "
                f"Je vais rester à l'arrêt pendant {fr_duration}, "
                "puis vous demander à nouveau avant de continuer."
            ),
            "es": (
                "He entendido su instrucción. "
                f"Permaneceré detenido durante {es_duration} "
                "y después volveré a preguntarle antes de continuar."
            ),
        }

        language = str(
            self.language or "en"
        ).strip().lower()

        if language not in texts:
            language = "en"

        msg = ExplanationText()

        msg.header.stamp = (
            self.get_clock().now().to_msg()
        )
        msg.context_stamp = msg.header.stamp

        msg.decision_id = (
            "REFUSAL_WAIT_" + uuid.uuid4().hex
        )

        msg.language = language
        msg.text = texts[language]

        msg.fallback_language = "en"
        msg.fallback_text = texts["en"]

        msg.provider = "g1_robot_collaboration"
        msg.model = ""
        msg.used_fallback = False
        msg.used_cache = False

        msg.speech_kind = "DIALOGUE"

        msg.navigation_goal_id = tx["goal"]
        msg.navigation_state = self.navigation_state
        msg.navigation_active = False
        msg.social_conflict_active = True

        try:
            self.directional_announcement_pub.publish(
                msg
            )

        except Exception as exc:
            self.get_logger().error(
                "[REFUSAL] Acknowledgement publication failed | "
                f"error={exc} | "
                "wait_timer_continues=True"
            )

    def _hold_after_denial(self):
        """
        Register an explicit human NO.

        Start exactly one deadline for this refusal.
        Keep the public goal G and the robot in WAIT.
        A spoken acknowledgement is informational.
        """
        super()._hold_after_denial()

        goal = str(
            self.active_navigation_goal_id or ""
        ).strip()

        if not goal or not self._scope:
            return

        previous = self._refusal_transaction

        if (
            previous is not None
            and previous["goal"] == goal
            and previous["scope"] == self._scope
        ):
            return

        duration = float(
            self.social_resolution_timeout_sec
        )

        now = time.monotonic()

        self._refusal_transaction = {
            "goal": goal,
            "scope": self._scope,
            "started_at": now,
            "deadline": now + duration,
            "duration": duration,
            "announced": False,
            "restarting": False,
        }

        self._released = False
        self._episode_mode = "D"
        self._nav_phase = "SOCIAL_DENIED"

        self.get_logger().info(
            "[REFUSAL] CONFIRMED | "
            f"goal={goal} | "
            f"duration={duration:.1f}s | "
            "next=ANNOUNCE_THEN_FRESH_DIALOGUE"
        )

        self._announce_refusal_wait()

    def _watch_refusal_restart(self):
        """
        Independent refusal watchdog owned by the final node.

        Returns True when it has started a new HRI dialogue.
        Returns False otherwise.

        Never starts navigation by itself.
        """
        started = getattr(
            self, "_refusal_restart_started_at", None
        )

        if started is None:
            return False

        goal = str(
            self.active_navigation_goal_id or ""
        ).strip()

        if (
            not goal
            or goal != getattr(
                self, "_refusal_restart_goal", ""
            )
            or self.pending_terminal_stop is not None
        ):
            self._refusal_restart_started_at = None
            self._refusal_restart_goal = ""
            return False

        elapsed = time.monotonic() - started

        if elapsed < 180.0:
            return False

        # The deadline is reached. Do not reset it while
        # waiting for a prerequisite.
        if not self._scope:
            self.get_logger().warning(
                "[REFUSAL RESTART] BLOCKED_NO_SOCIAL_SCOPE | "
                f"goal={goal} | elapsed={elapsed:.1f}s",
                throttle_duration_sec=10.0,
            )
            return False

        if (
            not self.nav2_cancel_confirmed
            or not self._stopped()
            or self._control is not None
        ):
            self.get_logger().warning(
                "[REFUSAL RESTART] BLOCKED_STOP_BARRIER | "
                f"goal={goal} | elapsed={elapsed:.1f}s",
                throttle_duration_sec=10.0,
            )
            return False

        if (
            self.pending_hri_close is not None
            or self.pending_dialogue_request_id
            or getattr(
                self, "_hri_detached_close", None
            ) is not None
            or self.dialogue_audio_recovery_pending
        ):
            self.get_logger().warning(
                "[REFUSAL RESTART] BLOCKED_HRI_TRANSACTION | "
                f"goal={goal} | elapsed={elapsed:.1f}s",
                throttle_duration_sec=10.0,
            )
            return False

        if not self._revoke_optional_directional_mask(
            "REFUSAL_DIALOGUE_RESTART"
        ):
            self.get_logger().warning(
                "[REFUSAL RESTART] BLOCKED_MASK_REVOCATION | "
                f"goal={goal} | elapsed={elapsed:.1f}s",
                throttle_duration_sec=10.0,
            )
            return False

        # If an old adaptation owns the refusal, close
        # that finished transaction before opening another.
        recovery = getattr(self, "_adaptation", None)

        if recovery is not None:
            if recovery.get("phase") != "DENIED":
                self.get_logger().warning(
                    "[REFUSAL RESTART] "
                    "BLOCKED_ACTIVE_ADAPTATION"
                )
                return False

            self._adaptation = None

        # Clear only the old negotiation, never G
        # or the real physical stop evidence.
        self._denied_scope = None
        self._handled.discard(self._scope)

        self.confirmed_human_intent = "UNKNOWN"
        self.confirmed_human_direction = "NONE"
        self.pending_human_intent = "UNKNOWN"
        self.pending_human_direction = "NONE"

        self.authorized_episode_key = None
        self._retained_same_person_consent = None

        self.dialogue_outcome = "NONE"
        self.dialogue_resolution_complete = False
        self.dialogue_retry_count = 0
        self.dialogue_speech_retry_count = 0
        self.dialogue_started_at = None

        self._direction_semantic_candidate = None
        self._direction_semantic_ready = False
        self._directional_finalization = None
        self._directional_announcement_complete = False

        self._hri_retry_after_close = None
        self._hri_retry_notice = "MISSED"
        self._hri_nav_fallback_active = False

        self._episode_mode = "D"
        self._direction = "NONE"
        self._released = False
        self._nav_phase = "HELD"

        # Disable the older, independent social timeout
        # for this completed refusal transaction.
        self._social_wait_started_at = None
        self._social_wait_goal_id = ""
        self._social_wait_timeout_triggered = False

        # Consume the old deadline before publishing
        # a new interaction.
        self._refusal_restart_started_at = None
        self._refusal_restart_goal = ""

        self.get_logger().info(
            "[REFUSAL RESTART] Deadline satisfied | "
            f"goal={goal} | elapsed={elapsed:.1f}s | "
            "action=FRESH_HUMAN_INITIATIVE | "
            "nav2_start=False"
        )

        self._start_human_initiative_window()

        return True

    def _timer_callback(self):
        """Own all global D waiting deadlines before mixin timers."""
        now = time.monotonic()

        # Applies to every other D WAIT, not only DENIED.
        if self._watch_mode_d_wait_180(now):
            return
        
        # Explicit refusal retains its existing HRI recovery.
        if self._watch_refusal_transaction(now):
            return

        return self._on_timer()
    
    # ------------------------------------------------------
    # INHERITED ROS BEHAVIOR
    # ------------------------------------------------------

    def _publish_wait(self, reason):
        """
        Publish WAIT and retain the start of the global waiting interval.

        Repeated WAIT messages and HRI question retries do not
        restart the 180-second clock.
        """
        if self._released:
            return ""

        result = super()._publish_wait(reason)
        now = time.monotonic()

        self._last_social_wait_at = now

        goal_id = str(
            self.active_navigation_goal_id or ""
        ).strip()

        if (
            result
            and goal_id
            and self._scope
            and self.pending_terminal_stop is None
        ):
            if (
                self._social_wait_started_at is None
                or self._social_wait_goal_id != goal_id
            ):
                self._social_wait_started_at = now
                self._social_wait_goal_id = goal_id
                self._social_wait_timeout_triggered = False

                self.get_logger().info(
                    "[SOCIAL WAIT] Global timer started | "
                    f"goal={goal_id} | "
                    f"timeout={self.social_resolution_timeout_sec:.1f}s"
                )

        return result
    
    def _enter_replanning(
        self,
        reason,
        directional_constraint=None,
    ):
        """Disable the inherited social replanning entry point."""
        return

    # ------------------------------------------------------
    # ROS INTERFACES USED BY POLICY MODULES
    # ------------------------------------------------------

    def _ros_publish_dialogue_request(self, request):
        """Publish a dialogue request."""
        self.dialogue_request_pub.publish(request)

    def _ros_publish_trajectory_request(self, request):
        """Publish a trajectory generation request."""
        self.trajectory_generation_request_pub.publish(request)

    def _ros_publish_explanation(self, context):
        """Publish an explanation context."""
        self.explanation_context_pub.publish(context)

    def _ros_cancel_speech(self, decision_id):
        """Request cancellation of the identified speech transaction."""
        self.speech_cancel_pub.publish(
            SpeechCancelRequest(decision_id=decision_id)
        )

    def _ros_query_generation_status(self, job):
        """Query the status of an exact trajectory generation transaction."""
        self.generation_status_request_pub.publish(
            String(
                data=json.dumps({
                    "version": 1,
                    "navigation_goal_id": job["goal"],
                    "request_id": job["id"],
                })
            )
        )