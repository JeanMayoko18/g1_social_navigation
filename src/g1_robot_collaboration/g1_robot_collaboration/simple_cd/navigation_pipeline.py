
"""Gate transactions and direct Nav2 navigation for Modes C and D.

Mode C:
    Autonomous NavigateToPose with direction NONE.

Mode D:
    NavigateToPose with an exact, previously confirmed LEFT/RIGHT
    permission and an operational directional Nav2 mask.

Nav2 owns planning, obstacle avoidance and path replanning.

This mixin never authorizes CandidateTrajectory or FollowPath.
"""

import json
import time

from g1_social_common.collaboration_states import (
    STATE_NAVIGATING,
)


class NavigationPipelineMixin:
    """Coordinate pathless Nav2 handoff and Gate acknowledgments."""

    # --------------------------------------------------
    # JSON MESSAGE VALIDATION
    # --------------------------------------------------

    @staticmethod
    def _json(msg, producer):
        """Decode a versioned JSON message from an expected producer."""
        try:
            data = json.loads(msg.data)

            if (
                isinstance(data, dict)
                and data.get("version") == 1
                and data.get("producer") == producer
            ):
                return data

        except (ValueError, TypeError, AttributeError):
            pass

        return None

    # --------------------------------------------------
    # GATE COMMANDS
    # --------------------------------------------------

    def _command(self, action, reason, path_id=""):
        """Build one consistent Gate command contract."""
        action = str(action or "").strip().upper()
        reason = str(reason or "").strip().upper()
        path_id = str(path_id or "").strip()

        if action == "RESUME" or path_id:
            self.get_logger().error(
                "[NAV PIPELINE] Path-based command rejected | "
                f"action={action} | path={path_id}"
            )
            return ""

        if action not in {"START", "WAIT", "REPLAN", "STOP"}:
            return ""

        mode = str(
            getattr(self, "_episode_mode", "") or ""
        ).strip().upper()

        direction = str(
            getattr(self, "_direction", "NONE") or "NONE"
        ).strip().upper()

        if direction not in {"NONE", "LEFT", "RIGHT"}:
            return ""

        gate_intent = str(
            self.confirmed_human_intent or "UNKNOWN"
        ).strip().upper()

        gate_direction = direction

        if action == "START":
            if not self._scope:
                if reason != "INITIAL_NAVIGATION_NO_SOCIAL_CONFLICT":
                    return ""

                gate_intent = "UNKNOWN"
                gate_direction = "NONE"

            elif mode == "C":
                if reason != "AUTONOMOUS_MODE_C_NAV2_HANDOFF":
                    return ""

                gate_intent = "UNKNOWN"
                gate_direction = "NONE"

            elif mode == "D":
                if reason != "CONFIRMED_PASSAGE_NAV2_HANDOFF":
                    return ""

                if direction not in {"LEFT", "RIGHT"}:
                    return ""

                if not (
                    self.confirmed_human_intent == "ALLOW_PASSAGE"
                    and self.dialogue_outcome == "ALLOW_CONFIRMED"
                    and self._permission_applies_to_current_episode()
                ):
                    return ""

                gate_intent = "ALLOW_PASSAGE"
                gate_direction = direction

            else:
                return ""

        request_id = self._publish_collaborative_decision(
            action=action,
            outcome="NAVIGATION_TRANSACTION",
            human_intent=gate_intent,
            directional_constraint=gate_direction,
            path_id="",
            reason=reason,
        )

        if request_id:
            self._control = {
                "action": action,
                "request_id": request_id,
                "goal": self.active_navigation_goal_id,
                "path": "",
                "sent": time.monotonic(),
            }

        return request_id
    
    # --------------------------------------------------
    # GATE ACKNOWLEDGMENTS
    # --------------------------------------------------


    def _on_gate_command_result(self, msg):
        """Reconcile exact Gate commands and confirmed social holds."""
        data = self._json(msg, "navigation_goal_gate")

        if data is None:
            return

        goal_id = str(
            data.get("navigation_goal_id") or ""
        ).strip()

        action = str(
            data.get("action") or ""
        ).strip().upper()

        status = str(
            data.get("status") or ""
        ).strip().upper()

        request_id = str(
            data.get("request_id") or ""
        ).strip()

        if (
            action == "GOAL_READY"
            and status == "READY"
            and goal_id
        ):
            self._ready_goals.add(goal_id)

        if goal_id != self.active_navigation_goal_id:
            return

        if action == "GOAL_READY" and status == "READY":
            self.gate_ready_goal_id = goal_id
            return

        pending = self._control

        matches_pending = bool(
            pending is not None
            and pending["goal"] == goal_id
            and pending["action"] == action
            and pending["request_id"] == request_id
        )

        # The initial social STOP barrier has its own request ID.
        # A confirmed Gate hold must release the matching local
        # transaction; otherwise direct handoff stays blocked.
        social_hold_confirmed = bool(
            not self._released
            and action == "WAIT"
            and status == "HOLD_CONFIRMED"
            and request_id
            and request_id
                == self.pending_stop_hold_request_id
        )

        if social_hold_confirmed:
            self.nav2_cancel_confirmed = True

            if matches_pending:
                self._control = None

            self._evaluate_stop_barrier()
            return

        if not matches_pending:
            return

        if status in {"REJECTED", "EXPIRED"}:
            self._control = None

            failure = (
                "GATE_"
                + str(
                    data.get("reason") or status
                ).strip().upper()
            )

            if self._nav_phase.startswith("DIRECT_NAV2_"):
                self._fail_direct_nav2_handoff(failure)
                return

            if action in {"WAIT", "REPLAN", "STOP"}:
                self.get_logger().error(
                    "[GOAL GATE] Control rejected | "
                    f"action={action} | reason={failure}"
                )
                return

            self._retry(failure)
            return

        if action == "WAIT":
            if status != "HOLD_CONFIRMED":
                return

            self._control = None
            self.nav2_cancel_confirmed = True

            if self._nav_phase == "DIRECT_NAV2_BLOCKED":
                return

            if self._nav_phase in {
                "DIRECT_NAV2_WAIT_MASK",
                "DIRECT_NAV2_WAIT_STOP",
            }:
                return

            if not self._released:
                self._nav_phase = "HELD"
                self._evaluate_stop_barrier()
                return

            # A released episode must not return to the
            # legacy HELD -> GENERATING pipeline.
            self._nav_phase = "DIRECT_NAV2_BLOCKED"
            return

        if action == "REPLAN":
            if status != "HOLD_CONFIRMED":
                return

            self._control = None
            self.nav2_cancel_confirmed = True

            if self._released:
                self._fail_direct_nav2_handoff(
                    "UNEXPECTED_REPLAN_TRANSACTION"
                )
            else:
                self._nav_phase = "HELD"

            return

        if action == "START":
            if status != "ACCEPTED":
                return

            self._control = None

            self.initial_start_authorized_goal_id = goal_id
            self._execution_sent_at = time.monotonic()

            if (
                self._nav_phase
                == "DIRECT_NAV2_START_PENDING"
            ):
                self._nav_phase = "DIRECT_NAV2_STARTING"
            else:
                self._nav_phase = "STARTING"

            self.get_logger().info(
                "[NAV2 HANDOFF] START accepted | "
                f"goal={goal_id} | "
                "physical_execution_confirmed=False"
            )
            return

        if action == "STOP" and status == "ACCEPTED":
            self._control = None
            return

        # Candidate-based RESUME is never an execution action.

    # --------------------------------------------------
    # PHYSICAL STOP VERIFICATION
    # --------------------------------------------------

    def _stopped(self):
        """Require fresh physical and Gate navigation-stop evidence."""
        now = time.monotonic()
        nav = self.latest_navigation_state

        return bool(
            nav is not None
            and nav.navigation_goal_id
                == self.active_navigation_goal_id
            and self._message_is_recent(nav, 1.0)
            and not self.navigation_active
            and self.stable_stop_confirmed
            and self.motion_state_received_at is not None
            and 0.0
                <= now - self.motion_state_received_at
                <= 0.5
            and self.motion_state_stamp_sec is not None
            and -0.1 <= (
                self.get_clock().now().nanoseconds / 1e9
                - self.motion_state_stamp_sec
            ) <= 0.5
        )

    # --------------------------------------------------
    # LEGACY TRAJECTORY INTERFACES
    # --------------------------------------------------

    def _generate(self):
        """Disable candidate generation in direct-navigation Modes C/D."""
        self.get_logger().warning(
            "[NAV PIPELINE] Legacy candidate generation disabled",
            throttle_duration_sec=5.0,
        )
        return False

    def _on_candidate_trajectory(self, msg):
        """Ignore candidate paths: Nav2 independently plans its routes."""
        return

    def _on_trajectory_generation_result(self, msg):
        """Ignore legacy planner results without granting execution."""
        return

    def _consume_generation(self):
        """Never convert a candidate into an executable authorization."""
        return False

    def _on_trajectory_validation(self, msg):
        """Ignore validations for the disabled candidate pipeline."""
        return

    def _trajectory_execution_allowed(self, path_id):
        """Never authorize execution of an external candidate path."""
        return False

    def _try_resume(self):
        """Disable RESUME(path_id) and FollowPath transactions."""
        return False

    # --------------------------------------------------
    # NAVIGATION RETRY
    # --------------------------------------------------

    def _retry(self, reason):
        """Keep direct-navigation failures held without entering BACKOFF."""
        self._candidate = None
        self._validation = None
        self._candidate_direction = ""
        self._candidate_scope = None
        self._early_validations.clear()

        if (
            self.pending_terminal_stop is not None
            or not self.active_navigation_goal_id
        ):
            return False

        if not self._released:
            # An unresolved social interaction retains
            # the navigation hold.
            self.get_logger().warning(
                "[NAV PIPELINE] Retry deferred | "
                f"reason={reason} | released=False"
            )
            return False

        self._fail_direct_nav2_handoff(
            "RETRY_REQUESTED_" + str(reason or "UNKNOWN")
        )

        return False

    # --------------------------------------------------
    # DIRECT NAV2 HANDOFF PREPARATION
    # --------------------------------------------------

    def _begin_direct_nav2_handoff(self, direction):
        """Prepare a pathless handoff using robot-relative direction."""
        direction = str(direction or "").strip().upper()

        mode = str(
            getattr(self, "_episode_mode", "") or ""
        ).strip().upper()

        if mode not in {"C", "D"}:
            return False

        if direction not in {"NONE", "LEFT", "RIGHT"}:
            return False

        if (
            not self._scope
            or not self.active_navigation_goal_id
            or self.active_navigation_goal_pose is None
            or self.pending_terminal_stop is not None
            or self._denied_scope
            or self.dialogue_outcome == "DENY_CONFIRMED"
            or self._control is not None
        ):
            return False

        if mode == "C":
            if direction != "NONE":
                return False

        else:
            finalization = getattr(
                self,
                "_directional_finalization",
                None,
            )

            candidate = (
                finalization.get("candidate")
                if isinstance(finalization, dict)
                else None
            )

            if not isinstance(candidate, dict):
                return False

            requested = str(
                candidate.get("direction") or ""
            ).strip().upper()

            resolved = str(
                candidate.get("resolved_robot_direction") or ""
            ).strip().upper()

            if not (
                direction in {"LEFT", "RIGHT"}
                and resolved == direction
                and requested == self._normalize_direction(
                    self.confirmed_human_direction
                )
                and self.confirmed_human_intent == "ALLOW_PASSAGE"
                and self.dialogue_outcome == "ALLOW_CONFIRMED"
                and self._permission_applies_to_current_episode()
                and finalization.get("phase") == "ANNOUNCED"
                and finalization.get("scope") == self._scope
                and candidate.get("scope") == self._scope
                and candidate.get(
                    "final_confirmation_received"
                ) is True
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
                return False

        if (
            self._generation is not None
            or getattr(
                self,
                "uncertain_generation_request_id",
                "",
            )
            or getattr(
                self,
                "pending_generation_request_id",
                "",
            )
        ):
            self.get_logger().error(
                "[DIRECT NAV2] Legacy generation still owns "
                "a pending transaction"
            )
            return False

        self._direction = direction

        self._nav_phase = (
            "DIRECT_NAV2_WAIT_MASK"
            if mode == "D"
            else "DIRECT_NAV2_WAIT_STOP"
        )

        self._candidate = None
        self._validation = None
        self._candidate_direction = ""
        self._candidate_scope = None
        self._early_validations.clear()

        self._released = True

        self.get_logger().info(
            "[DIRECT NAV2] Handoff prepared | "
            f"goal={self.active_navigation_goal_id} | "
            f"mode={mode} | "
            f"robot_direction={direction}"
        )

        return True
    
    # --------------------------------------------------
    # DIRECTIONAL AUTHORIZATION
    # --------------------------------------------------

    def _direct_directional_ready(self):
        """Check availability of the exact optional published mask.

        Publication does not prove costmap consumption.
        This predicate never grants navigation authorization.
        """
        request = getattr(self, "_directional_request", None)
        status = getattr(self, "_directional_status", None)
        scope = self._social_scope_key()
        direction = str(self._direction or "").strip().upper()

        if (
            direction not in {"LEFT", "RIGHT"}
            or request is None
            or status is None
            or not scope
            or len(scope) != 5
            or not self._permission_applies_to_current_episode()
            or self._normalize_direction(self.confirmed_human_direction)
                != direction
        ):
            return False

        try:
            request_identity = (
                str(request.request_id),
                str(request.navigation_goal_id),
                str(request.interaction_id),
                str(request.person_id),
                str(request.tracking_session_id),
                int(request.identity_generation),
                str(request.directional_constraint).strip().upper(),
            )

            status_identity = (
                str(status.request_id),
                str(status.navigation_goal_id),
                str(status.interaction_id),
                str(status.person_id),
                str(status.tracking_session_id),
                int(status.identity_generation),
                str(status.directional_constraint).strip().upper(),
            )

            return bool(
                str(request.action).strip().upper() == "INSTALL"
                and request.navigation_goal_id == scope[0]
                and request.tracking_session_id == scope[3]
                and request.person_id == scope[4]
                and request_identity == status_identity
                and request_identity[-1] == direction
                and str(status.state).strip().upper() == "PUBLISHED"
                and str(status.mask_id or "").strip()
                and self._message_is_recent(status, 1.0)
            )

        except (AttributeError, TypeError, ValueError):
            return False
    # --------------------------------------------------
    # DIRECT NAV2 START
    # --------------------------------------------------

    def _try_direct_nav2_handoff(self):
        """Start Nav2 only after exact social and stop barriers."""
        if self._nav_phase not in {
            "DIRECT_NAV2_WAIT_MASK",
            "DIRECT_NAV2_WAIT_STOP",
        }:
            return False

        if (
            not self._released
            or not self._scope
            or self._denied_scope
            or self.dialogue_outcome == "DENY_CONFIRMED"
            or not self.active_navigation_goal_id
            or self.active_navigation_goal_pose is None
            or self.pending_terminal_stop is not None
            or self._control is not None
            or self._generation is not None
            or getattr(
                self,
                "uncertain_generation_request_id",
                "",
            )
            or getattr(
                self,
                "pending_generation_request_id",
                "",
            )
        ):
            return False

        if self.gate_ready_goal_id != self.active_navigation_goal_id:
            return False

        autonomous_fallback = bool(
            self._episode_mode == "C"
            and getattr(
                self,
                "_hri_nav_fallback_active",
                False,
            )
        )

        if not autonomous_fallback and (
            self.pending_hri_close is not None
            or self.dialogue_audio_recovery_pending
            or getattr(
                self,
                "pending_dialogue_request_id",
                "",
            )
        ):
            return False

        if not self._stopped() or not self.nav2_cancel_confirmed:
            return False

        if (
            self.stop_request_stamp_sec is not None
            and (
                self.motion_state_stamp_sec is None
                or self.motion_state_stamp_sec
                    < self.stop_request_stamp_sec
            )
        ):
            return False

        mode = str(
            self._episode_mode or ""
        ).strip().upper()

        direction = str(
            self._direction or "NONE"
        ).strip().upper()

        if mode not in {"C", "D"}:
            return False

        if mode == "C":
            if direction != "NONE":
                self._fail_direct_nav2_handoff(
                    "MODE_C_DIRECTION_MISMATCH"
                )
                return False

        else:
            finalization = getattr(
                self,
                "_directional_finalization",
                None,
            )

            candidate = (
                finalization.get("candidate")
                if isinstance(finalization, dict)
                else None
            )

            if not isinstance(candidate, dict):
                self._fail_direct_nav2_handoff(
                    "MODE_D_CONFIRMATION_MISSING"
                )
                return False

            resolved = str(
                candidate.get("resolved_robot_direction") or ""
            ).strip().upper()

            requested = str(
                candidate.get("direction") or ""
            ).strip().upper()

            if not (
                direction in {"LEFT", "RIGHT"}
                and resolved == direction
                and requested == self._normalize_direction(
                    self.confirmed_human_direction
                )
                and self.confirmed_human_intent == "ALLOW_PASSAGE"
                and self.dialogue_outcome == "ALLOW_CONFIRMED"
                and self._permission_applies_to_current_episode()
                and finalization.get("phase") == "ANNOUNCED"
                and finalization.get("scope") == self._scope
                and candidate.get("scope") == self._scope
                and candidate.get(
                    "final_confirmation_received"
                ) is True
            ):
                self._fail_direct_nav2_handoff(
                    "MODE_D_PERMISSION_MISMATCH"
                )
                return False

            if not self._direct_directional_ready():
                # Adaptation owns the missing-enforcement recovery.
                # No START is authorized here.
                return False

        reason = (
            "CONFIRMED_PASSAGE_NAV2_HANDOFF"
            if mode == "D"
            else "AUTONOMOUS_MODE_C_NAV2_HANDOFF"
        )

        try:
            request_id = self._command(
                "START",
                reason,
                path_id="",
            )
        except RuntimeError as exc:
            self._fail_direct_nav2_handoff(
                f"START_SUBMISSION_FAILED:{exc}"
            )
            return False

        if not request_id:
            # NegotiatedAdaptationMixin can take ownership
            # of a failed directional preparation.
            if getattr(self, "_adaptation", None) is None:
                self._fail_direct_nav2_handoff(
                    "START_NOT_PUBLISHED"
                )
            return False

        self._nav_phase = "DIRECT_NAV2_START_PENDING"
        self._execution_sent_at = time.monotonic()

        self.get_logger().info(
            "[DIRECT NAV2] START submitted | "
            f"goal={self.active_navigation_goal_id} | "
            f"mode={mode} | "
            f"robot_direction={direction} | "
            f"request={request_id}"
        )

        return True
    
    # --------------------------------------------------
    # DIRECT NAV2 FAILURE HANDLING
    # --------------------------------------------------

    def _fail_direct_nav2_handoff(self, reason):
        """Block direct navigation without changing human permission."""
        reason = str(
            reason or "UNKNOWN_HANDOFF_FAILURE"
        ).strip().upper()

        previous_phase = str(
            self._nav_phase or ""
        )

        self._nav_phase = "DIRECT_NAV2_BLOCKED"

        self.get_logger().error(
            "[DIRECT NAV2] Handoff blocked | "
            f"goal={self.active_navigation_goal_id} | "
            f"direction={self._direction} | "
            f"reason={reason} | "
            "automatic_direction_fallback=False | "
            "legacy_path_fallback=False"
        )

        # A previously submitted START may still have reached
        # the Gate. Issue WAIT rather than assuming it failed.
        #
        # Preserve a currently outstanding WAIT transaction:
        # it is already the request that should stop Nav2.
        pending = self._control

        if (
            pending is not None
            and pending.get("action") == "WAIT"
        ):
            return

        if (
            not self.active_navigation_goal_id
            or self.pending_terminal_stop is not None
        ):
            return

        # Replace a pending START transaction with a newer
        # WAIT command. The Gate owns internal cancellation.
        if previous_phase in {
            "DIRECT_NAV2_START_PENDING",
            "DIRECT_NAV2_STARTING",
            "NAVIGATING",
        }:
            try:
                self._command(
                    "WAIT",
                    "DIRECT_NAV2_HANDOFF_BLOCKED_" + reason,
                    path_id="",
                )

            except RuntimeError as exc:
                self.get_logger().error(
                    "[DIRECT NAV2] WAIT publication failed | "
                    f"error={exc}"
                )
