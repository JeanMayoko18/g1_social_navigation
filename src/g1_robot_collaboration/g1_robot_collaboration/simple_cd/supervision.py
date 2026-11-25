
"""Supervise HRI, Nav2 handoff and directional-mask health."""

import copy
import time

from g1_social_common.collaboration_states import (
    STATE_NAVIGATING,
    STATE_STOPPING_FOR_SOCIAL_CONFLICT,
)


class SupervisionMixin:
    """Coordinate watchdogs without selecting or validating Nav2 paths."""

    _REVALIDATABLE_OBSERVATION_REASONS = frozenset({
        "INVALID_OR_STALE_SOCIAL_OBSERVATION",
        "NO_SOCIAL_OBSERVATION",
        "OBSERVATION_EXPIRED_DURING_VALIDATION",
        "INDIVIDUAL_SNAPSHOT_STALE",
        "INDIVIDUAL_SNAPSHOT_SOURCE_MISMATCH",
        "NO_INDIVIDUAL_EPISODE_SNAPSHOT",
        "INCOMPLETE_SOCIAL_COVERAGE",
        "UNRESOLVED_UNOBSERVED_PERSON",
        "UNRESOLVED_PERSON_ZONE_MISSING",
    })

    # --------------------------------------------------
    # DIRECT NAV2 HANDOFF SUPERVISION
    # --------------------------------------------------

    def _block_direct_handoff(self, reason):
        """Request a Gate hold without changing the authorized direction."""
        if self._nav_phase == "DIRECT_NAV2_BLOCKED":
            return

        self.get_logger().error(
            "[DIRECT NAV2] Hold required | "
            f"reason={reason} | "
            f"direction={self._direction}"
        )

        self._nav_phase = "DIRECT_NAV2_BLOCKED"

        # A pending START may already have reached the Gate.
        # Request WAIT instead of assuming that navigation stopped.
        self._control = None

        if (
            self.active_navigation_goal_id
            and self.pending_terminal_stop is None
        ):
            self._command(
                "WAIT",
                reason,
                path_id="",
            )

    def _watch_directional_constraint(self, now):
        """Supervise optional mask production without delaying Nav2 START."""
        direction = str(
            self._direction or "NONE"
        ).strip().upper()

        request = getattr(self, "_directional_request", None)

        if direction not in {"LEFT", "RIGHT"}:
            if request is not None:
                last_retry = getattr(
                    self,
                    "_optional_revoke_retry_at",
                    0.0,
                )

                if now - last_retry >= 1.0:
                    self._optional_revoke_retry_at = now
                    self._revoke_optional_directional_mask(
                        "UNCONSTRAINED_NAVIGATION"
                    )
            return True

        if request is None:
            return self._use_unconstrained_nav2(
                "OPTIONAL_MASK_REQUEST_MISSING"
            )

        scope = self._social_scope_key()

        if (
            not scope
            or len(scope) != 5
            or request.navigation_goal_id != scope[0]
            or request.tracking_session_id != scope[3]
            or request.person_id != scope[4]
            or str(request.action).strip().upper() != "INSTALL"
            or str(request.directional_constraint).strip().upper()
                != direction
            or not self._permission_applies_to_current_episode()
        ):
            return self._use_unconstrained_nav2(
                "OPTIONAL_MASK_SCOPE_MISMATCH"
            )

        request_id = str(request.request_id)

        if getattr(
            self,
            "_optional_mask_watch_id",
            "",
        ) != request_id:
            self._optional_mask_watch_id = request_id
            self._optional_mask_started_at = now

        status = getattr(self, "_directional_status", None)

        state = (
            str(status.state or "").strip().upper()
            if status is not None
            else ""
        )

        if state in {"FAILED", "REVOKED", "INACTIVE"}:
            return self._use_unconstrained_nav2(
                "OPTIONAL_MASK_" + state
            )

        ready = self._direct_directional_ready()

        if not ready:
            started_at = self._optional_mask_started_at

            if state in {"", "INSTALLING"} and now - started_at < 1.0:
                # Allow asynchronous mask construction.
                # Returning True does not claim that the mask is ready.
                # START remains unconstrained through the Gate contract.
                return True

            return self._use_unconstrained_nav2(
                "OPTIONAL_MASK_NOT_AVAILABLE"
            )

        lease = (
            int(request.validity_duration.sec)
            + int(request.validity_duration.nanosec) / 1e9
        )

        if lease <= 0.0:
            return self._use_unconstrained_nav2(
                "OPTIONAL_MASK_LEASE_INVALID"
            )

        last_renewal = getattr(
            self,
            "_directional_last_renewal_at",
            None,
        )

        if last_renewal is None:
            self._directional_last_renewal_at = now
            last_renewal = now

        if now - last_renewal < min(10.0, lease / 3.0):
            return True

        publisher = getattr(
            self,
            "directional_constraint_pub",
            None,
        )

        if publisher is None:
            return self._use_unconstrained_nav2(
                "OPTIONAL_MASK_PUBLISHER_MISSING"
            )

        try:
            renewal = copy.deepcopy(request)
            renewal.header.stamp = self.get_clock().now().to_msg()
            publisher.publish(renewal)

        except Exception as exc:
            self.get_logger().warning(
                "[DIRECTIONAL MASK] Renewal failed | "
                f"error={exc}"
            )
            return self._use_unconstrained_nav2(
                "OPTIONAL_MASK_RENEWAL_FAILED"
            )

        self._directional_last_renewal_at = now
        return True
    
    # --------------------------------------------------
    # MAIN TIMER
    # --------------------------------------------------

    def _watch_global_social_wait(self, now):
        """
        Watch social waiting.

        Confirmed NO:
            after 180 seconds, start a fresh HRI dialogue.

        Unresolved interaction without NO:
            preserve the existing autonomous announcement
            policy.

        Audio failure:
            never reinterpret it as human silence.
        """
        goal_id = str(
            self.active_navigation_goal_id or ""
        ).strip()

        if self._released:
            self._social_wait_started_at = None
            self._social_wait_goal_id = ""
            self._social_wait_timeout_triggered = False

            self._denial_wait_started_at = None
            self._denial_wait_goal = ""
            self._denial_wait_scope = None
            return

        if (
            not goal_id
            or not self._scope
            or self.pending_terminal_stop is not None
        ):
            return

        denied = bool(
            self._denied_scope == self._scope
            or self.dialogue_outcome == "DENY_CONFIRMED"
            or self.confirmed_human_intent == "DENY_PASSAGE"
        )

        # ----------------------------------------------
        # 1. EXPLICIT NO: NEW DIALOGUE AFTER 180 SECONDS
        # ----------------------------------------------

        if denied:
            started = getattr(
                self,
                "_denial_wait_started_at",
                None,
            )

            watched_goal = getattr(
                self,
                "_denial_wait_goal",
                "",
            )

            watched_scope = getattr(
                self,
                "_denial_wait_scope",
                None,
            )

            if (
                started is None
                or watched_goal != goal_id
                or watched_scope != self._scope
            ):
                self._denial_wait_started_at = now
                self._denial_wait_goal = goal_id
                self._denial_wait_scope = self._scope

                self.get_logger().info(
                    "[SOCIAL DENIAL] Timer initialized | "
                    f"goal={goal_id} | deadline_sec=180.0"
                )
                return

            elapsed = now - started

            if elapsed < 180.0:
                return

            if self._restart_denied_dialogue():
                self.get_logger().info(
                    "[SOCIAL DENIAL] Dialogue restart submitted | "
                    f"goal={goal_id} | "
                    f"elapsed={elapsed:.1f}s"
                )

            else:
                self.get_logger().warning(
                    "[SOCIAL DENIAL] Deadline reached; "
                    "restart pending | "
                    f"goal={goal_id} | "
                    f"elapsed={elapsed:.1f}s",
                    throttle_duration_sec=10.0,
                )

            return

        # ----------------------------------------------
        # 2. ORDINARY UNRESOLVED SOCIAL WAIT
        # ----------------------------------------------

        started = getattr(
            self,
            "_social_wait_started_at",
            None,
        )

        watched_goal = str(
            getattr(
                self,
                "_social_wait_goal_id",
                "",
            ) or ""
        ).strip()

        if (
            started is None
            or watched_goal != goal_id
        ):
            return

        if self.dialogue_outcome == "ALLOW_CONFIRMED":
            return

        if (
            getattr(
                self,
                "_social_wait_timeout_triggered",
                False,
            )
            or getattr(
                self,
                "_hri_nav_fallback_active",
                False,
            )
            or getattr(
                self,
                "_silence_pending",
                False,
            )
            or getattr(
                self,
                "_silence_announcement_id",
                "",
            )
        ):
            return

        elapsed = now - started

        if elapsed < float(
            self.social_resolution_timeout_sec
        ):
            return

        self._social_wait_timeout_triggered = True

        self.get_logger().warning(
            "[SOCIAL WAIT] Unresolved deadline reached | "
            f"goal={goal_id} | "
            f"elapsed={elapsed:.1f}s | "
            "next=ANNOUNCE_AUTONOMOUS_RESUME"
        )

        self._begin_autonomous_readaptation_announcement(
            "AUTONOMOUS_RESUME_AFTER_WAIT_TIMEOUT"
        )
        
    def _restart_denied_dialogue(self):
        """
        Restart social dialogue after the refusal timeout.

        Preserve the public goal, the social scope and
        the already established physical stop.

        Do not authorize navigation.
        """
        if (
            not self.active_navigation_goal_id
            or not self._scope
            or self.pending_terminal_stop is not None
        ):
            return False

        if (
            not self.nav2_cancel_confirmed
            or not self._stopped()
            or self._control is not None
        ):
            return False

        # A previous HRI transaction must be completely
        # detached before a new one is created.
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
            return False

        if not self._revoke_optional_directional_mask(
            "NEW_DIALOGUE_AFTER_DENIAL"
        ):
            return False

        goal = self.active_navigation_goal_id
        scope = self._scope

        # The old refusal applies only to the old
        # dialogue, not to the upcoming one.
        self._denied_scope = None
        self._handled.discard(scope)

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

        self._episode_mode = "D"
        self._direction = "NONE"
        self._released = False
        self._nav_phase = "HELD"

        self._denial_wait_started_at = None
        self._denial_wait_goal = ""
        self._denial_wait_scope = None

        self._social_wait_started_at = None
        self._social_wait_goal_id = ""
        self._social_wait_timeout_triggered = False

        self.get_logger().info(
            "[SOCIAL DENIAL] Restarting negotiation | "
            f"goal={goal} | "
            f"scope={scope} | "
            "navigation_start=False"
        )

        self._start_human_initiative_window()

        return True
    
    def _on_timer(self):
        """Supervise direct navigation and optional directional guidance."""
        now = time.monotonic()

        self._evaluate_stop_barrier()
        self._watch_hri_close(now)
        self._watch_terminal_stop(now)
        self._watch_global_social_wait(now)

        if (
            self.fsm_state == STATE_STOPPING_FOR_SOCIAL_CONFLICT
            and not self.nav2_cancel_confirmed
            and now - getattr(self, "_last_social_wait_at", 0.0) >= 3.0
        ):
            # Preserve the current stop transaction identity.
            self._publish_wait("SOCIAL_HOLD_ACK_RETRY")

        if self._silence_announcement_id:
            self._watch_silence_announcement(now)

        elif (
            not self._released
            and not self._denied_scope
            and not self._silence_pending
            and not self._silence_announcement_fault
        ):
            self._evaluate_initial_announcement_timeout(now)

        if (
            not self.active_navigation_goal_id
            or self.pending_terminal_stop is not None
        ):
            return

        control = self._control

        if control and now - control["sent"] >= 10.0:
            action = str(control["action"]).upper()

            if action == "WAIT":
                self.get_logger().error(
                    "[DIRECT NAV2] WAIT acknowledgment missing | "
                    "physical hold unconfirmed",
                    throttle_duration_sec=5.0,
                )
                return

            if (
                action == "START"
                or self._nav_phase.startswith("DIRECT_NAV2_")
            ):
                self._block_direct_handoff(
                    "DIRECT_NAV2_GATE_ACK_TIMEOUT"
                )
                return

            self.get_logger().error(
                "[NAV SUPERVISION] Gate acknowledgment missing | "
                f"action={action}",
                throttle_duration_sec=5.0,
            )
            return

        if not self._scope and not self._released:
            if (
                self.gate_ready_goal_id
                    == self.active_navigation_goal_id
                and self._nav_phase == "IDLE"
                and self._control is None
            ):
                self._command(
                    "START",
                    "INITIAL_NAVIGATION_NO_SOCIAL_CONFLICT",
                )
            return

        if not self._released:
            reuse_permission = (
                self._permission_applies_to_current_episode()
            )

            reuse_autonomous = bool(
                self.interaction == "none"
                and self._scope
                and getattr(self, "_reuse_autonomous_scope", None)
                    == self._scope
            )

            # Retry a deferred release without reopening speech or HRI.
            if (
                (reuse_permission or reuse_autonomous)
                and not self._denied_scope
                and self.fsm_state != STATE_STOPPING_FOR_SOCIAL_CONFLICT
                and not self.pending_speech_decision_id
                and self._control is None
                and self.nav2_cancel_confirmed
                and self._stopped()
            ):
                self._release_navigation(
                    self.confirmed_human_direction
                    if reuse_permission
                    else "NONE"
                )
            return

        phase = str(self._nav_phase or "")

        if phase in {
            "DIRECT_NAV2_WAIT_STOP",
            "DIRECT_NAV2_WAIT_MASK",
        }:
            # Check mask availability only when START can be considered.
            # There is no separate mask-wait deadline.
            self._try_direct_nav2_handoff()
            return

        if phase in {
            "DIRECT_NAV2_START_PENDING",
            "DIRECT_NAV2_STARTING",
        }:
            self._watch_directional_constraint(now)

            if phase == "DIRECT_NAV2_STARTING":
                nav = self.latest_navigation_state

                if (
                    nav is not None
                    and nav.navigation_goal_id
                        == self.active_navigation_goal_id
                    and self._message_is_recent(nav, 1.0)
                    and self.navigation_active
                    and str(self.navigation_state).strip().upper()
                        == "NAVIGATING"
                ):
                    self._nav_phase = "DIRECT_NAV2_EXECUTING"

                    self._transition_to(
                        STATE_NAVIGATING,
                        reason="DIRECT_NAV2_EXECUTION_CONFIRMED",
                    )
                    return

                if (
                    now - getattr(self, "_execution_sent_at", now)
                        >= 10.0
                    and self._control is None
                ):
                    self._block_direct_handoff(
                        "DIRECT_NAV2_EXECUTION_UNCONFIRMED"
                    )
            return

        if phase in {
            "DIRECT_NAV2_EXECUTING",
            "NAVIGATING",
            "EXECUTING",
        }:
            self._watch_directional_constraint(now)
            return

        if phase == "DIRECT_NAV2_BLOCKED":
            return

        self.get_logger().warning(
            "[NAV SUPERVISION] Unexpected released phase | "
            f"mode={self._episode_mode} | phase={phase}",
            throttle_duration_sec=5.0,
        )

    # --------------------------------------------------
    # LEGACY CANDIDATE REVALIDATION
    # --------------------------------------------------

    def _watch_unknown_candidate(self, now):
        """Revalidate one immutable candidate using newer social observations."""
        candidate = self._candidate

        if (
            candidate is None
            or self._nav_phase != "VALIDATING"
            or candidate.navigation_goal_id
                != self.active_navigation_goal_id
            or not str(candidate.path_id or "").strip()
            or not str(candidate.request_id or "").strip()
            or not candidate.path.header.frame_id
            or not candidate.path.poses
            or self.pending_terminal_stop is not None
        ):
            self._unknown_revalidation = None
            self._retry(
                "REVALIDATION_CANDIDATE_INVALID"
            )
            return

        state = getattr(
            self,
            "_unknown_revalidation",
            None,
        )

        if (
            state is None
            or state["path"] != candidate.path_id
            or state["goal"] != candidate.navigation_goal_id
            or state["scope"] != self._scope
        ):
            state = {
                "path": candidate.path_id,
                "goal": candidate.navigation_goal_id,
                "scope": self._scope,
                "deadline": now + 8.0,
                "last_source_ns": 0,
                "last_sent_at": 0.0,
                "requests": 0,
            }

            self._unknown_revalidation = state

            self.get_logger().info(
                "[PATH REVALIDATION] Retaining candidate | "
                f"path={candidate.path_id} | "
                "budget=8.0s"
            )

        if now >= state["deadline"]:
            self._unknown_revalidation = None
            self._retry(
                "FRESH_SOCIAL_REVALIDATION_EXPIRED"
            )
            return

        if (
            not self._stopped()
            or not self._fresh_observation_health_available()
            or not self._individual_scene_current()
            or not self.individual_source_complete
        ):
            return

        stamp_ns = self.health_stamp_ns

        if (
            stamp_ns is None
            or stamp_ns <= state["last_source_ns"]
            or self.individual_source_stamp_ns != stamp_ns
            or now - state["last_sent_at"] < 0.75
        ):
            return

        state["last_source_ns"] = stamp_ns
        state["last_sent_at"] = now
        state["requests"] += 1

        # Clear only validation evidence.
        self._validation = None

        self.mode_c_revalidation_pub.publish(
            copy.deepcopy(candidate)
        )

        self.get_logger().info(
            "[PATH REVALIDATION] Exact candidate requested | "
            f"path={candidate.path_id} | "
            f"source_ns={stamp_ns} | "
            f"request={state['requests']}"
        )

    # --------------------------------------------------
    # ASSISTANCE RECOVERY
    # --------------------------------------------------

    def _recover_missing_assistance(self, now, reason):
        """Recover from missing speech results without granting movement."""
        mode = str(
            getattr(self, "_episode_mode", "") or ""
        ).strip().upper()

        self._assistance_id = ""
        self._assistance_started = now

        if (
            not self._scope
            or not self._released
            or not self.active_navigation_goal_id
            or self.pending_terminal_stop is not None
        ):
            return

        if mode == "C":
            self._assisted = True
            self._attempts = 0

            self._retry(reason)

            self._next_attempt = max(
                self._next_attempt,
                now + 10.0,
            )

            self.get_logger().warning(
                "[MODE C] Assistance speech unavailable | "
                f"reason={reason} | "
                "next=FRESH_ROUTE_SEARCH | "
                "human_permission_inferred=False"
            )
            return

        if mode == "D":
            self._assisted = True
            self._recovery_phase = "ASSISTANCE_AUDIO_FAILURE"
            self._nav_phase = "ASSISTANCE_AUDIO_FAILURE"

            self.get_logger().warning(
                "[MODE D] Directional assistance audio unconfirmed | "
                f"reason={reason} | "
                "navigation_remains_held=True"
            )
            return

        self._nav_phase = "FAILED"

        self.get_logger().error(
            "[ASSISTANCE] Unknown recovery mode | "
            f"mode={mode} | reason={reason}"
        )

    # --------------------------------------------------
    # SILENCE ANNOUNCEMENT WATCHDOG
    # --------------------------------------------------


    def _watch_silence_announcement(self, now):
        """
        Recover from an autonomous-resume speech timeout.

        Keep the exact announcement transaction until the
        recovery handler has processed it.

        This is a technical speech timeout, not human silence.
        """
        announcement_id = str(
            getattr(self, "_silence_announcement_id", "") or ""
        ).strip()

        started_at = getattr(
            self,
            "_silence_announcement_started_at",
            None,
        )

        if not announcement_id or started_at is None:
            return

        if (
            now - started_at
            < float(self.speech_completion_timeout_sec)
        ):
            return

        self.get_logger().warning(
            "[HRI ANNOUNCEMENT] Completion timeout | "
            f"decision={announcement_id} | "
            "cause=TECHNICAL_SPEECH_TIMEOUT"
        )

        self._silence_announcement_fault = True

        try:
            self._ros_cancel_speech(announcement_id)
        except Exception as exc:
            self.get_logger().error(
                "[HRI ANNOUNCEMENT] Cancel failed | "
                f"decision={announcement_id} | error={exc}"
            )

        # The completion method owns transaction cleanup
        # and the existing autonomous-recovery handoff.
        self._complete_silence_announcement()
