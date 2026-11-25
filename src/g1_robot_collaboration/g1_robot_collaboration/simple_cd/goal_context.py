"""Buts publics, identité d'épisode et états Nav2."""

from g1_social_common.collaboration_states import (
    STATE_NAVIGATING,
)


class GoalContextMixin:

    def _social_scope_key(self, person_id=None):
        return self._scope or ("", "", "", "", "")

    def _mode_d_person_scope(self, person_id):
        """Associer le dialogue à la personne de l'épisode."""
        if self._scope and person_id == self._scope[4]:
            return self._scope
        return None
    def _permission_applies_to_current_episode(self, person_id=None):
        """Match confirmed consent by goal and tracked-person identity.

        Producer sessions and episode IDs are transaction metadata.
        They do not revoke consent for the same public goal,
        tracking session and person.
        """
        current = self._social_scope_key()
        authorized = self.authorized_episode_key

        if (
            not current
            or len(current) != 5
            or not authorized
            or len(authorized) != 5
        ):
            return False

        current_key = (
            str(current[0] or "").strip(),
            str(current[3] or "").strip(),
            str(current[4] or "").strip(),
        )

        authorized_key = (
            str(authorized[0] or "").strip(),
            str(authorized[3] or "").strip(),
            str(authorized[4] or "").strip(),
        )

        if person_id is not None:
            if str(person_id or "").strip() != current_key[2]:
                return False

        return bool(
            all(current_key)
            and current_key == authorized_key
            and current_key[0] == self.active_navigation_goal_id
            and self.confirmed_human_intent == "ALLOW_PASSAGE"
            and self.dialogue_outcome == "ALLOW_CONFIRMED"
            and self.pending_terminal_stop is None
            and not self.social_resolution_aborted
        )
    
    def _on_navigation_goal_context(self, msg):
        goal = str(msg.navigation_goal_id or "").strip()

        if not goal or not msg.goal.header.frame_id:
            return

        if goal == self.active_navigation_goal_id:
            self.active_navigation_goal_pose = msg.goal
            return

        if self.active_navigation_goal_id or self._generation:
            self.get_logger().error(
                "[SIMPLE] Previous goal is not terminal."
            )
            return

        self._goal_epoch = self.get_clock().now().nanoseconds

        super()._reset_episode()

        self._scope = None
        self._handled.clear()
        self._released = False
        self._denied_scope = None
        self._silence_pending = False
        self._silence_announcement_id = ""
        self._silence_announcement_started_at = None
        self._silence_announcement_fault = False

        self._control = None
        self._candidate = None
        self._validation = None
        self._nav_phase = "IDLE"

        self.active_navigation_goal_id = goal
        self.active_navigation_goal_pose = msg.goal

        self.gate_ready_goal_id = (
            goal if goal in self._ready_goals else ""
        )

        self.initial_start_authorized_goal_id = ""
        self.latest_navigation_state = None
        self._last_nav_ns = 0

        self.navigation_active = False
        self.navigation_state = "UNKNOWN"

        self._transition_to(
            STATE_NAVIGATING,
            reason="NEW_PUBLIC_GOAL",
        )

        self.get_logger().info(
            "[SIMPLE GOAL] "
            f"goal={goal} | epoch={self._goal_epoch}"
        )

    def _on_navigation_state(self, msg):
        """Track the public goal and actual Nav2 execution state."""
        goal_id = str(msg.navigation_goal_id or "").strip()
        if (
            not goal_id
            or goal_id != self.active_navigation_goal_id
            or not self._message_is_recent(msg, 1.0)
        ):
            return

        stamp = msg.header.stamp
        stamp_ns = (
            int(stamp.sec) * 1_000_000_000
            + int(stamp.nanosec)
        )
        if stamp_ns <= getattr(self, "_last_nav_ns", 0):
            return

        self._last_nav_ns = stamp_ns
        self.latest_navigation_state = msg
        self.navigation_state = str(msg.navigation_state or "").strip().upper()
        self.navigation_active = bool(msg.navigation_active)

        if self.navigation_state in {"SUCCEEDED", "ABORTED", "CANCELED"}:
            # Only the public goal's terminal state clears its identity.
            self.active_navigation_goal_id = ""
            self.active_navigation_goal_pose = None
            self.pending_terminal_stop = None
            self._control = None
            self._released = False
            self._scope = None
            self._nav_phase = "IDLE"
            self._denied_scope = None
            self._silence_pending = False
            self._silence_announcement_id = ""
            self._silence_announcement_started_at = None
            self._silence_announcement_fault = False

            # Preserve the parent FSM's existing terminal convention.
            self._transition_to(
                STATE_NAVIGATING,
                reason="PUBLIC_GOAL_TERMINAL",
            )
            return

        if self.navigation_active and self.navigation_state == "NAVIGATING":
            if self._nav_phase == "DIRECT_NAV2_STARTING":
                self._control = None
                self._nav_phase = "DIRECT_NAV2_EXECUTING"
                self._transition_to(
                    STATE_NAVIGATING,
                    reason="NAV2_EXECUTION_CONFIRMED",
                )
                self.get_logger().info(
                    "[NAV2 HANDOFF] Internal Nav2 goal is executing | "
                    f"goal={goal_id}"
                )
            elif self._nav_phase in {"RESUMING", "STARTING"}:
                self._control = None
                self._nav_phase = "EXECUTING"

        elif (
            self._released
            and self._nav_phase == "DIRECT_NAV2_EXECUTING"
            and self.navigation_state in {
                "WAITING", "SOCIAL_HOLD", "REPLANNING"
            }
        ):
            # A new hold must not be reported as active navigation.
            self._fail_direct_nav2_handoff(
                "NAV2_RETURNED_TO_HOLD"
            )

        elif (
            self._released
            and self._nav_phase == "EXECUTING"
            and self.navigation_state in {
                "WAITING", "SOCIAL_HOLD", "REPLANNING"
            }
        ):
            self._retry("NAVIGATION_EXECUTION_INTERRUPTED")

        self._evaluate_stop_barrier()

