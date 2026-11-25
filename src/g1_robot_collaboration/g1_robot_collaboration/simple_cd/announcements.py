"""Publish social explanations and track exact speech transactions."""

import time
import uuid

from g1_social_msgs.msg import ExplanationContext

from g1_social_common.collaboration_states import (
    STATE_ANNOUNCING_DETECTION,
)


class AnnouncementsMixin:

    def _explain(self, action, reason, person, summary, goal):
        """Publish an explanation request without claiming speech was produced."""
        context = ExplanationContext()

        context.header.stamp = (
            self.get_clock().now().to_msg()
        )

        context.decision_id = (
            "SIMPLE_SPEECH_" + uuid.uuid4().hex
        )

        context.navigation_goal_id = (
            self.active_navigation_goal_id
        )

        context.navigation_state = self.navigation_state
        context.navigation_active = bool(self.navigation_active)

        context.person_id = str(person or "")
        context.language = self.language
        context.action = action
        context.reason_code = reason

        context.situation_summary = summary
        context.robot_intention = goal
        context.explanation_goal = goal

        self._ros_publish_explanation(context)

        return context.decision_id

    def _assistance(self):
        """Request optional passage assistance without treating speech as clearance."""
        if (
            not self._scope
            or not self._released
            or not self.active_navigation_goal_id
            or self.pending_terminal_stop is not None
            or self.navigation_active
            or not self._stopped()
        ):
            return False

        mode = str(
            getattr(self, "_episode_mode", "") or ""
        ).strip().upper()

        if mode == "C":
            if self._assisted:
                # Resume bounded searching without repeating the request.
                self._attempts = 0
                self._retry("MODE_C_BOUNDED_REEVALUATION")
                return True

            self._assisted = True
            self._nav_phase = "ASSISTANCE"
            self._assistance_started = time.monotonic()

            try:
                self._assistance_id = self._explain(
                    "WAIT",
                    "PASSAGE_ASSISTANCE_MODE_C",
                    self.active_person_id,
                    "No executable route has been found yet.",
                    "Politely ask the person to move slightly if "
                    "convenient, without requesting an answer or "
                    "directional authorization. Remain stopped.",
                )

            except Exception as exc:
                self._assistance_id = ""

                self.get_logger().warning(
                    "[MODE C] Assistance announcement unavailable | "
                    f"error={exc}"
                )

            if not self._assistance_id:
                self._attempts = 0
                self._retry("MODE_C_ASSISTANCE_UNAVAILABLE")
                return False

            return True

        if mode != "D" or self._assisted:
            # Mode D has its own directional recovery policy.
            self._nav_phase = "FAILED"
            return False

        self._assisted = True
        self._nav_phase = "ASSISTANCE"
        self._assistance_started = time.monotonic()

        self._transition_to(
            STATE_ANNOUNCING_DETECTION,
            reason="PASSAGE_ASSISTANCE",
        )

        self._assistance_id = self._explain(
            "WAIT",
            "PASSAGE_ASSISTANCE",
            self.active_person_id,
            "The currently permitted passage has no executable route.",
            "Politely ask the person to move slightly toward their "
            "own right, if convenient, to free their own left. "
            "Do not claim permission for another direction.",
        )

        return bool(self._assistance_id)

    def _announce_no_feasible_route(self):
        """Report a confirmed two-sided planning failure once per recovery scope."""

        # Both directional searches must be exhausted first.
        if (
            getattr(self, "_recovery_phase", "")
            != "RIGHT_EXHAUSTED"
        ):
            self.get_logger().warning(
                "[ROUTE RECOVERY] Route-unavailable announcement rejected: "
                "both directions have not been established as exhausted."
            )
            return False

        if (
            not self._scope
            or not self.active_navigation_goal_id
            or self.navigation_active
            or getattr(self, "_no_feasible_route_id", "")
            or getattr(self, "_no_feasible_announced_scope", None)
                == self._scope
        ):
            return False

        self._no_feasible_announced_scope = self._scope
        self._recovery_phase = "NO_FEASIBLE_ROUTE_ANNOUNCING"
        self._nav_phase = "NO_FEASIBLE_ROUTE_ANNOUNCING"

        self._no_feasible_route_started = time.monotonic()

        self._no_feasible_route_id = self._explain(
            "WAIT",
            "NO_FEASIBLE_ROUTE",
            self.active_person_id,
            "Neither person-relative route has produced "
            "an executable path.",
            "Inform the person that neither side currently "
            "has a practicable path and that I remain stopped "
            "while searching for an alternative.",
        )

        return True


    def _on_speech_status(self, msg):
        """
        Track assistance announcements.

        Directional confirmation and autonomous recovery
        announcements are owned by the main ROS callback.
        """
        decision_id = str(msg.decision_id or "").strip()
        state = str(msg.state or "").strip().upper()

        if (
            self._assistance_id
            and decision_id == self._assistance_id
        ):
            if state not in {"COMPLETED", "FAILED", "CANCELED"}:
                return

            mode = str(
                getattr(self, "_episode_mode", "") or ""
            ).strip().upper()

            if mode == "C":
                self._assistance_id = ""
                self._attempts = 0

                # Start a new route search, not an automatic
                # authorization to execute an unvalidated path.
                self._retry(
                    "MODE_C_REPLAN_AFTER_ASSISTANCE"
                )
                return

            if mode == "D" and state == "COMPLETED":
                self._assistance_id = ""
                self._attempts = 0
                self._recovery_phase = "LEFT_AFTER_ASSISTANCE"

                self._retry(
                    "MODE_D_REPLAN_LEFT_AFTER_ASSISTANCE"
                )
                return

            if mode == "D":
                self._assistance_id = ""
                self._recovery_phase = "ASSISTANCE_AUDIO_FAILURE"
                self._nav_phase = "ASSISTANCE_AUDIO_FAILURE"
                return

        if not self._released:
            return super()._speech_status_callback(msg)
        
    def _speech_status_callback(self, msg):
        """Dispatch ROS speech status to the current assistance policy."""
        return self._on_speech_status(msg)