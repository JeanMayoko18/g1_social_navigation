"""Admission de l'événement social canonique."""

from g1_social_common.collaboration_states import (
    STATE_STOPPING_FOR_SOCIAL_CONFLICT,
)
from g1_social_common.social_navigation_events import (
    SOCIAL_EVENT_INTRUSION_CONFIRMED,
    normalize_social_navigation_event,
)


class SocialEventsMixin:

      def _on_social_decision(self, msg):
        """Admit an intrusion while preserving same-person consent."""
        event = normalize_social_navigation_event(msg.event)

        if event != SOCIAL_EVENT_INTRUSION_CONFIRMED:
            return

        if not self._is_fresh_social_decision(msg):
            return

        try:
            producer, episode, number = str(
                msg.decision_id
            ).split(":")

            number = int(number)
            tracking = str(msg.tracking_session_id or "").strip()
            person = str(msg.person_id or "").strip()

            stamp = msg.header.stamp
            stamp_ns = (
                int(stamp.sec) * 1_000_000_000
                + int(stamp.nanosec)
            )

            if (
                not all((producer, episode, tracking, person))
                or number <= 0
                or not self._validate_participant_context(
                    msg,
                    SOCIAL_EVENT_INTRUSION_CONFIRMED,
                )
            ):
                return

        except (ValueError, TypeError, AttributeError):
            return

        if number <= self._watermarks.get(producer, 0):
            return

        self._watermarks[producer] = number

        goal_id = str(self.active_navigation_goal_id or "").strip()

        if not goal_id or stamp_ns <= self._goal_epoch:
            return

        scope = (goal_id, producer, episode, tracking, person)

        if scope in self._handled or scope == self._scope:
            return

        if self._scope is not None:
            execution_confirmed = bool(
                self._released
                and self.navigation_active
                and str(self.navigation_state).strip().upper()
                    == "NAVIGATING"
                and self._nav_phase in {
                    "EXECUTING",
                    "NAVIGATING",
                    "DIRECT_NAV2_EXECUTING",
                }
            )

            if not execution_confirmed:
                return

        if self._assistance_id or self.pending_terminal_stop:
            return

        # Capture consent before resetting episode-local fields.
        authorized = self.authorized_episode_key

        keep_permission = bool(
            authorized
            and len(authorized) == 5
            and (authorized[0], authorized[3], authorized[4])
                == (goal_id, tracking, person)
            and self.confirmed_human_intent == "ALLOW_PASSAGE"
            and self.dialogue_outcome == "ALLOW_CONFIRMED"
        )

        retained_direction = (
            self._normalize_direction(self.confirmed_human_direction)
            if keep_permission
            else "NONE"
        )

        retained_interaction = (
            str(
                getattr(
                    self,
                    "_directional_confirmed_interaction_id",
                    "",
                ) or ""
            ).strip()
            if keep_permission
            else ""
        )

        # Autonomous Mode C resolution is reusable, but is not consent.
        previous_scope = self._scope

        reuse_autonomous_resolution = bool(
            self.interaction == "none"
            and self._released
            and previous_scope
            and len(previous_scope) == 5
            and (
                previous_scope[0],
                previous_scope[3],
                previous_scope[4],
            ) == (goal_id, tracking, person)
        )

        self._scope = scope
        self._released = False
        self._episode_mode = (
            "C" if self.interaction == "none" else "D"
        )

        self._denied_scope = None
        self._silence_pending = False
        self._silence_announcement_id = ""
        self._silence_announcement_started_at = None
        self._silence_announcement_fault = False
        self._hri_nav_fallback_active = False
        self._directional_retry_pending = False
        self._retained_same_person_consent = None
        self._reuse_autonomous_scope = (
            scope if reuse_autonomous_resolution else None
        )

        self._nav_phase = "SOCIAL"
        self._candidate = None
        self._validation = None
        self._control = None

        self.social_state_session = producer
        self.social_state_episode = episode
        self.social_state_sequence = number
        self.latest_social_decision = msg

        self.active_person_id = person
        self.active_decision_id = msg.decision_id
        self._active_actor_scope = scope
        self.social_episode_origin_goal_id = goal_id

        self.human_proximity_active = True
        self.social_observation_available = True
        self.social_observation_complete = bool(
            msg.observation_complete
        )
        self.social_loss_expired = False
        self.social_episode_retired = False
        self.waiting_for_observation_reacquisition = False

        self.announcement_social_decision = msg
        self.announcement_decision_id = msg.decision_id
        self.finished_announcement_attempts.clear()
        self.announcement_context_key = None
        self.current_announcement_action = ""

        self.active_interaction_id = ""
        self.pending_dialogue_request_id = ""
        self.pending_dialogue_scope = None
        self.mode_d_hri_owner = None
        self.pending_human_intent = "UNKNOWN"
        self.pending_human_direction = "NONE"

        self.dialogue_outcome = (
            "ALLOW_CONFIRMED" if keep_permission else "NONE"
        )
        self.confirmed_human_intent = (
            "ALLOW_PASSAGE" if keep_permission else "UNKNOWN"
        )
        self.confirmed_human_direction = retained_direction
        self.authorized_episode_key = (
            scope if keep_permission else None
        )
        self.dialogue_resolution_complete = keep_permission

        self._directional_confirmed_scope = (
            scope if keep_permission else None
        )
        self._directional_confirmed_interaction_id = (
            retained_interaction
        )

        # A new interruption still requires a real Gate hold
        # and fresh physical-stop evidence.
        self.nav2_cancel_requested = True
        self.nav2_cancel_confirmed = False
        self.pending_stop_hold_request_id = ""
        self.stable_stop_confirmed = False
        self.motion_state_received_at = None
        self.motion_state_stamp_sec = None
        self._physical_wait = None

        self.stop_request_stamp_sec = (
            self.get_clock().now().nanoseconds / 1e9
        )

        if keep_permission or reuse_autonomous_resolution:
            self.get_logger().info(
                "[SOCIAL MEMORY] Resolution retained | "
                f"goal={goal_id} | person={person} | "
                f"tracking_session={tracking} | episode={episode} | "
                f"human_permission={keep_permission} | "
                f"direction={retained_direction} | "
                "dialogue_restart_required=False"
            )

        self._transition_to(
            STATE_STOPPING_FOR_SOCIAL_CONFLICT,
            reason="NEW_SOCIAL_INTRUSION",
        )

        self._publish_wait("NEW_SOCIAL_INTRUSION")
        