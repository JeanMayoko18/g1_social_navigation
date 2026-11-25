#!/usr/bin/env python3

"""Sequential social policy with an independent navigation transaction."""

import json
import time
import uuid

from std_msgs.msg import String

from g1_social_msgs.msg import (
    ExplanationContext,
    SpeechCancelRequest,
)

from .collaboration_manager_node import CollaborationManager

from g1_social_common.collaboration_states import (
    STATE_NAVIGATING,
    STATE_WAITING,
    STATE_STOPPING_FOR_SOCIAL_CONFLICT,
    STATE_WAITING_FOR_HUMAN,
    STATE_ANNOUNCING_DETECTION,
)

from g1_social_common.social_navigation_events import (
    SOCIAL_EVENT_INTRUSION_CONFIRMED,
    normalize_social_navigation_event,
)


class SimpleCDCollaborationManager(CollaborationManager):
    """Keep social resolution separate from planning and execution."""

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

        super().__init__()

    def _initialize_social_reassessment(self):
        super()._initialize_social_reassessment()
        self.anticipatory_planning_enabled = False

    def _social_scope_key(self, person_id=None):
        return self._scope or ("", "", "", "", "")

    def _mode_d_person_scope(self, person_id):
        """Bind dialogue ownership to the admitted canonical situation."""
        if self._scope and person_id == self._scope[4]:
            return self._scope

        return None

    def _individual_situation_callback(self, msg):
        """Canonical intrusion events exclusively open social cycles."""
        return

    def _social_episode_retirement_callback(self, msg):
        """Retirement does not reopen a completed social interaction."""
        return

    def _navigation_goal_context_callback(self, msg):
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

    def _social_decision_callback(self, msg):
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
            tracking = str(msg.tracking_session_id).strip()
            person = str(msg.person_id).strip()

            stamp = msg.header.stamp

            stamp_ns = (
                int(stamp.sec) * 1_000_000_000
                + int(stamp.nanosec)
            )

            if (
                not all((producer, episode, tracking, person))
                or number <= 0
            ):
                return

            if not self._validate_participant_context(
                msg,
                SOCIAL_EVENT_INTRUSION_CONFIRMED,
            ):
                return

        except (ValueError, TypeError, AttributeError):
            return

        if number <= self._watermarks.get(producer, 0):
            return

        self._watermarks[producer] = number

        # Idle speech has no ownership of the navigation FSM.
        if not self.active_navigation_goal_id:
            key = (producer, episode, tracking, person)

            if key not in self._idle_seen:
                self._idle_seen.add(key)

                self._explain(
                    "OBSERVE",
                    "IDLE_PERSON_DETECTED",
                    person,
                    "A person is visible.",
                    "Briefly acknowledge the person without "
                    "promising navigation.",
                )

            return

        if stamp_ns <= self._goal_epoch:
            return

        scope = (
            self.active_navigation_goal_id,
            producer,
            episode,
            tracking,
            person,
        )

        if scope in self._handled or scope == self._scope:
            return

        # A second interaction cannot replace an unfinished
        # dialogue or announcement.
        if self._scope and not self._released:
            return

        if self._assistance_id or self.pending_terminal_stop:
            return

        self._scope = scope
        self._released = False
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
        self.social_episode_origin_goal_id = scope[0]

        self.human_proximity_active = True
        self.social_observation_available = True
        self.social_loss_expired = False
        self.social_episode_retired = False
        self.waiting_for_observation_reacquisition = False

        self.announcement_social_decision = msg
        self.announcement_decision_id = msg.decision_id
        self.finished_announcement_attempts.clear()

        self.active_interaction_id = ""
        self.pending_dialogue_request_id = ""
        self.mode_d_hri_owner = None

        self.dialogue_outcome = "NONE"
        self.confirmed_human_intent = "UNKNOWN"
        self.confirmed_human_direction = "NONE"
        self.authorized_episode_key = None

        self.nav2_cancel_requested = True
        self.nav2_cancel_confirmed = False

        self.stop_request_stamp_sec = (
            self.get_clock().now().nanoseconds / 1e9
        )

        self._transition_to(
            STATE_STOPPING_FOR_SOCIAL_CONFLICT,
            reason="NEW_SOCIAL_INTRUSION",
        )

        self._publish_wait("NEW_SOCIAL_INTRUSION")

    def _publish_wait(self, reason):
        """Do not reopen social waiting after resolution."""
        if self._released:
            return ""

        result = super()._publish_wait(reason)
        self._last_social_wait_at = time.monotonic()

        return result

    def _enter_social_resolution(self, *, speech_completed=False):
        if not speech_completed:
            self._request_terminal_stop(
                reason="INITIAL_ANNOUNCEMENT_FAILED"
            )
            return

        self.announcement_context_key = None
        self.current_announcement_action = ""

        if self.interaction == "none":
            self._release_navigation("NONE")
        else:
            self._start_human_initiative_window()

    def _start_human_initiative_window(self):
        if (
            not self._scope
            or self._released
            or self.pending_hri_close
        ):
            return

        request = self._new_dialogue_request()

        if request is None:
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

        self.active_dialogue_request_type = request.request_type
        self.waiting_for_human_initiative = True
        self.dialogue_outcome = "PENDING"
        self.dialogue_started_at = time.monotonic()

        self._transition_to(
            STATE_WAITING_FOR_HUMAN,
            reason="HUMAN_INITIATIVE",
        )

        self.dialogue_request_pub.publish(request)

    def _hri_dialogue_control_result_callback(self, msg):
        data = self._json(msg, "hri_manager")
        pending = self.pending_hri_close

        if (
            not data
            or not pending
            or any(
                data.get(key) != pending[key]
                for key in (
                    "close_id",
                    "interaction_id",
                    "request_id",
                )
            )
        ):
            return

        if data.get("status") != "CLOSED":
            return

        self.pending_hri_close = None
        self.hri_close_fault_reported = False
        self.pending_dialogue_request_id = ""
        self.pending_dialogue_scope = None
        self.dialogue_audio_recovery_pending = False
        self.active_interaction_id = ""
        self.mode_d_hri_owner = None
        self.dialogue_started_at = None

        if (
            self.dialogue_outcome == "ALLOW_CONFIRMED"
            and self._permission_applies_to_current_episode()
        ):
            self._release_navigation(
                self.confirmed_human_direction
            )

        elif self.dialogue_outcome == "DENY_CONFIRMED":
            self._request_terminal_stop(
                reason="HUMAN_DENIED_PASSAGE"
            )

    def _enter_replanning(
        self,
        reason,
        directional_constraint=None,
    ):
        """Parent dialogue fallbacks cannot reopen navigation policy."""
        return

    def _begin_autonomous_readaptation_announcement(self, reason):
        """An unanswered dialogue is not permission in this mode."""
        self._request_terminal_stop(reason=reason)

    def _release_navigation(self, direction):
        if (
            self._released
            or not self._scope
            or self.pending_terminal_stop
        ):
            return

        self._handled.add(self._scope)
        self._released = True

        self._direction = self._normalize_direction(direction)
        self._attempts = 0
        self._assisted = False

        self._candidate = None
        self._validation = None

        self._nav_phase = "BACKOFF"
        self._next_attempt = time.monotonic()

        self.nav2_cancel_requested = False
        self.nav2_cancel_confirmed = False
        self.pending_stop_hold_request_id = ""

        self._transition_to(
            STATE_NAVIGATING,
            reason="SOCIAL_RESOLUTION_COMPLETED",
        )

    @staticmethod
    def _json(msg, producer):
        try:
            data = json.loads(msg.data)

            if (
                isinstance(data, dict)
                and data.get("version") == 1
                and data.get("producer") == producer
            ):
                return data

        except (ValueError, TypeError):
            pass

        return None

    def _command(self, action, reason, path_id=""):
        request_id = self._publish_collaborative_decision(
            action=action,
            outcome="NAVIGATION_TRANSACTION",
            human_intent=self.confirmed_human_intent,
            directional_constraint=self._direction,
            path_id=path_id,
            reason=reason,
        )

        if request_id:
            self._control = {
                "action": action,
                "request_id": request_id,
                "goal": self.active_navigation_goal_id,
                "path": path_id,
                "sent": time.monotonic(),
            }

        return request_id

    def _gate_command_result_callback(self, msg):
        data = self._json(msg, "navigation_goal_gate")

        if not data:
            return

        if (
            data.get("action") == "GOAL_READY"
            and data.get("status") == "READY"
        ):
            self._ready_goals.add(
                data.get("navigation_goal_id")
            )

        if (
            data.get("navigation_goal_id")
            != self.active_navigation_goal_id
        ):
            return

        action = data.get("action")
        status = data.get("status")

        if action == "GOAL_READY" and status == "READY":
            self.gate_ready_goal_id = (
                self.active_navigation_goal_id
            )
            return

        if (
            not self._released
            and action == "WAIT"
            and status == "HOLD_CONFIRMED"
        ):
            if (
                data.get("request_id")
                == self.pending_stop_hold_request_id
            ):
                self.nav2_cancel_confirmed = True
                self._evaluate_stop_barrier()

            return

        pending = self._control

        if not pending or (
            action,
            data.get("request_id"),
        ) != (
            pending["action"],
            pending["request_id"],
        ):
            return

        if (
            action == "RESUME"
            and data.get("path_id") != pending["path"]
        ):
            return

        if status in {"REJECTED", "EXPIRED"}:
            self._control = None

            self._retry(
                "GATE_" + str(data.get("reason"))
            )

        elif action == "START" and status == "ACCEPTED":
            self._control = None

            self.initial_start_authorized_goal_id = (
                self.active_navigation_goal_id
            )

            self._nav_phase = "STARTING"
            self._execution_sent_at = time.monotonic()

        elif action == "REPLAN" and status == "HOLD_CONFIRMED":
            self._control = None
            self._nav_phase = "HELD"

        elif (
            action == "RESUME"
            and str(status).startswith("ACCEPTED")
        ):
            self._nav_phase = "RESUMING"

    def _stopped(self):
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
            and 0 <= now - self.motion_state_received_at <= 0.5
            and self.motion_state_stamp_sec is not None
            and -0.1 <= (
                self.get_clock().now().nanoseconds / 1e9
                - self.motion_state_stamp_sec
            ) <= 0.5
        )

    def _generate(self):
        if self._generation or not self._stopped():
            return

        request = self._build_trajectory_generation_request(
            generation_reason=(
                "ALTERNATIVE"
                if self._direction != "NONE"
                else "REPLAN"
            ),
            directional_constraint=self._direction,
        )

        if request is None:
            return

        self._attempts += 1
        self._candidate = None
        self._validation = None
        self._early_validations.clear()

        self._generation = {
            "id": request.request_id,
            "goal": request.navigation_goal_id,
            "scope": self._scope,
            "sent": time.monotonic(),
            "terminal": False,
            "query_at": 0.0,
        }

        self._nav_phase = "GENERATING"

        self.trajectory_generation_request_pub.publish(request)

        self.get_logger().info(
            "[NAV PIPELINE] "
            f"attempt={self._attempts} | "
            f"direction={self._direction} | "
            f"request={request.request_id}"
        )

    def _candidate_trajectory_callback(self, msg):
        job = self._generation

        if (
            not job
            or msg.request_id != job["id"]
            or msg.navigation_goal_id != job["goal"]
        ):
            return

        if (
            not msg.path_id
            or not msg.path.header.frame_id
            or not msg.path.poses
        ):
            return

        if (
            self._candidate is None
            and job["scope"] == self._scope
            and self._released
        ):
            self._candidate = msg

            self._validation = self._early_validations.pop(
                msg.path_id,
                None,
            )

            self._early_validations.clear()

        self._consume_generation()

    def _trajectory_generation_result_callback(self, msg):
        data = self._json(msg, "trajectory_manager")
        job = self._generation

        if not data or not job or (
            data.get("request_id"),
            data.get("navigation_goal_id"),
        ) != (
            job["id"],
            job["goal"],
        ):
            return

        status = data.get("status")

        if status == "TERMINATION_UNCONFIRMED":
            job["query_at"] = 0.0

        elif status == "FAILED":
            self._generation = None

            self._retry(
                str(data.get("reason", "PLANNING_FAILED"))
            )

        elif status == "TERMINATED":
            job["terminal"] = True
            job["terminal_at"] = time.monotonic()
            self._consume_generation()

    def _consume_generation(self):
        job = self._generation

        if not job or not job["terminal"]:
            return

        if (
            job["scope"] != self._scope
            or not self._released
        ):
            self._generation = None
            return

        if self._candidate is None:
            return

        self._generation = None
        self._nav_phase = "VALIDATING"
        self._validation_deadline = time.monotonic() + 2.0

        self._try_resume()

    def _trajectory_validation_callback(self, msg):
        if not self._released:
            return

        if (
            self._candidate is not None
            and msg.path_id != self._candidate.path_id
        ):
            return

        if (
            self._generation is None
            and self._candidate is None
        ):
            return

        if not self._message_is_recent(
            msg,
            self.trajectory_validation_max_age_sec,
        ):
            return

        if self._candidate is None:
            if len(self._early_validations) >= 16:
                self._early_validations.pop(
                    next(iter(self._early_validations))
                )

            self._early_validations[msg.path_id] = msg
            return

        self._validation = msg
        self._try_resume()

    def _trajectory_execution_allowed(self, path_id):
        candidate = self._candidate
        validation = self._validation

        return bool(
            self._released
            and not self.pending_terminal_stop
            and not self.pending_hri_close
            and not self._assistance_id
            and candidate is not None
            and validation is not None
            and candidate.navigation_goal_id
                == self.active_navigation_goal_id
            and candidate.path_id
                == validation.path_id
                == path_id
            and str(validation.validation).upper() == "VALID"
            and self._message_is_recent(
                validation,
                self.trajectory_validation_max_age_sec,
            )
        )

    def _try_resume(self):
        if (
            self._nav_phase != "VALIDATING"
            or self._candidate is None
            or self._validation is None
        ):
            return

        if (
            self._validation.path_id
            != self._candidate.path_id
        ):
            return

        if not self._trajectory_execution_allowed(
            self._candidate.path_id
        ):
            self._retry(
                "VALIDATION_"
                + str(self._validation.validation)
            )
            return

        if self._stopped():
            self._command(
                "RESUME",
                "EXACT_TRAJECTORY_VALID_AND_FRESH",
                self._candidate.path_id,
            )

            self._nav_phase = "RESUMING"

    def _retry(self, reason):
        self._candidate = None
        self._validation = None

        if not self._released:
            if (
                self._scope
                or not self.active_navigation_goal_id
            ):
                return

            self._released = True
            self._attempts = 0
            self._assisted = False
            self._direction = "NONE"

        self._nav_phase = "BACKOFF"
        self._next_attempt = time.monotonic() + 3.0

        self.get_logger().warning(
            "[NAV PIPELINE] "
            f"{reason} | social_fsm=NAVIGATING"
        )

    def _explain(
        self,
        action,
        reason,
        person,
        summary,
        goal,
    ):
        context = ExplanationContext()
        context.header.stamp = self.get_clock().now().to_msg()

        context.decision_id = (
            "SIMPLE_SPEECH_" + uuid.uuid4().hex
        )

        context.navigation_goal_id = (
            self.active_navigation_goal_id
        )

        context.navigation_state = self.navigation_state
        context.navigation_active = bool(self.navigation_active)

        context.person_id = person
        context.language = self.language
        context.action = action
        context.reason_code = reason

        context.situation_summary = summary
        context.robot_intention = goal
        context.explanation_goal = goal

        self.explanation_context_pub.publish(context)

        return context.decision_id

    def _assistance(self):
        if (
            self._assisted
            or self._scope is None
            or self.interaction != "dialogue"
        ):
            self._nav_phase = "FAILED"

            self.get_logger().error(
                "[NAV PIPELINE] Retry budget exhausted; "
                "no executable route. "
                "Social FSM remains NAVIGATING."
            )
            return

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
            "The navigation attempts did not produce "
            "an executable route.",
            "Politely ask the person to move slightly "
            "if convenient. Do not claim every route is blocked.",
        )

    def _speech_status_callback(self, msg):
        if (
            self._assistance_id
            and msg.decision_id == self._assistance_id
        ):
            safe_terminal = (
                msg.state in {"COMPLETED", "CANCELED"}
                or (
                    msg.state == "FAILED"
                    and str(msg.detail).startswith(
                        "AUDIO_RETRYABLE|"
                    )
                )
            )

            if safe_terminal:
                self._assistance_id = ""
                self._attempts = 0

                self._transition_to(
                    STATE_NAVIGATING,
                    reason="PASSAGE_ASSISTANCE_FINISHED",
                )

                self._retry("RETRY_AFTER_ASSISTANCE")

            return

        if not self._released:
            super()._speech_status_callback(msg)

    def _navigation_state_callback(self, msg):
        if (
            msg.navigation_goal_id
                != self.active_navigation_goal_id
            or not self._message_is_recent(msg, 1.0)
        ):
            return

        stamp_ns = (
            int(msg.header.stamp.sec) * 1_000_000_000
            + int(msg.header.stamp.nanosec)
        )

        if stamp_ns <= getattr(self, "_last_nav_ns", 0):
            return

        self._last_nav_ns = stamp_ns
        self.latest_navigation_state = msg

        self.navigation_state = str(
            msg.navigation_state
        ).strip().upper()

        self.navigation_active = bool(msg.navigation_active)

        if self.navigation_state in {
            "SUCCEEDED",
            "ABORTED",
            "CANCELED",
        }:
            self.active_navigation_goal_id = ""
            self.active_navigation_goal_pose = None
            self.pending_terminal_stop = None

            self._control = None
            self._released = False
            self._scope = None
            self._nav_phase = "IDLE"

            self._transition_to(
                STATE_NAVIGATING,
                reason="PUBLIC_GOAL_TERMINAL",
            )

        elif (
            self.navigation_active
            and self.navigation_state == "NAVIGATING"
        ):
            if self._nav_phase in {"RESUMING", "STARTING"}:
                self._control = None
                self._nav_phase = "EXECUTING"

        elif (
            self._released
            and self._nav_phase == "EXECUTING"
            and self.navigation_state in {
                "WAITING",
                "SOCIAL_HOLD",
                "REPLANNING",
            }
        ):
            self._retry("NAVIGATION_EXECUTION_INTERRUPTED")

        self._evaluate_stop_barrier()

    def _timer_callback(self):
        now = time.monotonic()

        self._evaluate_stop_barrier()
        self._watch_hri_close(now)
        self._watch_terminal_stop(now)

        if (
            self.fsm_state == STATE_STOPPING_FOR_SOCIAL_CONFLICT
            and not self.nav2_cancel_confirmed
            and now - getattr(
                self,
                "_last_social_wait_at",
                0.0,
            ) >= 3.0
        ):
            # A new identity forces a new hold acknowledgement.
            self._physical_wait = None
            self._publish_wait("SOCIAL_HOLD_ACK_RETRY")

        if not self._released:
            self._evaluate_initial_announcement_timeout(now)

        job = self._generation

        if job:
            if (
                job["terminal"]
                and now - job["terminal_at"] >= 2.0
            ):
                self._generation = None
                self._retry("NO_CANDIDATE_AFTER_TERMINATION")

            elif (
                now - job["sent"] >= 45.0
                and now >= job["query_at"]
            ):
                self.generation_status_request_pub.publish(
                    String(
                        data=json.dumps({
                            "version": 1,
                            "navigation_goal_id": job["goal"],
                            "request_id": job["id"],
                        })
                    )
                )

                job["query_at"] = now + 3.0

        if (
            not self.active_navigation_goal_id
            or self.pending_terminal_stop
        ):
            return

        control = self._control

        if control and now - control["sent"] >= 10.0:
            self._control = None
            self._retry("GATE_ACK_TIMEOUT")

            # Revoke uncertain execution without changing
            # the social FSM.
            self._command(
                "REPLAN",
                "REVOKE_UNACKNOWLEDGED_EXECUTION",
            )
            return

        if not self._scope and not self._released:
            if (
                self._nav_phase == "STARTING"
                and now - self._execution_sent_at >= 10.0
            ):
                self._retry("INITIAL_EXECUTION_UNCONFIRMED")
                return

            if (
                self.gate_ready_goal_id
                    == self.active_navigation_goal_id
                and self._nav_phase == "IDLE"
                and not self._control
            ):
                self._command(
                    "START",
                    "INITIAL_NAVIGATION_NO_SOCIAL_CONFLICT",
                )

            return

        if (
            not self._released
            or self._generation
            or self._control
        ):
            return

        if self._assistance_id:
            if (
                now - self._assistance_started
                >= self.speech_completion_timeout_sec
            ):
                self.speech_cancel_pub.publish(
                    SpeechCancelRequest(
                        decision_id=self._assistance_id
                    )
                )

                self._assistance_started = now

            return

        if (
            self._nav_phase == "BACKOFF"
            and now >= self._next_attempt
        ):
            self._command(
                "REPLAN",
                "NAVIGATION_RETRY",
            )

        elif self._nav_phase == "HELD" and self._stopped():
            if self._attempts >= self.route_search_max_attempts:
                self._assistance()
            else:
                self._generate()

        elif self._nav_phase == "VALIDATING":
            self._try_resume()

            if (
                self._nav_phase == "VALIDATING"
                and now >= self._validation_deadline
            ):
                self._retry("VALIDATION_TIMEOUT")