#!/usr/bin/env python3

import time

import rclpy
from rclpy.node import Node
import uuid
from g1_social_msgs.msg import (
    DialogueRequest,
    ExplanationText,
    HumanIntent,
    SpeechStatus,
    SpeechCancelRequest,
)
import re
import math

from hri_msgs.msg import LiveSpeech
import json
from std_msgs.msg import String
from g1_social_msgs.msg import SpeechCancelRequest
from g1_social_common.dialogue_texts import (
    get_dialogue_text,
    normalize_dialogue_language,
)
from std_msgs.msg import Bool

from g1_social_msgs.srv import SimulateHumanIntent
from g1_hri.core.hri_manager import HRIManager
from g1_hri.backends.ros4hri_backend import ROS4HRIBackend


class HRIManagerNode(Node):
    """
    HRI Manager.

    Responsibilities:
        1. Receive DialogueRequest from CollaborationManager.
        2. Manage the active HRI interaction.
        3. Produce HumanIntent.
        4. Handle interaction timeout.

    This node does NOT make navigation decisions.

    Architecture:

        CollaborationManager
                |
                | DialogueRequest
                v
            HRI Manager
                |
                | HumanIntent
                v
        CollaborationManager
    """

    def __init__(self):
        super().__init__("hri_manager")

        # ---------------------------------------------------------
        # Parameters
        # ---------------------------------------------------------

        self.declare_parameter(
            "enabled",
            True,
        )

        self.declare_parameter(
            "default_response_timeout_sec",
            5.0,
        )

        self.declare_parameter(
            "speech_start_timeout_sec",
            15.0,
        )

        self.declare_parameter(
            "speech_completion_timeout_sec",
            30.0,
        )
        
        self.declare_parameter(
            "speech_status",
            "/explainability/speech_status",
        )

        self.declare_parameter(
            "mock_mode",
            True,
        )

        self.declare_parameter("mock_response_enabled", False)
        self.declare_parameter("mock_response_intent", "ALLOW_PASSAGE")
        self.declare_parameter("mock_response_delay_sec", 2.0)

        # ---------------------------------------------------------
        # Parameters values
        # ---------------------------------------------------------

        self.enabled = bool(
            self.get_parameter("enabled").value
        )

        self.default_response_timeout_sec = float(
            self.get_parameter(
                "default_response_timeout_sec"
            ).value
        )

        self.mock_mode = bool(
            self.get_parameter("mock_mode").value
        )

        self.simulate_human_intent_srv = self.create_service(
            SimulateHumanIntent,
            "/hri/simulate_human_intent",
            self._simulate_human_intent_callback,
        )

        self.mock_response_enabled = (
            self.get_parameter("mock_response_enabled")
            .get_parameter_value()
            .bool_value
        )

        self.mock_response_intent = (
            self.get_parameter("mock_response_intent")
            .get_parameter_value()
            .string_value
        )

        self.speech_status = (
            self.get_parameter("speech_status")
            .get_parameter_value()
            .string_value
        )

        self.mock_response_delay_sec = (
            self.get_parameter("mock_response_delay_sec")
            .get_parameter_value()
            .double_value
        )

        self.speech_start_timeout_sec = float(
            self.get_parameter(
                "speech_start_timeout_sec"
            ).value
        )
        self.speech_completion_timeout_sec = float(
            self.get_parameter(
                "speech_completion_timeout_sec"
            ).value
        )

        self.pending_close_control = None
        self.closed_control_ids = {}
        self.audio_uncertain_decision_id = ""


        if self.speech_completion_timeout_sec <= 0.0:
            raise ValueError(
                "speech_completion_timeout_sec must be strictly positive."
            )

        if self.speech_start_timeout_sec <= 0.0:
            raise ValueError(
                "speech_start_timeout_sec must be strictly positive."
            )

        # ---------------------------------------------------------
        # HRI Core
        # ---------------------------------------------------------

        self.hri_manager = HRIManager(
            min_confidence=0.0,
        )

        # ---------------------------------------------------------
        # State
        # ---------------------------------------------------------

        self.active_interaction_id = ""
        self.active_person_id = ""
        self.active_request = None
        self.active_language = "en"
        self.active_dialogue_request_id = ""

        # Human-response timing starts ONLY when the human can actually answer.
        self.interaction_started_at = None
        self.waiting_for_human = False

        # Robot-initiated dialogue has a speech barrier.
        self.waiting_for_speech_completion = False
        self.pending_speech_decision_id = ""

        self.mock_response_at = None
        self.pending_speech_requested_at = None
        self.pending_speech_started = False
        self.pending_speech_started_at = None
        # Turn-taking cue.
        #
        # After robot speech completes:
        #
        #   anti-echo guard
        #       -> short audio cue
        #       -> human response window
        #
        # The human timeout starts only after the cue has completed.
        self.response_guard_sec = 0.20

        # Use an audible cue before opening the human response window.
        self.declare_parameter("turn_cue_enabled", True)
        self.turn_cue_enabled = bool(
            self.get_parameter("turn_cue_enabled").value
        )

        self.turn_cue_pending = False
        self.turn_cue_ready_at = None
        self.turn_cue_decision_id = ""
        self._turn_cue_authorized = False

        # ---------------------------------------------------------
        # ROS interfaces
        # ---------------------------------------------------------

        self.dialogue_request_sub = self.create_subscription(
            DialogueRequest,
            "/hri/dialogue_request",
            self._dialogue_request_callback,
            10,
        )

        self.speech_status_sub = self.create_subscription(
            SpeechStatus,
            self.speech_status,
            self._speech_status_callback,
            10,
        )

        self.human_intent_pub = self.create_publisher(
            HumanIntent,
            "/hri/human_intent",
            10,
        )

        self.speech_request_pub = self.create_publisher(
            ExplanationText,
            "/explainability/text",
            10,
        )
        
        self._initialize_audio_turn_protocol()

        self.asr_listening_pub = self.create_publisher(
            Bool,
            "/hri/asr_listening_enabled",
            10,
        )
        
        self.speech_cancel_pub = self.create_publisher(
            SpeechCancelRequest,
            "/explainability/speech_cancel_request",
            10,
        )

        # HRI owns the microphone turn-taking authority.
        #
        # Safe initial state:
        # the human microphone must remain disabled until an explicit
        # human-response window is opened.
        self.asr_listening_enabled = False

        # ---------------------------------------------------------
        # ROS4HRI backend
        # ---------------------------------------------------------

        self.ros4hri_backend = ROS4HRIBackend(
            self,
            self._ros4hri_intent_callback,
        )

        # ---------------------------------------------------------
        # Timer
        # ---------------------------------------------------------

        self.timer = self.create_timer(
            0.1,
            self._timer_callback,
        )

        self.get_logger().info(
            "HRI Manager started."
        )

        self.get_logger().info(
            f"HRI enabled: {self.enabled}"
        )

        self.get_logger().info(
            f"Mock mode: {self.mock_mode}"
        )

        self.speech_cancel_pub = self.create_publisher(
            SpeechCancelRequest,
            "/explainability/speech_cancel_request",
            10
        )

        self.dialogue_control_result_pub = (
            self.create_publisher(
                String,
                "/hri/dialogue_control_result",
                10
            )
        )

        self.dialogue_control_sub = self.create_subscription(
            String,
            "/hri/dialogue_control",
            self._dialogue_control_callback,
            10
        )

    # =========================================================
    # DialogueRequest
    # =========================================================

    def _set_asr_listening(self, enabled):
        """Publish gate diagnostics and send scoped ASR commands."""
        enabled = bool(enabled)

        self.asr_listening_enabled = enabled

        self.asr_listening_pub.publish(
            Bool(data=enabled)
        )

        if not hasattr(self, "_audio_turn"):
            return

        if enabled and self._audio_turn:
            self._audio_phase = "OPENING"

            self._audio_deadline = (
                time.monotonic()
                + self.asr_ready_timeout_sec
            )

            self._send_audio_command("LISTEN")

        elif not enabled:
            if self._audio_turn:
                self._send_audio_command("CANCEL")

            self._audio_turn = None
            self._audio_phase = "IDLE"
            self._audio_deadline = None
    def _publish_dialogue_rejection(self, request, reason):
        interaction_id = str(
            request.interaction_id or ""
        ).strip()

        request_id = str(
            request.request_id or ""
        ).strip()

        if not interaction_id or not request_id:
            self.get_logger().error(
                "[HRI] Cannot report rejection without "
                "interaction_id and request_id."
            )
            return

        msg = HumanIntent()

        msg.header.stamp = self.get_clock().now().to_msg()

        msg.interaction_id = interaction_id
        msg.request_id = request_id
        msg.person_id = str(request.person_id or "")

        msg.intent = "UNKNOWN"
        msg.direction = "NONE"
        msg.target = ""
        msg.constraint = ""
        msg.confidence = 0.0
        msg.utterance = str(reason)
        msg.source = "DIALOGUE_REQUEST_REJECTED"
        msg.modality = "INTERNAL"

        self.human_intent_pub.publish(msg)

        self.get_logger().warning(
            "[HRI] DIALOGUE_REJECTED | "
            f"interaction={interaction_id} | "
            f"request={request_id} | reason={reason}"
        )

    def _dialogue_request_callback(self, request):
        """
        Accept dialogue turns from CollaborationManager.

        A repeated interaction_id is permitted for a new
        dialogue turn, but not for a duplicate active request.

        Do not replace a robot speech transaction whose
        physical audio state remains unconfirmed.
        """
        if not request.request_id:
            self._reject_dialogue_request(
                request,
                "MISSING_TURN_ID"
            )
            return

        # Reject a previously closed exact turn, but allow
        # a new request_id within the same interaction.
        if any(
            c["interaction_id"] == request.interaction_id
            and c.get("request_id") == request.request_id
            for c in self.closed_control_ids.values()
        ):
            self._reject_dialogue_request(
                request,
                "DIALOGUE_TURN_ALREADY_CLOSED"
            )
            return

        if not self.enabled:
            self._reject_dialogue_request(
                request,
                "HRI_DISABLED"
            )
            return

        interaction_id = str(
            request.interaction_id or ""
        ).strip()

        if not interaction_id:
            self.get_logger().warning(
                "[HRI] DialogueRequest without interaction_id."
            )
            return

        if getattr(
            self,
            "audio_uncertain_decision_id",
            ""
        ):
            self._reject_dialogue_request(
                request,
                "AUDIO_TERMINATION_UNCONFIRMED"
            )

            self._set_asr_listening(False)

            self.get_logger().error(
                "[HRI] DIALOGUE_BLOCKED_AUDIO_UNCERTAIN | "
                f"decision={self.audio_uncertain_decision_id}"
            )
            return

        interaction_active = (
            self.waiting_for_human
            or self.waiting_for_speech_completion
            or self.turn_cue_pending
        )

        same_interaction = (
            interaction_id == self.active_interaction_id
        )

        same_request = bool(
            self.active_request is not None
            and same_interaction
            and self.active_request.request_id
            == request.request_id
        )

        if interaction_active and same_request:
            self.get_logger().debug(
                "[HRI] Duplicate active dialogue turn ignored."
            )
            return

        if interaction_active and not same_interaction:
            self.get_logger().warning(
                "[HRI] Another interaction is already active | "
                f"active={self.active_interaction_id} | "
                f"incoming={interaction_id}"
            )
            return

        if (
            self.waiting_for_speech_completion
            or self.turn_cue_pending
        ):
            self.get_logger().warning(
                "[HRI] New dialogue turn rejected while "
                "robot speech is in progress."
            )
            return

        self._set_asr_listening(False)

        self.active_interaction_id = interaction_id
        self.active_person_id = request.person_id
        self.active_request = request

        self.active_language = normalize_dialogue_language(
            getattr(request, "language", "en")
        )

        self.interaction_started_at = None
        self.waiting_for_human = False
        self.waiting_for_speech_completion = False

        self.pending_speech_decision_id = ""
        self.pending_speech_requested_at = None
        self.pending_speech_started = False
        self.pending_speech_started_at = None

        self.turn_cue_pending = False
        self.turn_cue_ready_at = None
        self.turn_cue_decision_id = ""

        self.mock_response_at = None
        self.unrecognized_response_received = False
        self.last_unrecognized_utterance = ""

        response_timeout = float(
            request.response_timeout
        )

        if response_timeout <= 0.0:
            response_timeout = (
                self.default_response_timeout_sec
            )

        self.active_response_timeout = response_timeout

        self.get_logger().info(
            "[HRI] DIALOGUE_TURN_ACCEPTED | "
            f"interaction={interaction_id} | "
            f"type={request.request_type} | "
            f"robot_initiated={request.robot_initiated}"
        )

        if request.robot_initiated:
            question = str(
                request.question or ""
            ).strip()

            if not question:
                self._publish_human_intent(
                    intent="UNKNOWN",
                    direction="NONE",
                    target="",
                    constraint="",
                    confidence=0.0,
                    utterance="",
                    source="ROBOT_SPEECH_FAILED",
                    modality="INTERNAL"
                )
                self._reset_interaction()
                return

            decision_id = (
                f"HRI_{interaction_id}_"
                f"{uuid.uuid4().hex[:12]}"
            )

            self.pending_speech_decision_id = decision_id
            self.waiting_for_speech_completion = True

            self.pending_speech_started = False
            self.pending_speech_started_at = None
            self.pending_speech_requested_at = time.monotonic()

            self._speak_question(
                question,
                decision_id=decision_id,
                language=self.active_language,
            )

            # Prepare capture during robot speech.
            # PREPARE does not authorize listening or audio retention.
            if not self.mock_response_enabled:
                self._prepare_audio_turn(with_cue=True)

            return

        self._open_human_response_window()

    def _speech_status_callback(self, msg):
        """Handle robot speech completion and authorize the human turn."""

        decision_id = str(msg.decision_id or "").strip()
        state = str(msg.state or "").upper()

        if not decision_id:
            return

        retryable = (
            state == "FAILED"
            and str(msg.detail).startswith("AUDIO_RETRYABLE|")
        )
        terminal_safe = state in {"COMPLETED", "CANCELED"} or retryable

        # Resolve an earlier cancellation or uncertain audio termination.
        if decision_id == self.audio_uncertain_decision_id:
            if not terminal_safe:
                if state == "FAILED" and self.pending_close_control is not None:
                    self._publish_close_result(
                        self.pending_close_control,
                        "TERMINATION_UNCONFIRMED",
                    )
                return

            self.audio_uncertain_decision_id = ""

            if self.pending_close_control is not None:
                command = self.pending_close_control
                self.pending_close_control = None
                self._reset_interaction()
                self._publish_close_result(command, "CLOSED")
            else:
                self._report_audio_event("AUDIO_TERMINATED")
                self._reset_interaction()

            return

        # Ignore playback events belonging to another request.
        if (
            not self.active_interaction_id
            or not self.waiting_for_speech_completion
            or decision_id != self.pending_speech_decision_id
        ):
            return

        if state == "STARTED":
            if not self.pending_speech_started:
                self.pending_speech_started = True
                self.pending_speech_started_at = time.monotonic()
            return

        if state == "COMPLETED":
            is_cue = decision_id == self.turn_cue_decision_id

            self.pending_speech_decision_id = ""
            self.waiting_for_speech_completion = False
            self.pending_speech_requested_at = None
            self.pending_speech_started = False
            self.pending_speech_started_at = None

            if is_cue:
                # The audible cue has finished. The person may now speak.
                self.turn_cue_decision_id = ""
                self._open_human_response_window()

            elif self.turn_cue_enabled:
                # The question has finished. Use the existing guard and cue.
                self.turn_cue_pending = True
                self.turn_cue_ready_at = (
                    time.monotonic() + self.response_guard_sec
                )

            else:
                # The question has finished. Release the cue requirement
                # for this turn before checking microphone readiness.
                self._audio_with_cue = False
                self._turn_cue_authorized = False
                self._open_human_response_window()

            return

        if state in {"FAILED", "CANCELED"}:
            if terminal_safe:
                self._report_audio_event("ROBOT_SPEECH_FAILED")
                self._report_audio_event("AUDIO_TERMINATED")
                self._reset_interaction()
            else:
                self._request_audio_reconciliation("ROBOT_SPEECH_FAILED")
                
    def _request_pending_speech_cancellation(self):
        """Request cancellation without assuming audio termination."""
        decision_id = str(
            self.pending_speech_decision_id or ""
        ).strip()

        if not decision_id:
            return

        self.audio_uncertain_decision_id = decision_id
        self._set_asr_listening(False)

        self.audio_uncertain_request_id = (
            self.active_dialogue_request_id
        )

        request = SpeechCancelRequest()
        request.decision_id = decision_id

        self.speech_cancel_pub.publish(request)


    def _request_turn_cue(self):
        """Emit the human-turn cue only after speech and capture readiness."""
        self.turn_cue_pending = False
        self.turn_cue_ready_at = None

        if not self.active_interaction_id:
            return

        if self.mock_response_enabled:
            self._emit_turn_cue()
            return

        # Called after robot speech COMPLETED and anti-echo guard.
        self._turn_cue_authorized = True

        if self._audio_turn is None:
            # Defensive fallback if preparation was not started.
            self._prepare_audio_turn(with_cue=True)
            return

        if self._audio_phase == "READY":
            self._emit_turn_cue()
            return

        if self._audio_phase == "PREPARING":
            # READY callback will emit the cue.
            return

        self.get_logger().error(
            "[HRI ASR] Unexpected audio phase before turn cue | "
            f"phase={self._audio_phase}"
        )

        self._report_audio_event(
            "DIALOGUE_REQUEST_REJECTED"
        )
        self._reset_interaction()

    def _emit_turn_cue(self):
        """Emit the cue while human-audio retention remains disabled."""
        if not self.active_interaction_id:
            return
        self._turn_cue_authorized = False
        self.turn_cue_pending = False
        self.turn_cue_ready_at = None

        self.turn_cue_decision_id = (
            f"HRI_TURN_CUE_{self.active_interaction_id}_"
            f"{uuid.uuid4().hex[:12]}"
        )

        self.waiting_for_speech_completion = True
        self.pending_speech_decision_id = (
            self.turn_cue_decision_id
        )

        self.pending_speech_requested_at = time.monotonic()
        self.pending_speech_started = False
        self.pending_speech_started_at = None

        msg = ExplanationText()

        msg.header.stamp = self.get_clock().now().to_msg()
        msg.decision_id = self.turn_cue_decision_id

        # The speech manager handles TURN_CUE as a tone.
        msg.language = "en"
        msg.text = ""
        msg.fallback_language = "en"
        msg.fallback_text = ""

        msg.provider = "g1_hri"
        msg.model = ""
        msg.used_fallback = False
        msg.used_cache = False
        msg.speech_kind = "TURN_CUE"

        msg.context_stamp = msg.header.stamp
        msg.navigation_state = ""
        msg.navigation_goal_id = ""
        msg.navigation_active = False
        msg.social_conflict_active = True

        self.speech_request_pub.publish(msg)

        self.get_logger().info(
            "Published human turn cue request | "
            f"interaction={self.active_interaction_id} | "
            f"decision_id={self.turn_cue_decision_id}"
        )

    def _open_human_response_window(self):
        """Enable recognition only after capture readiness and cue completion."""
        if not self.active_interaction_id:
            self.get_logger().warning(
                "Cannot open a response window without an active interaction"
            )

            self._set_asr_listening(False)
            return

        if (
            not self.mock_response_enabled
            and self._audio_phase != "READY"
        ):
            # Passive initiative has no preceding robot question or cue.
            self._prepare_audio_turn(
                with_cue=False,
            )
            return

        self.waiting_for_speech_completion = False
        self.pending_speech_decision_id = ""
        self.pending_speech_requested_at = None
        self.pending_speech_started = False
        self.pending_speech_started_at = None

        self.turn_cue_pending = False
        self.turn_cue_ready_at = None
        self.turn_cue_decision_id = ""

        self.waiting_for_human = True

        if self.mock_response_enabled:
            self.interaction_started_at = time.monotonic()

            self.mock_response_at = (
                self.interaction_started_at
                + self.mock_response_delay_sec
            )

        else:
            # The onset deadline starts on the LISTENING acknowledgment.
            self.interaction_started_at = None
            self.mock_response_at = None

        self._set_asr_listening(True)

        self.get_logger().info(
            "[HRI ASR] Human response window requested | "
            f"interaction={self.active_interaction_id}"
        )

    def _handle_mock_response(self):
        """Generate a simulated human response for testing."""

        self._publish_human_intent(
            intent=self.mock_response_intent,
            direction="NONE",
            target="",
            constraint="",
            confidence=0.95,
            utterance="Mock human response",
            source="RESPONSE_TO_ROBOT",
            modality="INTERNAL",
        )

        self.get_logger().info(
            "Mock human response published: "
            f"interaction={self.active_interaction_id}, "
            f"intent={self.mock_response_intent}"
        )

        self._reset_interaction()

    # =========================================================
    # Robot speech
    # =========================================================


    def _build_english_dialogue_fallback(self) -> str:
        """Build English fallback from explicit dialogue semantics."""
        request = self.active_request

        if request is None:
            return ""

        language = normalize_dialogue_language(
            getattr(request, "language", "en")
        )

        question = str(
            request.question or ""
        ).strip()

        request_type = str(
            request.request_type or ""
        ).strip().upper()

        context = str(
            request.context or ""
        ).strip().upper()

        if context == "AUTONOMOUS_ADAPTATION_PERMISSION":
            return (
                "I cannot guarantee passage on the requested side. "
                "May I freely look for another route "
                "while respecting social distance?"
            )

        if request_type in {
            "DIRECTION_REFERENCE_CLARIFICATION",
            "DIRECTION_FINAL_CONFIRMATION",
        }:
            direction = str(
                request.direction_to_confirm or ""
            ).strip().upper()

            reference = str(
                request.directional_reference or ""
            ).strip().upper()

            resolved = str(
                request.resolved_robot_direction or ""
            ).strip().upper()

            retry = int(request.directional_retry_index)

            if (
                direction not in {"LEFT", "RIGHT"}
                or retry not in {0, 1, 2}
            ):
                return ""

            # Already exact when the primary language is English.
            if language == "en":
                return question

            side = direction.lower()

            if request_type == "DIRECTION_REFERENCE_CLARIFICATION":
                if (
                    reference != "AMBIGUOUS"
                    or resolved != "NONE"
                ):
                    return ""

                questions = (
                    (
                        "To make sure I understand your instruction "
                        "correctly, do you mean "
                        f"your {side} or mine?"
                    ),
                    (
                        "Excuse me, I didn't receive your answer. "
                        "Could you please clarify whether you mean "
                        f"your {side} or mine?"
                    ),
                    (
                        "Thank you for your patience. "
                        "To avoid any misunderstanding, could you "
                        "please confirm whether you mean "
                        f"your {side} or mine?"
                    ),
                )

                return questions[retry]

            if reference == "PERSON":
                expected = (
                    "RIGHT" if direction == "LEFT"
                    else "LEFT"
                )

                if resolved != expected:
                    return ""

                statement = (
                    f"You would like me to pass on your {side}. "
                    "Assuming we are facing each other, "
                    f"that corresponds to my {resolved.lower()}. "
                )

            elif reference == "ROBOT":
                if resolved != direction:
                    return ""

                statement = (
                    f"You would like me to pass on my {side}. "
                )

            else:
                return ""

            prefixes = (
                "To make sure I respect your instruction, ",
                "Excuse me, I didn't receive your confirmation. ",
                "Thank you for your patience. ",
            )

            endings = (
                "Is that what you would like?",
                "Could you please confirm that this is correct?",
                "Could you please confirm your preference one last time?",
            )

            return (
                prefixes[retry]
                + statement
                + endings[retry]
            )

        if request_type == "PASSAGE_CLARIFICATION":
            return get_dialogue_text(
                "passage",
                "en",
            )

        if request_type == "INTENT_CONFIRMATION":
            intent = str(
                request.intent_to_confirm or ""
            ).strip().upper()

            direction = str(
                request.direction_to_confirm or "NONE"
            ).strip().upper()

            if intent == "DENY_PASSAGE":
                key = "confirm_deny"

            elif direction == "LEFT":
                key = "confirm_allow_left"

            elif direction == "RIGHT":
                key = "confirm_allow_right"

            else:
                key = "confirm_allow"

            return get_dialogue_text(key, "en")

        if request_type == "ALTERNATIVE_PROPOSAL":
            direction = str(
                request.direction_to_confirm or "NONE"
            ).strip().upper()

            if direction == "LEFT":
                key = "alternative_left"

            elif direction == "RIGHT":
                key = "alternative_right"

            else:
                self.get_logger().warning(
                    "Cannot construct English fallback for "
                    "ALTERNATIVE_PROPOSAL without LEFT/RIGHT."
                )
                return ""

            return get_dialogue_text(key, "en")

        self.get_logger().warning(
            "No English dialogue fallback mapping for "
            f"request_type={request_type!r}"
        )

        return ""

    def _speak_question(
        self,
        question,
        *,
        decision_id,
        language,
    ):
        """
        Publish one exact dialogue speech transaction through the Common Speech
        API.

        The preferred dialogue text keeps its configured language.

        An English fallback is rebuilt from the same structured DialogueRequest
        using g1_social_common. The Speech Manager may use it only when the
        preferred Piper voice is unavailable.
        """

        question = str(
            question or ""
        ).strip()

        if not question:
            self.get_logger().warning(
                "Cannot publish an empty dialogue question."
            )
            return

        if not self.active_interaction_id:
            self.get_logger().warning(
                "Cannot publish dialogue speech without "
                "an active interaction."
            )
            return

        decision_id = str(
            decision_id or ""
        ).strip()

        if not decision_id:
            self.get_logger().error(
                "Cannot publish dialogue speech without decision_id."
            )
            return

        normalized_language = (
            normalize_dialogue_language(
                language
            )
        )

        english_fallback = (
            self._build_english_dialogue_fallback()
        )

        msg = ExplanationText()

        msg.header.stamp = (
            self.get_clock().now().to_msg()
        )

        msg.decision_id = decision_id

        msg.language = (
            normalized_language
        )

        msg.text = question

        msg.fallback_language = "en"

        msg.fallback_text = (
            english_fallback
            if english_fallback
            else (
                question
                if normalized_language == "en"
                else ""
            )
        )

        msg.provider = "g1_hri"
        msg.model = ""

        msg.used_fallback = False
        msg.used_cache = False

        msg.speech_kind = "DIALOGUE"

        msg.context_stamp = (
            msg.header.stamp
        )

        msg.navigation_state = ""
        msg.navigation_goal_id = ""

        msg.navigation_active = False
        msg.social_conflict_active = True

        self.speech_request_pub.publish(
            msg
        )

        self.get_logger().info(
            "Published dialogue speech request | "
            f"decision_id={msg.decision_id} | "
            f"interaction={self.active_interaction_id} | "
            f"language={msg.language} | "
            f"english_fallback="
            f"{'YES' if msg.fallback_text else 'NO'}"
        )

        return msg.decision_id

    # =========================================================
    # Timer
    # =========================================================

    def _timer_callback(self):
        """
        Supervise HRI with bounded audio reconciliation.

        Human silence and audio failure remain distinct.

        Never invent CLOSED when audio termination is
        unconfirmed.

        Report an unresolved CLOSE after 10 seconds so
        CollaborationManager can execute its technical
        recovery policy.

        Keep ASR disabled while audio is uncertain.
        """
        now = time.monotonic()

        # ==========================================
        # A. AUDIO RECONCILIATION
        # ==========================================
        pending = self.pending_close_control

        uncertain_id = str(
            self.audio_uncertain_decision_id or ""
        ).strip()

        if pending is not None or uncertain_id:
            self._set_asr_listening(False)

            current_key = (
                str(
                    pending.get("close_id") or ""
                )
                if pending is not None
                else "",
                uncertain_id,
            )

            if getattr(
                self,
                "_audio_watch_key",
                None,
            ) != current_key:
                self._audio_watch_key = current_key
                self._audio_watch_started_at = now
                self._audio_watch_reported = False
                self._audio_watch_last_cancel_at = 0.0

            elapsed = (
                now - self._audio_watch_started_at
            )

            # Retry cancellation while terminal audio
            # evidence is still missing.
            if (
                uncertain_id
                and elapsed < 10.0
                and (
                    now - self._audio_watch_last_cancel_at
                    >= 2.0
                )
            ):
                cancel = SpeechCancelRequest()
                cancel.decision_id = uncertain_id
                cancel.reason = "AUDIO_RECONCILIATION"

                self.speech_cancel_pub.publish(cancel)

                self._audio_watch_last_cancel_at = now

            # Report the unresolved transaction once.
            if (
                elapsed >= 10.0
                and not self._audio_watch_reported
            ):
                self._audio_watch_reported = True

                self.get_logger().error(
                    "[HRI AUDIO] Reconciliation timeout | "
                    f"decision={uncertain_id} | "
                    "audio_terminated=False | "
                    "navigation_decision_owner="
                    "COLLABORATION_MANAGER"
                )

                if pending is not None:
                    self._publish_close_result(
                        pending,
                        "TERMINATION_UNCONFIRMED"
                    )

            # Keep the transaction available for a
            # possible late, exact terminal status.
            #
            # Do not claim CLOSED.
            # Do not reopen ASR.
            return

        # A previous uncertain transaction has been
        # resolved by the exact speech-status callback.
        self._audio_watch_key = None
        self._audio_watch_started_at = None
        self._audio_watch_reported = False
        self._audio_watch_last_cancel_at = 0.0

        # ==========================================
        # B. MOCK RESPONSE
        # ==========================================
        if (
            self.mock_response_enabled
            and self.waiting_for_human
            and self.mock_response_at is not None
            and now >= self.mock_response_at
        ):
            self._handle_mock_response()
            return

        # ==========================================
        # C. HUMAN TURN CUE
        # ==========================================
        if (
            self.turn_cue_pending
            and self.turn_cue_ready_at is not None
            and now >= self.turn_cue_ready_at
        ):
            self._request_turn_cue()
            return

        # ==========================================
        # D. ROBOT SPEECH
        # ==========================================
        if self.waiting_for_speech_completion:
            started_at = (
                self.pending_speech_started_at
                if self.pending_speech_started
                else self.pending_speech_requested_at
            )

            timeout = (
                self.speech_completion_timeout_sec
                if self.pending_speech_started
                else self.speech_start_timeout_sec
            )

            if (
                started_at is not None
                and now - started_at >= timeout
            ):
                self._request_audio_reconciliation(
                    "ROBOT_SPEECH_TIMEOUT"
                )

            return

        # ==========================================
        # E. MOCK HUMAN TIMEOUT
        # ==========================================
        if self.mock_response_enabled:
            if (
                self.waiting_for_human
                and self.interaction_started_at is not None
                and now - self.interaction_started_at
                    >= self.active_response_timeout
            ):
                self._handle_timeout()

            return

        # ==========================================
        # F. REAL AUDIO TURN DEADLINE
        # ==========================================
        if (
            not self._audio_turn
            or self._audio_deadline is None
            or now < self._audio_deadline
        ):
            return

        # A genuine absence of speech is human silence.
        if self._audio_phase == "LISTENING":
            self._handle_timeout()
            return

        # Every other audio-phase timeout is technical,
        # not human silence and not a refusal.
        self.get_logger().error(
            "[HRI ASR] Phase timeout | "
            f"phase={self._audio_phase}"
        )

        self._report_audio_event(
            "DIALOGUE_REQUEST_REJECTED"
        )

        self._reset_interaction()
        
    # =========================================================
    # Timeout
    # =========================================================

    def _handle_timeout(self):
        """
        End the human response window.

        Distinguish real silence from an unrecognized
        human response using the existing manager contract.
        """

        if not self.active_interaction_id:
            return

        if not self.waiting_for_human:
            return

        unrecognized = bool(
            getattr(
                self,
                "unrecognized_response_received",
                False,
            )
        )

        utterance = str(
            getattr(
                self,
                "last_unrecognized_utterance",
                "",
            )
        ).strip()

        self._set_asr_listening(False)

        if unrecognized:

            self.get_logger().warning(
                "[HRI] HUMAN_RESPONSE_NOT_UNDERSTOOD | "
                f"interaction={self.active_interaction_id}"
            )

            self._publish_human_intent(
                intent="REJECT",
                direction="NONE",
                target="",
                constraint="",
                confidence=0.0,
                utterance=utterance,
                source="RESPONSE_TO_ROBOT",
                modality="INTERNAL",
            )

        else:

            self.get_logger().info(
                "[HRI] HUMAN_RESPONSE_TIMEOUT | "
                f"interaction={self.active_interaction_id}"
            )

            self._publish_human_intent(
                intent="UNKNOWN",
                direction="NONE",
                target="",
                constraint="",
                confidence=0.0,
                utterance="",
                source="HUMAN_RESPONSE_TIMEOUT",
                modality="INTERNAL",
            )

        self._reset_interaction()
        
    # =========================================================
    # ROS4HRI Intent
    # =========================================================

    def _ros4hri_intent_callback(
        self,
        intent,
        data,
        confidence,
        source,
        utterance,
        modality,
    ):
        """
        Process the current ROS4HRI response.

        During a directional dialogue, distinguish:
          - "No" = negative answer;
          - "No, on your left/right" = correction of the
            robot's directional interpretation.

        Preserve the original utterance and the exact
        interaction/request identifiers.

        This method never authorizes navigation.
        """
        if not self.enabled:
            return

        if (
            not self.mock_response_enabled
            and self._audio_phase != "RESULT"
        ):
            self.get_logger().warning(
                "[HRI] Intent ignored outside validated ASR result"
            )
            return

        if (
            not self.waiting_for_human
            or not self.active_interaction_id
        ):
            self.get_logger().warning(
                "[HRI] Intent ignored: no human response window"
            )
            return

        # ----------------------------------------------
        # 1. Preserve normal intent processing
        # ----------------------------------------------

        hri_intent = self.hri_manager.process_intent(
            intent=intent,
            data=data,
            confidence=confidence,
            source=source,
            modality=modality,
            interaction_id=self.active_interaction_id,
            person_id=self.active_person_id,
            utterance=utterance,
        )

        normalized_intent = str(
            hri_intent.intent or "UNKNOWN"
        ).strip().upper()

        output_direction = str(
            hri_intent.direction or "NONE"
        ).strip().upper()

        # ----------------------------------------------
        # 2. Detect corrections only in directional turns
        # ----------------------------------------------

        request_type = (
            str(self.active_request.request_type or "")
            .strip()
            .upper()
            if self.active_request is not None
            else ""
        )

        directional_turn = request_type in {
            "DIRECTION_REFERENCE_CLARIFICATION",
            "DIRECTION_FINAL_CONFIRMATION",
        }

        original_utterance = str(
            hri_intent.utterance or utterance or ""
        ).strip()

        if directional_turn:
            normalized_text = re.sub(
                r"[,.!?;:]+",
                " ",
                original_utterance.lower(),
            )

            normalized_text = " ".join(
                normalized_text.split()
            )

            # The negation corrects the proposed side.
            # It is not a denial of passage.
            #
            # Match the complete utterance to avoid
            # confusing longer, unrelated sentences.
            correction = re.fullmatch(
                r"no\s+(?:on\s+)?your\s+(left|right)",
                normalized_text,
            )

            if correction is not None:
                corrected_direction = (
                    correction.group(1).upper()
                )

                normalized_intent = (
                    "REQUEST_LEFT"
                    if corrected_direction == "LEFT"
                    else "REQUEST_RIGHT"
                )

                output_direction = corrected_direction

                self.get_logger().info(
                    "[HRI DIRECTION] Contextual correction | "
                    f"request_type={request_type} | "
                    f"direction={corrected_direction} | "
                    "reference=ROBOT | "
                    "passage_denied=False"
                )

        # ----------------------------------------------
        # 3. Preserve UNKNOWN handling
        # ----------------------------------------------

        if normalized_intent == "UNKNOWN":
            self.unrecognized_response_received = True

            if original_utterance:
                self.last_unrecognized_utterance = (
                    original_utterance
                )

            self.get_logger().warning(
                "[HRI] HUMAN_RESPONSE_UNRECOGNIZED | "
                f"interaction={self.active_interaction_id} | "
                f"utterance={original_utterance!r}"
            )

            self._handle_timeout()
            return

        # ----------------------------------------------
        # 4. Publish the recognized response
        # ----------------------------------------------

        interaction_id = self.active_interaction_id

        self.waiting_for_human = False
        self._set_asr_listening(False)

        self._publish_human_intent(
            intent=normalized_intent,
            direction=output_direction,
            target=hri_intent.target,
            constraint=hri_intent.constraint,
            confidence=hri_intent.confidence,
            utterance=original_utterance,
            source=hri_intent.source,
            modality=hri_intent.modality,
        )

        self.get_logger().info(
            "[HRI] HUMAN_INTENT_ACCEPTED | "
            f"interaction={interaction_id} | "
            f"intent={normalized_intent} | "
            f"direction={output_direction}"
        )

        self._reset_interaction()

    # =========================================================
    # HumanIntent
    # =========================================================
    def _simulate_human_intent_callback(
        self,
        request,
        response,
    ):
        """
        Test-only service.

        Simulates a human response for the currently
        active HRI interaction.

        This interface is NOT part of the final HRI architecture.
        """

        if not self.enabled:
            response.success = False
            response.message = (
                "HRI Manager is disabled."
            )
            return response

        if not self.waiting_for_human:
            response.success = False
            response.message = (
                "No active HRI interaction."
            )
            return response

        if (
            request.interaction_id
            != self.active_interaction_id
        ):
            response.success = False
            response.message = (
                "Interaction ID does not match the active "
                "HRI interaction."
            )
            return response

        if not request.intent:
            response.success = False
            response.message = (
                "Human intent cannot be empty."
            )
            return response

        self._publish_human_intent(
            intent=request.intent,
            direction=request.direction,
            target=request.target,
            constraint=request.constraint,
            confidence=request.confidence,
            utterance=request.utterance,
            source="RESPONSE_TO_ROBOT",
            modality="INTERNAL",
        )

        self.get_logger().info(
            "Simulated human response: "
            f"interaction={request.interaction_id}, "
            f"intent={request.intent}"
        )

        self._reset_interaction()

        response.success = True
        response.message = (
            f"HumanIntent '{request.intent}' published."
        )

        return response
    
    def _publish_human_intent(
        self,
        intent,
        direction,
        target,
        constraint,
        confidence,
        utterance,
        source,
        modality="UNKNOWN"
    ):
        """
        Publishes a HumanIntent associated with the
        current dialogue interaction.
        """
        if not self.active_interaction_id:
            self.get_logger().warn(
                "Cannot publish HumanIntent: "
                "no active interaction."
            )
            return

        msg = HumanIntent()

        msg.header.stamp = self.get_clock().now().to_msg()

        msg.interaction_id = (
            self.active_interaction_id
        )

        msg.request_id = (
            self.active_request.request_id
            if self.active_request is not None
            else ""
        )

        msg.person_id = self.active_person_id
        msg.intent = intent
        msg.direction = direction
        msg.target = target
        msg.constraint = constraint
        msg.confidence = float(confidence)
        msg.utterance = utterance
        msg.source = source
        msg.modality = modality

        self.human_intent_pub.publish(msg)

        self.get_logger().info(
            "Published HumanIntent: "
            f"interaction={msg.interaction_id}, "
            f"intent={msg.intent}, "
            f"direction={msg.direction}, "
            f"confidence={msg.confidence:.2f}, "
            f"source={msg.source}"
        )
    
    def _dialogue_control_callback(self, msg):
        try:
            command = json.loads(msg.data)
        except (TypeError, ValueError):
            return

        if (
            not isinstance(command, dict)
            or command.get("version") != 1
            or command.get("command") != "CLOSE"
        ):
            return

        if (
            not command.get("close_id")
            or not command.get("interaction_id")
        ):
            return

        cached = self.closed_control_ids.get(
            command["close_id"]
        )

        if cached is not None:
            if (
                cached["interaction_id"]
                == command["interaction_id"]
            ):
                self._publish_close_result(
                    cached,
                    "CLOSED"
                )
            return

        if self.pending_close_control is not None:
            return

        if (
            self.active_interaction_id
            and self.active_interaction_id
            != command["interaction_id"]
        ):
            self._publish_close_result(
                command,
                "TERMINATION_UNCONFIRMED"
            )
            return

        self._set_asr_listening(False)

        speech_id = (
            self.audio_uncertain_decision_id
            or self.pending_speech_decision_id
        )

        if not speech_id:
            self._reset_interaction()

            self._publish_close_result(
                command,
                "CLOSED"
            )
            return

        self.pending_close_control = dict(
            command,
            speech_id=speech_id
        )

        self.audio_uncertain_decision_id = speech_id

        self.waiting_for_human = False
        self.waiting_for_speech_completion = False
        self.turn_cue_pending = False

        cancel = SpeechCancelRequest()
        cancel.decision_id = speech_id

        cancel.reason = str(
            command.get(
                "reason",
                "DIALOGUE_CLOSE"
            )
        )

        self.speech_cancel_pub.publish(cancel)

    def _reject_dialogue_request(self, request, reason):
        result = HumanIntent()

        result.header.stamp = (
            self.get_clock().now().to_msg()
        )

        result.interaction_id = request.interaction_id
        result.request_id = request.request_id
        result.person_id = request.person_id

        result.intent = "UNKNOWN"
        result.direction = "NONE"
        result.source = "DIALOGUE_REQUEST_REJECTED"
        result.modality = "INTERNAL"
        result.utterance = reason

        self.human_intent_pub.publish(result)

    def _report_audio_event(self, source):
        self._publish_human_intent(
            intent="UNKNOWN",
            direction="NONE",
            target="",
            constraint="",
            confidence=0.0,
            utterance="",
            source=source,
            modality="INTERNAL",
        )

    def _request_audio_reconciliation(self, source):
        self.audio_uncertain_decision_id = (
            self.pending_speech_decision_id
        )

        self.waiting_for_speech_completion = False
        self.waiting_for_human = False
        self.turn_cue_pending = False

        self._set_asr_listening(False)

        self._report_audio_event(source)

        cancel = SpeechCancelRequest()

        cancel.decision_id = (
            self.audio_uncertain_decision_id
        )

        cancel.reason = source

        self.speech_cancel_pub.publish(cancel)

    def _publish_close_result(self, command, status):
        data = {
            "version": 1,
            "producer": "hri_manager",
            "close_id": command["close_id"],
            "interaction_id": command["interaction_id"],
            "request_id": command.get("request_id", ""),
            "status": status,
        }

        self.dialogue_control_result_pub.publish(
            String(data=json.dumps(data))
        )

        if status == "CLOSED":
            self.closed_control_ids[
                command["close_id"]
            ] = dict(command)

            while len(self.closed_control_ids) > 64:
                self.closed_control_ids.pop(
                    next(iter(self.closed_control_ids))
                )

            
                
    # =========================================================
    # Reset
    # =========================================================

    def _reset_interaction(self):
        """
        Reset a completed or abandoned interaction.

        Never clear an uncertain audio transaction merely
        because the dialogue state is being reset.
        """

        self._set_asr_listening(False)

        self.active_interaction_id = ""
        self.active_person_id = ""
        self.active_request = None
        self.active_language = "en"

        self.mock_response_at = None

        self.pending_speech_requested_at = None
        self.pending_speech_started = False
        self.pending_speech_started_at = None

        self.interaction_started_at = None
        self.waiting_for_human = False

        self.waiting_for_speech_completion = False
        self.pending_speech_decision_id = ""

        self.turn_cue_pending = False
        self.turn_cue_ready_at = None
        self.turn_cue_decision_id = ""
        self.audio_uncertain_request_id = ""

        self.active_response_timeout = (
            self.default_response_timeout_sec
        )

        self.unrecognized_response_received = False
        self.last_unrecognized_utterance = ""
        self.active_dialogue_request_id = ""

        # Deliberately NOT reset:
        # self.audio_uncertain_decision_id

    def _initialize_audio_turn_protocol(self):
        """Initialize the ASR turn protocol and optional audible cue."""

        # Default to the existing audible-cue behavior.
        if not self.has_parameter("turn_cue_enabled"):
            self.declare_parameter("turn_cue_enabled", True)

        self.turn_cue_enabled = self.get_parameter(
            "turn_cue_enabled"
        ).value

        defaults = {
            "asr_onset_timeout_sec": 5.0,
            "asr_ready_timeout_sec": 3.0,
            "asr_utterance_timeout_sec": 10.0,
            "asr_transcription_timeout_sec": 10.0,
            "asr_intent_timeout_sec": 3.0,
            "asr_voice_id": "pc_user",
        }

        for name, default in defaults.items():
            self.declare_parameter(name, default)
            value = self.get_parameter(name).value

            if name == "asr_voice_id":
                value = str(value).strip()
                if not value:
                    raise ValueError("asr_voice_id must not be empty")
            else:
                value = float(value)
                if not math.isfinite(value) or value <= 0.0:
                    raise ValueError(
                        f"{name} must be finite and positive"
                    )

            setattr(self, name, value)

        self._audio_turn = None
        self._audio_phase = "IDLE"
        self._audio_deadline = None
        self._audio_last_sequence = 0
        self._audio_with_cue = False

        self._audio_command_pub = self.create_publisher(
            String,
            "/hri/asr_turn_command",
            10,
        )

        self._audio_status_sub = self.create_subscription(
            String,
            "/hri/asr_turn_status",
            self._audio_turn_status_callback,
            20,
        )

        self._validated_speech_pub = self.create_publisher(
            LiveSpeech,
            f"/humans/voices/{self.asr_voice_id}/speech",
            10,
        )

    def _send_audio_command(self, command):
        """Send a command for the currently owned audio turn."""
        if not self._audio_turn:
            return

        payload = dict(self._audio_turn)
        payload.update(
            version=1,
            command=command,
        )

        self._audio_command_pub.publish(
            String(data=json.dumps(payload))
        )

    def _prepare_audio_turn(self, *, with_cue):
        """Request capture readiness for one exact dialogue request."""
        if (
            not self.active_interaction_id
            or self.active_request is None
        ):
            return

        if self._audio_turn:
            return

        request_id = str(
            self.active_request.request_id or ""
        ).strip()

        if not request_id:
            self._report_audio_event(
                "DIALOGUE_REQUEST_REJECTED"
            )
            self._reset_interaction()
            return

        self._audio_turn = {
            "turn_id": uuid.uuid4().hex,
            "interaction_id": str(self.active_interaction_id),
            "request_id": request_id,
        }

        self._audio_last_sequence = 0
        self._audio_with_cue = with_cue
        self._audio_phase = "PREPARING"

        self._audio_deadline = (
            time.monotonic()
            + self.asr_ready_timeout_sec
        )

        self._send_audio_command("PREPARE")

    def _audio_turn_status_callback(self, msg):
        """Accept progress and results only from the current audio turn."""
        try:
            data = json.loads(msg.data)

            if (
                not isinstance(data, dict)
                or data.get("version") != 1
            ):
                return

            if (
                not self._audio_turn
                or any(
                    data.get(key) != value
                    for key, value in self._audio_turn.items()
                )
            ):
                return

            sequence = int(data.get("sequence", 0))

            if sequence <= self._audio_last_sequence:
                return

            state = str(data.get("state", ""))

        except (ValueError, TypeError, AttributeError):
            return

        self._audio_last_sequence = sequence

        now = time.monotonic()
        phase = self._audio_phase

        if state == "READY" and phase == "PREPARING":
            self._audio_phase = "READY"
            self._audio_deadline = None

            self.get_logger().info(
                "[HRI ASR] Capture ready before turn cue"
            )

            if self._audio_with_cue:
                # The microphone may be ready while the robot is still speaking.
                # Wait until question COMPLETED and anti-echo guard elapsed.
                if self._turn_cue_authorized:
                    self._emit_turn_cue()
            else:
                self._open_human_response_window()

        elif state == "LISTENING" and phase == "OPENING":
            self._audio_phase = "LISTENING"
            self.interaction_started_at = now

            self._audio_deadline = now + min(
                self.active_response_timeout,
                self.asr_onset_timeout_sec,
            )

        elif (
            state == "SPEECH_STARTED"
            and phase == "LISTENING"
        ):
            self._audio_phase = "RECORDING"

            self._audio_deadline = (
                now + self.asr_utterance_timeout_sec
            )

        elif (
            state == "TRANSCRIBING"
            and phase == "RECORDING"
        ):
            self._audio_phase = "TRANSCRIBING"

            self._audio_deadline = (
                now + self.asr_transcription_timeout_sec
            )

        elif state == "RESULT" and phase == "TRANSCRIBING":
            text = str(data.get("text", "")).strip()

            try:
                confidence = float(
                    data.get("confidence", 0.0)
                )
            except (ValueError, TypeError):
                confidence = float("nan")

            if (
                not text
                or not math.isfinite(confidence)
            ):
                self.unrecognized_response_received = True
                self._handle_timeout()
                return

            self._audio_phase = "RESULT"

            self._audio_deadline = (
                now + self.asr_intent_timeout_sec
            )

            language = str(data.get("language", ""))

            speech = LiveSpeech()
            speech.header.stamp = self.get_clock().now().to_msg()
            speech.incremental = ""
            speech.final = text
            speech.confidence = max(
                0.0,
                min(confidence, 1.0),
            )

            speech.locale = {
                "fr": "fr_FR",
                "en": "en_US",
                "es": "es_ES",
            }.get(language, language)

            speech.language = language

            self._validated_speech_pub.publish(speech)

        elif (
            state == "EMPTY"
            and phase in {
                "RECORDING",
                "TRANSCRIBING",
            }
        ):
            self.unrecognized_response_received = True
            self._handle_timeout()

        elif state == "FAILED":
            self.get_logger().error(
                "[HRI ASR] Capture failed | "
                f"reason={data.get('reason', '')}"
            )

            self._report_audio_event(
                "DIALOGUE_REQUEST_REJECTED"
            )
            self._reset_interaction()

        else:
            return

        self.get_logger().info(
            "[HRI ASR] "
            f"state={state} | "
            f"turn={data.get('turn_id')}"
        )

def main(args=None):
    rclpy.init(args=args)

    node = HRIManagerNode()

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