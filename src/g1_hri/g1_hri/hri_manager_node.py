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

        self.turn_cue_pending = False
        self.turn_cue_ready_at = None
        self.turn_cue_decision_id = ""

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

    def _set_asr_listening(
        self,
        enabled: bool,
    ) -> None:
        """
        Control the PC ASR microphone according to the HRI turn-taking state.

        True:
            The human response window is open and microphone capture is allowed.

        False:
            Robot speech, anti-echo guard, turn cue or non-listening state is
            active. The ASR must stop/discard any current microphone capture.

        HRI is the authority for this gate.
        """

        enabled = bool(
            enabled
        )

        changed = (
            enabled
            != self.asr_listening_enabled
        )

        self.asr_listening_enabled = (
            enabled
        )

        msg = Bool()
        msg.data = enabled

        self.asr_listening_pub.publish(
            msg
        )

        if changed:
            self.get_logger().info(
                "ASR listening gate | "
                f"enabled={enabled} | "
                f"interaction="
                f"{self.active_interaction_id or 'NONE'}"
            )

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

        if any(
            c["interaction_id"] == request.interaction_id
            for c in self.closed_control_ids.values()
        ):
            self._reject_dialogue_request(
                request,
                "INTERACTION_ALREADY_CLOSED"
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
                language=self.active_language
            )
            return

        self._open_human_response_window()

    def _speech_status_callback(self, msg):
        decision_id = str(
            msg.decision_id or ""
        ).strip()

        state = str(
            msg.state or ""
        ).upper()

        retryable = (
            state == "FAILED"
            and str(msg.detail).startswith(
                "AUDIO_RETRYABLE|"
            )
        )

        terminal_safe = (
            state in {"COMPLETED", "CANCELED"}
            or retryable
        )

        if not decision_id:
            return

        if decision_id == self.audio_uncertain_decision_id:
            if not terminal_safe:
                if (
                    state == "FAILED"
                    and self.pending_close_control is not None
                ):
                    self._publish_close_result(
                        self.pending_close_control,
                        "TERMINATION_UNCONFIRMED"
                    )
                return

            self.audio_uncertain_decision_id = ""

            if self.pending_close_control is not None:
                command = self.pending_close_control
                self.pending_close_control = None

                self._reset_interaction()

                self._publish_close_result(
                    command,
                    "CLOSED"
                )
            else:
                self._report_audio_event(
                    "AUDIO_TERMINATED"
                )
                self._reset_interaction()

            return

        if (
            not self.active_interaction_id
            or not self.waiting_for_speech_completion
            or decision_id != self.pending_speech_decision_id
        ):
            return

        if state == "STARTED":
            if not self.pending_speech_started:
                self.pending_speech_started = True
                self.pending_speech_started_at = (
                    time.monotonic()
                )
            return

        if state == "COMPLETED":
            is_cue = (
                decision_id == self.turn_cue_decision_id
            )

            self.pending_speech_decision_id = ""
            self.waiting_for_speech_completion = False

            self.pending_speech_requested_at = None
            self.pending_speech_started = False
            self.pending_speech_started_at = None

            if is_cue:
                self.turn_cue_decision_id = ""
                self._open_human_response_window()
            else:
                self.turn_cue_pending = True
                self.turn_cue_ready_at = (
                    time.monotonic()
                    + self.response_guard_sec
                )
            return

        if state in {"FAILED", "CANCELED"}:
            if terminal_safe:
                # Failure and exact terminal evidence
                # belong to the same turn.
                self._report_audio_event(
                    "ROBOT_SPEECH_FAILED"
                )

                self._report_audio_event(
                    "AUDIO_TERMINATED"
                )

                self._reset_interaction()
            else:
                self._request_audio_reconciliation(
                    "ROBOT_SPEECH_FAILED"
                )

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
        """
        Request the short audio cue announcing that the human may speak.

        The cue is routed through the Common Speech Manager so that it can use
        exactly the same audio-output policy as robot speech:

            Unitree
                -> bounded Unitree retry
                -> PC fallback

        This cue does not represent dialogue content and must never generate a
        HumanIntent.

        The human-response timeout starts only after the matching cue transaction
        reaches COMPLETED.

        If cue playback itself fails, HRI will fail open and start the human
        response window anyway. An ergonomic cue must never deadlock dialogue.
        """

        if not self.active_interaction_id:
            return

        self.turn_cue_pending = False
        self.turn_cue_ready_at = None

        self.turn_cue_decision_id = (
            f"HRI_TURN_CUE_{self.active_interaction_id}_"
            f"{uuid.uuid4().hex[:12]}"
        )

        # Reuse the existing speech barrier. While the cue is pending, another
        # DialogueRequest for the same interaction remains blocked.
        self.waiting_for_speech_completion = True

        self.pending_speech_decision_id = (
            self.turn_cue_decision_id
        )

        self.pending_speech_requested_at = (
            time.monotonic()
        )

        self.pending_speech_started = False
        self.pending_speech_started_at = None

        msg = ExplanationText()

        msg.header.stamp = (
            self.get_clock().now().to_msg()
        )

        msg.decision_id = (
            self.turn_cue_decision_id
        )

        # TURN_CUE is not linguistic content.
        #
        # The Speech Manager must intercept speech_kind=TURN_CUE before normal
        # text/Piper validation and synthesize a short tone instead.
        msg.language = "en"
        msg.text = ""

        msg.fallback_language = "en"
        msg.fallback_text = ""

        msg.provider = "g1_hri"
        msg.model = ""

        msg.used_fallback = False
        msg.used_cache = False

        msg.speech_kind = "TURN_CUE"

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
            "Published human turn cue request | "
            f"interaction={self.active_interaction_id} | "
            f"decision_id={self.turn_cue_decision_id}"
        )

    def _open_human_response_window(
        self,
    ):
        """
        Start the interval during which a human response is expected.

        Robot-initiated dialogue reaches this method only after:

            robot question COMPLETED
                -> anti-echo guard
                -> turn-taking cue COMPLETED

        If cue playback failed, this method is also used as the non-blocking
        fail-open path.

        Passive HUMAN_INITIATIVE_WINDOW enters here immediately because there is
        no preceding robot dialogue question.

        The human timeout and ASR microphone become active exactly here.
        """

        if not self.active_interaction_id:

            self.get_logger().warning(
                "Cannot open human response window without "
                "an active interaction."
            )

            self._set_asr_listening(
                False
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

        # ---------------------------------------------------------
        # This timestamp defines the true beginning of the human turn.
        #
        # Robot speech, anti-echo guard and TURN_CUE consume none of
        # the human-response timeout.
        # ---------------------------------------------------------

        self.interaction_started_at = (
            time.monotonic()
        )

        self.waiting_for_human = True

        if self.mock_response_enabled:

            self.mock_response_at = (
                self.interaction_started_at
                + self.mock_response_delay_sec
            )

        else:
            self.mock_response_at = None

        # ---------------------------------------------------------
        # LAST operation of the turn transition:
        #
        # only now may the microphone start accumulating samples.
        # g1_pc_asr opens a fresh InputStream after receiving True.
        # ---------------------------------------------------------

        self._set_asr_listening(
            True
        )

        self.get_logger().info(
            "Human response window opened | "
            f"interaction={self.active_interaction_id} | "
            f"timeout={self.active_response_timeout:.2f}s | "
            "human_may_speak=YES"
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
    def _build_english_dialogue_fallback(
        self,
    ) -> str:
        """
        Rebuild the currently active robot dialogue turn in English.

        Dialogue wording remains owned by g1_social_common.

        No free-form translation is performed here.
        """

        request = self.active_request

        if request is None:
            return ""

        request_type = str(
            request.request_type or ""
        ).strip().upper()

        # ---------------------------------------------------------
        # Passage clarification
        # ---------------------------------------------------------

        if request_type == "PASSAGE_CLARIFICATION":
            return get_dialogue_text(
                "passage",
                "en",
            )

        # ---------------------------------------------------------
        # Semantic confirmation
        # ---------------------------------------------------------

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

            return get_dialogue_text(
                key,
                "en",
            )

        # ---------------------------------------------------------
        # Alternative proposal
        # ---------------------------------------------------------

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
                    "ALTERNATIVE_PROPOSAL without LEFT/RIGHT direction."
                )
                return ""

            return get_dialogue_text(
                key,
                "en",
            )

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
        now = time.monotonic()

        if (
            self.pending_close_control is not None
            or self.audio_uncertain_decision_id
        ):
            self._set_asr_listening(False)
            return

        if (
            self.mock_response_enabled
            and self.waiting_for_human
            and self.mock_response_at is not None
            and now >= self.mock_response_at
        ):
            self._handle_mock_response()
            return

        if (
            self.turn_cue_pending
            and self.turn_cue_ready_at is not None
            and now >= self.turn_cue_ready_at
        ):
            self._request_turn_cue()
            return

        if self.waiting_for_speech_completion:
            started = (
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
                started is not None
                and now - started >= timeout
            ):
                self._request_audio_reconciliation(
                    "ROBOT_SPEECH_TIMEOUT"
                )
            return

        if (
            self.waiting_for_human
            and self.interaction_started_at is not None
            and now - self.interaction_started_at
            >= self.active_response_timeout
        ):
            self._handle_timeout()
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
        Process a human response in the currently open window.

        UNKNOWN records incomprehension but does not
        immediately terminate the interaction.
        """

        if not self.enabled:
            return

        if not self.waiting_for_human:
            self.get_logger().warning(
                "[HRI] Intent ignored: no human response window."
            )
            return

        if not self.active_interaction_id:
            return

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

        if normalized_intent == "UNKNOWN":

            self.unrecognized_response_received = True

            recognized_text = str(
                hri_intent.utterance or utterance or ""
            ).strip()

            if recognized_text:
                self.last_unrecognized_utterance = (
                    recognized_text
                )

            self.get_logger().warning(
                "[HRI] HUMAN_RESPONSE_UNRECOGNIZED | "
                f"interaction={self.active_interaction_id} | "
                f"utterance={recognized_text!r}"
            )

            # Keep the response window open.
            # Do not restart the timeout.
            return

        interaction_id = self.active_interaction_id

        self.waiting_for_human = False

        self._set_asr_listening(False)

        self._publish_human_intent(
            intent=hri_intent.intent,
            direction=hri_intent.direction,
            target=hri_intent.target,
            constraint=hri_intent.constraint,
            confidence=hri_intent.confidence,
            utterance=hri_intent.utterance,
            source=hri_intent.source,
            modality=hri_intent.modality,
        )

        self.get_logger().info(
            "[HRI] HUMAN_INTENT_ACCEPTED | "
            f"interaction={interaction_id} | "
            f"intent={normalized_intent}"
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