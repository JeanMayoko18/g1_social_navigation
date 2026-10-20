#!/usr/bin/env python3

"""
Common Speech Manager based on Piper TTS.

Responsibility
--------------
Receive ExplanationText requests, synthesize them with Piper, play the
generated audio, and publish the real speech lifecycle through SpeechStatus.

Flow
----
ExplanationText
    |
    v
Piper synthesis
    |
    v
WAV
    |
    +--> PC:      aplay
    |
    +--> Unitree: PCM 16 kHz mono 16-bit
                  -> unitree_audio_player
    |
    v
SpeechStatus
    |
    +--> STARTED
    +--> COMPLETED
    +--> FAILED

Important
---------
This node is deliberately independent from navigation and collaboration
policy.

It does NOT:
    - inspect NavigationState;
    - inspect SocialNavigationDecision;
    - decide WAIT / REPLAN / RESUME;
    - validate a social conflict;
    - decide whether an explanation is still semantically appropriate.

Those responsibilities belong upstream.

If an ExplanationText reaches this node, the Speech Manager only validates
the speech request itself and executes it.

Correlation invariant
---------------------
For every accepted request:

    ExplanationText.decision_id == SpeechStatus.decision_id

COMPLETED is emitted only after successful audio playback.
"""

import audioop
import importlib.util
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import wave
import threading

from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import rclpy

from rclpy.node import Node

from g1_social_msgs.msg import (
    ExplanationText,
    SpeechCancelRequest,
    SpeechStatus,
)

from g1_qos_manager.profiles import reliable_qos
from g1_qos_manager.diagnostics import log_qos_resolution
from g1_qos_manager.resolver import resolve_subscription_qos


# =============================================================================
# PIPER BACKENDS
# =============================================================================

PIPER_BACKEND_AUTO = "auto"
PIPER_BACKEND_MODULE = "module"
PIPER_BACKEND_EXECUTABLE = "executable"

SUPPORTED_PIPER_BACKENDS = {
    PIPER_BACKEND_AUTO,
    PIPER_BACKEND_MODULE,
    PIPER_BACKEND_EXECUTABLE,
}


# =============================================================================
# AUDIO BACKENDS
# =============================================================================

AUDIO_BACKEND_AUTO = "auto"
AUDIO_BACKEND_APLAY = "aplay"
AUDIO_BACKEND_UNITREE = "unitree"

SUPPORTED_AUDIO_BACKENDS = {
    AUDIO_BACKEND_AUTO,
    AUDIO_BACKEND_APLAY,
    AUDIO_BACKEND_UNITREE,
}

SPEECH_STARTED = "STARTED"
SPEECH_COMPLETED = "COMPLETED"
SPEECH_FAILED = "FAILED"
SPEECH_CANCELED = "CANCELED"


# =============================================================================
# LANGUAGES
# =============================================================================

from g1_social_common.dialogue_texts import (
    normalize_dialogue_language,
)

# =============================================================================
# SPEECH STATUS
# =============================================================================

SPEECH_STARTED = "STARTED"
SPEECH_COMPLETED = "COMPLETED"
SPEECH_FAILED = "FAILED"


# =============================================================================
# UNITREE AUDIO FORMAT
# =============================================================================

UNITREE_SAMPLE_RATE_HZ = 16000
UNITREE_CHANNEL_COUNT = 1
UNITREE_SAMPLE_WIDTH_BYTES = 2

# =============================================================================
# SPEECH PRIORITY
# =============================================================================

SPEECH_PRIORITY = {
    # Turn-taking cue must immediately follow the dialogue turn.
    "TURN_CUE": -1,

    # Human/robot interaction transactions.
    "DIALOGUE": 0,

    # Social-navigation synchronization.
    "SOCIAL_EXPLANATION": 1,

    # Navigation lifecycle.
    "NAVIGATION_RESULT": 2,
    "NAVIGATION_GOAL": 3,
    "INITIAL_POSE": 4,
}

DEFAULT_SPEECH_PRIORITY = 100

# =============================================================================
# NODE
# =============================================================================


class PiperTTSNode(Node):
    """
    Common Speech Manager using Piper.

    ExplanationText is treated as an already-authorized speech request.

    The node performs only:
        request validation,
        queue management,
        synthesis,
        audio playback,
        SpeechStatus publication.
    """

    def __init__(self) -> None:
        super().__init__("piper_tts")

        # =====================================================================
        # DEFAULT PATHS
        # =====================================================================

        environment_model_directory = os.environ.get(
            "G1_PIPER_MODEL_DIRECTORY",
            str(
                Path.home()
                / "g1_ws"
                / "models"
                / "piper"
            ),
        )

        # =====================================================================
        # PARAMETERS
        # =====================================================================

        self.declare_parameter(
            "enabled",
            True,
        )

        self.declare_parameter(
            "ros_distro",
            "auto",
        )

        self.declare_parameter(
            "input_topic",
            "/explainability/text",
        )

        self.declare_parameter(
            "status_topic",
            "/explainability/speech_status",
        )

        # ---------------------------------------------------------------------
        # Piper
        # ---------------------------------------------------------------------

        self.declare_parameter(
            "piper_model_directory",
            environment_model_directory,
        )

        self.declare_parameter(
            "piper_backend",
            PIPER_BACKEND_AUTO,
        )

        self.declare_parameter(
            "piper_executable",
            os.environ.get(
                "G1_PIPER_EXECUTABLE",
                "",
            ),
        )

        self.declare_parameter(
            "spanish_model",
            os.environ.get(
                "G1_PIPER_SPANISH_MODEL",
                "",
            ),
        )

        self.declare_parameter(
            "english_model",
            os.environ.get(
                "G1_PIPER_ENGLISH_MODEL",
                "",
            ),
        )

        # ---------------------------------------------------------------------
        # Voice
        # ---------------------------------------------------------------------

        self.declare_parameter(
            "speech_rate",
            1.0,
        )

        self.declare_parameter(
            "sentence_silence_sec",
            0.20,
        )

        # ---------------------------------------------------------------------
        # Audio
        # ---------------------------------------------------------------------

        self.declare_parameter(
            "audio_backend",
            AUDIO_BACKEND_AUTO,
        )

        self.declare_parameter(
            "audio_player",
            "aplay",
        )
        
        self.declare_parameter(
            "pc_fallback_request_topic",
            "/explainability/pc_audio_fallback/request",
        )

        self.declare_parameter(
            "pc_fallback_status_topic",
            "/explainability/pc_audio_fallback/status",
        )

        self.declare_parameter(
            "pc_fallback_timeout_sec",
            20.0,
        )

        # ---------------------------------------------------------------------
        # Unitree
        # ---------------------------------------------------------------------

        self.declare_parameter(
            "unitree_network_interface",
            "eth0",
        )

        self.declare_parameter(
            "unitree_volume",
            85,
        )

        self.declare_parameter(
            "unitree_stream_name",
            "g1_piper_tts",
        )

        # ---------------------------------------------------------------------
        # Queue / timeout
        # ---------------------------------------------------------------------

        self.declare_parameter(
            "speech_queue_size",
            10,
        )

        self.declare_parameter(
            "synthesis_timeout_sec",
            20.0,
        )

        # =====================================================================
        # GENERAL CONFIGURATION
        # =====================================================================

        self.enabled = bool(
            self.get_parameter("enabled").value
        )

        self.input_topic = str(
            self.get_parameter("input_topic").value
        ).strip()

        if not self.input_topic:
            self.input_topic = "/explainability/text"

        self.status_topic = str(
            self.get_parameter("status_topic").value
        ).strip()

        if not self.status_topic:
            self.status_topic = "/explainability/speech_status"

        # =====================================================================
        # ROS DISTRIBUTION
        # =====================================================================

        self.ros_distro_requested = str(
            self.get_parameter("ros_distro").value
        ).strip().lower()

        if (
            not self.ros_distro_requested
            or self.ros_distro_requested == "auto"
        ):
            self.ros_distro = os.environ.get(
                "ROS_DISTRO",
                "",
            ).strip().lower()

        else:
            self.ros_distro = self.ros_distro_requested

        if self.ros_distro not in {
            "foxy",
            "humble",
        }:
            raise RuntimeError(
                "ros_distro must resolve to foxy or humble, "
                f"received '{self.ros_distro}'."
            )

        # =====================================================================
        # MODEL DIRECTORY
        # =====================================================================

        model_directory_value = str(
            self.get_parameter(
                "piper_model_directory"
            ).value
        ).strip()

        self.piper_model_directory = Path(
            model_directory_value
            or environment_model_directory
        ).expanduser()

        # =====================================================================
        # VOICE MODELS
        # =====================================================================

        default_spanish_model = (
            self.piper_model_directory
            / "es_ES-davefx-medium.onnx"
        )

        default_english_model = (
            self.piper_model_directory
            / "en_US-ryan-medium.onnx"
        )

        spanish_model_value = str(
            self.get_parameter(
                "spanish_model"
            ).value
        ).strip()

        english_model_value = str(
            self.get_parameter(
                "english_model"
            ).value
        ).strip()

        self.spanish_model = Path(
            spanish_model_value
            or str(default_spanish_model)
        ).expanduser()

        self.english_model = Path(
            english_model_value
            or str(default_english_model)
        ).expanduser()

        # =====================================================================
        # VOICE PARAMETERS
        # =====================================================================

        self.speech_rate = float(
            self.get_parameter(
                "speech_rate"
            ).value
        )

        if not 0.5 <= self.speech_rate <= 2.0:
            self.get_logger().warn(
                "[TTS] speech_rate must be between "
                "0.5 and 2.0. Using 1.0."
            )

            self.speech_rate = 1.0

        self.sentence_silence_sec = float(
            self.get_parameter(
                "sentence_silence_sec"
            ).value
        )

        if not 0.0 <= self.sentence_silence_sec <= 2.0:
            self.get_logger().warn(
                "[TTS] sentence_silence_sec must be between "
                "0.0 and 2.0. Using 0.20."
            )

            self.sentence_silence_sec = 0.20

        # Piper:
        #   > 1.0 -> slower
        #   < 1.0 -> faster
        self.piper_length_scale = (
            1.0 / self.speech_rate
        )

        # =====================================================================
        # PIPER BACKEND
        # =====================================================================

        self.piper_backend_requested = str(
            self.get_parameter(
                "piper_backend"
            ).value
        ).strip().lower()

        if not self.piper_backend_requested:
            self.piper_backend_requested = (
                PIPER_BACKEND_AUTO
            )

        piper_executable_value = str(
            self.get_parameter(
                "piper_executable"
            ).value
        ).strip()

        self.piper_module_available = (
            importlib.util.find_spec("piper")
            is not None
        )

        self.piper_executable = (
            self._find_piper_executable(
                configured_value=piper_executable_value,
                model_directory=self.piper_model_directory,
            )
        )

        self.piper_backend = (
            self._resolve_piper_backend()
        )

        # =====================================================================
        # AUDIO CONFIGURATION
        # =====================================================================

        self.audio_backend_requested = str(
            self.get_parameter(
                "audio_backend"
            ).value
        ).strip().lower()

        if not self.audio_backend_requested:
            self.audio_backend_requested = (
                AUDIO_BACKEND_AUTO
            )

        self.audio_player = str(
            self.get_parameter(
                "audio_player"
            ).value
        ).strip()

        if not self.audio_player:
            self.audio_player = "aplay"

        self.audio_player_path: Optional[str] = None

        self.pc_fallback_request_topic = str(
            self.get_parameter(
                "pc_fallback_request_topic"
            ).value
        ).strip()

        self.pc_fallback_status_topic = str(
            self.get_parameter(
                "pc_fallback_status_topic"
            ).value
        ).strip()

        self.pc_fallback_timeout_sec = float(
            self.get_parameter(
                "pc_fallback_timeout_sec"
            ).value
        )

        if self.pc_fallback_timeout_sec <= 0.0:
            raise ValueError(
                "pc_fallback_timeout_sec must be strictly positive."
            )

        # =====================================================================
        # UNITREE CONFIGURATION
        # =====================================================================

        self.unitree_network_interface = str(
            self.get_parameter(
                "unitree_network_interface"
            ).value
        ).strip()

        self.unitree_volume = int(
            self.get_parameter(
                "unitree_volume"
            ).value
        )

        self.unitree_volume = max(
            0,
            min(
                100,
                self.unitree_volume,
            ),
        )

        self.unitree_stream_name = str(
            self.get_parameter(
                "unitree_stream_name"
            ).value
        ).strip()

        if not self.unitree_stream_name:
            self.unitree_stream_name = "g1_piper_tts"

        # =====================================================================
        # AUDIO BACKEND INITIALIZATION
        # =====================================================================

        self.audio_backend = (
            self._resolve_audio_backend()
        )

        self.audio_backend_ready = (
            self._initialize_audio_backend()
        )

        # =====================================================================
        # MODEL VALIDATION
        # =====================================================================

        self.spanish_model_ready = (
            self._validate_model(
                model=self.spanish_model,
                label="Spanish",
            )
        )

        self.english_model_ready = (
            self._validate_model(
                model=self.english_model,
                label="English",
            )
        )

        self.runtime_ready = (
            self.piper_backend is not None
            and self.audio_backend_ready
        )

        # =====================================================================
        # QUEUE CONFIGURATION
        # =====================================================================

        self.speech_queue_size = int(
            self.get_parameter(
                "speech_queue_size"
            ).value
        )

        self.synthesis_timeout_sec = float(
            self.get_parameter(
                "synthesis_timeout_sec"
            ).value
        )

        if self.speech_queue_size < 1:
            raise ValueError(
                "speech_queue_size must be positive."
            )

        if (
            not math.isfinite(
                self.synthesis_timeout_sec
            )
            or self.synthesis_timeout_sec <= 0.0
        ):
            raise ValueError(
                "synthesis_timeout_sec must be "
                "finite and positive."
            )

        # =====================================================================
        # RUNTIME STATE
        # =====================================================================

        self.messages_received = 0
        self.utterances_spoken = 0
        self.synthesis_failures = 0
        self.unsupported_languages = 0

        self.last_decision_id = ""

        self.speech_queue = deque()

        self.pending_speech_ids = set()

        self.completed_speech_ids = OrderedDict()

        self.audio_worker = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="piper_audio",
        )

        self.audio_future = None
        self.audio_phase = None
        self.active_speech = None

        # Protect the process handle shared by the ROS callback
        # and the background audio worker.
        self.playback_process_lock = threading.Lock()
        self.aplay_process = None

        self.unitree_process = None
        self.audio_output_uncertain = False

        self.tts_stopping = False

        # Remote PC fallback state.
        self.pc_fallback_waiting = False
        self.pc_fallback_decision_id = ""
        self.pc_fallback_requested_at = None
        self.pc_fallback_job = None


        # =====================================================================
        # QOS
        # =====================================================================

        input_qos = resolve_subscription_qos(
            self,
            self.input_topic,
            fallback_depth=10,
            prefer_sensor_data=False,
        )

        log_qos_resolution(
            self,
            self.input_topic,
            input_qos,
        )

        # =====================================================================
        # ROS INTERFACES
        # =====================================================================

        self.status_publisher = (
            self.create_publisher(
                SpeechStatus,
                self.status_topic,
                reliable_qos(depth=20),
            )
        )

        self.subscription = (
            self.create_subscription(
                ExplanationText,
                self.input_topic,
                self.explanation_callback,
                input_qos,
            )
        )

        self.pc_fallback_request_pub = (
            self.create_publisher(
                ExplanationText,
                self.pc_fallback_request_topic,
                reliable_qos(depth=10),
            )
        )

        self.pc_fallback_cancel_pub = self.create_publisher(
            SpeechCancelRequest,
            "/explainability/pc_audio_fallback/cancel_request",
            reliable_qos(depth=20),
        )

        self.pc_fallback_status_sub = (
            self.create_subscription(
                SpeechStatus,
                self.pc_fallback_status_topic,
                self._pc_fallback_status_callback,
                reliable_qos(depth=20),
            )
        )

        self.speech_cancel_sub = self.create_subscription(
            SpeechCancelRequest,
            "/explainability/speech_cancel_request",
            self._speech_cancel_callback,
            reliable_qos(depth=20),
        )

        self.audio_timer = self.create_timer(
            0.05,
            self._audio_tick,
        )

        self.diagnostics_timer = self.create_timer(
            5.0,
            self.log_diagnostics,
        )

        self._log_initialization()

    # =========================================================================
    # PIPER BACKEND
    # =========================================================================

    def _speech_priority(self, job) -> int:
        """Return the scheduling priority of a queued speech request."""

        message = job.get("message")

        if message is None:
            return DEFAULT_SPEECH_PRIORITY

        speech_kind = str(
            message.speech_kind or ""
        ).strip().upper()

        return SPEECH_PRIORITY.get(
            speech_kind,
            DEFAULT_SPEECH_PRIORITY,
        )


    def _pop_next_speech(self):
        """
        Remove the highest-priority queued speech.

        FIFO ordering is preserved between requests having the same priority.
        Active synthesis/playback is never preempted.
        """

        if not self.speech_queue:
            return None

        best_index = 0
        best_priority = self._speech_priority(
            self.speech_queue[0]
        )

        for index, job in enumerate(self.speech_queue):
            priority = self._speech_priority(job)

            if priority < best_priority:
                best_index = index
                best_priority = priority

        if best_index == 0:
            return self.speech_queue.popleft()

        self.speech_queue.rotate(-best_index)
        job = self.speech_queue.popleft()
        self.speech_queue.rotate(best_index)

        return job

    def _find_piper_executable(
        self,
        configured_value: str,
        model_directory: Path,
    ) -> Optional[Path]:
        """Locate a usable standalone Piper executable."""

        candidates = []

        if configured_value:
            candidates.append(
                Path(
                    configured_value
                ).expanduser()
            )

        environment_value = os.environ.get(
            "G1_PIPER_EXECUTABLE",
            "",
        ).strip()

        if environment_value:
            candidates.append(
                Path(
                    environment_value
                ).expanduser()
            )

        path_executable = shutil.which("piper")

        if path_executable:
            candidates.append(
                Path(path_executable)
            )

        candidates.append(
            model_directory / "piper"
        )

        for candidate in candidates:
            if (
                candidate.is_file()
                and os.access(
                    str(candidate),
                    os.X_OK,
                )
            ):
                return candidate

        return None

    def _resolve_piper_backend(
        self,
    ) -> Optional[str]:
        """Resolve the Piper execution backend."""

        requested = (
            self.piper_backend_requested
        )

        if requested not in SUPPORTED_PIPER_BACKENDS:
            self.get_logger().warn(
                "[TTS] Unsupported piper_backend="
                f"'{requested}'. Using auto."
            )

            requested = PIPER_BACKEND_AUTO

        if requested == PIPER_BACKEND_MODULE:
            if self.piper_module_available:
                return PIPER_BACKEND_MODULE

            self.get_logger().error(
                "[TTS] Piper Python module requested "
                "but unavailable."
            )

            return None

        if requested == PIPER_BACKEND_EXECUTABLE:
            if self.piper_executable is not None:
                return PIPER_BACKEND_EXECUTABLE

            self.get_logger().error(
                "[TTS] Piper executable requested "
                "but unavailable."
            )

            return None

        if self.piper_module_available:
            return PIPER_BACKEND_MODULE

        if self.piper_executable is not None:
            return PIPER_BACKEND_EXECUTABLE

        self.get_logger().error(
            "[TTS] No Piper backend is available."
        )

        return None

    # =========================================================================
    # AUDIO BACKEND
    # =========================================================================

    def _resolve_audio_backend(
        self,
    ) -> Optional[str]:
        """
        Resolve audio output.

        auto:
            Foxy   -> Unitree
            Humble -> aplay
        """

        requested = (
            self.audio_backend_requested
        )

        if requested not in SUPPORTED_AUDIO_BACKENDS:
            self.get_logger().error(
                "[TTS] Unsupported audio_backend="
                f"'{requested}'. Expected "
                "auto, unitree or aplay."
            )

            return None

        if requested == AUDIO_BACKEND_UNITREE:
            return AUDIO_BACKEND_UNITREE

        if requested == AUDIO_BACKEND_APLAY:
            return AUDIO_BACKEND_APLAY

        if self.ros_distro == "foxy":
            return AUDIO_BACKEND_UNITREE

        if self.ros_distro == "humble":
            return AUDIO_BACKEND_APLAY

        return None

    def _initialize_audio_backend(
        self,
    ) -> bool:
        """Initialize the selected output backend."""

        if self.audio_backend == AUDIO_BACKEND_APLAY:
            return self._initialize_aplay_backend()

        if self.audio_backend == AUDIO_BACKEND_UNITREE:
            return self._initialize_unitree_backend()

        return False

    def _initialize_aplay_backend(
        self,
    ) -> bool:
        """Locate aplay."""

        self.audio_player_path = shutil.which(
            self.audio_player
        )

        if self.audio_player_path is None:
            self.get_logger().error(
                "[TTS] Audio player not found: "
                f"{self.audio_player}"
            )

            return False

        return True

    def _initialize_unitree_backend(
        self,
    ) -> bool:
        """Validate Unitree audio support."""

        if not self.unitree_network_interface:
            self.get_logger().error(
                "[TTS] unitree_network_interface "
                "is empty."
            )

            return False

        if (
            importlib.util.find_spec(
                "unitree_sdk2py"
            )
            is None
        ):
            self.get_logger().error(
                "[TTS] unitree_sdk2py "
                "is unavailable."
            )

            return False

        return True

    # =========================================================================
    # REQUEST PROCESSING
    # =========================================================================

    def explanation_callback(
        self,
        msg: ExplanationText,
    ) -> None:
        """
        Validate and enqueue one speech/audio request.

        Normal speech:
            text
            -> language/model resolution
            -> Piper
            -> playback

        TURN_CUE:
            no linguistic text
            -> generated tone WAV
            -> playback

        Language semantics belong to g1_social_common.

        Voice selection belongs to this Speech Manager.

        If the preferred voice is unavailable, the request may fall back to
        an explicitly supplied English text and English voice.

        A voice fallback NEVER changes only the voice while keeping text in
        another language.
        """

        self.messages_received += 1

        decision_id = str(
            msg.decision_id or ""
        ).strip()

        if not decision_id:
            self._publish_speech_status(
                decision_id="",
                state=SPEECH_FAILED,
                detail=(
                    "The speech request has no "
                    "decision identifier."
                ),
            )
            return

        # Idempotency:
        # never queue or replay the same decision twice.
        if (
            decision_id in self.pending_speech_ids
            or decision_id in self.completed_speech_ids
        ):
            self.get_logger().info(
                "[TTS] Duplicate speech request "
                f"ignored: {decision_id}"
            )
            return

        # Refuse new requests while the physical audio state is uncertain.
        # Never queue a job that the audio scheduler cannot execute.
        if self.audio_output_uncertain:
            job = {
                "message": msg,
                "decision_id": decision_id,
            }

            self._finish_speech(
                job,
                success=False,
                detail=(
                    "AUDIO_OUTPUT_UNCONFIRMED: "
                    "Speech rejected because the previous "
                    "audio stop was not confirmed."
                ),
            )
            return

        speech_kind = str(
            msg.speech_kind or ""
        ).strip().upper()

        is_turn_cue = (
            speech_kind == "TURN_CUE"
        )

        # -----------------------------------------------------------------
        # Common node availability
        # -----------------------------------------------------------------

        if self.tts_stopping:
            job = {
                "message": msg,
                "decision_id": decision_id,
            }

            self._finish_speech(
                job,
                success=False,
                detail="TTS is stopping.",
            )
            return

        if not self.enabled:
            job = {
                "message": msg,
                "decision_id": decision_id,
            }

            self._finish_speech(
                job,
                success=False,
                detail="TTS is disabled.",
            )
            return

        # -----------------------------------------------------------------
        # TURN_CUE
        # -----------------------------------------------------------------
        #
        # TURN_CUE is audio, not language.
        #
        # It must therefore:
        #   - accept empty text;
        #   - not require Piper;
        #   - not require a voice model;
        #   - require only a usable audio backend.
        # -----------------------------------------------------------------

        if is_turn_cue:

            job = {
                "message": msg,
                "decision_id": decision_id,
                "text": "",
                "language": "",
                "model": None,
                "requested_language": "",
                "used_language_fallback": False,
                "received_at": time.monotonic(),
                "is_turn_cue": True,
            }

            if not self.audio_backend_ready:
                self._finish_speech(
                    job,
                    success=False,
                    detail=(
                        "Audio backend is unavailable "
                        "for the human turn cue."
                    ),
                )
                return

            if (
                len(self.speech_queue)
                >= self.speech_queue_size
            ):
                self._finish_speech(
                    job,
                    success=False,
                    detail="The speech queue is full.",
                )
                return

            self.pending_speech_ids.add(
                decision_id
            )

            self.speech_queue.append(
                job
            )

            self.get_logger().info(
                "[TTS] "
                f"QUEUED={decision_id} | "
                "KIND=TURN_CUE | "
                f"WAITING={len(self.speech_queue)}"
            )

            return

        # -----------------------------------------------------------------
        # Normal linguistic speech
        # -----------------------------------------------------------------

        preferred_text = str(
            msg.text or ""
        ).strip()

        if not preferred_text:
            self._publish_speech_status(
                decision_id=decision_id,
                state=SPEECH_FAILED,
                detail="The speech text is empty.",
            )
            return

        preferred_language = (
            normalize_dialogue_language(
                msg.language
            )
        )

        fallback_text = str(
            getattr(
                msg,
                "fallback_text",
                "",
            )
            or ""
        ).strip()

        fallback_language = str(
            getattr(
                msg,
                "fallback_language",
                "",
            )
            or ""
        ).strip()

        if fallback_language:
            fallback_language = (
                normalize_dialogue_language(
                    fallback_language
                )
            )

        # -------------------------------------------------------------
        # Preferred voice
        # -------------------------------------------------------------

        text = preferred_text
        language = preferred_language

        model = self._resolve_model(
            preferred_language
        )

        used_language_fallback = False

        # -------------------------------------------------------------
        # Explicit English fallback
        # -------------------------------------------------------------

        if model is None:

            if (
                fallback_text
                and fallback_language == "en"
            ):
                fallback_model = self._resolve_model(
                    "en"
                )

                if fallback_model is not None:
                    self.get_logger().warning(
                        "[TTS] Preferred voice unavailable; "
                        "using explicit English fallback | "
                        f"decision={decision_id} | "
                        f"requested_language={preferred_language!r}"
                    )

                    text = fallback_text
                    language = "en"
                    model = fallback_model
                    used_language_fallback = True

        job = {
            "message": msg,
            "decision_id": decision_id,
            "text": text,
            "language": language,
            "model": model,
            "requested_language": preferred_language,
            "used_language_fallback": (
                used_language_fallback
            ),
            "received_at": time.monotonic(),
            "is_turn_cue": False,
        }

        if not self.runtime_ready:
            self._finish_speech(
                job,
                success=False,
                detail=(
                    "TTS runtime is unavailable."
                ),
            )
            return

        if model is None:
            self.unsupported_languages += 1

            self._finish_speech(
                job,
                success=False,
                detail=(
                    "No configured voice exists for the requested "
                    "language and no usable English fallback "
                    "was provided."
                ),
            )
            return

        if (
            len(self.speech_queue)
            >= self.speech_queue_size
        ):
            self._finish_speech(
                job,
                success=False,
                detail="The speech queue is full.",
            )
            return

        self.pending_speech_ids.add(
            decision_id
        )

        self.speech_queue.append(
            job
        )

        self.get_logger().info(
            "[TTS] "
            f"QUEUED={decision_id} | "
            f"KIND={speech_kind or 'UNSPECIFIED'} | "
            f"REQUESTED_LANG={preferred_language} | "
            f"SPOKEN_LANG={language} | "
            f"LANG_FALLBACK={used_language_fallback} | "
            f"WAITING={len(self.speech_queue)}"
        )

    # =========================================================================
    # AUDIO STATE MACHINE
    # =========================================================================
    
    def _request_pc_audio_fallback(
        self,
        job,
        unitree_detail: str,
    ) -> None:
        """
        Delegate one already-authorized speech transaction to the PC.

        The robot remains the speech authority.

        The PC receives the original ExplanationText only after the Unitree
        backend has exhausted its allowed attempts.

        The robot keeps ownership of the official SpeechStatus lifecycle.
        """

        if job is None:
            return

        decision_id = str(
            job.get(
                "decision_id",
                "",
            )
        ).strip()

        if not decision_id:
            self._finish_speech(
                job,
                success=False,
                detail=(
                    "Cannot request PC audio fallback "
                    "without decision_id."
                ),
            )
            return

        message = job.get(
            "message"
        )

        if message is None:
            self._finish_speech(
                job,
                success=False,
                detail=(
                    "Cannot request PC audio fallback "
                    "without the original speech request."
                ),
            )
            return

        if self.pc_fallback_waiting:
            self._finish_speech(
                job,
                success=False,
                detail=(
                    "Another PC audio fallback transaction "
                    "is already pending."
                ),
            )
            return

        self.pc_fallback_waiting = True

        self.pc_fallback_decision_id = (
            decision_id
        )

        self.pc_fallback_requested_at = (
            time.monotonic()
        )

        self.pc_fallback_job = job

        self.get_logger().warning(
            "[TTS] Unitree audio exhausted; "
            "delegating playback to PC | "
            f"decision_id={decision_id} | "
            f"unitree_detail={unitree_detail}"
        )

        self.pc_fallback_request_pub.publish(
            message
        )

        self.get_logger().info(
            "[TTS] PC_FALLBACK_REQUESTED | "
            f"decision_id={decision_id} | "
            f"topic={self.pc_fallback_request_topic}"
        )

    def _clear_pc_audio_fallback(self) -> None:
        """
        Release ownership of the current PC fallback transaction.

        This function must not publish, cancel, or finalize
        any speech transaction.
        """
        self.pc_fallback_waiting = False
        self.pc_fallback_decision_id = ""
        self.pc_fallback_requested_at = None
        self.pc_fallback_job = None

    def _pc_fallback_status_callback(
        self,
        msg: SpeechStatus,
    ) -> None:
        """
        Process remote PC playback status.

        The robot is the sole publisher of the official
        SpeechStatus lifecycle.

        A remote CANCELED acknowledges a requested stop.
        A remote FAILED or COMPLETED cannot acknowledge
        cancellation.
        """

        if not self.pc_fallback_waiting:
            return

        decision_id = str(
            msg.decision_id or ""
        ).strip()

        if (
            not decision_id
            or decision_id != self.pc_fallback_decision_id
        ):
            return

        job = self.pc_fallback_job

        if job is None:
            self.audio_output_uncertain = True
            self._clear_pc_audio_fallback()
            return

        state = str(
            msg.state or ""
        ).strip().upper()

        detail = str(
            msg.detail or ""
        ).strip()

        cancel_requested = bool(
            job.get("cancel_requested", False)
        )

        # STARTED does not terminate the transaction.
        if state == SPEECH_STARTED:
            return

        # Ignore unknown or non-terminal statuses.
        if state not in {
            SPEECH_COMPLETED,
            SPEECH_FAILED,
            SPEECH_CANCELED,
        }:
            return

        # The transaction is now resolved by a terminal
        # response from the remote executor.
        self._clear_pc_audio_fallback()

        # -----------------------------------------------------
        # Requested cancellation
        # -----------------------------------------------------

        if cancel_requested:

            if state == SPEECH_CANCELED:

                job["cancel_confirmed"] = True

                self._finish_canceled_speech(
                    job,
                    detail=detail or (
                        "PC fallback cancellation confirmed."
                    ),
                )

                return

            # A remote FAILED or COMPLETED cannot prove
            # that cancellation was acknowledged.
            self.audio_output_uncertain = True

            self._finish_speech(
                job,
                success=False,
                detail=(
                    "Remote PC cancellation unconfirmed | "
                    f"state={state} | {detail}"
                ),
            )

            return

        # -----------------------------------------------------
        # No cancellation requested
        # -----------------------------------------------------

        if state == SPEECH_CANCELED:

            # Unexpected remote cancellation is treated
            # as a protocol inconsistency.
            self.audio_output_uncertain = True

            self._finish_speech(
                job,
                success=False,
                detail=(
                    "Unexpected PC cancellation | "
                    f"{detail}"
                ),
            )

            return

        if state == SPEECH_COMPLETED:

            self._finish_speech(
                job,
                success=True,
                detail=detail or (
                    "Speech delivered through PC fallback."
                ),
            )

            return

        # Normal remote execution failure.
        self._finish_speech(
            job,
            success=False,
            detail=detail or (
                "PC fallback playback failed."
            ),
        )

    def _speech_cancel_callback(
        self,
        msg: SpeechCancelRequest,
    ) -> None:
        """
        Request cancellation of one speech transaction.

        The decision_id is the correlation key.

        Guarantees:
        - Queued speech can be canceled immediately.
        - Active synthesis cannot proceed to playback.
        - Active local playback is asked to terminate.
        - The audio worker owns final cancellation confirmation.
        - Remote PC fallback is not falsely acknowledged.
        """

        decision_id = str(
            msg.decision_id or ""
        ).strip()

        if not decision_id:
            self.get_logger().warning(
                "[TTS CANCEL] Missing decision_id."
            )
            return

        if decision_id in self.completed_speech_ids:
            return

        # ---------------------------------------------------------
        # 1. Queued transaction
        # ---------------------------------------------------------

        for job in tuple(self.speech_queue):

            if job.get("decision_id") != decision_id:
                continue

            self.speech_queue.remove(job)

            job["cancel_requested"] = True
            job["cancel_confirmed"] = True

            self._finish_canceled_speech(
                job,
                detail="Queued speech canceled before execution.",
            )

            return

        # ---------------------------------------------------------
        # 2. Remote PC fallback transaction
        # ---------------------------------------------------------

        if (
            self.pc_fallback_waiting
            and self.pc_fallback_decision_id == decision_id
        ):
            job = self.pc_fallback_job

            if job is None:
                self.audio_output_uncertain = True
                return

            if job.get("cancel_requested", False):
                return

            job["cancel_requested"] = True
            job["cancel_confirmed"] = False
            job["pc_cancel_requested_at"] = time.monotonic()

            cancel_msg = SpeechCancelRequest()
            cancel_msg.decision_id = decision_id

            try:
                self.pc_fallback_cancel_pub.publish(
                    cancel_msg
                )

            except Exception as exc:
                self.audio_output_uncertain = True

                self.get_logger().error(
                    "[TTS CANCEL] Remote cancellation "
                    f"publication failed | decision_id={decision_id} | "
                    f"error={exc}"
                )
                return

            self.get_logger().warning(
                "[TTS CANCEL] Remote cancellation requested | "
                f"decision_id={decision_id}"
            )

            return
        
        # ---------------------------------------------------------
        # 3. Active local transaction
        # ---------------------------------------------------------

        job = self.active_speech

        if job is None:
            self.get_logger().warning(
                "[TTS CANCEL] Unknown speech transaction | "
                f"decision_id={decision_id}"
            )
            return

        if job.get("decision_id") != decision_id:
            return

        phase = self.audio_phase

        with self.playback_process_lock:

            if job.get("cancel_requested", False):
                return

            job["cancel_requested"] = True
            job["cancel_confirmed"] = False

            if phase in {
                "SYNTHESIS",
                "TURN_CUE_GENERATION",
            }:
                # The worker may complete its current calculation.
                # _audio_tick must suppress any resulting playback.
                pass

            elif phase == "PLAYBACK":

                if self.audio_backend == AUDIO_BACKEND_APLAY:
                    process = self.aplay_process

                elif self.audio_backend == AUDIO_BACKEND_UNITREE:
                    process = self.unitree_process

                else:
                    process = None

                if (
                    process is not None
                    and process.poll() is None
                ):
                    try:
                        process.terminate()

                    except ProcessLookupError:
                        pass

                    except Exception as exc:
                        self.get_logger().error(
                            "[TTS CANCEL] Unable to terminate "
                            f"playback process: {exc}"
                        )

                        self.audio_output_uncertain = True

        self.get_logger().warning(
            "[TTS CANCEL] Requested | "
            f"decision_id={decision_id} | "
            f"phase={phase} | "
            f"backend={self.audio_backend}"
        )
        
    def _audio_tick(self) -> None:
        """
        Advance the speech lifecycle.

        States:
            QUEUED
            SYNTHESIS / TURN_CUE_GENERATION
            PLAYBACK
            PC_FALLBACK_WAIT

        Terminal results:
            COMPLETED
            FAILED
            CANCELED

        An active cancellation is confirmed only after
        the worker has ended and its backend-specific
        cancellation conditions are satisfied.
        """

        now = time.monotonic()

        # =========================================================
        # 1. REMOTE PC FALLBACK
        # =========================================================

        if self.pc_fallback_waiting:

            job = self.pc_fallback_job

            if job is None:
                self.audio_output_uncertain = True
                self._clear_pc_audio_fallback()
                return

            decision_id = job["decision_id"]

            if job.get("cancel_requested", False):

                canceled_at = job.get(
                    "pc_cancel_requested_at"
                )

                if canceled_at is None:
                    canceled_at = now
                    job["pc_cancel_requested_at"] = now

                cancel_timeout_sec = 5.0

                if (
                    now - canceled_at
                    >= cancel_timeout_sec
                ):

                    self.audio_output_uncertain = True

                    self.get_logger().error(
                        "[TTS CANCEL] PC cancel timeout | "
                        f"decision_id={decision_id}"
                    )

                    self._clear_pc_audio_fallback()

                    self._finish_speech(
                        job,
                        success=False,
                        detail=(
                            "PC cancellation acknowledgement "
                            "timed out. Audio stop unconfirmed."
                        ),
                    )

                return

            requested_at = self.pc_fallback_requested_at

            if requested_at is not None:

                elapsed = now - requested_at

                if elapsed >= self.pc_fallback_timeout_sec:

                    # Timeout does not establish that
                    # remote playback has stopped.
                    self.audio_output_uncertain = True

                    self.get_logger().error(
                        "[TTS] PC fallback timeout | "
                        f"decision_id={decision_id}"
                    )

                    self._clear_pc_audio_fallback()

                    self._finish_speech(
                        job,
                        success=False,
                        detail=(
                            "PC fallback timed out. "
                            "Remote audio state unconfirmed."
                        ),
                    )

            return
        # =========================================================
        # 2. EXISTING LOCAL WORKER
        # =========================================================

        if self.audio_future is not None:

            if not self.audio_future.done():
                return

            future = self.audio_future
            phase = self.audio_phase
            job = self.active_speech

            self.audio_future = None
            self.audio_phase = None
            self.active_speech = None

            result = None
            worker_error = None

            try:
                result = future.result()

            except Exception as exc:
                worker_error = exc

            # =====================================================
            # 3. CANCELLATION AFTER WORKER COMPLETION
            # =====================================================

            canceled = bool(
                job is not None
                and job.get("cancel_requested", False)
            )

            if canceled:

                if phase in {
                    "SYNTHESIS",
                    "TURN_CUE_GENERATION",
                }:

                    if isinstance(result, Path):
                        self._remove_speech_wav(result)

                    self._finish_canceled_speech(
                        job,
                        detail=(
                            "Speech canceled before audio playback."
                        ),
                    )
                    return

                if phase == "PLAYBACK":

                    if self.audio_backend == AUDIO_BACKEND_UNITREE:

                        confirmed = bool(
                            job.get("cancel_confirmed", False)
                        )

                        if confirmed:
                            self._finish_canceled_speech(
                                job,
                                detail=(
                                    "Unitree cancellation acknowledged "
                                    "by the isolated audio player."
                                ),
                            )

                        else:
                            self.audio_output_uncertain = True

                            self._finish_speech(
                                job,
                                success=False,
                                detail=(
                                    "Unitree cancellation unconfirmed. "
                                    "Further playback inhibited."
                                ),
                            )

                        return

                    if self.audio_backend == AUDIO_BACKEND_APLAY:

                        if worker_error is None:
                            self._finish_canceled_speech(
                                job,
                                detail=(
                                    "Local playback worker ended "
                                    "after cancellation."
                                ),
                            )

                        else:
                            self.audio_output_uncertain = True

                            self._finish_speech(
                                job,
                                success=False,
                                detail=(
                                    "Local playback cancellation "
                                    f"failed: {worker_error}"
                                ),
                            )

                        return

                self.audio_output_uncertain = True

                self._finish_speech(
                    job,
                    success=False,
                    detail="Unknown cancellation state.",
                )
                return

            # =====================================================
            # 4. WORKER FAILURE
            # =====================================================

            if worker_error is not None:

                if phase == "PLAYBACK":
                    self.audio_output_uncertain = True

                self._finish_speech(
                    job,
                    success=False,
                    detail=(
                        "Audio worker failed: "
                        f"{type(worker_error).__name__}: "
                        f"{worker_error}"
                    ),
                )
                return

            # =====================================================
            # 5. SYNTHESIS COMPLETED
            # =====================================================

            if phase in {
                "SYNTHESIS",
                "TURN_CUE_GENERATION",
            }:

                wav_path = result

                if wav_path is None:

                    self._finish_speech(
                        job,
                        success=False,
                        detail=(
                            "Turn cue generation failed."
                            if phase == "TURN_CUE_GENERATION"
                            else "Speech synthesis failed."
                        ),
                    )
                    return

                if self.tts_stopping or self.audio_output_uncertain:

                    self._remove_speech_wav(wav_path)

                    self._finish_speech(
                        job,
                        success=False,
                        detail=(
                            "Audio playback inhibited "
                            "before starting."
                        ),
                    )
                    return

                is_turn_cue = bool(
                    job.get("is_turn_cue", False)
                )

                self._publish_speech_status(
                    decision_id=job["decision_id"],
                    state=SPEECH_STARTED,
                    detail=(
                        "Human turn cue playback starting."
                        if is_turn_cue
                        else "Speech playback starting."
                    ),
                )

                job["cancel_confirmed"] = False

                self.active_speech = job
                self.audio_phase = "PLAYBACK"

                self.audio_future = self.audio_worker.submit(
                    self._play_generated_audio,
                    wav_path,
                )

                return

            # =====================================================
            # 6. PLAYBACK COMPLETED
            # =====================================================

            if phase == "PLAYBACK":

                if (
                    isinstance(result, dict)
                    and result.get(
                        "pc_fallback_required", False
                    )
                ):

                    if self.audio_output_uncertain:

                        # No audio can start while the previous audio state
                        # remains unconfirmed. Resolve all queued transactions.
                        while self.speech_queue:
                            job = self.speech_queue.popleft()

                            self._finish_speech(
                                job,
                                success=False,
                                detail=(
                                    "AUDIO_OUTPUT_UNCONFIRMED: "
                                    "Queued speech rejected because "
                                    "audio playback is inhibited."
                                ),
                            )

                        self.get_logger().error(
                            "[TTS SAFETY] Playback inhibited: "
                            "previous audio stop is unconfirmed.",
                            throttle_duration_sec=2.0,
                        )
                        return

                    self._request_pc_audio_fallback(
                        job,
                        unitree_detail=str(
                            result.get("detail", "")
                        ),
                    )

                    return

                if (
                    isinstance(result, tuple)
                    and len(result) == 2
                ):
                    success = bool(result[0])
                    detail = str(result[1])

                else:
                    success = bool(result)
                    detail = (
                        "Speech playback completed."
                        if success
                        else "Speech playback failed."
                    )

                self._finish_speech(
                    job,
                    success=success,
                    detail=detail,
                )

                return

            self._finish_speech(
                job,
                success=False,
                detail=f"Unknown audio worker phase: {phase}",
            )

            return

        # =========================================================
        # 7. BLOCK NEW AUDIO IF OUTPUT IS UNCERTAIN
        # =========================================================

        if self.tts_stopping:
            return

        if self.audio_output_uncertain:
            self.get_logger().error(
                "[TTS SAFETY] Playback inhibited: "
                "previous audio stop is unconfirmed.",
                throttle_duration_sec=2.0,
            )
            return

        # =========================================================
        # 8. START NEXT QUEUED JOB
        # =========================================================

        if not self.speech_queue:
            return

        job = self._pop_next_speech()

        if job is None:
            return

        self.active_speech = job

        job["cancel_requested"] = False
        job["cancel_confirmed"] = False

        if job.get("is_turn_cue", False):

            self.audio_phase = "TURN_CUE_GENERATION"

            self.get_logger().info(
                "[TTS] TURN_CUE_GENERATION="
                f"{job['decision_id']}"
            )

            self.audio_future = self.audio_worker.submit(
                self._generate_turn_cue_wav,
            )

            return

        self.audio_phase = "SYNTHESIS"

        self.get_logger().info(
            "[TTS] SYNTHESIS="
            f"{job['decision_id']}"
        )

        self.audio_future = self.audio_worker.submit(
            self._synthesize_wav,
            job["text"],
            job["model"],
        )

        # =========================================================================
        # SPEECH STATUS
        # =========================================================================

    def _publish_speech_status(
        self,
        *,
        decision_id,
        state,
        detail
    ):
        """Publish a machine-readable classification for failed transactions."""
        detail = str(detail)

        if (
            state == SPEECH_FAILED
            and not detail.startswith(
                (
                    "AUDIO_RETRYABLE|",
                    "AUDIO_UNCERTAIN|"
                )
            )
        ):
            detail = "AUDIO_UNCERTAIN|" + detail

        status = SpeechStatus()

        status.header.stamp = (
            self.get_clock().now().to_msg()
        )

        status.header.frame_id = ""
        status.decision_id = decision_id
        status.state = state
        status.detail = detail

        self.status_publisher.publish(status)

        self.get_logger().info(
            "[TTS STATUS] "
            f"DECISION={decision_id} | "
            f"STATE={state} | "
            f"DETAIL={detail}"
        )
    def _finish_canceled_speech(self, job, detail):
        """
        Publish a single terminal CANCELED status.

        Canceled requests cannot be replayed using
        the same decision_id.
        """
        if job is None:
            return

        decision_id = str(
            job.get("decision_id", "")
        ).strip()

        if not decision_id:
            return

        if decision_id in self.completed_speech_ids:
            return

        self.pending_speech_ids.discard(decision_id)

        self.completed_speech_ids[decision_id] = None

        while len(self.completed_speech_ids) > 1000:
            self.completed_speech_ids.popitem(last=False)

        self._publish_speech_status(
            decision_id=decision_id,
            state=SPEECH_CANCELED,
            detail=detail,
        )

    def _finish_speech(
        self,
        job,
        success: bool,
        detail: str,
    ) -> None:
        """
        Finalize a speech transaction.

        COMPLETED is terminal.

        FAILED is retryable only when audio output is
        confirmed safe and no cancellation is pending.

        Uncertain failures are terminal for decision_id.
        """
        if job is None:
            return

        decision_id = str(
            job.get("decision_id", "")
        ).strip()

        if not decision_id:
            return

        if decision_id in self.completed_speech_ids:
            return

        self.pending_speech_ids.discard(
            decision_id
        )

        message = job.get("message")
        speech_kind = ""

        if message is not None:
            speech_kind = str(
                message.speech_kind or ""
            ).strip().upper()

        is_turn_cue = (
            speech_kind == "TURN_CUE"
        )

        # Never transform an uncertain result into success.
        if success and self.audio_output_uncertain:
            success = False

            detail = (
                "Speech completion rejected because "
                "audio output state is uncertain."
            )

        # Cancellation failures must not be replayed.
        cancel_requested = bool(
            job.get(
                "cancel_requested",
                False
            )
        )

        permanently_terminal = (
            success
            or self.audio_output_uncertain
            or cancel_requested
        )

        if permanently_terminal:
            self.completed_speech_ids[
                decision_id
            ] = None

            while len(self.completed_speech_ids) > 1000:
                self.completed_speech_ids.popitem(
                    last=False
                )

        if success:
            if not is_turn_cue:
                self.utterances_spoken += 1

            self.last_decision_id = decision_id

        else:
            self.synthesis_failures += 1

            if self.audio_output_uncertain:
                self.get_logger().error(
                    "[TTS SAFETY] FAILED, replay inhibited | "
                    f"decision_id={decision_id}"
                )

        if not success:
            retryable = (
                not self.audio_output_uncertain
                and not cancel_requested
            )

            detail = (
                "AUDIO_RETRYABLE|"
                if retryable
                else "AUDIO_UNCERTAIN|"
            ) + str(detail)

        self._publish_speech_status(
            decision_id=decision_id,
            state=(
                SPEECH_COMPLETED
                if success
                else SPEECH_FAILED
            ),
            detail=detail,
        )
        
    # =========================================================================
    # LANGUAGE / MODEL
    # =========================================================================

    def _resolve_model(
        self,
        language: str,
    ) -> Optional[Path]:
        """
        Resolve an available Piper model for a normalized language.

        Language validity belongs to g1_social_common.

        This method only answers whether this Speech Manager currently has
        a usable voice for the requested normalized language.
        """

        normalized = normalize_dialogue_language(
            language
        )

        if normalized == "en":

            if self.english_model_ready:
                return self.english_model

            self.get_logger().error(
                "[TTS] English fallback voice is unavailable."
            )

            return None

        if normalized == "es":

            if self.spanish_model_ready:
                return self.spanish_model

            self.get_logger().warning(
                "[TTS] Spanish voice is not configured."
            )

            return None

        # The language itself may be perfectly valid in g1_social_common.
        # It simply has no Piper voice configured in this Speech Manager.
        self.get_logger().warning(
            "[TTS] No configured Piper voice for normalized "
            f"language={normalized!r}."
        )

        return None
        
    # =========================================================================
    # PIPER COMMAND
    # =========================================================================

    def _build_piper_command(
        self,
        model: Path,
        output_file: str,
    ):
        """Build the Piper command."""

        common_arguments = [
            "--model",
            str(model),
            "--output_file",
            output_file,
            "--length_scale",
            str(self.piper_length_scale),
            "--sentence_silence",
            str(self.sentence_silence_sec),
        ]

        if (
            self.piper_backend
            == PIPER_BACKEND_MODULE
        ):
            return [
                sys.executable,
                "-m",
                "piper",
                *common_arguments,
            ]

        if (
            self.piper_backend
            == PIPER_BACKEND_EXECUTABLE
            and self.piper_executable
            is not None
        ):
            return [
                str(self.piper_executable),
                *common_arguments,
            ]

        return None

    # =========================================================================
    # SYNTHESIS
    # =========================================================================
        
    def _generate_turn_cue_wav(
        self,
    ) -> Optional[Path]:
        """
        Generate the short turn-taking cue announcing that the human may speak.

        Format:
            16000 Hz
            mono
            signed 16-bit PCM

        The cue is intentionally short and non-linguistic.

        A short fade-in/fade-out prevents hard waveform clicks.
        """

        sample_rate = 16000
        frequency_hz = 880.0
        duration_sec = 0.12
        amplitude = 0.22

        total_samples = int(
            sample_rate * duration_sec
        )

        if total_samples <= 0:
            return None

        directory = Path(
            tempfile.mkdtemp(
                prefix="g1_turn_cue_"
            )
        )

        wav_path = (
            directory
            / "turn_cue.wav"
        )

        keep_directory = False

        try:
            frames = bytearray()

            fade_samples = max(
                1,
                int(
                    sample_rate * 0.015
                ),
            )

            for index in range(
                total_samples
            ):
                time_sec = (
                    index
                    / float(sample_rate)
                )

                envelope = 1.0

                if index < fade_samples:
                    envelope = (
                        index
                        / float(fade_samples)
                    )

                remaining = (
                    total_samples
                    - index
                    - 1
                )

                if remaining < fade_samples:
                    envelope = min(
                        envelope,
                        remaining
                        / float(fade_samples),
                    )

                sample = (
                    math.sin(
                        2.0
                        * math.pi
                        * frequency_hz
                        * time_sec
                    )
                    * amplitude
                    * envelope
                )

                sample_value = int(
                    max(
                        -1.0,
                        min(
                            1.0,
                            sample,
                        ),
                    )
                    * 32767.0
                )

                frames.extend(
                    sample_value.to_bytes(
                        2,
                        byteorder="little",
                        signed=True,
                    )
                )

            with wave.open(
                str(wav_path),
                "wb",
            ) as audio:

                audio.setnchannels(1)
                audio.setsampwidth(2)
                audio.setframerate(
                    sample_rate
                )

                audio.writeframes(
                    bytes(frames)
                )

            if not wav_path.is_file():
                return None

            if wav_path.stat().st_size <= 44:
                return None

            keep_directory = True

            self.get_logger().info(
                "[TTS] Human turn cue WAV generated | "
                f"frequency={frequency_hz:.0f}Hz | "
                f"duration={duration_sec:.2f}s"
            )

            return wav_path

        except Exception as exc:
            self.get_logger().error(
                "[TTS] Turn cue generation failed: "
                f"{exc}"
            )

            return None

        finally:
            if not keep_directory:
                shutil.rmtree(
                    directory,
                    ignore_errors=True,
                )

    def _synthesize_wav(
        self,
        text: str,
        model: Path,
    ) -> Optional[Path]:
        """
        Generate one WAV.

        The caller owns cleanup when synthesis succeeds.
        """

        directory = Path(
            tempfile.mkdtemp(
                prefix="g1_piper_"
            )
        )

        wav_path = (
            directory
            / "speech.wav"
        )

        keep_directory = False

        try:
            command = (
                self._build_piper_command(
                    model=model,
                    output_file=str(
                        wav_path
                    ),
                )
            )

            if command is None:
                self.get_logger().error(
                    "[TTS] Unable to construct "
                    "the Piper command."
                )

                return None

            synthesis = subprocess.run(
                command,
                input=text.rstrip() + "\n",
                text=True,
                capture_output=True,
                check=False,
                timeout=(
                    self.synthesis_timeout_sec
                ),
            )

            if synthesis.returncode != 0:
                self.get_logger().error(
                    "[TTS] Piper synthesis failed: "
                    f"{synthesis.stderr.strip()}"
                )

                return None

            if not wav_path.is_file():
                self.get_logger().error(
                    "[TTS] Piper did not create "
                    "the WAV file."
                )

                return None

            wav_size = (
                wav_path.stat().st_size
            )

            if wav_size <= 44:
                self.get_logger().error(
                    "[TTS] Piper produced no "
                    "usable WAV audio."
                )

                return None

            self.get_logger().info(
                "[TTS] Generated WAV: "
                f"{wav_size} bytes"
            )

            keep_directory = True

            return wav_path

        except subprocess.TimeoutExpired:
            self.get_logger().error(
                "[TTS] Piper synthesis "
                "timed out."
            )

            return None

        except Exception as exc:
            self.get_logger().error(
                "[TTS] Synthesis failed: "
                f"{exc}"
            )

            return None

        finally:
            if not keep_directory:
                shutil.rmtree(
                    directory,
                    ignore_errors=True,
                )

    # =========================================================================
    # AUDIO ROUTING
    # =========================================================================

    def _play_generated_audio(
        self,
        wav_path: Path,
    ):
        """Play one generated WAV and preserve playback diagnostics."""

        try:
            return self._play_audio(
                wav_path
            )

        finally:
            self._remove_speech_wav(
                wav_path
            )
                
    @staticmethod
    def _remove_speech_wav(
        wav_path: Path,
    ) -> None:
        """Remove the temporary speech directory."""

        try:
            shutil.rmtree(
                wav_path.parent,
                ignore_errors=True,
            )

        except Exception:
            pass

    def _play_audio(self, wav_path: Path):
        """
        Route one audio transaction to its selected backend.

        Unitree policy:
        - At most two attempts for recoverable errors.
        - No retry after cancellation.
        - No retry when speaker state is uncertain.
        - PC fallback only after recoverable Unitree failures.
        - Never switch to fallback after cancellation.
        """

        def canceled_or_blocked():
            with self.playback_process_lock:
                job = self.active_speech

                canceled = bool(
                    job is not None
                    and job.get("cancel_requested", False)
                )

            return canceled, bool(
                self.audio_output_uncertain
            )

        canceled, blocked = canceled_or_blocked()

        if canceled:
            return False, "Audio canceled before playback."

        if blocked:
            return False, "Audio output state is uncertain."

        # =====================================================
        # LOCAL PC AUDIO
        # =====================================================

        if self.audio_backend == AUDIO_BACKEND_APLAY:

            success = self._play_with_aplay(wav_path)

            canceled, blocked = canceled_or_blocked()

            if canceled:
                return False, "Local playback cancellation requested."

            if blocked:
                return False, "Local audio output state is uncertain."

            if success:
                return (
                    True,
                    "Speech playback completed on local PC audio.",
                )

            return False, "Local PC audio playback failed."

        # =====================================================
        # UNITREE AUDIO
        # =====================================================

        if self.audio_backend == AUDIO_BACKEND_UNITREE:

            failures = []

            for attempt in (1, 2):

                canceled, blocked = canceled_or_blocked()

                if canceled:
                    return (
                        False,
                        "Unitree playback cancellation requested.",
                    )

                if blocked:
                    return (
                        False,
                        "Unitree speaker state is uncertain; "
                        "retry inhibited.",
                    )

                self.get_logger().info(
                    "[TTS] Unitree playback attempt "
                    f"{attempt}/2"
                )

                result = self._play_with_unitree(wav_path)

                if (
                    isinstance(result, tuple)
                    and len(result) == 2
                ):
                    success = bool(result[0])
                    detail = str(result[1])
                else:
                    success = bool(result)
                    detail = (
                        "Unitree playback completed."
                        if success
                        else "Unitree playback failed."
                    )

                canceled, blocked = canceled_or_blocked()

                # A canceled transaction must never
                # trigger another attempt or PC fallback.
                if canceled:
                    return False, detail

                # A failed or timed-out stop may leave the
                # physical speaker in an unknown state.
                if blocked:
                    return (
                        False,
                        "Unitree speaker state unconfirmed: "
                        + detail,
                    )

                if success:
                    return (
                        True,
                        (
                            "Speech playback completed on Unitree audio."
                            if attempt == 1
                            else
                            "Speech playback completed after one Unitree retry."
                        ),
                    )

                failures.append(detail)

                self.get_logger().warning(
                    "[TTS] Unitree playback failed | "
                    f"attempt={attempt}/2 | "
                    f"detail={detail}"
                )

            # =================================================
            # PC FALLBACK ONLY AFTER RECOVERABLE FAILURES
            # =================================================

            canceled, blocked = canceled_or_blocked()

            if canceled:
                return (
                    False,
                    "PC fallback suppressed: transaction canceled.",
                )

            if blocked:
                return (
                    False,
                    "PC fallback suppressed: "
                    "Unitree speaker state uncertain.",
                )

            return {
                "pc_fallback_required": True,
                "detail": " | ".join(failures),
            }

        # =====================================================
        # INVALID BACKEND
        # =====================================================

        return (
            False,
            "No valid audio backend is selected.",
        )
            # =========================================================================
            # APLAY
            # =========================================================================

    def _play_with_aplay(
        self,
        wav_path: Path,
    ) -> bool:
        """Play WAV through the PC audio device."""

        if self.audio_player_path is None:
            self.get_logger().error(
                "[TTS] aplay backend "
                "is unavailable."
            )

            return False

        try:
            with wave.open(
                str(wav_path),
                "rb",
            ) as audio:

                frame_rate = (
                    audio.getframerate()
                )

                if frame_rate <= 0:
                    return False

                duration_sec = (
                    audio.getnframes()
                    / float(frame_rate)
                )

            with self.playback_process_lock:
                job = self.active_speech

                if job is not None and job.get(
                    "cancel_requested", False
                ):
                    return False

                process = subprocess.Popen(
                    [
                        self.audio_player_path,
                        "-q",
                        str(wav_path),
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )

                self.aplay_process = process

            try:
                _, stderr = process.communicate(
                    timeout=max(10.0, duration_sec + 5.0)
                )

            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()

                self.get_logger().error(
                    "[TTS] aplay timed out."
                )
                return False

            finally:
                with self.playback_process_lock:
                    if self.aplay_process is process:
                        self.aplay_process = None

            if process.returncode != 0:
                self.get_logger().warning(
                    "[TTS] aplay ended without success | "
                    f"returncode={process.returncode} | "
                    f"stderr={(stderr or '').strip()}"
                )
                return False

            return True

        except subprocess.TimeoutExpired:
            self.get_logger().error(
                "[TTS] aplay timed out."
            )

            return False

        except Exception as exc:
            self.get_logger().error(
                "[TTS] aplay failed: "
                f"{exc}"
            )

            return False

    # =========================================================================
    # UNITREE
    # =========================================================================

    def _play_with_unitree(self, wav_path: Path):
        """
        Execute one isolated Unitree audio attempt.

        Protocol:
        - A subprocess that was never created cannot have started audio.
        - A launched subprocess must explicitly report successful completion.
        - CANCELED requires exit code 3 and UNITREE_CANCEL_CONFIRMED.
        - An ambiguous subprocess result inhibits future playback.
        - A cancellation never authorizes retry or PC fallback.

        The caller (_play_audio) decides whether to retry.
        """

        # ---------------------------------------------------------
        # 1. Convert WAV to Unitree PCM
        # ---------------------------------------------------------

        with self.playback_process_lock:
            job = self.active_speech

            if job is None:
                return False, "No active speech transaction."

            if job.get("cancel_requested", False):
                job["cancel_confirmed"] = True
                return (
                    False,
                    "Unitree canceled before PCM conversion.",
                )

            if self.audio_output_uncertain:
                return (
                    False,
                    "Previous audio output remains unconfirmed.",
                )

        pcm_data = self._convert_wav_for_unitree(wav_path)

        if not pcm_data:
            return False, "Unitree PCM conversion failed."

        pcm_path = wav_path.with_suffix(".pcm")
        process = None

        try:
            pcm_path.write_bytes(pcm_data)

            duration_sec = (
                len(pcm_data)
                / float(
                    UNITREE_SAMPLE_RATE_HZ
                    * UNITREE_CHANNEL_COUNT
                    * UNITREE_SAMPLE_WIDTH_BYTES
                )
            )

            client_timeout_sec = 5.0

            # Allow time for two SDK calls plus playback.
            # This is only a subprocess supervision timeout;
            # it is not an audio-stop acknowledgement.
            timeout_sec = max(
                15.0,
                duration_sec + 2.0 * client_timeout_sec + 5.0,
            )

            command = [
                sys.executable,
                "-m",
                "g1_robot_explainability.unitree_audio_player",
                "--interface",
                self.unitree_network_interface,
                "--pcm-file",
                str(pcm_path),
                "--volume",
                str(self.unitree_volume),
                "--stream-name",
                self.unitree_stream_name,
                "--client-timeout",
                str(client_timeout_sec),
            ]

            # -----------------------------------------------------
            # 2. Atomic check and process creation
            # -----------------------------------------------------

            with self.playback_process_lock:

                if self.active_speech is not job:
                    return (
                        False,
                        "Active speech transaction changed.",
                    )

                if job.get("cancel_requested", False):

                    # This attempt did not create a process.
                    job["cancel_confirmed"] = True

                    return (
                        False,
                        "Unitree canceled before process creation.",
                    )

                if self.audio_output_uncertain:
                    return (
                        False,
                        "Unitree playback inhibited: "
                        "audio state is uncertain.",
                    )

                try:
                    process = subprocess.Popen(
                        command,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                    )

                except Exception as exc:
                    # Popen failed before returning a child handle.
                    # No playback completion can be inferred.
                    self.get_logger().error(
                        "[TTS UNITREE] Cannot start player: "
                        f"{type(exc).__name__}: {exc}"
                    )

                    return (
                        False,
                        f"Unitree process creation failed: {exc}",
                    )

                self.unitree_process = process

            # -----------------------------------------------------
            # 3. Wait for isolated player
            # -----------------------------------------------------

            try:
                stdout, stderr = process.communicate(
                    timeout=timeout_sec
                )

            except subprocess.TimeoutExpired:

                self.audio_output_uncertain = True

                try:
                    process.kill()
                    process.communicate()

                except Exception as exc:
                    self.get_logger().error(
                        "[TTS UNITREE] Timeout cleanup failed: "
                        f"{exc}"
                    )

                return (
                    False,
                    "Unitree player timed out. "
                    "Physical audio stop was not confirmed.",
                )

            stdout = stdout or ""
            stderr = stderr or ""
            returncode = process.returncode

            # -----------------------------------------------------
            # 4. Classify subprocess result
            # -----------------------------------------------------

            cancellation_acknowledged = (
                returncode == 3
                and "UNITREE_CANCEL_CONFIRMED" in stdout
            )

            playback_completed = (
                returncode == 0
                and "UNITREE_PLAYBACK_OK" in stdout
            )

            prestream_failed = (
                returncode == 4
                and "UNITREE_PRESTREAM_FAILED:" in stderr
            )

            with self.playback_process_lock:
                canceled = bool(
                    job.get("cancel_requested", False)
                )

                # Acknowledgment is valid only for this exact
                # active job, not another speech transaction.
                same_job = (
                    self.active_speech is job
                )

                if canceled and cancellation_acknowledged and same_job:
                    job["cancel_confirmed"] = True

            # -----------------------------------------------------
            # 5. Cancellation requested by Speech Manager
            # -----------------------------------------------------

            if canceled:

                if cancellation_acknowledged and same_job:

                    self.get_logger().warning(
                        "[TTS UNITREE] Cancellation confirmed | "
                        f"decision_id={job['decision_id']}"
                    )

                    return (
                        False,
                        "Unitree cancellation acknowledged.",
                    )

                # Even a normal return code may race with a cancel.
                # Never transform it into confirmed CANCELED.
                self.audio_output_uncertain = True

                return (
                    False,
                    "Unitree cancellation not confirmed | "
                    f"returncode={returncode} | "
                    f"stderr={stderr.strip()}",
                )

            # The isolated player failed before making any PlayStream call.
            # It did not start a new audio stream in this attempt.
            # This failure must not poison the audio safety latch.
            if prestream_failed:

                detail = (
                    "Unitree preparation failed before PlayStream | "
                    f"{stderr.strip()}"
                )

                self.get_logger().warning(
                    "[TTS UNITREE] " + detail
                )

                return False, detail

            # -----------------------------------------------------
            # 6. Normal successful completion
            # -----------------------------------------------------

            if playback_completed:

                self.get_logger().info(
                    "[TTS UNITREE] Playback completed | "
                    f"decision_id={job['decision_id']}"
                )

                return (
                    True,
                    "Unitree playback completed.",
                )

            # -----------------------------------------------------
            # 7. Ambiguous result after process launch
            # -----------------------------------------------------

            # A nonzero SDK error cannot prove that PlayStream
            # never started.
            #
            # A zero return code without the completion marker
            # is also insufficient evidence of completion.
            #
            # Neither condition may trigger retry or PC fallback.

            self.audio_output_uncertain = True

            detail = (
                "Unitree playback state unconfirmed | "
                f"returncode={returncode}"
            )

            if stderr.strip():
                detail += f" | {stderr.strip()}"

            self.get_logger().error(
                f"[TTS UNITREE] {detail}"
            )

            return False, detail

        except Exception as exc:

            # Distinguish failures before and after Popen.
            # Once a process exists, an unhandled exception
            # leaves the speaker state uncertain.
            if process is not None:
                self.audio_output_uncertain = True

            self.get_logger().error(
                "[TTS UNITREE] Worker exception: "
                f"{type(exc).__name__}: {exc}"
            )

            return (
                False,
                f"Unitree playback error: {exc}",
            )

        finally:

            if process is not None:

                with self.playback_process_lock:

                    if self.unitree_process is process:
                        self.unitree_process = None

            try:
                pcm_path.unlink(missing_ok=True)

            except Exception:
                pass

    def _convert_wav_for_unitree(
        self,
        wav_path: Path,
    ) -> Optional[bytes]:
        """
        Convert Piper WAV into Unitree-compatible PCM.

        Output:
            16000 Hz
            mono
            signed 16-bit PCM
        """

        try:
            with wave.open(
                str(wav_path),
                "rb",
            ) as source:

                channels = (
                    source.getnchannels()
                )

                sample_width = (
                    source.getsampwidth()
                )

                sample_rate = (
                    source.getframerate()
                )

                pcm_data = (
                    source.readframes(
                        source.getnframes()
                    )
                )

            if channels not in (1, 2):
                self.get_logger().error(
                    "[TTS] Unsupported channel "
                    f"count: {channels}"
                )

                return None

            if sample_width not in (
                1,
                2,
                3,
                4,
            ):
                self.get_logger().error(
                    "[TTS] Unsupported sample "
                    f"width: {sample_width}"
                )

                return None

            original_format = (
                sample_rate,
                channels,
                sample_width,
            )

            # WAV 8-bit PCM is unsigned.
            if sample_width == 1:
                pcm_data = audioop.bias(
                    pcm_data,
                    1,
                    -128,
                )

            # Stereo -> mono.
            if channels == 2:
                pcm_data = audioop.tomono(
                    pcm_data,
                    sample_width,
                    0.5,
                    0.5,
                )

                channels = 1

            # Convert to signed 16-bit.
            if (
                sample_width
                != UNITREE_SAMPLE_WIDTH_BYTES
            ):
                pcm_data = audioop.lin2lin(
                    pcm_data,
                    sample_width,
                    UNITREE_SAMPLE_WIDTH_BYTES,
                )

                sample_width = (
                    UNITREE_SAMPLE_WIDTH_BYTES
                )

            # Resample to 16 kHz.
            if (
                sample_rate
                != UNITREE_SAMPLE_RATE_HZ
            ):
                pcm_data, _ = audioop.ratecv(
                    pcm_data,
                    sample_width,
                    channels,
                    sample_rate,
                    UNITREE_SAMPLE_RATE_HZ,
                    None,
                )

                sample_rate = (
                    UNITREE_SAMPLE_RATE_HZ
                )

            self.get_logger().info(
                "[TTS] Unitree PCM conversion: "
                f"{original_format[0]} Hz/"
                f"{original_format[1]} ch/"
                f"{original_format[2] * 8} bit "
                "-> "
                f"{sample_rate} Hz/"
                f"{channels} ch/"
                f"{sample_width * 8} bit"
            )

            return pcm_data

        except wave.Error as exc:
            self.get_logger().error(
                "[TTS] Invalid Piper WAV: "
                f"{exc}"
            )

            return None

        except Exception as exc:
            self.get_logger().error(
                "[TTS] WAV conversion failed: "
                f"{exc}"
            )

            return None

    # =========================================================================
    # MODEL VALIDATION
    # =========================================================================

    def _validate_model(
        self,
        model: Path,
        label: str,
    ) -> bool:
        """Validate model and Piper JSON configuration."""

        valid = True

        if not model.is_file():

            if self.enabled:
                self.get_logger().error(
                    f"[TTS] {label} model "
                    f"not found: {model}"
                )

            valid = False

        configuration = Path(
            str(model) + ".json"
        )

        if not configuration.is_file():

            if self.enabled:
                self.get_logger().error(
                    f"[TTS] {label} configuration "
                    f"not found: {configuration}"
                )

            valid = False

        return valid

    # =========================================================================
    # SHUTDOWN
    # =========================================================================

    def stop_audio(self) -> None:
        """
        Stop the Speech Manager with bounded worker shutdown.

        Safety policy:
        - Prevent new speech work immediately.
        - Cancel queued speech without starting playback.
        - Request cancellation of the active transaction.
        - Terminate the active local playback subprocess.
        - Allow the worker to report its final result.
        - Never claim Unitree cancellation without acknowledgment.
        - Never block indefinitely in ThreadPoolExecutor.shutdown().
        """

        if self.tts_stopping:
            return

        self.tts_stopping = True

        # ---------------------------------------------------------
        # 1. Stop periodic scheduling
        # ---------------------------------------------------------

        try:
            self.audio_timer.cancel()

        except Exception as exc:
            self.get_logger().warning(
                "[TTS SHUTDOWN] Unable to cancel audio timer: "
                f"{exc}"
            )

        # ---------------------------------------------------------
        # 2. Resolve queued transactions
        # ---------------------------------------------------------

        while self.speech_queue:

            job = self.speech_queue.popleft()

            job["cancel_requested"] = True
            job["cancel_confirmed"] = True

            self._finish_canceled_speech(
                job,
                detail=(
                    "Speech canceled during TTS shutdown "
                    "before execution."
                ),
            )

        # ---------------------------------------------------------
        # 3. Remote PC fallback
        # ---------------------------------------------------------

        if self.pc_fallback_waiting:

            job = self.pc_fallback_job

            decision_id = self.pc_fallback_decision_id

            if job is not None:
                job["cancel_requested"] = True
                job["cancel_confirmed"] = False

            # Shutdown cannot wait for an asynchronous ROS
            # acknowledgement while the executor is stopping.
            # Send cancellation, but never infer success.

            if decision_id:

                cancel_msg = SpeechCancelRequest()
                cancel_msg.decision_id = decision_id

                try:
                    self.pc_fallback_cancel_pub.publish(
                        cancel_msg
                    )

                    self.get_logger().warning(
                        "[TTS SHUTDOWN] PC cancel sent | "
                        f"decision_id={decision_id}"
                    )

                except Exception as exc:
                    self.get_logger().error(
                        "[TTS SHUTDOWN] PC cancel send failed | "
                        f"decision_id={decision_id} | "
                        f"error={exc}"
                    )

            # Publishing is not an acknowledgement.
            # The remote audio state remains unconfirmed.
            self.audio_output_uncertain = True

            self._clear_pc_audio_fallback()

            self._finish_speech(
                job,
                success=False,
                detail=(
                    "TTS shutdown during PC fallback. "
                    "Remote stop requested but unconfirmed."
                ),
            )

        # ---------------------------------------------------------
        # 4. Mark active transaction as canceled
        # ---------------------------------------------------------

        process = None
        phase = self.audio_phase
        backend = self.audio_backend

        with self.playback_process_lock:

            job = self.active_speech

            if job is not None:

                job["cancel_requested"] = True

                # Preserve an acknowledgment already obtained.
                # Otherwise confirmation belongs to the worker.
                job.setdefault("cancel_confirmed", False)

            if phase == "PLAYBACK":

                if backend == AUDIO_BACKEND_UNITREE:
                    process = self.unitree_process

                elif backend == AUDIO_BACKEND_APLAY:
                    process = self.aplay_process

            # Send the interruption while process identity
            # is still protected by the lock.
            if process is not None:

                try:
                    if process.poll() is None:
                        process.terminate()

                        self.get_logger().warning(
                            "[TTS SHUTDOWN] Playback "
                            "termination requested | "
                            f"backend={backend}"
                        )

                except ProcessLookupError:
                    pass

                except Exception as exc:

                    self.audio_output_uncertain = True

                    self.get_logger().error(
                        "[TTS SHUTDOWN] Process termination "
                        f"request failed: {exc}"
                    )

        # ---------------------------------------------------------
        # 5. Bound the worker completion wait
        # ---------------------------------------------------------

        # Unitree may need time to execute PlayStop through
        # its signal handler and SDK client.
        #
        # This deadline is NOT proof of physical audio stop.

        shutdown_timeout_sec = 8.0

        deadline = (
            time.monotonic() + shutdown_timeout_sec
        )

        future = self.audio_future

        if future is not None:

            while (
                not future.done()
                and time.monotonic() < deadline
            ):
                time.sleep(0.05)

            # -----------------------------------------------------
            # 6. Worker finished: process its actual result
            # -----------------------------------------------------

            if future.done():

                # _audio_tick owns the normal terminal-state
                # classification, including Unitree cancel ACK.
                self._audio_tick()

            # -----------------------------------------------------
            # 7. Worker still active: fail closed
            # -----------------------------------------------------

            else:

                self.audio_output_uncertain = True

                # Escalate only after the graceful deadline.
                # Killing a process is not an SDK PlayStop ACK.
                with self.playback_process_lock:

                    process = (
                        self.unitree_process
                        if backend == AUDIO_BACKEND_UNITREE
                        else self.aplay_process
                        if backend == AUDIO_BACKEND_APLAY
                        else None
                    )

                    if process is not None:

                        try:
                            if process.poll() is None:
                                process.kill()

                        except ProcessLookupError:
                            pass

                        except Exception as exc:
                            self.get_logger().error(
                                "[TTS SHUTDOWN] Forced process "
                                f"termination failed: {exc}"
                            )

                self.get_logger().error(
                    "[TTS SHUTDOWN] Audio worker did not "
                    "finish before deadline. "
                    "Audio output state remains uncertain."
                )

                # Do not publish CANCELED: there is no
                # confirmed backend-specific acknowledgment.
                if job is not None:

                    self._finish_speech(
                        job,
                        success=False,
                        detail=(
                            "TTS shutdown timed out while "
                            "waiting for the active worker. "
                            "Audio stop unconfirmed."
                        ),
                    )

                # Request cancellation for work that has
                # not yet begun.
                future.cancel()

        # ---------------------------------------------------------
        # 8. Shut down executor without an unbounded wait
        # ---------------------------------------------------------

        try:
            self.audio_worker.shutdown(
                wait=False,
                cancel_futures=True,
            )

        except TypeError:
            # Compatibility with older Python versions.
            self.audio_worker.shutdown(
                wait=False
            )

        # ---------------------------------------------------------
        # 9. Final report
        # ---------------------------------------------------------

        if self.audio_output_uncertain:

            self.get_logger().error(
                "[TTS SHUTDOWN] Finished with "
                "unconfirmed audio output state."
            )

        else:

            self.get_logger().info(
                "[TTS SHUTDOWN] Speech worker "
                "shutdown sequence completed."
            )

    # =========================================================================
    # REPORTING
    # =========================================================================

    def _log_initialization(
        self,
    ) -> None:
        """Report resolved Speech Manager configuration."""

        self.get_logger().info("")
        self.get_logger().info(
            "=" * 84
        )
        self.get_logger().info(
            "  COMMON SPEECH MANAGER - PIPER"
        )
        self.get_logger().info(
            "=" * 84
        )

        self.get_logger().info(
            f"  Input topic        : "
            f"{self.input_topic}"
        )

        self.get_logger().info(
            f"  Status topic       : "
            f"{self.status_topic}"
        )

        self.get_logger().info(
            "  Piper requested    : "
            f"{self.piper_backend_requested}"
        )

        self.get_logger().info(
            "  Piper resolved     : "
            f"{self.piper_backend or 'unavailable'}"
        )

        self.get_logger().info(
            "  Piper executable   : "
            f"{self.piper_executable or 'not found'}"
        )

        self.get_logger().info(
            "  ROS distro         : "
            f"{self.ros_distro}"
        )

        self.get_logger().info(
            "  Audio requested    : "
            f"{self.audio_backend_requested}"
        )

        self.get_logger().info(
            "  Audio resolved     : "
            f"{self.audio_backend or 'unavailable'}"
        )

        if (
            self.audio_backend
            == AUDIO_BACKEND_APLAY
        ):
            self.get_logger().info(
                "  Audio player       : "
                f"{self.audio_player_path or 'not found'}"
            )

        if (
            self.audio_backend
            == AUDIO_BACKEND_UNITREE
        ):
            self.get_logger().info(
                "  Unitree interface  : "
                f"{self.unitree_network_interface}"
            )

            self.get_logger().info(
                "  Unitree volume     : "
                f"{self.unitree_volume}"
            )

        self.get_logger().info(
            f"  Speech rate        : "
            f"{self.speech_rate:.2f}"
        )

        self.get_logger().info(
            "  Sentence silence   : "
            f"{self.sentence_silence_sec:.2f} s"
        )

        self.get_logger().info(
            f"  Spanish voice      : "
            f"{self.spanish_model}"
        )

        self.get_logger().info(
            f"  English voice      : "
            f"{self.english_model}"
        )

        self.get_logger().info(
            f"  Runtime ready      : "
            f"{self.runtime_ready}"
        )

        self.get_logger().info(
            f"  Enabled            : "
            f"{self.enabled}"
        )

        self.get_logger().info(
            "-" * 84
        )

    def log_diagnostics(
        self,
    ) -> None:
        """Periodic TTS health report."""

        active_decision = "NONE"

        if self.active_speech is not None:
            active_decision = (
                self.active_speech.get(
                    "decision_id",
                    "NONE",
                )
            )

        self.get_logger().info(
            "[TTS] "
            f"MESSAGES_RX={self.messages_received} | "
            f"SPOKEN={self.utterances_spoken} | "
            f"FAILURES={self.synthesis_failures} | "
            f"UNSUPPORTED_LANG="
            f"{self.unsupported_languages} | "
            f"PIPER={self.piper_backend or 'NONE'} | "
            f"AUDIO={self.audio_backend or 'NONE'} | "
            f"READY={self.runtime_ready} | "
            f"QUEUED={len(self.speech_queue)} | "
            f"PHASE={self.audio_phase or 'IDLE'} | "
            f"ACTIVE={active_decision} | "
            f"LAST={self.last_decision_id or 'NONE'}"
        )


# =============================================================================
# MAIN
# =============================================================================

def main(args=None) -> None:
    """Run the Common Speech Manager."""

    rclpy.init(
        args=args
    )

    node = PiperTTSNode()

    try:
        rclpy.spin(
            node
        )

    except KeyboardInterrupt:
        pass

    finally:

        try:
            node.stop_audio()

        finally:
            node.destroy_node()

            if rclpy.ok():
                rclpy.shutdown()


if __name__ == "__main__":
    main()