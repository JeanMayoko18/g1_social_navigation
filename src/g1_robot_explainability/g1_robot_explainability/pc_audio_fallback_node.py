#!/usr/bin/env python3

"""
PC audio fallback executor.

Responsibility
--------------
Execute an explicitly authorized remote speech transaction on the PC.

This node is NOT a speech authority.

It does not subscribe to:

    /explainability/text

It listens only to:

    /explainability/pc_audio_fallback/request

and reports execution status on:

    /explainability/pc_audio_fallback/status

Authority model
---------------
The robot Speech Manager remains the only component allowed to decide:

    - what must be spoken;
    - when speech must occur;
    - whether Unitree must be retried;
    - whether PC fallback is allowed;
    - what final SpeechStatus is exposed to HRI.

The PC node only executes one already-authorized request.

Supported transactions
----------------------
Normal speech:
    ExplanationText
        -> local Piper synthesis
        -> local PC aplay
        -> SpeechStatus

TURN_CUE:
    ExplanationText(speech_kind="TURN_CUE")
        -> generated short tone
        -> local PC aplay
        -> SpeechStatus

Correlation invariant
---------------------
request.decision_id == status.decision_id
"""

import importlib.util
import math
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import wave

from collections import OrderedDict
from pathlib import Path
from typing import Optional


import rclpy

from rclpy.node import Node

from g1_social_msgs.msg import (
    ExplanationText,
    SpeechCancelRequest,
    SpeechStatus,
)

from g1_social_common.dialogue_texts import (
    normalize_dialogue_language,
)

from g1_qos_manager.profiles import (
    reliable_qos,
)

from g1_qos_manager.diagnostics import (
    log_qos_resolution,
)

from g1_qos_manager.resolver import (
    resolve_subscription_qos,
)


# =============================================================================
# CONSTANTS
# =============================================================================

SPEECH_STARTED = "STARTED"
SPEECH_COMPLETED = "COMPLETED"
SPEECH_FAILED = "FAILED"
SPEECH_CANCELED = "CANCELED"

SPEECH_KIND_TURN_CUE = "TURN_CUE"

PIPER_BACKEND_AUTO = "auto"
PIPER_BACKEND_MODULE = "module"
PIPER_BACKEND_EXECUTABLE = "executable"

SUPPORTED_PIPER_BACKENDS = {
    PIPER_BACKEND_AUTO,
    PIPER_BACKEND_MODULE,
    PIPER_BACKEND_EXECUTABLE,
}


# =============================================================================
# NODE
# =============================================================================


class PCAudioFallbackNode(Node):
    """
    Passive PC-side audio fallback executor.

    The node never initiates speech by itself.
    """

    def __init__(self):
        super().__init__(
            "pc_audio_fallback"
        )

        # =====================================================================
        # PARAMETERS
        # =====================================================================

        self.declare_parameter(
            "enabled",
            True,
        )

        self.declare_parameter(
            "request_topic",
            "/explainability/pc_audio_fallback/request",
        )

        self.declare_parameter(
            "status_topic",
            "/explainability/pc_audio_fallback/status",
        )

        self.declare_parameter(
            "cancel_request_topic",
            "/explainability/pc_audio_fallback/cancel_request",
        )

        self.declare_parameter(
            "piper_model_directory",
            os.environ.get(
                "G1_PIPER_MODEL_DIRECTORY",
                str(
                    Path.home()
                    / "g1_ws"
                    / "models"
                    / "piper"
                ),
            ),
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

        self.declare_parameter(
            "audio_player",
            "aplay",
        )

        self.declare_parameter(
            "speech_rate",
            1.0,
        )

        self.declare_parameter(
            "sentence_silence_sec",
            0.20,
        )

        self.declare_parameter(
            "synthesis_timeout_sec",
            20.0,
        )

        self.declare_parameter(
            "queue_size",
            10,
        )

        # =====================================================================
        # PARAMETER VALUES
        # =====================================================================

        self.enabled = bool(
            self.get_parameter(
                "enabled"
            ).value
        )

        self.request_topic = str(
            self.get_parameter(
                "request_topic"
            ).value
        ).strip()

        self.status_topic = str(
            self.get_parameter(
                "status_topic"
            ).value
        ).strip()

        self.cancel_request_topic = str(
            self.get_parameter("cancel_request_topic").value
        ).strip()

        if not self.cancel_request_topic:
            raise ValueError(
                "cancel_request_topic cannot be empty."
            )
        
        self.piper_model_directory = Path(
            str(
                self.get_parameter(
                    "piper_model_directory"
                ).value
            )
        ).expanduser()

        self.piper_backend_requested = str(
            self.get_parameter(
                "piper_backend"
            ).value
        ).strip().lower()

        self.piper_executable_requested = str(
            self.get_parameter(
                "piper_executable"
            ).value
        ).strip()

        self.audio_player = str(
            self.get_parameter(
                "audio_player"
            ).value
        ).strip()

        self.speech_rate = float(
            self.get_parameter(
                "speech_rate"
            ).value
        )

        self.sentence_silence_sec = float(
            self.get_parameter(
                "sentence_silence_sec"
            ).value
        )

        self.synthesis_timeout_sec = float(
            self.get_parameter(
                "synthesis_timeout_sec"
            ).value
        )

        self.queue_size = int(
            self.get_parameter(
                "queue_size"
            ).value
        )

        # =====================================================================
        # VALIDATION
        # =====================================================================

        if not self.request_topic:
            raise ValueError(
                "request_topic cannot be empty."
            )

        if not self.status_topic:
            raise ValueError(
                "status_topic cannot be empty."
            )

        if not 0.5 <= self.speech_rate <= 2.0:
            raise ValueError(
                "speech_rate must be between 0.5 and 2.0."
            )

        if not 0.0 <= self.sentence_silence_sec <= 2.0:
            raise ValueError(
                "sentence_silence_sec must be between 0 and 2 seconds."
            )

        if (
            not math.isfinite(
                self.synthesis_timeout_sec
            )
            or self.synthesis_timeout_sec <= 0.0
        ):
            raise ValueError(
                "synthesis_timeout_sec must be finite and positive."
            )

        if self.queue_size < 1:
            raise ValueError(
                "queue_size must be positive."
            )

        self.piper_length_scale = (
            1.0 / self.speech_rate
        )

        # =====================================================================
        # MODELS
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
            or str(
                default_spanish_model
            )
        ).expanduser()

        self.english_model = Path(
            english_model_value
            or str(
                default_english_model
            )
        ).expanduser()

        self.spanish_model_ready = (
            self._validate_model(
                self.spanish_model,
                "Spanish",
            )
        )

        self.english_model_ready = (
            self._validate_model(
                self.english_model,
                "English",
            )
        )

        # =====================================================================
        # PIPER
        # =====================================================================

        self.piper_module_available = (
            importlib.util.find_spec(
                "piper"
            )
            is not None
        )

        self.piper_executable = (
            self._find_piper_executable()
        )

        self.piper_backend = (
            self._resolve_piper_backend()
        )

        # =====================================================================
        # LOCAL PC AUDIO
        # =====================================================================

        self.audio_player_path = (
            shutil.which(
                self.audio_player
            )
        )

        self.audio_ready = (
            self.audio_player_path
            is not None
        )

        if not self.audio_ready:
            self.get_logger().error(
                "[PC AUDIO FALLBACK] "
                f"Audio player not found: {self.audio_player!r}"
            )

        # =====================================================================
        # IDEMPOTENCY / WORKER
        # =====================================================================

        self.pending_ids = set()

        # decision_id -> (state, detail)
        self.terminal_results = OrderedDict()

        self.work_queue = queue.Queue(
            maxsize=self.queue_size
        )

        self.stop_event = (
            threading.Event()
        )

        # Protect the active transaction and its playback process.
        self.playback_lock = threading.Lock()

        # The worker owns this transaction while executing it.
        self.active_job = None

        # Handle of the local aplay subprocess.
        self.aplay_process = None

        # Cancellation is tracked per decision_id.
        self.cancel_requested_ids = set()

        # A failed or unconfirmed stop inhibits future playback.
        self.audio_output_uncertain = False

        self.worker = threading.Thread(
            target=self._worker_loop,
            name="pc_audio_fallback_worker",
            daemon=True,
        )

        # =====================================================================
        # ROS
        # =====================================================================

        request_qos = (
            resolve_subscription_qos(
                self,
                self.request_topic,
                fallback_depth=10,
                prefer_sensor_data=False,
            )
        )

        log_qos_resolution(
            self,
            self.request_topic,
            request_qos,
        )

        self.status_pub = (
            self.create_publisher(
                SpeechStatus,
                self.status_topic,
                reliable_qos(
                    depth=20
                ),
            )
        )

        self.request_sub = (
            self.create_subscription(
                ExplanationText,
                self.request_topic,
                self._request_callback,
                request_qos,
            )
        )

        self.cancel_sub = self.create_subscription(
            SpeechCancelRequest,
            self.cancel_request_topic,
            self._cancel_callback,
            reliable_qos(depth=20),
        )

        self.worker.start()

        self.get_logger().info(
            "=" * 84
        )

        self.get_logger().info(
            "PC AUDIO FALLBACK EXECUTOR READY"
        )

        self.get_logger().info(
            f"Request topic : {self.request_topic}"
        )

        self.get_logger().info(
            f"Status topic  : {self.status_topic}"
        )

        self.get_logger().info(
            "Authority     : PASSIVE EXECUTOR ONLY"
        )

        self.get_logger().info(
            f"Piper        : {self.piper_backend or 'UNAVAILABLE'}"
        )

        self.get_logger().info(
            f"aplay        : {self.audio_player_path or 'UNAVAILABLE'}"
        )

        self.get_logger().info(
            "=" * 84
        )

    # =========================================================================
    # REQUEST
    # =========================================================================

    def _cancel_callback(self, msg):
        """
        Register one remote cancellation.

        The worker owns terminal confirmation.
        Never publish CANCELED before process termination.
        """

        decision_id = str(
            msg.decision_id or ""
        ).strip()

        if not decision_id:
            return

        with self.playback_lock:

            if decision_id in self.terminal_results:
                return

            self.cancel_requested_ids.add(decision_id)

            job = self.active_job

            process = None

            if (
                job is not None
                and job.get("decision_id") == decision_id
            ):
                process = self.aplay_process

            if process is not None:
                try:
                    if process.poll() is None:
                        process.terminate()

                except ProcessLookupError:
                    pass

                except Exception as exc:
                    self.audio_output_uncertain = True

                    self.get_logger().error(
                        "[PC AUDIO FALLBACK] "
                        "Cancellation signal failed | "
                        f"decision_id={decision_id} | "
                        f"error={exc}"
                    )

        self.get_logger().warning(
            "[PC AUDIO FALLBACK] "
            f"CANCEL_REQUESTED={decision_id}"
        )
        
    def _request_callback(
        self,
        msg,
    ):
        """
        Accept one explicitly authorized remote playback request.

        Duplicate terminal requests replay their terminal status rather than
        replaying the audio.
        """

        decision_id = str(
            msg.decision_id or ""
        ).strip()

        if not decision_id:
            self._publish_status(
                decision_id="",
                state=SPEECH_FAILED,
                detail=(
                    "PC fallback request has no decision_id."
                ),
            )
            return

        if decision_id in self.terminal_results:
            state, detail = (
                self.terminal_results[
                    decision_id
                ]
            )

            self.get_logger().info(
                "[PC AUDIO FALLBACK] "
                "Duplicate terminal request; "
                "replaying status only | "
                f"decision_id={decision_id}"
            )

            self._publish_status(
                decision_id=decision_id,
                state=state,
                detail=detail,
            )
            return

        if decision_id in self.pending_ids:
            self.get_logger().info(
                "[PC AUDIO FALLBACK] "
                "Duplicate pending request ignored | "
                f"decision_id={decision_id}"
            )
            return

        if not self.enabled:
            self._store_and_publish_terminal(
                decision_id=decision_id,
                success=False,
                detail=(
                    "PC audio fallback executor is disabled."
                ),
            )
            return

        if not self.audio_ready:
            self._store_and_publish_terminal(
                decision_id=decision_id,
                success=False,
                detail=(
                    "PC audio output is unavailable."
                ),
            )
            return

        speech_kind = str(
            msg.speech_kind or ""
        ).strip().upper()

        job = {
            "decision_id": decision_id,
            "speech_kind": speech_kind,
            "text": str(
                msg.text or ""
            ).strip(),
            "language": str(
                msg.language or ""
            ).strip(),
            "fallback_text": str(
                getattr(
                    msg,
                    "fallback_text",
                    "",
                )
                or ""
            ).strip(),
            "fallback_language": str(
                getattr(
                    msg,
                    "fallback_language",
                    "",
                )
                or ""
            ).strip(),
        }

        try:
            self.work_queue.put_nowait(
                job
            )

        except queue.Full:
            self._store_and_publish_terminal(
                decision_id=decision_id,
                success=False,
                detail=(
                    "PC audio fallback queue is full."
                ),
            )
            return

        self.pending_ids.add(
            decision_id
        )

        self.get_logger().info(
            "[PC AUDIO FALLBACK] "
            f"QUEUED={decision_id} | "
            f"KIND={speech_kind or 'UNSPECIFIED'} | "
            f"WAITING={self.work_queue.qsize()}"
        )

    # =========================================================================
    # WORKER
    # =========================================================================

    def _worker_loop(self):
        """
        Execute fallback jobs sequentially while tracking
        ownership of the active audio transaction.
        """

        while not self.stop_event.is_set():

            try:
                job = self.work_queue.get(timeout=0.2)

            except queue.Empty:
                continue

            decision_id = job["decision_id"]

            try:
                with self.playback_lock:
                    self.active_job = job

                    canceled = (
                        decision_id in self.cancel_requested_ids
                    )

                    blocked = (
                        self.audio_output_uncertain
                        or self.stop_event.is_set()
                    )

                if canceled:
                    self._store_and_publish_terminal(
                        decision_id=decision_id,
                        success=False,
                        state=SPEECH_CANCELED,
                        detail=(
                            "PC fallback canceled before execution."
                        ),
                    )
                    continue

                if blocked:
                    self._store_and_publish_terminal(
                        decision_id=decision_id,
                        success=False,
                        detail=(
                            "PC fallback execution inhibited: "
                            "audio output state uncertain or stopping."
                        ),
                    )
                    continue

                self._execute_job(job)

            except Exception as exc:
                self._store_and_publish_terminal(
                    decision_id=decision_id,
                    success=False,
                    detail=(
                        "Unexpected PC audio fallback error: "
                        f"{exc}"
                    ),
                )

            finally:
                with self.playback_lock:
                    if self.active_job is job:
                        self.active_job = None

                self.work_queue.task_done()

    def _execute_job(self, job):
        """
        Execute one authorized PC fallback transaction.

        Cancellation policy:
        - Before playback: no audio starts, return CANCELED.
        - During playback: wait for aplay to end.
        - If stop cannot be verified: return FAILED.
        - Never publish COMPLETED after cancellation.
        """

        decision_id = job["decision_id"]
        speech_kind = job["speech_kind"]

        def canceled():
            with self.playback_lock:
                return (
                    decision_id in self.cancel_requested_ids
                    or self.stop_event.is_set()
                )

        def finish_canceled(detail):
            self._store_and_publish_terminal(
                decision_id=decision_id,
                success=False,
                state=SPEECH_CANCELED,
                detail=detail,
            )

        # An earlier cancellation may have arrived via DDS.
        if canceled():
            finish_canceled(
                "PC fallback canceled before synthesis."
            )
            return

        # ---------------------------------------------------------
        # 1. Generate WAV
        # ---------------------------------------------------------

        if speech_kind == SPEECH_KIND_TURN_CUE:
            wav_path = self._generate_turn_cue_wav()

        else:
            text, model, language = (
                self._resolve_speech_request(job)
            )

            if not text or model is None:
                self._store_and_publish_terminal(
                    decision_id=decision_id,
                    success=False,
                    detail="No usable PC Piper voice/text.",
                )
                return

            wav_path = self._synthesize_wav(
                text=text,
                model=model,
            )

        if wav_path is None:
            if canceled():
                finish_canceled(
                    "PC fallback canceled during synthesis."
                )
            else:
                self._store_and_publish_terminal(
                    decision_id=decision_id,
                    success=False,
                    detail="PC fallback audio generation failed.",
                )
            return

        # ---------------------------------------------------------
        # 2. Playback
        # ---------------------------------------------------------

        try:
            if canceled():
                finish_canceled(
                    "PC fallback canceled before playback."
                )
                return

            with self.playback_lock:
                blocked = self.audio_output_uncertain

            if blocked:
                self._store_and_publish_terminal(
                    decision_id=decision_id,
                    success=False,
                    detail="PC audio output state uncertain.",
                )
                return

            # A cancellation can race with STARTED.
            # The actual Popen guard remains in _play_with_aplay.
            if canceled():
                finish_canceled(
                    "PC fallback canceled before playback."
                )
                return

            self._publish_status(
                decision_id=decision_id,
                state=SPEECH_STARTED,
                detail="PC fallback playback starting.",
            )

            success = self._play_with_aplay(
                wav_path,
                decision_id=decision_id,
            )

            with self.playback_lock:
                was_canceled = (
                    decision_id in self.cancel_requested_ids
                    or self.stop_event.is_set()
                )
                uncertain = self.audio_output_uncertain
                process_active = (
                    self.aplay_process is not None
                )

            if was_canceled:
                if uncertain or process_active:
                    self._store_and_publish_terminal(
                        decision_id=decision_id,
                        success=False,
                        detail=(
                            "PC cancellation requested, "
                            "but audio stop is unconfirmed."
                        ),
                    )
                else:
                    finish_canceled(
                        "PC fallback playback stopped "
                        "after cancellation."
                    )
                return

            if uncertain or process_active:
                self._store_and_publish_terminal(
                    decision_id=decision_id,
                    success=False,
                    detail="PC playback state unconfirmed.",
                )
                return

            self._store_and_publish_terminal(
                decision_id=decision_id,
                success=bool(success),
                detail=(
                    "Speech delivered through PC fallback."
                    if success
                    else "PC fallback playback failed."
                ),
            )

        finally:
            self._remove_wav(wav_path)

    # =========================================================================
    # SPEECH RESOLUTION
    # =========================================================================

    def _resolve_speech_request(
        self,
        job,
    ):
        """
        Resolve text + language + Piper model.

        Preferred-language voice is tried first.

        If unavailable, the explicitly supplied English fallback may be used.

        The node never silently speaks foreign-language text with an English
        voice.
        """

        preferred_text = str(
            job.get(
                "text",
                "",
            )
            or ""
        ).strip()

        if not preferred_text:
            return "", None, ""

        preferred_language = (
            normalize_dialogue_language(
                job.get(
                    "language",
                    "",
                )
            )
        )

        preferred_model = (
            self._resolve_model(
                preferred_language
            )
        )

        if preferred_model is not None:
            return (
                preferred_text,
                preferred_model,
                preferred_language,
            )

        fallback_text = str(
            job.get(
                "fallback_text",
                "",
            )
            or ""
        ).strip()

        fallback_language = str(
            job.get(
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

        if (
            fallback_text
            and fallback_language == "en"
        ):
            english_model = (
                self._resolve_model(
                    "en"
                )
            )

            if english_model is not None:
                self.get_logger().warning(
                    "[PC AUDIO FALLBACK] "
                    "Using explicit English fallback."
                )

                return (
                    fallback_text,
                    english_model,
                    "en",
                )

        return "", None, ""

    # =========================================================================
    # PIPER
    # =========================================================================

    def _find_piper_executable(
        self,
    ) -> Optional[Path]:
        """
        Locate a standalone Piper executable.
        """

        candidates = []

        if self.piper_executable_requested:
            candidates.append(
                Path(
                    self.piper_executable_requested
                ).expanduser()
            )

        env_value = os.environ.get(
            "G1_PIPER_EXECUTABLE",
            "",
        ).strip()

        if env_value:
            candidates.append(
                Path(
                    env_value
                ).expanduser()
            )

        system_piper = (
            shutil.which(
                "piper"
            )
        )

        if system_piper:
            candidates.append(
                Path(
                    system_piper
                )
            )

        candidates.append(
            self.piper_model_directory
            / "piper"
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
    ):
        """
        Resolve local Piper execution mode.
        """

        requested = (
            self.piper_backend_requested
        )

        if (
            requested
            not in SUPPORTED_PIPER_BACKENDS
        ):
            self.get_logger().warning(
                "[PC AUDIO FALLBACK] "
                f"Unsupported piper_backend={requested!r}; "
                "using auto."
            )

            requested = (
                PIPER_BACKEND_AUTO
            )

        if (
            requested
            == PIPER_BACKEND_MODULE
        ):
            return (
                PIPER_BACKEND_MODULE
                if self.piper_module_available
                else None
            )

        if (
            requested
            == PIPER_BACKEND_EXECUTABLE
        ):
            return (
                PIPER_BACKEND_EXECUTABLE
                if self.piper_executable
                is not None
                else None
            )

        if self.piper_module_available:
            return PIPER_BACKEND_MODULE

        if self.piper_executable is not None:
            return (
                PIPER_BACKEND_EXECUTABLE
            )

        return None

    def _build_piper_command(
        self,
        model,
        output_file,
    ):
        """
        Build the Piper synthesis command.
        """

        common = [
            "--model",
            str(model),
            "--output_file",
            str(output_file),
            "--length_scale",
            str(
                self.piper_length_scale
            ),
            "--sentence_silence",
            str(
                self.sentence_silence_sec
            ),
        ]

        if (
            self.piper_backend
            == PIPER_BACKEND_MODULE
        ):
            return [
                sys.executable,
                "-m",
                "piper",
                *common,
            ]

        if (
            self.piper_backend
            == PIPER_BACKEND_EXECUTABLE
            and self.piper_executable
            is not None
        ):
            return [
                str(
                    self.piper_executable
                ),
                *common,
            ]

        return None

    def _synthesize_wav(
        self,
        text,
        model,
    ) -> Optional[Path]:
        """
        Synthesize one speech WAV locally on the PC.
        """

        directory = Path(
            tempfile.mkdtemp(
                prefix="g1_pc_fallback_"
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
                    output_file=wav_path,
                )
            )

            if command is None:
                return None

            result = subprocess.run(
                command,
                input=(
                    text.rstrip()
                    + "\n"
                ),
                text=True,
                capture_output=True,
                check=False,
                timeout=(
                    self.synthesis_timeout_sec
                ),
            )

            if result.returncode != 0:
                self.get_logger().error(
                    "[PC AUDIO FALLBACK] "
                    "Piper synthesis failed: "
                    f"{result.stderr.strip()}"
                )

                return None

            if not wav_path.is_file():
                return None

            if wav_path.stat().st_size <= 44:
                return None

            keep_directory = True

            return wav_path

        except subprocess.TimeoutExpired:
            self.get_logger().error(
                "[PC AUDIO FALLBACK] "
                "Piper synthesis timed out."
            )

            return None

        except Exception as exc:
            self.get_logger().error(
                "[PC AUDIO FALLBACK] "
                f"Piper synthesis error: {exc}"
            )

            return None

        finally:
            if not keep_directory:
                shutil.rmtree(
                    directory,
                    ignore_errors=True,
                )

    # =========================================================================
    # TURN CUE
    # =========================================================================

    def _generate_turn_cue_wav(
        self,
    ) -> Optional[Path]:
        """
        Generate the short non-linguistic 'your turn' cue.

        880 Hz
        120 ms
        16 kHz
        mono
        signed 16-bit
        """

        sample_rate = 16000
        frequency_hz = 880.0
        duration_sec = 0.12
        amplitude = 0.22

        sample_count = int(
            sample_rate
            * duration_sec
        )

        directory = Path(
            tempfile.mkdtemp(
                prefix="g1_pc_turn_cue_"
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
                    sample_rate
                    * 0.015
                ),
            )

            for index in range(
                sample_count
            ):
                t = (
                    index
                    / float(sample_rate)
                )

                envelope = 1.0

                if index < fade_samples:
                    envelope = (
                        index
                        / float(
                            fade_samples
                        )
                    )

                remaining = (
                    sample_count
                    - index
                    - 1
                )

                if remaining < fade_samples:
                    envelope = min(
                        envelope,
                        remaining
                        / float(
                            fade_samples
                        ),
                    )

                value = (
                    math.sin(
                        2.0
                        * math.pi
                        * frequency_hz
                        * t
                    )
                    * amplitude
                    * envelope
                )

                sample = int(
                    value
                    * 32767.0
                )

                frames.extend(
                    sample.to_bytes(
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

            return wav_path

        except Exception as exc:
            self.get_logger().error(
                "[PC AUDIO FALLBACK] "
                "Turn cue generation failed: "
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
    # APLAY
    # =========================================================================

    def _play_with_aplay(self, wav_path, decision_id=None):
        """
        Play one WAV through a supervised aplay subprocess.

        Guarantees:
        - Playback is associated with one decision_id.
        - Cancellation before Popen prevents playback.
        - A launched process is tracked until reaped.
        - Cancellation never produces a successful playback result.
        - Timeout or uncertain stop inhibits future playback.

        Returns:
            bool: True only for confirmed successful playback.
        """

        if self.audio_player_path is None:
            return False

        if decision_id is None:
            self.get_logger().error(
                "[PC AUDIO FALLBACK] Missing playback decision_id."
            )
            return False

        decision_id = str(decision_id).strip()

        if not decision_id:
            return False

        try:
            with wave.open(str(wav_path), "rb") as audio:
                frame_rate = audio.getframerate()

                if frame_rate <= 0:
                    return False

                duration_sec = (
                    audio.getnframes() / float(frame_rate)
                )

        except Exception as exc:
            self.get_logger().error(
                "[PC AUDIO FALLBACK] Invalid WAV: "
                f"{exc}"
            )
            return False

        process = None

        try:
            # -----------------------------------------------------
            # Atomic cancellation check and process creation
            # -----------------------------------------------------

            with self.playback_lock:

                if self.stop_event.is_set():
                    return False

                if self.audio_output_uncertain:
                    return False

                if decision_id in self.cancel_requested_ids:
                    return False

                if (
                    self.active_job is None
                    or self.active_job.get("decision_id") != decision_id
                ):
                    self.get_logger().error(
                        "[PC AUDIO FALLBACK] Playback transaction "
                        f"mismatch: {decision_id}"
                    )
                    return False

                if self.aplay_process is not None:
                    self.audio_output_uncertain = True
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

            # -----------------------------------------------------
            # Wait for playback
            # -----------------------------------------------------

            try:
                _, stderr = process.communicate(
                    timeout=max(5.0, duration_sec + 3.0)
                )

            except subprocess.TimeoutExpired:

                self.audio_output_uncertain = True

                try:
                    process.kill()
                    process.communicate(timeout=2.0)

                except Exception as exc:
                    self.get_logger().error(
                        "[PC AUDIO FALLBACK] Timeout cleanup "
                        f"failed: {exc}"
                    )

                return False

            # -----------------------------------------------------
            # Evaluate the actual process result
            # -----------------------------------------------------

            with self.playback_lock:

                canceled = (
                    decision_id in self.cancel_requested_ids
                    or self.stop_event.is_set()
                )

            if canceled:
                return False

            if process.returncode != 0:
                self.get_logger().warning(
                    "[PC AUDIO FALLBACK] aplay failed | "
                    f"decision_id={decision_id} | "
                    f"returncode={process.returncode} | "
                    f"stderr={(stderr or '').strip()}"
                )
                return False

            return True

        except Exception as exc:

            if process is not None:
                self.audio_output_uncertain = True

            self.get_logger().error(
                "[PC AUDIO FALLBACK] Playback exception | "
                f"decision_id={decision_id} | "
                f"{type(exc).__name__}: {exc}"
            )

            return False

        finally:

            if process is not None:

                with self.playback_lock:

                    if self.aplay_process is process:
                        self.aplay_process = None

    # =========================================================================
    # MODELS
    # =========================================================================

    def _resolve_model(
        self,
        language,
    ):
        """
        Resolve one locally available Piper model.
        """

        normalized = (
            normalize_dialogue_language(
                language
            )
        )

        if normalized == "en":
            if self.english_model_ready:
                return self.english_model

            return None

        if normalized == "es":
            if self.spanish_model_ready:
                return self.spanish_model

            return None

        return None

    def _validate_model(
        self,
        model,
        label,
    ):
        """
        Validate Piper model and JSON configuration.
        """

        if not model.is_file():
            self.get_logger().warning(
                "[PC AUDIO FALLBACK] "
                f"{label} model unavailable: {model}"
            )

            return False

        configuration = Path(
            str(model)
            + ".json"
        )

        if not configuration.is_file():
            self.get_logger().warning(
                "[PC AUDIO FALLBACK] "
                f"{label} model configuration unavailable: "
                f"{configuration}"
            )

            return False

        return True

    # =========================================================================
    # STATUS / IDEMPOTENCY
    # =========================================================================

    def _publish_status(
        self,
        *,
        decision_id,
        state,
        detail,
    ):
        """
        Publish one fallback-executor lifecycle event.
        """

        msg = SpeechStatus()

        msg.header.stamp = (
            self.get_clock()
            .now()
            .to_msg()
        )

        msg.header.frame_id = ""

        msg.decision_id = (
            decision_id
        )

        msg.state = state
        msg.detail = detail

        self.status_pub.publish(
            msg
        )

        self.get_logger().info(
            "[PC AUDIO FALLBACK STATUS] "
            f"DECISION={decision_id or 'NONE'} | "
            f"STATE={state} | "
            f"DETAIL={detail}"
        )

    def _store_and_publish_terminal(
        self,
        *,
        decision_id,
        success,
        detail,
        state=None,
    ):
        """
        Publish at most one terminal result per decision_id.

        CANCELED is permitted only when the caller has
        verified that playback never started or has ended.
        """

        decision_id = str(decision_id or "").strip()

        if not decision_id:
            return

        with self.playback_lock:

            if decision_id in self.terminal_results:
                return

            if state is None:
                state = (
                    SPEECH_COMPLETED
                    if success
                    else SPEECH_FAILED
                )

            if state not in {
                SPEECH_COMPLETED,
                SPEECH_FAILED,
                SPEECH_CANCELED,
            }:
                raise ValueError(
                    f"Invalid terminal speech state: {state}"
                )

            # Commit terminal result atomically against
            # the cancellation callback.
            cancellation_pending = (
                decision_id in self.cancel_requested_ids
                or self.stop_event.is_set()
            )

            if (
                cancellation_pending
                and state == SPEECH_COMPLETED
            ):
                state = SPEECH_FAILED
                detail = (
                    "Playback completion raced with "
                    "cancellation; final audio state "
                    "not confirmed."
                )
                self.audio_output_uncertain = True

            if (
                state == SPEECH_CANCELED
                and self.audio_output_uncertain
            ):
                state = SPEECH_FAILED
                detail = (
                    "Cancellation requested but "
                    "audio stop is unconfirmed."
                )
                
            self.pending_ids.discard(decision_id)

            self.terminal_results[decision_id] = (
                state,
                detail,
            )

            while len(self.terminal_results) > 1000:
                self.terminal_results.popitem(last=False)

        self._publish_status(
            decision_id=decision_id,
            state=state,
            detail=detail,
        )
    
    # =========================================================================
    # CLEANUP
    # =========================================================================

    @staticmethod
    def _remove_wav(
        wav_path,
    ):
        """
        Remove one temporary generated WAV directory.
        """

        try:
            shutil.rmtree(
                wav_path.parent,
                ignore_errors=True,
            )

        except Exception:
            pass

    def stop(
        self,
    ):
        """
        Stop the worker without accepting new execution work.
        """

        self.stop_event.set()

        if self.worker.is_alive():
            self.worker.join(
                timeout=2.0
            )


# =============================================================================
# MAIN
# =============================================================================


def main(
    args=None,
):
    rclpy.init(
        args=args
    )

    node = (
        PCAudioFallbackNode()
    )

    try:
        rclpy.spin(
            node
        )

    except KeyboardInterrupt:
        pass

    finally:
        node.stop()
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()