#!/usr/bin/env python3

import tempfile
import threading
import wave
import json

from std_msgs.msg import String
import numpy as np
import rclpy
import sounddevice as sd
from faster_whisper import WhisperModel
from collections import deque
from hri_msgs.msg import IdsList, LiveSpeech
from rclpy.node import Node
from std_msgs.msg import Bool


class PCASRNode(Node):
    """PC microphone -> Faster-Whisper -> ROS4HRI LiveSpeech."""

    LOCALES = {
        "fr": "fr_FR",
        "en": "en_US",
        "es": "es_ES",
    }

    def __init__(self):
        super().__init__("g1_pc_asr")

        # ------------------------------------------------------------------
        # Parameters
        # ------------------------------------------------------------------

        self.declare_parameter("voice_id", "pc_user")
        self.declare_parameter("language", "fr")
        self.declare_parameter("model", "small")
        self.declare_parameter("device", "cpu")
        self.declare_parameter("compute_type", "int8")

        self.declare_parameter("sample_rate", 16000)
        self.declare_parameter("record_seconds", 6.0)
        self.declare_parameter("chunk_ms", 30)
        self.declare_parameter("speech_start_rms", 180.0)
        self.declare_parameter("speech_end_rms", 100.0)
        self.declare_parameter("end_silence_sec", 0.55)
        self.declare_parameter("min_speech_sec", 0.20)
        self.declare_parameter("max_speech_sec", 8.0)
        self.declare_parameter("pre_roll_sec", 0.20)
        self.declare_parameter("audio_device", -1)

        self.declare_parameter("vad_filter", True)
        self.declare_parameter("min_rms", 25.0)

        
        # ------------------------------------------------------------------
        # Resolve 
        # ------------------------------------------------------------------
        self.chunk_ms = int(
            self.get_parameter("chunk_ms").value
        )

        self.speech_start_rms = float(
            self.get_parameter("speech_start_rms").value
        )

        self.speech_end_rms = float(
            self.get_parameter("speech_end_rms").value
        )

        self.end_silence_sec = float(
            self.get_parameter("end_silence_sec").value
        )

        self.min_speech_sec = float(
            self.get_parameter("min_speech_sec").value
        )

        self.max_speech_sec = float(
            self.get_parameter("max_speech_sec").value
        )

        self.pre_roll_sec = float(
            self.get_parameter("pre_roll_sec").value
        )

        self.voice_id = str(
            self.get_parameter("voice_id").value
        ).strip()

        self.language = str(
            self.get_parameter("language").value
        ).strip().lower()

        self.sample_rate = int(
            self.get_parameter("sample_rate").value
        )

        self.record_seconds = float(
            self.get_parameter("record_seconds").value
        )

        audio_device = int(
            self.get_parameter("audio_device").value
        )
        self.audio_device = None if audio_device < 0 else audio_device

        self.vad_filter = bool(
            self.get_parameter("vad_filter").value
        )

        self.min_rms = float(
            self.get_parameter("min_rms").value
        )

        if not self.voice_id:
            raise RuntimeError("voice_id must not be empty")

        if self.language not in {"fr", "en", "es", "auto"}:
            raise RuntimeError(
                "language must be one of: fr, en, es, auto"
            )

        # Each capture belongs to one exact HRI dialogue turn.
        self.audio_lock = threading.RLock()
        self.audio_turn = None
        self.audio_phase = "IDLE"
        self.audio_sequence = 0
        self.audio_queue = deque(maxlen=128)

        self.audio_command_sub = self.create_subscription(
            String,
            "/hri/asr_turn_command",
            self._audio_command_callback,
            10,
        )

        self.audio_status_pub = self.create_publisher(
            String,
            "/hri/asr_turn_status",
            20,
        )
        
        # ------------------------------------------------------------------
        # ROS4HRI publishers
        # ------------------------------------------------------------------

        self.voices_pub = self.create_publisher(
            IdsList,
            "/humans/voices/tracked",
            10,
        )

        # Publish periodically because consumers may start after this node.
        self.create_timer(1.0, self._publish_tracked_voice)

        # ------------------------------------------------------------------
        # Whisper
        # ------------------------------------------------------------------

        model_name = str(
            self.get_parameter("model").value
        )

        device = str(
            self.get_parameter("device").value
        )

        compute_type = str(
            self.get_parameter("compute_type").value
        )

        self.get_logger().info(
            f"Loading Faster-Whisper model '{model_name}' "
            f"on {device} ({compute_type})..."
        )

        self.model = WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
        )

        self.get_logger().info("Faster-Whisper model ready.")

        # ------------------------------------------------------------------
        # Worker
        # ------------------------------------------------------------------

        self.stop_event = threading.Event()

        self.worker = threading.Thread(
            target=self._recognition_loop,
            daemon=True,
        )
        self.worker.start()

        self.get_logger().info(
            "PC ASR ready: "
            f"voice_id={self.voice_id}, "
            f"language={self.language}, "
            f"topic=/humans/voices/{self.voice_id}/speech"
        )

    # ----------------------------------------------------------------------
    # ROS4HRI
    # ----------------------------------------------------------------------

    def _publish_tracked_voice(self):
        msg = IdsList()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.ids = [self.voice_id]
        self.voices_pub.publish(msg)

    # ----------------------------------------------------------------------
    # Audio + ASR
    # ----------------------------------------------------------------------

    def _audio_status_locked(self, state, **extra):
        """Publish ordered progress while holding audio_lock."""
        if self.audio_turn is None:
            return

        self.audio_sequence += 1

        payload = dict(self.audio_turn)
        payload.update(
            version=1,
            state=state,
            sequence=self.audio_sequence,
        )
        payload.update(extra)

        self.audio_status_pub.publish(
            String(data=json.dumps(payload))
        )

    def _audio_command_callback(self, msg):
        """Prepare, open or cancel one exactly identified audio turn."""
        try:
            data = json.loads(msg.data)

            if (
                not isinstance(data, dict)
                or data.get("version") != 1
            ):
                return

            keys = (
                "turn_id",
                "interaction_id",
                "request_id",
            )

            if any(
                not isinstance(data.get(key), str)
                or not data[key]
                for key in keys
            ):
                return

            turn = {
                key: data[key]
                for key in keys
            }

            command = data.get("command")

        except (ValueError, TypeError, AttributeError):
            return

        with self.audio_lock:
            if command == "PREPARE":
                if self.audio_turn == turn:
                    return

                self.audio_turn = turn
                self.audio_phase = "PREPARING"
                self.audio_sequence = 0
                self.audio_queue.clear()

            elif (
                self.audio_turn == turn
                and command == "LISTEN"
            ):
                if self.audio_phase not in {
                    "READY",
                    "RESULT",
                }:
                    return

                self.audio_queue.clear()
                self.audio_phase = "OPENING"

            elif (
                self.audio_turn == turn
                and command == "CANCEL"
            ):
                self.audio_turn = None
                self.audio_phase = "IDLE"
                self.audio_queue.clear()

    def _audio_callback(self, data, frames, timing, status):
        """Discard robot audio and retain only authorized human audio."""
        with self.audio_lock:
            if self.audio_turn is None:
                return

            if status:
                self.audio_phase = "FAILED"
                self.audio_queue.clear()

                self._audio_status_locked(
                    "FAILED",
                    reason="INPUT_STREAM_STATUS",
                )
                return

            if self.audio_phase == "PREPARING":
                # A real device callback confirms that capture is running.
                # This block is discarded.
                self.audio_phase = "READY"

                self._audio_status_locked("READY")
                return

            if self.audio_phase == "OPENING":
                self.audio_phase = "LISTENING"

                self._audio_status_locked("LISTENING")

                # Discard the block spanning the LISTEN transition.
                return

            if self.audio_phase not in {
                "LISTENING",
                "RECORDING",
            }:
                # Robot speech, cue audio and transcription-time audio
                # must never enter the next utterance.
                return

            if len(self.audio_queue) == self.audio_queue.maxlen:
                self.audio_phase = "FAILED"
                self.audio_queue.clear()

                self._audio_status_locked(
                    "FAILED",
                    reason="AUDIO_QUEUE_OVERFLOW",
                )
                return

            self.audio_queue.append((
                dict(self.audio_turn),
                data.reshape(-1).copy(),
            ))

    def _record_utterance(self):
        """Capture one bounded utterance from an already running stream."""
        chunk_sec = self.chunk_ms / 1000.0

        pre_roll = deque(
            maxlen=max(
                1,
                int(self.pre_roll_sec / chunk_sec),
            )
        )

        utterance = []
        owner = None
        recording = False
        silence_chunks = 0
        voiced_frames = 0
        total_frames = 0

        while (
            rclpy.ok()
            and not self.stop_event.is_set()
        ):
            with self.audio_lock:
                if (
                    owner is not None
                    and self.audio_turn != owner
                ):
                    return None

                if self.audio_phase not in {
                    "LISTENING",
                    "RECORDING",
                    "OPENING",
                }:
                    if owner is not None:
                        return None

                item = (
                    self.audio_queue.popleft()
                    if self.audio_queue
                    else None
                )

            if item is None:
                self.stop_event.wait(0.01)
                continue

            turn, chunk = item

            if owner is None:
                owner = turn

            if turn != owner:
                return None

            rms = self._rms(chunk)

            if not recording:
                pre_roll.append(chunk)

                if rms < self.speech_start_rms:
                    continue

                recording = True
                utterance.extend(pre_roll)
                pre_roll.clear()

                total_frames = sum(
                    block.size
                    for block in utterance
                )

                voiced_frames = chunk.size

                with self.audio_lock:
                    if (
                        self.audio_turn != owner
                        or self.audio_phase != "LISTENING"
                    ):
                        return None

                    self.audio_phase = "RECORDING"
                    self._audio_status_locked("SPEECH_STARTED")

            else:
                utterance.append(chunk)
                total_frames += chunk.size

                if rms >= self.speech_end_rms:
                    silence_chunks = 0
                    voiced_frames += chunk.size
                else:
                    silence_chunks += 1

            if total_frames >= int(
                self.max_speech_sec * self.sample_rate
            ):
                with self.audio_lock:
                    if self.audio_turn == owner:
                        self.audio_phase = "RESULT"
                        self.audio_queue.clear()

                        # Do not interpret an arbitrarily truncated sentence.
                        self._audio_status_locked(
                            "EMPTY",
                            reason="UTTERANCE_TOO_LONG",
                        )

                return None

            if (
                recording
                and silence_chunks * chunk_sec
                    >= self.end_silence_sec
            ):
                with self.audio_lock:
                    if (
                        self.audio_turn != owner
                        or self.audio_phase != "RECORDING"
                    ):
                        return None

                    self.audio_queue.clear()

                    if voiced_frames < int(
                        self.min_speech_sec * self.sample_rate
                    ):
                        self.audio_phase = "RESULT"

                        self._audio_status_locked(
                            "EMPTY",
                            reason="SPEECH_TOO_SHORT",
                        )
                        return None

                    self.audio_phase = "TRANSCRIBING"
                    self._audio_status_locked("TRANSCRIBING")

                return owner, np.concatenate(utterance)

        return None
    
    @staticmethod
    def _rms(audio):
        if audio.size == 0:
            return 0.0

        samples = audio.astype(np.float64)

        return float(
            np.sqrt(
                np.mean(samples * samples)
            )
        )

    def _transcribe(self, audio):
        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=True,
        ) as tmp:

            with wave.open(tmp.name, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.sample_rate)
                wf.writeframes(audio.tobytes())

            whisper_language = (
                None
                if self.language == "auto"
                else self.language
            )

            segments, info = self.model.transcribe(
                tmp.name,
                language=whisper_language,
                beam_size=5,
                vad_filter=self.vad_filter,
            )

            segments = list(segments)

            text = " ".join(
                segment.text.strip()
                for segment in segments
                if segment.text.strip()
            ).strip()

            detected_language = (
                info.language or whisper_language or "unknown"
            )

            valid_segments = [
                segment
                for segment in segments
                if segment.text.strip()
            ]

            if valid_segments:
                mean_no_speech = sum(
                    float(
                        getattr(segment, "no_speech_prob", 0.0)
                        or 0.0
                    )
                    for segment in valid_segments
                ) / len(valid_segments)

                confidence = max(
                    0.0,
                    min(1.0, 1.0 - mean_no_speech),
                )
            else:
                confidence = 0.0

            return (
                text,
                detected_language,
                confidence,
            )

    def _recognition_loop(self):
        """Keep capture warm and reject results from obsolete turns."""
        chunk_frames = max(
            1,
            int(
                self.sample_rate
                * self.chunk_ms
                / 1000.0
            ),
        )

        while (
            rclpy.ok()
            and not self.stop_event.is_set()
        ):
            try:
                with sd.InputStream(
                    samplerate=self.sample_rate,
                    channels=1,
                    dtype="int16",
                    device=self.audio_device,
                    blocksize=chunk_frames,
                    callback=self._audio_callback,
                ):
                    while (
                        rclpy.ok()
                        and not self.stop_event.is_set()
                    ):
                        captured = self._record_utterance()

                        if captured is None:
                            continue

                        owner, audio = captured

                        text, language, confidence = (
                            self._transcribe(audio)
                        )

                        with self.audio_lock:
                            if (
                                self.audio_turn != owner
                                or self.audio_phase != "TRANSCRIBING"
                            ):
                                self.get_logger().info(
                                    "Discarded stale ASR result"
                                )
                                continue

                            self.audio_phase = "RESULT"

                            # HRI validates ownership before forwarding
                            # the text to the existing ROS4HRI backend.
                            self._audio_status_locked(
                                "RESULT" if text.strip() else "EMPTY",
                                text=text.strip(),
                                language=language,
                                confidence=float(confidence),
                            )

            except Exception as exc:
                self.get_logger().error(
                    f"ASR loop error: {exc}"
                )

                with self.audio_lock:
                    self.audio_phase = "FAILED"
                    self.audio_queue.clear()

                    self._audio_status_locked(
                        "FAILED",
                        reason="ASR_WORKER_EXCEPTION",
                    )

                self.stop_event.wait(1.0)
    def destroy_node(self):
        self.stop_event.set()

        if (
            hasattr(self, "worker")
            and self.worker.is_alive()
        ):
            self.worker.join(timeout=2.0)

        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    node = PCASRNode()

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
