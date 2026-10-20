#!/usr/bin/env python3

import tempfile
import threading
import wave

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

        self.listening_enabled = threading.Event()

        self.asr_gate_sub = self.create_subscription(
            Bool,
            "/hri/asr_listening_enabled",
            self._listening_enabled_callback,
            10,
        )

        # Safe default:
        # do not listen until HRI explicitly opens a human-response window.
        self.listening_enabled.clear()

        # ------------------------------------------------------------------
        # ROS4HRI publishers
        # ------------------------------------------------------------------

        self.voices_pub = self.create_publisher(
            IdsList,
            "/humans/voices/tracked",
            10,
        )

        self.speech_pub = self.create_publisher(
            LiveSpeech,
            f"/humans/voices/{self.voice_id}/speech",
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

    def _publish_final_speech(
        self,
        text: str,
        language: str,
        confidence: float,
    ):
        text = text.strip()

        if not text:
            return

        msg = LiveSpeech()

        msg.header.stamp = self.get_clock().now().to_msg()

        msg.incremental = ""
        msg.final = text

        msg.confidence = max(
            0.0,
            min(float(confidence), 1.0),
        )

        msg.locale = self.LOCALES.get(
            language,
            language,
        )

        # Legacy field still present in the installed hri_msgs.
        msg.language = language

        self.speech_pub.publish(msg)

        self.get_logger().info(
            f'RECOGNIZED [{msg.locale}] '
            f'confidence={msg.confidence:.2f}: "{text}"'
        )

    # ----------------------------------------------------------------------
    # Audio + ASR
    # ----------------------------------------------------------------------

    def _listening_enabled_callback(
        self,
        msg: Bool,
    ):
        """
        Enable or disable microphone capture according to the HRI turn-taking
        authority.

        False:
            - immediately invalidate any active capture;
            - no microphone pre-roll is accumulated;
            - no recognition result is allowed to be published.

        True:
            - the next _record_utterance() call opens a fresh audio stream;
            - therefore audio emitted by the robot before this transition cannot
            leak into the next human utterance.
        """

        enabled = bool(
            msg.data
        )

        previous = (
            self.listening_enabled.is_set()
        )

        if enabled:
            self.listening_enabled.set()
        else:
            self.listening_enabled.clear()

        if enabled != previous:
            self.get_logger().info(
                "ASR listening gate changed | "
                f"enabled={enabled}"
            )

    def _record_utterance(self):
        """
        Capture exactly one human utterance while the HRI listening gate is open.

        Important:
            The microphone stream is opened only after HRI explicitly authorizes
            human speech.

            If HRI closes the gate while recording, the current utterance is
            discarded immediately.

            This prevents robot speech, anti-echo guard audio and the turn cue
            from leaking into the next Whisper transcription.
        """

        # ------------------------------------------------------------------
        # Wait until HRI explicitly allows human speech.
        # ------------------------------------------------------------------

        while (
            rclpy.ok()
            and not self.stop_event.is_set()
            and not self.listening_enabled.is_set()
        ):
            self.stop_event.wait(
                0.05
            )

        if (
            not rclpy.ok()
            or self.stop_event.is_set()
        ):
            return None

        chunk_frames = max(
            1,
            int(
                self.sample_rate
                * self.chunk_ms
                / 1000.0
            ),
        )

        pre_roll_chunks = max(
            1,
            int(
                self.pre_roll_sec
                / (
                    self.chunk_ms
                    / 1000.0
                )
            ),
        )

        end_silence_chunks = max(
            1,
            int(
                self.end_silence_sec
                / (
                    self.chunk_ms
                    / 1000.0
                )
            ),
        )

        min_speech_frames = int(
            self.min_speech_sec
            * self.sample_rate
        )

        max_speech_frames = int(
            self.max_speech_sec
            * self.sample_rate
        )

        pre_roll = deque(
            maxlen=pre_roll_chunks
        )

        utterance = []
        recording = False
        silence_chunks = 0
        voiced_frames = 0

        self.get_logger().debug(
            "Listening for human speech..."
        )

        # ------------------------------------------------------------------
        # Important:
        # Open the device only AFTER the HRI gate becomes true.
        #
        # That gives us a fresh PortAudio stream instead of retaining samples
        # produced while the robot was speaking.
        # ------------------------------------------------------------------

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="int16",
            device=self.audio_device,
            blocksize=chunk_frames,
        ) as stream:

            while (
                rclpy.ok()
                and not self.stop_event.is_set()
            ):

                # ----------------------------------------------------------
                # HRI may close the response window at any time.
                #
                # Drop absolutely everything captured so far.
                # ----------------------------------------------------------

                if not self.listening_enabled.is_set():

                    if recording or utterance or pre_roll:
                        self.get_logger().debug(
                            "ASR capture discarded because "
                            "the HRI listening gate closed."
                        )

                    return None

                data, overflowed = stream.read(
                    chunk_frames
                )

                if overflowed:
                    self.get_logger().warning(
                        "Microphone input overflow."
                    )

                # Check once more after the blocking read.
                if not self.listening_enabled.is_set():
                    return None

                chunk = (
                    data
                    .reshape(-1)
                    .copy()
                )

                rms = self._rms(
                    chunk
                )

                # ----------------------------------------------------------
                # Waiting for beginning of human speech.
                # ----------------------------------------------------------

                if not recording:

                    pre_roll.append(
                        chunk
                    )

                    if (
                        rms
                        >= self.speech_start_rms
                    ):
                        recording = True
                        silence_chunks = 0

                        utterance.extend(
                            pre_roll
                        )

                        pre_roll.clear()

                        voiced_frames += (
                            chunk.size
                        )

                        self.get_logger().debug(
                            "Speech started: "
                            f"rms={rms:.1f}"
                        )

                    continue

                # ----------------------------------------------------------
                # Active human utterance.
                # ----------------------------------------------------------

                utterance.append(
                    chunk
                )

                if (
                    rms
                    >= self.speech_end_rms
                ):
                    silence_chunks = 0
                    voiced_frames += (
                        chunk.size
                    )
                else:
                    silence_chunks += 1

                total_frames = sum(
                    block.size
                    for block in utterance
                )

                if (
                    total_frames
                    >= max_speech_frames
                ):
                    self.get_logger().warning(
                        "Maximum utterance duration reached."
                    )
                    break

                if (
                    silence_chunks
                    >= end_silence_chunks
                ):
                    break

        # ------------------------------------------------------------------
        # The gate may have closed exactly while InputStream was leaving.
        # Never return stale robot-contaminated audio.
        # ------------------------------------------------------------------

        if not self.listening_enabled.is_set():
            return None

        if not utterance:
            return None

        if (
            voiced_frames
            < min_speech_frames
        ):
            self.get_logger().debug(
                "Discarding too-short speech candidate."
            )
            return None

        audio = np.concatenate(
            utterance
        )

        self.get_logger().debug(
            "Utterance captured: "
            f"{audio.size / self.sample_rate:.2f}s"
        )

        return audio

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
        """
        Continuously recognize human speech while respecting the HRI listening
        authority.

        A recognition result is published only if the listening gate remains open
        throughout capture and transcription.

        This prevents a stale transcription from being emitted after the robot has
        already reclaimed the speaking turn.
        """

        while (
            rclpy.ok()
            and not self.stop_event.is_set()
        ):

            try:

                audio = (
                    self._record_utterance()
                )

                if (
                    audio is None
                    or audio.size == 0
                ):
                    continue

                # The turn may have been closed immediately after capture.
                if not self.listening_enabled.is_set():
                    self.get_logger().debug(
                        "Discarding captured audio because "
                        "the HRI gate closed before transcription."
                    )
                    continue

                text, language, confidence = (
                    self._transcribe(
                        audio
                    )
                )

                if not text:
                    continue

                # ----------------------------------------------------------
                # Critical race barrier.
                #
                # Whisper may take hundreds of milliseconds. During that time
                # HRI may have accepted a previous answer and started the next
                # robot utterance.
                #
                # Never publish a result belonging to the old listening turn.
                # ----------------------------------------------------------

                if not self.listening_enabled.is_set():
                    self.get_logger().debug(
                        "Discarding recognition result because "
                        "the HRI gate closed during transcription | "
                        f"text={text!r}"
                    )
                    continue

                self._publish_final_speech(
                    text,
                    language,
                    confidence,
                )

            except Exception as exc:

                self.get_logger().error(
                    f"ASR loop error: {exc}"
                )

                self.stop_event.wait(
                    1.0
                )

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
