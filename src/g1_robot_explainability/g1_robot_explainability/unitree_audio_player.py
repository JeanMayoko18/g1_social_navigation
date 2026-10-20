#!/usr/bin/env python3

"""
Play raw PCM audio through the integrated Unitree G1 speaker.

Purpose
-------
This module runs as an isolated subprocess without importing or
initializing rclpy.

This separation prevents conflicts between:

    ROS 2 DDS initialization
        and
    Unitree ChannelFactory initialization

Audio flow
----------
Piper TTS ROS node
        ↓
Piper WAV
        ↓
PCM conversion
    - 16000 Hz
    - mono
    - signed 16-bit little-endian
        ↓
this isolated process
        ↓
Unitree AudioClient.PlayStream()
        ↓
integrated G1 speaker

Expected PCM format
-------------------
sample rate:
    16000 Hz

channels:
    1 channel

sample width:
    signed 16-bit little-endian

Important
---------
The PCM format is intentionally fixed because it is the format expected
by the Unitree G1 audio streaming service.
"""

import argparse
import sys
import time
import signal

from pathlib import Path
from typing import Any

from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
)

from unitree_sdk2py.g1.audio.g1_audio_client import (
    AudioClient,
)


# =============================================================================
# UNITREE PCM FORMAT
# =============================================================================

PCM_SAMPLE_RATE_HZ = 16000
PCM_CHANNEL_COUNT = 1
PCM_SAMPLE_WIDTH_BYTES = 2

PCM_BYTES_PER_SECOND = (
    PCM_SAMPLE_RATE_HZ
    * PCM_CHANNEL_COUNT
    * PCM_SAMPLE_WIDTH_BYTES
)


# =============================================================================
# DEFAULT SETTINGS
# =============================================================================

DEFAULT_VOLUME = 85
DEFAULT_STREAM_NAME = "g1_piper_tts"

# Additional time allowed for the Unitree speaker to finish playing
# its internal buffered audio before PlayStop() is called.
DEFAULT_PLAYBACK_MARGIN_SEC = 0.75

DEFAULT_CLIENT_TIMEOUT_SEC = 10.0

class PlaybackCanceled(Exception):
    """Requested cancellation of the isolated audio playback."""

class PreStreamFailure(Exception):
    """
    Failure before any PlayStream call in this subprocess.

    No audio stream was requested by this playback attempt.
    This is not a playback completion or a stop acknowledgment.
    """

UNITREE_CANCELED_EXIT_CODE = 3


def _handle_cancel_signal(signum, frame):
    raise PlaybackCanceled(
        f"Audio cancellation requested by signal {signum}"
    )

# =============================================================================
# UNITREE RESULT HANDLING
# =============================================================================


def result_code(
    result: Any,
) -> int:
    """
    Extract the return code from a Unitree SDK result.

    Depending on the SDK method, a result can be returned as:

        0

    or:

        (0, response_data)
    """

    if isinstance(result, tuple):
        if not result:
            return -1

        return int(
            result[0]
        )

    return int(
        result
    )


def require_success(
    result: Any,
    operation: str,
) -> None:
    """Raise an exception when a Unitree SDK operation fails."""

    code = result_code(
        result
    )

    if code != 0:
        raise RuntimeError(
            f"{operation} failed with code {code}: "
            f"{result}"
        )


# =============================================================================
# PCM VALIDATION
# =============================================================================


def validate_pcm_file(
    pcm_file: Path,
) -> bytes:
    """Load and validate one raw PCM file."""

    if not pcm_file.is_file():
        raise FileNotFoundError(
            f"PCM file not found: {pcm_file}"
        )

    pcm_data = pcm_file.read_bytes()

    if not pcm_data:
        raise RuntimeError(
            f"PCM file is empty: {pcm_file}"
        )

    # Every signed 16-bit sample contains exactly two bytes.
    if len(pcm_data) % PCM_SAMPLE_WIDTH_BYTES != 0:
        raise RuntimeError(
            "PCM data size must be divisible by "
            f"{PCM_SAMPLE_WIDTH_BYTES}, received "
            f"{len(pcm_data)} bytes."
        )

    return pcm_data


# =============================================================================
# UNITREE AUDIO PLAYBACK
# =============================================================================

def play_pcm(
    *,
    interface: str,
    pcm_file: Path,
    volume: int,
    stream_name: str,
    playback_margin_sec: float,
    client_timeout_sec: float,
) -> None:
    """
    Play one complete PCM utterance through Unitree AudioClient.

    SIGTERM requests cancellation without interrupting an SDK
    call halfway through.

    An interrupted stream is acknowledged as canceled only
    after PlayStop() reports success.

    Raises:
        InterruptedError: cancellation confirmed
        RuntimeError: playback or stop not confirmed
    """

    import signal
    import threading

    cancel_event = threading.Event()

    previous_sigterm_handler = signal.getsignal(
        signal.SIGTERM
    )

    def request_cancel(signum, frame):
        cancel_event.set()

    signal.signal(
        signal.SIGTERM,
        request_cancel,
    )

    client = None
    stream_may_exist = False
    stop_confirmed = False

    try:
        pcm_data = validate_pcm_file(
            pcm_file
        )

        audio_duration_sec = (
            len(pcm_data)
            / float(PCM_BYTES_PER_SECOND)
        )

        print(
            "UNITREE_PCM_INFO "
            f"bytes={len(pcm_data)} "
            f"audio_duration={audio_duration_sec:.3f}s "
            f"client_timeout={client_timeout_sec:.3f}s",
            flush=True,
        )

        if cancel_event.is_set():
            raise InterruptedError(
                "Canceled before SDK initialization."
            )

        # -----------------------------------------------------
        # SDK initialization
        # -----------------------------------------------------

        ChannelFactoryInitialize(
            0,
            interface,
        )

        if cancel_event.is_set():
            raise InterruptedError(
                "Canceled before AudioClient initialization."
            )

        client = AudioClient()

        client.SetTimeout(
            client_timeout_sec
        )

        client.Init()

        if cancel_event.is_set():
            raise InterruptedError(
                "Canceled before audio stream creation."
            )

        require_success(
            client.SetVolume(volume),
            "SetVolume",
        )

        if cancel_event.is_set():
            raise InterruptedError(
                "Canceled before PlayStream."
            )

        stream_id = str(
            int(time.time() * 1000)
        )

        playback_started_at = time.monotonic()

        max_playstream_attempts = 2
        retry_delay_sec = 0.50

        result = None
        playstream_success = False

        # -----------------------------------------------------
        # PlayStream attempts
        # -----------------------------------------------------

        for attempt in range(
            1,
            max_playstream_attempts + 1,
        ):

            if cancel_event.is_set():
                break

            print(
                "UNITREE_PLAYSTREAM_ATTEMPT "
                f"attempt={attempt}/{max_playstream_attempts} "
                f"bytes={len(pcm_data)}",
                flush=True,
            )

            started_at = time.monotonic()

            # Mark this BEFORE calling the SDK:
            # the SDK may start output before it returns.
            stream_may_exist = True

            result = client.PlayStream(
                stream_name,
                stream_id,
                pcm_data,
            )

            elapsed = (
                time.monotonic() - started_at
            )

            code = result_code(result)

            print(
                "UNITREE_PLAYSTREAM_END "
                f"attempt={attempt}/{max_playstream_attempts} "
                f"elapsed={elapsed:.3f}s "
                f"code={code} "
                f"result={result!r}",
                flush=True,
            )

            # Cancellation always takes priority over retry.
            if cancel_event.is_set():
                break

            if code == 0:
                playstream_success = True
                break

            if attempt < max_playstream_attempts:
                print(
                    "UNITREE_PLAYSTREAM_RETRY "
                    f"after_code={code} "
                    f"delay={retry_delay_sec:.2f}s",
                    flush=True,
                )

                # Interruptible retry delay.
                cancel_event.wait(
                    retry_delay_sec
                )

        # -----------------------------------------------------
        # Cancellation after possible stream creation
        # -----------------------------------------------------

        if cancel_event.is_set():

            if stream_may_exist:

                require_success(
                    client.PlayStop(stream_name),
                    "PlayStop during cancellation",
                )

                stop_confirmed = True

            print(
                "UNITREE_CANCEL_CONFIRMED "
                f"stream_may_exist={stream_may_exist} "
                f"stop_confirmed={stop_confirmed}",
                flush=True,
            )

            raise InterruptedError(
                "Unitree playback cancellation confirmed."
            )

        # -----------------------------------------------------
        # Normal playback error
        # -----------------------------------------------------

        if not playstream_success:

            if stream_may_exist:

                require_success(
                    client.PlayStop(stream_name),
                    "PlayStop after PlayStream failure",
                )

                stop_confirmed = True

            require_success(
                result,
                "PlayStream",
            )

            raise RuntimeError(
                "PlayStream did not complete successfully."
            )

        # -----------------------------------------------------
        # Wait for the expected speaker playback duration
        # -----------------------------------------------------

        elapsed = (
            time.monotonic()
            - playback_started_at
        )

        remaining_playback_sec = max(
            0.0,
            audio_duration_sec
            - elapsed
            + playback_margin_sec,
        )

        if remaining_playback_sec > 0.0:

            cancel_event.wait(
                remaining_playback_sec
            )

        # Whether natural playback has finished or
        # cancellation occurred, explicitly stop the stream.
        require_success(
            client.PlayStop(stream_name),
            "PlayStop",
        )

        stop_confirmed = True

        if cancel_event.is_set():

            print(
                "UNITREE_CANCEL_CONFIRMED "
                "stream_may_exist=True "
                "stop_confirmed=True",
                flush=True,
            )

            raise InterruptedError(
                "Unitree playback cancellation confirmed."
            )

        total_elapsed = (
            time.monotonic()
            - playback_started_at
        )

        print(
            "UNITREE_PLAYBACK_OK "
            f"bytes={len(pcm_data)} "
            f"audio_duration={audio_duration_sec:.2f}s "
            f"elapsed={total_elapsed:.2f}s "
            f"margin={playback_margin_sec:.2f}s",
            flush=True,
        )

    except InterruptedError:
        # Cancellation was either before playback or
        # acknowledged after successful PlayStop.
        raise

    except Exception as exc:

        # No PlayStream call has been made by this subprocess.
        # Do not classify a preparation error as an
        # unconfirmed playback stop.
        if not stream_may_exist:
            raise PreStreamFailure(
                f"{type(exc).__name__}: {exc}"
            ) from exc

        # A stream may have started.
        # Preserve the existing emergency-stop policy.
        if (
            client is not None
            and not stop_confirmed
        ):
            try:
                require_success(
                    client.PlayStop(stream_name),
                    "Emergency PlayStop",
                )

            except Exception as stop_exc:
                print(
                    "UNITREE_EMERGENCY_STOP_FAILED "
                    f"error={stop_exc}",
                    file=sys.stderr,
                    flush=True,
                )

        raise

    finally:
        signal.signal(
            signal.SIGTERM,
            previous_sigterm_handler,
        )

# =============================================================================
# COMMAND-LINE ARGUMENTS
# =============================================================================


def parse_arguments():
    """Parse and return command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Play 16-kHz mono signed 16-bit PCM through "
            "the integrated Unitree G1 speaker."
        )
    )

    parser.add_argument(
        "--interface",
        required=True,
        help=(
            "Unitree network interface, normally eth0."
        ),
    )

    parser.add_argument(
        "--pcm-file",
        required=True,
        help=(
            "Path to the raw 16-kHz mono signed 16-bit PCM file."
        ),
    )

    parser.add_argument(
        "--volume",
        type=int,
        default=DEFAULT_VOLUME,
        help=(
            "Unitree speaker volume between 0 and 100."
        ),
    )

    parser.add_argument(
        "--stream-name",
        default=DEFAULT_STREAM_NAME,
        help=(
            "Application stream name sent to AudioClient."
        ),
    )

    parser.add_argument(
        "--playback-margin",
        type=float,
        default=DEFAULT_PLAYBACK_MARGIN_SEC,
        help=(
            "Additional waiting time before PlayStop."
        ),
    )

    parser.add_argument(
        "--client-timeout",
        type=float,
        default=DEFAULT_CLIENT_TIMEOUT_SEC,
        help=(
            "Unitree AudioClient timeout in seconds."
        ),
    )

    return parser.parse_args()


# =============================================================================
# ARGUMENT VALIDATION
# =============================================================================


def validate_arguments(
    arguments,
) -> None:
    """
    Validate command-line arguments.

    PCM samples contain two bytes, so each chunk must contain an even
    number of bytes.
    """

    if not arguments.interface.strip():
        raise ValueError(
            "The network interface cannot be empty."
        )

    if not 0 <= arguments.volume <= 100:
        raise ValueError(
            "Volume must be between 0 and 100."
        )

    if not arguments.stream_name.strip():
        raise ValueError(
            "Stream name cannot be empty."
        )

    if arguments.playback_margin < 0.0:
        raise ValueError(
            "Playback margin cannot be negative."
        )

    if arguments.client_timeout <= 0.0:
        raise ValueError(
            "Client timeout must be greater than zero."
        )




# =============================================================================
# MAIN
# =============================================================================


def main() -> int:
    """
    Run the isolated Unitree PCM player.

    Exit codes:
        0: playback completed
        1: playback failed or stop unconfirmed
        3: cancellation confirmed
    """

    arguments = parse_arguments()

    try:
        validate_arguments(
            arguments
        )

        play_pcm(
            interface=arguments.interface.strip(),
            pcm_file=Path(
                arguments.pcm_file
            ).expanduser(),
            volume=arguments.volume,
            stream_name=arguments.stream_name.strip(),
            playback_margin_sec=(
                arguments.playback_margin
            ),
            client_timeout_sec=(
                arguments.client_timeout
            ),
        )

        return 0

    except InterruptedError as exc:

        print(
            "UNITREE_PLAYBACK_CANCELED "
            f"reason={exc}",
            flush=True,
        )

        return 3

    except PreStreamFailure as exc:

        print(
            "UNITREE_PRESTREAM_FAILED: "
            f"{exc}",
            file=sys.stderr,
            flush=True,
        )

        return 4

    except Exception as exc:

        print(
            "UNITREE_PLAYBACK_ERROR: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )

        return 1

if __name__ == "__main__":
    raise SystemExit(
        main()
    )