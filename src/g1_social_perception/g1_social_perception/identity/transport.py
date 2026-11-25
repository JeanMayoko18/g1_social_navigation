"""ROS-independent validation and image decoding helpers."""
import numpy as np


def stamp_ns(stamp):
    sec, nano = int(stamp.sec), int(stamp.nanosec)
    if sec < 0 or not 0 <= nano < 1000000000:
        raise ValueError("Invalid source timestamp")
    value = sec * 1000000000 + nano
    if value <= 0:
        raise ValueError("Zero source timestamp")
    return value


def validate_envelope(msg):
    """Validate provenance and stage-specific identity metadata."""
    source = stamp_ns(msg.header.stamp)

    if (
        not msg.source_session_id.strip()
        or not msg.header.frame_id.strip()
        or not msg.image_header.frame_id.strip()
        or stamp_ns(msg.image_header.stamp) != source
    ):
        raise ValueError("Invalid source session or image provenance")

    if bool(msg.tracking_session_id.strip()) != bool(msg.identity_evaluated):
        raise ValueError(
            "Identity session must exist only after identity evaluation"
        )

    count = len(msg.detections)

    if any(
        len(getattr(msg, name)) != count
        for name in (
            "identity_states",
            "identity_confirmed",
            "identity_generations",
            "spatial_valid",
        )
    ):
        raise ValueError("Misaligned detection metadata")

    assigned_ids = set()

    for index, detection in enumerate(msg.detections):
        state = msg.identity_states[index]
        confirmed = msg.identity_confirmed[index]
        generation = msg.identity_generations[index]

        if not msg.identity_evaluated:
            if (
                detection.id
                or confirmed
                or generation != 0
                or state != "INSUFFICIENT"
            ):
                raise ValueError("Upstream stages cannot assign identity")
            continue

        if state not in {
            "NEW", "MATCHED", "RETURNING", "AMBIGUOUS", "INSUFFICIENT"
        }:
            raise ValueError("Invalid identity state")

        if state in {"AMBIGUOUS", "INSUFFICIENT"}:
            if detection.id or confirmed or generation != 0:
                raise ValueError(
                    "Unresolved evidence cannot carry an assigned identity"
                )
        elif (
            not detection.id
            or detection.id in assigned_ids
            or generation <= 0
        ):
            raise ValueError("Invalid assigned identity")

        if detection.id:
            assigned_ids.add(detection.id)

    if (
        msg.identity_evaluated
        and msg.source_observation_complete
        and not (
            all(msg.identity_confirmed)
            and all(msg.spatial_valid)
        )
    ):
        raise ValueError(
            "Complete identified evidence requires confirmed spatial observations"
        )

def decode_image(msg, depth=False):
    """Respect ROS row stride, endianness and channel encoding."""
    if depth:
        if msg.encoding not in ('16UC1', 'mono16'):
            raise ValueError("Expected 16-bit aligned depth")
        dtype = np.dtype('>u2' if msg.is_bigendian else '<u2')
        channels, itemsize = 1, 2
    else:
        if msg.encoding not in ('rgb8', 'bgr8'):
            raise ValueError("Expected rgb8 or bgr8")
        dtype, channels, itemsize = np.dtype('u1'), 3, 1
    w, h, step = int(msg.width), int(msg.height), int(msg.step)
    if min(w, h) <= 0 or step < w*channels*itemsize or len(msg.data) != h*step:
        raise ValueError("Invalid image shape/row stride/payload")
    shape = (h, w) if depth else (h, w, channels)
    strides = (step, itemsize) if depth else (step, channels, 1)
    return np.ndarray(shape, dtype=dtype, buffer=bytes(msg.data), strides=strides).copy()


class SourceOrder:
    """Enforce source order before and after identity attribution."""

    def __init__(self):
        self.session = None
        self.last_ns = None
        self.retired = set()

    def check(self, msg):
        source = stamp_ns(msg.header.stamp)
        session = (
            msg.source_session_id,
            msg.tracking_session_id,
        )

        if not session[0] or session in self.retired:
            raise ValueError("Invalid or retired source session")

        if (
            session == self.session
            and self.last_ns is not None
            and source <= self.last_ns
        ):
            raise ValueError("Duplicate or out-of-order source")

        return source

    def accept(self, msg):
        source = self.check(msg)
        session = (
            msg.source_session_id,
            msg.tracking_session_id,
        )

        if self.session and self.session != session:
            self.retired.add(self.session)

        self.session = session
        self.last_ns = source

