
#!/usr/bin/env python3
"""
person_spatial_tracking_node.py

Adapt the new identified PersonDetectionArray to the existing
social tracking outputs.

INPUT
    /social_perception/detections_identified
        g1_social_msgs/msg/PersonDetectionArray

OUTPUTS
    /social_perception/tracked_persons_odom
        g1_social_msgs/msg/TrackedPersonArray

    /social_perception/tracked_persons
        g1_social_msgs/msg/TrackedPersonArray

IDENTITY
    Identity is exclusively assigned by person_tracking.
    This node never creates, matches or confirms identities.

GEOMETRY
    Only actual, valid position observations enter motion estimates.
    Missing positions do not become zero-position observations.
    Historical tracks never refresh their observation timestamps.
    Robot-frame transforms use the source observation timestamp.

SOCIAL CONTRACT
    TrackedPersonArray preserves:
      tracking_session_id
      source_detection_count
      source_observation_complete

    Each TrackedPerson preserves:
      id, identity_generation
      last_observation_stamp
      observed_in_current_array
      tracking_state
      position, velocity, velocity_valid
      observation_count
"""

import copy
import math
import time
from dataclasses import dataclass

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time

from tf2_ros import (
    Buffer,
    TransformException,
    TransformListener,
)

from g1_social_msgs.msg import (
    FeatureMetadata,
    PersonDetectionArray,
    TrackedPerson,
    TrackedPersonArray,
)

from g1_social_perception.mathematical_model.geometry import (
    rotation_from_quaternion,
)

from g1_qos_manager.profiles import reliable_qos
from g1_qos_manager.resolver import resolve_subscription_qos
from g1_qos_manager.diagnostics import log_qos_resolution


# ============================================================================
# DATA / VALIDATION
# ============================================================================

VALID_IDENTITY_STATES = frozenset({
    "NEW",
    "MATCHED",
    "RETURNING",
    "AMBIGUOUS",
    "INSUFFICIENT",
})


def stamp_ns(stamp):
    sec = int(stamp.sec)
    nanosec = int(stamp.nanosec)

    if sec < 0 or not 0 <= nanosec < 1_000_000_000:
        raise ValueError("INVALID_TIMESTAMP")

    result = sec * 1_000_000_000 + nanosec

    if result <= 0:
        raise ValueError("ZERO_TIMESTAMP")

    return result


def finite_vector(values):
    vector = tuple(float(value) for value in values)

    if len(vector) != 3 or not all(
        math.isfinite(value) for value in vector
    ):
        raise ValueError("INVALID_VECTOR")

    return vector


def rotate(matrix, vector):
    return tuple(
        sum(matrix[i][j] * vector[j] for j in range(3))
        for i in range(3)
    )


def transform_pose(transform):
    """
    Read base_footprint <- odom from a timestamped TF.

    Position: p_base = R * p_odom + t
    Velocity: v_base = R * v_odom
    """

    q = transform.transform.rotation
    t = transform.transform.translation

    rotation = rotation_from_quaternion(
        float(q.x),
        float(q.y),
        float(q.z),
        float(q.w),
    )

    translation = finite_vector((t.x, t.y, t.z))

    return rotation, translation


def position_measurement(candidate, *, source_ns, frame):
    """
    Read only a genuine AVAILABLE odom position.

    An unavailable or zero-quality feature is not valid geometry.
    """

    metadata = candidate.position.metadata

    if (
        metadata.status != FeatureMetadata.AVAILABLE
        or metadata.quality <= 0.0
    ):
        return None

    if metadata.header.frame_id != frame:
        raise ValueError("POSITION_FRAME_MISMATCH")

    if stamp_ns(metadata.header.stamp) != source_ns:
        raise ValueError("POSITION_SOURCE_TIMESTAMP_MISMATCH")

    if not (
        math.isfinite(float(metadata.quality))
        and 0.0 < metadata.quality <= 1.0
    ):
        raise ValueError("INVALID_POSITION_QUALITY")

    p = candidate.position.value

    return finite_vector((p.x, p.y, p.z))


# ============================================================================
# DATA / TRACK RECORD
# ============================================================================

@dataclass
class SpatialTrack:
    person_id: str
    generation: int
    confirmed: bool

    class_id: int
    class_name: str
    confidence: float

    position: tuple
    velocity: tuple
    velocity_valid: bool

    stamp_ns: int
    received_monotonic: float

    observation_count: int

    # Historical base-frame coordinates correspond to the same
    # original source timestamp as position and stamp_ns.
    base_position: object = None
    base_velocity: object = None


# ============================================================================
# ROS NODE
# ============================================================================

class PersonSpatialTrackingNode(Node):

    def __init__(self):
        super().__init__("person_spatial_tracking")

        self._configure()
        self._initialize_state()
        self._create_interfaces()

        self.get_logger().info(
            "[SPATIAL] "
            f"input={self.input_topic} | "
            f"odom_output={self.odom_topic} | "
            f"robot_output={self.output_topic} | "
            "identity=INHERITED"
        )

    # ========================================================================
    # DATA / CONFIGURATION
    # ========================================================================

    def _configure(self):
        def parameter(name, default):
            return self.declare_parameter(name, default).value

        self.input_topic = parameter(
            "input_topic",
            "/social_perception/detections_identified",
        )

        self.odom_topic = parameter(
            "odom_output_topic",
            "/social_perception/tracked_persons_odom",
        )

        self.output_topic = parameter(
            "output_topic",
            "/social_perception/tracked_persons",
        )

        self.expected_frame = parameter("expected_frame", "odom")
        self.robot_frame = parameter(
            "robot_frame", "base_footprint"
        )

        self.alpha = float(
            parameter("velocity_filter_alpha", 0.35)
        )
        self.min_dt = float(
            parameter("minimum_velocity_dt_sec", 0.02)
        )
        self.max_dt = float(
            parameter("maximum_velocity_dt_sec", 1.0)
        )
        self.max_speed = float(
            parameter("maximum_speed_mps", 4.0)
        )

        self.freshness = float(
            parameter("observation_freshness_timeout_sec", 1.0)
        )
        self.retention = float(
            parameter("delete_timeout_sec", 5.0)
        )
        self.tf_timeout = float(
            parameter("transform_timeout_sec", 0.05)
        )

        self.log_period = float(
            parameter("log_period_sec", 5.0)
        )

        if (
            not all(
                math.isfinite(value)
                for value in (
                    self.alpha,
                    self.min_dt,
                    self.max_dt,
                    self.max_speed,
                    self.freshness,
                    self.retention,
                    self.tf_timeout,
                    self.log_period,
                )
            )
            or not 0.0 < self.alpha <= 1.0
            or not 0.0 < self.min_dt <= self.max_dt
            or min(
                self.max_speed,
                self.freshness,
                self.retention,
                self.log_period,
            ) <= 0.0
            or not 0.0 <= self.tf_timeout < self.freshness
            or self.expected_frame != "odom"
            or self.robot_frame != "base_footprint"
        ):
            raise ValueError("INVALID_SPATIAL_CONFIGURATION")

        if len({
            self.input_topic,
            self.odom_topic,
            self.output_topic,
        }) != 3:
            raise ValueError("SPATIAL_TOPIC_COLLISION")

    # ========================================================================
    # DATA / STATE
    # ========================================================================

    def _initialize_state(self):
        self.session = ""
        self.retired_sessions = set()

        self.last_source_ns = 0
        self.last_source_sequence = 0

        self.tracks = {}

        self.accepted = 0
        self.rejected = 0
        self.tf_failures = 0
        self.velocity_rejected = 0
        self.odom_published = 0
        self.base_published = 0

    # ========================================================================
    # ROS INTERFACES
    # ========================================================================

    def _create_interfaces(self):
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(
            self.tf_buffer,
            self,
        )

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

        self.odom_publisher = self.create_publisher(
            TrackedPersonArray,
            self.odom_topic,
            reliable_qos(depth=10),
        )

        self.publisher = self.create_publisher(
            TrackedPersonArray,
            self.output_topic,
            reliable_qos(depth=10),
        )

        self.subscription = self.create_subscription(
            PersonDetectionArray,
            self.input_topic,
            self.detections_callback,
            input_qos,
        )

        # Maintenance never generates new source observations.
        self.maintenance_timer = self.create_timer(
            0.5,
            self.expire_tracks,
        )

        self.diagnostics_timer = self.create_timer(
            self.log_period,
            self.log_diagnostics,
        )

    # ========================================================================
    # INPUT / VALIDATION
    # ========================================================================

    def fresh(self, source_ns):
        age = (
            self.get_clock().now().nanoseconds - source_ns
        ) / 1e9

        return -0.10 <= age <= self.freshness

    def _validate_input(self, msg):
        if not isinstance(msg, PersonDetectionArray):
            raise ValueError("INVALID_MESSAGE_TYPE")

        if (
            msg.processing_stage != PersonDetectionArray.IDENTITY
            or not msg.identity_evaluated
            or not str(msg.tracking_session_id).strip()
        ):
            raise ValueError("EXPECTED_IDENTIFIED_STAGE")

        if msg.header.frame_id != self.expected_frame:
            raise ValueError("EXPECTED_ODOM_FRAME")

        source_ns = stamp_ns(msg.header.stamp)

        if source_ns != stamp_ns(msg.image_header.stamp):
            raise ValueError("IMAGE_SOURCE_TIMESTAMP_MISMATCH")

        if not self.fresh(source_ns):
            raise ValueError("SOURCE_NOT_FRESH")

        session = str(msg.tracking_session_id).strip()

        if session in self.retired_sessions:
            raise ValueError("RETIRED_TRACKING_SESSION")

        if session == self.session:
            if (
                source_ns <= self.last_source_ns
                or int(msg.source_sequence)
                <= self.last_source_sequence
            ):
                raise ValueError("DUPLICATE_OR_REVERSED_SOURCE")

        if int(msg.source_sequence) < 1:
            raise ValueError("INVALID_SOURCE_SEQUENCE")

        if (
            int(msg.source_detection_count)
            != len(msg.candidates)
        ):
            raise ValueError("SOURCE_COUNT_MISMATCH")

        indices = set()
        person_ids = set()
        accepted = []

        for candidate in msg.candidates:
            index = int(candidate.source_index)

            if index in indices:
                raise ValueError("DUPLICATE_CANDIDATE_INDEX")

            indices.add(index)

            identity_state = str(candidate.identity_state)
            person_id = str(candidate.person_id).strip()

            if identity_state not in VALID_IDENTITY_STATES:
                raise ValueError("INVALID_IDENTITY_STATE")

            if identity_state in {
                "AMBIGUOUS", "INSUFFICIENT"
            }:
                if (
                    person_id
                    or candidate.identity_confirmed
                    or int(candidate.identity_generation) != 0
                ):
                    raise ValueError("UNRESOLVED_IDENTITY_HAS_KEY")
                continue

            if (
                not person_id
                or int(candidate.identity_generation) < 1
                or person_id in person_ids
            ):
                raise ValueError("INVALID_OR_DUPLICATE_IDENTITY")

            person_ids.add(person_id)

            # A valid identity without valid position does not
            # establish a spatial observation.
            position = position_measurement(
                candidate,
                source_ns=source_ns,
                frame=self.expected_frame,
            )

            if position is None:
                continue

            detection = candidate.source_detection
            confidence = float(detection.score)

            if (
                not math.isfinite(confidence)
                or not 0.0 <= confidence <= 1.0
            ):
                raise ValueError("INVALID_DETECTION_CONFIDENCE")

            accepted.append(
                (
                    candidate,
                    position,
                    person_id,
                    confidence,
                )
            )

        return source_ns, session, accepted

    # ========================================================================
    # PROCESSING / VELOCITY
    # ========================================================================

    def motion(
        self,
        previous,
        position,
        generation,
        source_ns,
        previous_frame_ns,
    ):
        """
        Differentiate two genuine observations in one continuous
        identity generation.

        An observation gap invalidates the velocity derivative.
        """

        zero = (0.0, 0.0, 0.0)

        if (
            previous is None
            or previous.generation != generation
            or previous.stamp_ns != previous_frame_ns
        ):
            return zero, False, False

        dt = (source_ns - previous.stamp_ns) / 1e9

        if not self.min_dt <= dt <= self.max_dt:
            return zero, False, False

        measured = tuple(
            (now - old) / dt
            for now, old in zip(
                position, previous.position
            )
        )

        speed = math.sqrt(
            sum(value * value for value in measured)
        )

        if not math.isfinite(speed) or speed > self.max_speed:
            return zero, False, True

        if previous.velocity_valid:
            velocity = tuple(
                self.alpha * current
                + (1.0 - self.alpha) * old
                for current, old in zip(
                    measured,
                    previous.velocity,
                )
            )
        else:
            velocity = measured

        return velocity, True, False

    # ========================================================================
    # PROCESSING / SOURCE-TIME TF
    # ========================================================================

    def _lookup_base_transform(self, msg):
        transform = self.tf_buffer.lookup_transform(
            self.robot_frame,
            self.expected_frame,
            Time.from_msg(msg.header.stamp),
            timeout=Duration(seconds=self.tf_timeout),
        )

        if (
            transform.header.frame_id != self.robot_frame
            or transform.child_frame_id != self.expected_frame
        ):
            raise ValueError("BASE_TF_FRAME_MISMATCH")

        return transform_pose(transform)

    # ========================================================================
    # PROCESSING / OBSERVATION
    # ========================================================================

    def detections_callback(self, msg):
        try:
            source_ns, session, accepted = (
                self._validate_input(msg)
            )

            changed = session != self.session

            previous_frame_ns = (
                0 if changed else self.last_source_ns
            )

            now = time.monotonic()

            # A new tracking session starts a new spatial history.
            # Retained tracks belong only to their original session.
            staged = (
                {}
                if changed
                else {
                    person_id: copy.deepcopy(track)
                    for person_id, track in self.tracks.items()
                    if (
                        0.0 <= (
                            source_ns - track.stamp_ns
                        ) / 1e9 <= self.retention
                        and 0.0 <= (
                            now - track.received_monotonic
                        ) <= self.retention
                    )
                }
            )

            projection = None

            if accepted:
                try:
                    projection = self._lookup_base_transform(
                        msg
                    )

                except (
                    TransformException,
                    ValueError,
                    TypeError,
                    OverflowError,
                ) as exc:
                    self.tf_failures += 1
                    self.get_logger().warning(
                        "[SPATIAL TF] "
                        "base_footprint projection unavailable: "
                        + str(exc),
                        throttle_duration_sec=2.0,
                    )

            if not self.fresh(source_ns):
                raise ValueError("SOURCE_EXPIRED_DURING_PROCESSING")

            rejected_velocity_count = 0

            for candidate, position, person_id, confidence in accepted:
                generation = int(
                    candidate.identity_generation
                )

                previous = staged.get(person_id)

                velocity, velocity_valid, velocity_rejected = (
                    self.motion(
                        previous,
                        position,
                        generation,
                        source_ns,
                        previous_frame_ns,
                    )
                )

                rejected_velocity_count += int(
                    velocity_rejected
                )

                count = (
                    previous.observation_count + 1
                    if (
                        previous is not None
                        and previous.generation == generation
                    )
                    else 1
                )

                detection = candidate.source_detection

                track = SpatialTrack(
                    person_id=person_id,
                    generation=generation,
                    confirmed=bool(
                        candidate.identity_confirmed
                    ),
                    class_id=int(detection.class_id),
                    class_name=str(
                        getattr(
                            detection,
                            "class_name",
                            "person",
                        )
                    ) or "person",
                    confidence=confidence,
                    position=position,
                    velocity=velocity,
                    velocity_valid=velocity_valid,
                    stamp_ns=source_ns,
                    received_monotonic=now,
                    observation_count=count,
                )

                if projection is not None:
                    rotation, translation = projection

                    rotated_position = rotate(
                        rotation,
                        position,
                    )

                    track.base_position = tuple(
                        rotated_position[i] + translation[i]
                        for i in range(3)
                    )

                    track.base_velocity = rotate(
                        rotation,
                        velocity,
                    )

                staged[person_id] = track

            # Source coverage is not the same as the number
            # of tracks retained from earlier observations.
            complete = bool(
                msg.source_observation_complete
                and len(accepted) == len(msg.candidates)
            )

            odom_output = self.build_array(
                msg=msg,
                tracks=staged,
                source_ns=source_ns,
                complete=complete,
                robot=False,
            )

            base_output = None

            if not accepted or projection is not None:
                base_output = self.build_array(
                    msg=msg,
                    tracks=staged,
                    source_ns=source_ns,
                    complete=complete,
                    robot=True,
                )

            # Commit only after successful validation and
            # construction of the output messages.
            if changed and self.session:
                self.retired_sessions.add(self.session)

            self.tracks = staged
            self.session = session
            self.last_source_ns = source_ns
            self.last_source_sequence = int(
                msg.source_sequence
            )

            self.accepted += 1
            self.velocity_rejected += rejected_velocity_count

            self.odom_publisher.publish(odom_output)
            self.odom_published += 1

            if base_output is not None:
                self.publisher.publish(base_output)
                self.base_published += 1

            if changed:
                self.get_logger().info(
                    "[SPATIAL] Inherited identity session="
                    + self.session
                )

        except (
            ValueError,
            TypeError,
            AttributeError,
            OverflowError,
        ) as exc:
            self.rejected += 1

            self.get_logger().warning(
                "[SPATIAL] Rejected observation: "
                + str(exc),
                throttle_duration_sec=2.0,
            )

    # ========================================================================
    # OUTPUT / TRACKED PERSON ARRAYS
    # ========================================================================

    def build_array(
        self,
        *,
        msg,
        tracks,
        source_ns,
        complete,
        robot,
    ):
        output = TrackedPersonArray()

        output.header = copy.deepcopy(msg.header)
        output.header.frame_id = (
            self.robot_frame
            if robot
            else self.expected_frame
        )

        output.tracking_session_id = (
            msg.tracking_session_id
        )

        output.source_detection_count = int(
            msg.source_detection_count
        )

        output.source_observation_complete = complete

        now_ros_ns = self.get_clock().now().nanoseconds

        for person_id, track in sorted(tracks.items()):
            if robot and track.base_position is None:
                output.source_observation_complete = False
                continue

            observed = track.stamp_ns == source_ns

            position = (
                track.base_position
                if robot
                else track.position
            )

            velocity = (
                track.base_velocity
                if robot
                else track.velocity
            )

            person = TrackedPerson()

            person.id = person_id
            person.class_id = track.class_id
            person.class_name = track.class_name
            person.confidence = float(
                track.confidence
            )

            person.tracking_session_id = (
                msg.tracking_session_id
            )

            person.identity_generation = (
                track.generation
            )

            seconds, nanoseconds = divmod(
                track.stamp_ns,
                1_000_000_000,
            )

            person.last_observation_stamp.sec = seconds
            person.last_observation_stamp.nanosec = nanoseconds

            person.observed_in_current_array = observed
            person.observation_count = (
                track.observation_count
            )

            person.time_since_last_seen = max(
                0.0,
                (now_ros_ns - track.stamp_ns) / 1e9,
            )

            if not observed:
                person.tracking_state = "LOST"
            elif track.confirmed:
                # Confirmation belongs to the identity engine.
                person.tracking_state = "TRACKED"
            else:
                person.tracking_state = "TENTATIVE"

            for axis, value in zip("xyz", position):
                setattr(
                    person.position,
                    axis,
                    float(value),
                )

            for axis, value in zip("xyz", velocity):
                setattr(
                    person.velocity,
                    axis,
                    float(value),
                )

            person.velocity_valid = bool(
                observed and track.velocity_valid
            )

            person.speed = float(
                math.hypot(
                    velocity[0],
                    velocity[1],
                )
            )

            person.motion_heading = (
                float(
                    math.atan2(
                        velocity[1],
                        velocity[0],
                    )
                )
                if person.speed > 1e-4
                else 0.0
            )

            output.persons.append(person)

        return output

    # ========================================================================
    # MAINTENANCE
    # ========================================================================

    def expire_tracks(self):
        """
        Remove expired history.

        No publication:
        a timer is not a new source observation.
        """

        now = time.monotonic()

        self.tracks = {
            person_id: track
            for person_id, track in self.tracks.items()
            if (
                0.0 <= (
                    now - track.received_monotonic
                ) <= self.retention
            )
        }

    # ========================================================================
    # DIAGNOSTICS
    # ========================================================================

    def log_diagnostics(self):
        self.get_logger().info(
            "[SPATIAL] "
            f"accepted={self.accepted} | "
            f"rejected={self.rejected} | "
            f"retained={len(self.tracks)} | "
            f"odom_tx={self.odom_published} | "
            f"base_tx={self.base_published} | "
            f"tf_failures={self.tf_failures} | "
            f"velocity_rejected={self.velocity_rejected}"
        )


# ============================================================================
# MAIN
# ============================================================================

def main(args=None):
    rclpy.init(args=args)

    node = None

    try:
        node = PersonSpatialTrackingNode()
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        if node is not None:
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
