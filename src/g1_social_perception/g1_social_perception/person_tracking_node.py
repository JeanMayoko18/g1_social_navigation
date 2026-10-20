"""
Track person detections across consecutive YOLO frames.

Responsibility:
    Associate current person detections with previously active tracks and assign
    persistent detection IDs.

Flow:
    INPUT
        /yolo/detections
            ↓
    PERSON FILTERING
            ↓
    TRACK ASSOCIATION
        IoU + center distance
            ↓
    TRACK UPDATE / CREATION / EXPIRATION
            ↓
    OUTPUT
        /yolo/detections_tracked

The node does not estimate depth and does not perform TF transforms.
It only manages temporal identity consistency.
"""

import copy
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
import time
import rclpy
from rclpy.node import Node

from yolo_msgs.msg import Detection
from yolo_msgs.msg import DetectionArray

from g1_qos_manager.diagnostics import log_qos_resolution
from g1_qos_manager.profiles import reliable_qos
from g1_qos_manager.resolver import resolve_subscription_qos


# =============================================================================
# RESULT OBJECTS
# =============================================================================


@dataclass
class TrackState:
    """
    Store the persistent state of one tracked person.

    Attributes:
        track_id:
            Internal numeric identifier.

        center_x / center_y:
            Last known 2D bounding-box center in pixels.

        width / height:
            Last known 2D bounding-box dimensions in pixels.

        missed_frames:
            Number of consecutive frames in which the track was not matched.

        age_frames:
            Total number of frames during which the track has existed.

        hits:
            Number of successful detection associations.
    """

    track_id: int

    center_x: float
    center_y: float

    width: float
    height: float

    missed_frames: int = 0
    age_frames: int = 1
    hits: int = 1


# =============================================================================
# TRACKING NODE
# =============================================================================


class PersonTrackingNode(Node):
    """
    Maintain persistent identities for person detections.

    Input:
        DetectionArray containing YOLO detections.

    Output:
        DetectionArray containing only person detections, each with a stable
        detection.id such as "person_1".
    """

    def __init__(self):
        super().__init__("person_tracking")

        # =====================================================================
        # INPUT / CONFIGURATION
        # =====================================================================

        self.declare_parameter(
            "input_topic",
            "/yolo/detections",
        )

        self.declare_parameter(
            "output_topic",
            "/yolo/detections_tracked",
        )

        self.declare_parameter(
            "person_class_id",
            0,
        )

        self.declare_parameter(
            "minimum_iou",
            0.20,
        )

        self.declare_parameter(
            "maximum_center_distance_px",
            100.0,
        )

        self.declare_parameter(
            "iou_weight",
            0.70,
        )

        self.declare_parameter(
            "center_weight",
            0.30,
        )

        self.declare_parameter(
            "minimum_match_score",
            0.35,
        )

        self.declare_parameter(
            "max_missed_frames",
            45,
        )

        self.input_topic = (
            self.get_parameter("input_topic")
            .get_parameter_value()
            .string_value
        )

        self.output_topic = (
            self.get_parameter("output_topic")
            .get_parameter_value()
            .string_value
        )

        self.person_class_id = (
            self.get_parameter("person_class_id")
            .get_parameter_value()
            .integer_value
        )

        self.minimum_iou = (
            self.get_parameter("minimum_iou")
            .get_parameter_value()
            .double_value
        )

        self.maximum_center_distance_px = (
            self.get_parameter("maximum_center_distance_px")
            .get_parameter_value()
            .double_value
        )

        self.iou_weight = (
            self.get_parameter("iou_weight")
            .get_parameter_value()
            .double_value
        )

        self.center_weight = (
            self.get_parameter("center_weight")
            .get_parameter_value()
            .double_value
        )

        self.minimum_match_score = (
            self.get_parameter("minimum_match_score")
            .get_parameter_value()
            .double_value
        )

        self.max_missed_frames = (
            self.get_parameter("max_missed_frames")
            .get_parameter_value()
            .integer_value
        )

        # =====================================================================
        # INTERNAL STATE
        # =====================================================================

        self.tracks: Dict[int, TrackState] = {}

        # ============================================================
        # OBSERVATION PROVENANCE - 2D
        # ============================================================

        self.last_input_stamp_sec = None
        self.last_input_received_monotonic = None

        self.accepted_observation_arrays = 0
        self.empty_observation_arrays = 0
        self.rejected_observation_timestamps = 0

        # Diagnostics.
        self.accepted_observation_arrays = 0
        self.empty_observation_arrays = 0
        self.rejected_observation_timestamps = 0
        self.stale_observation_cycles = 0

        # Maximum delay allowed between reception and publication.
        self.declare_parameter(
            "observation_freshness_timeout_sec",
            1.0,
        )

        self.observation_freshness_timeout_sec = float(
            self.get_parameter(
                "observation_freshness_timeout_sec"
            ).value
        )

        if (
            not math.isfinite(self.observation_freshness_timeout_sec)
            or self.observation_freshness_timeout_sec <= 0.0
        ):
            raise ValueError(
                "observation_freshness_timeout_sec must be finite and positive."
            )

        self.next_track_id = 1

        # Diagnostics counters.
        self.frames_received = 0
        self.detections_received = 0

        self.person_detections = 0

        self.matches = 0
        self.new_tracks = 0
        self.expired_tracks = 0

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

        output_qos = reliable_qos(
            depth=10,
        )

        # =====================================================================
        # ROS INTERFACES
        # =====================================================================

        self.subscription = self.create_subscription(
            DetectionArray,
            self.input_topic,
            self.detections_callback,
            input_qos,
        )

        self.publisher = self.create_publisher(
            DetectionArray,
            self.output_topic,
            output_qos,
        )

        # =====================================================================
        # REPORTING
        # =====================================================================

        self.diagnostics_timer = self.create_timer(
            5.0,
            self.log_diagnostics,
        )

        self.get_logger().info("")
        self.get_logger().info("=" * 82)
        self.get_logger().info(
            "  PERSON TRACKING - INITIALIZATION"
        )
        self.get_logger().info("=" * 82)

        self.get_logger().info(
            f"  Input topic              : {self.input_topic}"
        )
        self.get_logger().info(
            f"  Output topic             : {self.output_topic}"
        )
        self.get_logger().info(
            f"  Person class ID          : {self.person_class_id}"
        )
        self.get_logger().info(
            f"  Minimum IoU              : {self.minimum_iou:.2f}"
        )
        self.get_logger().info(
            f"  Maximum center distance  : "
            f"{self.maximum_center_distance_px:.1f} px"
        )
        self.get_logger().info(
            f"  Minimum match score      : "
            f"{self.minimum_match_score:.2f}"
        )
        self.get_logger().info(
            f"  Maximum missed frames    : {self.max_missed_frames}"
        )

        self.get_logger().info("-" * 82)

    # =========================================================================
    # COLLECTION
    # =========================================================================

    def detections_callback(
        self,
        msg: DetectionArray,
    ) -> None:
        """
        Associate YOLO 2D person detections with persistent tracks.

        Input:
            /yolo/detections

        Output:
            /yolo/detections_tracked

        Rules:
            - Process only genuine new source frames.
            - Preserve the original source timestamp and frame.
            - Use 2D bounding boxes, never bbox3d.
            - Associate detections using IoU and center distance.
            - Assign stable person IDs.
            - Expire missing tracks without inventing detections.
            - Reject malformed input without publishing a false
            empty observation.
        """

        self.frames_received += 1
        self.detections_received += len(msg.detections)

        # ------------------------------------------------------------
        # VALIDATE UPSTREAM OBSERVATION
        # ------------------------------------------------------------

        stamp = msg.header.stamp

        sec = int(stamp.sec)
        nanosec = int(stamp.nanosec)

        if (
            sec < 0
            or nanosec < 0
            or nanosec >= 1_000_000_000
        ):
            self.rejected_observation_timestamps += 1
            return

        stamp_sec = sec + nanosec * 1e-9

        if (
            stamp_sec <= 0.0
            or (
                self.last_input_stamp_sec is not None
                and stamp_sec <= self.last_input_stamp_sec
            )
        ):
            self.rejected_observation_timestamps += 1
            return

        if not str(msg.header.frame_id).strip():
            self.get_logger().warning(
                "[PERSON TRACKING] Rejected detection array "
                "with empty source frame.",
                throttle_duration_sec=2.0,
            )
            return

        # ------------------------------------------------------------
        # FILTER AND VALIDATE 2D PERSON DETECTIONS
        # ------------------------------------------------------------

        person_detections = []

        for detection in msg.detections:

            if int(detection.class_id) != self.person_class_id:
                continue

            center_x, center_y, width, height = (
                self.extract_bbox(detection)
            )

            if not all(
                math.isfinite(value)
                for value in (
                    center_x,
                    center_y,
                    width,
                    height,
                )
            ) or width <= 0.0 or height <= 0.0:

                self.get_logger().warning(
                    "[PERSON TRACKING] Rejected frame: "
                    "invalid person 2D bounding box.",
                    throttle_duration_sec=2.0,
                )
                return

            person_detections.append(detection)

        # The complete incoming frame is now accepted.

        self.last_input_stamp_sec = stamp_sec
        self.last_input_received_monotonic = time.monotonic()

        self.accepted_observation_arrays += 1
        self.person_detections += len(person_detections)

        if not person_detections:
            self.empty_observation_arrays += 1

        # ------------------------------------------------------------
        # ASSOCIATE WITH EXISTING TRACKS
        # ------------------------------------------------------------

        assignments = self.associate_detections(
            person_detections
        )

        matched_track_ids = set()
        matched_detection_indices = set()

        output_detections = [
            copy.deepcopy(detection)
            for detection in person_detections
        ]

        for track_id, detection_index in assignments:

            track = self.tracks.get(track_id)

            if track is None:
                continue

            detection = person_detections[detection_index]

            self.update_track(
                track,
                detection,
            )

            output_detections[detection_index].id = (
                f"person_{track_id}"
            )

            matched_track_ids.add(track_id)
            matched_detection_indices.add(detection_index)

            self.matches += 1

        # ------------------------------------------------------------
        # AGE UNMATCHED TRACKS
        # ------------------------------------------------------------

        for track_id, track in self.tracks.items():

            if track_id in matched_track_ids:
                continue

            track.missed_frames += 1
            track.age_frames += 1

        # ------------------------------------------------------------
        # CREATE NEW TRACKS FOR UNMATCHED DETECTIONS
        # ------------------------------------------------------------

        for detection_index, detection in enumerate(
            person_detections
        ):

            if detection_index in matched_detection_indices:
                continue

            track = self.create_track(detection)

            output_detections[detection_index].id = (
                f"person_{track.track_id}"
            )

        # ------------------------------------------------------------
        # REMOVE EXPIRED TRACKS
        # ------------------------------------------------------------

        self.remove_expired_tracks()

        # ------------------------------------------------------------
        # PUBLISH ACTUAL DETECTIONS
        # ------------------------------------------------------------

        output = copy.deepcopy(msg)

        # Only observed persons, never predicted or historical boxes.
        output.detections = output_detections

        # The original header remains unchanged.
        self.publisher.publish(output)

    # =========================================================================
    # PROCESSING / ASSOCIATION
    # =========================================================================

    def associate_detections(
        self,
        detections: List[Detection],
    ) -> List[Tuple[int, int]]:
        """
        Associate current detections with active tracks.

        Processing:
            Evaluates every track/detection pair using:
                - Intersection over Union.
                - 2D bounding-box center distance.

            The highest scoring non-conflicting pairs are selected greedily.

        Output:
            List of:
                (track_id, detection_index)

        Side effects:
            None.
        """

        candidates = []

        for track_id, track in self.tracks.items():
            for detection_index, detection in enumerate(
                detections
            ):
                iou = self.calculate_iou(
                    track,
                    detection,
                )

                center_distance = (
                    self.calculate_center_distance(
                        track,
                        detection,
                    )
                )

                # Reject obviously unrelated detections.
                if (
                    iou < self.minimum_iou
                    and center_distance
                    > self.maximum_center_distance_px
                ):
                    continue

                center_similarity = max(
                    0.0,
                    1.0
                    - (
                        center_distance
                        / self.maximum_center_distance_px
                    ),
                )

                score = (
                    self.iou_weight * iou
                    + self.center_weight
                    * center_similarity
                )

                if score < self.minimum_match_score:
                    continue

                candidates.append(
                    (
                        score,
                        track_id,
                        detection_index,
                    )
                )

        # Highest score first.
        candidates.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        assignments = []

        used_tracks: Set[int] = set()
        used_detections: Set[int] = set()

        for (
            _score,
            track_id,
            detection_index,
        ) in candidates:

            if track_id in used_tracks:
                continue

            if detection_index in used_detections:
                continue

            assignments.append(
                (
                    track_id,
                    detection_index,
                )
            )

            used_tracks.add(
                track_id
            )

            used_detections.add(
                detection_index
            )

        return assignments

    # =========================================================================
    # PROCESSING / TRACK STATE
    # =========================================================================

    def create_track(
        self,
        detection: Detection,
    ) -> TrackState:
        """
        Create a new persistent person track.

        Output:
            Newly created TrackState.

        Side effects:
            Inserts the track into self.tracks.
        """

        track_id = self.next_track_id

        self.next_track_id += 1

        center_x, center_y, width, height = (
            self.extract_bbox(
                detection
            )
        )

        track = TrackState(
            track_id=track_id,
            center_x=center_x,
            center_y=center_y,
            width=width,
            height=height,
        )

        self.tracks[
            track_id
        ] = track

        self.new_tracks += 1

        return track

    def update_track(
        self,
        track: TrackState,
        detection: Detection,
    ) -> None:
        """
        Update an existing track using a newly associated detection.

        Side effects:
            Mutates TrackState.
        """

        (
            center_x,
            center_y,
            width,
            height,
        ) = self.extract_bbox(
            detection
        )

        track.center_x = center_x
        track.center_y = center_y

        track.width = width
        track.height = height

        track.missed_frames = 0
        track.age_frames += 1
        track.hits += 1

    def remove_expired_tracks(
        self,
    ) -> None:
        """
        Remove tracks absent for too many consecutive frames.

        Side effects:
            Deletes expired TrackState objects.
        """

        expired_ids = [
            track_id
            for track_id, track in self.tracks.items()
            if track.missed_frames
            > self.max_missed_frames
        ]

        for track_id in expired_ids:
            del self.tracks[
                track_id
            ]

            self.expired_tracks += 1

    # =========================================================================
    # PROCESSING / GEOMETRY
    # =========================================================================

    @staticmethod
    def extract_bbox(
        detection: Detection,
    ) -> Tuple[
        float,
        float,
        float,
        float,
    ]:
        """
        Extract the YOLO 2D bounding box.

        Output:
            center_x, center_y, width, height
        """

        return (
            float(
                detection.bbox.center.position.x
            ),
            float(
                detection.bbox.center.position.y
            ),
            float(
                detection.bbox.size.x
            ),
            float(
                detection.bbox.size.y
            ),
        )

    def calculate_center_distance(
        self,
        track: TrackState,
        detection: Detection,
    ) -> float:
        """
        Compute Euclidean distance between bbox centers in pixels.

        Output:
            Pixel distance.
        """

        center_x, center_y, _, _ = (
            self.extract_bbox(
                detection
            )
        )

        return math.hypot(
            center_x - track.center_x,
            center_y - track.center_y,
        )

    def calculate_iou(
        self,
        track: TrackState,
        detection: Detection,
    ) -> float:
        """
        Compute Intersection over Union between track and detection boxes.

        Output:
            IoU in [0, 1].
        """

        detection_x, detection_y, detection_w, detection_h = (
            self.extract_bbox(
                detection
            )
        )

        track_x1 = (
            track.center_x
            - track.width / 2.0
        )

        track_y1 = (
            track.center_y
            - track.height / 2.0
        )

        track_x2 = (
            track.center_x
            + track.width / 2.0
        )

        track_y2 = (
            track.center_y
            + track.height / 2.0
        )

        detection_x1 = (
            detection_x
            - detection_w / 2.0
        )

        detection_y1 = (
            detection_y
            - detection_h / 2.0
        )

        detection_x2 = (
            detection_x
            + detection_w / 2.0
        )

        detection_y2 = (
            detection_y
            + detection_h / 2.0
        )

        intersection_x1 = max(
            track_x1,
            detection_x1,
        )

        intersection_y1 = max(
            track_y1,
            detection_y1,
        )

        intersection_x2 = min(
            track_x2,
            detection_x2,
        )

        intersection_y2 = min(
            track_y2,
            detection_y2,
        )

        intersection_width = max(
            0.0,
            intersection_x2
            - intersection_x1,
        )

        intersection_height = max(
            0.0,
            intersection_y2
            - intersection_y1,
        )

        intersection_area = (
            intersection_width
            * intersection_height
        )

        track_area = (
            track.width
            * track.height
        )

        detection_area = (
            detection_w
            * detection_h
        )

        union_area = (
            track_area
            + detection_area
            - intersection_area
        )

        if union_area <= 0.0:
            return 0.0

        return (
            intersection_area
            / union_area
        )

    # =========================================================================
    # OUTPUT / REPORTING
    # =========================================================================
    
    def log_diagnostics(self) -> None:
        """Report 2D tracking and source observation diagnostics."""

        now = time.monotonic()

        if self.last_input_received_monotonic is None:
            observation_age = float("inf")
        else:
            observation_age = max(
                0.0,
                now - self.last_input_received_monotonic,
            )

        self.get_logger().info(
            "[PERSON TRACKING 2D] "
            f"FRAMES_RX={self.frames_received} | "
            f"DETECTIONS_RX={self.detections_received} | "
            f"PERSONS_RX={self.person_detections} | "
            f"ACTIVE_TRACKS={len(self.tracks)} | "
            f"MATCHES={self.matches} | "
            f"NEW_TRACKS={self.new_tracks} | "
            f"EXPIRED_TRACKS={self.expired_tracks} | "
            f"OBS_ACCEPTED={self.accepted_observation_arrays} | "
            f"OBS_EMPTY={self.empty_observation_arrays} | "
            f"REJECTED_STAMPS="
            f"{self.rejected_observation_timestamps} | "
            f"OBS_AGE={observation_age:.2f}s"
        )


# =============================================================================
# ORCHESTRATION
# =============================================================================


def main(args=None):
    """Run the person tracking node."""

    rclpy.init(
        args=args
    )

    node = PersonTrackingNode()

    try:
        rclpy.spin(
            node
        )

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
