#!/usr/bin/env python3
'''
Data flow up to here
**************************

/yolo/detections
       │
       ▼
PersonTrackingNode
       │
       ▼
/yolo/detections_tracked
       │
       ▼
3D perception
       │
       ▼
/yolo/detections_3d_base
       │
       ▼
PersonSpatialTrackingNode
       │
       ▼
/social_perception/tracked_persons
       │
       │ TrackedPersonArray
       ▼
┌─────────────────────────┐
│ SpeakerAssociation      │
│                         │
│ person_1 → position 3D  │
│ person_2 → position 3D  │
│ person_3 → position 3D  │
└────────────┬────────────┘
             │
             ▼
      /speaker_association

'''
from typing import Dict, Optional

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Point
from hri_msgs.msg import IdsList
from g1_social_msgs.msg import (
    TrackedPersonArray,
    TrackedPerson,
    SpeakerAssociation,
)



class SpeakerAssociationNode(Node):

    def __init__(self):
        super().__init__("speaker_association")

        # ---------------------------------------------------------
        # Parameters
        # ---------------------------------------------------------

        self.declare_parameter("enabled", False)
        self.declare_parameter("association_distance_threshold", 2.0)
        self.declare_parameter("min_confidence", 0.0)

        self.enabled = self.get_parameter(
            "enabled"
        ).get_parameter_value().bool_value

        self.distance_threshold = self.get_parameter(
            "association_distance_threshold"
        ).get_parameter_value().double_value

        self.min_confidence = self.get_parameter(
            "min_confidence"
        ).get_parameter_value().double_value

        # ---------------------------------------------------------
        # Internal state
        # ---------------------------------------------------------

        self.voice_ids = []

        self.tracked_people: Dict[str, TrackedPerson] = {}

        # Future:
        # voice_positions[voice_id] = geometry_msgs/Point
        #
        # This dictionary is intentionally empty for now.
        # It will later be populated by the speech/localization
        # component.
        self.voice_positions: Dict[str, Point] = {}

        # ---------------------------------------------------------
        # Subscribers
        # ---------------------------------------------------------

        self.voice_ids_sub = self.create_subscription(
            IdsList,
            "/humans/voices/tracked",
            self._voice_ids_callback,
            10,
        )

        self.people_sub = self.create_subscription(
            TrackedPersonArray,
            "/social_perception/tracked_persons",
            self._tracked_people_callback,
            10,
        )

        # ---------------------------------------------------------
        # Publisher
        # ---------------------------------------------------------

        self.association_pub = self.create_publisher(
            SpeakerAssociation,
            "/speaker_association",
            10,
        )

        # ---------------------------------------------------------
        # Timer
        # ---------------------------------------------------------

        self.timer = self.create_timer(
            0.2,
            self._association_cycle,
        )

        self.get_logger().info(
            "SpeakerAssociation node initialized."
        )

        self.get_logger().info(
            f"enabled={self.enabled}, "
            f"distance_threshold={self.distance_threshold:.2f} m"
        )

    # =============================================================
    # ROS callbacks
    # =============================================================

    def _voice_ids_callback(self, msg: IdsList) -> None:
        self.voice_ids = list(msg.ids)

    def _tracked_people_callback(
        self,
        msg: TrackedPersonArray,
    ) -> None:

        self.tracked_people.clear()

        for person in msg.persons:

            if not person.id:
                continue

            self.tracked_people[person.id] = person

    # =============================================================
    # Association
    # =============================================================

    def _association_cycle(self) -> None:

        if not self.enabled:
            return

        if not self.voice_ids:
            return

        if not self.tracked_people:
            return

        # ---------------------------------------------------------
        # IMPORTANT:
        #
        # At the moment we do not have voice positions.
        #
        # Therefore we DO NOT perform an association.
        # ---------------------------------------------------------

        for voice_id in self.voice_ids:

            if voice_id not in self.voice_positions:
                continue

            voice_position = self.voice_positions[voice_id]

            association = self._associate_voice(
                voice_id,
                voice_position,
            )

            if association is not None:
                self.association_pub.publish(association)

    # =============================================================
    # Voice → Person association
    # =============================================================

    def _associate_voice(
        self,
        voice_id: str,
        voice_position: Point,
    ) -> Optional[SpeakerAssociation]:

        best_person_id = None
        best_distance = float("inf")

        for person_id, person in self.tracked_people.items():

            if person.tracking_state != "TRACKED":
                continue

            distance = self._distance(
                voice_position,
                person.position,
            )

            if distance < best_distance:
                best_distance = distance
                best_person_id = person_id

        if best_person_id is None:
            return None

        if best_distance > self.distance_threshold:
            return None

        confidence = self._distance_to_confidence(
            best_distance
        )

        if confidence < self.min_confidence:
            return None

        msg = SpeakerAssociation()

        msg.header.stamp = self.get_clock().now().to_msg()

        msg.voice_id = voice_id
        msg.person_id = best_person_id

        msg.distance = float(best_distance)
        msg.confidence = float(confidence)
        msg.valid = True

        return msg

    # =============================================================
    # Utilities
    # =============================================================

    @staticmethod
    def _distance(
        a: Point,
        b: Point,
    ) -> float:

        dx = a.x - b.x
        dy = a.y - b.y
        dz = a.z - b.z

        return (dx * dx + dy * dy + dz * dz) ** 0.5

    def _distance_to_confidence(
        self,
        distance: float,
    ) -> float:

        if self.distance_threshold <= 0.0:
            return 0.0

        confidence = 1.0 - (
            distance / self.distance_threshold
        )

        return max(0.0, min(1.0, confidence))


def main(args=None):

    rclpy.init(args=args)

    node = SpeakerAssociationNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()