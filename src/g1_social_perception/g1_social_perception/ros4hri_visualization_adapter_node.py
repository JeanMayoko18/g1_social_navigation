#!/usr/bin/env python3

"""
Bridge G1 tracked 3D person detections to ROS4HRI visualization interfaces.

FLOW
====
/yolo/detections_3d_base
        |
        v
ROS4HRIVisualizationAdapter
        |
        +--> /humans/persons/tracked
        |
        +--> TF: person_<id>

RESPONSIBILITIES
================
This node provides visualization-level ROS4HRI compatibility.

It:
    - consumes tracked 3D person detections;
    - publishes the currently tracked ROS4HRI person IDs;
    - publishes one TF frame for each tracked person.

It MUST NOT:
    - perform person detection;
    - perform tracking;
    - invent body, face, or skeleton observations;
    - own dialogue;
    - own HRI arbitration;
    - own navigation;
    - own speech.

The input detection is expected to already be expressed in the configured
reference frame, normally base_footprint.
"""

import re

import rclpy

from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from tf2_ros import TransformBroadcaster
from yolo_msgs.msg import DetectionArray
from std_msgs.msg import String
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    DurabilityPolicy,
    qos_profile_sensor_data,
)
from hri_msgs.msg import IdsList, NormalizedRegionOfInterest2D


class ROS4HRIVisualizationAdapter(Node):

    def __init__(self):
        super().__init__("ros4hri_visualization_adapter")

        self.declare_parameter(
            "input_topic",
            "/yolo/detections_3d_base",
        )
        self.declare_parameter(
            "persons_tracked_topic",
            "/humans/persons/tracked",
        )
        self.bodies_tracked_pub = self.create_publisher(
            IdsList,
            "/humans/bodies/tracked",
            10,
        )

        # One transient-local publisher is created per tracked person so that
        # ROS4HRI consumers can recover the Person -> Body association.
        self.body_id_publishers = {}

        self.declare_parameter(
            "person_class_id",
            0,
        )
        self.declare_parameter(
            "default_frame_id",
            "base_footprint",
        )
        self.image_width = 640.0
        self.image_height = 480.0

        self.body_roi_publishers = {}


        input_topic = str(
            self.get_parameter("input_topic").value
        )
        persons_tracked_topic = str(
            self.get_parameter(
                "persons_tracked_topic"
            ).value
        )

        self.person_class_id = int(
            self.get_parameter("person_class_id").value
        )
        self.default_frame_id = str(
            self.get_parameter("default_frame_id").value
        )

        self.persons_pub = self.create_publisher(
            IdsList,
            persons_tracked_topic,
            10,
        )

        self.tf_broadcaster = TransformBroadcaster(self)

        self.subscription = self.create_subscription(
            DetectionArray,
            input_topic,
            self._detections_callback,
            10,
        )

        self.get_logger().info(
            "ROS4HRI visualization adapter ready: "
            f"INPUT={input_topic} | "
            f"PERSONS={persons_tracked_topic}"
        )

    @staticmethod
    def _normalize_person_id(raw_id: str) -> str:
        """
        Return a ROS4HRI-safe person identifier.

        ROS4HRI person IDs are used as ROS namespace tokens, therefore they
        must not start with a digit.

        Example:
            YOLO id: person_1
            ROS4HRI id: person_1
            TF frame: person_person_1
        """
        value = str(raw_id).strip()

        value = re.sub(
            r"[^A-Za-z0-9_-]",
            "_",
            value,
        )

        if not value:
            return ""

        if value[0].isdigit():
            value = f"person_{value}"

        return value

    def _get_body_roi_publisher(self, body_id: str):
        publisher = self.body_roi_publishers.get(body_id)

        if publisher is None:
            publisher = self.create_publisher(
                NormalizedRegionOfInterest2D,
                f"/humans/bodies/{body_id}/roi",
                10,
            )
            self.body_roi_publishers[body_id] = publisher

        return publisher

    def _get_body_roi_publisher(self, body_id: str):
        publisher = self.body_roi_publishers.get(body_id)

        if publisher is None:
            publisher = self.create_publisher(
                NormalizedRegionOfInterest2D,
                f"/humans/bodies/{body_id}/roi",
                10,
            )
            self.body_roi_publishers[body_id] = publisher

        return publisher

    def _get_body_id_publisher(self, person_id: str):
        publisher = self.body_id_publishers.get(person_id)

        if publisher is None:
            qos = QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            )

            publisher = self.create_publisher(
                String,
                f"/humans/persons/{person_id}/body_id",
                qos,
            )

            self.body_id_publishers[person_id] = publisher

        return publisher

    def _publish_body_roi(self, detection, body_id: str, source_header) -> None:
        """Publish the YOLO 2D bounding box as a normalized ROS4HRI body ROI."""
        center_x = float(detection.bbox.center.position.x)
        center_y = float(detection.bbox.center.position.y)
        width = float(detection.bbox.size.x)
        height = float(detection.bbox.size.y)

        if width <= 0.0 or height <= 0.0:
            return

        xmin = (center_x - width / 2.0) / self.image_width
        ymin = (center_y - height / 2.0) / self.image_height
        xmax = (center_x + width / 2.0) / self.image_width
        ymax = (center_y + height / 2.0) / self.image_height

        roi = NormalizedRegionOfInterest2D()

        # The ROI belongs to the original RGB image coordinate system.
        # The DetectionArray itself has already been transformed to base_footprint,
        # so its frame_id must not be reused for this image-space ROI.
        roi.header.stamp = source_header.stamp
        roi.header.frame_id = "camera_color_optical_frame"

        roi.xmin = max(0.0, min(1.0, xmin))
        roi.ymin = max(0.0, min(1.0, ymin))
        roi.xmax = max(0.0, min(1.0, xmax))
        roi.ymax = max(0.0, min(1.0, ymax))

        roi.c = max(0.0, min(1.0, float(detection.score)))

        if roi.xmax <= roi.xmin or roi.ymax <= roi.ymin:
            return

        self._get_body_roi_publisher(body_id).publish(roi)

    def _detections_callback(
        self,
        msg: DetectionArray,
    ) -> None:

        tracked_ids = []
        tracked_body_ids = []

        for detection in msg.detections:

            if detection.class_id != self.person_class_id:
                continue

            person_id = self._normalize_person_id(
                detection.id
            )
            body_id = person_id

            tracked_ids.append(person_id)
            tracked_body_ids.append(body_id)

            body_id_msg = String()
            body_id_msg.data = body_id
            self._get_body_id_publisher(person_id).publish(body_id_msg)
            
            self._publish_body_roi(
                detection=detection,
                body_id=body_id,
                source_header=msg.header,
            )


            if not person_id:
                continue

            bbox3d = detection.bbox3d

            parent_frame = (
                bbox3d.frame_id.strip()
                or msg.header.frame_id.strip()
                or self.default_frame_id
            )

            transform = TransformStamped()

            transform.header.stamp = msg.header.stamp
            transform.header.frame_id = parent_frame
            transform.child_frame_id = f"body_{body_id}"

            transform.transform.translation.x = (
                bbox3d.center.position.x
            )
            transform.transform.translation.y = (
                bbox3d.center.position.y
            )
            transform.transform.translation.z = (
                bbox3d.center.position.z
            )

            transform.transform.translation.x = detection.bbox3d.center.position.x
            transform.transform.translation.y = detection.bbox3d.center.position.y
            transform.transform.translation.z = detection.bbox3d.center.position.z

            # Person-level localization does not provide a meaningful
            # ROS4HRI body orientation. Use an identity rotation rather
            # than propagating the 3D bounding-box orientation.
            #transform.transform.rotation.x = 0.0
            #transform.transform.rotation.y = 0.0
            #transform.transform.rotation.z = 0.0
            #transform.transform.rotation.w = 1.0
            transform.transform.rotation.x = detection.bbox3d.center.orientation.x
            transform.transform.rotation.y = detection.bbox3d.center.orientation.y
            transform.transform.rotation.z = detection.bbox3d.center.orientation.z
            transform.transform.rotation.w = detection.bbox3d.center.orientation.w

            self.tf_broadcaster.sendTransform(
                transform
            )

        tracked_msg = IdsList()
        tracked_msg.header.stamp = msg.header.stamp
        tracked_msg.header.frame_id = (
            msg.header.frame_id.strip()
            or self.default_frame_id
        )
        tracked_msg.ids = tracked_ids

        bodies_msg = IdsList()
        bodies_msg.header.stamp = msg.header.stamp
        bodies_msg.header.frame_id = (
            msg.header.frame_id.strip()
            or self.default_frame_id
        )
        bodies_msg.ids = tracked_body_ids

        self.persons_pub.publish(tracked_msg)
        self.bodies_tracked_pub.publish(bodies_msg)


def main(args=None):
    rclpy.init(args=args)

    node = ROS4HRIVisualizationAdapter()

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