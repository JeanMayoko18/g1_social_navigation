#!/usr/bin/env python3
from __future__ import annotations

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image


class CameraInfoPublisher(Node):
    def __init__(self) -> None:
        super().__init__("camera_info_publisher")

        self.image_topic = self.declare_parameter("image_topic", "/camera/image_raw").value
        self.camera_info_topic = self.declare_parameter("camera_info_topic", "/camera/camera_info").value
        self.frame_id = self.declare_parameter("frame_id", "camera_link").value

        self.width = int(self.declare_parameter("width", 1280).value)
        self.height = int(self.declare_parameter("height", 720).value)

        # Approximate intrinsics for first tests.
        # Replace later with real calibration.
        self.fx = float(self.declare_parameter("fx", 900.0).value)
        self.fy = float(self.declare_parameter("fy", 900.0).value)
        self.cx = float(self.declare_parameter("cx", self.width / 2.0).value)
        self.cy = float(self.declare_parameter("cy", self.height / 2.0).value)

        self.distortion_model = self.declare_parameter("distortion_model", "plumb_bob").value
        self.d = list(self.declare_parameter("d", [0.0, 0.0, 0.0, 0.0, 0.0]).value)

        self.pub = self.create_publisher(CameraInfo, self.camera_info_topic, qos_profile_sensor_data)
        self.sub = self.create_subscription(Image, self.image_topic, self._cb, qos_profile_sensor_data)

        self.get_logger().info(
            f"Publishing {self.camera_info_topic} from {self.image_topic} "
            f"({self.width}x{self.height}, fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy})"
        )

    def _cb(self, msg: Image) -> None:
        cam = CameraInfo()
        cam.header = msg.header
        cam.header.frame_id = self.frame_id

        cam.width = self.width
        cam.height = self.height
        cam.distortion_model = self.distortion_model
        cam.d = self.d

        cam.k = [
            self.fx, 0.0, self.cx,
            0.0, self.fy, self.cy,
            0.0, 0.0, 1.0
        ]

        cam.r = [
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0
        ]

        cam.p = [
            self.fx, 0.0, self.cx, 0.0,
            0.0, self.fy, self.cy, 0.0,
            0.0, 0.0, 1.0, 0.0
        ]

        self.pub.publish(cam)


def main() -> None:
    rclpy.init()
    node = CameraInfoPublisher()
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