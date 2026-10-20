#!/usr/bin/env python3
"""
cloud_y_flip.py

==============================================================
PointCloud2 Y-Axis Mirror Correction Node
==============================================================

Purpose
-------
This node fixes a lateral left/right mirror issue in a PointCloud2 stream.

In the observed system:
    - forward/backward is correct
    - left/right is inverted

This corresponds to a point cloud convention mismatch where:
    x is correct
    y must be negated
    z remains unchanged

This node subscribes to an input PointCloud2 topic, flips the Y axis
for every point, and republishes the corrected cloud.

Transformation applied
----------------------
For each point:
    x' = x
    y' = -y
    z' = z

Typical use
-----------
    /utlidar/cloud_livox_mid360
            -> pc2_restamper
            -> /cloud_now
            -> cloud_y_flip
            -> /cloud_fixed
            -> pointcloud_to_laserscan
            -> /scan_raw

Notes
-----
- This node assumes fields x, y, z are float32.
- Other fields are preserved.
- The frame_id is preserved.
- Only the spatial Y coordinate is modified.
"""

from __future__ import annotations

import math
import struct
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from sensor_msgs.msg import PointCloud2, PointField


class CloudYFlip(Node):
    """
    Flip the Y coordinate of every point in a PointCloud2 message.
    """

    def __init__(self) -> None:
        super().__init__("cloud_y_flip")

        # ----------------------------------------------------------
        # Parameters
        # ----------------------------------------------------------
        self.input_topic = self.declare_parameter(
            "input_cloud_topic",
            "/cloud_now"
        ).value

        self.output_topic = self.declare_parameter(
            "output_cloud_topic",
            "/cloud_fixed"
        ).value

        # ----------------------------------------------------------
        # QoS
        # ----------------------------------------------------------
        self.qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ----------------------------------------------------------
        # Interfaces
        # ----------------------------------------------------------
        self.pub = self.create_publisher(PointCloud2, self.output_topic, self.qos)
        self.sub = self.create_subscription(
            PointCloud2,
            self.input_topic,
            self.cb,
            self.qos,
        )

        self.get_logger().info(
            f"[cloud_y_flip] Flipping Y axis: {self.input_topic} -> {self.output_topic}"
        )

    def _find_field_offset(self, msg: PointCloud2, field_name: str) -> Optional[int]:
        """
        Return the byte offset of a named field in the PointCloud2 layout.
        """
        for field in msg.fields:
            if field.name == field_name:
                return int(field.offset)
        return None

    def cb(self, msg: PointCloud2) -> None:
        """
        Process one PointCloud2 message and negate the Y coordinate of each point.
        """
        x_offset = self._find_field_offset(msg, "x")
        y_offset = self._find_field_offset(msg, "y")
        z_offset = self._find_field_offset(msg, "z")

        if x_offset is None or y_offset is None or z_offset is None:
            self.get_logger().error(
                "Input cloud does not contain x/y/z fields. Message skipped."
            )
            return

        # We assume x, y, z are float32 as is standard for PointCloud2.
        # Create a mutable copy of the binary payload.
        data = bytearray(msg.data)
        point_step = int(msg.point_step)
        num_points = int(len(data) / point_step)

        for i in range(num_points):
            base = i * point_step

            # Read original y as float32.
            y_bytes = data[base + y_offset: base + y_offset + 4]
            y_val = struct.unpack("<f", y_bytes)[0]

            # Flip lateral axis.
            y_new = -y_val

            # Write corrected y back.
            data[base + y_offset: base + y_offset + 4] = struct.pack("<f", y_new)

        out = PointCloud2()
        out.header = msg.header
        out.height = msg.height
        out.width = msg.width
        out.fields = msg.fields
        out.is_bigendian = msg.is_bigendian
        out.point_step = msg.point_step
        out.row_step = msg.row_step
        out.is_dense = msg.is_dense
        out.data = bytes(data)

        self.pub.publish(out)


def main() -> None:
    """
    Standard ROS 2 node entry point.
    """
    rclpy.init()
    node = CloudYFlip()

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