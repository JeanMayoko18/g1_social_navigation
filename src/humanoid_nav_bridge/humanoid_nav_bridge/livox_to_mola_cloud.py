#!/usr/bin/env python3
"""
livox_to_mola_cloud.py

Convert Livox MID-360 PointCloud2 format:
    x, y, z, intensity, ring, time
    point_step = 22

into a standard PointCloud2 format for MOLA:
    x, y, z, intensity
    point_step = 16
"""

import math
import struct

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import PointCloud2, PointField


class LivoxToMolaCloud(Node):
    def __init__(self):
        super().__init__("livox_to_mola_cloud")

        self.declare_parameter("input_cloud_topic", "/utlidar/cloud_livox_mid360")
        self.declare_parameter("output_cloud_topic", "/mola/cloud_xyz_i")
        self.declare_parameter("output_frame_id", "mola_livox_frame")
        self.declare_parameter("max_points", 0)
        self.declare_parameter("skip_n", 1)

        self.input_topic = self.get_parameter("input_cloud_topic").value
        self.output_topic = self.get_parameter("output_cloud_topic").value
        self.output_frame_id = self.get_parameter("output_frame_id").value
        self.max_points = int(self.get_parameter("max_points").value)
        self.skip_n = max(1, int(self.get_parameter("skip_n").value))

        qos_sub = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )

        qos_pub = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )

        self.sub = self.create_subscription(
            PointCloud2,
            self.input_topic,
            self.cloud_callback,
            qos_sub,
        )

        self.pub = self.create_publisher(
            PointCloud2,
            self.output_topic,
            qos_pub,
        )

        self.get_logger().info(
            f"Livox -> MOLA cloud converter: {self.input_topic} -> {self.output_topic}"
        )

    def cloud_callback(self, msg: PointCloud2):
        field_offsets = {f.name: f.offset for f in msg.fields}

        required = ["x", "y", "z"]
        for name in required:
            if name not in field_offsets:
                self.get_logger().error(f"Missing required field: {name}")
                return

        x_off = field_offsets["x"]
        y_off = field_offsets["y"]
        z_off = field_offsets["z"]
        i_off = field_offsets.get("intensity", None)

        point_step = msg.point_step
        total_points = msg.width * msg.height

        if total_points <= 0 or point_step <= 0:
            return

        out_data = bytearray()
        count = 0

        raw = msg.data

        for idx in range(0, total_points, self.skip_n):
            if self.max_points > 0 and count >= self.max_points:
                break

            base = idx * point_step

            try:
                x = struct.unpack_from("<f", raw, base + x_off)[0]
                y = struct.unpack_from("<f", raw, base + y_off)[0]
                z = struct.unpack_from("<f", raw, base + z_off)[0]

                if i_off is not None:
                    intensity = struct.unpack_from("<f", raw, base + i_off)[0]
                else:
                    intensity = 0.0

            except struct.error:
                continue

            if not (
                math.isfinite(x)
                and math.isfinite(y)
                and math.isfinite(z)
                and math.isfinite(intensity)
            ):
                continue

            out_data.extend(struct.pack("<ffff", x, y, z, intensity))
            #out_data.extend(struct.pack("<ffff", x, y, z))
            count += 1

        out = PointCloud2()
        out.header = msg.header

        if self.output_frame_id:
            out.header.frame_id = self.output_frame_id

        out.height = 1
        out.width = count
        out.is_bigendian = False
        out.is_dense = True
        out.point_step = 16
        #out.point_step = 12
        out.row_step = out.point_step * count
        out.data = bytes(out_data)


        out.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            #PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = LivoxToMolaCloud()

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