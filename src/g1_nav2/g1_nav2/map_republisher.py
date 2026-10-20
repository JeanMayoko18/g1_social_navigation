#!/usr/bin/env python3

import rclpy

from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)


class MapRepublisher(Node):
    def __init__(self):
        super().__init__("map_republisher")

        self.declare_parameter(
            "input_topic",
            "/map",
        )

        self.declare_parameter(
            "output_topic",
            "/map_runtime",
        )

        input_topic = (
            self.get_parameter("input_topic")
            .get_parameter_value()
            .string_value
        )

        output_topic = (
            self.get_parameter("output_topic")
            .get_parameter_value()
            .string_value
        )

        if input_topic == output_topic:
            raise RuntimeError(
                "input_topic and output_topic must be different."
            )

        qos_in = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        qos_out = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.pub = self.create_publisher(
            OccupancyGrid,
            output_topic,
            qos_out,
        )

        self.sub = self.create_subscription(
            OccupancyGrid,
            input_topic,
            self.map_callback,
            qos_in,
        )

        self.map_count = 0

        self.get_logger().info(
            f"Map republisher ready: "
            f"{input_topic} -> {output_topic}"
        )

    def map_callback(self, msg):
        self.map_count += 1
        self.pub.publish(msg)

        self.get_logger().info(
            "Received /map and published /map_runtime "
            f"[update={self.map_count}, "
            f"size={msg.info.width}x{msg.info.height}, "
            f"resolution={msg.info.resolution:.3f}]"
        )


def main(args=None):
    rclpy.init(args=args)

    node = MapRepublisher()

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