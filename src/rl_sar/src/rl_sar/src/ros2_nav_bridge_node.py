#!/usr/bin/env python3

import socket

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


class Ros2NavBridgeNode(Node):
    def __init__(self):
        super().__init__("ros2_nav_bridge_node")

        self.cmd_vel_topic = self.declare_parameter(
            "cmd_vel_topic",
            "/cmd_vel_rlsar"
        ).value

        self.udp_host = self.declare_parameter(
            "udp_host",
            "127.0.0.1"
        ).value

        self.udp_port = self.declare_parameter(
            "udp_port",
            15000
        ).value

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self.sub = self.create_subscription(
            Twist,
            self.cmd_vel_topic,
            self.cmd_callback,
            10
        )

        self.get_logger().info(
            f"ROS2 nav bridge started: {self.cmd_vel_topic} "
            f"-> UDP {self.udp_host}:{self.udp_port}"
        )

    def cmd_callback(self, msg: Twist):
        vx = msg.linear.x
        vy = msg.linear.y
        wz = msg.angular.z

        data = f"{vx:.6f} {vy:.6f} {wz:.6f}\n".encode("utf-8")

        self.sock.sendto(data, (self.udp_host, int(self.udp_port)))

        self.get_logger().info(
            f"[UDP CMD] vx={vx:.3f} vy={vy:.3f} wz={wz:.3f}",
            throttle_duration_sec=0.3
        )


def main(args=None):
    rclpy.init(args=args)
    node = Ros2NavBridgeNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.sock.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()