#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

class ParamNode(Node):
    def __init__(self):
        super().__init__("param_node")
        self.declare_parameter("robot_name", "g1")
        self.declare_parameter("gazebo_model_name", "g1")

        self.get_logger().info(
            f"param_node ready: robot_name={self.get_parameter('robot_name').value} "
            f"gazebo_model_name={self.get_parameter('gazebo_model_name').value}"
        )

def main():
    rclpy.init()
    node = ParamNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
