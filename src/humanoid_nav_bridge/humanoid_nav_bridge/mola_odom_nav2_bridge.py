#!/usr/bin/env python3

import copy

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster


class MolaOdomNav2Bridge(Node):

    def __init__(self):
        super().__init__("mola_odom_nav2_bridge")

        # ----------------------------------------------------------
        # Topics and frames
        # ----------------------------------------------------------
        self.declare_parameter(
            "input_topic",
            "/odom_mola_2d",
        )

        self.declare_parameter(
            "output_topic",
            "/odom",
        )

        self.declare_parameter(
            "odom_frame",
            "odom",
        )

        self.declare_parameter(
            "base_frame",
            "base_footprint",
        )

        self.declare_parameter(
            "publish_tf",
            False,
        )

        # ----------------------------------------------------------
        # Read parameters
        # ----------------------------------------------------------
        self.input_topic = str(
            self.get_parameter(
                "input_topic"
            ).value
        )

        self.output_topic = str(
            self.get_parameter(
                "output_topic"
            ).value
        )

        self.odom_frame = str(
            self.get_parameter(
                "odom_frame"
            ).value
        )

        self.base_frame = str(
            self.get_parameter(
                "base_frame"
            ).value
        )

        self.publish_tf = bool(
            self.get_parameter(
                "publish_tf"
            ).value
        )

        # ----------------------------------------------------------
        # Diagnostics
        # ----------------------------------------------------------
        self.last_callback_time = None

        # ----------------------------------------------------------
        # ROS interfaces
        # ----------------------------------------------------------
        self.odom_pub = self.create_publisher(
            Odometry,
            self.output_topic,
            10,
        )

        self.tf_broadcaster = TransformBroadcaster(
            self
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            self.input_topic,
            self.odom_callback,
            10,
        )

        self.get_logger().info(
            "\n"
            "=========================================================\n"
            "              MOLA ODOM NAV2 BRIDGE\n"
            "=========================================================\n"
            f" Input topic         : {self.input_topic}\n"
            f" Output topic        : {self.output_topic}\n"
            f" Publish TF          : {self.publish_tf}\n"
            f" TF                  : {self.odom_frame} -> {self.base_frame}\n"
            f" Publication mode    : direct from MOLA callback\n"
            f" Timestamp policy    : ROS now\n"
            "========================================================="
        )

    def odom_callback(
        self,
        msg: Odometry,
    ):

        now = self.get_clock().now()

        # ----------------------------------------------------------
        # Diagnose callback delivery gaps.
        # This tells us whether the MOLA odometry itself stops
        # reaching this bridge for an abnormal amount of time.
        # ----------------------------------------------------------
        if self.last_callback_time is not None:

            dt = (
                now
                - self.last_callback_time
            ).nanoseconds * 1e-9

            if dt > 0.2:
                self.get_logger().warn(
                    f"[MOLA CALLBACK GAP] "
                    f"dt={dt:.3f}s"
                )

        self.last_callback_time = now

        # ----------------------------------------------------------
        # Use one single ROS timestamp for both /odom and TF.
        # ----------------------------------------------------------
        stamp = now.to_msg()

        # ----------------------------------------------------------
        # Publish Nav2 odometry
        # ----------------------------------------------------------
        out = Odometry()

        out.header.stamp = stamp
        out.header.frame_id = self.odom_frame
        out.child_frame_id = self.base_frame

        out.pose = copy.deepcopy(
            msg.pose
        )

        out.twist = copy.deepcopy(
            msg.twist
        )

        self.odom_pub.publish(
            out
        )

        # ----------------------------------------------------------
        # Publish matching odom -> base_footprint TF
        # ----------------------------------------------------------
        if self.publish_tf:

            tf_msg = TransformStamped()

            tf_msg.header.stamp = stamp
            tf_msg.header.frame_id = (
                self.odom_frame
            )
            tf_msg.child_frame_id = (
                self.base_frame
            )

            tf_msg.transform.translation.x = (
                out.pose.pose.position.x
            )

            tf_msg.transform.translation.y = (
                out.pose.pose.position.y
            )

            tf_msg.transform.translation.z = (
                out.pose.pose.position.z
            )

            tf_msg.transform.rotation = (
                out.pose.pose.orientation
            )

            self.tf_broadcaster.sendTransform(
                tf_msg
            )


def main(args=None):

    rclpy.init(args=args)

    node = MolaOdomNav2Bridge()

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