#!/usr/bin/env python3

import math
import sys

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped

from tf2_ros import TransformBroadcaster

from unitree_sdk2py.core.channel import (
    ChannelSubscriber,
    ChannelFactoryInitialize,
)

from unitree_sdk2py.idl.unitree_go.msg.dds_ import (
    SportModeState_,
)


class G1OdomPublisher(Node):

    def __init__(self):
        super().__init__("g1_odom_publisher")

        # ============================================================
        # Parameters
        # ============================================================

        self.declare_parameter(
            "odom_topic",
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
            "publish_rate",
            50.0,
        )

        self.declare_parameter(
            "publish_tf",
            True,
        )

        self.declare_parameter(
            "zero_local_origin",
            True,
        )

        self.declare_parameter(
            "max_abs_position",
            1000.0,
        )

        self.declare_parameter(
            "min_quaternion_norm",
            0.5,
        )

        self.declare_parameter(
            "max_quaternion_norm",
            1.5,
        )

        self.odom_topic = (
            self.get_parameter(
                "odom_topic"
            ).value
        )

        self.odom_frame = (
            self.get_parameter(
                "odom_frame"
            ).value
        )

        self.base_frame = (
            self.get_parameter(
                "base_frame"
            ).value
        )

        self.publish_rate = float(
            self.get_parameter(
                "publish_rate"
            ).value
        )

        self.publish_tf = bool(
            self.get_parameter(
                "publish_tf"
            ).value
        )

        self.zero_local_origin = bool(
            self.get_parameter(
                "zero_local_origin"
            ).value
        )

        self.max_abs_position = float(
            self.get_parameter(
                "max_abs_position"
            ).value
        )

        self.min_quaternion_norm = float(
            self.get_parameter(
                "min_quaternion_norm"
            ).value
        )

        self.max_quaternion_norm = float(
            self.get_parameter(
                "max_quaternion_norm"
            ).value
        )

        # ============================================================
        # ROS interfaces
        # ============================================================

        self.odom_pub = self.create_publisher(
            Odometry,
            self.odom_topic,
            10,
        )

        self.tf_broadcaster = None

        if self.publish_tf:
            self.tf_broadcaster = (
                TransformBroadcaster(
                    self
                )
            )

        # ============================================================
        # Latest native Unitree state
        #
        # The DDS callback only stores the latest state.
        # The ROS timer publishes independently at 50 Hz.
        # ============================================================

        self.latest_state = None

        self.valid_msg_count = 0
        self.invalid_msg_count = 0

        # ============================================================
        # Local navigation origin
        # ============================================================

        self.origin_initialized = False

        self.origin_x = 0.0
        self.origin_y = 0.0
        self.origin_yaw = 0.0

        # ============================================================
        # Native Unitree DDS subscriber
        # ============================================================

        self.odom_state_sub = ChannelSubscriber(
            "rt/odommodestate",
            SportModeState_,
        )

        self.odom_state_sub.Init(
            self.odom_state_handler,
            10,
        )

        # ============================================================
        # Independent ROS publication timer
        # ============================================================

        safe_rate = max(
            self.publish_rate,
            1.0,
        )

        self.timer = self.create_timer(
            1.0 / safe_rate,
            self.publish_odom,
        )

        # ============================================================
        # Status timer
        # ============================================================

        self.status_timer = (
            self.create_timer(
                2.0,
                self.status_check,
            )
        )

        self.get_logger().info(
            "\n"
            "=========================================================\n"
            "       G1 DIRECT NATIVE ODOM PUBLISHER\n"
            "=========================================================\n"
            f" DDS input         : rt/odommodestate\n"
            f" ROS odom topic    : {self.odom_topic}\n"
            f" Odom frame        : {self.odom_frame}\n"
            f" Base frame        : {self.base_frame}\n"
            f" Publish rate      : {self.publish_rate:.1f} Hz\n"
            f" Publish TF        : {self.publish_tf}\n"
            f" Zero local origin : {self.zero_local_origin}\n"
            "========================================================="
        )

    # ==============================================================
    # Utilities
    # ==============================================================

    @staticmethod
    def normalize_angle(angle):
        return math.atan2(
            math.sin(angle),
            math.cos(angle),
        )

    @staticmethod
    def quaternion_to_yaw(
        qw,
        qx,
        qy,
        qz,
    ):
        siny = 2.0 * (
            qw * qz
            +
            qx * qy
        )

        cosy = 1.0 - 2.0 * (
            qy * qy
            +
            qz * qz
        )

        return math.atan2(
            siny,
            cosy,
        )

    @staticmethod
    def yaw_to_quaternion(yaw):
        half_yaw = (
            0.5 * yaw
        )

        return (
            0.0,
            0.0,
            math.sin(half_yaw),
            math.cos(half_yaw),
        )

    # ==============================================================
    # Validate native Unitree state
    # ==============================================================

    def state_is_valid(
        self,
        px,
        py,
        vx,
        vy,
        wz,
        qw,
        qx,
        qy,
        qz,
    ):
        values = (
            px,
            py,
            vx,
            vy,
            wz,
            qw,
            qx,
            qy,
            qz,
        )

        for value in values:
            if not math.isfinite(value):
                return False

        if (
            abs(px)
            > self.max_abs_position
        ):
            return False

        if (
            abs(py)
            > self.max_abs_position
        ):
            return False

        qnorm = math.sqrt(
            qw * qw
            +
            qx * qx
            +
            qy * qy
            +
            qz * qz
        )

        if (
            qnorm
            < self.min_quaternion_norm
        ):
            return False

        if (
            qnorm
            > self.max_quaternion_norm
        ):
            return False

        return True

    # ==============================================================
    # Native Unitree DDS callback
    #
    # Keep it minimal.
    # No ROS publication here.
    # ==============================================================

    def odom_state_handler(
        self,
        msg,
    ):
        try:
            px = float(
                msg.position[0]
            )

            py = float(
                msg.position[1]
            )

            vx = float(
                msg.velocity[0]
            )

            vy = float(
                msg.velocity[1]
            )

            wz = float(
                msg.yaw_speed
            )

            # Unitree quaternion:
            # [w, x, y, z]

            qw = float(
                msg.imu_state.quaternion[0]
            )

            qx = float(
                msg.imu_state.quaternion[1]
            )

            qy = float(
                msg.imu_state.quaternion[2]
            )

            qz = float(
                msg.imu_state.quaternion[3]
            )

            if not self.state_is_valid(
                px,
                py,
                vx,
                vy,
                wz,
                qw,
                qx,
                qy,
                qz,
            ):
                self.invalid_msg_count += 1
                return

            # Store only simple Python values.
            # Do not retain the native DDS object itself.

            self.latest_state = (
                px,
                py,
                vx,
                vy,
                wz,
                qw,
                qx,
                qy,
                qz,
            )

            self.valid_msg_count += 1

        except Exception as exc:
            self.invalid_msg_count += 1

            self.get_logger().warn(
                f"Invalid Unitree odometry sample: {exc}"
            )

    # ==============================================================
    # Publish navigation odometry
    # ==============================================================

    def publish_odom(self):

        if self.latest_state is None:
            return

        (
            raw_x,
            raw_y,
            vx,
            vy,
            wz,
            qw,
            qx,
            qy,
            qz,
        ) = self.latest_state

        # ============================================================
        # Raw yaw
        # ============================================================

        raw_yaw = (
            self.quaternion_to_yaw(
                qw,
                qx,
                qy,
                qz,
            )
        )

        # ============================================================
        # Initialize local navigation origin
        # ============================================================

        if (
            self.zero_local_origin
            and
            not self.origin_initialized
        ):
            self.origin_x = (
                raw_x
            )

            self.origin_y = (
                raw_y
            )

            self.origin_yaw = (
                raw_yaw
            )

            self.origin_initialized = (
                True
            )

            self.get_logger().info(
                "Local navigation origin set | "
                f"x={self.origin_x:+.6f} "
                f"y={self.origin_y:+.6f} "
                f"yaw={math.degrees(self.origin_yaw):+.3f} deg"
            )

        # ============================================================
        # Local planar pose
        # ============================================================

        x = raw_x
        y = raw_y
        yaw = raw_yaw

        if (
            self.zero_local_origin
            and
            self.origin_initialized
        ):
            dx = (
                raw_x
                -
                self.origin_x
            )

            dy = (
                raw_y
                -
                self.origin_y
            )

            c = math.cos(
                -self.origin_yaw
            )

            s = math.sin(
                -self.origin_yaw
            )

            x = (
                c * dx
                -
                s * dy
            )

            y = (
                s * dx
                +
                c * dy
            )

            yaw = (
                self.normalize_angle(
                    raw_yaw
                    -
                    self.origin_yaw
                )
            )

        # ============================================================
        # Final navigation timestamp
        #
        # One exact timestamp shared by:
        #
        #   /odom
        #   odom -> base_footprint
        # ============================================================

        now = (
            self.get_clock()
            .now()
            .to_msg()
        )

        # ============================================================
        # Planar orientation
        #
        # We intentionally publish only yaw for Nav2.
        # Torso roll/pitch must not tilt base_footprint.
        # ============================================================

        (
            local_qx,
            local_qy,
            local_qz,
            local_qw,
        ) = self.yaw_to_quaternion(
            yaw
        )

        # ============================================================
        # /odom
        # ============================================================

        odom = Odometry()

        odom.header.stamp = (
            now
        )

        odom.header.frame_id = (
            self.odom_frame
        )

        odom.child_frame_id = (
            self.base_frame
        )

        odom.pose.pose.position.x = (
            x
        )

        odom.pose.pose.position.y = (
            y
        )

        odom.pose.pose.position.z = (
            0.0
        )

        odom.pose.pose.orientation.x = (
            local_qx
        )

        odom.pose.pose.orientation.y = (
            local_qy
        )

        odom.pose.pose.orientation.z = (
            local_qz
        )

        odom.pose.pose.orientation.w = (
            local_qw
        )

        # ============================================================
        # Native Unitree twist
        # ============================================================

        odom.twist.twist.linear.x = (
            vx
        )

        odom.twist.twist.linear.y = (
            vy
        )

        odom.twist.twist.linear.z = (
            0.0
        )

        odom.twist.twist.angular.x = (
            0.0
        )

        odom.twist.twist.angular.y = (
            0.0
        )

        odom.twist.twist.angular.z = (
            wz
        )

        # ============================================================
        # Covariances
        # ============================================================

        odom.pose.covariance[0] = (
            0.01
        )

        odom.pose.covariance[7] = (
            0.01
        )

        odom.pose.covariance[14] = (
            1e3
        )

        odom.pose.covariance[21] = (
            1e3
        )

        odom.pose.covariance[28] = (
            1e3
        )

        odom.pose.covariance[35] = (
            0.05
        )

        odom.twist.covariance[0] = (
            0.01
        )

        odom.twist.covariance[7] = (
            0.01
        )

        odom.twist.covariance[14] = (
            1e3
        )

        odom.twist.covariance[21] = (
            1e3
        )

        odom.twist.covariance[28] = (
            1e3
        )

        odom.twist.covariance[35] = (
            0.05
        )

        # ============================================================
        # Publish /odom
        # ============================================================

        self.odom_pub.publish(
            odom
        )

        # ============================================================
        # Publish matching TF
        # ============================================================

        if (
            self.publish_tf
            and
            self.tf_broadcaster
            is not None
        ):
            tf_msg = (
                TransformStamped()
            )

            tf_msg.header.stamp = (
                now
            )

            tf_msg.header.frame_id = (
                self.odom_frame
            )

            tf_msg.child_frame_id = (
                self.base_frame
            )

            tf_msg.transform.translation.x = (
                x
            )

            tf_msg.transform.translation.y = (
                y
            )

            tf_msg.transform.translation.z = (
                0.0
            )

            tf_msg.transform.rotation.x = (
                local_qx
            )

            tf_msg.transform.rotation.y = (
                local_qy
            )

            tf_msg.transform.rotation.z = (
                local_qz
            )

            tf_msg.transform.rotation.w = (
                local_qw
            )

            self.tf_broadcaster.sendTransform(
                tf_msg
            )

    # ==============================================================
    # Diagnostics
    # ==============================================================

    def status_check(self):

        if self.latest_state is None:
            self.get_logger().warn(
                "No valid rt/odommodestate data received yet."
            )

            return

        self.get_logger().info(
            "Native odometry | "
            f"valid={self.valid_msg_count} | "
            f"invalid={self.invalid_msg_count}"
        )


def main(args=None):

    # ============================================================
    # Unitree network interface
    # ============================================================

    network_interface = (
        "eth0"
    )

    if (
        len(sys.argv)
        > 1
    ):
        network_interface = (
            sys.argv[1]
        )

    print(
        "[G1 Odom] "
        "Initializing Unitree DDS on interface: "
        f"{network_interface}"
    )

    # Must be initialized before creating ChannelSubscriber.
    ChannelFactoryInitialize(
        0,
        network_interface,
    )

    # ============================================================
    # ROS
    # ============================================================

    rclpy.init(
        args=args
    )

    node = (
        G1OdomPublisher()
    )

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