#!/usr/bin/env python3

import rclpy
import math

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster


def yaw_from_quaternion(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def quaternion_from_yaw(yaw):
    qz = math.sin(yaw * 0.5)
    qw = math.cos(yaw * 0.5)
    return 0.0, 0.0, qz, qw


class MolaOdom2DBridge(Node):
    def __init__(self):
        super().__init__("mola_odom_2d_bridge")

        self.declare_parameter("input_odom_topic", "/lidar_odometry/pose")
        self.declare_parameter("output_odom_topic", "/mola_nav_odom")

        self.declare_parameter("output_frame_id", "mola_odom_test")
        self.declare_parameter("output_child_frame_id", "base_footprint_mola_test")

        self.declare_parameter("publish_tf", False)
        self.declare_parameter("force_2d", True)
        # Declare safety/filter parameters 
        self.declare_parameter("max_step_m", 0.25)
        self.declare_parameter("max_speed_mps", 1.2)
        self.declare_parameter("max_yaw_rate", 1.0)
        self.declare_parameter("compute_twist_from_pose", True)
        self.declare_parameter("reject_jumps", True)
        self.declare_parameter("hold_on_reject", True)
        self.declare_parameter("pose_cov_xy", 0.05)
        self.declare_parameter("pose_cov_yaw", 0.10)
        self.declare_parameter("twist_cov_vx", 0.10)
        self.declare_parameter("twist_cov_wz", 0.20)

        self.input_topic = self.get_parameter("input_odom_topic").value
        self.output_topic = self.get_parameter("output_odom_topic").value
        self.output_frame = self.get_parameter("output_frame_id").value
        self.output_child = self.get_parameter("output_child_frame_id").value
        self.publish_tf = bool(self.get_parameter("publish_tf").value)
        self.force_2d = bool(self.get_parameter("force_2d").value)

        self.pub = self.create_publisher(Odometry, self.output_topic, 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.max_step_m = float(self.get_parameter("max_step_m").value)
        self.max_speed_mps = float(self.get_parameter("max_speed_mps").value)
        self.max_yaw_rate = float(self.get_parameter("max_yaw_rate").value)

        self.compute_twist_from_pose = bool(self.get_parameter("compute_twist_from_pose").value)
        self.reject_jumps = bool(self.get_parameter("reject_jumps").value)
        self.hold_on_reject = bool(self.get_parameter("hold_on_reject").value)

        self.pose_cov_xy = float(self.get_parameter("pose_cov_xy").value)
        self.pose_cov_yaw = float(self.get_parameter("pose_cov_yaw").value)
        self.twist_cov_vx = float(self.get_parameter("twist_cov_vx").value)
        self.twist_cov_wz = float(self.get_parameter("twist_cov_wz").value)

        self.get_logger().info(
            f"MOLA 2D odom bridge: {self.input_topic} -> {self.output_topic}"
        )
        self.get_logger().info(
            f"TF test: {self.output_frame} -> {self.output_child}, publish_tf={self.publish_tf}"
        )
        self.origin_initialized = False
        self.x0 = 0.0
        self.y0 = 0.0
        self.yaw0 = 0.0

        self.sub = self.create_subscription(
            Odometry,
            self.input_topic,
            self.odom_callback,
            10,
        )

    def odom_callback(self, msg: Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation

        vx = 0.0
        vy = 0.0
        wz = 0.0

        yaw = yaw_from_quaternion(q)

        if not self.origin_initialized:
            self.x0 = p.x
            self.y0 = p.y
            self.yaw0 = yaw
            self.origin_initialized = True
            self.get_logger().info(
                f"MOLA origin initialized: "
                f"x0={self.x0:.3f}, y0={self.y0:.3f}, "
                f"yaw0={math.degrees(self.yaw0):.2f} deg"
            )

        dx = p.x - self.x0
        dy = p.y - self.y0

        c = math.cos(-self.yaw0)
        s = math.sin(-self.yaw0)

        x_rel = c * dx - s * dy
        y_rel = s * dx + c * dy
        yaw_rel = math.atan2(
            math.sin(yaw - self.yaw0),
            math.cos(yaw - self.yaw0)
        )

        if hasattr(self, "last_x"):
            dt = (msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9) - self.last_stamp

            dx = x_rel - self.last_x
            dy = y_rel - self.last_y
            dyaw = math.atan2(
                math.sin(yaw_rel - self.last_yaw),
                math.cos(yaw_rel - self.last_yaw)
            )

            step = math.hypot(dx, dy)
            speed = step / dt if dt > 1e-4 else 0.0
            yaw_rate = abs(dyaw / dt) if dt > 1e-4 else 0.0

            jump_detected = (
                step > self.max_step_m or
                speed > self.max_speed_mps or
                yaw_rate > self.max_yaw_rate
            )

            if self.reject_jumps and jump_detected:
                self.get_logger().warn(
                    f"MOLA jump rejected: step={step:.3f} m, "
                    f"speed={speed:.3f} m/s, yaw_rate={yaw_rate:.3f} rad/s"
                )

                if self.hold_on_reject:
                    x_rel = self.last_x
                    y_rel = self.last_y
                    yaw_rel = self.last_yaw
                    vx = 0.0
                    vy = 0.0
                    wz = 0.0
                else:
                    return
            else:
                vx = dx / dt if dt > 1e-4 else 0.0
                vy = dy / dt if dt > 1e-4 else 0.0
                wz = dyaw / dt if dt > 1e-4 else 0.0
        else:
            vx = 0.0
            vy = 0.0
            wz = 0.0

        qx, qy, qz, qw = quaternion_from_yaw(yaw_rel)

        out = Odometry()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = self.output_frame
        out.child_frame_id = self.output_child

        out.pose.pose.position.x = x_rel
        out.pose.pose.position.y = y_rel
        out.pose.pose.position.z = 0.0 if self.force_2d else p.z

        out.pose.pose.orientation.x = qx
        out.pose.pose.orientation.y = qy
        out.pose.pose.orientation.z = qz
        out.pose.pose.orientation.w = qw

        #out.twist = msg.twist

        out.twist.twist.linear.x = vx
        out.twist.twist.linear.y = vy
        out.twist.twist.linear.z = 0.0
        out.twist.twist.angular.x = 0.0
        out.twist.twist.angular.y = 0.0
        out.twist.twist.angular.z = wz

        out.pose.covariance = list(msg.pose.covariance)
        out.twist.covariance = list(msg.twist.covariance)

        # Make Z, roll, pitch very uncertain for Nav2-style 2D use
        out.pose.covariance[14] = 999.0   # z
        out.pose.covariance[21] = 999.0   # roll
        out.pose.covariance[28] = 999.0   # pitch

        self.pub.publish(out)

        if self.publish_tf:
            tf = TransformStamped()
            tf.header.stamp = out.header.stamp
            tf.header.frame_id = self.output_frame
            tf.child_frame_id = self.output_child

            tf.transform.translation.x = out.pose.pose.position.x
            tf.transform.translation.y = out.pose.pose.position.y
            tf.transform.translation.z = out.pose.pose.position.z
            tf.transform.rotation = out.pose.pose.orientation

            self.tf_broadcaster.sendTransform(tf)
    
        self.last_x = x_rel
        self.last_y = y_rel
        self.last_yaw = yaw_rel
        self.last_stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

def main(args=None):
    rclpy.init(args=args)
    node = MolaOdom2DBridge()
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