#!/usr/bin/env python3
import math
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped, Quaternion
from tf2_ros import TransformBroadcaster


def make_qos(reliable: bool, depth: int) -> QoSProfile:
    return QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE if reliable else ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
        depth=depth,
    )


def yaw_from_quat(q: Quaternion) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def quat_from_yaw(yaw: float) -> Quaternion:
    q = Quaternion()
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(yaw * 0.5)
    q.w = math.cos(yaw * 0.5)
    return q


def wrap_angle(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def lerp_angle(a: float, b: float, alpha: float) -> float:
    return wrap_angle(a + alpha * wrap_angle(b - a))


class SmartOdomFusion(Node):
    def __init__(self):
        super().__init__("smart_odom_fusion")

        # Topics
        self.declare_parameter("base_odom_topic", "/dog_odom_filtered")
        self.declare_parameter("lidar_odom_topic", "/odom_fastlio_2d")
        self.declare_parameter("output_odom_topic", "/odom")

        # Frames
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("base_frame", "base_footprint")
        self.declare_parameter("publish_tf", True)
        self.declare_parameter("tf_future_offset_sec", 0.05)

        # QoS
        self.declare_parameter("base_odom_reliable", False)
        self.declare_parameter("lidar_odom_reliable", False)
        self.declare_parameter("output_reliable", False)
        self.declare_parameter("qos_depth", 10)

        # Fusion parameters
        self.declare_parameter("lidar_timeout_sec", 0.7)
        self.declare_parameter("base_timeout_sec", 0.7)

        # Correction rate: small = smooth, large = aggressive.
        self.declare_parameter("offset_alpha_xy", 0.02)
        self.declare_parameter("offset_alpha_yaw", 0.02)

        # Reject lidar corrections if too large.
        self.declare_parameter("max_lidar_offset_xy", 2.0)
        self.declare_parameter("max_lidar_offset_yaw", 1.2)

        # Reject sudden lidar jumps between consecutive lidar odom messages.
        self.declare_parameter("max_lidar_step_m", 0.50)
        self.declare_parameter("max_lidar_yaw_step", 0.50)

        self.base_topic = self.get_parameter("base_odom_topic").value
        self.lidar_topic = self.get_parameter("lidar_odom_topic").value
        self.output_topic = self.get_parameter("output_odom_topic").value

        self.odom_frame = self.get_parameter("odom_frame").value
        self.base_frame = self.get_parameter("base_frame").value
        self.publish_tf = bool(self.get_parameter("publish_tf").value)
        self.tf_future_offset_sec = float(self.get_parameter("tf_future_offset_sec").value)

        self.qos_depth = int(self.get_parameter("qos_depth").value)

        self.base_timeout_sec = float(self.get_parameter("base_timeout_sec").value)
        self.lidar_timeout_sec = float(self.get_parameter("lidar_timeout_sec").value)

        self.offset_alpha_xy = float(self.get_parameter("offset_alpha_xy").value)
        self.offset_alpha_yaw = float(self.get_parameter("offset_alpha_yaw").value)

        self.max_lidar_offset_xy = float(self.get_parameter("max_lidar_offset_xy").value)
        self.max_lidar_offset_yaw = float(self.get_parameter("max_lidar_offset_yaw").value)

        self.max_lidar_step_m = float(self.get_parameter("max_lidar_step_m").value)
        self.max_lidar_yaw_step = float(self.get_parameter("max_lidar_yaw_step").value)

        base_qos = make_qos(bool(self.get_parameter("base_odom_reliable").value), self.qos_depth)
        lidar_qos = make_qos(bool(self.get_parameter("lidar_odom_reliable").value), self.qos_depth)
        output_qos = make_qos(bool(self.get_parameter("output_reliable").value), self.qos_depth)

        self.sub_base = self.create_subscription(Odometry, self.base_topic, self.cb_base, base_qos)
        self.sub_lidar = self.create_subscription(Odometry, self.lidar_topic, self.cb_lidar, lidar_qos)

        self.pub_odom = self.create_publisher(Odometry, self.output_topic, output_qos)
        self.tf_broadcaster = TransformBroadcaster(self) if self.publish_tf else None

        self.last_base: Optional[Odometry] = None
        self.last_lidar: Optional[Odometry] = None

        self.last_base_time = None
        self.last_lidar_time = None

        self.last_lidar_x = None
        self.last_lidar_y = None
        self.last_lidar_yaw = None

        self.offset_initialized = False
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.offset_yaw = 0.0

        self.timer = self.create_timer(1.0 / 30.0, self.publish_fused)

        self.get_logger().info("SmartOdomFusion started")
        self.get_logger().info(f" base odom : {self.base_topic}")
        self.get_logger().info(f" lidar odom: {self.lidar_topic}")
        self.get_logger().info(f" output    : {self.output_topic}")
        self.get_logger().info("Mode: dog/base odom continuity + slow FAST-LIO correction")

    def cb_base(self, msg: Odometry):
        self.last_base = msg
        self.last_base_time = self.get_clock().now()

    def cb_lidar(self, msg: Odometry):
        now = self.get_clock().now()

        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        yaw = yaw_from_quat(msg.pose.pose.orientation)

        if self.last_lidar_x is not None:
            step = math.hypot(x - self.last_lidar_x, y - self.last_lidar_y)
            dyaw = abs(wrap_angle(yaw - self.last_lidar_yaw))

            if step > self.max_lidar_step_m or dyaw > self.max_lidar_yaw_step:
                self.get_logger().warn(
                    f"FAST-LIO correction ignored: step={step:.3f}, dyaw={math.degrees(dyaw):.2f} deg"
                )
                return

        self.last_lidar = msg
        self.last_lidar_time = now
        self.last_lidar_x = x
        self.last_lidar_y = y
        self.last_lidar_yaw = yaw

    def publish_fused(self):
        now = self.get_clock().now()

        if self.last_base is None or self.last_base_time is None:
            return

        base_age = (now - self.last_base_time).nanoseconds * 1e-9
        if base_age > self.base_timeout_sec:
            self.get_logger().warn(f"Base odom timeout: {base_age:.2f}s")
            return

        base = self.last_base

        bx = base.pose.pose.position.x
        by = base.pose.pose.position.y
        bz = base.pose.pose.position.z
        byaw = yaw_from_quat(base.pose.pose.orientation)

        lidar_valid = False

        if self.last_lidar is not None and self.last_lidar_time is not None:
            lidar_age = (now - self.last_lidar_time).nanoseconds * 1e-9
            lidar_valid = lidar_age <= self.lidar_timeout_sec

        if lidar_valid:
            lx = self.last_lidar.pose.pose.position.x
            ly = self.last_lidar.pose.pose.position.y
            lyaw = yaw_from_quat(self.last_lidar.pose.pose.orientation)

            target_offset_x = lx - bx
            target_offset_y = ly - by
            target_offset_yaw = wrap_angle(lyaw - byaw)

            offset_norm = math.hypot(target_offset_x, target_offset_y)
            offset_yaw_abs = abs(target_offset_yaw)

            if offset_norm <= self.max_lidar_offset_xy and offset_yaw_abs <= self.max_lidar_offset_yaw:
                if not self.offset_initialized:
                    self.offset_x = target_offset_x
                    self.offset_y = target_offset_y
                    self.offset_yaw = target_offset_yaw
                    self.offset_initialized = True
                    self.get_logger().info(
                        f"FAST-LIO correction initialized: "
                        f"dx={self.offset_x:.3f}, dy={self.offset_y:.3f}, "
                        f"dyaw={math.degrees(self.offset_yaw):.2f} deg"
                    )
                else:
                    self.offset_x = (1.0 - self.offset_alpha_xy) * self.offset_x + self.offset_alpha_xy * target_offset_x
                    self.offset_y = (1.0 - self.offset_alpha_xy) * self.offset_y + self.offset_alpha_xy * target_offset_y
                    self.offset_yaw = lerp_angle(self.offset_yaw, target_offset_yaw, self.offset_alpha_yaw)
            else:
                self.get_logger().warn(
                    f"FAST-LIO offset ignored: norm={offset_norm:.3f}, "
                    f"yaw={math.degrees(offset_yaw_abs):.2f} deg"
                )

        if not self.offset_initialized:
            fx = bx
            fy = by
            fyaw = byaw
        else:
            fx = bx + self.offset_x
            fy = by + self.offset_y
            fyaw = wrap_angle(byaw + self.offset_yaw)

        out = Odometry()
        out.header.stamp = now.to_msg()
        out.header.frame_id = self.odom_frame
        out.child_frame_id = self.base_frame

        out.pose.pose.position.x = fx
        out.pose.pose.position.y = fy
        out.pose.pose.position.z = bz
        out.pose.pose.orientation = quat_from_yaw(fyaw)

        out.pose.covariance = base.pose.covariance
        out.twist = base.twist

        self.pub_odom.publish(out)

        if self.publish_tf:
            self.publish_tf_msg(out)

    def publish_tf_msg(self, odom: Odometry):
        t = TransformStamped()
        t.header.stamp = (
            self.get_clock().now() + Duration(seconds=self.tf_future_offset_sec)
        ).to_msg()
        t.header.frame_id = self.odom_frame
        t.child_frame_id = self.base_frame

        t.transform.translation.x = odom.pose.pose.position.x
        t.transform.translation.y = odom.pose.pose.position.y
        t.transform.translation.z = odom.pose.pose.position.z
        t.transform.rotation = odom.pose.pose.orientation

        self.tf_broadcaster.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    node = SmartOdomFusion()
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