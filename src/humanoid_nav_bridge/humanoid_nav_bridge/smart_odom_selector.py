#!/usr/bin/env python3
"""
smart_odom_selector.py

Select the best available odometry source for Nav2 and publish a stable:

    /odom
    TF: odom -> base_footprint

Priority policy:

    1) FAST-LIO odometry       /odom_fastlio_2d
    2) Artificial odometry     /odom_artificial
    3) dog_odom fallback       /repub/dog_odom

Why this node exists:
---------------------
robot_localization EKF is not a real priority/fallback switch. If FAST-LIO stops,
EKF may continue predicting for a while, but it does not automatically replace
FAST-LIO pose with dog_odom pose in the way we need.

This node explicitly manages source priority and fallback.

Main idea:
----------
- Use FAST-LIO when it is recent and valid.
- If FAST-LIO is lost, use artificial odom.
- If artificial odom is also not usable, use dog_odom.
- When switching sources, apply an offset so Nav2 does not see a sudden jump.
"""


from ament_index_python import resources
import math
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.time import Time

from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy


def yaw_from_quat(q) -> float:
    """Extract planar yaw from quaternion."""
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


def quat_from_yaw(yaw: float):
    """Create quaternion from planar yaw."""
    return 0.0, 0.0, math.sin(yaw * 0.5), math.cos(yaw * 0.5)


def normalize_angle(a: float) -> float:
    """Normalize angle to [-pi, pi]."""
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


class SmartOdomSelector(Node):
    def __init__(self):
        super().__init__("smart_odom_selector")

        # ============================================================
        # Frames
        # ============================================================
        self.odom_frame = self.declare_parameter(
            "odom_frame", "odom"
        ).value

        self.base_frame = self.declare_parameter(
            "base_frame", "base_footprint"
        ).value

        # ============================================================
        # Input topics
        # ============================================================
        self.fastlio_topic = self.declare_parameter(
            "fastlio_topic", "/odom_fastlio_2d"
        ).value

        self.artificial_topic = self.declare_parameter(
            "artificial_topic", "/odom_artificial"
        ).value

        self.dog_topic = self.declare_parameter(
            "dog_topic", "/repub/dog_odom"
        ).value

        # ============================================================
        # Output topic
        # ============================================================
        self.output_odom_topic = self.declare_parameter(
            "output_odom_topic", "/odom"
        ).value

        # ============================================================
        # Timeouts
        # ============================================================
        self.fastlio_timeout = float(
            self.declare_parameter("fastlio_timeout", 1.0).value
        )

        self.artificial_timeout = float(
            self.declare_parameter("artificial_timeout", 0.7).value
        )

        self.dog_timeout = float(
            self.declare_parameter("dog_timeout", 1.0).value
        )

        # ============================================================
        # Validation thresholds
        # ============================================================
        self.max_abs_xy = float(
            self.declare_parameter("max_abs_xy", 80.0).value
        )

        self.max_jump_xy = float(
            self.declare_parameter("max_jump_xy", 2.0).value
        )
        self.max_jump_yaw = float(
            self.declare_parameter("max_jump_yaw", 0.50).value
        )

        # Number of valid FAST-LIO samples required before switching
        # back to FAST-LIO after a fallback.
        self.fastlio_recovery_samples = int(
            self.declare_parameter("fastlio_recovery_samples", 3).value
        )

        self.publish_tf_enabled = bool(
            self.declare_parameter("publish_tf", True).value
        )

        self.output_rate_hz = float(
            self.declare_parameter("output_rate_hz", 30.0).value
        )

        # ============================================================
        # Latest messages
        # ============================================================
        self.fastlio_msg: Optional[Odometry] = None
        self.artificial_msg: Optional[Odometry] = None
        self.dog_msg: Optional[Odometry] = None

        self.fastlio_time: Optional[Time] = None
        self.artificial_time: Optional[Time] = None
        self.dog_time: Optional[Time] = None

        # ============================================================
        # Source ownership state
        # ============================================================
        self.current_source: Optional[str] = None

        # Last output pose published to Nav2.
        # Used to avoid jumps when switching sources.
        self.last_out_x = 0.0
        self.last_out_y = 0.0
        self.last_out_yaw = 0.0
        self.has_last_output = False

        # Offset applied to the currently selected source.
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.offset_yaw = 0.0

        # FAST-LIO recovery counter.
        self.fastlio_valid_count = 0

        # ============================================================
        # ROS interfaces
        # ============================================================
        self.odom_pub = self.create_publisher(
            Odometry,
            self.output_odom_topic,
            20,
        )

        self.tf_broadcaster = TransformBroadcaster(self)

        self.create_subscription(
            Odometry,
            self.fastlio_topic,
            self.cb_fastlio,
            20,
        )

        self.create_subscription(
            Odometry,
            self.artificial_topic,
            self.cb_artificial,
            20,
        )

        dog_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.create_subscription(
            Odometry, 
            self.dog_topic, 
            self.cb_dog, 
            dog_qos
        )

        self.create_timer(
            1.0 / self.output_rate_hz,
            self.tick,
        )

        self.last_source_log_time = self.get_clock().now()
        self.source_log_period = 5.0
        self.last_logged_source = None

        self.get_logger().info(
            "SmartOdomSelector started:\n"
            f"  priority 1 : {self.fastlio_topic}\n"
            f"  priority 2 : {self.artificial_topic}\n"
            f"  priority 3 : {self.dog_topic}\n"
            f"  output     : {self.output_odom_topic}\n"
            f"  TF         : {'ON' if self.publish_tf_enabled else 'OFF'} "
            f"({self.odom_frame}->{self.base_frame})"
        )

    # ============================================================
    # Callbacks
    # ============================================================
    def cb_fastlio(self, msg: Odometry):
        self.fastlio_msg = msg
        self.fastlio_time = self.get_clock().now()

    def cb_artificial(self, msg: Odometry):
        self.artificial_msg = msg
        self.artificial_time = self.get_clock().now()

    def cb_dog(self, msg: Odometry):
        self.dog_msg = msg
        self.dog_time = self.get_clock().now()

    # ============================================================
    # Validation helpers
    # ============================================================
    def recent(self, stamp: Optional[Time], timeout: float) -> bool:
        if stamp is None:
            return False

        age = (self.get_clock().now() - stamp).nanoseconds * 1e-9
        return age <= timeout

    def valid_pose(self, msg: Odometry) -> bool:
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation

        values = [
            p.x, p.y, p.z,
            q.x, q.y, q.z, q.w,
        ]

        if not all(math.isfinite(v) for v in values):
            return False

        if abs(p.x) > self.max_abs_xy or abs(p.y) > self.max_abs_xy:
            return False

        q_norm = math.sqrt(
            q.x * q.x +
            q.y * q.y +
            q.z * q.z +
            q.w * q.w
        )

        if q_norm < 0.5 or q_norm > 1.5:
            return False

        return True

    def source_is_valid(self, name: str) -> bool:
        if name == "fastlio":
            return (
                self.fastlio_msg is not None
                and self.recent(self.fastlio_time, self.fastlio_timeout)
                and self.valid_pose(self.fastlio_msg)
            )

        if name == "artificial":
            return (
                self.artificial_msg is not None
                and self.recent(self.artificial_time, self.artificial_timeout)
                and self.valid_pose(self.artificial_msg)
            )

        if name == "dog":
            return (
                self.dog_msg is not None
                and self.recent(self.dog_time, self.dog_timeout)
                and self.valid_pose(self.dog_msg)
            )

        return False

    def get_msg(self, name: str) -> Optional[Odometry]:
        if name == "fastlio":
            return self.fastlio_msg
        if name == "artificial":
            return self.artificial_msg
        if name == "dog":
            return self.dog_msg
        return None

    # ============================================================
    # Source selection policy
    # ============================================================
    def choose_source(self) -> Tuple[Optional[str], Optional[Odometry]]:
        """
        Priority logic:

            FAST-LIO > artificial odom > dog_odom

        FAST-LIO recovery is protected by a small validation counter.
        This avoids switching back to FAST-LIO after only one isolated sample.
        """

        fastlio_ok = self.source_is_valid("fastlio")
        artificial_ok = self.source_is_valid("artificial")
        dog_ok = self.source_is_valid("dog")

        if fastlio_ok:
            self.fastlio_valid_count += 1
        else:
            self.fastlio_valid_count = 0

        # If we are already using FAST-LIO, keep it as long as valid.
        if self.current_source == "fastlio" and fastlio_ok:
            return "fastlio", self.fastlio_msg

        # If FAST-LIO was lost and comes back, require several valid samples.
        if fastlio_ok and self.fastlio_valid_count >= self.fastlio_recovery_samples:
            return "fastlio", self.fastlio_msg

        # Second priority: artificial odom.
        if artificial_ok:
            return "artificial", self.artificial_msg

        # Third priority: dog_odom.
        if dog_ok:
            return "dog", self.dog_msg

        return None, None

    # ============================================================
    # Continuity logic
    # ============================================================
    def compute_switch_offset(self, new_source: str, msg: Odometry):
        """
        Compute offset when switching source.

        Without this, switching from FAST-LIO to dog_odom could create
        a sudden jump because their coordinate origins are different.

        We force:

            new_output_pose == last_output_pose

        at the instant of switching.
        """

        src_x = msg.pose.pose.position.x
        src_y = msg.pose.pose.position.y
        src_yaw = yaw_from_quat(msg.pose.pose.orientation)

        if not self.has_last_output:
            self.offset_x = -src_x
            self.offset_y = -src_y
            self.offset_yaw = -src_yaw
            return

        self.offset_x = self.last_out_x - src_x
        self.offset_y = self.last_out_y - src_y
        self.offset_yaw = normalize_angle(self.last_out_yaw - src_yaw)

        self.get_logger().warn(
            f"Odometry source switch: {self.current_source} -> {new_source} | "
            f"offset x={self.offset_x:.3f}, y={self.offset_y:.3f}, "
            f"yaw={self.offset_yaw:.3f}"
        )

    def apply_offset(self, src: Odometry) -> Odometry:
        """
        Convert selected source odometry into the common output frame:

            header.frame_id = odom
            child_frame_id  = base_footprint

        and apply the current continuity offset.
        """

        now = self.get_clock().now().to_msg()

        src_x = src.pose.pose.position.x
        src_y = src.pose.pose.position.y
        src_yaw = yaw_from_quat(src.pose.pose.orientation)

        out_x = src_x + self.offset_x
        out_y = src_y + self.offset_y
        out_yaw = normalize_angle(src_yaw + self.offset_yaw)

        qx, qy, qz, qw = quat_from_yaw(out_yaw)

        out = Odometry()
        out.header.stamp = now
        out.header.frame_id = self.odom_frame
        out.child_frame_id = self.base_frame

        out.pose.pose.position.x = out_x
        out.pose.pose.position.y = out_y
        out.pose.pose.position.z = 0.0

        out.pose.pose.orientation.x = qx
        out.pose.pose.orientation.y = qy
        out.pose.pose.orientation.z = qz
        out.pose.pose.orientation.w = qw

        out.twist.twist.linear.x = src.twist.twist.linear.x
        out.twist.twist.linear.y = 0.0
        out.twist.twist.linear.z = 0.0

        out.twist.twist.angular.x = 0.0
        out.twist.twist.angular.y = 0.0
        out.twist.twist.angular.z = src.twist.twist.angular.z

        out.pose.covariance = src.pose.covariance
        out.twist.covariance = src.twist.covariance

        return out

    def jump_is_acceptable(self, odom: Odometry) -> bool:
        """
        Reject sudden output jumps.

        This is a final safety layer before publishing TF.
        """

        if not self.has_last_output:
            return True

        dx = odom.pose.pose.position.x - self.last_out_x
        dy = odom.pose.pose.position.y - self.last_out_y
        dist = math.sqrt(dx * dx + dy * dy)

        yaw = yaw_from_quat(odom.pose.pose.orientation)
        dyaw = abs(normalize_angle(yaw - self.last_out_yaw))

        if dist > self.max_jump_xy:
            self.get_logger().error(
                f"Rejected XY odom jump: dist={dist:.3f} m"
            )
            return False

        if dyaw > self.max_jump_yaw:
            self.get_logger().error(
                f"Rejected YAW odom jump: dyaw={dyaw:.3f} rad"
            )
            return False

        return True

    # ============================================================
    # Publishing
    # ============================================================
    def publish_tf(self, odom: Odometry):
        tfm = TransformStamped()

        tfm.header.stamp = odom.header.stamp
        tfm.header.frame_id = self.odom_frame
        tfm.child_frame_id = self.base_frame

        tfm.transform.translation.x = odom.pose.pose.position.x
        tfm.transform.translation.y = odom.pose.pose.position.y
        tfm.transform.translation.z = 0.0
        tfm.transform.rotation = odom.pose.pose.orientation

        self.tf_broadcaster.sendTransform(tfm)

    def tick(self):
        source, msg = self.choose_source()
        now = self.get_clock().now()

        if msg is None or source is None:
            self.get_logger().warn(
                "No valid odometry source available. TF odom->base_footprint not updated.",
            )
            return

        # Detect source switch.
        if source != self.current_source:
            self.compute_switch_offset(source, msg)
            self.current_source = source

        source_changed = (
            source != self.last_logged_source
        )

        dt = (
            now - self.last_source_log_time
        ).nanoseconds * 1e-9

        odom = self.apply_offset(msg)

        if not self.valid_pose(odom):
            self.get_logger().error(
                f"Rejected invalid output odom from source: {source}"
            )
            return
    
        if not self.jump_is_acceptable(odom):
            self.get_logger().error(
                f"Rejected odom jump from source {source}: "
                f"x={odom.pose.pose.position.x:.2f}, "
                f"y={odom.pose.pose.position.y:.2f}"
            )
            return

        self.odom_pub.publish(odom)

        if self.publish_tf_enabled:
            self.publish_tf(odom)

        self.last_out_x = odom.pose.pose.position.x
        self.last_out_y = odom.pose.pose.position.y
        self.last_out_yaw = yaw_from_quat(odom.pose.pose.orientation)
        self.has_last_output = True

        if source_changed or dt >= self.source_log_period:
            self.get_logger().info(
                f"Using odom source: {source}"
            )

            self.last_source_log_time = now
            self.last_logged_source = source
    


def main(args=None):
    rclpy.init(args=args)

    node = SmartOdomSelector()

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