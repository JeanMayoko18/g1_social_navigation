# File: humanoid_nav_bridge/real_odommodestate_to_odom.py
#!/usr/bin/env python3
"""
odommodestate_to_odom.py  (ROS 2 Humble)

Convert Unitree internal state topic:
  /odommodestate   (unitree_go/msg/SportModeState)

into a standard ROS odometry:
  /odommodestate_odom   (nav_msgs/msg/Odometry)

Why this exists
---------------
Nav2 / SLAM / robot_localization and most ROS tooling expect nav_msgs/Odometry.
Unitree publishes a rich internal state (SportModeState) that *contains* everything
we need for planar navigation (x,y,yaw, vx,vy,wz), but it is not Odometry.

This node is intentionally simple:
- no integration
- no gating
- no TF publishing
It only repackages fields into nav_msgs/Odometry, so other blocks can consume it.

Key fields mapping 
-----------------------------------
SportModeState.position[0], [1], [2]      -> pose.position (x,y,z)
SportModeState.imu_state.quaternion[4]    -> pose.orientation
SportModeState.velocity[0], [1], [2]      -> twist.linear (x,y,z)
SportModeState.yaw_speed                  -> twist.angular.z

IMPORTANT: quaternion order
---------------------------
Unitree arrays are often [w, x, y, z].
The echo looks like:
  quaternion:
  - 0.99987
  - -0.0036
  - 0.0127
  - 0.0088
=> This strongly suggests [w, x, y, z].

We provide a parameter "quat_order" so the switch can be done if needed:
  quat_order: "wxyz" (default) or "xyzw"

QoS note
--------
The `ros2 topic info /odommodestate -v` shows:
- publisher RELIABLE
- some subscriber BEST_EFFORT

For robustness with bare DDS apps, we subscribe BEST_EFFORT by default.
(Can be changed if needed.)

Run
Better run it through the pipeline but just run as below 

---
ros2 run humanoid_nav_bridge odommodestate_to_odom
"""

from __future__ import annotations

from typing import List, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion, TwistWithCovariance, Twist
from unitree_go.msg import SportModeState


import math

def clamp_pi(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a

def quat_to_yaw(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)

def yaw_to_quat(yaw: float) -> Quaternion:
    q = Quaternion()
    h = 0.5 * yaw
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(h)
    q.w = math.cos(h)
    return q

def make_qos(reliability: str = "best_effort", depth: int = 5) -> QoSProfile:
    rel = ReliabilityPolicy.BEST_EFFORT if reliability.lower() == "best_effort" else ReliabilityPolicy.RELIABLE
    return QoSProfile(
        reliability=rel,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
        depth=depth,
    )


def quat_from_array(q: List[float], order: str) -> Quaternion:
    """
    Convert an array[4] into geometry_msgs/Quaternion.

    order:
      - "wxyz": q=[w,x,y,z]
      - "xyzw": q=[x,y,z,w]
    """
    if len(q) != 4:
        # fallback safe identity
        out = Quaternion()
        out.w = 1.0
        return out

    out = Quaternion()
    if order.lower() == "wxyz":
        w, x, y, z = q
        out.x = float(x)
        out.y = float(y)
        out.z = float(z)
        out.w = float(w)
    else:  # "xyzw"
        x, y, z, w = q
        out.x = float(x)
        out.y = float(y)
        out.z = float(z)
        out.w = float(w)
    return out


class OdomModeStateToOdom(Node):
    def __init__(self) -> None:
        super().__init__("odommodestate_to_odom")

        # -----------------------------
        # Parameters
        # -----------------------------
        self.input_topic = self.declare_parameter("input_topic", "/odommodestate").value
        self.output_topic = self.declare_parameter("output_topic", "/odommodestate_odom").value

        self.odom_frame = self.declare_parameter("odom_frame", "odom").value
        self.child_frame = self.declare_parameter("child_frame", "base_footprint").value
        self.base_frame = self.declare_parameter("base_frame", "base_footprint").value


        # "wxyz" is the safest guess for Unitree
        self.quat_order = str(self.declare_parameter("quat_order", "wxyz").value).strip().lower()

        # QoS
        self.sub_reliability = str(self.declare_parameter("sub_reliability", "best_effort").value)
        self.sub_depth = int(self.declare_parameter("sub_depth", 5).value)

        self.pub_reliability = str(self.declare_parameter("pub_reliability", "reliable").value)
        self.pub_depth = int(self.declare_parameter("pub_depth", 10).value)

        self.sub_qos = make_qos(self.sub_reliability, self.sub_depth)
        self.pub_qos = make_qos(self.pub_reliability, self.pub_depth)

        # -----------------------------
        # ROS Interfaces
        # -----------------------------
        self.pub = self.create_publisher(Odometry, self.output_topic, self.pub_qos)
        self.sub = self.create_subscription(SportModeState, self.input_topic, self.cb, self.sub_qos)

        self.get_logger().info(
            "OdomModeStateToOdom started:\n"
            f"  in : {self.input_topic} (SportModeState)\n"
            f"  out: {self.output_topic} (nav_msgs/Odometry)\n"
            f"  frames: {self.odom_frame} -> {self.child_frame}\n"
            f"  quat_order={self.quat_order}\n"
            f"  QoS sub={self.sub_reliability} pub={self.pub_reliability}"
        )

    def cb(self, msg: SportModeState) -> None:
        out = Odometry()

        # Timestamp: SportModeState has stamp but in your echo it is 0.
        # So we use node time (safe).
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = self.odom_frame
        out.child_frame_id = self.child_frame

        # Pose (x,y,z + quaternion)
        # position is a float array of size 3 in your echo
        if hasattr(msg, "position") and len(msg.position) >= 3:
            out.pose.pose.position.x = float(msg.position[0])
            out.pose.pose.position.y = float(msg.position[1])
            #out.pose.pose.position.z = float(msg.position[2])
            out.pose.pose.position.z = 0.0
        else:
            out.pose.pose.position.x = 0.0
            out.pose.pose.position.y = 0.0
            out.pose.pose.position.z = 0.0

        # IMU quaternion
        q_arr = []
        try:
            q_arr = list(msg.imu_state.quaternion)
        except Exception:
            q_arr = [1.0, 0.0, 0.0, 0.0]  # fallback wxyz-ish
        #out.pose.pose.orientation = quat_from_array(q_arr, self.quat_order)
        q_full = quat_from_array(q_arr, self.quat_order)
        yaw = quat_to_yaw(q_full.x, q_full.y, q_full.z, q_full.w)
        out.pose.pose.orientation = yaw_to_quat(yaw)

        # Twist
        tw = Twist()
        if hasattr(msg, "velocity") and len(msg.velocity) >= 3:
            tw.linear.x = float(msg.velocity[0])
            #tw.linear.y = float(msg.velocity[1])
            #tw.linear.z = float(msg.velocity[2])
            tw.linear.y = 0.0
            tw.linear.z = 0.0

        # yaw_speed is explicitly provided (rad/s)
        if hasattr(msg, "yaw_speed"):
            tw.angular.z = float(msg.yaw_speed)

        twc = TwistWithCovariance()
        twc.twist = tw
        twc.covariance = [0.0] * 36
        twc.covariance[0] = 0.10
        twc.covariance[7] = 0.10
        twc.covariance[35] = 0.20
        out.twist = twc

        # Pose covariance (EKF-friendly; not all zeros)
        out.pose.covariance = [0.0] * 36
        out.pose.covariance[0] = 0.05
        out.pose.covariance[7] = 0.05
        out.pose.covariance[14] = 0.20
        out.pose.covariance[21] = 0.30
        out.pose.covariance[28] = 0.30
        out.pose.covariance[35] = 0.10

        self.pub.publish(out)


def main() -> None:
    rclpy.init()
    node = OdomModeStateToOdom()
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