#!/usr/bin/env python3
"""
real_odom_tf_bridge_cmd_gated_pro.py (ROS 2 Humble)

PRODUCTION odom + TF bridge for floating-base humanoids (Unitree G1 etc.)
with command-gated integration + optional driver pose passthrough.

Key upgrade in this version
---------------------------
✅ Adds a planar navigation base frame (`base_footprint`) while preserving the real body frame (`robot_center`).

Nav2 expects a *planar* base frame:
  - roll = 0, pitch = 0, z = 0
  - only x, y, yaw matter

A humanoid RL policy produces continuous roll/pitch oscillations during walking.
If Nav2 uses a non-planar base (robot_center), LaserScan transforms become "tilted"
and costmaps generate fake obstacles -> spinning, recoveries, never reaching goal.

Therefore we publish TWO TF transforms:
  1) odom -> base_footprint  (PLANAR: x,y,yaw)
  2) base_footprint -> robot_center (BODY: roll,pitch,z, yaw=0)

Odometry integration strategy
----------------------------
- Integrate SE(2) pose ONLY from measured twist (/dog_odom.twist), never from cmd_vel.
- Use cmd_vel only as a gating signal to decide whether we should integrate.
- Optionally integrate lateral velocity vy if it is real (not sway noise).

Inputs
------
- /dog_odom        (nav_msgs/Odometry)      : measured motion
- /cmd_vel_rlsar   (geometry_msgs/Twist)    : command gating only

Outputs
-------
- /odom (nav_msgs/Odometry)
- TF: odom -> base_frame (planar), base_frame -> body_frame (roll/pitch)
- /odom_bridge/status (std_msgs/String) debug telemetry

Author: Jean Chrysostome Mayoko Biong
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped, Twist, TwistWithCovariance
from std_msgs.msg import String
import tf2_ros


# ============================================================
# QoS helper
# ============================================================
def _make_qos(reliable: bool, depth: int) -> QoSProfile:
    return QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE if reliable else ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
        depth=depth,
    )


# ============================================================
# Math helpers
# ============================================================
def _clamp_angle_pi(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def _quat_from_yaw(yaw: float) -> Tuple[float, float, float, float]:
    """Quaternion for yaw-only rotation (REP-103)."""
    half = 0.5 * yaw
    return (0.0, 0.0, math.sin(half), math.cos(half))


def _yaw_from_quat(x: float, y: float, z: float, w: float) -> float:
    """Yaw from quaternion (REP-103)."""
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def _rpy_from_quat(x: float, y: float, z: float, w: float) -> Tuple[float, float, float]:
    """
    Convert quaternion -> roll, pitch, yaw (REP-103, ZYX convention).
    Used to extract roll/pitch for the body frame.
    """
    # roll (x-axis)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis)
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    # yaw (z-axis)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def _quat_from_rpy(roll: float, pitch: float, yaw: float) -> Tuple[float, float, float, float]:
    """Convert roll, pitch, yaw -> quaternion (ZYX convention)."""
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)

    qw = cy * cp * cr + sy * sp * sr
    qx = cy * cp * sr - sy * sp * cr
    qy = sy * cp * sr + cy * sp * cr
    qz = sy * cp * cr - cy * sp * sr
    return qx, qy, qz, qw


# ============================================================
# Data structures
# ============================================================
@dataclass
class GateDecision:
    integrate: bool
    reason: str
    cmd_age: float
    cmd_vx: float
    cmd_wz: float


# ============================================================
# Node
# ============================================================
class RealOdomTfBridgeCmdGatedPro(Node):
    def __init__(self) -> None:
        super().__init__("real_odom_tf_bridge_cmd_gated")

        # ----------------------------------------------------
        # Topics
        # ----------------------------------------------------
        self.input_odom_topic = self.declare_parameter("input_odom_topic", "/dog_odom").value
        self.cmd_vel_topic = self.declare_parameter("cmd_vel_topic", "/cmd_vel_rlsar").value
        self.output_odom_topic = self.declare_parameter("output_odom_topic", "/odom").value

        # ----------------------------------------------------
        # Frames (IMPORTANT for humanoids + Nav2)
        # ----------------------------------------------------
        # odom_frame: parent of the odometry chain (e.g., "odom")
        # base_frame: planar navigation base frame (recommended: "base_footprint")
        # body_frame: real floating body frame (recommended: "robot_center")
        self.odom_frame = self.declare_parameter("odom_frame", "odom").value
        self.base_frame = self.declare_parameter("base_frame", "base_footprint").value
        self.body_frame = self.declare_parameter("body_frame", "robot_center").value

        # ----------------------------------------------------
        # Pose source (what pose do we publish as planar?)
        # ----------------------------------------------------
        # integrate: x,y,yaw from our SE(2) integration (recommended for humanoid RL)
        # driver   : x,y,yaw from driver pose (can be unstable for Nav2 if noisy)
        # hybrid   : x,y from driver, yaw from integrated yaw
        self.pose_source = str(self.declare_parameter("pose_source", "integrate").value).strip().lower()
        if self.pose_source not in ("integrate", "driver", "hybrid"):
            self.get_logger().warn(f"Invalid pose_source='{self.pose_source}', forcing 'integrate'")
            self.pose_source = "integrate"

        # Optional: subtract initial driver pose so Nav2 starts around (0,0,0)
        self.zero_at_start = bool(self.declare_parameter("zero_at_start", True).value)
        self._zero_ready = False
        self._x0 = 0.0
        self._y0 = 0.0
        self._yaw0 = 0.0

        # ----------------------------------------------------
        # Time policy
        # ----------------------------------------------------
        # If False: use driver odom header stamp for TF/odom (better coherence)
        # If True : use node clock stamps (can be safer if driver stamps are broken)
        self.use_node_time = bool(self.declare_parameter("use_node_time", False).value)

        # Rotation confirmation window (allow using cmd yaw for a short time until measurement follows)
        self.rot_confirm_timeout = float(self.declare_parameter("rot_confirm_timeout", 0.35).value)
        self._rot_cmd_start_time: Optional[Time] = None

        # ----------------------------------------------------
        # Yaw-rate remap (only if your robot reports yaw in angular.y)
        # ----------------------------------------------------
        self.remap_yawrate_y_to_z = bool(self.declare_parameter("remap_yawrate_y_to_z", False).value)

        # ----------------------------------------------------
        # Command gating (cmd_vel is used ONLY as "intent", not for integration)
        # ----------------------------------------------------
        self.cmd_eps_vx = float(self.declare_parameter("cmd_eps_vx", 0.02).value)
        self.cmd_eps_wz = float(self.declare_parameter("cmd_eps_wz", 0.02).value)
        self.cmd_timeout_sec = float(self.declare_parameter("cmd_timeout_sec", 0.25).value)

        # Freeze yaw when idle (recommended)
        self.freeze_yaw_when_idle = bool(self.declare_parameter("freeze_yaw_when_idle", True).value)

        # ----------------------------------------------------
        # Turn-in-place rule
        # ----------------------------------------------------
        # If Nav2 commands a strong rotation with small forward speed, treat it as pure rotation.
        self.turn_wz_thresh = float(self.declare_parameter("turn_wz_thresh", 0.25).value)
        self.turn_vx_small = float(self.declare_parameter("turn_vx_small", 0.04).value)

        # ----------------------------------------------------
        # Filtering measured twist (deadband + EMA)
        # ----------------------------------------------------
        self.enable_filter = bool(self.declare_parameter("enable_filter", True).value)
        self.deadband_vx = float(self.declare_parameter("deadband_vx", 0.02).value)
        self.deadband_wz = float(self.declare_parameter("deadband_wz", 0.02).value)
        self.ema_alpha = float(self.declare_parameter("ema_alpha", 0.25).value)

        # Optional lateral velocity (vy)
        self.enable_vy = bool(self.declare_parameter("enable_vy", False).value)
        self.deadband_vy = float(self.declare_parameter("deadband_vy", 0.02).value)
        self.ema_alpha_vy = float(self.declare_parameter("ema_alpha_vy", self.ema_alpha).value)

        # Measured eps thresholds (what counts as "real motion")
        self.meas_eps_vx = float(self.declare_parameter("meas_eps_vx", 0.03).value)
        self.meas_eps_vy = float(self.declare_parameter("meas_eps_vy", 0.03).value)
        self.meas_eps_wz = float(self.declare_parameter("meas_eps_wz", 0.05).value)

        # ----------------------------------------------------
        # Integration safety
        # ----------------------------------------------------
        self.dt_max = float(self.declare_parameter("dt_max", 0.20).value)

        # ----------------------------------------------------
        # Publishing
        # ----------------------------------------------------
        self.publish_tf = bool(self.declare_parameter("publish_tf", True).value)
        self.publish_odom = bool(self.declare_parameter("publish_odom", True).value)
        self.publish_status = bool(self.declare_parameter("publish_status", True).value)

        self.tf_rate_hz = float(self.declare_parameter("tf_rate_hz", 50.0).value)
        self.odom_rate_hz = float(self.declare_parameter("odom_rate_hz", 50.0).value)
        self.status_rate_hz = float(self.declare_parameter("status_rate_hz", 5.0).value)

        # Watchdog
        self.timeout_sec = float(self.declare_parameter("timeout_sec", 1.0).value)

        # ----------------------------------------------------
        # QoS
        # ----------------------------------------------------
        self.odom_sub_reliable = bool(self.declare_parameter("odom_sub_reliable", True).value)
        self.cmd_sub_reliable = bool(self.declare_parameter("cmd_sub_reliable", False).value)
        self.pub_reliable = bool(self.declare_parameter("pub_reliable", True).value)
        self.qos_depth = int(self.declare_parameter("qos_depth", 10).value)

        odom_sub_qos = _make_qos(self.odom_sub_reliable, self.qos_depth)
        cmd_sub_qos = _make_qos(self.cmd_sub_reliable, self.qos_depth)
        pub_qos = _make_qos(self.pub_reliable, self.qos_depth)

        # ----------------------------------------------------
        # ROS interfaces
        # ----------------------------------------------------
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        self.odom_pub = self.create_publisher(Odometry, self.output_odom_topic, pub_qos)
        self.status_pub = self.create_publisher(String, "/odom_bridge/status", pub_qos)

        self.odom_sub = self.create_subscription(Odometry, self.input_odom_topic, self._cb_odom, odom_sub_qos)
        self.cmd_sub = self.create_subscription(Twist, self.cmd_vel_topic, self._cb_cmd, cmd_sub_qos)

        self.get_logger().info(
            f"QoS: odom_sub={'RELIABLE' if self.odom_sub_reliable else 'BEST_EFFORT'} | "
            f"cmd_sub={'RELIABLE' if self.cmd_sub_reliable else 'BEST_EFFORT'} | "
            f"pub={'RELIABLE' if self.pub_reliable else 'BEST_EFFORT'} | depth={self.qos_depth}"
        )

        # ----------------------------------------------------
        # State: last messages
        # ----------------------------------------------------
        self._last_odom: Optional[Odometry] = None
        self._last_cmd: Optional[Twist] = None
        self._last_odom_time: Optional[Time] = None
        self._last_cmd_time: Optional[Time] = None

        # ----------------------------------------------------
        # State: integrated SE(2)
        # ----------------------------------------------------
        self._x_int = 0.0
        self._y_int = 0.0
        self._yaw_int = 0.0
        self._last_integration_time = self.get_clock().now()

        # EMA filter state
        self._vx_f = 0.0
        self._vy_f = 0.0
        self._wz_f = 0.0

        # Debug snapshot of last integration twist (for status)
        self._last_gate = GateDecision(False, "init", 999.0, 0.0, 0.0)
        self._last_twist_vx = 0.0
        self._last_twist_vy = 0.0
        self._last_twist_wz = 0.0

        # ----------------------------------------------------
        # Timers
        # ----------------------------------------------------
        self.create_timer(0.5, self._watchdog_tick)

        if self.publish_tf and self.tf_rate_hz > 0:
            self.create_timer(1.0 / self.tf_rate_hz, self._publish_tf_tick)

        if self.publish_odom and self.odom_rate_hz > 0:
            self.create_timer(1.0 / self.odom_rate_hz, self._publish_odom_tick)

        if self.publish_status and self.status_rate_hz > 0:
            self.create_timer(1.0 / self.status_rate_hz, self._publish_status_tick)

        self._last_info_log_time = 0.0
        self._info_log_period = 2.0

        # Startup log
        self.get_logger().info(
            "Cmd-gated PRO bridge started:\n"
            f"  pose_source={self.pose_source}\n"
            f"  /dog_odom={self.input_odom_topic}\n"
            f"  cmd_vel={self.cmd_vel_topic}\n"
            f"  /odom={self.output_odom_topic}\n"
            f"  TF: {self.odom_frame}->{self.base_frame}->{self.body_frame}\n"
            f"  enable_vy={self.enable_vy}\n"
        )

    # ============================================================
    # Callbacks
    # ============================================================
    def _cb_odom(self, msg: Odometry) -> None:
        self._last_odom = msg
        self._last_odom_time = self.get_clock().now()

        # Lock zero-at-start offsets from driver pose (only once)
        if self.zero_at_start and not self._zero_ready:
            self._x0 = float(msg.pose.pose.position.x)
            self._y0 = float(msg.pose.pose.position.y)
            q = msg.pose.pose.orientation
            self._yaw0 = _yaw_from_quat(float(q.x), float(q.y), float(q.z), float(q.w))
            self._zero_ready = True
            self.get_logger().info(
                f"zero_at_start locked: x0={self._x0:.3f} y0={self._y0:.3f} yaw0={self._yaw0:.3f}"
            )

    def _cb_cmd(self, msg: Twist) -> None:
        self._last_cmd = msg
        self._last_cmd_time = self.get_clock().now()

    # ============================================================
    # Time helpers
    # ============================================================
    def _now(self) -> Time:
        return self.get_clock().now()

    def _stamp_msg(self):
        """
        Return a timestamp consistent with sensor time when possible.

        Priority:
          1) Use last received odom header stamp (best for Nav2 TF coherence)
          2) Fallback to node clock
        """
        if (not self.use_node_time) and (self._last_odom is not None):
            return self._last_odom.header.stamp
        return self.get_clock().now().to_msg()

    # ============================================================
    # Gating decision
    # ============================================================
    def _decide_gate(self) -> GateDecision:
        if self._last_cmd is None or self._last_cmd_time is None:
            return GateDecision(False, "no_cmd_yet", 999.0, 0.0, 0.0)

        cmd_age = (self._now() - self._last_cmd_time).nanoseconds * 1e-9
        cmd_vx = float(self._last_cmd.linear.x)
        cmd_wz = float(self._last_cmd.angular.z)

        if cmd_age > self.cmd_timeout_sec:
            return GateDecision(False, "cmd_stale", cmd_age, cmd_vx, cmd_wz)

        if (abs(cmd_vx) > self.cmd_eps_vx) or (abs(cmd_wz) > self.cmd_eps_wz):
            return GateDecision(True, "cmd_active", cmd_age, cmd_vx, cmd_wz)

        return GateDecision(False, "cmd_below_eps", cmd_age, cmd_vx, cmd_wz)

    # ============================================================
    # Twist correction + filtering
    # ============================================================
    def _correct_and_filter_twist(self, odom: Odometry) -> Tuple[float, float, float]:
        """
        Returns a filtered planar twist (vx, vy, wz) in the robot base frame.

        - vx: forward velocity (m/s)
        - vy: lateral velocity (m/s), optional (enable_vy)
        - wz: yaw rate (rad/s)
        """
        vx = float(odom.twist.twist.linear.x)
        vy = float(odom.twist.twist.linear.y)

        wy = float(odom.twist.twist.angular.y)
        wz = float(odom.twist.twist.angular.z)
        wz_used = wy if self.remap_yawrate_y_to_z else wz

        if not self.enable_filter:
            return vx, (vy if self.enable_vy else 0.0), wz_used

        # Deadbands
        if abs(vx) < self.deadband_vx:
            vx = 0.0

        if self.enable_vy:
            if abs(vy) < self.deadband_vy:
                vy = 0.0
        else:
            vy = 0.0

        if abs(wz_used) < self.deadband_wz:
            wz_used = 0.0

        # EMA filtering
        self._vx_f = self.ema_alpha * vx + (1.0 - self.ema_alpha) * self._vx_f

        if self.enable_vy:
            a = self.ema_alpha_vy
            self._vy_f = a * vy + (1.0 - a) * self._vy_f
        else:
            self._vy_f = 0.0

        self._wz_f = self.ema_alpha * wz_used + (1.0 - self.ema_alpha) * self._wz_f

        return self._vx_f, self._vy_f, self._wz_f

    def _apply_turn_in_place_rule(self, vx: float, vy: float) -> Tuple[float, float]:
        """
        If Nav2 is commanding a turn-in-place, force translation to zero.
        This avoids integrating tiny humanoid drift during rotation.
        """
        if self._last_cmd is None:
            return vx, vy

        cmd_vx = float(self._last_cmd.linear.x)
        cmd_wz = float(self._last_cmd.angular.z)

        if abs(cmd_wz) > self.turn_wz_thresh and abs(cmd_vx) < self.turn_vx_small:
            return 0.0, 0.0

        return vx, vy

    # ============================================================
    # SE(2) integration
    # ============================================================
    def _integrate_se2(self, vx: float, vy: float, wz: float, integrate_yaw: bool) -> None:
        """
        Integrate a planar SE(2) state (x, y, yaw) using base-frame twist.

        Twist is expressed in the robot base frame:
          dx = (vx*cos(yaw) - vy*sin(yaw)) * dt
          dy = (vx*sin(yaw) + vy*cos(yaw)) * dt
        """
        now = self._now()
        dt = (now - self._last_integration_time).nanoseconds * 1e-9
        self._last_integration_time = now

        if dt < 0.0 or dt > self.dt_max:
            return

        if integrate_yaw:
            self._yaw_int = _clamp_angle_pi(self._yaw_int + wz * dt)

        cy = math.cos(self._yaw_int)
        sy = math.sin(self._yaw_int)

        self._x_int += (vx * cy - vy * sy) * dt
        self._y_int += (vx * sy + vy * cy) * dt

    # ============================================================
    # Pose selection (driver/integrate/hybrid) + zero_at_start
    # ============================================================
    def _select_pose(self) -> Tuple[float, float, float]:
        """
        Returns (x, y, yaw) for the planar navigation frame.

        integrate: from integrated SE(2)
        driver   : from driver pose (may include artifacts for humanoids)
        hybrid   : x,y from driver pose, yaw from integrated yaw
        """
        if self._last_odom is None:
            return 0.0, 0.0, 0.0

        if self.pose_source == "integrate":
            x, y, yaw = self._x_int, self._y_int, self._yaw_int
        elif self.pose_source == "driver":
            p = self._last_odom.pose.pose.position
            q = self._last_odom.pose.pose.orientation
            x = float(p.x)
            y = float(p.y)
            yaw = _yaw_from_quat(float(q.x), float(q.y), float(q.z), float(q.w))
        else:  # hybrid
            p = self._last_odom.pose.pose.position
            x = float(p.x)
            y = float(p.y)
            yaw = self._yaw_int

        if self.zero_at_start and self._zero_ready:
            x -= self._x0
            y -= self._y0
            yaw = _clamp_angle_pi(yaw - self._yaw0)

        return x, y, yaw

    # ============================================================
    # Watchdog
    # ============================================================
    def _watchdog_tick(self) -> None:
        now = self._now()

        if self._last_odom_time is None:
            self.get_logger().warn("No /dog_odom received yet...")
        else:
            odom_age = (now - self._last_odom_time).nanoseconds * 1e-9
            if odom_age > self.timeout_sec:
                self.get_logger().warn(f"/dog_odom stale: age={odom_age:.2f}s")

        if self._last_cmd_time is None:
            self.get_logger().warn(f"No {self.cmd_vel_topic} yet -> bridge will freeze.")

    # ============================================================
    # Publishing: TF
    # ============================================================
    def _publish_tf_tick(self) -> None:
        """
        Publish TF:
          1) odom -> base_frame (planar: x,y,yaw)
          2) base_frame -> body_frame (roll,pitch,z; yaw=0)

        This prevents Nav2 costmaps from being polluted by roll/pitch oscillations.
        """
        if self._last_odom is None:
            return

        stamp = self._stamp_msg()

        # -----------------------------
        # (1) PLANAR nav frame
        # -----------------------------
        x, y, yaw = self._select_pose()
        qx, qy, qz, qw = _quat_from_yaw(yaw)

        tf_planar = TransformStamped()
        tf_planar.header.stamp = stamp
        tf_planar.header.frame_id = self.odom_frame
        tf_planar.child_frame_id = self.base_frame
        tf_planar.transform.translation.x = float(x)
        tf_planar.transform.translation.y = float(y)
        tf_planar.transform.translation.z = 0.0
        tf_planar.transform.rotation.x = qx
        tf_planar.transform.rotation.y = qy
        tf_planar.transform.rotation.z = qz
        tf_planar.transform.rotation.w = qw
        self.tf_broadcaster.sendTransform(tf_planar)

        # -----------------------------
        # (2) BODY frame (roll/pitch)
        # -----------------------------
        # Extract roll/pitch/z from driver pose.
        # IMPORTANT: yaw must be 0 here to avoid double-yaw.
        p = self._last_odom.pose.pose.position
        q = self._last_odom.pose.pose.orientation
        roll, pitch, _ = _rpy_from_quat(float(q.x), float(q.y), float(q.z), float(q.w))
        bqx, bqy, bqz, bqw = _quat_from_rpy(roll, pitch, 0.0)

        tf_body = TransformStamped()
        tf_body.header.stamp = stamp
        tf_body.header.frame_id = self.base_frame
        tf_body.child_frame_id = self.body_frame
        tf_body.transform.translation.x = 0.0
        tf_body.transform.translation.y = 0.0

        # For 2D navigation, z is irrelevant. Keeping it can be useful for visualization.
        # If you suspect z noise affects any consumer, set this to 0.0.
        tf_body.transform.translation.z = float(p.z)

        tf_body.transform.rotation.x = bqx
        tf_body.transform.rotation.y = bqy
        tf_body.transform.rotation.z = bqz
        tf_body.transform.rotation.w = bqw
        self.tf_broadcaster.sendTransform(tf_body)

    # ============================================================
    # Publishing: /odom
    # ============================================================
    def _publish_odom_tick(self) -> None:
        """
        Publish Nav2-compatible /odom.

        The core rule:
          - Integrate pose only from measured motion (/dog_odom.twist)
          - Use cmd_vel only to decide whether we *should* integrate (gating)
        """
        if self._last_odom is None:
            return

        # (1) Command gating decision
        gate = self._decide_gate()
        self._last_gate = gate
        cmd_active = gate.integrate  # True if cmd is fresh and above eps

        # (2) Filter measured twist
        vx_meas, vy_meas, wz_meas = self._correct_and_filter_twist(self._last_odom)

        # Apply turn-in-place: no translation during in-place rotation
        vx_meas, vy_meas = self._apply_turn_in_place_rule(vx_meas, vy_meas)

        # (3) Command twist (for intent + rotation window)
        vx_cmd = float(gate.cmd_vx)
        wz_cmd = float(gate.cmd_wz)

        if abs(wz_cmd) > self.turn_wz_thresh and abs(vx_cmd) < self.turn_vx_small:
            vx_cmd = 0.0

        # Detect rotation command intent
        rot_cmd_active = cmd_active and (abs(wz_cmd) > self.cmd_eps_wz)

        # Measured evidence above noise thresholds
        yaw_meas_active = abs(wz_meas) > self.meas_eps_wz
        xy_meas_active = cmd_active and (
            abs(vx_meas) > self.meas_eps_vx or
            (self.enable_vy and abs(vy_meas) > self.meas_eps_vy)
        )

        # Rotation confirmation timing window
        now = self._now()

        if rot_cmd_active and self._rot_cmd_start_time is None:
            self._rot_cmd_start_time = now
        if (not rot_cmd_active) and self._rot_cmd_start_time is not None:
            self._rot_cmd_start_time = None

        rot_cmd_age = 0.0
        if self._rot_cmd_start_time is not None:
            rot_cmd_age = (now - self._rot_cmd_start_time).nanoseconds * 1e-9

        allow_cmd_yaw_fallback = rot_cmd_active and (rot_cmd_age <= self.rot_confirm_timeout)

        # Choose yaw rate for integration
        if yaw_meas_active:
            wz_for_yaw = wz_meas
        elif allow_cmd_yaw_fallback:
            wz_for_yaw = wz_cmd
        else:
            wz_for_yaw = 0.0

        # Decide whether to integrate
        integrate_xy = xy_meas_active
        integrate_yaw = cmd_active and (yaw_meas_active or allow_cmd_yaw_fallback)

        # Values actually used for SE(2) integration
        vx_int = vx_meas if integrate_xy else 0.0
        vy_int = vy_meas if (integrate_xy and self.enable_vy) else 0.0
        wz_int = wz_for_yaw if integrate_yaw else 0.0

        # Store for debug/status (IMPORTANT!)
        self._last_twist_vx = vx_int
        self._last_twist_vy = vy_int
        self._last_twist_wz = wz_int

        # (5) Integrate pose ONCE
        if integrate_xy or integrate_yaw:
            self._integrate_se2(
                vx=vx_int,
                vy=vy_int,
                wz=wz_int,
                integrate_yaw=integrate_yaw,
            )
        else:
            # If idle and you want zero yaw drift, keep it frozen.
            # (If you ever want passive yaw drift, you can change this policy.)
            if not self.freeze_yaw_when_idle:
                # Integrate yaw from measurement even when idle (not recommended for humanoids).
                if abs(wz_meas) > self.meas_eps_wz:
                    self._integrate_se2(vx=0.0, vy=0.0, wz=wz_meas, integrate_yaw=True)

        # (6) Pose to publish
        x, y, yaw = self._select_pose()
        qx, qy, qz, qw = _quat_from_yaw(yaw)

        # (7) Publish twist consistent with integration policy
        vx_pub = vx_meas if integrate_xy else 0.0
        vy_pub = vy_meas if (integrate_xy and self.enable_vy) else 0.0
        wz_pub = wz_for_yaw if integrate_yaw else 0.0

        out = Odometry()
        out.header.stamp = self._stamp_msg()
        out.header.frame_id = self.odom_frame
        out.child_frame_id = self.base_frame  # Nav2 base frame (planar)

        out.pose.pose.position.x = float(x)
        out.pose.pose.position.y = float(y)
        out.pose.pose.position.z = 0.0
        out.pose.pose.orientation.x = qx
        out.pose.pose.orientation.y = qy
        out.pose.pose.orientation.z = qz
        out.pose.pose.orientation.w = qw

        # Conservative covariances (tune if needed)
        out.pose.covariance = [0.0] * 36
        out.pose.covariance[0] = 0.05
        out.pose.covariance[7] = 0.05
        out.pose.covariance[35] = 0.10

        tw = Twist()
        tw.linear.x = float(vx_pub)
        tw.linear.y = float(vy_pub)
        tw.angular.z = float(wz_pub)

        twc = TwistWithCovariance()
        twc.twist = tw
        twc.covariance = [0.0] * 36
        twc.covariance[0] = 0.10
        twc.covariance[35] = 0.20
        out.twist = twc

        self.odom_pub.publish(out)

        # (8) Debug logging (throttled)
        now_s = time.time()
        if (now_s - self._last_info_log_time) >= self._info_log_period:
            self._last_info_log_time = now_s
            self.get_logger().info(
                f"gate={int(gate.integrate)}({gate.reason}) "
                f"xy={int(integrate_xy)} yaw={int(integrate_yaw)} "
                f"cmd=({vx_cmd:+.2f},{wz_cmd:+.2f}) "
                f"meas=({vx_meas:+.2f},{vy_meas:+.2f},{wz_meas:+.2f}) "
                f"int=({vx_int:+.2f},{vy_int:+.2f},{wz_int:+.2f}) "
                f"pose=({x:+.2f},{y:+.2f},{yaw:+.2f})"
            )

    # ============================================================
    # Publishing: status
    # ============================================================
    def _publish_status_tick(self) -> None:
        now = self._now()

        odom_age = 999.0
        if self._last_odom_time is not None:
            odom_age = (now - self._last_odom_time).nanoseconds * 1e-9

        cmd_age = 999.0
        if self._last_cmd_time is not None:
            cmd_age = (now - self._last_cmd_time).nanoseconds * 1e-9

        gate = self._last_gate
        x, y, yaw = self._select_pose()

        msg = String()
        msg.data = (
            "{"
            f"\"pose_source\":\"{self.pose_source}\","
            f"\"integrate\":{1 if gate.integrate else 0},"
            f"\"reason\":\"{gate.reason}\","
            f"\"odom_age\":{odom_age:.3f},"
            f"\"cmd_age\":{cmd_age:.3f},"
            f"\"cmd_vx\":{gate.cmd_vx:.3f},"
            f"\"cmd_wz\":{gate.cmd_wz:.3f},"
            f"\"enable_vy\":{1 if self.enable_vy else 0},"
            f"\"twist_vx\":{self._last_twist_vx:.3f},"
            f"\"twist_vy\":{self._last_twist_vy:.3f},"
            f"\"twist_wz\":{self._last_twist_wz:.3f},"
            f"\"x\":{x:.3f},"
            f"\"y\":{y:.3f},"
            f"\"yaw\":{yaw:.3f}"
            "}"
        )
        self.status_pub.publish(msg)


def main() -> None:
    rclpy.init()
    node = RealOdomTfBridgeCmdGatedPro()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()