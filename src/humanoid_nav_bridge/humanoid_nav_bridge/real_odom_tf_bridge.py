#!/usr/bin/env python3
"""
real_odom_tf_bridge.py (ROS 2 Humble) - FINAL

Cmd-gated planar odom + humanoid-safe TF helper for Unitree G1 (RL locomotion).

Goals
-----
1) Provide a *planar* odom suitable for Nav2 / robot_localization:
   - stable x,y,yaw
   - twist consistent with pose evolution
   - covariances non-zero (EKF-friendly)
2) Avoid fake obstacle motion caused by roll/pitch oscillations:
   - publish odom->base_footprint planar
   - publish base_footprint->robot_center roll/pitch only (yaw=0)
3) Robust yaw-rate sourcing:
   yawrate_source: "quat"|"z"|"y"|"auto"
   - quat: dyaw/dt from odom.pose orientation
   - z/y: angular.z or angular.y
   - auto: pick most energetic source when cmd_active (with hysteresis)
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
# Helpers
# ============================================================
def _make_qos(reliable: bool, depth: int) -> QoSProfile:
    return QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE if reliable else ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
        depth=depth,
    )


def _clamp_angle_pi(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def _quat_from_yaw(yaw: float) -> Tuple[float, float, float, float]:
    half = 0.5 * yaw
    return (0.0, 0.0, math.sin(half), math.cos(half))


def _yaw_from_quat(x: float, y: float, z: float, w: float) -> float:
    """Yaw from quaternion (REP-103)."""
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def _rpy_from_quat(x: float, y: float, z: float, w: float) -> Tuple[float, float, float]:
    """Quaternion -> roll, pitch, yaw (REP-103, ZYX)."""
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def _quat_from_rpy(roll: float, pitch: float, yaw: float) -> Tuple[float, float, float, float]:
    """RPY -> quaternion (ZYX)."""
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
class RealOdomTfBridgeCmdGated(Node):
    def __init__(self) -> None:
        super().__init__("real_odom_tf_bridge_cmd_gated")

        # ---------------------------
        # Topics / frames
        # ---------------------------
        self.input_odom_topic = self.declare_parameter("input_odom_topic", "/dog_odom").value
        self.gate_cmd_vel_topic = self.declare_parameter("gate_cmd_vel_topic", "/cmd_vel_nav").value

        # Publish a *source* odom (EKF-friendly). Consider naming it /odom_rl
        self.output_odom_topic = self.declare_parameter("output_odom_topic", "/odom").value

        self.odom_frame = self.declare_parameter("odom_frame", "odom").value
        self.base_frame = self.declare_parameter("base_frame", "base_footprint").value
        self.body_frame = self.declare_parameter("body_frame", "robot_center").value

        # ---------------------------
        # Pose source
        # ---------------------------
        self.pose_source = str(self.declare_parameter("pose_source", "integrate").value).strip().lower()
        if self.pose_source not in ("integrate", "driver", "hybrid"):
            self.get_logger().warn(f"Invalid pose_source='{self.pose_source}', forcing 'integrate'")
            self.pose_source = "integrate"

        self.zero_at_start = bool(self.declare_parameter("zero_at_start", True).value)
        self._zero_ready = False
        self._x0 = 0.0
        self._y0 = 0.0
        self._yaw0 = 0.0

        # ---------------------------
        # Time / stamping
        # ---------------------------
        self.use_node_time = bool(self.declare_parameter("use_node_time", False).value)

        # ---------------------------
        # Yaw-rate source
        # ---------------------------
        # quat|z|y|auto
        self.yawrate_source = str(self.declare_parameter("yawrate_source", "auto").value).strip().lower()
        if self.yawrate_source not in ("quat", "z", "y", "auto"):
            self.get_logger().warn(f"Invalid yawrate_source='{self.yawrate_source}', forcing 'auto'")
            self.yawrate_source = "auto"

        # Auto selection hysteresis (prevents source switching every tick)
        self.yawrate_auto_hold_sec = float(self.declare_parameter("yawrate_auto_hold_sec", 1.0).value)
        self._wz_src_locked: str = "z"
        self._wz_src_lock_time: Time = self.get_clock().now()

        # For quat yaw rate
        self._last_quat_yaw: Optional[float] = None
        self._last_quat_time: Optional[Time] = None

        # ---------------------------
        # Command gating
        # ---------------------------
        self.cmd_eps_vx = float(self.declare_parameter("cmd_eps_vx", 0.02).value)
        self.cmd_eps_wz = float(self.declare_parameter("cmd_eps_wz", 0.02).value)
        self.cmd_timeout_sec = float(self.declare_parameter("cmd_timeout_sec", 0.25).value)

        # Anti-chatter
        self.gate_on_hold_sec  = float(self.declare_parameter("gate_on_hold_sec", 0.0).value)
        self.gate_off_hold_sec = float(self.declare_parameter("gate_off_hold_sec", 0.10).value)
        self._gate_state = False
        self._gate_last_switch_time = self.get_clock().now()

        # Turn-in-place rule (Nav2 intent)
        self.turn_wz_thresh = float(self.declare_parameter("turn_wz_thresh", 0.25).value)
        self.turn_vx_small  = float(self.declare_parameter("turn_vx_small", 0.04).value)

        # ---------------------------
        # Filtering measured twist
        # ---------------------------
        self.enable_filter = bool(self.declare_parameter("enable_filter", True).value)
        self.deadband_vx = float(self.declare_parameter("deadband_vx", 0.02).value)
        self.deadband_wz = float(self.declare_parameter("deadband_wz", 0.02).value)
        self.ema_alpha   = float(self.declare_parameter("ema_alpha", 0.25).value)

        # Optional vy
        self.enable_vy = bool(self.declare_parameter("enable_vy", False).value)
        self.deadband_vy = float(self.declare_parameter("deadband_vy", 0.02).value)
        self.ema_alpha_vy = float(self.declare_parameter("ema_alpha_vy", self.ema_alpha).value)

        # Evidence thresholds (ignore sway/noise at rest)
        self.meas_eps_vx = float(self.declare_parameter("meas_eps_vx", 0.03).value)
        self.meas_eps_vy = float(self.declare_parameter("meas_eps_vy", 0.03).value)
        self.meas_eps_wz = float(self.declare_parameter("meas_eps_wz", 0.05).value)

        # Integration safety
        self.dt_max = float(self.declare_parameter("dt_max", 0.20).value)

        # Rotation confirmation (measured yaw can lag)
        self.rot_confirm_timeout = float(self.declare_parameter("rot_confirm_timeout", 0.35).value)
        self._rot_cmd_start_time: Optional[Time] = None

        # ---------------------------
        # Publishing
        # ---------------------------
        self.publish_tf = bool(self.declare_parameter("publish_tf", True).value)
        self.publish_odom = bool(self.declare_parameter("publish_odom", True).value)
        self.publish_status = bool(self.declare_parameter("publish_status", True).value)

        self.tf_rate_hz = float(self.declare_parameter("tf_rate_hz", 50.0).value)
        self.odom_rate_hz = float(self.declare_parameter("odom_rate_hz", 50.0).value)
        self.status_rate_hz = float(self.declare_parameter("status_rate_hz", 5.0).value)

        # QoS
        self.odom_sub_reliable     = bool(self.declare_parameter("odom_sub_reliable", True).value)
        self.gate_cmd_sub_reliable = bool(self.declare_parameter("gate_cmd_sub_reliable", True).value)
        self.pub_reliable          = bool(self.declare_parameter("pub_reliable", True).value)
        self.qos_depth             = int(self.declare_parameter("qos_depth", 10).value)
        self.gate_cmd_qos_depth    = int(self.declare_parameter("gate_cmd_qos_depth", 1).value)

        odom_sub_qos = _make_qos(self.odom_sub_reliable, self.qos_depth)
        gate_cmd_qos = _make_qos(self.gate_cmd_sub_reliable, self.gate_cmd_qos_depth)
        pub_qos      = _make_qos(self.pub_reliable, self.qos_depth)

        # ROS interfaces
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.odom_pub   = self.create_publisher(Odometry, self.output_odom_topic, pub_qos)
        self.status_pub = self.create_publisher(String, "/odom_bridge/status", pub_qos)

        self.odom_sub = self.create_subscription(Odometry, self.input_odom_topic, self._cb_odom, odom_sub_qos)
        self.gate_sub = self.create_subscription(Twist, self.gate_cmd_vel_topic, self._cb_gate_cmd, gate_cmd_qos)

        # State
        self._last_odom: Optional[Odometry] = None
        self._last_odom_time: Optional[Time] = None
        self._last_gate_cmd: Optional[Twist] = None
        self._last_gate_cmd_time: Optional[Time] = None

        # Integrated SE(2)
        self._x_int = 0.0
        self._y_int = 0.0
        self._yaw_int = 0.0
        self._last_integration_time = self.get_clock().now()

        # EMA state
        self._vx_f = 0.0
        self._vy_f = 0.0
        self._wz_f = 0.0

        # Debug
        self._last_gate = GateDecision(False, "init", 999.0, 0.0, 0.0)
        self._last_twist_vx = 0.0
        self._last_twist_wz = 0.0
        self._last_wz_src = "?"
        self._last_wz_y = 0.0
        self._last_wz_z = 0.0
        self._last_wz_quat = 0.0
        self._last_dyaw_dt = 0.0

        # Timers
        if self.publish_tf and self.tf_rate_hz > 0:
            self.create_timer(1.0 / self.tf_rate_hz, self._publish_tf_tick)
        if self.publish_odom and self.odom_rate_hz > 0:
            self.create_timer(1.0 / self.odom_rate_hz, self._publish_odom_tick)
        if self.publish_status and self.status_rate_hz > 0:
            self.create_timer(1.0 / self.status_rate_hz, self._publish_status_tick)

        self._last_info_log_time = 0.0
        self._info_log_period = 2.0

        self.get_logger().info(
            "Bridge FINAL started:\n"
            f"  /dog_odom={self.input_odom_topic}\n"
            f"  gate_cmd={self.gate_cmd_vel_topic}\n"
            f"  /odom_out={self.output_odom_topic}\n"
            f"  pose_source={self.pose_source}\n"
            f"  yawrate_source={self.yawrate_source}\n"
            f"  frames: {self.odom_frame}->{self.base_frame}->{self.body_frame}\n"
            f"  publish_tf={self.publish_tf} (disable if EKF publishes TF)\n"
        )

    # ============================================================
    # Callbacks
    # ============================================================
    def _cb_odom(self, msg: Odometry) -> None:
        self._last_odom = msg
        self._last_odom_time = self.get_clock().now()

        # latch zero_at_start offsets using driver pose
        if self.zero_at_start and not self._zero_ready:
            self._x0 = float(msg.pose.pose.position.x)
            self._y0 = float(msg.pose.pose.position.y)
            q = msg.pose.pose.orientation
            self._yaw0 = _yaw_from_quat(float(q.x), float(q.y), float(q.z), float(q.w))
            self._zero_ready = True
            self.get_logger().info(
                f"zero_at_start locked: x0={self._x0:.3f} y0={self._y0:.3f} yaw0={self._yaw0:.3f}"
            )

    def _cb_gate_cmd(self, msg: Twist) -> None:
        self._last_gate_cmd = msg
        self._last_gate_cmd_time = self.get_clock().now()

    # ============================================================
    # Time helpers
    # ============================================================
    def _now(self) -> Time:
        return self.get_clock().now()

    def _stamp_msg(self):
        # Prefer sensor stamp for coherence with SLAM/AMCL (unless forced)
        if (not self.use_node_time) and (self._last_odom is not None):
            return self._last_odom.header.stamp
        return self.get_clock().now().to_msg()

    # ============================================================
    # Gating
    # ============================================================
    def _decide_gate(self) -> GateDecision:
        if self._last_gate_cmd is None or self._last_gate_cmd_time is None:
            self._gate_state = False
            return GateDecision(False, "no_cmd_yet", 999.0, 0.0, 0.0)

        now = self._now()
        cmd_age = (now - self._last_gate_cmd_time).nanoseconds * 1e-9

        cmd_vx = float(self._last_gate_cmd.linear.x)
        # cmd_wz from command frame: always angular.z (Nav2 convention)
        cmd_wz = float(self._last_gate_cmd.angular.z)

        # stale -> force OFF (with hold)
        if cmd_age > self.cmd_timeout_sec:
            if self._gate_state:
                since = (now - self._gate_last_switch_time).nanoseconds * 1e-9
                if since < self.gate_off_hold_sec:
                    return GateDecision(True, "cmd_stale_hold", cmd_age, cmd_vx, cmd_wz)
            self._gate_state = False
            self._gate_last_switch_time = now
            return GateDecision(False, "cmd_stale", cmd_age, cmd_vx, cmd_wz)

        active = (abs(cmd_vx) > self.cmd_eps_vx) or (abs(cmd_wz) > self.cmd_eps_wz)
        since = (now - self._gate_last_switch_time).nanoseconds * 1e-9

        if active and not self._gate_state:
            if self.gate_on_hold_sec > 0.0 and since < self.gate_on_hold_sec:
                return GateDecision(False, "cmd_active_hold_on", cmd_age, cmd_vx, cmd_wz)
            self._gate_state = True
            self._gate_last_switch_time = now
            return GateDecision(True, "cmd_active", cmd_age, cmd_vx, cmd_wz)

        if (not active) and self._gate_state:
            if self.gate_off_hold_sec > 0.0 and since < self.gate_off_hold_sec:
                return GateDecision(True, "cmd_below_eps_hold_off", cmd_age, cmd_vx, cmd_wz)
            self._gate_state = False
            self._gate_last_switch_time = now
            return GateDecision(False, "cmd_below_eps", cmd_age, cmd_vx, cmd_wz)

        return GateDecision(self._gate_state, "cmd_active" if self._gate_state else "cmd_below_eps",
                            cmd_age, cmd_vx, cmd_wz)

    # ============================================================
    # Yaw-rate measurement
    # ============================================================
    def _wz_from_quat(self, odom: Odometry) -> float:
        """Compute dyaw/dt from pose quaternion."""
        q = odom.pose.pose.orientation
        yaw = _yaw_from_quat(float(q.x), float(q.y), float(q.z), float(q.w))

        now = self._now()
        if self._last_quat_yaw is None or self._last_quat_time is None:
            self._last_quat_yaw = yaw
            self._last_quat_time = now
            return 0.0

        dt = (now - self._last_quat_time).nanoseconds * 1e-9
        self._last_quat_time = now
        if dt <= 1e-4 or dt > self.dt_max:
            self._last_quat_yaw = yaw
            return 0.0

        dyaw = _clamp_angle_pi(yaw - self._last_quat_yaw)
        self._last_quat_yaw = yaw
        return dyaw / dt

    def _pick_wz_source_auto(self, cmd_active: bool, wz_y: float, wz_z: float, wz_quat: float) -> str:
        """
        Auto selection: when cmd_active, pick the most energetic source.
        Use hysteresis so we don't switch too often.
        """
        now = self._now()

        # if not commanded, keep last (doesn't matter much because we won't integrate yaw anyway)
        if not cmd_active:
            return self._wz_src_locked

        # Hold current source for a minimum time
        held = (now - self._wz_src_lock_time).nanoseconds * 1e-9
        if held < self.yawrate_auto_hold_sec:
            return self._wz_src_locked

        # "Energy" proxy: absolute value (you can replace by EMA/variance if you want)
        ay = abs(wz_y)
        az = abs(wz_z)
        aq = abs(wz_quat)

        # Choose max-energy source
        best = "z"
        bestv = az
        if ay > bestv:
            best, bestv = "y", ay
        if aq > bestv:
            best, bestv = "quat", aq

        # Lock it
        self._wz_src_locked = best
        self._wz_src_lock_time = now
        return best

    def _get_wz_meas(self, odom: Odometry, cmd_active: bool) -> Tuple[str, float, float, float, float]:
        """
        Return (wz_src, wz_used, wz_y, wz_z, wz_quat)
        """
        wy = float(odom.twist.twist.angular.y)
        wz = float(odom.twist.twist.angular.z)
        wq = float(self._wz_from_quat(odom))

        if self.yawrate_source == "y":
            src = "y"
        elif self.yawrate_source == "z":
            src = "z"
        elif self.yawrate_source == "quat":
            src = "quat"
        else:
            src = self._pick_wz_source_auto(cmd_active, wy, wz, wq)

        used = {"y": wy, "z": wz, "quat": wq}[src]
        return src, used, wy, wz, wq

    # ============================================================
    # Twist correction + filtering
    # ============================================================
    def _correct_and_filter_twist(self, odom: Odometry, cmd_active: bool) -> Tuple[float, float, float, str]:
        """
        Returns filtered (vx, vy, wz) in base frame + wz_src.
        """
        vx = float(odom.twist.twist.linear.x)
        vy = float(odom.twist.twist.linear.y)

        wz_src, wz_used, wz_y, wz_z, wz_quat = self._get_wz_meas(odom, cmd_active)

        # Save for logs/status
        self._last_wz_src = wz_src
        self._last_wz_y = wz_y
        self._last_wz_z = wz_z
        self._last_wz_quat = wz_quat
        self._last_dyaw_dt = wz_quat  # dyaw/dt == wz_quat

        if not self.enable_filter:
            return vx, (vy if self.enable_vy else 0.0), wz_used, wz_src

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

        # EMA
        self._vx_f = self.ema_alpha * vx + (1.0 - self.ema_alpha) * self._vx_f
        if self.enable_vy:
            a = self.ema_alpha_vy
            self._vy_f = a * vy + (1.0 - a) * self._vy_f
        else:
            self._vy_f = 0.0
        self._wz_f = self.ema_alpha * wz_used + (1.0 - self.ema_alpha) * self._wz_f

        return self._vx_f, self._vy_f, self._wz_f, wz_src

    def _apply_turn_in_place_rule(self, vx: float, vy: float, cmd_vx: float, cmd_wz: float) -> Tuple[float, float, bool]:
        """
        If Nav2 commands a turn-in-place, force translation to 0.
        """
        turn = (abs(cmd_wz) > self.turn_wz_thresh) and (abs(cmd_vx) < self.turn_vx_small)
        if turn:
            return 0.0, 0.0, True
        return vx, vy, False

    # ============================================================
    # SE(2) integration
    # ============================================================
    def _integrate_se2(self, vx: float, vy: float, wz: float, integrate_yaw: bool) -> None:
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
    # TF publishing
    # ============================================================
    def _publish_tf_tick(self) -> None:
        if self._last_odom is None:
            return

        stamp = self._stamp_msg()

        # 1) odom -> base_footprint (planar)
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

        # 2) base_footprint -> robot_center (roll/pitch only, yaw=0)
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
        tf_body.transform.translation.z = float(p.z)
        tf_body.transform.rotation.x = bqx
        tf_body.transform.rotation.y = bqy
        tf_body.transform.rotation.z = bqz
        tf_body.transform.rotation.w = bqw
        self.tf_broadcaster.sendTransform(tf_body)

    # ============================================================
    # Odom publishing (EKF-friendly)
    # ============================================================
    def _publish_odom_tick(self) -> None:
        if self._last_odom is None:
            return

        gate = self._decide_gate()
        self._last_gate = gate
        cmd_active = gate.integrate

        # Command intent
        vx_cmd = float(gate.cmd_vx)
        wz_cmd = float(gate.cmd_wz)

        # Measured twist
        vx_meas, vy_meas, wz_meas, wz_src = self._correct_and_filter_twist(self._last_odom, cmd_active)

        # Turn-in-place: forbid translation
        vx_meas, vy_meas, turn_in_place_cmd = self._apply_turn_in_place_rule(vx_meas, vy_meas, vx_cmd, wz_cmd)

        # Evidence checks
        rot_cmd_active = cmd_active and (abs(wz_cmd) > self.cmd_eps_wz)
        yaw_meas_active = abs(wz_meas) > self.meas_eps_wz

        xy_meas_active = cmd_active and (
            abs(vx_meas) > self.meas_eps_vx or (self.enable_vy and abs(vy_meas) > self.meas_eps_vy)
        )
        integrate_xy = (not turn_in_place_cmd) and xy_meas_active

        # Rotation confirmation window: allow cmd yaw briefly until measurement follows
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

        # Never integrate yaw unless command is active
        integrate_yaw = cmd_active and (yaw_meas_active or allow_cmd_yaw_fallback)

        vx_int = vx_meas if integrate_xy else 0.0
        vy_int = vy_meas if (integrate_xy and self.enable_vy) else 0.0
        wz_int = wz_for_yaw if integrate_yaw else 0.0

        self._last_twist_vx = vx_int
        self._last_twist_wz = wz_int

        # Integrate
        if integrate_xy or integrate_yaw:
            self._integrate_se2(vx=vx_int, vy=vy_int, wz=wz_int, integrate_yaw=integrate_yaw)

        # Pose to publish
        x, y, yaw = self._select_pose()
        qx, qy, qz, qw = _quat_from_yaw(yaw)

        # Publish twist consistent with integration decision
        vx_pub = vx_meas if integrate_xy else 0.0
        vy_pub = vy_meas if (integrate_xy and self.enable_vy) else 0.0
        wz_pub = wz_for_yaw if integrate_yaw else 0.0

        out = Odometry()
        out.header.stamp = self._stamp_msg()
        out.header.frame_id = self.odom_frame
        out.child_frame_id = self.base_frame

        out.pose.pose.position.x = float(x)
        out.pose.pose.position.y = float(y)
        out.pose.pose.position.z = 0.0
        out.pose.pose.orientation.x = qx
        out.pose.pose.orientation.y = qy
        out.pose.pose.orientation.z = qz
        out.pose.pose.orientation.w = qw

        # EKF-friendly (not perfect, but NOT all zeros)
        # Tune these if you know your real noise.
        out.pose.covariance = [0.0] * 36
        out.pose.covariance[0]  = 0.05  # x
        out.pose.covariance[7]  = 0.05  # y
        out.pose.covariance[14] = 999.0 # z (unused in planar)
        out.pose.covariance[21] = 999.0 # roll
        out.pose.covariance[28] = 999.0 # pitch
        out.pose.covariance[35] = 0.10  # yaw

        tw = Twist()
        tw.linear.x = float(vx_pub)
        tw.linear.y = float(vy_pub)
        tw.angular.z = float(wz_pub)

        twc = TwistWithCovariance()
        twc.twist = tw
        twc.covariance = [0.0] * 36
        twc.covariance[0]  = 0.10 # vx
        twc.covariance[7]  = 0.10 # vy
        twc.covariance[35] = 0.20 # wz
        out.twist = twc

        self.odom_pub.publish(out)

        # Logs (clear)
        now_s = time.time()
        if (now_s - self._last_info_log_time) >= self._info_log_period:
            self._last_info_log_time = now_s
            self.get_logger().info(
                "gate=%d(%s) xy=%d yaw=%d turn=%d "
                "wz_src=%s wz_y=%+.3f wz_z=%+.3f wz_quat(dyaw/dt)=%+.3f wz_used=%+.3f "
                "cmd(vx,wz)=(%+.2f,%+.2f) meas(vx,vy,wz)=(%+.2f,%+.2f,%+.2f) "
                "int(vx,vy,wz)=(%+.2f,%+.2f,%+.2f) pose(x,y,yaw)=(%+.2f,%+.2f,%+.2f)"
                % (
                    int(cmd_active), gate.reason,
                    int(integrate_xy), int(integrate_yaw), int(turn_in_place_cmd),
                    wz_src, self._last_wz_y, self._last_wz_z, self._last_wz_quat, wz_for_yaw,
                    vx_cmd, wz_cmd,
                    vx_meas, vy_meas, wz_meas,
                    vx_int, vy_int, wz_int,
                    x, y, yaw
                )
            )

    # ============================================================
    # Status (JSON-ish)
    # ============================================================
    def _publish_status_tick(self) -> None:
        now = self._now()

        odom_age = 999.0
        if self._last_odom_time is not None:
            odom_age = (now - self._last_odom_time).nanoseconds * 1e-9

        gate = self._last_gate
        x, y, yaw = self._select_pose()

        msg = String()
        msg.data = (
            "{"
            f"\"pose_source\":\"{self.pose_source}\","
            f"\"yawrate_source\":\"{self.yawrate_source}\","
            f"\"wz_src\":\"{self._last_wz_src}\","
            f"\"wz_y\":{self._last_wz_y:.4f},"
            f"\"wz_z\":{self._last_wz_z:.4f},"
            f"\"wz_quat\":{self._last_wz_quat:.4f},"
            f"\"dyaw_dt\":{self._last_dyaw_dt:.4f},"
            f"\"integrate\":{1 if gate.integrate else 0},"
            f"\"reason\":\"{gate.reason}\","
            f"\"odom_age\":{odom_age:.3f},"
            f"\"cmd_vx\":{gate.cmd_vx:.3f},"
            f"\"cmd_wz\":{gate.cmd_wz:.3f},"
            f"\"twist_vx\":{self._last_twist_vx:.3f},"
            f"\"twist_wz\":{self._last_twist_wz:.3f},"
            f"\"x\":{x:.3f},"
            f"\"y\":{y:.3f},"
            f"\"yaw\":{yaw:.3f}"
            "}"
        )
        self.status_pub.publish(msg)


def main() -> None:
    rclpy.init()
    node = RealOdomTfBridgeCmdGated()
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