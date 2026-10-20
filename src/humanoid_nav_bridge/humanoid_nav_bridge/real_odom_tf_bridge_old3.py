#!/usr/bin/env python3
"""
real_odom_tf_bridge.py (ROS 2 Humble)

PRODUCTION odom + TF bridge for floating-base humanoids (Unitree G1 etc.)
with command-gated integration + optional driver pose passthrough.

Why "shorter" can still be "more complete"
------------------------------------------
Some previous versions had duplicated logic (yaw integrated twice), risky projections
(sqrt(vx^2+vy^2)), or partially unused code paths.

This PRO version is "complete" but intentionally structured:
- clear separation of: gating, twist correction, pose source choice, integration, publishing
- optional: use driver pose OR integrated pose OR hybrid (driver xy + integrated yaw)
- always publishes TF + /odom (pose can be frozen when idle)

Inputs
------
- /dog_odom        (nav_msgs/Odometry)
- /cmd_vel_rlsar   (geometry_msgs/Twist)

Outputs
-------
- /odom (nav_msgs/Odometry)
- TF: odom_frame -> base_frame
- /odom_bridge/status (std_msgs/String): JSON-ish debug telemetry

Key parameters
--------------
pose_source: "integrate" | "driver" | "hybrid"
  - integrate: x,y,yaw from integrating twist (recommended for RL humanoids)
  - driver   : x,y,yaw from driver pose (but can be frozen when idle)
  - hybrid   : x,y from driver pose, yaw from integrated yaw

remap_yawrate_y_to_z (bool):
  - True  => use angular.y as yaw rate
  - False => use angular.z as yaw rate

Command gating:
  integrate if |cmd_vx|>cmd_eps_vx OR |cmd_wz|>cmd_eps_wz and cmd is fresh (cmd_timeout_sec)
  else freeze pose (and optionally freeze yaw).

Author: Jean Chrysostome Mayoko Biong
"""

from __future__ import annotations

import math
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
import time

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

# ============================================================
# RPY helpers (no external deps)
# ============================================================

def _rpy_from_quat(x: float, y: float, z: float, w: float) -> Tuple[float, float, float]:
    """
    Convert quaternion -> roll, pitch, yaw (REP-103, ZYX convention).

    This is needed because a floating-base humanoid base frame typically
    has continuous roll/pitch oscillations from balancing / walking.

    We will use:
      - yaw for the planar navigation frame (base_footprint)
      - roll/pitch for the body frame (robot_center)
    """
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw

def _log_warn_throttle(self, text: str, period_s: float = 2.0) -> None:
    now_s = time.time()
    key = "_last_warn_log_time"
    last = getattr(self, key, 0.0)
    if (now_s - last) >= period_s:
        setattr(self, key, now_s)
        self.get_logger().warn(text)


def _quat_from_rpy(roll: float, pitch: float, yaw: float) -> Tuple[float, float, float, float]:
    """
    Convert roll, pitch, yaw -> quaternion (ZYX convention).

    We'll use this to publish base_footprint->robot_center with yaw=0,
    keeping roll/pitch only.
    """
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)

    # ZYX: q = qz * qy * qx
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

        # ----------------------------------------------------
        # Topics / frames
        # ----------------------------------------------------
        self.input_odom_topic = self.declare_parameter("input_odom_topic", "/dog_odom").value
        self.cmd_vel_topic = self.declare_parameter("cmd_vel_topic", "/cmd_vel_rlsar").value
        self.output_odom_topic = self.declare_parameter("output_odom_topic", "/odom").value
        # Topic used ONLY for gating (Nav2 intent)
        self.gate_cmd_vel_topic = self.declare_parameter("gate_cmd_vel_topic", "/cmd_vel_nav").value

        # Frames:
        #   - base_frame is the planar frame used by Nav2 (recommended: base_footprint)
        #   - body_frame is the real floating body frame (recommended: robot_center)
        self.odom_frame = self.declare_parameter("odom_frame", "odom").value

        # Nav2 base frame (planar)
        self.base_frame = self.declare_parameter("base_frame", "base_footprint").value

        # Real body frame (roll/pitch) published as a child of base_frame
        self.body_frame = self.declare_parameter("body_frame", "robot_center").value

        # ----------------------------------------------------
        # Pose source choice (this is what you felt "missing")
        # ----------------------------------------------------
        # integrate: x,y,yaw from twist integration (recommended humanoid)
        # driver   : x,y,yaw from driver pose (can be frozen)
        # hybrid   : x,y from driver, yaw from integration
        self.pose_source = str(self.declare_parameter("pose_source", "integrate").value).strip().lower()
        if self.pose_source not in ("integrate", "driver", "hybrid"):
            self.get_logger().warn(f"Invalid pose_source='{self.pose_source}', forcing 'integrate'")
            self.pose_source = "integrate"

        # Optional: subtract initial pose so Nav2 starts near (0,0,0)
        self.zero_at_start = bool(self.declare_parameter("zero_at_start", True).value)
        self._zero_ready = False
        self._x0 = 0.0
        self._y0 = 0.0
        self._yaw0 = 0.0

        # ----------------------------------------------------
        # Time policy
        # ----------------------------------------------------
        #self.use_node_time = bool(self.declare_parameter("use_node_time", True).value)
        self.use_node_time = bool(self.declare_parameter("use_node_time", False).value)
        self.rot_confirm_timeout = float(self.declare_parameter("rot_confirm_timeout", 0.35).value)
        self._rot_cmd_start_time = None  # type: Optional[rclpy.time.Time]

        # ----------------------------------------------------
        # Keep control: yaw-rate remap
        # ----------------------------------------------------
        #self.remap_yawrate_y_to_z = bool(self.declare_parameter("remap_yawrate_y_to_z", False).value)
        self.remap_yawrate_y_to_z = bool(self.declare_parameter("remap_yawrate_y_to_z", True).value)

        # ----------------------------------------------------
        # Command gating
        # ----------------------------------------------------
        self.cmd_eps_vx = float(self.declare_parameter("cmd_eps_vx", 0.02).value)
        self.cmd_eps_wz = float(self.declare_parameter("cmd_eps_wz", 0.02).value)
        self.cmd_timeout_sec = float(self.declare_parameter("cmd_timeout_sec", 0.25).value)

        # Freeze yaw when idle (recommended)
        self.freeze_yaw_when_idle = bool(self.declare_parameter("freeze_yaw_when_idle", True).value)

        # ----------------------------------------------------
        # Turn-in-place rule
        # ----------------------------------------------------
        self.turn_wz_thresh = float(self.declare_parameter("turn_wz_thresh", 0.25).value)
        self.turn_vx_small = float(self.declare_parameter("turn_vx_small", 0.04).value)

        # ----------------------------------------------------
        # Filtering measured twist
        # ----------------------------------------------------
        self.enable_filter = bool(self.declare_parameter("enable_filter", True).value)
        self.deadband_vx = float(self.declare_parameter("deadband_vx", 0.02).value)
        self.deadband_wz = float(self.declare_parameter("deadband_wz", 0.02).value)
        self.ema_alpha = float(self.declare_parameter("ema_alpha", 0.25).value)

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

        # ----------------------------------------------------
        # Watchdog
        # ----------------------------------------------------
        self.timeout_sec = float(self.declare_parameter("timeout_sec", 1.0).value)

        # ----------------------------------------------------
        # QoS
        # ----------------------------------------------------
        # QoS knobs (separated: odom vs cmd)
        self.odom_sub_reliable     = bool(self.declare_parameter("odom_sub_reliable", True).value)
        self.cmd_sub_reliable      = bool(self.declare_parameter("cmd_sub_reliable", False).value)
        # QoS for the gate cmd subscription
        self.gate_cmd_sub_reliable = bool(self.declare_parameter("gate_cmd_sub_reliable", True).value) 


        self.pub_reliable      = bool(self.declare_parameter("pub_reliable", True).value)
        self.qos_depth         = int(self.declare_parameter("qos_depth", 10).value)
        # (Optionnel) deducated depth, 1 is okay for cmd_vel
        self.gate_cmd_qos_depth = int(self.declare_parameter(
            "gate_cmd_qos_depth", 1
        ).value)

        odom_sub_qos     = _make_qos(self.odom_sub_reliable, self.qos_depth)
        cmd_sub_qos      = _make_qos(self.cmd_sub_reliable,  self.qos_depth)
        cmd_gate_qos     = _make_qos(self.gate_cmd_sub_reliable, self.gate_cmd_qos_depth)
        pub_qos          = _make_qos(self.pub_reliable,      self.qos_depth)




        # ----------------------------------------------------
        # ROS interfaces
        # ----------------------------------------------------
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        # Publishers
        self.odom_pub   = self.create_publisher(Odometry, self.output_odom_topic, pub_qos)
        self.status_pub = self.create_publisher(String, "/odom_bridge/status", pub_qos)

        # Subscriptions (different QoS!)
        self.odom_sub = self.create_subscription(Odometry, self.input_odom_topic, self._cb_odom, odom_sub_qos)
        self.cmd_sub  = self.create_subscription(Twist,    self.cmd_vel_topic,    self._cb_cmd,  cmd_sub_qos)
        self.gate_sub = self.create_subscription(Twist, self.gate_cmd_vel_topic, self._cb_gate_cmd, cmd_gate_qos)



        self.get_logger().info(
            f"QoS: odom_sub={'RELIABLE' if self.odom_sub_reliable else 'BEST_EFFORT'} | "
            f"cmd_sub={'RELIABLE' if self.cmd_sub_reliable else 'BEST_EFFORT'} | "
            f"pub={'RELIABLE' if self.pub_reliable else 'BEST_EFFORT'} | depth={self.qos_depth}"
        )
        # ----------------------------------------------------
        # State: last messages + times
        # ----------------------------------------------------
        self._last_odom: Optional[Odometry] = None
        self._last_cmd: Optional[Twist] = None
        self._last_odom_time: Optional[Time] = None
        self._last_cmd_time: Optional[Time] = None
        # 
        self._last_gate_cmd: Optional[Twist] = None
        self._last_gate_cmd_time: Optional[Time] = None

        # ----------------------------------------------------
        # State: integrated SE(2)
        # ----------------------------------------------------
        self._x_int = 0.0
        self._y_int = 0.0
        self._yaw_int = 0.0
        self._last_integration_time = self.get_clock().now()

        # EMA filter state
        self._vx_f = 0.0
        self._wz_f = 0.0

        self.meas_eps_vx = float(self.declare_parameter("meas_eps_vx", 0.03).value)
        self.meas_eps_wz = float(self.declare_parameter("meas_eps_wz", 0.05).value)

        # Debug
        self._last_gate = GateDecision(False, "init", 999.0, 0.0, 0.0)
        self._last_twist_vx = 0.0
        self._last_twist_wz = 0.0

        # ----------------------------------------------------
        # Timers
        # ----------------------------------------------------
        self._watchdog_timer = self.create_timer(0.5, self._watchdog_tick)

        self._tf_timer = None
        if self.publish_tf and self.tf_rate_hz > 0:
            self._tf_timer = self.create_timer(1.0 / self.tf_rate_hz, self._publish_tf_tick)

        self._odom_timer = None
        if self.publish_odom and self.odom_rate_hz > 0:
            self._odom_timer = self.create_timer(1.0 / self.odom_rate_hz, self._publish_odom_tick)

        self._status_timer = None
        if self.publish_status and self.status_rate_hz > 0:
            self._status_timer = self.create_timer(1.0 / self.status_rate_hz, self._publish_status_tick)

        self._last_info_log_time = 0.0
        self._info_log_period = 2.0  # seconds



        # ----------------------------------------------------
        # Startup log
        # ----------------------------------------------------
        self.get_logger().info(
            "Cmd-gated PRO bridge (FULL) started:\n"
            f"  pose_source={self.pose_source} (integrate|driver|hybrid)\n"
            f"  /dog_odom={self.input_odom_topic}\n"
            f"  cmd_vel={self.cmd_vel_topic}\n"
            f"  /odom={self.output_odom_topic}\n"
            f"  frames: {self.odom_frame}->{self.base_frame}\n"
            f"  remap_yawrate_y_to_z={self.remap_yawrate_y_to_z}\n"
            f"  gating: eps_vx={self.cmd_eps_vx}, eps_wz={self.cmd_eps_wz}, cmd_timeout={self.cmd_timeout_sec}s\n"
            f"  freeze_yaw_when_idle={self.freeze_yaw_when_idle}\n"
            f"  turn-in-place: wz>{self.turn_wz_thresh} & vx<{self.turn_vx_small} => force vx=0\n"
        )

        # ----------------------------------------------------
        # Optional lateral velocity handling (vy)
        # ----------------------------------------------------
        # Humanoid odom can have non-zero lateral velocity due to sway / foot slippage.
        # Nav2 is 2D, but if vy is real (not only noise), ignoring it can create drift.
        self.enable_vy = bool(self.declare_parameter("enable_vy", False).value)

        # Deadband and EMA for vy (to remove sway noise)
        self.deadband_vy = float(self.declare_parameter("deadband_vy", 0.02).value)
        self.ema_alpha_vy = float(self.declare_parameter("ema_alpha_vy", self.ema_alpha).value)

        # Minimum measured vy to accept as real motion when commanded
        self.meas_eps_vy = float(self.declare_parameter("meas_eps_vy", 0.03).value)

        # EMA state for vy
        self._vy_f = 0.0

        # Anti-picks / anti-chatter (optional)
        self.gate_on_hold_sec  = float(self.declare_parameter("gate_on_hold_sec", 0.0).value)   # ex: 0.05
        self.gate_off_hold_sec = float(self.declare_parameter("gate_off_hold_sec", 0.10).value) # ex: 0.10

        self._gate_state = False
        self._gate_last_switch_time = self._now()

    # ============================================================
    # Callbacks
    # ============================================================
    def _cb_odom(self, msg: Odometry) -> None:
        self._last_odom = msg
        self._last_odom_time = self.get_clock().now()

        # latch zero-at-start using driver pose (only for offsets)
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
        self._last_gate_cmd_time = self._now()

    # et dans _decide_gate(): utiliser _last_gate_cmd au lieu de _last_cmd

    def _cb_cmd(self, msg: Twist) -> None:
        self._last_cmd = msg
        self._last_cmd_time = self.get_clock().now()

    # ============================================================
    # Time helpers
    # ============================================================
    def _now(self) -> Time:
        return self.get_clock().now()

    def _stamp_msg_old(self):
        return self._now().to_msg() if self.use_node_time else self._now().to_msg()

    def _stamp_msg(self):
        """
        Return a timestamp consistent with sensor time when possible.

        Priority:
        1) Use last received odom header stamp (best for Nav2 / AMCL coherence)
        2) Fallback to node clock if no odom yet
        """

        if not self.use_node_time and self._last_odom is not None:
            return self._last_odom.header.stamp

        # Fallback
        return self.get_clock().now().to_msg()

    def _log_info_throttle(self, text: str, period_s: float = 2.0) -> None:
        """Simple manual throttle for rclpy logger (Humble has no throttle arg)."""
        now_s = time.time()
        last = getattr(self, "_last_info_log_time", 0.0)
        if (now_s - last) >= period_s:
            self._last_info_log_time = now_s
            self.get_logger().info(text)

    def _log_warn_throttle(self, text: str, period_s: float = 2.0) -> None:
        """Manual throttle for warn logs (rclpy has no throttle keyword arg)."""
        now_s = time.time()
        last = getattr(self, "_last_warn_log_time", 0.0)
        if (now_s - last) >= period_s:
            self._last_warn_log_time = now_s
            self.get_logger().warn(text)
            
    # ============================================================
    # Gating decision
    # ============================================================
    def _decide_gate_old(self) -> GateDecision:
        if self._last_gate_cmd is None or self._last_gate_cmd_time is None:
            return GateDecision(False, "no_cmd_yet", 999.0, 0.0, 0.0)

        cmd_age = (self._now() - self._last_gate_cmd_time).nanoseconds * 1e-9
        cmd_vx = float(self._last_gate_cmd.linear.x)
        cmd_wz = float(self._last_gate_cmd.angular.z)

        if cmd_age > self.cmd_timeout_sec:
            return GateDecision(False, "cmd_stale", cmd_age, cmd_vx, cmd_wz)

        if (abs(cmd_vx) > self.cmd_eps_vx) or (abs(cmd_wz) > self.cmd_eps_wz):
            return GateDecision(True, "cmd_active", cmd_age, cmd_vx, cmd_wz)

        return GateDecision(False, "cmd_below_eps", cmd_age, cmd_vx, cmd_wz)

    def _decide_gate(self) -> GateDecision:
        if self._last_gate_cmd is None or self._last_gate_cmd_time is None:
            self._gate_state = False
            return GateDecision(False, "no_cmd_yet", 999.0, 0.0, 0.0)

        now = self._now()
        cmd_age = (now - self._last_gate_cmd_time).nanoseconds * 1e-9

        cmd_vx = float(self._last_gate_cmd.linear.x)

        # B) respect same yaw convention as the rest of the node
        cmd_wz = float(
            self._last_gate_cmd.angular.y if self.remap_yawrate_y_to_z
            else self._last_gate_cmd.angular.z
        )

        # stale -> force OFF
        if cmd_age > self.cmd_timeout_sec:
            if self._gate_state:
                # allow OFF only if hold satisfied
                since = (now - self._gate_last_switch_time).nanoseconds * 1e-9
                if since < self.gate_off_hold_sec:
                    return GateDecision(True, "cmd_stale_hold", cmd_age, cmd_vx, cmd_wz)

            self._gate_state = False
            self._gate_last_switch_time = now
            return GateDecision(False, "cmd_stale", cmd_age, cmd_vx, cmd_wz)

        # base gate condition (your original eps logic)
        active = (abs(cmd_vx) > self.cmd_eps_vx) or (abs(cmd_wz) > self.cmd_eps_wz)

        # A) Anti-picks / anti-chatter via temporal hysteresis
        since = (now - self._gate_last_switch_time).nanoseconds * 1e-9

        if active and not self._gate_state:
            # OFF -> ON only if held long enough
            if self.gate_on_hold_sec > 0.0 and since < self.gate_on_hold_sec:
                return GateDecision(False, "cmd_active_hold_on", cmd_age, cmd_vx, cmd_wz)
            self._gate_state = True
            self._gate_last_switch_time = now
            return GateDecision(True, "cmd_active", cmd_age, cmd_vx, cmd_wz)

        if (not active) and self._gate_state:
            # ON -> OFF only if held long enough
            if self.gate_off_hold_sec > 0.0 and since < self.gate_off_hold_sec:
                return GateDecision(True, "cmd_below_eps_hold_off", cmd_age, cmd_vx, cmd_wz)
            self._gate_state = False
            self._gate_last_switch_time = now
            return GateDecision(False, "cmd_below_eps", cmd_age, cmd_vx, cmd_wz)

        # unchanged state
        return GateDecision(self._gate_state, "cmd_active" if self._gate_state else "cmd_below_eps",
                            cmd_age, cmd_vx, cmd_wz)


    # ============================================================
    # Twist correction + filtering 
    # ============================================================
    def _correct_and_filter_twist_old(self, odom: Odometry) -> Tuple[float, float]:
        vx = float(odom.twist.twist.linear.x)

        wy = float(odom.twist.twist.angular.y)
        wz = float(odom.twist.twist.angular.z)
        wz_used = wy if self.remap_yawrate_y_to_z else wz

        if not self.enable_filter:
            return vx, wz_used

        if abs(vx) < self.deadband_vx:
            vx = 0.0
        if abs(wz_used) < self.deadband_wz:
            wz_used = 0.0

        self._vx_f = self.ema_alpha * vx + (1.0 - self.ema_alpha) * self._vx_f
        self._wz_f = self.ema_alpha * wz_used + (1.0 - self.ema_alpha) * self._wz_f

        return self._vx_f, self._wz_f
    
    def _correct_and_filter_twist(self, odom: Odometry) -> Tuple[float, float, float]:
        """
        Returns a filtered planar twist (vx, vy, wz) expressed in the robot base frame.

        Notes:
        - vx, vy are assumed to be base-frame velocities (as in nav_msgs/Odometry).
        - wz is yaw rate (angular.z by default, optionally angular.y if remapped).
        - We apply deadbands + EMA to suppress humanoid sway noise.
        """
        vx = float(odom.twist.twist.linear.x)
        vy = float(odom.twist.twist.linear.y)

        wy = float(odom.twist.twist.angular.y)
        wz = float(odom.twist.twist.angular.z)
        wz_used = wy if self.remap_yawrate_y_to_z else wz

        if not self.enable_filter:
            # If filtering is disabled, we still honor enable_vy by either returning
            # the raw vy or forcing it to zero.
            return vx, (vy if self.enable_vy else 0.0), wz_used

        # -------------------------
        # Deadbands (noise removal)
        # -------------------------
        if abs(vx) < self.deadband_vx:
            vx = 0.0

        if self.enable_vy:
            if abs(vy) < self.deadband_vy:
                vy = 0.0
        else:
            vy = 0.0

        if abs(wz_used) < self.deadband_wz:
            wz_used = 0.0

        # -------------------------
        # EMA low-pass filter
        # -------------------------
        self._vx_f = self.ema_alpha * vx + (1.0 - self.ema_alpha) * self._vx_f

        if self.enable_vy:
            a = self.ema_alpha_vy
            self._vy_f = a * vy + (1.0 - a) * self._vy_f
        else:
            self._vy_f = 0.0

        self._wz_f = self.ema_alpha * wz_used + (1.0 - self.ema_alpha) * self._wz_f

        return self._vx_f, self._vy_f, self._wz_f

    def _apply_turn_in_place_rule_old(self, vx: float) -> float:
        if self._last_cmd is None:
            return vx
        cmd_vx = float(self._last_cmd.linear.x)
        cmd_wz = float(self._last_cmd.angular.z)
        if abs(cmd_wz) > self.turn_wz_thresh and abs(cmd_vx) < self.turn_vx_small:
            return 0.0
        return vx
    
    def _apply_turn_in_place_rule(self, vx: float, vy: float) -> Tuple[float, float]:

        """
        If Nav2 is commanding a turn-in-place, force translation to zero.
        This avoids integrating tiny lateral drift while rotating.
        """
                
        msg = self._last_gate_cmd   # <-- au lieu de _last_cmd
        if msg is None:
            return vx, vy
        
        cmd_vx = float(msg.linear.x)
        cmd_wz = float(msg.angular.y if self.remap_yawrate_y_to_z else msg.angular.z)

        if abs(cmd_wz) > self.turn_wz_thresh and abs(cmd_vx) < self.turn_vx_small:
            return 0.0, 0.0
        return vx, vy

    # ============================================================
    # SE(2) integration
    # ============================================================
    def _integrate_se2_old(self, vx: float, wz: float, integrate_yaw: bool) -> None:
        now = self._now()
        dt = (now - self._last_integration_time).nanoseconds * 1e-9
        self._last_integration_time = now

        if dt < 0.0 or dt > self.dt_max:
            return

        if integrate_yaw:
            self._yaw_int = _clamp_angle_pi(self._yaw_int + wz * dt)

        self._x_int += vx * math.cos(self._yaw_int) * dt
        self._y_int += vx * math.sin(self._yaw_int) * dt

    def _integrate_se2(self, vx: float, vy: float, wz: float, integrate_yaw: bool) -> None:
        """
        Integrate a planar SE(2) state (x, y, yaw) using base-frame twist.

        The twist (vx, vy, wz) is assumed to be expressed in the robot base frame:
        - vx: forward (m/s)
        - vy: left    (m/s)   (optional; can be forced to 0 for diff-drive-like behavior)
        - wz: yaw rate (rad/s)

        World-frame increments:
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
        Returns (x, y, yaw) according to pose_source:
          integrate: from (_x_int, _y_int, _yaw_int)
          driver   : from last odom pose
          hybrid   : x,y from driver pose, yaw from _yaw_int
        Applies zero_at_start offsets if enabled.
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
    def _watchdog_tick_old(self) -> None:
        now = self._now()

        if self._last_odom_time is None:
            self.get_logger().warn("No /dog_odom received yet...", throttle_duration_sec=2.0)
        else:
            odom_age = (now - self._last_odom_time).nanoseconds * 1e-9
            if odom_age > self.timeout_sec:
                self.get_logger().warn(f"/dog_odom stale: age={odom_age:.2f}s", throttle_duration_sec=2.0)

        if self._last_cmd_time is None:
            self.get_logger().warn(f"No {self.cmd_vel_topic} yet -> bridge will freeze.", throttle_duration_sec=2.0)

    def _watchdog_tick(self) -> None:
        now = self._now()

        if self._last_odom_time is None:
            self._log_warn_throttle("No /dog_odom received yet...", period_s=2.0)
        else:
            odom_age = (now - self._last_odom_time).nanoseconds * 1e-9
            if odom_age > self.timeout_sec:
                self._log_warn_throttle(f"/dog_odom stale: age={odom_age:.2f}s", period_s=2.0)

        if self._last_cmd_time is None:
            self._log_warn_throttle(f"No {self.cmd_vel_topic} yet -> bridge will freeze.", period_s=2.0)
            
    # ============================================================
    # Publishing
    # ============================================================
    def _publish_tf_tick_old(self) -> None:
        if self._last_odom is None:
            return

        x, y, yaw = self._select_pose()
        qx, qy, qz, qw = _quat_from_yaw(yaw)

        tfm = TransformStamped()
        #tfm.header.stamp = self._last_odom.header.stamp
        tfm.header.stamp = self._stamp_msg()
        tfm.header.frame_id = self.odom_frame
        tfm.child_frame_id = self.base_frame
        tfm.transform.translation.x = float(x)
        tfm.transform.translation.y = float(y)
        tfm.transform.translation.z = 0.0
        tfm.transform.rotation.x = qx
        tfm.transform.rotation.y = qy
        tfm.transform.rotation.z = qz
        tfm.transform.rotation.w = qw

        self.tf_broadcaster.sendTransform(tfm)


    def _publish_tf_tick(self) -> None:
        """
        Publish a TF chain suitable for Nav2 on a floating-base humanoid.

        Why two frames?
        --------------
        Nav2 assumes a planar robot model:
        - roll = 0
        - pitch = 0
        - z = 0
        - only x,y,yaw matter

        A humanoid stabilized by an RL policy continuously oscillates in roll/pitch,
        especially while walking. If Nav2 uses that oscillating frame as its base,
        the laser scan gets "tilted" in TF transforms, generating fake obstacles.

        Therefore we publish:
        1) odom -> base_footprint  (PLANAR: x,y,yaw only)
        2) base_footprint -> robot_center (BODY: roll/pitch/z only, yaw=0)
        """
        if self._last_odom is None:
            return

        stamp = self._stamp_msg()

        # -----------------------------
        # 1) Planar pose for navigation
        # -----------------------------
        # This pose is computed by your selected pose source (integrate/driver/hybrid),
        # but it MUST be planar for Nav2: x,y,yaw only.
        x, y, yaw = self._select_pose()
        qx, qy, qz, qw = _quat_from_yaw(yaw)

        tf_planar = TransformStamped()
        tf_planar.header.stamp = stamp
        tf_planar.header.frame_id = self.odom_frame
        tf_planar.child_frame_id = self.base_frame  # e.g., base_footprint
        tf_planar.transform.translation.x = float(x)
        tf_planar.transform.translation.y = float(y)
        tf_planar.transform.translation.z = 0.0     # planar nav frame
        tf_planar.transform.rotation.x = qx
        tf_planar.transform.rotation.y = qy
        tf_planar.transform.rotation.z = qz
        tf_planar.transform.rotation.w = qw

        self.tf_broadcaster.sendTransform(tf_planar)

        # ---------------------------------------
        # 2) Body frame carrying roll/pitch motion
        # ---------------------------------------
        # We take roll/pitch/z from the *driver odom pose* (not the integrated planar pose).
        # The key point: yaw must be 0 here to avoid "double yaw" (odom->base + base->body).
        p = self._last_odom.pose.pose.position
        q = self._last_odom.pose.pose.orientation
        roll, pitch, _yaw_driver = _rpy_from_quat(float(q.x), float(q.y), float(q.z), float(q.w))

        # body transform keeps roll/pitch, but yaw=0
        bqx, bqy, bqz, bqw = _quat_from_rpy(roll, pitch, 0.0)

        tf_body = TransformStamped()
        tf_body.header.stamp = stamp
        tf_body.header.frame_id = self.base_frame   # base_footprint
        tf_body.child_frame_id = self.body_frame    # robot_center
        tf_body.transform.translation.x = 0.0
        tf_body.transform.translation.y = 0.0
        tf_body.transform.translation.z = float(p.z)  # optional (safe for 2D nav)
        tf_body.transform.rotation.x = bqx
        tf_body.transform.rotation.y = bqy
        tf_body.transform.rotation.z = bqz
        tf_body.transform.rotation.w = bqw

        self.tf_broadcaster.sendTransform(tf_body)


    def _publish_odom_tick_old(self) -> None:
        """
        Publish Nav2-compatible /odom at a fixed rate.

        Humanoid RL design principle
        -----------------------------
        For floating-base humanoids stabilized by an RL policy:

        - The commanded velocity (/cmd_vel_rlsar) is NOT guaranteed
        to immediately produce real motion (deadzone, contact,
        balance recovery, etc.).
        - Therefore integrating pose from cmd_vel directly can cause:
            -> TF "moves"
            -> Robot does NOT move physically
            -> Nav2 thinks goal is reached
            -> Replanning chaos / abort

        SOLUTION:
        ---------
        - Use cmd_vel ONLY as a gating signal (robot is being commanded).
        - Integrate pose ONLY from measured twist (/dog_odom.twist).
        - Ignore micro-oscillations using eps thresholds.
        - Publish twist consistent with integrated pose.
        """

        # ------------------------------------------------------------
        # (0) Safety: we need at least one odom message
        # ------------------------------------------------------------
        if self._last_odom is None:
            return

        # ------------------------------------------------------------
        # (1) Command gating decision
        #     -> Are we actively commanded by Nav2?
        # ------------------------------------------------------------
        gate = self._decide_gate()
        self._last_gate = gate

        # ------------------------------------------------------------
        # (2) Measured twist from /dog_odom
        #     - optional yaw remap
        #     - deadband
        #     - EMA filtering
        # ------------------------------------------------------------
        vx_meas, wz_meas = self._correct_and_filter_twist(self._last_odom)

        # Apply turn-in-place correction on measured velocity
        # (ensures vx=0 when strong rotation command)
        vx_meas = self._apply_turn_in_place_rule(vx_meas)

        # ------------------------------------------------------------
        # (3) Command twist (for debug only, not integration)
        # ------------------------------------------------------------
        vx_cmd = float(gate.cmd_vx)
        wz_cmd = float(gate.cmd_wz)

        # Optional strict turn rule also applied to cmd (for debug clarity)
        if abs(wz_cmd) > self.turn_wz_thresh and abs(vx_cmd) < self.turn_vx_small:
            vx_cmd = 0.0

        # ------------------------------------------------------------
        # (4) Integration decision
        #
        # IMPORTANT:
        # We integrate ONLY if:
        #   - robot is actively commanded
        #   - measured twist is above noise threshold
        #
        # This prevents:
        #   "TF reaches goal while robot stands still"
        # ------------------------------------------------------------
        if gate.integrate and (
            abs(vx_meas) > self.meas_eps_vx or
            abs(wz_meas) > self.meas_eps_wz
        ):
            vx_int = vx_meas
            wz_int = wz_meas
            integrate_yaw = True
        else:
            vx_int = 0.0
            wz_int = 0.0
            integrate_yaw = (not self.freeze_yaw_when_idle)

        # Store for debug telemetry
        self._last_twist_vx = vx_int
        self._last_twist_wz = wz_int

        # ------------------------------------------------------------
        # (5) SE(2) integration step
        # ------------------------------------------------------------
        if gate.integrate:
            # Normal case: integrate measured motion
            self._integrate_se2(
                vx=vx_int,
                wz=wz_int,
                integrate_yaw=integrate_yaw
            )
        else:
            # Idle case:
            # Optionally allow yaw drift from measurement (usually disabled)
            if integrate_yaw and abs(wz_meas) > 0.0:
                self._integrate_se2(
                    vx=0.0,
                    wz=wz_meas,
                    integrate_yaw=True
                )

        # ------------------------------------------------------------
        # (6) Select pose to publish
        #     (integrate / driver / hybrid)
        # ------------------------------------------------------------
        x, y, yaw = self._select_pose()
        qx, qy, qz, qw = _quat_from_yaw(yaw)

        # ------------------------------------------------------------
        # (7) Twist publication strategy
        #
        # We publish measured twist when active,
        # otherwise publish zeros.
        #
        # Why?
        # Because pose evolution = measured motion.
        # Publishing cmd twist would break consistency.
        # ------------------------------------------------------------
        if gate.integrate:
            vx_pub, wz_pub = vx_meas, wz_meas
        else:
            vx_pub, wz_pub = 0.0, 0.0

        # ------------------------------------------------------------
        # (8) Build /odom message
        # ------------------------------------------------------------
        out = Odometry()
        out.header.stamp = self._stamp_msg()
        out.header.frame_id = self.odom_frame
        out.child_frame_id = self.base_frame 

        # Pose
        out.pose.pose.position.x = float(x)
        out.pose.pose.position.y = float(y)
        out.pose.pose.position.z = 0.0
        out.pose.pose.orientation.x = qx
        out.pose.pose.orientation.y = qy
        out.pose.pose.orientation.z = qz
        out.pose.pose.orientation.w = qw

        # Conservative covariance for humanoid
        out.pose.covariance = [0.0] * 36
        out.pose.covariance[0] = 0.05   # x
        out.pose.covariance[7] = 0.05   # y
        out.pose.covariance[35] = 0.10  # yaw

        # Twist
        tw = Twist()
        tw.linear.x = float(vx_pub)
        tw.angular.z = float(wz_pub)

        twc = TwistWithCovariance()
        twc.twist = tw
        twc.covariance = [0.0] * 36
        twc.covariance[0] = 0.10
        twc.covariance[35] = 0.20

        out.twist = twc

        self.odom_pub.publish(out)

        # ------------------------------------------------------------
        # (9) Debug telemetry (throttled)
        # ------------------------------------------------------------
        self._log_info_throttle(
            (
                f"integrate={int(gate.integrate)}({gate.reason}) "
                f"cmd=({vx_cmd:+.2f},{wz_cmd:+.2f}) "
                f"meas=({vx_meas:+.2f},{wz_meas:+.2f}) "
                f"used_int=({vx_int:+.2f},{wz_int:+.2f}) "
                f"pub_twist=({vx_pub:+.2f},{wz_pub:+.2f}) "
                f"pose=({x:+.2f},{y:+.2f},{yaw:+.2f})"
            ),
            period_s=2.0
        )
    

        
    #def _publish_odom_tick(self):
    #    return self._publish_odom_tick_old()

    def _publish_odom_tick(self) -> None:
        """
        Publish Nav2-compatible /odom at a fixed rate.

        Why this matters for RL humanoids
        --------------------------------
        With an RL stabilized humanoid, the base twist measured from the robot
        (e.g., /dog_odom.twist) can be:
        - noisy at rest (micro oscillations),
        - biased during walking (COM stabilization effects),
        - occasionally inconsistent with what Nav2 expects (planar SE(2) motion).

        Nav2 primarily needs *self-consistency*:
        - The pose we publish in /odom.pose must match the twist in /odom.twist,
        - and the TF odom->base must reflect *real* motion.

        The classic failure mode:
        - If you integrate pose from commanded velocity (cmd_vel),
            TF can move even when the robot is physically stuck (deadzone).
        - Nav2 thinks it progressed, reaches the goal virtually,
            then replans when the robot finally starts moving -> oscillations / abort.

        Durable strategy implemented here
        ---------------------------------
        1) Compute a filtered "measured" twist from /dog_odom
        (deadband + EMA + yawrate remap if needed).
        2) Use command gating to decide whether we are "actively commanded".
        3) Integrate pose ONLY when there is evidence of real motion
        (measured twist above eps), not just because cmd_vel is non-zero.
        4) Publish twist that matches the pose source (consistency).
        """
        if self._last_odom is None:
            return

        # ------------------------------------------------------------
        # (1) Command gating: are we currently being commanded?
        #     gate.integrate typically becomes True when cmd_vel is recent
        #     and above some thresholds (your implementation).
        # ------------------------------------------------------------
        gate = self._decide_gate()
        self._last_gate = gate

        # ------------------------------------------------------------
        # (2) Compute "measured" twist from robot odom:
        #     This is your controlled output: remap yawrate, deadband, EMA...
        # ------------------------------------------------------------

        # (2) Compute filtered measured twist (vx, vy, wz) from robot odom
        vx_meas, vy_meas, wz_meas = self._correct_and_filter_twist(self._last_odom)

        # Apply turn-in-place rule: no translation during in-place rotation
        vx_meas, vy_meas = self._apply_turn_in_place_rule(vx_meas, vy_meas)

        # ------------------------------------------------------------
        # (3) Compute "command" twist (what Nav2 is requesting).
        #     gate.cmd_vx, gate.cmd_wz should reflect the post-mux/adapted command.
        # ------------------------------------------------------------
        vx_cmd = float(gate.cmd_vx)
        wz_cmd = float(gate.cmd_wz)

        # Detect strict turn-in-place intent from Nav2 gate cmd
        turn_in_place_cmd = (abs(wz_cmd) > self.turn_wz_thresh) and (abs(vx_cmd) < self.turn_vx_small)

        # If turning in place, never integrate translation (even if RL reports vx)
        if turn_in_place_cmd:
            vx_meas, vy_meas = 0.0, 0.0

        # Optional: enforce same turn-in-place rule on command twist
        if abs(wz_cmd) > self.turn_wz_thresh and abs(vx_cmd) < self.turn_vx_small:
            vx_cmd = 0.0

        # ------------------------------------------------------------
        # (4) Decide integration twist (Humanoid-Safe Version)
        #
        # OBJECTIVE:
        #   - Never integrate noise when robot is idle.
        #   - Allow rotation only when truly commanded.
        #   - Prevent "virtual spinning lidar".
        #
        # PRINCIPLE:
        #   A floating-base humanoid always produces micro-oscillations.
        #   We MUST ignore those unless a real command is active.
        # ------------------------------------------------------------

        # Are we currently receiving a valid velocity command?
        cmd_active = gate.integrate   # True only if cmd_vel is fresh + above eps


        # ------------------------------------------------------------
        # Detect rotation command intent
        # ------------------------------------------------------------
        rot_cmd_active = cmd_active and (abs(wz_cmd) > self.cmd_eps_wz)

        # Measured yaw motion above noise threshold
        yaw_meas_active = abs(wz_meas) > self.meas_eps_wz

        # Measured translation evidence (vx or vy)
        xy_meas_active = cmd_active and (
            abs(vx_meas) > self.meas_eps_vx or
            (self.enable_vy and abs(vy_meas) > self.meas_eps_vy)
        )

        # ------------------------------------------------------------
        # Rotation confirmation timing
        #
        # WHY:
        #   When a rotation command is issued,
        #   measured yaw may lag a bit.
        #
        #   We allow a short fallback window where
        #   commanded yaw is used until measurement confirms.
        # ------------------------------------------------------------

        now = self._now()

        # Start rotation timing window
        if rot_cmd_active and self._rot_cmd_start_time is None:
            self._rot_cmd_start_time = now

        # Reset when rotation command stops
        if (not rot_cmd_active) and self._rot_cmd_start_time is not None:
            self._rot_cmd_start_time = None

        rot_cmd_age = 0.0
        if self._rot_cmd_start_time is not None:
            rot_cmd_age = (now - self._rot_cmd_start_time).nanoseconds * 1e-9


        # Allow fallback to command yaw only for a short duration
        allow_cmd_yaw_fallback = (
            rot_cmd_active and
            (rot_cmd_age <= self.rot_confirm_timeout)
        )


        # ------------------------------------------------------------
        # Choose yaw rate used for integration
        # ------------------------------------------------------------

        if yaw_meas_active:
            # Prefer measured yaw when confirmed
            wz_for_yaw = wz_meas

        elif allow_cmd_yaw_fallback:
            # Short fallback to commanded yaw (avoid delay)
            wz_for_yaw = wz_cmd

        else:
            # Otherwise freeze yaw
            wz_for_yaw = 0.0


        # ------------------------------------------------------------
        # Integration decisions
        # ------------------------------------------------------------

        # Integrate translation only if:
        #   - command active
        #   - measured forward motion confirmed

        #integrate_xy = xy_meas_active
        
        # Integrate translation only if command active AND not turn-in-place
        integrate_xy = (not turn_in_place_cmd) and xy_meas_active


        # CRITICAL RULE:
        #   Never integrate yaw unless command is active.
        #
        #   This prevents:
        #       - idle yaw drift
        #       - lidar sliding
        #       - infinite replanning
        integrate_yaw = cmd_active and (
            yaw_meas_active or allow_cmd_yaw_fallback
        )


        # Values actually used for SE(2) integration
        vx_int = vx_meas if integrate_xy else 0.0
        vy_int = vy_meas if (integrate_xy and self.enable_vy) else 0.0
        wz_int = wz_for_yaw if integrate_yaw else 0.0

        # Store for /odom_bridge/status (and for debugging)
        self._last_twist_vx = vx_int
        self._last_twist_wz = wz_int

        # ------------------------------------------------------------
        # (5) Integrate SE(2) pose ONCE
        # ------------------------------------------------------------

        if integrate_xy or integrate_yaw:
            self._integrate_se2(
                vx=vx_int,
                vy=vy_int,
                wz=wz_int,
                integrate_yaw=integrate_yaw
            )

        # ------------------------------------------------------------
        # (6) Select pose to publish
        # ------------------------------------------------------------

        x, y, yaw = self._select_pose()
        qx, qy, qz, qw = _quat_from_yaw(yaw)


        # ------------------------------------------------------------
        # (7) Publish twist CONSISTENT with integration
        #
        # WHY:
        #   Pose and twist must match.
        #   Publishing cmd twist would break consistency.
        # ------------------------------------------------------------

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

        # Simple covariances (tune if needed)
        out.pose.covariance = [0.0] * 36
        out.pose.covariance[0] = 0.05
        out.pose.covariance[7] = 0.05
        out.pose.covariance[35] = 0.10

        tw = Twist()
        tw.linear.x = float(vx_pub)
        tw.linear.y = float(vy_pub)   # new (optional)
        tw.angular.z = float(wz_pub)

        twc = TwistWithCovariance()
        twc.twist = tw
        twc.covariance = [0.0] * 36
        twc.covariance[0] = 0.10
        twc.covariance[35] = 0.20
        out.twist = twc

        self.odom_pub.publish(out)

        # ------------------------------------------------------------
        # (8) Debug logging
        # ------------------------------------------------------------
        now_s = time.time()
        if (now_s - self._last_info_log_time) >= self._info_log_period:
            self._last_info_log_time = now_s
            self.get_logger().info(
                f"gate={int(gate.integrate)}({gate.reason}) "
                f"xy={int(integrate_xy)} yaw={int(integrate_yaw)} "
                f"cmd=({vx_cmd:+.2f},{wz_cmd:+.2f}) meas=({vx_meas:+.2f},{wz_meas:+.2f}) "
                f"used_int=({vx_int:+.2f},{wz_int:+.2f}) pub=({vx_pub:+.2f},{wz_pub:+.2f}) "
                f"pose=({x:+.2f},{y:+.2f},{yaw:+.2f})"
                f"meas=({vx_meas:+.2f},{vy_meas:+.2f},{wz_meas:+.2f}) "
                f"used_int=({vx_int:+.2f},{vy_int:+.2f},{wz_int:+.2f}) "
            )

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
            f"\"yaw_remap\":{1 if self.remap_yawrate_y_to_z else 0},"
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