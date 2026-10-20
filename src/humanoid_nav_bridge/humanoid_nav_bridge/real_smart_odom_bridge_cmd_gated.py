#!/usr/bin/env python3
"""
smart_odom_bridge_cmd_gated.py

===============================================================================
PURPOSE
===============================================================================

This node is designed to be the UNIQUE odometry authority for navigation.

In a complex real robot system, several odometry-like sources may exist:
  - visual odometry
  - odom reconstructed from robot state
  - SLAM relocation odometry
  - mapping odometry
  - raw dog_odom from the locomotion stack

However, Nav2 expects a clean and stable odometry interface:
  - one odometry topic
  - one odom -> base_footprint TF publisher
  - no competing TF publishers

This node solves that problem by:

  1) Listening to multiple odometry candidates
  2) Selecting the best currently available source
  3) Deciding whether the source pose is trustworthy enough to reuse directly
  4) Otherwise integrating motion from filtered twist
  5) Publishing:
       - /odom              (or /odom/raw if desired)
       - TF odom -> base_footprint
       - TF base_footprint -> robot_center

===============================================================================
WHY THIS NODE IS USEFUL
===============================================================================

Different sources behave differently:

- Some sources provide a pose that is already meaningful and stable
  (for example visual odometry or SLAM-based odometry).

- Some sources provide twist or pose that may drift even when the robot is
  effectively resting, especially on RL-based locomotion systems.

- Some sources may encode yaw rate in angular.y instead of angular.z.

- The locomotion controller itself may generate body motion or oscillation
  that pollutes raw odometry when the robot is just balancing in place.

Because of this, blindly forwarding one odometry source to Nav2 is often not
robust enough.

This node adds:
- source prioritization
- gating by command activity
- twist filtering
- optional pose trust / pose rejection
- optional deadbanding and EMA smoothing
- consistent TF publishing

===============================================================================
TARGET TF TREE
===============================================================================

The intended TF chain is:

    map -> odom -> base_footprint -> robot_center -> livox_frame

where:

- map -> odom
    comes from AMCL or SLAM, not from this node

- odom -> base_footprint
    comes ONLY from this node

- base_footprint -> robot_center
    also comes from this node

- robot_center -> livox_frame
    comes from a static transform publisher

===============================================================================
VERY IMPORTANT RULE
===============================================================================

Only ONE node must publish the main navigation TF:

    odom -> base_footprint

If another node also publishes the same TF, RViz, Nav2 and TF consumers may
see conflicting transforms and produce unstable behavior.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped, Twist, TwistWithCovariance
from std_msgs.msg import String
import tf2_ros


# =============================================================================
# QoS helper
# =============================================================================
def _make_qos(reliable: bool, depth: int) -> QoSProfile:
    """
    Create a small QoS profile helper.

    Why this helper exists:
    - Odometry topics may be published either as RELIABLE or BEST_EFFORT
      depending on the source.
    - Having a dedicated function keeps the code shorter and cleaner.
    """
    return QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE if reliable else ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
        depth=depth,
    )


# =============================================================================
# Angle helpers
# =============================================================================
def _clamp_pi(a: float) -> float:
    """
    Normalize an angle to the interval [-pi, pi].

    This is important whenever yaw is integrated over time, otherwise the angle
    can grow unbounded and make debugging harder.
    """
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def _yaw_from_quat(x: float, y: float, z: float, w: float) -> float:
    """
    Extract planar yaw from a quaternion.

    This function is used when we need the heading of a pose source or of an
    already built odometry message.
    """
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def _quat_from_yaw(yaw: float) -> Tuple[float, float, float, float]:
    """
    Build a quaternion representing a pure planar yaw rotation.
    """
    h = 0.5 * yaw
    return (0.0, 0.0, math.sin(h), math.cos(h))


def _rpy_from_quat(x: float, y: float, z: float, w: float) -> Tuple[float, float, float]:
    """
    Convert quaternion to roll, pitch, yaw.

    This is used mainly for the body TF:
        base_footprint -> robot_center

    We keep roll and pitch there, but yaw must stay in odom -> base_footprint.
    """
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)

    yaw = _yaw_from_quat(x, y, z, w)
    return roll, pitch, yaw


def _quat_from_rpy(roll: float, pitch: float, yaw: float) -> Tuple[float, float, float, float]:
    """
    Build quaternion from roll, pitch, yaw.
    """
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


# =============================================================================
# Small container for each source
# =============================================================================
@dataclass
class SrcState:
    """
    Store:
    - the latest Odometry message for a source
    - the local reception time of that message

    We use reception time, not just the message stamp, because source freshness
    for selection is based on "when did this node last receive a usable sample?"
    """
    msg: Optional[Odometry] = None
    t: Optional[Time] = None


class SmartOdomBridgeCmdGated(Node):
    def __init__(self) -> None:
        super().__init__("smart_odom_bridge_cmd_gated")

        # =========================================================================
        # FRAME PARAMETERS
        # =========================================================================
        # odom_frame:
        #   The local odometry frame used by Nav2 / controller logic.
        #
        # base_frame:
        #   The planar robot base frame used for navigation.
        #   This is typically base_footprint.
        #
        # body_frame:
        #   The body frame above the planar base. This can carry z / roll / pitch.
        self.odom_frame = self.declare_parameter("odom_frame", "odom").value
        self.base_frame = self.declare_parameter("base_frame", "base_footprint").value
        self.body_frame = self.declare_parameter("body_frame", "robot_center").value
        self.frame = self.declare_parameter("frame", "base").value

        # =========================================================================
        # OUTPUT TOPIC
        # =========================================================================
        # This is the odometry topic that navigation will consume.
        # If you want to stay standard, use /odom.
        self.output_odom_topic = self.declare_parameter("output_odom_topic", "/odom").value

        # =========================================================================
        # SOURCE POLICY
        # =========================================================================
        # use_source_pose_for_trusted_sources:
        #   If True, some sources may directly provide the pose used for output.
        #   If False, all sources will only provide twist evidence and the node
        #   will always rely on internal integration.
        self.use_source_pose_for_trusted_sources = bool(
            self.declare_parameter("use_source_pose_for_trusted_sources", True).value
        )

        # Sources ordered by priority:
        #   src1 is preferred over src2, etc.
        #
        # Example strategy:
        #   src1 = visual odom
        #   src5 = dog_odom fallback
        self.src1 = self.declare_parameter("src1", "/odom/visual").value
        self.src2 = self.declare_parameter("src2", "/odommodestate/odom").value
        self.src3 = self.declare_parameter("src3", "/unitree/slam_relocation/odom").value
        self.src4 = self.declare_parameter("src4", "/unitree/slam_mapping/odom").value
        self.src5 = self.declare_parameter("src5", "/dog_odom").value

        # For each source, decide whether its POSE can be trusted directly.
        #
        # Important:
        # - If a source is trusted, its pose is copied directly to output odom.
        # - If not trusted, the node will integrate filtered twist instead.
        self.trust_src1_pose = bool(self.declare_parameter("trust_src1_pose", False).value)
        self.trust_src2_pose = bool(self.declare_parameter("trust_src2_pose", True).value)
        self.trust_src3_pose = bool(self.declare_parameter("trust_src3_pose", True).value)
        self.trust_src4_pose = bool(self.declare_parameter("trust_src4_pose", True).value)
        self.trust_src5_pose = bool(self.declare_parameter("trust_src5_pose", False).value)

        # How long a source can remain silent before being considered stale.
        self.source_timeout_sec = float(self.declare_parameter("source_timeout_sec", 0.5).value)

        # =========================================================================
        # SOURCE-SPECIFIC TWIST CONVENTION
        # =========================================================================
        # Some robot sources may encode yaw rate in angular.y rather than angular.z.
        #
        # This remap is kept only for src5 (typically dog_odom) because that is the
        # source that often needs special handling.
        self.remap_src5_yawrate_y_to_z = bool(
            self.declare_parameter("remap_src5_yawrate_y_to_z", False).value
        )

        # =========================================================================
        # COMMAND GATE PARAMETERS
        # =========================================================================
        # The gate command tells the node whether the robot is currently supposed
        # to move according to navigation.
        #
        # Idea:
        # - if Nav2 is not commanding motion, we should be very careful about
        #   integrating noisy measured twist
        # - if Nav2 IS commanding motion, then measured twist becomes more relevant
        #self.gate_cmd_vel_topic = self.declare_parameter("gate_cmd_vel_topic", "/cmd_vel_nav").value
        self.gate_cmd_vel_topic = self.declare_parameter(
            "gate_cmd_vel_topic", "/cmd_vel_nav_scaled"
        ).value
        self.cmd_eps_vx = float(self.declare_parameter("cmd_eps_vx", 0.02).value)
        self.cmd_eps_wz = float(self.declare_parameter("cmd_eps_wz", 0.02).value)
        self.cmd_timeout_sec = float(self.declare_parameter("cmd_timeout_sec", 0.25).value)

        # =========================================================================
        # MEASURED TWIST FILTERING
        # =========================================================================
        # Small measured twists are often noise when the robot is standing still.
        #
        # We therefore:
        # - apply deadbands
        # - optionally smooth the result with EMA
        self.enable_filter = bool(self.declare_parameter("enable_filter", True).value)
        self.deadband_vx = float(self.declare_parameter("deadband_vx", 0.02).value)
        self.deadband_wz = float(self.declare_parameter("deadband_wz", 0.02).value)
        self.ema_alpha = float(self.declare_parameter("ema_alpha", 0.25).value)

        # Evidence thresholds:
        # measured motion must be above these limits to be considered "real"
        self.meas_eps_vx = float(self.declare_parameter("meas_eps_vx", 0.03).value)
        self.meas_eps_wz = float(self.declare_parameter("meas_eps_wz", 0.05).value)

        # Turn-in-place detection:
        # if navigation clearly asks for turning with almost no forward velocity,
        # we force forward motion integration to zero
        self.turn_wz_thresh = float(self.declare_parameter("turn_wz_thresh", 0.25).value)
        self.turn_vx_small = float(self.declare_parameter("turn_vx_small", 0.04).value)

        # Maximum integration step allowed.
        # If the timer jumps too much, skip integration rather than producing a
        # huge artificial jump.
        self.dt_max = float(self.declare_parameter("dt_max", 0.20).value)

        # =========================================================================
        # TF PUBLICATION CONTROL
        # =========================================================================
        # publish_tf_main:
        #   If True, publish odom -> base_footprint.
        #
        # publish_tf_body:
        #   If True, publish base_footprint -> robot_center.
        self.publish_tf_main = bool(self.declare_parameter("publish_tf_main", True).value)
        self.publish_tf_body = bool(self.declare_parameter("publish_tf_body", True).value)

        # =========================================================================
        # PUBLICATION RATES
        # =========================================================================
        self.odom_rate_hz = float(self.declare_parameter("odom_rate_hz", 50.0).value)
        self.tf_rate_hz = float(self.declare_parameter("tf_rate_hz", 50.0).value)

        # =========================================================================
        # STATUS OUTPUT
        # =========================================================================
        # Optional textual debug output published as a JSON-like string.
        self.publish_status = bool(self.declare_parameter("publish_status", True).value)
        self.status_rate_hz = float(self.declare_parameter("status_rate_hz", 5.0).value)

        # =========================================================================
        # QOS
        # =========================================================================
        self.sub_reliable = bool(self.declare_parameter("sub_reliable", True).value)
        self.pub_reliable = bool(self.declare_parameter("pub_reliable", True).value)
        self.qos_depth = int(self.declare_parameter("qos_depth", 10).value)
        self.dog_odom_reliable = bool(
            self.declare_parameter("dog_odom_reliable", False).value
        )

        sub_qos = _make_qos(self.sub_reliable, self.qos_depth)
        pub_qos = _make_qos(self.pub_reliable, self.qos_depth)
        cmd_qos = _make_qos(True, 1)
        dog_odom_qos = _make_qos(
            self.dog_odom_reliable,
            self.qos_depth
        )

        # =========================================================================
        # INTERNAL STATE
        # =========================================================================
        # We keep one state object per source.
        self._src: Dict[str, SrcState] = {
            t: SrcState() for t in [self.src1, self.src2, self.src3, self.src4, self.src5]
        }

        # Last navigation command used for gating decisions
        self._last_gate_cmd: Optional[Twist] = None
        self._last_gate_time: Optional[Time] = None

        # Internal integrated pose used only when source pose is not trusted
        self._x_int = 0.0
        self._y_int = 0.0
        self._yaw_int = 0.0
        self._last_int_time = self.get_clock().now()

        # EMA filter internal states
        self._vx_f = 0.0
        self._wz_f = 0.0

        # Last odom message published by this node
        # This is reused when publishing the main TF so both remain consistent.
        self._last_odom_msg: Optional[Odometry] = None

        # Last debug/status dictionary
        self._last_debug = {}

        # =========================================================================
        # IDLE / MOVING HYSTERESIS
        # =========================================================================
        # We do not want the bridge to switch too fast between "robot moving"
        # and "robot idle" because RL locomotion and VO can oscillate around zero.
        self.idle_confirm_sec = float(self.declare_parameter("idle_confirm_sec", 0.30).value)
        self.motion_confirm_sec = float(self.declare_parameter("motion_confirm_sec", 0.12).value)

        # If enabled, force published twist to zero while idle.
        # This is usually desirable for Nav2 when raw odom is noisy at standstill.
        self.zero_twist_when_idle = bool(
            self.declare_parameter("zero_twist_when_idle", True).value
        )

        # If enabled, keep internal yaw frozen while idle.
        self.freeze_yaw_when_idle = bool(
            self.declare_parameter("freeze_yaw_when_idle", True).value
        )

        self.freeze_pose_when_idle = bool(
            self.declare_parameter("freeze_pose_when_idle", True).value
        )

        self._idle_anchor_valid = False
        self._idle_anchor_x = 0.0
        self._idle_anchor_y = 0.0
        self._idle_anchor_yaw = 0.0
        self._prev_motion_state = "idle"

        # Optional: while idle, low-pass the published pose toward a stable anchor.
        # For now we keep it simple: freeze integrated pose and zero twist.
        self._motion_state = "idle"   # "idle" or "moving"
        self._candidate_state = "idle"
        self._candidate_since = self._now()


        # =========================================================================
        # ROS INTERFACES
        # =========================================================================
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        self.odom_pub = self.create_publisher(Odometry, self.output_odom_topic, pub_qos)

        self.status_pub = None
        if self.publish_status:
            self.status_pub = self.create_publisher(String, "/smart_odom_bridge/status", pub_qos)

        # Subscribe to all odometry candidate sources
        for topic in self._src.keys():
            self.create_subscription(
                Odometry,
                topic,
                lambda msg, t=topic: self._cb_src(t, msg),
                dog_odom_qos,
                #sub_qos,
            )

        # Subscribe to navigation command used as motion gate
        self.create_subscription(Twist, self.gate_cmd_vel_topic, self._cb_gate, cmd_qos)

        # Main odometry timer
        self.create_timer(1.0 / self.odom_rate_hz, self._tick_odom)

        # TF timers
        if self.publish_tf_main and self.tf_rate_hz > 0.0:
            self.create_timer(1.0 / self.tf_rate_hz, self._tick_tf_main)

        if self.publish_tf_body and self.tf_rate_hz > 0.0:
            self.create_timer(1.0 / self.tf_rate_hz, self._tick_tf_body)

        # Status timer
        if self.publish_status and self.status_rate_hz > 0.0:
            self.create_timer(1.0 / self.status_rate_hz, self._tick_status)

        self.get_logger().info(
            "Smart odom bridge started:\n"
            f"  source priority: {self.src1} > {self.src2} > {self.src3} > {self.src4} > {self.src5}\n"
            f"  output odom: {self.output_odom_topic}\n"
            f"  gate cmd: {self.gate_cmd_vel_topic}\n"
            f"  TF main: {'ON' if self.publish_tf_main else 'OFF'} ({self.odom_frame}->{self.base_frame})\n"
            f"  TF body: {'ON' if self.publish_tf_body else 'OFF'} ({self.base_frame}->{self.body_frame})"
        )

        # =========================================================================
        # SOURCE LOCK WHILE MOVING
        # =========================================================================
        #
        # Problem this solves:
        # --------------------
        #
        # In real systems, multiple odometry sources may exist:
        #
        #   src1 : visual odometry
        #   src2 : reconstructed odom
        #   src3 : SLAM relocation
        #   src4 : mapping odom
        #   src5 : dog_odom (locomotion stack)
        #
        # The bridge normally selects the first *fresh* source based on priority.
        #
        # However, during motion, a temporary dropout of the best source
        # (for example visual odometry losing tracking) can cause the bridge
        # to switch to another source mid-trajectory.
        #
        # That leads to:
        #
        #   odom source jump
        #   TF discontinuity
        #   Nav2 control instability
        #
        # Goal:
        # -----
        # Prevent the bridge from switching odometry source in the middle of a
        # navigation trajectory.
        #
        # Why this matters:
        # -----------------
        # If src1 = visual odometry is selected at the start of motion, but later
        # drops for a short moment, the default priority-based selection may fall
        # back to src5 = /dog_odom.
        #
        # On humanoid RL systems this is dangerous because:
        #   - src1 may provide the cleanest NAVIGATION pose
        #   - src5 may provide useful motion activity, but its POSE may drift
        #
        # If the bridge switches from src1 pose to src5 integrated motion
        # mid-trajectory, Nav2 may see:
        #   - TF discontinuity
        #   - frozen odom
        #   - jumps in the local frame
        #
        # Strategy:
        # ---------
        #   • when robot enters MOVING:
        #       lock current source
        #
        #   • while MOVING:
        #       keep using that same source as long as it stays reasonably fresh
        #
        #   • when robot returns to IDLE:
        #       unlock source and go back to normal priority selection
        #
        self.lock_source_while_moving = bool(
            self.declare_parameter("lock_source_while_moving", True).value
        )

        # Maximum allowed age for a locked source.
        # If the locked source becomes older than this threshold,
        # the lock is considered unusable and the bridge falls back
        # to normal source selection.
        self.locked_source_timeout_sec = float(
            self.declare_parameter("locked_source_timeout_sec", 1.0).value
        )

        # Name of currently locked source, or None when unlocked.
        self._locked_source: Optional[str] = None

        # =========================================================================
        # MOTION EVIDENCE FROM POSE DELTA
        # =========================================================================
        #
        # Problem this solves:
        # --------------------
        # Some odometry sources (especially visual odometry) may provide a good POSE
        # but a poor / unstable TWIST.
        #
        # If the bridge decides "moving vs idle" using only twist,
        # then RViz / Nav2 may see a frozen TF even though the pose is evolving.
        #
        # Strategy:
        # ---------
        # Estimate motion directly from the delta between:
        #   - current pose
        #   - previous pose
        #
        # This gives:
        #   - planar speed estimate
        #   - yaw-rate estimate
        #
        # If either is above threshold, we treat it as evidence of real movement.
        #
        self.pose_evidence_vxy = float(
            self.declare_parameter("pose_evidence_vxy", 0.02).value
        )
        self.pose_evidence_wz = float(
            self.declare_parameter("pose_evidence_wz", 0.05).value
        )

        # Previous pose sample used for delta-based motion estimation.
        self._last_pose_x: Optional[float] = None
        self._last_pose_y: Optional[float] = None
        self._last_pose_yaw: Optional[float] = None
        self._last_pose_time: Optional[Time] = None

        # =========================================================================
        # OPTIONAL: USE src5 TWIST AS AUXILIARY MOTION EVIDENCE
        # =========================================================================
        #
        # Motivation:
        # -----------
        # On the humanoid RL setup:
        #
        #   - src1 (/odom/visual) may be the best source for NAVIGATION POSE
        #   - src5 (/dog_odom) may still be useful to detect that locomotion is
        #     physically happening, even if its pose should NOT be trusted
        #
        # This flag allows using src5 TWIST only as a "robot is moving" hint.
        #
        self.use_src5_twist_as_aux_evidence = bool(
            self.declare_parameter("use_src5_twist_as_aux_evidence", True).value
        )


        # =========================================================================
        # FALLBACK INTEGRATION WHEN TRUSTED POSE IS "STUCK"
        # =========================================================================
        #
        # Problem:
        # --------
        # A trusted pose source (typically visual odometry) may remain nearly flat
        # while the robot is physically moving. In that case:
        #
        #   - src1 is still selected
        #   - src1 pose is still published
        #   - Nav2 sees almost no progress
        #
        # To avoid that, we maintain a small fallback integrated pose driven by
        # src5 twist, used only when trusted pose appears stuck during motion.
        #
        self.enable_trusted_pose_fallback = bool(
            self.declare_parameter("enable_trusted_pose_fallback", True).value
        )

        # If True, when trusted pose is judged "stuck" during motion,
        # we integrate a backup pose using src5 twist.
        self.fallback_use_src5_twist = bool(
            self.declare_parameter("fallback_use_src5_twist", True).value
        )

        # Persistent fallback pose state
        self._fallback_x = 0.0
        self._fallback_y = 0.0
        self._fallback_yaw = 0.0
        self._fallback_initialized = False
        self._fallback_last_time = self.get_clock().now()


        # =========================================================================
        # GATE RELEASE HOLD
        # =========================================================================
        #
        # Problem:
        # --------
        # Near the goal, Nav2 often reduces command magnitude below epsilon
        # or briefly stops publishing commands before the physical robot has
        # fully stopped.
        #
        # On RL-based locomotion systems, this is common:
        #   - command becomes very small
        #   - gate would normally become False
        #   - but the robot is still physically moving due to controller dynamics
        #
        # If the bridge drops immediately from MOVING to IDLE:
        #   - odom can freeze too early
        #   - source lock can be released too early
        #   - Nav2 sees late progress / relocalization artifacts
        #
        # Strategy:
        # ---------
        # Keep a short grace period after the last active command.
        #
        # During that time, the bridge may still consider movement intentional
        # even if the instantaneous gate is already False.
        #
        self.gate_release_hold_sec = float(
            self.declare_parameter("gate_release_hold_sec", 0.80).value
        )

        # =========================================================================
        # COMMAND / MEASUREMENT FUSION FOR FALLBACK NAVIGATION ODOM
        # =========================================================================
        #
        # Why these parameters exist:
        # ---------------------------
        # The fallback odom should not blindly trust:
        #
        #   - cmd_vel only  -> tends to overestimate real displacement
        #   - measured twist only -> tends to be noisy and can drift
        #
        # We therefore use a fused model:
        #
        #   fused = alpha * scaled_command + (1 - alpha) * measured_twist
        #
        # Translation and rotation are calibrated separately because the humanoid
        # robot often behaves differently in linear motion vs angular motion.
        #

        # ------------------------------------------------------------
        # Command model calibration
        # ------------------------------------------------------------
        #
        # These scales compensate systematic mismatch between:
        #   - commanded motion
        #   - real robot motion
        #
        # Example:
        # If RViz reaches the goal before the physical robot,
        # linear_cmd_scale is probably too high.
        #
        self.linear_cmd_scale = float(
            self.declare_parameter("linear_cmd_scale", 0.70).value
        )
        self.angular_cmd_scale = float(
            self.declare_parameter("angular_cmd_scale", 0.85).value
        )

        # ------------------------------------------------------------
        # Fusion weights during commanded motion
        # ------------------------------------------------------------
        #
        # alpha = 1.0  -> trust command only
        # alpha = 0.0  -> trust measured twist only
        #
        self.alpha_lin = float(
            self.declare_parameter("alpha_lin", 0.80).value
        )
        self.alpha_ang = float(
            self.declare_parameter("alpha_ang", 0.60).value
        )

        # Small artificial uncertainty around the calibrated command model.
        # This keeps the synthetic odom slightly imperfect, which is often more
        # realistic than a perfectly deterministic command integrator.
        self.linear_noise_ratio = float(
            self.declare_parameter("linear_noise_ratio", 0.005).value
        )
        self.angular_noise_ratio = float(
            self.declare_parameter("angular_noise_ratio", 0.03).value
        )

    def _now(self) -> Time:
        """
        Convenience helper to get current node time.
        """
        return self.get_clock().now()

    def _cb_src(self, topic: str, msg: Odometry) -> None:
        """
        Store the latest message and reception time for a source.

        We do not process the source immediately here.
        The actual selection happens later in the main odom tick.
        """
        self._src[topic].msg = msg
        self._src[topic].t = self._now()

    def _cb_gate(self, msg: Twist) -> None:
        """
        Store the latest navigation command used for gating.

        This command is not directly used as output odometry.
        It only helps us decide whether measured motion should be interpreted as
        intentional robot motion or likely noise.
        """
        self._last_gate_cmd = msg
        self._last_gate_time = self._now()

    def _integrate_fused(self, vx_cmd: float, wz_cmd: float, vx_meas: float, wz_meas: float) -> None:
        """
        Integrate odom using a fusion of:
            - commanded motion
            - measured motion

        Why this helps:
        ----------------
        Command gives smoothness.
        Measured twist keeps the odom closer to the physical robot.

        This is especially useful on humanoid RL robots where:
            - command execution is not perfectly linear
            - gait and stabilization create mismatch between command and reality
        """
        now = self._now()
        dt = (now - self._last_int_time).nanoseconds * 1e-9
        self._last_int_time = now

        if dt <= 0.0 or dt > 0.20:
            return

        # Remove tiny command values before scaling/noise
        if abs(vx_cmd) < self.cmd_eps_vx:
            vx_cmd = 0.0
        if abs(wz_cmd) < self.cmd_eps_wz:
            wz_cmd = 0.0

        # Deadband on measured motion
        if abs(vx_meas) < self.deadband_vx:
            vx_meas = 0.0
        if abs(wz_meas) < self.deadband_wz:
            wz_meas = 0.0

        # Calibrated command model
        vx_model = self.linear_cmd_scale * vx_cmd
        wz_model = self.angular_cmd_scale * wz_cmd

        # Small synthetic noise only on command model
        vx_model = vx_model * (
            1.0 + self._rng.uniform(-self.linear_noise_ratio, self.linear_noise_ratio)
        )
        wz_model = wz_model * (
            1.0 + self._rng.uniform(-self.angular_noise_ratio, self.angular_noise_ratio)
        )

        # Fusion
        vx_fused = self.alpha_lin * vx_model + (1.0 - self.alpha_lin) * vx_meas
        wz_fused = self.alpha_ang * wz_model + (1.0 - self.alpha_ang) * wz_meas

        self._yaw = _clamp_pi(self._yaw + wz_fused * dt)
        self._x += vx_fused * math.cos(self._yaw) * dt
        self._y += vx_fused * math.sin(self._yaw) * dt

    def _integrate_meas(self, vx: float, wz: float) -> None:
        """
        Integrate odom from measured twist during coasting / braking phase.
        """
        now = self._now()
        dt = (now - self._last_int_time).nanoseconds * 1e-9
        self._last_int_time = now

        if dt <= 0.0 or dt > 0.20:
            return

        if abs(vx) < self.deadband_vx:
            vx = 0.0
        if abs(wz) < self.deadband_wz:
            wz = 0.0

        self._yaw = _clamp_pi(self._yaw + wz * dt)
        self._x += vx * math.cos(self._yaw) * dt
        self._y += vx * math.sin(self._yaw) * dt

    def _update_motion_state(self, gate: bool, evidence: bool) -> str:
        """
        Robust moving/idle state machine with hysteresis.

        Rules:
        - moving requires gate AND evidence for a short confirmation window
        - idle is entered when gate is false OR evidence is false, but only after
          a slightly longer confirmation window
        """
        now = self._now()

        desired = "moving" if (gate and evidence) else "idle"

        if desired != self._candidate_state:
            self._candidate_state = desired
            self._candidate_since = now
            return self._motion_state

        dt = (now - self._candidate_since).nanoseconds * 1e-9

        if desired == "moving":
            if dt >= self.motion_confirm_sec:
                self._motion_state = "moving"
        else:
            if dt >= self.idle_confirm_sec:
                self._motion_state = "idle"

        return self._motion_state
    
    def _pick_active_source(self) -> Tuple[str, Optional[Odometry], float]:
        """
        Pick the highest-priority source that is still fresh.

        Returns:
            (source_name, odom_msg, age_sec)

        If no valid source is found:
            ("none", None, 999.0)
        """
        now = self._now()
        for topic in [self.src1, self.src2, self.src3, self.src4, self.src5]:
            st = self._src.get(topic)
            if st and st.msg and st.t:
                age = (now - st.t).nanoseconds * 1e-9
                if age <= self.source_timeout_sec:
                    return topic, st.msg, age
        return "none", None, 999.0

    def _gate_effective(self, gate: bool, cmd_age: float) -> bool:
        """
        Return an effective motion gate.

        Why this helper exists:
        -----------------------
        The raw gate returned by _gate_active() is strict:
            gate = True only if the command is both recent and above epsilon

        On a real RL robot, physical motion may continue for a short time
        after Nav2 command:
            - falls below epsilon
            - becomes temporarily stale
            - or goes to zero near the goal

        If we dropped immediately to gate=False, the bridge could:
            - exit MOVING too early
            - freeze odom too early
            - release locked source too early

        Effective gate logic:
        ---------------------
        - If raw gate is already True:
              effective gate is True
        - Else:
              keep a short grace period after the last command

        Args:
            gate: raw gate from _gate_active()
            cmd_age: age of last command in seconds

        Returns:
            True if movement should still be considered intentionally active.
        """
        if gate:
            return True

        return cmd_age <= self.gate_release_hold_sec
    
    def _extract_wz_from_source(self, src_name: str, odom: Odometry) -> float:
        """
        Extract yaw rate from a source.

        Normally yaw rate is in angular.z, but some sources may incorrectly
        encode it in angular.y. For now we handle that explicitly for src5.
        """
        wy = float(odom.twist.twist.angular.y)
        wz = float(odom.twist.twist.angular.z)

        if src_name == self.src5 and self.remap_src5_yawrate_y_to_z:
            return wy
        return wz

    def _get_source_if_fresh(self, topic: str) -> Tuple[Optional[str], Optional[Odometry], float]:
        """
        Return a specific source if it is still fresh enough.

        Why this helper exists:
        -----------------------
        During MOVING, if source locking is enabled, we want to continue using
        the currently locked source as long as it still publishes recent data.

        Returns:
            (topic, odom_msg, age_sec) if source is fresh enough
            (None, None, age_sec) otherwise
        """
        st = self._src.get(topic)

        if st is None or st.msg is None or st.t is None:
            return None, None, 999.0

        now = self._now()
        age = (now - st.t).nanoseconds * 1e-9

        if age <= self.locked_source_timeout_sec:
            return topic, st.msg, age

        return None, None, age
    
    def _pose_motion_evidence(self, odom: Odometry) -> Tuple[bool, float, float]:
        """
        Estimate motion evidence from pose delta.

        Why this is needed:
        -------------------
        Some sources may provide:
            - good pose
            - poor or unreliable twist

        If we only inspect twist, the bridge may believe the robot is idle
        while the pose is clearly changing.

        Method:
        -------
        Compare current pose with previous pose:
            dx, dy, dyaw over dt

        Then estimate:
            planar speed   = sqrt(dx^2 + dy^2) / dt
            yaw rate       = abs(dyaw) / dt

        Returns:
            evidence_from_pose : bool
            vxy_est            : estimated planar speed [m/s]
            wz_est             : estimated yaw rate [rad/s]
        """
        now = self._now()

        x = float(odom.pose.pose.position.x)
        y = float(odom.pose.pose.position.y)
        q = odom.pose.pose.orientation
        yaw = _yaw_from_quat(float(q.x), float(q.y), float(q.z), float(q.w))

        # First sample: initialize history and report no evidence yet
        if (
            self._last_pose_x is None
            or self._last_pose_y is None
            or self._last_pose_yaw is None
            or self._last_pose_time is None
        ):
            self._last_pose_x = x
            self._last_pose_y = y
            self._last_pose_yaw = yaw
            self._last_pose_time = now
            return False, 0.0, 0.0

        dt = (now - self._last_pose_time).nanoseconds * 1e-9

        # Invalid or too-large dt: reset history safely 
        if dt <= 0.0 or dt > self.dt_max:
            self._last_pose_x = x
            self._last_pose_y = y
            self._last_pose_yaw = yaw
            self._last_pose_time = now
            return False, 0.0, 0.0

        dx = x - self._last_pose_x
        dy = y - self._last_pose_y
        dyaw = _clamp_pi(yaw - self._last_pose_yaw)

        vxy_est = math.sqrt(dx * dx + dy * dy) / dt
        wz_est = abs(dyaw) / dt

        # Update history
        self._last_pose_x = x
        self._last_pose_y = y
        self._last_pose_yaw = yaw
        self._last_pose_time = now

        evidence = (vxy_est > self.pose_evidence_vxy) or (wz_est > self.pose_evidence_wz)
        return evidence, vxy_est, wz_est
    
    def _src5_twist_evidence(self) -> Tuple[bool, float, float]:
        """
        Use src5 (/dog_odom) TWIST only as auxiliary motion evidence.

        Important:
        ----------
        This helper does NOT trust src5 pose.
        It only asks:
            "Is locomotion apparently happening physically?"

        This is useful on humanoid RL systems where:
            - dog_odom pose may drift due to stabilization
            - but dog_odom twist may still indicate real body motion

        Returns:
            evidence_from_src5 : bool
            vx_src5            : linear.x from src5
            wz_src5            : yaw-rate extracted from src5
        """
        if not self.use_src5_twist_as_aux_evidence:
            return False, 0.0, 0.0

        st = self._src.get(self.src5)
        if st is None or st.msg is None or st.t is None:
            return False, 0.0, 0.0

        now = self._now()
        age = (now - st.t).nanoseconds * 1e-9
        if age > self.source_timeout_sec:
            return False, 0.0, 0.0

        vx = float(st.msg.twist.twist.linear.x)
        wz = self._extract_wz_from_source(self.src5, st.msg)

        evidence = (abs(vx) > self.meas_eps_vx) or (abs(wz) > self.meas_eps_wz)
        return evidence, vx, wz
    
    def _reset_fallback_pose_from(self, x: float, y: float, yaw: float) -> None:
        """
        Reset fallback pose state from a trusted reference pose.

        This keeps fallback aligned with the trusted source before any temporary
        fallback integration starts.
        """
        self._fallback_x = x
        self._fallback_y = y
        self._fallback_yaw = yaw
        self._fallback_initialized = True
        self._fallback_last_time = self._now()


    def _integrate_fallback_pose(self, vx: float, wz: float) -> None:
        """
        Integrate a backup planar pose using fallback twist.

        Why:
        ----
        If trusted pose source is selected but appears nearly frozen while the
        robot is clearly moving, we need a temporary backup motion model.

        This fallback is intentionally simple:
            yaw += wz * dt
            x   += vx * cos(yaw) * dt
            y   += vx * sin(yaw) * dt
        """
        now = self._now()
        dt = (now - self._fallback_last_time).nanoseconds * 1e-9
        self._fallback_last_time = now

        if dt <= 0.0 or dt > self.dt_max:
            return

        self._fallback_yaw = _clamp_pi(self._fallback_yaw + wz * dt)
        self._fallback_x += vx * math.cos(self._fallback_yaw) * dt
        self._fallback_y += vx * math.sin(self._fallback_yaw) * dt

    def _integrate_fallback_pose_fused(
        self,
        vx_cmd: float,
        wz_cmd: float,
        vx_meas: float,
        wz_meas: float,
    ) -> None:
        """
        Integrate fallback pose using a fusion of:
            - calibrated command
            - measured twist

        Why this helper exists:
        -----------------------
        Using command only tends to make RViz reach the goal too early.
        Using measured twist only tends to inject noise and late residual motion.

        A fused model is a better compromise for local navigation odom.

        Important behavior:
        -------------------
        - translation uses linear_cmd_scale and alpha_lin
        - rotation uses angular_cmd_scale and alpha_ang
        - tiny commands and tiny measurements are suppressed
        - a small synthetic uncertainty can be injected on the command model
        """
        now = self._now()
        dt = (now - self._fallback_last_time).nanoseconds * 1e-9
        self._fallback_last_time = now

        if dt <= 0.0 or dt > self.dt_max:
            return

        # Remove tiny command values
        if abs(vx_cmd) < self.cmd_eps_vx:
            vx_cmd = 0.0
        if abs(wz_cmd) < self.cmd_eps_wz:
            wz_cmd = 0.0

        # Remove tiny measured values
        if abs(vx_meas) < self.deadband_vx:
            vx_meas = 0.0
        if abs(wz_meas) < self.deadband_wz:
            wz_meas = 0.0

        # Calibrated command model
        vx_model = self.linear_cmd_scale * vx_cmd
        wz_model = self.angular_cmd_scale * wz_cmd

        # Small artificial uncertainty on the command model
        vx_model *= (
            1.0 + self._rng.uniform(-self.linear_noise_ratio, self.linear_noise_ratio)
        )
        wz_model *= (
            1.0 + self._rng.uniform(-self.angular_noise_ratio, self.angular_noise_ratio)
        )

        # Fused motion model
        vx_fused = self.alpha_lin * vx_model + (1.0 - self.alpha_lin) * vx_meas
        wz_fused = self.alpha_ang * wz_model + (1.0 - self.alpha_ang) * wz_meas

        self._fallback_yaw = _clamp_pi(self._fallback_yaw + wz_fused * dt)
        self._fallback_x += vx_fused * math.cos(self._fallback_yaw) * dt
        self._fallback_y += vx_fused * math.sin(self._fallback_yaw) * dt

    def _integrate_fallback_pose_measured(
        self,
        vx_meas: float,
        wz_meas: float,
    ) -> None:
        """
        Integrate fallback pose from measured twist only.

        Why this helper exists:
        -----------------------
        When Nav2 command has already dropped near the goal, the robot may still
        be physically moving. In that case, command-based integration becomes
        too optimistic or even zero, while measured twist is the correct source
        of residual motion.

        This helper is therefore used in the "coast / braking" phase.
        """
        now = self._now()
        dt = (now - self._fallback_last_time).nanoseconds * 1e-9
        self._fallback_last_time = now

        if dt <= 0.0 or dt > self.dt_max:
            return

        if abs(vx_meas) < self.deadband_vx:
            vx_meas = 0.0
        if abs(wz_meas) < self.deadband_wz:
            wz_meas = 0.0

        self._fallback_yaw = _clamp_pi(self._fallback_yaw + wz_meas * dt)
        self._fallback_x += vx_meas * math.cos(self._fallback_yaw) * dt
        self._fallback_y += vx_meas * math.sin(self._fallback_yaw) * dt

    def _gate_active(self) -> Tuple[bool, str, float, float, float]:
        """
        Decide whether the motion gate is active.

        Returns:
            gate_active
            reason_string
            command_age_sec
            vx_cmd
            wz_cmd

        The gate is active only if:
        - a recent command exists
        - and either vx or wz is above the configured epsilon
        """
        if self._last_gate_cmd is None or self._last_gate_time is None:
            return False, "no_cmd", 999.0, 0.0, 0.0

        now = self._now()
        cmd_age = (now - self._last_gate_time).nanoseconds * 1e-9
        vx_cmd = float(self._last_gate_cmd.linear.x)
        wz_cmd = float(self._last_gate_cmd.angular.z)

        if cmd_age > self.cmd_timeout_sec:
            return False, "cmd_stale", cmd_age, vx_cmd, wz_cmd

        active = (abs(vx_cmd) > self.cmd_eps_vx) or (abs(wz_cmd) > self.cmd_eps_wz)
        return active, ("cmd_active" if active else "cmd_below_eps"), cmd_age, vx_cmd, wz_cmd

    def _correct_filter(self, src_name: str, odom: Odometry) -> Tuple[float, float]:
        """
        Filter measured twist from the selected source.

        Steps:
        1. Read vx and wz from the source
        2. Apply deadband to remove tiny noise
        3. Apply EMA smoothing if enabled

        Returns:
            filtered_vx, filtered_wz
        """
        vx = float(odom.twist.twist.linear.x)
        wz = self._extract_wz_from_source(src_name, odom)

        if not self.enable_filter:
            return vx, wz

        if abs(vx) < self.deadband_vx:
            vx = 0.0
        if abs(wz) < self.deadband_wz:
            wz = 0.0

        self._vx_f = self.ema_alpha * vx + (1.0 - self.ema_alpha) * self._vx_f
        self._wz_f = self.ema_alpha * wz + (1.0 - self.ema_alpha) * self._wz_f

        return self._vx_f, self._wz_f

    def _integrate(self, vx: float, wz: float, integrate_yaw: bool) -> None:
        """
        Integrate planar motion into the internal odom state.

        This path is used only when the currently selected source pose is NOT
        trusted and we prefer to reconstruct motion from measured twist.

        The integration is very simple:
            yaw += wz * dt
            x += vx * cos(yaw) * dt
            y += vx * sin(yaw) * dt
        """
        now = self._now()
        dt = (now - self._last_int_time).nanoseconds * 1e-9
        self._last_int_time = now

        if dt < 0.0 or dt > self.dt_max:
            return

        if integrate_yaw:
            self._yaw_int = _clamp_pi(self._yaw_int + wz * dt)

        self._x_int += vx * math.cos(self._yaw_int) * dt
        self._y_int += vx * math.sin(self._yaw_int) * dt

    def _is_source_pose_trusted(self, src_name: str) -> bool:
        """
        Decide whether the currently selected source pose should be copied
        directly to output odom.
        """
        if not self.use_source_pose_for_trusted_sources:
            return False

        if src_name == self.src1:
            return self.trust_src1_pose
        if src_name == self.src2:
            return self.trust_src2_pose
        if src_name == self.src3:
            return self.trust_src3_pose
        if src_name == self.src4:
            return self.trust_src4_pose
        if src_name == self.src5:
            return self.trust_src5_pose

        return False

    def _tick_odom(self) -> None:
        """
        Main odometry production loop.

        Robust strategy used here:
        --------------------------
        1. Select odometry source
           - normal priority selection
           - OR locked source if robot is already moving

        2. Read command gate
           - tells whether navigation is currently asking for motion

        3. Read and filter measured twist from selected source

        4. Build robust motion evidence using:
           - twist from selected source
           - pose delta from selected source
           - optional auxiliary twist evidence from src5 (/dog_odom)

        5. Compute robust motion state: idle / moving
           with hysteresis to avoid noisy toggling

        6. If selected source pose is trusted:
           - use source pose directly for published odom
           - optionally freeze pose while idle

        7. Otherwise:
           - integrate planar motion from filtered twist
           - but ONLY when robot is truly moving

        Goal:
        -----
        Provide Nav2 with a stable odom -> base_footprint transform
        that follows the robot when moving, yet stays quiet while idle.
        """

        # ------------------------------------------------------------
        # 1) SOURCE SELECTION WITH OPTIONAL LOCK
        # ------------------------------------------------------------
        #
        # If robot is already moving and source locking is enabled,
        # try to keep the previously locked source.
        #
        # Otherwise fall back to normal priority-based selection.
        #
        src_name = None
        odom = None
        src_age = 999.0

        if (
            self.lock_source_while_moving
            and self._motion_state == "moving"
            and self._locked_source
        ):
            src_name, odom, src_age = self._get_source_if_fresh(self._locked_source)

        if odom is None:
            src_name, odom, src_age = self._pick_active_source()

        if odom is None:
            return

        # ------------------------------------------------------------
        # 2) GATE FROM COMMAND ACTIVITY
        # ------------------------------------------------------------
        #
        # gate=True means:
        #   the navigation stack is actively commanding motion
        #
        # This helps distinguish:
        #   - intentional movement
        #   - passive noise / balancing / drift
        #
        gate, gate_reason, cmd_age, vx_cmd, wz_cmd = self._gate_active()

        # ------------------------------------------------------------
        # Effective gate
        # ------------------------------------------------------------
        #
        # The raw gate may drop too early near the goal.
        # gate_effective keeps a short grace period so the bridge can continue
        # following residual physical motion after command magnitude shrinks.
        #
        gate_effective = self._gate_effective(gate, cmd_age)

        # Pure turn-in-place request:
        # if command asks for a clear rotation with little forward speed,
        # we suppress forward integration from measured twist.
        turn_in_place_cmd = (abs(wz_cmd) > self.turn_wz_thresh) and (abs(vx_cmd) < self.turn_vx_small)

        # ------------------------------------------------------------
        # 3) FILTERED TWIST FROM SELECTED SOURCE
        # ------------------------------------------------------------
        #
        # This is the classical motion signal:
        #   - use current source twist
        #   - apply deadband + EMA
        #
        vx_meas, wz_meas = self._correct_filter(src_name, odom)

        # For pure turning commands, do not allow forward evidence to come from
        # small noisy linear velocity.
        if turn_in_place_cmd:
            vx_meas = 0.0

        # ------------------------------------------------------------
        # 4) ROBUST MOTION EVIDENCE
        # ------------------------------------------------------------
        #
        # Evidence 1:
        #   twist evidence from CURRENT selected source
        #
        # Evidence 2:
        #   pose-delta evidence from CURRENT selected source
        #
        # Evidence 3:
        #   auxiliary twist evidence from src5 (/dog_odom)
        #
        # Why combine all three?
        # ----------------------
        # On humanoid RL robots:
        #
        #   • VO may provide good pose but poor twist
        #   • dog_odom may provide poor pose but useful locomotion activity
        #
        # So a single evidence channel is not robust enough.
        #
        evidence_from_selected_twist = (
            (abs(vx_meas) > self.meas_eps_vx) or (abs(wz_meas) > self.meas_eps_wz)
        )

        evidence_from_pose, vxy_pose_est, wz_pose_est = self._pose_motion_evidence(odom)

        evidence_from_src5_twist, vx_src5, wz_src5 = self._src5_twist_evidence()

        evidence = (
            evidence_from_selected_twist
            or evidence_from_pose
            or evidence_from_src5_twist
        )

        # ------------------------------------------------------------
        # Trusted-pose-stuck detection
        # ------------------------------------------------------------
        #
        # This detects the situation:
        #
        #   - robot is commanded to move
        #   - auxiliary RL evidence says movement is happening
        #   - but trusted pose itself shows almost no motion
        #
        # Typical case:
        #   src1 = visual odom
        #   src5 = dog_odom
        #
        trusted_pose_stuck = (
            self.enable_trusted_pose_fallback
            and gate
            and evidence_from_src5_twist
            and (not evidence_from_pose)
        )

        # ------------------------------------------------------------
        # 5) ROBUST IDLE / MOVING STATE
        # ------------------------------------------------------------
        #
        #
        # moving requires:
        #   effective_gate AND evidence
        #
        # effective_gate is slightly more tolerant than the raw gate:
        # it allows a short post-command tail so the bridge does not stop
        # tracking motion too early near the goal.

        # This avoids integrating motion when:
        #   - command exists but robot is physically stuck
        #   - source is noisy but robot is not actually commanded
        #
        #motion_state = self._update_motion_state(gate, evidence)
        motion_state = self._update_motion_state(gate_effective, evidence)
        moving = (motion_state == "moving")

        entered_idle = (self._prev_motion_state != "idle" and motion_state == "idle")
        entered_moving = (self._prev_motion_state != "moving" and motion_state == "moving")

        # ------------------------------------------------------------
        # 6) SOURCE LOCK MANAGEMENT
        # ------------------------------------------------------------
        #
        # Lock current source when entering MOVING.
        # Release lock when IDLE.
        #
        if self.lock_source_while_moving:
            if entered_moving:
                self._locked_source = src_name
            elif motion_state == "idle":
                self._locked_source = None



        # ------------------------------------------------------------
        # 7) MOTION REGIMES FOR FALLBACK NAVIGATION ODOM
        # ------------------------------------------------------------
        #
        # For non-trusted sources we integrate only when robot is truly moving.
        #
        # We distinguish three regimes:
        #
        #   1) commanded_motion
        #      Nav2 is still sending a meaningful command.
        #
        #   2) coast_motion
        #      Raw command is no longer meaningful, but the robot is still
        #      physically moving during braking / final residual motion.
        #
        #   3) idle
        #      No meaningful command and no convincing physical motion.
        #
        # Why this separation matters:
        # ----------------------------
        # - Using command only makes RViz reach the goal too early.
        # - Stopping too early causes teleportation after AMCL correction.
        # - Continuing to integrate angular command too long causes excessive
        #   final orientation correction after the robot is already stopped.
        #
        commanded_motion = gate

        coast_motion = (
            (not gate)
            and gate_effective
            and (
                evidence_from_selected_twist
                or evidence_from_pose
                or evidence_from_src5_twist
            )
        )

        # Default measured integration inputs for the non-trusted path
        vx_int = vx_meas if (moving and not turn_in_place_cmd) else 0.0
        wz_int = wz_meas if moving else 0.0

        use_source_pose = self._is_source_pose_trusted(src_name)

        if not use_source_pose:
            # --------------------------------------------------------
            # NON-TRUSTED SOURCE PATH
            # --------------------------------------------------------
            #
            # We do not publish source pose directly.
            #
            # Current behavior in this branch:
            #   - while command is active, integrate filtered measured twist
            #   - during coast phase, keep integrating residual measured motion
            #   - when idle, stop integrating
            #
            # Note:
            # -----
            # This branch is mostly for non-trusted sources such as /dog_odom.
            # The more advanced fused fallback is mainly used inside the trusted
            # branch when trusted pose becomes stuck.
            #
            integrate_yaw_now = (
                (moving or (turn_in_place_cmd and gate_effective))
                if not self.freeze_yaw_when_idle
                else moving
            )

            self._integrate(
                vx=vx_int if (commanded_motion or coast_motion) else 0.0,
                wz=wz_int if (commanded_motion or coast_motion) else 0.0,
                integrate_yaw=integrate_yaw_now,
            )

        # ------------------------------------------------------------
        # 8) BUILD OUTPUT ODOM MESSAGE
        # ------------------------------------------------------------
        out = Odometry()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = self.odom_frame
        out.child_frame_id = self.base_frame

        '''
                if use_source_pose:
                    # --------------------------------------------------------
                    # TRUSTED SOURCE POSE PATH
                    # --------------------------------------------------------
                    #
                    # Use source pose directly for odom publication.
                    # While idle, optionally freeze the published pose
                    # at the last stable anchor to avoid micro-motion.
                    #
                    raw_x = float(odom.pose.pose.position.x)
                    raw_y = float(odom.pose.pose.position.y)
                    raw_q = odom.pose.pose.orientation

                    raw_yaw = _yaw_from_quat(
                        float(raw_q.x), float(raw_q.y), float(raw_q.z), float(raw_q.w)
                    )

                    # Save anchor when entering idle
                    if entered_idle:
                        self._idle_anchor_x = raw_x
                        self._idle_anchor_y = raw_y
                        self._idle_anchor_yaw = raw_yaw
                        self._idle_anchor_valid = True

                    # If entering moving and no anchor exists yet, initialize it
                    if entered_moving and not self._idle_anchor_valid:
                        self._idle_anchor_x = raw_x
                        self._idle_anchor_y = raw_y
                        self._idle_anchor_yaw = raw_yaw
                        self._idle_anchor_valid = True

                    # Freeze published pose while idle if requested
                    if self.freeze_pose_when_idle and (not moving) and self._idle_anchor_valid:
                        x_out = self._idle_anchor_x
                        y_out = self._idle_anchor_y
                        yaw_out = self._idle_anchor_yaw
                    else:
                        x_out = raw_x
                        y_out = raw_y
                        yaw_out = raw_yaw

                    # Publish planar yaw only in odom -> base_footprint
                    qx, qy, qz, qw = _quat_from_yaw(yaw_out)

                    out.pose.pose.position.x = x_out
                    out.pose.pose.position.y = y_out
                    out.pose.pose.position.z = 0.0
                    out.pose.pose.orientation.x = qx
                    out.pose.pose.orientation.y = qy
                    out.pose.pose.orientation.z = qz
                    out.pose.pose.orientation.w = qw

                    # Use source twist, but zero it while idle if requested
                    out.twist = odom.twist

                    if self.zero_twist_when_idle and not moving:
                        out.twist.twist.linear.x = 0.0
                        out.twist.twist.linear.y = 0.0
                        out.twist.twist.linear.z = 0.0
                        out.twist.twist.angular.x = 0.0
                        out.twist.twist.angular.y = 0.0
                        out.twist.twist.angular.z = 0.0
        '''

        if use_source_pose:
            # --------------------------------------------------------
            # TRUSTED SOURCE POSE PATH
            # --------------------------------------------------------
            #
            # Normal behavior:
            #   publish trusted source pose directly
            #
            # Special robust behavior:
            #   if trusted pose appears stuck while the robot is clearly moving,
            #   switch temporarily to fallback integrated pose driven by src5 twist
            #
            raw_x = float(odom.pose.pose.position.x)
            raw_y = float(odom.pose.pose.position.y)
            raw_q = odom.pose.pose.orientation

            raw_yaw = _yaw_from_quat(
                float(raw_q.x), float(raw_q.y), float(raw_q.z), float(raw_q.w)
            )

            # Keep fallback aligned with trusted pose whenever fallback is not needed
            if not trusted_pose_stuck:
                self._reset_fallback_pose_from(raw_x, raw_y, raw_yaw)

            # Save idle anchor when entering idle
            if entered_idle:
                self._idle_anchor_x = raw_x
                self._idle_anchor_y = raw_y
                self._idle_anchor_yaw = raw_yaw
                self._idle_anchor_valid = True

            # Initialize anchor if entering moving and none exists
            if entered_moving and not self._idle_anchor_valid:
                self._idle_anchor_x = raw_x
                self._idle_anchor_y = raw_y
                self._idle_anchor_yaw = raw_yaw
                self._idle_anchor_valid = True

            # ------------------------------
            # CASE A: trusted pose is stuck
            # ------------------------------
            if trusted_pose_stuck and self.fallback_use_src5_twist:
                # ----------------------------------------------------
                # FALLBACK REGIMES WHEN TRUSTED POSE IS STUCK
                # ----------------------------------------------------
                #
                # We are in a special situation:
                #
                #   - selected trusted pose (e.g. visual odom) does not evolve
                #   - but auxiliary evidence says the robot is really moving
                #
                # In that case we temporarily stop trusting the frozen pose
                # and publish a fallback synthetic pose instead.
                #
                # We again distinguish:
                #
                #   A) commanded fallback
                #      use fused command + measured twist
                #
                #   B) coast fallback
                #      command has faded, but residual physical motion continues
                #
                #   C) idle fallback
                #      no more update
                #
                if not self._fallback_initialized:
                    self._reset_fallback_pose_from(raw_x, raw_y, raw_yaw)

                commanded_fallback = gate

                coast_fallback = (
                    (not gate)
                    and gate_effective
                    and (
                        evidence_from_selected_twist
                        or evidence_from_pose
                        or evidence_from_src5_twist
                    )
                )

                # For pure turning commands, suppress artificial forward motion
                # both on command side and measured side.
                if turn_in_place_cmd:
                    vx_cmd_fb = 0.0
                    vx_meas_fb = 0.0
                else:
                    vx_cmd_fb = vx_cmd
                    vx_meas_fb = vx_src5 if evidence_from_src5_twist else vx_meas

                # For angular motion, prefer src5 when available because it
                # usually reflects physical locomotion activity better than a
                # frozen trusted pose source.
                wz_meas_fb = wz_src5 if evidence_from_src5_twist else wz_meas

                if commanded_fallback:
                    # Normal commanded navigation phase:
                    # fused command model + measured activity
                    self._integrate_fallback_pose_fused(
                        vx_cmd=vx_cmd_fb,
                        wz_cmd=wz_cmd,
                        vx_meas=vx_meas_fb,
                        wz_meas=wz_meas_fb,
                    )

                elif coast_fallback:
                    # Final braking / residual motion:
                    # measured motion only
                    self._integrate_fallback_pose_measured(
                        vx_meas=vx_meas_fb,
                        wz_meas=wz_meas_fb,
                    )

                # else: true idle -> keep fallback frozen

                x_out = self._fallback_x
                y_out = self._fallback_y
                yaw_out = self._fallback_yaw

            # ------------------------------
            # CASE B: standard trusted pose
            # ------------------------------
            else:
                if self.freeze_pose_when_idle and (not moving) and self._idle_anchor_valid:
                    x_out = self._idle_anchor_x
                    y_out = self._idle_anchor_y
                    yaw_out = self._idle_anchor_yaw
                else:
                    x_out = raw_x
                    y_out = raw_y
                    yaw_out = raw_yaw

            # Publish only planar yaw in odom -> base_footprint
            qx, qy, qz, qw = _quat_from_yaw(yaw_out)

            out.pose.pose.position.x = x_out
            out.pose.pose.position.y = y_out
            out.pose.pose.position.z = 0.0
            out.pose.pose.orientation.x = qx
            out.pose.pose.orientation.y = qy
            out.pose.pose.orientation.z = qz
            out.pose.pose.orientation.w = qw

            # Keep source twist as output twist, but sanitize while idle
            out.twist = odom.twist

            if self.zero_twist_when_idle and not moving:
                out.twist.twist.linear.x = 0.0
                out.twist.twist.linear.y = 0.0
                out.twist.twist.linear.z = 0.0
                out.twist.twist.angular.x = 0.0
                out.twist.twist.angular.y = 0.0
                out.twist.twist.angular.z = 0.0
        else:
            # --------------------------------------------------------
            # INTEGRATED POSE PATH
            # --------------------------------------------------------
            #
            # Source pose is not trusted, so publish internal integrated pose.
            #
            qx, qy, qz, qw = _quat_from_yaw(self._yaw_int)

            out.pose.pose.position.x = float(self._x_int)
            out.pose.pose.position.y = float(self._y_int)
            out.pose.pose.position.z = 0.0
            out.pose.pose.orientation.x = qx
            out.pose.pose.orientation.y = qy
            out.pose.pose.orientation.z = qz
            out.pose.pose.orientation.w = qw

            tw = TwistWithCovariance()
            tw.twist.linear.x = float(vx_int if gate else 0.0)
            tw.twist.linear.y = 0.0
            tw.twist.linear.z = 0.0
            tw.twist.angular.x = 0.0
            tw.twist.angular.y = 0.0
            tw.twist.angular.z = float(wz_int if gate else 0.0)

            # Placeholder covariance
            tw.covariance = [0.0] * 36
            tw.covariance[0] = 0.20
            tw.covariance[7] = 0.20
            tw.covariance[35] = 0.40
            out.twist = tw

        # Pose covariance placeholders
        out.pose.covariance = [0.0] * 36
        out.pose.covariance[0] = 0.10
        out.pose.covariance[7] = 0.10
        out.pose.covariance[35] = 0.30

        # ------------------------------------------------------------
        # 9) PUBLISH
        # ------------------------------------------------------------
        self.odom_pub.publish(out)
        self._last_odom_msg = out

        # Save motion state for transition detection next cycle
        self._prev_motion_state = motion_state

        # ------------------------------------------------------------
        # 10) DEBUG STATE
        # ------------------------------------------------------------
        self._last_debug = {
            "src": src_name,
            "src_age": src_age,
            "locked_source": self._locked_source if self._locked_source else "",
            "gate": int(gate),
            "gate_effective": int(gate_effective),
            "gate_reason": gate_reason,
            "cmd_age": cmd_age,
            "vx_cmd": vx_cmd,
            "wz_cmd": wz_cmd,
            "vx_meas": vx_meas,
            "wz_meas": wz_meas,
            "commanded_motion": int(commanded_motion),
            "coast_motion": int(coast_motion),
            "evidence_sel_twist": int(evidence_from_selected_twist),
            "evidence_pose": int(evidence_from_pose),
            "evidence_src5": int(evidence_from_src5_twist),
            "evidence": int(evidence),
            "motion_state": motion_state,
            "trusted_pose_stuck": int(trusted_pose_stuck),
            "fallback_x": self._fallback_x,
            "fallback_y": self._fallback_y,
            "fallback_yaw": self._fallback_yaw,
            "moving": int(moving),
            "turn": int(turn_in_place_cmd),
            "use_source_pose": int(use_source_pose),
            "idle_anchor_valid": int(self._idle_anchor_valid),
            "vxy_pose_est": vxy_pose_est,
            "wz_pose_est": wz_pose_est,
            "vx_src5": vx_src5,
            "wz_src5": wz_src5,
            "x": out.pose.pose.position.x,
            "y": out.pose.pose.position.y,
            "yaw": _yaw_from_quat(
                float(out.pose.pose.orientation.x),
                float(out.pose.pose.orientation.y),
                float(out.pose.pose.orientation.z),
                float(out.pose.pose.orientation.w),
            ),
        }

    def _tick_odom_old(self) -> None:
        """
        Main odometry production loop.

        Strategy used here:
        1. Select the best available source
        2. Use the navigation command as a motion gate
        3. Filter measured twist
        4. Build a robust motion state: moving / idle
        5. If source pose is trusted:
            - keep reading raw pose continuously
            - freeze the PUBLISHED pose while idle
            - publish yaw-only orientation in odom -> base_footprint
        6. Otherwise integrate motion only when truly moving
        """

        src_name, odom, src_age = self._pick_active_source()
        if odom is None:
            return

        gate, gate_reason, cmd_age, vx_cmd, wz_cmd = self._gate_active()

        # Detect a pure turning intent from navigation.
        # In that case, forward integration should not create fake linear motion.
        turn_in_place_cmd = (abs(wz_cmd) > self.turn_wz_thresh) and (abs(vx_cmd) < self.turn_vx_small)

        vx_meas, wz_meas = self._correct_filter(src_name, odom)

        if turn_in_place_cmd:
            vx_meas = 0.0

        # Motion evidence from measurements.
        # We do not want to trust command intent alone because the robot can be
        # commanded but still physically stuck, or command can be stale / noisy.
        evidence = (abs(vx_meas) > self.meas_eps_vx) or (abs(wz_meas) > self.meas_eps_wz)

        # Robust motion state with hysteresis.
        motion_state = self._update_motion_state(gate, evidence)
        moving = (motion_state == "moving")

        entered_idle = (self._prev_motion_state != "idle" and motion_state == "idle")
        entered_moving = (self._prev_motion_state != "moving" and motion_state == "moving")

        # Integration inputs are allowed only when the robot is truly moving.
        vx_int = vx_meas if (moving and not turn_in_place_cmd) else 0.0
        wz_int = wz_meas if moving else 0.0

        use_source_pose = self._is_source_pose_trusted(src_name)

        if not use_source_pose:
            # Non-trusted source path:
            # integrate from filtered twist rather than copying the source pose.
            self._integrate(
                vx=vx_int,
                wz=wz_int,
                integrate_yaw=(moving or (turn_in_place_cmd and gate)) if not self.freeze_yaw_when_idle else moving
            )

        out = Odometry()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = self.odom_frame
        out.child_frame_id = self.base_frame

        if use_source_pose:
            # ------------------------------------------------------------------
            # Trusted source pose path
            # ------------------------------------------------------------------
            # Important idea:
            # - we ALWAYS keep reading the raw source pose
            # - but while idle, we do NOT necessarily publish that raw pose
            # - instead, we can freeze the published navigation pose to an anchor
            #
            # This prevents stabilization micro-motion from polluting Nav2.
            raw_x = float(odom.pose.pose.position.x)
            raw_y = float(odom.pose.pose.position.y)
            raw_q = odom.pose.pose.orientation
            raw_yaw = _yaw_from_quat(
                float(raw_q.x), float(raw_q.y), float(raw_q.z), float(raw_q.w)
            )

            # If we just entered idle, store the current trusted pose as the
            # standstill anchor. This anchor will be republished while idle.
            if entered_idle:
                self._idle_anchor_x = raw_x
                self._idle_anchor_y = raw_y
                self._idle_anchor_yaw = raw_yaw
                self._idle_anchor_valid = True

            # If we just entered moving, we simply stop freezing the trusted pose.
            # If desired later, a smooth blend-back can be added here.
            if entered_moving and not self._idle_anchor_valid:
                self._idle_anchor_x = raw_x
                self._idle_anchor_y = raw_y
                self._idle_anchor_yaw = raw_yaw
                self._idle_anchor_valid = True

            # Decide what pose should be PUBLISHED.
            #
            # While idle:
            #   publish the frozen anchor pose
            # While moving:
            #   publish the live trusted source pose
            if self.freeze_pose_when_idle and (not moving) and self._idle_anchor_valid:
                x_out = self._idle_anchor_x
                y_out = self._idle_anchor_y
                yaw_out = self._idle_anchor_yaw
            else:
                x_out = raw_x
                y_out = raw_y
                yaw_out = raw_yaw

            # Publish ONLY yaw in odom -> base_footprint.
            # Roll and pitch belong in base_footprint -> body_frame.
            qx, qy, qz, qw = _quat_from_yaw(yaw_out)

            out.pose.pose.position.x = x_out
            out.pose.pose.position.y = y_out
            out.pose.pose.position.z = 0.0
            out.pose.pose.orientation.x = qx
            out.pose.pose.orientation.y = qy
            out.pose.pose.orientation.z = qz
            out.pose.pose.orientation.w = qw

            # Forward source twist, but sanitize it if the robot is idle.
            out.twist = odom.twist

            if self.zero_twist_when_idle and not moving:
                out.twist.twist.linear.x = 0.0
                out.twist.twist.linear.y = 0.0
                out.twist.twist.linear.z = 0.0
                out.twist.twist.angular.x = 0.0
                out.twist.twist.angular.y = 0.0
                out.twist.twist.angular.z = 0.0

        else:
            # ------------------------------------------------------------------
            # Integrated pose path
            # ------------------------------------------------------------------
            # Used when source pose is not trusted enough for direct publication.
            qx, qy, qz, qw = _quat_from_yaw(self._yaw_int)

            out.pose.pose.position.x = float(self._x_int)
            out.pose.pose.position.y = float(self._y_int)
            out.pose.pose.position.z = 0.0
            out.pose.pose.orientation.x = qx
            out.pose.pose.orientation.y = qy
            out.pose.pose.orientation.z = qz
            out.pose.pose.orientation.w = qw

            tw = TwistWithCovariance()
            tw.twist.linear.x = float(vx_int if gate else 0.0)
            tw.twist.linear.y = 0.0
            tw.twist.linear.z = 0.0
            tw.twist.angular.x = 0.0
            tw.twist.angular.y = 0.0
            tw.twist.angular.z = float(wz_int if gate else 0.0)

            tw.covariance = [0.0] * 36
            tw.covariance[0] = 0.20
            tw.covariance[7] = 0.20
            tw.covariance[35] = 0.40
            out.twist = tw

        # Pose covariance placeholders.
        out.pose.covariance = [0.0] * 36
        out.pose.covariance[0] = 0.10
        out.pose.covariance[7] = 0.10
        out.pose.covariance[35] = 0.30

        self.odom_pub.publish(out)
        self._last_odom_msg = out

        # Keep the previous state for transition detection on the next tick.
        self._prev_motion_state = motion_state

        # Save debug state for status publication.
        self._last_debug = {
            "src": src_name,
            "src_age": src_age,
            "gate": int(gate),
            "gate_reason": gate_reason,
            "cmd_age": cmd_age,
            "vx_cmd": vx_cmd,
            "wz_cmd": wz_cmd,
            "vx_meas": vx_meas,
            "wz_meas": wz_meas,
            "evidence": int(evidence),
            "motion_state": motion_state,
            "moving": int(moving),
            "turn": int(turn_in_place_cmd),
            "use_source_pose": int(use_source_pose),
            "idle_anchor_valid": int(self._idle_anchor_valid),
            "x": out.pose.pose.position.x,
            "y": out.pose.pose.position.y,
            "yaw": _yaw_from_quat(
                float(out.pose.pose.orientation.x),
                float(out.pose.pose.orientation.y),
                float(out.pose.pose.orientation.z),
                float(out.pose.pose.orientation.w),
            ),
        }
        
    def _tick_tf_main(self) -> None:
        """
        Publish the MAIN navigation TF:

            odom -> base_footprint

        This is the TF consumed by Nav2 for local motion.

        IMPORTANT:
        - This TF must be unique in the system.
        - It must be consistent with the odometry message published by this node.
        """
        if self._last_odom_msg is None:
            return

        tfm = TransformStamped()
        tfm.header.stamp = self._last_odom_msg.header.stamp
        tfm.header.frame_id = self.odom_frame
        tfm.child_frame_id = self.base_frame

        tfm.transform.translation.x = float(self._last_odom_msg.pose.pose.position.x)
        tfm.transform.translation.y = float(self._last_odom_msg.pose.pose.position.y)
        tfm.transform.translation.z = 0.0

        tfm.transform.rotation = self._last_odom_msg.pose.pose.orientation

        self.tf_broadcaster.sendTransform(tfm)

    def _tick_tf_body(self) -> None:
        """
        Publish the body attitude TF:

            base_footprint -> robot_center

        What is stored here:
        - z
        - roll
        - pitch

        What is NOT stored here:
        - yaw

        Why yaw is forced to zero here:
        - yaw must belong to odom -> base_footprint
        - splitting yaw across both transforms would corrupt the navigation frame
        """
        src_name, odom, _ = self._pick_active_source()
        if odom is None:
            return

        p = odom.pose.pose.position
        q = odom.pose.pose.orientation

        roll, pitch, _ = _rpy_from_quat(float(q.x), float(q.y), float(q.z), float(q.w))
        bqx, bqy, bqz, bqw = _quat_from_rpy(roll, pitch, 0.0)

        tfm = TransformStamped()
        tfm.header.stamp = self.get_clock().now().to_msg()
        tfm.header.frame_id = self.base_frame
        tfm.child_frame_id = self.body_frame

        tfm.transform.translation.x = 0.0
        tfm.transform.translation.y = 0.0
        tfm.transform.translation.z = float(p.z)

        tfm.transform.rotation.x = bqx
        tfm.transform.rotation.y = bqy
        tfm.transform.rotation.z = bqz
        tfm.transform.rotation.w = bqw

        self.tf_broadcaster.sendTransform(tfm)

    def _tick_status(self) -> None:
        """
        Publish a compact human-readable status string.

        This is very helpful to understand:
        - which source is currently selected
        - whether integration is active
        - whether source pose is directly trusted
        """
        if self.status_pub is None or not self._last_debug:
            return

        d = self._last_debug
        msg = String()
        msg.data = (
            "{"
            f"\"src\":\"{d['src']}\","
            f"\"src_age\":{d['src_age']:.3f},"
            f"\"locked_source\":\"{d['locked_source']}\","
            f"\"gate\":{d['gate']},"
            f"\"gate_effective\":{d['gate_effective']},"
            f"\"gate_reason\":\"{d['gate_reason']}\","
            f"\"cmd_age\":{d['cmd_age']:.3f},"
            f"\"vx_cmd\":{d['vx_cmd']:.3f},"
            f"\"wz_cmd\":{d['wz_cmd']:.3f},"
            f"\"vx_meas\":{d['vx_meas']:.3f},"
            f"\"wz_meas\":{d['wz_meas']:.3f},"
            f"\"commanded_motion\":{d['commanded_motion']},"
            f"\"coast_motion\":{d['coast_motion']},"
            f"\"trusted_pose_stuck\":{d['trusted_pose_stuck']},"
            f"\"fallback_x\":{d['fallback_x']:.3f},"
            f"\"fallback_y\":{d['fallback_y']:.3f},"
            f"\"fallback_yaw\":{d['fallback_yaw']:.3f},"
            f"\"evidence_sel_twist\":{d['evidence_sel_twist']},"
            f"\"evidence_pose\":{d['evidence_pose']},"
            f"\"evidence_src5\":{d['evidence_src5']},"
            f"\"evidence\":{d['evidence']},"
            f"\"motion_state\":\"{d['motion_state']}\","
            f"\"moving\":{d['moving']},"
            f"\"turn\":{d['turn']},"
            f"\"use_source_pose\":{d['use_source_pose']},"
            f"\"vxy_pose_est\":{d['vxy_pose_est']:.3f},"
            f"\"wz_pose_est\":{d['wz_pose_est']:.3f},"
            f"\"vx_src5\":{d['vx_src5']:.3f},"
            f"\"wz_src5\":{d['wz_src5']:.3f},"
            f"\"x\":{d['x']:.3f},"
            f"\"y\":{d['y']:.3f},"
            f"\"yaw\":{d['yaw']:.3f}"
            "}"
        )

        self.status_pub.publish(msg)


def main() -> None:
    """
    Standard ROS 2 entry point.
    """
    rclpy.init()
    node = SmartOdomBridgeCmdGated()
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