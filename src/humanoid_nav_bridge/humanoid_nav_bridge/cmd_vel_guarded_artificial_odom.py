#!/usr/bin/env python3
"""
cmd_vel_guarded_artificial_odom.py

===============================================================================
PURPOSE
===============================================================================

This node creates a SYNTHETIC NAVIGATION ODOMETRY for Nav2.

It is specifically useful in difficult real-robot situations where:

  - visual odometry is unstable, delayed, oscillatory, or inconsistent
  - raw locomotion odometry drifts because of RL stabilization behavior
  - Nav2 needs a smooth and monotonic odom to properly reduce goal distance

Instead of trusting a problematic pose source directly, this node builds
an odom estimate from the NAVIGATION COMMAND itself:

    cmd_vel_nav_scaled  -->  integrated navigation odom

BUT with an important safety mechanism:

    dog_odom twist  -->  evidence that robot is actually moving

So the odom is NOT blindly integrated forever.
It advances only when the robot appears to be really executing motion.

===============================================================================
WHY THIS NODE CAN HELP
===============================================================================

In your current system, the core issue is:

  - The physical robot walks
  - But the odom consumed by Nav2 does not evolve consistently
  - Therefore Nav2 thinks the robot is not progressing correctly
  - Then recoveries / relocalization / overshoot may happen

This node changes the logic:

  1) Use cmd_vel as the primary "intent of motion"
  2) Integrate that command into a planar odom pose
  3) Add configurable artificial noise to avoid an unrealistically perfect odom
  4) Only integrate when there is real motion evidence from dog_odom twist
  5) Publish a stable /odom + TF(odom -> base_footprint)

This often gives Nav2 a much more usable control odom on RL-based robots.

===============================================================================
IMPORTANT LIMITATION
===============================================================================

This odom is NOT a true physical state estimator.

It is a CONTROL / NAVIGATION odom.

That means:
  - it is often better for short-horizon navigation control
  - but it is not a replacement for a globally accurate localization system

In practice, this is acceptable because:
  - AMCL / SLAM can still provide map -> odom
  - this node only provides the local odom -> base_footprint behavior

===============================================================================
TARGET TF TREE
===============================================================================

    map -> odom -> base_footprint -> robot_center -> sensors...

Where:
  - map -> odom comes from AMCL or SLAM
  - odom -> base_footprint comes from THIS NODE
  - base_footprint -> robot_center comes from your existing body TF logic

===============================================================================
DESIGN SUMMARY
===============================================================================

Inputs:
  - /cmd_vel_nav_scaled   : navigation command to integrate
  - /dog_odom             : motion evidence only (twist), not trusted as pose

Outputs:
  - /odom
  - TF odom -> base_footprint

Key idea:
  - integrate cmd_vel only when:
        command is recent enough
        AND robot seems physically moving
  - keep a short post-command grace period
        because RL robots may keep moving after command becomes small
"""

from __future__ import annotations

import math
import random
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from geometry_msgs.msg import Twist, TransformStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import String
import tf2_ros


# =============================================================================
# QoS helper
# =============================================================================
def _make_qos(reliable: bool, depth: int) -> QoSProfile:
    """
    Small helper to create QoS profiles.

    Why this exists:
    ----------------
    Robot topics may use RELIABLE or BEST_EFFORT depending on the driver.
    Keeping QoS creation centralized avoids repeating boilerplate.
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
    Normalize angle into [-pi, pi].

    This keeps yaw numerically stable and easier to debug.
    """
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def _quat_from_yaw(yaw: float) -> Tuple[float, float, float, float]:
    """
    Create a quaternion representing a pure planar yaw rotation.
    """
    h = 0.5 * yaw
    return (0.0, 0.0, math.sin(h), math.cos(h))


class CmdVelGuardedArtificialOdom(Node):
    def __init__(self) -> None:
        super().__init__("cmd_vel_guarded_artificial_odom")

        # =========================================================================
        # FRAME / TOPIC PARAMETERS
        # =========================================================================
        #
        # odom_frame:
        #   local odometry frame used by Nav2
        #
        # base_frame:
        #   planar robot frame used for navigation
        #
        # output_odom_topic:
        #   odom topic published by this node
        #
        self.odom_frame = self.declare_parameter("odom_frame", "odom").value
        self.base_frame = self.declare_parameter("base_frame", "base_footprint").value
        self.output_odom_topic = self.declare_parameter("output_odom_topic", "/odom").value

        # =========================================================================
        # INPUT TOPICS
        # =========================================================================
        #
        # cmd_topic:
        #   navigation command to integrate
        #
        # dog_odom_topic:
        #   locomotion odom used ONLY as motion evidence, not as trusted pose
        #
        self.cmd_topic = self.declare_parameter("cmd_topic", "/cmd_vel_nav_scaled").value
        self.dog_odom_topic = self.declare_parameter("dog_odom_topic", "/dog_odom").value

        # =========================================================================
        # COMMAND VALIDITY / HOLD
        # =========================================================================
        #
        # cmd_timeout_sec:
        #   if no command arrives for longer than this, the command is stale
        #
        # gate_release_hold_sec:
        #   after command becomes weak or stale, keep a short grace period
        #   because RL-based robots may still move physically for a while
        #
        self.cmd_timeout_sec = float(self.declare_parameter("cmd_timeout_sec", 1.0).value)
        self.gate_release_hold_sec = float(self.declare_parameter("gate_release_hold_sec", 1.5).value)

        # =========================================================================
        # COMMAND EPSILONS
        # =========================================================================
        #
        # Small commands near zero should not always be treated as an active motion
        # request. These thresholds define when command is considered "active".
        #
        self.cmd_eps_vx = float(self.declare_parameter("cmd_eps_vx", 0.005).value)
        self.cmd_eps_wz = float(self.declare_parameter("cmd_eps_wz", 0.005).value)

        # =========================================================================
        # ARTIFICIAL NOISE MODEL
        # =========================================================================
        #
        # This implements your supervisor's recommendation:
        #
        #   linear noise ~ 5%
        #   angular noise ~ 15%
        #
        # Why add noise?
        # --------------
        # A perfectly integrated cmd_vel odom may look unrealistically ideal.
        # Small artificial noise makes it more plausible and less brittle.
        #
        self.linear_noise_ratio = float(self.declare_parameter("linear_noise_ratio", 0.05).value)
        self.angular_noise_ratio = float(self.declare_parameter("angular_noise_ratio", 0.15).value)
        self.random_seed = int(self.declare_parameter("random_seed", 42).value)

        rng = random.Random(self.random_seed)
        self._rng = rng

        # =========================================================================
        # MOTION EVIDENCE FROM dog_odom TWIST
        # =========================================================================
        #
        # We do not want to integrate cmd_vel if the robot is not actually moving.
        # So we inspect dog_odom twist and require some evidence of real motion.
        #
        self.use_motion_evidence = bool(self.declare_parameter("use_motion_evidence", True).value)

        self.meas_eps_vx = float(self.declare_parameter("meas_eps_vx", 0.03).value)
        self.meas_eps_wz = float(self.declare_parameter("meas_eps_wz", 0.05).value)

        self.deadband_vx = float(self.declare_parameter("deadband_vx", 0.02).value)
        self.deadband_wz = float(self.declare_parameter("deadband_wz", 0.04).value)
        self.ema_alpha = float(self.declare_parameter("ema_alpha", 0.25).value)

        # If dog_odom uses angular.y instead of angular.z for yaw rate,
        # this can be enabled.
        self.remap_yawrate_y_to_z = bool(
            self.declare_parameter("remap_yawrate_y_to_z", False).value
        )

        # =========================================================================
        # TURN-IN-PLACE HANDLING
        # =========================================================================
        #
        # If navigation requests mainly rotation with very small forward speed,
        # do not inject fake forward displacement.
        #
        self.turn_wz_thresh = float(self.declare_parameter("turn_wz_thresh", 0.25).value)
        self.turn_vx_small = float(self.declare_parameter("turn_vx_small", 0.04).value)

        # =========================================================================
        # IDLE / MOVING HYSTERESIS
        # =========================================================================
        #
        # Without hysteresis, the node may toggle too quickly between moving/idle
        # due to small command fluctuations or noisy twist.
        #
        self.motion_confirm_sec = float(self.declare_parameter("motion_confirm_sec", 0.10).value)
        self.idle_confirm_sec = float(self.declare_parameter("idle_confirm_sec", 1.0).value)

        # =========================================================================
        # PUBLICATION CONTROL
        # =========================================================================
        self.publish_tf = bool(self.declare_parameter("publish_tf", True).value)
        self.publish_status = bool(self.declare_parameter("publish_status", True).value)

        self.odom_rate_hz = float(self.declare_parameter("odom_rate_hz", 50.0).value)
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
        # /cmd_vel_nav_scaled usually comes from local Nav2/scaler, keep RELIABLE
        cmd_qos = _make_qos(True, 1)

        dog_odom_qos = _make_qos(
            self.dog_odom_reliable,
            self.qos_depth
        )

        # =========================================================================
        # INTERNAL SYNTHETIC ODOM STATE
        # =========================================================================
        #
        # This is the pose Nav2 will see.
        #
        self._x = 0.0
        self._y = 0.0
        self._yaw = 0.0

        self._last_int_time = self.get_clock().now()

        # =========================================================================
        # INTERNAL COMMAND STATE
        # =========================================================================
        self._last_cmd: Optional[Twist] = None
        self._last_cmd_time: Optional[Time] = None

        # =========================================================================
        # INTERNAL dog_odom STATE
        # =========================================================================
        self._last_dog_odom: Optional[Odometry] = None
        self._last_dog_odom_time: Optional[Time] = None

        # Filter state for dog_odom twist
        self._vx_f = 0.0
        self._wz_f = 0.0

        # =========================================================================
        # INTERNAL MOVING / IDLE STATE
        # =========================================================================
        self._motion_state = "idle"
        self._candidate_state = "idle"
        self._candidate_since = self.get_clock().now()

        # =========================================================================
        # LAST PUBLISHED ODOM
        # =========================================================================
        self._last_odom_msg: Optional[Odometry] = None
        self._last_debug = {}

        # =========================================================================
        # ROS INTERFACES
        # =========================================================================
        self.cmd_sub = self.create_subscription(
            Twist,
            self.cmd_topic,
            self._cb_cmd,
            cmd_qos,
        )

        self.dog_odom_sub = self.create_subscription(
            Odometry,
            self.dog_odom_topic,
            self._cb_dog_odom,
            dog_odom_qos,
            #sub_qos,
        )

        self.odom_pub = self.create_publisher(
            Odometry,
            self.output_odom_topic,
            pub_qos,
        )

        self.status_pub = None
        if self.publish_status:
            self.status_pub = self.create_publisher(
                String,
                "/cmd_vel_guarded_artificial_odom/status",
                pub_qos,
            )

        self.tf_broadcaster = None
        if self.publish_tf:
            self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)


        self.create_timer(1.0 / self.odom_rate_hz, self._tick_odom)

        if self.publish_status and self.status_rate_hz > 0.0:
            self.create_timer(1.0 / self.status_rate_hz, self._tick_status)

        self.get_logger().info(
            "CmdVel guarded artificial odom started:\n"
            f"  cmd topic          : {self.cmd_topic}\n"
            f"  dog_odom topic     : {self.dog_odom_topic}\n"
            f"  output odom topic  : {self.output_odom_topic}\n"
            f"  TF publish         : {'ON' if self.publish_tf else 'OFF'} ({self.odom_frame}->{self.base_frame})\n"
            f"  linear noise ratio : {self.linear_noise_ratio:.3f}\n"
            f"  angular noise ratio: {self.angular_noise_ratio:.3f}"
        )

        # Minimum measured motion required to keep coast mode alive.
        # This avoids prolonging local odom motion because of tiny residual
        # oscillations near the goal.
        self.coast_meas_vx_min = float(
            self.declare_parameter("coast_meas_vx_min", 0.08).value
        )
        self.coast_meas_wz_min = float(
            self.declare_parameter("coast_meas_wz_min", 0.12).value
        )

        # =========================================================================
        # ONLINE ADAPTIVE CALIBRATION OF COMMAND SCALES
        # =========================================================================
        #
        # Goal:
        # -----
        # Reduce the mismatch between:
        #   - navigation odom predicted from command
        #   - real robot motion measured from odom / twist
        #
        # Why this matters:
        # -----------------
        # If the synthetic navigation odom moves faster than the real robot,
        # Nav2 will think the goal is reached too early and stop the robot
        # before the physical platform actually reaches the target.
        #
        # Strategy:
        # ---------
        # Estimate, online and slowly, how much real motion is actually produced
        # by a given command.
        #
        # Example:
        #   if vx_cmd = 0.40 m/s but robot usually produces only 0.28 m/s,
        #   then linear_cmd_scale should converge toward ~0.70
        #
        self.enable_adaptive_scale = bool(
            self.declare_parameter("enable_adaptive_scale", True).value
        )

        # Learning rates:
        # small values = slow and stable adaptation
        self.adapt_lr_lin = float(
            self.declare_parameter("adapt_lr_lin", 0.02).value
        )
        self.adapt_lr_ang = float(
            self.declare_parameter("adapt_lr_ang", 0.02).value
        )

        # Only adapt when commanded motion is sufficiently large.
        # This avoids learning from tiny noisy commands near zero.
        self.adapt_min_cmd_vx = float(
            self.declare_parameter("adapt_min_cmd_vx", 0.08).value
        )
        self.adapt_min_cmd_wz = float(
            self.declare_parameter("adapt_min_cmd_wz", 0.12).value
        )

        # Plausible allowed bounds for the adaptive scales.
        # These bounds protect against bad measurements or transient spikes.
        self.linear_cmd_scale_min = float(
            self.declare_parameter("linear_cmd_scale_min", 0.35).value
        )
        self.linear_cmd_scale_max = float(
            self.declare_parameter("linear_cmd_scale_max", 1.00).value
        )
        self.angular_cmd_scale_min = float(
            self.declare_parameter("angular_cmd_scale_min", 0.30).value
        )
        self.angular_cmd_scale_max = float(
            self.declare_parameter("angular_cmd_scale_max", 1.00).value
        )

        # Initial calibration factors used by synthetic odom integration.
        # These scales represent how much of the commanded motion is expected
        # to be really executed by the robot.
        self.linear_cmd_scale = float(
            self.declare_parameter("linear_cmd_scale", 0.75).value
        )
        self.angular_cmd_scale = float(
            self.declare_parameter("angular_cmd_scale", 0.70).value
        )

        self.physical_turn_min = 0.18
        self.physical_move_min = 0.05
    
        # =========================================================================
        # CONTEXTUAL YAW FUSION
        # =========================================================================
        #
        # Goal:
        # -----
        # Keep synthetic odom mainly command-driven, but prevent yaw from staying
        # unrealistically straight when the real robot rotates more than expected.
        #
        # Strategy:
        # ---------
        # Blend commanded angular velocity and measured angular velocity:
        #
        #   wz_model = alpha_yaw * wz_cmd_model + (1 - alpha_yaw) * wz_meas
        #
        # where alpha_yaw is adapted depending on context:
        #   - cruise / strong forward motion -> trust command more
        #   - near-goal / low-speed / coast / high mismatch -> trust measurement more
        #
        self.enable_contextual_yaw_fusion = bool(
            self.declare_parameter("enable_contextual_yaw_fusion", True).value
        )

        # Base fusion factor:
        #   1.0 -> pure command
        #   0.0 -> pure measured yaw-rate
        self.alpha_yaw_base = float(
            self.declare_parameter("alpha_yaw_base", 0.60).value
        )

        # Low-speed threshold:
        # below this, we reduce alpha_yaw and trust measured yaw more
        self.yaw_low_speed_vx_thresh = float(
            self.declare_parameter("yaw_low_speed_vx_thresh", 0.10).value
        )

        # Additional reduction of alpha_yaw when vx command is small
        self.alpha_yaw_low_speed_penalty = float(
            self.declare_parameter("alpha_yaw_low_speed_penalty", 0.25).value
        )

        # If coast mode is active, measured yaw is often more informative
        self.alpha_yaw_coast_penalty = float(
            self.declare_parameter("alpha_yaw_coast_penalty", 0.20).value
        )

        # If commanded and measured yaw disagree a lot, reduce alpha_yaw
        self.yaw_mismatch_thresh = float(
            self.declare_parameter("yaw_mismatch_thresh", 0.08).value
        )
        self.alpha_yaw_mismatch_penalty = float(
            self.declare_parameter("alpha_yaw_mismatch_penalty", 0.30).value
        )

        # Clamp final fusion factor
        self.alpha_yaw_min = float(
            self.declare_parameter("alpha_yaw_min", 0.15).value
        )
        self.alpha_yaw_max = float(
            self.declare_parameter("alpha_yaw_max", 0.90).value
        )

        self._last_alpha_yaw = 1.0
        self._last_wz_model = 0.0
        self._last_vx_model = 0.0



    # =========================================================================
    # BASIC HELPERS
    # =========================================================================
    def _now(self) -> Time:
        return self.get_clock().now()

    def _cb_cmd(self, msg: Twist) -> None:
        """
        Store latest navigation command.
        """
        self._last_cmd = msg
        self._last_cmd_time = self._now()

    def _clamp(self, v: float, lo: float, hi: float) -> float:
        """
        Clamp a scalar value into a closed interval [lo, hi].
        """
        return max(lo, min(hi, v))
    
    def _cb_dog_odom(self, msg: Odometry) -> None:
        """
        Store latest dog_odom sample.

        Important:
        ----------
        We do NOT trust dog_odom pose here.
        Only twist is used as physical-motion evidence.
        """
        self._last_dog_odom = msg
        self._last_dog_odom_time = self._now()

    def _extract_wz_from_dog_odom(self, odom: Odometry) -> float:
        """
        Extract yaw-rate from dog_odom.
        """
        wy = float(odom.twist.twist.angular.y)
        wz = float(odom.twist.twist.angular.z)

        if self.remap_yawrate_y_to_z:
            return wy
        return wz

    def _raw_gate(self) -> Tuple[bool, str, float, float, float]:
        """
        Compute the raw command gate.

        A raw gate is active only when:
          - a recent command exists
          - and its magnitude is above configured epsilons
        """
        if self._last_cmd is None or self._last_cmd_time is None:
            return False, "no_cmd", 999.0, 0.0, 0.0

        now = self._now()
        cmd_age = (now - self._last_cmd_time).nanoseconds * 1e-9
        vx_cmd = float(self._last_cmd.linear.x)
        wz_cmd = float(self._last_cmd.angular.z)

        if cmd_age > self.cmd_timeout_sec:
            return False, "cmd_stale", cmd_age, vx_cmd, wz_cmd

        active = (abs(vx_cmd) > self.cmd_eps_vx) or (abs(wz_cmd) > self.cmd_eps_wz)
        return active, ("cmd_active" if active else "cmd_below_eps"), cmd_age, vx_cmd, wz_cmd

    def _effective_gate(self, raw_gate: bool, cmd_age: float) -> bool:
        """
        Compute a more tolerant gate.

        Why:
        ----
        Near the goal, Nav2 may reduce commands below epsilon before the robot
        has physically stopped. RL locomotion may continue for a short time.

        So we allow a short grace period after command becomes weak or stale.
        """
        if raw_gate:
            return True

        return cmd_age <= self.gate_release_hold_sec

    def _filtered_dog_twist(self) -> Tuple[float, float]:
        """
        Read dog_odom twist, apply deadband + EMA.

        Returns:
            filtered_vx, filtered_wz
        """
        if self._last_dog_odom is None or self._last_dog_odom_time is None:
            return 0.0, 0.0

        vx = float(self._last_dog_odom.twist.twist.linear.x)
        wz = self._extract_wz_from_dog_odom(self._last_dog_odom)

        # Deadband
        if abs(vx) < self.deadband_vx:
            vx = 0.0
        if abs(wz) < self.deadband_wz:
            wz = 0.0

        # EMA
        self._vx_f = self.ema_alpha * vx + (1.0 - self.ema_alpha) * self._vx_f
        self._wz_f = self.ema_alpha * wz + (1.0 - self.ema_alpha) * self._wz_f

        return self._vx_f, self._wz_f

    def _motion_evidence(self, vx_meas: float, wz_meas: float) -> bool:
        """
        Decide whether the robot appears to be physically moving.

        If use_motion_evidence is disabled, we always accept integration.
        """
        if not self.use_motion_evidence:
            return True

        return (abs(vx_meas) > self.meas_eps_vx) or (abs(wz_meas) > self.meas_eps_wz)

    def _update_motion_state(self, gate_effective: bool, evidence: bool) -> str:
        """
        Robust moving/idle state machine with hysteresis.

        MOVING requires:
            gate_effective AND evidence

        IDLE is entered only after a longer confirmation window.
        """
        now = self._now()
        desired = "moving" if (gate_effective and evidence) else "idle"

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
    
    def _compute_contextual_alpha_yaw(
        self,
        vx_cmd: float,
        wz_cmd_model: float,
        wz_meas: float,
        coast_motion: bool,
    ) -> float:
        """
        Compute a contextual blending factor for yaw integration.

        Returned value:
            alpha_yaw in [alpha_yaw_min, alpha_yaw_max]

        Interpretation:
            alpha_yaw close to 1.0:
                trust commanded yaw more

            alpha_yaw close to 0.0:
                trust measured yaw more

        We reduce alpha_yaw when:
            - commanded forward speed is low
            - coast mode is active
            - measured yaw differs significantly from commanded yaw model

        This is especially useful near the goal, where small velocity commands
        and residual robot rotations make pure command-driven yaw too optimistic.
        """
        alpha = self.alpha_yaw_base

        # 1) Low-speed regime: near goal / delicate approach
        if abs(vx_cmd) < self.yaw_low_speed_vx_thresh:
            alpha -= self.alpha_yaw_low_speed_penalty

        # 2) Coast regime: command has weakened, but robot still moves
        if coast_motion:
            alpha -= self.alpha_yaw_coast_penalty

        # 3) Yaw mismatch regime: robot rotates differently from expected
        yaw_err = abs(wz_meas - wz_cmd_model)
        if yaw_err > self.yaw_mismatch_thresh:
            alpha -= self.alpha_yaw_mismatch_penalty

        alpha = self._clamp(alpha, self.alpha_yaw_min, self.alpha_yaw_max)
        return alpha

    def _update_adaptive_scales(
        self,
        vx_cmd: float,
        wz_cmd: float,
        vx_meas: float,
        wz_meas: float,
        gate: bool,
    ) -> None:
        """
        Update command calibration scales online from measured motion.

        Purpose:
        --------
        Learn, slowly and safely, how much real motion the robot actually
        produces for a given commanded motion.

        This is useful because the humanoid RL locomotion stack may:
            - under-execute translation
            - under-execute rotation
            - vary with floor, stabilization, and gait state

        Update rule:
        ------------
        We compare absolute measured motion to absolute commanded motion:

            instantaneous_linear_gain  = |vx_meas| / |vx_cmd|
            instantaneous_angular_gain = |wz_meas| / |wz_cmd|

        Then we apply a slow exponential update to:
            self.linear_cmd_scale
            self.angular_cmd_scale

        Safety:
        -------
        - only adapt when command is really active
        - ignore tiny commands
        - clamp learned scales into plausible bounds
        """
        if not self.enable_adaptive_scale:
            return

        if not gate:
            return

        eps = 1e-6

        # ------------------------------------------------------------
        # Linear adaptation
        # ------------------------------------------------------------
        # Linear adaptive update intentionally disabled for now.
        # Reason:
        #   the dominant mismatch observed in experiments is angular, not linear.
        # Keeping linear_cmd_scale fixed makes the synthetic odom easier to
        # interpret and avoids creating new translation bias while we fix yaw.

        #if abs(vx_cmd) >= self.adapt_min_cmd_vx:
        #    gain_lin_inst = abs(vx_meas) / max(abs(vx_cmd), eps)

        #    self.linear_cmd_scale = (
        #        (1.0 - self.adapt_lr_lin) * self.linear_cmd_scale
        #        + self.adapt_lr_lin * gain_lin_inst
        #    )

        #    self.linear_cmd_scale = self._clamp(
        #        self.linear_cmd_scale,
        #        self.linear_cmd_scale_min,
        #        self.linear_cmd_scale_max,
        #    )

        # ------------------------------------------------------------
        # Angular adaptation
        # ------------------------------------------------------------
        if abs(wz_cmd) >= self.adapt_min_cmd_wz:
            gain_ang_inst = abs(wz_meas) / max(abs(wz_cmd), eps)

            self.angular_cmd_scale = (
                (1.0 - self.adapt_lr_ang) * self.angular_cmd_scale
                + self.adapt_lr_ang * gain_ang_inst
            )

            self.angular_cmd_scale = self._clamp(
                self.angular_cmd_scale,
                self.angular_cmd_scale_min,
                self.angular_cmd_scale_max,
            )

    def _integrate_cmd(
        self,
        vx_cmd: float,
        wz_cmd: float,
        vx_meas: float,
        wz_meas: float,
        coast_motion: bool = False,
    ) -> None:
        """
        Integrate synthetic odom from navigation command, with contextual yaw fusion.

        Main idea:
        ----------
        - Translation remains mostly command-driven
        - Yaw is corrected using a contextual blend between:
              commanded yaw model
              measured yaw-rate from dog_odom

        Why:
        ----
        In your robot, the main mismatch appears to be angular:
            the real robot may rotate more than the command suggests,
            especially near the goal or during residual motion.

        So we keep the synthetic odom philosophy, but make yaw less blind
        to real execution.
        """
        now = self._now()
        dt = (now - self._last_int_time).nanoseconds * 1e-9
        self._last_int_time = now

        if dt <= 0.0 or dt > 0.20:
            return

        # Remove tiny commands
        if abs(vx_cmd) < self.cmd_eps_vx:
            vx_cmd = 0.0
        if abs(wz_cmd) < self.cmd_eps_wz:
            wz_cmd = 0.0

        # Base command-driven model
        vx_cmd_model = self.linear_cmd_scale * vx_cmd
        wz_cmd_model = self.angular_cmd_scale * wz_cmd


        # Translation: keep mostly synthetic / command-based
        vx_model = vx_cmd_model

        # Yaw: contextual fusion
        if self.enable_contextual_yaw_fusion:
            alpha_yaw = self._compute_contextual_alpha_yaw(
                vx_cmd=vx_cmd,
                wz_cmd_model=wz_cmd_model,
                wz_meas=wz_meas,
                coast_motion=coast_motion,
            )
            wz_model = alpha_yaw * wz_cmd_model + (1.0 - alpha_yaw) * wz_meas
        else:
            alpha_yaw = 1.0
            wz_model = wz_cmd_model

        # Add artificial noise
        vx_noisy = vx_model * (
            1.0 + self._rng.uniform(-self.linear_noise_ratio, self.linear_noise_ratio)
        )
        wz_noisy = wz_model * (
            1.0 + self._rng.uniform(-self.angular_noise_ratio, self.angular_noise_ratio)
        )


        # Integrate
        self._yaw = _clamp_pi(self._yaw + wz_noisy * dt)
        self._x += vx_noisy * math.cos(self._yaw) * dt
        self._y += vx_noisy * math.sin(self._yaw) * dt

        # Save for debugging
        self._last_alpha_yaw = alpha_yaw
        self._last_wz_model = wz_model
        self._last_vx_model = vx_model

    def _coast_motion_allowed(self, raw_gate: bool, cmd_age: float, vx_meas: float, wz_meas: float) -> bool:
        """
        Decide whether synthetic odom is allowed to continue integrating
        during the short post-command coast phase.

        Why this exists:
        ----------------
        Near the goal, Nav2 may reduce commands to zero before the robot has
        fully stopped physically. A short coast phase helps the odom keep
        following the residual real motion.

        But this coast must remain strict:
            - only for a short time
            - only if measured motion is still significant

        Otherwise small residual oscillations would keep moving /odom forever.
        """
        if raw_gate:
            return False

        if cmd_age > self.gate_release_hold_sec:
            return False

        translational_coast = abs(vx_meas) > self.coast_meas_vx_min
        rotational_coast = abs(wz_meas) > self.coast_meas_wz_min

        return translational_coast or rotational_coast
    
    # =========================================================================
    # MAIN ODOM LOOP
    # =========================================================================
    def _tick_odom(self) -> None:
        """
        Main synthetic odom update loop.

        Global idea:
        ------------
        This node publishes a CONTROL / NAVIGATION odom, not a true physical
        state estimator.

        The synthetic odom is built from:
            - cmd_vel as motion intention
            - dog_odom twist as evidence that the robot is really moving

        Why this loop exists:
        ---------------------
        On your humanoid RL robot:
            - visual odometry may be unstable or delayed
            - raw dog_odom pose may drift
            - Nav2 still needs a smooth local odom to control motion

        So we:
            1) inspect the navigation command
            2) inspect dog_odom twist
            3) decide if the robot should be considered MOVING or IDLE
            4) integrate cmd_vel only when appropriate
            5) publish /odom and TF(odom -> base_footprint)

        Important refinement added here:
        --------------------------------
        We now separate:
            - translation evidence
            - rotation evidence

        This is critical because:
            - during pure turning, small residual linear twist should NOT create
              fake forward motion
            - during forward motion, translational evidence is the main clue

        This version is therefore stricter and more robust than a naive
        "one evidence for everything" implementation.
        """

        # ------------------------------------------------------------
        # 1) READ RAW COMMAND GATE
        # ------------------------------------------------------------
        #
        # raw_gate:
        #   True only if a recent command exists and its magnitude is above
        #   configured command epsilons.
        #
        # gate_reason:
        #   useful debug info ("cmd_active", "cmd_below_eps", "cmd_stale", ...)
        #
        raw_gate, gate_reason, cmd_age, vx_cmd, wz_cmd = self._raw_gate()

        # ------------------------------------------------------------
        # 2) BUILD EFFECTIVE GATE
        # ------------------------------------------------------------
        #
        # Why this matters:
        # -----------------
        # Near the goal, Nav2 may reduce command magnitude very quickly.
        # But the real RL robot may still be physically moving for a short time.
        #
        # Therefore:
        #   effective gate = raw gate + short grace period
        #
        gate_effective = self._effective_gate(raw_gate, cmd_age)

        # ------------------------------------------------------------
        # 3) READ FILTERED dog_odom TWIST
        # ------------------------------------------------------------
        #
        # Important:
        # ----------
        # We do NOT trust dog_odom pose here.
        # We only use dog_odom twist as "physical movement evidence".
        #
        vx_meas, wz_meas = self._filtered_dog_twist()

        # ------------------------------------------------------------
        # 3-bis) ONLINE ADAPTIVE CALIBRATION
        # ------------------------------------------------------------
        #
        # Learn slowly how much real motion the robot actually executes
        # for the commanded velocity.
        #
        self._update_adaptive_scales(
            vx_cmd=vx_cmd,
            wz_cmd=wz_cmd,
            vx_meas=vx_meas,
            wz_meas=wz_meas,
            gate=raw_gate,
        )

        # ------------------------------------------------------------
        # 4) DETECT PURE TURN COMMAND
        # ------------------------------------------------------------
        #
        # If angular command is large while forward command is very small,
        # we treat this as a turn-in-place request.
        #
        # Why this is necessary:
        # ----------------------
        # Without this, small noisy residual vx from dog_odom could cause
        # fake translational integration during rotations.
        #
        turn_in_place_cmd = (
            abs(wz_cmd) > self.turn_wz_thresh
            and abs(vx_cmd) < self.turn_vx_small
        )

        # ------------------------------------------------------------
        # 5) SEPARATE MOTION EVIDENCE CHANNELS
        # ------------------------------------------------------------
        #
        # translation_evidence:
        #   robot appears to be moving linearly
        #
        # rotation_evidence:
        #   robot appears to be rotating
        #
        # Why separate them:
        # ------------------
        # A single boolean evidence signal is too crude for your robot.
        #
        # Example:
        #   - command asks for pure turn
        #   - dog_odom shows tiny residual vx
        #   - old logic might keep integrating unwanted translation
        #
        translation_evidence = abs(vx_meas) > self.meas_eps_vx
        rotation_evidence = abs(wz_meas) > self.meas_eps_wz

        # ------------------------------------------------------------
        # 6) BUILD FINAL EVIDENCE FOR THIS COMMAND TYPE
        # ------------------------------------------------------------
        #
        # If motion evidence is disabled entirely, always allow integration.
        #
        # Otherwise:
        #
        #   - for pure turn commands:
        #         require angular evidence only
        #
        #   - for normal motion:
        #         accept either translational or angular evidence
        #
        # Why allow angular evidence in normal motion:
        # --------------------------------------------
        # During curved motion, or when the robot reorients while advancing,
        # angular activity can still be a valid sign that locomotion is ongoing.
        #
        if not self.use_motion_evidence:
            evidence = True
        else:
            if turn_in_place_cmd:
                evidence = rotation_evidence
            else:
                evidence = translation_evidence or rotation_evidence

        # ------------------------------------------------------------
        # 6-bis) STRICT COAST MOTION
        # ------------------------------------------------------------
        #
        # A short post-command continuation is allowed only if the robot still
        # shows sufficiently strong measured motion.
        #
        coast_motion = self._coast_motion_allowed(
            raw_gate=raw_gate,
            cmd_age=cmd_age,
            vx_meas=vx_meas,
            wz_meas=wz_meas,
        )

        # ------------------------------------------------------------
        # 7) UPDATE ROBUST IDLE / MOVING STATE
        # ------------------------------------------------------------
        #
        # moving requires:
        #   effective_gate AND evidence
        #
        # This means:
        #   - command intention must still be active enough
        #   - robot must also appear to move physically
        #
        effective_motion_permission = raw_gate or coast_motion
        motion_state = self._update_motion_state(effective_motion_permission, evidence)
        #motion_state = self._update_motion_state(gate_effective, evidence)
        moving = (motion_state == "moving")

        # ------------------------------------------------------------
        # 8) INTEGRATE ONLY THE RELEVANT COMMAND COMPONENTS
        # ------------------------------------------------------------
        #
        # Key logic:
        #
        #   if MOVING:
        #       integrate cmd_vel
        #
        #   but:
        #       - for pure turning: integrate yaw only
        #       - for normal motion: integrate both vx and wz
        #
        vx_cmd_int = 0.0
        wz_cmd_int = 0.0
        if moving:
            if raw_gate:
                # Normal case: a valid current navigation command exists.
                if turn_in_place_cmd:
                    vx_cmd_int = 0.0
                    wz_cmd_int = wz_cmd
                else:
                    vx_cmd_int = vx_cmd
                    wz_cmd_int = wz_cmd

            elif coast_motion:
                # Coast case: command has already weakened or gone stale,
                # but the robot still appears to move physically.
                #
                # In this phase we should NOT reuse a large old command blindly.
                # Instead we integrate a reduced motion based on current measured twist.
                if turn_in_place_cmd:
                    vx_cmd_int = 0.0
                    wz_cmd_int = wz_meas
                else:
                    vx_cmd_int = vx_meas
                    wz_cmd_int = wz_meas
            else:
                vx_cmd_int = 0.0
                wz_cmd_int = 0.0

            #self._integrate_cmd(vx_cmd_int, wz_cmd_int, vx_meas, wz_meas,)

            self._integrate_cmd(
                vx_cmd=vx_cmd_int,
                wz_cmd=wz_cmd_int,
                vx_meas=vx_meas,
                wz_meas=wz_meas,
                coast_motion=coast_motion,
            )

        # ------------------------------------------------------------
        # 9) BUILD ODOM MESSAGE
        # ------------------------------------------------------------
        #
        # The synthetic state (_x, _y, _yaw) is the odom pose seen by Nav2.
        #
        now = self._now()
        odom = Odometry()
        odom.header.stamp = now.to_msg()
        odom.header.frame_id = self.odom_frame
        odom.child_frame_id = self.base_frame

        qx, qy, qz, qw = _quat_from_yaw(self._yaw)

        odom.pose.pose.position.x = float(self._x)
        odom.pose.pose.position.y = float(self._y)
        odom.pose.pose.position.z = 0.0
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw

        # ------------------------------------------------------------
        # 10) BUILD ODOM TWIST
        # ------------------------------------------------------------
        #
        # We publish the commanded twist shape while moving.
        # When idle, twist is zeroed.
        #
        # This makes the odom message more coherent for navigation consumers.
        #
        #odom.twist.twist.linear.x = float(vx_cmd_int if moving else 0.0)
        #odom.twist.twist.linear.y = 0.0
        #odom.twist.twist.linear.z = 0.0
        #odom.twist.twist.angular.x = 0.0
        #odom.twist.twist.angular.y = 0.0
        #odom.twist.twist.angular.z = float(wz_cmd_int if moving else 0.0)

        odom.twist.twist.linear.x = float(self._last_vx_model if moving else 0.0)
        odom.twist.twist.linear.y = 0.0
        odom.twist.twist.linear.z = 0.0
        odom.twist.twist.angular.x = 0.0
        odom.twist.twist.angular.y = 0.0
        odom.twist.twist.angular.z = float(self._last_wz_model if moving else 0.0)
        # ------------------------------------------------------------
        # 11) PLACEHOLDER COVARIANCES
        # ------------------------------------------------------------
        #
        # These can be tuned later if needed.
        #
        odom.pose.covariance = [0.0] * 36
        odom.pose.covariance[0] = 0.10
        odom.pose.covariance[7] = 0.10
        odom.pose.covariance[35] = 0.30

        odom.twist.covariance = [0.0] * 36
        odom.twist.covariance[0] = 0.20
        odom.twist.covariance[7] = 0.20
        odom.twist.covariance[35] = 0.40

        # ------------------------------------------------------------
        # 12) PUBLISH /ODOM
        # ------------------------------------------------------------
        self.odom_pub.publish(odom)
        self._last_odom_msg = odom

        # ------------------------------------------------------------
        # 13) PUBLISH TF odom -> base_footprint
        # ------------------------------------------------------------
        #
        # This is the local navigation TF consumed by Nav2.
        #
        if self.publish_tf and self.tf_broadcaster is not None:
            tfm = TransformStamped()
            tfm.header.stamp = odom.header.stamp
            tfm.header.frame_id = self.odom_frame
            tfm.child_frame_id = self.base_frame

            tfm.transform.translation.x = float(self._x)
            tfm.transform.translation.y = float(self._y)
            tfm.transform.translation.z = 0.0
            tfm.transform.rotation.x = qx
            tfm.transform.rotation.y = qy
            tfm.transform.rotation.z = qz
            tfm.transform.rotation.w = qw

            self.tf_broadcaster.sendTransform(tfm)

        # ------------------------------------------------------------
        # 14) SAVE DEBUG STATE
        # ------------------------------------------------------------
        #
        # This is very important for field debugging.
        # It allows you to inspect:
        #   - whether command was active
        #   - whether grace period was active
        #   - whether translation or rotation evidence dominated
        #   - whether the node considered the robot moving
        #
        self._last_debug = {
            "raw_gate": int(raw_gate),
            "gate_effective": int(gate_effective),
            "gate_reason": gate_reason,
            "cmd_age": cmd_age,
            "vx_cmd": vx_cmd,
            "wz_cmd": wz_cmd,
            "vx_meas": vx_meas,
            "wz_meas": wz_meas,
            "coast_motion": int(coast_motion),
            "linear_cmd_scale": self.linear_cmd_scale,
            "angular_cmd_scale": self.angular_cmd_scale,
            "vx_cmd_int": vx_cmd_int,
            "wz_cmd_int": wz_cmd_int,
            "translation_evidence": int(translation_evidence),
            "rotation_evidence": int(rotation_evidence),
            "evidence": int(evidence),
            "motion_state": motion_state,
            "moving": int(moving),
            "turn": int(turn_in_place_cmd),
            "x": self._x,
            "y": self._y,
            "yaw": self._yaw,
            "alpha_yaw": self._last_alpha_yaw,
            "wz_model": self._last_wz_model,
        }

    # =========================================================================
    # STATUS OUTPUT
    # =========================================================================
    def _tick_status(self) -> None:
        if self.status_pub is None or not self._last_debug:
            return

        d = self._last_debug
        msg = String()
        msg.data = (
            "{"
                f"\"raw_gate\":{d['raw_gate']},"
                f"\"gate_effective\":{d['gate_effective']},"
                f"\"gate_reason\":\"{d['gate_reason']}\","
                f"\"cmd_age\":{d['cmd_age']:.3f},"
                f"\"vx_cmd\":{d['vx_cmd']:.3f},"
                f"\"wz_cmd\":{d['wz_cmd']:.3f},"
                f"\"vx_meas\":{d['vx_meas']:.3f},"
                f"\"wz_meas\":{d['wz_meas']:.3f},"
                f"\"coast_motion\":{d['coast_motion']},"
                f"\"linear_cmd_scale\":{d['linear_cmd_scale']:.3f},"
                f"\"angular_cmd_scale\":{d['angular_cmd_scale']:.3f},"
                f"\"vx_cmd_int\":{d['vx_cmd_int']:.3f},"
                f"\"wz_cmd_int\":{d['wz_cmd_int']:.3f},"
                f"\"translation_evidence\":{d['translation_evidence']},"
                f"\"rotation_evidence\":{d['rotation_evidence']},"
                f"\"evidence\":{d['evidence']},"
                f"\"motion_state\":\"{d['motion_state']}\","
                f"\"moving\":{d['moving']},"
                f"\"turn\":{d['turn']},"
                f"\"x\":{d['x']:.3f},"
                f"\"y\":{d['y']:.3f},"
                f"\"yaw\":{d['yaw']:.3f},"
                f"\"alpha_yaw\":{d['alpha_yaw']:.3f},"
                f"\"wz_model\":{d['wz_model']:.3f}"
            "}"
        )
        self.status_pub.publish(msg)


def main() -> None:
    rclpy.init()
    node = CmdVelGuardedArtificialOdom()
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