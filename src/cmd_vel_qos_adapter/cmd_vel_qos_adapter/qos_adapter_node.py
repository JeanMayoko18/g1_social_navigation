#!/usr/bin/env python3
"""
cmd_vel_qos_adapter.py

--------------------------------------------------------------------------------
PURPOSE
--------------------------------------------------------------------------------
This node is a small "QoS bridge" for /cmd_vel-like topics.

It is designed for setups where the navigation stack (e.g., Nav2 + velocity_smoother)
and the locomotion controller (e.g., rl_sar) do not agree on DDS reliability:

- Nav2 often publishes velocity commands using RELIABLE QoS.
- rl_sar may subscribe using "SystemDefaultsQoS()", which can resolve to BEST_EFFORT
  depending on the RMW implementation, network conditions, or platform defaults.

A RELIABLE publisher and a BEST_EFFORT subscriber *may* fail to communicate or behave
inconsistently across machines. This adapter makes the system robust by accepting
both QoS types on the input and republishing in a QoS that rl_sar reliably receives.

--------------------------------------------------------------------------------
WHAT IT DOES
--------------------------------------------------------------------------------
Input:
  - Subscribes to the same input topic TWICE:
      1) with RELIABLE QoS   (preferred source)
      2) with BEST_EFFORT QoS (fallback source)

Output:
  - Republishes the received Twist messages to an output topic using BEST_EFFORT QoS.

Selection policy:
  - If a RELIABLE message was seen "recently", BEST_EFFORT messages are ignored
    for a short time window (to prevent duplicates and prefer the stronger source).
  - Otherwise, BEST_EFFORT messages are accepted and republished.
  - Duplicate commands are suppressed within a small time window to avoid
    double-publishing when both subscriptions receive the same command.

Typical pipeline:
  Nav2 controller output (/cmd_vel_nav or /cmd_vel)  --->  QoS adapter  --->  /cmd_vel_rlsar  --->  rl_sar

--------------------------------------------------------------------------------
WHEN TO USE
--------------------------------------------------------------------------------
Use this node when:
  - You see Nav2 generating commands (controller_server active), but the robot
    does not move and rl_sar shows no cmd_vel reception.
  - 'ros2 topic info' indicates QoS mismatch between publisher and subscriber.
  - You run Nav2 on a different PC or network where QoS defaults may differ.

--------------------------------------------------------------------------------
PARAMETERS
--------------------------------------------------------------------------------
- in_topic (string):
    Input command topic to listen to (default: "/cmd_vel")
- out_topic (string):
    Output topic to republish towards rl_sar (default: "/cmd_vel_rlsar")
- prefer_reliable_window_sec (float):
    For how many seconds after receiving a RELIABLE message we prefer RELIABLE
    and ignore BEST_EFFORT (default: 0.25)
- dedup_window_sec (float):
    Suppress publishing the exact same Twist signature within this time window
    (default: 0.05)

Notes:
- This node does not modify the command values, it only repackages them.
- BEST_EFFORT is chosen for output because it is commonly accepted by locomotion
  controllers and avoids strict reliability constraints on WiFi / lossy links.
"""

import rclpy
import time
import math
from typing import Optional, Tuple

from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from geometry_msgs.msg import Twist

def angle_wrap(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def yaw_from_quaternion(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def _twist_signature(msg: Twist) -> Tuple[float, float, float, float, float, float]:
    """
    Create a small immutable "signature" for a Twist message.

    Why:
      - When we subscribe twice (RELIABLE + BEST_EFFORT), the same command can arrive
        through both subscriptions.
      - We want to avoid double-publishing identical commands back-to-back.

    Here we use exact float equality. For cmd_vel commands this is generally fine
    because many stacks publish fixed step values (e.g., 0.1, 0.0, 0.5).
    """
    return (
        float(msg.linear.x), float(msg.linear.y), float(msg.linear.z),
        float(msg.angular.x), float(msg.angular.y), float(msg.angular.z),
    )


class CmdVelQoSAdapter(Node):
    """
    CmdVelQoSAdapter

    High-level behavior:
      1) Subscribe to `in_topic` with RELIABLE QoS (preferred).
      2) Subscribe to `in_topic` with BEST_EFFORT QoS (fallback).
      3) Republish to `out_topic` using BEST_EFFORT QoS.

    Preference logic:
      - RELIABLE messages always pass through immediately.
      - BEST_EFFORT messages are accepted only if RELIABLE hasn't been seen recently.
      - Duplicate suppression prevents republishing the same command twice.

    This makes the adapter "future-proof":
      - If the publisher is RELIABLE, the RELIABLE subscription will match.
      - If the publisher is BEST_EFFORT, the BEST_EFFORT subscription will match.
      - If the publisher changes (different PC / different DDS), the adapter still works.
    """

    def __init__(self):
        super().__init__("cmd_vel_qos_adapter")

        # -----------------------
        # Parameters
        # -----------------------
        # Topics are parameters so you can reuse the same node for multiple pipelines.
        # Example:
        #   in_topic := /cmd_vel_nav
        #   out_topic := /cmd_vel_rlsar
        self.declare_parameter("in_topic", "/cmd_vel_mux")
        self.declare_parameter("out_topic", "/cmd_vel_rlsar")
        self.declare_parameter(
            "yaw_feedback_odom_topic",
            "/dog_odom_filtered"
        )
        

        # How long a RELIABLE message keeps "priority" over BEST_EFFORT.
        # If RELIABLE has been seen within this time window, ignore BEST_EFFORT.
        self.declare_parameter("prefer_reliable_window_sec", 0.25)

        # Drop repeated publishing of the exact same Twist signature within this window.
        # Helps avoid duplicates when both subs receive the same command.
        self.declare_parameter("dedup_window_sec", 0.05)

        self.declare_parameter("enable_straight_bias_correction", True)
        self.declare_parameter("straight_cmd_vx_thresh", 0.08)
        self.declare_parameter("straight_cmd_wz_thresh", 0.05)
        self.declare_parameter("straight_wz_bias", -0.02)

        self.declare_parameter("use_yaw_feedback", True)
        self.declare_parameter("yaw_feedback_kp", 0.20)
        self.declare_parameter("yaw_feedback_max_correction", 0.04)


        self.declare_parameter("cmd_vx_scale", 1.0)
        self.declare_parameter("cmd_vy_scale", 1.0)
        self.declare_parameter("cmd_wz_scale", 1.0)

        self.declare_parameter("cmd_vx_bias", 0.0)
        self.declare_parameter("cmd_vy_bias", 0.0)
        self.declare_parameter("cmd_wz_bias", 0.0)
        
        

        # -----------------------
        # Parameters reader
        # -----------------------
        self.in_topic = self.get_parameter("in_topic").get_parameter_value().string_value
        self.out_topic = self.get_parameter("out_topic").get_parameter_value().string_value

        self.prefer_reliable_window = float(
            self.get_parameter("prefer_reliable_window_sec").get_parameter_value().double_value
        )
        self.dedup_window = float(
            self.get_parameter("dedup_window_sec").get_parameter_value().double_value
        )

        self.enable_straight_bias_correction = bool(
            self.get_parameter("enable_straight_bias_correction").value
        )

        self.straight_cmd_vx_thresh = float(
            self.get_parameter("straight_cmd_vx_thresh").value
        )

        self.straight_cmd_wz_thresh = float(
            self.get_parameter("straight_cmd_wz_thresh").value
        )

        self.straight_wz_bias = float(
            self.get_parameter("straight_wz_bias").value
        )
        self.yaw_feedback_odom_topic = (
            self.get_parameter("yaw_feedback_odom_topic").value
        )

        self.use_yaw_feedback = bool(
            self.get_parameter("use_yaw_feedback").value
        )

        self.yaw_feedback_kp = float(
            self.get_parameter("yaw_feedback_kp").value
        )

        self.yaw_feedback_max_correction = float(
            self.get_parameter("yaw_feedback_max_correction").value
        )

        self.cmd_vx_scale = float(self.get_parameter("cmd_vx_scale").value)
        self.cmd_vy_scale = float(self.get_parameter("cmd_vy_scale").value)
        self.cmd_wz_scale = float(self.get_parameter("cmd_wz_scale").value)

        self.cmd_vx_bias = float(self.get_parameter("cmd_vx_bias").value)
        self.cmd_vy_bias = float(self.get_parameter("cmd_vy_bias").value)
        self.cmd_wz_bias = float(self.get_parameter("cmd_wz_bias").value)

        # -----------------------
        # QoS profiles
        # -----------------------
        # For velocity commands, KEEP_LAST(1) is enough because we only care about
        # the latest command. We do not want queues building up.
        qos_reliable = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        qos_pub_reliable = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        qos_best_effort = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        qos_odom_feedback = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        # Publisher towards rl_sar: BEST_EFFORT
        # Rationale:
        #  - Many locomotion/low-level controllers accept BEST_EFFORT by default.
        #  - BEST_EFFORT tends to work better on lossy networks (WiFi).
        #self.pub = self.create_publisher(Twist, self.out_topic, qos_best_effort)
        self.pub = self.create_publisher(Twist, self.out_topic, qos_pub_reliable)

        self.sub_odom_feedback = self.create_subscription(
            Odometry,
            self.yaw_feedback_odom_topic,
            self._cb_odom_feedback,
            qos_odom_feedback
        )



        # -----------------------
        # Internal state (timestamps + dedup)
        # -----------------------
        # Time (monotonic) when we last received a RELIABLE message.
        self._last_reliable_time: float = -1.0

        # Time (monotonic) when we last published ANY message.
        self._last_published_time: float = -1.0

        # Signature of the last published Twist. Used to suppress duplicates.
        self._last_published_sig: Optional[Tuple[float, float, float, float, float, float]] = None

        # -----------------------
        # Yaw feedback state
        # -----------------------
        self.current_yaw = None
        self.straight_yaw_ref = None
        self.straight_hold_active = False

        # -----------------------
        # Subscriptions (same topic, different QoS)
        # -----------------------
        # RELIABLE subscription: preferred if available.
        self.sub_rel = self.create_subscription(Twist, self.in_topic, self._cb_rel, qos_reliable)

        # BEST_EFFORT subscription: fallback if RELIABLE is not available.
        self.sub_be = self.create_subscription(Twist, self.in_topic, self._cb_be, qos_best_effort)
        

        self.get_logger().info(
            f"QoS Adapter started | in={self.in_topic} (REL+BE) -> out={self.out_topic} (BE) "
            f"| prefer_reliable_window={self.prefer_reliable_window:.3f}s | dedup_window={self.dedup_window:.3f}s"
        )

    def _should_drop_duplicate(self, sig: Tuple[float, float, float, float, float, float], now: float) -> bool:
        """
        Decide whether to drop a publish attempt because it is a duplicate.

        We drop only if:
          - the signature is identical to the last published signature AND
          - the last publish happened within dedup_window_sec.

        This prevents "double publish" when:
          - the same command arrives from both subscriptions,
          - or the publisher repeats the same command at high frequency.
        """
        if self._last_published_sig is None:
            return False
        if sig != self._last_published_sig:
            return False
        if (now - self._last_published_time) <= self.dedup_window:
            return True
        return False

    def _cb_odom_feedback(self, msg: Odometry):
        """
        Read current yaw from filtered odometry.

        This yaw is used only for command-level feedback correction
        during commanded straight walking.
        """
        self.current_yaw = yaw_from_quaternion(msg.pose.pose.orientation)


    def _apply_straight_bias_correction(self, msg: Twist) -> Twist:
        """
        Apply command-level correction before sending cmd_vel to rl_sar.

        This function is now used as a "G1 command adapter", inspired by the DimOS
        G1 MuJoCo controller logic.

        DimOS applies command adaptation before feeding the command to the policy:
            command[0] = command[0] * 2
            command[1] = command[1] * 2
            command[2] += drift_compensation[2]

        Here we keep the same idea, but make it fully configurable:
            vx_out = vx_in * cmd_vx_scale + cmd_vx_bias
            vy_out = vy_in * cmd_vy_scale + cmd_vy_bias
            wz_out = wz_in * cmd_wz_scale + cmd_wz_bias

        Safety design:
        - If enable_straight_bias_correction is False:
            return the original Twist unchanged.
        - If True:
            apply global scale/bias corrections.
        - The additional straight_wz_bias is applied only during commanded
            straight walking, so intentional Nav2 rotations are not modified.
        - Optional yaw feedback is also applied only during straight walking.
        """

        # ------------------------------------------------------------
        # 0) Always start by copying the original command.
        #    This guarantees that when correction is disabled, the output
        #    is exactly the same as the input.
        # ------------------------------------------------------------
        corrected = Twist()
        corrected.linear.x = msg.linear.x
        corrected.linear.y = msg.linear.y
        corrected.linear.z = msg.linear.z
        corrected.angular.x = msg.angular.x
        corrected.angular.y = msg.angular.y
        corrected.angular.z = msg.angular.z

        # ------------------------------------------------------------
        # 1) Master switch.
        #    False = pure QoS adapter behavior, no command modification.
        # ------------------------------------------------------------
        if not self.enable_straight_bias_correction:
            self.straight_hold_active = False
            self.straight_yaw_ref = None
            return corrected

        # ------------------------------------------------------------
        # 2) Global command adaptation inspired by DimOS.
        #
        #    This changes the command distribution seen by the RL policy.
        #    It is useful when the policy expects a scaled or biased command
        #    space compared with the raw Nav2 /cmd_vel.
        #
        #    Recommended safe initial values:
        #      cmd_vx_scale = 1.0
        #      cmd_vy_scale = 1.0
        #      cmd_wz_scale = 1.0
        #      cmd_vx_bias  = 0.0
        #      cmd_vy_bias  = 0.0
        #      cmd_wz_bias  = 0.0
        #
        #    DimOS simulation uses stronger values, but they should not be
        #    copied directly to the real robot without progressive testing.
        # ------------------------------------------------------------
        corrected.linear.x = (
            float(msg.linear.x) * self.cmd_vx_scale + self.cmd_vx_bias
        )
        corrected.linear.y = (
            float(msg.linear.y) * self.cmd_vy_scale + self.cmd_vy_bias
        )
        corrected.angular.z = (
            float(msg.angular.z) * self.cmd_wz_scale + self.cmd_wz_bias
        )

        # Use the original angular command to decide whether Nav2 intended
        # a real rotation. This avoids treating an internally added cmd_wz_bias
        # as a user/navigation rotation command.
        raw_cmd_vx = float(msg.linear.x)
        raw_cmd_wz = float(msg.angular.z)

        walking_straight = (
            abs(raw_cmd_vx) > self.straight_cmd_vx_thresh and
            abs(raw_cmd_wz) < self.straight_cmd_wz_thresh
        )

        # ------------------------------------------------------------
        # 3) If the command is not straight walking, keep only the global
        #    DimOS-like scale/bias correction and reset yaw-hold state.
        #
        #    This means Nav2 rotations remain mostly controlled by Nav2,
        #    except for any global cmd_wz_scale/cmd_wz_bias you explicitly set.
        # ------------------------------------------------------------
        if not walking_straight:
            self.straight_hold_active = False
            self.straight_yaw_ref = None
            return corrected

        # ------------------------------------------------------------
        # 4) Straight-walking feed-forward yaw correction.
        #
        #    This is your empirical correction for the G1 drift during
        #    commanded straight walking.
        #
        #    Example:
        #      straight_wz_bias = -0.04
        #
        #    It is applied only when:
        #      - vx is active
        #      - raw wz is close to zero
        # ------------------------------------------------------------
        wz_out = float(corrected.angular.z) + self.straight_wz_bias

        # ------------------------------------------------------------
        # 5) Optional yaw feedback correction.
        #
        #    This tries to hold the heading measured from filtered odometry.
        #    It is intentionally enabled only during straight walking.
        # ------------------------------------------------------------
        if self.use_yaw_feedback and self.current_yaw is not None:
            if not self.straight_hold_active:
                self.straight_yaw_ref = self.current_yaw
                self.straight_hold_active = True
                self.get_logger().info(
                    f"[YAW_HOLD] reference initialized: "
                    f"{math.degrees(self.straight_yaw_ref):.2f} deg"
                )

            yaw_error = angle_wrap(self.current_yaw - self.straight_yaw_ref)

            wz_fb = -self.yaw_feedback_kp * yaw_error
            wz_fb = max(
                -self.yaw_feedback_max_correction,
                min(self.yaw_feedback_max_correction, wz_fb),
            )

            wz_out += wz_fb

            self.get_logger().info(
                f"[YAW_FB] "
                f"yaw_error={math.degrees(yaw_error):.2f} deg "
                f"cmd_wz_bias={self.cmd_wz_bias:.3f} "
                f"straight_wz_bias={self.straight_wz_bias:.3f} "
                f"wz_fb={wz_fb:.3f} "
                f"wz_out={wz_out:.3f}",
                throttle_duration_sec=0.5,
            )

        corrected.angular.z = wz_out

        self.get_logger().info(
            f"[CMD_ADAPT] "
            f"vx_in={msg.linear.x:.3f} vx_out={corrected.linear.x:.3f} "
            f"vy_in={msg.linear.y:.3f} vy_out={corrected.linear.y:.3f} "
            f"wz_in={msg.angular.z:.3f} wz_out={corrected.angular.z:.3f} "
            f"straight={walking_straight}",
            throttle_duration_sec=0.5,
        )

        return corrected

    def _publish(self, msg: Twist, source: str):
        """
        Publish a Twist to the output topic with BEST_EFFORT, with dedup filtering.

        `source` is only used for debug logs (RELIABLE or BEST_EFFORT).
        """
        now = time.monotonic()
        sig = _twist_signature(msg)
        
        out_msg = self._apply_straight_bias_correction(msg)
        out_sig = _twist_signature(out_msg)

        # Drop very recent duplicates to avoid republishing the exact same command twice.
        if self._should_drop_duplicate(out_sig, now):
            return


        self.pub.publish(out_msg)
        self._last_published_time = now
        #self._last_published_sig = sig
        self._last_published_sig = out_sig

        

        # Debug-level logs reduce console spam when publishing at 20-50Hz.
        self.get_logger().debug(f"Published from {source}: {sig}")
        # Keep logs lightweight; INFO is ok for debug sessions

        self.get_logger().info(
          f"IN={source:<11} OUT=BE "
          f"lin.x={msg.linear.x:.3f} "
          f"ang.z_in={msg.angular.z:.3f} "
          f"ang.z_out={out_msg.angular.z:.3f}",
          throttle_duration_sec=0.5,
        )

    def _cb_rel(self, msg: Twist):
        """
        Callback for the RELIABLE subscription.

        Behavior:
          - Mark that RELIABLE has been seen "now".
          - Publish immediately because RELIABLE is always the preferred source.
        """
        self._last_reliable_time = time.monotonic()
        self._publish(msg, source="RELIABLE")
        self.get_logger().info(f"IN  REL lin.x={msg.linear.x:.3f} ang.z={msg.angular.z:.3f}")

    def _cb_be(self, msg: Twist):
        """
        Callback for the BEST_EFFORT subscription.

        Behavior:
          - If RELIABLE was seen recently (within prefer_reliable_window_sec),
            ignore this message to avoid duplicates / weaker source.
          - Otherwise, publish it (fallback mode).
        """
        now = time.monotonic()

        # If RELIABLE messages are active recently, ignore BEST_EFFORT messages.
        if self._last_reliable_time > 0.0 and (now - self._last_reliable_time) <= self.prefer_reliable_window:
            return

        self._publish(msg, source="BEST_EFFORT")


def main():
    """
    Standard rclpy entrypoint.
    """
    rclpy.init()
    node = CmdVelQoSAdapter()
    try:
        rclpy.spin(node)
    finally:
        # Always destroy the node before shutting down rclpy.
        node.destroy_node()
        if rclpy.ok():
          rclpy.shutdown()


if __name__ == "__main__":
    main()