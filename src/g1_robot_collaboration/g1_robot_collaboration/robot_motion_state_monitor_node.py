#!/usr/bin/env python3

"""
Robot motion-state monitor.

Purpose
-------
Provide the CollaborationManager with a conservative, temporally filtered
confirmation that navigation locomotion has stopped.

Inputs
------
/cmd_vel_rlsar
    Command sent toward the G1 locomotion interface.

/odom
    Canonical navigation odometry selected by the navigation bringup.

Output
------
/social_navigation/robot_motion_state
    RobotMotionState

Important
---------
This node does NOT interpret Nav2 cancellation.

A zero command alone is not sufficient.
A single low-speed odometry sample alone is not sufficient.

stable_stop_confirmed becomes true only when:

    command_quiet
        AND
    odometry_quiet

remain continuously true for stable_dwell_sec.

Small motions caused by the RL standing/balance controller are tolerated
through configurable velocity thresholds.
"""

import math
import time

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node

from g1_social_msgs.msg import RobotMotionState


class RobotMotionStateMonitorNode(Node):

    def __init__(self) -> None:
        super().__init__("robot_motion_state_monitor")

        # ================================================================
        # PARAMETERS
        # ================================================================

        self.declare_parameter(
            "cmd_vel_topic",
            "/cmd_vel_rlsar",
        )
        

        self.declare_parameter(
            "odom_topic",
            "/odom",
        )

        self.declare_parameter(
            "output_topic",
            "/social_navigation/robot_motion_state",
        )

        # Initial conservative values.
        #
        # These are deliberately parameters: the next physical recording
        # must be used to calibrate them against the normal RL standing
        # oscillation of the real G1.
        self.declare_parameter(
            "cmd_linear_threshold",
            0.03,
        )

        self.declare_parameter(
            "cmd_angular_threshold",
            0.04,
        )

        self.declare_parameter(
            "odom_linear_threshold",
            0.05,
        )

        self.declare_parameter(
            "odom_angular_threshold",
            0.08,
        )

        self.declare_parameter(
            "stable_dwell_sec",
            0.75,
        )

        # Freshness guards.
        #
        # Odometry must always remain fresh.
        #
        # /cmd_vel_rlsar may legitimately become silent when no locomotion
        # command is being produced. Command silence is accepted only
        # after at least one command has previously been observed.
        self.declare_parameter(
            "cmd_timeout_sec",
            0.50,
        )

        self.declare_parameter(
            "odom_timeout_sec",
            0.50,
        )

        self.declare_parameter(
            "publish_frequency_hz",
            20.0,
        )

        # ================================================================
        # RESOLVE PARAMETERS
        # ================================================================

        self.cmd_vel_topic = str(
            self.get_parameter("cmd_vel_topic").value
        )

        self.odom_topic = str(
            self.get_parameter("odom_topic").value
        )

        self.output_topic = str(
            self.get_parameter("output_topic").value
        )

        self.cmd_linear_threshold = float(
            self.get_parameter(
                "cmd_linear_threshold"
            ).value
        )

        self.cmd_angular_threshold = float(
            self.get_parameter(
                "cmd_angular_threshold"
            ).value
        )
        self.monitor_started_at = time.monotonic()

        self.odom_linear_threshold = float(
            self.get_parameter(
                "odom_linear_threshold"
            ).value
        )

        self.odom_angular_threshold = float(
            self.get_parameter(
                "odom_angular_threshold"
            ).value
        )

        self.stable_dwell_sec = float(
            self.get_parameter(
                "stable_dwell_sec"
            ).value
        )

        self.cmd_timeout_sec = float(
            self.get_parameter(
                "cmd_timeout_sec"
            ).value
        )

        self.odom_timeout_sec = float(
            self.get_parameter(
                "odom_timeout_sec"
            ).value
        )

        publish_frequency_hz = float(
            self.get_parameter(
                "publish_frequency_hz"
            ).value
        )

        if publish_frequency_hz <= 0.0:
            raise ValueError(
                "publish_frequency_hz must be > 0"
            )

        if self.stable_dwell_sec <= 0.0:
            raise ValueError(
                "stable_dwell_sec must be > 0"
            )

        # ================================================================
        # STATE
        # ================================================================

        self.latest_cmd = None
        self.latest_odom = None

        self.latest_cmd_received_at = None
        self.latest_odom_received_at = None

        self.stable_since = None
        self.last_timer_evaluation_at = None

        self.was_stable_stop_confirmed = False

        # ================================================================
        # ROS INTERFACES
        # ================================================================

        self.publisher = self.create_publisher(
            RobotMotionState,
            self.output_topic,
            10,
        )

        self.cmd_sub = self.create_subscription(
            Twist,
            self.cmd_vel_topic,
            self._cmd_callback,
            10,
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            self.odom_topic,
            self._odom_callback,
            10,
        )

        self.timer = self.create_timer(
            1.0 / publish_frequency_hz,
            self._timer_callback,
        )

        self.get_logger().info(
            "RobotMotionStateMonitor ready | "
            f"cmd={self.cmd_vel_topic} | "
            f"odom={self.odom_topic} | "
            f"output={self.output_topic} | "
            f"dwell={self.stable_dwell_sec:.2f}s"
        )

    # ====================================================================
    # INPUT CALLBACKS
    # ====================================================================
    
    def _odom_stamp_is_fresh(self, msg):
        """Check the odometry source timestamp, not only its arrival time."""
        stamp = msg.header.stamp

        stamp_sec = (
            float(stamp.sec)
            + float(stamp.nanosec) * 1e-9
        )

        age = (
            self.get_clock().now().nanoseconds / 1e9
            - stamp_sec
        )

        return (
            stamp_sec > 0.0
            and math.isfinite(age)
            and -0.10 <= age <= self.odom_timeout_sec
        )
    
    def _cmd_callback(self, msg):
        self.latest_cmd = msg
        self.latest_cmd_received_at = time.monotonic()

        quiet = (
            math.hypot(
                float(msg.linear.x),
                float(msg.linear.y),
            ) <= self.cmd_linear_threshold
            and abs(
                float(msg.angular.z)
            ) <= self.cmd_angular_threshold
        )

        if not quiet:
            self.stable_since = None

    def _odom_callback(self, msg):
        now = time.monotonic()

        # Delayed measurements must not look fresh on arrival.
        if not self._odom_stamp_is_fresh(msg):
            self.latest_odom = None
            self.latest_odom_received_at = None
            self.stable_since = None
            return

        # Repeated or out-of-order measurements cannot establish
        # a continuous interval of fresh physical observations.
        if self.latest_odom is not None:
            old = self.latest_odom.header.stamp
            new = msg.header.stamp

            old_ns = (
                old.sec * 1000000000
                + old.nanosec
            )

            new_ns = (
                new.sec * 1000000000
                + new.nanosec
            )

            if new_ns <= old_ns:
                self.latest_odom = None
                self.latest_odom_received_at = None
                self.stable_since = None
                return

        # A new sample cannot retroactively fill an observation gap.
        if (
            self.latest_odom_received_at is None
            or (
                now - self.latest_odom_received_at
            ) > self.odom_timeout_sec
        ):
            self.stable_since = None

        twist = msg.twist.twist

        quiet = (
            math.hypot(
                float(twist.linear.x),
                float(twist.linear.y),
            ) <= self.odom_linear_threshold
            and abs(
                float(twist.angular.z)
            ) <= self.odom_angular_threshold
        )

        # Catch movement even if a later quiet sample arrives
        # before the next timer evaluation.
        if not quiet:
            self.stable_since = None

        self.latest_odom = msg
        self.latest_odom_received_at = now

    # ====================================================================
    # ESTIMATION
    # ====================================================================

    def _timer_callback(self) -> None:
        """
        Publish a conservative physical-stop estimate.

        Command-side policy:

            - fresh command:
                evaluate the current Twist;

            - stale command after at least one observed command:
                accept command silence only if the LAST known Twist was already
                below the configured quiet thresholds;

            - no command ever observed:
                command state remains unknown and cannot confirm a stop.

        Physical stop additionally requires fresh quiet odometry continuously
        for stable_dwell_sec.
        """

        now_monotonic = time.monotonic()

        # Time without evaluation must not count as confirmed stability.
        if (
            self.last_timer_evaluation_at is not None
            and (
                now_monotonic
                - self.last_timer_evaluation_at
            ) > self.odom_timeout_sec
        ):
            self.stable_since = None

        self.last_timer_evaluation_at = now_monotonic

        # ================================================================
        # INPUT FRESHNESS
        # ================================================================

        cmd_seen = (
            self.latest_cmd is not None
            and self.latest_cmd_received_at is not None
        )

        cmd_fresh = (
            cmd_seen
            and (
                now_monotonic
                - self.latest_cmd_received_at
            )
            <= self.cmd_timeout_sec
        )

        odom_fresh = (
            self.latest_odom is not None
            and self.latest_odom_received_at is not None
            and self._odom_stamp_is_fresh(
                self.latest_odom
            )
            and (
                now_monotonic
                - self.latest_odom_received_at
            ) <= self.odom_timeout_sec
        )

        # ================================================================
        # COMMAND STATE
        # ================================================================

        cmd_linear = 0.0
        cmd_angular = 0.0

        if cmd_seen:

            # Always evaluate the last command that was actually observed.
            #
            # If the stream later becomes silent, that silence may count as
            # command-side inactivity only when this last command was already
            # quiet.
            cmd_linear = math.hypot(
                float(self.latest_cmd.linear.x),
                float(self.latest_cmd.linear.y),
            )

            cmd_angular = abs(
                float(self.latest_cmd.angular.z)
            )

            last_command_quiet = (
                cmd_linear
                <= self.cmd_linear_threshold
                and
                cmd_angular
                <= self.cmd_angular_threshold
            )

            if cmd_fresh:

                command_quiet = last_command_quiet

            else:

                # Event-driven command stream:
                #
                # silence is acceptable only when the last known command
                # was already a stop/quiet command.
                command_quiet = last_command_quiet

        else:

            # No locomotion command has ever been observed.
            #
            # Do not infer a stop from silence alone.
            command_quiet = False

        # ================================================================
        # MEASURED NAVIGATION MOTION
        # ================================================================

        linear_speed = 0.0
        angular_speed = 0.0

        if odom_fresh:

            twist = self.latest_odom.twist.twist

            linear_speed = math.hypot(
                float(twist.linear.x),
                float(twist.linear.y),
            )

            angular_speed = abs(
                float(twist.angular.z)
            )

            odometry_quiet = (
                linear_speed
                <= self.odom_linear_threshold
                and
                angular_speed
                <= self.odom_angular_threshold
            )

        else:

            odometry_quiet = False

        # ================================================================
        # TEMPORAL CONFIRMATION
        # ================================================================

        stop_candidate = (
            cmd_seen
            and command_quiet
            and odom_fresh
            and odometry_quiet
        )

        if stop_candidate:

            if self.stable_since is None:
                self.stable_since = now_monotonic

        else:

            self.stable_since = None

        if self.stable_since is None:

            stable_duration = 0.0

        else:

            stable_duration = max(
                0.0,
                now_monotonic - self.stable_since,
            )

        stable_stop_confirmed = (
            stop_candidate
            and stable_duration
            >= self.stable_dwell_sec
        )

        # ================================================================
        # PUBLISH
        # ================================================================

        msg = RobotMotionState()

        msg.header.stamp = (
            self.get_clock()
            .now()
            .to_msg()
        )

        msg.stable_stop_confirmed = bool(
            stable_stop_confirmed
        )

        msg.linear_speed = float(
            linear_speed
        )

        msg.angular_speed = float(
            angular_speed
        )

        msg.command_quiet = bool(
            command_quiet
        )

        msg.odometry_quiet = bool(
            odometry_quiet
        )

        msg.stable_duration = float(
            stable_duration
        )

        msg.source = (
            "cmd_vel_rlsar+canonical_odom_temporal_monitor"
        )

        self.publisher.publish(msg)

        # ================================================================
        # TRANSITION LOGGING
        # ================================================================

        if (
            stable_stop_confirmed
            != self.was_stable_stop_confirmed
        ):

            if stable_stop_confirmed:

                self.get_logger().info(
                    "[ROBOT MOTION] "
                    "STABLE_STOP_CONFIRMED | "
                    f"linear={linear_speed:.4f} m/s | "
                    f"angular={angular_speed:.4f} rad/s | "
                    f"dwell={stable_duration:.3f}s | "
                    f"cmd_fresh={cmd_fresh}"
                )

            else:

                self.get_logger().warning(
                    "[ROBOT MOTION] "
                    "STABLE_STOP_RELEASED | "
                    f"cmd_seen={cmd_seen} | "
                    f"cmd_fresh={cmd_fresh} | "
                    f"odom_fresh={odom_fresh} | "
                    f"command_quiet={command_quiet} | "
                    f"odometry_quiet={odometry_quiet}"
                )

            self.was_stable_stop_confirmed = (
                stable_stop_confirmed
            )


def main(args=None) -> None:

    rclpy.init(args=args)

    node = RobotMotionStateMonitorNode()

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