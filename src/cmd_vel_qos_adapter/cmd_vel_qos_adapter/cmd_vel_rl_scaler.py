#!/usr/bin/env python3

"""
cmd_vel_nav_scaler.py

Purpose
-------
This node rescales navigation velocity commands before they are sent further
down the real robot command pipeline.

Why it is useful
----------------
Some robots, especially RL-based locomotion systems, do not respond well to
small velocity commands:

- very small linear commands may produce no motion at all
- moderate commands may produce unstable or weak walking
- only commands above a certain threshold may produce reliable motion

Nav2, however, often generates smooth and small velocity values that are
perfectly valid mathematically, but not always effective on the real robot.

This node solves that mismatch by:
- removing tiny commands (stop deadzone)
- preserving small commands near the goal (approach zone)
- boosting larger commands into an effective locomotion range
- clamping the result to a safe maximum

Typical use
-----------
/cmd_vel  --->  cmd_vel_nav_scaler  --->  /cmd_vel_nav_scaled

Then /cmd_vel_nav_scaled can be passed to twist_mux, while teleop commands
remain unchanged on their own branch.

Important note
--------------
This node only changes the command magnitude.
It does NOT check whether the robot is truly moving or whether odometry is valid.
"""
from nav_msgs.msg import Odometry, Path
import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String
import json


class CmdVelNavScaler(Node):
    def __init__(self):
        super().__init__("cmd_vel_nav_scaler")

        # ------------------------------------------------------------------
        # Parameters: topics
        # ------------------------------------------------------------------
        self.declare_parameter("input_topic", "/cmd_vel")
        self.declare_parameter("output_topic", "/cmd_vel_nav_scaled")

        # ------------------------------------------------------------------
        # Parameters: linear scaling
        # ------------------------------------------------------------------
        self.declare_parameter("linear_stop_deadzone", 0.03)
        self.declare_parameter("linear_approach_threshold", 0.12)
        self.declare_parameter("linear_approach_gain", 1.0)
        self.declare_parameter("linear_base_boost", 0.30)  # kept for compatibility
        self.declare_parameter("linear_gain", 1.0)
        self.declare_parameter("max_linear", 0.39)

        self.declare_parameter("linear_min_effective", 0.25)
        self.declare_parameter("linear_full_threshold", 0.20)

        # Parameters
        self.declare_parameter("enable_adaptive_linear_scale", False)
        self.declare_parameter("linear_cmd_scale", 1.0)
        self.declare_parameter("linear_cmd_scale_min", 0.45)
        self.declare_parameter("linear_cmd_scale_max", 1.0)
        self.declare_parameter("linear_adapt_lr", 0.02)
        self.declare_parameter("adapt_min_cmd_vx", 0.12)
        self.declare_parameter("adapt_min_odom_vx", 0.03)

        self.declare_parameter(
            "enable_forward_heading_compensation",
            False,
        )

        self.declare_parameter(
            "forward_heading_compensation_wz",
            0.0,
        )

        self.declare_parameter(
            "forward_heading_compensation_min_vx",
            0.08,
        )

        # ------------------------------------------------------------------
        # Parameters: angular scaling
        # ------------------------------------------------------------------
        # angular_deadzone:
        #   Commands smaller than this are forced to zero.
        #
        # angular_approach_threshold:
        #   Small heading corrections near goal are preserved softly.
        #
        # angular_approach_gain:
        #   Gain used in the soft correction zone.
        #
        # angular_min_effective:
        #   Minimum angular speed that really makes the robot turn.
        #
        # angular_full_threshold:
        #   Command value from which we assume the robot should already be
        #   in the effective turning regime.
        #
        # angular_gain:
        #   Gain applied above angular_full_threshold.
        #
        # max_angular:
        #   Final safety clamp.
        self.declare_parameter("angular_deadzone", 0.05)
        self.declare_parameter("angular_approach_threshold", 0.12)
        self.declare_parameter("angular_approach_gain", 1.0)
        self.declare_parameter("angular_min_effective", 0.12)
        self.declare_parameter("angular_full_threshold", 0.25)
        self.declare_parameter("angular_gain", 0.3)
        self.declare_parameter("max_angular", 0.20)
        self.declare_parameter(
            "g1_forward_heading_deadband_deg",
            2.0,
        )
        self.declare_parameter("forward_mode_threshold", 0.08)
        self.declare_parameter("turn_mode_threshold", 0.05)

        self.declare_parameter("g1_turn_enter_wz", 0.10)
        self.declare_parameter("g1_turn_exit_wz", 0.04)
        self.declare_parameter("g1_align_max_wz", 0.08)
        self.declare_parameter("g1_forward_max_vx", 0.16)
        self.declare_parameter("g1_policy", "state_machine")
        self.declare_parameter("g1_use_path_heading_error", True)
        self.declare_parameter("g1_align_enter_heading_deg", 10.0)
        self.declare_parameter("g1_align_exit_heading_deg", 4.0)
        self.declare_parameter("g1_forward_soft_wz", 0.03)
        self.declare_parameter("odom_debug_topic", "/odom_normalizer/debug")
        self.declare_parameter("path_topic", "/transformed_global_plan")
        self.declare_parameter("g1_align_source", "path_error")
        self.declare_parameter("g1_align_kp", 0.4)
        self.declare_parameter("g1_forward_wz_deadband", 0.05)

        # ------------------------------------------------------------------
        # Read parameters
        # ------------------------------------------------------------------
        self.input_topic = self.get_parameter("input_topic").value
        self.output_topic = self.get_parameter("output_topic").value

        self.linear_stop_deadzone = float(self.get_parameter("linear_stop_deadzone").value)
        self.linear_approach_threshold = float(self.get_parameter("linear_approach_threshold").value)
        self.linear_approach_gain = float(self.get_parameter("linear_approach_gain").value)
        self.linear_base_boost = float(self.get_parameter("linear_base_boost").value)
        self.linear_gain = float(self.get_parameter("linear_gain").value)
        self.max_linear = float(self.get_parameter("max_linear").value)
        self.linear_min_effective = float(self.get_parameter("linear_min_effective").value)
        self.linear_full_threshold = float(self.get_parameter("linear_full_threshold").value)
        self.enable_adaptive_linear_scale = bool(self.get_parameter("enable_adaptive_linear_scale").value)
        self.linear_cmd_scale = float(
            self.get_parameter("linear_cmd_scale").value
        )

        self.linear_cmd_scale_min = float(
            self.get_parameter("linear_cmd_scale_min").value
        )
        self.linear_cmd_scale_max = float(
            self.get_parameter("linear_cmd_scale_max").value
        )
        self.linear_adapt_lr = float(
            self.get_parameter("linear_adapt_lr").value
        )
        self.adapt_min_cmd_vx = float(
            self.get_parameter("adapt_min_cmd_vx").value
        )
        self.adapt_min_odom_vx = float(
            self.get_parameter("adapt_min_odom_vx").value
        )
        self.forward_mode_threshold = float(
            self.get_parameter("forward_mode_threshold").value
        )
        self.turn_mode_threshold = float(
            self.get_parameter("turn_mode_threshold").value
        )
        self.g1_use_path_heading_error = bool(
            self.get_parameter("g1_use_path_heading_error").value
        )
        self.g1_align_enter_heading_deg = float(
            self.get_parameter("g1_align_enter_heading_deg").value
        )
        self.g1_align_exit_heading_deg = float(
            self.get_parameter("g1_align_exit_heading_deg").value
        )
        self.g1_forward_soft_wz = float(
            self.get_parameter("g1_forward_soft_wz").value
        )
        self.g1_forward_heading_deadband_deg = float(
            self.get_parameter(
                "g1_forward_heading_deadband_deg"
            ).value
        )
        self.odom_debug_topic = str(self.get_parameter("odom_debug_topic").value)
        self.angular_deadzone = float(self.get_parameter("angular_deadzone").value)
        self.angular_approach_threshold = float(self.get_parameter("angular_approach_threshold").value)
        self.angular_approach_gain = float(self.get_parameter("angular_approach_gain").value)
        self.angular_min_effective = float(self.get_parameter("angular_min_effective").value)
        self.angular_full_threshold = float(self.get_parameter("angular_full_threshold").value)
        self.angular_gain = float(self.get_parameter("angular_gain").value)
        self.max_angular = float(self.get_parameter("max_angular").value)
        self.g1_policy = str(self.get_parameter("g1_policy").value)
        self.g1_turn_enter_wz = float(self.get_parameter("g1_turn_enter_wz").value)
        self.g1_turn_exit_wz = float(self.get_parameter("g1_turn_exit_wz").value)
        self.g1_align_max_wz = float(self.get_parameter("g1_align_max_wz").value)
        self.g1_forward_max_vx = float(self.get_parameter("g1_forward_max_vx").value)
        self.path_topic = str(self.get_parameter("path_topic").value)
        self.g1_align_source = str(self.get_parameter("g1_align_source").value)
        self.g1_align_kp = float(self.get_parameter("g1_align_kp").value)
        self.g1_forward_wz_deadband = float(self.get_parameter("g1_forward_wz_deadband" ).value)

        self.enable_forward_heading_compensation = bool(
            self.get_parameter(
                "enable_forward_heading_compensation"
            ).value
        )

        self.forward_heading_compensation_wz = float(
            self.get_parameter(
                "forward_heading_compensation_wz"
            ).value
        )

        self.forward_heading_compensation_min_vx = float(
            self.get_parameter(
                "forward_heading_compensation_min_vx"
            ).value
        )

        self.last_path = None
        self.last_robot_yaw = None
        self.last_path_heading_error_deg = None
        self.last_odom_vx = 0.0
        self.last_robot_x = None
        self.last_robot_y = None
        self.last_path_heading_error_deg = None
        self.mode = "IDLE"

        # ------------------------------------------------------------------
        # ROS interfaces
        # ------------------------------------------------------------------
        self.sub = self.create_subscription(Twist, self.input_topic, self.cb, 10)
        self.odom_debug_sub = self.create_subscription(
            String,
            self.odom_debug_topic,
            self.cb_odom_debug,
            10,
        )
        self.path_sub = self.create_subscription(
            Path,
            self.path_topic,
            self.cb_path,
            10,
        )
        self.odom_sub = self.create_subscription(
            Odometry,
            "/odom",
            self.cb_odom,
            10,
        )
        self.pub = self.create_publisher(Twist, self.output_topic, 10)

        self.get_logger().info(
            "\n"
            "=========================================================\n"
            "            G1 NAVIGATION COMMAND SCALER\n"
            "=========================================================\n"
            f" Input topic           : {self.input_topic}\n"
            f" Output topic          : {self.output_topic}\n"
            f" G1 policy             : {self.g1_policy}\n"
            f" Path topic            : {self.path_topic}\n"
            f" Heading source        : "
            f"{'Path heading' if self.g1_use_path_heading_error else 'Command wz'}\n"

            "========================================================="
        )

    @staticmethod
    def clamp(v: float, vmin: float, vmax: float) -> float:
        """Clamp value v to the interval [vmin, vmax]."""
        return max(vmin, min(vmax, v))

    def wrap_angle(self, a):
        return math.atan2(math.sin(a), math.cos(a))

    def format_section(self, title, fields):
        lines = []
        lines.append("")
        lines.append("=" * 56)
        lines.append(title)
        lines.append("=" * 56)

        for key, value in fields:
            lines.append(f"{key:<18}: {value}")

        lines.append("=" * 56)
        return "\n".join(lines)

    def update_path_heading_error(self):
        if self.last_path is None:
            return

        if self.last_robot_yaw is None:
            return

        if self.last_robot_x is None:
            return

        if self.last_robot_y is None:
            return

        poses = self.last_path.poses

        if len(poses) < 2:
            return

        # ------------------------------------------------------------
        # 1. Find the path point closest to the current robot position.
        # ------------------------------------------------------------
        closest_index = min(
            range(len(poses)),
            key=lambda i: math.hypot(
                poses[i].pose.position.x - self.last_robot_x,
                poses[i].pose.position.y - self.last_robot_y,
            ),
        )

        p0 = poses[closest_index].pose.position

        # ------------------------------------------------------------
        # 2. Find a point approximately 0.50 m ahead on the path.
        # ------------------------------------------------------------
        p1 = None

        for pose_stamped in poses[closest_index + 1:]:
            p = pose_stamped.pose.position

            distance = math.hypot(
                p.x - p0.x,
                p.y - p0.y,
            )

            if distance >= 0.50:
                p1 = p
                break

        # ------------------------------------------------------------
        # 3. If less than 0.50 m remains, use the final path point.
        # ------------------------------------------------------------
        if p1 is None:
            p1 = poses[-1].pose.position

        dx = p1.x - p0.x
        dy = p1.y - p0.y

        if math.hypot(dx, dy) < 1e-3:
            return

        # ------------------------------------------------------------
        # 4. Local path direction, expressed in odom.
        # ------------------------------------------------------------
        path_heading = math.atan2(
            dy,
            dx,
        )

        # ------------------------------------------------------------
        # 5. Robot heading error relative to the path.
        # Both quantities are expressed in odom.
        # ------------------------------------------------------------
        err = self.wrap_angle(
            path_heading - self.last_robot_yaw
        )

        self.last_path_heading_error_deg = math.degrees(
            err
        )
            
    
    def cb_odom(self, msg):
        self.last_odom_vx = float(
            msg.twist.twist.linear.x
        )

        self.last_robot_x = float(
            msg.pose.pose.position.x
        )

        self.last_robot_y = float(
            msg.pose.pose.position.y
        )

        q = msg.pose.pose.orientation

        siny = 2.0 * (
            q.w * q.z
            + q.x * q.y
        )

        cosy = 1.0 - 2.0 * (
            q.y * q.y
            + q.z * q.z
        )

        self.last_robot_yaw = math.atan2(
            siny,
            cosy,
        )

        # First update the heading error with the latest odometry.
        self.update_path_heading_error()

        # Safe formatting when no path has been received yet.
        if self.last_path_heading_error_deg is None:
            path_error_text = "N/A"
        else:
            path_error_text = (
                f"{self.last_path_heading_error_deg:+.2f} deg"
            )

        self.get_logger().info(
            (
                f"Robot odom: "
                f"x={self.last_robot_x:+.3f} "
                f"y={self.last_robot_y:+.3f} "
                f"yaw={math.degrees(self.last_robot_yaw):+.2f} deg | "
                f"Path error={path_error_text}"
            ),
            throttle_duration_sec=0.5,
        )

    def cb_path(self, msg):
        self.last_path = msg

        self.get_logger().info(
            f"Transformed global plan frame: {msg.header.frame_id}",
            throttle_duration_sec=2.0,
        )

        self.update_path_heading_error()
    
    def cb_odom_debug(self, msg):
        try:
            data = json.loads(msg.data)

            if "path_heading_error_deg" in data:
                self.last_path_heading_error_deg = float(
                    data["path_heading_error_deg"]
                )

        except Exception:
            return  

    def apply_g1_motion_compatibility(
        self,
        out: Twist,
        raw_msg: Twist,
    ):
        """
        Adapt DWB commands to the locomotion characteristics of the Unitree G1.

        The purpose is not to replace the Nav2 controller.

        Rules
        -----
        1. Forward-dominant motion:
        suppress small parasitic yaw commands.

        2. Curved forward motion:
        preserve forward motion but limit yaw.

        3. Rotation-dominant motion:
        allow in-place rotation when forward command is negligible.
        """
        raw_wz = float(raw_msg.angular.z)

        # Use the scaled linear command to decide whether
        # forward motion is effectively requested.
        vx_abs = abs(out.linear.x)

        # Keep raw Nav2 angular command to interpret
        # whether yaw intention is small or significant.
        wz_abs = abs(raw_wz)
        vx_abs = abs(out.linear.x)

        # ============================================================
        # CASE 1 / 2:
        # Nav2 is requesting actual forward motion.
        # ============================================================
        if vx_abs >= self.forward_mode_threshold:

            # --------------------------------------------------------
            # Small wz while moving forward:
            # considered undesirable DWB curvature for the G1.
            # --------------------------------------------------------
            if wz_abs <= self.g1_forward_wz_deadband:
                self.mode = "FORWARD_STRAIGHT"
                out.angular.z = 0.0

            # --------------------------------------------------------
            # Significant yaw request:
            # preserve the intention, but limit the angular component.
            # --------------------------------------------------------
            else:
                self.mode = "FORWARD_CURVED"
                out.angular.z = self.clamp(
                    out.angular.z,
                    -self.g1_forward_soft_wz,
                    self.g1_forward_soft_wz,
                )

            return out

        # ============================================================
        # CASE 3:
        # Almost no forward request + significant angular request.
        # This is treated as a genuine rotation request.
        # ============================================================
        if wz_abs >= self.turn_mode_threshold:
            self.mode = "ROTATE"
            out.linear.x = 0.0

            out.angular.z = self.clamp(
                out.angular.z,
                -self.g1_align_max_wz,
                self.g1_align_max_wz,
            )

            return out

        # ============================================================
        # Otherwise:
        # insignificant command.
        # ============================================================
        self.mode = "IDLE"
        out.linear.x = 0.0
        out.angular.z = 0.0

        return out
    def apply_g1_exclusive_policy(self, out):
        """
        Legacy G1 exclusive policy.

        If forward walking is significant, suppress yaw.
        If yaw is significant and forward walking is weak, rotate in place.
        Otherwise stop.
        """
        # ==========================================================
        # Humanoid walking policy
        #
        # The Unitree G1 behaves much more reliably when forward
        # walking and turning are separated.
        #
        # Forward walking:
        #   prioritize vx and suppress yaw.
        #
        # Turning:
        #   stop forward motion first, then rotate in place.
        # ==========================================================

        vx_abs = abs(out.linear.x)
        wz_abs = abs(out.angular.z)

        FORWARD_THRESHOLD = self.forward_mode_threshold
        TURN_THRESHOLD = self.turn_mode_threshold

        # ----------------------------------------------------------
        # Mode 1: Forward walking
        #
        # Any significant forward command suppresses yaw completely.
        # ----------------------------------------------------------
        if vx_abs >= FORWARD_THRESHOLD:
            out.angular.z = 0.0

        # ----------------------------------------------------------
        # Mode 2: Rotation in place
        #
        # Only allow rotation when almost not walking.
        # ----------------------------------------------------------
        elif wz_abs >= TURN_THRESHOLD:

            out.linear.x = 0.0

            out.angular.z = self.clamp(
                out.angular.z,
                -0.08,
                0.08,
            )

        # ----------------------------------------------------------
        # Mode 3: Idle
        # ----------------------------------------------------------
        else:

            out.linear.x = 0.0
            out.angular.z = 0.0

        return out

    def scale_linear(self, v: float) -> float:
        """
        Scale linear.x using 4 zones.

        Zone 1:
            True stop.

        Zone 2:
            Any real motion request below the G1 locomotion threshold
            is promoted to the minimum effective speed.

        Zone 3:
            Smooth transition above the minimum effective speed.

        Zone 4:
            Continue scaling without discontinuity.
        """

        a = abs(v)

        # ----------------------------------------------------------
        # Zone 1: true stop
        # ----------------------------------------------------------
        if a < self.linear_stop_deadzone:
            return 0.0

        s = math.copysign(1.0, v)

        # ----------------------------------------------------------
        # Zone 2: compensate Unitree deadzone
        # ----------------------------------------------------------
        if a < self.linear_approach_threshold:
            return s * self.linear_min_effective

        # ----------------------------------------------------------
        # Value reached at the end of Zone 3
        # ----------------------------------------------------------
        transition_end = (
            self.linear_min_effective
            + self.linear_gain
            * (
                self.linear_full_threshold
                - self.linear_approach_threshold
            )
        )

        # ----------------------------------------------------------
        # Zone 3: smooth transition
        # ----------------------------------------------------------
        if a < self.linear_full_threshold:
            t = (
                (a - self.linear_approach_threshold)
                / max(
                    1e-6,
                    self.linear_full_threshold
                    - self.linear_approach_threshold,
                )
            )

            out = (
                (1.0 - t)
                * self.linear_min_effective
                + t
                * transition_end
            )

            return s * self.clamp(
                out,
                0.0,
                self.max_linear,
            )

        # ----------------------------------------------------------
        # Zone 4: continue from Zone 3 without a jump
        # ----------------------------------------------------------
        out = (
            transition_end
            + self.linear_gain
            * (
                a
                - self.linear_full_threshold
            )
        )

        return s * self.clamp(
            out,
            0.0,
            self.max_linear,
        )
    def scale_angular(self, w: float) -> float:
        """
        Scale angular.z using 4 zones.

        Zone 1: stop zone
            Very small angular commands are forced to zero.

        Zone 2: final approach zone
            Small heading corrections are preserved gently.

        Zone 3: transition zone
            Commands are ramped toward the minimum effective turning speed.

        Zone 4: effective turning zone
            Output stays in a range that really makes the robot turn,
            with a final safety clamp.
        """
        a = abs(w)

        if a < self.angular_deadzone:
            return 0.0

        s = math.copysign(1.0, w)

        # Zone 2: small heading correction
        if a < self.angular_approach_threshold:
            out = self.angular_approach_gain * a
            out = self.clamp(out, 0.0, self.max_angular)
            return s * out

        # Zone 3: transition toward effective turning speed
        if a < self.angular_full_threshold:
            t = (
                (a - self.angular_approach_threshold)
                / max(1e-6, (self.angular_full_threshold - self.angular_approach_threshold))
            )
            out = (1.0 - t) * self.angular_approach_threshold + t * self.angular_min_effective
            out = self.clamp(out, 0.0, self.max_angular)
            return s * out

        # Zone 4: effective turning
        out = self.angular_min_effective + self.angular_gain * (a - self.angular_full_threshold)
        out = self.clamp(out, 0.0, self.max_angular)
        return s * out

    def cb(self, msg: Twist):
        """
        Temporary passthrough mode with optional forward heading compensation.

        Input:
            Incoming Twist command.

        Processing:
            Copies the incoming command unchanged, then optionally adds a fixed
            angular compensation during forward straight motion.

        Output:
            Compensated Twist command.

        Side effects:
            Publishes one Twist message and writes a diagnostic log.
        """
        out = Twist()

        # ------------------------------------------------------------------
        # Exact passthrough of all Twist components
        # ------------------------------------------------------------------

        out.linear.x = msg.linear.x
        out.linear.y = msg.linear.y
        out.linear.z = msg.linear.z

        out.angular.x = msg.angular.x
        out.angular.y = msg.angular.y
        out.angular.z = msg.angular.z

        # ------------------------------------------------------------------
        # Forward heading compensation
        # ------------------------------------------------------------------

        applied_compensation_wz = 0.0

        if (
            self.enable_forward_heading_compensation
            and abs(out.linear.x)
            >= self.forward_heading_compensation_min_vx
            and abs(msg.angular.z) < 1e-6
        ):
            applied_compensation_wz = (
                self.forward_heading_compensation_wz
            )

            out.angular.z += (
                applied_compensation_wz
            )

            self.mode = "FORWARD_HEADING_COMPENSATION"

        else:
            self.mode = "PASSTHROUGH"

        # ------------------------------------------------------------------
        # Publish
        # ------------------------------------------------------------------

        self.pub.publish(
            out
        )

        # ------------------------------------------------------------------
        # Diagnostic output
        # ------------------------------------------------------------------

        self.get_logger().info(
            self.format_section(
                "G1 NAVIGATION SCALER",
                [
                    (
                        "Mode",
                        self.mode,
                    ),
                    (
                        "Input",
                        f"vx={msg.linear.x:+.3f}  "
                        f"wz={msg.angular.z:+.3f}",
                    ),
                    (
                        "Compensation",
                        f"wz={applied_compensation_wz:+.3f}",
                    ),
                    (
                        "Output",
                        f"vx={out.linear.x:+.3f}  "
                        f"wz={out.angular.z:+.3f}",
                    ),
                ],
            ),
            throttle_duration_sec=0.5,
        )

    def cb_old(self, msg: Twist):
        """
        Callback for incoming Twist commands.

        This node only keeps planar motion:
        - linear.x
        - angular.z

        All other components are forced to zero.
        """
        out = Twist()

        out.linear.x = self.scale_linear(msg.linear.x)
        if self.enable_adaptive_linear_scale:
            cmd_abs = abs(out.linear.x)
            odom_abs = abs(self.last_odom_vx)

            if cmd_abs > self.adapt_min_cmd_vx and odom_abs > self.adapt_min_odom_vx:
                ratio = odom_abs / max(cmd_abs, 1e-6)

                self.linear_cmd_scale = (
                    (1.0 - self.linear_adapt_lr) * self.linear_cmd_scale
                    + self.linear_adapt_lr * ratio
                )

                self.linear_cmd_scale = self.clamp(
                    self.linear_cmd_scale,
                    self.linear_cmd_scale_min,
                    self.linear_cmd_scale_max,
                )

        out.linear.x *= self.linear_cmd_scale
        out.linear.y = 0.0
        out.linear.z = 0.0

        out.angular.x = 0.0
        out.angular.y = 0.0
        out.angular.z = self.scale_angular(msg.angular.z)

        ################################################################################
        # ==========================================================
        # Final G1 command shaping
        # ==========================================================
        # Principle:
        # - allow gentle yaw while walking
        # - forbid almost-in-place rotation from Nav2
        # - do not reduce vx because of wz here; DWB already computed
        #   its trajectory assuming its own vx/wz pair.

        # --------------------------------------------------
        # G1 ALIGN / FORWARD state machine and EXCLUSIVE Approaches
        # --------------------------------------------------
        if self.g1_policy == "exclusive":
            out = self.apply_g1_exclusive_policy(out)

        elif self.g1_policy == "motion_compatibility":
            out = self.apply_g1_motion_compatibility(
                out,
                msg,
            )

        else:
            self.get_logger().warn(
                f"Unknown g1_policy='{self.g1_policy}', "
                "using motion_compatibility",
                throttle_duration_sec=2.0,
            )

            out = self.apply_g1_motion_compatibility(
                out,
                msg,
            )

        self.pub.publish(out)

        self.get_logger().info(
            self.format_section(
                "G1 NAVIGATION SCALER",
                [
                    (
                        "Mode",
                        getattr(
                            self,
                            "motion_mode",
                            "NA",
                        ),
                    ),
                    (
                        "Input",
                        f"vx={msg.linear.x:+.3f}  "
                        f"wz={msg.angular.z:+.3f}",
                    ),
                    (
                        "Output",
                        f"vx={out.linear.x:+.3f}  "
                        f"wz={out.angular.z:+.3f}",
                    ),
                ],
            ),
            throttle_duration_sec=0.5,
        )


def main():
    rclpy.init()
    node = CmdVelNavScaler()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()