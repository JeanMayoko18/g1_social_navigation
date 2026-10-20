#!/usr/bin/env python3
"""
dog_odom_nav_filter.py

Purpose
-------
Improve the yaw/orientation consistency of dog_odom for Nav2 by using
raw IMU information in a way that is more robust to walking impacts.

Main idea
---------
During walking, foot-ground impacts can perturb:
- gyro_z (spurious yaw-rate)
- acceleration-based orientation correction

Instead of trusting the original yaw blindly, this node:
1) reads dog_odom orientation
2) reads dog_imu_raw
3) estimates impact perturbation from | ||acc|| - g |
4) corrects gyro_z using:
      gyro_z_corr = gyro_z - bias_z - k_g * perturbation
5) integrates a yaw estimate from corrected gyro_z
6) fuses odom yaw and imu-integrated yaw adaptively
7) republishes a filtered odometry for Nav2

Design choice
-------------
This node focuses mainly on improving YAW only.
It keeps the original x, y, z and twist linear components from dog_odom.

This is usually the safest first step for Nav2:
- avoid fake yaw turns during walking
- keep odom continuous
- do not rebuild full odometry from IMU

Topics
------
Inputs:
- /dog_odom           nav_msgs/msg/Odometry
- /dog_imu_raw        sensor_msgs/msg/Imu
- /cmd_vel_rlsar      geometry_msgs/msg/Twist   (optional, can be disabled)

Outputs:
- /dog_odom_filtered  nav_msgs/msg/Odometry
- /tf                 optional odom -> base frame

Important
---------
Make sure only ONE source publishes odom -> base frame TF if publish_tf=true.
"""


import rclpy
import math
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration

from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Twist, TransformStamped, Quaternion
from tf2_ros import TransformBroadcaster
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def norm3(x: float, y: float, z: float) -> float:
    return math.sqrt(x * x + y * y + z * z)


def angle_wrap(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def angle_lerp(a: float, b: float, w: float) -> float:
    """
    Interpolate angles safely.
    w=0 -> a
    w=1 -> b
    """
    d = angle_wrap(b - a)
    return angle_wrap(a + w * d)

def make_qos(reliable: bool, depth: int) -> QoSProfile:
    return QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE
        if reliable else ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
        depth=depth,
    )

def quaternion_to_euler(qx: float, qy: float, qz: float, qw: float):
    """Convert quaternion to roll, pitch, yaw."""
    # roll
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # pitch
    sinp = 2.0 * (qw * qy - qz * qx)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    # yaw
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> Quaternion:
    """Convert roll, pitch, yaw to quaternion."""
    q = Quaternion()

    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    q.w = cr * cp * cy + sr * sp * sy
    q.x = sr * cp * cy - cr * sp * sy
    q.y = cr * sp * cy + sr * cp * sy
    q.z = cr * cp * sy - sr * sp * cy
    return q
    
class DogOdomNavFilter(Node):
    def __init__(self):
        super().__init__("dog_odom_nav_filter")
        # -----------------------------
        # Parameters
        # -----------------------------
        self.declare_parameter("odom_in_topic", "/repub/dog_odom")
        self.declare_parameter("imu_topic", "/dog_imu_raw")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel_rlsar")
        self.declare_parameter("odom_out_topic", "/dog_odom_filtered")

        self.declare_parameter("use_cmd_context", True)
        self.declare_parameter("publish_tf", False)

        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("base_frame", "base_footprint")

        # Gravity magnitude used for impact/noise estimation
        self.declare_parameter("gravity", 9.81)

        # Initial bias calibration
        self.declare_parameter("bias_calib_duration_sec", 5.0)
        self.declare_parameter("bias_calib_max_samples", 2000)

        # Exponential moving averages
        self.declare_parameter("pert_alpha", 0.15)
        self.declare_parameter("gyro_alpha", 0.20)

        # Adaptive fusion thresholds based on filtered perturbation
        self.declare_parameter("pert_low", 0.20)
        self.declare_parameter("pert_high", 1.00)

        # g-sensitivity coefficient: gyro_z disturbance due to impacts
        self.declare_parameter("g_sensitivity_k", 0.0)

        # Optional gyro_z hard clamp after correction/filtering
        self.declare_parameter("max_abs_wz", 2.5)

        # Extra command-aware protection
        self.declare_parameter("cmd_vx_walk_thresh", 0.08)
        self.declare_parameter("cmd_wz_turn_thresh", 0.12)

        # When driving straight, amplify trust in the corrected yaw
        self.declare_parameter("straight_motion_extra_weight", 0.20)

        # When rotating on purpose, reduce trust in impact-based suppression
        self.declare_parameter("turn_motion_weight_scale", 0.50)

        # If true, also replace twist.angular.z with corrected filtered value
        self.declare_parameter("replace_twist_wz", True)

        # Optional debug logging
        self.declare_parameter("debug_log_period_sec", 1.0)

        self.declare_parameter("use_fixed_gyro_bias", True)
        self.declare_parameter("gyro_bias_z", 0.00421)

        self.declare_parameter("use_signed_g_sensitivity", True)
        self.declare_parameter("straight_hold_base_weight", 0.20)
        self.declare_parameter("straight_hold_max_weight", 0.55)
        self.declare_parameter("straight_hold_cmd_wz_thresh", 0.05)
        self.declare_parameter("straight_hold_wz_thresh", 0.08)

        self.declare_parameter("static_confirm_samples", 10)
        self.declare_parameter("moving_confirm_samples", 25)
        self.declare_parameter("static_cmd_vx_thresh", 0.05)
        self.declare_parameter("static_cmd_wz_thresh", 0.05)
        self.declare_parameter("static_wz_thresh", 0.15)

        self.declare_parameter("rebuild_xy", True)
        self.declare_parameter("linear_scale", 1.2545)
        self.declare_parameter("hold_weight", 0.6)
        self.declare_parameter("straight_imu_ref", 0.0)
        
        self.declare_parameter("odom_sub_reliable", False)
        self.declare_parameter("imu_sub_reliable", False)
        self.declare_parameter("cmd_sub_reliable", False)
        self.declare_parameter("odom_pub_reliable", False)
        self.declare_parameter("qos_depth", 10)

        # -----------------------------
        # Read parameters
        # -----------------------------
        self.odom_in_topic = self.get_parameter("odom_in_topic").value
        self.imu_topic = self.get_parameter("imu_topic").value
        self.cmd_vel_topic = self.get_parameter("cmd_vel_topic").value
        self.odom_out_topic = self.get_parameter("odom_out_topic").value

        self.use_cmd_context = bool(self.get_parameter("use_cmd_context").value)
        self.publish_tf = bool(self.get_parameter("publish_tf").value)

        self.odom_frame = self.get_parameter("odom_frame").value
        self.base_frame = self.get_parameter("base_frame").value

        self.gravity = float(self.get_parameter("gravity").value)

        self.bias_calib_duration_sec = float(
            self.get_parameter("bias_calib_duration_sec").value
        )
        self.bias_calib_max_samples = int(
            self.get_parameter("bias_calib_max_samples").value
        )

        self.pert_alpha = float(self.get_parameter("pert_alpha").value)
        self.gyro_alpha = float(self.get_parameter("gyro_alpha").value)

        self.pert_low = float(self.get_parameter("pert_low").value)
        self.pert_high = float(self.get_parameter("pert_high").value)

        self.g_sensitivity_k = float(self.get_parameter("g_sensitivity_k").value)
        self.max_abs_wz = float(self.get_parameter("max_abs_wz").value)

        self.cmd_vx_walk_thresh = float(
            self.get_parameter("cmd_vx_walk_thresh").value
        )
        self.cmd_wz_turn_thresh = float(
            self.get_parameter("cmd_wz_turn_thresh").value
        )

        self.straight_motion_extra_weight = float(
            self.get_parameter("straight_motion_extra_weight").value
        )
        self.turn_motion_weight_scale = float(
            self.get_parameter("turn_motion_weight_scale").value
        )

        self.replace_twist_wz = bool(self.get_parameter("replace_twist_wz").value)

        self.debug_log_period_sec = float(
            self.get_parameter("debug_log_period_sec").value
        )

        self.use_fixed_gyro_bias = bool(self.get_parameter("use_fixed_gyro_bias").value)
        self.gyro_bias_z_param = float(self.get_parameter("gyro_bias_z").value)


        self.static_confirm_samples = int(self.get_parameter("static_confirm_samples").value)
        self.moving_confirm_samples = int(self.get_parameter("moving_confirm_samples").value)
        self.static_cmd_vx_thresh = float(self.get_parameter("static_cmd_vx_thresh").value)
        self.static_cmd_wz_thresh = float(self.get_parameter("static_cmd_wz_thresh").value)
        self.static_wz_thresh = float(self.get_parameter("static_wz_thresh").value)

        self.use_signed_g_sensitivity = bool(
            self.get_parameter("use_signed_g_sensitivity").value
        )
        self.straight_hold_base_weight = float(
            self.get_parameter("straight_hold_base_weight").value
        )
        self.straight_hold_max_weight = float(
            self.get_parameter("straight_hold_max_weight").value
        )
        self.straight_hold_cmd_wz_thresh = float(
            self.get_parameter("straight_hold_cmd_wz_thresh").value
        )
        self.straight_hold_wz_thresh = float(
            self.get_parameter("straight_hold_wz_thresh").value
        )

        self.rebuild_xy = bool(self.get_parameter("rebuild_xy").value)
        self.linear_scale = float(self.get_parameter("linear_scale").value)

        self.filtered_pose_initialized = False
        self.x_f = 0.0
        self.y_f = 0.0
        self.z_f = 0.0
        self.last_filter_stamp = None
        self.hold_weight = float(self.get_parameter("hold_weight").value)
        self.straight_ref_samples = []
        self.straight_ref_min_samples = 10
        self.straight_imu_ref = float(self.get_parameter("straight_imu_ref").value)

        self.odom_sub_reliable = bool(self.get_parameter("odom_sub_reliable").value)
        self.imu_sub_reliable = bool(self.get_parameter("imu_sub_reliable").value)
        self.cmd_sub_reliable = bool(self.get_parameter("cmd_sub_reliable").value)
        self.odom_pub_reliable = bool(self.get_parameter("odom_pub_reliable").value)
        self.qos_depth = int(self.get_parameter("qos_depth").value)
        self.last_motion_debug_time = self.get_clock().now()
        self.motion_debug_period_sec = 60.0

        # -----------------------------
        # Internal state
        # -----------------------------
        self.last_imu_stamp: Optional[Time] = None
        self.last_odom_msg: Optional[Odometry] = None
        self.last_cmd_msg: Optional[Twist] = None

        self.have_imu = False
        self.have_odom = False

        # Bias calibration state
        self.bias_samples = []
        self.bias_start_time = self.get_clock().now()

        if self.use_fixed_gyro_bias:
            self.bias_z = self.gyro_bias_z_param
            self.bias_calibrated = True
            self.get_logger().info(
                f"Using fixed gyro bias: {self.bias_z:.6f} rad/s"
            )
        else:
            self.bias_z = 0.0
            self.bias_calibrated = False
            self.get_logger().info(
                "Using startup gyro bias calibration"
            )

        self.static_counter = 0
        self.moving_counter = 0
        self.is_static_confirmed = False

        self.straight_yaw_hold_active = False
        self.straight_yaw_ref = 0.0

        self.last_odom_time = None
        self.declare_parameter("max_odom_age_sec", 0.2)
        self.max_odom_age_sec = float(self.get_parameter("max_odom_age_sec").value)

        # Filter state
        self.pert_f = 0.0
        self.wz_corr_f = 0.0

        # IMU integrated yaw state
        self.yaw_imu_initialized = False
        self.yaw_imu = 0.0

        # Debug state
        self.last_debug_time = self.get_clock().now()

        # -----------------------------
        # ROS interfaces
        # -----------------------------
        odom_sub_qos = make_qos(self.odom_sub_reliable, self.qos_depth)
        imu_sub_qos = make_qos(self.imu_sub_reliable, self.qos_depth)
        cmd_sub_qos = make_qos(self.cmd_sub_reliable, 1)
        odom_pub_qos = make_qos(self.odom_pub_reliable, self.qos_depth)

        self.sub_odom = self.create_subscription(
            Odometry, self.odom_in_topic, self.cb_odom, odom_sub_qos
        )

        self.sub_imu = self.create_subscription(
            Imu, self.imu_topic, self.cb_imu, imu_sub_qos
        )

        self.sub_cmd = self.create_subscription(
            Twist, self.cmd_vel_topic, self.cb_cmd, cmd_sub_qos
        )

        self.pub_odom = self.create_publisher(
            Odometry, self.odom_out_topic, odom_pub_qos
        )

        self.tf_broadcaster = TransformBroadcaster(self) if self.publish_tf else None

        self.print_config()

        self.pert_signed_f = 0.0



    def log_throttled(self, key: str, level: str, period_sec: float, message: str):
        """
        Generic throttled logger.

        key:
            Unique name for this log message family.

        level:
            "debug", "info", "warn", or "error".

        period_sec:
            Minimum delay between two prints of the same key.

        message:
            Text to print.
        """

        now = self.get_clock().now()

        if not hasattr(self, "_last_log_times"):
            self._last_log_times = {}

        last_time = self._last_log_times.get(key)

        if last_time is None:
            should_log = True
        else:
            elapsed = (now - last_time).nanoseconds * 1e-9
            should_log = elapsed >= period_sec

        if not should_log:
            return

        self._last_log_times[key] = now

        if level == "debug":
            self.get_logger().debug(message)
        elif level == "info":
            self.get_logger().info(message)
        elif level == "warn":
            self.get_logger().warn(message)
        elif level == "error":
            self.get_logger().error(message)
        else:
            self.get_logger().info(message)


    def print_config(self):
        self.get_logger().info(
            "\n"
            "========================================\n"
            " dog_odom_nav_filter configuration\n"
            "========================================\n"
            f" odom_in_topic       : {self.odom_in_topic}\n"
            f" imu_topic           : {self.imu_topic}\n"
            f" cmd_vel_topic       : {self.cmd_vel_topic}\n"
            f" odom_out_topic      : {self.odom_out_topic}\n"
            f" use_cmd_context     : {self.use_cmd_context}\n"
            f" publish_tf          : {self.publish_tf}\n"
            f" g_sensitivity_k     : {self.g_sensitivity_k:.6f}\n"
            f" bias_calib_duration : {self.bias_calib_duration_sec:.2f} s\n"
            "========================================"
        )
    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    def cb_cmd(self, msg: Twist):
        self.last_cmd_msg = msg

    def cb_odom(self, msg: Odometry):
        self.last_odom_msg = msg
        self.have_odom = True
        self.last_odom_time = self.get_clock().now()

        # If IMU yaw not initialized yet, initialize it from odom yaw
        q = msg.pose.pose.orientation
        roll, pitch, yaw_odom = quaternion_to_euler(q.x, q.y, q.z, q.w)

        if not self.yaw_imu_initialized:
            self.yaw_imu = yaw_odom
            self.yaw_imu_initialized = True

    def cb_imu(self, msg: Imu):
        now = self.get_clock().now()
        self.have_imu = True

        # Use message timestamp if available; otherwise use node clock.
        imu_stamp = Time.from_msg(msg.header.stamp) if (
            msg.header.stamp.sec != 0 or msg.header.stamp.nanosec != 0
        ) else now

        # --------------------------------------------------------------
        # Step 1: bias calibration during initial static interval
        # --------------------------------------------------------------
        if not self.bias_calibrated:
            elapsed = (now - self.bias_start_time).nanoseconds * 1e-9
            self.bias_samples.append(float(msg.angular_velocity.z))

            enough_time = elapsed >= self.bias_calib_duration_sec
            enough_samples = len(self.bias_samples) >= self.bias_calib_max_samples

            if enough_time or enough_samples:
                if self.bias_samples:
                    self.bias_z = sum(self.bias_samples) / len(self.bias_samples)
                else:
                    self.bias_z = 0.0
                self.bias_calibrated = True
                self.get_logger().info(
                    f"Gyro z bias calibrated: bias_z={self.bias_z:.6f} rad/s "
                    f"using {len(self.bias_samples)} samples"
                )

        # --------------------------------------------------------------
        # Step 2: dt
        # --------------------------------------------------------------
        if self.last_imu_stamp is None:
            self.last_imu_stamp = imu_stamp
            return

        dt = (imu_stamp - self.last_imu_stamp).nanoseconds * 1e-9
        self.last_imu_stamp = imu_stamp

        if dt <= 0.0 or dt > 0.2:
            # Skip strange dt values
            return

        # --------------------------------------------------------------
        # Step 3: impact perturbation from acceleration magnitude
        # --------------------------------------------------------------
        ax = float(msg.linear_acceleration.x)
        ay = float(msg.linear_acceleration.y)
        az = float(msg.linear_acceleration.z)
        acc_norm = norm3(ax, ay, az)

        #perturbation = abs(acc_norm - self.gravity)
        #self.pert_f = self.pert_alpha * perturbation + (1.0 - self.pert_alpha) * self.pert_f

        perturbation_signed = acc_norm - self.gravity
        perturbation_abs = abs(perturbation_signed)

        self.pert_f = (
            self.pert_alpha * perturbation_abs
            + (1.0 - self.pert_alpha) * self.pert_f
        )

        self.pert_signed_f = (
            self.pert_alpha * perturbation_signed
            + (1.0 - self.pert_alpha) * self.pert_signed_f
        )

        # --------------------------------------------------------------
        # Step 4: corrected gyro_z
        # --------------------------------------------------------------
        wz_raw = float(msg.angular_velocity.z)
        #wz_corr = wz_raw - self.bias_z - self.g_sensitivity_k * self.pert_f

        if self.use_signed_g_sensitivity:
            wz_corr = wz_raw - self.bias_z - self.g_sensitivity_k * self.pert_signed_f
        else:
            wz_corr = wz_raw - self.bias_z - self.g_sensitivity_k * self.pert_f

        wz_corr = clamp(wz_corr, -self.max_abs_wz, self.max_abs_wz)
        


        # Low-pass corrected gyro_z
        self.wz_corr_f = self.gyro_alpha * wz_corr + (1.0 - self.gyro_alpha) * self.wz_corr_f

        # --------------------------------------------------------------
        # Step 5: integrate yaw from IMU
        # --------------------------------------------------------------
        if not self.yaw_imu_initialized and self.last_odom_msg is not None:
            q = self.last_odom_msg.pose.pose.orientation
            _, _, yaw_init = quaternion_to_euler(q.x, q.y, q.z, q.w)
            self.yaw_imu = yaw_init
            self.yaw_imu_initialized = True

        if self.last_odom_time is not None:
            odom_age = (now - self.last_odom_time).nanoseconds * 1e-9
            if odom_age > self.max_odom_age_sec:
                return
        
        if self.yaw_imu_initialized:
            self.yaw_imu = angle_wrap(self.yaw_imu + self.wz_corr_f * dt)
        
        # --------------------------------------------------------------
        # Step 6: if odom available, produce filtered odom
        # --------------------------------------------------------------
        if self.last_odom_msg is not None and self.yaw_imu_initialized:
            filtered = self.build_filtered_odom(self.last_odom_msg)
            self.pub_odom.publish(filtered)

            if self.publish_tf:
                self.publish_odom_tf(filtered)

        # --------------------------------------------------------------
        # Step 7: periodic debug log
        # --------------------------------------------------------------
        if self.debug_log_period_sec > 0.0:
            if (now - self.last_debug_time) >= Duration(seconds=self.debug_log_period_sec):
                self.last_debug_time = now

                self.log_throttled(
                    key="imu_filter_state",
                    level="info",
                    period_sec=10.0,
                    message=(
                        "\n"
                        "----------------------------------------\n"
                        " IMU filter state\n"
                        "----------------------------------------\n"
                        f" pert_abs_f       : {self.pert_f:.3f}\n"
                        f" pert_signed_f    : {self.pert_signed_f:.3f}\n"
                        f" wz_raw           : {wz_raw:.3f} rad/s\n"
                        f" wz_corr_f        : {self.wz_corr_f:.3f} rad/s\n"
                        f" bias_z           : {self.bias_z:.4f} rad/s\n"
                        f" g_sensitivity_k  : {self.g_sensitivity_k:.4f}\n"
                        f" yaw_imu          : {math.degrees(self.yaw_imu):.2f} deg\n"
                        f" bias_calibrated  : {self.bias_calibrated}\n"
                        "----------------------------------------"
                    )
                )

    # ------------------------------------------------------------------
    # Core filtering logic
    # ------------------------------------------------------------------
    def build_filtered_odom(self, odom_msg: Odometry) -> Odometry:
        """
        Build a filtered odometry message.

        Main idea:
        1. Read the original dog_odom pose and twist.
        2. Decide the current motion context:
        - static
        - walking straight
        - intentional turn
        - uncertain/default motion
        3. Compute a corrected yaw.
        4. Optionally rebuild x,y using the corrected yaw and forward velocity.
        5. Publish a physically more coherent odometry message for Nav2.
        """

        out = Odometry()
        out.header = odom_msg.header
        out.child_frame_id = odom_msg.child_frame_id

        # ----------------------------------------------------------
        # 1. Extract original orientation from dog_odom
        # ----------------------------------------------------------
        q = odom_msg.pose.pose.orientation
        roll, pitch, yaw_odom = quaternion_to_euler(q.x, q.y, q.z, q.w)

        # ----------------------------------------------------------
        # 2. Read command context
        # ----------------------------------------------------------
        cmd_vx = 0.0
        cmd_wz = 0.0

        if self.last_cmd_msg is not None:
            cmd_vx = float(self.last_cmd_msg.linear.x)
            cmd_wz = float(self.last_cmd_msg.angular.z)

        # ----------------------------------------------------------
        # 3. Detect static mode with hysteresis
        # ----------------------------------------------------------

        static_candidate = (
            #self.last_cmd_msg is not None and
            abs(cmd_vx) < self.static_cmd_vx_thresh and
            abs(cmd_wz) < self.static_cmd_wz_thresh and
            abs(self.wz_corr_f) < self.static_wz_thresh
        )


        if static_candidate:
            self.static_counter += 1
            self.moving_counter = 0
        else:
            self.moving_counter += 1
            self.static_counter = 0

        # Enter static mode only after several stable samples
        if not self.is_static_confirmed:
            if self.static_counter >= self.static_confirm_samples:
                self.is_static_confirmed = True

        # Exit static mode only after sustained non-static behavior
        else:
            if self.moving_counter >= self.moving_confirm_samples:
                self.is_static_confirmed = False

        # ----------------------------------------------------------
        # 4. Detect motion type
        # ----------------------------------------------------------
        walking_straight = (
            abs(cmd_vx) >= self.cmd_vx_walk_thresh and
            abs(cmd_wz) < self.cmd_wz_turn_thresh
        )

        intentional_turn = abs(cmd_wz) >= self.cmd_wz_turn_thresh
        active_motion_command = walking_straight or intentional_turn

        if active_motion_command:
            self.is_static_confirmed = False
            self.static_counter = 0
            #self.moving_counter = 0


        # ----------------------------------------------------------
        # 5. Manage straight-line yaw hold
        # ----------------------------------------------------------
        # When the robot starts a commanded straight motion, store the
        # initial yaw as a reference. This helps reduce fake yaw drift
        # caused by walking impacts when cmd_wz is close to zero.


        # When straight motion starts, capture both references at the same instant.
        if walking_straight and not self.straight_yaw_hold_active:
            self.straight_yaw_ref = yaw_odom
            self.straight_imu_ref = self.yaw_imu
            self.straight_yaw_hold_active = True



        # Disable yaw hold when the robot is no longer commanded to go straight.
        if not walking_straight:
            self.straight_yaw_hold_active = False
            self.straight_ref_samples = []

        # ----------------------------------------------------------
        # 6. Compute perturbation-based weight for covariance/debug
        # ----------------------------------------------------------
        # This weight is not the main yaw fusion weight anymore.
        # It is mainly used to increase yaw covariance when impacts are high.
        if self.pert_high <= self.pert_low:
            w = 1.0 if self.pert_f >= self.pert_low else 0.0
        else:
            w = (self.pert_f - self.pert_low) / (self.pert_high - self.pert_low)
            w = clamp(w, 0.0, 1.0)

        if self.use_cmd_context:
            if walking_straight:
                w = clamp(w + self.straight_motion_extra_weight, 0.0, 1.0)

            if intentional_turn:
                w = clamp(w * self.turn_motion_weight_scale, 0.0, 1.0)

        # ----------------------------------------------------------
        # 7. Select final yaw according to motion context
        # ----------------------------------------------------------
        if self.is_static_confirmed:
            # At rest, experiments showed that raw dog_odom yaw is more stable
            # than a purely integrated IMU yaw.
            yaw_final = yaw_odom

        elif walking_straight and self.straight_yaw_hold_active:
            # IMU yaw corrected by bias + g-sensitivity,
            # expressed relative to the beginning of the straight segment.
            yaw_imu_relative = angle_wrap(
                self.straight_yaw_ref + angle_wrap(self.yaw_imu - self.straight_imu_ref)
            )

            # During straight walking, do NOT force yaw to a fixed reference.
            # The goal is only to remove impact-induced yaw drift.
            yaw_final = yaw_imu_relative
            
            '''
                    
            elif walking_straight and self.straight_yaw_hold_active:
                yaw_final = self.straight_yaw_ref

            '''
        elif intentional_turn:
            # During intentional turns, avoid blocking the real rotation.
            # Only apply a very weak correction from the IMU-integrated yaw.
            yaw_final = angle_lerp(yaw_odom, self.yaw_imu, 0.10)

        else:
            # Conservative default behavior for uncertain motion.
            yaw_final = angle_lerp(yaw_odom, self.yaw_imu, 0.15)


        now = self.get_clock().now()

        if (now - self.last_motion_debug_time) >= Duration(seconds=self.motion_debug_period_sec):
            self.last_motion_debug_time = now

            self.log_throttled(
                key="motion_context",
                level="info",
                period_sec=10.0,
                message=(
                    "\n"
                    "----------------------------------------\n"
                    " Motion context\n"
                    "----------------------------------------\n"
                    f" cmd_vx           : {cmd_vx:.3f} m/s\n"
                    f" cmd_wz           : {cmd_wz:.3f} rad/s\n"
                    f" static_confirmed : {self.is_static_confirmed}\n"
                    f" walking_straight : {walking_straight}\n"
                    f" yaw_hold_active  : {self.straight_yaw_hold_active}\n"
                    f" yaw_odom         : {math.degrees(yaw_odom):.2f} deg\n"
                    f" yaw_ref          : {math.degrees(self.straight_yaw_ref):.2f} deg\n"
                    f" yaw_final        : {math.degrees(yaw_final):.2f} deg\n"
                    f" rebuild_xy       : {self.rebuild_xy}\n"
                    "----------------------------------------"
                )
            )

        # ----------------------------------------------------------
        # 8. Rebuild XY using corrected yaw
        # ----------------------------------------------------------
        # This is essential for Nav2:
        # if yaw is corrected but x,y remain from raw dog_odom,
        # the resulting odometry can be physically inconsistent.
        #
        # Here, x,y are rebuilt using:
        #   v_forward from dog_odom twist
        #   yaw_final from the corrected yaw logic
        #
        # This avoids integrating IMU acceleration for position, which would drift.
        stamp = Time.from_msg(odom_msg.header.stamp)

        if self.last_filter_stamp is None:
            self.last_filter_stamp = stamp
            self.x_f = odom_msg.pose.pose.position.x
            self.y_f = odom_msg.pose.pose.position.y
            self.z_f = odom_msg.pose.pose.position.z
            self.filtered_pose_initialized = True

        dt_pos = (stamp - self.last_filter_stamp).nanoseconds * 1e-9
        self.last_filter_stamp = stamp

        if dt_pos <= 0.0 or dt_pos > 0.2:
            dt_pos = 0.0

        if self.rebuild_xy and self.filtered_pose_initialized:
            v_forward = float(odom_msg.twist.twist.linear.x) * self.linear_scale

            self.x_f += v_forward * math.cos(yaw_final) * dt_pos
            self.y_f += v_forward * math.sin(yaw_final) * dt_pos
            self.z_f = odom_msg.pose.pose.position.z

            out.pose.pose.position.x = self.x_f
            out.pose.pose.position.y = self.y_f
            out.pose.pose.position.z = self.z_f
        else:
            out.pose.pose.position = odom_msg.pose.pose.position

        # ----------------------------------------------------------
        # 9. Rebuild quaternion using original roll/pitch and corrected yaw
        # ----------------------------------------------------------
        q_new = euler_to_quaternion(roll, pitch, yaw_final)
        out.pose.pose.orientation = q_new

        # ----------------------------------------------------------
        # 10. Copy covariance and increase yaw covariance under perturbation
        # ----------------------------------------------------------
        out.pose.covariance = odom_msg.pose.covariance

        cov = list(out.pose.covariance)
        yaw_var = cov[35]
        extra_yaw_var = 0.05 * w
        cov[35] = float(yaw_var + extra_yaw_var)
        out.pose.covariance = cov

        # ----------------------------------------------------------
        # 11. Copy twist and optionally replace angular.z
        # ----------------------------------------------------------
        out.twist.twist = odom_msg.twist.twist
        out.twist.covariance = odom_msg.twist.covariance

        if self.replace_twist_wz:
            out.twist.twist.angular.z = self.wz_corr_f

        return out

    def publish_odom_tf(self, odom_msg: Odometry):
        """
        Broadcast odom -> base_frame transform.
        """
        t = TransformStamped()
        t.header.stamp = odom_msg.header.stamp
        t.header.frame_id = self.odom_frame
        t.child_frame_id = self.base_frame

        t.transform.translation.x = odom_msg.pose.pose.position.x
        t.transform.translation.y = odom_msg.pose.pose.position.y
        t.transform.translation.z = odom_msg.pose.pose.position.z
        t.transform.rotation = odom_msg.pose.pose.orientation

        self.tf_broadcaster.sendTransform(t)


    def publish_odom_tf(self, odom_msg: Odometry):
        """
        Broadcast odom -> base_frame transform.
        """
        if self.tf_broadcaster is None:
            return

        t = TransformStamped()
        t.header = odom_msg.header
        t.header.frame_id = self.odom_frame
        t.child_frame_id = self.base_frame

        t.transform.translation.x = odom_msg.pose.pose.position.x
        t.transform.translation.y = odom_msg.pose.pose.position.y
        t.transform.translation.z = odom_msg.pose.pose.position.z
        t.transform.rotation = odom_msg.pose.pose.orientation

        self.tf_broadcaster.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    node = DogOdomNavFilter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("Shutting down dog_odom_nav_filter")
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()