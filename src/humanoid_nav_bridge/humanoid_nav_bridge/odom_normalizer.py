#!/usr/bin/env python3
"""
Adaptive Motion Consistency Estimator (AMCE) for dog_odom.

Principle
---------
dog_odom remains the primary odometry source.

This node:
1. Converts dog_odom into a local frame.
2. Computes incremental motion between consecutive dog_odom samples.
3. Estimates the expected motion from the current command and learned G1 response gains.
4. Computes Motion Consistency Indices (MCI) for dx, dy and dyaw.
5. Blends dog_odom increments with expected increments according to MCI.
6. Integrates the corrected increments into /odom_dog_norm.

cmd_vel is never integrated directly as odometry.
It is only used as a weak physical consistency reference.


Adaptive Odometry Calibration Framework (AOCF).

Pipeline
--------
1. TMIR: represent raw dog_odom as local temporal increments.
2. MPM: predict command-conditioned body-frame motion.
3. AMCE: estimate component-wise motion consistency.
4. AFI: fuse measured and expected body-frame increments.
5. AHC: calibrate the navigation heading.
6. ATC: calibrate the navigation twist.

dog_odom remains the primary odometry source.
cmd_vel is used only as a temporally validated physical reference.
"""
import threading
import copy
import math
from std_msgs.msg import String
import json
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup


def make_qos(reliable: bool, depth: int) -> QoSProfile:
    """
    Create a ROS 2 QoS profile.

    Role in the navigation pipeline
    -------------------------------
    The odom normalizer receives raw odometry from the robot and publishes
    corrected odometry for Nav2. Different topics may require different QoS:

        /repub/dog_odom  ---> subscriber
        /cmd_vel_rlsar   ---> subscriber
        /odom            ---> publisher
        /odom_dog_norm   ---> publisher

    BEST_EFFORT is typically used for high-rate sensor data, whereas
    RELIABLE is preferred for control and navigation information.
    """
    return QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE if reliable else ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
        depth=depth,
    )


def valid_quaternion(q) -> bool:
    """
    Verify that an incoming quaternion is numerically valid.

    Role in the navigation pipeline
    -------------------------------
    The raw Unitree odometry provides the robot orientation as a quaternion.
    Before extracting yaw, the quaternion must:

      - contain only finite values,
      - have a non-zero norm.

    Rejecting invalid quaternions prevents publishing corrupted odometry
    and avoids injecting invalid orientations into Nav2, AMCL and TF.
    """
    vals = [q.x, q.y, q.z, q.w]
    if not all(math.isfinite(v) for v in vals):
        return False

    n = math.sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w)
    return n > 1e-6


def yaw_from_quaternion(q) -> float:
    """
    Extract the planar yaw angle from a quaternion.

    Role in the navigation pipeline
    -------------------------------
    Nav2 operates in a planar (2D) navigation space. The incoming Unitree
    odometry is fully 3D, but only the heading around the vertical axis
    (yaw) is required by the normalizer.

    Pipeline:

        dog_odom quaternion
                ↓
          yaw extraction
                ↓
        incremental yaw correction
                ↓
          corrected navigation odometry
    """
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)

    return math.atan2(siny, cosy)


def quaternion_from_yaw(yaw: float):
    """
    Build a 2D quaternion from a corrected yaw angle.

    Role in the navigation pipeline
    -------------------------------
    After the normalizer estimates the corrected planar heading, the result
    must be converted back into a quaternion before publishing the output
    odometry and TF.

    Pipeline:

        corrected yaw
              ↓
      quaternion reconstruction
              ↓
      /odom or /odom_dog_norm
              ↓
      TF odom -> base_footprint
              ↓
      AMCL / Nav2 / RViz
    """
    return (
        0.0,
        0.0,
        math.sin(yaw * 0.5),
        math.cos(yaw * 0.5),
    )

class OdomNormalizer(Node):
    def __init__(self):
        super().__init__("odom_normalizer")

        # ------------------------------------------------------------
        # Topics and frames
        # ------------------------------------------------------------
        self.declare_parameter("input_odom_topic", "/repub/dog_odom")
        self.declare_parameter("output_odom_topic", "/odom_dog_norm")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel_nav_scaled")
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("base_frame", "base_footprint")
        self.declare_parameter("publish_tf", False)


        # ------------------------------------------------------------
        # QoS and time policy
        # ------------------------------------------------------------
        self.declare_parameter("use_node_time", True)
        self.declare_parameter("sub_reliable", False)
        self.declare_parameter("cmd_reliable", True)
        self.declare_parameter("pub_reliable", True)
        self.declare_parameter("qos_depth", 10)

        # ------------------------------------------------------------
        # 2D projection
        # ------------------------------------------------------------
        self.declare_parameter("force_2d", True)
        self.declare_parameter("zero_z", True)
        self.declare_parameter("zero_roll_pitch", True)
        self.declare_parameter("remap_yawrate_y_to_z", False)

        # ------------------------------------------------------------
        # Startup behavior
        # ------------------------------------------------------------
        self.declare_parameter("startup_freeze_sec", 5.0)
        self.declare_parameter("startup_ramp_sec", 2.0)

        # ------------------------------------------------------------
        # Hard increment safety limits
        # ------------------------------------------------------------
        self.declare_parameter("max_step_m", 0.12)
        self.declare_parameter("max_step_yaw", 0.12)

        # ------------------------------------------------------------
        # Low-pass filtering on corrected increments
        # ------------------------------------------------------------
        self.declare_parameter("enable_lowpass", True)
        self.declare_parameter("position_alpha", 0.70)
        self.declare_parameter("yaw_alpha", 0.70)
        self.declare_parameter("yaw_observation_gain", 0.03)
        self.declare_parameter("max_yaw_observation_correction", 0.0015)

        # ------------------------------------------------------------
        # Twist estimation
        # ------------------------------------------------------------
        self.declare_parameter("compute_twist_from_corrected_pose", True)
        self.declare_parameter("max_vx", 0.8)
        self.declare_parameter("max_vy", 0.4)
        self.declare_parameter("max_wz", 1.2)

        # ------------------------------------------------------------
        # Output yaw-rate filtering
        # ------------------------------------------------------------
        self.declare_parameter("filter_output_wz", True)
        self.declare_parameter("ema_wz_alpha", 0.25)
        self.declare_parameter("max_output_wz", 0.35)

        # Output linear velocity filtering
        self.declare_parameter("filter_output_vx", True)
        self.declare_parameter("ema_vx_alpha", 0.20)
        self.declare_parameter("max_output_vx", 0.25)

        self.declare_parameter("filter_output_vy", True)
        self.declare_parameter("ema_vy_alpha", 0.15)
        self.declare_parameter("max_output_vy", 0.08)

        # ------------------------------------------------------------
        # AMCE parameters
        # ------------------------------------------------------------
        self.declare_parameter("enable_amce", True)
        self.declare_parameter("amce_alpha", 0.02)

        self.declare_parameter("straight_cmd_vx_min", 0.08)
        self.declare_parameter("straight_cmd_wz_max", 0.08)
        self.declare_parameter("turn_cmd_wz_min", 0.08)

        self.declare_parameter("vx_gain_min", 0.35)
        self.declare_parameter("vx_gain_max", 1.20)
        self.declare_parameter("wz_gain_min", 0.30)
        self.declare_parameter("wz_gain_max", 1.30)

        self.declare_parameter("mci_x_min", 0.60)
        self.declare_parameter("mci_y_min", 0.20)
        self.declare_parameter("mci_yaw_min", 0.05)

        self.declare_parameter("mci_x_sensitivity", 8.0)
        self.declare_parameter("mci_y_sensitivity", 16.0)
        self.declare_parameter("mci_yaw_sensitivity", 10.0)

        self.declare_parameter("max_expected_vx_margin", 0.06)
        self.declare_parameter("max_expected_vy", 0.025)
        self.declare_parameter("max_expected_wz_margin", 0.08)
        self.declare_parameter("max_expected_wz_straight", 0.008)

        self.declare_parameter("enable_straight_yaw_gate", True)
        self.declare_parameter("straight_yaw_gate_wz", 0.008)
        self.declare_parameter("yaw_correction_mode", "raw_increment")

        # ------------------------------------------------------------
        # Stationary hold
        # ------------------------------------------------------------
        self.declare_parameter("stationary_hold_enabled", True)
        self.declare_parameter("stationary_cmd_vx_max", 0.02)
        self.declare_parameter("stationary_cmd_wz_max", 0.02)
        self.declare_parameter("stationary_raw_step_max", 0.003)
        self.declare_parameter("stationary_raw_yaw_max", 0.003)
        self.declare_parameter("stationary_zero_twist", True)
        self.declare_parameter("cmd_timeout_sec", 0.35)
        self.declare_parameter("max_state_age_sec", 0.5)
        self.declare_parameter("odom_period_alpha", 0.02)
        self.declare_parameter("odom_gap_factor", 5.0)
        self.declare_parameter("enable_amce_pose_fusion", False)
        self.declare_parameter("straight_raw_wz_deadband", 0.0)
        self.declare_parameter("output_frequency_hz", 45.0)
        self.declare_parameter("straight_yaw_confirm_samples", 5)
        self.declare_parameter("straight_yaw_release_samples", 2)
        self.declare_parameter("odom_period_init_samples", 20)

        # ------------------------------------------------------------
        # Raw-to-corrected yaw alignment
        # ------------------------------------------------------------
        self.yaw_alignment_initialized = False
        self.yaw_raw_to_corrected_offset = 0.0

        # ------------------------------------------------------------
        # Callback groups
        # ------------------------------------------------------------
        self.odom_cb_group = MutuallyExclusiveCallbackGroup()
        self.cmd_cb_group = MutuallyExclusiveCallbackGroup()
        self.timer_cb_group = MutuallyExclusiveCallbackGroup()

        # ------------------------------------------------------------
        # Read parameters
        # ------------------------------------------------------------
        self.input_odom_topic = self.get_parameter("input_odom_topic").value
        self.output_odom_topic = self.get_parameter("output_odom_topic").value
        self.cmd_vel_topic = self.get_parameter("cmd_vel_topic").value
        self.odom_frame = self.get_parameter("odom_frame").value
        self.base_frame = self.get_parameter("base_frame").value

        self.use_node_time = bool(self.get_parameter("use_node_time").value)
        self.sub_reliable = bool(self.get_parameter("sub_reliable").value)
        self.cmd_reliable = bool(self.get_parameter("cmd_reliable").value)
        self.pub_reliable = bool(self.get_parameter("pub_reliable").value)
        self.qos_depth = int(self.get_parameter("qos_depth").value)

        self.force_2d = bool(self.get_parameter("force_2d").value)
        self.zero_z = bool(self.get_parameter("zero_z").value)
        self.zero_roll_pitch = bool(self.get_parameter("zero_roll_pitch").value)
        self.remap_yawrate_y_to_z = bool(self.get_parameter("remap_yawrate_y_to_z").value)

        self.startup_freeze_sec = float(self.get_parameter("startup_freeze_sec").value)
        self.startup_ramp_sec = float(self.get_parameter("startup_ramp_sec").value)

        self.max_step_m = float(self.get_parameter("max_step_m").value)
        self.max_step_yaw = float(self.get_parameter("max_step_yaw").value)

        self.enable_lowpass = bool(self.get_parameter("enable_lowpass").value)
        self.position_alpha = float(self.get_parameter("position_alpha").value)
        self.yaw_alpha = float(self.get_parameter("yaw_alpha").value)

        self.compute_twist_from_corrected_pose = bool(
            self.get_parameter("compute_twist_from_corrected_pose").value
        )
        self.max_vx = float(self.get_parameter("max_vx").value)
        self.max_vy = float(self.get_parameter("max_vy").value)
        self.max_wz = float(self.get_parameter("max_wz").value)

        self.filter_output_wz = bool(self.get_parameter("filter_output_wz").value)
        self.ema_wz_alpha = float(self.get_parameter("ema_wz_alpha").value)
        self.max_output_wz = float(self.get_parameter("max_output_wz").value)

        self.filter_output_vx = bool(self.get_parameter("filter_output_vx").value)
        self.ema_vx_alpha = float(self.get_parameter("ema_vx_alpha").value)
        self.max_output_vx = float(self.get_parameter("max_output_vx").value)

        self.filter_output_vy = bool(self.get_parameter("filter_output_vy").value)
        self.ema_vy_alpha = float(self.get_parameter("ema_vy_alpha").value)
        self.max_output_vy = float(self.get_parameter("max_output_vy").value)

        self.enable_amce = bool(self.get_parameter("enable_amce").value)
        self.amce_alpha = float(self.get_parameter("amce_alpha").value)

        self.straight_cmd_vx_min = float(self.get_parameter("straight_cmd_vx_min").value)
        self.straight_cmd_wz_max = float(self.get_parameter("straight_cmd_wz_max").value)
        self.turn_cmd_wz_min = float(self.get_parameter("turn_cmd_wz_min").value)

        self.vx_gain_min = float(self.get_parameter("vx_gain_min").value)
        self.vx_gain_max = float(self.get_parameter("vx_gain_max").value)
        self.wz_gain_min = float(self.get_parameter("wz_gain_min").value)
        self.wz_gain_max = float(self.get_parameter("wz_gain_max").value)

        self.mci_x_min = float(self.get_parameter("mci_x_min").value)
        self.mci_y_min = float(self.get_parameter("mci_y_min").value)
        self.mci_yaw_min = float(self.get_parameter("mci_yaw_min").value)

        self.mci_x_sensitivity = float(self.get_parameter("mci_x_sensitivity").value)
        self.mci_y_sensitivity = float(self.get_parameter("mci_y_sensitivity").value)
        self.mci_yaw_sensitivity = float(self.get_parameter("mci_yaw_sensitivity").value)

        self.max_expected_vx_margin = float(self.get_parameter("max_expected_vx_margin").value)
        self.max_expected_vy = float(self.get_parameter("max_expected_vy").value)
        self.max_expected_wz_margin = float(self.get_parameter("max_expected_wz_margin").value)
        self.max_expected_wz_straight = float(self.get_parameter("max_expected_wz_straight").value)
        self.odom_period_alpha = float(self.get_parameter("odom_period_alpha").value)
        self.odom_gap_factor = float(self.get_parameter("odom_gap_factor").value)
        self.enable_straight_yaw_gate = bool(self.get_parameter("enable_straight_yaw_gate").value)
        self.straight_yaw_gate_wz = float(self.get_parameter("straight_yaw_gate_wz").value)
        self.publish_tf = bool(self.get_parameter("publish_tf").value)
        self.yaw_observation_gain = float(self.get_parameter("yaw_observation_gain").value)
        self.max_yaw_observation_correction = float(self.get_parameter("max_yaw_observation_correction").value)
        self.yaw_correction_mode = str(self.get_parameter("yaw_correction_mode").value)
        self.stationary_hold_enabled = bool(self.get_parameter("stationary_hold_enabled").value)
        self.stationary_cmd_vx_max = float(self.get_parameter("stationary_cmd_vx_max").value)
        self.stationary_cmd_wz_max = float(self.get_parameter("stationary_cmd_wz_max").value)
        self.stationary_raw_step_max = float(self.get_parameter("stationary_raw_step_max").value)
        self.stationary_raw_yaw_max = float(self.get_parameter("stationary_raw_yaw_max").value)
        self.stationary_zero_twist = bool(self.get_parameter("stationary_zero_twist").value)
        self.cmd_timeout_sec = float(self.get_parameter("cmd_timeout_sec").value)
        self.max_state_age_sec = float(self.get_parameter("max_state_age_sec").value)
        self.enable_amce_pose_fusion = bool(self.get_parameter("enable_amce_pose_fusion").value)
        self.straight_raw_wz_deadband = float(self.get_parameter("straight_raw_wz_deadband").value)
        self.output_frequency_hz = float(self.get_parameter("output_frequency_hz").value)
        self.straight_yaw_confirm_samples = int(self.get_parameter("straight_yaw_confirm_samples").value)
        self.straight_yaw_release_samples = int(self.get_parameter("straight_yaw_release_samples").value)
        self.odom_period_init_samples = int(self.get_parameter("odom_period_init_samples").value)
        
        # ------------------------------------------------------------
        # Raw local origin
        # ------------------------------------------------------------
        self.have_origin = False
        self.origin_x = 0.0
        self.origin_y = 0.0
        self.origin_yaw = 0.0

        # ------------------------------------------------------------
        # Previous raw local pose
        # ------------------------------------------------------------
        self.have_prev_raw = False
        self.prev_raw_x = 0.0
        self.prev_raw_y = 0.0
        self.prev_raw_yaw = 0.0

        # ------------------------------------------------------------
        # Corrected odometry state
        # ------------------------------------------------------------
        self.has_output = False
        self.last_x = 0.0
        self.last_y = 0.0
        self.last_yaw = 0.0
        self.last_time = None

        # ------------------------------------------------------------
        # Last command
        # ------------------------------------------------------------
        self.last_cmd_vx = 0.0
        self.last_cmd_wz = 0.0
        self.last_cmd_time = None

        # ------------------------------------------------------------
        # Learned G1 response model
        # ------------------------------------------------------------
        self.vx_gain_est = 0.60
        self.wz_gain_est = 0.60

        # ------------------------------------------------------------
        # Motion Consistency Indices
        # ------------------------------------------------------------
        self.mci_x = 1.0
        self.mci_y = 1.0
        self.mci_yaw = 1.0

        # ------------------------------------------------------------
        # Smoothed innovation errors
        # ------------------------------------------------------------
        self.err_vx = 0.0
        self.err_vy = 0.0
        self.err_wz = 0.0

        # ------------------------------------------------------------
        # Filtered output yaw rate and VX/VY
        # ------------------------------------------------------------
        self.wz_f = 0.0
        self.vx_f = 0.0
        self.vy_f = 0.0

        self.start_time = self.get_clock().now()

        self.tf_broadcaster = TransformBroadcaster(self)

        # ------------------------------------------------------------
        # Latest corrected navigation state for periodic publication
        # ------------------------------------------------------------
        self.state_lock = threading.Lock()
        self.latest_output_state = None

        self.prev_msg_stamp_sec = None
        self.prev_arrival_time = None
        self.nominal_odom_period = None
        self.odom_period_samples = []

        self.straight_yaw_exceed_count = 0
        self.straight_yaw_below_count = 0
        self.straight_yaw_motion_confirmed = False

        # Publish navigation odometry / TF independently from raw dog_odom timing.
        #self.output_timer_period = 1.0 / 30.0
        self.output_timer_period = (
            1.0
            / max(
                self.output_frequency_hz,
                1.0,
            )
        )
        

        self.output_timer = self.create_timer(
            self.output_timer_period,
            self.publish_navigation_state,
            callback_group=self.timer_cb_group,
        )

        # ------------------------------------------------------------
        # ROS interfaces
        # ------------------------------------------------------------
        self.pub = self.create_publisher(
            Odometry,
            self.output_odom_topic,
            make_qos(self.pub_reliable, self.qos_depth),
        )
        self.debug_pub = self.create_publisher(
            String,
            "/odom_normalizer/debug",
            10,
        )

        self.sub = self.create_subscription(
            Odometry,
            self.input_odom_topic,
            self.cb_odom,
            make_qos(self.sub_reliable, self.qos_depth),
            callback_group=self.odom_cb_group,
        )

        self.cmd_sub = self.create_subscription(
            Twist,
            self.cmd_vel_topic,
            self.cb_cmd,
            make_qos(self.cmd_reliable, self.qos_depth),
            callback_group=self.cmd_cb_group,
        )

        self.get_logger().info(
            "\n"
            "========================================\n"
            " Adaptive Odometry Calibration Framework (AOCF)\n"
            "========================================\n"
            f" input_odom_topic     : {self.input_odom_topic}\n"
            f" output_odom_topic    : {self.output_odom_topic}\n"
            f" cmd_vel_topic        : {self.cmd_vel_topic}\n"
            f" enable_amce          : {self.enable_amce}\n"
            f" mci_min x/y/yaw      : {self.mci_x_min}/{self.mci_y_min}/{self.mci_yaw_min}\n"
            f" startup_freeze_sec   : {self.startup_freeze_sec}\n"
            f" startup_ramp_sec     : {self.startup_ramp_sec}\n"
            "========================================"
        )

    # ====================================================================
    # Others functions
    # ====================================================================
    def wrap_angle(self, a: float) -> float:
        """Normalize an angle to [-pi, pi] to avoid artificial yaw jumps."""
        return math.atan2(math.sin(a), math.cos(a))

    def clamp(self, v: float, lo: float, hi: float) -> float:
        """Limit a value to a safe range."""
        return max(lo, min(hi, v))

    def odom_increment_to_body(
        self,
        dx_odom: float,
        dy_odom: float,
        heading_prev_raw: float,
    ):
        """
        Rotate a translation increment from F_O=odom into
        F_B=base_footprint.

        The previous raw heading is used because the measured increment
        belongs to the interval [k-1, k].
        """
        c = math.cos(heading_prev_raw)
        s = math.sin(heading_prev_raw)

        dx_body = c * dx_odom + s * dy_odom
        dy_body = -s * dx_odom + c * dy_odom

        return dx_body, dy_body

    def update_nominal_odom_period(self, msg_period):
        if (
            msg_period is None
            or not math.isfinite(msg_period)
            or msg_period <= 0.0
        ):
            return False

        # ------------------------------------------------------------
        # Initialization phase:
        # learn the nominal source period from several samples.
        # No gap detection yet.
        # ------------------------------------------------------------
        if self.nominal_odom_period is None:
            self.odom_period_samples.append(
                msg_period
            )

            if (
                len(self.odom_period_samples)
                < self.odom_period_init_samples
            ):
                return False

            samples = sorted(
                self.odom_period_samples
            )

            n = len(samples)

            if n % 2 == 1:
                nominal = samples[n // 2]
            else:
                nominal = 0.5 * (
                    samples[n // 2 - 1]
                    + samples[n // 2]
                )

            self.nominal_odom_period = nominal

            self.odom_period_samples.clear()

            self.get_logger().info(
                (
                    "[ODOM PERIOD] "
                    f"Nominal period initialized at "
                    f"{self.nominal_odom_period:.6f}s "
                    f"({1.0 / self.nominal_odom_period:.2f} Hz)"
                )
            )

            return False

        # ------------------------------------------------------------
        # Normal operation:
        # detect a source gap relative to learned cadence.
        # ------------------------------------------------------------
        gap_threshold = (
            self.nominal_odom_period
            * self.odom_gap_factor
        )

        source_gap = (
            msg_period > gap_threshold
        )

        # ------------------------------------------------------------
        # Adapt nominal period only from normal samples.
        # ------------------------------------------------------------
        if not source_gap:
            a = self.clamp(
                self.odom_period_alpha,
                0.0,
                1.0,
            )

            self.nominal_odom_period = (
                (1.0 - a)
                * self.nominal_odom_period
                + a
                * msg_period
            )

        return source_gap

    def publish_navigation_state(self):
        """
        Publish the latest complete corrected navigation state.

        cb_odom() creates complete immutable snapshots.
        The timer only takes a reference to the latest snapshot.
        """

        # ------------------------------------------------------------
        # Atomically obtain the latest complete state snapshot.
        # ------------------------------------------------------------
        with self.state_lock:
            if self.latest_output_state is None:
                return

            state = self.latest_output_state

        # The lock is released here.
        # cb_odom() may replace self.latest_output_state, but this local
        # 'state' reference remains valid and is never modified by cb_odom().

        latest_pose = state["pose"]
        latest_twist = state["twist"]

        latest_pose_covariance = state["pose_covariance"]
        latest_twist_covariance = state["twist_covariance"]

        latest_state_update_time_ns = state["update_time_ns"]

        now = self.get_clock().now()

        state_age = (
            now.nanoseconds
            - latest_state_update_time_ns
        ) * 1e-9

        if state_age > self.max_state_age_sec:
            self.get_logger().warn(
                (
                    f"Corrected odometry state is stale: "
                    f"age={state_age:.3f}s. "
                    f"Holding last pose."
                ),
                throttle_duration_sec=2.0,
            )

        # ------------------------------------------------------------
        # Build a completely new Odometry message.
        # Do not mutate objects stored inside the shared snapshot.
        # ------------------------------------------------------------
        out = Odometry()

        out.header.stamp = now.to_msg()
        out.header.frame_id = self.odom_frame
        out.child_frame_id = self.base_frame

        # Pose
        out.pose.pose.position.x = latest_pose.position.x
        out.pose.pose.position.y = latest_pose.position.y
        out.pose.pose.position.z = latest_pose.position.z

        out.pose.pose.orientation.x = latest_pose.orientation.x
        out.pose.pose.orientation.y = latest_pose.orientation.y
        out.pose.pose.orientation.z = latest_pose.orientation.z
        out.pose.pose.orientation.w = latest_pose.orientation.w

        # Twist
        out.twist.twist.linear.x = latest_twist.linear.x
        out.twist.twist.linear.y = latest_twist.linear.y
        out.twist.twist.linear.z = latest_twist.linear.z

        out.twist.twist.angular.x = latest_twist.angular.x
        out.twist.twist.angular.y = latest_twist.angular.y
        out.twist.twist.angular.z = latest_twist.angular.z

        # Covariances
        out.pose.covariance = list(
            latest_pose_covariance
        )

        out.twist.covariance = list(
            latest_twist_covariance
        )

        # ------------------------------------------------------------
        # Stale state:
        # hold pose but never keep publishing stale motion.
        # ------------------------------------------------------------
        if state_age > self.max_state_age_sec:
            out.twist.twist.linear.x = 0.0
            out.twist.twist.linear.y = 0.0
            out.twist.twist.linear.z = 0.0

            out.twist.twist.angular.x = 0.0
            out.twist.twist.angular.y = 0.0
            out.twist.twist.angular.z = 0.0

        self.pub.publish(out)

        if self.publish_tf:
            tf_msg = TransformStamped()

            tf_msg.header.stamp = out.header.stamp
            tf_msg.header.frame_id = self.odom_frame
            tf_msg.child_frame_id = self.base_frame

            tf_msg.transform.translation.x = (
                out.pose.pose.position.x
            )
            tf_msg.transform.translation.y = (
                out.pose.pose.position.y
            )
            tf_msg.transform.translation.z = (
                out.pose.pose.position.z
            )

            tf_msg.transform.rotation.x = (
                out.pose.pose.orientation.x
            )
            tf_msg.transform.rotation.y = (
                out.pose.pose.orientation.y
            )
            tf_msg.transform.rotation.z = (
                out.pose.pose.orientation.z
            )
            tf_msg.transform.rotation.w = (
                out.pose.pose.orientation.w
            )

            self.tf_broadcaster.sendTransform(
                tf_msg
            )
            
    def body_increment_to_odom(
        self,
        dx_body: float,
        dy_body: float,
        body_increment_to_odom: float,
    ):
        """
        Rotate a corrected translation increment from
        F_B=base_footprint back into F_O=odom.

        The previous corrected heading is used so that the corrected
        trajectory remains internally consistent.
        """
        c = math.cos(body_increment_to_odom)
        s = math.sin(body_increment_to_odom)

        dx_odom = c * dx_body - s * dy_body
        dy_odom = s * dx_body + c * dy_body

        return dx_odom, dy_odom
        
    def update_output_vx_filter(self, vx_pose: float):
        """Smooth the published forward velocity."""
        if not self.filter_output_vx:
            return self.clamp(vx_pose, -self.max_vx, self.max_vx)

        self.vx_f = (
            self.ema_vx_alpha * vx_pose
            + (1.0 - self.ema_vx_alpha) * self.vx_f
        )

        return self.clamp(self.vx_f, -self.max_output_vx, self.max_output_vx)

    def get_active_cmd(self):
        """
        Return the latest command only if it is recent enough.
        Otherwise return zero command to avoid integrating stale cmd_vel.
        """
        if self.last_cmd_time is None:
            return 0.0, 0.0, False

        age = (self.get_clock().now() - self.last_cmd_time).nanoseconds * 1e-9

        if age > self.cmd_timeout_sec:
            return 0.0, 0.0, False

        return self.last_cmd_vx, self.last_cmd_wz, True
        
    def update_output_vy_filter(self, vy_pose: float):
        """Smooth and strongly limit the published lateral velocity."""
        if not self.filter_output_vy:
            return self.clamp(vy_pose, -self.max_vy, self.max_vy)

        self.vy_f = (
            self.ema_vy_alpha * vy_pose
            + (1.0 - self.ema_vy_alpha) * self.vy_f
        )

        return self.clamp(self.vy_f, -self.max_output_vy, self.max_output_vy)
        
    def publish_debug(self, data: dict):
        """Publish the internal normalizer pipeline state for bag analysis."""
        msg = String()
        msg.data = json.dumps(data)
        self.debug_pub.publish(msg)

    def apply_stationary_hold(
        self,
        dx: float,
        dy: float,
        dyaw: float,
        cmd_vx: float,
        cmd_wz: float,
        cmd_active: bool,
    ):
        """
        Reject tiny dog_odom drift when the robot is commanded to stand still.

        This prevents AMCL/Nav2 from seeing fake motion while the G1 is physically
        standing in place.
        """
        if not self.stationary_hold_enabled:
            return dx, dy, dyaw, False

        cmd_is_zero = (
            not cmd_active
            or (
                abs(cmd_vx) <= self.stationary_cmd_vx_max
                and abs(cmd_wz) <= self.stationary_cmd_wz_max
            )
        )

        raw_step = math.hypot(dx, dy)

        raw_is_small = (
            raw_step <= self.stationary_raw_step_max
            and abs(dyaw) <= self.stationary_raw_yaw_max
        )

        if cmd_is_zero and raw_is_small:
            return 0.0, 0.0, 0.0, True

        return dx, dy, dyaw, False

    def startup_alpha(self) -> float:
        """Compute the startup freeze/ramp factor applied to odometry increments."""
        elapsed = (self.get_clock().now() - self.start_time).nanoseconds * 1e-9

        if elapsed < self.startup_freeze_sec:
            return 0.0

        ramp = max(self.startup_ramp_sec, 1e-6)

        if elapsed < self.startup_freeze_sec + ramp:
            return (elapsed - self.startup_freeze_sec) / ramp

        return 1.0

    def cb_cmd(self, msg: Twist):
        """Store the latest velocity command used as motion-consistency reference."""
        self.last_cmd_vx = float(msg.linear.x)
        self.last_cmd_wz = float(msg.angular.z)
        self.last_cmd_time = self.get_clock().now()

    def normalize_raw_pose(self, msg: Odometry):
        """Convert raw dog_odom pose into a local 2D frame starting at zero."""
        raw_x = float(msg.pose.pose.position.x)
        raw_y = float(msg.pose.pose.position.y)
        raw_yaw = yaw_from_quaternion(msg.pose.pose.orientation)

        if not self.have_origin:
            self.origin_x = raw_x
            self.origin_y = raw_y
            self.origin_yaw = raw_yaw
            self.have_origin = True

            self.get_logger().info(
                f"Odom local origin set: x={raw_x:.3f}, "
                f"y={raw_y:.3f}, yaw={raw_yaw:.3f}"
            )

            return 0.0, 0.0, 0.0

        dx = raw_x - self.origin_x
        dy = raw_y - self.origin_y

        c = math.cos(-self.origin_yaw)
        s = math.sin(-self.origin_yaw)

        local_x = c * dx - s * dy
        local_y = s * dx + c * dy
        local_yaw = self.wrap_angle(raw_yaw - self.origin_yaw)

        return local_x, local_y, local_yaw


    def compute_raw_increment(
        self,
        raw_x: float,
        raw_y: float,
        raw_yaw: float,
    ):
        """
        Compute the raw temporal increment in F_O=odom and return
        the raw heading at the beginning of the interval.
        """
        if not self.have_prev_raw:
            self.prev_raw_x = raw_x
            self.prev_raw_y = raw_y
            self.prev_raw_yaw = raw_yaw
            self.have_prev_raw = True

            return (
                False,
                0.0,
                0.0,
                0.0,
                raw_yaw,
            )

        heading_prev_raw = self.prev_raw_yaw

        dx_odom = raw_x - self.prev_raw_x
        dy_odom = raw_y - self.prev_raw_y
        dyaw = self.wrap_angle(
            raw_yaw - self.prev_raw_yaw
        )

        self.prev_raw_x = raw_x
        self.prev_raw_y = raw_y
        self.prev_raw_yaw = raw_yaw

        return (
            True,
            dx_odom,
            dy_odom,
            dyaw,
            heading_prev_raw,
        )
        
    def expected_motion(
        self,
        dt: float,
        cmd_vx: float,
        cmd_wz: float,
        cmd_active: bool,
    ):
        """
        Estimate expected body-frame motion from the validated command
        snapshot used by the current odometry callback.
        """
        if (
            not cmd_active
            or dt is None
            or dt <= 1e-4
        ):
            return (
                0.0,  # expected_dx
                0.0,  # expected_dy
                0.0,  # expected_dyaw
                0.0,  # expected_vx
                0.0,  # expected_vy
                0.0,  # expected_wz
            )

        expected_vx = (
            cmd_vx * self.vx_gain_est
        )

        expected_vy = 0.0

        is_straight_command = (
            abs(cmd_vx) >= self.straight_cmd_vx_min
            and abs(cmd_wz) <= self.straight_cmd_wz_max
        )

        if is_straight_command:
            expected_wz = 0.0
        else:
            expected_wz = (
                cmd_wz * self.wz_gain_est
            )

        expected_dx = expected_vx * dt
        expected_dy = expected_vy * dt
        expected_dyaw = expected_wz * dt

        return (
            expected_dx,
            expected_dy,
            expected_dyaw,
            expected_vx,
            expected_vy,
            expected_wz,
        )
        
    def update_response_model(
        self,
        vx_meas: float,
        wz_meas: float,
        cmd_vx: float,
        cmd_wz: float,
        cmd_active: bool,
    ):
        """
        Adapt G1 response gains only from an active, validated command.

        Measurements are expressed in F_B=base_footprint.
        """
        if not self.enable_amce:
            return

        if not cmd_active:
            return

        a = self.clamp(
            self.amce_alpha,
            0.0,
            1.0,
        )

        is_forward_reference = (
            abs(cmd_vx) >= self.straight_cmd_vx_min
        )

        if is_forward_reference:
            ratio = (
                abs(vx_meas)
                / max(abs(cmd_vx), 1e-6)
            )

            ratio = self.clamp(
                ratio,
                self.vx_gain_min,
                self.vx_gain_max,
            )

            self.vx_gain_est = (
                (1.0 - a) * self.vx_gain_est
                + a * ratio
            )

        is_turn_reference = (
            abs(cmd_wz) >= self.turn_cmd_wz_min
        )

        if is_turn_reference:
            ratio = (
                abs(wz_meas)
                / max(abs(cmd_wz), 1e-6)
            )

            ratio = self.clamp(
                ratio,
                self.wz_gain_min,
                self.wz_gain_max,
            )

            self.wz_gain_est = (
                (1.0 - a) * self.wz_gain_est
                + a * ratio
            )

    def update_mci(self, vx_meas: float, vy_meas: float, wz_meas: float,
                expected_vx: float, expected_vy: float, expected_wz: float):
        """Update Motion Consistency Indices deciding trust in raw odom vs expected motion."""
        if not self.enable_amce:
            self.mci_x = 1.0
            self.mci_y = 1.0
            self.mci_yaw = 1.0
            return

        a = self.clamp(self.amce_alpha, 0.0, 1.0)

        self.err_vx = (1.0 - a) * self.err_vx + a * abs(vx_meas - expected_vx)
        self.err_vy = (1.0 - a) * self.err_vy + a * abs(vy_meas - expected_vy)
        self.err_wz = (1.0 - a) * self.err_wz + a * abs(wz_meas - expected_wz)

        raw_mci_x = math.exp(-self.mci_x_sensitivity * self.err_vx)
        raw_mci_y = math.exp(-self.mci_y_sensitivity * self.err_vy)
        raw_mci_yaw = math.exp(-self.mci_yaw_sensitivity * self.err_wz)

        self.mci_x = self.clamp(raw_mci_x, self.mci_x_min, 1.0)
        self.mci_y = self.clamp(raw_mci_y, self.mci_y_min, 1.0)
        self.mci_yaw = self.clamp(raw_mci_yaw, self.mci_yaw_min, 1.0)


    def apply_amce(
        self,
        dx: float,
        dy: float,
        dyaw: float,
        dt: float,
        cmd_vx: float,
        cmd_wz: float,
        cmd_active: bool,
    ):
        """
        Blend measured and expected increments in F_B=base_footprint.
        """
        if (
            not self.enable_amce
            or dt is None
            or dt <= 1e-4
            or not cmd_active
        ):
            return dx, dy, dyaw

        vx_meas = dx / dt
        vy_meas = dy / dt
        wz_meas = dyaw / dt

        self.update_response_model(
            vx_meas,
            wz_meas,
            cmd_vx,
            cmd_wz,
            cmd_active,
        )

        (
            expected_dx,
            expected_dy,
            expected_dyaw,
            expected_vx,
            expected_vy,
            expected_wz,
        ) = self.expected_motion(
            dt,
            cmd_vx,
            cmd_wz,
            cmd_active,
        )

        self.update_mci(
            vx_meas,
            vy_meas,
            wz_meas,
            expected_vx,
            expected_vy,
            expected_wz,
        )

        if not self.enable_amce_pose_fusion:
           return dx, dy, dyaw

        dx = (
            self.mci_x * dx
            + (1.0 - self.mci_x) * expected_dx
        )

        dy = (
            self.mci_y * dy
            + (1.0 - self.mci_y) * expected_dy
        )

        dyaw = (
            self.mci_yaw * dyaw
            + (1.0 - self.mci_yaw)
            * expected_dyaw
        )

        return dx, dy, dyaw
        
    def limit_increment(self, dx: float, dy: float, dyaw: float):
        """Clamp abnormal position/yaw increments to reject dog_odom spikes."""
        jump_detected = False
        step = math.sqrt(dx * dx + dy * dy)

        if step > self.max_step_m:
            scale = self.max_step_m / max(step, 1e-9)
            dx *= scale
            dy *= scale
            jump_detected = True

        if abs(dyaw) > self.max_step_yaw:
            dyaw = self.clamp(dyaw, -self.max_step_yaw, self.max_step_yaw)
            jump_detected = True

        return dx, dy, dyaw, jump_detected, step

    def lowpass_increment(self, dx: float, dy: float, dyaw: float):
        """Optionally attenuate corrected increments before integration."""
        if not self.enable_lowpass:
            return dx, dy, dyaw

        a_pos = self.clamp(self.position_alpha, 0.0, 1.0)
        a_yaw = self.clamp(self.yaw_alpha, 0.0, 1.0)

        return a_pos * dx, a_pos * dy, a_yaw * dyaw


    def apply_straight_yaw_gate_old(
        self,
        dyaw: float,
        dt: float,
        cmd_vx: float,
        cmd_wz: float,
        cmd_active: bool,
    ):
        """
        Reject parasitic yaw accumulation during commanded straight walking.

        STRAIGHT:
            - yaw below measured noise threshold is rejected;
            - isolated excursions above the threshold are rejected;
            - persistent excursions are accepted only after temporal confirmation.

        TURN / IDLE:
            - the straight-motion gate is reset;
            - yaw is allowed to follow the measured odometry.
        """
        if not self.enable_straight_yaw_gate:
            return dyaw

        if dt is None or dt <= 1e-4:
            return dyaw

        is_straight_motion = (
            cmd_active
            and abs(cmd_vx) >= self.straight_cmd_vx_min
            and abs(cmd_wz) <= self.straight_cmd_wz_max
        )

        # ------------------------------------------------------------
        # TURN or IDLE:
        # do not apply the straight-motion yaw filter.
        # ------------------------------------------------------------
        if not is_straight_motion:
            self.straight_yaw_exceed_count = 0
            self.straight_yaw_below_count = 0
            self.straight_yaw_motion_confirmed = False

            return dyaw

        # ------------------------------------------------------------
        # STRAIGHT mode.
        # ------------------------------------------------------------
        measured_wz = dyaw / dt

        # ------------------------------------------------------------
        # Below the experimentally measured parasitic yaw rate.
        # ------------------------------------------------------------
        if abs(measured_wz) <= self.straight_yaw_gate_wz:
            self.straight_yaw_exceed_count = 0

            if self.straight_yaw_motion_confirmed:
                self.straight_yaw_below_count += 1

                # Use a small temporal release hysteresis so that one sample
                # below the threshold does not immediately reset a real rotation.
                if (
                    self.straight_yaw_below_count
                    < self.straight_yaw_release_samples
                ):
                    return dyaw

            self.straight_yaw_below_count = 0
            self.straight_yaw_motion_confirmed = False

            return 0.0

        # ------------------------------------------------------------
        # Above the noise threshold.
        # ------------------------------------------------------------
        self.straight_yaw_below_count = 0

        if not self.straight_yaw_motion_confirmed:
            self.straight_yaw_exceed_count += 1

            if (
                self.straight_yaw_exceed_count
                < self.straight_yaw_confirm_samples
            ):
                # Do not integrate isolated or short-lived excursions.
                return 0.0

            # Persistent yaw motion has now been confirmed.
            self.straight_yaw_motion_confirmed = True

        return dyaw

    def apply_straight_yaw_gate(
        self,
        dyaw: float,
        raw_yaw: float,
        prev_yaw: float,
        dt: float,
        cmd_vx: float,
        cmd_wz: float,
        cmd_active: bool,
    ):
        """
        Manage yaw drift without re-injecting accumulated raw-yaw divergence.

        STRAIGHT:
            - corrected yaw is locked;
            - raw->corrected yaw offset is continuously updated.

        TURN:
            - raw yaw is first realigned into the corrected-yaw frame;
            - only the new physical rotation is integrated;
            - accumulated raw-yaw drift is not suddenly reintroduced.
        """

        if not self.enable_straight_yaw_gate:
            return dyaw

        if dt is None or dt <= 1e-4:
            return dyaw

        # ------------------------------------------------------------
        # Initialize alignment once.
        # ------------------------------------------------------------
        if not self.yaw_alignment_initialized:
            self.yaw_raw_to_corrected_offset = self.wrap_angle(
                prev_yaw - raw_yaw
            )
            self.yaw_alignment_initialized = True

        is_forward_motion = (
            cmd_active
            and abs(cmd_vx) >= self.straight_cmd_vx_min
        )

        is_negligible_turn_command = (
            abs(cmd_wz) <= self.straight_cmd_wz_max
        )

        # ------------------------------------------------------------
        # STRAIGHT:
        # Keep corrected yaw fixed and continuously absorb raw-yaw drift
        # into the alignment offset.
        # ------------------------------------------------------------
        if (
            is_forward_motion
            and is_negligible_turn_command
        ):
            self.yaw_raw_to_corrected_offset = self.wrap_angle(
                prev_yaw - raw_yaw
            )

            return 0.0

        # ------------------------------------------------------------
        # TURN:
        # Realign raw yaw into the corrected-yaw frame.
        # ------------------------------------------------------------
        aligned_raw_yaw = self.wrap_angle(
            raw_yaw
            + self.yaw_raw_to_corrected_offset
        )

        # Compute only the yaw increment relative to the current corrected yaw.
        aligned_dyaw = self.wrap_angle(
            aligned_raw_yaw
            - prev_yaw
        )

        # ------------------------------------------------------------
        # Remove the measured parasitic yaw-rate component.
        # ------------------------------------------------------------
        measured_wz = aligned_dyaw / dt

        parasitic_wz = abs(
            self.straight_yaw_gate_wz
        )

        if abs(measured_wz) <= parasitic_wz:
            corrected_wz = 0.0

        else:
            corrected_wz = (
                measured_wz
                - math.copysign(
                    parasitic_wz,
                    measured_wz,
                )
            )

        return corrected_wz * dt
        
    def update_output_wz_filter(self, wz_pose: float):
        """Smooth only the published angular velocity, not the integrated pose yaw."""
        if not self.filter_output_wz:
            return self.clamp(wz_pose, -self.max_wz, self.max_wz)

        self.wz_f = (
            self.ema_wz_alpha * wz_pose
            + (1.0 - self.ema_wz_alpha) * self.wz_f
        )

        return self.clamp(
            self.wz_f,
            -self.max_output_wz,
            self.max_output_wz,
        )    
    # ********************************************************************

    # ====================================================================
    # Comment cb_odom
    # ====================================================================

    def cb_odom(self, msg: Odometry):
        # 1) Reject invalid raw orientation before using dog_odom.
        if not valid_quaternion(msg.pose.pose.orientation):
            self.get_logger().warn("Dropping dog_odom: invalid quaternion")
            return

        # 2) Select output timestamp policy.
        # T_arrival: local time when this dog_odom message enters cb_odom().
        arrival_time = self.get_clock().now()
        callback_time = arrival_time 
        msg_stamp_sec = (
            float(msg.header.stamp.sec)
            + float(msg.header.stamp.nanosec) * 1e-9
        )
        arrival_sec = arrival_time.nanoseconds * 1e-9
        msg_age = arrival_sec - msg_stamp_sec
        msg_period = None
        if self.prev_msg_stamp_sec is not None:
            msg_period = (
                msg_stamp_sec
                - self.prev_msg_stamp_sec
            )
        
        source_gap = self.update_nominal_odom_period(
            msg_period
        )

        arrival_period = None
        if self.prev_arrival_time is not None:
            arrival_period = (
                arrival_time
                - self.prev_arrival_time
            ).nanoseconds * 1e-9

        self.prev_msg_stamp_sec = msg_stamp_sec
        self.prev_arrival_time = arrival_time

        if (
            arrival_period is not None
            and arrival_period > 0.1
        ):
            self.get_logger().warn(
                (
                    f"[ODOM CALLBACK GAP] "
                    f"dt_arrival={arrival_period:.4f}s | "
                    f"dt_msg="
                    f"{msg_period if msg_period is not None else float('nan'):.4f}s"
                )
            )
        # 3) Convert raw dog_odom pose into the local normalized odom frame.
        raw_x, raw_y, raw_yaw = self.normalize_raw_pose(msg)

        if not all(math.isfinite(v) for v in [raw_x, raw_y, raw_yaw]):
            self.get_logger().warn("Dropping dog_odom: invalid normalized pose")
            return

        # 4) Compute time delta and raw incremental motion.
        dt = None
        dt_arrival = None

        if self.last_time is not None:
            dt_arrival = (
                callback_time - self.last_time
            ).nanoseconds * 1e-9


        # Prefer the physical sampling interval carried by dog_odom.
        # Its absolute timestamp is shifted, but its relative period was
        # measured to be stable around 51 Hz.
        if (
            msg_period is not None
            and math.isfinite(msg_period)
            and msg_period > 0.0
        ):
            dt = msg_period
        else:
            dt = dt_arrival

        if source_gap:
            self.get_logger().warn(
                (
                    "[ODOM SOURCE GAP] "
                    f"dt_msg={msg_period:.4f}s | "
                    f"nominal={self.nominal_odom_period:.4f}s. "
                    "Resynchronizing raw reference."
                )
            )

            # Resynchronize raw odometry baseline.
            self.prev_raw_x = raw_x
            self.prev_raw_y = raw_y
            self.prev_raw_yaw = raw_yaw
            self.have_prev_raw = True

            # Reset temporal yaw validation because continuity was lost.
            self.straight_yaw_exceed_count = 0
            self.straight_yaw_below_count = 0
            self.straight_yaw_motion_confirmed = False

            self.yaw_alignment_initialized = False
            self.yaw_raw_to_corrected_offset = 0.0

            # Do not keep velocity-filter memory across a discontinuity.
            self.vx_f = 0.0
            self.vy_f = 0.0
            self.wz_f = 0.0

            self.last_time = callback_time

            return
    
        (
            has_increment,
            dx_odom_raw,
            dy_odom_raw,
            dyaw_raw,
            heading_prev_raw,
        ) = self.compute_raw_increment(
            raw_x,
            raw_y,
            raw_yaw,
        )

        dx_body_raw = 0.0
        dy_body_raw = 0.0

        if has_increment:
            (
                dx_body_raw,
                dy_body_raw,
            ) = self.odom_increment_to_body(
                dx_odom_raw,
                dy_odom_raw,
                heading_prev_raw,
            )         

        cmd_vx, cmd_wz, cmd_active = self.get_active_cmd()

        # 5) Keep previous corrected pose for twist reconstruction.
        prev_x = self.last_x
        prev_y = self.last_y
        prev_yaw = self.last_yaw

        # 6) Initialize debug variables for the full correction pipeline.
        raw_dx = raw_dy = raw_dyaw = None
        vx_meas = vy_meas = wz_meas = None

        expected_dx = expected_dy = expected_dyaw = None
        expected_vx = expected_vy = expected_wz = None

        amce_dx = amce_dy = amce_dyaw = None
        limited_dx = limited_dy = limited_dyaw = None
        lowpass_dx = lowpass_dy = lowpass_dyaw = None
        ramped_dx = ramped_dy = ramped_dyaw = None

        jump_detected = False
        step = 0.0

        # Final corrected increment expressed in F_B = base_footprint.
        # These values will be used directly to publish the twist.
        final_dx_body = 0.0
        final_dy_body = 0.0
        final_dyaw_body = 0.0

        # 7) Compute startup freeze/ramp factor.
        alpha_startup = self.startup_alpha()

        # 8) Create debug dictionary used later by the bag analyzer.

        dbg = {
            # Timing and command snapshot
            "dt": dt,
            "has_increment": has_increment,
            "cmd_vx": cmd_vx,
            "cmd_wz": cmd_wz,
            "cmd_active": cmd_active,

            # Raw increment in F_O = odom
            "raw_dx_odom": None,
            "raw_dy_odom": None,
            "raw_dyaw": None,

            # Same raw increment in F_B = base_footprint
            "raw_dx_body": None,
            "raw_dy_body": None,
            "raw_vx_body": None,
            "raw_vy_body": None,
            "raw_wz": None,

            # Expected motion in F_B
            "expected_dx_body": None,
            "expected_dy_body": None,
            "expected_dyaw": None,
            "expected_vx_body": None,
            "expected_vy_body": None,
            "expected_wz": None,

            # AMCE output in F_B
            "mci_x": self.mci_x,
            "mci_y": self.mci_y,
            "mci_yaw": self.mci_yaw,
            "amce_dx_body": None,
            "amce_dy_body": None,
            "amce_dyaw": None,

            # Limiter output in F_B
            "limited_dx_body": None,
            "limited_dy_body": None,
            "limited_dyaw": None,
            "jump_detected": False,
            "jump_step": None,

            # Stationary hold
            "stationary_hold_enabled": (
                self.stationary_hold_enabled
            ),
            "stationary_hold_active": False,

            # Low-pass and heading gate in F_B
            "lowpass_dx_body": None,
            "lowpass_dy_body": None,
            "lowpass_dyaw": None,
            "dyaw_before_straight_gate": None,
            "dyaw_after_straight_gate": None,
            "straight_yaw_gate_active": False,

            # Startup ramp in F_B
            "startup_alpha": alpha_startup,
            "startup_ramp": None,
            "ramped_dx_body": None,
            "ramped_dy_body": None,
            "ramped_dyaw": None,

            # Corrected increment and pose in F_O
            "corrected_dx_odom": None,
            "corrected_dy_odom": None,
            "corrected_x": None,
            "corrected_y": None,
            "corrected_yaw": None,

            # Published twist in F_B
            "published_vx": None,
            "published_vy": None,
            "published_wz": None,
        }

        # 9) Store the same raw motion in both useful frames.
        if has_increment:
            # Raw temporal increment in F_O = odom.
            dbg["raw_dx_odom"] = dx_odom_raw
            dbg["raw_dy_odom"] = dy_odom_raw
            dbg["raw_dyaw"] = dyaw_raw

            # Same physical increment expressed in F_B = base_footprint.
            dbg["raw_dx_body"] = dx_body_raw
            dbg["raw_dy_body"] = dy_body_raw

            if dt is not None and dt > 1e-4:
                dbg["raw_vx_body"] = dx_body_raw / dt
                dbg["raw_vy_body"] = dy_body_raw / dt
                dbg["raw_wz"] = dyaw_raw / dt
        
        
        # 10) Compute expected motion from current command.
        
        if dt is not None and dt > 1e-4:
            (
                expected_dx,
                expected_dy,
                expected_dyaw,
                expected_vx,
                expected_vy,
                expected_wz,
            ) = self.expected_motion(
                dt,
                cmd_vx,
                cmd_wz,
                cmd_active,
            )

        else:
            expected_dx = expected_dy = expected_dyaw = None
            expected_vx = expected_vy = expected_wz = None

        # 11) During startup freeze or first sample, publish zero pose/twist.
        if alpha_startup <= 0.0 or not has_increment:
            corrected_x = 0.0
            corrected_y = 0.0
            corrected_yaw = 0.0
            dt_for_twist = None

        else:
            # 12) Store expected motion in debug once the correction pipeline is active.
            if dt is not None and dt > 1e-4:

                dbg["expected_dx_body"] = expected_dx
                dbg["expected_dy_body"] = expected_dy
                dbg["expected_dyaw"] = expected_dyaw

                dbg["expected_vx_body"] = expected_vx
                dbg["expected_vy_body"] = expected_vy
                dbg["expected_wz"] = expected_wz

            # 13) Blend raw dog_odom increment with expected motion.
            #     AMCE operates in F_B = base_footprint.
            dx_body, dy_body, dyaw_body = self.apply_amce(
                dx_body_raw,
                dy_body_raw,
                dyaw_raw,
                dt,
                cmd_vx,
                cmd_wz,
                cmd_active,
            )

            dbg["amce_dx_body"] = dx_body
            dbg["amce_dy_body"] = dy_body
            dbg["amce_dyaw"] = dyaw_body

            # MCI values must be stored after apply_amce(), because apply_amce()
            # updates them.
            dbg["mci_x"] = self.mci_x
            dbg["mci_y"] = self.mci_y
            dbg["mci_yaw"] = self.mci_yaw

            # 14) Hard limiter, still in F_B.
            (
                dx_body,
                dy_body,
                dyaw_body,
                jump_detected,
                step,
            ) = self.limit_increment(
                dx_body,
                dy_body,
                dyaw_body,
            )

            dbg["limited_dx_body"] = dx_body
            dbg["limited_dy_body"] = dy_body
            dbg["limited_dyaw"] = dyaw_body
            dbg["jump_detected"] = jump_detected
            dbg["jump_step"] = step

            # Stationary hold, still in F_B.
            (
                dx_body,
                dy_body,
                dyaw_body,
                stationary_hold_active,
            ) = self.apply_stationary_hold(
                dx_body,
                dy_body,
                dyaw_body,
                cmd_vx,
                cmd_wz,
                cmd_active,
            )

            dbg["stationary_hold_enabled"] = self.stationary_hold_enabled
            dbg["stationary_hold_active"] = stationary_hold_active
            dbg["stationary_raw_step_max"] = self.stationary_raw_step_max
            dbg["stationary_raw_yaw_max"] = self.stationary_raw_yaw_max


            # 15) Keep pose increments unchanged.
            # Smoothing is applied later only to the published twist.
            dbg["post_filter_dx_body"] = dx_body
            dbg["post_filter_dy_body"] = dy_body
            dbg["post_filter_dyaw"] = dyaw_body

            # 15b) Straight-yaw gate.
            dyaw_before_straight_gate = dyaw_body

            dyaw_body = self.apply_straight_yaw_gate(
            dyaw_body,
            raw_yaw,
            prev_yaw,
            dt,
            cmd_vx,
            cmd_wz,
            cmd_active,
        )

            straight_yaw_gate_active = (
                self.enable_straight_yaw_gate
                and cmd_active
                and dt is not None
                and dt > 1e-4
                and abs(cmd_vx) >= self.straight_cmd_vx_min
                and abs(cmd_wz) <= self.straight_cmd_wz_max
            )

            dbg["straight_yaw_gate_enabled"] = (
                self.enable_straight_yaw_gate
            )
            dbg["straight_yaw_gate_active"] = (
                straight_yaw_gate_active
            )
            dbg["straight_yaw_gate_wz"] = (
                self.straight_yaw_gate_wz
            )
            dbg["dyaw_before_straight_gate"] = (
                dyaw_before_straight_gate
            )
            dbg["dyaw_after_straight_gate"] = (
                dyaw_body
            )

            # 16) Startup ramp, still in F_B.
            ramp = self.clamp(alpha_startup, 0.0, 1.0)

            dx_body *= ramp
            dy_body *= ramp
            dyaw_body *= ramp

            dbg["startup_ramp"] = ramp
            dbg["ramped_dx_body"] = dx_body
            dbg["ramped_dy_body"] = dy_body
            dbg["ramped_dyaw"] = dyaw_body

            # Final corrected body-frame increment used by ATC/twist publication.
            # Final body-frame increment after all AOCF corrections.
            # This increment is integrated into pose and also used
            # directly to reconstruct the published body-frame twist.
            final_dx_body = dx_body
            final_dy_body = dy_body
            final_dyaw_body = dyaw_body

            # 17) Integrate corrected increment into normalized odometry pose.
            # Convert corrected body increment back into F_O = odom.
            (
                dx_odom_corrected,
                dy_odom_corrected,
            ) = self.body_increment_to_odom(
                dx_body,
                dy_body,
                prev_yaw,
            )

            corrected_x = prev_x + dx_odom_corrected
            corrected_y = prev_y + dy_odom_corrected
            # ------------------------------------------------------------
            # Yaw prediction/correction model
            #
            # Translation remains incremental, but yaw is not obtained by
            # blindly accumulating raw dog_odom yaw increments.
            #
            # We first predict yaw from the command-consistent increment,
            # then use dog_odom yaw only as a weak bounded observation.
            # ------------------------------------------------------------
            if self.yaw_correction_mode == "predicted_cmd":
                predicted_yaw = self.wrap_angle(
                    prev_yaw + expected_dyaw
                )

                yaw_error = self.wrap_angle(
                    raw_yaw - predicted_yaw
                )

                yaw_correction = self.clamp(
                    self.yaw_observation_gain * yaw_error,
                    -self.max_yaw_observation_correction,
                    self.max_yaw_observation_correction,
                )

                corrected_yaw = self.wrap_angle(
                    predicted_yaw + yaw_correction
                )
            
            else:
                corrected_yaw = self.wrap_angle(
                    prev_yaw + dyaw_body
                )

            dbg["corrected_dx_odom"] = dx_odom_corrected
            dbg["corrected_dy_odom"] = dy_odom_corrected

            dbg["corrected_x"] = corrected_x
            dbg["corrected_y"] = corrected_y
            dbg["corrected_yaw"] = corrected_yaw

            dt_for_twist = dt

            if jump_detected:
                self.get_logger().warn(
                    f"dog_odom increment limited: step={step:.3f}, "
                    f"dyaw={math.degrees(dyaw_body):+.2f} deg"
                )

        # 18) Build output odometry message from corrected pose.
        out = copy.deepcopy(msg)
        out.header.stamp = callback_time.to_msg()
        out.header.frame_id = self.odom_frame
        out.child_frame_id = self.base_frame

        out.pose.pose.position.x = float(corrected_x)
        out.pose.pose.position.y = float(corrected_y)

        # 19) Force Nav2-compatible 2D pose.
        if self.force_2d and self.zero_z:
            out.pose.pose.position.z = 0.0

        if self.force_2d and self.zero_roll_pitch:
            qx, qy, qz, qw = quaternion_from_yaw(corrected_yaw)
            out.pose.pose.orientation.x = qx
            out.pose.pose.orientation.y = qy
            out.pose.pose.orientation.z = qz
            out.pose.pose.orientation.w = qw

        # 20) Force Nav2-compatible 2D twist.
        if self.force_2d:
            out.twist.twist.linear.z = 0.0
            out.twist.twist.angular.x = 0.0
            out.twist.twist.angular.y = 0.0

        # 21) Publish twist directly from the final corrected increment
        # expressed in F_B = base_footprint.
        if (
            self.compute_twist_from_corrected_pose
            and self.has_output
            and dt_for_twist is not None
            and dt_for_twist > 1e-4
        ):
            vx_body = final_dx_body / dt_for_twist
            vy_body = final_dy_body / dt_for_twist
            wz_body = final_dyaw_body / dt_for_twist

            out.twist.twist.linear.x = (
                self.update_output_vx_filter(vx_body)
            )

            out.twist.twist.linear.y = (
                self.update_output_vy_filter(vy_body)
            )

            out.twist.twist.angular.z = (
                self.update_output_wz_filter(wz_body)
            )

            dbg["published_vx"] = out.twist.twist.linear.x
            dbg["published_vy"] = out.twist.twist.linear.y
            dbg["published_wz"] = out.twist.twist.angular.z
    
        else:
            # First valid output or startup freeze: publish zero twist.
            out.twist.twist.linear.x = 0.0
            out.twist.twist.linear.y = 0.0
            out.twist.twist.angular.z = 0.0

            dbg["published_vx"] = 0.0
            dbg["published_vy"] = 0.0
            dbg["published_wz"] = 0.0
    
        if self.stationary_zero_twist and dbg.get("stationary_hold_active", False):
            out.twist.twist.linear.x = 0.0
            out.twist.twist.linear.y = 0.0
            out.twist.twist.angular.z = 0.0

            dbg["published_vx"] = 0.0
            dbg["published_vy"] = 0.0
            dbg["published_wz"] = 0.0
    
        # 22) Set 2D navigation covariance.
        out.pose.covariance = [0.0] * 36
        out.pose.covariance[0] = 0.05
        out.pose.covariance[7] = 0.05
        out.pose.covariance[14] = 999.0
        out.pose.covariance[21] = 999.0
        out.pose.covariance[28] = 999.0
        out.pose.covariance[35] = 0.10

        out.twist.covariance = [0.0] * 36
        out.twist.covariance[0] = 0.10
        out.twist.covariance[7] = 0.10
        out.twist.covariance[14] = 999.0
        out.twist.covariance[21] = 999.0
        out.twist.covariance[28] = 999.0
        out.twist.covariance[35] = 0.20

        state_ready_time = self.get_clock().now()

        processing_time = (
            state_ready_time - arrival_time
        ).nanoseconds * 1e-9

        # 23) Publish debug and corrected odometry.
        #self.publish_debug(dbg)

        # Store the latest corrected navigation state.
        # ------------------------------------------------------------
        # Build a complete navigation-state snapshot locally.
        # ------------------------------------------------------------

        new_state = {
            "pose": copy.deepcopy(
                out.pose.pose
            ),
            "twist": copy.deepcopy(
                out.twist.twist
            ),
            "pose_covariance": list(
                out.pose.covariance
            ),
            "twist_covariance": list(
                out.twist.covariance
            ),
            "update_time_ns": state_ready_time.nanoseconds,
        }

        # ------------------------------------------------------------
        # Atomically replace the state shared with the publication timer.
        # ------------------------------------------------------------
        with self.state_lock:
            self.latest_output_state = new_state

        if processing_time > 0.05:
            self.get_logger().warn(
                (
                    f"[AOCF SLOW] "
                    f"processing={processing_time:.4f}s"
                )
            )

        # 24) Save state for the next odometry increment.
        self.last_x = corrected_x
        self.last_y = corrected_y
        self.last_yaw = corrected_yaw
        self.last_time = callback_time
        self.has_output = True
        

def main(args=None):
    rclpy.init(args=args)

    node = OdomNormalizer()

    executor = MultiThreadedExecutor(
        num_threads=3
    )

    executor.add_node(node)

    try:
        executor.spin()

    except KeyboardInterrupt:
        pass

    finally:
        executor.shutdown()
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()