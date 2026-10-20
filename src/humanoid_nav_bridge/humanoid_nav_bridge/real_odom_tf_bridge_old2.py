#!/usr/bin/env python3
"""
real_odom_tf_bridge.py (ROS 2 Humble)

Production bridge for a real robot:
- Subscribe to /dog_odom (nav_msgs/Odometry)
- Publish /odom (nav_msgs/Odometry) for Nav2
- Publish dynamic TF: odom -> base_frame

✅ Fixes included (Unitree/G1 case)
1) Non-REP103 odom fix:
   The driver publishes yaw rate in twist.angular.y, so we remap it to twist.angular.z.
2) 2D planar Nav2 enforcement:
   Force linear.y=0, linear.z=0, angular.x=0, angular.y=0.
3) Robust option (recommended):
   Rebuild a yaw-only orientation by integrating corrected angular.z:
       yaw := ∫ wz dt
   This avoids "weird" quaternions coming from drivers with different axis conventions.

ROS Parameters
--------------
input_odom_topic (str)   default: /dog_odom
output_odom_topic (str)  default: /odom

odom_frame (str)         default: odom
base_frame (str)         default: robot_center

use_node_time (bool)     default: True
force_child_frame (bool) default: True

zero_at_start (bool)       default: True   # Reset (x,y,yaw) at startup (recommended for Nav2)
planarize_twist (bool)     default: True   # Force vy/vz/wx/wy = 0
kill_linear_y (bool)       default: True   # Set linear.y=0 (stability)
remap_yawrate_y_to_z (bool) default: True  # angular.y -> angular.z

integrate_yaw (bool)        default: True  # (recommended) yaw := ∫ wz dt (after remap)
yaw_init_from_input (bool)  default: False # If True, initialize yaw from driver quaternion (often wrong)

publish_tf (bool)         default: True
publish_odom (bool)       default: True

tf_rate_hz (float)        default: 50.0
odom_rate_hz (float)      default: 50.0
timeout_sec (float)       default: 1.0

QoS
---
sub_reliable (bool) default: True
pub_reliable (bool) default: True
qos_depth (int)     default: 10

Author: Jean Chrysostome Mayoko Biong 
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
import tf2_ros
from geometry_msgs.msg import Twist, TwistWithCovariance


def _make_qos(reliable: bool, depth: int) -> QoSProfile:
    """Helper to build a QoS profile that works on most real-robot DDS setups."""
    return QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE if reliable else ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
        depth=depth,
    )


def _clamp_angle_pi(a: float) -> float:
    """Normalize an angle to [-pi, pi]."""
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def _yaw_from_quat(x: float, y: float, z: float, w: float) -> float:
    """Compute yaw from a quaternion under the REP-103 convention (Z-up)."""
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def _quat_from_yaw(yaw: float) -> Tuple[float, float, float, float]:
    """Build a yaw-only quaternion (roll=pitch=0)."""
    half = 0.5 * yaw
    return (0.0, 0.0, math.sin(half), math.cos(half))


class RealOdomTfBridge(Node):
    """
    Bridge a robot Odometry stream into:
      - a standardized /odom topic
      - a stable TF transform: odom_frame -> base_frame

    The node can "repair" odometry that does not follow REP-103:
    - remap yaw rate from angular.y to angular.z
    - force planar motion for Nav2
    - optionally rebuild yaw-only orientation by integrating yaw rate
    """

    def __init__(self) -> None:
        super().__init__("real_odom_tf_bridge")

        # ----------------------------
        # Parameters: topics + frames
        # ----------------------------
        self.input_odom_topic = self.declare_parameter("input_odom_topic", "/dog_odom").value
        self.output_odom_topic = self.declare_parameter("output_odom_topic", "/odom").value

        self.odom_frame = self.declare_parameter("odom_frame", "odom").value
        self.base_frame = self.declare_parameter("base_frame", "robot_center").value

        # ----------------------------
        # Timestamping policy
        # ----------------------------
        # On real robots, using node time for TF/odom is usually more stable than
        # trusting driver timestamps (which can drift or jump).
        self.use_node_time = bool(self.declare_parameter("use_node_time", True).value)

        # ----------------------------
        # Frame enforcement
        # ----------------------------
        # If True, ignore msg.child_frame_id and always publish base_frame.
        self.force_child_frame = bool(self.declare_parameter("force_child_frame", True).value)

        # ----------------------------
        # Start at origin
        # ----------------------------
        # If True, subtract the initial (x,y,yaw) so Nav2 starts near (0,0,0).
        self.zero_at_start = bool(self.declare_parameter("zero_at_start", True).value)

        # ----------------------------
        # Twist correction knobs
        # ----------------------------
        self.planarize_twist = bool(self.declare_parameter("planarize_twist", True).value)
        self.kill_linear_y = bool(self.declare_parameter("kill_linear_y", True).value)
        self.remap_yawrate_y_to_z = bool(self.declare_parameter("remap_yawrate_y_to_z", True).value)

        # ----------------------------
        # Orientation reconstruction (robust)
        # ----------------------------
        self.integrate_yaw = bool(self.declare_parameter("integrate_yaw", True).value)
        self.yaw_init_from_input = bool(self.declare_parameter("yaw_init_from_input", False).value)

        # ----------------------------
        # Publish toggles
        # ----------------------------
        self.publish_tf = bool(self.declare_parameter("publish_tf", True).value)
        self.publish_odom = bool(self.declare_parameter("publish_odom", True).value)

        # ----------------------------
        # Rates + watchdog timeout
        # ----------------------------
        self.tf_rate_hz = float(self.declare_parameter("tf_rate_hz", 50.0).value)
        self.odom_rate_hz = float(self.declare_parameter("odom_rate_hz", 50.0).value)
        self.timeout_sec = float(self.declare_parameter("timeout_sec", 1.0).value)

        # ----------------------------
        # QoS knobs
        # ----------------------------
        self.sub_reliable = bool(self.declare_parameter("sub_reliable", True).value)
        self.pub_reliable = bool(self.declare_parameter("pub_reliable", True).value)
        self.qos_depth = int(self.declare_parameter("qos_depth", 10).value)

        sub_qos = _make_qos(self.sub_reliable, self.qos_depth)
        pub_qos = _make_qos(self.pub_reliable, self.qos_depth)

        # ----------------------------
        # ROS interfaces
        # ----------------------------
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.odom_pub = self.create_publisher(Odometry, self.output_odom_topic, pub_qos)
        self.sub = self.create_subscription(Odometry, self.input_odom_topic, self._cb, sub_qos)

        # ----------------------------
        # State
        # ----------------------------
        self._last_msg: Optional[Odometry] = None
        self._last_rx_time: Optional[Time] = None

        # Zero-at-start offsets (latched on first odom)
        self._zero_ready = False
        self._x0 = 0.0
        self._y0 = 0.0
        self._yaw0 = 0.0

        # Yaw integration state
        self._yaw_est = 0.0
        self._yaw_inited = False
        self._last_yaw_time: Optional[Time] = None

        # ----------------------------
        # Timers
        # ----------------------------
        if self.publish_tf and self.tf_rate_hz > 0.0:
            self._tf_timer = self.create_timer(1.0 / self.tf_rate_hz, self._publish_tf_tick)
        else:
            self._tf_timer = None

        if self.publish_odom and self.odom_rate_hz > 0.0:
            self._odom_timer = self.create_timer(1.0 / self.odom_rate_hz, self._publish_odom_tick)
        else:
            self._odom_timer = None

        self._watchdog_timer = self.create_timer(0.5, self._watchdog_tick)

        # ----------------------------------------------------
        # NEW: Planar SE(2) projected state (Publication model)
        # ----------------------------------------------------
        # Instead of trusting driver pose (floating base),
        # we reconstruct a consistent planar state by integrating
        # the horizontal projection of the COM velocity.
        #
        # This is a rigorous SE(3) -> SE(2) projection.
        #
        # Previous approach (kept for traceability):
        #   x = msg.pose.pose.position.x
        #   y = msg.pose.pose.position.y
        #
        # We DO NOT remove that code from file,
        # but we do not use it anymore.
        # ----------------------------------------------------

        self._x_est = 0.0
        self._y_est = 0.0
        self._yaw_est = 0.0
        self._last_integration_time = self.get_clock().now()

        self._last_filtered = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        # ----------------------------
        # Startup log
        # ----------------------------
        self.get_logger().info(
            "RealOdomTfBridge started:\n"
            f"  input_odom_topic={self.input_odom_topic}\n"
            f"  output_odom_topic={self.output_odom_topic}\n"
            f"  frames: odom={self.odom_frame}, base={self.base_frame}\n"
            f"  use_node_time={self.use_node_time}\n"
            f"  force_child_frame={self.force_child_frame}\n"
            f"  zero_at_start={self.zero_at_start}\n"
            f"  remap_yawrate_y_to_z={self.remap_yawrate_y_to_z}\n"
            f"  planarize_twist={self.planarize_twist}, kill_linear_y={self.kill_linear_y}\n"
            f"  integrate_yaw={self.integrate_yaw}, yaw_init_from_input={self.yaw_init_from_input}\n"
            f"  publish_tf={self.publish_tf} @ {self.tf_rate_hz} Hz\n"
            f"  publish_odom={self.publish_odom} @ {self.odom_rate_hz} Hz\n"
            f"  qos: sub_reliable={self.sub_reliable}, pub_reliable={self.pub_reliable}, depth={self.qos_depth}\n"
        )

        # ----------------------------
        # Humanoid RL filtering
        # ----------------------------
        self.enable_filter = bool(self.declare_parameter("enable_filter", True).value)
        self.deadband_vx = float(self.declare_parameter("deadband_vx", 0.02).value)
        self.deadband_wz = float(self.declare_parameter("deadband_wz", 0.02).value)
        self.ema_alpha = float(self.declare_parameter("ema_alpha", 0.2).value)
        self.freeze_after_sec = float(self.declare_parameter("freeze_after_sec", 0.5).value)

        # Filter state
        self._vx_f = 0.0
        self._wz_f = 0.0
        self._last_motion_time = self.get_clock().now()


    # ----------------------------
    # Subscriber callback
    # ----------------------------
    def _cb(self, msg: Odometry) -> None:
        """Store the latest odometry message and initialize offsets/estimators when needed."""
        self._last_msg = msg
        self._last_rx_time = self.get_clock().now()

        # Latch zero-at-start offsets on the first message
        if self.zero_at_start and not self._zero_ready:
            self._x0 = float(msg.pose.pose.position.x)
            self._y0 = float(msg.pose.pose.position.y)

            # We only use yaw from the input quaternion as an initial offset (not as ground truth).
            q = msg.pose.pose.orientation
            self._yaw0 = _yaw_from_quat(float(q.x), float(q.y), float(q.z), float(q.w))

            self._zero_ready = True
            self.get_logger().info(
                f"zero_at_start locked: x0={self._x0:.3f}, y0={self._y0:.3f}, yaw0={self._yaw0:.3f} rad"
            )

        # Initialize yaw integrator once
        if self.integrate_yaw and (not self._yaw_inited):
            if self.yaw_init_from_input:
                q = msg.pose.pose.orientation
                self._yaw_est = _yaw_from_quat(float(q.x), float(q.y), float(q.z), float(q.w))
            else:
                self._yaw_est = 0.0

            self._yaw_inited = True
            self._last_yaw_time = self.get_clock().now()

            self.get_logger().info(
                f"yaw integrator init: yaw_est={self._yaw_est:.3f} rad "
                f"(yaw_init_from_input={self.yaw_init_from_input})"
            )

    def _integrate_planar_state(self, vx, wz):
        now = self.get_clock().now()
        dt = (now - self._last_integration_time).nanoseconds * 1e-9
        self._last_integration_time = now

        if dt < 0.0 or dt > 0.2:
            return

        # integrate yaw first
        self._yaw_est = _clamp_angle_pi(self._yaw_est + wz * dt)

        # integrate planar position
        self._x_est += vx * math.cos(self._yaw_est) * dt
        self._y_est += vx * math.sin(self._yaw_est) * dt


    def _integrate_projected_state(self, vx_raw, vy_raw, wz):
        """
        Integrate planar state using projected COM velocity.

        Mathematical model:

            dx/dt = v_long * cos(theta)
            dy/dt = v_long * sin(theta)
            dtheta/dt = wz

        """

        now = self.get_clock().now()
        dt = (now - self._last_integration_time).nanoseconds * 1e-9
        self._last_integration_time = now

        if dt < 0.0 or dt > 0.2:
            return

        # Project horizontal COM velocity
        v_long = self._project_com_velocity(vx_raw, vy_raw)

        # Deadband
        if abs(v_long) < self.deadband_vx:
            v_long = 0.0

        if abs(wz) < self.deadband_wz:
            wz = 0.0

        # Integrate yaw
        self._yaw_est = _clamp_angle_pi(self._yaw_est + wz * dt)

        # Integrate planar position
        self._x_est += v_long * math.cos(self._yaw_est) * dt
        self._y_est += v_long * math.sin(self._yaw_est) * dt

    # ----------------------------------------------------
    # NEW: Horizontal COM projection
    # ----------------------------------------------------
    def _project_com_velocity(self, vx, vy):
        """
        Project 3D COM velocity onto horizontal plane.

        v_proj = sqrt(vx^2 + vy^2)
        v_long = v_proj * sign(vx)

        This avoids lateral oscillation generating fake translation.
        """

        v_proj = math.sqrt(vx*vx + vy*vy)

        if vx < 0:
            v_proj = -v_proj

        return v_proj
    
    # ----------------------------
    # Watchdog
    # ----------------------------
    def _watchdog_tick(self) -> None:
        """Warn if the incoming odom stream is stale."""
        if self._last_rx_time is None:
            self.get_logger().warn("No odom received yet...", throttle_duration_sec=2.0)
            return

        age = (self.get_clock().now() - self._last_rx_time).nanoseconds * 1e-9
        if age > self.timeout_sec:
            self.get_logger().warn(
                f"Odometry stale: age={age:.2f}s (timeout={self.timeout_sec:.2f}s).",
                throttle_duration_sec=2.0,
            )

    # ----------------------------
    # Time / frame helpers
    # ----------------------------
    def _choose_stamp(self, msg: Odometry):
        """Choose outgoing timestamp based on policy."""
        if self.use_node_time:
            return self.get_clock().now().to_msg()
        return msg.header.stamp

    def _choose_child_frame(self, msg: Odometry) -> str:
        """Choose TF child frame based on force_child_frame policy."""
        if self.force_child_frame:
            return self.base_frame
        incoming = (msg.child_frame_id or "").strip()
        return incoming if incoming else self.base_frame

    # ----------------------------
    # Yaw estimator
    # ----------------------------
    def _compute_yaw_est(self, wz: float) -> float:
        """
        Integrate yaw using corrected wz (angular.z after remap).

        We guard against large dt jumps to avoid "teleport" yaw updates.
        """
        if not self.integrate_yaw:
            return 0.0

        if not self._yaw_inited or self._last_yaw_time is None:
            self._yaw_inited = True
            self._last_yaw_time = self.get_clock().now()
            return self._yaw_est

        now = self.get_clock().now()
        dt = (now - self._last_yaw_time).nanoseconds * 1e-9
        self._last_yaw_time = now

        # Guard against timing jumps (startup / pauses / clock changes)
        if dt < 0.0 or dt > 0.2:
            dt = 0.0

        # Freeze if stable
        if self.enable_filter:
            idle_time = (self.get_clock().now() - self._last_motion_time).nanoseconds * 1e-9
            if idle_time > self.freeze_after_sec:
                return self._yaw_est
    
        self._yaw_est = _clamp_angle_pi(self._yaw_est + float(wz) * dt)
        return self._yaw_est

    # ----------------------------
    # Pose extraction (planar)
    # ----------------------------
    def _extract_xy_and_yaw_quat(
        self,
        msg: Odometry,
        wz_corrected: float,
    ) -> Tuple[float, float, float, float, float, float, float]:
        """
        Return a planar pose:
          (x, y, z=0) + yaw-only quaternion.

        yaw source:
        - integrate_yaw=True: yaw := ∫ wz dt (recommended)
        - integrate_yaw=False: yaw from input quaternion (may be wrong if axes differ)
        """
        #x = float(msg.pose.pose.position.x)
        #y = float(msg.pose.pose.position.y)

        self._x_est
        self._y_est
        self._yaw_est

        # Apply zero-at-start translation
        if self.zero_at_start and self._zero_ready:
            x -= self._x0
            y -= self._y0

        if self.integrate_yaw:
            yaw = self._compute_yaw_est(wz_corrected)
            if self.zero_at_start and self._zero_ready:
                yaw = _clamp_angle_pi(yaw - self._yaw0)
        else:
            q = msg.pose.pose.orientation
            yaw = _yaw_from_quat(float(q.x), float(q.y), float(q.z), float(q.w))
            if self.zero_at_start and self._zero_ready:
                yaw = _clamp_angle_pi(yaw - self._yaw0)

        qx, qy, qz, qw = _quat_from_yaw(yaw)
        return x, y, 0.0, qx, qy, qz, qw

    # ----------------------------
    # Twist correction
    # ----------------------------
    def _correct_twist(self, msg: Odometry):
        """
        Extract and correct twist for Nav2.

        Returns:
          (vx, vy, vz, wx, wy, wz_corrected)

        If remap_yawrate_y_to_z=True:
          wz_corrected := angular.y
        """
        vx = float(msg.twist.twist.linear.x)
        vy = float(msg.twist.twist.linear.y)
        vz = float(msg.twist.twist.linear.z)

        wx = float(msg.twist.twist.angular.x)
        wy = float(msg.twist.twist.angular.y)
        wz = float(msg.twist.twist.angular.z)

        # Unitree/G1 driver quirk: yaw rate is in angular.y
        wz_corrected = wy if self.remap_yawrate_y_to_z else wz

        if self.planarize_twist:
            wx = 0.0
            wy = 0.0
            vz = 0.0
            if self.kill_linear_y:
                vy = 0.0

        # ----------------------------
        # Humanoid RL filtering
        # ----------------------------
        if self.enable_filter:

            # Deadband
            if abs(vx) < self.deadband_vx:
                vx = 0.0

            if abs(wz_corrected) < self.deadband_wz:
                wz_corrected = 0.0

            # EMA filter
            self._vx_f = self.ema_alpha * vx + (1 - self.ema_alpha) * self._vx_f
            self._wz_f = self.ema_alpha * wz_corrected + (1 - self.ema_alpha) * self._wz_f

            vx = self._vx_f
            wz_corrected = self._wz_f

            # Freeze detection
            if abs(vx) > 0.0 or abs(wz_corrected) > 0.0:
                self._last_motion_time = self.get_clock().now()


        return vx, vy, vz, wx, wy, wz_corrected

    # ----------------------------
    # TF publishing tick
    # ----------------------------
    def _publish_tf_tick_old(self) -> None:
        """Publish TF odom_frame -> base_frame from the latest odom."""
        if self._last_msg is None:
            return

        msg = self._last_msg
        stamp = self._choose_stamp(msg)
        child = self._choose_child_frame(msg)

        #vx, vy, vz, wx, wy, wz_corr = self._last_filtered
        #x, y, z, qx, qy, qz, qw = self._extract_xy_and_yaw_quat(msg, wz_corr)

        stamp = self.get_clock().now().to_msg()

        x = self._x_est
        y = self._y_est
        qx, qy, qz, qw = _quat_from_yaw(self._yaw_est)

        tfm = TransformStamped()
        tfm.header.stamp = stamp
        tfm.header.frame_id = self.odom_frame
        tfm.child_frame_id = child
        tfm.transform.translation.x = x
        tfm.transform.translation.y = y
        tfm.transform.translation.z = z
        tfm.transform.rotation.x = qx
        tfm.transform.rotation.y = qy
        tfm.transform.rotation.z = qz
        tfm.transform.rotation.w = qw

        self.tf_broadcaster.sendTransform(tfm)

        self.get_logger().info(
            f"TF {self.odom_frame}->{child} (x={x:.3f}, y={y:.3f})",
            throttle_duration_sec=2.0,
        )

    def _publish_tf_tick(self) -> None:
        if self._last_msg is None:
            return

        stamp = self.get_clock().now().to_msg()
        child = self.base_frame

        x = self._x_est
        y = self._y_est
        z = 0.0
        qx, qy, qz, qw = _quat_from_yaw(self._yaw_est)

        tfm = TransformStamped()
        tfm.header.stamp = stamp
        tfm.header.frame_id = self.odom_frame
        tfm.child_frame_id = child
        tfm.transform.translation.x = x
        tfm.transform.translation.y = y
        tfm.transform.translation.z = z
        tfm.transform.rotation.x = qx
        tfm.transform.rotation.y = qy
        tfm.transform.rotation.z = qz
        tfm.transform.rotation.w = qw

        self.tf_broadcaster.sendTransform(tfm)
        
    # ----------------------------
    # /odom publishing tick
    # ----------------------------
    def _publish_odom_tick_old(self) -> None:
        """Publish corrected /odom for Nav2."""
        if self._last_msg is None:
            return

        msg = self._last_msg
        stamp = self._choose_stamp(msg)
        child = self._choose_child_frame(msg)
        
        # ------------------------------------------
        # NEW: SE(2) projected integration
        # ------------------------------------------
        #x, y, z, qx, qy, qz, qw = self._extract_xy_and_yaw_quat(msg, wz_corr)

        vx, vy, vz, wx, wy, wz_corr = self._correct_twist(msg)
        # integrate projected SE(2)
        self._integrate_projected_state(
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
            wz_corr
        )

        # store filtered once
        self._last_filtered = (vx, vy, vz, wx, wy, wz_corr)

        # build pose
        x = self._x_est
        y = self._y_est
        z = 0.0
        qx, qy, qz, qw = _quat_from_yaw(self._yaw_est)


        out = Odometry()
        out.header.stamp = stamp
        out.header.frame_id = self.odom_frame
        out.child_frame_id = child

        # Pose (planar)
        out.pose.pose.position.x = x
        out.pose.pose.position.y = y
        out.pose.pose.position.z = z
        out.pose.pose.orientation.x = qx
        out.pose.pose.orientation.y = qy
        out.pose.pose.orientation.z = qz
        out.pose.pose.orientation.w = qw
        #out.pose.covariance = msg.pose.covariance
        
        # Realistic covariance for humanoid
        out.pose.covariance = [0.0]*36
        out.pose.covariance[0] = 0.05   # x
        out.pose.covariance[7] = 0.05   # y
        out.pose.covariance[35] = 0.1   # yaw

        # Twist (corrected / planar) — SAFE COPY (no side effects)
        tw = Twist()
        tw.linear.x = vx
        tw.linear.y = vy
        tw.linear.z = vz
        tw.angular.x = wx
        tw.angular.y = wy
        tw.angular.z = wz_corr

        twc = TwistWithCovariance()
        twc.twist = tw
        #twc.covariance = msg.twist.covariance

        twc.covariance = [0.0]*36

        # linear velocity uncertainty
        twc.covariance[0] = 0.1   # vx
        twc.covariance[7] = 0.2   # vy (should be high for humanoid)

        # angular velocity uncertainty
        twc.covariance[35] = 0.2  # wz

        out.twist = twc

        self.odom_pub.publish(out)

        self.get_logger().info(
            f"/odom published (vx={vx:.3f}, wz={wz_corr:.3f})",
            throttle_duration_sec=2.0,
        )

    def _publish_odom_tick(self) -> None:
        if self._last_msg is None:
            return

        msg = self._last_msg
        stamp = self.get_clock().now().to_msg()
        child = self.base_frame

        # Filter twist ONCE
        vx, vy, vz, wx, wy, wz_corr = self._correct_twist(msg)

        # Integrate projected SE(2) using FILTERED values
        self._integrate_projected_state(
            vx,
            vy,
            wz_corr
        )

        # Build pose
        x = self._x_est
        y = self._y_est
        z = 0.0
        qx, qy, qz, qw = _quat_from_yaw(self._yaw_est)

        out = Odometry()
        out.header.stamp = stamp
        out.header.frame_id = self.odom_frame
        out.child_frame_id = child

        out.pose.pose.position.x = x
        out.pose.pose.position.y = y
        out.pose.pose.position.z = z
        out.pose.pose.orientation.x = qx
        out.pose.pose.orientation.y = qy
        out.pose.pose.orientation.z = qz
        out.pose.pose.orientation.w = qw

        # Pose covariance (humanoid realistic)
        out.pose.covariance = [0.0]*36
        out.pose.covariance[0]  = 0.05
        out.pose.covariance[7]  = 0.05
        out.pose.covariance[35] = 0.1

        # Twist
        tw = Twist()
        tw.linear.x  = vx
        tw.linear.y  = 0.0
        tw.linear.z  = 0.0
        tw.angular.x = 0.0
        tw.angular.y = 0.0
        tw.angular.z = wz_corr

        twc = TwistWithCovariance()
        twc.twist = tw
        twc.covariance = [0.0]*36
        twc.covariance[0]  = 0.1
        twc.covariance[7]  = 0.2
        twc.covariance[35] = 0.2

        out.twist = twc

        self.odom_pub.publish(out)

def main() -> None:
    """Node entry point."""
    rclpy.init()
    node = RealOdomTfBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()