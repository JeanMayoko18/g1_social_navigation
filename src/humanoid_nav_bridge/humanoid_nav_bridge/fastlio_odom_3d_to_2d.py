#!/usr/bin/env python3

"""
fastlio_odom_3d_to_2d.py

Convert FAST-LIO 3D odometry:

    /Odometry
    frame: camera_init -> body

into a Nav2-compatible planar odometry:

    /odom
    frame: odom -> base_footprint

The node preserves:
    - x
    - y
    - yaw

and removes:
    - z
    - roll
    - pitch

This allows Nav2 to operate on a standard 2D odometry source
while still benefiting from FAST-LIO LiDAR-Inertial odometry.
"""
#from datetime import dt
#from geometry_msgs import msg
import math

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
from rclpy.duration import Duration
from humanoid_nav_bridge.logger_utils import make_section


def yaw_from_quaternion(q):
    """
    Extract yaw angle from a ROS quaternion.

    Parameters
    ----------
    q : geometry_msgs.msg.Quaternion

    Returns
    -------
    float
        Yaw angle in radians.
    """
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def quaternion_from_yaw(yaw):
    """
    Build a planar quaternion from a yaw angle.

    Roll and pitch are forced to zero.

    Parameters
    ----------
    yaw : float

    Returns
    -------
    tuple
        (qx, qy, qz, qw)
    """
    qz = math.sin(yaw * 0.5)
    qw = math.cos(yaw * 0.5)

    return 0.0, 0.0, qz, qw


class FastLioOdom3DTo2D(Node):
    """
    Bridge node converting FAST-LIO 3D odometry into
    a 2D Nav2-compatible odometry source.
    """

    def __init__(self):
        super().__init__("fastlio_odom_3d_to_2d")

        # ---------------------------------------------------------
        # Parameters
        # ---------------------------------------------------------

        self.declare_parameter(
            "input_odom_topic",
            "/Odometry"
        )

        self.declare_parameter(
            "output_odom_topic",
            "/odom"
        )

        self.declare_parameter(
            "odom_frame",
            "odom"
        )

        self.declare_parameter(
            "base_frame",
            "base_footprint"
        )

        self.declare_parameter(
            "publish_tf",
            False
        )
        self.declare_parameter("publish_rate", 30.0)
        self.declare_parameter("max_odom_age", 0.80)
        self.declare_parameter("max_step_m", 0.35)
        self.declare_parameter("max_speed_mps", 2.5)
        self.declare_parameter("max_yaw_rate", 6.0)
        self.declare_parameter("max_reject_count", 30)

        self.declare_parameter("hold_last_valid_on_jump", True)
        self.declare_parameter("reset_after_reject_limit", False)
        # ---------------------------------------------------------
        # Read Parameters
        # ---------------------------------------------------------
        self.input_odom_topic = self.get_parameter(
            "input_odom_topic"
        ).value

        self.output_odom_topic = self.get_parameter(
            "output_odom_topic"
        ).value

        self.odom_frame = self.get_parameter(
            "odom_frame"
        ).value

        self.base_frame = self.get_parameter(
            "base_frame"
        ).value

        self.publish_tf = self.get_parameter(
            "publish_tf"
        ).value

        self.publish_rate = float(self.get_parameter("publish_rate").value)
        self.max_odom_age = float(self.get_parameter("max_odom_age").value)
        self.max_step_m = float(self.get_parameter("max_step_m").value)
        self.max_speed_mps = float(self.get_parameter("max_speed_mps").value)
        self.max_yaw_rate = float(self.get_parameter("max_yaw_rate").value)
        self.max_reject_count = int(self.get_parameter("max_reject_count").value)


        self.last_vx = 0.0
        self.last_vy = 0.0
        self.last_wz = 0.0
        self.last_valid_time = None
        self.last_old_odom_warn_time = None
        self.old_odom_warn_period = 2.0

        # ---------------------------------------------------------
        # Publisher
        # ---------------------------------------------------------

        self.odom_pub = self.create_publisher(
            Odometry,
            self.output_odom_topic,
            20
        )

        # ---------------------------------------------------------
        # TF broadcaster
        # ---------------------------------------------------------

        self.tf_broadcaster = TransformBroadcaster(self)

        # ---------------------------------------------------------
        # Subscriber
        # ---------------------------------------------------------

        self.odom_sub = self.create_subscription(
            Odometry,
            self.input_odom_topic,
            self.odom_callback,
            20
        )
        
        self.pub_timer = self.create_timer(
            1.0 / self.publish_rate,
            self.publish_last_odom
        )

        self.last_x_nav = None
        self.last_y_nav = None
        self.last_yaw_nav = None
        self.max_fastlio_yaw_jump = 0.50
        self.max_fastlio_xy_jump = 1.00
        
        title = "FAST-LIO 3D -> 2D bridge configuration"
        fields = {
            "input_odom_topic": self.input_odom_topic,
            "output_odom_topic": self.output_odom_topic,
        }

        self.get_logger().info(make_section(title, fields))

        self.origin_initialized = False
        self.x0 = 0.0
        self.y0 = 0.0
        self.yaw0 = 0.0

        self.reject_count = 0
        self.reject_count = 0
        self.log_throttle_sec = 40.0
        
    def throttled_log(self, key, period_sec, level, title, fields=None, inline=False):
        fields = fields or {}

        now_sec = self.get_clock().now().nanoseconds * 1e-9
        last = getattr(self, key, None)

        if last is None or (now_sec - last) >= period_sec:
            setattr(self, key, now_sec)

            if inline:
                details = ", ".join(f"{k}={v}" for k, v in fields.items())
                text = f"{title}: {details}" if details else title
            else:
                text = make_section(title, fields)

            if level == "warn":
                self.get_logger().warn(text)
            elif level == "error":
                self.get_logger().error(text)
            else:
                self.get_logger().info(text)
                
    def publish_last_odom(self):
        """
        Publish the last valid FAST-LIO planar odometry at a fixed rate.

        This function is called by a ROS 2 timer, for example at 30 Hz.

        It does NOT process incoming FAST-LIO data directly.
        It only republishes the last valid state stored by odom_callback().

        Purpose:
            - decouple Nav2 from irregular FAST-LIO publication bursts;
            - provide a stable /odom_fastlio_2d frequency;
            - stop publishing if FAST-LIO becomes too old, so that a fallback
            odometry source can take over.
        """

        # =========================================================
        # 1) DO NOT PUBLISH BEFORE THE FIRST VALID FAST-LIO ORIGIN
        # =========================================================
        # The local 2D odometry frame is initialized in odom_callback().
        # Before that, we do not know the initial x, y, yaw reference.

        if not self.origin_initialized:
            return

        # =========================================================
        # 2) DO NOT PUBLISH IF NO VALID STATE HAS BEEN STORED YET
        # =========================================================
        # last_x_nav contains the last accepted planar x position.
        # last_valid_time tells when this state was last updated.

        if self.last_x_nav is None or self.last_valid_time is None:
            return

        # =========================================================
        # 3) CHECK THE AGE OF THE LAST VALID FAST-LIO STATE
        # =========================================================
        # FAST-LIO may stop, freeze, publish late, or produce rejected jumps.
        # If the last accepted state is too old, we stop publishing.
        #
        # This is important because publishing an old fixed odom forever
        # would make Nav2 believe that FAST-LIO is still healthy.

        now_clock = self.get_clock().now()
        age = (now_clock - self.last_valid_time).nanoseconds * 1e-9

        if age > self.max_odom_age:
            if (
                self.last_old_odom_warn_time is None or
                (now_clock - self.last_old_odom_warn_time).nanoseconds * 1e-9
                > self.old_odom_warn_period
            ):
                self.get_logger().warn(
                    f"FAST-LIO odom too old: age={age:.3f}s. "
                    f"Stop publishing for fallback."
                )
                self.last_old_odom_warn_time = now_clock

            return
        # Convert current ROS time to message timestamp.
        # We use current time because this node republishes the latest valid
        # state at a regular frequency for Nav2.
        now = now_clock.to_msg()

        # =========================================================
        # 4) CONVERT STORED YAW INTO A PLANAR QUATERNION
        # =========================================================
        # Nav2 needs a quaternion orientation.
        # Roll and pitch are forced to zero.

        qx, qy, qz, qw = quaternion_from_yaw(self.last_yaw_nav)

        # =========================================================
        # 5) BUILD THE NAV2-COMPATIBLE ODOMETRY MESSAGE
        # =========================================================

        odom_msg = Odometry()

        # ---------------------------------------------------------
        # Header
        # ---------------------------------------------------------
        # frame_id is the odometry frame, normally "odom".
        # child_frame_id is the robot base frame, normally "base_footprint".

        odom_msg.header.stamp = now
        odom_msg.header.frame_id = self.odom_frame
        odom_msg.child_frame_id = self.base_frame

        # ---------------------------------------------------------
        # Pose: use the last valid planar FAST-LIO state
        # ---------------------------------------------------------
        # z is forced to 0.0 because Nav2 works in 2D.

        odom_msg.pose.pose.position.x = self.last_x_nav
        odom_msg.pose.pose.position.y = self.last_y_nav
        odom_msg.pose.pose.position.z = 0.0

        odom_msg.pose.pose.orientation.x = qx
        odom_msg.pose.pose.orientation.y = qy
        odom_msg.pose.pose.orientation.z = qz
        odom_msg.pose.pose.orientation.w = qw

        # ---------------------------------------------------------
        # Pose covariance
        # ---------------------------------------------------------
        # Small values for x, y, yaw mean we trust these planar variables.
        # Very large values for z, roll, pitch mean they are not used/trusted.
        #
        # Covariance index meaning in 6D:
        #   0  -> x
        #   7  -> y
        #   14 -> z
        #   21 -> roll
        #   28 -> pitch
        #   35 -> yaw

        odom_msg.pose.covariance = [0.0] * 36
        odom_msg.pose.covariance[0] = 0.05
        odom_msg.pose.covariance[7] = 0.05
        odom_msg.pose.covariance[14] = 999.0
        odom_msg.pose.covariance[21] = 999.0
        odom_msg.pose.covariance[28] = 999.0
        odom_msg.pose.covariance[35] = 0.10

        # =========================================================
        # 6) SET TWIST FROM THE LAST VALID ESTIMATED VELOCITIES
        # =========================================================
        # These velocities were computed in odom_callback() using the
        # difference between consecutive accepted FAST-LIO poses.

        odom_msg.twist.twist.linear.x = self.last_vx
        odom_msg.twist.twist.linear.y = self.last_vy
        odom_msg.twist.twist.linear.z = 0.0

        odom_msg.twist.twist.angular.x = 0.0
        odom_msg.twist.twist.angular.y = 0.0
        odom_msg.twist.twist.angular.z = self.last_wz

        # ---------------------------------------------------------
        # Twist covariance
        # ---------------------------------------------------------
        # We trust planar vx, vy, wz moderately.
        # z velocity, roll rate, and pitch rate are ignored.

        odom_msg.twist.covariance = [0.0] * 36
        odom_msg.twist.covariance[0] = 0.20
        odom_msg.twist.covariance[7] = 0.20
        odom_msg.twist.covariance[14] = 999.0
        odom_msg.twist.covariance[21] = 999.0
        odom_msg.twist.covariance[28] = 999.0
        odom_msg.twist.covariance[35] = 0.30

        # =========================================================
        # 7) PUBLISH REGULAR 2D ODOMETRY
        # =========================================================
        # This is the only place where odometry is published.
        # Therefore /odom_fastlio_2d becomes regular even if /Odometry
        # from FAST-LIO arrives in bursts.

        self.odom_pub.publish(odom_msg)

        # =========================================================
        # 8) OPTIONAL TF PUBLICATION
        # =========================================================
        # Only publish TF here, never in odom_callback().
        #
        # For your architecture with smart_odom_selector and AMCL:
        # usually keep publish_tf = False here.
        #
        # Let the final odometry source or EKF publish:
        #     odom -> base_footprint
        #
        # AMCL should publish:
        #     map -> odom

        if self.publish_tf:
            tf_msg = TransformStamped()
            tf_msg.header.stamp = now
            tf_msg.header.frame_id = self.odom_frame
            tf_msg.child_frame_id = self.base_frame

            tf_msg.transform.translation.x = self.last_x_nav
            tf_msg.transform.translation.y = self.last_y_nav
            tf_msg.transform.translation.z = 0.0

            tf_msg.transform.rotation.x = qx
            tf_msg.transform.rotation.y = qy
            tf_msg.transform.rotation.z = qz
            tf_msg.transform.rotation.w = qw

            self.tf_broadcaster.sendTransform(tf_msg)


    def odom_callback(self, msg: Odometry):
        """
        Receive FAST-LIO 3D odometry and convert it internally to a valid
        planar 2D odometry state.

        This callback does NOT publish odometry.
        It only performs three steps:

            1. Compute planar pose from FAST-LIO 3D pose
            2. Filter abnormal jumps
            3. Store the last valid state

        The actual publication is done periodically by publish_last_odom().
        """

        # =========================================================
        # 1) READ CURRENT TIME AND FAST-LIO 3D POSE
        # =========================================================

        now_clock = self.get_clock().now()
        stamp = now_clock.nanoseconds * 1e-9

        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation

        # Extract yaw from FAST-LIO quaternion.
        # Roll and pitch are ignored because Nav2 needs planar motion.
        yaw = yaw_from_quaternion(orientation)

        # =========================================================
        # 2) INITIALIZE LOCAL ODOM ORIGIN
        # =========================================================

        if not self.origin_initialized:
            self.x0 = position.x
            self.y0 = position.y
            self.yaw0 = yaw

            self.origin_initialized = True

            self.last_x_nav = 0.0
            self.last_y_nav = 0.0
            self.last_yaw_nav = 0.0

            self.last_vx = 0.0
            self.last_vy = 0.0
            self.last_wz = 0.0

            self.last_stamp = stamp
            self.last_valid_time = now_clock

            self.get_logger().info(
                f"FAST-LIO origin initialized: "
                f"x0={self.x0:.3f}, y0={self.y0:.3f}, "
                f"yaw0={math.degrees(self.yaw0):.2f} deg"
            )
            return

        # =========================================================
        # 3) CALCULATE PLANAR POSITION RELATIVE TO INITIAL ORIGIN
        # =========================================================

        dx = position.x - self.x0
        dy = position.y - self.y0

        # Rotate FAST-LIO coordinates into the local odom frame.
        # This makes the first FAST-LIO pose become:
        #   x = 0, y = 0, yaw = 0
        c = math.cos(-self.yaw0)
        s = math.sin(-self.yaw0)

        x_nav = c * dx - s * dy
        y_nav = s * dx + c * dy

        # Normalize yaw relative to the initial yaw.
        yaw_nav = math.atan2(
            math.sin(yaw - self.yaw0),
            math.cos(yaw - self.yaw0)
        )

        # =========================================================
        # 4) BASIC SAFETY FILTER AGAINST LARGE ABSOLUTE VALUES
        # =========================================================

        if abs(x_nav) > 50.0 or abs(y_nav) > 50.0:
            self.get_logger().error(
                f"FAST-LIO odom rejected: abnormal absolute pose "
                f"x={x_nav:.2f}, y={y_nav:.2f}"
            )
            return

        # =========================================================
        # 5) COMPUTE DELTA TIME
        # =========================================================

        dt = stamp - self.last_stamp

        if dt < 0.02:
            self.get_logger().debug(
                f"FAST-LIO sample ignored: dt too small ({dt:.4f}s)"
            )
            return
        # Default twist values.
        # They remain zero if the sample is invalid or repeated.
        vx = 0.0
        vy = 0.0
        wz = 0.0

        # =====================================================
        # 6) CALCULATE STEP BETWEEN CURRENT AND PREVIOUS STATE
        # =====================================================

        dx_step = x_nav - self.last_x_nav
        dy_step = y_nav - self.last_y_nav

        dyaw = math.atan2(
            math.sin(yaw_nav - self.last_yaw_nav),
            math.cos(yaw_nav - self.last_yaw_nav)
        )

        step = math.hypot(dx_step, dy_step)
        speed = step / dt
        yaw_rate = abs(dyaw / dt)

        # =====================================================
        # 7) FILTER FAST-LIO JUMPS
        # =====================================================

        jump_detected = (
            step > self.max_step_m or
            speed > self.max_speed_mps or
            yaw_rate > self.max_yaw_rate
        )

        if jump_detected:
            self.reject_count += 1

            self.throttled_log(
                key="last_jump_reject_warn_time",
                period_sec=self.log_throttle_sec,
                level="warn",
                title="FAST-LIO jump rejected",
                fields={
                    "reject_count": f"{self.reject_count}/{self.max_reject_count}",
                    "step": f"{step:.3f} m",
                    "speed": f"{speed:.3f} m/s",
                    "yaw_rate": f"{yaw_rate:.3f} rad/s",
                },
                inline=True,
            )

            x_nav = self.last_x_nav
            y_nav = self.last_y_nav
            yaw_nav = self.last_yaw_nav

            vx = 0.0
            vy = 0.0
            wz = 0.0

            if self.reject_count >= self.max_reject_count:
                self.throttled_log(
                    key="last_unhealthy_error_time",
                    period_sec=self.log_throttle_sec,
                    level="error",
                    title="FAST-LIO unhealthy",
                    fields={
                        "reason": "too many consecutive rejected jumps",
                        "reject_count": f"{self.reject_count}/{self.max_reject_count}",
                        "action": "keeping last valid state and waiting for stable odometry again",
                    },
                    inline=True,
                )
                return
        else:
            self.reject_count = 0

            vx = dx_step / dt
            vy = dy_step / dt
            wz = dyaw / dt
        # =========================================================
        # 8) STORE LAST VALID PLANAR STATE
        # =========================================================
        # No publishing is done here.
        # publish_last_odom() will publish this stored state at a fixed rate.

        self.last_x_nav = x_nav
        self.last_y_nav = y_nav
        self.last_yaw_nav = yaw_nav

        self.last_vx = vx
        self.last_vy = vy
        self.last_wz = wz
        
        self.last_stamp = stamp
        self.last_valid_time = now_clock


def main(args=None):
    """
    Node entry point.
    """
    rclpy.init(args=args)

    node = FastLioOdom3DTo2D()

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