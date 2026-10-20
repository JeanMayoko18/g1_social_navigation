#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry
import tf2_ros


# ---------------------------
# Utils
# ---------------------------
def quat_to_yaw(q):
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    )


def yaw_to_quat(yaw):
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


def norm_angle(a):
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


# ---------------------------
# Bridge Node
# ---------------------------
class MujocoToGazeboBridge(Node):
    def __init__(self):
        super().__init__("mujoco_to_gazebo_bridge")

        # Frames
        self.map_frame = self.declare_parameter("map_frame", "map").value
        self.odom_frame = self.declare_parameter("odom_frame", "odom").value
        self.base_frame = self.declare_parameter("base_frame", "base_footprint").value
        self.tf_buffer = tf2_ros.Buffer(cache_time=rclpy.duration.Duration(seconds=10))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Topics
        self.mujoco_pose_topic = self.declare_parameter(
            "mujoco_pose_topic", "/mujoco/base_pose"
        ).value

        self.odom_topic = self.declare_parameter(
            "odom_topic", "/odom"
        ).value

        # Options
        #self.publish_map_odom = self.declare_parameter(
        #    "publish_map_odom_identity", False
        #).value

        self.declare_parameter(
            "publish_map_odom_identity",
            True
        )

        self.publish_map_odom_identity = (
            self.get_parameter("publish_map_odom_identity")
            .get_parameter_value()
            .bool_value
        )

        self.rate_hz = float(self.declare_parameter("rate_hz", 50.0).value)

        # State
        self.last_pose_msg = None
        self.prev = None  # (t_sec, x, y, yaw)


        # Bootstrap TF state
        self.bootstrap_sent = False
        self.map_odom_owned_by_slam = False

        # ROS interfaces
        self.create_subscription(
            PoseStamped,
            self.mujoco_pose_topic,
            self.pose_cb,
            10
        )

        self.odom_pub = self.create_publisher(Odometry, self.odom_topic, 10)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.static_tf_broadcaster = tf2_ros.StaticTransformBroadcaster(self)

        #if self.publish_map_odom:
        #    self._publish_static_map_odom()
        

        self.create_timer(1.0 / max(1e-6, self.rate_hz), self.publish)

        self.get_logger().info(
            f"Bridge: {self.mujoco_pose_topic} → TF({self.odom_frame}->{self.base_frame}) + {self.odom_topic}"
        )

    # ---------------------------
    def _publish_static_map_odom(self):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.map_frame
        t.child_frame_id = self.odom_frame
        t.transform.rotation.w = 1.0
        self.static_tf_broadcaster.sendTransform(t)

    def _maybe_publish_map_odom(self):
        if self.map_odom_owned_by_slam:
            return

        try:
            # Si SLAM publie déjà map→odom, on s'efface
            self.tf_buffer.lookup_transform(
                self.map_frame,
                self.odom_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1)
            )
            self.map_odom_owned_by_slam = True
            self.get_logger().info("SLAM detected: map->odom now owned by SLAM")
            return

        except Exception:
            pass  # SLAM pas encore prêt

        # 🔵 Bootstrap TF UNE SEULE FOIS
        if not self.bootstrap_sent:
            t = TransformStamped()
            t.header.stamp = rclpy.time.Time().to_msg()  # static TF
            t.header.frame_id = self.map_frame
            t.child_frame_id = self.odom_frame
            t.transform.rotation.w = 1.0

            self.static_tf_broadcaster.sendTransform(t)
            self.bootstrap_sent = True

            self.get_logger().info("Bootstrap TF map->odom published by bridge")

    # ---------------------------
    def pose_cb(self, msg: PoseStamped):
        self.last_pose_msg = msg
        self.get_logger().info(
            f"RX base_pose x={msg.pose.position.x:.3f} y={msg.pose.position.y:.3f}",
            throttle_duration_sec=1.0
        )

    # ---------------------------
    def publish_old(self):
        if self.last_pose_msg is None:
            return

        msg = self.last_pose_msg
        stamp = msg.header.stamp

        now = self.get_clock().now()
        if now.nanoseconds == 0:
            return
        stamp = now.to_msg()

        stamp = self.get_clock().now().to_msg()
        if stamp.sec == 0 and stamp.nanosec == 0:
            stamp = self.get_clock().now().to_msg()

        x = float(msg.pose.position.x)
        y = float(msg.pose.position.y)

        yaw = norm_angle(quat_to_yaw(msg.pose.orientation))

        # velocities
        vx = vy = wz = 0.0
        t_sec = float(stamp.sec) + 1e-9 * float(stamp.nanosec)

        if self.prev is not None:
            t0, x0, y0, yaw0 = self.prev
            dt = max(1e-6, t_sec - t0)
            vx = (x - x0) / dt
            vy = (y - y0) / dt
            wz = norm_angle(yaw - yaw0) / dt

        self.prev = (t_sec, x, y, yaw)

        qx, qy, qz, qw = yaw_to_quat(yaw)

        # TF
        tf = TransformStamped()
        tf.header.stamp = stamp
        tf.header.frame_id = self.odom_frame
        tf.child_frame_id = self.base_frame
        tf.transform.translation.x = x
        tf.transform.translation.y = y
        tf.transform.rotation.x = qx
        tf.transform.rotation.y = qy
        tf.transform.rotation.z = qz
        tf.transform.rotation.w = qw
        self.get_logger().info(
            f"PUB TF {self.odom_frame}->{self.base_frame} x={x:.3f} y={y:.3f}",
            throttle_duration_sec=1.0
        )

        self.tf_broadcaster.sendTransform(tf)

        # Odometry
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self.odom_frame
        odom.child_frame_id = self.base_frame

        odom.pose.pose.position.x = x
        odom.pose.pose.position.y = y
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw

        odom.twist.twist.linear.x = vx
        odom.twist.twist.linear.y = vy
        odom.twist.twist.angular.z = wz

        self.odom_pub.publish(odom)

        if self.publish_map_odom_identity:
            self._maybe_publish_map_odom()


    def publish(self):
        """Publish TF(odom->base) + /odom from the latest MuJoCo PoseStamped.

        Robust to missing /clock and to PoseStamped with stamp=0.
        """
        msg = self.last_pose_msg
        if msg is None:
            return

        # ----------------------------
        # Robust timestamp selection
        # ----------------------------
        now = self.get_clock().now()
        stamp_msg = msg.header.stamp

        # Prefer ROS time if valid; otherwise fall back to message stamp; if still 0 -> use wall-time anyway
        if now.nanoseconds != 0:
            stamp = now.to_msg()
            t_sec = now.nanoseconds * 1e-9
        elif (stamp_msg.sec != 0) or (stamp_msg.nanosec != 0):
            stamp = stamp_msg
            t_sec = float(stamp.sec) + 1e-9 * float(stamp.nanosec)
        else:
            # Last resort: do not drop the update (dropping causes Nav2 to never see odom/tf)
            now2 = self.get_clock().now()
            stamp = now2.to_msg()
            t_sec = now2.nanoseconds * 1e-9

        # ----------------------------
        # Pose extraction
        # ----------------------------
        x = float(msg.pose.position.x)
        y = float(msg.pose.position.y)
        yaw = norm_angle(quat_to_yaw(msg.pose.orientation))
        qx, qy, qz, qw = yaw_to_quat(yaw)

        # ----------------------------
        # Velocity estimation
        # ----------------------------
        vx = vy = wz = 0.0
        if self.prev is not None:
            t0, x0, y0, yaw0 = self.prev
            dt = max(1e-6, t_sec - t0)
            vx = (x - x0) / dt
            vy = (y - y0) / dt
            wz = norm_angle(yaw - yaw0) / dt
        self.prev = (t_sec, x, y, yaw)

        # ----------------------------
        # TF: odom -> base_frame
        # ----------------------------
        tfm = TransformStamped()
        tfm.header.stamp = stamp
        tfm.header.frame_id = self.odom_frame
        tfm.child_frame_id = self.base_frame
        tfm.transform.translation.x = x
        tfm.transform.translation.y = y
        tfm.transform.translation.z = 0.0
        tfm.transform.rotation.x = qx
        tfm.transform.rotation.y = qy
        tfm.transform.rotation.z = qz
        tfm.transform.rotation.w = qw
        self.tf_broadcaster.sendTransform(tfm)

        # Optional debug (keep throttled)
        self.get_logger().info(
            f"PUB TF {self.odom_frame}->{self.base_frame} x={x:.3f} y={y:.3f}",
            throttle_duration_sec=1.0
        )

        # ----------------------------
        # Odometry
        # ----------------------------
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self.odom_frame
        odom.child_frame_id = self.base_frame

        odom.pose.pose.position.x = x
        odom.pose.pose.position.y = y
        odom.pose.pose.position.z = 0.0
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw

        odom.twist.twist.linear.x = vx
        odom.twist.twist.linear.y = vy
        odom.twist.twist.angular.z = wz

        self.odom_pub.publish(odom)

        # ----------------------------
        # Bootstrap map->odom if requested
        # ----------------------------
        if self.publish_map_odom_identity:
            self._maybe_publish_map_odom()

def main():
    rclpy.init()
    node = MujocoToGazeboBridge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
