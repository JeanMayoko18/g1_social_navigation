#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist


def qos(reliable=True, depth=10):
    return QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE if reliable else ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
        depth=depth,
    )


class ScanAmclGate(Node):
    def __init__(self):
        super().__init__("scan_amcl_gate")

        self.declare_parameter("input_scan_topic", "/scan")
        self.declare_parameter("output_scan_topic", "/scan_amcl")
        self.declare_parameter("cmd_topic", "/cmd_vel_nav_scaled")

        self.declare_parameter("linear_stop_thresh", 0.02)
        self.declare_parameter("angular_stop_thresh", 0.03)
        self.declare_parameter("stationary_hold_sec", 1.0)
        self.declare_parameter("output_rate_hz", 10.0)

        self.input_scan_topic = self.get_parameter("input_scan_topic").value
        self.output_scan_topic = self.get_parameter("output_scan_topic").value
        self.cmd_topic = self.get_parameter("cmd_topic").value

        self.linear_stop_thresh = float(self.get_parameter("linear_stop_thresh").value)
        self.angular_stop_thresh = float(self.get_parameter("angular_stop_thresh").value)
        self.stationary_hold_sec = float(self.get_parameter("stationary_hold_sec").value)
        self.output_rate_hz = float(self.get_parameter("output_rate_hz").value)

        self.last_cmd_time = self.get_clock().now()
        self.last_publish_time = self.get_clock().now()
        self.is_moving = False

        self.pub = self.create_publisher(
            LaserScan,
            self.output_scan_topic,
            qos(True, 10),
        )

        self.sub_scan = self.create_subscription(
            LaserScan,
            self.input_scan_topic,
            self.on_scan,
            qos(True, 10),
        )

        self.sub_cmd = self.create_subscription(
            Twist,
            self.cmd_topic,
            self.on_cmd,
            qos(False, 10),
        )

        self.get_logger().info(
            f"ScanAmclGate started: {self.input_scan_topic} -> {self.output_scan_topic}, "
            f"cmd={self.cmd_topic}"
        )

    def on_cmd(self, msg: Twist):
        vx = abs(msg.linear.x)
        wz = abs(msg.angular.z)

        moving = vx > self.linear_stop_thresh or wz > self.angular_stop_thresh

        if moving:
            self.is_moving = True
            self.last_cmd_time = self.get_clock().now()
        else:
            # Do not immediately declare stationary.
            # Wait stationary_hold_sec after last movement command.
            now = self.get_clock().now()
            dt = (now - self.last_cmd_time).nanoseconds * 1e-9
            if dt >= self.stationary_hold_sec:
                self.is_moving = False

    def on_scan(self, msg: LaserScan):
        now = self.get_clock().now()

        if self.is_moving:
            return

        dt_pub = (now - self.last_publish_time).nanoseconds * 1e-9
        if dt_pub < 1.0 / self.output_rate_hz:
            return

        out = LaserScan()
        out.header = msg.header
        out.header.stamp = now.to_msg()
        out.header.frame_id = "nav_laser_frame"

        out.angle_min = msg.angle_min

        if msg.angle_increment <= 0.0 or not math.isfinite(msg.angle_increment):
            self.get_logger().warn("Dropping scan_amcl: invalid angle_increment")
            return
        out.angle_increment = msg.angle_increment

        n = len(msg.ranges)
        out.angle_max = out.angle_min + (n - 1) * out.angle_increment

        out.time_increment = 0.0
        out.scan_time = 1.0 / self.output_rate_hz
        
        out.range_min = 0.45
        out.range_max = 8.0
        invalid_value = out.range_max

        clean_ranges = []

        for r in msg.ranges:
            if not math.isfinite(r):
                clean_ranges.append(invalid_value)
            elif r < out.range_min:
                clean_ranges.append(invalid_value)
            elif r > out.range_max:
                clean_ranges.append(invalid_value)
            else:
                clean_ranges.append(float(r))

        if len(clean_ranges) < 10:
            self.get_logger().warn("Dropping scan_amcl: too few ranges")
            return

        expected = int(round((out.angle_max - out.angle_min) / out.angle_increment)) + 1
        if abs(expected - len(clean_ranges)) > 2:
            self.get_logger().warn(
                f"Dropping scan_amcl: inconsistent size expected={expected}, got={len(clean_ranges)}"
            )
            return

        out.ranges = clean_ranges
        out.intensities = []

        self.pub.publish(out)
        self.last_publish_time = now

    def on_scan_o(self, msg: LaserScan):
        now = self.get_clock().now()

        if self.is_moving:
            return

        dt_pub = (now - self.last_publish_time).nanoseconds * 1e-9
        if dt_pub < 1.0 / self.output_rate_hz:
            return

        n = len(msg.ranges)
        if n < 10:
            self.get_logger().warn("Dropping scan_amcl: too few ranges")
            return

        if msg.angle_increment <= 0.0 or not math.isfinite(msg.angle_increment):
            self.get_logger().warn("Dropping scan_amcl: invalid angle_increment")
            return

        out = LaserScan()
        out.header.stamp = now.to_msg()
        out.header.frame_id = "nav_laser_frame"

        out.angle_min = msg.angle_min
        out.angle_increment = msg.angle_increment
        out.angle_max = msg.angle_min + (n - 1) * msg.angle_increment

        out.time_increment = 0.0
        out.scan_time = 1.0 / self.output_rate_hz

        out.range_min = 0.45
        out.range_max = 8.0
        invalid_value = out.range_max

        clean_ranges = []
        for r in msg.ranges:
            if not math.isfinite(r):
                clean_ranges.append(invalid_value)
            elif r < out.range_min:
                clean_ranges.append(invalid_value)
            elif r > out.range_max:
                clean_ranges.append(invalid_value)
            else:
                clean_ranges.append(float(r))

        expected = int(round(
            (out.angle_max - out.angle_min) / out.angle_increment
        )) + 1

        if len(clean_ranges) != expected:
            self.get_logger().warn(
                f"Dropping scan_amcl: inconsistent size expected={expected}, got={len(clean_ranges)}"
            )
            return

        out.ranges = clean_ranges
        out.intensities = []

        self.pub.publish(out)
        self.last_publish_time = now

def main(args=None):
    rclpy.init(args=args)
    node = ScanAmclGate()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()