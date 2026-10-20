#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan
from copy import deepcopy
from rclpy.duration import Duration
from tf2_ros import Buffer, TransformListener
from rclpy.time import Time
import math
import json
from std_msgs.msg import String
from humanoid_nav_bridge.logger_utils import make_section


def make_qos(reliable: bool, depth: int) -> QoSProfile:
    return QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE if reliable else ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
        depth=depth,
    )

class ScanQosRelay(Node):
    def __init__(self):
        super().__init__("scan_qos_relay")

        self.input_scan_topic = self.declare_parameter(
            "input_scan_topic", "/scan_robot"
        ).value

        self.output_scan_topic = self.declare_parameter(
            "output_scan_topic", "/scan"
        ).value

        self.override_stamp = bool(
            self.declare_parameter("override_stamp", False).value
        )

        self.sub_reliable = bool(
            self.declare_parameter("sub_reliable", False).value
        )

        self.pub_reliable = bool(
            self.declare_parameter("pub_reliable", False).value
        )

        self.qos_depth = int(
            self.declare_parameter("qos_depth", 10).value
        )

        self.max_input_age_sec = float(
            self.declare_parameter("max_input_age_sec", 0.15).value
        )
        self.min_finite_beams = int(
            self.declare_parameter("min_finite_beams", 20).value
        )
        self.output_frame_id = self.declare_parameter(
            "output_frame_id",
            "nav_laser_frame_test"
        ).value

        # Temporal scan jump filter
        self.scan_jump_filter_enabled = bool(
            self.declare_parameter("scan_jump_filter_enabled", True).value
        )
        self.scan_jump_threshold = float(
            self.declare_parameter("scan_jump_threshold", 0.8).value
        )
        self.scan_jump_hold_frames = int(
            self.declare_parameter("scan_jump_hold_frames", 3).value
        )
        self.stamp_past_offset_sec = float(
            self.declare_parameter("stamp_past_offset_sec", 0.03).value)
        

        self.tf_buffer = Buffer()

        self.tf_listener = TransformListener(
            self.tf_buffer,
            self,
        )

        self.last_ranges = None
        self.jump_counter = []

        sub_qos = make_qos(self.sub_reliable, self.qos_depth)
        pub_qos = make_qos(self.pub_reliable, self.qos_depth)

        self.pub = self.create_publisher(
            LaserScan,
            self.output_scan_topic,
            pub_qos,
        )
        self.pub_rviz = self.create_publisher(
            LaserScan,
            "/scan_rviz",
            make_qos(True, self.qos_depth),  # RELIABLE only for RViz on PC
        )

        self.rviz_pub_hz = float(self.declare_parameter("rviz_pub_hz", 2.0).value)
        self.rviz_period = (
            1.0 / self.rviz_pub_hz if self.rviz_pub_hz > 0.0 else 0.0
        )
        self.last_rviz_pub = self.get_clock().now()

        self.sub = self.create_subscription(
            LaserScan,
            self.input_scan_topic,
            self.callback,
            sub_qos,
        )
        self.declare_parameter("startup_drop_sec", 5.0)
        self.startup_drop_sec = float(self.get_parameter("startup_drop_sec").value)
        self.start_time = self.get_clock().now()
        self.log_pub = self.create_publisher(String, "/system/log_event", 10)

        self.publish_log(
            "info",
            "Scan QoS relay configuration",
            {
                "input_scan_topic": self.input_scan_topic,
                "input_qos": "RELIABLE" if self.sub_reliable else "BEST_EFFORT",
                "output_scan_topic": self.output_scan_topic,
                "output_qos": "RELIABLE" if self.pub_reliable else "BEST_EFFORT",
                "output_frame_id": self.output_frame_id,
                "override_stamp": self.override_stamp,
                "startup_drop_sec": self.startup_drop_sec,
            }
        )


    def publish_log(self, level: str, title: str, fields=None):
        fields = fields or {}

        text = make_section(title, fields)

        if level == "warn":
            self.get_logger().warn(text)
        elif level == "error":
            self.get_logger().error(text)
        else:
            self.get_logger().info(text)

        msg = String()
        msg.data = json.dumps({
            "level": level,
            "source": self.get_name(),
            "title": title,
            "fields": fields,
            "message": text,
        })
        self.log_pub.publish(msg)

    def apply_temporal_jump_filter(self, ranges):
        if not self.scan_jump_filter_enabled:
            self.last_ranges = list(ranges)
            return ranges

        if self.last_ranges is None or len(self.last_ranges) != len(ranges):
            self.last_ranges = list(ranges)
            self.jump_counter = [0] * len(ranges)
            return ranges

        filtered = list(ranges)

        for i, curr in enumerate(ranges):
            prev = self.last_ranges[i]

            if not math.isfinite(prev) or not math.isfinite(curr):
                self.jump_counter[i] = 0
                continue

            if abs(curr - prev) > self.scan_jump_threshold:
                if self.jump_counter[i] < self.scan_jump_hold_frames:
                    filtered[i] = prev
                    self.jump_counter[i] += 1
                else:
                    self.jump_counter[i] = 0
            else:
                self.jump_counter[i] = 0

        self.last_ranges = list(filtered)
        return filtered
  
    def callback(self, msg: LaserScan):
        out = deepcopy(msg)

        now = self.get_clock().now()

        input_stamp = rclpy.time.Time.from_msg(msg.header.stamp)
        input_age = (now - input_stamp).nanoseconds * 1e-9

        if input_age > self.max_input_age_sec:
            self.get_logger().warn(
                f"Dropping stale scan: age={input_age:.3f}s > {self.max_input_age_sec:.3f}s"
            )
            return

        startup_age = (
            now - self.start_time
        ).nanoseconds * 1e-9

        if startup_age < self.startup_drop_sec:
            return

        # Preserve the acquisition timestamp. Restamping at the relay
        # publication time can make the scan newer than the latest
        # odom -> base_footprint transform and cause TF future
        # extrapolation in AMCL.
        #if self.override_stamp:
        #    out.header.stamp = now.to_msg()
        
        if self.override_stamp:
            out.header.stamp = now.to_msg()
        else:
            out.header.stamp = msg.header.stamp

        if self.output_frame_id:
            out.header.frame_id = self.output_frame_id

        # AMCL/Nav2-compatible range limits
        out.range_min = 0.45
        out.range_max = 5.0
        #invalid_value = out.range_max
        invalid_value = float("inf")

        old_ranges = list(out.ranges)
        old_n = len(old_ranges)

        if old_n <= 1:
            self.get_logger().warn("Dropping scan: too few ranges")
            return

        # Clean input ranges
        clean_ranges = []
        for r in old_ranges:
            if not math.isfinite(r):
                clean_ranges.append(invalid_value)
            elif r < out.range_min:
                clean_ranges.append(invalid_value)
            elif r > out.range_max:
                clean_ranges.append(invalid_value)
            else:
                clean_ranges.append(float(r))

        finite_count = sum(1 for r in clean_ranges if math.isfinite(r))

        if finite_count < self.min_finite_beams:
            self.get_logger().warn(
                f"Dropping sparse scan: finite={finite_count} < {self.min_finite_beams}"
            )
            return

        # Keep the real number of beams and preserve the upstream angular resolution.
        # Do not stretch a partial scan to 360 degrees, otherwise AMCL/Nav2 will
        # believe that the measured sector covers the full circle.
        n = len(clean_ranges)

        if not math.isfinite(out.angle_min) or not math.isfinite(out.angle_increment):
            self.get_logger().warn("Dropping scan: invalid angle_min or angle_increment")
            return

        if out.angle_increment <= 0.0:
            self.get_logger().warn("Dropping scan: non-positive angle_increment")
            return

        out.angle_max = out.angle_min + (n - 1) * out.angle_increment

        out.ranges = self.apply_temporal_jump_filter(clean_ranges)
        out.intensities = []
        out.time_increment = 0.0
        out.scan_time = 0.1

        expected = round(
            (out.angle_max - out.angle_min) / out.angle_increment
        ) + 1

        if len(out.ranges) != expected:
            self.publish_log(
                "warn",
                "LaserScan geometry mismatch",
                {
                    "ranges": len(out.ranges),
                    "expected": expected,
                    "angle_min": f"{out.angle_min:.6f}",
                    "angle_max": f"{out.angle_max:.6f}",
                    "angle_increment": f"{out.angle_increment:.9f}",
                    "frame_id": out.header.frame_id,
                },
            )
            return

        self.pub.publish(out)        # Main scan for AMCL/Nav2

        if self.rviz_period > 0.0:
            if (now - self.last_rviz_pub).nanoseconds * 1e-9 >= self.rviz_period:
                self.pub_rviz.publish(out) # Reliable debug scan for RViz on PC
                self.last_rviz_pub = now
            
    def callback_360(self, msg: LaserScan):
        out = deepcopy(msg)

        age = (self.get_clock().now() - self.start_time).nanoseconds * 1e-9
        if age < self.startup_drop_sec:
            return

        # Force current timestamp and output frame
        out.header.stamp = self.get_clock().now().to_msg()

        if self.output_frame_id:
            out.header.frame_id = self.output_frame_id

        # Basic range limits for AMCL/Nav2
        out.range_min = 0.45
        out.range_max = 8.0
        #invalid_value = out.range_max + 1.0
        invalid_value = out.range_max

        old_ranges = list(out.ranges)
        old_n = len(old_ranges)

        if old_n <= 1:
            self.get_logger().warn("Dropping scan: too few ranges")
            return

        # Clean input ranges
        clean_old = []
        for r in old_ranges:
            if not math.isfinite(r):
                clean_old.append(invalid_value)
            elif r < out.range_min:
                clean_old.append(invalid_value)
            elif r > out.range_max:
                clean_old.append(invalid_value)
            else:
                clean_old.append(float(r))

        # Force exactly 360 beams
        TARGET_N = 360
        new_ranges = []

        for i in range(TARGET_N):
            j = round(i * (old_n - 1) / (TARGET_N - 1))
            new_ranges.append(clean_old[j])

        # 360 beams over 360 degrees:
        # angle_min = -pi
        # angle_max = -pi + 359 * 1 degree
        out.angle_min = -math.pi
        out.angle_increment = 2.0 * math.pi / TARGET_N
        out.angle_max = out.angle_min + (TARGET_N - 1) * out.angle_increment

        out.ranges = new_ranges
        out.intensities = []
        out.time_increment = 0.0
        out.scan_time = 0.1

        expected = round(
            (out.angle_max - out.angle_min) / out.angle_increment
        ) + 1

        if len(out.ranges) != expected:
            self.publish_log(
                "warn",
                "LaserScan geometry mismatch",
                {
                    "ranges": len(out.ranges),
                    "expected": expected,
                    "angle_min": f"{out.angle_min:.6f}",
                    "angle_max": f"{out.angle_max:.6f}",
                    "angle_increment": f"{out.angle_increment:.9f}",
                    "frame_id": out.header.frame_id,
                },
            )
            return
        
        self.pub.publish(out)

def main(args=None):
    rclpy.init(args=args)
    node = ScanQosRelay()

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