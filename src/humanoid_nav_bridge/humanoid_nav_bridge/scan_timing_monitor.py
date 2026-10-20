#!/usr/bin/env python3
"""
Lightweight timing monitor for the G1 navigation pipeline.

Purpose
-------
This node monitors selected ROS 2 topics without processing every message
in depth.

It can report:

1. Estimated message frequency.
2. Message header age relative to the local ROS clock.
3. Received and sampled message counts.
4. Availability of the laser-to-base transform at the latest scan stamp.
5. Availability and age of the latest map-to-odom transform.

Resource-saving strategy
------------------------
- PointCloud2 monitoring can be disabled.
- Only one message out of N is processed statistically.
- QoS queues use depth 1.
- TF listeners are created only when TF checks are enabled.
- TF lookup is performed periodically, not for every LaserScan.
- Only the latest scan timestamp is retained for TF checking.
"""
from typing import Any, Dict, Optional

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from rclpy.time import Time

from geometry_msgs.msg import PoseWithCovarianceStamped
from sensor_msgs.msg import LaserScan, PointCloud2
from tf2_ros import Buffer, TransformException, TransformListener


class ScanTimingMonitor(Node):
    """
    Lightweight monitor for scan, point cloud, AMCL and TF timing.
    """

    def __init__(self) -> None:
        super().__init__("scan_timing_monitor")

        # ============================================================
        # Enable or disable subscriptions
        # ============================================================
        self.enable_cloud = bool(
            self.declare_parameter(
                "enable_cloud",
                False,
            ).value
        )

        self.enable_scan_robot = bool(
            self.declare_parameter(
                "enable_scan_robot",
                True,
            ).value
        )

        self.enable_scan = bool(
            self.declare_parameter(
                "enable_scan",
                True,
            ).value
        )

        self.enable_amcl = bool(
            self.declare_parameter(
                "enable_amcl",
                True,
            ).value
        )

        # ============================================================
        # Topic names
        # ============================================================
        self.cloud_topic = str(
            self.declare_parameter(
                "cloud_topic",
                "/cloud_nav_deskewed",
            ).value
        )

        self.scan_robot_topic = str(
            self.declare_parameter(
                "scan_robot_topic",
                "/scan_robot",
            ).value
        )

        self.scan_topic = str(
            self.declare_parameter(
                "scan_topic",
                "/scan",
            ).value
        )

        self.amcl_topic = str(
            self.declare_parameter(
                "amcl_topic",
                "/amcl_pose",
            ).value
        )

        # ============================================================
        # Sampling and reporting
        # ============================================================
        self.report_period_sec = max(
            0.5,
            float(
                self.declare_parameter(
                    "report_period_sec",
                    5.0,
                ).value
            ),
        )

        self.sample_every_n = max(
            1,
            int(
                self.declare_parameter(
                    "sample_every_n",
                    10,
                ).value
            ),
        )

        self.age_change_threshold = max(
            0.0,
            float(
                self.declare_parameter(
                    "age_change_threshold",
                    0.20,
                ).value
            ),
        )

        self.hz_change_threshold = max(
            0.0,
            float(
                self.declare_parameter(
                    "hz_change_threshold",
                    1.00,
                ).value
            ),
        )

        self.stale_timeout_sec = max(
            0.1,
            float(
                self.declare_parameter(
                    "stale_timeout_sec",
                    2.0,
                ).value
            ),
        )

        # ============================================================
        # TF checking
        # ============================================================
        self.enable_tf_check = bool(
            self.declare_parameter(
                "enable_tf_check",
                False,
            ).value
        )

        self.enable_laser_tf_check = bool(
            self.declare_parameter(
                "enable_laser_tf_check",
                True,
            ).value
        )

        self.enable_map_odom_tf_check = bool(
            self.declare_parameter(
                "enable_map_odom_tf_check",
                True,
            ).value
        )

        self.tf_target_frame = str(
            self.declare_parameter(
                "tf_target_frame",
                "base_footprint",
            ).value
        )

        self.tf_source_frame = str(
            self.declare_parameter(
                "tf_source_frame",
                "nav_laser_fixed",
            ).value
        )

        self.map_frame = str(
            self.declare_parameter(
                "map_frame",
                "map",
            ).value
        )

        self.odom_frame = str(
            self.declare_parameter(
                "odom_frame",
                "odom",
            ).value
        )

        self.tf_lookup_timeout_sec = max(
            0.0,
            float(
                self.declare_parameter(
                    "tf_lookup_timeout_sec",
                    0.0,
                ).value
            ),
        )

        self.enable_odom_scan_tf_check = bool(
            self.declare_parameter(
                "enable_odom_scan_tf_check",
                True,
            ).value
        )

        self.enable_map_scan_tf_check = bool(
            self.declare_parameter(
                "enable_map_scan_tf_check",
                True,
            ).value
        )

        self.scan_frame = str(
            self.declare_parameter(
                "scan_frame",
                "nav_laser_fixed",
            ).value
        )

        self.immediate_tf_sample_every_n = max(
            1,
            int(
                self.declare_parameter(
                    "immediate_tf_sample_every_n",
                    10,
                ).value
            ),
        )

        # ============================================================
        # Runtime state
        # ============================================================
        self.stats: Dict[str, Dict[str, Any]] = {}

        # Count every callback without performing all calculations.
        self.receive_counters: Dict[str, int] = {}

        # Used to suppress repeated logs.
        self.last_report_signature: Dict[str, Dict[str, Any]] = {}

        # Keep ROS subscriptions alive.
        self.subscriptions_ = []

        # Latest /scan stamp retained for periodic TF checking.
        self.last_scan_stamp = None

        self.last_odom_scan_tf_ok = None
        self.last_odom_scan_tf_error = ""

        self.last_map_scan_tf_ok = None
        self.last_map_scan_tf_error = ""

        self.immediate_tf_counter = 0
        self.immediate_tf_ok_count = 0
        self.immediate_tf_fail_count = 0
        self.last_immediate_tf_ok = None
        self.last_immediate_tf_error = ""
        self.last_immediate_scan_age = None

        # Latest TF check results.
        self.last_tf_ok: Optional[bool] = None
        self.last_tf_error = ""
        self.last_tf_scan_age: Optional[float] = None

        self.last_map_odom_ok: Optional[bool] = None
        self.last_map_odom_error = ""
        self.last_map_odom_age: Optional[float] = None

        # TF resources are created only when required.
        self.tf_buffer: Optional[Buffer] = None
        self.tf_listener: Optional[TransformListener] = None

        if self.enable_tf_check:
            self.tf_buffer = Buffer()

            self.tf_listener = TransformListener(
                self.tf_buffer,
                self,
                spin_thread=False,
            )

        # ============================================================
        # QoS profiles
        # ============================================================
        sensor_best_effort_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ============================================================
        # Conditional subscriptions
        # ============================================================
        if self.enable_cloud:
            self.subscriptions_.append(
                self.create_subscription(
                    PointCloud2,
                    self.cloud_topic,
                    self.cb_cloud,
                    sensor_best_effort_qos,
                )
            )

        if self.enable_scan_robot:
            self.subscriptions_.append(
                self.create_subscription(
                    LaserScan,
                    self.scan_robot_topic,
                    self.cb_scan_robot,
                    sensor_best_effort_qos,
                )
            )

        if self.enable_scan:
            self.subscriptions_.append(
                self.create_subscription(
                    LaserScan,
                    self.scan_topic,
                    self.cb_scan,
                    sensor_best_effort_qos,
                )
            )

        if self.enable_amcl:
            self.subscriptions_.append(
                self.create_subscription(
                    PoseWithCovarianceStamped,
                    self.amcl_topic,
                    self.cb_amcl,
                    reliable_qos,
                )
            )

        # ============================================================
        # Periodic report timer
        # ============================================================
        self.timer = self.create_timer(
            self.report_period_sec,
            self.report,
        )

        self.get_logger().info(
            "\n"
            "=========================================================\n"
            "             G1 LIGHTWEIGHT TIMING MONITOR\n"
            "=========================================================\n"
            f" enable_cloud             : {self.enable_cloud}\n"
            f" enable_scan_robot        : {self.enable_scan_robot}\n"
            f" enable_scan              : {self.enable_scan}\n"
            f" enable_amcl              : {self.enable_amcl}\n"
            f" enable_tf_check          : {self.enable_tf_check}\n"
            f" enable_laser_tf_check    : "
            f"{self.enable_laser_tf_check}\n"
            f" enable_map_odom_tf_check : "
            f"{self.enable_map_odom_tf_check}\n"
            f" cloud_topic              : {self.cloud_topic}\n"
            f" scan_robot_topic         : {self.scan_robot_topic}\n"
            f" scan_topic               : {self.scan_topic}\n"
            f" amcl_topic               : {self.amcl_topic}\n"
            f" laser TF                 : "
            f"{self.tf_target_frame} <- {self.tf_source_frame}\n"
            f" localization TF          : "
            f"{self.map_frame} <- {self.odom_frame}\n"
            f" report_period_sec        : {self.report_period_sec}\n"
            f" sample_every_n           : {self.sample_every_n}\n"
            f" stale_timeout_sec        : {self.stale_timeout_sec}\n"
            f" TF listener created      : "
            f"{self.tf_listener is not None}\n"
            "========================================================="
        )

    # ================================================================
    # Topic callbacks
    # ================================================================

    def cb_cloud(self, msg: PointCloud2) -> None:
        """
        Process only the PointCloud2 header timestamp.

        The full message is still deserialized by DDS, which is why
        cloud monitoring should remain disabled during normal operation.
        """
        self.update(
            "cloud_nav_deskewed",
            msg.header.stamp,
        )

    def cb_scan_robot(self, msg: LaserScan) -> None:
        """
        Update statistics for the pre-relay LaserScan topic.
        """
        self.update(
            "scan_robot",
            msg.header.stamp,
        )
    
    def cb_scan(self, msg: LaserScan) -> None:
        """
        Update statistics for /scan and retain its latest timestamp.

        No TF lookup is performed in this callback.
        """
        self.update("scan", msg.header.stamp)

        if not self.enable_tf_check:
            return

        self.last_scan_stamp = msg.header.stamp

        if msg.header.frame_id:
            self.scan_frame = msg.header.frame_id

        # Sample only one scan out of N to keep the monitor lightweight.
        self.immediate_tf_counter += 1

        if (
            self.immediate_tf_counter
            % self.immediate_tf_sample_every_n
            != 0
        ):
            return

        if self.tf_buffer is None:
            return

        scan_time = Time.from_msg(msg.header.stamp)
        now = self.get_clock().now()

        self.last_immediate_scan_age = (
            now - scan_time
        ).nanoseconds * 1e-9

        try:
            available = self.tf_buffer.can_transform(
                self.odom_frame,
                self.scan_frame,
                scan_time,
                timeout=Duration(seconds=0.0),
            )

            if available:
                self.immediate_tf_ok_count += 1
                self.last_immediate_tf_ok = True
                self.last_immediate_tf_error = ""
            else:
                self.immediate_tf_fail_count += 1
                self.last_immediate_tf_ok = False
                self.last_immediate_tf_error = (
                    "transform unavailable at scan arrival"
                )

        except TransformException as ex:
            self.immediate_tf_fail_count += 1
            self.last_immediate_tf_ok = False
            self.last_immediate_tf_error = self.compact_error(
                str(ex)
            )
            
    def check_transform_at_scan_stamp(
        self,
        key: str,
        target_frame: str,
        source_frame: str,
    ) -> bool:
        """
        Check whether target_frame <- source_frame is available at the
        exact timestamp of the latest received LaserScan.

        The lookup uses a zero timeout to avoid blocking the monitor.
        """
        if not self.enable_tf_check:
            return False

        if self.tf_buffer is None:
            return False

        if self.last_scan_stamp is None:
            state = {
                "status": "no_scan",
            }

            if self.should_log(key, state):
                self.get_logger().info(
                    f"TF {target_frame} <- {source_frame}: "
                    "no scan timestamp available"
                )
                return True

            return False

        scan_time = Time.from_msg(self.last_scan_stamp)
        now = self.get_clock().now()

        scan_age = (
            now - scan_time
        ).nanoseconds * 1e-9

        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                scan_time,
                timeout=Duration(
                    seconds=self.tf_lookup_timeout_sec
                ),
            )

            tf_time = Time.from_msg(transform.header.stamp)

            tf_age = (
                now - tf_time
            ).nanoseconds * 1e-9

            state = {
                "status": "ok",
                "age": round(scan_age, 3),
            }

            if not self.should_log(key, state):
                return False

            self.get_logger().info(
                f"TF {target_frame} <- {source_frame}: "
                f"OK at scan stamp | "
                f"scan_age={scan_age:+.3f}s | "
                f"returned_tf_age={tf_age:+.3f}s"
            )

            return True

        except TransformException as ex:
            error = self.compact_error(str(ex))

            state = {
                "status": "failed",
                "age": round(scan_age, 3),
                "error": error,
            }

            if not self.should_log(key, state):
                return False

            self.get_logger().warn(
                f"TF {target_frame} <- {source_frame}: "
                f"FAILED at scan stamp | "
                f"scan_age={scan_age:+.3f}s | "
                f"{error}"
            )

            return True

    def cb_amcl(
        self,
        msg: PoseWithCovarianceStamped,
    ) -> None:
        """
        Update timing statistics for AMCL pose publications.
        """
        self.update(
            "amcl_pose",
            msg.header.stamp,
        )

    # ================================================================
    # Statistics
    # ================================================================

    def update(self, name: str, stamp) -> None:
        """
        Update timing statistics for one topic.

        Only one message out of sample_every_n is processed.

        The estimated frequency compensates for the sampling ratio.
        """
        received = self.receive_counters.get(name, 0) + 1
        self.receive_counters[name] = received

        # Avoid timestamp and frequency calculations for every message.
        if received % self.sample_every_n != 0:
            return

        now = self.get_clock().now()
        msg_time = Time.from_msg(stamp)

        age_sec = (
            now - msg_time
        ).nanoseconds * 1e-9

        now_sec = now.nanoseconds * 1e-9

        stats = self.stats.setdefault(
            name,
            {
                "samples": 0,
                "received": 0,
                "last_receive_time": None,
                "last_sample_wall_time": None,
                "hz": 0.0,
                "age": None,
                "min_age": None,
                "max_age": None,
            },
        )

        previous_sample_time = stats["last_receive_time"]

        if previous_sample_time is not None:
            dt = now_sec - previous_sample_time

            if dt > 1e-6:
                sampled_hz = 1.0 / dt

                # Restore the approximate original topic frequency.
                estimated_hz = (
                    sampled_hz
                    * float(self.sample_every_n)
                )

                if stats["hz"] <= 0.0:
                    stats["hz"] = estimated_hz
                else:
                    stats["hz"] = (
                        0.8 * stats["hz"]
                        + 0.2 * estimated_hz
                    )

        stats["last_receive_time"] = now_sec
        stats["last_sample_wall_time"] = now_sec
        stats["samples"] += 1
        stats["received"] = received
        stats["age"] = age_sec

        if (
            stats["min_age"] is None
            or age_sec < stats["min_age"]
        ):
            stats["min_age"] = age_sec

        if (
            stats["max_age"] is None
            or age_sec > stats["max_age"]
        ):
            stats["max_age"] = age_sec

    # ================================================================
    # Log suppression
    # ================================================================

    def should_log(
        self,
        key: str,
        state: Dict[str, Any],
    ) -> bool:
        """
        Return True only when the monitored state changed enough.
        """
        old = self.last_report_signature.get(key)

        if old is None:
            self.last_report_signature[key] = state
            return True

        if old.get("status") != state.get("status"):
            self.last_report_signature[key] = state
            return True

        if old.get("error") != state.get("error"):
            self.last_report_signature[key] = state
            return True

        old_hz = float(old.get("hz", 0.0))
        new_hz = float(state.get("hz", 0.0))

        if (
            abs(old_hz - new_hz)
            >= self.hz_change_threshold
        ):
            self.last_report_signature[key] = state
            return True

        old_age = float(old.get("age", 0.0))
        new_age = float(state.get("age", 0.0))

        if (
            abs(old_age - new_age)
            >= self.age_change_threshold
        ):
            self.last_report_signature[key] = state
            return True

        return False

    @staticmethod
    def compact_error(error: str, max_length: int = 180) -> str:
        """
        Collapse multiline TF errors and limit their log length.
        """
        compact = " ".join(error.split())

        if len(compact) <= max_length:
            return compact

        return compact[:max_length] + "..."

    # ================================================================
    # Topic reporting
    # ================================================================

    def report_one(self, name: str) -> bool:
        """
        Report one topic.

        Returns True when a log line was emitted.
        """
        stats = self.stats.get(name)

        if stats is None:
            received = self.receive_counters.get(name, 0)

            if received > 0:
                # Messages exist, but sample_every_n may not yet have
                # been reached.
                state = {
                    "status": "waiting_for_sample",
                }

                if self.should_log(name, state):
                    self.get_logger().info(
                        f"{name:20s}: "
                        f"waiting for sample | "
                        f"received={received}/"
                        f"{self.sample_every_n}"
                    )
                    return True

                return False

            state = {
                "status": "no_messages",
            }

            if self.should_log(name, state):
                self.get_logger().warn(
                    f"{name:20s}: no messages"
                )
                return True

            return False

        now_sec = (
            self.get_clock().now().nanoseconds
            * 1e-9
        )

        last_sample_time = stats.get(
            "last_sample_wall_time"
        )

        is_stale = (
            last_sample_time is None
            or (
                now_sec - last_sample_time
                > self.stale_timeout_sec
            )
        )

        status = "stale" if is_stale else "ok"

        age = (
            float(stats["age"])
            if stats["age"] is not None
            else 0.0
        )

        state = {
            "status": status,
            "hz": round(float(stats["hz"]), 2),
            "age": round(age, 2),
        }

        if not self.should_log(name, state):
            return False

        log_message = (
            f"{name:20s}: "
            f"status={status:5s} | "
            f"hz={stats['hz']:6.2f} | "
            f"age={age:+.3f}s | "
            f"min={stats['min_age']:+.3f}s | "
            f"max={stats['max_age']:+.3f}s | "
            f"received={stats['received']} | "
            f"samples={stats['samples']}"
        )

        if is_stale:
            self.get_logger().warn(log_message)
        else:
            self.get_logger().info(log_message)

        return True

    # ================================================================
    # TF reporting
    # ================================================================

    def check_laser_tf(self) -> bool:
        """
        Check target_frame <- source_frame at the latest scan stamp.

        Returns True when a log line was emitted.
        """
        if not self.enable_tf_check:
            return False

        if not self.enable_laser_tf_check:
            return False

        if self.tf_buffer is None:
            return False

        if self.last_scan_stamp is None:
            state = {
                "status": "no_scan",
            }

            if self.should_log("tf_laser_base", state):
                self.get_logger().info(
                    f"TF {self.tf_target_frame} <- "
                    f"{self.tf_source_frame}: "
                    "no scan timestamp available"
                )
                return True

            return False

        scan_time = Time.from_msg(
            self.last_scan_stamp
        )

        now = self.get_clock().now()

        self.last_tf_scan_age = (
            now - scan_time
        ).nanoseconds * 1e-9

        try:
            self.tf_buffer.lookup_transform(
                self.tf_target_frame,
                self.tf_source_frame,
                scan_time,
                timeout=Duration(
                    seconds=self.tf_lookup_timeout_sec
                ),
            )

            self.last_tf_ok = True
            self.last_tf_error = ""

        except TransformException as ex:
            self.last_tf_ok = False
            self.last_tf_error = self.compact_error(
                str(ex)
            )

        if self.last_tf_ok:
            state = {
                "status": "ok",
                "age": round(
                    self.last_tf_scan_age,
                    2,
                ),
            }

            if not self.should_log(
                "tf_laser_base",
                state,
            ):
                return False

            self.get_logger().info(
                f"TF {self.tf_target_frame} <- "
                f"{self.tf_source_frame}: "
                f"OK | "
                f"scan_age="
                f"{self.last_tf_scan_age:+.3f}s"
            )

            return True

        state = {
            "status": "failed",
            "age": round(
                self.last_tf_scan_age,
                2,
            ),
            "error": self.last_tf_error,
        }

        if not self.should_log(
            "tf_laser_base",
            state,
        ):
            return False

        self.get_logger().warn(
            f"TF {self.tf_target_frame} <- "
            f"{self.tf_source_frame}: "
            f"FAILED | "
            f"scan_age="
            f"{self.last_tf_scan_age:+.3f}s | "
            f"{self.last_tf_error}"
        )

        return True

    def check_map_odom_tf(self) -> bool:
        """
        Check the latest available map <- odom transform.

        Returns True when a log line was emitted.
        """
        if not self.enable_tf_check:
            return False

        if not self.enable_map_odom_tf_check:
            return False

        if self.tf_buffer is None:
            return False

        try:
            transform = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.odom_frame,
                Time(),
                timeout=Duration(
                    seconds=self.tf_lookup_timeout_sec
                ),
            )

            transform_time = Time.from_msg(
                transform.header.stamp
            )

            now = self.get_clock().now()

            self.last_map_odom_age = (
                now - transform_time
            ).nanoseconds * 1e-9

            self.last_map_odom_ok = True
            self.last_map_odom_error = ""

        except TransformException as ex:
            self.last_map_odom_ok = False
            self.last_map_odom_error = (
                self.compact_error(str(ex))
            )

        if self.last_map_odom_ok:
            age = (
                self.last_map_odom_age
                if self.last_map_odom_age is not None
                else 0.0
            )

            state = {
                "status": "ok",
                "age": round(age, 2),
            }

            if not self.should_log(
                "tf_map_odom",
                state,
            ):
                return False

            self.get_logger().info(
                f"TF {self.map_frame} <- "
                f"{self.odom_frame}: "
                f"OK | age={age:+.3f}s"
            )

            return True

        state = {
            "status": "failed",
            "error": self.last_map_odom_error,
        }

        if not self.should_log(
            "tf_map_odom",
            state,
        ):
            return False

        self.get_logger().warn(
            f"TF {self.map_frame} <- "
            f"{self.odom_frame}: "
            f"FAILED | "
            f"{self.last_map_odom_error}"
        )

        return True

    # ================================================================
    # Periodic report
    # ================================================================

    def report(self) -> None:
        """
        Periodically report topic and TF timing information.
        """
        any_log = False

        if self.enable_cloud:
            any_log = (
                self.report_one(
                    "cloud_nav_deskewed"
                )
                or any_log
            )

        if self.enable_scan_robot:
            any_log = (
                self.report_one(
                    "scan_robot"
                )
                or any_log
            )

        if self.enable_scan:
            any_log = (
                self.report_one("scan")
                or any_log
            )

        if self.enable_amcl:
            any_log = (
                self.report_one("amcl_pose")
                or any_log
            )


        if self.enable_tf_check:
            if self.enable_odom_scan_tf_check:
                any_log = (
                    self.check_transform_at_scan_stamp(
                        key="tf_odom_scan",
                        target_frame=self.odom_frame,
                        source_frame=self.scan_frame,
                    )
                    or any_log
                )

            if self.enable_map_scan_tf_check:
                any_log = (
                    self.check_transform_at_scan_stamp(
                        key="tf_map_scan",
                        target_frame=self.map_frame,
                        source_frame=self.scan_frame,
                    )
                    or any_log
                )

            if self.enable_map_odom_tf_check:
                any_log = (
                    self.check_map_odom_tf()
                    or any_log
                )

        if self.enable_tf_check:
            total_immediate = (
                self.immediate_tf_ok_count
                + self.immediate_tf_fail_count
            )

            if total_immediate > 0:
                immediate_success_ratio = (
                    100.0
                    * self.immediate_tf_ok_count
                    / total_immediate
                )

                state = {
                    "status": (
                        "ok"
                        if self.immediate_tf_fail_count == 0
                        else "partial"
                    ),
                    "ok": self.immediate_tf_ok_count,
                    "failed": self.immediate_tf_fail_count,
                }

                if self.should_log(
                    "tf_immediate_odom_scan",
                    state,
                ):
                    message = (
                        f"TF immediate {self.odom_frame} <- "
                        f"{self.scan_frame}: "
                        f"ok={self.immediate_tf_ok_count} | "
                        f"failed={self.immediate_tf_fail_count} | "
                        f"success={immediate_success_ratio:.1f}% | "
                        f"last_scan_age="
                        f"{self.last_immediate_scan_age:+.3f}s"
                    )

                    if self.immediate_tf_fail_count > 0:
                        self.get_logger().warn(message)
                    else:
                        self.get_logger().info(message)

                    any_log = True
                    
        if any_log:
            self.get_logger().info(
                "---- timing monitor updated ----"
            )

def main(args=None) -> None:
    """
    ROS 2 node entry point.
    """
    rclpy.init(args=args)

    node = ScanTimingMonitor()

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