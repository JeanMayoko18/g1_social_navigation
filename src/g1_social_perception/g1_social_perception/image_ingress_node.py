#!/usr/bin/env python3

"""
Network image ingress for distributed G1 perception.

FLOW
====
Remote hardware topics
    |
    | DDS network boundary
    v
ImageIngress
    |
    +-- local RGB image
    +-- local aligned depth image
    `-- local aligned-depth CameraInfo

RESPONSIBILITIES
================
This node is the single remote image reader for a destination host.

It republishes selected remote camera streams onto perception-local
topics. Downstream perception nodes MUST use the local topics whenever
the ingress is active.

The node does not:
    - start camera hardware;
    - perform inference;
    - process images;
    - perform depth fusion;
    - compress or modify image payloads.

RGB and depth forwarding can be enabled independently so deployments
can avoid transporting streams that are processed on the robot.
"""

import rclpy
import time
from rclpy.node import Node

from sensor_msgs.msg import CameraInfo, Image

from g1_qos_manager.resolver import resolve_subscription_qos
from g1_qos_manager.profiles import reliable_qos, sensor_data_qos


class ImageIngressNode(Node):

    def __init__(self) -> None:
        super().__init__("image_ingress")

        self.enable_rgb = bool(
            self.declare_parameter(
                "enable_rgb",
                True,
            ).value
        )

        self.enable_depth = bool(
            self.declare_parameter(
                "enable_depth",
                True,
            ).value
        )

        self.enable_camera_info = bool(
            self.declare_parameter(
                "enable_camera_info",
                True,
            ).value
        )

        self.rgb_input_topic = self.declare_parameter(
            "rgb_input_topic",
            "/camera/camera/color/image_raw",
        ).value

        self.depth_input_topic = self.declare_parameter(
            "depth_input_topic",
            "/camera/camera/aligned_depth_to_color/image_raw",
        ).value

        self.camera_info_input_topic = self.declare_parameter(
            "camera_info_input_topic",
            "/camera/camera/aligned_depth_to_color/camera_info",
        ).value

        self.rgb_output_topic = self.declare_parameter(
            "rgb_output_topic",
            "/perception/ingress/color/image_raw",
        ).value

        self.depth_output_topic = self.declare_parameter(
            "depth_output_topic",
            "/perception/ingress/aligned_depth/image_raw",
        ).value

        self.camera_info_output_topic = self.declare_parameter(
            "camera_info_output_topic",
            "/perception/ingress/aligned_depth/camera_info",
        ).value

        self._subscriptions = []
        self._publishers = []
        
        self._relay_stats = {}
        self._diagnostics_period_sec = 5.0

        self._diagnostics_timer = self.create_timer(
            self._diagnostics_period_sec,
            self._report_relay_stats,
        )

        if self.enable_rgb:
            self._create_relay(
                Image,
                self.rgb_input_topic,
                self.rgb_output_topic,
            )

        if self.enable_depth:
            self._create_relay(
                Image,
                self.depth_input_topic,
                self.depth_output_topic,
            )

        if self.enable_camera_info:
            self._create_relay(
                CameraInfo,
                self.camera_info_input_topic,
                self.camera_info_output_topic,
            )

        self.get_logger().info(
            "ImageIngress configured: "
            f"rgb={self.enable_rgb}, "
            f"depth={self.enable_depth}, "
            f"camera_info={self.enable_camera_info}"
        )

    def _report_relay_stats(self) -> None:

        now = time.monotonic()

        for topic, stats in self._relay_stats.items():

            elapsed = max(
                now - stats["last_report_time"],
                1e-6,
            )

            rx_delta = (
                stats["received"]
                - stats["last_report_received"]
            )

            tx_delta = (
                stats["published"]
                - stats["last_report_published"]
            )

            rx_hz = rx_delta / elapsed
            tx_hz = tx_delta / elapsed

            last_rx = stats["last_received"]

            silence = (
                now - last_rx
                if last_rx is not None
                else float("inf")
            )

            self.get_logger().info(
                f"[IMAGE INGRESS] "
                f"TOPIC={topic} | "
                f"RX={stats['received']} | "
                f"TX={stats['published']} | "
                f"RX_HZ={rx_hz:.1f} | "
                f"TX_HZ={tx_hz:.1f} | "
                f"SILENCE={silence:.2f}s | "
                f"ERRORS={stats['errors']}"
            )

            stats["last_report_received"] = stats["received"]
            stats["last_report_published"] = stats["published"]
            stats["last_report_time"] = now
            
    def _create_relay(
        self,
        message_type,
        input_topic: str,
        output_topic: str,
    ) -> None:
        """Relay camera messages while bounding RGB buffering."""
        queue_depth = (
            1
            if output_topic == self.rgb_output_topic
            else 5
        )

        subscription_qos = sensor_data_qos(
            depth=queue_depth
        )

        if output_topic == self.depth_output_topic:
            publisher_qos = sensor_data_qos(
                depth=queue_depth
            )
            qos_description = "BEST_EFFORT -> BEST_EFFORT"
        else:
            publisher_qos = reliable_qos(
                depth=queue_depth
            )
            qos_description = "BEST_EFFORT -> RELIABLE"

        publisher = self.create_publisher(
            message_type,
            output_topic,
            publisher_qos,
        )

        self._relay_stats[output_topic] = {
            "received": 0,
            "published": 0,
            "errors": 0,
            "last_received": None,
            "last_report_received": 0,
            "last_report_published": 0,
            "last_report_time": time.monotonic(),
        }

        def relay_callback(msg) -> None:
            stats = self._relay_stats[output_topic]
            stats["received"] += 1
            stats["last_received"] = time.monotonic()

            try:
                # Forward the original header without restamping.
                publisher.publish(msg)
                stats["published"] += 1

            except Exception as exc:
                stats["errors"] += 1

                self.get_logger().error(
                    "[IMAGE INGRESS] Publish exception | "
                    f"topic={output_topic} | "
                    f"error={exc}",
                    throttle_duration_sec=2.0,
                )

        subscription = self.create_subscription(
            message_type,
            input_topic,
            relay_callback,
            subscription_qos,
        )

        self._publishers.append(publisher)
        self._subscriptions.append(subscription)

        self.get_logger().info(
            "Image ingress relay: "
            f"{input_topic} -> {output_topic} | "
            f"qos={qos_description} | "
            f"depth={queue_depth}"
        )
        
def main(args=None) -> None:
    rclpy.init(args=args)

    node = ImageIngressNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()