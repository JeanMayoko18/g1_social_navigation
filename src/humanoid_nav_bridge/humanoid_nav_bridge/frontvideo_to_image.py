#!/usr/bin/env python3
from __future__ import annotations

import time
import threading

import av
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from unitree_go.msg import Go2FrontVideoData


class FrontVideoToImage(Node):
    """
    Decode Unitree H264 video stream into sensor_msgs/Image.

    Strategy:
    - subscription receives H264 chunks quickly
    - decoder extracts frames
    - only the most recent decoded frame is kept
    - a timer publishes the latest frame at a bounded rate

    This avoids spending too much time publishing every decoded frame and
    helps keep latency low.
    """

    def __init__(self) -> None:
        super().__init__("frontvideo_to_image")

        self.out_topic = self.declare_parameter("out_topic", "/camera/image_raw").value
        self.frame_id = self.declare_parameter("frame_id", "camera_link").value
        self.publish_rate = float(self.declare_parameter("publish_rate", 15.0).value)
        self.log_period_s = float(self.declare_parameter("log_period_s", 2.0).value)

        self.pub = self.create_publisher(Image, self.out_topic, qos_profile_sensor_data)

        self.sub = self.create_subscription(
            Go2FrontVideoData,
            "/frontvideostream",
            self.cb,
            qos_profile_sensor_data,
        )

        self.codec = av.CodecContext.create("h264", "r")

        self._lock = threading.Lock()
        self._latest_img = None
        self._latest_shape = None

        self._rx_msgs = 0
        self._decoded_frames = 0
        self._published_frames = 0
        self._bad_packets = 0
        self._t_last_log = time.time()

        timer_period = 1.0 / max(self.publish_rate, 1.0)
        self.timer = self.create_timer(timer_period, self.publish_latest)

        self.get_logger().info(
            f"Listening /frontvideostream -> publishing {self.out_topic} "
            f"(frame_id={self.frame_id}, target_pub_rate={self.publish_rate:.1f} Hz)"
        )

    def cb(self, msg: Go2FrontVideoData) -> None:
        self._rx_msgs += 1
        data = bytes(msg.data)

        try:
            packets = self.codec.parse(data)
        except Exception:
            self._bad_packets += 1
            return

        newest = None
        newest_shape = None

        for pkt in packets:
            try:
                frames = self.codec.decode(pkt)
            except av.error.InvalidDataError:
                self._bad_packets += 1
                continue
            except Exception:
                self._bad_packets += 1
                continue

            for frame in frames:
                try:
                    img = frame.to_ndarray(format="bgr24")
                except Exception:
                    self._bad_packets += 1
                    continue

                newest = img
                newest_shape = img.shape
                self._decoded_frames += 1

        if newest is not None:
            with self._lock:
                self._latest_img = newest
                self._latest_shape = newest_shape

        self.maybe_log()

    def publish_latest(self) -> None:
        with self._lock:
            if self._latest_img is None or self._latest_shape is None:
                return
            img = self._latest_img
            h, w = self._latest_shape[:2]

        out = Image()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = self.frame_id
        out.height = h
        out.width = w
        out.encoding = "bgr8"
        out.is_bigendian = 0
        out.step = w * 3
        out.data = img.tobytes()

        self.pub.publish(out)
        self._published_frames += 1
        self.maybe_log()

    def maybe_log(self) -> None:
        now = time.time()
        if now - self._t_last_log < self.log_period_s:
            return
        self._t_last_log = now
        self.get_logger().info(
            "rx_msgs=%d decoded_frames=%d published_frames=%d bad_packets=%d"
            % (
                self._rx_msgs,
                self._decoded_frames,
                self._published_frames,
                self._bad_packets,
            )
        )


def main() -> None:
    rclpy.init()
    node = FrontVideoToImage()
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