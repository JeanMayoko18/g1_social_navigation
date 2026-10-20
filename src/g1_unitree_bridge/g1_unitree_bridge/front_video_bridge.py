#!/usr/bin/env python3

#source /opt/ros/humble/setup.bash
#source ~/g1_ws/src/unitree_ros2/cyclonedds_ws/install/setup.bash

import av
import rclpy

from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from unitree_go.msg import Go2FrontVideoData


class FrontVideoBridge(Node):
    """Convert Unitree H.264 front video stream to ROS 2 Image messages."""

    def __init__(self) -> None:
        super().__init__('front_video_bridge')

        self.bridge = CvBridge()
        self.codec = av.CodecContext.create('h264', 'r')

        self.publisher = self.create_publisher(
            Image,
            '/camera/front/image_raw',
            10,
        )

        self.subscription = self.create_subscription(
            Go2FrontVideoData,
            '/frontvideostream',
            self.video_callback,
            10,
        )

        self.received_packets = 0
        self.decoded_frames = 0
        self.decode_errors = 0

        self.get_logger().info(
            'Front video bridge started: '
            '/frontvideostream -> /camera/front/image_raw'
        )

    def video_callback(self, msg: Go2FrontVideoData) -> None:
        """Feed Unitree H.264 fragments to the persistent decoder."""

        if not msg.video720p:
            return

        self.received_packets += 1
        payload = bytes(msg.video720p)

        try:
            packets = self.codec.parse(payload)

            for packet in packets:
                try:
                    frames = self.codec.decode(packet)
                except av.error.InvalidDataError:
                    self.decode_errors += 1
                    continue

                for frame in frames:
                    image = frame.to_ndarray(format='bgr24')

                    if self.decoded_frames == 0:
                        self.get_logger().info(
                            f'Decoded image: '
                            f'shape={image.shape}, '
                            f'dtype={image.dtype}, '
                            f'ndim={image.ndim}, '
                            f'contiguous={image.flags["C_CONTIGUOUS"]}'
                        )

                    ros_image = self.bridge.cv2_to_imgmsg(
                        image,
                        encoding='bgr8',
                    )

                    ros_image.header.stamp = self.get_clock().now().to_msg()
                    ros_image.header.frame_id = 'front_camera'

                    self.publisher.publish(ros_image)
                    self.decoded_frames += 1

                    if self.decoded_frames % 30 == 0:
                        self.get_logger().info(
                            f'Decoded frames: {self.decoded_frames}, '
                            f'packets: {self.received_packets}, '
                            f'decode errors: {self.decode_errors}'
                        )

        except av.error.FFmpegError as error:
            self.decode_errors += 1

            if self.decode_errors % 100 == 1:
                self.get_logger().warning(
                    f'H.264 stream decoding error: {error}'
                )


def main(args=None) -> None:
    """Run the Unitree front video bridge node."""

    rclpy.init(args=args)
    node = FrontVideoBridge()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()