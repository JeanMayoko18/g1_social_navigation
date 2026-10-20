#!/usr/bin/env python3

import math

import numpy as np

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)

from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2

from tf2_ros import Buffer
from tf2_ros import TransformException
from tf2_ros import TransformListener


class CameraObstacleNode(Node):
    """
    Extract low and nearby obstacles from an aligned depth image.

    Inputs
    ------
    Depth image:
        RealSense aligned depth image.

    Camera info:
        Camera intrinsic calibration corresponding to the aligned image.

    TF:
        Transform from the depth optical frame to the configured output frame.

    Processing
    ----------
    1. Convert depth pixels to metric depth.
    2. Back-project valid pixels into 3D camera coordinates.
    3. Transform points into base_footprint.
    4. Keep only points located in the configured low-obstacle ROI.

    Output
    ------
    sensor_msgs/msg/PointCloud2 containing obstacle points in base_footprint.

    Side effects
    ------------
    Publishes a filtered point cloud for use by Nav2.
    """

    def __init__(self):
        super().__init__('camera_obstacle_node')

        # --------------------------------------------------------------
        # Parameters
        # --------------------------------------------------------------

        self.declare_parameter(
            'depth_topic',
            '/camera/camera/aligned_depth_to_color/image_raw',
        )

        self.declare_parameter(
            'camera_info_topic',
            '/camera/camera/aligned_depth_to_color/camera_info',
        )

        self.declare_parameter(
            'output_topic',
            '/camera/obstacles',
        )

        self.declare_parameter(
            'output_frame',
            'base_footprint',
        )

        # Initial experimental ROI.
        self.declare_parameter('min_x', 0.15)
        self.declare_parameter('max_x', 1.50)

        self.declare_parameter('min_y', -0.60)
        self.declare_parameter('max_y', 0.60)

        # --------------------------------------------------------------
        # Ground / obstacle-height configuration
        # --------------------------------------------------------------
        #
        # IMPORTANT:
        # In the current G1 TF tree:
        #
        #     base_footprint == pelvis
        #
        # Therefore z=0 in base_footprint is approximately located at
        # pelvis height, not at floor level.
        #
        # Approximate pelvis height above the physical floor:
        #
        #     H_pelvis ~= 0.78 m
        #
        # The physical floor therefore corresponds approximately to:
        #
        #     z_floor_in_base = -0.78 m
        #
        # We keep obstacle heights expressed in intuitive physical
        # coordinates relative to the floor, then convert them internally
        # to base_footprint coordinates.
        #
        self.declare_parameter(
            'ground_z_in_output_frame',
            -0.78,
        )

        # Desired low-obstacle ROI:
        #   2 cm to 45 cm above the physical floor.
        self.declare_parameter(
            'min_obstacle_height',
            0.02,
        )

        self.declare_parameter(
            'max_obstacle_height',
            0.45,
        )


        self.declare_parameter('min_depth', 0.10)
        self.declare_parameter('max_depth', 2.00)

        # Pixel subsampling to reduce CPU/network load.
        self.declare_parameter('pixel_stride', 4)

        # RealSense Z16 is normally expressed in millimetres.
        self.declare_parameter('depth_scale', 0.001)

        self.depth_topic = str(
            self.get_parameter('depth_topic').value
        )

        self.camera_info_topic = str(
            self.get_parameter('camera_info_topic').value
        )

        self.output_topic = str(
            self.get_parameter('output_topic').value
        )

        self.output_frame = str(
            self.get_parameter('output_frame').value
        )

        # --------------------------------------------------------------
        # Runtime state
        # --------------------------------------------------------------

        self.camera_info = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(
            self.tf_buffer,
            self,
        )

        # --------------------------------------------------------------
        # QoS
        # --------------------------------------------------------------

        depth_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        camera_info_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        output_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        # --------------------------------------------------------------
        # ROS interfaces
        # --------------------------------------------------------------

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self.camera_info_callback,
            camera_info_qos,
        )

        self.depth_sub = self.create_subscription(
            Image,
            self.depth_topic,
            self.depth_callback,
            depth_qos,
        )

        self.cloud_pub = self.create_publisher(
            PointCloud2,
            self.output_topic,
            output_qos,
        )

        self.get_logger().info(
            'Camera obstacle node ready: '
            f'{self.depth_topic} -> {self.output_topic}'
        )

    # ------------------------------------------------------------------
    # INPUT / CONFIGURATION
    # ------------------------------------------------------------------

    def camera_info_callback(self, msg):
        """Store the most recent camera calibration."""
        self.camera_info = msg

    # ------------------------------------------------------------------
    # PROCESSING
    # ------------------------------------------------------------------

    def depth_callback(self, msg):
        """Convert one depth image into a filtered obstacle cloud."""

        if self.camera_info is None:
            return

        depth = self._decode_depth_image(msg)

        if depth is None:
            return

        points_camera = self._backproject_depth(depth)

        if points_camera.size == 0:
            self._publish_empty_cloud(msg)
            return

        points_base = self._transform_points(
            points_camera,
            msg.header.frame_id,
            msg.header.stamp,
        )

        if points_base is None:
            return

        roi_mask = self._build_obstacle_roi_mask(points_base)

        obstacle_points_camera = points_camera[roi_mask]

        self._publish_cloud(
            obstacle_points_camera,
            msg.header,
        )

    def _decode_depth_image(self, msg):
        """Decode ROS depth Image into a floating-point depth array."""

        height = msg.height
        width = msg.width

        if msg.encoding in ('16UC1', 'mono16'):

            raw = np.frombuffer(
                msg.data,
                dtype=np.uint16,
            ).reshape(height, width)

            depth_scale = float(
                self.get_parameter('depth_scale').value
            )

            return raw.astype(np.float32) * depth_scale

        if msg.encoding == '32FC1':

            return np.frombuffer(
                msg.data,
                dtype=np.float32,
            ).reshape(height, width)

        self.get_logger().warning(
            f'Unsupported depth encoding: {msg.encoding}',
            throttle_duration_sec=5.0,
        )

        return None

    def _backproject_depth(self, depth):
        """Back-project sampled depth pixels into camera optical coordinates."""

        info = self.camera_info

        fx = info.k[0]
        fy = info.k[4]
        cx = info.k[2]
        cy = info.k[5]

        if fx <= 0.0 or fy <= 0.0:
            return np.empty((0, 3), dtype=np.float32)

        stride = max(
            1,
            int(self.get_parameter('pixel_stride').value),
        )

        min_depth = float(
            self.get_parameter('min_depth').value
        )

        max_depth = float(
            self.get_parameter('max_depth').value
        )

        v, u = np.mgrid[
            0:depth.shape[0]:stride,
            0:depth.shape[1]:stride
        ]

        z = depth[
            0:depth.shape[0]:stride,
            0:depth.shape[1]:stride
        ]

        valid = (
            np.isfinite(z)
            & (z >= min_depth)
            & (z <= max_depth)
        )

        if not np.any(valid):
            return np.empty((0, 3), dtype=np.float32)

        z = z[valid]
        u = u[valid].astype(np.float32)
        v = v[valid].astype(np.float32)

        # ROS optical frame:
        # X = right
        # Y = down
        # Z = forward
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        return np.column_stack((x, y, z)).astype(np.float32)

    def _transform_points(
        self,
        points,
        source_frame,
        stamp,
    ):
        """Transform camera points into output_frame."""

        query_time = rclpy.time.Time.from_msg(stamp)

        if not self.tf_buffer.can_transform(
            self.output_frame,
            source_frame,
            query_time,
            timeout=Duration(seconds=0.02),
        ):
            return None

        try:
            transform = self.tf_buffer.lookup_transform(
                self.output_frame,
                source_frame,
                query_time,
            )

        except TransformException as exc:
            self.get_logger().warning(
                f'TF {source_frame} -> {self.output_frame} unavailable: {exc}',
                throttle_duration_sec=2.0,
            )
            return None


        t = transform.transform.translation
        q = transform.transform.rotation

        rotation = self._quaternion_to_rotation_matrix(
            q.x,
            q.y,
            q.z,
            q.w,
        )

        translation = np.array(
            [t.x, t.y, t.z],
            dtype=np.float32,
        )

        return points @ rotation.T + translation

    @staticmethod
    def _quaternion_to_rotation_matrix(x, y, z, w):
        """Convert quaternion to a 3x3 rotation matrix."""

        norm = math.sqrt(
            x * x
            + y * y
            + z * z
            + w * w
        )

        if norm == 0.0:
            return np.eye(3, dtype=np.float32)

        x /= norm
        y /= norm
        z /= norm
        w /= norm

        return np.array(
            [
                [
                    1.0 - 2.0 * (y * y + z * z),
                    2.0 * (x * y - z * w),
                    2.0 * (x * z + y * w),
                ],
                [
                    2.0 * (x * y + z * w),
                    1.0 - 2.0 * (x * x + z * z),
                    2.0 * (y * z - x * w),
                ],
                [
                    2.0 * (x * z - y * w),
                    2.0 * (y * z + x * w),
                    1.0 - 2.0 * (x * x + y * y),
                ],
            ],
            dtype=np.float32,
        )

    def _build_obstacle_roi_mask(self, points_base):
        """
        Select points located inside the configured obstacle ROI.

        Vertical filtering is expressed physically relative to the floor,
        even though output_frame is currently base_footprint and that frame
        is coincident with pelvis.

        Conversion:
            z_base = z_floor_in_base + obstacle_height

        With:
            z_floor_in_base ~= -0.78 m

        Examples:
            2 cm above floor:
                -0.78 + 0.02 = -0.76 m

            45 cm above floor:
                -0.78 + 0.45 = -0.33 m
        """

        min_x = float(self.get_parameter('min_x').value)
        max_x = float(self.get_parameter('max_x').value)

        min_y = float(self.get_parameter('min_y').value)
        max_y = float(self.get_parameter('max_y').value)

        ground_z = float(
            self.get_parameter(
                'ground_z_in_output_frame'
            ).value
        )

        min_obstacle_height = float(
            self.get_parameter(
                'min_obstacle_height'
            ).value
        )

        max_obstacle_height = float(
            self.get_parameter(
                'max_obstacle_height'
            ).value
        )

        # ----------------------------------------------------------
        # Convert physical obstacle heights above floor into the
        # current output-frame coordinates.
        #
        # Current approximation:
        #
        #     floor in base_footprint ~= -0.78 m
        #
        # Minimum:
        #
        #     z_min = -0.78 + 0.02
        #           = -0.76 m
        #
        # Maximum:
        #
        #     z_max = -0.78 + 0.45
        #           = -0.33 m
        #
        # So the camera keeps points physically located between
        # approximately 2 cm and 45 cm above the floor.
        # ----------------------------------------------------------
        min_z = ground_z + min_obstacle_height
        max_z = ground_z + max_obstacle_height

        return (
            (points_base[:, 0] >= min_x)
            & (points_base[:, 0] <= max_x)
            & (points_base[:, 1] >= min_y)
            & (points_base[:, 1] <= max_y)
            & (points_base[:, 2] >= min_z)
            & (points_base[:, 2] <= max_z)
        )
    # ------------------------------------------------------------------
    # OUTPUT
    # ------------------------------------------------------------------

    def _publish_cloud(self, points_camera, source_header):
        """
        Publish selected obstacle points in the original camera frame.

        Keeping the camera frame allows Nav2 ObstacleLayer to use the
        actual sensor origin for marking and raytracing/clearing.
        """

        cloud = point_cloud2.create_cloud_xyz32(
            source_header,
            points_camera.tolist(),
        )

        self.cloud_pub.publish(cloud)

    def _publish_empty_cloud(self, msg):
        """Publish an empty observation in the original camera frame."""

        cloud = point_cloud2.create_cloud_xyz32(
            msg.header,
            [],
        )

        self.cloud_pub.publish(cloud)

def main(args=None):
    rclpy.init(args=args)

    node = CameraObstacleNode()

    try:
        rclpy.spin(node)

    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()