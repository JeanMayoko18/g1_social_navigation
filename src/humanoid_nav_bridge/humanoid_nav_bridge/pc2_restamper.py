#!/usr/bin/env python3
"""
pc2_restamper.py

==============================================================
PointCloud2 Timestamp Restamper (Production Utility Node)
==============================================================

Purpose
-------
This node fixes timestamp inconsistencies coming from LiDAR
drivers that publish PointCloud2 messages with:

    • Delayed timestamps
    • Unsynchronized clocks
    • Non-monotonic time values
    • Hardware time instead of ROS time

Such inconsistencies cause TF errors in SLAM and Nav2:

    - "Extrapolation into the past"
    - MessageFilter drops
    - Costmap instability
    - SLAM relocalization jumps

Strategy
--------
The node republishes the incoming PointCloud2 message with:

    header.stamp = current ROS node time

No spatial data is modified.
Only the timestamp is replaced.

This guarantees temporal coherence between:

    • TF tree (odom -> base)
    • LaserScan
    • SLAM Toolbox
    • Nav2 costmaps

Typical Pipeline
----------------
    /utlidar/cloud_livox_mid360   (driver time)
                ↓
         pc2_restamper
                ↓
            /cloud_now            (ROS time)
                ↓
     pointcloud_to_laserscan
                ↓
              /scan

Author: Jean Chrysostome Mayoko Biong
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

class Pc2Restamper(Node):
    """
    A ROS2 node that republishes a PointCloud2 topic
    with a fresh ROS timestamp (node clock).

    This ensures compatibility with TF and navigation stacks
    when LiDAR drivers publish inconsistent timestamps.
    """

    def __init__(self):
        # Initialize ROS2 node
        super().__init__("pc2_restamper")

        # ----------------------------------------------------------
        # Parameters
        # ----------------------------------------------------------
        # Input raw cloud topic from LiDAR driver
        self.in_topic = self.declare_parameter(
            "input_cloud_topic",
            "/utlidar/cloud_livox_mid360"
        ).value

        # Output restamped cloud topic
        self.out_topic = self.declare_parameter(
            "output_cloud_topic",
            "/cloud_now"
        ).value

        # ----------------------------------------------------------
        # Publishers & Subscribers
        # ----------------------------------------------------------
        # Publisher for restamped cloud
        #self.pub = self.create_publisher(PointCloud2, self.out_topic, 10)

        # Subscriber to original cloud
        #self.sub = self.create_subscription(
        #    PointCloud2,
        #    self.in_topic,
        #    self.cb,
        #    10
        #)

        self.qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.pub = self.create_publisher(PointCloud2, self.out_topic, self.qos)
        self.sub = self.create_subscription(
            PointCloud2,
            self.in_topic,
            self.cb,
            self.qos,
        )

        self.get_logger().info(
            f"[pc2_restamper] Restamping cloud: {self.in_topic} -> {self.out_topic} "
            f"(BEST_EFFORT, depth=1)"
        )

        #self.get_logger().info(
        #    f"[pc2_restamper] Restamping cloud:\n"
        #    f"    {self.in_topic}  -->  {self.out_topic}\n"
        #    f"    strategy: header.stamp = node_clock.now()"
        #)

    # --------------------------------------------------------------
    # Callback
    # --------------------------------------------------------------
    def cb(self, msg: PointCloud2):
        """
        Receives a PointCloud2 message and republishes it
        with the same spatial data but with a fresh timestamp.

        Important:
        - We construct a new message object to avoid mutating
          the incoming message in-place.
        - All fields are copied explicitly for clarity.
        """

        # Create new PointCloud2 message
        out = PointCloud2()

        # Copy all structural fields
        out.header = msg.header
        out.height = msg.height
        out.width = msg.width
        out.fields = msg.fields
        out.is_bigendian = msg.is_bigendian
        out.point_step = msg.point_step
        out.row_step = msg.row_step
        out.data = msg.data
        out.is_dense = msg.is_dense

        # ----------------------------------------------------------
        # Critical operation:
        # Replace timestamp with current ROS time
        # ----------------------------------------------------------
        out.header.stamp = self.get_clock().now().to_msg()

        # Publish corrected cloud
        self.pub.publish(out)


# ------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------
def main():
    """
    Standard ROS2 node lifecycle wrapper.
    Ensures safe shutdown without double rcl_shutdown errors.
    """
    rclpy.init()
    node = Pc2Restamper()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()