"""
profiles.py — Standard ROS 2 QoS profiles for the G1 workspace.

Responsibility:
    Provide reusable QoSProfile factories for sensor streams, reliable
    application data, and latched/static data.

Flow:
    INPUT / CONFIGURATION
        ↓
    PROFILE CONSTRUCTION
        ↓
    RESULT OBJECT
        ↓
    RETURN TO CALLER
"""

from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    DurabilityPolicy,
    HistoryPolicy,
)


def sensor_data_qos(depth: int = 1) -> QoSProfile:
    """
    Build a low-latency QoS profile for high-rate sensor streams.

    Inputs:
        depth:
            KEEP_LAST queue depth.

    Processing:
        Uses BEST_EFFORT reliability and VOLATILE durability to avoid
        blocking high-rate streams such as RGB, depth, LiDAR, and scans.

    Output:
        Configured QoSProfile.

    Called by:
        Nodes subscribing to or publishing high-bandwidth sensor data.

    Side effects:
        None.
    """

    return QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
        depth=depth,
    )


def reliable_qos(depth: int = 10) -> QoSProfile:
    """
    Build a reliable volatile QoS profile.

    Inputs:
        depth:
            KEEP_LAST queue depth.

    Processing:
        Uses RELIABLE delivery for application-level data where message
        loss is undesirable.

    Output:
        Configured QoSProfile.

    Called by:
        Nodes handling detections, commands, states, and similar data.

    Side effects:
        None.
    """

    return QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
        depth=depth,
    )


def transient_local_qos(depth: int = 1) -> QoSProfile:
    """
    Build a reliable transient-local QoS profile.

    Inputs:
        depth:
            KEEP_LAST queue depth.

    Processing:
        Uses RELIABLE delivery with TRANSIENT_LOCAL durability so late
        subscribers can receive the most recently retained data.

    Output:
        Configured QoSProfile.

    Called by:
        Nodes publishing static or latched-like state.

    Side effects:
        None.
    """

    return QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
        history=HistoryPolicy.KEEP_LAST,
        depth=depth,
    )