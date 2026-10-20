"""
resolver.py — Automatic ROS 2 subscription QoS resolution.

Responsibility:
    Inspect the QoS offered by publishers on a topic and build a compatible
    subscription QoS profile.

Flow:
    INPUT / CONFIGURATION
        ↓
    ROS GRAPH INSPECTION
        ↓
    COMPATIBILITY RESOLUTION
        ↓
    RESULT OBJECT
        ↓
    RETURN TO CALLER
"""

from typing import Iterable

from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    DurabilityPolicy,
    HistoryPolicy,
)

from g1_qos_manager.profiles import (
    reliable_qos,
    sensor_data_qos,
)


def _resolve_reliability(
    publisher_infos: Iterable,
    prefer_sensor_data: bool,
) -> ReliabilityPolicy:
    """
    Resolve a subscription reliability policy.

    Inputs:
        publisher_infos:
            Publisher endpoint information returned by
            get_publishers_info_by_topic().

        prefer_sensor_data:
            Whether BEST_EFFORT should be preferred for sensor streams.

    Processing:
        - If any publisher is BEST_EFFORT, the subscriber must use
          BEST_EFFORT to remain compatible with all discovered publishers.
        - If every publisher is RELIABLE, RELIABLE can be used.
        - UNKNOWN or SYSTEM_DEFAULT values fall back according to
          prefer_sensor_data.

    Output:
        Selected ReliabilityPolicy.

    Called by:
        resolve_subscription_qos().

    Side effects:
        None.
    """

    reliabilities = {
        info.qos_profile.reliability
        for info in publisher_infos
    }

    if ReliabilityPolicy.BEST_EFFORT in reliabilities:
        return ReliabilityPolicy.BEST_EFFORT

    if reliabilities and reliabilities == {ReliabilityPolicy.RELIABLE}:
        return ReliabilityPolicy.RELIABLE

    return (
        ReliabilityPolicy.BEST_EFFORT
        if prefer_sensor_data
        else ReliabilityPolicy.RELIABLE
    )


def _resolve_durability(
    publisher_infos: Iterable,
) -> DurabilityPolicy:
    """
    Resolve a subscription durability policy.

    Inputs:
        publisher_infos:
            Publisher endpoint information returned by
            get_publishers_info_by_topic().

    Processing:
        Prefer VOLATILE whenever at least one publisher is VOLATILE.
        Use TRANSIENT_LOCAL only if every discovered publisher offers it.

    Output:
        Selected DurabilityPolicy.

    Called by:
        resolve_subscription_qos().

    Side effects:
        None.
    """

    durabilities = {
        info.qos_profile.durability
        for info in publisher_infos
    }

    if DurabilityPolicy.VOLATILE in durabilities:
        return DurabilityPolicy.VOLATILE

    if (
        durabilities
        and durabilities == {DurabilityPolicy.TRANSIENT_LOCAL}
    ):
        return DurabilityPolicy.TRANSIENT_LOCAL

    return DurabilityPolicy.VOLATILE


def resolve_subscription_qos(
    node,
    topic_name: str,
    fallback_depth: int = 10,
    prefer_sensor_data: bool = False,
) -> QoSProfile:
    """
    Build a subscription QoS compatible with discovered publishers.

    Inputs:
        node:
            ROS 2 node used to inspect the ROS graph.

        topic_name:
            Topic whose publishers must be inspected.

        fallback_depth:
            KEEP_LAST queue depth used for the subscription.

        prefer_sensor_data:
            If True, use BEST_EFFORT as fallback when no publisher is
            discovered or publisher reliability cannot be resolved.

    Processing:
        1. Query publishers with get_publishers_info_by_topic().
        2. If no publisher exists yet, return a deterministic fallback.
        3. Resolve reliability conservatively.
        4. Resolve durability conservatively.
        5. Build a KEEP_LAST QoS profile.

    Output:
        QoSProfile ready for create_subscription().

    Called by:
        Any G1 workspace node before creating a subscription.

    Side effects:
        Reads the current ROS 2 graph.
    """

    publisher_infos = node.get_publishers_info_by_topic(topic_name)

    if not publisher_infos:
        if prefer_sensor_data:
            return sensor_data_qos(depth=fallback_depth)

        return reliable_qos(depth=fallback_depth)

    reliability = _resolve_reliability(
        publisher_infos,
        prefer_sensor_data,
    )

    durability = _resolve_durability(
        publisher_infos,
    )

    return QoSProfile(
        reliability=reliability,
        durability=durability,
        history=HistoryPolicy.KEEP_LAST,
        depth=fallback_depth,
    )