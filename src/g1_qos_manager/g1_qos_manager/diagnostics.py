"""
diagnostics.py — QoS diagnostics for the G1 workspace.

Responsibility:
    Provide reusable logging helpers to inspect publisher QoS endpoints and
    report the QoS profile selected by a node.

Flow:
    INPUT / CONFIGURATION
        ↓
    ROS GRAPH INSPECTION
        ↓
    QOS DESCRIPTION
        ↓
    LOG OUTPUT
"""

from typing import Iterable, Optional

from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    DurabilityPolicy,
    HistoryPolicy,
)


def reliability_to_string(policy) -> str:
    """
    Convert a ROS 2 reliability policy to a readable string.

    Inputs:
        policy:
            ReliabilityPolicy value.

    Processing:
        Maps known reliability values to stable text labels.

    Output:
        Human-readable reliability name.

    Called by:
        describe_qos(), log_topic_qos().

    Side effects:
        None.
    """

    if policy == ReliabilityPolicy.RELIABLE:
        return "RELIABLE"

    if policy == ReliabilityPolicy.BEST_EFFORT:
        return "BEST_EFFORT"

    if policy == ReliabilityPolicy.SYSTEM_DEFAULT:
        return "SYSTEM_DEFAULT"

    if policy == ReliabilityPolicy.UNKNOWN:
        return "UNKNOWN"

    return str(policy)


def durability_to_string(policy) -> str:
    """
    Convert a ROS 2 durability policy to a readable string.

    Inputs:
        policy:
            DurabilityPolicy value.

    Processing:
        Maps known durability values to stable text labels.

    Output:
        Human-readable durability name.

    Called by:
        describe_qos(), log_topic_qos().

    Side effects:
        None.
    """

    if policy == DurabilityPolicy.VOLATILE:
        return "VOLATILE"

    if policy == DurabilityPolicy.TRANSIENT_LOCAL:
        return "TRANSIENT_LOCAL"

    if policy == DurabilityPolicy.SYSTEM_DEFAULT:
        return "SYSTEM_DEFAULT"

    if policy == DurabilityPolicy.UNKNOWN:
        return "UNKNOWN"

    return str(policy)


def history_to_string(policy) -> str:
    """
    Convert a ROS 2 history policy to a readable string.

    Inputs:
        policy:
            HistoryPolicy value.

    Processing:
        Maps known history values to stable text labels.

    Output:
        Human-readable history name.

    Called by:
        describe_qos().

    Side effects:
        None.
    """

    if policy == HistoryPolicy.KEEP_LAST:
        return "KEEP_LAST"

    if policy == HistoryPolicy.KEEP_ALL:
        return "KEEP_ALL"

    if policy == HistoryPolicy.SYSTEM_DEFAULT:
        return "SYSTEM_DEFAULT"

    if policy == HistoryPolicy.UNKNOWN:
        return "UNKNOWN"

    return str(policy)


def describe_qos(qos: QoSProfile) -> str:
    """
    Build a compact human-readable QoS description.

    Inputs:
        qos:
            QoSProfile to describe.

    Processing:
        Extracts reliability, durability, history, and queue depth.

    Output:
        Single-line QoS description.

    Called by:
        Nodes and diagnostic helpers.

    Side effects:
        None.
    """

    return (
        f"reliability={reliability_to_string(qos.reliability)} | "
        f"durability={durability_to_string(qos.durability)} | "
        f"history={history_to_string(qos.history)} | "
        f"depth={qos.depth}"
    )


def log_publishers_qos(
    node,
    topic_name: str,
    publisher_infos: Optional[Iterable] = None
) -> None:
    """
    Log QoS profiles offered by publishers on a topic.

    Inputs:
        node:
            ROS 2 node providing graph access and logging.

        topic_name:
            Topic to inspect.

        publisher_infos:
            Optional pre-fetched publisher endpoint information. If omitted,
            the ROS graph is queried.

    Processing:
        Reads all publisher endpoints and logs their offered QoS profiles.

    Output:
        None.

    Called by:
        Nodes during initialization or diagnostics.

    Side effects:
        Writes diagnostic messages to the ROS logger.
    """

    if publisher_infos is None:
        publisher_infos = node.get_publishers_info_by_topic(topic_name)

    publisher_infos = list(publisher_infos)

    if not publisher_infos:
        node.get_logger().warn(
            f"[QOS] TOPIC={topic_name} | PUBLISHERS=0 | "
            "Using fallback subscription QoS."
        )
        return

    for index, info in enumerate(publisher_infos, start=1):
        qos = info.qos_profile

        node.get_logger().info(
            f"[QOS] TOPIC={topic_name} | "
            f"PUBLISHER={index}/{len(publisher_infos)} | "
            f"NODE={info.node_namespace}/{info.node_name} | "
            f"RELIABILITY={reliability_to_string(qos.reliability)} | "
            f"DURABILITY={durability_to_string(qos.durability)} | "
            f"HISTORY={history_to_string(qos.history)} | "
            f"DEPTH={qos.depth}"
        )


def log_selected_qos(
    node,
    topic_name: str,
    qos: QoSProfile,
    role: str = "SUBSCRIBER",
) -> None:
    """
    Log the QoS profile selected for a ROS endpoint.

    Inputs:
        node:
            ROS 2 node providing the logger.

        topic_name:
            Topic associated with the QoS profile.

        qos:
            QoSProfile selected for the endpoint.

        role:
            Endpoint role, typically SUBSCRIBER or PUBLISHER.

    Processing:
        Formats the selected QoS configuration.

    Output:
        None.

    Called by:
        Nodes immediately before endpoint creation.

    Side effects:
        Writes one diagnostic message to the ROS logger.
    """

    node.get_logger().info(
        f"[QOS] TOPIC={topic_name} | "
        f"ROLE={role.upper()} | "
        f"SELECTED | "
        f"RELIABILITY={reliability_to_string(qos.reliability)} | "
        f"DURABILITY={durability_to_string(qos.durability)} | "
        f"HISTORY={history_to_string(qos.history)} | "
        f"DEPTH={qos.depth}"
    )


def log_qos_resolution(
    node,
    topic_name: str,
    selected_qos: QoSProfile,
    publisher_infos: Optional[Iterable] = None
) -> None:
    """
    Log both offered publisher QoS and the selected subscription QoS.

    Inputs:
        node:
            ROS 2 node providing graph access and logging.

        topic_name:
            Topic being resolved.

        selected_qos:
            QoSProfile selected for the subscription.

        publisher_infos:
            Optional pre-fetched publisher endpoint information.

    Processing:
        Logs discovered publisher QoS profiles, then logs the selected
        subscriber QoS profile.

    Output:
        None.

    Called by:
        Nodes after resolve_subscription_qos().

    Side effects:
        Writes diagnostic messages to the ROS logger.
    """

    log_publishers_qos(
        node=node,
        topic_name=topic_name,
        publisher_infos=publisher_infos,
    )

    log_selected_qos(
        node=node,
        topic_name=topic_name,
        qos=selected_qos,
        role="SUBSCRIBER",
    )
