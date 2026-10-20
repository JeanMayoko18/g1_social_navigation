"""
Runtime RViz configuration generator.

Responsibility:
    Coordinate the generation of a runtime RViz configuration adapted to the
    currently available ROS 2 publishers.

Flow:
    INPUT / CONFIGURATION
        ↓
    ROS GRAPH COLLECTION
        ↓
    TOPIC SPECIFICATION
        ↓
    RVIZ CONFIGURATION GENERATION
        ↓
    CLEAN SHUTDOWN

This executable does not implement QoS resolution itself.
That responsibility belongs to g1_qos_manager.rviz_qos.
"""

import argparse
import time
from typing import Dict, List

import rclpy
from rclpy.node import Node

from g1_qos_manager.rviz_qos import generate_rviz_qos_config


# =============================================================================
# NODE
# =============================================================================


class RvizQosGenerator(Node):
    """
    Temporary ROS 2 node used to inspect the ROS graph.

    Inputs:
        ROS graph publisher endpoint information.

    Processing:
        Provides access to get_publishers_info_by_topic() for the generator.

    Output:
        None directly.

    Called by:
        main().

    Side effects:
        Creates a temporary ROS 2 node while the RViz configuration is built.
    """

    def __init__(self) -> None:
        super().__init__("rviz_qos_generator")


# =============================================================================
# INPUT / CONFIGURATION
# =============================================================================


def create_argument_parser() -> argparse.ArgumentParser:
    """
    Create the command-line argument parser.

    Inputs:
        None.

    Processing:
        Defines all runtime parameters required by the RViz generator.

    Output:
        Configured ArgumentParser.

    Called by:
        main().

    Side effects:
        None.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Generate an RViz configuration adapted to active ROS 2 "
            "publisher QoS profiles."
        )
    )

    # -------------------------------------------------------------------------
    # Files
    # -------------------------------------------------------------------------

    parser.add_argument(
        "--template",
        required=True,
        help="Path to the RViz template file.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Path to the generated runtime RViz configuration.",
    )

    # -------------------------------------------------------------------------
    # ROS graph discovery
    # -------------------------------------------------------------------------

    parser.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help=(
            "Maximum time in seconds to wait for publishers before "
            "using fallback QoS profiles."
        ),
    )

    # -------------------------------------------------------------------------
    # Visualization topics
    # -------------------------------------------------------------------------

    parser.add_argument(
        "--rgb-topic",
        default="/camera/camera/color/image_raw",
        help="RGB image topic displayed in RViz.",
    )

    parser.add_argument(
        "--depth-topic",
        default="/camera/camera/aligned_depth_to_color/image_raw",
        help="Aligned depth image topic displayed in RViz.",
    )

    parser.add_argument(
        "--yolo-topic",
        default="/yolo/dbg_image",
        help="YOLO annotated image topic displayed in RViz.",
    )

    parser.add_argument(
        "--markers-topic",
        default="/person_3d_markers",
        help="3D person marker topic displayed in RViz.",
    )

    # -------------------------------------------------------------------------
    # RViz global configuration
    # -------------------------------------------------------------------------

    parser.add_argument(
        "--fixed-frame",
        default="base_footprint",
        help="RViz fixed reference frame.",
    )

    parser.add_argument(
        "--enable-ros4hri",
        default="false",
        choices=["true", "false"],
        help=(
            "Enable ROS4HRI displays in the generated "
            "RViz configuration."
        ),
    )

    return parser


def parse_arguments():
    """
    Parse application arguments while preserving ROS 2 compatibility.

    Inputs:
        Process command-line arguments.

    Processing:
        Parses application-specific arguments and ignores additional ROS 2
        arguments such as --ros-args and node remappings.

    Output:
        Parsed application arguments.

    Called by:
        main().

    Side effects:
        Prints ignored ROS-specific arguments when present.
    """

    parser = create_argument_parser()

    args, ros_args = parser.parse_known_args()

    if ros_args:
        print(
            "[RVIZ QOS] Ignoring ROS-specific CLI arguments: "
            + " ".join(ros_args)
        )

    return args


# =============================================================================
# COLLECTION
# =============================================================================


def wait_for_publishers(
    node: Node,
    topics: List[str],
    timeout_sec: float,
) -> None:
    """
    Wait for requested publishers to become visible in the ROS graph.

    Inputs:
        node:
            Temporary ROS 2 graph inspection node.

        topics:
            Topic names whose publishers should be discovered.

        timeout_sec:
            Maximum discovery duration.

    Processing:
        Repeatedly spins the node and checks publisher endpoint information.

    Output:
        None.

    Called by:
        main().

    Side effects:
        Emits warnings for topics whose publishers are not discovered before
        timeout.

    Notes:
        Missing publishers are not considered fatal because the QoS resolver
        provides deterministic fallback profiles.
    """

    deadline = time.monotonic() + timeout_sec
    remaining = set(topics)

    while remaining and time.monotonic() < deadline:
        rclpy.spin_once(
            node,
            timeout_sec=0.1,
        )

        ready_topics = {
            topic
            for topic in remaining
            if node.get_publishers_info_by_topic(topic)
        }

        remaining -= ready_topics

    for topic in sorted(remaining):
        node.get_logger().warn(
            f"[RVIZ QOS] No publisher discovered for {topic}. "
            "Fallback QoS will be used."
        )


# =============================================================================
# PROCESSING / CONFIGURATION RESOLUTION
# =============================================================================


def build_topic_specs(args) -> Dict[str, dict]:
    """
    Build RViz topic specifications.

    Inputs:
        Parsed command-line arguments.

    Processing:
        Associates each RViz template prefix with:
            - topic name,
            - desired RViz queue depth,
            - fallback QoS behavior.

    Output:
        Dictionary consumed by generate_rviz_qos_config().

    Called by:
        main().

    Side effects:
        None.
    """

    return {
        "RGB": {
            "topic": args.rgb_topic,
            "depth": 1,
            "prefer_sensor_data": True,
        },

        "DEPTH": {
            "topic": args.depth_topic,
            "depth": 1,
            "prefer_sensor_data": True,
        },

        "YOLO": {
            "topic": args.yolo_topic,
            "depth": 1,
            "prefer_sensor_data": False,
        },

        "MARKERS": {
            "topic": args.markers_topic,
            "depth": 10,
            "prefer_sensor_data": False,
        },
    }


def get_required_topics(
    topic_specs: Dict[str, dict],
) -> List[str]:
    """
    Extract topic names requiring ROS graph discovery.

    Inputs:
        Topic specification dictionary.

    Processing:
        Extracts and deduplicates topic names.

    Output:
        List of topic names.

    Called by:
        main().

    Side effects:
        None.
    """

    return list(
        dict.fromkeys(
            spec["topic"]
            for spec in topic_specs.values()
        )
    )


# =============================================================================
# OUTPUT / REPORTING
# =============================================================================


def log_configuration(
    node: Node,
    args,
    topic_specs: Dict[str, dict],
) -> None:
    """
    Report the generator configuration before processing.

    Inputs:
        ROS node, parsed arguments and topic specifications.

    Processing:
        Formats the active runtime configuration.

    Output:
        None.

    Called by:
        main().

    Side effects:
        Writes informational logs.
    """

    node.get_logger().info("")
    node.get_logger().info("=" * 78)
    node.get_logger().info(
        "  RVIZ RUNTIME CONFIGURATION GENERATOR"
    )
    node.get_logger().info("=" * 78)

    node.get_logger().info(
        f"  Template           : {args.template}"
    )

    node.get_logger().info(
        f"  Runtime output     : {args.output}"
    )

    node.get_logger().info(
        f"  Fixed frame        : {args.fixed_frame}"
    )

    node.get_logger().info(
        f"  Discovery timeout  : {args.timeout:.2f} s"
    )

    node.get_logger().info("-" * 78)

    for prefix, spec in topic_specs.items():
        node.get_logger().info(
            f"  {prefix:<10} : {spec['topic']}"
        )

    node.get_logger().info("-" * 78)


# =============================================================================
# ORCHESTRATION
# =============================================================================


def main() -> None:
    """
    Run the runtime RViz configuration generation workflow.

    Flow:
        1. Parse configuration.
        2. Initialize ROS 2.
        3. Create temporary graph inspection node.
        4. Build topic specifications.
        5. Wait for publishers.
        6. Generate the runtime RViz configuration.
        7. Destroy the temporary node.
    """

    # -------------------------------------------------------------------------
    # INPUT / CONFIGURATION
    # -------------------------------------------------------------------------

    args = parse_arguments()

    # -------------------------------------------------------------------------
    # ROS INITIALIZATION
    # -------------------------------------------------------------------------

    rclpy.init()

    node = RvizQosGenerator()

    try:
        # ---------------------------------------------------------------------
        # CONFIGURATION RESOLUTION
        # ---------------------------------------------------------------------

        topic_specs = build_topic_specs(
            args
        )

        topics = get_required_topics(
            topic_specs
        )

        log_configuration(
            node,
            args,
            topic_specs,
        )

        # ---------------------------------------------------------------------
        # COLLECTION
        # ---------------------------------------------------------------------

        wait_for_publishers(
            node=node,
            topics=topics,
            timeout_sec=args.timeout,
        )

        # ---------------------------------------------------------------------
        # PROCESSING + OUTPUT
        # ---------------------------------------------------------------------
        generate_rviz_qos_config(
            node=node,
            template_path=args.template,
            output_path=args.output,
            topic_specs=topic_specs,
            fixed_frame=args.fixed_frame,
            enable_ros4hri=(
                args.enable_ros4hri == "true"
            ),
        )

    finally:
        # ---------------------------------------------------------------------
        # CLEANUP
        # ---------------------------------------------------------------------

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()