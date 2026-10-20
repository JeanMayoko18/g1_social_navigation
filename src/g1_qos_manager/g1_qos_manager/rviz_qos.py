"""
RViz QoS configuration generation utilities.

Responsibility:
    Resolve QoS profiles for RViz subscriptions and generate a complete
    runtime RViz configuration from a static template.

Flow:
    INPUT / CONFIGURATION
        ↓
    QOS RESOLUTION
        ↓
    RVIZ VALUE CONVERSION
        ↓
    TEMPLATE SUBSTITUTION
        ↓
    VALIDATION
        ↓
    RUNTIME CONFIGURATION OUTPUT

This module does not create its own ROS 2 node.

The caller must provide a Node instance capable of inspecting the ROS graph.
"""

from pathlib import Path
from typing import Dict

from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    ReliabilityPolicy,
)

from g1_qos_manager.resolver import resolve_subscription_qos


# =============================================================================
# INPUT / CONFIGURATION
# =============================================================================


DEFAULT_FIXED_FRAME = "base_footprint"


# =============================================================================
# RVIZ QOS VALUE CONVERSION
# =============================================================================


def _rviz_reliability(
    policy,
) -> str:
    """
    Convert ROS 2 reliability policy into RViz configuration syntax.

    Inputs:
        ROS 2 ReliabilityPolicy.

    Processing:
        Maps supported ROS reliability values to RViz strings.

    Output:
        "Best Effort" or "Reliable".

    Called by:
        _build_topic_replacements().

    Side effects:
        None.
    """

    if policy == ReliabilityPolicy.BEST_EFFORT:
        return "Best Effort"

    return "Reliable"


def _rviz_durability(
    policy,
) -> str:
    """
    Convert ROS 2 durability policy into RViz configuration syntax.

    Inputs:
        ROS 2 DurabilityPolicy.

    Output:
        "Transient Local" or "Volatile".

    Called by:
        _build_topic_replacements().

    Side effects:
        None.
    """

    if policy == DurabilityPolicy.TRANSIENT_LOCAL:
        return "Transient Local"

    return "Volatile"


def _rviz_history(
    policy,
) -> str:
    """
    Convert ROS 2 history policy into RViz configuration syntax.

    Inputs:
        ROS 2 HistoryPolicy.

    Output:
        "Keep All" or "Keep Last".

    Called by:
        _build_topic_replacements().

    Side effects:
        None.
    """

    if policy == HistoryPolicy.KEEP_ALL:
        return "Keep All"

    return "Keep Last"


# =============================================================================
# PROCESSING / QOS RESOLUTION
# =============================================================================


def _resolve_topic_qos(
    node: Node,
    topic_name: str,
    depth: int,
    prefer_sensor_data: bool,
):
    """
    Resolve a compatible subscriber QoS for one RViz topic.

    Inputs:
        node:
            ROS graph inspection node.

        topic_name:
            ROS topic to inspect.

        depth:
            Desired subscription queue depth.

        prefer_sensor_data:
            Whether SENSOR_DATA semantics should be preferred when fallback
            resolution is required.

    Processing:
        Delegates actual QoS discovery to resolve_subscription_qos().

    Output:
        Resolved QoSProfile.

    Called by:
        generate_rviz_qos_config().

    Side effects:
        Inspects ROS publisher endpoint information.
    """

    return resolve_subscription_qos(
        node,
        topic_name,
        fallback_depth=depth,
        prefer_sensor_data=prefer_sensor_data,
    )


def _build_topic_replacements(
    prefix: str,
    topic_name: str,
    depth: int,
    qos,
) -> Dict[str, str]:
    """
    Build all RViz template substitutions for one topic.

    Inputs:
        prefix:
            Template namespace such as RGB, DEPTH, YOLO or MARKERS.

        topic_name:
            ROS topic name.

        depth:
            RViz subscription queue depth.

        qos:
            Resolved ROS QoSProfile.

    Processing:
        Converts QoS policies into RViz-compatible strings.

    Output:
        Placeholder-to-value mapping.

    Called by:
        generate_rviz_qos_config().

    Side effects:
        None.
    """

    return {
        f"__{prefix}_TOPIC__":
            topic_name,

        f"__{prefix}_DEPTH__":
            str(depth),

        f"__{prefix}_RELIABILITY__":
            _rviz_reliability(
                qos.reliability
            ),

        f"__{prefix}_DURABILITY__":
            _rviz_durability(
                qos.durability
            ),

        f"__{prefix}_HISTORY__":
            _rviz_history(
                qos.history
            ),
    }


# =============================================================================
# TEMPLATE PROCESSING
# =============================================================================


def _replace_placeholders(
    text: str,
    replacements: Dict[str, str],
) -> str:
    """
    Replace placeholders inside an RViz template.

    Inputs:
        text:
            Current RViz template content.

        replacements:
            Placeholder-to-value mapping.

    Processing:
        Applies all requested substitutions.

    Output:
        Updated template text.

    Called by:
        generate_rviz_qos_config().

    Side effects:
        None.
    """

    for placeholder, value in replacements.items():
        text = text.replace(
            placeholder,
            value,
        )

    return text


def _validate_template_resolution(
    text: str,
) -> None:
    """
    Ensure no unresolved g1_qos_manager placeholders remain.

    Inputs:
        Final generated RViz configuration text.

    Processing:
        Searches for placeholder markers.

    Output:
        None.

    Called by:
        generate_rviz_qos_config().

    Raises:
        RuntimeError:
            If unresolved placeholders remain.

    Side effects:
        None.
    """

    if "__" not in text:
        return

    unresolved = {
        token
        for token in text.split()
        if token.startswith("__")
        and token.endswith("__")
    }

    if unresolved:
        unresolved_text = ", ".join(
            sorted(unresolved)
        )

        raise RuntimeError(
            "RViz template contains unresolved placeholders: "
            f"{unresolved_text}"
        )


# =============================================================================
# OUTPUT / REPORTING
# =============================================================================


def _log_topic_resolution(
    node: Node,
    topic_name: str,
    depth: int,
    qos,
) -> None:
    """
    Report the QoS selected for one RViz topic.

    Inputs:
        Resolved topic configuration.

    Output:
        None.

    Called by:
        generate_rviz_qos_config().

    Side effects:
        Writes an informational ROS log.
    """

    node.get_logger().info(
        f"[RVIZ QOS] TOPIC={topic_name} | "
        f"RELIABILITY={_rviz_reliability(qos.reliability)} | "
        f"DURABILITY={_rviz_durability(qos.durability)} | "
        f"HISTORY={_rviz_history(qos.history)} | "
        f"DEPTH={depth}"
    )


# =============================================================================
# RESULT OBJECT / PUBLIC API
# =============================================================================


def generate_rviz_qos_config(
    node: Node,
    template_path: str,
    output_path: str,
    topic_specs: Dict[str, dict],
    fixed_frame: str = DEFAULT_FIXED_FRAME,
    enable_ros4hri: bool = False,
) -> Path:
    """
    Generate an RViz configuration adapted to active ROS publishers.

    Inputs:
        node:
            ROS 2 node used to inspect publisher endpoints.

        template_path:
            Source RViz template containing placeholders.

        output_path:
            Destination of the generated runtime RViz configuration.

        topic_specs:
            Mapping between RViz placeholder prefixes and topic definitions.

            Example:

                {
                    "RGB": {
                        "topic": "/camera/camera/color/image_raw",
                        "depth": 1,
                        "prefer_sensor_data": True,
                    }
                }

        fixed_frame:
            Fixed coordinate frame used by RViz.

    Processing:
        1. Read the template.
        2. Inject the fixed frame.
        3. Resolve each topic QoS.
        4. Convert ROS QoS values into RViz strings.
        5. Replace all topic placeholders.
        6. Validate that no expected placeholders remain.
        7. Write the runtime configuration.

    Output:
        Path to the generated RViz configuration.

    Called by:
        generate_rviz_config.main().

    Side effects:
        - Inspects ROS publisher endpoints.
        - Reads the RViz template file.
        - Creates the output directory when necessary.
        - Writes the runtime RViz configuration.
        - Emits diagnostic logs.
    """

    # -------------------------------------------------------------------------
    # INPUT
    # -------------------------------------------------------------------------

    template = Path(
        template_path
    )

    output = Path(
        output_path
    )

    if not template.is_file():
        raise FileNotFoundError(
            f"RViz template does not exist: {template}"
        )

    # -------------------------------------------------------------------------
    # TEMPLATE COLLECTION
    # -------------------------------------------------------------------------

    text = template.read_text(
        encoding="utf-8"
    )

    # -------------------------------------------------------------------------
    # GLOBAL CONFIGURATION
    # -------------------------------------------------------------------------

    text = _replace_placeholders(
        text,
        {
            "__FIXED_FRAME__": fixed_frame,
        },
    )

    node.get_logger().info(
        f"[RVIZ CONFIG] FIXED_FRAME={fixed_frame}"
    )

    # -------------------------------------------------------------------------
    # OPTIONAL ROS4HRI VISUALIZATION
    # -------------------------------------------------------------------------

    ros4hri_displays = ""

    if enable_ros4hri:
        ros4hri_displays = """
    # ==========================================================================
    # ROS4HRI VISUALIZATION
    #
    # Visualization-only ROS4HRI displays.
    #
    # These plugins consume ROS4HRI data directly. They do not own perception,
    # identity management, dialogue, navigation, explanation, or speech.
    # ==========================================================================
    - Class: hri_rviz/Humans
      Enabled: true
      Name: ROS4HRI Humans
      Value: true

    - Class: hri_rviz/Skeletons3D
      Enabled: true
      Name: ROS4HRI Skeletons
      Value: true

    - Class: hri_rviz/TF_HRI
      Enabled: true
      Name: ROS4HRI TF
      Value: true
"""

    text = _replace_placeholders(
        text,
        {
            "__ROS4HRI_DISPLAYS__": ros4hri_displays,
        },
    )

    node.get_logger().info(
        "[RVIZ CONFIG] ROS4HRI displays="
        + ("enabled" if enable_ros4hri else "disabled")
    )

    # -------------------------------------------------------------------------
    # QOS RESOLUTION
    # -------------------------------------------------------------------------

    for prefix, spec in topic_specs.items():

        topic_name = spec["topic"]

        depth = int(
            spec.get(
                "depth",
                10,
            )
        )

        prefer_sensor_data = bool(
            spec.get(
                "prefer_sensor_data",
                False,
            )
        )

        qos = _resolve_topic_qos(
            node=node,
            topic_name=topic_name,
            depth=depth,
            prefer_sensor_data=prefer_sensor_data,
        )

        # ---------------------------------------------------------------------
        # TEMPLATE SUBSTITUTION
        # ---------------------------------------------------------------------

        replacements = _build_topic_replacements(
            prefix=prefix,
            topic_name=topic_name,
            depth=depth,
            qos=qos,
        )

        text = _replace_placeholders(
            text,
            replacements,
        )

        # ---------------------------------------------------------------------
        # REPORTING
        # ---------------------------------------------------------------------

        _log_topic_resolution(
            node=node,
            topic_name=topic_name,
            depth=depth,
            qos=qos,
        )

    # -------------------------------------------------------------------------
    # VALIDATION
    # -------------------------------------------------------------------------

    _validate_template_resolution(
        text
    )

    # -------------------------------------------------------------------------
    # OUTPUT
    # -------------------------------------------------------------------------

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.write_text(
        text,
        encoding="utf-8",
    )

    node.get_logger().info(
        "[RVIZ QOS] Runtime configuration generated: "
        f"{output}"
    )

    return output