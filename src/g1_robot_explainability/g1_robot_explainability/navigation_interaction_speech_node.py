#!/usr/bin/env python3

"""Publish deterministic speech messages for navigation interactions."""

from collections import deque
from typing import Deque, Set, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)

from action_msgs.msg import GoalStatus, GoalStatusArray
from geometry_msgs.msg import PoseWithCovarianceStamped

from g1_qos_manager.profiles import reliable_qos
from g1_social_msgs.msg import ExplanationText


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

GoalEventKey = Tuple[str, str]


class NavigationInteractionSpeechNode(Node):
    """
    Publish deterministic speech messages for navigation events.

    Supported events:
        - RViz initial pose received;
        - navigation goal received;
        - navigation goal succeeded;
        - navigation goal aborted;
        - navigation goal canceled.

    Navigation goals are tracked by UUID. Each semantic event is published
    at most once for a given goal UUID.
    """

    # -----------------------------------------------------------------------
    # Semantic goal events
    # -----------------------------------------------------------------------

    EVENT_RECEIVED = "RECEIVED"
    EVENT_SUCCEEDED = "SUCCEEDED"
    EVENT_ABORTED = "ABORTED"
    EVENT_CANCELED = "CANCELED"

    # -----------------------------------------------------------------------
    # Supported languages
    # -----------------------------------------------------------------------

    LANGUAGE_ENGLISH = "en"
    LANGUAGE_SPANISH = "es"

    SUPPORTED_LANGUAGES = {
        LANGUAGE_ENGLISH,
        LANGUAGE_SPANISH,
    }

    def __init__(self) -> None:
        """Initialize parameters, internal state, and ROS interfaces."""

        super().__init__("navigation_interaction_speech")

        self._declare_parameters()
        self._read_parameters()
        self._initialize_state()
        self._create_ros_interfaces()
        self._log_configuration()

    # -----------------------------------------------------------------------
    # Parameter configuration
    # -----------------------------------------------------------------------

    def _declare_parameters(self) -> None:
        """Declare all configurable node parameters."""

        # ROS topics.
        self.declare_parameter(
            "initial_pose_topic",
            "/initialpose",
        )
        self.declare_parameter(
            "goal_status_topic",
            "/navigate_to_pose/_action/status",
        )
        self.declare_parameter(
            "output_topic",
            "/explainability/text",
        )

        # General behavior.
        self.declare_parameter(
            "language",
            self.LANGUAGE_ENGLISH,
        )
        self.declare_parameter(
            "enabled",
            True,
        )

        # Individual speech-event controls.
        self.declare_parameter(
            "enable_initial_pose_speech",
            True,
        )
        self.declare_parameter(
            "enable_goal_speech",
            True,
        )
        self.declare_parameter(
            "enable_goal_result_speech",
            True,
        )

        # Memory protection.
        self.declare_parameter(
            "maximum_remembered_goals",
            100,
        )

    def _read_parameters(self) -> None:
        """Read and validate declared parameters."""

        self.initial_pose_topic = self._get_string_parameter(
            "initial_pose_topic"
        )
        self.goal_status_topic = self._get_string_parameter(
            "goal_status_topic"
        )
        self.output_topic = self._get_string_parameter(
            "output_topic"
        )

        self.language = (
            self._get_string_parameter("language")
            .strip()
            .lower()
        )

        if self.language not in self.SUPPORTED_LANGUAGES:
            self.get_logger().warning(
                "[NAVIGATION SPEECH] "
                f"Unsupported language='{self.language}'. "
                "English will be used."
            )
            self.language = self.LANGUAGE_ENGLISH

        self.enabled = self._get_boolean_parameter(
            "enabled"
        )
        self.enable_initial_pose_speech = (
            self._get_boolean_parameter(
                "enable_initial_pose_speech"
            )
        )
        self.enable_goal_speech = (
            self._get_boolean_parameter(
                "enable_goal_speech"
            )
        )
        self.enable_goal_result_speech = (
            self._get_boolean_parameter(
                "enable_goal_result_speech"
            )
        )

        self.maximum_remembered_goals = max(
            1,
            self._get_integer_parameter(
                "maximum_remembered_goals"
            ),
        )

    def _get_string_parameter(self, name: str) -> str:
        """Return a string parameter value."""

        return (
            self.get_parameter(name)
            .get_parameter_value()
            .string_value
        )

    def _get_boolean_parameter(self, name: str) -> bool:
        """Return a boolean parameter value."""

        return (
            self.get_parameter(name)
            .get_parameter_value()
            .bool_value
        )

    def _get_integer_parameter(self, name: str) -> int:
        """Return an integer parameter value."""

        return (
            self.get_parameter(name)
            .get_parameter_value()
            .integer_value
        )

    # -----------------------------------------------------------------------
    # Internal state
    # -----------------------------------------------------------------------

    def _initialize_state(self) -> None:
        """Initialize UUID tracking and event-deduplication state."""

        # UUIDs observed in ACCEPTED or EXECUTING state.
        #
        # A terminal status is ignored unless the same UUID was first
        # observed as active. This prevents stale action statuses from being
        # announced when the node starts.
        self.observed_goal_ids: Set[str] = set()

        # UUIDs that already reached a terminal state.
        self.closed_goal_ids: Set[str] = set()

        # Semantic events already processed.
        #
        # Example:
        #     ("a51c...", "RECEIVED")
        #     ("a51c...", "SUCCEEDED")
        self.announced_goal_events: Set[GoalEventKey] = set()

        # Ordered UUID history used to limit memory consumption.
        self.goal_history: Deque[str] = deque()

        # Last processed initial-pose identifier.
        self.last_initial_pose_id = ""

    # -----------------------------------------------------------------------
    # ROS interfaces
    # -----------------------------------------------------------------------

    def _create_ros_interfaces(self) -> None:
        """Create publishers and subscriptions."""

        input_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.publisher = self.create_publisher(
            ExplanationText,
            self.output_topic,
            reliable_qos(depth=10),
        )

        self.initial_pose_subscription = self.create_subscription(
            PoseWithCovarianceStamped,
            self.initial_pose_topic,
            self.initial_pose_callback,
            input_qos,
        )

        self.goal_status_subscription = self.create_subscription(
            GoalStatusArray,
            self.goal_status_topic,
            self.goal_status_callback,
            input_qos,
        )

    def _log_configuration(self) -> None:
        """Display the effective runtime configuration."""

        self.get_logger().info("")
        self.get_logger().info(
            "=" * 80
        )
        self.get_logger().info(
            "  NAVIGATION INTERACTION SPEECH - INITIALIZATION"
        )
        self.get_logger().info(
            "=" * 80
        )
        self.get_logger().info(
            f"  Enabled                  : {self.enabled}"
        )
        self.get_logger().info(
            f"  Language                 : {self.language}"
        )
        self.get_logger().info(
            f"  Initial pose topic       : {self.initial_pose_topic}"
        )
        self.get_logger().info(
            f"  Goal status topic        : {self.goal_status_topic}"
        )
        self.get_logger().info(
            f"  Speech output topic      : {self.output_topic}"
        )
        self.get_logger().info(
            "  Initial pose speech      : "
            f"{self.enable_initial_pose_speech}"
        )
        self.get_logger().info(
            f"  Goal reception speech    : {self.enable_goal_speech}"
        )
        self.get_logger().info(
            "  Goal result speech       : "
            f"{self.enable_goal_result_speech}"
        )
        self.get_logger().info(
            "  Maximum remembered goals : "
            f"{self.maximum_remembered_goals}"
        )
        self.get_logger().info(
            "-" * 80
        )

    # -----------------------------------------------------------------------
    # Initial-pose handling
    # -----------------------------------------------------------------------

    def initial_pose_callback(
        self,
        msg: PoseWithCovarianceStamped,
    ) -> None:
        """
        Announce a new RViz 2D Pose Estimate.

        The message timestamp is used as the event identifier to avoid
        announcing the same initial-pose message multiple times.
        """

        if not self.enabled:
            return

        if not self.enable_initial_pose_speech:
            return

        initial_pose_id = self._initial_pose_identifier(
            msg
        )

        if initial_pose_id == self.last_initial_pose_id:
            return

        self.last_initial_pose_id = initial_pose_id

        self._publish_speech(
            decision_id=f"initial_pose_{initial_pose_id}",
            english_text=(
                "I have received my initial position. "
                "I am ready for navigation. "
                "Please select your destination on the map."
            ),
            spanish_text=(
                "He recibido mi posición inicial. "
                "Estoy listo para navegar. "
                "Por favor, seleccione su destino en el mapa."
            ),
            speech_kind="INITIAL_POSE",
        )

    @staticmethod
    def _initial_pose_identifier(
        msg: PoseWithCovarianceStamped,
    ) -> str:
        """Build a stable identifier from the initial-pose timestamp."""

        return (
            f"{msg.header.stamp.sec}_"
            f"{msg.header.stamp.nanosec}"
        )

    # -----------------------------------------------------------------------
    # Navigation-goal lifecycle
    # -----------------------------------------------------------------------

    def goal_status_callback(
        self,
        msg: GoalStatusArray,
    ) -> None:
        """
        Process public NavigateToPose status transitions by UUID.

        Each goal follows this logical lifecycle:

            ACCEPTED or EXECUTING
                -> remember UUID
                -> optionally announce RECEIVED once

            SUCCEEDED
                -> close UUID
                -> optionally announce SUCCEEDED once

            ABORTED
                -> close UUID
                -> optionally announce ABORTED once

            CANCELED
                -> close UUID
                -> optionally announce CANCELED once

        Status arrays may contain old entries. A terminal state is therefore
        ignored when its UUID was never observed as active by this node.
        """

        if not self.enabled:
            return

        for goal_status in msg.status_list:
            goal_id = self._goal_uuid_to_hex(
                goal_status
            )

            if not goal_id:
                continue

            status = goal_status.status

            if self._is_active_status(status):
                self._handle_active_goal(
                    goal_id
                )
                continue

            if self._is_terminal_status(status):
                self._handle_terminal_goal(
                    goal_id=goal_id,
                    status=status,
                )

    @staticmethod
    def _goal_uuid_to_hex(
        goal_status: GoalStatus,
    ) -> str:
        """Convert a ROS action goal UUID into a hexadecimal string."""

        return bytes(
            goal_status.goal_info.goal_id.uuid
        ).hex()

    @staticmethod
    def _is_active_status(status: int) -> bool:
        """Return whether a goal status represents an active goal."""

        return status in (
            GoalStatus.STATUS_ACCEPTED,
            GoalStatus.STATUS_EXECUTING,
        )

    @staticmethod
    def _is_terminal_status(status: int) -> bool:
        """Return whether a goal status represents a terminal goal."""

        return status in (
            GoalStatus.STATUS_SUCCEEDED,
            GoalStatus.STATUS_ABORTED,
            GoalStatus.STATUS_CANCELED,
        )

    def _handle_active_goal(
        self,
        goal_id: str,
    ) -> None:
        """Register an active goal and announce its reception once."""

        # A closed UUID must never become active again.
        if goal_id in self.closed_goal_ids:
            return

        first_observation = (
            goal_id not in self.observed_goal_ids
        )

        if first_observation:
            self._remember_goal(
                goal_id
            )

        if not self.enable_goal_speech:
            return

        if not first_observation:
            return

        self._announce_goal_event(
            goal_id=goal_id,
            event_name=self.EVENT_RECEIVED,
            english_text=(
                "I have received your destination. "
                "Navigation will begin now."
            ),
            spanish_text=(
                "He recibido su destino. "
                "La navegación comenzará ahora."
            ),
        )

    def _handle_terminal_goal(
        self,
        *,
        goal_id: str,
        status: int,
    ) -> None:
        """Process a terminal status for a previously active UUID."""

        # Ignore stale terminal entries already present when the node starts.
        if goal_id not in self.observed_goal_ids:
            return

        # Ignore repeated terminal states for a closed UUID.
        if goal_id in self.closed_goal_ids:
            return

        if status == GoalStatus.STATUS_SUCCEEDED:
            self._close_goal(
                goal_id=goal_id,
                event_name=self.EVENT_SUCCEEDED,
                english_text=(
                    "I have reached the destination."
                ),
                spanish_text=(
                    "He llegado al destino."
                ),
            )
            return

        if status == GoalStatus.STATUS_ABORTED:
            self._close_goal(
                goal_id=goal_id,
                event_name=self.EVENT_ABORTED,
                english_text=(
                    "I could not reach the destination. "
                    "Please send me a new goal."
                ),
                spanish_text=(
                    "No pude alcanzar el destino. "
                    "Por favor, envíeme un nuevo objetivo."
                ),
            )
            return

        if status == GoalStatus.STATUS_CANCELED:
            self._close_goal(
                goal_id=goal_id,
                event_name=self.EVENT_CANCELED,
                english_text=(
                    "The navigation goal was canceled."
                ),
                spanish_text=(
                    "El objetivo de navegación fue cancelado."
                ),
            )

    def _close_goal(
        self,
        *,
        goal_id: str,
        event_name: str,
        english_text: str,
        spanish_text: str,
    ) -> None:
        """Close a goal UUID and optionally announce its final result."""

        event_key = self._goal_event_key(
            goal_id,
            event_name,
        )

        if event_key in self.announced_goal_events:
            return

        # Close the UUID even when result speech is disabled.
        self.closed_goal_ids.add(
            goal_id
        )

        if not self.enable_goal_result_speech:
            self.announced_goal_events.add(
                event_key
            )
            return

        self._announce_goal_event(
            goal_id=goal_id,
            event_name=event_name,
            english_text=english_text,
            spanish_text=spanish_text,
        )

    # -----------------------------------------------------------------------
    # Event deduplication
    # -----------------------------------------------------------------------

    def _announce_goal_event(
        self,
        *,
        goal_id: str,
        event_name: str,
        english_text: str,
        spanish_text: str,
    ) -> None:
        """Publish one semantic event at most once for a goal UUID."""

        event_key = self._goal_event_key(
            goal_id,
            event_name,
        )

        if event_key in self.announced_goal_events:
            return

        # Register the event before publishing it. This prevents duplicate
        # publications if another status update arrives immediately.
        self.announced_goal_events.add(
            event_key
        )

        decision_id = (
            f"navigation_goal_"
            f"{event_name.lower()}_"
            f"{goal_id}"
        )

        self._publish_speech(
            decision_id=decision_id,
            english_text=english_text,
            spanish_text=spanish_text,
            speech_kind=(
                "NAVIGATION_GOAL"
                if event_name == self.EVENT_RECEIVED
                else "NAVIGATION_RESULT"
            ),
        )
        

    @staticmethod
    def _goal_event_key(
        goal_id: str,
        event_name: str,
    ) -> GoalEventKey:
        """Create the semantic deduplication key for a goal event."""

        return (
            goal_id,
            event_name,
        )

    def _remember_goal(
        self,
        goal_id: str,
    ) -> None:
        """Remember an active UUID while keeping memory bounded."""

        if goal_id in self.observed_goal_ids:
            return

        self.observed_goal_ids.add(
            goal_id
        )
        self.goal_history.append(
            goal_id
        )

        self._prune_goal_history()

    def _prune_goal_history(self) -> None:
        """Remove the oldest UUIDs when the configured limit is exceeded."""

        while (
            len(self.goal_history)
            > self.maximum_remembered_goals
        ):
            expired_goal_id = (
                self.goal_history.popleft()
            )

            self.observed_goal_ids.discard(
                expired_goal_id
            )
            self.closed_goal_ids.discard(
                expired_goal_id
            )

            self.announced_goal_events = {
                event_key
                for event_key in self.announced_goal_events
                if event_key[0] != expired_goal_id
            }

    # -----------------------------------------------------------------------
    # Speech publication
    # -----------------------------------------------------------------------

    def _publish_speech(
        self,
        decision_id: str,
        english_text: str,
        spanish_text: str,
        *,
        speech_kind: str,
    ) -> None:
        """Publish one deterministic speech-ready explanation."""

        output = ExplanationText()

        output.header.stamp = (
            self.get_clock()
            .now()
            .to_msg()
        )
        output.header.frame_id = "map"

        output.decision_id = decision_id
        output.language = self.language
        output.text = self._select_text(
            english_text=english_text,
            spanish_text=spanish_text,
        )

        output.provider = "deterministic"
        output.model = "navigation_interaction_v2"
        output.used_fallback = False
        output.used_cache = False

        output.speech_kind = speech_kind
        output.context_stamp = output.header.stamp

        self.publisher.publish(
            output
        )

        self.get_logger().info(
            "[NAVIGATION SPEECH] "
            f"PUBLISHED={decision_id} | "
            f"LANG={self.language}"
        )

    def _select_text(
        self,
        *,
        english_text: str,
        spanish_text: str,
    ) -> str:
        """Select the speech text matching the configured language."""

        if self.language == self.LANGUAGE_SPANISH:
            return spanish_text

        return english_text


def main(args=None) -> None:
    """Run the navigation interaction speech node."""

    rclpy.init(args=args)

    node = NavigationInteractionSpeechNode()

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