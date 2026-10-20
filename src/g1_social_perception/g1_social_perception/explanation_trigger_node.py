"""
Filter social-navigation decisions before the explainability pipeline.

Responsibility:
    Receive deterministic SocialNavigationDecision events and forward only
    those that should generate a natural-language explanation.

Flow:
    INPUT
        /social_navigation/decision
            ↓
    VALIDATION
            ↓
    EXPLANATION FLAG CHECK
            ↓
    DUPLICATE PROTECTION
            ↓
    PRIORITY / EVENT POLICY
            ↓
    RESULT
            ↓
    OUTPUT
        /explainability/decision_request

Important:
    This node does not:
        - make navigation decisions,
        - modify navigation actions,
        - call an LLM,
        - generate natural-language text,
        - perform TTS.

    Navigation authority remains entirely in
    social_navigation_decision_node.py.
"""

from g1_social_common.navigation_states import (
    is_inactive_navigation_state,
    is_navigating_state,
)

from typing import Optional
import math
import rclpy
from rclpy.node import Node

from g1_social_msgs.msg import SocialNavigationDecision

from g1_qos_manager.diagnostics import log_qos_resolution
from g1_qos_manager.profiles import reliable_qos
from g1_qos_manager.resolver import resolve_subscription_qos


# =============================================================================
# DECISION CONSTANTS
# =============================================================================


EVENT_INTRUSION_CONFIRMED = "INTRUSION_CONFIRMED"
EVENT_OBSERVATION_LOST = "OBSERVATION_LOST"
EVENT_CLEAR = "CLEAR"


# =============================================================================
# NODE
# =============================================================================


class ExplanationTriggerNode(Node):
    """
    Forward explainable navigation events to the explainability stack.

    Input:
        /social_navigation/decision

    Output:
        /explainability/decision_request

    Only events explicitly marked with requires_explanation=True are eligible.
    """

    def __init__(self):
        super().__init__(
            "explanation_trigger"
        )

        # =====================================================================
        # INPUT / CONFIGURATION
        # =====================================================================

        self.declare_parameter(
            "input_topic",
            "/social_navigation/decision",
        )

        self.declare_parameter(
            "output_topic",
            "/explainability/decision_request",
        )

        self.declare_parameter(
            "forward_replan",
            True,
        )

        self.declare_parameter(
            "forward_presence",
            True,
        )

        self.declare_parameter(
            "forward_path_clear",
            False,
        )

        # Protect against accidental publication of the same decision twice.
        self.declare_parameter(
            "drop_duplicate_decision_ids",
            True,
        )

        # Maximum number of recently forwarded IDs kept in memory.
        self.declare_parameter(
            "decision_history_size",
            100,
        )

        # ---------------------------------------------------------------------
        # Resolve parameters
        # ---------------------------------------------------------------------

        self.input_topic = (
            self.get_parameter("input_topic")
            .get_parameter_value()
            .string_value
        )

        self.output_topic = (
            self.get_parameter("output_topic")
            .get_parameter_value()
            .string_value
        )

        self.forward_replan = (
            self.get_parameter("forward_replan")
            .get_parameter_value()
            .bool_value
        )

        self.forward_presence = (
            self.get_parameter("forward_presence")
            .get_parameter_value()
            .bool_value
        )

        self.forward_path_clear = (
            self.get_parameter("forward_path_clear")
            .get_parameter_value()
            .bool_value
        )

        self.drop_duplicate_decision_ids = (
            self.get_parameter("drop_duplicate_decision_ids")
            .get_parameter_value()
            .bool_value
        )

        self.decision_history_size = (
            self.get_parameter("decision_history_size")
            .get_parameter_value()
            .integer_value
        )

        # =====================================================================
        # VALIDATION
        # =====================================================================

        self.decision_history_size = max(
            1,
            self.decision_history_size,
        )

        # =====================================================================
        # RUNTIME STATE
        # =====================================================================

        # Ordered list is sufficient here because event frequency is low.
        self.forwarded_decision_ids = []

        self.last_forwarded_decision_id: Optional[str] = None

        # ---------------------------------------------------------------------
        # Diagnostics
        # ---------------------------------------------------------------------

        self.decisions_received = 0
        self.requests_forwarded = 0

        self.filtered_not_requested = 0
        self.filtered_action_policy = 0
        self.filtered_duplicates = 0
        self.invalid_decisions = 0

        # =====================================================================
        # QOS
        # =====================================================================

        input_qos = resolve_subscription_qos(
            self,
            self.input_topic,
            fallback_depth=10,
            prefer_sensor_data=False,
        )

        log_qos_resolution(
            self,
            self.input_topic,
            input_qos,
        )

        output_qos = reliable_qos(
            depth=10,
        )

        # =====================================================================
        # ROS INTERFACES
        # =====================================================================

        self.subscription = self.create_subscription(
            SocialNavigationDecision,
            self.input_topic,
            self.decision_callback,
            input_qos,
        )

        self.publisher = self.create_publisher(
            SocialNavigationDecision,
            self.output_topic,
            output_qos,
        )

        self.diagnostics_timer = self.create_timer(
            5.0,
            self.log_diagnostics,
        )

        # =====================================================================
        # OUTPUT / REPORTING
        # =====================================================================

        self.get_logger().info("")
        self.get_logger().info("=" * 84)
        self.get_logger().info(
            "  EXPLANATION TRIGGER - INITIALIZATION"
        )
        self.get_logger().info("=" * 84)

        self.get_logger().info(
            f"  Decision input            : {self.input_topic}"
        )

        self.get_logger().info(
            f"  Explanation request       : {self.output_topic}"
        )

        self.get_logger().info(
            f"  Forward REPLAN            : {self.forward_replan}"
        )

        self.get_logger().info(
            f"  Forward presence          : {self.forward_presence}"
        )

        self.get_logger().info(
            "  Forward PATH_CLEAR        : False "
            "(speech suppressed)"
        )

        self.get_logger().info(
            f"  Drop duplicate IDs        : "
            f"{self.drop_duplicate_decision_ids}"
        )

        self.get_logger().info(
            f"  Decision history size     : "
            f"{self.decision_history_size}"
        )

        self.get_logger().info("-" * 84)

    # =========================================================================
    # COLLECTION
    # =========================================================================

    def decision_callback(
        self,
        msg: SocialNavigationDecision,
    ) -> None:
        """
        Process one deterministic social-navigation decision.

        Inputs:
            SocialNavigationDecision.

        Processing:
            1. Validate the event.
            2. Require requires_explanation=True.
            3. Apply explanation event policy.
            4. Reject already-forwarded decision IDs.
            5. Forward the unchanged decision.

        Output:
            SocialNavigationDecision on /explainability/decision_request.

        Side effects:
            Updates event-history and diagnostics counters.
        """

        self.decisions_received += 1

        # ---------------------------------------------------------------------
        # VALIDATION
        # ---------------------------------------------------------------------

        if not self._is_valid_decision(
            msg
        ):
            self.invalid_decisions += 1
            return

        # ---------------------------------------------------------------------
        # EXPLANATION REQUEST FLAG
        # ---------------------------------------------------------------------

        if not msg.requires_explanation:
            self.filtered_not_requested += 1
            return

        # ---------------------------------------------------------------------
        # EVENT POLICY
        # ---------------------------------------------------------------------

        if not self._is_action_enabled(
            msg
        ):
            self.filtered_action_policy += 1
            return

        # ---------------------------------------------------------------------
        # DUPLICATE PROTECTION
        # ---------------------------------------------------------------------

        if self._is_duplicate(
            msg.decision_id
        ):
            self.filtered_duplicates += 1

            self.get_logger().warn(
                "[EXPLANATION TRIGGER] "
                f"Duplicate decision ignored: {msg.decision_id}",
                throttle_duration_sec=2.0,
            )

            return

        # ---------------------------------------------------------------------
        # OUTPUT
        # ---------------------------------------------------------------------

        self.publisher.publish(
            msg
        )

        self._register_forwarded_decision(
            msg.decision_id
        )

        self.requests_forwarded += 1

        self.get_logger().info(
            "[EXPLANATION TRIGGER] "
            f"FORWARDED={msg.decision_id} | "
            f"EVENT={msg.event} | "
            f"STATE={msg.navigation_state} | "
            f"ACTIVE={msg.navigation_active} | "
            f"GOAL={msg.navigation_goal_id or 'NONE'} | "
            f"PERSON={msg.person_id or 'NONE'}"
        )

    # =========================================================================
    # PROCESSING / VALIDATION
    # =========================================================================

    def _is_valid_decision(
        self,
        msg: SocialNavigationDecision,
    ) -> bool:
        """Validate a factual social-navigation event."""

        if not msg.decision_id.strip():
            return False

        if msg.event not in (
            EVENT_INTRUSION_CONFIRMED,
            EVENT_OBSERVATION_LOST,
            EVENT_CLEAR,
        ):
            return False

        # Non-speaking state-management events are valid.
        if not msg.requires_explanation:
            return True

        # Only a confirmed intrusion may request an explanation.
        if msg.event != EVENT_INTRUSION_CONFIRMED:
            return False

        if not msg.social_conflict_active:
            return False

        if not msg.person_id.strip():
            return False

        try:
            distance = float(msg.person_distance)
            social_radius = float(msg.social_radius)
        except (TypeError, ValueError, OverflowError):
            return False

        if (
            not math.isfinite(distance)
            or not math.isfinite(social_radius)
            or distance < 0.0
            or social_radius <= 0.0
        ):
            return False

        return True

    # =========================================================================
    # PROCESSING / EXPLANATION POLICY
    # =========================================================================

    def _is_action_enabled(
        self,
        msg: SocialNavigationDecision,
    ) -> bool:
        """Select factual social events eligible for explanation."""

        if not msg.requires_explanation:
            return False

        if msg.event != EVENT_INTRUSION_CONFIRMED:
            return False

        if not msg.social_conflict_active:
            return False

        # Explanation generation is independent from Nav2 activity.
        return True

    # =========================================================================
    # PROCESSING / DUPLICATE PROTECTION
    # =========================================================================

    def _is_duplicate(
        self,
        decision_id: str,
    ) -> bool:
        """
        Return True when a decision has already been forwarded.

        Duplicate filtering can be disabled through configuration.
        """

        if not self.drop_duplicate_decision_ids:
            return False

        return (
            decision_id
            in self.forwarded_decision_ids
        )

    def _register_forwarded_decision(
        self,
        decision_id: str,
    ) -> None:
        """
        Remember one successfully forwarded decision ID.

        The history is bounded to prevent unbounded memory growth.
        """

        self.last_forwarded_decision_id = (
            decision_id
        )

        self.forwarded_decision_ids.append(
            decision_id
        )

        excess = (
            len(self.forwarded_decision_ids)
            - self.decision_history_size
        )

        if excess > 0:
            del self.forwarded_decision_ids[
                :excess
            ]

    # =========================================================================
    # OUTPUT / REPORTING
    # =========================================================================

    def log_diagnostics(
        self,
    ) -> None:
        """
        Report explainability-trigger health.

        This provides a clear distinction between:
            - navigation decisions received,
            - decisions explicitly requesting explanation,
            - explanation requests actually forwarded.
        """

        self.get_logger().info(
            "[EXPLANATION TRIGGER] "
            f"DECISIONS_RX={self.decisions_received} | "
            f"REQUESTS_TX={self.requests_forwarded} | "
            f"NO_EXPLANATION={self.filtered_not_requested} | "
            f"POLICY_FILTERED={self.filtered_action_policy} | "
            f"DUPLICATES={self.filtered_duplicates} | "
            f"INVALID={self.invalid_decisions} | "
            f"LAST="
            f"{self.last_forwarded_decision_id or 'NONE'}"
        )


# =============================================================================
# ORCHESTRATION
# =============================================================================


def main(args=None):
    """Run the explanation trigger node."""

    rclpy.init(
        args=args
    )

    node = ExplanationTriggerNode()

    try:
        rclpy.spin(
            node
        )

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
