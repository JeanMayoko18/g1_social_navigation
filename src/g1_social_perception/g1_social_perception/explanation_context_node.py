"""
Build structured explainability context from navigation decision requests.

Responsibility:
    Convert deterministic SocialNavigationDecision events into a stable,
    provider-independent ExplanationContext message.

Flow:
    INPUT
        /explainability/decision_request
            ↓
    VALIDATION
            ↓
    LANGUAGE RESOLUTION
            ↓
    SEMANTIC CONTEXT CONSTRUCTION
            ↓
    RESULT OBJECT
        ExplanationContext
            ↓
    OUTPUT
        /explainability/context

Important:
    This node does not:
        - make navigation decisions,
        - modify navigation actions,
        - call an LLM,
        - generate final natural-language speech,
        - perform TTS.

    Its responsibility is only to prepare deterministic context for the
    downstream explainability providers.
"""

from g1_social_common.navigation_states import (
    is_inactive_navigation_state,
    is_navigating_state,
)

import math

import rclpy
from rclpy.node import Node

from g1_social_msgs.msg import (
    ExplanationContext,
    SocialNavigationDecision,
)

from g1_qos_manager.diagnostics import log_qos_resolution
from g1_qos_manager.profiles import reliable_qos
from g1_qos_manager.resolver import resolve_subscription_qos


# =============================================================================
# CONSTANTS
# =============================================================================


ACTION_REPLAN = "REPLAN"
ACTION_KEEP_PATH = "KEEP_PATH"

REASON_SOCIAL_ZONE_INTRUSION = "SOCIAL_ZONE_INTRUSION"
REASON_PATH_CLEAR = "PATH_CLEAR"

EVENT_INTRUSION_CONFIRMED = "INTRUSION_CONFIRMED"


# =============================================================================
# NODE
# =============================================================================


class ExplanationContextNode(Node):
    """
    Convert explainable navigation decisions into structured LLM context.

    Input:
        /explainability/decision_request

    Output:
        /explainability/context
    """

    def __init__(self):
        super().__init__(
            "explanation_context"
        )

        # =====================================================================
        # INPUT / CONFIGURATION
        # =====================================================================

        self.declare_parameter(
            "input_topic",
            "/explainability/decision_request",
        )

        self.declare_parameter(
            "output_topic",
            "/explainability/context",
        )

        self.declare_parameter(
            "language",
            "es",
        )

        self.declare_parameter(
            "fallback_language",
            "es",
        )

        self.declare_parameter(
            "supported_languages",
            ["es", "en", "fr"],
        )
        self.declare_parameter(
            "default_language",
            "es",
        )

        self.declare_parameter(
            "include_measurements",
            True,
        )

        self.declare_parameter(
            "brief_explanations",
            True,
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

        self.language = (
            self.get_parameter("language")
            .get_parameter_value()
            .string_value
            .lower()
        )

        self.fallback_language = (
            self.get_parameter("fallback_language")
            .get_parameter_value()
            .string_value
            .lower()
        )

        self.include_measurements = (
            self.get_parameter("include_measurements")
            .get_parameter_value()
            .bool_value
        )

        self.brief_explanations = (
            self.get_parameter("brief_explanations")
            .get_parameter_value()
            .bool_value
        )

        self.supported_languages = {
            str(language).strip().lower()
            for language in self.get_parameter(
                "supported_languages"
            ).value
            if str(language).strip()
        }

        self.default_language = (
            str(
                self.get_parameter(
                    "default_language"
                ).value
            )
            .strip()
            .lower()
        )

        if not self.supported_languages:
            raise RuntimeError(
                "Parameter 'supported_languages' must not be empty."
            )

        if self.default_language not in self.supported_languages:
            raise RuntimeError(
                "Parameter 'default_language' must belong to "
                "'supported_languages'."
            )

        # =====================================================================
        # VALIDATION
        # =====================================================================

        if self.fallback_language not in self.supported_languages:
            self.get_logger().warn(
                "[EXPLANATION CONTEXT] "
                f"Unsupported fallback_language="
                f"'{self.fallback_language}'. "
                "Falling back to 'es'."
            )

            self.fallback_language = "es"

        if self.language not in self.supported_languages:
            self.get_logger().warn(
                "[EXPLANATION CONTEXT] "
                f"Unsupported language='{self.language}'. "
                f"Using fallback='{self.fallback_language}'."
            )

            self.language = self.fallback_language

        # =====================================================================
        # RUNTIME STATE / DIAGNOSTICS
        # =====================================================================

        self.decisions_received = 0
        self.contexts_published = 0
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
            ExplanationContext,
            self.output_topic,
            output_qos,
        )

        self.diagnostics_timer = self.create_timer(
            5.0,
            self.log_diagnostics,
        )

        # =====================================================================
        # REPORTING
        # =====================================================================

        self.get_logger().info("")
        self.get_logger().info("=" * 84)
        self.get_logger().info(
            "  EXPLANATION CONTEXT - INITIALIZATION"
        )
        self.get_logger().info("=" * 84)

        self.get_logger().info(
            f"  Decision input        : {self.input_topic}"
        )

        self.get_logger().info(
            f"  Context output        : {self.output_topic}"
        )

        self.get_logger().info(
            f"  Language              : {self.language}"
        )

        self.get_logger().info(
            f"  Include measurements  : {self.include_measurements}"
        )

        self.get_logger().info(
            f"  Brief explanations    : {self.brief_explanations}"
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
        Process one explanation request.

        Inputs:
            SocialNavigationDecision already filtered by explanation_trigger.

        Processing:
            - Validate deterministic decision fields.
            - Resolve semantic situation summary.
            - Resolve robot intention.
            - Resolve explanation goal.
            - Build ExplanationContext.

        Output:
            ExplanationContext on /explainability/context.

        Side effects:
            Publishes context and updates diagnostics counters.
        """

        self.decisions_received += 1

        if not self._is_valid_decision(
            msg
        ):
            self.invalid_decisions += 1
            return

        context = self._build_context(
            msg
        )

        self.publisher.publish(
            context
        )

        self.contexts_published += 1

        self.get_logger().info(
            "[EXPLANATION CONTEXT] "
            f"PUBLISHED={context.decision_id} | "
            f"LANG={context.language} | "
            f"ACTION={context.action} | "
            f"REASON={context.reason_code} | "
            f"STATE={context.navigation_state} | "
            f"GOAL={context.navigation_goal_id or 'NONE'} | "
            f"ACTIVE={context.navigation_active} | "
            f"REPLAN={context.requires_replan}"
        )

    # =========================================================================
    # PROCESSING / CONTEXT CONSTRUCTION
    # =========================================================================

    def _build_context(
        self,
        msg: SocialNavigationDecision,
    ) -> ExplanationContext:
        """
        Build one provider-independent explainability context.

        Inputs:
            Deterministic navigation decision.

        Output:
            ExplanationContext.

        Called by:
            decision_callback().

        Side effects:
            None.
        """

        context = ExplanationContext()

        # ---------------------------------------------------------------------
        # ORIGINAL DECISION DATA
        # ---------------------------------------------------------------------

        context.header = msg.header
        context.decision_id = msg.decision_id

        # Preserve the navigation snapshot associated with this observation.
        context.navigation_state = msg.navigation_state
        context.navigation_goal_id = msg.navigation_goal_id
        context.navigation_active = msg.navigation_active
        context.social_conflict_active = msg.social_conflict_active

        context.language = (
            self.language
        )

        # ExplanationContext still exposes the legacy semantic fields expected
        # by downstream explanation providers. Derive them from factual event data.

        if msg.navigation_active and msg.navigation_goal_id.strip():
            context.action = "REPLAN"
            context.requires_replan = True
        else:
            context.action = "OBSERVE"
            context.requires_replan = False

        context.reason_code = "SOCIAL_ZONE_INTRUSION"

        context.person_id = (
            msg.person_id
        )

        context.zone_type = (
            msg.zone_type
        )

        context.tracking_state = (
            msg.tracking_state
        )

        context.person_position = (
            msg.person_position
        )

        context.person_distance = float(
            msg.person_distance
        )

        context.social_radius = float(
            msg.social_radius
        )

        context.person_speed = float(
            msg.person_speed
        )

        context.occupancy_ratio = float(
            msg.occupancy_ratio
        )

        context.requires_replan = (
            msg.navigation_active
            and bool(msg.navigation_goal_id.strip())
        )

        # ---------------------------------------------------------------------
        # EXPLAINABILITY SEMANTICS
        # ---------------------------------------------------------------------

        context.situation_summary = (
            self._build_situation_summary(
                msg
            )
        )

        context.robot_intention = (
            self._build_robot_intention(
                msg
            )
        )

        context.explanation_goal = (
            self._build_explanation_goal(
                msg
            )
        )

        return context

    # =========================================================================
    # PROCESSING / SEMANTIC CONTEXT
    # =========================================================================

    def _is_speech_context_consistent(
        self,
        msg: SocialNavigationDecision,
    ) -> bool:
        """Accept a confirmed intrusion independently of Nav2 activity."""

        return (
            msg.event == EVENT_INTRUSION_CONFIRMED
            and msg.social_conflict_active
            and msg.requires_explanation
            and bool(msg.person_id.strip())
        )
    
    def _build_situation_summary(
        self,
        msg: SocialNavigationDecision,
    ) -> str:
        """Describe observed facts without inferring motion from KEEP_PATH."""

        if not (
            msg.social_conflict_active
            and msg.event == EVENT_INTRUSION_CONFIRMED
        ):
            return (
                "No confirmed social intrusion is supplied for explanation."
            )

        summary = (
            "A tracked person was observed inside the configured "
            "intrusion boundary."
        )

        if msg.navigation_active:
            summary += (
                " A navigation goal is executing."
            )
        else:
            summary += (
                " No navigation goal is currently executing."
            )

        if self.include_measurements:
            summary += (
                f" Observed person distance={msg.person_distance:.2f} m;"
                f" effective social radius={msg.social_radius:.2f} m;"
                f" person speed={msg.person_speed:.2f} m/s;"
                f" zone type={msg.zone_type}."
            )

        return summary

    def _build_robot_intention(
        self,
        msg: SocialNavigationDecision,
    ) -> str:

        if msg.navigation_active and msg.navigation_goal_id.strip():
            return (
                "Maintain the social restriction while the current navigation "
                "goal is handled by the collaboration and navigation layers."
            )

        return (
            "Acknowledge the person's presence without assuming that a "
            "navigation goal is currently executing."
        )

    def _build_explanation_goal(
        self,
        msg: SocialNavigationDecision,
    ) -> str:
        """Build a language instruction consistent with the goal lifecycle."""

        language_name = (
            "Spanish"
            if self.language == "es"
            else "English"
        )

        if (
            msg.navigation_active
            and bool(msg.navigation_goal_id.strip())
        ):
            purpose = (
                "Explain that a person was detected inside the configured "
                "intrusion boundary and that an adapted path is being "
                "requested for the current navigation goal. "
                "Do not claim that replanning has already succeeded."
            )
        else:
            purpose = (
                "Explain that a person was detected inside the configured "
                "intrusion boundary while no navigation goal is executing. "
                "Explain conditionally that, if navigation starts while "
                "the person remains nearby, the robot will need to adapt "
                "its movement to maintain separation. "
                "Do not say that a new path is being requested."
            )

        length_instruction = (
            "Use one or two short natural sentences."
            if self.brief_explanations
            else "Keep the explanation concise and factual."
        )

        return (
            f"{purpose} "
            f"Respond in {language_name}. "
            f"{length_instruction} "
            "Use only the supplied facts. "
            "Do not announce that the restriction has cleared. "
            "Do not announce that navigation is resuming. "
            "Do not repeat goal reception, arrival, cancellation, "
            "or failure announcements. "
            "Do not include internal UUIDs or machine-readable state names."
        )
    
    # =========================================================================
    # VALIDATION
    # =========================================================================

    def _is_valid_decision(
        self,
        msg: SocialNavigationDecision,
    ) -> bool:
        """Validate an explanation request before context construction."""

        if not msg.decision_id.strip():
            return False

        if msg.event != EVENT_INTRUSION_CONFIRMED:
            return False

        if not msg.requires_explanation:
            return False

        if not msg.social_conflict_active:
            return False

        if not msg.person_id.strip():
            return False

        values = (
            msg.person_distance,
            msg.social_radius,
            msg.person_speed,
            msg.occupancy_ratio,
        )

        if not all(math.isfinite(float(value)) for value in values):
            return False

        return True

    # =========================================================================
    # OUTPUT / REPORTING
    # =========================================================================

    def log_diagnostics(
        self,
    ) -> None:
        """Report context-generation health."""

        self.get_logger().info(
            "[EXPLANATION CONTEXT] "
            f"DECISIONS_RX={self.decisions_received} | "
            f"CONTEXTS_TX={self.contexts_published} | "
            f"INVALID={self.invalid_decisions} | "
            f"LANG={self.language}"
        )


# =============================================================================
# ORCHESTRATION
# =============================================================================


def main(args=None):
    """Run the explanation context node."""

    rclpy.init(
        args=args
    )

    node = ExplanationContextNode()

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
