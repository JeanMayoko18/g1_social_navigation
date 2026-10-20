"""
Generate natural-language explanations from structured explainability context.

Responsibility:
    Receive deterministic ExplanationContext messages, select the configured
    explanation provider, generate an explanation, and publish the final text.
Architecture

g1_robot_explainability/
│
├── llm_provider.py
│      │
│      ├── LLMProvider
│      └── LLMExplanationResult
│
├── generative_llm_provider.py
│      │
│      └── GenerativeLLMProvider
│            └── build_prompt()
│
├── deterministic_fallback.py
│      │
│      └── DeterministicFallbackProvider
│
└── providers/
       │
       ├── openai_provider.py
       │      └── OpenAIProvider
       │
       ├── gemini_provider.py
       │      └── GeminiProvider
       │
       └── local_provider.py        
              └── LocalProvider

Flow:

    INPUT
        /explainability/context
            ↓
    VALIDATION
            ↓
    PROVIDER SELECTION
            ↓
    EXPLANATION GENERATION
            ↓
    PROVIDER FAILURE?
        ├── NO  → use provider result
        └── YES → deterministic fallback
            ↓
    RESULT
            ↓
    OUTPUT
        /explainability/text

Important:
    This node never:
        - chooses a navigation action,
        - modifies a navigation decision,
        - requests replanning,
        - changes social-navigation parameters.

    It only explains an upstream deterministic decision.
"""
import copy
from g1_social_common.navigation_states import (
    is_inactive_navigation_state,
    is_navigating_state,
)

from typing import Optional
import math
import rclpy
from rclpy.node import Node
import time

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import json
import os
from datetime import datetime, timezone

from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)

from g1_social_msgs.msg import (
    ExplanationContext,
    ExplanationText,
    NavigationState,
)

from g1_qos_manager.diagnostics import log_qos_resolution
from g1_qos_manager.profiles import reliable_qos
from g1_qos_manager.resolver import resolve_subscription_qos


from g1_robot_explainability.deterministic_fallback import (
    DeterministicFallbackProvider,
    REASON_DIALOGUE_SPEECH_UNAVAILABLE,
    REASON_NO_HUMAN_RESPONSE,
    REASON_SOCIAL_ZONE_INTRUSION,
)

from g1_robot_explainability.llm_provider import (
    LLMExplanationResult,
    LLMProvider,
)
from g1_robot_explainability.providers.openai_provider import (
    OpenAIProvider,
)

from g1_robot_explainability.providers.gemini_provider import (
    GeminiProvider,
)
from g1_robot_explainability.explanation_cache import (
    ExplanationCache,
)
from g1_robot_explainability.providers.local_provider import (
    LocalProvider,
)
from g1_robot_explainability.generative_llm_provider import (
    has_invalid_speech_perspective,
)
from dataclasses import dataclass



# =============================================================================
# PROVIDER CONSTANTS
# =============================================================================


PROVIDER_DETERMINISTIC = "deterministic"
PROVIDER_OPENAI = "openai"
PROVIDER_GEMINI = "gemini"
PROVIDER_LOCAL = "local"

SUPPORTED_PROVIDERS = {
    PROVIDER_DETERMINISTIC,
    PROVIDER_OPENAI,
    PROVIDER_GEMINI,
    PROVIDER_LOCAL,
}


@dataclass
class ProviderAttempt:
    """Result of one provider stage in the explanation chain."""

    stage: str
    success: bool
    result: Optional[LLMExplanationResult] = None
    failure_kind: str = ""
    failure_detail: str = ""

# =============================================================================
# NODE
# =============================================================================


class ExplanationNode(Node):
    """
    Generate final natural-language explanations.

    Input:
        /explainability/context

    Output:
        /explainability/text
    """

    def __init__(self):
        super().__init__(
            "explanation_node"
        )

        # =====================================================================
        # INPUT / CONFIGURATION
        # =====================================================================

        self.declare_parameter(
            "input_topic",
            "/explainability/context",
        )

        self.declare_parameter(
            "output_topic",
            "/explainability/text",
        )

        self.declare_parameter(
            "provider",
            PROVIDER_DETERMINISTIC,
        )

        self.declare_parameter(
            "fallback_enabled",
            True,
        )
        self.declare_parameter(
            "provider_diagnostics_enabled",
            True,
        )

        self.declare_parameter(
            "provider_diagnostics_file",
            "~/g1_ws/log/explainability/provider_chain.jsonl",
        )

        self.declare_parameter(
            "openai_model",
            "gpt-5-mini",
        )

        self.declare_parameter(
            "openai_api_key_env",
            "OPENAI_API_KEY",
        )

        self.declare_parameter(
            "gemini_model",
            "gemini-2.5-flash",
        )

        self.declare_parameter(
            "gemini_api_key_env",
            "GEMINI_API_KEY",
        )

        self.declare_parameter(
            "max_output_tokens",
            120,
        )

        self.declare_parameter(
            "temperature",
            0.2,
        )

        self.declare_parameter(
            "cache_enabled",
            True,
        )

        self.declare_parameter(
            "cache_file",
            "~/.cache/g1_robot_explainability/explanations.json",
        )

        self.declare_parameter(
            "local_model",
            "qwen3-4b",
        )
        self.declare_parameter(
            "local_backend",
            "llama_cpp",
        )

        self.declare_parameter(
            "local_api_key_env",
            "",
        )

        self.declare_parameter(
            "local_generation_timeout_sec",
            30.0,
        )

        self.declare_parameter(
            "max_speech_words",
            30,
        )

        self.declare_parameter(
            "local_base_url",
            "http://127.0.0.1:8080/v1",
        )

        self.declare_parameter(
            "local_api_key",
            "local-no-key-required",
        )

        self.declare_parameter(
            "local_availability_timeout_sec",
            1.0,
        )
        self.declare_parameter(
            "allow_online_fallback",
            False,
        )

        self.declare_parameter(
            "online_fallback_provider",
            PROVIDER_OPENAI,
        )

        self.declare_parameter(
            "supported_languages",
            ["es", "en", "fr"],
        )
        self.declare_parameter(
            "default_language",
            "es",
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

        self.provider_name = (
            self.get_parameter("provider")
            .get_parameter_value()
            .string_value
            .strip()
            .lower()
        )

        self.max_speech_words = int(
            self.get_parameter(
                "max_speech_words"
            ).value
        )

        if self.max_speech_words <= 0:
            raise ValueError(
                "max_speech_words must be strictly positive."
            )

        self.fallback_enabled = (
            self.get_parameter("fallback_enabled")
            .get_parameter_value()
            .bool_value
        )
        self.allow_online_fallback = (
            self.get_parameter("allow_online_fallback")
            .get_parameter_value()
            .bool_value
        )

        self.online_fallback_provider_name = (
            self.get_parameter("online_fallback_provider")
            .get_parameter_value()
            .string_value
            .strip()
            .lower()
        )

        self.provider_diagnostics_enabled = (
            self.get_parameter("provider_diagnostics_enabled")
            .get_parameter_value()
            .bool_value
        )

        self.provider_diagnostics_file = (
            self.get_parameter("provider_diagnostics_file")
            .get_parameter_value()
            .string_value
        )

        self.openai_model = (
            self.get_parameter("openai_model")
            .get_parameter_value()
            .string_value
        )

        self.openai_api_key_env = (
            self.get_parameter("openai_api_key_env")
            .get_parameter_value()
            .string_value
        )

        self.gemini_model = (
            self.get_parameter("gemini_model")
            .get_parameter_value()
            .string_value
        )

        self.gemini_api_key_env = (
            self.get_parameter("gemini_api_key_env")
            .get_parameter_value()
            .string_value
        )

        self.max_output_tokens = (
            self.get_parameter("max_output_tokens")
            .get_parameter_value()
            .integer_value
        )

        self.temperature = (
            self.get_parameter("temperature")
            .get_parameter_value()
            .double_value
        )

        self.cache_enabled = (
            self.get_parameter("cache_enabled")
            .get_parameter_value()
            .bool_value
        )

        self.cache_file = (
            self.get_parameter("cache_file")
            .get_parameter_value()
            .string_value
        )

        self.local_model = (
            self.get_parameter("local_model")
            .get_parameter_value()
            .string_value
        )

        self.local_base_url = (
            self.get_parameter("local_base_url")
            .get_parameter_value()
            .string_value
        )

        self.local_api_key = (
            self.get_parameter("local_api_key")
            .get_parameter_value()
            .string_value
        )

        self.local_availability_timeout_sec = (
            self.get_parameter(
                "local_availability_timeout_sec"
            )
            .get_parameter_value()
            .double_value
        )

        self.local_backend = (
            self.get_parameter("local_backend")
            .get_parameter_value()
            .string_value
            .strip()
            .lower()
        )

        self.local_api_key_env = (
            self.get_parameter("local_api_key_env")
            .get_parameter_value()
            .string_value
        )

        self.local_generation_timeout_sec = (
            self.get_parameter(
                "local_generation_timeout_sec"
            )
            .get_parameter_value()
            .double_value
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

        if self.provider_name not in SUPPORTED_PROVIDERS:
            self.get_logger().warn(
                "[EXPLANATION] "
                f"Unsupported provider='{self.provider_name}'. "
                "Using deterministic provider."
            )

            self.provider_name = (
                PROVIDER_DETERMINISTIC
            )

        if self.online_fallback_provider_name not in (
            PROVIDER_OPENAI,
            PROVIDER_GEMINI,
        ):
            raise ValueError(
                "online_fallback_provider must be "
                "'openai' or 'gemini'."
            )

        if (
            self.allow_online_fallback
            and self.provider_name != PROVIDER_LOCAL
        ):
            self.get_logger().warning(
                "[EXPLANATION] Online fallback is enabled but "
                f"primary provider is '{self.provider_name}'. "
                "Online fallback is only applied when the "
                "primary provider is local."
            )

        # =====================================================================
        # PROVIDERS
        # =====================================================================

        self.fallback_provider = (
            DeterministicFallbackProvider()
        )

        self.provider = (
            self._create_provider(
                self.provider_name
            )
        )

        self.online_fallback_provider = None

        if (
            self.allow_online_fallback
            and self.provider_name == PROVIDER_LOCAL
        ):
            self.online_fallback_provider = (
                self._create_provider(
                    self.online_fallback_provider_name
                )
            )

        # =====================================================================
        # RUNTIME STATE / DIAGNOSTICS
        # =====================================================================

        self.requests_received = 0

        self.provider_successes = 0
        self.fallback_uses = 0
        self.failures = 0

        self.messages_published = 0

        self.last_decision_id: Optional[str] = None

        # -------------------------------------------------------------------------
        # SEMANTIC EXPLANATION CACHE
        # -------------------------------------------------------------------------

        self.explanation_cache = ExplanationCache(
            cache_file=self.cache_file
        )

        self.cache_hits = 0
        self.cache_misses = 0

        # =====================================================================
        # LIVE NAVIGATION CONTEXT
        # =====================================================================

        self.declare_parameter(
            "navigation_state_topic",
            "/social_navigation/navigation_state",
        )
        self.declare_parameter(
            "navigation_state_timeout_sec",
            1.0,
        )
        self.declare_parameter(
            "maximum_context_age_sec",
            10.0,
        )
        self.declare_parameter(
            "generation_watchdog_timeout_sec",
            5.0,
        )
        self.navigation_state_timeout_sec = float(
            self.get_parameter(
                "navigation_state_timeout_sec"
            ).value
        )
        self.maximum_context_age_sec = float(
            self.get_parameter(
                "maximum_context_age_sec"
            ).value
        )

        self.generation_watchdog_timeout_sec = float(
            self.get_parameter(
                "generation_watchdog_timeout_sec"
            ).value
        )

        if (
            not math.isfinite(
                self.generation_watchdog_timeout_sec
            )
            or self.generation_watchdog_timeout_sec <= 0.0
        ):
            raise ValueError(
                "generation_watchdog_timeout_sec must be "
                "finite and strictly positive."
            )

        for name, value in (
            (
                "navigation_state_timeout_sec",
                self.navigation_state_timeout_sec,
            ),
            (
                "maximum_context_age_sec",
                self.maximum_context_age_sec,
            ),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"{name} must be finite and strictly positive."
                )

        self.latest_navigation_state = None
        self.navigation_state_received_at = None

        # Increment on every observed navigation-context transition.
        # This also rejects a response if the state changes and later returns
        # to its previous value while generation is running.
        self.navigation_revision = 0
        self.navigation_signature = None

        navigation_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.navigation_state_subscription = self.create_subscription(
            NavigationState,
            self.get_parameter("navigation_state_topic").value,
            self.navigation_state_callback,
            navigation_qos,
        )

        # =====================================================================
        # NON-BLOCKING GENERATION
        # =====================================================================

        # Only provider inference runs in this worker.
        # ROS callbacks and final publication remain on the ROS executor.
        self.generation_worker = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="explanation_generation",
        )

        self.generation_future = None
        self.generation_context = None
        self.generation_revision = None
        self.generation_received_at = None
        self.generation_started_at = None

        # Keep at most one pending request: the newest valid context.
        self.pending_context = None

        # Bounded process-local deduplication.
        self.seen_decision_ids = OrderedDict()
        self.maximum_remembered_decisions = 1000

        self.generation_timer = self.create_timer(
            0.05,
            self._generation_tick,
        )

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
            ExplanationContext,
            self.input_topic,
            self.context_callback,
            input_qos,
        )

        self.publisher = self.create_publisher(
            ExplanationText,
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
            "  ROBOT EXPLANATION - INITIALIZATION"
        )
        self.get_logger().info("=" * 84)

        self.get_logger().info(
            f"  Context input       : {self.input_topic}"
        )

        self.get_logger().info(
            f"  Text output         : {self.output_topic}"
        )

        self.get_logger().info(
            f"  Provider            : {self.provider_name}"
        )

        self.get_logger().info(
            f"  Fallback enabled    : {self.fallback_enabled}"
        )

        self.get_logger().info(
            f"  Online fallback     : "
            f"{self.allow_online_fallback}"
        )

        self.get_logger().info(
            f"  Online provider     : "
            f"{self.online_fallback_provider_name}"
        )

        self.get_logger().info(
            f"  Provider available  : "
            f"{self.provider.is_available()}"
        )

        self.get_logger().info("-" * 84)

    # =========================================================================
    # INPUT / COLLECTION
    # =========================================================================

    def context_callback(
        self,
        msg: ExplanationContext,
    ) -> None:
        """Queue a valid context without blocking ROS subscriptions."""
        self.requests_received += 1

        if not self._is_valid_context(msg):
            self.failures += 1
            return

        if msg.decision_id in self.seen_decision_ids:
            return

        self.seen_decision_ids[msg.decision_id] = None

        while (
            len(self.seen_decision_ids)
            > self.maximum_remembered_decisions
        ):
            self.seen_decision_ids.popitem(last=False)

        # Replace any request that has not started yet.
        # Never build an unbounded queue of obsolete explanations.
        self.pending_context = (
            msg,
            self.navigation_revision,
            time.monotonic(),
        )

    # =========================================================================
    # PROCESSING / PROVIDER SELECTION
    # =========================================================================
    
    def navigation_state_callback(
        self,
        msg: NavigationState,
    ) -> None:
        """Track navigation independently from provider inference."""
        signature = (
            msg.navigation_state,
            msg.navigation_goal_id,
            bool(msg.navigation_active),
        )

        if signature != self.navigation_signature:
            self.navigation_revision += 1
            self.navigation_signature = signature

        self.latest_navigation_state = msg
        self.navigation_state_received_at = time.monotonic()

    def _stamp_is_fresh(
        self,
        stamp,
        maximum_age: float,
    ) -> bool:
        """Check source age using the shared ROS clock."""
        stamp_ns = (
            int(stamp.sec) * 1_000_000_000
            + int(stamp.nanosec)
        )

        if stamp_ns <= 0:
            return False

        age = (
            self.get_clock().now().nanoseconds - stamp_ns
        ) / 1_000_000_000.0

        # Allow only a small clock skew into the future.
        return -0.20 <= age <= maximum_age

    def _context_matches_navigation(
        self,
        context: ExplanationContext,
    ) -> bool:
        """
        Validate an ExplanationContext against the latest navigation state.

        Semantic rules:

            OBSERVE / no public goal
                Navigation matching is unnecessary.

            WAIT / REPLAN with a retained public goal but inactive Nav2
                The latest NavigationState is treated as authoritative state,
                not as a heartbeat.

                A WAITING state may legitimately remain unchanged for several
                seconds while social interaction or speech is taking place.
                Therefore its reception timestamp must not invalidate the
                explanation merely because no newer NavigationState was emitted.

            Active navigation
                Require a recent NavigationState because the robot is physically
                executing navigation and stale state must not authorize speech
                about a possibly obsolete execution state.
        """

        goal_id = str(
            context.navigation_goal_id or ""
        ).strip()

        # ---------------------------------------------------------------------
        # No active Nav2 and no public navigation goal.
        #
        # Typical OBSERVE case.
        # ---------------------------------------------------------------------

        if (
            not context.navigation_active
            and not goal_id
        ):
            return True

        state = self.latest_navigation_state

        if state is None:
            return False

        # ---------------------------------------------------------------------
        # The explanation request itself must still be fresh.
        # ---------------------------------------------------------------------

        if not self._stamp_is_fresh(
            context.header.stamp,
            self.maximum_context_age_sec,
        ):
            return False

        state_goal_id = str(
            state.navigation_goal_id or ""
        ).strip()

        # ---------------------------------------------------------------------
        # Held public goal, but Nav2 is no longer physically active.
        #
        # IMPORTANT:
        # NavigationState is stateful here, not a heartbeat.
        #
        # Do NOT reject WAIT simply because the last WAITING publication is
        # older than navigation_state_timeout_sec.
        # ---------------------------------------------------------------------

        if not context.navigation_active:

            return (
                context.navigation_state
                == state.navigation_state
                and goal_id
                == state_goal_id
                and not bool(
                    state.navigation_active
                )
            )

        # ---------------------------------------------------------------------
        # Physically active Nav2 execution.
        #
        # Here freshness is required because stale execution state could make
        # the explanation inconsistent with what the robot is currently doing.
        # ---------------------------------------------------------------------

        if self.navigation_state_received_at is None:
            return False

        if (
            time.monotonic()
            - self.navigation_state_received_at
            > self.navigation_state_timeout_sec
        ):
            return False

        if not self._stamp_is_fresh(
            state.header.stamp,
            self.navigation_state_timeout_sec,
        ):
            return False

        return (
            context.navigation_state
            == state.navigation_state
            and goal_id
            == state_goal_id
            and bool(
                context.navigation_active
            )
            == bool(
                state.navigation_active
            )
        )

    def _generation_tick(
        self,
    ) -> None:
        """
        Collect completed inference and start the newest valid request.

        Provider generation is guarded by a node-level watchdog.

        A provider that blocks indefinitely must never block the explanation
        pipeline indefinitely. On watchdog expiration, the current provider
        attempt is abandoned logically and a deterministic fallback is generated
        immediately on the ROS thread.

        The underlying Python worker cannot be forcefully killed safely, therefore
        a new worker is created after a timeout and the stale future is detached.
        Its eventual result, if any, is ignored.
        """

        now = time.monotonic()

        # =====================================================================
        # ACTIVE GENERATION
        # =====================================================================

        if self.generation_future is not None:

            # -----------------------------------------------------------------
            # Completed normally.
            # -----------------------------------------------------------------

            if self.generation_future.done():

                future = self.generation_future
                context = self.generation_context
                revision = self.generation_revision
                received_at = self.generation_received_at

                self.generation_future = None
                self.generation_context = None
                self.generation_revision = None
                self.generation_received_at = None
                self.generation_started_at = None

                try:
                    result = future.result()

                except Exception as exc:
                    self.failures += 1

                    self.get_logger().error(
                        "[EXPLANATION] Generation failed: "
                        f"{type(exc).__name__}: {exc}"
                    )

                    result = None

                if result is not None:
                    self._publish_if_current(
                        context,
                        revision,
                        received_at,
                        result,
                    )

                # Continue below so a newer pending request may start immediately.

            # -----------------------------------------------------------------
            # Watchdog.
            # -----------------------------------------------------------------

            else:

                if self.generation_started_at is None:
                    self.generation_started_at = now

                elapsed = (
                    now
                    - self.generation_started_at
                )

                if (
                    elapsed
                    < self.generation_watchdog_timeout_sec
                ):
                    return

                context = self.generation_context
                revision = self.generation_revision
                received_at = self.generation_received_at

                self.get_logger().error(
                    "[EXPLANATION] Provider generation watchdog expired | "
                    f"decision_id="
                    f"{getattr(context, 'decision_id', '')!r} | "
                    f"elapsed={elapsed:.2f}s | "
                    "forcing deterministic fallback."
                )

                # -------------------------------------------------------------
                # Detach the blocked worker.
                #
                # Future.cancel() normally cannot stop a thread that is already
                # executing. We therefore abandon this executor and create a new
                # one for subsequent requests.
                # -------------------------------------------------------------

                stale_worker = self.generation_worker

                try:
                    self.generation_future.cancel()
                except Exception:
                    pass

                try:
                    stale_worker.shutdown(
                        wait=False,
                        cancel_futures=True,
                    )
                except TypeError:
                    # Python versions without cancel_futures support.
                    stale_worker.shutdown(
                        wait=False
                    )
                except Exception:
                    pass

                self.generation_worker = ThreadPoolExecutor(
                    max_workers=1,
                    thread_name_prefix="explanation_generation",
                )

                self.generation_future = None
                self.generation_context = None
                self.generation_revision = None
                self.generation_received_at = None
                self.generation_started_at = None

                # -------------------------------------------------------------
                # Deterministic fallback is local and bounded.
                # -------------------------------------------------------------

                try:
                    fallback_result = (
                        self.fallback_provider.generate(
                            context
                        )
                    )

                except Exception as exc:
                    self.failures += 1

                    self.get_logger().error(
                        "[EXPLANATION] Deterministic fallback after "
                        "provider watchdog failed: "
                        f"{type(exc).__name__}: {exc}"
                    )

                    fallback_result = None

                if (
                    fallback_result is not None
                    and fallback_result.success
                ):
                    self.fallback_uses += 1

                    self.get_logger().warning(
                        "[EXPLANATION CHAIN] "
                        "STAGE=deterministic | "
                        "RESULT=SUCCESS | "
                        "TRIGGER=PROVIDER_WATCHDOG"
                    )

                    self._publish_if_current(
                        context,
                        revision,
                        received_at,
                        fallback_result,
                    )

                else:
                    self.failures += 1

                # Continue below for any pending newer request.

        # =====================================================================
        # NEW REQUEST
        # =====================================================================

        if self.pending_context is None:
            return

        context, revision, received_at = (
            self.pending_context
        )

        self.pending_context = None

        if (
            revision != self.navigation_revision
            or (
                time.monotonic()
                - received_at
                > self.maximum_context_age_sec
            )
            or not self._context_matches_navigation(
                context
            )
        ):
            return

        self.generation_context = context
        self.generation_revision = revision
        self.generation_received_at = received_at
        self.generation_started_at = (
            time.monotonic()
        )

        self.generation_future = (
            self.generation_worker.submit(
                self._generate_explanation,
                context,
            )
        )

    def _build_english_fallback_text(
        self,
        context: ExplanationContext,
    ) -> str:
        """
        Build a deterministic English fallback for the exact same semantic
        explanation context.

        This fallback is intended only for the Speech Manager when the preferred
        Piper voice is unavailable.

        It never changes navigation or collaboration policy.
        """

        try:
            fallback_context = copy.deepcopy(
                context
            )

            fallback_context.language = "en"

            result = self.fallback_provider.generate(
                fallback_context
            )

        except Exception as exc:
            self.get_logger().error(
                "[EXPLANATION] Unable to build English speech fallback | "
                f"decision={context.decision_id!r} | "
                f"error={type(exc).__name__}: {exc}"
            )
            return ""

        if (
            result is None
            or not result.success
            or not str(result.text or "").strip()
        ):
            self.get_logger().warning(
                "[EXPLANATION] Deterministic English speech fallback "
                "is unavailable | "
                f"decision={context.decision_id!r}"
            )
            return ""

        fallback_text = self._compose_final_text(
            context=fallback_context,
            semantic_text=str(
                result.text
            ).strip(),
        )

        if not self._is_valid_explanation_text(
            fallback_text
        ):
            self.get_logger().warning(
                "[EXPLANATION] English speech fallback rejected by "
                "speech validation | "
                f"decision={context.decision_id!r}"
            )
            return ""

        return fallback_text

    def _publish_if_current(
        self,
        context: ExplanationContext,
        revision: int,
        received_at: float,
        result: Optional[LLMExplanationResult],
    ) -> None:
        """
        Publish a successful explanation only while its source context remains
        valid.

        The preferred explanation keeps its requested language.

        An independent deterministic English fallback is attached to the same
        ExplanationText so the Speech Manager can use it if the preferred Piper
        voice is unavailable.
        """

        # ---------------------------------------------------------------------
        # CONTEXT VALIDITY
        # ---------------------------------------------------------------------

        if (
            revision != self.navigation_revision
            or (
                time.monotonic()
                - received_at
                > self.maximum_context_age_sec
            )
            or not self._context_matches_navigation(
                context
            )
        ):
            self.get_logger().debug(
                "[EXPLANATION] Discarding obsolete generated explanation: "
                f"DECISION={context.decision_id}"
            )
            return

        # ---------------------------------------------------------------------
        # PROVIDER RESULT VALIDITY
        # ---------------------------------------------------------------------

        if (
            result is None
            or not result.success
            or not str(
                result.text or ""
            ).strip()
            or not self._is_valid_explanation_text(
                result.text
            )
        ):
            self.failures += 1

            self.get_logger().warning(
                "[EXPLANATION] Empty or unsuccessful result discarded: "
                f"DECISION={context.decision_id}"
            )
            return

        # ---------------------------------------------------------------------
        # PREFERRED TEXT
        # ---------------------------------------------------------------------

        final_text = self._compose_final_text(
            context=context,
            semantic_text=result.text.strip(),
        )

        if not self._is_valid_explanation_text(
            final_text
        ):
            self.failures += 1
            return

        preferred_language = str(
            result.language
            or context.language
            or ""
        ).strip().lower()

        # ---------------------------------------------------------------------
        # ENGLISH SPEECH FALLBACK
        # ---------------------------------------------------------------------

        fallback_text = ""

        if preferred_language != "en":
            fallback_text = (
                self._build_english_fallback_text(
                    context
                )
            )

        # English primary text is already its own fallback.
        elif final_text:
            fallback_text = final_text

        # ---------------------------------------------------------------------
        # PUBLICATION
        # ---------------------------------------------------------------------

        output = ExplanationText()

        output.header.frame_id = (
            context.header.frame_id
        )

        output.header.stamp = (
            self.get_clock().now().to_msg()
        )

        output.decision_id = (
            context.decision_id
        )

        output.language = preferred_language
        output.text = final_text

        output.fallback_language = "en"
        output.fallback_text = fallback_text

        output.provider = (
            result.provider
        )

        output.model = (
            result.model
        )

        output.used_fallback = bool(
            result.used_fallback
        )

        output.used_cache = (
            result.provider == "cache"
        )

        # Preserve the original context for downstream transaction correlation.
        output.speech_kind = (
            "SOCIAL_EXPLANATION"
        )

        output.context_stamp = (
            context.header.stamp
        )

        output.navigation_state = (
            context.navigation_state
        )

        output.navigation_goal_id = (
            context.navigation_goal_id
        )

        output.navigation_active = bool(
            context.navigation_active
        )

        output.social_conflict_active = bool(
            context.social_conflict_active
        )

        self.publisher.publish(
            output
        )

        self.messages_published += 1

        self.last_decision_id = (
            context.decision_id
        )

        self.get_logger().info(
            "[EXPLANATION] "
            f"DECISION={context.decision_id} | "
            f"STATE={context.navigation_state} | "
            f"GOAL={context.navigation_goal_id or 'NONE'} | "
            f"PROVIDER={result.provider} | "
            f"LANG={preferred_language} | "
            f"EN_FALLBACK={'YES' if fallback_text else 'NO'}"
        )

        self.get_logger().info(
            f"[EXPLANATION TEXT] {final_text}"
        )

    def stop_generation(self) -> None:
        """Stop accepting work when the ROS node shuts down."""
        self.generation_timer.cancel()
        self.pending_context = None

        if self.generation_future is not None:
            self.generation_future.cancel()

        # Running network calls must finish before destroying the node.
        # Provider clients must therefore use finite network timeouts.
        self.generation_worker.shutdown(wait=True)

    def _is_valid_explanation_text(
        self,
        text: str,
    ) -> bool:
        """
        Validate speech-facing social explanation text.

        Social explanations are real-time speech barriers. They must remain
        concise and use the required conversational perspective.
        """
        normalized = text.strip()

        if not normalized:
            return False

        if has_invalid_speech_perspective(normalized):
            self.get_logger().warn(
                "[EXPLANATION] Rejected text with invalid speech perspective: "
                f"{normalized}"
            )
            return False

        word_count = len(normalized.split())

        if word_count > self.max_speech_words:
            self.get_logger().warn(
                "[EXPLANATION] Rejected overly long speech: "
                f"WORDS={word_count} | "
                f"MAX={self.max_speech_words} | "
                f"TEXT={normalized}"
            )
            return False

        return True

    def _create_provider(
        self,
        provider_name: str,
    ) -> LLMProvider:
        """
        Instantiate the configured explanation provider.

        Inputs:
            Provider identifier.

        Output:
            LLMProvider implementation.

        Notes:
            OpenAI, Gemini and local providers will be added incrementally.
            Until then, unavailable providers automatically fall back to the
            deterministic implementation.
        """

        if provider_name == PROVIDER_DETERMINISTIC:
            return self.fallback_provider

        if provider_name == PROVIDER_OPENAI:
            return self._create_openai_provider()

        if provider_name == PROVIDER_GEMINI:
            return self._create_gemini_provider()

        if provider_name == PROVIDER_LOCAL:
            return self._create_local_provider()

        return self.fallback_provider

    def _compose_final_text(
        self,
        context: ExplanationContext,
        semantic_text: str,
    ) -> str:
        """
        Compose the reusable semantic explanation with current measurements.

        Measurements are added after provider generation/cache lookup so that
        cached semantic explanations never contain stale numerical values.

        If adding measurements would exceed the speech budget, preserve the
        semantic human-first explanation and omit the measurements.
        
        This composition is provider-independent:
                - OpenAI
                - Gemini
                - Local
                - semantic cache
                - deterministic fallback

        all receive the current measurements in exactly the same way.
        """

        semantic_text = semantic_text.strip()

        measurement_text = self._build_measurement_text(
            context
        ).strip()

        if not measurement_text:
            return semantic_text

        candidate = (
            f"{semantic_text} {measurement_text}"
        ).strip()

        if len(candidate.split()) <= self.max_speech_words:
            return candidate

        self.get_logger().debug(
            "[EXPLANATION] Measurements omitted from speech "
            "because the final text would exceed "
            f"MAX_WORDS={self.max_speech_words}."
        )

        return semantic_text

    @staticmethod
    def _build_measurement_text(
        context: ExplanationContext,
    ) -> str:
        """
        Build a non-cacheable sentence from current sensor measurements.
        """

        if (
            context.reason_code.strip().upper()
            != REASON_SOCIAL_ZONE_INTRUSION
        ):
            return ""

        try:
            distance = float(
                context.person_distance
            )

            social_radius = float(
                context.social_radius
            )

        except (TypeError, ValueError):
            return ""

        if (
            not math.isfinite(distance)
            or not math.isfinite(social_radius)
            or distance < 0.0
            or social_radius <= 0.0
        ):
            return ""

        language = (
            context.language
            .strip()
            .lower()
        )

        if language == "en":
            return (
                f"You are about {distance:.2f} meters away; "
                f"my social boundary is {social_radius:.2f} meters."
            )

        if language == "fr":
            return (
                f"Vous êtes à environ {distance:.2f} mètres ; "
                f"ma limite sociale est de {social_radius:.2f} mètres."
            )

        return (
            f"Estás aproximadamente a {distance:.2f} metros; "
            f"mi límite social es de {social_radius:.2f} metros."
        )

    @staticmethod
    def _stamp_nanoseconds(stamp) -> int:
        """Convert a ROS timestamp into integer nanoseconds."""
        return (
            int(stamp.sec) * 1_000_000_000
            + int(stamp.nanosec)
        )

    def _create_openai_provider(
        self,
    ) -> LLMProvider:
        """Create the configured OpenAI explanation provider."""

        return OpenAIProvider(
            model_name=self.openai_model,
            api_key_env=self.openai_api_key_env,
            max_output_tokens=self.max_output_tokens,
        )

    def _create_gemini_provider(
        self,
    ) -> LLMProvider:
        """Create the configured Gemini explanation provider."""

        return GeminiProvider(
            model_name=self.gemini_model,
            api_key_env=self.gemini_api_key_env,
            max_output_tokens=self.max_output_tokens,
            temperature=self.temperature,
        )

    def _create_local_provider(
        self,
    ) -> LLMProvider:
        """Create the configured local explanation provider."""

        return LocalProvider(
            model_name=self.local_model,
            backend=self.local_backend,
            base_url=self.local_base_url,
            api_key=self.local_api_key,
            api_key_env=self.local_api_key_env,
            max_output_tokens=self.max_output_tokens,
            temperature=self.temperature,
            availability_timeout_sec=(
                self.local_availability_timeout_sec
            ),
            generation_timeout_sec=(
                self.local_generation_timeout_sec
            ),
        )

    # =========================================================================
    # PROCESSING / EXPLANATION GENERATION
    # =========================================================================

    def _try_provider(
        self,
        provider: LLMProvider,
        provider_label: str,
        context: ExplanationContext,
    ) -> ProviderAttempt:
        """Try one provider stage and preserve the exact failure reason."""

        try:
            available = provider.is_available()
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"

            self.get_logger().error(
                "[EXPLANATION CHAIN] "
                f"STAGE={provider_label} | "
                "RESULT=FAILED | "
                "KIND=AVAILABILITY_CHECK_EXCEPTION | "
                f"DETAIL={detail}"
            )

            return ProviderAttempt(
                stage=provider_label,
                success=False,
                failure_kind="AVAILABILITY_CHECK_EXCEPTION",
                failure_detail=detail,
            )

        if not available:
            availability_error = getattr(
                provider,
                "_availability_error",
                None,
            )

            if callable(availability_error):
                try:
                    detail = availability_error()
                except Exception:
                    detail = "Provider reported itself unavailable."
            else:
                detail = "Provider reported itself unavailable."

                
            self.get_logger().warning(
                "[EXPLANATION CHAIN] "
                f"STAGE={provider_label} | "
                "RESULT=FAILED | "
                "KIND=UNAVAILABLE | "
                f"DETAIL={detail}"
            )

            return ProviderAttempt(
                stage=provider_label,
                success=False,
                failure_kind="UNAVAILABLE",
                failure_detail=detail,
            )

        try:
            result = provider.generate(context)

        except TimeoutError as exc:
            detail = f"{type(exc).__name__}: {exc}"

            self.get_logger().error(
                "[EXPLANATION CHAIN] "
                f"STAGE={provider_label} | "
                "RESULT=FAILED | "
                "KIND=TIMEOUT | "
                f"DETAIL={detail}"
            )

            return ProviderAttempt(
                stage=provider_label,
                success=False,
                failure_kind="TIMEOUT",
                failure_detail=detail,
            )

        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"

            self.get_logger().error(
                "[EXPLANATION CHAIN] "
                f"STAGE={provider_label} | "
                "RESULT=FAILED | "
                "KIND=EXCEPTION | "
                f"DETAIL={detail}"
            )

            return ProviderAttempt(
                stage=provider_label,
                success=False,
                failure_kind="EXCEPTION",
                failure_detail=detail,
            )

        if result is None:
            detail = "Provider returned no result."

            self.get_logger().error(
                "[EXPLANATION CHAIN] "
                f"STAGE={provider_label} | "
                "RESULT=FAILED | "
                "KIND=NO_RESULT | "
                f"DETAIL={detail}"
            )

            return ProviderAttempt(
                stage=provider_label,
                success=False,
                failure_kind="NO_RESULT",
                failure_detail=detail,
            )

        if not result.success:
            detail = result.error or "Provider returned success=False."

            self.get_logger().warning(
                "[EXPLANATION CHAIN] "
                f"STAGE={provider_label} | "
                "RESULT=FAILED | "
                "KIND=GENERATION_FAILED | "
                f"DETAIL={detail}"
            )

            return ProviderAttempt(
                stage=provider_label,
                success=False,
                failure_kind="GENERATION_FAILED",
                failure_detail=detail,
            )

        text = result.text.strip()

        if not text:
            detail = "Provider returned empty text."

            self.get_logger().warning(
                "[EXPLANATION CHAIN] "
                f"STAGE={provider_label} | "
                "RESULT=FAILED | "
                "KIND=EMPTY_TEXT | "
                f"DETAIL={detail}"
            )

            return ProviderAttempt(
                stage=provider_label,
                success=False,
                failure_kind="EMPTY_TEXT",
                failure_detail=detail,
            )

        if not self._is_valid_explanation_text(text):
            detail = "Generated text was rejected by speech validation."

            self.get_logger().warning(
                "[EXPLANATION CHAIN] "
                f"STAGE={provider_label} | "
                "RESULT=FAILED | "
                "KIND=SPEECH_VALIDATION | "
                f"DETAIL={detail}"
            )

            return ProviderAttempt(
                stage=provider_label,
                success=False,
                failure_kind="SPEECH_VALIDATION",
                failure_detail=detail,
            )

        self.get_logger().info(
            "[EXPLANATION CHAIN] "
            f"STAGE={provider_label} | "
            "RESULT=SUCCESS | "
            f"MODEL={result.model or 'NONE'}"
        )

        return ProviderAttempt(
            stage=provider_label,
            success=True,
            result=result,
        )

    def _store_provider_result_in_cache(
        self,
        context: ExplanationContext,
        result: LLMExplanationResult,
    ) -> None:
        """
        Store a valid generative-provider result in the semantic cache.

        Deterministic and cache-originated results are never stored.
        """

        if not self.cache_enabled:
            return

        if result.provider in (
            "deterministic",
            "cache",
        ):
            return

        self.explanation_cache.put(
            context=context,
            text=result.text.strip(),
            provider=result.provider,
            model=result.model,
        )

        self.get_logger().info(
            "[EXPLANATION CACHE] "
            f"STORED | "
            f"PROVIDER={result.provider} | "
            f"MODEL={result.model or 'NONE'} | "
            f"ACTION={context.action} | "
            f"REASON={context.reason_code} | "
            f"ZONE={context.zone_type} | "
            f"LANG={context.language}"
        )
    
    def _write_provider_diagnostic(
        self,
        *,
        context: ExplanationContext,
        stage: str,
        provider: str,
        model: str,
        result: str,
        failure_kind: str = "",
        failure_detail: str = "",
        next_stage: str = "",
    ) -> None:
        """Persist one provider-chain event as one JSONL record."""

        if not self.provider_diagnostics_enabled:
            return

        path = os.path.abspath(
            os.path.expanduser(self.provider_diagnostics_file)
        )

        try:
            os.makedirs(
                os.path.dirname(path),
                exist_ok=True,
            )

            record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "decision_id": context.decision_id,
                "language": context.language,
                "stage": stage,
                "provider": provider,
                "model": model,
                "result": result,
                "failure_kind": failure_kind,
                "failure_detail": failure_detail,
                "next_stage": next_stage,
            }

            with open(path, "a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )

        except Exception as exc:
            # Diagnostics must never break explanation generation.
            self.get_logger().error(
                "[EXPLANATION CHAIN] "
                "Unable to persist provider diagnostic: "
                f"{type(exc).__name__}: {exc}"
            )
            
    def _generate_explanation(
        self,
        context: ExplanationContext,
    ) -> Optional[LLMExplanationResult]:
        """
        Generate an explanation using the configured provider chain.

        Provider policy:
            1. Semantic cache, when enabled.
            2. Primary configured provider.
            3. Optional online fallback, only when:
                - the primary provider is local,
                - allow_online_fallback is true.
            4. Deterministic fallback, when enabled.

        Every provider-chain transition is persisted to the JSONL diagnostic
        file when provider diagnostics are enabled.

        Online inference is explicit opt-in. Merely having OpenAI or Gemini
        credentials in the process environment never authorizes online use.

        Provider failures affect explanation generation only. They never
        modify or delay the upstream navigation decision.
        """

        # =====================================================================
        # SEMANTIC CACHE
        # =====================================================================

        if self.cache_enabled:
            cached = self.explanation_cache.get(context)

            if cached is not None:
                cached_text = str(
                    cached.get("text", "")
                ).strip()

                if self._is_valid_explanation_text(cached_text):
                    self.cache_hits += 1

                    self.get_logger().info(
                        "[EXPLANATION CACHE] "
                        f"HIT | "
                        f"ACTION={context.action} | "
                        f"REASON={context.reason_code} | "
                        f"ZONE={context.zone_type} | "
                        f"LANG={context.language}"
                    )

                    self._write_provider_diagnostic(
                        context=context,
                        stage="cache",
                        provider="cache",
                        model=str(cached.get("model", "")),
                        result="SUCCESS",
                        next_stage="publish",
                    )

                    return LLMExplanationResult(
                        success=True,
                        text=cached_text,
                        provider="cache",
                        model=cached.get("model", ""),
                        language=context.language,
                        error=None,
                        used_fallback=False,
                    )

                self.get_logger().warning(
                    "[EXPLANATION CACHE] "
                    "Ignoring cached text with invalid speech perspective."
                )

                self._write_provider_diagnostic(
                    context=context,
                    stage="cache",
                    provider="cache",
                    model=str(cached.get("model", "")),
                    result="FAILED",
                    failure_kind="SPEECH_VALIDATION",
                    failure_detail=(
                        "Cached text was rejected by speech validation."
                    ),
                    next_stage=self.provider_name,
                )

            self.cache_misses += 1

            self.get_logger().info(
                "[EXPLANATION CACHE] "
                f"MISS | "
                f"ACTION={context.action} | "
                f"REASON={context.reason_code} | "
                f"ZONE={context.zone_type} | "
                f"LANG={context.language}"
            )

        # =====================================================================
        # STAGE 1 - PRIMARY PROVIDER
        # =====================================================================

        primary_attempt = self._try_provider(
            provider=self.provider,
            provider_label=self.provider_name,
            context=context,
        )

        if primary_attempt.success:
            result = primary_attempt.result

            if result is None:
                # Defensive guard. _try_provider() guarantees that a successful
                # attempt contains a result, but keep the chain safe if that
                # contract changes later.
                self._write_provider_diagnostic(
                    context=context,
                    stage="primary",
                    provider=self.provider_name,
                    model="",
                    result="FAILED",
                    failure_kind="INTERNAL_CONTRACT",
                    failure_detail=(
                        "Successful provider attempt contained no result."
                    ),
                    next_stage="deterministic",
                )
                return None

            if not result.used_fallback:
                self.provider_successes += 1

            self._store_provider_result_in_cache(
                context=context,
                result=result,
            )

            self._write_provider_diagnostic(
                context=context,
                stage="primary",
                provider=result.provider or self.provider_name,
                model=result.model or "",
                result="SUCCESS",
                next_stage="publish",
            )

            return result

        # Preserve the complete chain for the final ROS diagnostic.
        failure_chain = [
            (
                primary_attempt.stage,
                primary_attempt.failure_kind,
                primary_attempt.failure_detail,
            )
        ]

        # Determine where a failed primary stage will go next.
        if (
            self.provider_name == PROVIDER_LOCAL
            and self.allow_online_fallback
            and self.online_fallback_provider is not None
        ):
            primary_next_stage = self.online_fallback_provider_name
        elif self.fallback_enabled:
            primary_next_stage = PROVIDER_DETERMINISTIC
        else:
            primary_next_stage = "failed"

        self._write_provider_diagnostic(
            context=context,
            stage="primary",
            provider=self.provider_name,
            model=(
                self.local_model
                if self.provider_name == PROVIDER_LOCAL
                else ""
            ),
            result="FAILED",
            failure_kind=primary_attempt.failure_kind,
            failure_detail=primary_attempt.failure_detail,
            next_stage=primary_next_stage,
        )

        # =====================================================================
        # STAGE 2 - OPTIONAL ONLINE FALLBACK
        # =====================================================================

        if self.provider_name == PROVIDER_LOCAL:
            if (
                self.allow_online_fallback
                and self.online_fallback_provider is not None
            ):
                self.get_logger().warning(
                    "[EXPLANATION CHAIN] "
                    f"LOCAL_FAILED={primary_attempt.failure_kind} | "
                    f"DETAIL={primary_attempt.failure_detail} | "
                    f"NEXT={self.online_fallback_provider_name}"
                )

                online_attempt = self._try_provider(
                    provider=self.online_fallback_provider,
                    provider_label=self.online_fallback_provider_name,
                    context=context,
                )

                if online_attempt.success:
                    result = online_attempt.result

                    if result is None:
                        self._write_provider_diagnostic(
                            context=context,
                            stage="online",
                            provider=self.online_fallback_provider_name,
                            model="",
                            result="FAILED",
                            failure_kind="INTERNAL_CONTRACT",
                            failure_detail=(
                                "Successful online provider attempt "
                                "contained no result."
                            ),
                            next_stage=(
                                PROVIDER_DETERMINISTIC
                                if self.fallback_enabled
                                else "failed"
                            ),
                        )

                        if not self.fallback_enabled:
                            return None

                    else:
                        self.provider_successes += 1

                        self._store_provider_result_in_cache(
                            context=context,
                            result=result,
                        )

                        self._write_provider_diagnostic(
                            context=context,
                            stage="online",
                            provider=(
                                result.provider
                                or self.online_fallback_provider_name
                            ),
                            model=result.model or "",
                            result="SUCCESS",
                            next_stage="publish",
                        )

                        return result

                else:
                    failure_chain.append(
                        (
                            online_attempt.stage,
                            online_attempt.failure_kind,
                            online_attempt.failure_detail,
                        )
                    )

                    self._write_provider_diagnostic(
                        context=context,
                        stage="online",
                        provider=self.online_fallback_provider_name,
                        model="",
                        result="FAILED",
                        failure_kind=online_attempt.failure_kind,
                        failure_detail=online_attempt.failure_detail,
                        next_stage=(
                            PROVIDER_DETERMINISTIC
                            if self.fallback_enabled
                            else "failed"
                        ),
                    )

            else:
                # Online inference is deliberately not attempted unless it has
                # been explicitly authorized by configuration.
                self.get_logger().info(
                    "[EXPLANATION CHAIN] "
                    "STAGE=online | RESULT=SKIPPED | "
                    "REASON=ONLINE_FALLBACK_NOT_AUTHORIZED"
                )

                self._write_provider_diagnostic(
                    context=context,
                    stage="online",
                    provider=self.online_fallback_provider_name,
                    model="",
                    result="SKIPPED",
                    failure_kind="NOT_AUTHORIZED",
                    failure_detail=(
                        "Online fallback is disabled by configuration."
                    ),
                    next_stage=(
                        PROVIDER_DETERMINISTIC
                        if self.fallback_enabled
                        else "failed"
                    ),
                )

        # =====================================================================
        # STAGE 3 - DETERMINISTIC FALLBACK
        # =====================================================================

        if not self.fallback_enabled:
            self._write_provider_diagnostic(
                context=context,
                stage="deterministic",
                provider=PROVIDER_DETERMINISTIC,
                model="",
                result="SKIPPED",
                failure_kind="DISABLED",
                failure_detail=(
                    "Deterministic fallback is disabled by configuration."
                ),
                next_stage="failed",
            )

            self.get_logger().error(
                "[EXPLANATION CHAIN] "
                "FINAL=FAILED | "
                "REASON=DETERMINISTIC_FALLBACK_DISABLED | "
                f"DECISION={context.decision_id} | "
                f"CHAIN={failure_chain}"
            )

            return None

        deterministic_attempt = self._try_provider(
            provider=self.fallback_provider,
            provider_label=PROVIDER_DETERMINISTIC,
            context=context,
        )

        if deterministic_attempt.success:
            result = deterministic_attempt.result

            if result is None:
                self._write_provider_diagnostic(
                    context=context,
                    stage="deterministic",
                    provider=PROVIDER_DETERMINISTIC,
                    model="",
                    result="FAILED",
                    failure_kind="INTERNAL_CONTRACT",
                    failure_detail=(
                        "Successful deterministic provider attempt "
                        "contained no result."
                    ),
                    next_stage="failed",
                )

                self.get_logger().error(
                    "[EXPLANATION CHAIN] "
                    "FINAL=FAILED | "
                    f"DECISION={context.decision_id} | "
                    "REASON=DETERMINISTIC_RESULT_MISSING"
                )

                return None

            # Do not mutate LLMExplanationResult here. The deterministic
            # provider owns the metadata describing its own result.
            self.fallback_uses += 1

            self._write_provider_diagnostic(
                context=context,
                stage="deterministic",
                provider=result.provider or PROVIDER_DETERMINISTIC,
                model=result.model or "",
                result="SUCCESS",
                next_stage="publish",
            )

            self.get_logger().warning(
                "[EXPLANATION CHAIN] "
                "FINAL=DETERMINISTIC | "
                f"DECISION={context.decision_id} | "
                f"PREVIOUS_FAILURES={failure_chain}"
            )

            return result

        failure_chain.append(
            (
                deterministic_attempt.stage,
                deterministic_attempt.failure_kind,
                deterministic_attempt.failure_detail,
            )
        )

        self._write_provider_diagnostic(
            context=context,
            stage="deterministic",
            provider=PROVIDER_DETERMINISTIC,
            model="",
            result="FAILED",
            failure_kind=deterministic_attempt.failure_kind,
            failure_detail=deterministic_attempt.failure_detail,
            next_stage="failed",
        )

        self.get_logger().error(
            "[EXPLANATION CHAIN] "
            "FINAL=FAILED | "
            f"DECISION={context.decision_id} | "
            f"CHAIN={failure_chain}"
        )

        return None
    
    # =========================================================================
    # VALIDATION
    # =========================================================================

    def _is_valid_context(
        self,
        context: ExplanationContext,
    ) -> bool:
        """
        Validate request metadata and the shared semantic contract.

        Navigation matching and freshness remain checked separately.
        """
        decision_id = str(
            context.decision_id or ""
        ).strip()

        language = str(
            context.language or ""
        ).strip().lower()

        if not decision_id:
            return False

        if language not in self.supported_languages:
            return False

        return DeterministicFallbackProvider.is_supported_context(
            context
        )
    
    # =========================================================================
    # OUTPUT / REPORTING
    # =========================================================================

    def log_diagnostics(
        self,
    ) -> None:
        """Report explanation-generation health."""

        self.get_logger().info(
            "[EXPLANATION] "
            f"REQUESTS_RX={self.requests_received} | "
            f"PROVIDER_SUCCESS={self.provider_successes} | "
            f"CACHE_HITS={self.cache_hits} | "
            f"CACHE_MISSES={self.cache_misses} | "
            f"FALLBACK={self.fallback_uses} | "
            f"FAILURES={self.failures} | "
            f"TEXT_TX={self.messages_published} | "
            f"PROVIDER={self.provider_name} | "
            f"LAST={self.last_decision_id or 'NONE'}"
        )


# =============================================================================
# TEMPORARY PROVIDER PLACEHOLDER
# =============================================================================

class UnavailableProvider(LLMProvider):
    """
    Placeholder used for providers not yet configured.

    This allows provider selection to already exist without introducing hard
    dependencies on OpenAI, Gemini or local inference libraries.
    """

    def __init__(
        self,
        provider_name: str,
        reason: str,
    ) -> None:
        super().__init__(
            model_name=""
        )

        self.provider_name = (
            provider_name
        )

        self.reason = (
            reason
        )

    def is_available(
        self,
    ) -> bool:
        """Return False because this provider is not implemented yet."""

        return False

    def generate(
        self,
        context: ExplanationContext,
    ) -> LLMExplanationResult:
        """
        Return an explicit unavailable-provider result.
        """

        return LLMExplanationResult(
            success=False,
            text="",
            provider=self.provider_name,
            model=self.model_name,
            language=context.language,
            error=self.reason,
            used_fallback=False,
        )

# =============================================================================
# ORCHESTRATION
# =============================================================================


def main(args=None):
    """Run the explanation generation node."""

    rclpy.init(
        args=args
    )

    node = ExplanationNode()

    try:
        rclpy.spin(
            node
        )

    except KeyboardInterrupt:
        pass


    finally:
        node.stop_generation()
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()
