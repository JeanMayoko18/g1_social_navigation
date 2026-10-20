"""
Common interface for explainability language-model providers.

Responsibility:
    Define a provider-independent contract for generating a natural-language
    explanation from a deterministic ExplanationContext.

Flow:
                         LLMProvider
                             │
              ┌──────────────┴───────────────┐
              │                              │
              ▼                              ▼
   GenerativeLLMProvider          DeterministicFallbackProvider
              │                          templates ES/EN
              │
        build_prompt()
              │
      ┌───────┼─────────┐
      ▼       ▼         ▼
   OpenAI   Gemini     Local
                        │
                        ▼
                  Qwen / llama.cpp

Important:
    Providers are explanation components only.

    They must never:
        - modify the navigation decision,
        - choose a navigation action,
        - request replanning,
        - override deterministic robot behavior.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from g1_social_msgs.msg import ExplanationContext


# =============================================================================
# RESULT OBJECTS
# =============================================================================


@dataclass
class LLMExplanationResult:
    """
    Result returned by an explainability provider.

    Attributes:
        success:
            True when a usable explanation was produced.

        text:
            Final natural-language explanation.

        provider:
            Provider identifier such as:
                openai
                gemini
                local
                deterministic

        model:
            Model identifier when applicable.

        language:
            Output language.

        error:
            Error information when generation failed.

        used_fallback:
            True when deterministic fallback generation was used.
    """

    success: bool

    text: str

    provider: str

    model: str

    language: str

    error: Optional[str] = None

    used_fallback: bool = False


# =============================================================================
# PROVIDER INTERFACE
# =============================================================================


class LLMProvider(ABC):
    """
    Abstract interface implemented by every explanation provider.

    A provider receives deterministic context and returns natural-language text.

    The interface intentionally knows nothing about ROS publishers,
    navigation actions or TTS.
    """

    provider_name = "unknown"

    def __init__(
        self,
        model_name: str = "",
    ) -> None:
        """
        Initialize provider metadata.

        Inputs:
            model_name:
                Provider-specific model identifier.

        Side effects:
            None.
        """

        self.model_name = model_name

    @abstractmethod
    def generate(
        self,
        context: ExplanationContext,
    ) -> LLMExplanationResult:
        """
        Generate a natural-language explanation.

        Inputs:
            context:
                Deterministic ExplanationContext generated upstream.

        Processing:
            Provider-specific.

        Output:
            LLMExplanationResult.

        Side effects:
            May perform a local inference or remote API call depending on the
            concrete implementation.
        """

        raise NotImplementedError

    def is_available(
        self,
    ) -> bool:
        """
        Return whether the provider is currently usable.

        Concrete providers may override this to check:
            - API credentials,
            - local model availability,
            - inference server availability,
            - required dependencies.
        """

        return True

    def get_provider_name(
        self,
    ) -> str:
        """Return provider identifier."""

        return self.provider_name

    def get_model_name(
        self,
    ) -> str:
        """Return configured model identifier."""

        return self.model_name
