"""
Google Gemini explanation provider.

Responsibility:
    Generate a short natural-language explanation from deterministic
    ExplanationContext data using the Gemini API.

Important:
    This provider explains an existing deterministic navigation decision.
    It must never select or modify robot navigation behavior.
"""

import os

from g1_social_msgs.msg import ExplanationContext

from g1_robot_explainability.llm_provider import (
    LLMExplanationResult,
)

from g1_robot_explainability.generative_llm_provider import (
    GenerativeLLMProvider,
)


class GeminiProvider(GenerativeLLMProvider):
    """Generate robot decision explanations using Google Gemini."""

    provider_name = "gemini"

    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        api_key_env: str = "GEMINI_API_KEY",
        max_output_tokens: int = 120,
        temperature: float = 0.2,
    ) -> None:
        super().__init__(
            model_name=model_name
        )

        self.api_key_env = api_key_env

        self.api_key = os.getenv(
            self.api_key_env,
            "",
        ).strip()

        self.max_output_tokens = max(
            32,
            int(max_output_tokens),
        )

        self.temperature = float(
            temperature
        )

        self._client = None
        self._types = None
        self._import_error = None

        try:
            from google import genai
            from google.genai import types

            self._types = types

            if self.api_key:
                self._client = genai.Client(
                    api_key=self.api_key
                )

        except Exception as exc:
            self._import_error = str(exc)

    # =========================================================================
    # PROVIDER STATUS
    # =========================================================================

    def is_available(
        self,
    ) -> bool:
        """Return whether Gemini inference can currently be used."""

        return (
            self._client is not None
            and self._types is not None
            and bool(self.api_key)
            and self._import_error is None
        )

    # =========================================================================
    # GENERATION
    # =========================================================================

    def generate(
        self,
        context: ExplanationContext,
    ) -> LLMExplanationResult:
        """Generate one speech-friendly explanation using Gemini."""

        if not self.is_available():
            return LLMExplanationResult(
                success=False,
                text="",
                provider=self.provider_name,
                model=self.model_name,
                language=context.language,
                error=self._availability_error(),
                used_fallback=False,
            )

        prompt = self._build_prompt(
            context
        )

        try:
            response = self._client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=self._types.GenerateContentConfig(
                    max_output_tokens=self.max_output_tokens,
                    temperature=self.temperature,
                ),
            )

            text = (
                response.text
                if response.text
                else ""
            ).strip()

            if not text:
                return LLMExplanationResult(
                    success=False,
                    text="",
                    provider=self.provider_name,
                    model=self.model_name,
                    language=context.language,
                    error="Gemini returned an empty explanation.",
                    used_fallback=False,
                )

            return LLMExplanationResult(
                success=True,
                text=text,
                provider=self.provider_name,
                model=self.model_name,
                language=context.language,
                error=None,
                used_fallback=False,
            )

        except Exception as exc:
            return LLMExplanationResult(
                success=False,
                text="",
                provider=self.provider_name,
                model=self.model_name,
                language=context.language,
                error=str(exc),
                used_fallback=False,
            )

    # =========================================================================
    # PROMPT CONSTRUCTION
    # =========================================================================

    @staticmethod
    def _build_prompt(
        context: ExplanationContext,
    ) -> str:
        """Build Gemini-specific input from the shared semantic prompt."""

        return GenerativeLLMProvider.build_prompt(
            context
        )

    # =========================================================================
    # UTILITY
    # =========================================================================

    def _availability_error(
        self,
    ) -> str:
        """Describe why Gemini is unavailable."""

        if self._import_error:
            return (
                "Google GenAI Python SDK unavailable: "
                f"{self._import_error}"
            )

        if not self.api_key:
            return (
                f"Environment variable "
                f"{self.api_key_env} is not configured."
            )

        return "Gemini provider is unavailable."
