"""
OpenAI explanation provider.

Responsibility:
    Generate a short natural-language explanation from deterministic
    ExplanationContext data using the OpenAI Responses API.

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


class OpenAIProvider(GenerativeLLMProvider):
    """Generate robot decision explanations using OpenAI."""

    provider_name = "openai"

    def __init__(
        self,
        model_name: str = "gpt-5-mini",
        api_key_env: str = "OPENAI_API_KEY",
        max_output_tokens: int = 120,
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

        self._client = None
        self._import_error = None

        try:
            from openai import OpenAI

            if self.api_key:
                self._client = OpenAI(
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
        """Return whether OpenAI inference can currently be used."""

        return (
            self._client is not None
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
        """
        Generate a short explanation using the OpenAI Responses API.
        """

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
            response = self._client.responses.create(
                model=self.model_name,
                instructions=(
                    "You are the explanation component of a service robot. "
                    "Explain only the deterministic navigation decision "
                    "provided to you. "
                    "Never change the action, invent sensor information, "
                    "or recommend a different navigation behavior. "
                    "Produce speech-friendly text only."
                ),
                input=prompt,
                max_output_tokens=self.max_output_tokens,
            )

            text = (
                response.output_text
                if response.output_text
                else ""
            ).strip()

            if not text:
                return LLMExplanationResult(
                    success=False,
                    text="",
                    provider=self.provider_name,
                    model=self.model_name,
                    language=context.language,
                    error="OpenAI returned an empty explanation.",
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
        self,
        context: ExplanationContext,
    ) -> str:
        """Build OpenAI-specific input from the shared semantic prompt."""

        base_prompt = self.build_prompt(
            context
        )

        return (
            f"{base_prompt}\n"
            "Avoid markdown formatting."
        )

    # =========================================================================
    # UTILITY
    # =========================================================================

    def _availability_error(
        self,
    ) -> str:
        """Describe why the provider is unavailable."""

        if self._import_error:
            return (
                "OpenAI Python SDK unavailable: "
                f"{self._import_error}"
            )

        if not self.api_key:
            return (
                f"Environment variable "
                f"{self.api_key_env} is not configured."
            )

        return "OpenAI provider is unavailable."
