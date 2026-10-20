"""
Local generative explanation provider.

Responsibility:
    Generate natural-language robot explanations using a locally hosted
    OpenAI-compatible inference server.

Supported backends:
    - llama_cpp:
        Direct llama.cpp OpenAI-compatible server.
        Availability endpoint: /health

    - open_webui:
        Open WebUI OpenAI-compatible API connected to local inference
        backends such as Ollama.
        Availability endpoint: /api/models

Flow:
    ExplanationContext
        ↓
    shared semantic prompt
        ↓
    LocalProvider
        ↓
    selected local backend
        ├── llama.cpp
        └── Open WebUI
                ├── Qwen3
                └── GPT-OSS
        ↓
    local LLM
        ↓
    LLMExplanationResult

Important:
    This provider explains an already determined navigation decision.

    It never:
        - chooses a navigation action,
        - modifies a navigation action,
        - requests replanning,
        - controls Nav2.

    Only the final assistant message content is returned as explanation text.
    Model reasoning content is intentionally ignored.
"""

import os

from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from g1_social_msgs.msg import ExplanationContext

from g1_robot_explainability.generative_llm_provider import (
    GenerativeLLMProvider,
)
from g1_robot_explainability.llm_provider import (
    LLMExplanationResult,
)


BACKEND_LLAMA_CPP = "llama_cpp"
BACKEND_OPEN_WEBUI = "open_webui"

SUPPORTED_LOCAL_BACKENDS = {
    BACKEND_LLAMA_CPP,
    BACKEND_OPEN_WEBUI,
}


class LocalProvider(GenerativeLLMProvider):
    """
    Generate explanations using a local OpenAI-compatible inference backend.

    The backend controls transport-specific behavior such as availability
    checks and authentication. The model remains independently configurable.
    """

    provider_name = "local"

    def __init__(
        self,
        model_name: str = "qwen3-4b",
        backend: str = BACKEND_LLAMA_CPP,
        base_url: str = "http://127.0.0.1:8080/v1",
        api_key: str = "local-no-key-required",
        api_key_env: str = "",
        max_output_tokens: int = 120,
        temperature: float = 0.2,
        availability_timeout_sec: float = 1.0,
        generation_timeout_sec: float = 30.0,
    ) -> None:
        """
        Initialize the local provider.

        Inputs:
            model_name:
                Model identifier exposed by the selected local backend.

            backend:
                Local inference backend:
                    - llama_cpp
                    - open_webui

            base_url:
                OpenAI-compatible API base URL.

                Examples:
                    llama.cpp:
                        http://127.0.0.1:8080/v1

                    Open WebUI:
                        http://127.0.0.1:3000/api

            api_key:
                Direct API key.

                This remains available for compatibility with local servers
                such as llama.cpp that may expect a dummy key.

            api_key_env:
                Optional environment variable containing the API key.

                When non-empty, this takes precedence over api_key.

                Example:
                    OPEN_WEBUI_API_KEY

            max_output_tokens:
                Maximum generated explanation length.

            temperature:
                Sampling temperature.

            availability_timeout_sec:
                Maximum wait for the backend availability endpoint.

            generation_timeout_sec:
                Maximum duration of one model-generation request.

        Side effects:
            Creates an OpenAI-compatible client when the SDK is available.
        """

        super().__init__(
            model_name=model_name
        )

        self.backend = (
            backend
            .strip()
            .lower()
        )

        if self.backend not in SUPPORTED_LOCAL_BACKENDS:
            raise ValueError(
                "Unsupported local backend "
                f"'{self.backend}'. "
                "Expected one of: "
                f"{sorted(SUPPORTED_LOCAL_BACKENDS)}"
            )

        self.base_url = (
            base_url.rstrip("/")
        )

        self.api_key_env = (
            api_key_env.strip()
        )

        self.api_key = (
            self._resolve_api_key(
                direct_api_key=api_key,
            )
        )

        self.max_output_tokens = max(
            32,
            int(max_output_tokens),
        )

        self.temperature = float(
            temperature
        )

        self.availability_timeout_sec = max(
            0.1,
            float(availability_timeout_sec),
        )

        self.generation_timeout_sec = max(
            1.0,
            float(generation_timeout_sec),
        )

        self._client = None
        self._import_error = None
        self._last_availability_error = ""

        if (
            self.backend == BACKEND_OPEN_WEBUI
            and not self.api_key
        ):
            self._import_error = (
                "Open WebUI requires an API key. "
                f"Environment variable "
                f"'{self.api_key_env}' is empty or unavailable."
            )
            return

        try:
            from openai import OpenAI

            self._client = OpenAI(
                base_url=self.base_url,
                api_key=(
                    self.api_key
                    or "local-no-key-required"
                ),
                timeout=self.generation_timeout_sec,
                max_retries=0,
            )

        except Exception as exc:
            self._import_error = str(exc)

    # =========================================================================
    # PROVIDER STATUS
    # =========================================================================

    def is_available(
        self,
    ) -> bool:
        """
        Check whether the configured local inference backend is reachable.

        Preserve the latest availability failure so the provider chain can
        report whether the failure came from authentication, HTTP transport,
        timeout, connectivity, or local client initialization.
        """
        self._last_availability_error = ""

        if self._client is None:
            self._last_availability_error = (
                self._import_error
                or "OpenAI-compatible client is not initialized."
            )
            return False

        if self._import_error is not None:
            self._last_availability_error = self._import_error
            return False

        if (
            self.backend == BACKEND_OPEN_WEBUI
            and not self.api_key
        ):
            self._last_availability_error = (
                "Open WebUI API key is unavailable. "
                f"Environment variable '{self.api_key_env}' is not set."
            )
            return False

        request = Request(
            self._availability_url(),
            headers=self._availability_headers(),
            method="GET",
        )

        try:
            with urlopen(
                request,
                timeout=self.availability_timeout_sec,
            ) as response:
                if 200 <= response.status < 300:
                    return True

                self._last_availability_error = (
                    f"Availability endpoint returned HTTP {response.status}."
                )
                return False

        except HTTPError as exc:
            self._last_availability_error = (
                f"Availability endpoint returned HTTP {exc.code}: "
                f"{exc.reason}"
            )
            return False

        except URLError as exc:
            self._last_availability_error = (
                f"Unable to reach availability endpoint: {exc.reason}"
            )
            return False

        except TimeoutError:
            self._last_availability_error = (
                "Availability request timed out after "
                f"{self.availability_timeout_sec:.2f} seconds."
            )
            return False

        except OSError as exc:
            self._last_availability_error = (
                f"Availability request failed: "
                f"{type(exc).__name__}: {exc}"
            )
            return False
        
    # =========================================================================
    # GENERATION
    # =========================================================================

    def generate(
        self,
        context: ExplanationContext,
    ) -> LLMExplanationResult:
        """
        Generate an explanation using the configured local model.

        Processing:
            1. Verify backend availability.
            2. Build the shared semantic prompt.
            3. Send the prompt to the selected local backend.
            4. Extract only the final assistant content.
            5. Return LLMExplanationResult.

        reasoning_content is intentionally ignored.
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
            request_kwargs = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You explain deterministic navigation "
                            "decisions made by a service robot. "
                            "You do not select or modify robot actions. "
                            "Return only the final robot explanation. "
                            "Do not expose reasoning."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                "max_tokens": self.max_output_tokens,
                "temperature": self.temperature,
            }

            # Open WebUI can forward these options to compatible local
            # backends such as Ollama/Qwen3.
            #
            # They are intentionally not sent to llama.cpp because support
            # depends on the specific llama.cpp version/model template.
            if self.backend == BACKEND_OPEN_WEBUI:
                request_kwargs["extra_body"] = {
                    "chat_template_kwargs": {
                        "enable_thinking": False,
                    },
                    "reasoning_effort": "none",
                }

            response = (
                self._client.chat.completions.create(
                    **request_kwargs
                )
            )

            text = ""

            if response.choices:
                message = (
                    response.choices[0].message
                )

                # Only final assistant content is robot speech.
                #
                # Open WebUI/Ollama may additionally expose
                # reasoning_content. It must never be published.
                text = (
                    message.content
                    or ""
                ).strip()

            if not text:
                finish_reason = (
                    response.choices[0].finish_reason
                    if response.choices
                    else "unknown"
                )

                return LLMExplanationResult(
                    success=False,
                    text="",
                    provider=self.provider_name,
                    model=self.model_name,
                    language=context.language,
                    error=(
                        "Local model returned an empty explanation. "
                        f"finish_reason={finish_reason}"
                    ),
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
    # PROMPT ADAPTER
    # =========================================================================

    def _build_prompt(
        self,
        context: ExplanationContext,
    ) -> str:
        """
        Build local-model input from the shared semantic prompt.

        Backend-specific prompt adaptations may be added here later without
        duplicating the shared navigation explanation semantics.
        """

        return self.build_prompt(
            context
        )

    # =========================================================================
    # AUTHENTICATION
    # =========================================================================

    def _resolve_api_key(
        self,
        direct_api_key: str,
    ) -> str:
        """
        Resolve the API key.

        Priority:
            1. api_key_env when configured and populated.
            2. direct api_key parameter.

        The key itself is never logged here.
        """

        if self.api_key_env:
            environment_value = (
                os.environ.get(
                    self.api_key_env,
                    "",
                )
                .strip()
            )

            if environment_value:
                return environment_value

            # For Open WebUI an explicitly configured environment variable
            # must not silently fall back to a dummy/direct key.
            if self.backend == BACKEND_OPEN_WEBUI:
                return ""

        return (
            direct_api_key
            or ""
        ).strip()

    # =========================================================================
    # BACKEND-SPECIFIC AVAILABILITY
    # =========================================================================

    def _availability_url(
        self,
    ) -> str:
        """Build the availability endpoint for the configured backend."""

        if self.backend == BACKEND_OPEN_WEBUI:
            return (
                self.base_url
                + "/models"
            )

        return self._llama_cpp_health_url()

    def _availability_headers(
        self,
    ) -> dict:
        """Build HTTP headers required by the availability request."""

        headers = {
            "Accept": "application/json",
        }

        if (
            self.backend == BACKEND_OPEN_WEBUI
            and self.api_key
        ):
            headers["Authorization"] = (
                f"Bearer {self.api_key}"
            )

        return headers

    def _llama_cpp_health_url(
        self,
    ) -> str:
        """Build llama.cpp health endpoint from the configured API URL."""

        if self.base_url.endswith("/v1"):
            return (
                self.base_url[:-3]
                + "/health"
            )

        return (
            self.base_url
            + "/health"
        )

    # =========================================================================
    # DIAGNOSTICS
    # =========================================================================

    def _availability_error(
        self,
    ) -> str:
        """Describe why local inference is unavailable."""

        if self._last_availability_error:
            return (
                "Local provider unavailable: "
                f"{self._last_availability_error}"
            )

        if self._import_error:
            return (
                "Local provider unavailable: "
                f"{self._import_error}"
            )

        if (
            self.backend == BACKEND_OPEN_WEBUI
            and not self.api_key
        ):
            return (
                "Open WebUI API key unavailable. "
                f"Set environment variable '{self.api_key_env}'."
            )

        return (
            "Local inference backend "
            f"'{self.backend}' is unavailable at "
            f"{self.base_url}."
        )
