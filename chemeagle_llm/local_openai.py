"""Local vLLM/Ollama backend using an OpenAI-compatible endpoint."""

from __future__ import annotations

from typing import Any

from .config import BackendConfig
from .errors import BackendConfigurationError
from .openai_compatible import OpenAICompatibleBackend


class LocalOpenAIBackend(OpenAICompatibleBackend):
    provider_name = "local"

    def _build_client(self) -> Any:
        if not self.config.base_url:
            raise BackendConfigurationError("Local backend requires VLLM_BASE_URL")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise BackendConfigurationError(
                "Local backend requires the 'openai' Python package"
            ) from exc
        return OpenAI(
            base_url=self.config.base_url,
            api_key=self.config.api_key or "EMPTY",
            timeout=self.config.timeout,
            max_retries=0,
        )
