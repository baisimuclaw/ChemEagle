"""Azure OpenAI backend with lazy credential and SDK validation."""

from __future__ import annotations

from typing import Any

from .config import BackendConfig
from .errors import BackendConfigurationError
from .openai_compatible import OpenAICompatibleBackend


class AzureOpenAIBackend(OpenAICompatibleBackend):
    provider_name = "azure"

    def _build_client(self) -> Any:
        if not self.config.api_key or not self.config.azure_endpoint:
            raise BackendConfigurationError(
                "Azure backend requires AZURE_OPENAI_API_KEY (or API_KEY) and "
                "AZURE_OPENAI_ENDPOINT (or AZURE_ENDPOINT)"
            )
        try:
            from openai import AzureOpenAI
        except ImportError as exc:
            raise BackendConfigurationError(
                "Azure backend requires the 'openai' Python package"
            ) from exc
        return AzureOpenAI(
            api_key=self.config.api_key,
            azure_endpoint=self.config.azure_endpoint,
            api_version=self.config.azure_api_version,
            timeout=self.config.timeout,
            max_retries=0,
        )
