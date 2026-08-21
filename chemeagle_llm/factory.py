"""Backend factory kept free of eager optional imports."""

from __future__ import annotations

from typing import Any, Optional

from .base import LLMBackend
from .config import BackendConfig


def create_backend(
    provider: Optional[str] = None,
    *,
    config: Optional[BackendConfig] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    client: Any = None,
) -> LLMBackend:
    resolved = config or BackendConfig.from_env(
        provider, model=model, base_url=base_url, api_key=api_key
    )
    if resolved.provider == "azure":
        from .azure import AzureOpenAIBackend

        return AzureOpenAIBackend(resolved, client=client)
    if resolved.provider == "local":
        from .local_openai import LocalOpenAIBackend

        return LocalOpenAIBackend(resolved, client=client)
    if resolved.provider == "codex":
        from .codex_app_server import CodexAppServerBackend

        return CodexAppServerBackend(resolved, client=client)
    raise ValueError(f"Unknown backend provider: {resolved.provider}")
