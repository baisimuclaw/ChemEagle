"""Configuration resolution with explicit, documented precedence."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Mapping, Optional


def _first(env: Mapping[str, str], *names: str, default: Optional[str] = None) -> Optional[str]:
    for name in names:
        value = env.get(name)
        if value:
            return value
    return default


def _env_bool(env: Mapping[str, str], name: str, default: bool) -> bool:
    value = env.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


@dataclass
class BackendConfig:
    provider: str = "azure"
    model: Optional[str] = None
    api_key: Optional[str] = field(default=None, repr=False)
    base_url: Optional[str] = None
    azure_endpoint: Optional[str] = None
    azure_api_version: str = "2024-10-21"
    timeout: float = 180.0
    synthesis_timeout: float = 600.0
    tool_timeout: float = 3600.0
    max_retries: int = 3
    supports_tools: bool = True
    supports_response_format: bool = True
    supports_response_format_with_tools: bool = True
    supports_json_schema: bool = False
    codex_binary: str = "codex"
    codex_cwd: Optional[str] = None
    codex_min_version: Optional[str] = None

    @classmethod
    def from_env(
        cls,
        provider: Optional[str] = None,
        *,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        env: Optional[Mapping[str, str]] = None,
    ) -> "BackendConfig":
        values = os.environ if env is None else env
        selected = (provider or values.get("CHEMEAGLE_LLM_PROVIDER") or "azure").lower()
        if selected in {"vllm", "ollama", "openai-compatible", "openai_compatible"}:
            selected = "local"
        if selected not in {"azure", "local", "codex"}:
            raise ValueError(
                "CHEMEAGLE_LLM_PROVIDER must be one of: azure, codex, local"
            )

        default_timeout = "240" if selected == "codex" else "180"
        timeout = float(values.get("CHEMEAGLE_LLM_TIMEOUT", default_timeout))
        synthesis_timeout = float(
            values.get("CHEMEAGLE_LLM_SYNTHESIS_TIMEOUT", "600")
        )
        tool_timeout = float(values.get("CHEMEAGLE_LLM_TOOL_TIMEOUT", "3600"))
        retries = int(values.get("CHEMEAGLE_LLM_MAX_RETRIES", "3"))
        common_model = model or values.get("CHEMEAGLE_LLM_MODEL")

        if selected == "azure":
            return cls(
                provider=selected,
                model=common_model,
                api_key=api_key or _first(values, "AZURE_OPENAI_API_KEY", "API_KEY"),
                azure_endpoint=_first(values, "AZURE_OPENAI_ENDPOINT", "AZURE_ENDPOINT"),
                azure_api_version=_first(
                    values, "AZURE_OPENAI_API_VERSION", "API_VERSION", default="2024-10-21"
                ) or "2024-10-21",
                timeout=timeout,
                synthesis_timeout=synthesis_timeout,
                tool_timeout=tool_timeout,
                max_retries=retries,
                supports_tools=True,
                supports_response_format=True,
                supports_response_format_with_tools=True,
                supports_json_schema=_env_bool(values, "AZURE_OPENAI_JSON_SCHEMA", False),
            )

        if selected == "local":
            return cls(
                provider=selected,
                model=common_model or _first(
                    values,
                    "VLLM_MODEL",
                    "OLLAMA_MODEL",
                    default="Qwen/Qwen3-VL-32B-Instruct",
                ),
                api_key=api_key or _first(
                    values, "VLLM_API_KEY", "OLLAMA_API_KEY", default="EMPTY"
                ),
                base_url=base_url or _first(
                    values,
                    "VLLM_BASE_URL",
                    "OLLAMA_BASE_URL",
                    default="http://localhost:8000/v1",
                ),
                timeout=timeout,
                synthesis_timeout=synthesis_timeout,
                tool_timeout=tool_timeout,
                max_retries=retries,
                supports_tools=_env_bool(values, "VLLM_SUPPORTS_TOOLS", True),
                supports_response_format=_env_bool(
                    values, "VLLM_SUPPORTS_RESPONSE_FORMAT", True
                ),
                supports_response_format_with_tools=_env_bool(
                    values, "VLLM_SUPPORTS_RESPONSE_FORMAT_WITH_TOOLS", False
                ),
                supports_json_schema=_env_bool(values, "VLLM_SUPPORTS_JSON_SCHEMA", False),
            )

        return cls(
            provider=selected,
            model=common_model or values.get("CODEX_MODEL"),
            timeout=timeout,
            synthesis_timeout=synthesis_timeout,
            tool_timeout=tool_timeout,
            max_retries=retries,
            supports_tools=True,
            supports_response_format=True,
            supports_response_format_with_tools=True,
            supports_json_schema=True,
            codex_binary=values.get("CODEX_BIN", "codex"),
            codex_cwd=values.get("CHEMEAGLE_CODEX_CWD"),
            codex_min_version=values.get("CHEMEAGLE_CODEX_MIN_VERSION", "0.146.0"),
        )
