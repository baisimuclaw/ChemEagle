"""Shared implementation for Azure and local OpenAI-compatible services."""

from __future__ import annotations

import logging
import random
import time
from typing import Any, Dict, Optional

from .base import BaseLLMBackend, parse_json_content
from .config import BackendConfig
from .errors import (
    BackendConfigurationError,
    BackendCancelledError,
    BackendError,
    BackendRateLimitError,
    InvalidResponseError,
    UnsupportedCapabilityError,
)
from .types import LLMRequest, LLMResponse, LLMToolCall

logger = logging.getLogger(__name__)


class OpenAICompatibleBackend(BaseLLMBackend):
    def __init__(self, config: BackendConfig, *, client: Any = None):
        self.config = config
        self._client = client

    def _build_client(self) -> Any:
        raise NotImplementedError

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = self._build_client()
        return self._client

    def _request_kwargs(self, request: LLMRequest) -> Dict[str, Any]:
        model = request.model or self.config.model
        if not model:
            raise BackendConfigurationError(
                f"No model configured for {self.provider_name}; pass model= or set CHEMEAGLE_LLM_MODEL"
            )
        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": request.messages,
        }
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.timeout is not None:
            kwargs["timeout"] = request.timeout
        if request.tools:
            if not self.config.supports_tools:
                raise UnsupportedCapabilityError(
                    f"{self.provider_name} is configured without function-tool support"
                )
            kwargs["tools"] = request.tools
            if request.tool_choice is not None:
                kwargs["tool_choice"] = request.tool_choice
        allow_response_format = (
            not request.tools or self.config.supports_response_format_with_tools
        )
        if request.output_schema and allow_response_format:
            if self.config.supports_json_schema:
                kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "chemeagle_output",
                        "strict": True,
                        "schema": request.output_schema,
                    },
                }
            elif self.config.supports_response_format:
                kwargs["response_format"] = {"type": "json_object"}
        elif (
            request.json_mode
            and allow_response_format
            and self.config.supports_response_format
        ):
            kwargs["response_format"] = {"type": "json_object"}
        kwargs.update(request.extra)
        return kwargs

    @staticmethod
    def _normalise(response: Any) -> LLMResponse:
        try:
            message = response.choices[0].message
        except (AttributeError, IndexError, TypeError) as exc:
            raise InvalidResponseError("OpenAI-compatible response has no assistant message") from exc
        calls = []
        for call in getattr(message, "tool_calls", None) or []:
            function = getattr(call, "function", None)
            calls.append(
                LLMToolCall(
                    id=str(getattr(call, "id", "")),
                    name=str(getattr(function, "name", "")),
                    arguments=str(getattr(function, "arguments", "{}") or "{}"),
                )
            )
        usage_obj = getattr(response, "usage", None)
        if usage_obj is None:
            usage: Dict[str, Any] = {}
        elif hasattr(usage_obj, "model_dump"):
            usage = usage_obj.model_dump()
        elif isinstance(usage_obj, dict):
            usage = dict(usage_obj)
        else:
            usage = {}
        return LLMResponse(
            content=getattr(message, "content", None),
            tool_calls=calls,
            model=getattr(response, "model", None),
            usage=usage,
        )

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        name = type(exc).__name__
        status = getattr(exc, "status_code", None)
        return status in {429, 500, 502, 503, 504} or name in {
            "APIConnectionError",
            "APITimeoutError",
            "InternalServerError",
            "RateLimitError",
        }

    def generate(self, request: LLMRequest) -> LLMResponse:
        kwargs = self._request_kwargs(request)
        attempts = max(1, self.config.max_retries)
        last: Optional[Exception] = None
        for attempt in range(attempts):
            normalised: Optional[LLMResponse] = None
            if request.cancel_event is not None and request.cancel_event.is_set():
                raise BackendCancelledError(
                    f"{self.provider_name} request was cancelled"
                )
            try:
                response = self.client.chat.completions.create(**kwargs)
                if request.cancel_event is not None and request.cancel_event.is_set():
                    raise BackendCancelledError(
                        f"{self.provider_name} request was cancelled"
                    )
                normalised = self._normalise(response)
                if (request.json_mode or request.output_schema) and not normalised.tool_calls:
                    parse_json_content(normalised, request.output_schema)
                return normalised
            except InvalidResponseError as exc:
                if attempt == attempts - 1:
                    raise
                last = exc
                invalid_text = normalised.content if normalised is not None else None
                correction_messages = list(request.messages)
                if invalid_text:
                    correction_messages.append(
                        {"role": "assistant", "content": invalid_text}
                    )
                correction_messages.append(
                    {
                        "role": "user",
                        "content": (
                            "The previous answer was not valid JSON matching the requested "
                            "schema. Return a corrected JSON object only."
                        ),
                    }
                )
                kwargs = dict(kwargs)
                kwargs["messages"] = correction_messages
                logger.warning(
                    "%s returned invalid structured output; requesting correction",
                    self.provider_name,
                )
                continue
            except Exception as exc:
                if not self._is_retryable(exc) or attempt == attempts - 1:
                    if type(exc).__name__ == "RateLimitError" or getattr(exc, "status_code", None) == 429:
                        raise BackendRateLimitError(
                            f"{self.provider_name} rate limit reached"
                        ) from exc
                    if isinstance(exc, BackendError):
                        raise
                    raise BackendError(f"{self.provider_name} request failed: {exc}") from exc
                last = exc
                delay = min(30.0, (2**attempt) + random.uniform(0.0, 0.5))
                logger.warning(
                    "%s request failed transiently (%s); retrying in %.2fs",
                    self.provider_name,
                    type(exc).__name__,
                    delay,
                )
                time.sleep(delay)
        raise BackendError(f"{self.provider_name} request failed: {last}")

    def close(self) -> None:
        if self._client is not None and hasattr(self._client, "close"):
            self._client.close()
