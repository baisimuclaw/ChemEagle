"""Provider-neutral language-model backends for ChemEAGLE."""

from .base import BaseLLMBackend, LLMBackend, bind_image_tools, parse_json_content
from .config import BackendConfig
from .context import (
    backend_model,
    backend_scope,
    get_active_backend,
    get_request_cache,
    peek_active_backend,
)
from .errors import (
    AuthenticationError,
    BackendConfigurationError,
    BackendCancelledError,
    BackendError,
    BackendProcessError,
    BackendRateLimitError,
    BackendTimeoutError,
    InvalidResponseError,
    ToolExecutionError,
    UnsupportedCapabilityError,
)
from .factory import create_backend
from .types import LLMRequest, LLMResponse, LLMToolCall, LLMToolOutput

__all__ = [
    "AuthenticationError",
    "BackendConfig",
    "BackendConfigurationError",
    "BackendCancelledError",
    "BackendError",
    "BackendProcessError",
    "BackendRateLimitError",
    "BackendTimeoutError",
    "BaseLLMBackend",
    "InvalidResponseError",
    "LLMBackend",
    "LLMRequest",
    "LLMResponse",
    "LLMToolCall",
    "LLMToolOutput",
    "ToolExecutionError",
    "UnsupportedCapabilityError",
    "backend_scope",
    "backend_model",
    "bind_image_tools",
    "create_backend",
    "get_active_backend",
    "get_request_cache",
    "parse_json_content",
    "peek_active_backend",
]
