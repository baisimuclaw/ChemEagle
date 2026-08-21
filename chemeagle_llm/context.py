"""Request-scoped backend propagation for the existing nested agent API."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator, Optional

from .base import LLMBackend
from .factory import create_backend


_ACTIVE_BACKEND: ContextVar[Optional[LLMBackend]] = ContextVar(
    "chemeagle_active_llm_backend", default=None
)


@contextmanager
def backend_scope(backend: LLMBackend) -> Iterator[LLMBackend]:
    token = _ACTIVE_BACKEND.set(backend)
    try:
        yield backend
    finally:
        _ACTIVE_BACKEND.reset(token)


def peek_active_backend() -> Optional[LLMBackend]:
    """Return the request-scoped backend without implicitly creating one."""
    return _ACTIVE_BACKEND.get()


def get_active_backend(
    backend: Optional[LLMBackend] = None,
    *,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> LLMBackend:
    if backend is not None:
        return backend
    active = _ACTIVE_BACKEND.get()
    if active is not None:
        return active
    return create_backend(
        provider, model=model, base_url=base_url, api_key=api_key
    )


def backend_model(backend: LLMBackend, legacy_default: Optional[str]) -> Optional[str]:
    """Select a configured model while preserving legacy Azure/local defaults.

    Codex discovers its account model when none is explicitly configured, so a
    legacy Azure deployment name must never be forwarded to it accidentally.
    """
    configured = getattr(getattr(backend, "config", None), "model", None)
    if configured:
        return configured
    if getattr(backend, "provider_name", None) == "codex":
        return None
    return legacy_default
