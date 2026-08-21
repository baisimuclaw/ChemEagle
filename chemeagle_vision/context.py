"""Request-scoped propagation of the selected vision backend."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator, Optional

from .base import VisionBackend


_ACTIVE_VISION_BACKEND: ContextVar[Optional[VisionBackend]] = ContextVar(
    "chemeagle_active_vision_backend", default=None
)
_DEFAULT_BACKEND: Optional[VisionBackend] = None
_DEFAULT_LOCK = threading.Lock()


@contextmanager
def vision_scope(backend: VisionBackend) -> Iterator[VisionBackend]:
    token = _ACTIVE_VISION_BACKEND.set(backend)
    try:
        yield backend
    finally:
        _ACTIVE_VISION_BACKEND.reset(token)


def peek_active_vision_backend() -> Optional[VisionBackend]:
    return _ACTIVE_VISION_BACKEND.get()


def get_active_vision_backend(backend: Optional[VisionBackend] = None) -> VisionBackend:
    if backend is not None:
        return backend
    active = _ACTIVE_VISION_BACKEND.get()
    if active is not None:
        return active
    global _DEFAULT_BACKEND
    with _DEFAULT_LOCK:
        if _DEFAULT_BACKEND is None:
            from .factory import create_vision_backend

            _DEFAULT_BACKEND = create_vision_backend()
        return _DEFAULT_BACKEND
