"""Request-scoped propagation of the selected vision backend."""

from __future__ import annotations

import threading
import copy
import hashlib
import json
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Dict, Iterator, Optional

from .base import VisionBackend
from .codec import encode_value


_ACTIVE_VISION_BACKEND: ContextVar[Optional[VisionBackend]] = ContextVar(
    "chemeagle_active_vision_backend", default=None
)
_REQUEST_CACHE: ContextVar[Optional[Dict[str, Any]]] = ContextVar(
    "chemeagle_vision_request_cache", default=None
)
_REQUEST_CACHE_LOCK: ContextVar[Optional[threading.RLock]] = ContextVar(
    "chemeagle_vision_request_cache_lock", default=None
)
_DEFAULT_BACKEND: Optional[VisionBackend] = None
_DEFAULT_LOCK = threading.Lock()


@contextmanager
def vision_scope(backend: VisionBackend) -> Iterator[VisionBackend]:
    token = _ACTIVE_VISION_BACKEND.set(backend)
    cache_token = _REQUEST_CACHE.set({})
    cache_lock_token = _REQUEST_CACHE_LOCK.set(threading.RLock())
    try:
        yield backend
    finally:
        _REQUEST_CACHE_LOCK.reset(cache_lock_token)
        _REQUEST_CACHE.reset(cache_token)
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


def _cache_key(method: str, params: Dict[str, Any]) -> str:
    encoded = json.dumps(
        encode_value(params),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"{method}:{hashlib.sha256(encoded).hexdigest()}"


def _empty_prediction(value: Any) -> bool:
    return value is None or (
        isinstance(value, (list, dict, tuple)) and len(value) == 0
    )


def call_vision(method: str, params: Dict[str, Any]) -> Any:
    """Run one deterministic vision call with request-local reuse and bounded retries.

    The cache contains the complete unmodified prediction.  It exists only
    inside one top-level ChemEAGLE request and prevents a Codex retry from
    repeating an already successful GPU inference.
    """
    backend = get_active_vision_backend()
    cache = _REQUEST_CACHE.get()
    cache_lock = _REQUEST_CACHE_LOCK.get()
    key = _cache_key(method, params) if cache is not None else None

    def invoke() -> Any:
        if key is not None and key in cache:
            return copy.deepcopy(cache[key])

        result = backend.call(method, params)
        # Empty inference is intentionally not cached.  The upstream-facing
        # agent helpers own the single three-attempt loop, so retries are never
        # multiplied.
        if key is not None and not _empty_prediction(result):
            cache[key] = copy.deepcopy(result)
        return result

    # Context copies share this RLock object.  The check, inference, and cache
    # insertion therefore form one operation even when Codex issues identical
    # dynamic-tool calls concurrently.
    if cache_lock is not None:
        with cache_lock:
            return invoke()
    return invoke()
