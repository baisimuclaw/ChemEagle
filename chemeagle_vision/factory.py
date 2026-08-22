"""Vision backend factory with lazy heavy imports."""

from __future__ import annotations

from typing import Any, Optional

from .base import VisionBackend
from .config import VisionConfig


def create_vision_backend(
    provider: Optional[str] = None,
    *,
    config: Optional[VisionConfig] = None,
    **overrides: Any,
) -> VisionBackend:
    resolved = config or VisionConfig.from_env(provider, **overrides)
    if resolved.provider == "local":
        from .runtime import LocalVisionBackend

        return LocalVisionBackend(resolved)
    from .remote import RemoteVisionBackend

    return RemoteVisionBackend(resolved)

