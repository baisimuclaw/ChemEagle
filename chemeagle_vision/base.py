"""Provider-neutral vision backend contract."""

from __future__ import annotations

from typing import Any, Dict, Protocol, runtime_checkable


@runtime_checkable
class VisionBackend(Protocol):
    provider_name: str

    def call(self, method: str, params: Dict[str, Any]) -> Any: ...

    def health(self) -> Dict[str, Any]: ...

    def close(self) -> None: ...


class BaseVisionBackend:
    provider_name = "base"

    def health(self) -> Dict[str, Any]:
        result = self.call("health", {})
        if not isinstance(result, dict):
            raise RuntimeError("Vision health response must be a dictionary")
        return result

    def close(self) -> None:
        return None

