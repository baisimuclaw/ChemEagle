"""Version-isolated mapping for Codex App Server experimental dynamic tools."""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List

from .errors import UnsupportedCapabilityError


def openai_tools_to_dynamic(
    tools: Iterable[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    dynamic = []
    for entry in tools:
        function = entry.get("function") if entry.get("type") == "function" else None
        if not isinstance(function, dict):
            raise UnsupportedCapabilityError(
                "Codex only supports function tool definitions"
            )
        name = function.get("name")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", name):
            raise UnsupportedCapabilityError(
                f"Invalid Codex dynamic tool name: {name!r}"
            )
        schema = function.get("parameters") or {"type": "object"}
        if not isinstance(schema, dict):
            raise UnsupportedCapabilityError(
                f"Codex dynamic tool {name!r} requires an object input schema"
            )
        dynamic.append(
            {
                "type": "function",
                "name": name,
                "description": str(function.get("description") or "ChemEAGLE tool"),
                "inputSchema": schema,
            }
        )
    return dynamic


def tool_result(text: str, *, success: bool) -> Dict[str, Any]:
    return {
        "contentItems": [{"type": "inputText", "text": text}],
        "success": success,
    }

