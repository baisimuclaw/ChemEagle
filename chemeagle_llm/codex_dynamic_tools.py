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


def tool_result(
    text: str,
    *,
    success: bool,
    supplemental_content: Iterable[Dict[str, Any]] = (),
) -> Dict[str, Any]:
    content_items = [{"type": "inputText", "text": text}]
    for item in supplemental_content:
        item_type = item.get("type")
        if item_type == "text" and isinstance(item.get("text"), str):
            content_items.append({"type": "inputText", "text": item["text"]})
        elif item_type == "image_url":
            image_url = item.get("image_url")
            url = image_url.get("url") if isinstance(image_url, dict) else image_url
            if not isinstance(url, str):
                raise UnsupportedCapabilityError(
                    "Supplemental tool images require a string image URL"
                )
            content_items.append({"type": "inputImage", "imageUrl": url})
        else:
            raise UnsupportedCapabilityError(
                f"Unsupported supplemental tool content type: {item_type!r}"
            )
    return {"contentItems": content_items, "success": success}
