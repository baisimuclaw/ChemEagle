"""Small provider-neutral request and response models."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from .errors import InvalidResponseError


@dataclass(frozen=True)
class LLMToolCall:
    id: str
    name: str
    arguments: str = "{}"

    def parsed_arguments(self) -> Dict[str, Any]:
        try:
            value = json.loads(self.arguments or "{}")
        except (TypeError, json.JSONDecodeError) as exc:
            raise InvalidResponseError(
                f"Tool {self.name!r} returned invalid JSON arguments"
            ) from exc
        if not isinstance(value, dict):
            raise InvalidResponseError(
                f"Tool {self.name!r} arguments must be a JSON object"
            )
        return value

    def as_openai_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": self.arguments},
        }


@dataclass
class LLMResponse:
    content: Optional[str] = None
    tool_calls: List[LLMToolCall] = field(default_factory=list)
    model: Optional[str] = None
    usage: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def assistant_message(self) -> Dict[str, Any]:
        message: Dict[str, Any] = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            message["tool_calls"] = [call.as_openai_dict() for call in self.tool_calls]
        return message


@dataclass
class LLMToolOutput:
    """A tool value plus multimodal evidence for the model's continuation.

    ``value`` remains the machine-readable function result.  Supplemental
    content becomes the annotated user message in the upstream-compatible
    second completion, so agents can inspect the marked image without
    rerunning the underlying vision model.
    """

    value: Any
    supplemental_content: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class LLMRequest:
    messages: List[Dict[str, Any]]
    model: Optional[str] = None
    tools: List[Dict[str, Any]] = field(default_factory=list)
    tool_choice: Optional[Any] = None
    json_mode: bool = False
    output_schema: Optional[Dict[str, Any]] = None
    temperature: Optional[float] = None
    timeout: Optional[float] = None
    cancel_event: Optional[Any] = None
    tool_followup_content: List[Dict[str, Any]] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)


ToolExecutor = Mapping[str, Any]
