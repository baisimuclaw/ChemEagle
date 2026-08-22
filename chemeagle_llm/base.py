"""Backend contract and shared tool/JSON handling."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from contextvars import copy_context
from dataclasses import replace
from typing import Any, Callable, Dict, Mapping, Protocol, runtime_checkable

from .errors import InvalidResponseError, ToolExecutionError, UnsupportedCapabilityError
from .types import LLMRequest, LLMResponse, LLMToolOutput


def validate_json_value(value: Any, schema: Dict[str, Any]) -> None:
    """Validate JSON with jsonschema when installed, with a small stdlib fallback."""
    try:
        import jsonschema  # type: ignore
    except ImportError:
        jsonschema = None

    if jsonschema is not None:
        try:
            jsonschema.validate(value, schema)
        except jsonschema.ValidationError as exc:  # type: ignore[attr-defined]
            if exc.validator == "type":
                expected = exc.validator_value
                path = "$" + "".join(f"[{item!r}]" for item in exc.absolute_path)
                raise InvalidResponseError(
                    f"{path}: expected JSON {expected}, got {type(exc.instance).__name__}"
                ) from exc
            raise InvalidResponseError(f"JSON schema validation failed: {exc.message}") from exc
        return

    type_map = {
        "object": dict,
        "array": list,
        "string": str,
        "number": (int, float),
        "integer": int,
        "boolean": bool,
        "null": type(None),
    }

    def validate(current: Any, rule: Dict[str, Any], path: str) -> None:
        expected = rule.get("type")
        allowed = expected if isinstance(expected, list) else [expected]
        allowed = [name for name in allowed if name in type_map]
        if allowed:
            matches = any(isinstance(current, type_map[name]) for name in allowed)
            if isinstance(current, bool) and any(
                name in {"number", "integer"} for name in allowed
            ) and "boolean" not in allowed:
                matches = False
            if not matches:
                raise InvalidResponseError(
                    f"{path}: expected JSON {expected}, got {type(current).__name__}"
                )
        if "const" in rule and current != rule["const"]:
            raise InvalidResponseError(f"{path}: value does not match schema const")
        if "enum" in rule and current not in rule["enum"]:
            raise InvalidResponseError(f"{path}: value is not in the schema enum")
        if isinstance(current, dict):
            missing = [key for key in rule.get("required", []) if key not in current]
            if missing:
                raise InvalidResponseError(
                    f"{path}: JSON object is missing required fields: {missing}"
                )
            properties = rule.get("properties") or {}
            for key, child_rule in properties.items():
                if key in current and isinstance(child_rule, dict):
                    validate(current[key], child_rule, f"{path}.{key}")
            if rule.get("additionalProperties") is False:
                extras = sorted(set(current) - set(properties))
                if extras:
                    raise InvalidResponseError(
                        f"{path}: JSON object has unexpected fields: {extras}"
                    )
        if isinstance(current, list) and isinstance(rule.get("items"), dict):
            for index, item in enumerate(current):
                validate(item, rule["items"], f"{path}[{index}]")

    validate(value, schema, "$")


def parse_json_content(response: LLMResponse, schema: Dict[str, Any] | None = None) -> Any:
    if not response.content:
        raise InvalidResponseError("Model returned no content for a JSON response")
    text = response.content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        preview = text[:240].replace("\n", " ")
        raise InvalidResponseError(f"Model returned invalid JSON: {preview!r}") from exc
    if schema:
        validate_json_value(value, schema)
    return value


def bind_image_tools(
    image_path: str,
    handlers: Mapping[str, Callable[[str], Any]],
) -> Dict[str, Callable[..., Any]]:
    """Bind image tools to the trusted caller-supplied image path.

    Tool-call arguments are model output and therefore untrusted. Existing
    ChemEAGLE tools only need the image selected by the caller, so deliberately
    ignore a model-provided ``image_path`` instead of permitting arbitrary file
    access.
    """

    def bind(handler: Callable[[str], Any]) -> Callable[..., Any]:
        # Codex App Server invokes dynamic tools on its protocol-reader thread.
        # Capture the caller's request context so request-scoped LLM/vision
        # backends are not silently replaced by process defaults there.
        caller_context = copy_context()

        def invoke(**_arguments: Any) -> Any:
            return caller_context.copy().run(handler, image_path)

        return invoke

    return {name: bind(handler) for name, handler in handlers.items()}


def tool_input_schemas(tools: list[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    schemas: Dict[str, Dict[str, Any]] = {}
    for entry in tools:
        function = entry.get("function") if entry.get("type") == "function" else None
        if not isinstance(function, dict) or not isinstance(function.get("name"), str):
            continue
        schema = function.get("parameters") or {"type": "object"}
        if isinstance(schema, dict):
            schemas[function["name"]] = schema
    return schemas


@runtime_checkable
class LLMBackend(Protocol):
    provider_name: str

    def generate(self, request: LLMRequest) -> LLMResponse: ...

    def run_tool_loop(
        self,
        request: LLMRequest,
        tools: list[Dict[str, Any]],
        executor: Mapping[str, Any],
    ) -> LLMResponse: ...

    def close(self) -> None: ...


class BaseLLMBackend(ABC):
    provider_name = "base"

    @abstractmethod
    def generate(self, request: LLMRequest) -> LLMResponse:
        raise NotImplementedError

    def run_tool_loop(
        self,
        request: LLMRequest,
        tools: list[Dict[str, Any]],
        executor: Mapping[str, Any],
    ) -> LLMResponse:
        first = self.generate(replace(request, tools=list(tools)))
        if not first.tool_calls:
            if request.tool_followup_content:
                messages = [
                    message
                    for message in request.messages
                    if message.get("role") in {"system", "developer"}
                ]
                messages.append(
                    {
                        "role": "user",
                        "content": request.tool_followup_content,
                    }
                )
                messages.append(first.assistant_message())
                return self.generate(
                    replace(
                        request,
                        messages=messages,
                        tools=[],
                        tool_choice=None,
                        tool_followup_content=[],
                    )
                )
            return first

        schemas = tool_input_schemas(tools)
        assistant_message = first.assistant_message()
        tool_messages = []
        supplemental_content = []
        for call in first.tool_calls:
            handler = executor.get(call.name)
            if handler is None:
                raise ToolExecutionError(f"Model requested unknown tool {call.name!r}")
            arguments = call.parsed_arguments()
            try:
                schema = schemas.get(call.name)
                if schema is not None:
                    validate_json_value(arguments, schema)
                result = handler(**arguments)
                if isinstance(result, LLMToolOutput):
                    supplemental_content.extend(result.supplemental_content)
                    result = result.value
            except Exception as exc:
                raise ToolExecutionError(f"Tool {call.name!r} failed: {exc}") from exc
            tool_messages.append(
                {
                    "role": "tool",
                    "name": call.name,
                    "tool_call_id": call.id,
                    "content": json.dumps(result, ensure_ascii=False),
                }
            )
        if supplemental_content:
            # Match the upstream two-completion contract: the second request
            # replaces the original figure with its annotated copy, followed
            # by the assistant function calls and their tool outputs.
            messages = [
                message
                for message in request.messages
                if message.get("role") in {"system", "developer"}
            ]
            messages.append({"role": "user", "content": supplemental_content})
        else:
            messages = list(request.messages)
        messages.append(assistant_message)
        messages.extend(tool_messages)
        return self.generate(
            replace(
                request,
                messages=messages,
                tools=[],
                tool_choice=None,
                tool_followup_content=[],
            )
        )

    def parse_json(self, response: LLMResponse, schema: Dict[str, Any] | None = None) -> Any:
        return parse_json_content(response, schema)

    def close(self) -> None:
        return None

    def account_status(self) -> Dict[str, Any]:
        raise UnsupportedCapabilityError(
            f"{self.provider_name} does not expose account status"
        )
