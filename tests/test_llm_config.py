from __future__ import annotations

import json
import threading
import unittest
from contextvars import ContextVar
from types import SimpleNamespace

from chemeagle_llm.base import bind_image_tools, parse_json_content
from chemeagle_llm.config import BackendConfig
from chemeagle_llm.errors import (
    BackendCancelledError,
    BackendConfigurationError,
    InvalidResponseError,
    ToolExecutionError,
)
from chemeagle_llm.factory import create_backend
from chemeagle_llm.types import LLMRequest, LLMResponse


class _FakeCompletions:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


class _FakeClient:
    def __init__(self, responses):
        self.chat = SimpleNamespace(completions=_FakeCompletions(responses))
        self.closed = False

    def close(self):
        self.closed = True


def _response(content=None, tool_calls=None, model="fake-model"):
    message = SimpleNamespace(content=content, tool_calls=tool_calls or [])
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)], model=model, usage=None
    )


def _tool_call(name, arguments, call_id="call-1"):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


class BackendConfigTests(unittest.TestCase):
    def test_default_is_backward_compatible_azure_without_early_validation(self):
        config = BackendConfig.from_env(env={})
        self.assertEqual(config.provider, "azure")
        self.assertIsNone(config.api_key)
        backend = create_backend(config=config, client=_FakeClient([]))
        self.assertEqual(backend.provider_name, "azure")

    def test_aliases_and_explicit_precedence(self):
        config = BackendConfig.from_env(
            "vllm",
            model="explicit-model",
            base_url="http://explicit/v1",
            api_key="explicit-secret",
            env={
                "VLLM_MODEL": "environment-model",
                "VLLM_BASE_URL": "http://environment/v1",
                "VLLM_API_KEY": "environment-secret",
            },
        )
        self.assertEqual(config.provider, "local")
        self.assertEqual(config.model, "explicit-model")
        self.assertEqual(config.base_url, "http://explicit/v1")
        self.assertEqual(config.api_key, "explicit-secret")
        self.assertNotIn("explicit-secret", repr(config))

    def test_azure_aliases_and_legacy_environment_names(self):
        aliases = BackendConfig.from_env(
            "azure",
            env={
                "AZURE_OPENAI_API_KEY": "new-secret",
                "API_KEY": "legacy-secret",
                "AZURE_OPENAI_ENDPOINT": "https://new.example",
                "AZURE_ENDPOINT": "https://legacy.example",
                "AZURE_OPENAI_API_VERSION": "2026-01-01",
            },
        )
        self.assertEqual(aliases.api_key, "new-secret")
        self.assertEqual(aliases.azure_endpoint, "https://new.example")
        self.assertEqual(aliases.azure_api_version, "2026-01-01")

    def test_unknown_provider_rejected(self):
        with self.assertRaises(ValueError):
            BackendConfig.from_env("mystery", env={})

    def test_tool_timeout_has_independent_environment_setting(self):
        config = BackendConfig.from_env(
            "codex",
            env={
                "CHEMEAGLE_LLM_TIMEOUT": "90",
                "CHEMEAGLE_LLM_TOOL_TIMEOUT": "1234",
            },
        )
        self.assertEqual(config.timeout, 90)
        self.assertEqual(config.tool_timeout, 1234)

    def test_final_synthesis_uses_upstream_compatible_timeout(self):
        self.assertEqual(
            BackendConfig.from_env("codex", env={}).synthesis_timeout,
            600,
        )
        configured = BackendConfig.from_env(
            "codex",
            env={"CHEMEAGLE_LLM_SYNTHESIS_TIMEOUT": "720"},
        )
        self.assertEqual(configured.synthesis_timeout, 720)

    def test_codex_defaults_to_four_minute_response_window(self):
        self.assertEqual(BackendConfig.from_env("codex", env={}).timeout, 240)
        self.assertEqual(BackendConfig.from_env("azure", env={}).timeout, 180)
        self.assertEqual(BackendConfig.from_env("codex", env={}).max_retries, 3)

    def test_credentials_are_validated_only_when_client_is_needed(self):
        backend = create_backend(config=BackendConfig(provider="azure", model="m"))
        with self.assertRaises(BackendConfigurationError):
            backend.generate(LLMRequest(model="m", messages=[{"role": "user", "content": "x"}]))


class OpenAICompatibleContractTests(unittest.TestCase):
    def test_azure_request_mapping_uses_the_same_normalised_contract(self):
        client = _FakeClient([_response('{"provider": "azure"}')])
        backend = create_backend(
            config=BackendConfig(
                provider="azure",
                model="azure-deployment",
                api_key="secret",
                azure_endpoint="https://azure.example",
                max_retries=1,
            ),
            client=client,
        )
        result = backend.generate(
            LLMRequest(
                messages=[{"role": "user", "content": "json"}],
                json_mode=True,
            )
        )
        self.assertEqual(parse_json_content(result), {"provider": "azure"})
        self.assertEqual(
            client.chat.completions.calls[0]["model"], "azure-deployment"
        )

    def test_json_request_and_normalised_response(self):
        client = _FakeClient([_response('{"ok": true}')])
        config = BackendConfig(
            provider="local",
            model="qwen",
            base_url="http://localhost:8000/v1",
            api_key="EMPTY",
            max_retries=1,
        )
        backend = create_backend(config=config, client=client)
        response = backend.generate(
            LLMRequest(
                messages=[{"role": "user", "content": "Return JSON"}],
                json_mode=True,
                temperature=0,
            )
        )
        self.assertEqual(parse_json_content(response), {"ok": True})
        call = client.chat.completions.calls[0]
        self.assertEqual(call["model"], "qwen")
        self.assertEqual(call["temperature"], 0)
        self.assertEqual(call["response_format"], {"type": "json_object"})

    def test_tool_loop_executes_only_registered_mapping(self):
        client = _FakeClient(
            [
                _response(tool_calls=[_tool_call("lookup", {"value": 4})]),
                _response('{"answer": 8}'),
            ]
        )
        backend = create_backend(
            config=BackendConfig(
                provider="local",
                model="qwen",
                base_url="http://localhost:8000/v1",
                max_retries=1,
            ),
            client=client,
        )
        result = backend.run_tool_loop(
            LLMRequest(messages=[{"role": "user", "content": "double"}], json_mode=True),
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "description": "test",
                        "parameters": {"type": "object"},
                    },
                }
            ],
            executor={"lookup": lambda value: {"answer": value * 2}},
        )
        self.assertEqual(parse_json_content(result), {"answer": 8})
        followup = client.chat.completions.calls[1]["messages"]
        self.assertEqual(followup[-1]["role"], "tool")
        self.assertEqual(json.loads(followup[-1]["content"]), {"answer": 8})

    def test_invalid_json_is_not_silently_accepted(self):
        client = _FakeClient([_response("not json")])
        backend = create_backend(
            config=BackendConfig(
                provider="local",
                model="qwen",
                base_url="http://localhost:8000/v1",
                max_retries=1,
            ),
            client=client,
        )
        with self.assertRaises(InvalidResponseError):
            backend.generate(
                LLMRequest(messages=[{"role": "user", "content": "json"}], json_mode=True)
            )

        with self.assertRaises(InvalidResponseError):
            parse_json_content(
                LLMResponse(content='{\"ok\": false, \"extra\": 1}'),
                {
                    "type": "object",
                    "properties": {"ok": {"const": True}},
                    "required": ["ok"],
                    "additionalProperties": False,
                },
            )

    def test_invalid_json_is_corrected_and_capability_flags_omit_response_format(self):
        client = _FakeClient([_response("not json"), _response('{"ok": true}')])
        backend = create_backend(
            config=BackendConfig(
                provider="local",
                model="qwen",
                base_url="http://localhost:8000/v1",
                max_retries=2,
                supports_response_format_with_tools=False,
            ),
            client=client,
        )
        result = backend.generate(
            LLMRequest(messages=[{"role": "user", "content": "json"}], json_mode=True)
        )
        self.assertEqual(parse_json_content(result), {"ok": True})
        self.assertIn("previous answer", client.chat.completions.calls[1]["messages"][-1]["content"])

        tool_client = _FakeClient([_response(tool_calls=[_tool_call("lookup", {})])])
        tool_backend = create_backend(
            config=BackendConfig(
                provider="local",
                model="qwen",
                base_url="http://localhost:8000/v1",
                max_retries=1,
                supports_response_format_with_tools=False,
            ),
            client=tool_client,
        )
        tool_backend.generate(
            LLMRequest(
                messages=[{"role": "user", "content": "tool"}],
                tools=[{"type": "function", "function": {"name": "lookup"}}],
                json_mode=True,
            )
        )
        self.assertNotIn("response_format", tool_client.chat.completions.calls[0])

    def test_unknown_tool_bad_arguments_cancellation_and_bound_path(self):
        backend = create_backend(
            config=BackendConfig(
                provider="local",
                model="qwen",
                base_url="http://localhost:8000/v1",
                max_retries=1,
            ),
            client=_FakeClient([_response(tool_calls=[_tool_call("unknown", {})])]),
        )
        with self.assertRaises(ToolExecutionError):
            backend.run_tool_loop(
                LLMRequest(messages=[{"role": "user", "content": "tool"}]),
                [],
                {},
            )

        bad_call = SimpleNamespace(
            id="bad", function=SimpleNamespace(name="lookup", arguments="[")
        )
        backend = create_backend(
            config=BackendConfig(
                provider="local",
                model="qwen",
                base_url="http://localhost:8000/v1",
                max_retries=1,
            ),
            client=_FakeClient([_response(tool_calls=[bad_call])]),
        )
        with self.assertRaises(InvalidResponseError):
            backend.run_tool_loop(
                LLMRequest(messages=[{"role": "user", "content": "tool"}]),
                [],
                {"lookup": lambda: None},
            )

        event = SimpleNamespace(is_set=lambda: True)
        with self.assertRaises(BackendCancelledError):
            backend.generate(
                LLMRequest(
                    messages=[{"role": "user", "content": "cancel"}],
                    cancel_event=event,
                )
            )

        seen = []
        executor = bind_image_tools(
            "/trusted/input.png", {"read": lambda path: seen.append(path) or "ok"}
        )
        self.assertEqual(executor["read"](image_path="/etc/passwd"), "ok")
        self.assertEqual(seen, ["/trusted/input.png"])

        marker = ContextVar("test_image_tool_context", default="default")
        token = marker.set("remote-request")
        try:
            contextual = bind_image_tools(
                "/trusted/input.png", {"read": lambda _path: marker.get()}
            )
        finally:
            marker.reset(token)
        threaded_result = []
        thread = threading.Thread(
            target=lambda: threaded_result.append(contextual["read"]())
        )
        thread.start()
        thread.join()
        self.assertEqual(threaded_result, ["remote-request"])


if __name__ == "__main__":
    unittest.main()
