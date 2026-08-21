from __future__ import annotations

import contextlib
import json
import os
import queue
import threading
import time
import unittest
from unittest import mock

from chemeagle_llm.codex_app_server import (
    CodexAppServerBackend,
    CodexAppServerClient,
)
from chemeagle_llm.config import BackendConfig
from chemeagle_llm.errors import (
    AuthenticationError,
    BackendCancelledError,
    BackendProcessError,
    BackendRateLimitError,
    BackendTimeoutError,
    UnsupportedCapabilityError,
)
from chemeagle_llm.types import LLMRequest


class _QueueStream:
    def __init__(self):
        self.lines = queue.Queue()

    def put(self, message):
        self.lines.put(json.dumps(message) + "\n")

    def close(self):
        self.lines.put(None)

    def __iter__(self):
        return self

    def __next__(self):
        item = self.lines.get(timeout=5)
        if item is None:
            raise StopIteration
        return item


class _ScriptedStdin:
    def __init__(self, callback):
        self.callback = callback

    def write(self, value):
        self.callback(json.loads(value))
        return len(value)

    def flush(self):
        return None


class _FakeProcess:
    def __init__(self, script):
        self.stdout = _QueueStream()
        self.stderr = _QueueStream()
        self.stdin = _ScriptedStdin(script.handle)
        self.returncode = None
        script.process = self

    def poll(self):
        return self.returncode


class _CodexScript:
    def __init__(
        self,
        *,
        account_type="chatgpt",
        requested_tool=None,
        tool_arguments=None,
        rate_limited=False,
        turn_texts=None,
        hold_turn=False,
        overload_method=None,
        close_method=None,
        ignore_method=None,
    ):
        self.process = None
        self.account_type = account_type
        self.requested_tool = requested_tool
        self.tool_arguments = {"value": 3} if tool_arguments is None else tool_arguments
        self.rate_limited = rate_limited
        self.turn_texts = list(turn_texts or ['{"reactions": []}'])
        self.hold_turn = hold_turn
        self.overload_method = overload_method
        self.close_method = close_method
        self.ignore_method = ignore_method
        self.overload_count = 0
        self.turn_count = 0
        self.initialized_received = False
        self.login_types = []
        self.interrupts = []
        self.dynamic_tools = []
        self.dynamic_reply = None
        self.turn_input = None
        self.output_schema = None

    def send(self, message):
        self.process.stdout.put(message)

    def response(self, request, result):
        self.send({"id": request["id"], "result": result})

    def finish_turn(self, turn_id="turn-1"):
        index = min(max(self.turn_count - 1, 0), len(self.turn_texts) - 1)
        self.send(
            {
                "method": "turn/completed",
                "params": {
                    "threadId": "thread-1",
                    "turn": {
                        "id": turn_id,
                        "status": "completed",
                        "items": [
                            {
                                "id": "message-1",
                                "type": "agentMessage",
                                "text": self.turn_texts[index],
                            }
                        ],
                    },
                },
            }
        )

    def handle(self, message):
        if "method" not in message:
            if message.get("id") == 900:
                self.dynamic_reply = message.get("result")
                self.finish_turn(f"turn-{self.turn_count}")
            return
        method = message["method"]
        if "id" not in message:  # initialized notification
            if method == "initialized":
                self.initialized_received = True
            return
        if method == self.close_method:
            self.process.returncode = 17
            self.process.stdout.close()
            return
        if method == self.ignore_method:
            return
        if method == self.overload_method and self.overload_count == 0:
            self.overload_count += 1
            self.send(
                {
                    "id": message["id"],
                    "error": {"code": -32001, "message": "overloaded"},
                }
            )
            return
        if method == "initialize":
            self.response(message, {"userAgent": "fake", "codexHome": "/fake"})
        elif method == "account/read":
            account = None
            if self.account_type:
                account = {"type": self.account_type}
                if self.account_type == "chatgpt":
                    account.update({"email": "test@example.com", "planType": "pro"})
            self.response(message, {"account": account, "requiresOpenaiAuth": True})
        elif method == "account/rateLimits/read":
            used = 100 if self.rate_limited else 1
            self.response(message, {"rateLimits": {"primary": {"usedPercent": used}}})
        elif method == "account/login/start":
            login_type = message["params"]["type"]
            self.login_types.append(login_type)
            if login_type == "chatgptDeviceCode":
                result = {
                    "type": login_type,
                    "loginId": "login-1",
                    "verificationUrl": "https://example.test/device",
                    "userCode": "ABCD-EFGH",
                }
            else:
                result = {
                    "type": login_type,
                    "loginId": "login-1",
                    "authUrl": "https://example.test/oauth?secret=value",
                }
            self.response(message, result)
            self.send(
                {
                    "method": "account/login/completed",
                    "params": {"loginId": "login-1", "success": True},
                }
            )
        elif method == "account/logout":
            self.response(message, {})
        elif method == "model/list":
            self.response(
                message,
                {
                    "data": [
                        {
                            "id": "available-model",
                            "model": "available-model",
                            "displayName": "Available",
                            "hidden": False,
                            "isDefault": True,
                        }
                    ],
                    "nextCursor": None,
                },
            )
        elif method == "thread/start":
            self.dynamic_tools = message["params"].get("dynamicTools") or []
            self.response(
                message,
                {"thread": {"id": "thread-1"}, "model": "available-model"},
            )
        elif method == "turn/start":
            self.turn_count += 1
            turn_id = f"turn-{self.turn_count}"
            self.turn_input = message["params"].get("input")
            self.output_schema = message["params"].get("outputSchema")
            self.response(message, {"turn": {"id": turn_id, "status": "inProgress"}})
            if self.dynamic_tools:
                self.send(
                    {
                        "id": 900,
                        "method": "item/tool/call",
                        "params": {
                            "threadId": "thread-1",
                            "turnId": turn_id,
                            "callId": "call-1",
                            "tool": self.requested_tool or self.dynamic_tools[0]["name"],
                            "arguments": self.tool_arguments,
                        },
                    }
                )
            elif not self.hold_turn:
                self.finish_turn(turn_id)
        elif method == "turn/interrupt":
            self.interrupts.append(message["params"])
            self.response(message, {})
        else:
            self.send(
                {
                    "id": message["id"],
                    "error": {"code": -32601, "message": f"unknown {method}"},
                }
            )


class CodexBackendTests(unittest.TestCase):
    def make_backend(self, *, account_type="chatgpt", max_retries=1, **script_kwargs):
        config = BackendConfig(
            provider="codex", model="available-model", timeout=5, max_retries=max_retries
        )
        script = _CodexScript(account_type=account_type, **script_kwargs)
        process = _FakeProcess(script)
        client = CodexAppServerClient(config, process=process)
        return CodexAppServerBackend(config, client=client), script

    def test_binary_version_is_checked_before_managed_spawn(self):
        config = BackendConfig(
            provider="codex", codex_binary="codex", codex_min_version="0.146.0"
        )
        client = CodexAppServerClient(config)
        old = mock.Mock(stdout="codex-cli 0.100.0", stderr="")
        with mock.patch("chemeagle_llm.codex_app_server.subprocess.run", return_value=old):
            with self.assertRaises(UnsupportedCapabilityError):
                client._check_version()
        with mock.patch(
            "chemeagle_llm.codex_app_server.subprocess.run",
            side_effect=FileNotFoundError,
        ):
            with self.assertRaises(BackendProcessError):
                client._check_version()

    def test_managed_spawn_uses_stdio_controlled_cwd_and_strips_api_keys(self):
        captured = {}

        class Spawned:
            def __init__(self):
                self.returncode = None

            def poll(self):
                return self.returncode

            def terminate(self):
                self.returncode = 0

            def wait(self, timeout=None):
                return self.returncode

        def factory(command, **kwargs):
            captured["command"] = command
            captured.update(kwargs)
            return Spawned()

        config = BackendConfig(provider="codex", codex_min_version="0.146.0")
        client = CodexAppServerClient(config, process_factory=factory)
        version = mock.Mock(stdout="codex-cli 0.146.0", stderr="")
        with mock.patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "platform-secret",
                "AZURE_OPENAI_API_KEY": "azure-secret",
                "VLLM_API_KEY": "local-secret",
            },
        ), mock.patch(
            "chemeagle_llm.codex_app_server.subprocess.run", return_value=version
        ):
            client._spawn()
        try:
            self.assertEqual(captured["command"], ["codex", "app-server", "--stdio"])
            self.assertTrue(captured["cwd"].startswith("/tmp/chemeagle-codex-"))
            self.assertNotIn("OPENAI_API_KEY", captured["env"])
            self.assertNotIn("AZURE_OPENAI_API_KEY", captured["env"])
            self.assertNotIn("VLLM_API_KEY", captured["env"])
        finally:
            client.close()

    def test_text_image_schema_and_subscription_account(self):
        backend, script = self.make_backend()
        schema = {"type": "object", "required": ["reactions"]}
        try:
            response = backend.generate(
                LLMRequest(
                    messages=[
                        {"role": "system", "content": "Extract chemistry."},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Read the image"},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": "data:image/png;base64,AAAA"},
                                },
                            ],
                        },
                    ],
                    output_schema=schema,
                )
            )
            self.assertEqual(json.loads(response.content), {"reactions": []})
            self.assertEqual(script.output_schema, schema)
            self.assertTrue(any(item["type"] == "image" for item in script.turn_input))
            self.assertEqual(backend.models()[0]["id"], "available-model")
            self.assertTrue(script.initialized_received)
        finally:
            backend.close()

    def test_dynamic_tool_bridge_is_whitelisted(self):
        backend, script = self.make_backend()
        try:
            response = backend.run_tool_loop(
                LLMRequest(
                    messages=[{"role": "user", "content": "Use lookup"}],
                    json_mode=True,
                ),
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "description": "safe lookup",
                            "parameters": {
                                "type": "object",
                                "properties": {"value": {"type": "integer"}},
                            },
                        },
                    }
                ],
                executor={"lookup": lambda value: {"doubled": value * 2}},
            )
            self.assertEqual(json.loads(response.content), {"reactions": []})
            self.assertEqual(script.dynamic_reply["success"], True)
            tool_payload = json.loads(script.dynamic_reply["contentItems"][0]["text"])
            self.assertEqual(tool_payload, {"doubled": 6})
        finally:
            backend.close()

    def test_dynamic_tool_may_issue_nested_rpc_without_reader_deadlock(self):
        backend, script = self.make_backend()
        try:
            response = backend.run_tool_loop(
                LLMRequest(
                    messages=[{"role": "user", "content": "Use lookup"}],
                    json_mode=True,
                ),
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "parameters": {
                                "type": "object",
                                "properties": {"value": {"type": "integer"}},
                            },
                        },
                    }
                ],
                executor={
                    "lookup": lambda **_kwargs: {
                        "model": backend.models()[0]["id"]
                    }
                },
            )
            self.assertEqual(json.loads(response.content), {"reactions": []})
            payload = json.loads(script.dynamic_reply["contentItems"][0]["text"])
            self.assertEqual(payload, {"model": "available-model"})
        finally:
            backend.close()

    def test_api_key_account_is_rejected_instead_of_silently_billed(self):
        backend, _ = self.make_backend(account_type="apiKey")
        try:
            with self.assertRaises(AuthenticationError):
                backend.generate(
                    LLMRequest(messages=[{"role": "user", "content": "hello"}])
                )
        finally:
            backend.close()

    def test_missing_account_is_rejected(self):
        backend, _ = self.make_backend(account_type=None)
        try:
            with self.assertRaises(AuthenticationError):
                backend.generate(LLMRequest(messages=[{"role": "user", "content": "x"}]))
        finally:
            backend.close()

    def test_browser_and_device_code_login_events(self):
        for device_code, expected_type in (
            (False, "chatgpt"),
            (True, "chatgptDeviceCode"),
        ):
            with self.subTest(device_code=device_code):
                backend, script = self.make_backend()
                try:
                    started = backend.login(device_code=device_code)
                    completed = backend.wait_login(
                        started["loginId"], after=started["eventCursor"]
                    )
                    self.assertTrue(completed["success"])
                    self.assertEqual(script.login_types, [expected_type])
                finally:
                    backend.close()

    def _run_dynamic_failure(self, **script_kwargs):
        backend, script = self.make_backend(**script_kwargs)
        try:
            backend.run_tool_loop(
                LLMRequest(messages=[{"role": "user", "content": "tool"}], json_mode=True),
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "parameters": {
                                "type": "object",
                                "properties": {"value": {"type": "integer"}},
                            },
                        },
                    }
                ],
                executor={"lookup": lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("boom"))},
            )
            return script.dynamic_reply
        finally:
            backend.close()

    def test_dynamic_tool_unknown_arguments_and_exception_are_structured_failures(self):
        unknown = self._run_dynamic_failure(requested_tool="not-allowed")
        self.assertFalse(unknown["success"])
        self.assertIn("disallowed", unknown["contentItems"][0]["text"])

        invalid = self._run_dynamic_failure(tool_arguments="not-an-object")
        self.assertFalse(invalid["success"])
        self.assertIn("JSON object", invalid["contentItems"][0]["text"])

        wrong_type = self._run_dynamic_failure(tool_arguments={"value": "bad"})
        self.assertFalse(wrong_type["success"])
        self.assertIn("expected JSON integer", wrong_type["contentItems"][0]["text"])

        failed = self._run_dynamic_failure()
        self.assertFalse(failed["success"])
        self.assertIn("RuntimeError", failed["contentItems"][0]["text"])

    def test_structured_output_is_corrected_within_retry_limit(self):
        backend, script = self.make_backend(
            max_retries=2,
            turn_texts=["not-json", '{"reactions": []}'],
        )
        try:
            response = backend.generate(
                LLMRequest(messages=[{"role": "user", "content": "json"}], json_mode=True)
            )
            self.assertEqual(json.loads(response.content), {"reactions": []})
            self.assertEqual(script.turn_count, 2)
            self.assertEqual(script.output_schema["type"], "object")
        finally:
            backend.close()

    def test_overload_retry_timeout_eof_rate_limit_and_cancel(self):
        backend, script = self.make_backend(
            max_retries=2,
            overload_method="model/list",
        )
        try:
            with mock.patch("chemeagle_llm.codex_app_server.time.sleep"):
                self.assertEqual(backend.models()[0]["id"], "available-model")
            self.assertEqual(script.overload_count, 1)
        finally:
            backend.close()

        backend, _ = self.make_backend(ignore_method="model/list")
        try:
            with self.assertRaises(BackendTimeoutError):
                backend.client.request("model/list", {}, timeout=0.01)
        finally:
            backend.close()

        backend, _ = self.make_backend(close_method="model/list")
        try:
            with self.assertRaises(BackendProcessError):
                backend.models()
        finally:
            backend.close()

        backend, _ = self.make_backend(rate_limited=True)
        try:
            with self.assertRaises(BackendRateLimitError):
                backend.generate(LLMRequest(messages=[{"role": "user", "content": "x"}]))
        finally:
            backend.close()

        backend, script = self.make_backend(hold_turn=True)
        cancel_event = threading.Event()
        timer = threading.Timer(0.03, cancel_event.set)
        timer.start()
        try:
            with self.assertRaises(BackendCancelledError):
                backend.generate(
                    LLMRequest(
                        messages=[{"role": "user", "content": "wait"}],
                        cancel_event=cancel_event,
                    )
                )
            self.assertEqual(script.interrupts[0]["turnId"], "turn-1")
        finally:
            timer.cancel()
            backend.close()


if __name__ == "__main__":
    unittest.main()
