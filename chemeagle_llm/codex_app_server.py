"""Native Codex App Server backend.

This module deliberately uses the documented stdio JSONL protocol.  It never
reads Codex credential files and removes API-key environment fallbacks from the
managed child so a ChatGPT subscription route cannot silently become an API-
billed route.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import random
import re
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import Future
from contextlib import contextmanager
from dataclasses import replace
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Tuple

from .base import (
    BaseLLMBackend,
    parse_json_content,
    tool_input_schemas,
    validate_json_value,
)
from .config import BackendConfig
from .codex_dynamic_tools import openai_tools_to_dynamic, tool_result
from .errors import (
    AuthenticationError,
    BackendCancelledError,
    BackendConfigurationError,
    BackendProcessError,
    BackendRateLimitError,
    BackendTimeoutError,
    InvalidResponseError,
    ToolExecutionError,
    UnsupportedCapabilityError,
)
from .types import LLMRequest, LLMResponse

logger = logging.getLogger(__name__)

_VERSION_RE = re.compile(r"(\d+)\.(\d+)\.(\d+)")
_REDACT_RE = re.compile(
    r"(?i)(access[_ -]?token|refresh[_ -]?token|api[_ -]?key|authorization)"
    r"\s*[:=]\s*([^\s,}]+)"
)


def _redact(text: str) -> str:
    text = _REDACT_RE.sub(r"\1=<redacted>", text)
    return re.sub(r"(https?://[^\s?]+)\?[^\s]+", r"\1?<redacted>", text)


def _trace(message: str) -> None:
    if os.environ.get("CHEMEAGLE_TRACE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        print(f"[ChemEAGLE Codex] {message}", file=sys.stderr, flush=True)


def _version_tuple(value: str) -> Tuple[int, int, int]:
    match = _VERSION_RE.search(value)
    if not match:
        raise BackendProcessError(f"Could not parse Codex version from {value!r}")
    return tuple(int(part) for part in match.groups())  # type: ignore[return-value]


class JsonRpcError(BackendProcessError):
    def __init__(self, code: Optional[int], message: str, data: Any = None):
        super().__init__(f"Codex App Server RPC error {code}: {message}")
        self.code = code
        self.message = message
        self.data = data


class CodexAppServerClient:
    """Thread-safe synchronous client around the App Server's stdio protocol."""

    def __init__(
        self,
        config: BackendConfig,
        *,
        process: Any = None,
        process_factory: Any = None,
    ):
        self.config = config
        self._process = process
        self._process_factory = process_factory or subprocess.Popen
        self._owns_process = process is None
        self._started = False
        self._closed = False
        self._next_id = 1
        self._pending: Dict[Any, "queue.Queue[Dict[str, Any]]"] = {}
        self._pending_lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._events: List[Dict[str, Any]] = []
        self._event_condition = threading.Condition()
        self._stderr_lines: List[str] = []
        self._reader_thread: Optional[threading.Thread] = None
        self._stderr_thread: Optional[threading.Thread] = None
        self._tool_executor: Optional[Mapping[str, Any]] = None
        self._tool_lock = threading.Lock()
        self._turn_activity_lock = threading.Lock()
        self._turn_activity: Dict[str, Dict[str, Any]] = {}
        self._workspace: Optional[tempfile.TemporaryDirectory[str]] = None

    @property
    def process(self) -> Any:
        if self._process is None:
            raise BackendProcessError("Codex App Server is not running")
        return self._process

    def _check_version(self) -> None:
        try:
            result = subprocess.run(
                [self.config.codex_binary, "--version"],
                check=True,
                capture_output=True,
                text=True,
                timeout=15,
            )
        except FileNotFoundError as exc:
            raise BackendProcessError(
                f"Codex binary {self.config.codex_binary!r} was not found"
            ) from exc
        except (subprocess.SubprocessError, OSError) as exc:
            raise BackendProcessError(f"Could not inspect Codex version: {exc}") from exc
        actual_text = (result.stdout or result.stderr).strip()
        minimum = self.config.codex_min_version
        if minimum and _version_tuple(actual_text) < _version_tuple(minimum):
            raise UnsupportedCapabilityError(
                f"Codex App Server >= {minimum} is required; found {actual_text}"
            )

    def _spawn(self) -> None:
        self._check_version()
        if self.config.codex_cwd:
            cwd = os.path.abspath(os.path.expanduser(self.config.codex_cwd))
            if cwd in {os.path.abspath(os.sep), os.path.abspath(os.path.expanduser("~"))}:
                raise BackendConfigurationError(
                    "CHEMEAGLE_CODEX_CWD must be a dedicated directory, not / or the user home"
                )
            os.makedirs(cwd, exist_ok=True)
        else:
            self._workspace = tempfile.TemporaryDirectory(prefix="chemeagle-codex-")
            cwd = self._workspace.name
        child_env = dict(os.environ)
        for secret_name in (
            "OPENAI_API_KEY",
            "CODEX_API_KEY",
            "API_KEY",
            "AZURE_OPENAI_API_KEY",
            "VLLM_API_KEY",
            "OLLAMA_API_KEY",
        ):
            child_env.pop(secret_name, None)
        try:
            self._process = self._process_factory(
                [self.config.codex_binary, "app-server", "--stdio"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                bufsize=1,
                cwd=cwd,
                env=child_env,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            if self._workspace is not None:
                self._workspace.cleanup()
                self._workspace = None
            raise BackendProcessError(f"Could not start Codex App Server: {exc}") from exc

    def start(self) -> None:
        if self._started:
            return
        if self._closed:
            raise BackendProcessError("Codex App Server client is closed")
        if self._process is None:
            self._spawn()
        if not self.process.stdin or not self.process.stdout:
            raise BackendProcessError("Codex App Server stdio pipes are unavailable")
        self._reader_thread = threading.Thread(
            target=self._reader_loop, name="chemeagle-codex-reader", daemon=True
        )
        self._reader_thread.start()
        if self.process.stderr is not None:
            self._stderr_thread = threading.Thread(
                target=self._stderr_loop, name="chemeagle-codex-stderr", daemon=True
            )
            self._stderr_thread.start()
        self._started = True
        try:
            self.request(
                "initialize",
                {
                    "clientInfo": {
                        "name": "chemeagle",
                        "title": "ChemEAGLE",
                        "version": "0.2.0",
                    },
                    "capabilities": {"experimentalApi": True},
                },
            )
            self.notify("initialized", {})
        except Exception:
            self.close()
            raise

    def _stderr_loop(self) -> None:
        try:
            for line in self.process.stderr:
                safe = _redact(line.rstrip())
                if safe:
                    self._stderr_lines.append(safe)
                    del self._stderr_lines[:-100]
                    logger.debug("codex app-server: %s", safe)
        except Exception:
            return

    def _reader_loop(self) -> None:
        try:
            for line in self.process.stdout:
                if not line.strip():
                    continue
                try:
                    message = json.loads(line)
                except json.JSONDecodeError as exc:
                    self._fail_pending(
                        BackendProcessError("Codex App Server emitted invalid JSONL")
                    )
                    logger.error("Invalid Codex JSONL: %s", _redact(str(exc)))
                    continue
                if "id" in message and ("result" in message or "error" in message):
                    with self._pending_lock:
                        waiter = self._pending.get(message["id"])
                    if waiter is not None:
                        waiter.put(message)
                elif "id" in message and "method" in message:
                    self._handle_server_request(message)
                elif "method" in message:
                    self._mark_notification_activity(message)
                    with self._event_condition:
                        self._events.append(message)
                        if len(self._events) > 10000:
                            del self._events[:1000]
                        self._event_condition.notify_all()
        except Exception as exc:
            self._fail_pending(BackendProcessError(f"Codex reader failed: {exc}"))
        finally:
            if not self._closed:
                code = self.process.poll()
                details = self._stderr_lines[-1] if self._stderr_lines else "no stderr"
                self._fail_pending(
                    BackendProcessError(
                        f"Codex App Server exited unexpectedly ({code}): {details}"
                    )
                )

    def _fail_pending(self, exc: Exception) -> None:
        with self._pending_lock:
            waiters = list(self._pending.values())
        for waiter in waiters:
            waiter.put({"_client_error": exc})
        with self._event_condition:
            self._events.append({"_client_error": exc})
            self._event_condition.notify_all()

    def _write(self, message: Dict[str, Any]) -> None:
        encoded = json.dumps(message, ensure_ascii=False, separators=(",", ":"))
        with self._write_lock:
            try:
                self.process.stdin.write(encoded + "\n")
                self.process.stdin.flush()
            except (BrokenPipeError, OSError, ValueError) as exc:
                raise BackendProcessError("Could not write to Codex App Server") from exc

    def request(
        self,
        method: str,
        params: Optional[Dict[str, Any]] = None,
        *,
        timeout: Optional[float] = None,
        retry_overload: bool = True,
    ) -> Dict[str, Any]:
        if not self._started and method != "initialize":
            self.start()
        attempts = max(1, self.config.max_retries if retry_overload else 1)
        for attempt in range(attempts):
            with self._pending_lock:
                request_id = self._next_id
                self._next_id += 1
                waiter: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=1)
                self._pending[request_id] = waiter
            self._write({"id": request_id, "method": method, "params": params or {}})
            try:
                message = waiter.get(timeout=timeout or self.config.timeout)
            except queue.Empty as exc:
                raise BackendTimeoutError(f"Timed out waiting for Codex RPC {method}") from exc
            finally:
                with self._pending_lock:
                    self._pending.pop(request_id, None)
            if "_client_error" in message:
                raise message["_client_error"]
            if "error" not in message:
                result = message.get("result")
                return result if isinstance(result, dict) else {"value": result}
            error = message.get("error") or {}
            code = error.get("code")
            rpc_error = JsonRpcError(code, str(error.get("message", "unknown error")), error.get("data"))
            if code != -32001 or not retry_overload or attempt == attempts - 1:
                raise rpc_error
            delay = min(10.0, (2**attempt) + random.uniform(0.0, 0.5))
            logger.warning("Codex App Server overloaded; retrying in %.2fs", delay)
            time.sleep(delay)
        raise BackendProcessError(f"Codex RPC {method} failed")

    def notify(self, method: str, params: Optional[Dict[str, Any]] = None) -> None:
        self._write({"method": method, "params": params or {}})

    def event_cursor(self) -> int:
        with self._event_condition:
            return len(self._events)

    def wait_notification(
        self,
        method: str,
        *,
        after: int = 0,
        predicate: Any = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        deadline = time.monotonic() + (timeout or self.config.timeout)
        cursor = after
        with self._event_condition:
            while True:
                while cursor < len(self._events):
                    message = self._events[cursor]
                    cursor += 1
                    if "_client_error" in message:
                        raise message["_client_error"]
                    if message.get("method") == method and (
                        predicate is None or predicate(message.get("params") or {})
                    ):
                        return message
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise BackendTimeoutError(
                        f"Timed out waiting for Codex notification {method}"
                    )
                self._event_condition.wait(remaining)

    @contextmanager
    def tool_executor(self, executor: Optional[Mapping[str, Any]]) -> Iterator[None]:
        with self._tool_lock:
            previous = self._tool_executor
            self._tool_executor = executor
        try:
            yield
        finally:
            with self._tool_lock:
                self._tool_executor = previous

    def _send_result(self, request_id: Any, result: Dict[str, Any]) -> None:
        self._write({"id": request_id, "result": result})

    def _send_error(self, request_id: Any, code: int, message: str) -> None:
        self._write({"id": request_id, "error": {"code": code, "message": message}})

    def _handle_server_request(self, message: Dict[str, Any]) -> None:
        # Tool handlers may invoke another ChemEAGLE LLM agent using this same
        # connection. Never block the sole JSONL reader while such nested RPCs
        # are waiting for responses.
        threading.Thread(
            target=self._handle_server_request_sync,
            args=(message,),
            name="chemeagle-codex-server-request",
            daemon=True,
        ).start()

    def _handle_server_request_sync(self, message: Dict[str, Any]) -> None:
        method = message.get("method")
        request_id = message.get("id")
        params = message.get("params") or {}
        if method == "item/tool/call":
            try:
                self._mark_tool_started(str(params.get("turnId") or ""))
                self._execute_tool_call(request_id, params)
            finally:
                self._mark_tool_finished(str(params.get("turnId") or ""))
            return

        if method in {"applyPatchApproval", "execCommandApproval"}:
            self._send_result(
                request_id,
                {"decision": {"denied": {"rejection": "ChemEAGLE inference forbids shell and file edits"}}},
            )
        elif method in {
            "item/commandExecution/requestApproval",
            "item/fileChange/requestApproval",
        }:
            self._send_result(request_id, {"decision": "decline"})
        elif method == "item/tool/requestUserInput":
            self._send_result(request_id, {"answers": {}})
        elif method == "currentTime/read":
            self._send_result(request_id, {"unixTimestampMs": int(time.time() * 1000)})
        else:
            self._send_error(request_id, -32601, f"Unsupported server request: {method}")

    def _execute_tool_call(self, request_id: Any, params: Dict[str, Any]) -> None:
        with self._tool_lock:
            executor = self._tool_executor
        tool_name = params.get("tool")
        handler = executor.get(tool_name) if executor is not None else None
        if handler is None:
            self._send_result(
                request_id,
                tool_result(f"Unknown or disallowed tool: {tool_name}", success=False),
            )
            _trace(f"tool/result-sent name={tool_name} success=false reason=unknown-tool")
            return
        arguments = params.get("arguments")
        if not isinstance(arguments, dict):
            self._send_result(
                request_id,
                tool_result("Tool arguments must be a JSON object", success=False),
            )
            _trace(f"tool/result-sent name={tool_name} success=false reason=invalid-arguments")
            return
        started_at = time.monotonic()
        _trace(f"tool/start name={tool_name}")
        try:
            result = handler(**arguments)
            text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
            _trace(
                f"tool/complete name={tool_name} elapsed={time.monotonic() - started_at:.3f}s "
                f"payload_chars={len(text)}"
            )
            self._send_result(request_id, tool_result(text, success=True))
            _trace(f"tool/result-sent name={tool_name} success=true")
        except Exception as exc:
            _trace(
                f"tool/error name={tool_name} elapsed={time.monotonic() - started_at:.3f}s "
                f"type={type(exc).__name__}"
            )
            self._send_result(
                request_id,
                tool_result(
                    f"Tool failed: {type(exc).__name__}: {exc}", success=False
                ),
            )
            _trace(f"tool/result-sent name={tool_name} success=false reason=tool-error")

    def register_turn_activity(self, turn_id: str) -> None:
        now = time.monotonic()
        with self._turn_activity_lock:
            self._turn_activity.setdefault(
                turn_id,
                {"active_tools": 0, "active_since": None, "last_activity": now},
            )

    def _mark_notification_activity(self, message: Dict[str, Any]) -> None:
        """Reset a turn's silence window when App Server reports progress."""
        params = message.get("params") or {}
        turn_id = params.get("turnId") or params.get("turn_id")
        for key in ("turn", "item"):
            nested = params.get(key)
            if not turn_id and isinstance(nested, dict):
                turn_id = nested.get("turnId") or nested.get("turn_id")
                if key == "turn":
                    turn_id = turn_id or nested.get("id")
        if not turn_id:
            return
        now = time.monotonic()
        with self._turn_activity_lock:
            state = self._turn_activity.setdefault(
                str(turn_id),
                {"active_tools": 0, "active_since": None, "last_activity": now},
            )
            state["last_activity"] = now

    def _mark_tool_started(self, turn_id: str) -> None:
        if not turn_id:
            return
        now = time.monotonic()
        with self._turn_activity_lock:
            state = self._turn_activity.setdefault(
                turn_id,
                {"active_tools": 0, "active_since": None, "last_activity": now},
            )
            if not state["active_tools"]:
                state["active_since"] = now
            state["active_tools"] += 1
            state["last_activity"] = now

    def _mark_tool_finished(self, turn_id: str) -> None:
        if not turn_id:
            return
        now = time.monotonic()
        with self._turn_activity_lock:
            state = self._turn_activity.get(turn_id)
            if state is None:
                return
            state["active_tools"] = max(0, state["active_tools"] - 1)
            state["last_activity"] = now
            if not state["active_tools"]:
                state["active_since"] = None

    def turn_activity(self, turn_id: str) -> Tuple[int, float, Optional[float]]:
        now = time.monotonic()
        with self._turn_activity_lock:
            state = self._turn_activity.get(turn_id)
            if state is None:
                return 0, now, None
            return (
                int(state["active_tools"]),
                float(state["last_activity"]),
                state["active_since"],
            )

    def clear_turn_activity(self, turn_id: str) -> None:
        with self._turn_activity_lock:
            self._turn_activity.pop(turn_id, None)

    def account(self, *, refresh: bool = False) -> Dict[str, Any]:
        return self.request("account/read", {"refreshToken": refresh})

    def login_start(self, *, device_code: bool = False) -> Dict[str, Any]:
        login_type = "chatgptDeviceCode" if device_code else "chatgpt"
        return self.request("account/login/start", {"type": login_type})

    def wait_login(self, login_id: str, *, after: int = 0) -> Dict[str, Any]:
        message = self.wait_notification(
            "account/login/completed",
            after=after,
            predicate=lambda params: params.get("loginId") == login_id,
            timeout=max(self.config.timeout, 600.0),
        )
        return message.get("params") or {}

    def logout(self) -> Dict[str, Any]:
        return self.request("account/logout", {})

    def models(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        cursor: Optional[str] = None
        while True:
            params: Dict[str, Any] = {"limit": 100}
            if cursor:
                params["cursor"] = cursor
            result = self.request("model/list", params)
            rows.extend(result.get("data") or [])
            cursor = result.get("nextCursor")
            if not cursor:
                return rows

    def rate_limits(self) -> Dict[str, Any]:
        return self.request("account/rateLimits/read", {})

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        process = self._process
        if process is not None and self._owns_process and process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=5)
            except Exception:
                try:
                    process.kill()
                except Exception:
                    pass
        if self._workspace is not None:
            self._workspace.cleanup()
            self._workspace = None
        if self._owns_process:
            current = threading.current_thread()
            for thread in (self._reader_thread, self._stderr_thread):
                if thread is not None and thread is not current and thread.is_alive():
                    thread.join(timeout=1)
            if process is not None:
                for name in ("stdin", "stdout", "stderr"):
                    stream = getattr(process, name, None)
                    if stream is not None:
                        try:
                            stream.close()
                        except Exception:
                            pass


def _messages_to_codex_input(messages: Iterable[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    system_parts: List[str] = []
    text_parts: List[str] = []
    images: List[Dict[str, Any]] = []
    for message in messages:
        role = str(message.get("role", "user"))
        content = message.get("content")
        destination = system_parts if role in {"system", "developer"} else text_parts
        if isinstance(content, str):
            destination.append(f"[{role}]\n{content}")
        elif isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") in {"text", "input_text"}:
                    destination.append(f"[{role}]\n{part.get('text', '')}")
                elif part.get("type") in {"image_url", "input_image"}:
                    image = part.get("image_url")
                    url = image.get("url") if isinstance(image, dict) else part.get("image_url")
                    if not isinstance(url, str) or not url.startswith("data:image/"):
                        raise UnsupportedCapabilityError(
                            "Codex image inputs must be inline data:image/... URLs"
                        )
                    images.append({"type": "image", "url": url})
        tool_calls = message.get("tool_calls")
        if tool_calls:
            text_parts.append(
                f"[{role} tool calls]\n{json.dumps(tool_calls, ensure_ascii=False)}"
            )
    if not text_parts:
        text_parts.append("[user]\nComplete the requested ChemEAGLE inference task.")
    inputs = [{"type": "text", "text": "\n\n".join(text_parts)}, *images]
    return "\n\n".join(system_parts), inputs


def _rate_limit_reached(payload: Dict[str, Any]) -> Optional[str]:
    snapshots = [payload.get("rateLimits")]
    snapshots.extend((payload.get("rateLimitsByLimitId") or {}).values())
    for snapshot in snapshots:
        if not isinstance(snapshot, dict):
            continue
        reached = snapshot.get("rateLimitReachedType")
        if reached:
            return str(reached)
        if snapshot.get("spendControlReached") is True:
            return "spend_control_reached"
        for key in ("primary", "secondary"):
            window = snapshot.get(key)
            if isinstance(window, dict) and (window.get("usedPercent") or 0) >= 100:
                return f"{key}_window_exhausted"
    return None


class CodexAppServerBackend(BaseLLMBackend):
    provider_name = "codex"

    def __init__(
        self,
        config: BackendConfig,
        *,
        client: Optional[CodexAppServerClient] = None,
    ):
        self.config = config
        self.client = client or CodexAppServerClient(config)

    def _require_subscription(self) -> Dict[str, Any]:
        account = self.client.account(refresh=False)
        details = account.get("account")
        if not isinstance(details, dict):
            raise AuthenticationError(
                "Codex is not signed in. Run: python -m chemeagle_llm.codex_auth login"
            )
        if details.get("type") != "chatgpt":
            raise AuthenticationError(
                "Codex backend requires ChatGPT subscription login; the active Codex account uses "
                f"{details.get('type')!r}. Sign out and log in with ChatGPT."
            )
        try:
            limits = self.client.rate_limits()
        except JsonRpcError:
            limits = {}
        reached = _rate_limit_reached(limits)
        if reached:
            raise BackendRateLimitError(f"Codex subscription limit reached: {reached}")
        return details

    def account_status(self) -> Dict[str, Any]:
        account = self.client.account(refresh=True)
        try:
            account["rateLimits"] = self.client.rate_limits()
        except JsonRpcError as exc:
            account["rateLimitsError"] = str(exc)
        return account

    def models(self) -> List[Dict[str, Any]]:
        return self.client.models()

    def login(self, *, device_code: bool = False) -> Dict[str, Any]:
        cursor = self.client.event_cursor()
        started = self.client.login_start(device_code=device_code)
        started["eventCursor"] = cursor
        return started

    def wait_login(self, login_id: str, *, after: int = 0) -> Dict[str, Any]:
        return self.client.wait_login(login_id, after=after)

    def logout(self) -> Dict[str, Any]:
        return self.client.logout()

    def _wait_for_turn(
        self,
        *,
        thread_id: str,
        turn_id: str,
        after: int,
        timeout: float,
        tool_timeout: Optional[float] = None,
        cancel_event: Any = None,
    ) -> Dict[str, Any]:
        last_activity = time.monotonic()

        def interrupt() -> None:
            try:
                self.client.request(
                    "turn/interrupt",
                    {"threadId": thread_id, "turnId": turn_id},
                    timeout=min(5.0, timeout),
                    retry_overload=False,
                )
            except BackendProcessError:
                pass

        while True:
            if cancel_event is not None and cancel_event.is_set():
                try:
                    interrupt()
                finally:
                    raise BackendCancelledError("Codex turn was cancelled")

            active_tools, activity_at, active_since = self.client.turn_activity(turn_id)
            last_activity = max(last_activity, activity_at)
            now = time.monotonic()
            if active_tools:
                if (
                    tool_timeout is not None
                    and active_since is not None
                    and now - active_since >= tool_timeout
                ):
                    interrupt()
                    raise ToolExecutionError(
                        f"Codex dynamic tool did not finish within {tool_timeout:g}s"
                    )
                remaining = 0.2
            else:
                remaining = timeout - (now - last_activity)
            if remaining <= 0:
                interrupt()
                raise BackendTimeoutError(
                    f"Codex produced no turn completion within {timeout:g}s of its last recorded activity"
                )
            try:
                return self.client.wait_notification(
                    "turn/completed",
                    after=after,
                    predicate=lambda params: (
                        params.get("threadId") == thread_id
                        and (params.get("turn") or {}).get("id") == turn_id
                    ),
                    timeout=min(0.2, max(remaining, 0.001)),
                )
            except BackendTimeoutError:
                continue

    def _run(
        self,
        request: LLMRequest,
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
        executor: Optional[Mapping[str, Any]] = None,
    ) -> LLMResponse:
        if request.cancel_event is not None and request.cancel_event.is_set():
            raise BackendCancelledError("Codex request was cancelled")
        self.client.start()
        self._require_subscription()
        system_text, inputs = _messages_to_codex_input(request.messages)
        instructions = (
            "You are the language-model reasoning backend inside ChemEAGLE, a chemical "
            "information-extraction pipeline. Do not inspect files, execute shell commands, "
            "edit the workspace, browse the web, or call tools other than the explicitly "
            "registered ChemEAGLE functions. Return only the requested task result."
        )
        if system_text:
            instructions += "\n\nApplication instructions:\n" + system_text
        thread_params: Dict[str, Any] = {
            "cwd": self.config.codex_cwd or (
                self.client._workspace.name if self.client._workspace is not None else os.getcwd()
            ),
            "sandbox": "read-only",
            "approvalPolicy": "never",
            "baseInstructions": instructions,
            "ephemeral": True,
        }
        selected_model = request.model or self.config.model
        if selected_model:
            thread_params["model"] = selected_model
        if tools:
            if executor is None:
                raise UnsupportedCapabilityError(
                    "Codex dynamic tools require an explicit whitelist executor"
                )
            thread_params["dynamicTools"] = openai_tools_to_dynamic(tools)

        with self.client.tool_executor(executor):
            started = self.client.request("thread/start", thread_params)
            thread = started.get("thread") or {}
            thread_id = thread.get("id")
            if not thread_id:
                raise BackendProcessError("Codex thread/start returned no thread id")
            _trace(
                f"thread/start id={thread_id} model="
                f"{started.get('model') or selected_model or 'default'}"
            )
            output_schema = request.output_schema
            attempts = max(1, min(self.config.max_retries, 3))
            correction_retry = False
            for attempt in range(attempts):
                if correction_retry:
                    turn_inputs = [
                        {
                            "type": "text",
                            "text": (
                                "The previous answer was not valid JSON matching the requested "
                                "schema. Return a corrected JSON object only."
                            ),
                        }
                    ]
                else:
                    turn_inputs = inputs
                correction_retry = False
                turn_params: Dict[str, Any] = {
                    "threadId": thread_id,
                    "input": turn_inputs,
                }
                if output_schema:
                    turn_params["outputSchema"] = output_schema
                if selected_model:
                    turn_params["model"] = selected_model
                cursor = self.client.event_cursor()
                started_turn = self.client.request("turn/start", turn_params)
                turn_id = str((started_turn.get("turn") or {}).get("id") or "")
                if not turn_id:
                    raise BackendProcessError("Codex turn/start returned no turn id")
                self.client.register_turn_activity(turn_id)
                turn_started_at = time.monotonic()
                _trace(
                    f"turn/start id={turn_id} attempt={attempt + 1}/{attempts} "
                    f"tools={len(tools or [])} timeout={request.timeout or self.config.timeout:g}s"
                )
                try:
                    completed = self._wait_for_turn(
                        thread_id=thread_id,
                        turn_id=turn_id,
                        after=cursor,
                        timeout=request.timeout or self.config.timeout,
                        tool_timeout=self.config.tool_timeout if tools else None,
                        cancel_event=request.cancel_event,
                    )
                except BackendTimeoutError as exc:
                    _trace(
                        f"turn/error id={turn_id} type={type(exc).__name__} "
                        f"elapsed={time.monotonic() - turn_started_at:.3f}s"
                    )
                    if attempt + 1 < attempts:
                        logger.warning(
                            "Codex turn timed out; resubmitting the original request (%d/%d)",
                            attempt + 2,
                            attempts,
                        )
                        continue
                    raise
                except Exception as exc:
                    _trace(
                        f"turn/error id={turn_id} type={type(exc).__name__} "
                        f"elapsed={time.monotonic() - turn_started_at:.3f}s"
                    )
                    raise
                finally:
                    self.client.clear_turn_activity(turn_id)
                turn = (completed.get("params") or {}).get("turn") or {}
                status = turn.get("status")
                _trace(
                    f"turn/complete id={turn_id} status={status} "
                    f"elapsed={time.monotonic() - turn_started_at:.3f}s"
                )
                if status == "cancelled":
                    raise BackendCancelledError("Codex turn was cancelled")
                if status != "completed":
                    error = turn.get("error") or {}
                    info = error.get("codexErrorInfo")
                    message = str(
                        error.get("message") or f"Codex turn ended with {status}"
                    )
                    if info == "usageLimitExceeded" or "usage limit" in message.lower():
                        raise BackendRateLimitError(message)
                    raise BackendProcessError(message)
                agent_messages = [
                    item.get("text")
                    for item in turn.get("items") or []
                    if isinstance(item, dict)
                    and item.get("type") == "agentMessage"
                    and item.get("text")
                ]
                if not agent_messages:
                    raise InvalidResponseError("Codex completed without an agent message")
                response = LLMResponse(
                    content=agent_messages[-1],
                    model=started.get("model") or selected_model,
                    metadata={"thread_id": thread_id, "turn_id": turn.get("id")},
                )
                if request.json_mode or request.output_schema:
                    try:
                        parse_json_content(response, request.output_schema)
                    except InvalidResponseError:
                        if attempt + 1 < attempts:
                            logger.warning(
                                "Codex returned invalid structured output; requesting correction"
                            )
                            correction_retry = True
                            continue
                        raise
                return response
        raise InvalidResponseError("Codex structured-output retries were exhausted")

    def generate(self, request: LLMRequest) -> LLMResponse:
        if request.tools:
            raise UnsupportedCapabilityError(
                "Use run_tool_loop() for Codex requests containing function tools"
            )
        return self._run(request)

    def run_tool_loop(
        self,
        request: LLMRequest,
        tools: list[Dict[str, Any]],
        executor: Mapping[str, Any],
    ) -> LLMResponse:
        schemas = tool_input_schemas(tools)
        call_cache: Dict[str, Future[Any]] = {}
        call_cache_lock = threading.Lock()

        def bind(name: str, handler: Any) -> Any:
            def invoke(**arguments: Any) -> Any:
                schema = schemas.get(name)
                if schema is not None:
                    validate_json_value(arguments, schema)
                cache_key = name + ":" + json.dumps(
                    arguments,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                with call_cache_lock:
                    future = call_cache.get(cache_key)
                    owns_call = future is None
                    if future is None:
                        future = Future()
                        call_cache[cache_key] = future
                if owns_call:
                    try:
                        future.set_result(handler(**arguments))
                    except BaseException as exc:
                        future.set_exception(exc)
                        future.exception()
                        with call_cache_lock:
                            if call_cache.get(cache_key) is future:
                                call_cache.pop(cache_key, None)
                        raise
                else:
                    _trace(f"tool/cache-hit name={name}")
                return future.result()

            return invoke

        validated_executor = {
            name: bind(name, handler)
            for name, handler in executor.items()
            if name in schemas
        }
        return self._run(
            replace(
                request,
                tools=[],
                timeout=request.timeout or self.config.timeout,
            ),
            tools=tools,
            executor=validated_executor,
        )

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> "CodexAppServerBackend":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()
