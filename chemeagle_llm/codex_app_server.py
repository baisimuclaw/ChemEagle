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
    remap_fallback_tool_result,
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
from .types import LLMRequest, LLMResponse, LLMToolOutput

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


class _ToolExecutorRegistration:
    """Bind one allowlist to every App Server thread created in a scope.

    A ChemEAGLE dynamic tool may itself start a nested Codex turn.  App Server
    requests include the owning ``threadId``, so routing by that id prevents a
    nested turn from temporarily replacing the outer turn's allowlist.
    """

    def __init__(
        self,
        client: "CodexAppServerClient",
        executor: Optional[Mapping[str, Any]],
    ) -> None:
        self.client = client
        self.executor = executor
        self._thread_ids: List[str] = []

    def bind(self, thread_id: str) -> None:
        if self.executor is None or thread_id in self._thread_ids:
            return
        with self.client._tool_lock:
            self.client._tool_executors.setdefault(thread_id, []).append(
                self.executor
            )
        self._thread_ids.append(thread_id)

    def close(self) -> None:
        if self.executor is None:
            return
        with self.client._tool_lock:
            for thread_id in reversed(self._thread_ids):
                stack = self.client._tool_executors.get(thread_id)
                if not stack:
                    continue
                for index in range(len(stack) - 1, -1, -1):
                    if stack[index] is self.executor:
                        stack.pop(index)
                        break
                if not stack:
                    self.client._tool_executors.pop(thread_id, None)
        self._thread_ids.clear()


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
        self._tool_executors: Dict[str, List[Mapping[str, Any]]] = {}
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
            try:
                waiter.put_nowait({"_client_error": exc})
            except queue.Full:
                # A real reply already won the race.  Never let one full
                # single-item waiter block the sole JSONL reader thread.
                continue
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
        attempts = min(
            3,
            max(1, self.config.max_retries if retry_overload else 1),
        )
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
    def tool_executor(
        self, executor: Optional[Mapping[str, Any]]
    ) -> Iterator[_ToolExecutorRegistration]:
        registration = _ToolExecutorRegistration(self, executor)
        try:
            yield registration
        finally:
            registration.close()

    def _send_result(self, request_id: Any, result: Dict[str, Any]) -> None:
        self._write({"id": request_id, "result": result})

    def _send_error(self, request_id: Any, code: int, message: str) -> None:
        self._write({"id": request_id, "error": {"code": code, "message": message}})

    def _handle_server_request(self, message: Dict[str, Any]) -> None:
        # Tool handlers may invoke another ChemEAGLE LLM agent using this same
        # connection. Never block the sole JSONL reader while such nested RPCs
        # are waiting for responses.
        if message.get("method") == "item/tool/call":
            params = message.get("params") or {}
            self._record_tool_trace(
                str(params.get("turnId") or ""),
                {
                    "call_id": str(params.get("callId") or ""),
                    "name": str(params.get("tool") or ""),
                    "arguments": params.get("arguments"),
                    "output": None,
                    "supplemental_content": [],
                    "success": None,
                    "failure_kind": None,
                },
            )
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
        thread_id = str(params.get("threadId") or "")
        with self._tool_lock:
            stack = self._tool_executors.get(thread_id) or []
            executor = stack[-1] if stack else None
        tool_name = params.get("tool")
        handler = executor.get(tool_name) if executor is not None else None
        if handler is None:
            failure = f"Unknown or disallowed tool: {tool_name}"
            self._complete_tool_trace(
                str(params.get("turnId") or ""),
                str(params.get("callId") or ""),
                output=failure,
                supplemental_content=[],
                success=False,
                failure_kind="unknown_tool",
            )
            self._send_result(
                request_id,
                tool_result(failure, success=False),
            )
            _trace(f"tool/result-sent name={tool_name} success=false reason=unknown-tool")
            return
        arguments = params.get("arguments")
        if getattr(executor, "ignore_tool_arguments", False):
            arguments = {}
        if not isinstance(arguments, dict):
            failure = "Tool arguments must be a JSON object"
            self._complete_tool_trace(
                str(params.get("turnId") or ""),
                str(params.get("callId") or ""),
                output=failure,
                supplemental_content=[],
                success=False,
                failure_kind="invalid_arguments",
            )
            self._send_result(
                request_id,
                tool_result(failure, success=False),
            )
            _trace(f"tool/result-sent name={tool_name} success=false reason=invalid-arguments")
            return
        started_at = time.monotonic()
        _trace(f"tool/start name={tool_name}")
        turn_id = str(params.get("turnId") or "")
        call_id = str(params.get("callId") or "")
        try:
            result = handler(**arguments)
            supplemental_content = []
            if isinstance(result, LLMToolOutput):
                supplemental_content = result.supplemental_content
                result = result.value
            text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
            _trace(
                f"tool/complete name={tool_name} elapsed={time.monotonic() - started_at:.3f}s "
                f"payload_chars={len(text)}"
            )
            self._complete_tool_trace(
                turn_id,
                call_id,
                output=text,
                supplemental_content=supplemental_content,
                success=True,
                failure_kind=None,
            )
            self._send_result(request_id, tool_result(text, success=True))
            _trace(f"tool/result-sent name={tool_name} success=true")
        except Exception as exc:
            failure = f"Tool failed: {type(exc).__name__}: {exc}"
            self._complete_tool_trace(
                turn_id,
                call_id,
                output=failure,
                supplemental_content=[],
                success=False,
                failure_kind="tool_error",
            )
            _trace(
                f"tool/error name={tool_name} elapsed={time.monotonic() - started_at:.3f}s "
                f"type={type(exc).__name__}"
            )
            self._send_result(
                request_id,
                tool_result(failure, success=False),
            )
            _trace(f"tool/result-sent name={tool_name} success=false reason=tool-error")

    def register_turn_activity(self, turn_id: str) -> None:
        now = time.monotonic()
        with self._turn_activity_lock:
            self._turn_activity.setdefault(
                turn_id,
                {
                    "active_tools": 0,
                    "active_since": None,
                    "tool_elapsed": 0.0,
                    "last_activity": now,
                    "tool_trace": [],
                },
            )

    def _record_tool_trace(self, turn_id: str, entry: Dict[str, Any]) -> None:
        if not turn_id:
            return
        now = time.monotonic()
        with self._turn_activity_lock:
            state = self._turn_activity.setdefault(
                turn_id,
                {
                    "active_tools": 0,
                    "active_since": None,
                    "tool_elapsed": 0.0,
                    "last_activity": now,
                    "tool_trace": [],
                },
            )
            state.setdefault("tool_trace", []).append(entry)
            state["last_activity"] = now

    def _complete_tool_trace(
        self,
        turn_id: str,
        call_id: str,
        *,
        output: str,
        supplemental_content: List[Dict[str, Any]],
        success: bool,
        failure_kind: Optional[str],
    ) -> None:
        if not turn_id:
            return
        now = time.monotonic()
        with self._turn_activity_lock:
            state = self._turn_activity.get(turn_id)
            if state is None:
                return
            for entry in reversed(state.get("tool_trace") or []):
                if entry.get("call_id") == call_id:
                    entry["output"] = output
                    entry["supplemental_content"] = supplemental_content
                    entry["success"] = success
                    entry["failure_kind"] = failure_kind
                    break
            state["last_activity"] = now

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
                {
                    "active_tools": 0,
                    "active_since": None,
                    "tool_elapsed": 0.0,
                    "last_activity": now,
                    "tool_trace": [],
                },
            )
            state["last_activity"] = now

    def _mark_tool_started(self, turn_id: str) -> None:
        if not turn_id:
            return
        now = time.monotonic()
        with self._turn_activity_lock:
            state = self._turn_activity.setdefault(
                turn_id,
                {
                    "active_tools": 0,
                    "active_since": None,
                    "tool_elapsed": 0.0,
                    "last_activity": now,
                    "tool_trace": [],
                },
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
                active_since = state.get("active_since")
                if active_since is not None:
                    state["tool_elapsed"] = float(
                        state.get("tool_elapsed", 0.0)
                    ) + max(0.0, now - float(active_since))
                state["active_since"] = None

    def turn_activity(
        self, turn_id: str
    ) -> Tuple[int, float, Optional[float], float]:
        now = time.monotonic()
        with self._turn_activity_lock:
            state = self._turn_activity.get(turn_id)
            if state is None:
                return 0, now, None, 0.0
            return (
                int(state["active_tools"]),
                float(state["last_activity"]),
                state["active_since"],
                float(state.get("tool_elapsed", 0.0)),
            )

    def clear_turn_activity(self, turn_id: str) -> List[Dict[str, Any]]:
        with self._turn_activity_lock:
            state = self._turn_activity.pop(turn_id, None) or {}
        return list(state.get("tool_trace") or [])

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


def _inline_image_url(part: Dict[str, Any]) -> str:
    image = part.get("image_url")
    url = image.get("url") if isinstance(image, dict) else image
    if not isinstance(url, str) or not url.startswith("data:image/"):
        raise UnsupportedCapabilityError(
            "Codex image inputs must be inline data:image/... URLs"
        )
    return url


def _message_content_items(role: str, content: Any) -> List[Dict[str, Any]]:
    output = role == "assistant"
    text_type = "output_text" if output else "input_text"
    items: List[Dict[str, Any]] = []
    if isinstance(content, str):
        items.append({"type": text_type, "text": content})
    elif isinstance(content, list):
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") in {"text", "input_text", "output_text"}:
                items.append({"type": text_type, "text": str(part.get("text", ""))})
            elif part.get("type") in {"image_url", "input_image"}:
                if output:
                    raise UnsupportedCapabilityError(
                        "Assistant history cannot contain input images"
                    )
                items.append(
                    {"type": "input_image", "image_url": _inline_image_url(part)}
                )
    return items


def _message_to_turn_inputs(message: Dict[str, Any]) -> List[Dict[str, Any]]:
    inputs: List[Dict[str, Any]] = []
    content = message.get("content")
    if isinstance(content, str):
        inputs.append({"type": "text", "text": content})
    elif isinstance(content, list):
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") in {"text", "input_text"}:
                inputs.append({"type": "text", "text": str(part.get("text", ""))})
            elif part.get("type") in {"image_url", "input_image"}:
                inputs.append({"type": "image", "url": _inline_image_url(part)})
    return inputs


def _message_to_response_items(message: Dict[str, Any]) -> List[Dict[str, Any]]:
    role = str(message.get("role", "user"))
    if role == "tool":
        call_id = message.get("tool_call_id")
        if not isinstance(call_id, str) or not call_id:
            raise UnsupportedCapabilityError(
                "Tool history requires a non-empty tool_call_id"
            )
        content = message.get("content")
        output = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
        return [
            {
                "type": "function_call_output",
                "call_id": call_id,
                "output": output,
            }
        ]

    if role not in {"user", "assistant"}:
        raise UnsupportedCapabilityError(
            f"Unsupported Codex conversation-history role: {role!r}"
        )
    items: List[Dict[str, Any]] = []
    content_items = _message_content_items(role, message.get("content"))
    if content_items:
        items.append({"type": "message", "role": role, "content": content_items})
    for call in message.get("tool_calls") or []:
        function = call.get("function") if isinstance(call, dict) else None
        call_id = call.get("id") if isinstance(call, dict) else None
        if not isinstance(function, dict) or not isinstance(call_id, str) or not call_id:
            raise UnsupportedCapabilityError(
                "Assistant tool-call history requires id and function fields"
            )
        name = function.get("name")
        arguments = function.get("arguments", "{}")
        if not isinstance(name, str) or not name:
            raise UnsupportedCapabilityError(
                "Assistant tool-call history requires a function name"
            )
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, ensure_ascii=False)
        items.append(
            {
                "type": "function_call",
                "call_id": call_id,
                "name": name,
                "arguments": arguments,
            }
        )
    return items


def _messages_to_codex_context(
    messages: Iterable[Dict[str, Any]],
) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
    system_parts: List[str] = []
    conversation: List[Dict[str, Any]] = []
    for message in messages:
        role = str(message.get("role", "user"))
        if role in {"system", "developer"}:
            content = message.get("content")
            if isinstance(content, str):
                system_parts.append(content)
            elif isinstance(content, list):
                system_parts.extend(
                    str(part.get("text", ""))
                    for part in content
                    if isinstance(part, dict)
                    and part.get("type") in {"text", "input_text"}
                )
            continue
        conversation.append(message)

    # Consecutive user messages at the end belong to the current turn. Every
    # earlier role is injected as a native Responses item, preserving assistant
    # function calls and function-call outputs without textual role labels.
    split = len(conversation)
    while split > 0 and conversation[split - 1].get("role", "user") == "user":
        split -= 1
    history_messages = conversation[:split]
    current_messages = conversation[split:]
    history_items: List[Dict[str, Any]] = []
    for message in history_messages:
        history_items.extend(_message_to_response_items(message))
    turn_inputs: List[Dict[str, Any]] = []
    for message in current_messages:
        turn_inputs.extend(_message_to_turn_inputs(message))
    return "\n\n".join(system_parts), history_items, turn_inputs


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


def _codex_error_kind(info: Any) -> Optional[str]:
    if isinstance(info, str):
        return info
    if isinstance(info, dict) and info:
        return str(next(iter(info)))
    return None


_RETRYABLE_TURN_ERRORS = {
    "serverOverloaded",
    "internalServerError",
    "httpConnectionFailed",
    "responseStreamConnectionFailed",
    "responseStreamDisconnected",
    "responseTooManyFailedAttempts",
}


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
        self._resolved_models: List[str] = []
        self._resolved_models_lock = threading.Lock()

    def _record_resolved_model(self, model: Optional[str]) -> None:
        if not model:
            return
        with self._resolved_models_lock:
            if model not in self._resolved_models:
                self._resolved_models.append(model)

    def runtime_metadata(self) -> Dict[str, Any]:
        with self._resolved_models_lock:
            resolved = list(self._resolved_models)
        return {
            "provider": self.provider_name,
            "configured_model": self.config.model,
            "resolved_models": resolved,
        }

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
        turn_started = time.monotonic()
        last_activity = turn_started
        total_non_tool_timeout = max(timeout * 3.0, timeout)

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

            active_tools, activity_at, active_since, tool_elapsed = (
                self.client.turn_activity(turn_id)
            )
            last_activity = max(last_activity, activity_at)
            now = time.monotonic()
            current_tool_elapsed = (
                max(0.0, now - active_since)
                if active_tools and active_since is not None
                else 0.0
            )
            non_tool_elapsed = max(
                0.0,
                now - turn_started - tool_elapsed - current_tool_elapsed,
            )
            if non_tool_elapsed >= total_non_tool_timeout:
                interrupt()
                raise BackendTimeoutError(
                    "Codex turn exceeded its absolute non-tool runtime limit "
                    f"of {total_non_tool_timeout:g}s"
                )
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
        system_text, history_items, inputs = _messages_to_codex_context(
            request.messages
        )
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

        with self.client.tool_executor(executor) as tool_registration:
            def start_thread() -> Tuple[Dict[str, Any], str, Optional[str]]:
                started_thread = self.client.request("thread/start", thread_params)
                thread = started_thread.get("thread") or {}
                new_thread_id = thread.get("id")
                if not new_thread_id:
                    raise BackendProcessError(
                        "Codex thread/start returned no thread id"
                    )
                tool_registration.bind(str(new_thread_id))
                model = started_thread.get("model") or selected_model
                self._record_resolved_model(model)
                if history_items:
                    self.client.request(
                        "thread/inject_items",
                        {
                            "threadId": str(new_thread_id),
                            "items": history_items,
                        },
                    )
                _trace(
                    f"thread/start id={new_thread_id} model={model or 'default'} "
                    f"history_items={len(history_items)}"
                )
                return started_thread, str(new_thread_id), model

            started, thread_id, resolved_model = start_thread()
            # Codex outputSchema enables strict Structured Outputs, unlike the
            # upstream Azure ``json_object`` mode.  A generic ``{"type":
            # "object"}`` is not a valid strict schema, so only send schemas
            # that the caller explicitly supplied.  Generic JSON mode remains
            # locally validated as an object, matching Azure JSON mode.
            output_schema = request.output_schema
            validation_schema = output_schema
            if validation_schema is None and request.json_mode:
                validation_schema = {"type": "object"}
            attempts = max(1, min(self.config.max_retries, 3))
            correction_retry = False
            for attempt in range(attempts):
                if correction_retry:
                    turn_inputs = [
                        {
                            "type": "text",
                            "text": (
                                "The previous answer was not a valid JSON object"
                                + (
                                    " matching the requested schema"
                                    if output_schema is not None
                                    else ""
                                )
                                + ". Return a corrected JSON object only."
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
                    tool_trace: List[Dict[str, Any]] = []
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
                        # Interrupted turns remain in Codex conversation history.
                        # A fresh ephemeral thread makes this a genuinely
                        # independent retry while the outer tool-call cache
                        # still avoids duplicate GPU inference.
                        started, thread_id, resolved_model = start_thread()
                        continue
                    raise
                except Exception as exc:
                    _trace(
                        f"turn/error id={turn_id} type={type(exc).__name__} "
                        f"elapsed={time.monotonic() - turn_started_at:.3f}s"
                    )
                    raise
                finally:
                    tool_trace = self.client.clear_turn_activity(turn_id)
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
                    error_kind = _codex_error_kind(info)
                    message = str(
                        error.get("message") or f"Codex turn ended with {status}"
                    )
                    if error_kind == "usageLimitExceeded" or "usage limit" in message.lower():
                        raise BackendRateLimitError(message)
                    if error_kind == "unauthorized":
                        raise AuthenticationError(message)
                    if (
                        error_kind in _RETRYABLE_TURN_ERRORS
                        and attempt + 1 < attempts
                    ):
                        logger.warning(
                            "Codex turn failed transiently (%s); resubmitting "
                            "the original request (%d/%d)",
                            error_kind,
                            attempt + 2,
                            attempts,
                        )
                        started, thread_id, resolved_model = start_thread()
                        continue
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
                    model=resolved_model,
                    metadata={
                        "thread_id": thread_id,
                        "turn_id": turn.get("id"),
                        "tool_trace": tool_trace,
                    },
                )
                should_validate_json = request.json_mode or request.output_schema
                if should_validate_json and tools and (
                    tool_trace or request.defer_json_validation_for_tools
                ):
                    # Match the upstream two-completion contract.  Once a
                    # dynamic tool has run, run_tool_loop must inspect that
                    # tool trace before any JSON-only correction can begin.
                    # A correction turn has tools disabled and preserves the
                    # completed function-call history explicitly.
                    should_validate_json = False
                if should_validate_json:
                    try:
                        parse_json_content(response, validation_schema)
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
                effective_arguments = (
                    {} if request.ignore_tool_arguments else arguments
                )
                schema = schemas.get(name)
                if (
                    schema is not None
                    and not request.ignore_tool_arguments
                    and request.validate_tool_arguments
                ):
                    validate_json_value(effective_arguments, schema)
                cache_key = name + ":" + json.dumps(
                    effective_arguments,
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
                        future.set_result(handler(**effective_arguments))
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

        class PolicyExecutor(dict):
            ignore_tool_arguments = request.ignore_tool_arguments

            def get(self, key: Any, default: Any = None) -> Any:
                handler = super().get(key)
                if handler is not None:
                    return handler
                if request.unknown_tool_policy == "fallback" and len(self) == 1:
                    fallback_name, fallback_handler = next(iter(self.items()))

                    def invoke_fallback(**arguments: Any) -> Any:
                        return remap_fallback_tool_result(
                            fallback_handler(**arguments),
                            fallback_name,
                            str(key),
                        )

                    return invoke_fallback
                return default

        validated_executor = PolicyExecutor(
            {
                name: bind(name, handler)
                for name, handler in executor.items()
                if name in schemas
            }
        )
        first = self._run(
            replace(
                request,
                tools=[],
                timeout=request.timeout or self.config.timeout,
            ),
            tools=tools,
            executor=validated_executor,
        )
        trace = getattr(first, "metadata", {}).get("tool_trace") or []
        supplemental_content = list(request.tool_followup_content)

        def tool_history_messages() -> List[Dict[str, Any]]:
            calls = [
                {
                    "id": entry["call_id"],
                    "type": "function",
                    "function": {
                        "name": entry["name"],
                        "arguments": json.dumps(
                            entry.get("arguments") or {},
                            ensure_ascii=False,
                        ),
                    },
                }
                for entry in trace
                if entry.get("call_id") and entry.get("name")
            ]
            messages: List[Dict[str, Any]] = []
            if calls:
                messages.append(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": calls,
                    }
                )
            messages.extend(
                {
                    "role": "tool",
                    "name": entry["name"],
                    "tool_call_id": entry["call_id"],
                    "content": str(entry.get("output") or ""),
                }
                for entry in trace
                if entry.get("call_id") and entry.get("name")
            )
            return messages

        failed_trace = [
            entry
            for entry in trace
            if entry.get("success") is False
            and not (
                request.unknown_tool_policy == "skip"
                and entry.get("failure_kind") == "unknown_tool"
            )
        ]
        if failed_trace:
            failed = failed_trace[0]
            raise ToolExecutionError(
                "Codex dynamic tool failed: "
                f"{failed.get('name')}: {failed.get('output')}"
            )
        if request.require_tool_call and not any(
            entry.get("success") is True for entry in trace
        ):
            raise ToolExecutionError(
                "Model did not call a required ChemEAGLE tool"
            )
        if (
            not trace
            and request.followup_on_no_tool_call
            and not supplemental_content
        ):
            messages = list(request.messages)
            messages.append(first.assistant_message())
            return self.generate(
                replace(
                    request,
                    messages=messages,
                    tools=[],
                    tool_choice=None,
                    followup_on_no_tool_call=False,
                    defer_json_validation_for_tools=False,
                    timeout=request.timeout or self.config.timeout,
                )
            )
        if not supplemental_content:
            for entry in trace:
                supplemental_content.extend(
                    entry.get("supplemental_content") or []
                )
        if not supplemental_content:
            if trace and (request.json_mode or request.output_schema):
                validation_schema = request.output_schema
                if validation_schema is None and request.json_mode:
                    validation_schema = {"type": "object"}
                try:
                    parse_json_content(first, validation_schema)
                except InvalidResponseError:
                    # Upstream validates only after function results are
                    # available.  Correct the final answer with tools disabled
                    # while retaining the native function-call/result history.
                    messages = list(request.messages)
                    messages.extend(tool_history_messages())
                    if first.content:
                        messages.append(
                            {"role": "assistant", "content": first.content}
                        )
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "The previous answer was not a valid JSON object"
                                + (
                                    " matching the requested schema"
                                    if request.output_schema is not None
                                    else ""
                                )
                                + ". Return a corrected JSON object only."
                            ),
                        }
                    )
                    return self.generate(
                        replace(
                            request,
                            messages=messages,
                            tools=[],
                            tool_choice=None,
                            tool_followup_content=[],
                            followup_on_no_tool_call=False,
                            defer_json_validation_for_tools=False,
                            timeout=request.timeout or self.config.timeout,
                        )
                    )
            return first

        # The upstream product-variant R-group agent performs two completions.
        # Its second message list contains the annotated user image first,
        # followed by the assistant's native function calls and the matching
        # function-call outputs. Rebuild that history rather than placing the
        # annotation inside a tool result or flattening roles into prose.
        messages = [
            message
            for message in request.messages
            if message.get("role") in {"system", "developer"}
        ]
        messages.append({"role": "user", "content": supplemental_content})
        if trace:
            messages.extend(tool_history_messages())
        else:
            messages.append(first.assistant_message())
        return self.generate(
            replace(
                request,
                messages=messages,
                tools=[],
                tool_choice=None,
                tool_followup_content=[],
                followup_on_no_tool_call=False,
                defer_json_validation_for_tools=False,
                timeout=request.timeout or self.config.timeout,
            )
        )

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> "CodexAppServerBackend":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()
