"""Persistent SSH and Slurm-over-SSH clients for the offline vision worker."""

from __future__ import annotations

import json
import os
import queue
import shlex
import subprocess
import threading
from typing import Any, Dict, List, Optional

from .codec import decode_value, encode_value
from .config import VisionConfig
from .errors import (
    VisionBackendError,
    VisionConfigurationError,
    VisionProcessError,
    VisionProtocolError,
    VisionTimeoutError,
)


class RemoteVisionBackend:
    def __init__(self, config: VisionConfig, *, process_factory: Any = None):
        config.validate_remote()
        self.config = config
        self.provider_name = config.provider
        self._process_factory = process_factory or subprocess.Popen
        self._process: Any = None
        self._started = False
        self._closed = False
        self._next_id = 1
        self._start_lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._pending_lock = threading.Lock()
        self._pending: Dict[int, "queue.Queue[Dict[str, Any]]"] = {}
        self._stderr_lines: List[str] = []
        self._reader_thread: Optional[threading.Thread] = None
        self._stderr_thread: Optional[threading.Thread] = None

    @property
    def process(self) -> Any:
        if self._process is None:
            raise VisionProcessError("Remote vision worker is not running")
        return self._process

    def worker_argv(self) -> List[str]:
        argv = [
            self.config.remote_python,
            "-u",
            "-m",
            "chemeagle_vision.worker",
            "--stdio",
            "--device",
            self.config.device,
        ]
        if self.config.offline:
            argv.append("--offline")
        if self.config.model_dir:
            argv.extend(["--model-dir", self.config.model_dir])
        return argv

    def remote_argv(self) -> List[str]:
        worker = self.worker_argv()
        if self.config.provider != "slurm-ssh":
            return worker
        argv = ["srun", "--quiet", "--unbuffered"]
        optional = (
            ("--account", self.config.slurm_account),
            ("--qos", self.config.slurm_qos),
            ("--reservation", self.config.slurm_reservation),
            ("--partition", self.config.slurm_partition),
        )
        for flag, value in optional:
            if value:
                argv.extend([flag, value])
        argv.extend(
            [
                f"--gres=gpu:{self.config.slurm_gpu_type}:{self.config.slurm_gpus}",
                f"--cpus-per-task={self.config.slurm_cpus}",
                f"--mem={self.config.slurm_memory}",
                f"--time={self.config.slurm_time}",
            ]
        )
        return [*argv, *worker]

    def remote_command(self) -> str:
        remote_dir = self.config.remote_dir
        if not remote_dir:
            raise VisionConfigurationError("Remote working directory is required")
        environment = [
            "PYTHONNOUSERSITE=1",
            "CHEMEAGLE_OFFLINE=1" if self.config.offline else "CHEMEAGLE_OFFLINE=0",
            "HF_HUB_OFFLINE=1" if self.config.offline else "HF_HUB_OFFLINE=0",
            "TRANSFORMERS_OFFLINE=1" if self.config.offline else "TRANSFORMERS_OFFLINE=0",
        ]
        command = ["env", *environment, *self.remote_argv()]
        return f"cd {shlex.quote(remote_dir)} && exec {shlex.join(command)}"

    def submit_command(self) -> str:
        """Return the command executed on the externally reachable SSH host.

        Some clusters prohibit long-running scheduler clients on login nodes and
        provide an internal submit host instead.  In that case the outer SSH
        connection starts a second, stdio-preserving SSH connection before srun.
        """
        command = self.remote_command()
        submit_host = self.config.slurm_submit_host
        if self.config.provider != "slurm-ssh" or not submit_host:
            return command
        nested_ssh = [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ServerAliveInterval=30",
            "-o",
            "ServerAliveCountMax=3",
            submit_host,
            command,
        ]
        return f"exec {shlex.join(nested_ssh)}"

    def ssh_command(self) -> List[str]:
        argv = [self.config.ssh_binary]
        if self.config.ssh_config:
            argv.extend(["-F", os.path.expanduser(self.config.ssh_config)])
        argv.extend(
            [
                "-o",
                "BatchMode=yes",
                "-o",
                "ServerAliveInterval=30",
                "-o",
                "ServerAliveCountMax=3",
                self.config.ssh_host or "",
                self.submit_command(),
            ]
        )
        return argv

    def _spawn(self) -> None:
        try:
            self._process = self._process_factory(
                self.ssh_command(),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                bufsize=1,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise VisionProcessError(f"Could not start remote vision worker: {exc}") from exc
        if not self.process.stdin or not self.process.stdout:
            raise VisionProcessError("Remote vision worker stdio is unavailable")
        self._reader_thread = threading.Thread(
            target=self._reader_loop, name="chemeagle-vision-reader", daemon=True
        )
        self._reader_thread.start()
        if self.process.stderr is not None:
            self._stderr_thread = threading.Thread(
                target=self._stderr_loop, name="chemeagle-vision-stderr", daemon=True
            )
            self._stderr_thread.start()

    def start(self) -> None:
        with self._start_lock:
            if self._started:
                return
            if self._closed:
                raise VisionProcessError("Remote vision backend is closed")
            self._spawn()
            self._started = True
            try:
                status = self._request(
                    "health", {}, timeout=self.config.startup_timeout
                )
                if not isinstance(status, dict):
                    raise VisionProtocolError(
                        "Remote vision worker returned invalid health data"
                    )
                if self.config.device.startswith("cuda") and not status.get(
                    "cuda_available"
                ):
                    raise VisionConfigurationError(
                        "Remote worker started but CUDA is unavailable; check Slurm GPU "
                        "allocation and the remote PyTorch installation"
                    )
            except Exception:
                self.close()
                raise

    def _reader_loop(self) -> None:
        try:
            for line in self.process.stdout:
                if not line.strip():
                    continue
                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    self._fail_pending(
                        VisionProtocolError("Remote vision worker emitted invalid JSONL")
                    )
                    continue
                request_id = message.get("id")
                if not isinstance(request_id, int):
                    continue
                with self._pending_lock:
                    waiter = self._pending.get(request_id)
                if waiter is not None:
                    waiter.put(message)
        except Exception as exc:
            self._fail_pending(VisionProcessError(f"Vision reader failed: {exc}"))
        finally:
            if not self._closed:
                details = self._stderr_lines[-1] if self._stderr_lines else "no stderr"
                self._fail_pending(
                    VisionProcessError(
                        f"Remote vision worker exited unexpectedly: {details}"
                    )
                )

    def _stderr_loop(self) -> None:
        try:
            for line in self.process.stderr:
                value = line.rstrip()
                if value:
                    self._stderr_lines.append(value)
                    del self._stderr_lines[:-100]
        except Exception:
            return

    def _fail_pending(self, exc: Exception) -> None:
        with self._pending_lock:
            waiters = list(self._pending.values())
        for waiter in waiters:
            try:
                waiter.put_nowait({"exception": exc})
            except queue.Full:
                continue

    def _request(self, method: str, params: Dict[str, Any], *, timeout: float) -> Any:
        with self._pending_lock:
            request_id = self._next_id
            self._next_id += 1
            waiter: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=1)
            self._pending[request_id] = waiter
        message = {
            "id": request_id,
            "method": method,
            "params": encode_value(params),
        }
        try:
            with self._write_lock:
                self.process.stdin.write(
                    json.dumps(message, ensure_ascii=False, allow_nan=False) + "\n"
                )
                self.process.stdin.flush()
            try:
                response = waiter.get(timeout=timeout)
            except queue.Empty as exc:
                raise VisionTimeoutError(
                    f"Remote vision method {method!r} timed out after {timeout:g}s"
                ) from exc
        finally:
            with self._pending_lock:
                self._pending.pop(request_id, None)
        if "exception" in response:
            raise response["exception"]
        error = response.get("error")
        if isinstance(error, dict):
            error_type = error.get("type", "RemoteError")
            message = error.get("message", "unknown remote vision failure")
            raise VisionBackendError(f"{error_type}: {message}")
        if "result" not in response:
            raise VisionProtocolError("Remote vision response has no result")
        return decode_value(response["result"])

    def call(self, method: str, params: Dict[str, Any]) -> Any:
        self.start()
        return self._request(method, params, timeout=self.config.timeout)

    def health(self) -> Dict[str, Any]:
        self.start()
        value = self._request("health", {}, timeout=self.config.timeout)
        if not isinstance(value, dict):
            raise VisionProtocolError("Remote vision health response is not a dictionary")
        return value

    def close(self) -> None:
        if self._closed:
            return
        if self._process is not None and self._started and self.process.poll() is None:
            try:
                self._request("shutdown", {}, timeout=5)
            except Exception:
                pass
        self._closed = True
        if self._process is not None and self.process.poll() is None:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except Exception:
                try:
                    self.process.kill()
                    self.process.wait(timeout=5)
                except Exception:
                    pass
