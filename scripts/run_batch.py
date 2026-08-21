#!/usr/bin/env python3
"""Run independent ChemEAGLE image jobs with durable local checkpoints."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parents[1]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_default(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return repr(value)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, default=json_default)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{utc_now()} {message}\n")
        handle.flush()


def load_manifest(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    items = manifest.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("Manifest must contain a non-empty 'items' list")
    seen = set()
    for item in items:
        if not isinstance(item, dict) or not item.get("id") or not item.get("image"):
            raise ValueError("Every manifest item requires string 'id' and 'image'")
        item_id = str(item["id"])
        if item_id in seen:
            raise ValueError(f"Duplicate manifest item id: {item_id}")
        seen.add(item_id)
        image = Path(str(item["image"]))
        if not image.is_absolute():
            image = (path.parent / image).resolve()
        if not image.is_file():
            raise FileNotFoundError(f"Input image does not exist: {image}")
        item["id"] = item_id
        item["image"] = str(image)
    return manifest


def selected_environment() -> Dict[str, str]:
    safe = {}
    for key, value in os.environ.items():
        if not key.startswith("CHEMEAGLE_") and key not in {"CODEX_MODEL"}:
            continue
        upper = key.upper()
        if any(word in upper for word in ("KEY", "TOKEN", "SECRET", "AUTH")):
            continue
        safe[key] = value
    return dict(sorted(safe.items()))


def git_revision() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def run_child(manifest_path: Path, run_dir: Path, item_id: str) -> int:
    manifest = load_manifest(manifest_path)
    item = next((row for row in manifest["items"] if row["id"] == item_id), None)
    if item is None:
        raise ValueError(f"Manifest has no item {item_id!r}")

    os.chdir(REPO_ROOT)
    sys.path.insert(0, str(REPO_ROOT))
    os.environ["CHEMEAGLE_TRACE"] = "1"
    os.environ["CHEMEAGLE_CACHE_DIR"] = str(run_dir / "cache" / item_id)
    os.environ["CHEMEAGLE_VISION_REMOTE_LOG"] = str(
        run_dir / "logs" / f"{item_id}.remote.log"
    )

    from chemeagle_llm import create_backend
    from chemeagle_vision import create_vision_backend
    from main import ChemEagle

    started_at = utc_now()
    started = time.monotonic()
    report: Dict[str, Any] = {
        "id": item_id,
        "item": item,
        "status": "running",
        "started_at": started_at,
        "finished_at": None,
        "elapsed_seconds": None,
        "vision_health": None,
        "result": None,
    }
    llm = None
    vision = None
    try:
        llm = create_backend("codex")
        vision = create_vision_backend("slurm-ssh")
        report["vision_health"] = vision.health()
        print(f"[batch:{item_id}] vision health: {report['vision_health']}", flush=True)
        report["result"] = ChemEagle(
            str(item["image"]),
            backend=llm,
            vision_backend=vision,
        )
        report["status"] = "succeeded"
        return_code = 0
    except Exception as exc:
        report.update(
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        error_path = run_dir / "errors" / f"{item_id}.txt"
        error_path.parent.mkdir(parents=True, exist_ok=True)
        error_path.write_text(report["traceback"], encoding="utf-8")
        print(f"[batch:{item_id}] failed: {type(exc).__name__}: {exc}", flush=True)
        return_code = 1
    finally:
        if vision is not None:
            vision.close()
        if llm is not None:
            llm.close()
        report["finished_at"] = utc_now()
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        write_json(run_dir / "results" / f"{item_id}.json", report)
    return return_code


def initial_status(
    manifest: Dict[str, Any], run_dir: Path, max_parallel: int
) -> Dict[str, Any]:
    items = {}
    for item in manifest["items"]:
        result_path = run_dir / "results" / f"{item['id']}.json"
        state = "pending"
        if result_path.is_file():
            try:
                previous = json.loads(result_path.read_text(encoding="utf-8"))
                if previous.get("status") == "succeeded":
                    state = "succeeded"
            except (OSError, json.JSONDecodeError):
                pass
        items[item["id"]] = {
            "state": state,
            "pid": None,
            "started_at": None,
            "finished_at": None,
            "exit_code": None,
            "image": item["image"],
        }
    return {
        "state": "running",
        "started_at": utc_now(),
        "updated_at": utc_now(),
        "finished_at": None,
        "max_parallel": max_parallel,
        "run_dir": str(run_dir),
        "items": items,
    }


def run_parent(
    manifest_path: Path, run_dir: Path, max_parallel: int, poll_seconds: float
) -> int:
    if max_parallel < 1:
        raise ValueError("--max-parallel must be at least 1")
    manifest = load_manifest(manifest_path)
    run_dir.mkdir(parents=True, exist_ok=True)
    for name in ("logs", "results", "errors", "cache"):
        (run_dir / name).mkdir(exist_ok=True)

    workflow_log = run_dir / "workflow.log"
    status_path = run_dir / "status.json"
    status = initial_status(manifest, run_dir, max_parallel)
    write_json(
        run_dir / "config.json",
        {
            "created_at": utc_now(),
            "git_revision": git_revision(),
            "python": sys.executable,
            "manifest": str(manifest_path),
            "max_parallel": max_parallel,
            "environment": selected_environment(),
        },
    )
    write_json(status_path, status)
    append_log(workflow_log, f"batch started; max_parallel={max_parallel}")

    pending: List[Dict[str, Any]] = [
        item
        for item in manifest["items"]
        if status["items"][item["id"]]["state"] != "succeeded"
    ]
    running: Dict[int, Dict[str, Any]] = {}
    stop_requested = False

    def request_stop(signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True
        append_log(workflow_log, f"received signal {signum}; stopping children")

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    while pending or running:
        while pending and len(running) < max_parallel and not stop_requested:
            item = pending.pop(0)
            item_id = item["id"]
            combined_log = run_dir / "logs" / f"{item_id}.log"
            log_handle = combined_log.open("a", encoding="utf-8", buffering=1)
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--manifest",
                    str(manifest_path),
                    "--run-dir",
                    str(run_dir),
                    "--child",
                    item_id,
                ],
                cwd=REPO_ROOT,
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            running[process.pid] = {
                "process": process,
                "item": item,
                "log_handle": log_handle,
            }
            status["items"][item_id].update(
                {"state": "running", "pid": process.pid, "started_at": utc_now()}
            )
            status["updated_at"] = utc_now()
            write_json(status_path, status)
            append_log(workflow_log, f"started {item_id}; pid={process.pid}")

        for pid, active in list(running.items()):
            process = active["process"]
            return_code = process.poll()
            if return_code is None:
                continue
            active["log_handle"].close()
            item_id = active["item"]["id"]
            state = "succeeded" if return_code == 0 else "failed"
            status["items"][item_id].update(
                {
                    "state": state,
                    "pid": None,
                    "finished_at": utc_now(),
                    "exit_code": return_code,
                }
            )
            status["updated_at"] = utc_now()
            write_json(status_path, status)
            append_log(workflow_log, f"finished {item_id}; state={state}")
            del running[pid]

        if stop_requested:
            for active in running.values():
                active["process"].terminate()
            for active in running.values():
                try:
                    active["process"].wait(timeout=45)
                except subprocess.TimeoutExpired:
                    active["process"].kill()
                active["log_handle"].close()
            for item_state in status["items"].values():
                if item_state["state"] == "running":
                    item_state.update(
                        {"state": "interrupted", "pid": None, "finished_at": utc_now()}
                    )
            status.update(
                {"state": "interrupted", "updated_at": utc_now(), "finished_at": utc_now()}
            )
            write_json(status_path, status)
            return 130
        if pending or running:
            time.sleep(poll_seconds)

    failed = [key for key, value in status["items"].items() if value["state"] != "succeeded"]
    status.update(
        {
            "state": "completed" if not failed else "completed_with_failures",
            "updated_at": utc_now(),
            "finished_at": utc_now(),
        }
    )
    write_json(status_path, status)
    append_log(workflow_log, f"batch finished; failures={failed}")
    return 0 if not failed else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--max-parallel", type=int, default=1)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--child", help=argparse.SUPPRESS)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest_path = args.manifest.expanduser().resolve()
    run_dir = args.run_dir.expanduser().resolve()
    if args.child:
        return run_child(manifest_path, run_dir, args.child)
    return run_parent(manifest_path, run_dir, args.max_parallel, args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
