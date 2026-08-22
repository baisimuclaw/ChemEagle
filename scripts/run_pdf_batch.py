#!/usr/bin/env python3
"""Run resumable end-to-end ChemEAGLE jobs starting from PDF files.

Each manifest item is one PDF.  The runner first invokes the upstream
``pdf_extraction.run_pdf`` implementation, records every extracted figure/table,
and then passes every extracted PNG through ``ChemEagle``.  PDF jobs are isolated
in child processes so ``--max-parallel`` also limits concurrent Codex sessions,
remote Slurm workers, and PDF extraction models.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.run_batch import (
    REPO_ROOT,
    append_log,
    git_revision,
    selected_environment,
    utc_now,
    write_json,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    items = manifest.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("Manifest must contain a non-empty 'items' list")

    seen = set()
    for item in items:
        if not isinstance(item, dict) or not item.get("id") or not item.get("pdf"):
            raise ValueError("Every PDF manifest item requires string 'id' and 'pdf'")
        item_id = str(item["id"])
        if item_id in seen:
            raise ValueError(f"Duplicate manifest item id: {item_id}")
        seen.add(item_id)

        pdf = Path(str(item["pdf"]))
        if not pdf.is_absolute():
            pdf = (path.parent / pdf).resolve()
        if not pdf.is_file():
            raise FileNotFoundError(f"Input PDF does not exist: {pdf}")
        if pdf.suffix.lower() != ".pdf":
            raise ValueError(f"Input is not a PDF file: {pdf}")
        item["id"] = item_id
        item["pdf"] = str(pdf)
    return manifest


def extraction_record_path(run_dir: Path, item_id: str) -> Path:
    return run_dir / "extraction" / f"{item_id}.json"


def extracted_image_dir(run_dir: Path, item_id: str) -> Path:
    return run_dir / "extracted" / item_id


def discover_images(image_dir: Path) -> List[Path]:
    return sorted(
        path.resolve()
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".png"
    )


def reusable_extraction(
    record_path: Path, pdf_path: Path, model_size: str
) -> Optional[Dict[str, Any]]:
    if not record_path.is_file():
        return None
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if record.get("status") != "succeeded" or record.get("model_size") != model_size:
        return None
    if record.get("pdf_sha256") != sha256_file(pdf_path):
        return None
    images = record.get("images")
    if not isinstance(images, list) or not images:
        return None
    if not all(Path(str(image["path"])).is_file() for image in images):
        return None
    return record


def extract_pdf(
    item: Dict[str, Any],
    run_dir: Path,
    model_size: str,
    extractor: Optional[Callable[..., Any]] = None,
) -> Dict[str, Any]:
    item_id = item["id"]
    pdf_path = Path(item["pdf"])
    record_path = extraction_record_path(run_dir, item_id)
    cached = reusable_extraction(record_path, pdf_path, model_size)
    if cached is not None:
        print(
            f"[pdf-batch:{item_id}] reusing {cached['image_count']} extracted images",
            flush=True,
        )
        return cached

    image_dir = extracted_image_dir(run_dir, item_id)
    image_dir.parent.mkdir(parents=True, exist_ok=True)
    if image_dir.exists() and list(image_dir.iterdir()):
        raise RuntimeError(
            f"Extraction directory is non-empty without a reusable manifest: {image_dir}. "
            "Use a new run directory to avoid mixing stale PDF crops."
        )
    if image_dir.exists():
        image_dir.rmdir()
    temporary_image_dir = image_dir.with_name(f".{item_id}.{os.getpid()}.tmp")
    temporary_image_dir.mkdir()

    if extractor is None:
        from pdf_extraction import run_pdf

        extractor = run_pdf

    started_at = utc_now()
    started = time.monotonic()
    print(f"[pdf-batch:{item_id}] extracting {pdf_path}", flush=True)
    extractor(
        pdf_dir=str(pdf_path),
        image_dir=str(temporary_image_dir),
        model_size=model_size,
    )
    images = discover_images(temporary_image_dir)
    if not images:
        raise RuntimeError(f"PDF extraction produced no PNG files: {pdf_path}")
    os.replace(temporary_image_dir, image_dir)
    images = discover_images(image_dir)

    record = {
        "id": item_id,
        "status": "succeeded",
        "pdf": str(pdf_path),
        "pdf_sha256": sha256_file(pdf_path),
        "model_size": model_size,
        "started_at": started_at,
        "finished_at": utc_now(),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "image_count": len(images),
        "images": [
            {
                "id": path.stem,
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in images
        ],
    }
    write_json(record_path, record)
    print(f"[pdf-batch:{item_id}] extracted {len(images)} images", flush=True)
    return record


def successful_result(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("status") == "succeeded"
    except (OSError, json.JSONDecodeError):
        return False


def run_image(
    *,
    item: Dict[str, Any],
    image: Dict[str, Any],
    run_dir: Path,
) -> Dict[str, Any]:
    item_id = item["id"]
    image_id = image["id"]
    result_path = run_dir / "results" / item_id / f"{image_id}.json"
    if successful_result(result_path):
        print(f"[pdf-batch:{item_id}] reusing successful image {image_id}", flush=True)
        return json.loads(result_path.read_text(encoding="utf-8"))

    log_path = run_dir / "logs" / item_id / f"{image_id}.log"
    remote_log = run_dir / "logs" / item_id / f"{image_id}.remote.log"
    os.environ["CHEMEAGLE_CACHE_DIR"] = str(run_dir / "cache" / item_id / image_id)
    os.environ["CHEMEAGLE_VISION_REMOTE_LOG"] = str(remote_log)

    started = time.monotonic()
    report: Dict[str, Any] = {
        "id": image_id,
        "paper_id": item_id,
        "image": image,
        "status": "running",
        "started_at": utc_now(),
        "finished_at": None,
        "elapsed_seconds": None,
        "vision_health": None,
        "llm_backend": None,
        "result": None,
    }
    llm = None
    vision = None
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", buffering=1) as image_log:
        with contextlib.redirect_stdout(image_log), contextlib.redirect_stderr(image_log):
            try:
                from chemeagle_llm import create_backend
                from chemeagle_vision import create_vision_backend
                from main import ChemEagle

                print(f"[pdf-batch:{item_id}:{image_id}] started", flush=True)
                llm = create_backend("codex")
                vision = create_vision_backend("slurm-ssh")
                report["vision_health"] = vision.health()
                print(
                    f"[pdf-batch:{item_id}:{image_id}] vision health: "
                    f"{report['vision_health']}",
                    flush=True,
                )
                report["result"] = ChemEagle(
                    image["path"],
                    backend=llm,
                    vision_backend=vision,
                )
                report["status"] = "succeeded"
            except Exception as exc:
                report.update(
                    {
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
                error_path = run_dir / "errors" / item_id / f"{image_id}.txt"
                error_path.parent.mkdir(parents=True, exist_ok=True)
                error_path.write_text(report["traceback"], encoding="utf-8")
                print(
                    f"[pdf-batch:{item_id}:{image_id}] failed: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
            finally:
                if llm is not None:
                    runtime_metadata = getattr(llm, "runtime_metadata", None)
                    if callable(runtime_metadata):
                        report["llm_backend"] = runtime_metadata()
                if vision is not None:
                    vision.close()
                if llm is not None:
                    llm.close()
                report["finished_at"] = utc_now()
                report["elapsed_seconds"] = round(time.monotonic() - started, 3)
                write_json(result_path, report)
    return report


def write_paper_report(
    item: Dict[str, Any],
    extraction: Dict[str, Any],
    image_reports: List[Dict[str, Any]],
    run_dir: Path,
) -> Dict[str, Any]:
    failures = [report["id"] for report in image_reports if report["status"] != "succeeded"]
    report = {
        "id": item["id"],
        "item": item,
        "status": "succeeded" if not failures else "completed_with_failures",
        "extraction": extraction,
        "image_count": len(image_reports),
        "succeeded_count": len(image_reports) - len(failures),
        "failed_count": len(failures),
        "failed_images": failures,
        "image_results": [
            {
                "id": image_report["id"],
                "status": image_report["status"],
                "vision_health": image_report.get("vision_health"),
                "result_path": str(
                    run_dir
                    / "results"
                    / item["id"]
                    / f"{image_report['id']}.json"
                ),
            }
            for image_report in image_reports
        ],
        "finished_at": utc_now(),
    }
    write_json(run_dir / "paper_results" / f"{item['id']}.json", report)
    return report


def run_child(
    manifest_path: Path,
    run_dir: Path,
    item_id: str,
    model_size: str,
    extract_only: bool,
) -> int:
    manifest = load_manifest(manifest_path)
    item = next((row for row in manifest["items"] if row["id"] == item_id), None)
    if item is None:
        raise ValueError(f"Manifest has no item {item_id!r}")

    os.chdir(REPO_ROOT)
    sys.path.insert(0, str(REPO_ROOT))
    os.environ["CHEMEAGLE_TRACE"] = "1"

    try:
        extraction = extract_pdf(item, run_dir, model_size)
        if extract_only:
            return 0

        image_reports = []
        total_images = len(extraction["images"])
        for index, image in enumerate(extraction["images"], start=1):
            print(
                f"[pdf-batch:{item_id}] image {index}/{total_images} "
                f"started: {image['id']}",
                flush=True,
            )
            image_report = run_image(item=item, image=image, run_dir=run_dir)
            image_reports.append(image_report)
            print(
                f"[pdf-batch:{item_id}] image {index}/{total_images} "
                f"finished: {image['id']} status={image_report['status']}",
                flush=True,
            )
        paper_report = write_paper_report(item, extraction, image_reports, run_dir)
        return 0 if paper_report["status"] == "succeeded" else 1
    except Exception:
        error_path = run_dir / "errors" / f"{item_id}.txt"
        error_path.parent.mkdir(parents=True, exist_ok=True)
        error_path.write_text(traceback.format_exc(), encoding="utf-8")
        traceback.print_exc()
        return 1


def phase_result_path(run_dir: Path, item_id: str, extract_only: bool) -> Path:
    if extract_only:
        return extraction_record_path(run_dir, item_id)
    return run_dir / "paper_results" / f"{item_id}.json"


def completed_for_phase(run_dir: Path, item_id: str, extract_only: bool) -> bool:
    path = phase_result_path(run_dir, item_id, extract_only)
    if not path.is_file():
        return False
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("status") == "succeeded"
    except (OSError, json.JSONDecodeError):
        return False


def initial_status(
    manifest: Dict[str, Any], run_dir: Path, max_parallel: int, extract_only: bool
) -> Dict[str, Any]:
    items = {}
    for item in manifest["items"]:
        state = "succeeded" if completed_for_phase(
            run_dir, item["id"], extract_only
        ) else "pending"
        items[item["id"]] = {
            "state": state,
            "pid": None,
            "started_at": None,
            "finished_at": None,
            "exit_code": None,
            "pdf": item["pdf"],
        }
    return {
        "state": "running",
        "phase": "extract" if extract_only else "full",
        "started_at": utc_now(),
        "updated_at": utc_now(),
        "finished_at": None,
        "max_parallel": max_parallel,
        "run_dir": str(run_dir),
        "items": items,
    }


def run_parent(
    manifest_path: Path,
    run_dir: Path,
    max_parallel: int,
    poll_seconds: float,
    model_size: str,
    extract_only: bool,
) -> int:
    if max_parallel < 1:
        raise ValueError("--max-parallel must be at least 1")
    manifest = load_manifest(manifest_path)
    run_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "logs",
        "results",
        "paper_results",
        "errors",
        "cache",
        "extracted",
        "extraction",
    ):
        (run_dir / name).mkdir(exist_ok=True)

    phase = "extract" if extract_only else "full"
    workflow_log = run_dir / f"{phase}_workflow.log"
    status_path = run_dir / f"{phase}_status.json"
    status = initial_status(manifest, run_dir, max_parallel, extract_only)
    write_json(
        run_dir / f"{phase}_config.json",
        {
            "created_at": utc_now(),
            "phase": phase,
            "git_revision": git_revision(),
            "python": sys.executable,
            "manifest": str(manifest_path),
            "max_parallel": max_parallel,
            "model_size": model_size,
            "environment": selected_environment(),
        },
    )
    write_json(status_path, status)
    append_log(workflow_log, f"PDF batch started; phase={phase}; max_parallel={max_parallel}")

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
            log_path = run_dir / "logs" / f"{item_id}.{phase}.log"
            log_handle = log_path.open("a", encoding="utf-8", buffering=1)
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--manifest",
                str(manifest_path),
                "--run-dir",
                str(run_dir),
                "--model-size",
                model_size,
                "--child",
                item_id,
            ]
            if extract_only:
                command.append("--extract-only")
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            process = subprocess.Popen(
                command,
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
    append_log(workflow_log, f"PDF batch finished; phase={phase}; failures={failed}")
    return 0 if not failed else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--max-parallel", type=int, default=1)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--model-size", choices=("base", "large"), default="large")
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="Run the real upstream PDF figure/table extraction without invoking Codex or GPU workers",
    )
    parser.add_argument("--child", help=argparse.SUPPRESS)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest_path = args.manifest.expanduser().resolve()
    run_dir = args.run_dir.expanduser().resolve()
    if args.child:
        return run_child(
            manifest_path,
            run_dir,
            args.child,
            args.model_size,
            args.extract_only,
        )
    return run_parent(
        manifest_path,
        run_dir,
        args.max_parallel,
        args.poll_seconds,
        args.model_size,
        args.extract_only,
    )


if __name__ == "__main__":
    raise SystemExit(main())
