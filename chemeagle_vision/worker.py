"""Offline JSONL worker launched through SSH, optionally underneath Slurm."""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import traceback
from typing import Any, Dict, Iterable, TextIO

from .codec import decode_value, encode_value
from .config import VisionConfig
from .runtime import VisionRuntime


def serve_stdio(
    runtime: VisionRuntime,
    *,
    stdin: TextIO = sys.stdin,
    stdout: TextIO = sys.stdout,
    stderr: TextIO = sys.stderr,
) -> int:
    for line in stdin:
        if not line.strip():
            continue
        request_id: Any = None
        try:
            request = json.loads(line)
            request_id = request.get("id")
            method = request.get("method")
            if not isinstance(request_id, int) or not isinstance(method, str):
                raise ValueError("Worker requests require integer id and string method")
            params = decode_value(request.get("params", {}))
            if method == "shutdown":
                response = {"id": request_id, "result": {"status": "closing"}}
                stdout.write(json.dumps(response, allow_nan=False) + "\n")
                stdout.flush()
                return 0
            with contextlib.redirect_stdout(stderr):
                result = runtime.call(method, params)
            response = {"id": request_id, "result": encode_value(result)}
        except Exception as exc:
            traceback.print_exc(file=stderr)
            response = {
                "id": request_id,
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
        stdout.write(json.dumps(response, ensure_ascii=False, allow_nan=False) + "\n")
        stdout.flush()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ChemEAGLE offline vision worker")
    parser.add_argument("--stdio", action="store_true", help="serve JSONL over stdio")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-dir")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--health", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = VisionConfig(
        provider="local",
        device=args.device,
        model_dir=args.model_dir,
        offline=args.offline,
    )
    runtime = VisionRuntime(config)
    if args.health:
        print(json.dumps(runtime.call("health", {}), ensure_ascii=False))
        return 0
    if not args.stdio:
        raise SystemExit("Use --stdio or --health")
    return serve_stdio(runtime)


if __name__ == "__main__":
    raise SystemExit(main())
