"""Command-line authentication and diagnostics for the Codex backend."""

from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from typing import Any, Dict, Optional, Sequence

from .config import BackendConfig
from .codex_app_server import CodexAppServerBackend
from .errors import BackendError


def _safe_account(payload: Dict[str, Any]) -> Dict[str, Any]:
    account = payload.get("account")
    safe_account = None
    if isinstance(account, dict):
        safe_account = {
            key: account.get(key)
            for key in ("type", "email", "planType")
            if account.get(key) is not None
        }
    safe: Dict[str, Any] = {
        "requiresOpenaiAuth": payload.get("requiresOpenaiAuth"),
        "account": safe_account,
    }
    if "rateLimits" in payload:
        safe["rateLimits"] = payload["rateLimits"]
    if "rateLimitsError" in payload:
        safe["rateLimitsError"] = payload["rateLimitsError"]
    return safe


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage ChemEAGLE Codex authentication")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status", help="Show the active Codex account and subscription limits")
    login = sub.add_parser("login", help="Sign in with ChatGPT/Codex subscription")
    login.add_argument("--device-code", action="store_true", help="Use headless device-code login")
    login.add_argument("--no-open", action="store_true", help="Do not open the browser automatically")
    sub.add_parser("logout", help="Sign out the active Codex account")
    sub.add_parser("models", help="List models visible to the active Codex account")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    backend = CodexAppServerBackend(BackendConfig.from_env("codex"))
    try:
        if args.command == "status":
            print(json.dumps(_safe_account(backend.account_status()), ensure_ascii=False, indent=2))
        elif args.command == "logout":
            backend.logout()
            print("Codex account signed out.")
        elif args.command == "models":
            rows = [
                {
                    "id": row.get("id"),
                    "model": row.get("model"),
                    "displayName": row.get("displayName"),
                    "isDefault": row.get("isDefault"),
                    "inputModalities": row.get("inputModalities"),
                }
                for row in backend.models()
                if not row.get("hidden")
            ]
            print(json.dumps(rows, ensure_ascii=False, indent=2))
        else:
            started = backend.login(device_code=args.device_code)
            cursor = int(started.pop("eventCursor", 0))
            login_id = str(started.get("loginId"))
            if args.device_code:
                print(f"Open: {started.get('verificationUrl')}")
                print(f"Enter code: {started.get('userCode')}")
            else:
                auth_url = str(started.get("authUrl"))
                print(f"Open this URL to sign in:\n{auth_url}")
                if not args.no_open:
                    webbrowser.open(auth_url)
            completed = backend.wait_login(login_id, after=cursor)
            if not completed.get("success"):
                raise BackendError(str(completed.get("error") or "Codex login failed"))
            print("Codex ChatGPT login completed.")
        return 0
    except BackendError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        backend.close()


if __name__ == "__main__":
    raise SystemExit(main())
