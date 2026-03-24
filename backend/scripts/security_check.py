"""Basic repository security checks for CI."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _run_git_ls_files() -> list[str]:
    proc = subprocess.run(
        ["git", "ls-files"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def _check_env_files(tracked: list[str]) -> list[str]:
    failures: list[str] = []
    forbidden = {".env", "backend/.env", "frontend/.env"}
    for path in tracked:
        if path in forbidden:
            p = ROOT / path
            text = p.read_text(encoding="utf-8", errors="ignore") if p.exists() else ""
            risky = [
                line
                for line in text.splitlines()
                if "=" in line and not line.strip().startswith("#") and line.split("=", 1)[1].strip() not in {"", "change-me-in-production"}
            ]
            if risky:
                failures.append(f"{path} appears to contain non-empty secrets.")
    return failures


def _check_default_secret_reference() -> list[str]:
    failures: list[str] = []
    candidates = [
        ROOT / "backend" / "app_config.py",
        ROOT / "backend" / "config.py",  # backward compatibility
    ]
    cfg = next((p for p in candidates if p.exists()), None)
    if cfg is None:
        failures.append("Missing backend app config file; cannot verify SECRET_KEY guard.")
        return failures

    text = cfg.read_text(encoding="utf-8", errors="ignore")
    if "APP_ENV" not in text or "Invalid SECRET_KEY for production" not in text:
        failures.append(f"{cfg.relative_to(ROOT)} missing production SECRET_KEY guard.")
    return failures


def main() -> int:
    failures: list[str] = []
    try:
        tracked = _run_git_ls_files()
    except Exception as exc:
        print(f"[security-check] Failed to inspect git files: {exc}")
        return 2

    failures.extend(_check_env_files(tracked))
    failures.extend(_check_default_secret_reference())

    if failures:
        print("[security-check] FAILED")
        for item in failures:
            print(f"- {item}")
        return 1

    print("[security-check] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
