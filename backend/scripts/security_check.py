from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

SKIP_PREFIXES = (
    ".git/",
    "backend/.venv/",
    "frontend/node_modules/",
    "backend/exports/",
)
SKIP_SUFFIXES = (
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".pdf",
    ".dxf",
    ".zip",
)

PATTERNS = [
    ("openrouter_live_key", re.compile(r"sk-or-v1-[A-Za-z0-9]{20,}")),
    ("openai_live_key", re.compile(r"\bsk-[A-Za-z0-9]{24,}\b")),
    (
        "github_token_assignment",
        re.compile(r"(?i)(github_token|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})"),
    ),
    (
        "generic_api_key_assignment",
        re.compile(r"(?i)\bapi[_-]?key\b\s*[:=]\s*[\"']?[A-Za-z0-9_\-/]{24,}"),
    ),
]


def _tracked_files() -> list[Path]:
    try:
        result = subprocess.run(
            ["git", "ls-files"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        rel_paths = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        return [ROOT / rel for rel in rel_paths]
    except Exception:
        return [p for p in ROOT.rglob("*") if p.is_file()]


def _should_skip(path: Path) -> bool:
    rel = path.relative_to(ROOT).as_posix()
    if any(rel.startswith(prefix) for prefix in SKIP_PREFIXES):
        return True
    if rel.endswith(SKIP_SUFFIXES):
        return True
    return False


def _line_allowlist(line: str) -> bool:
    stripped = line.strip()
    if stripped.startswith("#"):
        return True
    if stripped.startswith("OPENROUTER_API_KEY=") and stripped.endswith("="):
        return True
    if stripped.startswith("OPENROUTER_API_KEY_SECONDARY=") and stripped.endswith("="):
        return True
    if "example" in stripped.lower() and "key" in stripped.lower():
        return True
    return False


def main() -> int:
    findings: list[str] = []
    scanned = 0

    for file_path in _tracked_files():
        if _should_skip(file_path):
            continue

        rel = file_path.relative_to(ROOT).as_posix()
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        scanned += 1
        for i, line in enumerate(content.splitlines(), start=1):
            if _line_allowlist(line):
                continue
            for label, pattern in PATTERNS:
                if pattern.search(line):
                    findings.append(f"{rel}:{i}: possible {label}")
                    break

    if findings:
        print("security-check: FAILED")
        print(f"scanned_files={scanned} findings={len(findings)}")
        for finding in findings[:50]:
            print(f"- {finding}")
        return 1

    print("security-check: PASSED")
    print(f"scanned_files={scanned} findings=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
