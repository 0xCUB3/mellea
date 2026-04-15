"""Code search using ripgrep with grep fallback."""

from __future__ import annotations

import subprocess

_SKIP_DIRS = [
    ".git",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    "build",
    "dist",
    ".tox",
    ".eggs",
    ".mypy_cache",
]

_MAX_MATCHES = 30
_TIMEOUT_S = 5


def search_code(query: str, *, repo_root: str) -> str:
    """Search for a regex pattern across all files in a repo using ripgrep."""
    globs = [f"--glob=!{d}" for d in _SKIP_DIRS]
    cmd = [
        "rg",
        "--no-heading",
        "--line-number",
        "--max-columns=200",
        "--max-count=5",
        "--color=never",
        *globs,
        query,
        repo_root,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=_TIMEOUT_S)
        output = result.stdout.strip()
    except FileNotFoundError:
        return _fallback_search(query, repo_root)
    except subprocess.TimeoutExpired:
        return f"Search timed out after {_TIMEOUT_S}s. Try a more specific query."

    if not output:
        return "No matches found."

    lines = output.splitlines()
    if len(lines) > _MAX_MATCHES:
        lines = lines[:_MAX_MATCHES]
        return "\n".join(lines) + f"\n... ({_MAX_MATCHES} of many matches shown)"
    return "\n".join(lines)


def _fallback_search(query: str, repo_root: str) -> str:
    """Fallback to grep if ripgrep is not installed."""
    import re
    from pathlib import Path

    skip = set(_SKIP_DIRS)
    pattern = re.compile(query, re.IGNORECASE)
    matches: list[str] = []
    for p in Path(repo_root).rglob("*"):
        if any(s in p.parts for s in skip):
            continue
        if not p.is_file() or p.stat().st_size > 500_000:
            continue
        try:
            text = p.read_text(errors="replace")
            for i, line in enumerate(text.splitlines(), 1):
                if pattern.search(line):
                    rel = p.relative_to(repo_root)
                    matches.append(f"{rel}:{i}: {line[:200]}")
                    if len(matches) >= _MAX_MATCHES:
                        return "\n".join(matches)
        except (OSError, UnicodeDecodeError):
            continue
    return "\n".join(matches) if matches else "No matches found."
