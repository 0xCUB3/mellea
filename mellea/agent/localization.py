"""Lightweight repository localization helpers for mcode."""

from __future__ import annotations

import re
from pathlib import Path

_CODE_SUFFIXES = {
    ".c",
    ".cc",
    ".cpp",
    ".cs",
    ".go",
    ".h",
    ".hpp",
    ".java",
    ".js",
    ".jsx",
    ".kt",
    ".m",
    ".mm",
    ".php",
    ".py",
    ".rb",
    ".rs",
    ".scala",
    ".sh",
    ".swift",
    ".ts",
    ".tsx",
}


def _query_tokens(query: str) -> list[str]:
    return [token for token in re.findall(r"[A-Za-z0-9_]+", query.lower()) if len(token) >= 3]


def _candidate_files(repo_root: Path) -> list[Path]:
    candidates: list[Path] = []
    for path in repo_root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in _CODE_SUFFIXES:
            continue
        if any(part.startswith(".") for part in path.relative_to(repo_root).parts):
            continue
        candidates.append(path)
    return candidates


def format_candidate_files(repo_root: str, query: str, top_n: int = 6) -> str:
    """Return a short ranked list of likely files to inspect first."""
    root = Path(repo_root)
    if not root.exists():
        return ""

    tokens = _query_tokens(query)
    if not tokens:
        return ""

    ranked: list[tuple[int, str]] = []
    for path in _candidate_files(root):
        rel = path.relative_to(root).as_posix()
        rel_lower = rel.lower()
        score = sum(rel_lower.count(token) for token in tokens)
        if score:
            ranked.append((score, rel))

    if not ranked:
        return ""

    ranked.sort(key=lambda item: (-item[0], item[1]))
    lines = ["Likely files to inspect first:"]
    for _, rel in ranked[: max(1, top_n)]:
        lines.append(f"- {rel}")
    return "\n".join(lines)
