"""File finding and directory listing tools."""

from __future__ import annotations

from pathlib import Path

_SKIP_DIRS = {
    ".git",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    "build",
    "dist",
    ".tox",
    ".eggs",
}
_MAX_RESULTS = 50


def find_file(pattern: str, *, repo_root: str) -> str:
    """Find files matching a glob pattern."""
    root = Path(repo_root)
    matches: list[str] = []
    for p in root.rglob(pattern):
        if any(s in p.parts for s in _SKIP_DIRS):
            continue
        if p.is_file():
            matches.append(str(p.relative_to(root)))
            if len(matches) >= _MAX_RESULTS:
                break
    if not matches:
        return f"No files matching '{pattern}' found."
    result = "\n".join(sorted(matches))
    if len(matches) >= _MAX_RESULTS:
        result += f"\n... (showing first {_MAX_RESULTS} matches)"
    return result


def list_dir(path: str = ".", *, repo_root: str) -> str:
    """List directory contents."""
    full_path = Path(repo_root) / path
    if not full_path.is_dir():
        return f"Error: not a directory: {path}"
    entries: list[str] = []
    for p in sorted(full_path.iterdir()):
        if p.name.startswith(".") and p.name in _SKIP_DIRS:
            continue
        kind = "dir" if p.is_dir() else "file"
        entries.append(f"  {p.name:<40} [{kind}]")
    return f"{path}/\n" + "\n".join(entries) if entries else f"{path}/ (empty)"
