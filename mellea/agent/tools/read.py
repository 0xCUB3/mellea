"""File reading with line numbers and range selection."""

from __future__ import annotations

from pathlib import Path

_MAX_LINES = 200


def read_file(
    path: str, start_line: int = 1, end_line: int | None = None, *, repo_root: str
) -> str:
    """Read a file with line numbers, capped at 200 lines."""
    full_path = Path(repo_root) / path if not Path(path).is_absolute() else Path(path)
    try:
        content = full_path.read_text(errors="replace")
    except FileNotFoundError:
        return f"Error: file not found: {path}"

    lines = content.splitlines()
    total = len(lines)

    start = max(1, start_line) - 1
    end = min(total, end_line or total)
    selected = lines[start:end]

    if len(selected) > _MAX_LINES:
        selected = selected[:_MAX_LINES]
        end = start + _MAX_LINES
        truncated = True
    else:
        truncated = False

    numbered = [f"{start + i + 1:>4}: {line}" for i, line in enumerate(selected)]
    header = f"{path} ({total} lines total, showing {start + 1}-{end})"
    if truncated:
        header += f" [truncated to {_MAX_LINES} lines]"

    return header + "\n" + "\n".join(numbered)
