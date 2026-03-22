"""String replacement editing with tree-sitter syntax validation."""

from __future__ import annotations

import os
from pathlib import Path

_EXTENSION_TO_LANGUAGE: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".jsx": "javascript",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".c": "c",
    ".cpp": "cpp",
    ".h": "c",
    ".hpp": "cpp",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".kt": "kotlin",
    ".scala": "scala",
    ".sh": "bash",
    ".bash": "bash",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json",
    ".toml": "toml",
}


def _count_errors(tree) -> int:
    """Count ERROR nodes in a tree-sitter parse tree."""
    count = 0
    stack = [tree.root_node]
    while stack:
        node = stack.pop()
        if node.type == "ERROR":
            count += 1
        stack.extend(node.children)
    return count


def _syntax_details(path: str, content: str) -> tuple[int, str | None] | None:
    """Return syntax error count and first error message for known languages."""
    ext = Path(path).suffix.lower()
    lang = _EXTENSION_TO_LANGUAGE.get(ext)
    if lang is None:
        return None
    try:
        from tree_sitter_language_pack import get_parser

        parser = get_parser(lang)
    except (ImportError, Exception):
        return None
    tree = parser.parse(content.encode("utf-8"))
    error_count = _count_errors(tree)
    if error_count > 0:
        stack = [tree.root_node]
        while stack:
            node = stack.pop()
            if node.type == "ERROR":
                row, col = node.start_point
                return error_count, f"Syntax error at line {row + 1}, column {col}"
            stack.extend(node.children)
    return 0, None


def _validate_syntax(path: str, content: str) -> str | None:
    """Return error message if content has syntax errors, else None."""
    syntax = _syntax_details(path, content)
    if syntax is None:
        return None
    _, message = syntax
    return message


def _fuzzy_find(content: str, old_str: str) -> str | None:
    """Try whitespace-normalized matching. Returns the actual substring or None."""
    import re

    def normalize(s: str) -> str:
        return re.sub(r"[ \t]+", " ", s).strip()

    norm_old = normalize(old_str)
    old_lines = old_str.splitlines()
    content_lines = content.splitlines()
    n = len(old_lines)
    if n == 0:
        return None

    for i in range(len(content_lines) - n + 1):
        candidate = "\n".join(content_lines[i : i + n])
        if normalize(candidate) == norm_old:
            return candidate
    return None


def _format_edit_result(status: str, path: str, detail: str) -> str:
    """Return edit results in a consistent status-first format."""
    return f"EDIT {status}\n{path}\n{detail}"


def str_replace_edit(path: str, old_str: str, new_str: str, *, repo_root: str) -> str:
    """Replace a unique occurrence of old_str with new_str in a file.

    Validates syntax after edit using tree-sitter. Reverts on syntax errors.
    """
    try:
        full_path = (
            Path(repo_root) / path if not Path(path).is_absolute() else Path(path)
        )
        content = full_path.read_text(errors="replace")
    except FileNotFoundError:
        return _format_edit_result("REJECTED", path, "Error: file not found.")
    except OSError as e:
        return _format_edit_result("REJECTED", path, f"Error reading file: {e}")

    count = content.count(old_str)
    match_mode = "exact"
    if count == 0:
        if os.environ.get("MCODE_FUZZY_EDIT", "1") == "1":
            match = _fuzzy_find(content, old_str)
            if match is not None:
                old_str = match
                count = 1
                match_mode = "fallback"
        if count == 0:
            lines = content.splitlines()
            preview = "\n".join(lines[:30])
            return _format_edit_result(
                "REJECTED",
                path,
                f"Error: old_str not found. First 30 lines:\n{preview}",
            )
    if count > 1:
        return _format_edit_result(
            "REJECTED",
            path,
            f"Error: old_str appears {count} times. "
            "Provide more context to make the match unique.",
        )

    pre_syntax = _syntax_details(str(full_path), content)
    new_content = content.replace(old_str, new_str, 1)
    post_syntax = _syntax_details(str(full_path), new_content)

    pre_error_count = 0 if pre_syntax is None else pre_syntax[0]
    post_error_count = 0 if post_syntax is None else post_syntax[0]
    post_error = None if post_syntax is None else post_syntax[1]

    if post_error_count > pre_error_count and post_error is not None:
        return _format_edit_result(
            "REJECTED",
            path,
            f"Edit rejected: introduces syntax error. {post_error}. File unchanged.",
        )

    try:
        full_path.write_text(new_content)
    except OSError as e:
        return _format_edit_result("REJECTED", path, f"Error writing file: {e}")
    old_lines = old_str.count("\n") + 1
    new_lines = new_str.count("\n") + 1
    match_note = (
        "Used fallback match after exact replacement missed."
        if match_mode == "fallback"
        else "Used exact match."
    )
    msg = (
        f"Successfully replaced {old_lines} lines with {new_lines} lines. "
        f"{match_note}"
    )
    if os.environ.get("MCODE_ECHO_POSITION", "0") == "1":
        edited_lines = new_content.splitlines()
        idx = new_content.find(new_str)
        if idx >= 0:
            line_num = new_content[:idx].count("\n") + 1
            start = max(0, line_num - 3)
            end = min(len(edited_lines), line_num + new_lines + 2)
            snippet = "\n".join(
                f"{start + i + 1:>4}: {ln}"
                for i, ln in enumerate(edited_lines[start:end])
            )
            msg += f"\n\nContext around edit (lines {start + 1}-{end}):\n{snippet}"
    return _format_edit_result("APPLIED", path, msg)
