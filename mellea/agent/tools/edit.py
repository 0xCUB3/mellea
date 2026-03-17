"""String replacement editing with tree-sitter syntax validation."""

from __future__ import annotations

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


def _validate_syntax(path: str, content: str) -> str | None:
    """Return error message if content has new syntax errors, else None."""
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
    errors = _count_errors(tree)
    if errors > 0:
        stack = [tree.root_node]
        while stack:
            node = stack.pop()
            if node.type == "ERROR":
                row, col = node.start_point
                return f"Syntax error at line {row + 1}, column {col}"
            stack.extend(node.children)
    return None


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
        return f"Error: file not found: {path}"
    except OSError as e:
        return f"Error reading {path}: {e}"

    count = content.count(old_str)
    if count == 0:
        import os

        if os.environ.get("MCODE_FUZZY_EDIT", "0") == "1":
            match = _fuzzy_find(content, old_str)
            if match is not None:
                old_str = match
                count = 1
        if count == 0:
            lines = content.splitlines()
            preview = "\n".join(lines[:30])
            return f"Error: old_str not found in {path}. First 30 lines:\n{preview}"
    if count > 1:
        return (
            f"Error: old_str appears {count} times in {path}. "
            "Provide more context to make the match unique."
        )

    pre_errors = _validate_syntax(str(full_path), content)
    new_content = content.replace(old_str, new_str, 1)
    post_error = _validate_syntax(str(full_path), new_content)

    if post_error and not pre_errors:
        return f"Edit rejected: introduces syntax error. {post_error}. File unchanged."

    try:
        full_path.write_text(new_content)
    except OSError as e:
        return f"Error writing {path}: {e}"
    old_lines = old_str.count("\n") + 1
    new_lines = new_str.count("\n") + 1

    import os

    msg = f"Successfully replaced {old_lines} lines with {new_lines} lines in {path}."
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
            msg += (
                f"\n\nContext around edit ({path}, lines {start + 1}-{end}):\n{snippet}"
            )
    return msg
