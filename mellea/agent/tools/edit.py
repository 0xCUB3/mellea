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

    count = content.count(old_str)
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

    full_path.write_text(new_content)
    old_lines = old_str.count("\n") + 1
    new_lines = new_str.count("\n") + 1
    return f"Successfully replaced {old_lines} lines with {new_lines} lines in {path}."
