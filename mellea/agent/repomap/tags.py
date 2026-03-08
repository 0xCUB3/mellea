"""Tree-sitter based tag extraction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Tag:
    """A symbol tag extracted from source code."""

    rel_path: str
    name: str
    line: int
    kind: str  # "def" or "ref"


_EXT_TO_LANG: dict[str, str] = {
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
    ".sh": "bash",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json",
    ".toml": "toml",
}

# tree-sitter node types that represent definitions, per language.
_DEF_NODE_TYPES: dict[str, set[str]] = {
    "python": {"function_definition", "class_definition"},
    "javascript": {"function_declaration", "class_declaration", "method_definition"},
    "typescript": {"function_declaration", "class_declaration", "method_definition"},
    "tsx": {"function_declaration", "class_declaration", "method_definition"},
    "java": {"method_declaration", "class_declaration", "interface_declaration"},
    "go": {"function_declaration", "method_declaration", "type_declaration"},
    "rust": {"function_item", "struct_item", "impl_item", "enum_item", "trait_item"},
    "c": {"function_definition", "struct_specifier"},
    "cpp": {"function_definition", "class_specifier", "struct_specifier"},
    "ruby": {"method", "class", "module"},
}

_NAME_FIELD = "name"


def extract_tags(path: str, *, repo_root: str = "") -> list[Tag]:
    """Extract definition tags from a source file using tree-sitter."""
    ext = Path(path).suffix.lower()
    lang = _EXT_TO_LANG.get(ext)
    if lang is None:
        return []

    try:
        from tree_sitter_language_pack import get_parser

        parser = get_parser(lang)
    except (ImportError, Exception):
        return []

    try:
        content = Path(path).read_text(errors="replace")
    except OSError:
        return []

    tree = parser.parse(content.encode("utf-8"))
    def_types = _DEF_NODE_TYPES.get(lang, set())
    rel = str(Path(path).relative_to(repo_root)) if repo_root else path

    tags: list[Tag] = []
    stack = [tree.root_node]
    while stack:
        node = stack.pop()
        if node.type in def_types:
            name_node = node.child_by_field_name(_NAME_FIELD)
            if name_node:
                tags.append(
                    Tag(
                        rel_path=rel,
                        name=name_node.text.decode("utf-8", errors="replace"),
                        line=node.start_point[0] + 1,
                        kind="def",
                    )
                )
        stack.extend(reversed(node.children))
    return tags
