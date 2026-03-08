"""Agent tools for code search, editing, reading, and navigation."""

from __future__ import annotations

from mellea.agent.tools.edit import str_replace_edit
from mellea.agent.tools.navigate import find_file, list_dir
from mellea.agent.tools.read import read_file
from mellea.agent.tools.search import search_code
from mellea.backends.tools import MelleaTool


def make_agent_tools(repo_root: str) -> list[MelleaTool]:
    """Create the standard set of agent tools bound to a repo root."""

    def _search(query: str) -> str:
        """Search for a regex pattern across all files in the repo using ripgrep."""
        return search_code(query, repo_root=repo_root)

    def _edit(path: str, old_str: str, new_str: str) -> str:
        """Replace a unique occurrence of old_str with new_str in a file."""
        return str_replace_edit(path, old_str, new_str, repo_root=repo_root)

    def _read(path: str, start_line: int = 1, end_line: int | None = None) -> str:
        """Read a file with line numbers, capped at 200 lines."""
        return read_file(path, start_line, end_line, repo_root=repo_root)

    def _find(pattern: str) -> str:
        """Find files matching a glob pattern."""
        return find_file(pattern, repo_root=repo_root)

    def _list(path: str = ".") -> str:
        """List directory contents."""
        return list_dir(path, repo_root=repo_root)

    tools = {
        "search_code": _search,
        "edit": _edit,
        "read_file": _read,
        "find_file": _find,
        "list_dir": _list,
    }
    return [MelleaTool.from_callable(fn, name=name) for name, fn in tools.items()]
