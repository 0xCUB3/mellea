"""Agent tools for code search, editing, reading, and navigation."""

from __future__ import annotations

from collections.abc import Callable

from mellea.agent.tools.bash import run_command
from mellea.agent.tools.edit import str_replace_edit
from mellea.agent.tools.navigate import find_file, list_dir
from mellea.agent.tools.read import read_file
from mellea.agent.tools.search import search_code
from mellea.agent.tools.testing import run_tests
from mellea.backends.tools import MelleaTool


def make_agent_tools(
    repo_root: str,
    *,
    test_cmds: list[str] | None = None,
    test_fn: Callable[[str], str] | None = None,
) -> list[MelleaTool]:
    """Create the standard set of agent tools bound to a repo root.

    If *test_fn* is provided it is used as the test runner instead of the
    default subprocess-based ``run_tests``.  The callable receives a single
    command string and must return the combined output.
    """
    import os

    read_counts: dict[str, int] = {}
    read_nudge = os.environ.get("MCODE_READ_NUDGE", "1") == "1"

    def _search(query: str) -> str:
        """Search for a regex pattern across all files in the repo using ripgrep."""
        return search_code(query, repo_root=repo_root)

    def _edit(path: str, old_str: str, new_str: str) -> str:
        """Replace a unique occurrence of old_str with new_str in a file."""
        return str_replace_edit(path, old_str, new_str, repo_root=repo_root)

    def _read(path: str, start_line: int = 1, end_line: int | None = None) -> str:
        """Read a file with line numbers, capped at 200 lines."""
        result = read_file(path, start_line, end_line, repo_root=repo_root)
        if read_nudge:
            count = read_counts.get(path, 0) + 1
            read_counts[path] = count
            if count >= 3:
                result = (
                    f"[Note: you have read {path} {count} times. "
                    "Consider searching other files or making your edit.]\n" + result
                )
        return result

    def _find(pattern: str) -> str:
        """Find files matching a glob pattern."""
        return find_file(pattern, repo_root=repo_root)

    def _list(path: str = ".") -> str:
        """List directory contents."""
        return list_dir(path, repo_root=repo_root)

    use_bash = os.environ.get("MELLEA_BASH_TOOL", "1") == "1"

    def _bash(command: str) -> str:
        """Run a bash command in the repo and return its output. Use for grep, find, git, python, pytest, or any shell command."""
        return run_command(command, repo_root=repo_root)

    tools: dict[str, object] = {
        "search_code": _search,
        "edit": _edit,
        "read_file": _read,
        "find_file": _find,
        "list_dir": _list,
    }

    if use_bash:
        tools["bash"] = _bash

    if test_fn is not None:

        def _run_tests_fn(test_cmd: str = "default") -> str:
            """Run tests to check if your fix works. Pass 'default' to run the task's test suite, or a custom pytest/test command."""
            return test_fn(test_cmd)

        tools["run_tests"] = _run_tests_fn

    elif test_cmds:

        def _run_tests(test_cmd: str = "default") -> str:
            """Run tests to check if your fix works. Pass 'default' to run the task's test suite, or a custom pytest/test command."""
            return run_tests(test_cmd, repo_root=repo_root, test_cmds=test_cmds)

        tools["run_tests"] = _run_tests

    return [MelleaTool.from_callable(fn, name=name) for name, fn in tools.items()]
