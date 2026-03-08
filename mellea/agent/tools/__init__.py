"""Agent tools for code search, editing, reading, and navigation."""

from __future__ import annotations

from functools import partial, wraps

from mellea.agent.tools.edit import str_replace_edit
from mellea.agent.tools.navigate import find_file, list_dir
from mellea.agent.tools.read import read_file
from mellea.agent.tools.search import search_code
from mellea.backends.tools import MelleaTool


def _bind(func, **kwargs):
    """Create a partial that preserves __name__ and __doc__ for from_callable."""
    bound = partial(func, **kwargs)
    bound = wraps(func)(bound)
    return bound


def make_agent_tools(repo_root: str) -> list[MelleaTool]:
    """Create the standard set of agent tools bound to a repo root."""
    bound = {
        "search_code": _bind(search_code, repo_root=repo_root),
        "edit": _bind(str_replace_edit, repo_root=repo_root),
        "read_file": _bind(read_file, repo_root=repo_root),
        "find_file": _bind(find_file, repo_root=repo_root),
        "list_dir": _bind(list_dir, repo_root=repo_root),
    }
    return [MelleaTool.from_callable(fn, name=name) for name, fn in bound.items()]
