from __future__ import annotations

from pathlib import Path

from mellea.agent.tools import make_agent_tools


def test_make_agent_tools_returns_mellea_tools(tmp_path: Path) -> None:
    tools = make_agent_tools(str(tmp_path))
    names = {t.name for t in tools}
    assert "bash" in names
    assert "search_code" in names
    assert "edit" in names
    assert "read_file" in names
    assert "find_file" in names
    assert "list_dir" in names
    assert len(tools) == 6


def test_tools_are_callable(tmp_path: Path) -> None:
    tools = make_agent_tools(str(tmp_path))
    for t in tools:
        assert callable(t.run)
