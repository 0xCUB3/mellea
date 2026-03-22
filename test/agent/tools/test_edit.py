from __future__ import annotations

from pathlib import Path

from mellea.agent.tools import make_agent_tools
from mellea.agent.tools.edit import str_replace_edit


def test_edit_replaces_unique_string(tmp_path: Path) -> None:
    f = tmp_path / "foo.py"
    f.write_text("def hello():\n    return 1\n")
    result = str_replace_edit(
        str(f), old_str="return 1", new_str="return 2", repo_root=str(tmp_path)
    )
    assert "return 2" in f.read_text()
    assert result.startswith(f"$ edit {f}\nAPPLIED\n")
    assert "Successfully" in result


def test_edit_rejects_nonunique_match(tmp_path: Path) -> None:
    f = tmp_path / "foo.py"
    f.write_text("x = 1\nx = 1\n")
    result = str_replace_edit(
        str(f), old_str="x = 1", new_str="x = 2", repo_root=str(tmp_path)
    )
    assert "x = 1" in f.read_text()
    assert "multiple" in result.lower() or "unique" in result.lower()


def test_edit_rejects_missing_match(tmp_path: Path) -> None:
    f = tmp_path / "foo.py"
    f.write_text("x = 1\n")
    result = str_replace_edit(
        str(f), old_str="y = 2", new_str="y = 3", repo_root=str(tmp_path)
    )
    assert "x = 1" in f.read_text()
    assert result.startswith(f"$ edit {f}\nREJECTED\n")
    assert "not found" in result.lower()


def test_edit_rejects_syntax_error(tmp_path: Path) -> None:
    f = tmp_path / "foo.py"
    f.write_text("def hello():\n    return 1\n")
    result = str_replace_edit(
        str(f), old_str="return 1", new_str="return (", repo_root=str(tmp_path)
    )
    assert "return 1" in f.read_text()
    assert "syntax" in result.lower() or "error" in result.lower()


def test_edit_works_for_non_python(tmp_path: Path) -> None:
    f = tmp_path / "foo.js"
    f.write_text("function hello() {\n  return 1;\n}\n")
    result = str_replace_edit(
        str(f), old_str="return 1;", new_str="return 2;", repo_root=str(tmp_path)
    )
    assert "return 2;" in f.read_text()
    assert "Successfully" in result


def test_coding_tools_include_shell_by_default(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.delenv("MELLEA_BASH_TOOL", raising=False)

    tool_map = {tool.name: tool for tool in make_agent_tools(str(tmp_path))}

    assert "bash" in tool_map
    result = tool_map["bash"].run("echo hello")
    assert result.startswith("$ echo hello\nPASSED\n")
    assert "hello" in result


def test_edit_uses_forgiving_match_when_exact_string_misses(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.delenv("MCODE_FUZZY_EDIT", raising=False)
    f = tmp_path / "foo.py"
    f.write_text("def hello():\n    return 1\n")

    result = str_replace_edit(
        str(f),
        old_str="def hello():\n  return 1",
        new_str="def hello():\n  return 2",
        repo_root=str(tmp_path),
    )

    assert f.read_text() == "def hello():\n  return 2\n"
    assert "fallback" in result.lower()


def test_edit_rolls_back_when_forgiving_match_breaks_syntax(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.delenv("MCODE_FUZZY_EDIT", raising=False)
    f = tmp_path / "foo.py"
    original = "def hello():\n    return 1\n"
    f.write_text(original)

    result = str_replace_edit(
        str(f),
        old_str="def hello():\n  return 1",
        new_str="def hello():\n  return (",
        repo_root=str(tmp_path),
    )

    assert f.read_text() == original
    assert "syntax" in result.lower()
    assert "unchanged" in result.lower()
