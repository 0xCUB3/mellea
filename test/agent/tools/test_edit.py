from __future__ import annotations

from pathlib import Path

from mellea.agent.tools.edit import str_replace_edit


def test_edit_replaces_unique_string(tmp_path: Path) -> None:
    f = tmp_path / "foo.py"
    f.write_text("def hello():\n    return 1\n")
    result = str_replace_edit(
        str(f), old_str="return 1", new_str="return 2", repo_root=str(tmp_path)
    )
    assert "return 2" in f.read_text()
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
