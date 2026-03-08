from __future__ import annotations

from pathlib import Path

from mellea.agent.tools.navigate import find_file, list_dir


def test_find_file_by_name(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("")
    (tmp_path / "src" / "util.py").write_text("")
    result = find_file("main.py", repo_root=str(tmp_path))
    assert "main.py" in result
    assert "util.py" not in result


def test_find_file_glob(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("")
    (tmp_path / "b.py").write_text("")
    (tmp_path / "c.js").write_text("")
    result = find_file("*.py", repo_root=str(tmp_path))
    assert "a.py" in result
    assert "b.py" in result
    assert "c.js" not in result


def test_list_dir(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "foo.py").write_text("hello")
    result = list_dir(str(tmp_path), repo_root=str(tmp_path))
    assert "src" in result
    assert "foo.py" in result
