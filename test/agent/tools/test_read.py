from __future__ import annotations

from pathlib import Path

from mellea.agent.tools.read import read_file


def test_read_file_with_line_numbers(tmp_path: Path) -> None:
    f = tmp_path / "foo.py"
    f.write_text("line1\nline2\nline3\n")
    result = read_file(str(f), repo_root=str(tmp_path))
    assert "1:" in result or "1 " in result
    assert "line1" in result
    assert "3 lines" in result or "3)" in result


def test_read_file_range(tmp_path: Path) -> None:
    f = tmp_path / "foo.py"
    f.write_text("\n".join(f"line{i}" for i in range(1, 301)))
    result = read_file(str(f), start_line=10, end_line=20, repo_root=str(tmp_path))
    assert "line10" in result
    assert "line20" in result
    assert "line9" not in result


def test_read_file_caps_at_200(tmp_path: Path) -> None:
    f = tmp_path / "foo.py"
    f.write_text("\n".join(f"line{i}" for i in range(1, 501)))
    result = read_file(str(f), repo_root=str(tmp_path))
    assert "200" in result or "truncated" in result.lower()
