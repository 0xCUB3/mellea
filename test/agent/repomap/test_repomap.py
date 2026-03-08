from __future__ import annotations

from pathlib import Path

from mellea.agent.repomap import build_repo_map


def test_build_repo_map_python(tmp_path: Path) -> None:
    (tmp_path / "main.py").write_text(
        "from util import helper\n\ndef main():\n    helper()\n"
    )
    (tmp_path / "util.py").write_text("def helper():\n    return 42\n")
    result = build_repo_map(str(tmp_path), query="main function", max_tokens=2000)
    assert "main.py" in result
    assert "def main" in result
    assert "util.py" in result
    assert "def helper" in result


def test_build_repo_map_empty_dir(tmp_path: Path) -> None:
    result = build_repo_map(str(tmp_path), query="anything")
    assert result == "" or "No source files" in result
