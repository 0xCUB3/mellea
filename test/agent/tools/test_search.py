from __future__ import annotations

from pathlib import Path

from mellea.agent.tools.search import search_code


def test_search_finds_match(tmp_path: Path) -> None:
    (tmp_path / "foo.py").write_text("def hello_world():\n    pass\n")
    (tmp_path / "bar.py").write_text("x = 1\n")
    result = search_code("hello_world", repo_root=str(tmp_path))
    assert "foo.py" in result
    assert "hello_world" in result
    assert "bar.py" not in result


def test_search_no_matches(tmp_path: Path) -> None:
    (tmp_path / "foo.py").write_text("x = 1\n")
    result = search_code("nonexistent_symbol", repo_root=str(tmp_path))
    assert "No matches" in result


def test_search_caps_results(tmp_path: Path) -> None:
    for i in range(50):
        (tmp_path / f"f{i}.py").write_text(f"match_me_{i} = {i}\n")
    result = search_code("match_me", repo_root=str(tmp_path))
    assert result.count("\n") <= 31


def test_search_skips_git_dir(tmp_path: Path) -> None:
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "config").write_text("match_me\n")
    (tmp_path / "real.py").write_text("match_me\n")
    result = search_code("match_me", repo_root=str(tmp_path))
    assert "real.py" in result
    assert ".git" not in result
