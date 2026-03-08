from __future__ import annotations

from mellea.agent.repomap.graph import rank_files
from mellea.agent.repomap.tags import Tag


def test_rank_files_returns_sorted_paths() -> None:
    tags = [
        Tag("a.py", "Foo", 1, "def"),
        Tag("b.py", "Foo", 5, "ref"),
        Tag("b.py", "Bar", 1, "def"),
        Tag("c.py", "Bar", 3, "ref"),
        Tag("c.py", "Foo", 7, "ref"),
    ]
    ranked = rank_files(tags, seed_files=["a.py"])
    assert isinstance(ranked, list)
    assert len(ranked) > 0
    assert ranked[0] == "a.py"


def test_rank_files_empty_tags() -> None:
    ranked = rank_files([], seed_files=[])
    assert ranked == []
