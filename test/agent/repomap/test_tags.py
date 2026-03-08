from __future__ import annotations

from pathlib import Path

from mellea.agent.repomap.tags import extract_tags


def test_extract_python_defs(tmp_path: Path) -> None:
    f = tmp_path / "foo.py"
    f.write_text(
        "class Foo:\n    def bar(self):\n        pass\n\ndef baz():\n    pass\n"
    )
    tags = extract_tags(str(f))
    def_names = [t.name for t in tags if t.kind == "def"]
    assert "Foo" in def_names
    assert "bar" in def_names
    assert "baz" in def_names


def test_extract_javascript_defs(tmp_path: Path) -> None:
    f = tmp_path / "foo.js"
    f.write_text("function hello() {}\nclass World {}\n")
    tags = extract_tags(str(f))
    def_names = [t.name for t in tags if t.kind == "def"]
    assert "hello" in def_names
    assert "World" in def_names


def test_extract_unknown_extension(tmp_path: Path) -> None:
    f = tmp_path / "foo.xyz"
    f.write_text("whatever content\n")
    tags = extract_tags(str(f))
    assert tags == []
