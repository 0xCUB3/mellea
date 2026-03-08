"""Repo map: structural code analysis for LLM context."""

from __future__ import annotations

from pathlib import Path

from mellea.agent.repomap.graph import rank_files
from mellea.agent.repomap.render import render_skeleton
from mellea.agent.repomap.tags import Tag, extract_tags

_SKIP_DIRS = {
    ".git",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    "build",
    "dist",
    ".tox",
    ".eggs",
}

_SOURCE_EXTS = {
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".java",
    ".go",
    ".rs",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".rb",
    ".sh",
    ".yaml",
    ".yml",
    ".json",
    ".toml",
}


def build_repo_map(
    repo_root: str, query: str, *, max_tokens: int = 4096, bm25_top_n: int = 30
) -> str:
    """Build a structural map of a repository for LLM context.

    Combines BM25 text ranking with tree-sitter structural analysis
    and PageRank to produce a condensed skeleton of the most relevant files.
    """
    root = Path(repo_root)
    source_files: list[str] = []
    for p in root.rglob("*"):
        if any(s in p.parts for s in _SKIP_DIRS):
            continue
        if p.is_file() and p.suffix.lower() in _SOURCE_EXTS:
            source_files.append(str(p))

    if not source_files:
        return ""

    from mellea.agent.repomap._bm25 import rank_bm25

    seed_files = rank_bm25(source_files, query, repo_root, top_n=bm25_top_n)

    all_tags: list[Tag] = []
    tags_by_file: dict[str, list[Tag]] = {}
    for fpath in source_files:
        ftags = extract_tags(fpath, repo_root=repo_root)
        if ftags:
            rel = str(Path(fpath).relative_to(repo_root))
            all_tags.extend(ftags)
            tags_by_file[rel] = ftags

    seed_rels = [str(Path(f).relative_to(repo_root)) for f in seed_files]
    ranked = rank_files(all_tags, seed_files=seed_rels)

    return render_skeleton(tags_by_file, ranked, max_tokens=max_tokens)
