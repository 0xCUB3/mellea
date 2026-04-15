"""Skeleton renderer for ranked files."""

from __future__ import annotations

from mellea.agent.repomap.tags import Tag


def render_skeleton(
    tags_by_file: dict[str, list[Tag]],
    ranked_files: list[str],
    *,
    max_tokens: int = 4096,
) -> str:
    """Render a condensed skeleton of the top-ranked files."""
    lines: list[str] = []
    token_estimate = 0

    for rel_path in ranked_files:
        file_tags = tags_by_file.get(rel_path, [])
        if not file_tags:
            continue
        file_lines = [f"{rel_path}:"]
        for t in sorted(file_tags, key=lambda t: t.line):
            file_lines.append(f"  {t.line}: {t.kind} {t.name}")

        block = "\n".join(file_lines)
        block_tokens = len(block) // 4
        if token_estimate + block_tokens > max_tokens:
            break
        lines.append(block)
        token_estimate += block_tokens

    return "\n\n".join(lines)
