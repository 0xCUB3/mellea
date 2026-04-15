from __future__ import annotations

import math
import re
from pathlib import Path


def _tokenize(text: str) -> list[str]:
    """Split text into lowercase tokens."""
    return re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", text.lower())


def rank_bm25(
    paths: list[str],
    query: str,
    repo_root: str,
    *,
    top_n: int = 30,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[str]:
    """Rank files by BM25 relevance to a query string."""
    query_tokens = _tokenize(query)
    if not query_tokens or not paths:
        return paths[:top_n]

    # Build document token lists.
    doc_tokens: list[list[str]] = []
    valid_paths: list[str] = []
    for p in paths:
        try:
            text = Path(p).read_text(errors="replace")
            tokens = _tokenize(text)
            doc_tokens.append(tokens)
            valid_paths.append(p)
        except OSError:
            continue

    if not valid_paths:
        return []

    n_docs = len(valid_paths)
    avg_dl = sum(len(d) for d in doc_tokens) / max(1, n_docs)

    # IDF per query token.
    df: dict[str, int] = {}
    for qt in set(query_tokens):
        df[qt] = sum(1 for d in doc_tokens if qt in set(d))

    idf: dict[str, float] = {}
    for qt in set(query_tokens):
        idf[qt] = math.log((n_docs - df[qt] + 0.5) / (df[qt] + 0.5) + 1.0)

    # Score each document.
    scores: list[tuple[float, str]] = []
    for tokens, path in zip(doc_tokens, valid_paths):
        score = 0.0
        dl = len(tokens)
        tf_map: dict[str, int] = {}
        for t in tokens:
            tf_map[t] = tf_map.get(t, 0) + 1
        for qt in query_tokens:
            tf = tf_map.get(qt, 0)
            num = tf * (k1 + 1)
            den = tf + k1 * (1 - b + b * dl / avg_dl)
            score += idf.get(qt, 0) * num / den
        scores.append((score, path))

    scores.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in scores[:top_n]]
