"""PageRank-based file ranking on symbol graphs."""

from __future__ import annotations

from mellea.agent.repomap.tags import Tag


def _pagerank(
    nodes: list[str],
    edges: list[tuple[str, str]],
    personalization: dict[str, float],
    *,
    alpha: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> dict[str, float]:
    """Pure-Python personalized PageRank (no scipy needed)."""
    n = len(nodes)
    if n == 0:
        return {}

    idx = {node: i for i, node in enumerate(nodes)}
    out_degree: dict[int, int] = dict.fromkeys(range(n), 0)
    in_edges: dict[int, list[int]] = {i: [] for i in range(n)}

    for src, dst in edges:
        si, di = idx[src], idx[dst]
        out_degree[si] += 1
        in_edges[di].append(si)

    # Normalize personalization.
    p_sum = sum(personalization.values())
    p = [personalization.get(node, 0.0) / p_sum for node in nodes]

    rank = list(p)

    for _ in range(max_iter):
        new_rank = [0.0] * n
        dangling_sum = sum(rank[i] for i in range(n) if out_degree[i] == 0)

        for i in range(n):
            s = 0.0
            for j in in_edges[i]:
                s += rank[j] / out_degree[j]
            new_rank[i] = alpha * (s + dangling_sum * p[i]) + (1 - alpha) * p[i]

        diff = sum(abs(new_rank[i] - rank[i]) for i in range(n))
        rank = new_rank
        if diff < n * tol:
            break

    return {nodes[i]: rank[i] for i in range(n)}


def rank_files(tags: list[Tag], seed_files: list[str], *, top_n: int = 30) -> list[str]:
    """Rank files by relevance using PageRank on a symbol-reference graph."""
    if not tags:
        return list(seed_files)[:top_n]

    all_files: set[str] = set()
    defs: dict[str, list[str]] = {}

    for t in tags:
        all_files.add(t.rel_path)
        if t.kind == "def":
            defs.setdefault(t.name, []).append(t.rel_path)

    edges: list[tuple[str, str]] = []
    for t in tags:
        if t.kind == "ref" and t.name in defs:
            for def_file in defs[t.name]:
                if def_file != t.rel_path:
                    edges.append((t.rel_path, def_file))

    nodes = sorted(all_files)
    if not nodes:
        return list(seed_files)[:top_n]

    n = len(nodes)
    seed_set = set(seed_files)
    personalization = {node: 100.0 if node in seed_set else 1.0 / n for node in nodes}

    pr = _pagerank(nodes, edges, personalization)
    ranked = sorted(pr, key=lambda x: pr[x], reverse=True)
    return ranked[:top_n]
