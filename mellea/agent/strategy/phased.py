"""Phased tool access control."""

from __future__ import annotations

_EXPLORE_TOOLS = {"search_code", "read_file", "find_file", "list_dir"}
_COMMIT_TOOLS = {"edit", "final_answer"}
_DEFAULT_PHASES = (0.4, 0.8, 1.0)


def get_available_tools(
    all_tool_names: list[str],
    turn: int,
    budget: int,
    *,
    phases: tuple[float, ...] = _DEFAULT_PHASES,
) -> list[str]:
    """Return the tool names available at a given turn based on phase boundaries.

    Phases (percentage of budget):
        Phase 1 (0 to phases[0]): explore tools only.
        Phase 2 (phases[0] to phases[1]): all tools.
        Phase 3 (phases[1] to 1.0): edit + final_answer only.
    """
    assert len(phases) == 3, "phases must have exactly 3 values"
    progress = turn / max(1, budget)

    if progress <= phases[0]:
        return [t for t in all_tool_names if t in _EXPLORE_TOOLS]
    elif progress <= phases[1]:
        return list(all_tool_names)
    else:
        return [t for t in all_tool_names if t in _COMMIT_TOOLS]
