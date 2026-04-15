"""Loop detection for repeated tool calls."""

from __future__ import annotations


def detect_loop(
    history: list[tuple[str, tuple]],
    *,
    nudge_threshold: int = 2,
    force_threshold: int = 3,
) -> str | None:
    """Check if the most recent tool calls are repeated.

    Returns:
        None: no loop detected.
        "nudge": same call repeated nudge_threshold times.
        "force_switch": same call repeated force_threshold times.
    """
    if len(history) < nudge_threshold:
        return None

    last = history[-1]
    repeat_count = 0
    for call in reversed(history):
        if call == last:
            repeat_count += 1
        else:
            break

    if repeat_count >= force_threshold:
        return "force_switch"
    if repeat_count >= nudge_threshold:
        return "nudge"
    return None


NUDGE_MESSAGE = (
    "You already tried this exact tool call. "
    "Try a different approach: read a file, search for something else, "
    "or make an edit."
)

FORCE_MESSAGE = (
    "You have repeated the same tool call 3 times. You MUST call a different tool now."
)
