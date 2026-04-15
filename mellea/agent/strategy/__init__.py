"""Strategy components for agent behavior."""

from __future__ import annotations

from dataclasses import dataclass

from .phased import get_available_tools as _phased_get_available_tools


@dataclass(frozen=True)
class ToolInvocation:
    """Records one tool invocation for phase tracking."""

    name: str
    status: str = "completed"


@dataclass(frozen=True)
class ToolPhaseState:
    """Summarizes tool activity within the current loop budget."""

    turn: int
    budget: int
    invocations: tuple[ToolInvocation, ...] = ()
    malformed_tool_calls: int = 0
    final_answer_blocks: int = 0

    @property
    def has_edit(self) -> bool:
        """Return whether any invocation edited the workspace."""
        return any(call.name == "edit" for call in self.invocations)

    @property
    def progress(self) -> float:
        """Return normalized progress through the current budget."""
        return self.turn / max(1, self.budget)

    @property
    def last_tool_name(self) -> str | None:
        """Return the most recent tool name, if any."""
        if not self.invocations:
            return None
        return self.invocations[-1].name

    @property
    def repeated_tool_streak(self) -> int:
        """Return the trailing streak length for the latest tool name."""
        last_tool_name = self.last_tool_name
        if last_tool_name is None:
            return 0
        streak = 0
        for call in reversed(self.invocations):
            if call.name != last_tool_name:
                break
            streak += 1
        return streak


def get_available_tools(
    all_tool_names: list[str],
    turn: int,
    budget: int,
    *,
    state: ToolPhaseState | None = None,
    policy: object | None = None,
    phases: tuple[float, ...] = (0.4, 0.8, 1.0),
) -> list[str]:
    """Return the currently available tools for phased access control."""
    del state, policy
    return _phased_get_available_tools(
        all_tool_names,
        turn=turn,
        budget=budget,
        phases=phases,
    )
