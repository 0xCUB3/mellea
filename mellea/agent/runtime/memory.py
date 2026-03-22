"""Reusable memory primitives for condensing long text agent trajectories."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

Message = dict[str, str]


def _copy_messages(messages: Sequence[Message]) -> tuple[Message, ...]:
    return tuple(dict(message) for message in messages)


def _normalize_items(items: Sequence[str]) -> tuple[str, ...]:
    return tuple(item for item in items if item)


@dataclass(frozen=True)
class WorkingMemory:
    """Structured state that can survive history condensation."""

    summary: str = ""
    facts: tuple[str, ...] = field(default_factory=tuple)
    hypotheses: tuple[str, ...] = field(default_factory=tuple)
    next_steps: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        """Normalize optional sequence fields into filtered tuples."""
        object.__setattr__(self, "facts", _normalize_items(self.facts))
        object.__setattr__(self, "hypotheses", _normalize_items(self.hypotheses))
        object.__setattr__(self, "next_steps", _normalize_items(self.next_steps))

    def to_dict(self) -> dict[str, str | list[str]]:
        """Return a serializable snapshot."""
        return {
            "summary": self.summary,
            "facts": list(self.facts),
            "hypotheses": list(self.hypotheses),
            "next_steps": list(self.next_steps),
        }

    def is_empty(self) -> bool:
        """Return True when the memory carries no information."""
        return not (self.summary or self.facts or self.hypotheses or self.next_steps)

    def as_message(self, *, omitted_messages: int = 0) -> Message:
        """Render the memory as a structured user reminder."""
        lines = ["Condensed context:"]
        if omitted_messages:
            noun = "message" if omitted_messages == 1 else "messages"
            lines.append(f"{omitted_messages} earlier {noun} were condensed.")
        if self.summary:
            lines.extend(["Summary:", self.summary])
        if self.facts:
            lines.append("Facts:")
            lines.extend(f"- {fact}" for fact in self.facts)
        if self.hypotheses:
            lines.append("Hypotheses:")
            lines.extend(f"- {hypothesis}" for hypothesis in self.hypotheses)
        if self.next_steps:
            lines.append("Next steps:")
            lines.extend(f"- {step}" for step in self.next_steps)
        return {"role": "user", "content": "\n".join(lines)}


@dataclass(frozen=True)
class CondensedState:
    """Condensed trajectory state plus a recent verbatim tail."""

    working_memory: WorkingMemory = field(default_factory=WorkingMemory)
    recent_messages: tuple[Message, ...] = field(default_factory=tuple)
    omitted_messages: int = 0

    def __post_init__(self) -> None:
        """Copy recent messages so callers cannot mutate stored state."""
        object.__setattr__(self, "recent_messages", _copy_messages(self.recent_messages))

    def is_empty(self) -> bool:
        """Return True when no condensation state is present."""
        return self.omitted_messages == 0 and self.working_memory.is_empty() and not self.recent_messages


def should_condense_messages(
    messages: Sequence[Message],
    *,
    max_messages: int,
    preserve_recent: int = 4,
    preserve_head: int = 2,
) -> bool:
    """Return True when a trajectory is long enough to justify condensation."""
    if max_messages <= preserve_head + preserve_recent:
        return len(messages) > max_messages
    return len(messages) > max_messages and len(messages) > preserve_head + preserve_recent


def condense_messages(
    messages: Sequence[Message],
    *,
    working_memory: WorkingMemory,
    preserve_recent: int = 4,
    preserve_head: int = 2,
) -> CondensedState:
    """Drop the middle of a long trajectory while preserving a recent tail."""
    total = len(messages)
    if total <= preserve_head:
        return CondensedState(working_memory=working_memory)

    tail_start = max(preserve_head, total - preserve_recent)
    recent_messages = messages[tail_start:]
    omitted_messages = max(0, tail_start - preserve_head)
    return CondensedState(
        working_memory=working_memory,
        recent_messages=recent_messages,
        omitted_messages=omitted_messages,
    )


def materialize_messages(
    *,
    system_prompt: str,
    goal: str,
    condensed_state: CondensedState | None = None,
) -> list[Message]:
    """Create the message history for a text loop from optional condensed state."""
    messages: list[Message] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": goal},
    ]
    if condensed_state is None:
        return messages

    if condensed_state.omitted_messages or not condensed_state.working_memory.is_empty():
        messages.append(
            condensed_state.working_memory.as_message(
                omitted_messages=condensed_state.omitted_messages
            )
        )

    messages.extend(dict(message) for message in condensed_state.recent_messages)
    return messages
