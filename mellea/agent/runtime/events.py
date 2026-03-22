"""Structured event primitives for agent runtimes."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from mellea.agent.runtime.workspace import Workspace


def _copy_mapping(mapping: Mapping[str, Any] | None) -> dict[str, Any]:
    return dict(mapping or {})


class TerminationReason(StrEnum):
    """Common reasons an agent runtime can terminate."""

    FINAL_ANSWER = "final_answer"
    MAX_TURNS = "max_turns"
    TOOL_ERROR = "tool_error"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class ToolCallEvent:
    """Records the start of a tool action."""

    tool_name: str
    arguments: Mapping[str, Any]
    call_id: str | None = None
    kind: str = field(init=False, default="tool_call")

    def as_dict(self) -> dict[str, Any]:
        """Return a serializable representation of the event."""
        return {
            "kind": self.kind,
            "tool_name": self.tool_name,
            "arguments": _copy_mapping(self.arguments),
            "call_id": self.call_id,
        }


@dataclass(frozen=True)
class ToolResultEvent:
    """Records the result of a tool action."""

    tool_name: str
    status: str
    output: Any
    call_id: str | None = None
    duration_ms: int | None = None
    kind: str = field(init=False, default="tool_result")

    def as_dict(self) -> dict[str, Any]:
        """Return a serializable representation of the event."""
        return {
            "kind": self.kind,
            "tool_name": self.tool_name,
            "call_id": self.call_id,
            "status": self.status,
            "output": self.output,
            "duration_ms": self.duration_ms,
        }


@dataclass(frozen=True)
class SummaryEvent:
    """Captures a structured narrative summary."""

    message: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    kind: str = field(init=False, default="summary")

    def as_dict(self) -> dict[str, Any]:
        """Return a serializable representation of the event."""
        return {
            "kind": self.kind,
            "message": self.message,
            "metadata": _copy_mapping(self.metadata),
        }


@dataclass(frozen=True)
class TerminationEvent:
    """Captures why a runtime stopped."""

    reason: TerminationReason
    detail: str | None = None
    kind: str = field(init=False, default="termination")

    def as_dict(self) -> dict[str, Any]:
        """Return a serializable representation of the event."""
        return {
            "kind": self.kind,
            "reason": self.reason.value,
            "detail": self.detail,
        }


RuntimeEvent = ToolCallEvent | ToolResultEvent | SummaryEvent | TerminationEvent


class EventLog:
    """Small event log that can be used independently of benchmark integrations."""

    def __init__(
        self,
        *,
        workspace: Workspace | None = None,
        events: Iterable[RuntimeEvent] | None = None,
    ) -> None:
        """Initialize an event log with optional workspace context and seed events."""
        self.workspace = workspace
        self._events = list(events or [])

    def emit(self, event: RuntimeEvent) -> RuntimeEvent:
        """Append an event and return it for call-site convenience."""
        self._events.append(event)
        return event

    def to_dicts(self) -> list[dict[str, Any]]:
        """Serialize all events in insertion order."""
        return [event.as_dict() for event in self._events]

    def snapshot(self) -> dict[str, Any]:
        """Return a serializable snapshot of workspace context and events."""
        return {
            "workspace": self.workspace.describe() if self.workspace else None,
            "events": self.to_dicts(),
        }

    def final_reason(self) -> TerminationReason | None:
        """Return the last recorded termination reason, if any."""
        for event in reversed(self._events):
            if isinstance(event, TerminationEvent):
                return event.reason
        return None

    def __iter__(self) -> Iterator[RuntimeEvent]:
        """Iterate over recorded events."""
        return iter(self._events)

    def __len__(self) -> int:
        """Return the number of recorded events."""
        return len(self._events)
