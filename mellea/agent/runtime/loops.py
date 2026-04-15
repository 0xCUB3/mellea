"""Small reusable observe/act/verify loop helpers."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar

from mellea.agent.runtime.events import (
    EventLog,
    SummaryEvent,
    TerminationEvent,
    TerminationReason,
)
from mellea.agent.runtime.memory import (
    CondensedState,
    Message,
    WorkingMemory,
    condense_messages,
    materialize_messages,
)

StateT = TypeVar("StateT")
ObservationT = TypeVar("ObservationT")
ActionT = TypeVar("ActionT")
ResultT = TypeVar("ResultT")


@dataclass(frozen=True)
class LoopBudget:
    """Explicit turn and retry limits for a reusable runtime loop."""

    max_turns: int
    max_retries_per_turn: int = 0

    def __post_init__(self) -> None:
        """Reject invalid loop budgets early."""
        if self.max_turns < 1:
            raise ValueError("max_turns must be at least 1")
        if self.max_retries_per_turn < 0:
            raise ValueError("max_retries_per_turn cannot be negative")


@dataclass(frozen=True)
class LoopContext:
    """Turn-local execution context shared across observe/act/verify callables."""

    turn: int
    attempt: int
    budget: LoopBudget


@dataclass(frozen=True)
class LoopContinue(Generic[StateT]):
    """Continue to the next turn with updated loop state."""

    state: StateT


@dataclass(frozen=True)
class LoopRetry(Generic[StateT]):
    """Retry the current turn with updated loop state."""

    state: StateT
    detail: str | None = None
    termination_reason: TerminationReason = TerminationReason.RUNTIME_ERROR


@dataclass(frozen=True)
class LoopTerminate(Generic[StateT, ResultT]):
    """Terminate the loop with an explicit reason and optional result."""

    state: StateT
    reason: TerminationReason
    result: ResultT | None = None
    detail: str | None = None


@dataclass(frozen=True)
class LoopResult(Generic[StateT, ResultT]):
    """Structured outcome from a reusable runtime loop."""

    state: StateT
    budget: LoopBudget
    result: ResultT | None
    completed: bool
    turns: int
    retries: int
    termination: TerminationEvent


@dataclass(frozen=True)
class CondensationConfig:
    """Configuration for condensing loop message history."""

    working_memory: WorkingMemory
    max_messages: int
    preserve_recent: int = 4
    preserve_head: int = 2


ObserveFn = Callable[[StateT, LoopContext], ObservationT | Awaitable[ObservationT]]
ActFn = Callable[[StateT, ObservationT, LoopContext], ActionT | Awaitable[ActionT]]
VerifyFn = Callable[
    [StateT, ObservationT, ActionT, LoopContext],
    LoopContinue[StateT] | LoopRetry[StateT] | LoopTerminate[StateT, ResultT]
    | Awaitable[LoopContinue[StateT] | LoopRetry[StateT] | LoopTerminate[StateT, ResultT]],
]
PrepareTurnFn = Callable[[StateT, LoopContext], StateT | Awaitable[StateT]]


async def _resolve(value):
    if isinstance(value, Awaitable):
        return await value
    return value


def _emit_termination(
    *,
    event_log: EventLog | None,
    reason: TerminationReason,
    detail: str,
) -> TerminationEvent:
    event = TerminationEvent(reason=reason, detail=detail)
    if event_log is not None:
        event_log.emit(event)
    return event


def build_loop_messages(
    *,
    system_prompt: str,
    goal: str,
    recent_messages: Sequence[Message] = (),
    condensed_state: CondensedState | None = None,
    condensation: CondensationConfig | None = None,
    event_log: EventLog | None = None,
) -> tuple[list[Message], CondensedState | None]:
    """Materialize loop messages with optional condensation and event emission."""
    active_condensed_state = condensed_state

    if active_condensed_state is None and condensation is not None:
        history = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": goal},
            *(dict(message) for message in recent_messages),
        ]
        active_condensed_state = condense_messages(
            history,
            working_memory=condensation.working_memory,
            max_messages=condensation.max_messages,
            preserve_recent=condensation.preserve_recent,
            preserve_head=condensation.preserve_head,
        )
        if active_condensed_state.omitted_messages > 0 and event_log is not None:
            event_log.emit(
                SummaryEvent(
                    message="Condensed loop history",
                    metadata={
                        "omitted_messages": active_condensed_state.omitted_messages,
                        "max_messages": condensation.max_messages,
                    },
                )
            )
    elif active_condensed_state is None and recent_messages:
        active_condensed_state = CondensedState(
            recent_messages=tuple(dict(message) for message in recent_messages),
            show_reminder=False,
        )

    return (
        materialize_messages(
            system_prompt=system_prompt,
            goal=goal,
            condensed_state=active_condensed_state,
        ),
        active_condensed_state,
    )


async def run_observe_act_verify_loop(
    *,
    initial_state: StateT,
    budget: LoopBudget,
    observe: ObserveFn[StateT, ObservationT],
    act: ActFn[StateT, ObservationT, ActionT],
    verify: VerifyFn[StateT, ObservationT, ActionT, ResultT],
    prepare_turn: PrepareTurnFn[StateT] | None = None,
    event_log: EventLog | None = None,
) -> LoopResult[StateT, ResultT]:
    """Run a reusable async loop with explicit budgets and termination."""
    state = initial_state
    retries = 0

    for turn in range(1, budget.max_turns + 1):
        if prepare_turn is not None:
            prepare_ctx = LoopContext(turn=turn, attempt=1, budget=budget)
            try:
                state = await _resolve(prepare_turn(state, prepare_ctx))
            except Exception as exc:
                termination = _emit_termination(
                    event_log=event_log,
                    reason=TerminationReason.RUNTIME_ERROR,
                    detail=f"{type(exc).__name__}: {exc}",
                )
                return LoopResult(
                    state=state,
                    budget=budget,
                    result=None,
                    completed=False,
                    turns=turn,
                    retries=retries,
                    termination=termination,
                )

        max_attempts = 1 + budget.max_retries_per_turn
        for attempt in range(1, max_attempts + 1):
            ctx = LoopContext(turn=turn, attempt=attempt, budget=budget)
            try:
                observation = await _resolve(observe(state, ctx))
                action = await _resolve(act(state, observation, ctx))
                decision = await _resolve(verify(state, observation, action, ctx))
            except Exception as exc:
                termination = _emit_termination(
                    event_log=event_log,
                    reason=TerminationReason.RUNTIME_ERROR,
                    detail=f"{type(exc).__name__}: {exc}",
                )
                return LoopResult(
                    state=state,
                    budget=budget,
                    result=None,
                    completed=False,
                    turns=turn,
                    retries=retries,
                    termination=termination,
                )

            if isinstance(decision, LoopRetry):
                if attempt >= max_attempts:
                    termination = _emit_termination(
                        event_log=event_log,
                        reason=decision.termination_reason,
                        detail=decision.detail or "Loop retry budget exhausted.",
                    )
                    return LoopResult(
                        state=decision.state,
                        budget=budget,
                        result=None,
                        completed=False,
                        turns=turn,
                        retries=retries,
                        termination=termination,
                    )

                if event_log is not None and decision.detail is not None:
                    event_log.emit(
                        SummaryEvent(
                            message="Retrying loop turn",
                            metadata={
                                "turn": turn,
                                "attempt": attempt,
                                "detail": decision.detail,
                            },
                        )
                    )

                retries += 1
                state = decision.state
                continue

            if isinstance(decision, LoopContinue):
                state = decision.state
                break

            termination = _emit_termination(
                event_log=event_log,
                reason=decision.reason,
                detail=decision.detail or "",
            )
            return LoopResult(
                state=decision.state,
                budget=budget,
                result=decision.result,
                completed=decision.reason is TerminationReason.FINAL_ANSWER,
                turns=turn,
                retries=retries,
                termination=termination,
            )

    termination = _emit_termination(
        event_log=event_log,
        reason=TerminationReason.MAX_TURNS,
        detail=f"Loop budget exhausted after {budget.max_turns} turns.",
    )
    return LoopResult(
        state=state,
        budget=budget,
        result=None,
        completed=False,
        turns=budget.max_turns,
        retries=retries,
        termination=termination,
    )
