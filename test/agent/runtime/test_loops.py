from __future__ import annotations

import pytest

from mellea.agent.runtime.events import EventLog, SummaryEvent, TerminationReason
from mellea.agent.runtime.loops import (
    CondensationConfig,
    LoopBudget,
    LoopContinue,
    LoopRetry,
    LoopTerminate,
    build_loop_messages,
    run_observe_act_verify_loop,
)
from mellea.agent.runtime.memory import WorkingMemory


@pytest.mark.asyncio
async def test_generic_observe_act_verify_loop_runs_without_benchmark_scaffolding():
    async def observe(state: dict[str, int], ctx) -> int:
        del ctx
        return state["value"]

    async def act(state: dict[str, int], observation: int, ctx) -> int:
        del state, ctx
        return observation + 1

    async def verify(state: dict[str, int], observation: int, action: int, ctx):
        del observation, ctx
        next_state = {"value": action}
        if action >= 3:
            return LoopTerminate(
                state=next_state,
                result=f"done:{action}",
                reason=TerminationReason.FINAL_ANSWER,
                detail="Reached the configured target.",
            )
        return LoopContinue(state=next_state)

    result = await run_observe_act_verify_loop(
        initial_state={"value": 0},
        budget=LoopBudget(max_turns=5),
        observe=observe,
        act=act,
        verify=verify,
    )

    assert result.budget == LoopBudget(max_turns=5)
    assert result.completed is True
    assert result.result == "done:3"
    assert result.state == {"value": 3}
    assert result.turns == 3
    assert result.retries == 0
    assert result.termination.reason is TerminationReason.FINAL_ANSWER


@pytest.mark.asyncio
async def test_loop_result_tracks_retry_usage_and_budget_termination():
    attempts: list[tuple[int, int]] = []
    event_log = EventLog()

    async def observe(state: dict[str, tuple[tuple[int, int], ...]], ctx) -> str:
        del state
        return f"turn-{ctx.turn}"

    async def act(state: dict[str, tuple[tuple[int, int], ...]], observation: str, ctx) -> str:
        del state, observation
        attempts.append((ctx.turn, ctx.attempt))
        return "retry"

    async def verify(state: dict[str, tuple[tuple[int, int], ...]], observation: str, action: str, ctx):
        del observation, action
        next_state = {"attempts": state["attempts"] + ((ctx.turn, ctx.attempt),)}
        if ctx.attempt <= ctx.budget.max_retries_per_turn:
            return LoopRetry(state=next_state, detail=f"retry turn {ctx.turn} attempt {ctx.attempt}")
        return LoopContinue(state=next_state)

    result = await run_observe_act_verify_loop(
        initial_state={"attempts": ()},
        budget=LoopBudget(max_turns=2, max_retries_per_turn=1),
        observe=observe,
        act=act,
        verify=verify,
        event_log=event_log,
    )

    assert attempts == [(1, 1), (1, 2), (2, 1), (2, 2)]
    assert result.budget == LoopBudget(max_turns=2, max_retries_per_turn=1)
    assert result.completed is False
    assert result.turns == 2
    assert result.retries == 2
    assert result.state == {"attempts": ((1, 1), (1, 2), (2, 1), (2, 2))}
    assert result.termination.reason is TerminationReason.MAX_TURNS
    assert result.termination.detail == "Loop budget exhausted after 2 turns."
    assert event_log.to_dicts() == [
        {
            "kind": "summary",
            "message": "Retrying loop turn",
            "metadata": {"turn": 1, "attempt": 1, "detail": "retry turn 1 attempt 1"},
        },
        {
            "kind": "summary",
            "message": "Retrying loop turn",
            "metadata": {"turn": 2, "attempt": 1, "detail": "retry turn 2 attempt 1"},
        },
        {
            "kind": "termination",
            "reason": "max_turns",
            "detail": "Loop budget exhausted after 2 turns.",
        },
    ]


def test_build_loop_messages_can_condense_history_and_emit_runtime_events():
    event_log = EventLog()

    rendered, condensed = build_loop_messages(
        system_prompt="sys",
        goal="Recover the service",
        recent_messages=[
            {"role": "assistant", "content": "I checked the logs."},
            {"role": "user", "content": "[read_logs] upstream returned 502"},
            {"role": "assistant", "content": "The API cannot reach the database."},
            {"role": "user", "content": "[check_env] DB_HOST is unset in staging"},
            {"role": "assistant", "content": "This looks like deploy config drift."},
            {"role": "user", "content": "[list_deploys] latest deploy started at 14:32"},
        ],
        condensation=CondensationConfig(
            working_memory=WorkingMemory(
                summary="Investigated the outage and narrowed it to configuration drift.",
                facts=("DB_HOST is unset in staging.",),
                next_steps=("Restore DB_HOST and rerun health checks.",),
            ),
            max_messages=5,
            preserve_recent=2,
        ),
        event_log=event_log,
    )

    assert condensed is not None
    assert condensed.omitted_messages == 4
    assert len(rendered) == 5
    assert rendered[:2] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "Recover the service"},
    ]
    assert "Condensed context" in rendered[2]["content"]
    assert rendered[-2:] == [
        {"role": "assistant", "content": "This looks like deploy config drift."},
        {"role": "user", "content": "[list_deploys] latest deploy started at 14:32"},
    ]

    summary_events = [event for event in event_log if isinstance(event, SummaryEvent)]
    assert len(summary_events) == 1
    assert summary_events[0].message == "Condensed loop history"
    assert summary_events[0].metadata == {
        "omitted_messages": 4,
        "max_messages": 5,
    }
