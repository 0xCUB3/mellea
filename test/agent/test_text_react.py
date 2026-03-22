from __future__ import annotations

import asyncio

from mellea.agent.runtime import EventLog, TerminationReason
from mellea.agent.runtime.memory import CondensedState, WorkingMemory
from mellea.agent.text_react import text_react
from mellea.backends import ModelOption
from mellea.core.base import AbstractMelleaTool


class _FakeResponseMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeResponseMessage(content)


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def __init__(self) -> None:
        self.kwargs = None
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.kwargs = kwargs
        self.calls.append(kwargs)
        if _FakeAsyncOpenAI.responses:
            return _FakeResponse(_FakeAsyncOpenAI.responses.pop(0))
        return _FakeResponse(
            '<tool_call>{"name": "final_answer", "arguments": {"answer": "ok"}}</tool_call>'
        )


class _FakeChat:
    def __init__(self) -> None:
        self.completions = _FakeCompletions()


class _FakeAsyncOpenAI:
    last = None
    responses: list[str] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.chat = _FakeChat()
        type(self).last = self


class _FakeBackend:
    _model_id = "model-x"


class _EchoTool(AbstractMelleaTool):
    name = "echo"

    def run(self, *, text: str) -> str:
        return f"echo:{text}"

    @property
    def as_json_tool(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "echo",
                "description": "Echo text back to the model.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                    },
                    "required": ["text"],
                },
            },
        }


def test_text_react_uses_model_options(monkeypatch):
    monkeypatch.setattr("openai.AsyncOpenAI", _FakeAsyncOpenAI)
    _FakeAsyncOpenAI.responses = []

    answer, done = asyncio.run(
        text_react(
            goal="solve it",
            backend=_FakeBackend(),
            system_prompt="sys",
            model_options={
                ModelOption.MAX_NEW_TOKENS: 321,
                ModelOption.TEMPERATURE: 0.4,
                ModelOption.SEED: 99,
            },
            loop_budget=1,
        )
    )

    assert (answer, done) == ("ok", True)
    assert _FakeAsyncOpenAI.last is not None
    kwargs = _FakeAsyncOpenAI.last.chat.completions.kwargs
    assert kwargs is not None
    assert kwargs["model"] == "model-x"
    assert kwargs["messages"][:2] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "solve it"},
    ]
    assert kwargs["max_tokens"] == 321
    assert kwargs["temperature"] == 0.4
    assert kwargs["timeout"] == 120
    assert kwargs["seed"] == 99


def test_text_react_accepts_condensed_state(monkeypatch):
    monkeypatch.setattr("openai.AsyncOpenAI", _FakeAsyncOpenAI)
    _FakeAsyncOpenAI.responses = []

    answer, done = asyncio.run(
        text_react(
            goal="Recover the service",
            backend=_FakeBackend(),
            system_prompt="sys",
            condensed_state=CondensedState(
                working_memory=WorkingMemory(
                    summary="The deploy introduced a staging config mismatch.",
                    facts=("DB_HOST is unset.",),
                    hypotheses=("The release job skipped env sync.",),
                    next_steps=("Restore the variable before retrying.",),
                ),
                recent_messages=(
                    {"role": "assistant", "content": "I narrowed the outage to staging config."},
                    {"role": "user", "content": "[check_env] DB_HOST is unset"},
                ),
                omitted_messages=6,
            ),
            loop_budget=1,
        )
    )

    assert (answer, done) == ("ok", True)
    kwargs = _FakeAsyncOpenAI.last.chat.completions.kwargs
    assert kwargs is not None
    assert kwargs["messages"][:5] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "Recover the service"},
        {
            "role": "user",
            "content": (
                "Condensed context:\n"
                "6 earlier messages were condensed.\n"
                "Summary:\n"
                "The deploy introduced a staging config mismatch.\n"
                "Facts:\n"
                "- DB_HOST is unset.\n"
                "Hypotheses:\n"
                "- The release job skipped env sync.\n"
                "Next steps:\n"
                "- Restore the variable before retrying."
            ),
        },
        {"role": "assistant", "content": "I narrowed the outage to staging config."},
        {"role": "user", "content": "[check_env] DB_HOST is unset"},
    ]


def test_text_react_emits_runtime_events_for_tool_calls(monkeypatch):
    monkeypatch.setattr("openai.AsyncOpenAI", _FakeAsyncOpenAI)
    _FakeAsyncOpenAI.responses = [
        '<tool_call>{"name": "echo", "arguments": {"text": "ping"}}</tool_call>',
        '<tool_call>{"name": "final_answer", "arguments": {"answer": "done"}}</tool_call>',
    ]
    event_log = EventLog()

    answer, done = asyncio.run(
        text_react(
            goal="Debug the service",
            backend=_FakeBackend(),
            system_prompt="sys",
            tools=[_EchoTool()],
            loop_budget=3,
            event_log=event_log,
        )
    )

    assert (answer, done) == ("done", True)
    assert event_log.to_dicts() == [
        {
            "kind": "tool_call",
            "tool_name": "echo",
            "arguments": {"text": "ping"},
            "call_id": None,
        },
        {
            "kind": "tool_result",
            "tool_name": "echo",
            "call_id": None,
            "status": "completed",
            "output": "echo:ping",
            "duration_ms": None,
        },
        {
            "kind": "termination",
            "reason": "final_answer",
            "detail": "Model returned a final answer.",
        },
    ]
    assert event_log.final_reason() is TerminationReason.FINAL_ANSWER
