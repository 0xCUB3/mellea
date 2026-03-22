from __future__ import annotations

import asyncio

from mellea.agent.runtime.memory import CondensedState, WorkingMemory
from mellea.agent.text_react import text_react
from mellea.backends import ModelOption


class _FakeResponseMessage:
    content = '<tool_call>{"name": "final_answer", "arguments": {"answer": "ok"}}</tool_call>'


class _FakeChoice:
    message = _FakeResponseMessage()


class _FakeResponse:
    choices = [_FakeChoice()]


class _FakeCompletions:
    def __init__(self) -> None:
        self.kwargs = None

    async def create(self, **kwargs):
        self.kwargs = kwargs
        return _FakeResponse()


class _FakeChat:
    def __init__(self) -> None:
        self.completions = _FakeCompletions()


class _FakeAsyncOpenAI:
    last = None

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.chat = _FakeChat()
        type(self).last = self


class _FakeBackend:
    _model_id = "model-x"


def test_text_react_uses_model_options(monkeypatch):
    monkeypatch.setattr("openai.AsyncOpenAI", _FakeAsyncOpenAI)

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
