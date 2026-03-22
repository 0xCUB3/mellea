"""Text-based ReAct loop that doesn't depend on API tool calling.

Uses plain chat completions and parses tool calls from the model's
text output. Works with any model on any inference server.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass

from mellea.agent.runtime.events import (
    EventLog,
    TerminationReason,
    ToolCallEvent,
    ToolResultEvent,
)
from mellea.agent.runtime.loops import (
    LoopBudget,
    LoopContinue,
    LoopRetry,
    LoopTerminate,
    build_loop_messages,
    run_observe_act_verify_loop,
)
from mellea.agent.runtime.memory import CondensedState, Message
from mellea.agent.text_tool_calling import format_tools_for_prompt, parse_tool_calls
from mellea.backends.model_options import ModelOption
from mellea.core.base import AbstractMelleaTool
from mellea.core.utils import FancyLogger


@dataclass(frozen=True)
class _TextReactState:
    messages: tuple[Message, ...]


@dataclass(frozen=True)
class _ModelTurn:
    text: str = ""
    error: str | None = None


async def text_react(
    goal: str,
    backend: object,
    *,
    tools: list[AbstractMelleaTool] | None = None,
    system_prompt: str = "",
    model_options: dict | None = None,
    loop_budget: int = 15,
    on_turn: Callable[[int, int, list[dict]], list[dict]] | None = None,
    condensed_state: CondensedState | None = None,
    event_log: EventLog | None = None,
    max_retries_per_turn: int = 0,
) -> tuple[str, bool]:
    """Run a text-based ReAct loop.

    Returns (final_answer_text, completed).
    """
    from openai import AsyncOpenAI

    tools = tools or []
    tool_map = {t.name: t for t in tools}

    tool_prompt = format_tools_for_prompt(tool_map)

    base_url = os.environ.get("OPENAI_BASE_URL")
    api_key = os.environ.get("OPENAI_API_KEY", "unused")
    model_id = getattr(backend, "_model_id", None)
    if model_id is None:
        model_id = os.environ.get("MCODE_MODEL_ID", "unknown")

    client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    max_tokens = 4096
    if model_options is not None and model_options.get(ModelOption.MAX_NEW_TOKENS) is not None:
        max_tokens = int(model_options[ModelOption.MAX_NEW_TOKENS])
    else:
        raw = os.environ.get("MCODE_MAX_NEW_TOKENS")
        if raw:
            max_tokens = int(raw)

    temperature = 0.0
    if model_options is not None and model_options.get(ModelOption.TEMPERATURE) is not None:
        temperature = float(model_options[ModelOption.TEMPERATURE])

    seed = None
    if model_options is not None:
        seed = model_options.get(ModelOption.SEED)

    full_system = system_prompt
    if tool_prompt:
        full_system = f"{system_prompt}\n\n{tool_prompt}"

    initial_messages, _ = build_loop_messages(
        system_prompt=full_system,
        goal=goal,
        condensed_state=condensed_state,
        event_log=event_log,
    )

    async def prepare_turn(state: _TextReactState, ctx) -> _TextReactState:
        FancyLogger.get_logger().info(f"## TEXT REACT TURN {ctx.turn}")
        if on_turn is None:
            return state
        messages = on_turn(ctx.turn, ctx.budget.max_turns, [dict(message) for message in state.messages])
        return _TextReactState(messages=tuple(dict(message) for message in messages))

    async def observe(state: _TextReactState, ctx) -> list[Message]:
        del ctx
        return [dict(message) for message in state.messages]

    async def act(state: _TextReactState, observation: list[Message], ctx) -> _ModelTurn:
        del state, ctx
        try:
            response = await client.chat.completions.create(
                model=model_id,
                messages=observation,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=120,
                seed=seed,
            )
            return _ModelTurn(text=response.choices[0].message.content or "")
        except Exception as e:
            return _ModelTurn(error=str(e))

    async def verify(
        state: _TextReactState,
        observation: list[Message],
        action: _ModelTurn,
        ctx,
    ):
        del observation
        messages = [dict(message) for message in state.messages]

        if action.error is not None:
            FancyLogger.get_logger().warning(f"API error: {action.error}")
            if ctx.attempt <= ctx.budget.max_retries_per_turn:
                return LoopRetry(state=state, detail=f"API error: {action.error}")
            messages.append({"role": "assistant", "content": f"Error: {action.error}"})
            return LoopContinue(state=_TextReactState(messages=tuple(messages)))

        text = action.text
        messages.append({"role": "assistant", "content": text})

        tool_calls = parse_tool_calls(text)
        if not tool_calls:
            # No tool call in response, model is done or confused
            return LoopContinue(state=_TextReactState(messages=tuple(messages)))

        for call in tool_calls:
            name = call["name"]
            args = call["arguments"]

            if name == "final_answer":
                answer = args.get("answer", args.get("explanation", str(args)))
                FancyLogger.get_logger().info(
                    f"Tool final_answer returned: {str(answer)[:200]}"
                )
                return LoopTerminate(
                    state=_TextReactState(messages=tuple(messages)),
                    result=answer,
                    reason=TerminationReason.FINAL_ANSWER,
                    detail="Model returned a final answer.",
                )

            tool = tool_map.get(name)
            if event_log is not None:
                event_log.emit(ToolCallEvent(tool_name=name, arguments=args))
            if tool is None:
                result = f"Error: unknown tool '{name}'"
                FancyLogger.get_logger().warning(result)
                status = "error"
            else:
                FancyLogger.get_logger().info(
                    f"Calling tool: {name} with args: {str(args)[:200]}"
                )
                try:
                    result = str(tool.run(**args))
                    status = "completed"
                except Exception as e:
                    result = f"Error: {type(e).__name__}: {e}"
                    FancyLogger.get_logger().warning(f"Tool {name} raised: {result}")
                    status = "error"
                FancyLogger.get_logger().info(
                    f"Tool {name} returned: {str(result)[:200]}"
                )

            if event_log is not None:
                event_log.emit(
                    ToolResultEvent(
                        tool_name=name,
                        status=status,
                        output=result,
                    )
                )
            messages.append({"role": "user", "content": f"[{name}] {result}"})

        return LoopContinue(state=_TextReactState(messages=tuple(messages)))

    result = await run_observe_act_verify_loop(
        initial_state=_TextReactState(messages=tuple(initial_messages)),
        budget=LoopBudget(
            max_turns=loop_budget,
            max_retries_per_turn=max_retries_per_turn,
        ),
        prepare_turn=prepare_turn,
        observe=observe,
        act=act,
        verify=verify,
        event_log=event_log,
    )
    return str(result.result or ""), result.completed
