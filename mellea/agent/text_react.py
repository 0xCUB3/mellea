"""Text-based ReAct loop that doesn't depend on API tool calling.

Uses plain chat completions and parses tool calls from the model's
text output. Works with any model on any inference server.
"""

from __future__ import annotations

import os
from collections.abc import Callable

from mellea.agent.text_tool_calling import format_tools_for_prompt, parse_tool_calls
from mellea.backends.model_options import ModelOption
from mellea.core.base import AbstractMelleaTool, ModelToolCall
from mellea.core.utils import FancyLogger
from mellea.stdlib.components.chat import ToolMessage


async def text_react(
    goal: str,
    backend: object,
    *,
    tools: list[AbstractMelleaTool] | None = None,
    system_prompt: str = "",
    model_options: dict | None = None,
    loop_budget: int = 15,
    on_turn: Callable[[int, int, list[dict]], list[dict]] | None = None,
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

    messages: list[dict] = [
        {"role": "system", "content": full_system},
        {"role": "user", "content": goal},
    ]

    for turn in range(1, loop_budget + 1):
        FancyLogger.get_logger().info(f"## TEXT REACT TURN {turn}")

        if on_turn is not None:
            messages = on_turn(turn, loop_budget, messages)

        try:
            response = await client.chat.completions.create(
                model=model_id,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=120,
                seed=seed,
            )
            text = response.choices[0].message.content or ""
        except Exception as e:
            FancyLogger.get_logger().warning(f"API error: {e}")
            messages.append({"role": "assistant", "content": f"Error: {e}"})
            continue

        messages.append({"role": "assistant", "content": text})

        tool_calls = parse_tool_calls(text)
        if not tool_calls:
            # No tool call in response, model is done or confused
            if turn == loop_budget:
                break
            continue

        for call in tool_calls:
            name = call["name"]
            args = call["arguments"]

            if name == "final_answer":
                answer = args.get("answer", args.get("explanation", str(args)))
                FancyLogger.get_logger().info(
                    f"Tool final_answer returned: {str(answer)[:200]}"
                )
                return answer, True

            tool = tool_map.get(name)
            if tool is None:
                result = f"Error: unknown tool '{name}'"
                FancyLogger.get_logger().warning(result)
            else:
                FancyLogger.get_logger().info(
                    f"Calling tool: {name} with args: {str(args)[:200]}"
                )
                try:
                    result = str(tool.run(**args))
                except Exception as e:
                    result = f"Error: {type(e).__name__}: {e}"
                    FancyLogger.get_logger().warning(f"Tool {name} raised: {result}")
                FancyLogger.get_logger().info(
                    f"Tool {name} returned: {str(result)[:200]}"
                )

            messages.append({"role": "user", "content": f"[{name}] {result}"})

    return "", False
