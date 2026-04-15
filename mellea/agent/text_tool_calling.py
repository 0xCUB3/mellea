"""Text-based tool calling that bypasses vLLM's tool call parsers.

Instead of relying on the API's native tool calling interface (which
requires vLLM to parse model-specific tool call formats), this module
embeds tool schemas in the prompt and parses tool calls from the
model's plain text output using XML-delimited JSON blocks.

The model outputs tool calls as:
    <tool_call>
    {"name": "search_code", "arguments": {"query": "def foo"}}
    </tool_call>

This works with any model and any vLLM version since it's just text.
"""

from __future__ import annotations

import json
import re
from typing import Any


def format_tools_for_prompt(tools: dict[str, Any]) -> str:
    """Format tool schemas as a text block for inclusion in a prompt."""
    if not tools:
        return ""

    lines = ["You have the following tools available:\n"]
    for name, tool in tools.items():
        schema = tool.as_json_tool
        func = schema.get("function", {})
        desc = func.get("description", "")
        params = func.get("parameters", {})
        props = params.get("properties", {})
        required = params.get("required", [])

        param_parts = []
        for pname, pinfo in props.items():
            ptype = pinfo.get("type", "any")
            req = "(required)" if pname in required else "(optional)"
            param_parts.append(f"    {pname}: {ptype} {req}")

        param_str = "\n".join(param_parts) if param_parts else "    (no parameters)"
        lines.append(f"- {name}: {desc}\n  Parameters:\n{param_str}\n")

    lines.append(
        "To call a tool, output exactly this format:\n"
        "<tool_call>\n"
        '{"name": "tool_name", "arguments": {"arg1": "value1"}}\n'
        "</tool_call>\n\n"
        "Call one tool at a time. After the tool result, "
        "continue reasoning and call another tool or provide "
        "your final answer.\n"
        "When you are done, call the final_answer tool with "
        "your explanation."
    )
    return "\n".join(lines)


def parse_tool_calls(text: str) -> list[dict[str, Any]]:
    """Extract tool calls from model text output.

    Returns a list of dicts with 'name' and 'arguments' keys.
    """
    pattern = r"<tool_call>\s*(.*?)\s*</tool_call>"
    matches = re.findall(pattern, text, re.DOTALL)

    calls = []
    for match in matches:
        try:
            parsed = json.loads(match.strip())
            name = parsed.get("name")
            args = parsed.get("arguments", {})
            if name:
                calls.append({"name": name, "arguments": args})
        except (json.JSONDecodeError, AttributeError):
            continue
    return calls
