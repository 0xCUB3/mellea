"""Test execution tool for running repo tests during agent repair."""

from __future__ import annotations

import subprocess
from pathlib import Path

from mellea.agent.tools.bash import format_tool_result

_TIMEOUT_S = 120
_MAX_OUTPUT = 4000


def run_tests(test_cmd: str, *, repo_root: str, test_cmds: list[str]) -> str:
    """Run a test command in the repo and return output.

    The agent can pass a custom test_cmd or use 'default' to run the
    pre-configured test commands for this task.
    """
    root = Path(repo_root)
    if not root.is_dir():
        return format_tool_result(
            test_cmd,
            "ERROR",
            f"Error: repo root not found: {repo_root}",
        )

    # If agent asks for default, run the task's test commands
    if test_cmd.strip().lower() == "default":
        cmds = test_cmds
    else:
        cmds = [test_cmd]

    outputs: list[str] = []
    for cmd in cmds:
        if not cmd.strip():
            continue
        try:
            result = subprocess.run(
                ["bash", "-c", cmd],
                cwd=str(root),
                capture_output=True,
                text=True,
                timeout=_TIMEOUT_S,
            )
            out = result.stdout + result.stderr
            if len(out) > _MAX_OUTPUT:
                out = out[-_MAX_OUTPUT:]
            status = (
                "PASSED"
                if result.returncode == 0
                else f"FAILED (exit {result.returncode})"
            )
            outputs.append(format_tool_result(cmd, status, out))
        except subprocess.TimeoutExpired:
            outputs.append(format_tool_result(cmd, f"TIMEOUT after {_TIMEOUT_S}s", ""))
        except OSError as e:
            outputs.append(format_tool_result(cmd, "ERROR", f"Error: {e}"))

    if outputs:
        return "\n---\n".join(outputs)
    return format_tool_result(test_cmd, "SKIPPED", "No test commands available.")
