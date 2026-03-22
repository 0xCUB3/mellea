"""Sandboxed bash command execution."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

_MAX_OUTPUT = 10_000
_DEFAULT_TIMEOUT = 30

# Commands that could damage the host or escape the repo.
_BLOCKED_PREFIXES = (
    "rm -rf /",
    "rm -rf /*",
    "mkfs",
    "dd if=",
    ":(){",
    "chmod -R 777 /",
    "sudo",
    "reboot",
    "shutdown",
    "kill -9 -1",
    "pkill",
)


def _truncate_output(output: str) -> str:
    """Cap command output while preserving both the start and the end."""
    if len(output) <= _MAX_OUTPUT:
        return output

    half = _MAX_OUTPUT // 2
    return (
        output[:half]
        + f"\n\n[... truncated {len(output) - _MAX_OUTPUT} chars ...]\n\n"
        + output[-half:]
    )


def _format_command_output(command: str, status: str, output: str) -> str:
    """Return command output in the same status-first shape as test execution."""
    body = output.rstrip() if output.strip() else "(no output)"
    return f"$ {command}\n{status}\n{body}"


def _run_shell_command(
    command: str,
    *,
    repo_root: str,
    timeout: int | None = None,
    allowed_dirs: list[str] | None = None,
) -> tuple[str, str]:
    """Run a shell command inside the repo root and return status plus output.

    The command runs in a subprocess with cwd set to *repo_root*.
    Output (stdout + stderr) is capped and the process is killed
    after *timeout* seconds.

    If *allowed_dirs* is provided, the command is rejected when it
    references absolute paths outside those directories.
    """
    timeout = timeout or int(
        os.environ.get("MCODE_BASH_TIMEOUT", str(_DEFAULT_TIMEOUT))
    )

    cmd_lower = command.strip().lower()
    for blocked in _BLOCKED_PREFIXES:
        if cmd_lower.startswith(blocked):
            return "BLOCKED", f"Error: command blocked for safety: {command[:80]}"

    if allowed_dirs is not None:
        import shlex

        try:
            tokens = shlex.split(command)
        except ValueError:
            tokens = command.split()
        for token in tokens:
            if token.startswith("/") and not any(
                token.startswith(d) for d in allowed_dirs
            ):
                return (
                    "ERROR",
                    f"Error: absolute path '{token}' is outside allowed "
                    f"directories. Use relative paths from the repo root.",
                )

    root = Path(repo_root).resolve()
    if not root.is_dir():
        return "ERROR", f"Error: repo root does not exist: {repo_root}"

    try:
        result = subprocess.run(
            ["bash", "-c", command],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "HOME": str(root), "LC_ALL": "C"},
        )
        output = _truncate_output(result.stdout + result.stderr)
    except subprocess.TimeoutExpired:
        return f"TIMEOUT after {timeout}s", ""
    except Exception as e:
        return "ERROR", f"Error: {type(e).__name__}: {e}"

    status = "PASSED" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
    return status, output


def run_command(
    command: str,
    *,
    repo_root: str,
    timeout: int | None = None,
    allowed_dirs: list[str] | None = None,
) -> str:
    """Run a shell command and return structured tool output."""
    status, output = _run_shell_command(
        command,
        repo_root=repo_root,
        timeout=timeout,
        allowed_dirs=allowed_dirs,
    )
    return _format_command_output(command, status, output)


def run_bash(
    command: str,
    *,
    repo_root: str,
    timeout: int | None = None,
    allowed_dirs: list[str] | None = None,
) -> str:
    """Run a bash command inside the repo root and return raw command output."""
    status, output = _run_shell_command(
        command,
        repo_root=repo_root,
        timeout=timeout,
        allowed_dirs=allowed_dirs,
    )

    if output.strip():
        return output
    if status == "PASSED":
        return "(no output, exit code 0)"
    if status.startswith("FAILED (exit "):
        exit_code = status.removeprefix("FAILED (exit ").removesuffix(")")
        return f"(no output, exit code {exit_code})"
    if status.startswith("TIMEOUT"):
        return f"Error: command timed out after {status.removeprefix('TIMEOUT after ')}"
    return output or f"Error: command failed with status {status}"
