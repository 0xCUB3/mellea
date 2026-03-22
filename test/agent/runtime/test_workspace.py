from __future__ import annotations

from pathlib import Path

from mellea.agent.runtime import (
    EventLog,
    SafetyPolicy,
    SessionMetadata,
    SummaryEvent,
    TerminationEvent,
    TerminationReason,
    ToolCallEvent,
    ToolResultEvent,
    Workspace,
)


def test_workspace_describes_execution_context():
    workspace = Workspace(
        cwd="/repo/project",
        safety_policy=SafetyPolicy(
            mode="workspace-write",
            network_access=False,
            writable_roots=("/repo/project",),
        ),
        session=SessionMetadata(
            session_id="sess-123",
            executor="codex",
            branch="task3-coding-tools",
            metadata={"loop_budget": 8},
        ),
        metadata={"task_id": "task-4"},
    )

    assert workspace.resolve("src/../README.md") == Path("/repo/project/README.md")
    assert workspace.contains("/repo/project/src/module.py")
    assert not workspace.contains("/tmp/other.txt")
    assert workspace.describe() == {
        "cwd": "/repo/project",
        "safety_policy": {
            "mode": "workspace-write",
            "network_access": False,
            "writable_roots": ["/repo/project"],
        },
        "session": {
            "session_id": "sess-123",
            "executor": "codex",
            "branch": "task3-coding-tools",
            "metadata": {"loop_budget": 8},
        },
        "metadata": {"task_id": "task-4"},
    }


def test_tool_actions_emit_structured_events():
    event_log = EventLog()
    tool_call = ToolCallEvent(
        tool_name="bash",
        arguments={"command": "pwd"},
        call_id="call-1",
    )
    tool_result = ToolResultEvent(
        tool_name="bash",
        call_id="call-1",
        status="completed",
        output="/repo/project\n",
        duration_ms=12,
    )

    event_log.emit(tool_call)
    event_log.emit(tool_result)

    assert list(event_log) == [tool_call, tool_result]
    assert event_log.to_dicts() == [
        {
            "kind": "tool_call",
            "tool_name": "bash",
            "arguments": {"command": "pwd"},
            "call_id": "call-1",
        },
        {
            "kind": "tool_result",
            "tool_name": "bash",
            "call_id": "call-1",
            "status": "completed",
            "output": "/repo/project\n",
            "duration_ms": 12,
        },
    ]


def test_event_log_works_without_benchmark_stack():
    workspace = Workspace(
        cwd="/repo/project",
        safety_policy=SafetyPolicy(mode="read-only"),
        session=SessionMetadata(session_id="sess-456"),
    )
    event_log = EventLog(workspace=workspace)

    event_log.emit(SummaryEvent(message="Started agent loop", metadata={"turn": 1}))
    event_log.emit(
        TerminationEvent(
            reason=TerminationReason.FINAL_ANSWER,
            detail="Model returned a final answer",
        )
    )

    assert event_log.final_reason() is TerminationReason.FINAL_ANSWER
    assert event_log.snapshot() == {
        "workspace": {
            "cwd": "/repo/project",
            "safety_policy": {
                "mode": "read-only",
                "network_access": None,
                "writable_roots": [],
            },
            "session": {
                "session_id": "sess-456",
                "executor": None,
                "branch": None,
                "metadata": {},
            },
            "metadata": {},
        },
        "events": [
            {
                "kind": "summary",
                "message": "Started agent loop",
                "metadata": {"turn": 1},
            },
            {
                "kind": "termination",
                "reason": "final_answer",
                "detail": "Model returned a final answer",
            },
        ],
    }
