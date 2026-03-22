from __future__ import annotations

from mellea.agent.runtime.memory import (
    CondensedState,
    WorkingMemory,
    condense_messages,
    materialize_messages,
    should_condense_messages,
)


def test_long_trajectories_condense_into_structured_summaries():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "Recover the service"},
        {"role": "assistant", "content": "I will inspect the failing path."},
        {"role": "user", "content": "[read_logs] upstream returned 502 for /items"},
        {"role": "assistant", "content": "The API cannot reach the database."},
        {"role": "user", "content": "[check_env] DB_HOST is unset in staging"},
        {"role": "assistant", "content": "This looks like deploy config drift."},
        {"role": "user", "content": "[list_deploys] latest deploy started at 14:32"},
    ]
    memory = WorkingMemory(
        summary="Investigated the outage and narrowed it to configuration drift after the latest deploy.",
        facts=(
            "Requests to /items fail with a 502 upstream error.",
            "DB_HOST is unset in staging.",
        ),
        hypotheses=("The latest deploy skipped environment sync.",),
        next_steps=("Restore DB_HOST and rerun health checks.",),
    )

    assert should_condense_messages(messages, max_messages=6, preserve_recent=2)

    condensed = condense_messages(messages, working_memory=memory, preserve_recent=2)
    rendered = materialize_messages(
        system_prompt="sys",
        goal="Recover the service",
        condensed_state=condensed,
    )

    assert condensed.omitted_messages == 4
    assert condensed.recent_messages == tuple(messages[-2:])
    assert rendered[:2] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "Recover the service"},
    ]
    assert rendered[2]["role"] == "user"
    assert "Condensed context" in rendered[2]["content"]
    assert "Summary:" in rendered[2]["content"]
    assert "- Requests to /items fail with a 502 upstream error." in rendered[2]["content"]
    assert "Hypotheses:" in rendered[2]["content"]
    assert "Next steps:" in rendered[2]["content"]
    assert rendered[-2:] == messages[-2:]


def test_working_memory_retains_key_facts_hypotheses_and_next_steps():
    memory = WorkingMemory(
        summary="The repo is partially migrated to a new runtime API.",
        facts=("Task 4 introduced runtime event primitives.", "The text loop is append-only."),
        hypotheses=("Memory can be injected as a synthetic user reminder.",),
        next_steps=("Keep recent turns verbatim.", "Avoid model-driven summarization."),
    )

    assert memory.to_dict() == {
        "summary": "The repo is partially migrated to a new runtime API.",
        "facts": [
            "Task 4 introduced runtime event primitives.",
            "The text loop is append-only.",
        ],
        "hypotheses": ["Memory can be injected as a synthetic user reminder."],
        "next_steps": [
            "Keep recent turns verbatim.",
            "Avoid model-driven summarization.",
        ],
    }

    reminder = memory.as_message(omitted_messages=5)

    assert reminder["role"] == "user"
    assert "5 earlier messages were condensed." in reminder["content"]
    assert "- Task 4 introduced runtime event primitives." in reminder["content"]
    assert "- Memory can be injected as a synthetic user reminder." in reminder["content"]
    assert "- Keep recent turns verbatim." in reminder["content"]


def test_materialize_messages_without_condensed_state_keeps_default_shape():
    assert materialize_messages(system_prompt="sys", goal="Solve it") == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "Solve it"},
    ]

    condensed = CondensedState(
        working_memory=WorkingMemory(summary="Already investigated."),
        recent_messages=(
            {"role": "assistant", "content": "I checked the logs."},
            {"role": "user", "content": "[grep] timeout in worker-3"},
        ),
        omitted_messages=2,
    )

    assert materialize_messages(
        system_prompt="sys",
        goal="Solve it",
        condensed_state=condensed,
    ) == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "Solve it"},
        {
            "role": "user",
            "content": "Condensed context:\n2 earlier messages were condensed.\nSummary:\nAlready investigated.",
        },
        {"role": "assistant", "content": "I checked the logs."},
        {"role": "user", "content": "[grep] timeout in worker-3"},
    ]
