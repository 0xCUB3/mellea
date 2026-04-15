from __future__ import annotations

from mellea.agent.strategy.phased import get_available_tools


def test_phase1_explore_only() -> None:
    all_tools = [
        "search_code",
        "read_file",
        "find_file",
        "list_dir",
        "edit",
        "final_answer",
    ]
    available = get_available_tools(all_tools, turn=1, budget=15)
    assert "search_code" in available
    assert "read_file" in available
    assert "edit" not in available
    assert "final_answer" not in available


def test_phase2_all_tools() -> None:
    all_tools = [
        "search_code",
        "read_file",
        "find_file",
        "list_dir",
        "edit",
        "final_answer",
    ]
    available = get_available_tools(all_tools, turn=8, budget=15)
    assert "edit" in available
    assert "search_code" in available


def test_phase3_commit_only() -> None:
    all_tools = [
        "search_code",
        "read_file",
        "find_file",
        "list_dir",
        "edit",
        "final_answer",
    ]
    available = get_available_tools(all_tools, turn=14, budget=15)
    assert "edit" in available
    assert "final_answer" in available
    assert "search_code" not in available


def test_small_budget() -> None:
    all_tools = ["search_code", "read_file", "edit", "final_answer"]
    available = get_available_tools(all_tools, turn=1, budget=3)
    assert "edit" not in available
    available = get_available_tools(all_tools, turn=2, budget=3)
    assert "edit" in available
    available = get_available_tools(all_tools, turn=3, budget=3)
    assert "edit" in available
    assert "search_code" not in available


def test_custom_phases() -> None:
    all_tools = ["search_code", "edit", "final_answer"]
    available = get_available_tools(
        all_tools, turn=1, budget=10, phases=(0.5, 1.0, 1.0)
    )
    assert "edit" not in available
