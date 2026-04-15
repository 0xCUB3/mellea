from __future__ import annotations

from mellea.agent.capabilities import OrchestratorContract
from mellea.agent.localization import format_candidate_files
from mellea.agent.strategy import ToolInvocation, ToolPhaseState, get_available_tools


def test_mcode_capability_contract_surface():
    contract = OrchestratorContract.from_tool_names(
        ["search_code", "edit", "run_tests", "final_answer"],
        default_verification_commands=["pytest -q"],
    )

    assert contract.verification_required is True
    assert contract.route_for_tool("run_tests").requested_family == "verification"
    assert contract.snapshot()["default_verification_commands"] == ["pytest -q"]


def test_mcode_strategy_surface_accepts_state():
    phase_state = ToolPhaseState(
        turn=2,
        budget=10,
        invocations=(ToolInvocation("search_code"), ToolInvocation("read_file")),
    )

    tools = get_available_tools(
        ["search_code", "read_file", "edit", "run_tests", "final_answer"],
        turn=phase_state.turn,
        budget=phase_state.budget,
        state=phase_state,
    )

    assert "search_code" in tools
    assert phase_state.has_edit is False
    assert phase_state.repeated_tool_streak == 1


def test_mcode_localization_surface_returns_text(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "parser.py").write_text("def parse_error_report():\n    return None\n")
    (tmp_path / "pkg" / "models.py").write_text("class Model:\n    pass\n")

    text = format_candidate_files(str(tmp_path), "parse error report", top_n=2)

    assert text.startswith("Likely files to inspect first:")
    assert "pkg/parser.py" in text
