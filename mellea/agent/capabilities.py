"""Compatibility capability contract primitives for mcode."""

from __future__ import annotations

from dataclasses import dataclass

_DEFAULT_FALLBACK_ROUTE = "bundled_tool_fallback"
_FAMILY_BY_TOOL = {
    "search_code": "repository_exploration",
    "read_file": "repository_exploration",
    "find_file": "repository_exploration",
    "list_dir": "repository_exploration",
    "edit": "editing",
    "run_tests": "verification",
    "probe_python": "verification",
    "bash": "shell",
    "final_answer": "submission",
}


@dataclass(frozen=True)
class CapabilityRoute:
    """Describes how a tool is routed by the orchestrator."""

    requested_family: str
    mode: str = _DEFAULT_FALLBACK_ROUTE


@dataclass(frozen=True)
class OrchestratorContract:
    """Small tool-family contract used by mcode's phased prompting."""

    tool_names: tuple[str, ...]
    default_verification_commands: tuple[str, ...] = ()
    fallback_route: str = _DEFAULT_FALLBACK_ROUTE

    @classmethod
    def from_tool_names(
        cls,
        tool_names: list[str] | tuple[str, ...],
        *,
        default_verification_commands: list[str] | tuple[str, ...] = (),
        **_: object,
    ) -> OrchestratorContract:
        """Build a contract from a tool-name list."""
        return cls(
            tool_names=tuple(tool_names),
            default_verification_commands=tuple(default_verification_commands),
        )

    @property
    def verification_required(self) -> bool:
        """Return whether the contract requires a verification step."""
        return bool(self.default_verification_commands)

    def route_for_tool(self, tool_name: str) -> CapabilityRoute:
        """Return the default family mapping for a tool."""
        return CapabilityRoute(
            requested_family=_FAMILY_BY_TOOL.get(tool_name, "other"),
            mode=self.fallback_route,
        )

    def snapshot(self) -> dict[str, object]:
        """Return a serializable contract snapshot."""
        return {
            "phases": ["diagnose", "edit", "verify", "submit"],
            "tool_names": list(self.tool_names),
            "family_by_tool": {
                name: self.route_for_tool(name).requested_family for name in self.tool_names
            },
            "adapter_families": [],
            "default_verification_commands": list(self.default_verification_commands),
            "verification_required": self.verification_required,
            "fallback_route": self.fallback_route,
        }
