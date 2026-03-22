"""Workspace metadata primitives for agent runtimes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _normalize_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve(strict=False)


def _copy_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    return dict(metadata or {})


@dataclass(frozen=True)
class SafetyPolicy:
    """Describes the safety envelope for a runtime workspace."""

    mode: str
    network_access: bool | None = None
    writable_roots: tuple[str | Path, ...] = ()

    def __post_init__(self) -> None:
        """Normalize writable roots to absolute string paths."""
        object.__setattr__(
            self,
            "writable_roots",
            tuple(str(_normalize_path(path)) for path in self.writable_roots),
        )

    def describe(self) -> dict[str, Any]:
        """Return a serializable description of the safety policy."""
        return {
            "mode": self.mode,
            "network_access": self.network_access,
            "writable_roots": list(self.writable_roots),
        }


@dataclass(frozen=True)
class SessionMetadata:
    """Execution metadata that travels with a runtime workspace."""

    session_id: str | None = None
    executor: str | None = None
    branch: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def describe(self) -> dict[str, Any]:
        """Return a serializable description of the execution session."""
        return {
            "session_id": self.session_id,
            "executor": self.executor,
            "branch": self.branch,
            "metadata": _copy_metadata(self.metadata),
        }


@dataclass(frozen=True)
class Workspace:
    """Normalized workspace context for agent execution."""

    cwd: str | Path
    safety_policy: SafetyPolicy
    session: SessionMetadata
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize the workspace cwd to an absolute path."""
        object.__setattr__(self, "cwd", _normalize_path(self.cwd))

    def resolve(self, path: str | Path) -> Path:
        """Resolve a path relative to the workspace cwd."""
        candidate = Path(path)
        if candidate.is_absolute():
            return candidate.resolve(strict=False)
        return (self.cwd / candidate).resolve(strict=False)

    def contains(self, path: str | Path) -> bool:
        """Return whether a path stays within the workspace cwd."""
        candidate = self.resolve(path)
        try:
            candidate.relative_to(self.cwd)
        except ValueError:
            return False
        return True

    def describe(self) -> dict[str, Any]:
        """Return a serializable description of the workspace."""
        return {
            "cwd": str(self.cwd),
            "safety_policy": self.safety_policy.describe(),
            "session": self.session.describe(),
            "metadata": _copy_metadata(self.metadata),
        }
