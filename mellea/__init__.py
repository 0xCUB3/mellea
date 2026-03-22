"""Mellea."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .backends import model_ids
    from .stdlib.components.genslot import generative
    from .stdlib.session import MelleaSession, start_session

_EXPORTS = {
    "MelleaSession": ("mellea.stdlib.session", "MelleaSession"),
    "generative": ("mellea.stdlib.components.genslot", "generative"),
    "model_ids": ("mellea.backends.model_ids", None),
    "start_session": ("mellea.stdlib.session", "start_session"),
}

__all__ = ["MelleaSession", "generative", "model_ids", "start_session"]


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    module = import_module(module_name)
    value = module if attr_name is None else getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
