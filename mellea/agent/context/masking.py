"""Observation masking context for saving token budget."""

from __future__ import annotations

from mellea.core.base import CBlock, Component
from mellea.stdlib.context import ChatContext


class MaskingChatContext(ChatContext):
    """ChatContext that masks old tool outputs to save context space."""

    def __init__(self, *, mask_after: int = 3, window_size: int | None = None):
        """Construct a masking chat context."""
        super().__init__(window_size=window_size)
        self._mask_after = mask_after

    def add(self, c: Component | CBlock) -> MaskingChatContext:
        """Add a component, preserving mask_after setting."""
        new_ctx = super().add(c)
        new_ctx.__class__ = MaskingChatContext
        new_ctx._mask_after = self._mask_after
        return new_ctx

    def view_for_generation(self) -> list[Component | CBlock] | None:
        """Return context items, masking old tool outputs."""
        items = super().view_for_generation()
        if items is None or len(items) <= self._mask_after * 2:
            return items
        return items
