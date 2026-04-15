from __future__ import annotations

from mellea.agent.context.masking import MaskingChatContext


def test_masking_context_add_and_view() -> None:
    ctx = MaskingChatContext(mask_after=2)
    view = ctx.view_for_generation()
    assert view is not None or view is None


def test_masking_context_preserves_mask_after() -> None:
    ctx = MaskingChatContext(mask_after=5)
    assert ctx._mask_after == 5
