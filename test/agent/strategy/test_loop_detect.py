from __future__ import annotations

from mellea.agent.strategy.loop_detect import detect_loop


def test_no_loop() -> None:
    history = [
        ("search_code", ("foo",)),
        ("read_file", ("bar.py",)),
        ("search_code", ("baz",)),
    ]
    action = detect_loop(history)
    assert action is None


def test_detect_double_repeat() -> None:
    history = [("search_code", ("foo",)), ("search_code", ("foo",))]
    action = detect_loop(history)
    assert action == "nudge"


def test_detect_triple_repeat() -> None:
    history = [
        ("search_code", ("foo",)),
        ("search_code", ("foo",)),
        ("search_code", ("foo",)),
    ]
    action = detect_loop(history)
    assert action == "force_switch"
