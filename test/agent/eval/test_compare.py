from __future__ import annotations

import sqlite3
from pathlib import Path

from mellea.eval.compare import compare_runs, format_comparison


def _make_db(path: Path, results: dict[str, bool]) -> None:
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE task_results (task_id TEXT, passed INTEGER)")
    for task_id, passed in results.items():
        conn.execute("INSERT INTO task_results VALUES (?, ?)", (task_id, int(passed)))
    conn.commit()
    conn.close()


def test_compare_runs_basic(tmp_path: Path) -> None:
    baseline_dir = tmp_path / "baseline"
    baseline_dir.mkdir()
    _make_db(
        baseline_dir / "shard-0.db", {"task_a": True, "task_b": False, "task_c": True}
    )

    candidate_dir = tmp_path / "candidate"
    candidate_dir.mkdir()
    _make_db(
        candidate_dir / "shard-0.db", {"task_a": True, "task_b": True, "task_c": False}
    )

    report = compare_runs(str(baseline_dir), str(candidate_dir))
    assert report["baseline_passed"] == 2
    assert report["candidate_passed"] == 2
    assert "task_b" in report["gained"]
    assert "task_c" in report["lost"]
    assert report["net_change"] == 0


def test_compare_runs_with_filter(tmp_path: Path) -> None:
    baseline_dir = tmp_path / "baseline"
    baseline_dir.mkdir()
    _make_db(baseline_dir / "shard-0.db", {"task_a": True, "task_b": False})

    candidate_dir = tmp_path / "candidate"
    candidate_dir.mkdir()
    _make_db(candidate_dir / "shard-0.db", {"task_a": True, "task_b": True})

    report = compare_runs(str(baseline_dir), str(candidate_dir), task_ids=["task_a"])
    assert report["baseline_total"] == 1
    assert report["candidate_total"] == 1


def test_format_comparison() -> None:
    report = {
        "baseline_total": 10,
        "candidate_total": 10,
        "baseline_passed": 3,
        "candidate_passed": 4,
        "gained": ["task_x"],
        "lost": [],
        "unchanged_pass": ["task_a", "task_b", "task_c"],
        "net_change": 1,
    }
    text = format_comparison(report)
    assert "+1" in text
    assert "task_x" in text
