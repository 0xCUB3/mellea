"""A/B comparison of benchmark run results."""

from __future__ import annotations

import sqlite3
from pathlib import Path


def compare_runs(
    baseline_dir: str, candidate_dir: str, *, task_ids: list[str] | None = None
) -> dict:
    """Compare two benchmark run directories and return a diff report.

    Each directory should contain one or more .db files with task results.
    """
    baseline = _load_results_from_dir(baseline_dir, task_ids)
    candidate = _load_results_from_dir(candidate_dir, task_ids)

    all_tasks = sorted(set(baseline) | set(candidate))
    gained: list[str] = []
    lost: list[str] = []
    unchanged_pass: list[str] = []
    unchanged_fail: list[str] = []

    for task in all_tasks:
        b = baseline.get(task, False)
        c = candidate.get(task, False)
        if not b and c:
            gained.append(task)
        elif b and not c:
            lost.append(task)
        elif b and c:
            unchanged_pass.append(task)
        else:
            unchanged_fail.append(task)

    return {
        "baseline_total": len(baseline),
        "candidate_total": len(candidate),
        "baseline_passed": sum(baseline.values()),
        "candidate_passed": sum(candidate.values()),
        "gained": gained,
        "lost": lost,
        "unchanged_pass": unchanged_pass,
        "net_change": len(gained) - len(lost),
    }


def _load_results_from_dir(
    dir_path: str, task_ids: list[str] | None
) -> dict[str, bool]:
    """Load results from all .db files in a directory."""
    results: dict[str, bool] = {}
    db_dir = Path(dir_path)
    for db_file in sorted(db_dir.glob("*.db")):
        results.update(_load_results(str(db_file), task_ids))
    return results


def _load_results(db_path: str, task_ids: list[str] | None) -> dict[str, bool]:
    """Load task results from a single SQLite database."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        # Try common column names
        cols = [
            row[1] for row in conn.execute("PRAGMA table_info(task_results)").fetchall()
        ]
        if not cols:
            return {}

        passed_col = (
            "passed" if "passed" in cols else "resolved" if "resolved" in cols else None
        )
        if passed_col is None:
            return {}

        if task_ids:
            placeholders = ",".join("?" for _ in task_ids)
            rows = conn.execute(
                f"SELECT task_id, {passed_col} FROM task_results WHERE task_id IN ({placeholders})",
                task_ids,
            ).fetchall()
        else:
            rows = conn.execute(
                f"SELECT task_id, {passed_col} FROM task_results"
            ).fetchall()
        return {r["task_id"]: bool(r[passed_col]) for r in rows}
    except Exception:
        return {}
    finally:
        conn.close()


def format_comparison(report: dict) -> str:
    """Format a comparison report as readable text."""
    lines = [
        f"Baseline: {report['baseline_passed']}/{report['baseline_total']} passed",
        f"Candidate: {report['candidate_passed']}/{report['candidate_total']} passed",
        f"Net change: {report['net_change']:+d}",
    ]
    if report["gained"]:
        lines.append(f"\nGained ({len(report['gained'])}):")
        for t in report["gained"]:
            lines.append(f"  + {t}")
    if report["lost"]:
        lines.append(f"\nLost ({len(report['lost'])}):")
        for t in report["lost"]:
            lines.append(f"  - {t}")
    if report["unchanged_pass"]:
        lines.append(f"\nStill passing ({len(report['unchanged_pass'])}):")
        for t in report["unchanged_pass"]:
            lines.append(f"  = {t}")
    return "\n".join(lines)
