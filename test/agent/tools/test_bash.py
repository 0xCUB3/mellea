"""Tests for the bash tool."""

from __future__ import annotations

import os
import tempfile

from mellea.agent.tools.bash import run_bash


def test_basic_command():
    with tempfile.TemporaryDirectory() as d:
        result = run_bash("echo hello", repo_root=d)
        assert "hello" in result


def test_cwd_is_repo_root():
    with tempfile.TemporaryDirectory() as d:
        result = run_bash("pwd", repo_root=d)
        assert os.path.realpath(d) in os.path.realpath(result.strip())


def test_timeout():
    with tempfile.TemporaryDirectory() as d:
        result = run_bash("sleep 60", repo_root=d, timeout=1)
        assert "timed out" in result


def test_blocked_command():
    with tempfile.TemporaryDirectory() as d:
        result = run_bash("rm -rf /", repo_root=d)
        assert "blocked" in result


def test_output_truncation():
    with tempfile.TemporaryDirectory() as d:
        result = run_bash("yes | head -20000", repo_root=d, timeout=5)
        assert "truncated" in result


def test_allowed_dirs():
    with tempfile.TemporaryDirectory() as d:
        result = run_bash("cat /etc/passwd", repo_root=d, allowed_dirs=["/tmp"])
        assert "outside allowed" in result
