"""Compatibility wrapper for Problem 3 Step 28 revision action optimization."""

from __future__ import annotations

from src.revision_action_optimizer import (  # noqa: F401
    build_revision_action_library,
    optimize_revision_actions,
    run_revision_action_optimization,
)

__all__ = [
    "build_revision_action_library",
    "optimize_revision_actions",
    "run_revision_action_optimization",
]
