"""Compatibility wrapper for Problem 3 Step 30 robustness analysis."""

from __future__ import annotations

from src.third_question_robustness import (  # noqa: F401
    build_parameter_grid,
    run_robustness_simulation,
    run_third_question_robustness,
)

run_problem3_bootstrap = run_third_question_robustness
run_parameter_sensitivity = run_third_question_robustness

__all__ = [
    "build_parameter_grid",
    "run_robustness_simulation",
    "run_third_question_robustness",
    "run_problem3_bootstrap",
    "run_parameter_sensitivity",
]
