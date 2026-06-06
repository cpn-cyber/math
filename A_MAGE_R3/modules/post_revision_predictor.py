"""Compatibility wrapper for Problem 3 Step 29 post-revision prediction."""

from __future__ import annotations

from src.quality_prediction_after_revision import (  # noqa: F401
    predict_after_revision as simulate_post_revision_features,
    predict_after_revision,
    run_quality_prediction_after_revision,
)

predict_post_revision_quality = run_quality_prediction_after_revision

__all__ = [
    "simulate_post_revision_features",
    "predict_after_revision",
    "predict_post_revision_quality",
    "run_quality_prediction_after_revision",
]
