"""Robust preprocessing utilities for Problem 2 Step 14."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from modules.deep_quality_features import DEEP_FEATURE_COLUMNS
from modules.quality_label_builder import SURFACE_FEATURE_COLUMNS


NON_FEATURE_COLUMNS = {
    "paper_id",
    "filename",
    "Q_label",
    "Q_source",
    "label_note",
    "review_focus",
    "candidate_used",
    "candidate_sections",
    "step12b_flag",
}


def select_problem2_numeric_features(features: pd.DataFrame) -> list[str]:
    """Select the 19 surface and 6 deep numeric feature columns."""
    expected = list(SURFACE_FEATURE_COLUMNS) + list(DEEP_FEATURE_COLUMNS)
    return [column for column in expected if column in features.columns]


def build_feature_variance_filter(
    feature_matrix: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Compute variance diagnostics and model-use suggestions."""
    rows: list[dict[str, Any]] = []
    n_rows = len(feature_matrix)
    for feature in feature_columns:
        series = pd.to_numeric(feature_matrix[feature], errors="coerce")
        valid = series.dropna()
        missing_count = int(series.isna().sum())
        zero_count = int(series.fillna(np.nan).eq(0).sum())
        unique_count = int(valid.nunique())
        variance = float(valid.var(ddof=0)) if not valid.empty else np.nan
        std = float(valid.std(ddof=0)) if not valid.empty else np.nan
        min_value = float(valid.min()) if not valid.empty else np.nan
        max_value = float(valid.max()) if not valid.empty else np.nan

        value_counts = valid.value_counts(dropna=True)
        min_class_count = int(value_counts.min()) if not value_counts.empty else 0
        min_class_ratio = min_class_count / n_rows if n_rows else 0

        if unique_count <= 1 or variance == 0:
            variance_flag = "constant_feature"
            use_in_model = "False"
            recommendation = "exclude from main correlation ranking and later modeling"
        elif feature == "appendix_code_presence":
            variance_flag = "low_variance_feature"
            use_in_model = "Caution"
            recommendation = "keep in raw matrix, use cautiously because it is near-constant"
        elif unique_count <= 2 and min_class_ratio <= 0.2:
            variance_flag = "low_variance_feature"
            use_in_model = "Caution"
            recommendation = "small-sample binary/near-constant feature, interpret cautiously"
        elif pd.notna(std) and std <= 0.02:
            variance_flag = "low_variance_feature"
            use_in_model = "Caution"
            recommendation = "very low spread after extraction, interpret cautiously"
        else:
            variance_flag = "normal"
            use_in_model = "True"
            recommendation = "usable"

        if feature == "reference_norm_rate" and variance_flag == "constant_feature":
            recommendation = "confirmed constant feature; exclude from main correlation ranking and PLS"

        rows.append(
            {
                "feature_name": feature,
                "missing_count": missing_count,
                "zero_count": zero_count,
                "unique_count": unique_count,
                "variance": variance,
                "std": std,
                "min": min_value,
                "max": max_value,
                "variance_flag": variance_flag,
                "use_in_model": use_in_model,
                "recommendation": recommendation,
            }
        )
    return pd.DataFrame(rows)


def robust_scale_features(
    features: pd.DataFrame,
    feature_columns: list[str] | None = None,
    epsilon: float = 1e-6,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Robust-scale features with median and MAD.

    z_ij = (x_ij - median_j) / (MAD_j + epsilon)
    """
    feature_columns = feature_columns or select_problem2_numeric_features(features)
    scaled = features.copy()
    rows: list[dict[str, Any]] = []
    for feature in feature_columns:
        series = pd.to_numeric(features[feature], errors="coerce")
        median = float(series.median()) if not series.dropna().empty else np.nan
        mad = float((series - median).abs().median()) if pd.notna(median) else np.nan
        if pd.isna(mad):
            scaled[feature] = np.nan
            scale_flag = "missing"
        elif mad == 0:
            scaled[feature] = (series - median) / (mad + epsilon)
            scale_flag = "mad_zero"
        else:
            scaled[feature] = (series - median) / (mad + epsilon)
            scale_flag = "ok"
        rows.append(
            {
                "feature_name": feature,
                "median": median,
                "mad": mad,
                "epsilon": epsilon,
                "scale_flag": scale_flag,
            }
        )
    return scaled, pd.DataFrame(rows)


def merge_problem2_feature_matrix(
    features_with_q: pd.DataFrame,
    surface_raw: pd.DataFrame,
    deep_features: pd.DataFrame,
    step12_flags: pd.DataFrame,
    candidate_usage: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Merge surface features, deep features, Q_label, and audit markers."""
    features_with_q = features_with_q.copy()
    surface_raw = surface_raw.copy()
    deep_features = deep_features.copy()
    step12_flags = step12_flags.copy()
    candidate_usage = candidate_usage.copy() if candidate_usage is not None else pd.DataFrame()

    for frame in [features_with_q, surface_raw, deep_features, step12_flags, candidate_usage]:
        if "paper_id" in frame.columns:
            frame["paper_id"] = frame["paper_id"].astype(str)

    base_cols = ["paper_id", "filename", "Q_label", "Q_source", "label_note"]
    base = features_with_q[[column for column in base_cols if column in features_with_q.columns]].drop_duplicates("paper_id")
    surface = surface_raw[["paper_id", *[column for column in SURFACE_FEATURE_COLUMNS if column in surface_raw.columns]]]
    deep_cols = [
        "paper_id",
        *[column for column in DEEP_FEATURE_COLUMNS if column in deep_features.columns],
        "review_focus",
        "step12b_flag",
        "candidate_sections",
    ]
    deep = deep_features[[column for column in deep_cols if column in deep_features.columns]]

    flag_cols = ["paper_id", "feature_quality_flag"]
    flags = step12_flags[[column for column in flag_cols if column in step12_flags.columns]]
    flags = flags.rename(columns={"feature_quality_flag": "step12b_feature_quality_flag"})

    if not candidate_usage.empty and {"paper_id", "candidate_section"}.issubset(candidate_usage.columns):
        candidate = (
            candidate_usage.groupby("paper_id")["candidate_section"]
            .apply(lambda values: ",".join(sorted(set(map(str, values)))))
            .reset_index()
        )
        candidate["candidate_used"] = True
        candidate = candidate.rename(columns={"candidate_section": "candidate_sections_from_step12"})
    else:
        candidate = pd.DataFrame(columns=["paper_id", "candidate_sections_from_step12", "candidate_used"])

    merged = base.merge(surface, on="paper_id", how="left")
    merged = merged.merge(deep, on="paper_id", how="left")
    merged = merged.merge(flags, on="paper_id", how="left")
    merged = merged.merge(candidate, on="paper_id", how="left")
    merged["candidate_used"] = merged["candidate_used"].fillna(False).astype(bool)
    if "review_focus" in merged.columns:
        merged["review_focus"] = merged["review_focus"].fillna(False).astype(bool)
    else:
        merged["review_focus"] = False
    if "candidate_sections" in merged.columns:
        merged["candidate_sections"] = merged["candidate_sections"].fillna("")
    if "candidate_sections_from_step12" in merged.columns:
        merged["candidate_sections_from_step12"] = merged["candidate_sections_from_step12"].fillna("")
    return merged
