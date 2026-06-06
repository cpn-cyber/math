"""Quality audit for Problem 2 Step 12 outputs.

This module audits Appendix 2 surface features, weak labels, and candidate
section usage. It does not recompute features, labels, or run any model.
"""

from __future__ import annotations

from pathlib import Path
import logging
from typing import Any

import numpy as np
import pandas as pd

from modules.appendix2_pipeline import get_problem2_config, resolve_project_path
from modules.quality_label_builder import SURFACE_FEATURE_COLUMNS


LOGGER_NAME = "A_MAGE_R3.problem2.step12_quality_audit"

EXPECTED_Q_SOURCE = "Problem1 sealed evaluation system"
EXPECTED_LABEL_NOTE = "weak-supervised label, not official ground truth"
EXPECTED_CANDIDATES = [
    ("2-2", "results"),
    ("2-3", "results"),
    ("2-10", "abstract"),
]


def setup_step12_audit_logger(log_path: Path) -> logging.Logger:
    """Configure a dedicated Step 12B audit logger."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(LOGGER_NAME)
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False

    handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(handler)
    return logger


def _read_excel(path: Path, *, required: bool = True) -> pd.DataFrame:
    """Read an Excel file with a clear error for missing required inputs."""
    path = Path(path)
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required Step 12B input is missing: {path}")
        return pd.DataFrame()
    return pd.read_excel(path)


def _natural_key(value: Any) -> list[Any]:
    """Natural sort key for paper ids such as 2-1 and 2-10."""
    import re

    text = str(value)
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def _feature_stats(raw_features: pd.DataFrame) -> pd.DataFrame:
    """Compute requested descriptive statistics for each surface feature."""
    rows: list[dict[str, Any]] = []
    for feature in SURFACE_FEATURE_COLUMNS:
        series = pd.to_numeric(raw_features.get(feature), errors="coerce")
        rows.append(
            {
                "feature_name": feature,
                "missing_count": int(series.isna().sum()),
                "zero_count": int(series.fillna(np.nan).eq(0).sum()),
                "min": float(series.min()) if not series.dropna().empty else np.nan,
                "max": float(series.max()) if not series.dropna().empty else np.nan,
                "mean": float(series.mean()) if not series.dropna().empty else np.nan,
                "median": float(series.median()) if not series.dropna().empty else np.nan,
                "std": float(series.std(ddof=0)) if not series.dropna().empty else np.nan,
                "unique_count": int(series.dropna().nunique()),
            }
        )
    return pd.DataFrame(rows)


def _constant_checks(normalized_features: pd.DataFrame) -> pd.DataFrame:
    """Detect constant and near-constant feature columns."""
    rows: list[dict[str, Any]] = []
    for feature in SURFACE_FEATURE_COLUMNS:
        series = pd.to_numeric(normalized_features.get(feature), errors="coerce")
        valid = series.dropna()
        std = float(valid.std(ddof=0)) if not valid.empty else np.nan
        unique_count = int(valid.nunique())
        is_constant = unique_count <= 1
        is_near_constant = (not is_constant) and (unique_count <= 2 or (not np.isnan(std) and std <= 0.02))
        rows.append(
            {
                "feature_name": feature,
                "std_normalized": std,
                "unique_count": unique_count,
                "is_constant": is_constant,
                "is_near_constant": is_near_constant,
                "audit_note": "constant" if is_constant else ("near_constant" if is_near_constant else "ok"),
            }
        )
    return pd.DataFrame(rows)


def _extreme_value_checks(raw_features: pd.DataFrame, normalized_features: pd.DataFrame) -> pd.DataFrame:
    """Find out-of-range normalized values and extreme raw outliers."""
    rows: list[dict[str, Any]] = []
    for feature in SURFACE_FEATURE_COLUMNS:
        norm_series = pd.to_numeric(normalized_features.get(feature), errors="coerce")
        for index, value in norm_series.items():
            if pd.notna(value) and (float(value) < -1e-9 or float(value) > 1 + 1e-9):
                rows.append(
                    {
                        "paper_id": str(normalized_features.loc[index, "paper_id"]),
                        "filename": str(normalized_features.loc[index, "filename"]),
                        "feature_name": feature,
                        "issue_type": "normalized_out_of_range",
                        "value": float(value),
                        "threshold": "[0,1]",
                    }
                )

        raw_series = pd.to_numeric(raw_features.get(feature), errors="coerce")
        valid = raw_series.dropna()
        if len(valid) < 4:
            continue
        q1 = float(valid.quantile(0.25))
        q3 = float(valid.quantile(0.75))
        iqr = q3 - q1
        if iqr <= 0:
            continue
        lower = q1 - 3.0 * iqr
        upper = q3 + 3.0 * iqr
        for index, value in raw_series.items():
            if pd.notna(value) and (float(value) < lower or float(value) > upper):
                rows.append(
                    {
                        "paper_id": str(raw_features.loc[index, "paper_id"]),
                        "filename": str(raw_features.loc[index, "filename"]),
                        "feature_name": feature,
                        "issue_type": "raw_extreme_iqr3",
                        "value": float(value),
                        "threshold": f"<{lower:.6g} or >{upper:.6g}",
                    }
                )
    return pd.DataFrame(
        rows,
        columns=["paper_id", "filename", "feature_name", "issue_type", "value", "threshold"],
    )


def _q_label_summary(q_labels: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """Summarize Q_label distribution and return weak-distribution flag."""
    q = pd.to_numeric(q_labels.get("Q_label"), errors="coerce")
    valid = q.dropna()
    q_range = float(valid.max() - valid.min()) if not valid.empty else np.nan
    summary = pd.DataFrame(
        [
            {
                "valid_count": int(valid.count()),
                "min": float(valid.min()) if not valid.empty else np.nan,
                "max": float(valid.max()) if not valid.empty else np.nan,
                "mean": float(valid.mean()) if not valid.empty else np.nan,
                "std": float(valid.std(ddof=0)) if not valid.empty else np.nan,
                "range": q_range,
                "unique_count": int(valid.nunique()),
            }
        ]
    )
    weak_distribution = bool(valid.count() < 10 or valid.nunique() <= 2 or (pd.notna(q_range) and q_range < 5))
    return summary, weak_distribution


def _candidate_checks(candidate_usage: pd.DataFrame) -> pd.DataFrame:
    """Check candidate existence and 0.5 usage weight."""
    rows: list[dict[str, Any]] = []
    if candidate_usage.empty:
        for paper_id, section in EXPECTED_CANDIDATES:
            rows.append(
                {
                    "paper_id": paper_id,
                    "candidate_section": section,
                    "exists": False,
                    "all_weight_0_5": False,
                    "usage_rows": 0,
                    "audit_note": "candidate missing",
                }
            )
        return pd.DataFrame(rows)

    usage = candidate_usage.copy()
    usage["paper_id"] = usage["paper_id"].astype(str)
    usage["candidate_section"] = usage["candidate_section"].astype(str)
    usage["candidate_weight"] = pd.to_numeric(usage.get("candidate_weight"), errors="coerce")
    for paper_id, section in EXPECTED_CANDIDATES:
        subset = usage.loc[
            usage["paper_id"].eq(paper_id) & usage["candidate_section"].eq(section)
        ]
        exists = not subset.empty
        all_weight_0_5 = bool(exists and subset["candidate_weight"].eq(0.5).all())
        rows.append(
            {
                "paper_id": paper_id,
                "candidate_section": section,
                "exists": exists,
                "all_weight_0_5": all_weight_0_5,
                "usage_rows": int(len(subset)),
                "used_features": ",".join(sorted(subset["feature_name"].astype(str).unique())) if exists else "",
                "audit_note": "ok" if exists and all_weight_0_5 else "candidate missing or weight mismatch",
            }
        )
    return pd.DataFrame(rows)


def _paper_quality_flags(
    features_with_q: pd.DataFrame,
    candidate_usage: pd.DataFrame,
    anomalies: pd.DataFrame,
    weak_label_warning: bool,
) -> pd.DataFrame:
    """Generate paper-level feature quality flags."""
    candidate_ids = set(candidate_usage.get("paper_id", pd.Series(dtype=str)).astype(str)) if not candidate_usage.empty else set()
    anomaly_ids = set(anomalies.get("paper_id", pd.Series(dtype=str)).astype(str)) if not anomalies.empty else set()
    rows: list[dict[str, Any]] = []
    for _, row in features_with_q.iterrows():
        paper_id = str(row.get("paper_id"))
        feature_missing = [
            feature
            for feature in SURFACE_FEATURE_COLUMNS
            if feature not in features_with_q.columns or pd.isna(row.get(feature))
        ]
        q_valid = pd.notna(pd.to_numeric(pd.Series([row.get("Q_label")]), errors="coerce").iloc[0])
        reasons: list[str] = []
        if feature_missing:
            reasons.append("missing surface feature values")
        if not q_valid:
            reasons.append("invalid Q_label")
        if paper_id in candidate_ids:
            reasons.append("candidate section used")
        if paper_id in anomaly_ids:
            reasons.append("extreme feature value detected")
        if weak_label_warning:
            reasons.append("weak Q_label distribution")

        if feature_missing or not q_valid or paper_id in anomaly_ids:
            flag = "need_review"
        elif weak_label_warning:
            flag = "weak_label_warning"
        elif paper_id in candidate_ids:
            flag = "partial_candidate"
        else:
            flag = "normal"

        rows.append(
            {
                "paper_id": paper_id,
                "filename": str(row.get("filename")),
                "feature_quality_flag": flag,
                "surface_feature_missing_count": len(feature_missing),
                "Q_label_valid": q_valid,
                "candidate_used": paper_id in candidate_ids,
                "extreme_feature_count": int((anomalies.get("paper_id", pd.Series(dtype=str)).astype(str).eq(paper_id)).sum()) if not anomalies.empty else 0,
                "quality_notes": "; ".join(reasons) if reasons else "all checks passed",
            }
        )
    return pd.DataFrame(rows).sort_values("paper_id", key=lambda series: series.map(_natural_key)).reset_index(drop=True)


def _integrity_checks(
    raw_features: pd.DataFrame,
    normalized_features: pd.DataFrame,
    q_labels: pd.DataFrame,
    features_with_q: pd.DataFrame,
    candidate_check: pd.DataFrame,
    q_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Build pass/fail integrity check rows."""
    feature_cols = set(SURFACE_FEATURE_COLUMNS)
    normalized_numeric = normalized_features[list(feature_cols & set(normalized_features.columns))].apply(pd.to_numeric, errors="coerce")
    checks = [
        {
            "check_item": "features_with_q_has_10_papers",
            "passed": len(features_with_q) == 10,
            "detail": f"row_count={len(features_with_q)}",
        },
        {
            "check_item": "paper_id_no_duplicate_or_missing",
            "passed": features_with_q["paper_id"].notna().all() and not features_with_q["paper_id"].duplicated().any(),
            "detail": f"duplicates={features_with_q.loc[features_with_q['paper_id'].duplicated(), 'paper_id'].astype(str).tolist()}",
        },
        {
            "check_item": "all_19_surface_features_exist",
            "passed": feature_cols.issubset(set(features_with_q.columns)),
            "detail": f"missing_columns={sorted(feature_cols - set(features_with_q.columns))}",
        },
        {
            "check_item": "normalized_features_no_missing",
            "passed": int(normalized_numeric.isna().sum().sum()) == 0,
            "detail": f"missing_cells={int(normalized_numeric.isna().sum().sum())}",
        },
        {
            "check_item": "normalized_features_in_0_1",
            "passed": bool(((normalized_numeric >= -1e-9) & (normalized_numeric <= 1 + 1e-9)).all().all()),
            "detail": "checked all normalized feature cells",
        },
        {
            "check_item": "q_label_has_10_valid_scores",
            "passed": int(q_summary.loc[0, "valid_count"]) == 10,
            "detail": f"valid_count={int(q_summary.loc[0, 'valid_count'])}",
        },
        {
            "check_item": "q_source_expected",
            "passed": q_labels.get("Q_source", pd.Series(dtype=str)).astype(str).eq(EXPECTED_Q_SOURCE).all(),
            "detail": f"unique={q_labels.get('Q_source', pd.Series(dtype=str)).astype(str).unique().tolist()}",
        },
        {
            "check_item": "label_note_expected",
            "passed": q_labels.get("label_note", pd.Series(dtype=str)).astype(str).eq(EXPECTED_LABEL_NOTE).all(),
            "detail": f"unique={q_labels.get('label_note', pd.Series(dtype=str)).astype(str).unique().tolist()}",
        },
        {
            "check_item": "expected_candidates_present",
            "passed": bool(candidate_check["exists"].all()),
            "detail": candidate_check.to_dict(orient="records"),
        },
        {
            "check_item": "candidate_weight_is_0_5",
            "passed": bool(candidate_check["all_weight_0_5"].all()),
            "detail": candidate_check[["paper_id", "candidate_section", "all_weight_0_5"]].to_dict(orient="records"),
        },
    ]
    return pd.DataFrame(checks)


def run_step12_quality_audit(config_path: Path | None = None) -> dict[str, Any]:
    """Run Step 12B audit and save the workbook/log outputs."""
    problem2_config = get_problem2_config(config_path)
    tables_dir = resolve_project_path(problem2_config["problem2_output_tables_dir"])
    logs_dir = resolve_project_path(problem2_config["problem2_output_logs_dir"])
    tables_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    log_path = logs_dir / "appendix2_step12_quality_audit.log"
    logger = setup_step12_audit_logger(log_path)
    logger.info("Starting Problem 2 Step 12B quality audit")

    raw_path = tables_dir / "appendix2_surface_features_raw.xlsx"
    normalized_path = tables_dir / "appendix2_surface_features_normalized.xlsx"
    q_path = tables_dir / "appendix2_q_labels.xlsx"
    features_q_path = tables_dir / "appendix2_features_with_q.xlsx"
    candidate_path = tables_dir / "appendix2_candidate_usage_report.xlsx"
    refined_report_path = tables_dir / "appendix2_section_split_report_refined.xlsx"
    section_audit_path = tables_dir / "appendix2_section_quality_audit.xlsx"
    output_path = tables_dir / "appendix2_step12_quality_audit.xlsx"

    raw_features = _read_excel(raw_path)
    normalized_features = _read_excel(normalized_path)
    q_labels = _read_excel(q_path)
    features_with_q = _read_excel(features_q_path)
    candidate_usage = _read_excel(candidate_path)
    refined_report = _read_excel(refined_report_path)
    section_audit = _read_excel(section_audit_path)

    for frame in [raw_features, normalized_features, q_labels, features_with_q, candidate_usage, refined_report, section_audit]:
        if "paper_id" in frame.columns:
            frame["paper_id"] = frame["paper_id"].astype(str)

    stats = _feature_stats(raw_features)
    constants = _constant_checks(normalized_features)
    anomalies = _extreme_value_checks(raw_features, normalized_features)
    q_summary, weak_label_warning = _q_label_summary(q_labels)
    candidate_check = _candidate_checks(candidate_usage)
    paper_flags = _paper_quality_flags(features_with_q, candidate_usage, anomalies, weak_label_warning)
    integrity = _integrity_checks(raw_features, normalized_features, q_labels, features_with_q, candidate_check, q_summary)

    fatal_failed = integrity.loc[
        ~integrity["passed"]
        & integrity["check_item"].isin(
            [
                "features_with_q_has_10_papers",
                "paper_id_no_duplicate_or_missing",
                "all_19_surface_features_exist",
                "normalized_features_no_missing",
                "normalized_features_in_0_1",
                "q_label_has_10_valid_scores",
                "q_source_expected",
                "label_note_expected",
                "expected_candidates_present",
                "candidate_weight_is_0_5",
            ]
        )
    ]
    can_enter_step13 = fatal_failed.empty

    summary = pd.DataFrame(
        [
            {
                "paper_count": len(features_with_q),
                "surface_feature_count": len(SURFACE_FEATURE_COLUMNS),
                "normalized_missing_cells": int(
                    normalized_features[SURFACE_FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce").isna().sum().sum()
                ),
                "q_label_min": float(q_summary.loc[0, "min"]),
                "q_label_max": float(q_summary.loc[0, "max"]),
                "q_label_mean": float(q_summary.loc[0, "mean"]),
                "q_label_std": float(q_summary.loc[0, "std"]),
                "q_label_range": float(q_summary.loc[0, "range"]),
                "weak_label_warning": weak_label_warning,
                "constant_feature_count": int(constants["is_constant"].sum()),
                "near_constant_feature_count": int(constants["is_near_constant"].sum()),
                "extreme_anomaly_count": len(anomalies),
                "candidate_usage_rows": len(candidate_usage),
                "can_enter_step13": can_enter_step13,
            }
        ]
    )

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="audit_summary", index=False)
        integrity.to_excel(writer, sheet_name="integrity_checks", index=False)
        paper_flags.to_excel(writer, sheet_name="paper_quality_flags", index=False)
        stats.to_excel(writer, sheet_name="feature_stats", index=False)
        constants.to_excel(writer, sheet_name="constant_checks", index=False)
        anomalies.to_excel(writer, sheet_name="extreme_value_checks", index=False)
        q_summary.to_excel(writer, sheet_name="q_label_summary", index=False)
        candidate_check.to_excel(writer, sheet_name="candidate_check", index=False)

    logger.info("Audit summary: %s", summary.to_dict(orient="records")[0])
    logger.info("Integrity checks: %s", integrity[["check_item", "passed"]].to_dict(orient="records"))
    logger.info("Feature flags: %s", paper_flags[["paper_id", "feature_quality_flag", "quality_notes"]].to_dict(orient="records"))
    if not constants.loc[constants["is_constant"] | constants["is_near_constant"]].empty:
        logger.warning(
            "Constant or near-constant features: %s",
            constants.loc[constants["is_constant"] | constants["is_near_constant"], ["feature_name", "audit_note"]].to_dict(orient="records"),
        )
    if not anomalies.empty:
        logger.warning("Extreme feature anomalies: %s", anomalies.to_dict(orient="records"))
    logger.info("Step 12B audit saved: %s", output_path)
    logger.info("Can enter Step 13: %s", can_enter_step13)

    return {
        "summary": summary,
        "integrity": integrity,
        "paper_flags": paper_flags,
        "feature_stats": stats,
        "constant_checks": constants,
        "anomalies": anomalies,
        "q_summary": q_summary,
        "candidate_check": candidate_check,
        "can_enter_step13": can_enter_step13,
        "output_path": output_path,
        "log_path": log_path,
    }
