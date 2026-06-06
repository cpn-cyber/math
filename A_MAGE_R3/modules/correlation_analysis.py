"""Correlation analysis and visualisation for Problem 2 Step 14."""

from __future__ import annotations

from pathlib import Path
import logging
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:  # scipy is available in the bundled runtime, but keep a safe fallback.
    from scipy.stats import pearsonr, spearmanr
except ImportError:  # pragma: no cover
    pearsonr = None
    spearmanr = None

from modules.appendix2_pipeline import get_problem2_config, resolve_project_path
from modules.deep_quality_features import DEEP_FEATURE_COLUMNS
from modules.quality_label_builder import SURFACE_FEATURE_COLUMNS
from modules.robust_preprocessing import (
    build_feature_variance_filter,
    merge_problem2_feature_matrix,
    robust_scale_features,
    select_problem2_numeric_features,
)


LOGGER_NAME = "A_MAGE_R3.problem2.robust_correlation"
NEGATIVE_FEATURES = {"stacking_penalty"}


def setup_correlation_logger(log_path: Path) -> logging.Logger:
    """Configure Step 14 logger."""
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


def _natural_sort_key(value: Any) -> list[Any]:
    """Natural sort key for paper IDs."""
    import re

    text = str(value)
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def _load_step12_flags(audit_path: Path) -> pd.DataFrame:
    """Load Step 12B paper flags."""
    if not Path(audit_path).exists():
        return pd.DataFrame(columns=["paper_id", "feature_quality_flag"])
    return pd.read_excel(audit_path, sheet_name="paper_quality_flags")


def _correlation_pair(feature: pd.Series, target: pd.Series) -> tuple[float, float, float, float]:
    """Compute Spearman and Pearson correlations with p-values."""
    data = pd.DataFrame({"x": pd.to_numeric(feature, errors="coerce"), "y": pd.to_numeric(target, errors="coerce")}).dropna()
    if len(data) < 3 or data["x"].nunique() <= 1 or data["y"].nunique() <= 1:
        return np.nan, np.nan, np.nan, np.nan
    if spearmanr is None or pearsonr is None:
        x_rank = data["x"].rank(method="average").to_numpy(dtype=float)
        y_rank = data["y"].rank(method="average").to_numpy(dtype=float)
        x = data["x"].to_numpy(dtype=float)
        y = data["y"].to_numpy(dtype=float)
        spearman_corr = float(np.corrcoef(x_rank, y_rank)[0, 1])
        pearson_corr = float(np.corrcoef(x, y)[0, 1])
        return spearman_corr, np.nan, pearson_corr, np.nan
    spearman_result = spearmanr(data["x"], data["y"])
    pearson_result = pearsonr(data["x"], data["y"])
    return float(spearman_result.statistic), float(spearman_result.pvalue), float(pearson_result.statistic), float(pearson_result.pvalue)


def _feature_group(feature: str) -> str:
    """Return surface/deep group."""
    if feature in DEEP_FEATURE_COLUMNS:
        return "deep"
    return "surface"


def _direction(corr: float) -> str:
    """Classify correlation direction."""
    if pd.isna(corr) or abs(corr) < 0.2:
        return "weak"
    return "positive" if corr > 0 else "negative"


def _interpretation(feature: str, corr: float, variance_flag: str, use_in_model: str) -> str:
    """Build a concise interpretation."""
    if use_in_model == "False":
        return "低方差或常数特征，不作为主要解释依据"
    if feature == "stacking_penalty":
        if pd.isna(corr):
            return "堆砌惩罚为负向指标，但该样本下相关性无法稳定估计"
        if corr < 0:
            return "堆砌惩罚越高，弱监督质量分越低"
        return "堆砌惩罚越高但质量分未下降，需谨慎解释"
    if pd.isna(corr) or abs(corr) < 0.2:
        return "与弱监督质量分相关较弱，小样本下仅作参考"
    label = {
        "task_coverage": "任务覆盖率",
        "data_credibility": "数据可信度",
        "method_fit": "方法匹配度",
        "formula_explanation": "公式解释度",
        "result_closure": "结果闭环度",
    }.get(feature, feature)
    if corr > 0:
        return f"{label}越高，弱监督质量分越高"
    return f"{label}越高，弱监督质量分越低，需结合特征含义解释"


def calculate_feature_correlations(
    feature_matrix: pd.DataFrame,
    feature_columns: list[str],
    variance_filter: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate feature-Q correlations."""
    variance_lookup = variance_filter.set_index("feature_name").to_dict(orient="index")
    rows: list[dict[str, Any]] = []
    target = feature_matrix["Q_label"]
    for feature in feature_columns:
        info = variance_lookup.get(feature, {})
        spearman_corr, spearman_p, pearson_corr, pearson_p = _correlation_pair(feature_matrix[feature], target)
        variance_flag = str(info.get("variance_flag", "normal"))
        use_in_model = str(info.get("use_in_model", "True"))
        rows.append(
            {
                "feature_name": feature,
                "feature_group": _feature_group(feature),
                "is_negative_feature": feature in NEGATIVE_FEATURES,
                "use_in_model": use_in_model,
                "spearman_corr": spearman_corr,
                "spearman_abs": abs(spearman_corr) if pd.notna(spearman_corr) else np.nan,
                "pearson_corr": pearson_corr,
                "p_value": spearman_p,
                "pearson_p_value": pearson_p,
                "variance_flag": variance_flag,
                "feature_direction": _direction(spearman_corr),
                "interpretation": _interpretation(feature, spearman_corr, variance_flag, use_in_model),
            }
        )
    result = pd.DataFrame(rows)
    return result.sort_values(["use_in_model", "spearman_abs"], ascending=[False, False]).reset_index(drop=True)


def _save_heatmap(
    scaled_matrix: pd.DataFrame,
    correlation: pd.DataFrame,
    chart_path: Path,
) -> None:
    """Save 10-paper by top-feature robust-scaled heatmap."""
    valid = correlation.loc[correlation["use_in_model"].ne("False")].dropna(subset=["spearman_abs"])
    top_features = valid.sort_values("spearman_abs", ascending=False)["feature_name"].head(12).tolist()
    if not top_features:
        top_features = [column for column in scaled_matrix.columns if column not in {"paper_id", "filename", "Q_label"}][:12]
    heat = scaled_matrix[top_features].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy()
    heat = np.clip(heat, -5, 5)
    labels = scaled_matrix["paper_id"].astype(str).tolist()

    chart_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(9, len(top_features) * 0.7), 5.5), dpi=180)
    im = ax.imshow(heat, aspect="auto", cmap="coolwarm", vmin=-3, vmax=3)
    ax.set_xticks(range(len(top_features)))
    ax.set_xticklabels(top_features, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title("Appendix 2 Robust-Scaled Feature Heatmap")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="robust z")
    fig.tight_layout()
    fig.savefig(chart_path)
    plt.close(fig)


def _save_correlation_bar(correlation: pd.DataFrame, chart_path: Path) -> None:
    """Save Top Spearman absolute correlation bar chart."""
    valid = correlation.loc[correlation["use_in_model"].ne("False")].dropna(subset=["spearman_abs"])
    top = valid.sort_values("spearman_abs", ascending=False).head(12).iloc[::-1]
    chart_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 6), dpi=180)
    colors = ["#b64b4b" if row["spearman_corr"] < 0 else "#3569a8" for _, row in top.iterrows()]
    ax.barh(top["feature_name"], top["spearman_abs"], color=colors)
    ax.set_xlabel("|Spearman correlation|")
    ax.set_title("Top Appendix 2 Feature-Q Correlations")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(chart_path)
    plt.close(fig)


def _save_q_distribution(feature_matrix: pd.DataFrame, chart_path: Path) -> None:
    """Save Q_label distribution chart."""
    q = pd.to_numeric(feature_matrix["Q_label"], errors="coerce").dropna()
    chart_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=180)
    ax.hist(q, bins=min(6, max(3, len(q) // 2)), color="#4f7cac", edgecolor="white", alpha=0.85)
    ax.axvline(q.mean(), color="#b64b4b", linestyle="--", linewidth=1.5, label=f"mean={q.mean():.2f}")
    ax.set_xlabel("Q_label")
    ax.set_ylabel("Paper count")
    ax.set_title("Appendix 2 Weak Label Distribution")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(chart_path)
    plt.close(fig)


def run_step14_robust_correlation(config_path: Path | None = None) -> dict[str, Any]:
    """Run Step 14 robust scaling and correlation analysis."""
    problem2_config = get_problem2_config(config_path)
    tables_dir = resolve_project_path(problem2_config["problem2_output_tables_dir"])
    charts_dir = resolve_project_path(problem2_config["problem2_output_charts_dir"])
    logs_dir = resolve_project_path(problem2_config["problem2_output_logs_dir"])
    tables_dir.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    log_path = logs_dir / "robust_correlation.log"
    logger = setup_correlation_logger(log_path)
    epsilon = float(problem2_config.get("robust_scale_epsilon", 1e-6))
    logger.info("Starting Step 14 robust preprocessing and correlation analysis; epsilon=%s", epsilon)
    if spearmanr is None or pearsonr is None:
        logger.warning("scipy is unavailable; Spearman/Pearson coefficients are computed by rank/numpy fallback and p_value columns are NaN.")

    features_with_q = pd.read_excel(tables_dir / "appendix2_features_with_q.xlsx")
    surface_raw = pd.read_excel(tables_dir / "appendix2_surface_features_raw.xlsx")
    surface_normalized = pd.read_excel(tables_dir / "appendix2_surface_features_normalized.xlsx")
    deep_features = pd.read_excel(tables_dir / "appendix2_deep_quality_features_auto.xlsx")
    step12_flags = _load_step12_flags(tables_dir / "appendix2_step12_quality_audit.xlsx")
    candidate_usage = pd.read_excel(tables_dir / "appendix2_candidate_usage_report.xlsx")

    feature_matrix = merge_problem2_feature_matrix(
        features_with_q=features_with_q,
        surface_raw=surface_raw,
        deep_features=deep_features,
        step12_flags=step12_flags,
        candidate_usage=candidate_usage,
    )
    feature_matrix = feature_matrix.sort_values("paper_id", key=lambda series: series.map(_natural_sort_key)).reset_index(drop=True)
    feature_columns = select_problem2_numeric_features(feature_matrix)
    variance_filter = build_feature_variance_filter(feature_matrix, feature_columns)
    scaled_matrix, scale_diagnostics = robust_scale_features(feature_matrix, feature_columns, epsilon=epsilon)

    for _, row in scale_diagnostics.loc[scale_diagnostics["scale_flag"].eq("mad_zero")].iterrows():
        logger.warning("MAD=0 for feature %s; scaled with epsilon and flagged for low variance review.", row["feature_name"])

    correlation = calculate_feature_correlations(feature_matrix, feature_columns, variance_filter)
    stacking_row = correlation.loc[correlation["feature_name"].eq("stacking_penalty")]
    if not stacking_row.empty and pd.notna(stacking_row.iloc[0]["spearman_corr"]) and float(stacking_row.iloc[0]["spearman_corr"]) > 0:
        logger.warning("stacking_penalty is positively correlated with Q_label in this sample; interpret cautiously.")

    raw_output_path = tables_dir / "appendix2_feature_matrix_raw.xlsx"
    scaled_output_path = tables_dir / "appendix2_feature_matrix_robust_scaled.xlsx"
    correlation_output_path = tables_dir / "appendix2_correlation_analysis.xlsx"
    variance_output_path = tables_dir / "appendix2_feature_variance_filter.xlsx"
    heatmap_path = charts_dir / "appendix2_feature_heatmap.png"
    bar_path = charts_dir / "appendix2_correlation_bar.png"
    q_dist_path = charts_dir / "appendix2_q_distribution.png"

    feature_matrix.to_excel(raw_output_path, index=False)
    scaled_matrix.to_excel(scaled_output_path, index=False)
    variance_filter.to_excel(variance_output_path, index=False)
    with pd.ExcelWriter(correlation_output_path, engine="openpyxl") as writer:
        correlation.to_excel(writer, sheet_name="correlation", index=False)
        scale_diagnostics.to_excel(writer, sheet_name="robust_scale_diagnostics", index=False)
        pd.DataFrame(
            [
                {
                    "paper_count": len(feature_matrix),
                    "feature_count": len(feature_columns),
                    "epsilon": epsilon,
                    "q_min": float(pd.to_numeric(feature_matrix["Q_label"], errors="coerce").min()),
                    "q_max": float(pd.to_numeric(feature_matrix["Q_label"], errors="coerce").max()),
                    "q_mean": float(pd.to_numeric(feature_matrix["Q_label"], errors="coerce").mean()),
                    "q_std": float(pd.to_numeric(feature_matrix["Q_label"], errors="coerce").std(ddof=0)),
                }
            ]
        ).to_excel(writer, sheet_name="summary", index=False)

    _save_heatmap(scaled_matrix, correlation, heatmap_path)
    _save_correlation_bar(correlation, bar_path)
    _save_q_distribution(feature_matrix, q_dist_path)

    caution_features = variance_filter.loc[variance_filter["use_in_model"].ne("True"), ["feature_name", "variance_flag", "use_in_model"]]
    top8 = correlation.loc[correlation["use_in_model"].ne("False")].sort_values("spearman_abs", ascending=False).head(8)
    logger.info("Merged feature count=%s", len(feature_columns))
    logger.info("Caution/excluded features: %s", caution_features.to_dict(orient="records"))
    logger.info("Spearman Top 8: %s", top8[["feature_name", "spearman_corr", "spearman_abs"]].to_dict(orient="records"))
    logger.info("Saved outputs to %s and %s", tables_dir, charts_dir)
    logger.info("Finished Step 14")

    return {
        "feature_matrix": feature_matrix,
        "scaled_matrix": scaled_matrix,
        "variance_filter": variance_filter,
        "correlation": correlation,
        "scale_diagnostics": scale_diagnostics,
        "top8": top8,
        "caution_features": caution_features,
        "raw_output_path": raw_output_path,
        "scaled_output_path": scaled_output_path,
        "correlation_output_path": correlation_output_path,
        "variance_output_path": variance_output_path,
        "heatmap_path": heatmap_path,
        "bar_path": bar_path,
        "q_dist_path": q_dist_path,
        "log_path": log_path,
    }
