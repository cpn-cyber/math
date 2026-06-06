"""Small-sample validation and final key-feature index for Problem 2 Step 18."""

from __future__ import annotations

from pathlib import Path
import logging
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from modules.appendix2_pipeline import get_problem2_config, load_config, resolve_project_path
from modules.pls_vip_model import calculate_vip, fit_pls_model


LOGGER_NAME = "A_MAGE_R3.problem2.small_sample_validation"
NEGATIVE_FEATURES = {"stacking_penalty"}
EXCLUDED_FEATURES = {"reference_norm_rate", "appendix_code_presence"}
FOCUS_PAPERS = {"2-1", "2-2", "2-7", "2-10"}


def setup_validation_logger(log_path: Path) -> logging.Logger:
    """Configure the Step 18 logger."""
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


def _spearman_corr(x: pd.Series | np.ndarray, y: pd.Series | np.ndarray) -> float:
    """Calculate Spearman rank correlation."""
    try:
        from scipy.stats import spearmanr

        value = spearmanr(x, y, nan_policy="omit").correlation
        return float(value) if value is not None else np.nan
    except Exception:
        x_rank = pd.Series(x).rank(method="average")
        y_rank = pd.Series(y).rank(method="average")
        return float(x_rank.corr(y_rank)) if x_rank.nunique() > 1 and y_rank.nunique() > 1 else np.nan


def _metrics(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> dict[str, float]:
    """Calculate MAE, RMSE, R2, and Spearman."""
    y = pd.to_numeric(pd.Series(y_true), errors="coerce").to_numpy(dtype=float)
    pred = pd.to_numeric(pd.Series(y_pred), errors="coerce").to_numpy(dtype=float)
    mae = float(np.mean(np.abs(y - pred)))
    rmse = float(np.sqrt(np.mean((y - pred) ** 2)))
    sst = float(np.sum((y - y.mean()) ** 2))
    sse = float(np.sum((y - pred) ** 2))
    r2 = np.nan if sst == 0 else 1.0 - sse / sst
    spearman = _spearman_corr(y, pred)
    return {"MAE": mae, "RMSE": rmse, "R2": float(r2), "Spearman": float(spearman)}


def _natural_sort_key(value: Any) -> list[Any]:
    """Natural sort key for paper IDs."""
    import re

    text = str(value)
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def _feature_group(feature_set: pd.DataFrame, feature_name: str) -> str:
    """Return feature group from the PLS feature set."""
    row = feature_set.loc[feature_set["feature_name"].eq(feature_name)]
    if row.empty:
        return "deep" if feature_name in NEGATIVE_FEATURES else "surface"
    return str(row["feature_group"].iloc[0])


def _coef_sign(model: Any, feature_count: int) -> np.ndarray:
    """Return coefficient signs from a fitted sklearn PLS model."""
    coef = np.asarray(getattr(model, "coef_", np.zeros(feature_count)), dtype=float).reshape(-1)
    if coef.size != feature_count:
        coef = coef[:feature_count]
    return np.sign(coef)


def _minmax(values: pd.Series) -> pd.Series:
    """Min-max normalize a numeric series."""
    series = pd.to_numeric(values, errors="coerce")
    valid = series.dropna()
    if valid.empty:
        return series * np.nan
    minimum = float(valid.min())
    maximum = float(valid.max())
    if maximum == minimum:
        return series.apply(lambda value: np.nan if pd.isna(value) else 1.0)
    return (series - minimum) / (maximum - minimum)


def _safe_prediction(model: Any, features: pd.DataFrame) -> np.ndarray:
    """Predict with sklearn PLS and flatten output."""
    return np.asarray(model.predict(features.to_numpy(dtype=float))).reshape(-1)


def _load_selected_features(feature_set: pd.DataFrame) -> list[str]:
    """Load selected PLS feature columns."""
    return feature_set.loc[feature_set["use_in_pls"].astype(bool), "feature_name"].astype(str).tolist()


def run_loocv(features: pd.DataFrame, quality_label: pd.Series, model_config: dict | None = None) -> pd.DataFrame:
    """Run leave-one-out cross validation for a fixed PLS component count."""
    n_components = int((model_config or {}).get("n_components", 1))
    rows: list[dict[str, Any]] = []
    for test_index in range(len(features)):
        train_mask = np.ones(len(features), dtype=bool)
        train_mask[test_index] = False
        model = fit_pls_model(features.iloc[train_mask], quality_label.iloc[train_mask], n_components)
        pred = float(_safe_prediction(model, features.iloc[[test_index]])[0])
        rows.append({"test_index": test_index, "Q_true": float(quality_label.iloc[test_index]), "Q_pred": pred})
    return pd.DataFrame(rows)


def _bootstrap_raw_records(
    features: pd.DataFrame,
    quality_label: pd.Series,
    feature_columns: list[str],
    bootstrap_B: int,
    random_seed: int,
    logger: logging.Logger | None = None,
) -> tuple[pd.DataFrame, int, int]:
    """Run bootstrap PLS fits and return long VIP/sign records."""
    rng = np.random.default_rng(random_seed)
    records: list[dict[str, Any]] = []
    skipped = 0
    n = len(features)
    for b in range(int(bootstrap_B)):
        sample_indices = rng.integers(0, n, size=n)
        unique_count = int(len(np.unique(sample_indices)))
        y_sample = quality_label.iloc[sample_indices].reset_index(drop=True)
        if unique_count < 3 or y_sample.nunique() < 2:
            skipped += 1
            continue
        x_sample = features.iloc[sample_indices].reset_index(drop=True)
        try:
            model = fit_pls_model(x_sample, y_sample, 1)
            vip = calculate_vip(model, x_sample, y_sample)
            signs = _coef_sign(model, len(feature_columns))
        except Exception as exc:  # pragma: no cover - model edge case
            skipped += 1
            if logger is not None:
                logger.warning("Bootstrap iteration %s skipped: %s", b, exc)
            continue
        for idx, feature in enumerate(feature_columns):
            records.append(
                {
                    "bootstrap_id": b,
                    "feature_name": feature,
                    "VIP": float(vip.loc[vip["feature_name"].eq(feature), "VIP"].iloc[0]),
                    "coef_sign": int(signs[idx]),
                    "VIP_gt_1": bool(float(vip.loc[vip["feature_name"].eq(feature), "VIP"].iloc[0]) > 1.0),
                    "unique_papers": unique_count,
                }
            )
    valid = len(set(row["bootstrap_id"] for row in records))
    return pd.DataFrame(records), valid, skipped


def run_bootstrap_stability(features: pd.DataFrame, quality_label: pd.Series, bootstrap_B: int) -> pd.DataFrame:
    """Run bootstrap stability analysis and return aggregate feature statistics."""
    feature_columns = list(features.columns)
    records, _, _ = _bootstrap_raw_records(features, quality_label, feature_columns, bootstrap_B, random_seed=2026)
    return _summarize_bootstrap(records, feature_columns, pd.DataFrame({"feature_name": feature_columns}))


def _summarize_bootstrap(records: pd.DataFrame, feature_columns: list[str], feature_set: pd.DataFrame) -> pd.DataFrame:
    """Aggregate bootstrap VIP and coefficient sign stability."""
    rows: list[dict[str, Any]] = []
    for feature in feature_columns:
        subset = records.loc[records["feature_name"].eq(feature)]
        if subset.empty:
            rows.append(
                {
                    "feature_name": feature,
                    "feature_group": _feature_group(feature_set, feature),
                    "mean_VIP": np.nan,
                    "std_VIP": np.nan,
                    "P_VIP_gt_1": np.nan,
                    "positive_sign_rate": np.nan,
                    "negative_sign_rate": np.nan,
                    "zero_sign_rate": np.nan,
                    "sign_consistency": np.nan,
                    "valid_bootstrap_count": 0,
                    "bootstrap_selected": False,
                }
            )
            continue
        sign_counts = subset["coef_sign"].value_counts(normalize=True)
        positive = float(sign_counts.get(1, 0.0))
        negative = float(sign_counts.get(-1, 0.0))
        zero = float(sign_counts.get(0, 0.0))
        sign_consistency = max(positive, negative)
        p_vip = float(subset["VIP_gt_1"].mean())
        rows.append(
            {
                "feature_name": feature,
                "feature_group": _feature_group(feature_set, feature),
                "mean_VIP": float(subset["VIP"].mean()),
                "std_VIP": float(subset["VIP"].std(ddof=1)),
                "P_VIP_gt_1": p_vip,
                "positive_sign_rate": positive,
                "negative_sign_rate": negative,
                "zero_sign_rate": zero,
                "sign_consistency": sign_consistency,
                "valid_bootstrap_count": int(subset["bootstrap_id"].nunique()),
                "bootstrap_selected": bool(p_vip > 0.6 and sign_consistency > 0.7),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_VIP", ascending=False).reset_index(drop=True)


def run_leave_one_sample_sensitivity(features: pd.DataFrame, quality_label: pd.Series) -> pd.DataFrame:
    """Run deletion-one-sample sensitivity analysis with internal LOOCV."""
    paper_ids = [str(i) for i in range(len(features))]
    return _delete_one_sensitivity_core(features, quality_label, paper_ids, list(features.columns), pd.DataFrame())


def _fit_full_vip(features: pd.DataFrame, quality_label: pd.Series, feature_columns: list[str]) -> pd.DataFrame:
    """Fit full PLS(A=1) and return VIP table."""
    model = fit_pls_model(features, quality_label, 1)
    vip = calculate_vip(model, features, quality_label)
    signs = _coef_sign(model, len(feature_columns))
    vip["coef_sign"] = signs
    vip["VIP_rank"] = vip["VIP"].rank(ascending=False, method="first").astype(int)
    return vip.sort_values("VIP_rank").reset_index(drop=True)


def _delete_one_sensitivity_core(
    features: pd.DataFrame,
    quality_label: pd.Series,
    paper_ids: list[str],
    feature_columns: list[str],
    feature_set: pd.DataFrame,
    baseline_metrics: dict[str, float] | None = None,
    full_top5: set[str] | None = None,
) -> pd.DataFrame:
    """Deletion-one-sample sensitivity implementation."""
    baseline_metrics = baseline_metrics or {"RMSE": np.nan, "Spearman": np.nan}
    if full_top5 is None:
        full_vip = _fit_full_vip(features, quality_label, feature_columns)
        full_top5 = set(full_vip.head(5)["feature_name"])
        baseline_top1 = str(full_vip.iloc[0]["feature_name"])
    else:
        full_vip = _fit_full_vip(features, quality_label, feature_columns)
        baseline_top1 = str(full_vip.iloc[0]["feature_name"])

    rows: list[dict[str, Any]] = []
    for remove_index, paper_id in enumerate(paper_ids):
        keep_mask = np.ones(len(features), dtype=bool)
        keep_mask[remove_index] = False
        x_sub = features.iloc[keep_mask].reset_index(drop=True)
        y_sub = quality_label.iloc[keep_mask].reset_index(drop=True)
        loo = run_loocv(x_sub, y_sub, {"n_components": 1})
        metrics = _metrics(loo["Q_true"], loo["Q_pred"])
        vip_sub = _fit_full_vip(x_sub, y_sub, feature_columns)
        top5 = set(vip_sub.head(5)["feature_name"])
        jaccard = len(top5 & full_top5) / len(top5 | full_top5) if top5 | full_top5 else np.nan
        top1_changed = str(vip_sub.iloc[0]["feature_name"]) != baseline_top1
        rmse_delta = metrics["RMSE"] - float(baseline_metrics.get("RMSE", np.nan))
        spearman_delta = metrics["Spearman"] - float(baseline_metrics.get("Spearman", np.nan))
        high_influence = bool(
            (pd.notna(rmse_delta) and abs(rmse_delta) >= max(1.0, 0.10 * float(baseline_metrics.get("RMSE", 0.0))))
            or (pd.notna(spearman_delta) and abs(spearman_delta) >= 0.20)
            or (pd.notna(jaccard) and jaccard < 0.60)
            or top1_changed
        )
        reason_parts: list[str] = []
        if pd.notna(rmse_delta) and abs(rmse_delta) >= max(1.0, 0.10 * float(baseline_metrics.get("RMSE", 0.0))):
            reason_parts.append("rmse_delta")
        if pd.notna(spearman_delta) and abs(spearman_delta) >= 0.20:
            reason_parts.append("spearman_delta")
        if pd.notna(jaccard) and jaccard < 0.60:
            reason_parts.append("top5_jaccard_low")
        if top1_changed:
            reason_parts.append("top1_changed")
        if str(paper_id) in FOCUS_PAPERS:
            reason_parts.append("focus_paper")
        rows.append(
            {
                "removed_paper_id": paper_id,
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"],
                "Spearman": metrics["Spearman"],
                "RMSE_delta_vs_full_LOOCV": rmse_delta,
                "Spearman_delta_vs_full_LOOCV": spearman_delta,
                "VIP_top5": ",".join(vip_sub.head(5)["feature_name"].astype(str)),
                "top5_jaccard_vs_full": jaccard,
                "top1_feature": str(vip_sub.iloc[0]["feature_name"]),
                "top1_changed": top1_changed,
                "high_influence_sample": high_influence,
                "influence_reason": ";".join(reason_parts) if reason_parts else "stable",
            }
        )
    return pd.DataFrame(rows)


def _interpret_final_feature(row: pd.Series) -> str:
    """Build final key feature interpretation."""
    feature = str(row["feature_name"])
    if feature == "reference_norm_rate":
        return "Excluded constant feature; not used in final key-feature ranking."
    if feature == "appendix_code_presence":
        return "Excluded from main ranking because it is low-variance in the small sample."
    if feature in {"total_chars", "page_count"}:
        return "Length/information-carrying signal; interpret as completeness-related, not longer-is-better."
    if feature == "stacking_penalty":
        return "Negative constraint feature; larger value means higher stacking risk."
    if feature in {"method_fit", "task_coverage", "section_coverage", "objective_constraint_completeness"}:
        return "Core quality feature linking task coverage, method fit, and modeling completeness."
    if feature == "figure_table_explanation_rate":
        return "Result presentation quality feature; reflects whether figures/tables are explained."
    return "Auxiliary explanatory feature for weak-supervised quality differences."


def _build_final_key_index(
    correlation: pd.DataFrame,
    grey: pd.DataFrame,
    key_pre: pd.DataFrame,
    bootstrap: pd.DataFrame,
    feature_set: pd.DataFrame,
) -> pd.DataFrame:
    """Build final K_j key-feature index."""
    table = correlation[["feature_name", "feature_group", "spearman_abs"]].copy()
    table = table.merge(grey[["feature_name", "grey_norm"]], on="feature_name", how="left")
    table = table.merge(bootstrap[["feature_name", "mean_VIP", "P_VIP_gt_1", "sign_consistency", "bootstrap_selected"]], on="feature_name", how="left")
    feature_info = feature_set[["feature_name", "use_in_pls", "variance_flag"]].copy()
    table = table.merge(feature_info, on="feature_name", how="left")
    table["VIP_norm"] = _minmax(table["mean_VIP"])
    table["spearman_abs"] = pd.to_numeric(table["spearman_abs"], errors="coerce")
    table["grey_norm"] = pd.to_numeric(table["grey_norm"], errors="coerce")
    table["sign_consistency"] = pd.to_numeric(table["sign_consistency"], errors="coerce")
    include_mask = table["use_in_pls"].astype(str).eq("True")
    table["K_final"] = (
        0.35 * table["spearman_abs"].fillna(0.0)
        + 0.25 * table["grey_norm"].fillna(0.0)
        + 0.25 * table["VIP_norm"].fillna(0.0)
        + 0.15 * table["sign_consistency"].fillna(0.0)
    )
    table.loc[~include_mask, "K_final"] = np.nan
    rankable = table.loc[include_mask].copy()
    rankable["K_rank"] = rankable["K_final"].rank(ascending=False, method="first").astype(int)
    table = table.merge(rankable[["feature_name", "K_rank"]], on="feature_name", how="left")
    table["final_key_feature"] = (
        table["K_rank"].le(8)
        & include_mask
        & table["variance_flag"].astype(str).ne("constant_feature")
        & table["variance_flag"].astype(str).ne("low_variance_feature")
    )
    table["interpretation"] = table.apply(_interpret_final_feature, axis=1)
    columns = [
        "feature_name",
        "feature_group",
        "spearman_abs",
        "grey_norm",
        "mean_VIP",
        "VIP_norm",
        "sign_consistency",
        "P_VIP_gt_1",
        "K_final",
        "K_rank",
        "bootstrap_selected",
        "final_key_feature",
        "interpretation",
    ]
    return table[columns].sort_values(["K_rank", "feature_name"], na_position="last").reset_index(drop=True)


def _save_bootstrap_boxplot(records: pd.DataFrame, final_index: pd.DataFrame, chart_path: Path) -> None:
    """Save bootstrap VIP boxplot for top features."""
    chart_path.parent.mkdir(parents=True, exist_ok=True)
    top_features = final_index.dropna(subset=["K_rank"]).sort_values("K_rank").head(10)["feature_name"].astype(str).tolist()
    data = [records.loc[records["feature_name"].eq(feature), "VIP"].to_numpy(dtype=float) for feature in top_features]
    fig, ax = plt.subplots(figsize=(10, 5.8), dpi=180)
    ax.boxplot(data, labels=top_features, vert=True, patch_artist=True)
    ax.axhline(1.0, color="#333333", linestyle="--", linewidth=1)
    ax.set_ylabel("Bootstrap VIP")
    ax.set_title("Bootstrap VIP Stability for Top Features")
    ax.tick_params(axis="x", labelrotation=45)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(chart_path)
    plt.close(fig)


def _save_key_index_bar(final_index: pd.DataFrame, chart_path: Path) -> None:
    """Save K_final Top 10 bar chart."""
    chart_path.parent.mkdir(parents=True, exist_ok=True)
    top = final_index.dropna(subset=["K_final"]).sort_values("K_final", ascending=False).head(10).iloc[::-1]
    colors = ["#b64b4b" if feature in NEGATIVE_FEATURES else "#4f7cac" for feature in top["feature_name"]]
    fig, ax = plt.subplots(figsize=(9, 6), dpi=180)
    ax.barh(top["feature_name"], top["K_final"], color=colors)
    ax.set_xlabel("K_final")
    ax.set_title("Final Key Feature Index Top 10")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(chart_path)
    plt.close(fig)


def _save_delete_one_chart(delete_one: pd.DataFrame, chart_path: Path) -> None:
    """Save deletion-one RMSE/Spearman sensitivity chart."""
    chart_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = delete_one.sort_values("removed_paper_id", key=lambda series: series.map(_natural_sort_key))
    x = np.arange(len(ordered))
    fig, ax1 = plt.subplots(figsize=(10, 5.5), dpi=180)
    ax1.bar(x - 0.18, ordered["RMSE_delta_vs_full_LOOCV"], width=0.36, label="RMSE delta", color="#4f7cac")
    ax1.set_ylabel("RMSE delta")
    ax1.axhline(0, color="#333333", linewidth=1)
    ax2 = ax1.twinx()
    ax2.plot(x + 0.18, ordered["Spearman_delta_vs_full_LOOCV"], marker="o", color="#b64b4b", label="Spearman delta")
    ax2.set_ylabel("Spearman delta")
    ax1.set_xticks(x)
    ax1.set_xticklabels(ordered["removed_paper_id"])
    ax1.set_title("Delete-One Sample Sensitivity")
    ax1.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(chart_path)
    plt.close(fig)


def _save_model_summary_chart(summary_metrics: pd.DataFrame, chart_path: Path) -> None:
    """Save PLS vs QAF metric comparison chart."""
    chart_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = summary_metrics.loc[summary_metrics["metric"].isin(["MAE", "RMSE", "R2", "Spearman"])].copy()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), dpi=180)
    error_metrics = metrics.loc[metrics["metric"].isin(["MAE", "RMSE"])]
    axes[0].bar(error_metrics["metric"] + "_PLS", error_metrics["PLS"].astype(float), color="#9a9a9a", label="PLS")
    axes[0].bar(error_metrics["metric"] + "_QAF", error_metrics["QAF"].astype(float), color="#4f7cac", label="QAF")
    axes[0].set_title("Error Metrics")
    axes[0].tick_params(axis="x", labelrotation=25)
    axes[0].grid(axis="y", alpha=0.25)
    score_metrics = metrics.loc[metrics["metric"].isin(["R2", "Spearman"])]
    axes[1].bar(score_metrics["metric"] + "_PLS", score_metrics["PLS"].astype(float), color="#9a9a9a")
    axes[1].bar(score_metrics["metric"] + "_QAF", score_metrics["QAF"].astype(float), color="#4f7cac")
    axes[1].set_title("Association Metrics")
    axes[1].tick_params(axis="x", labelrotation=25)
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("PLS vs QAF Stability Summary")
    fig.tight_layout()
    fig.savefig(chart_path)
    plt.close(fig)


def _build_model_stability_summary(
    pls_prediction: pd.DataFrame,
    qaf_prediction: pd.DataFrame,
    bootstrap: pd.DataFrame,
    high_influence: pd.DataFrame,
) -> pd.DataFrame:
    """Build model stability summary table."""
    pls_metrics = _metrics(pls_prediction["Q_true"], pls_prediction["Q_pred_loo"])
    qaf_metrics = _metrics(qaf_prediction["Q_true"], qaf_prediction["Q_hat_qaf"])
    rows = []
    for metric in ["MAE", "RMSE", "R2", "Spearman"]:
        rows.append(
            {
                "section": "model_metrics",
                "metric": metric,
                "PLS": pls_metrics[metric],
                "QAF": qaf_metrics[metric],
                "note": "QAF is conservative; do not overstate performance.",
            }
        )
    rows.extend(
        [
            {
                "section": "qaf_improvement",
                "metric": "MAE_improved",
                "PLS": pls_metrics["MAE"],
                "QAF": qaf_metrics["MAE"],
                "note": str(qaf_metrics["MAE"] < pls_metrics["MAE"]),
            },
            {
                "section": "bootstrap",
                "metric": "bootstrap_selected_feature_count",
                "PLS": np.nan,
                "QAF": np.nan,
                "note": int(bootstrap["bootstrap_selected"].sum()),
            },
            {
                "section": "delete_one",
                "metric": "high_influence_samples",
                "PLS": np.nan,
                "QAF": np.nan,
                "note": ",".join(high_influence["removed_paper_id"].astype(str).tolist()) if not high_influence.empty else "none",
            },
            {
                "section": "overall_judgment",
                "metric": "small_sample_stability",
                "PLS": np.nan,
                "QAF": np.nan,
                "note": "Limited prediction ability; use VIP/K_final for interpretation, QAF as conservative correction, and report high-influence samples.",
            },
            {
                "section": "writing_advice",
                "metric": "advice",
                "PLS": np.nan,
                "QAF": np.nan,
                "note": "Do not claim strong generalization; emphasize weak-supervised label, N=10 limitation, and auditability.",
            },
        ]
    )
    return pd.DataFrame(rows)


def run_step18_small_sample_validation(config_path: Path | None = None) -> dict[str, Any]:
    """Run Step 18 bootstrap, delete-one sensitivity, and final key index."""
    problem2_config = get_problem2_config(config_path)
    tables_dir = resolve_project_path(problem2_config["problem2_output_tables_dir"])
    charts_dir = resolve_project_path(problem2_config["problem2_output_charts_dir"])
    logs_dir = resolve_project_path(problem2_config["problem2_output_logs_dir"])
    tables_dir.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    log_path = logs_dir / "small_sample_validation.log"
    logger = setup_validation_logger(log_path)
    logger.info("Starting Step 18; this step does not run pairwise checks or paper writing.")

    try:
        project_config = load_config(config_path) if config_path else {}
    except RuntimeError:
        project_config = {}
    random_seed = int(project_config.get("project", {}).get("random_seed", 2026))
    bootstrap_B = int(problem2_config.get("bootstrap_B", 1000))

    feature_matrix = pd.read_excel(tables_dir / "appendix2_feature_matrix_robust_scaled.xlsx")
    q_labels = pd.read_excel(tables_dir / "appendix2_q_labels.xlsx")
    correlation = pd.read_excel(tables_dir / "appendix2_correlation_analysis.xlsx")
    grey = pd.read_excel(tables_dir / "appendix2_grey_relation.xlsx")
    key_pre = pd.read_excel(tables_dir / "appendix2_key_features_preliminary.xlsx")
    vip_scores = pd.read_excel(tables_dir / "pls_vip_scores.xlsx")
    feature_set = pd.read_excel(tables_dir / "pls_model_feature_set.xlsx")
    pls_prediction = pd.read_excel(tables_dir / "pls_prediction_results.xlsx")
    qaf_prediction = pd.read_excel(tables_dir / "qaf_prediction_results.xlsx")
    qaf_audit = pd.read_excel(tables_dir / "qaf_adjustment_audit.xlsx")

    for frame in [feature_matrix, q_labels, correlation, grey, key_pre, vip_scores, feature_set, pls_prediction, qaf_prediction, qaf_audit]:
        if "paper_id" in frame.columns:
            frame["paper_id"] = frame["paper_id"].astype(str)

    feature_matrix = feature_matrix.sort_values("paper_id", key=lambda series: series.map(_natural_sort_key)).reset_index(drop=True)
    feature_columns = _load_selected_features(feature_set)
    features = feature_matrix[feature_columns].apply(pd.to_numeric, errors="coerce")
    quality_label = pd.to_numeric(feature_matrix["Q_label"], errors="coerce")
    if features.isna().any().any():
        missing = features.isna().sum()
        raise ValueError(f"Selected PLS features contain NaN values: {missing[missing > 0].to_dict()}")
    if quality_label.isna().any():
        raise ValueError("Q_label contains NaN values.")

    bootstrap_records, valid_bootstrap_count, skipped_bootstrap_count = _bootstrap_raw_records(
        features,
        quality_label,
        feature_columns,
        bootstrap_B,
        random_seed,
        logger,
    )
    bootstrap_summary = _summarize_bootstrap(bootstrap_records, feature_columns, feature_set)
    baseline_metrics = _metrics(pls_prediction["Q_true"], pls_prediction["Q_pred_loo"])
    full_vip = _fit_full_vip(features, quality_label, feature_columns)
    full_top5 = set(full_vip.head(5)["feature_name"])
    delete_one = _delete_one_sensitivity_core(
        features,
        quality_label,
        feature_matrix["paper_id"].astype(str).tolist(),
        feature_columns,
        feature_set,
        baseline_metrics=baseline_metrics,
        full_top5=full_top5,
    )
    high_influence = delete_one.loc[delete_one["high_influence_sample"]].copy()
    final_index = _build_final_key_index(correlation, grey, key_pre, bootstrap_summary, feature_set)
    model_summary = _build_model_stability_summary(pls_prediction, qaf_prediction, bootstrap_summary, high_influence)

    bootstrap_path = tables_dir / "bootstrap_vip_stability.xlsx"
    delete_one_path = tables_dir / "delete_one_sensitivity.xlsx"
    final_index_path = tables_dir / "key_feature_index_final.xlsx"
    summary_path = tables_dir / "model_stability_summary.xlsx"
    high_influence_path = tables_dir / "high_influence_samples.xlsx"
    bootstrap_chart_path = charts_dir / "bootstrap_vip_boxplot.png"
    final_index_chart_path = charts_dir / "key_feature_index_final.png"
    delete_one_chart_path = charts_dir / "delete_one_sensitivity.png"
    model_summary_chart_path = charts_dir / "model_stability_summary.png"

    bootstrap_summary.to_excel(bootstrap_path, index=False)
    delete_one.to_excel(delete_one_path, index=False)
    final_index.to_excel(final_index_path, index=False)
    high_influence.to_excel(high_influence_path, index=False)
    with pd.ExcelWriter(summary_path) as writer:
        model_summary.to_excel(writer, sheet_name="summary", index=False)
        bootstrap_summary.loc[bootstrap_summary["bootstrap_selected"]].to_excel(writer, sheet_name="stable_features", index=False)
        high_influence.to_excel(writer, sheet_name="high_influence", index=False)
        final_index.head(10).to_excel(writer, sheet_name="top_key_features", index=False)

    _save_bootstrap_boxplot(bootstrap_records, final_index, bootstrap_chart_path)
    _save_key_index_bar(final_index, final_index_chart_path)
    _save_delete_one_chart(delete_one, delete_one_chart_path)
    _save_model_summary_chart(model_summary, model_summary_chart_path)

    bootstrap_top = bootstrap_summary.sort_values(["bootstrap_selected", "mean_VIP"], ascending=[False, False]).head(10)
    final_top8 = final_index.dropna(subset=["K_rank"]).sort_values("K_rank").head(8)
    final_key_features = final_index.loc[final_index["final_key_feature"]]
    delete_21 = delete_one.loc[delete_one["removed_paper_id"].eq("2-1")]
    stability_judgment = "limited but auditable; use K_final for interpretation, not strong prediction"

    logger.info("Bootstrap B=%s valid=%s skipped=%s", bootstrap_B, valid_bootstrap_count, skipped_bootstrap_count)
    logger.info("Bootstrap stable features: %s", bootstrap_summary.loc[bootstrap_summary["bootstrap_selected"]].to_dict(orient="records"))
    logger.info("Final K Top 8: %s", final_top8.to_dict(orient="records"))
    logger.info("Final key features: %s", final_key_features["feature_name"].astype(str).tolist())
    logger.info("High influence samples: %s", high_influence.to_dict(orient="records"))
    logger.info("Delete 2-1 row: %s", delete_21.to_dict(orient="records"))
    logger.info("Stability judgment: %s", stability_judgment)
    logger.info("Saved outputs: %s, %s, %s, %s, %s", bootstrap_path, delete_one_path, final_index_path, summary_path, high_influence_path)
    logger.info("Finished Step 18")

    return {
        "bootstrap_summary": bootstrap_summary,
        "bootstrap_records": bootstrap_records,
        "valid_bootstrap_count": valid_bootstrap_count,
        "skipped_bootstrap_count": skipped_bootstrap_count,
        "delete_one": delete_one,
        "final_index": final_index,
        "model_summary": model_summary,
        "high_influence": high_influence,
        "bootstrap_top": bootstrap_top,
        "final_top8": final_top8,
        "final_key_features": final_key_features,
        "delete_21": delete_21,
        "stability_judgment": stability_judgment,
        "bootstrap_path": bootstrap_path,
        "delete_one_path": delete_one_path,
        "final_index_path": final_index_path,
        "summary_path": summary_path,
        "high_influence_path": high_influence_path,
        "bootstrap_chart_path": bootstrap_chart_path,
        "final_index_chart_path": final_index_chart_path,
        "delete_one_chart_path": delete_one_chart_path,
        "model_summary_chart_path": model_summary_chart_path,
        "log_path": log_path,
    }
