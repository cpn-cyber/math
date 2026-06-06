"""Quality adjustment factor model for Problem 2 Step 17."""

from __future__ import annotations

from pathlib import Path
import logging
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from modules.appendix2_pipeline import get_problem2_config, resolve_project_path


LOGGER_NAME = "A_MAGE_R3.problem2.quality_adjustment_factor"
POSITIVE_DEEP_FEATURES = [
    "task_coverage",
    "data_credibility",
    "method_fit",
    "formula_explanation",
    "result_closure",
]
NEGATIVE_DEEP_FEATURE = "stacking_penalty"
FOCUS_PAPERS = {"2-1", "2-2", "2-7", "2-10"}


def setup_qaf_logger(log_path: Path) -> logging.Logger:
    """Configure the Step 17 logger."""
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


def _metrics(y_true: pd.Series, y_pred: pd.Series | np.ndarray) -> dict[str, float]:
    """Calculate MAE, RMSE, R2, and Spearman."""
    y = pd.to_numeric(y_true, errors="coerce").to_numpy(dtype=float)
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


def _as_bool(value: Any) -> bool:
    """Convert common table values to bool."""
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def calculate_quality_adjustment_factor(
    deep_features: pd.DataFrame,
    eta: float,
    clip_min: float,
    clip_max: float,
) -> pd.DataFrame:
    """Calculate u_i, centered u_i, and quality adjustment factor phi_i."""
    required = [*POSITIVE_DEEP_FEATURES, NEGATIVE_DEEP_FEATURE]
    missing = [column for column in required if column not in deep_features.columns]
    if missing:
        raise ValueError(f"Deep quality feature table missing required columns: {missing}")

    scores = deep_features.copy()
    for column in required:
        scores[column] = pd.to_numeric(scores[column], errors="coerce")
    if scores[required].isna().any().any():
        missing_counts = scores[required].isna().sum()
        raise ValueError(f"Deep quality features contain NaN values: {missing_counts[missing_counts > 0].to_dict()}")

    scores["positive_deep_mean"] = scores[POSITIVE_DEEP_FEATURES].mean(axis=1)
    scores["u_i"] = scores["positive_deep_mean"] - scores[NEGATIVE_DEEP_FEATURE]
    scores["u_centered_i"] = scores["u_i"] - float(scores["u_i"].mean())
    scores["phi_i"] = (1.0 + float(eta) * scores["u_centered_i"]).clip(lower=float(clip_min), upper=float(clip_max))
    return scores


def apply_quality_adjustment(
    base_prediction: pd.Series,
    phi: pd.Series,
    q_min: float,
    q_max: float,
) -> pd.Series:
    """Apply quality adjustment to the base prediction and clip to label range."""
    adjusted = pd.to_numeric(base_prediction, errors="coerce") * pd.to_numeric(phi, errors="coerce")
    return adjusted.clip(lower=float(q_min), upper=float(q_max))


def _audit_note(row: pd.Series, high_stacking_threshold: float, high_length_constraint: bool) -> str:
    """Build per-paper audit note."""
    notes: list[str] = []
    paper_id = str(row["paper_id"])
    if paper_id == "2-1":
        notes.append("Step16 high-error paper; check whether QAF improves underprediction")
    if paper_id in {"2-2", "2-7", "2-10"} or _as_bool(row.get("review_focus", False)):
        notes.append("review_focus sample; adjusted result should be interpreted cautiously")
    if _as_bool(row.get("candidate_flag", False)):
        notes.append("candidate section participated in previous features")
    if float(row.get("stacking_penalty", 0.0)) >= high_stacking_threshold:
        notes.append("high stacking_penalty; QAF should constrain superficial stacking risk")
    if high_length_constraint:
        notes.append("high length signal with below-average deep quality; QAF constrains length-driven prediction")
    if not notes:
        notes.append("normal QAF adjustment")
    return "; ".join(notes)


def _build_qaf_prediction(
    pls_prediction: pd.DataFrame,
    qaf_scores: pd.DataFrame,
    feature_matrix: pd.DataFrame,
    eta: float,
    clip_min: float,
    clip_max: float,
    q_min: float,
    q_max: float,
) -> pd.DataFrame:
    """Build QAF prediction table for one eta."""
    base = pls_prediction.copy()
    for frame in [base, qaf_scores, feature_matrix]:
        if "paper_id" in frame.columns:
            frame["paper_id"] = frame["paper_id"].astype(str)

    qaf = calculate_quality_adjustment_factor(qaf_scores, eta, clip_min, clip_max)
    selected = base.merge(
        qaf[
            [
                "paper_id",
                *POSITIVE_DEEP_FEATURES,
                NEGATIVE_DEEP_FEATURE,
                "positive_deep_mean",
                "u_i",
                "u_centered_i",
                "phi_i",
            ]
        ],
        on="paper_id",
        how="left",
    )
    selected["Q_tilde"] = pd.to_numeric(selected["Q_pred_loo"], errors="coerce")
    selected["Q_hat_qaf"] = apply_quality_adjustment(selected["Q_tilde"], selected["phi_i"], q_min, q_max)
    selected["adjustment_value"] = selected["Q_hat_qaf"] - selected["Q_tilde"]
    selected["adjustment_direction"] = np.where(
        selected["adjustment_value"].gt(0),
        "up",
        np.where(selected["adjustment_value"].lt(0), "down", "none"),
    )
    selected["abs_error_pls"] = pd.to_numeric(selected["abs_error"], errors="coerce")
    selected["abs_error_qaf"] = (pd.to_numeric(selected["Q_true"], errors="coerce") - selected["Q_hat_qaf"]).abs()
    selected["error_change"] = selected["abs_error_pls"] - selected["abs_error_qaf"]

    length_cols = [column for column in ["total_chars", "page_count"] if column in feature_matrix.columns]
    length_flags = pd.DataFrame({"paper_id": selected["paper_id"], "high_length_low_deep": False})
    if length_cols:
        length_info = feature_matrix[["paper_id", *length_cols]].copy()
        length_info["paper_id"] = length_info["paper_id"].astype(str)
        length_signal = length_info[length_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
        length_threshold = float(length_signal.quantile(0.75))
        length_info["high_length_low_deep"] = length_signal.ge(length_threshold)
        length_flags = length_info[["paper_id", "high_length_low_deep"]]
    selected = selected.merge(length_flags, on="paper_id", how="left")
    selected["high_length_low_deep"] = selected["high_length_low_deep"].fillna(False)
    selected.loc[selected["u_centered_i"].ge(0), "high_length_low_deep"] = False

    high_stacking_threshold = float(selected[NEGATIVE_DEEP_FEATURE].quantile(0.75))
    selected["audit_note"] = selected.apply(
        lambda row: _audit_note(row, high_stacking_threshold, bool(row["high_length_low_deep"])),
        axis=1,
    )
    columns = [
        "paper_id",
        "filename",
        "Q_true",
        "Q_tilde",
        "u_i",
        "u_centered_i",
        "phi_i",
        "Q_hat_qaf",
        "adjustment_value",
        "adjustment_direction",
        "abs_error_pls",
        "abs_error_qaf",
        "error_change",
        "review_focus",
        "candidate_flag",
        "audit_note",
    ]
    return selected[columns].sort_values("paper_id", key=lambda series: series.map(_natural_sort_key)).reset_index(drop=True)


def _select_eta(eta_table: pd.DataFrame, base_metrics: dict[str, float]) -> tuple[float, str]:
    """Select eta with conservative small-sample rules."""
    candidates = eta_table.copy()
    improved = candidates.loc[(candidates["RMSE_change_vs_pls"] > 0) | (candidates["MAE_change_vs_pls"] > 0)]
    if improved.empty:
        eta = float(candidates.sort_values("eta").iloc[0]["eta"])
        return eta, "no_overall_error_improvement_select_smallest_eta"

    best = improved.sort_values(["RMSE", "MAE", "eta"], ascending=[True, True, True]).iloc[0]
    rmse_gain_ratio = float(best["RMSE_change_vs_pls"]) / float(base_metrics["RMSE"]) if base_metrics["RMSE"] else 0.0
    mae_gain_ratio = float(best["MAE_change_vs_pls"]) / float(base_metrics["MAE"]) if base_metrics["MAE"] else 0.0
    if rmse_gain_ratio < 0.01 and mae_gain_ratio < 0.01:
        eta = float(improved.sort_values("eta").iloc[0]["eta"])
        return eta, "minor_error_gain_select_smaller_eta_to_avoid_over_adjustment"
    return float(best["eta"]), "selected_by_lower_rmse_mae"


def _build_eta_selection(
    pls_prediction: pd.DataFrame,
    deep_features: pd.DataFrame,
    feature_matrix: pd.DataFrame,
    q_min: float,
    q_max: float,
    eta_grid: list[float],
    clip_min: float,
    clip_max: float,
) -> tuple[pd.DataFrame, float, pd.DataFrame]:
    """Evaluate eta candidates and return eta table plus selected prediction."""
    base_metrics = _metrics(pls_prediction["Q_true"], pls_prediction["Q_pred_loo"])
    rows: list[dict[str, Any]] = []
    predictions: dict[float, pd.DataFrame] = {}
    for eta in eta_grid:
        prediction = _build_qaf_prediction(pls_prediction, deep_features, feature_matrix, eta, clip_min, clip_max, q_min, q_max)
        predictions[float(eta)] = prediction
        metrics = _metrics(prediction["Q_true"], prediction["Q_hat_qaf"])
        row = {
            "eta": float(eta),
            **metrics,
            "MAE_change_vs_pls": base_metrics["MAE"] - metrics["MAE"],
            "RMSE_change_vs_pls": base_metrics["RMSE"] - metrics["RMSE"],
            "max_abs_adjustment": float(prediction["adjustment_value"].abs().max()),
            "paper_2_1_error_pls": np.nan,
            "paper_2_1_error_qaf": np.nan,
            "paper_2_1_improved": False,
        }
        paper_21 = prediction.loc[prediction["paper_id"].eq("2-1")]
        if not paper_21.empty:
            row["paper_2_1_error_pls"] = float(paper_21["abs_error_pls"].iloc[0])
            row["paper_2_1_error_qaf"] = float(paper_21["abs_error_qaf"].iloc[0])
            row["paper_2_1_improved"] = bool(row["paper_2_1_error_qaf"] < row["paper_2_1_error_pls"])
        rows.append(row)

    eta_table = pd.DataFrame(rows)
    selected_eta, selection_note = _select_eta(eta_table, base_metrics)
    eta_table["selected"] = eta_table["eta"].eq(selected_eta)
    eta_table["selection_note"] = eta_table["selected"].apply(lambda value: selection_note if value else "")
    for key, value in base_metrics.items():
        eta_table[f"PLS_base_{key}"] = value
    return eta_table, selected_eta, predictions[selected_eta]


def _build_adjustment_audit(prediction: pd.DataFrame, eta_table: pd.DataFrame, vip_table: pd.DataFrame) -> pd.DataFrame:
    """Build QAF adjustment audit rows."""
    rows: list[dict[str, Any]] = []
    selected_eta = float(eta_table.loc[eta_table["selected"], "eta"].iloc[0])
    selected_eta_row = eta_table.loc[eta_table["selected"]].iloc[0].to_dict()

    def add_row(audit_item: str, paper_rows: pd.DataFrame, note: str) -> None:
        for _, row in paper_rows.iterrows():
            rows.append(
                {
                    "audit_item": audit_item,
                    "paper_id": row["paper_id"],
                    "filename": row["filename"],
                    "Q_true": row["Q_true"],
                    "Q_tilde": row["Q_tilde"],
                    "Q_hat_qaf": row["Q_hat_qaf"],
                    "adjustment_value": row["adjustment_value"],
                    "abs_error_pls": row["abs_error_pls"],
                    "abs_error_qaf": row["abs_error_qaf"],
                    "error_change": row["error_change"],
                    "audit_note": note,
                }
            )

    rows.append(
        {
            "audit_item": "selected_eta",
            "paper_id": "ALL",
            "filename": "",
            "Q_true": np.nan,
            "Q_tilde": np.nan,
            "Q_hat_qaf": np.nan,
            "adjustment_value": selected_eta,
            "abs_error_pls": selected_eta_row["PLS_base_MAE"],
            "abs_error_qaf": selected_eta_row["MAE"],
            "error_change": selected_eta_row["MAE_change_vs_pls"],
            "audit_note": f"selected eta={selected_eta}; {selected_eta_row.get('selection_note', '')}",
        }
    )
    add_row("max_up_adjustment", prediction.sort_values("adjustment_value", ascending=False).head(3), "largest upward QAF adjustments")
    add_row("max_down_adjustment", prediction.sort_values("adjustment_value", ascending=True).head(3), "largest downward QAF adjustments")

    paper_21 = prediction.loc[prediction["paper_id"].eq("2-1")]
    if not paper_21.empty:
        improved = bool(paper_21["abs_error_qaf"].iloc[0] < paper_21["abs_error_pls"].iloc[0])
        add_row("paper_2_1_improvement", paper_21, f"2-1 improved={improved}")

    review_focus = prediction.loc[prediction["paper_id"].isin(["2-2", "2-7", "2-10"]) | prediction["review_focus"].apply(_as_bool)]
    add_row("review_focus_caution", review_focus, "review_focus papers require cautious interpretation")

    penalty_threshold = prediction["audit_note"].str.contains("high stacking_penalty", na=False)
    add_row("high_stacking_penalty_constraint", prediction.loc[penalty_threshold], "high stacking_penalty papers audited for downward constraint")

    length_constraint = prediction["audit_note"].str.contains("high length signal", na=False)
    add_row("length_signal_constraint", prediction.loc[length_constraint], "high length/page signal but below-average deep quality")

    stacking_vip = vip_table.loc[vip_table["feature_name"].eq("stacking_penalty")]
    if not stacking_vip.empty:
        rows.append(
            {
                "audit_item": "stacking_penalty_vip_context",
                "paper_id": "ALL",
                "filename": "",
                "Q_true": np.nan,
                "Q_tilde": np.nan,
                "Q_hat_qaf": np.nan,
                "adjustment_value": np.nan,
                "abs_error_pls": np.nan,
                "abs_error_qaf": np.nan,
                "error_change": np.nan,
                "audit_note": f"stacking_penalty VIP={float(stacking_vip['VIP'].iloc[0]):.6f}; kept as QAF negative constraint even if PLS importance is limited",
            }
        )
    return pd.DataFrame(rows)


def _save_waterfall(prediction: pd.DataFrame, chart_path: Path) -> None:
    """Save adjustment waterfall-like bar chart."""
    chart_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = prediction.sort_values("adjustment_value")
    colors = ["#b64b4b" if value < 0 else "#4f7cac" for value in ordered["adjustment_value"]]
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=180)
    ax.bar(ordered["paper_id"], ordered["adjustment_value"], color=colors)
    ax.axhline(0, color="#333333", linewidth=1)
    ax.set_ylabel("QAF adjustment value")
    ax.set_title("QAF Adjustment by Paper")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(chart_path)
    plt.close(fig)


def _save_true_vs_pred(prediction: pd.DataFrame, chart_path: Path) -> None:
    """Save Q_true vs Q_hat_qaf chart."""
    chart_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=180)
    colors = np.where(prediction["review_focus"].apply(_as_bool), "#c44e52", "#4f7cac")
    ax.scatter(prediction["Q_true"], prediction["Q_hat_qaf"], c=colors, s=55, alpha=0.9, edgecolor="white", linewidth=0.8)
    min_value = float(min(prediction["Q_true"].min(), prediction["Q_hat_qaf"].min()))
    max_value = float(max(prediction["Q_true"].max(), prediction["Q_hat_qaf"].max()))
    padding = max((max_value - min_value) * 0.08, 1.0)
    ax.plot([min_value - padding, max_value + padding], [min_value - padding, max_value + padding], color="#444444", linestyle="--", linewidth=1)
    for _, row in prediction.iterrows():
        ax.text(row["Q_true"], row["Q_hat_qaf"], str(row["paper_id"]), fontsize=7, ha="left", va="bottom")
    ax.set_xlabel("Q_true")
    ax.set_ylabel("Q_hat_qaf")
    ax.set_title("QAF: True vs Adjusted Prediction")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(chart_path)
    plt.close(fig)


def _save_before_after_error(prediction: pd.DataFrame, chart_path: Path) -> None:
    """Save before/after absolute error comparison."""
    chart_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = prediction.sort_values("abs_error_pls", ascending=False)
    x = np.arange(len(ordered))
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=180)
    width = 0.38
    ax.bar(x - width / 2, ordered["abs_error_pls"], width=width, label="PLS", color="#9a9a9a")
    ax.bar(x + width / 2, ordered["abs_error_qaf"], width=width, label="QAF", color="#4f7cac")
    ax.set_xticks(x)
    ax.set_xticklabels(ordered["paper_id"], rotation=0)
    ax.set_ylabel("Absolute error")
    ax.set_title("Absolute Error Before and After QAF")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(chart_path)
    plt.close(fig)


def run_step17_quality_adjustment(config_path: Path | None = None) -> dict[str, Any]:
    """Run Step 17 quality adjustment factor model."""
    problem2_config = get_problem2_config(config_path)
    tables_dir = resolve_project_path(problem2_config["problem2_output_tables_dir"])
    charts_dir = resolve_project_path(problem2_config["problem2_output_charts_dir"])
    logs_dir = resolve_project_path(problem2_config["problem2_output_logs_dir"])
    tables_dir.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    log_path = logs_dir / "quality_adjustment_factor.log"
    logger = setup_qaf_logger(log_path)
    logger.info("Starting Step 17 QAF; this step does not run Bootstrap, pairwise checks, or paper writing.")

    pls_prediction = pd.read_excel(tables_dir / "pls_prediction_results.xlsx")
    deep_features = pd.read_excel(tables_dir / "appendix2_deep_quality_features_auto.xlsx")
    feature_matrix = pd.read_excel(tables_dir / "appendix2_feature_matrix_robust_scaled.xlsx")
    q_labels = pd.read_excel(tables_dir / "appendix2_q_labels.xlsx")
    vip_table = pd.read_excel(tables_dir / "pls_vip_scores.xlsx")
    step12_audit = pd.read_excel(tables_dir / "appendix2_step12_quality_audit.xlsx")
    evidence = pd.read_excel(tables_dir / "appendix2_deep_quality_evidence.xlsx")

    for frame in [pls_prediction, deep_features, feature_matrix, q_labels, vip_table, step12_audit, evidence]:
        if "paper_id" in frame.columns:
            frame["paper_id"] = frame["paper_id"].astype(str)

    q_min = float(pd.to_numeric(q_labels["Q_label"], errors="coerce").min())
    q_max = float(pd.to_numeric(q_labels["Q_label"], errors="coerce").max())
    eta_grid = [float(value) for value in problem2_config.get("qaf_eta_grid", [0.05, 0.10, 0.15])]
    clip_min = float(problem2_config.get("qaf_clip_min", 0.90))
    clip_max = float(problem2_config.get("qaf_clip_max", 1.10))

    eta_table, selected_eta, selected_prediction = _build_eta_selection(
        pls_prediction,
        deep_features,
        feature_matrix,
        q_min,
        q_max,
        eta_grid,
        clip_min,
        clip_max,
    )

    qaf_score_base = calculate_quality_adjustment_factor(deep_features, selected_eta, clip_min, clip_max)
    qaf_scores = qaf_score_base[["paper_id", "filename", *POSITIVE_DEEP_FEATURES, NEGATIVE_DEEP_FEATURE, "positive_deep_mean", "u_i", "u_centered_i"]].copy()
    for eta in eta_grid:
        qaf_scores[f"phi_eta_{eta:.2f}"] = calculate_quality_adjustment_factor(deep_features, eta, clip_min, clip_max)["phi_i"]
    qaf_scores["selected_eta"] = selected_eta
    selected_phi = selected_prediction[["paper_id", "phi_i"]].rename(columns={"phi_i": "selected_phi_i"})
    qaf_scores = qaf_scores.merge(selected_phi, on="paper_id", how="left")
    qaf_scores = qaf_scores.sort_values("paper_id", key=lambda series: series.map(_natural_sort_key)).reset_index(drop=True)

    audit_table = _build_adjustment_audit(selected_prediction, eta_table, vip_table)

    qaf_scores_path = tables_dir / "qaf_scores.xlsx"
    eta_path = tables_dir / "qaf_eta_selection.xlsx"
    prediction_path = tables_dir / "qaf_prediction_results.xlsx"
    audit_path = tables_dir / "qaf_adjustment_audit.xlsx"
    waterfall_path = charts_dir / "qaf_waterfall.png"
    true_vs_pred_path = charts_dir / "qaf_true_vs_pred.png"
    before_after_path = charts_dir / "qaf_before_after_error.png"

    qaf_scores.to_excel(qaf_scores_path, index=False)
    eta_table.to_excel(eta_path, index=False)
    selected_prediction.to_excel(prediction_path, index=False)
    audit_table.to_excel(audit_path, index=False)
    _save_waterfall(selected_prediction, waterfall_path)
    _save_true_vs_pred(selected_prediction, true_vs_pred_path)
    _save_before_after_error(selected_prediction, before_after_path)

    selected_eta_row = eta_table.loc[eta_table["selected"]].iloc[0].to_dict()
    phi_range = (float(selected_prediction["phi_i"].min()), float(selected_prediction["phi_i"].max()))
    paper_21 = selected_prediction.loc[selected_prediction["paper_id"].eq("2-1")]
    paper_21_improved = bool(paper_21["abs_error_qaf"].iloc[0] < paper_21["abs_error_pls"].iloc[0]) if not paper_21.empty else False
    max_up = selected_prediction.sort_values("adjustment_value", ascending=False).head(3)
    max_down = selected_prediction.sort_values("adjustment_value", ascending=True).head(3)
    stacking_constrained = selected_prediction.loc[selected_prediction["audit_note"].str.contains("high stacking_penalty", na=False)]

    logger.info("Q_label range for clipping: min=%.6f max=%.6f", q_min, q_max)
    logger.info("Eta selection table: %s", eta_table.to_dict(orient="records"))
    logger.info("Selected eta: %.6f", selected_eta)
    logger.info("Selected phi range: %.6f~%.6f", phi_range[0], phi_range[1])
    logger.info("Selected metrics: %s", selected_eta_row)
    logger.info("2-1 improved: %s row=%s", paper_21_improved, paper_21.to_dict(orient="records"))
    logger.info("Max upward adjustments: %s", max_up.to_dict(orient="records"))
    logger.info("Max downward adjustments: %s", max_down.to_dict(orient="records"))
    logger.info("Stacking constrained rows: %s", stacking_constrained.to_dict(orient="records"))
    logger.info("Saved outputs: %s, %s, %s, %s, %s, %s, %s", qaf_scores_path, eta_path, prediction_path, audit_path, waterfall_path, true_vs_pred_path, before_after_path)
    logger.info("Finished Step 17")

    return {
        "qaf_scores": qaf_scores,
        "eta_table": eta_table,
        "selected_eta": selected_eta,
        "selected_eta_row": selected_eta_row,
        "prediction": selected_prediction,
        "audit": audit_table,
        "phi_range": phi_range,
        "paper_21_improved": paper_21_improved,
        "paper_21": paper_21,
        "max_up": max_up,
        "max_down": max_down,
        "stacking_constrained": stacking_constrained,
        "qaf_scores_path": qaf_scores_path,
        "eta_path": eta_path,
        "prediction_path": prediction_path,
        "audit_path": audit_path,
        "waterfall_path": waterfall_path,
        "true_vs_pred_path": true_vs_pred_path,
        "before_after_path": before_after_path,
        "log_path": log_path,
    }
