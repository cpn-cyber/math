"""PLS-VIP small-sample prediction model for Problem 2 Step 16."""

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
from modules.deep_quality_features import DEEP_FEATURE_COLUMNS
from modules.quality_label_builder import SURFACE_FEATURE_COLUMNS


LOGGER_NAME = "A_MAGE_R3.problem2.pls_vip_model"
NEGATIVE_FEATURES = {"stacking_penalty"}
NON_NUMERIC_MARKERS = {"review_focus", "candidate_used", "candidate_sections", "candidate_sections_from_step12"}


def setup_pls_logger(log_path: Path) -> logging.Logger:
    """Configure the Step 16 logger."""
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


def _import_sklearn() -> tuple[Any, Any, Any]:
    """Import sklearn objects lazily and raise a clear dependency message."""
    try:
        from sklearn.cross_decomposition import PLSRegression
        from sklearn.metrics import mean_absolute_error, mean_squared_error
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "scikit-learn is required for Step 16 PLSRegression. "
            "Install it with: python -m pip install scikit-learn"
        ) from exc
    return PLSRegression, mean_absolute_error, mean_squared_error


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


def _feature_group(feature: str) -> str:
    """Return the Problem 2 feature group."""
    return "deep" if feature in DEEP_FEATURE_COLUMNS else "surface"


def _candidate_flag(row: pd.Series) -> bool:
    """Return whether the paper used a candidate section."""
    if "candidate_used" in row.index:
        value = row["candidate_used"]
        if isinstance(value, str):
            return value.strip().lower() in {"true", "1", "yes"}
        return bool(value)
    text = str(row.get("candidate_sections_from_step12", "") or "")
    return bool(text.strip())


def _natural_sort_key(value: Any) -> list[Any]:
    """Natural sort key for paper IDs."""
    import re

    text = str(value)
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def build_pls_feature_set(
    feature_matrix: pd.DataFrame,
    variance_filter: pd.DataFrame,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Build and audit the feature set used by the main PLS model."""
    expected_features = [*SURFACE_FEATURE_COLUMNS, *DEEP_FEATURE_COLUMNS]
    variance_lookup = variance_filter.set_index("feature_name").to_dict(orient="index")

    length_corr = np.nan
    if {"total_chars", "page_count"}.issubset(feature_matrix.columns):
        length_corr = abs(float(pd.to_numeric(feature_matrix["total_chars"]).corr(pd.to_numeric(feature_matrix["page_count"]))))

    rows: list[dict[str, Any]] = []
    for feature in expected_features:
        info = variance_lookup.get(feature, {})
        variance_flag = str(info.get("variance_flag", "missing" if feature not in feature_matrix.columns else "normal"))
        step14_use = str(info.get("use_in_model", "False" if feature not in feature_matrix.columns else "True"))
        use_in_pls = True
        exclude_reason = ""
        caution_parts: list[str] = []

        if feature not in feature_matrix.columns:
            use_in_pls = False
            exclude_reason = "feature_missing_in_scaled_matrix"
        elif feature == "reference_norm_rate" or step14_use == "False" or variance_flag == "constant_feature":
            use_in_pls = False
            exclude_reason = "constant_feature_excluded_by_step14"
        elif feature == "appendix_code_presence":
            use_in_pls = False
            exclude_reason = "low_variance_feature_excluded_from_main_pls"
            caution_parts.append("near_constant_caution")
        elif variance_flag == "low_variance_feature" or step14_use == "Caution":
            caution_parts.append("low_variance_caution")

        if feature in NEGATIVE_FEATURES:
            caution_parts.append("negative_risk_feature")
        if feature in {"total_chars", "page_count"} and pd.notna(length_corr) and length_corr >= 0.8:
            caution_parts.append("possible_collinearity_length_feature")

        rows.append(
            {
                "feature_name": feature,
                "feature_group": _feature_group(feature),
                "use_in_pls": bool(use_in_pls),
                "exclude_reason": exclude_reason,
                "caution_flag": ";".join(caution_parts),
                "variance_flag": variance_flag,
                "step14_use_in_model": step14_use,
            }
        )

    if logger is not None:
        if pd.notna(length_corr):
            logger.info("total_chars/page_count absolute Pearson correlation on robust matrix: %.6f", length_corr)
        if pd.notna(length_corr) and length_corr >= 0.8:
            logger.info("Length features may be collinear; both are retained for auditability.")
        logger.info("PLS feature set rows: %s", rows)
    return pd.DataFrame(rows)


def _select_feature_columns(feature_set: pd.DataFrame) -> list[str]:
    """Return selected PLS feature columns."""
    return feature_set.loc[feature_set["use_in_pls"].astype(bool), "feature_name"].astype(str).tolist()


def _prepare_xy(
    feature_matrix: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare numeric X and y matrices for PLS."""
    missing = [feature for feature in feature_columns if feature not in feature_matrix.columns]
    if missing:
        raise ValueError(f"PLS feature columns missing from matrix: {missing}")
    if "Q_label" not in feature_matrix.columns:
        raise ValueError("appendix2_feature_matrix_robust_scaled.xlsx missing Q_label")

    x = feature_matrix[feature_columns].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(feature_matrix["Q_label"], errors="coerce")
    if x.isna().any().any():
        missing_counts = x.isna().sum()
        raise ValueError(f"PLS features contain NaN values: {missing_counts[missing_counts > 0].to_dict()}")
    if y.isna().any():
        raise ValueError("Q_label contains NaN values.")
    return x, y


def fit_pls_model(features: pd.DataFrame, quality_label: pd.Series, n_components: int):
    """Fit a PLS regression model with sklearn.cross_decomposition.PLSRegression."""
    PLSRegression, _, _ = _import_sklearn()
    # The input matrix is Step 14's robust-scaled matrix. Some binary/sparse
    # features can still become very large when MAD is zero, so sklearn's
    # internal scaling is kept on to balance feature columns before PLS fitting.
    model = PLSRegression(n_components=int(n_components), scale=True, max_iter=1000, tol=1e-06)
    model.fit(features.to_numpy(dtype=float), quality_label.to_numpy(dtype=float).reshape(-1, 1))
    return model


def _predict_pls(model: Any, features: pd.DataFrame) -> np.ndarray:
    """Predict with a fitted sklearn PLS model and flatten output."""
    return np.asarray(model.predict(features.to_numpy(dtype=float))).reshape(-1)


def _metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """Calculate small-sample prediction metrics."""
    _, mean_absolute_error, mean_squared_error = _import_sklearn()
    y_values = y_true.to_numpy(dtype=float)
    mae = float(mean_absolute_error(y_values, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_values, y_pred)))
    sst = float(np.sum((y_values - y_values.mean()) ** 2))
    sse = float(np.sum((y_values - y_pred) ** 2))
    r2_loo = np.nan if sst == 0 else 1.0 - sse / sst
    spearman = _spearman_corr(y_values, y_pred)
    return {"MAE": mae, "RMSE": rmse, "R2_LOO": float(r2_loo), "Spearman": float(spearman)}


def select_pls_components_by_loocv(
    features: pd.DataFrame,
    quality_label: pd.Series,
    grid: list[int],
    improvement_threshold: float = 0.05,
) -> tuple[int, pd.DataFrame, dict[int, np.ndarray]]:
    """Select PLS component count by LOOCV."""
    n_samples = len(features)
    predictions_by_component: dict[int, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    max_allowed = max(1, min(n_samples - 1, features.shape[1]))

    for candidate in sorted(set(int(value) for value in grid)):
        if candidate < 1 or candidate > max_allowed:
            continue
        preds = np.zeros(n_samples, dtype=float)
        for test_index in range(n_samples):
            train_mask = np.ones(n_samples, dtype=bool)
            train_mask[test_index] = False
            model = fit_pls_model(features.iloc[train_mask], quality_label.iloc[train_mask], candidate)
            preds[test_index] = _predict_pls(model, features.iloc[[test_index]])[0]

        metrics = _metrics(quality_label, preds)
        rows.append({"n_components": candidate, **metrics})
        predictions_by_component[candidate] = preds

    if not rows:
        raise ValueError("No valid PLS component candidate is available.")

    selection = pd.DataFrame(rows).sort_values("n_components").reset_index(drop=True)
    best_row = selection.sort_values(["RMSE", "n_components"], ascending=[True, True]).iloc[0]
    best_component = int(best_row["n_components"])

    if {1, 2}.issubset(set(selection["n_components"])):
        rmse_1 = float(selection.loc[selection["n_components"].eq(1), "RMSE"].iloc[0])
        rmse_2 = float(selection.loc[selection["n_components"].eq(2), "RMSE"].iloc[0])
        improvement = (rmse_1 - rmse_2) / rmse_1 if rmse_1 > 0 else 0.0
        if rmse_2 < rmse_1 and improvement < improvement_threshold:
            best_component = 1
            selection["selection_note"] = selection["n_components"].apply(
                lambda value: "selected_simpler_model_small_rmse_gain" if int(value) == 1 else "rmse_gain_below_threshold"
            )
        else:
            selection["selection_note"] = selection["n_components"].apply(
                lambda value: "selected_lowest_rmse" if int(value) == best_component else ""
            )
    else:
        selection["selection_note"] = selection["n_components"].apply(
            lambda value: "selected_lowest_rmse" if int(value) == best_component else ""
        )
    selection["selected"] = selection["n_components"].eq(best_component)
    return best_component, selection, predictions_by_component


def calculate_vip(pls_model: Any, features: pd.DataFrame, quality_label: pd.Series | None = None) -> pd.DataFrame:
    """Calculate VIP scores for a fitted PLS model."""
    t = np.asarray(pls_model.x_scores_, dtype=float)
    w = np.asarray(pls_model.x_weights_, dtype=float)
    q = np.asarray(pls_model.y_loadings_, dtype=float)
    if q.ndim == 1:
        q = q.reshape(1, -1)

    p = w.shape[0]
    component_count = w.shape[1]
    ss_y = np.zeros(component_count, dtype=float)
    for component in range(component_count):
        t_col = t[:, component]
        q_col = q[:, component]
        ss_y[component] = float(np.sum(t_col**2) * np.sum(q_col**2))
    total_ss = float(np.sum(ss_y))
    if total_ss == 0:
        vip_values = np.zeros(p, dtype=float)
    else:
        vip_values = np.zeros(p, dtype=float)
        for feature_index in range(p):
            weight_sum = 0.0
            for component in range(component_count):
                w_col = w[:, component]
                denom = float(np.sum(w_col**2))
                normalized_weight = 0.0 if denom == 0 else float(w[feature_index, component] ** 2 / denom)
                weight_sum += ss_y[component] * normalized_weight
            vip_values[feature_index] = float(np.sqrt(p * weight_sum / total_ss))
    return pd.DataFrame({"feature_name": list(features.columns), "VIP": vip_values})


def _build_prediction_table(feature_matrix: pd.DataFrame, y_true: pd.Series, y_pred: np.ndarray) -> pd.DataFrame:
    """Build LOOCV prediction result table."""
    table = feature_matrix[["paper_id", "filename"]].copy()
    table["Q_true"] = y_true.to_numpy(dtype=float)
    table["Q_pred_loo"] = y_pred
    table["residual"] = table["Q_true"] - table["Q_pred_loo"]
    table["abs_error"] = table["residual"].abs()
    table["rank_true"] = table["Q_true"].rank(ascending=False, method="first").astype(int)
    table["rank_pred"] = table["Q_pred_loo"].rank(ascending=False, method="first").astype(int)
    table["review_focus"] = feature_matrix.apply(lambda row: bool(row.get("review_focus", False)), axis=1)
    table["candidate_flag"] = feature_matrix.apply(_candidate_flag, axis=1)
    return table.sort_values("rank_true").reset_index(drop=True)


def _vip_interpretation(row: pd.Series) -> str:
    """Build one-sentence VIP interpretation."""
    feature = str(row["feature_name"])
    vip = float(row["VIP"])
    if feature in {"total_chars", "page_count"}:
        return "VIP靠前时解释为信息承载量与弱监督质量标签同步变化，不能解释为篇幅越长越好。"
    if feature in {"method_fit", "section_coverage", "task_coverage", "objective_constraint_completeness"}:
        return "VIP靠前支持方法匹配、任务覆盖与结构完整性是质量差异的重要来源。"
    if feature == "stacking_penalty":
        return "堆砌惩罚是负向风险特征；即使VIP不高，后续仍可作为QAF约束项。"
    if bool(row.get("important_by_vip")):
        return "VIP大于1，说明该特征在PLS潜变量中具有较强解释贡献。"
    return "VIP未超过1，作为辅助解释特征谨慎使用。"


def _build_vip_table(
    vip: pd.DataFrame,
    correlation: pd.DataFrame,
    key_pre: pd.DataFrame,
    feature_set: pd.DataFrame,
) -> pd.DataFrame:
    """Merge VIP with Step14/Step15 diagnostics."""
    table = vip.copy()
    table["feature_group"] = table["feature_name"].map(_feature_group)
    table["VIP_rank"] = table["VIP"].rank(ascending=False, method="first").astype(int)
    table["important_by_vip"] = table["VIP"].gt(1.0)
    table = table.merge(
        correlation[["feature_name", "spearman_corr"]],
        on="feature_name",
        how="left",
    )
    table = table.merge(
        key_pre[["feature_name", "grey_norm", "K_pre"]],
        on="feature_name",
        how="left",
    )
    table = table.merge(
        feature_set[["feature_name", "caution_flag"]],
        on="feature_name",
        how="left",
    )
    table["interpretation"] = table.apply(_vip_interpretation, axis=1)
    columns = [
        "feature_name",
        "feature_group",
        "VIP",
        "VIP_rank",
        "important_by_vip",
        "spearman_corr",
        "grey_norm",
        "K_pre",
        "caution_flag",
        "interpretation",
    ]
    return table[columns].sort_values("VIP_rank").reset_index(drop=True)


def _save_true_vs_pred(prediction: pd.DataFrame, chart_path: Path) -> None:
    """Save Q_true vs Q_pred LOOCV scatter chart."""
    chart_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=180)
    colors = np.where(prediction["review_focus"], "#c44e52", "#4f7cac")
    ax.scatter(prediction["Q_true"], prediction["Q_pred_loo"], c=colors, s=55, alpha=0.88, edgecolor="white", linewidth=0.8)
    min_value = float(min(prediction["Q_true"].min(), prediction["Q_pred_loo"].min()))
    max_value = float(max(prediction["Q_true"].max(), prediction["Q_pred_loo"].max()))
    padding = max((max_value - min_value) * 0.08, 1.0)
    ax.plot([min_value - padding, max_value + padding], [min_value - padding, max_value + padding], color="#444444", linestyle="--", linewidth=1)
    for _, row in prediction.iterrows():
        ax.text(row["Q_true"], row["Q_pred_loo"], str(row["paper_id"]), fontsize=7, ha="left", va="bottom")
    ax.set_xlabel("Q_true")
    ax.set_ylabel("Q_pred_LOOCV")
    ax.set_title("PLS LOOCV: True vs Predicted Q_label")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(chart_path)
    plt.close(fig)


def _save_vip_bar(vip_table: pd.DataFrame, chart_path: Path) -> None:
    """Save VIP Top 10 bar chart."""
    chart_path.parent.mkdir(parents=True, exist_ok=True)
    top = vip_table.sort_values("VIP", ascending=False).head(10).iloc[::-1]
    colors = ["#b64b4b" if feature in NEGATIVE_FEATURES else "#4f7cac" for feature in top["feature_name"]]
    fig, ax = plt.subplots(figsize=(9, 6), dpi=180)
    ax.barh(top["feature_name"], top["VIP"], color=colors)
    ax.axvline(1.0, color="#333333", linestyle="--", linewidth=1)
    ax.set_xlabel("VIP")
    ax.set_title("PLS VIP Top 10 Features")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(chart_path)
    plt.close(fig)


def run_step16_pls_vip_prediction(config_path: Path | None = None) -> dict[str, Any]:
    """Run Step 16 PLS-LOOCV and VIP feature importance analysis."""
    problem2_config = get_problem2_config(config_path)
    tables_dir = resolve_project_path(problem2_config["problem2_output_tables_dir"])
    charts_dir = resolve_project_path(problem2_config["problem2_output_charts_dir"])
    logs_dir = resolve_project_path(problem2_config["problem2_output_logs_dir"])
    tables_dir.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    log_path = logs_dir / "pls_vip_model.log"
    logger = setup_pls_logger(log_path)
    logger.info("Starting Step 16 PLS-VIP; this step does not run QAF, Bootstrap, or pairwise ranking checks.")
    logger.info("PLS uses Step14 robust-scaled X and sklearn PLSRegression(scale=True) for column balance.")

    feature_matrix = pd.read_excel(tables_dir / "appendix2_feature_matrix_robust_scaled.xlsx")
    q_labels = pd.read_excel(tables_dir / "appendix2_q_labels.xlsx")
    variance_filter = pd.read_excel(tables_dir / "appendix2_feature_variance_filter.xlsx")
    correlation = pd.read_excel(tables_dir / "appendix2_correlation_analysis.xlsx")
    key_pre = pd.read_excel(tables_dir / "appendix2_key_features_preliminary.xlsx")

    for frame in [feature_matrix, q_labels, variance_filter, correlation, key_pre]:
        if "paper_id" in frame.columns:
            frame["paper_id"] = frame["paper_id"].astype(str)
    feature_matrix = feature_matrix.sort_values("paper_id", key=lambda series: series.map(_natural_sort_key)).reset_index(drop=True)
    if {"paper_id", "Q_label"}.issubset(q_labels.columns):
        q_check = feature_matrix[["paper_id", "Q_label"]].merge(
            q_labels[["paper_id", "Q_label"]].rename(columns={"Q_label": "Q_label_from_q_file"}),
            on="paper_id",
            how="left",
        )
        mismatch = q_check.loc[
            ~np.isclose(
                pd.to_numeric(q_check["Q_label"], errors="coerce"),
                pd.to_numeric(q_check["Q_label_from_q_file"], errors="coerce"),
                equal_nan=False,
            )
        ]
        if not mismatch.empty:
            raise ValueError(f"Q_label mismatch between feature matrix and appendix2_q_labels.xlsx: {mismatch.to_dict(orient='records')}")
        logger.info("Q_label consistency check passed against appendix2_q_labels.xlsx.")

    feature_set = build_pls_feature_set(feature_matrix, variance_filter, logger=logger)
    feature_columns = _select_feature_columns(feature_set)
    x, y = _prepare_xy(feature_matrix, feature_columns)

    component_grid = [int(value) for value in problem2_config.get("pls_components_grid", [1, 2])]
    best_component, component_selection, predictions_by_component = select_pls_components_by_loocv(x, y, component_grid)
    best_prediction = predictions_by_component[best_component]
    prediction_table = _build_prediction_table(feature_matrix, y, best_prediction)

    final_model = fit_pls_model(x, y, best_component)
    vip_raw = calculate_vip(final_model, x, y)
    vip_table = _build_vip_table(vip_raw, correlation, key_pre, feature_set)

    component_path = tables_dir / "pls_component_selection.xlsx"
    prediction_path = tables_dir / "pls_prediction_results.xlsx"
    vip_path = tables_dir / "pls_vip_scores.xlsx"
    feature_set_path = tables_dir / "pls_model_feature_set.xlsx"
    true_pred_chart_path = charts_dir / "pls_true_vs_pred.png"
    vip_chart_path = charts_dir / "pls_vip_bar.png"

    component_selection.to_excel(component_path, index=False)
    prediction_table.to_excel(prediction_path, index=False)
    vip_table.to_excel(vip_path, index=False)
    feature_set.to_excel(feature_set_path, index=False)
    _save_true_vs_pred(prediction_table, true_pred_chart_path)
    _save_vip_bar(vip_table, vip_chart_path)

    metrics_row = component_selection.loc[component_selection["selected"]].iloc[0].to_dict()
    excluded = feature_set.loc[~feature_set["use_in_pls"].astype(bool), ["feature_name", "exclude_reason", "caution_flag"]]
    cautions = feature_set.loc[feature_set["caution_flag"].astype(str).ne(""), ["feature_name", "caution_flag"]]
    vip_top8 = vip_table.head(8)
    vip_gt1 = vip_table.loc[vip_table["important_by_vip"]]
    large_error_threshold = float(prediction_table["abs_error"].mean() + prediction_table["abs_error"].std(ddof=0))
    large_errors = prediction_table.loc[prediction_table["abs_error"].ge(large_error_threshold)].sort_values("abs_error", ascending=False)

    logger.info("Selected PLS feature count: %s", len(feature_columns))
    logger.info("Excluded features: %s", excluded.to_dict(orient="records"))
    logger.info("Caution features: %s", cautions.to_dict(orient="records"))
    logger.info("Component selection: %s", component_selection.to_dict(orient="records"))
    logger.info("Selected component metrics: %s", metrics_row)
    logger.info("VIP Top 8: %s", vip_top8.to_dict(orient="records"))
    logger.info("VIP > 1 features: %s", vip_gt1.to_dict(orient="records"))
    logger.info("Large prediction errors threshold=%.6f rows=%s", large_error_threshold, large_errors.to_dict(orient="records"))
    logger.info("Saved outputs: %s, %s, %s, %s, %s, %s", component_path, prediction_path, vip_path, feature_set_path, true_pred_chart_path, vip_chart_path)
    logger.info("Finished Step 16")

    return {
        "feature_set": feature_set,
        "feature_columns": feature_columns,
        "component_selection": component_selection,
        "best_component": best_component,
        "metrics_row": metrics_row,
        "prediction_table": prediction_table,
        "vip_table": vip_table,
        "vip_top8": vip_top8,
        "vip_gt1": vip_gt1,
        "excluded": excluded,
        "cautions": cautions,
        "large_errors": large_errors,
        "component_path": component_path,
        "prediction_path": prediction_path,
        "vip_path": vip_path,
        "feature_set_path": feature_set_path,
        "true_pred_chart_path": true_pred_chart_path,
        "vip_chart_path": vip_chart_path,
        "log_path": log_path,
    }
