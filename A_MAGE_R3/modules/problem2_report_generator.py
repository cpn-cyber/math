"""Problem 2 final audit and report helpers."""

from __future__ import annotations

from pathlib import Path
import logging
from typing import Any

import numpy as np
import pandas as pd

from modules.appendix2_pipeline import get_problem2_config, resolve_project_path


LOGGER_NAME = "A_MAGE_R3.problem2.final_audit"
EXPECTED_Q_SOURCE = "Problem1 sealed evaluation system"
EXPECTED_LABEL_NOTE = "weak-supervised label, not official ground truth"
EXPECTED_HIGH_INFLUENCE = {"2-1", "2-3", "2-4", "2-6", "2-8", "2-10"}
EXPECTED_FINAL_KEYS = [
    "total_chars",
    "method_fit",
    "page_count",
    "section_coverage",
    "objective_constraint_completeness",
    "task_coverage",
    "figure_table_explanation_rate",
    "conclusion_echo_rate",
]
EXPECTED_CHARTS = [
    "appendix2_feature_heatmap.png",
    "appendix2_correlation_bar.png",
    "appendix2_q_distribution.png",
    "grey_relation_bar.png",
    "key_features_preliminary_bar.png",
    "pls_true_vs_pred.png",
    "pls_vip_bar.png",
    "qaf_waterfall.png",
    "qaf_true_vs_pred.png",
    "qaf_before_after_error.png",
    "bootstrap_vip_boxplot.png",
    "key_feature_index_final.png",
    "delete_one_sensitivity.png",
    "pairwise_ranking_accuracy_bar.png",
    "pairwise_feature_weight_bar.png",
]
INPUT_TABLES = [
    "appendix2_q_labels.xlsx",
    "appendix2_features_with_q.xlsx",
    "appendix2_surface_features_raw.xlsx",
    "appendix2_surface_features_normalized.xlsx",
    "appendix2_candidate_usage_report.xlsx",
    "appendix2_step12_quality_audit.xlsx",
    "appendix2_deep_quality_features_auto.xlsx",
    "appendix2_deep_quality_review_template.xlsx",
    "appendix2_deep_quality_evidence.xlsx",
    "appendix2_feature_matrix_raw.xlsx",
    "appendix2_feature_matrix_robust_scaled.xlsx",
    "appendix2_correlation_analysis.xlsx",
    "appendix2_feature_variance_filter.xlsx",
    "appendix2_grey_relation.xlsx",
    "appendix2_key_features_preliminary.xlsx",
    "pls_component_selection.xlsx",
    "pls_prediction_results.xlsx",
    "pls_vip_scores.xlsx",
    "pls_model_feature_set.xlsx",
    "qaf_scores.xlsx",
    "qaf_eta_selection.xlsx",
    "qaf_prediction_results.xlsx",
    "qaf_adjustment_audit.xlsx",
    "bootstrap_vip_stability.xlsx",
    "delete_one_sensitivity.xlsx",
    "key_feature_index_final.xlsx",
    "model_stability_summary.xlsx",
    "high_influence_samples.xlsx",
    "pairwise_ranking_dataset.xlsx",
    "pairwise_ranking_check.xlsx",
    "pairwise_group_validation.xlsx",
]
PAPER_LEVEL_TABLES = [
    "appendix2_q_labels.xlsx",
    "appendix2_features_with_q.xlsx",
    "appendix2_surface_features_raw.xlsx",
    "appendix2_surface_features_normalized.xlsx",
    "appendix2_deep_quality_features_auto.xlsx",
    "appendix2_feature_matrix_raw.xlsx",
    "appendix2_feature_matrix_robust_scaled.xlsx",
    "pls_prediction_results.xlsx",
    "qaf_scores.xlsx",
    "qaf_prediction_results.xlsx",
]


def _setup_logger(log_path: Path) -> logging.Logger:
    """Configure final audit logger."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(LOGGER_NAME)
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(handler)
    return logger


def _status(condition: bool) -> str:
    """Return PASS/FAIL."""
    return "PASS" if bool(condition) else "FAIL"


def _warn_status(condition: bool) -> str:
    """Return PASS/WARN."""
    return "PASS" if bool(condition) else "WARN"


def _read_excel(path: Path, sheet_name: str | int = 0) -> pd.DataFrame:
    """Read an Excel file with pandas."""
    return pd.read_excel(path, sheet_name=sheet_name)


def _natural_sort_key(value: Any) -> list[Any]:
    """Natural sort key for paper IDs."""
    import re

    text = str(value)
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def _paper_ids(df: pd.DataFrame, column: str = "paper_id") -> list[str]:
    """Return sorted paper IDs from a table."""
    if column not in df.columns:
        return []
    return sorted(df[column].dropna().astype(str).tolist(), key=_natural_sort_key)


def _check_float(value: Any, expected: float, tolerance: float = 1e-6) -> bool:
    """Compare numeric values with tolerance."""
    try:
        return abs(float(value) - float(expected)) <= tolerance
    except Exception:
        return False


def _make_audit_row(category: str, item: str, status: str, details: str) -> dict[str, Any]:
    """Build one audit row."""
    return {"category": category, "item": item, "status": status, "details": details}


def _file_integrity(tables_dir: Path, charts_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    """Check table and chart file presence/non-empty."""
    rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for name in INPUT_TABLES:
        path = tables_dir / name
        exists = path.exists()
        size = path.stat().st_size if exists else 0
        rows.append({"file_name": name, "path": str(path), "exists": exists, "size_bytes": size, "non_empty": size > 0})
        audit_rows.append(_make_audit_row("file_integrity", name, _status(exists and size > 0), f"exists={exists}, size={size}"))

    chart_rows: list[dict[str, Any]] = []
    for name in EXPECTED_CHARTS:
        path = charts_dir / name
        exists = path.exists()
        size = path.stat().st_size if exists else 0
        chart_rows.append({"chart_name": name, "path": str(path), "exists": exists, "size_bytes": size, "non_empty": size > 0})
        audit_rows.append(_make_audit_row("chart_integrity", name, _status(exists and size > 0), f"exists={exists}, size={size}"))
    return pd.DataFrame(rows), pd.DataFrame(chart_rows), audit_rows


def _sample_consistency(tables_dir: Path) -> tuple[pd.DataFrame, list[dict[str, Any]], list[str]]:
    """Check paper-level table consistency."""
    expected_ids = [f"2-{i}" for i in range(1, 11)]
    q_labels = _read_excel(tables_dir / "appendix2_q_labels.xlsx")
    q_lookup = q_labels.set_index(q_labels["paper_id"].astype(str))["Q_label"].to_dict()
    rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for name in PAPER_LEVEL_TABLES:
        path = tables_dir / name
        df = _read_excel(path)
        ids = _paper_ids(df)
        missing = sorted(set(expected_ids) - set(ids), key=_natural_sort_key)
        extra = sorted(set(ids) - set(expected_ids), key=_natural_sort_key)
        duplicate_count = int(df["paper_id"].duplicated().sum()) if "paper_id" in df.columns else np.nan
        order_ok = ids == expected_ids
        q_consistent = True
        q_columns = [column for column in ["Q_label", "Q_true"] if column in df.columns]
        mismatches: list[str] = []
        for _, row in df.iterrows():
            paper_id = str(row.get("paper_id", ""))
            if paper_id not in q_lookup:
                continue
            for column in q_columns:
                if not _check_float(row[column], q_lookup[paper_id], tolerance=1e-6):
                    q_consistent = False
                    mismatches.append(f"{paper_id}:{column}")
        has_review_focus = "review_focus" in df.columns
        has_candidate = bool({"candidate_flag", "candidate_used", "candidate_sections_from_step12", "candidate_sections"}.intersection(df.columns))
        ok = len(ids) == 10 and not missing and not extra and duplicate_count == 0 and q_consistent
        rows.append(
            {
                "file_name": name,
                "row_count": len(df),
                "paper_id_count": len(ids),
                "missing_ids": ",".join(missing),
                "extra_ids": ",".join(extra),
                "duplicate_count": duplicate_count,
                "order_ok": order_ok,
                "q_label_consistent": q_consistent,
                "q_label_mismatch": ",".join(mismatches),
                "review_focus_retained": has_review_focus,
                "candidate_marker_retained": has_candidate,
            }
        )
        audit_rows.append(_make_audit_row("sample_consistency", name, _status(ok), f"ids={len(ids)}, missing={missing}, extra={extra}, duplicates={duplicate_count}, q_consistent={q_consistent}"))
    return pd.DataFrame(rows), audit_rows, expected_ids


def _label_checks(tables_dir: Path) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Audit weak-label source and risky wording."""
    rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    q_labels = _read_excel(tables_dir / "appendix2_q_labels.xlsx")
    q_source_ok = bool(q_labels["Q_source"].astype(str).eq(EXPECTED_Q_SOURCE).all())
    label_note_ok = bool(q_labels["label_note"].astype(str).eq(EXPECTED_LABEL_NOTE).all())
    rows.append({"check_item": "Q_source", "status": _status(q_source_ok), "details": EXPECTED_Q_SOURCE})
    rows.append({"check_item": "label_note", "status": _status(label_note_ok), "details": EXPECTED_LABEL_NOTE})
    audit_rows.append(_make_audit_row("label_expression", "Q_source", _status(q_source_ok), EXPECTED_Q_SOURCE))
    audit_rows.append(_make_audit_row("label_expression", "label_note", _status(label_note_ok), EXPECTED_LABEL_NOTE))

    risky_records: list[dict[str, Any]] = []
    risky_terms = ["human score", "expert score", "official score", "official quality", "official label", "ground truth"]
    for name in INPUT_TABLES:
        path = tables_dir / name
        if not path.exists() or path.stat().st_size == 0:
            continue
        try:
            sheets = pd.read_excel(path, sheet_name=None)
        except Exception:
            continue
        for sheet_name, df in sheets.items():
            for column in df.columns:
                column_text = str(column).lower()
                if any(term in column_text for term in risky_terms):
                    risky_records.append({"file_name": name, "sheet": sheet_name, "field": str(column), "value": "", "risk": "risky_column_name"})
            object_df = df.select_dtypes(include=["object"])
            for column in object_df.columns:
                for value in object_df[column].dropna().astype(str).unique():
                    lower = value.lower()
                    if lower == EXPECTED_LABEL_NOTE.lower() or "not official" in lower:
                        continue
                    if any(term in lower for term in risky_terms):
                        risky_records.append({"file_name": name, "sheet": sheet_name, "field": str(column), "value": value[:300], "risk": "risky_text"})
    risky = pd.DataFrame(risky_records, columns=["file_name", "sheet", "field", "value", "risk"])
    risky_ok = risky.empty
    rows.append({"check_item": "risky_label_wording", "status": _status(risky_ok), "details": "No output writes Q_i as official/human/expert/ground-truth score." if risky_ok else f"{len(risky)} risky entries"})
    audit_rows.append(_make_audit_row("label_expression", "risky_label_wording", _status(risky_ok), f"risky_entries={len(risky)}"))
    if not risky.empty:
        rows = rows + risky.to_dict(orient="records")
    return pd.DataFrame(rows), audit_rows


def _feature_usage_checks(tables_dir: Path) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Audit special feature handling."""
    rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    variance = _read_excel(tables_dir / "appendix2_feature_variance_filter.xlsx")
    pls_set = _read_excel(tables_dir / "pls_model_feature_set.xlsx")
    key_final = _read_excel(tables_dir / "key_feature_index_final.xlsx")
    vip = _read_excel(tables_dir / "pls_vip_scores.xlsx")

    def add(item: str, condition: bool, details: str) -> None:
        rows.append({"check_item": item, "status": _status(condition), "details": details})
        audit_rows.append(_make_audit_row("feature_usage", item, _status(condition), details))

    ref_var = variance.loc[variance["feature_name"].eq("reference_norm_rate")]
    ref_constant = not ref_var.empty and str(ref_var["variance_flag"].iloc[0]) == "constant_feature"
    add("reference_norm_rate_constant", ref_constant, ref_var.to_dict(orient="records"))

    ref_pls = pls_set.loc[pls_set["feature_name"].eq("reference_norm_rate")]
    ref_not_pls = not ref_pls.empty and not bool(ref_pls["use_in_pls"].iloc[0])
    ref_key = key_final.loc[key_final["feature_name"].eq("reference_norm_rate")]
    ref_not_key = not ref_key.empty and pd.isna(ref_key["K_final"].iloc[0]) and not bool(ref_key["final_key_feature"].iloc[0])
    add("reference_norm_rate_excluded_from_pls_and_kfinal", ref_not_pls and ref_not_key, f"pls={ref_pls.to_dict(orient='records')}, kfinal={ref_key.to_dict(orient='records')}")

    appendix_var = variance.loc[variance["feature_name"].eq("appendix_code_presence")]
    appendix_pls = pls_set.loc[pls_set["feature_name"].eq("appendix_code_presence")]
    appendix_ok = (
        not appendix_var.empty
        and str(appendix_var["variance_flag"].iloc[0]) == "low_variance_feature"
        and not appendix_pls.empty
        and not bool(appendix_pls["use_in_pls"].iloc[0])
    )
    add("appendix_code_presence_caution_or_excluded", appendix_ok, f"variance={appendix_var.to_dict(orient='records')}, pls={appendix_pls.to_dict(orient='records')}")

    stacking_vip = vip.loc[vip["feature_name"].eq("stacking_penalty")]
    stacking_key = key_final.loc[key_final["feature_name"].eq("stacking_penalty")]
    stacking_ok = (
        not stacking_vip.empty
        and "negative" in str(stacking_vip.get("caution_flag", pd.Series([""])).iloc[0]).lower()
        and not stacking_key.empty
        and "negative" in str(stacking_key["interpretation"].iloc[0]).lower()
    )
    add("stacking_penalty_negative_risk", stacking_ok, f"vip={stacking_vip.to_dict(orient='records')}, kfinal={stacking_key.to_dict(orient='records')}")

    length_rows = key_final.loc[key_final["feature_name"].isin(["total_chars", "page_count"])]
    length_ok = not length_rows.empty and length_rows["interpretation"].astype(str).str.contains("not longer-is-better", case=False, regex=False).all()
    add("length_feature_interpretation", length_ok, length_rows[["feature_name", "interpretation"]].to_dict(orient="records"))
    return pd.DataFrame(rows), audit_rows


def _model_consistency_checks(tables_dir: Path) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Audit model result consistency with sealed Step16-Step19 facts."""
    rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []

    def add(item: str, condition: bool, details: str) -> None:
        rows.append({"check_item": item, "status": _status(condition), "details": details})
        audit_rows.append(_make_audit_row("model_consistency", item, _status(condition), details))

    q = _read_excel(tables_dir / "appendix2_q_labels.xlsx")
    q_min = float(q["Q_label"].min())
    q_max = float(q["Q_label"].max())
    add("q_label_range", _check_float(q_min, 35.892975, 1e-5) and _check_float(q_max, 70.222043, 1e-5), f"{q_min:.6f}~{q_max:.6f}")

    pls = _read_excel(tables_dir / "pls_component_selection.xlsx")
    pls_selected = pls.loc[pls["selected"].astype(bool)].iloc[0]
    pls_ok = (
        int(pls_selected["n_components"]) == 1
        and _check_float(pls_selected["MAE"], 9.196813, 1e-6)
        and _check_float(pls_selected["RMSE"], 11.176952, 1e-6)
        and _check_float(pls_selected["R2_LOO"], -0.344322, 1e-6)
        and _check_float(pls_selected["Spearman"], 0.393939, 1e-6)
    )
    add("pls_metrics", pls_ok, pls_selected.to_dict())

    qaf = _read_excel(tables_dir / "qaf_eta_selection.xlsx")
    qaf_selected = qaf.loc[qaf["selected"].astype(bool)].iloc[0]
    qaf_scores = _read_excel(tables_dir / "qaf_prediction_results.xlsx")
    qaf_ok = (
        _check_float(qaf_selected["eta"], 0.05, 1e-9)
        and _check_float(qaf_scores["phi_i"].min(), 0.973304, 1e-6)
        and _check_float(qaf_scores["phi_i"].max(), 1.018579, 1e-6)
        and _check_float(qaf_selected["MAE"], 9.122450, 1e-6)
        and _check_float(qaf_selected["RMSE"], 11.049784, 1e-6)
        and _check_float(qaf_selected["R2"], -0.313906, 1e-6)
        and _check_float(qaf_selected["Spearman"], 0.406061, 1e-6)
    )
    add("qaf_metrics", qaf_ok, {**qaf_selected.to_dict(), "phi_min": qaf_scores["phi_i"].min(), "phi_max": qaf_scores["phi_i"].max()})

    boot = _read_excel(tables_dir / "bootstrap_vip_stability.xlsx")
    bootstrap_ok = int(boot["valid_bootstrap_count"].min()) == 1000 and int(boot["valid_bootstrap_count"].max()) == 1000
    add("bootstrap_valid_count", bootstrap_ok, f"min={boot['valid_bootstrap_count'].min()}, max={boot['valid_bootstrap_count'].max()}")

    kfinal = _read_excel(tables_dir / "key_feature_index_final.xlsx")
    top8 = kfinal.dropna(subset=["K_rank"]).sort_values("K_rank").head(8)["feature_name"].astype(str).tolist()
    final_keys = kfinal.loc[kfinal["final_key_feature"].astype(bool), "feature_name"].astype(str).tolist()
    add("kfinal_top8", top8 == EXPECTED_FINAL_KEYS, ", ".join(top8))
    add("final_key_feature", final_keys == EXPECTED_FINAL_KEYS, ", ".join(final_keys))

    high = _read_excel(tables_dir / "high_influence_samples.xlsx")
    high_set = set(high["removed_paper_id"].astype(str).tolist())
    add("high_influence_samples", EXPECTED_HIGH_INFLUENCE.issubset(high_set), ", ".join(sorted(high_set, key=_natural_sort_key)))

    pairwise = _read_excel(tables_dir / "pairwise_ranking_check.xlsx").iloc[0]
    pair_ok = (
        int(pairwise["total_pairs"]) == 45
        and int(pairwise["near_tie_pairs"]) == 3
        and _check_float(pairwise["overall_pairwise_accuracy"], 0.622222, 1e-6)
        and _check_float(pairwise["group_validation_accuracy_mean"], 0.622222, 1e-6)
        and _check_float(pairwise["accuracy_without_near_tie"], 0.595238, 1e-6)
    )
    group = _read_excel(tables_dir / "pairwise_group_validation.xlsx")
    group_21 = group.loc[group["heldout_paper"].astype(str).eq("2-1")]
    pair_21_ok = not group_21.empty and _check_float(group_21["pairwise_accuracy"].iloc[0], 0.888889, 1e-6)
    add("pairwise_accuracy", pair_ok and pair_21_ok, {**pairwise.to_dict(), "paper_2_1_group": group_21.to_dict(orient="records")})
    return pd.DataFrame(rows), audit_rows


def _key_results(tables_dir: Path) -> pd.DataFrame:
    """Collect key result facts for markdown and audit workbook."""
    rows: list[dict[str, Any]] = []
    q = _read_excel(tables_dir / "appendix2_q_labels.xlsx")
    rows.append({"section": "Q_label", "item": "range", "value": f"{q['Q_label'].min():.6f} ~ {q['Q_label'].max():.6f}"})

    corr = _read_excel(tables_dir / "appendix2_correlation_analysis.xlsx")
    corr_top = corr.loc[corr["use_in_model"].astype(str).ne("False")].sort_values("spearman_abs", ascending=False).head(8)
    rows.append({"section": "Step14", "item": "Spearman Top 8", "value": "; ".join(f"{r.feature_name}:{r.spearman_corr:.6f}" for r in corr_top.itertuples())})

    kpre = _read_excel(tables_dir / "appendix2_key_features_preliminary.xlsx")
    kpre_top = kpre.dropna(subset=["K_pre"]).sort_values("K_pre", ascending=False).head(8)
    rows.append({"section": "Step15", "item": "K_pre Top 8", "value": "; ".join(f"{r.feature_name}:{r.K_pre:.6f}" for r in kpre_top.itertuples())})

    pls = _read_excel(tables_dir / "pls_component_selection.xlsx")
    pls_selected = pls.loc[pls["selected"].astype(bool)].iloc[0]
    rows.append({"section": "Step16", "item": "PLS metrics", "value": f"A={int(pls_selected['n_components'])}, MAE={pls_selected['MAE']:.6f}, RMSE={pls_selected['RMSE']:.6f}, R2_LOO={pls_selected['R2_LOO']:.6f}, Spearman={pls_selected['Spearman']:.6f}"})
    vip_top = _read_excel(tables_dir / "pls_vip_scores.xlsx").sort_values("VIP_rank").head(8)
    rows.append({"section": "Step16", "item": "VIP Top 8", "value": "; ".join(f"{r.feature_name}:{r.VIP:.6f}" for r in vip_top.itertuples())})

    qaf = _read_excel(tables_dir / "qaf_eta_selection.xlsx")
    qaf_selected = qaf.loc[qaf["selected"].astype(bool)].iloc[0]
    rows.append({"section": "Step17", "item": "QAF metrics", "value": f"eta={qaf_selected['eta']:.2f}, MAE={qaf_selected['MAE']:.6f}, RMSE={qaf_selected['RMSE']:.6f}, R2={qaf_selected['R2']:.6f}, Spearman={qaf_selected['Spearman']:.6f}"})

    kfinal = _read_excel(tables_dir / "key_feature_index_final.xlsx")
    kfinal_top = kfinal.dropna(subset=["K_rank"]).sort_values("K_rank").head(8)
    rows.append({"section": "Step18", "item": "K_final Top 8", "value": "; ".join(f"{r.feature_name}:{r.K_final:.6f}" for r in kfinal_top.itertuples())})
    final_keys = kfinal.loc[kfinal["final_key_feature"].astype(bool), "feature_name"].astype(str).tolist()
    rows.append({"section": "Step18", "item": "final_key_feature", "value": ", ".join(final_keys)})

    pair = _read_excel(tables_dir / "pairwise_ranking_check.xlsx").iloc[0]
    rows.append({"section": "Step19", "item": "pairwise accuracy", "value": f"total_pairs={int(pair['total_pairs'])}, near_tie={int(pair['near_tie_pairs'])}, overall={pair['overall_pairwise_accuracy']:.6f}, group_mean={pair['group_validation_accuracy_mean']:.6f}, no_near_tie={pair['accuracy_without_near_tie']:.6f}"})
    rows.append({"section": "Conclusion", "item": "writeable conclusion", "value": "Key features have moderate auxiliary ranking support; PLS/QAF are limited and auditable, not strong prediction models."})
    return pd.DataFrame(rows)


def _warnings_table() -> pd.DataFrame:
    """Return writing warning rows."""
    warnings = [
        ("Q_i is a weak-supervised label from the Problem 1 sealed intelligent evaluation system, not official ground truth."),
        ("PLS/QAF prediction ability is limited; do not describe them as strong prediction models."),
        ("R2_LOO is negative and must be reported honestly."),
        ("QAF is a conservative calibration with small improvement, not a significant performance boost."),
        ("Pairwise ranking is auxiliary validation only; 45 pairs are not 45 independent samples."),
        ("total_chars and page_count represent information-carrying amount/completeness, not longer-is-better."),
        ("There are multiple high-influence samples, so conclusions should be conservative."),
        ("Small-sample stability is auditable but limited."),
    ]
    return pd.DataFrame({"warning_id": range(1, len(warnings) + 1), "warning": warnings})


def _write_markdown(key_results: pd.DataFrame, warnings: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    """Write key-result and warning markdown files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    key_path = output_dir / "problem2_key_results.md"
    warnings_path = output_dir / "problem2_writing_warnings.md"

    key_lines = [
        "# 第二问关键结果汇总",
        "",
        "本文件只汇总 Step 12-Step 19 已生成的真实结果，不重新计算模型。",
        "",
    ]
    for section, group in key_results.groupby("section", sort=False):
        key_lines.append(f"## {section}")
        for row in group.itertuples(index=False):
            key_lines.append(f"- **{row.item}**：{row.value}")
        key_lines.append("")
    key_lines.extend(
        [
            "## 最终可写结论",
            "- 第二问以第一问封版智能评估系统输出的 `Q_label` 作为弱监督质量标签，而非官方真实质量分。",
            "- `method_fit`、`section_coverage`、`task_coverage`、`objective_constraint_completeness`、`figure_table_explanation_rate` 等指标对质量差异具有较稳定解释力。",
            "- `total_chars` 与 `page_count` 只能解释为信息承载量和完整性相关，不代表篇幅越长越好。",
            "- PLS 和 QAF 的预测能力有限，应作为关键特征解释、稳健性审计和保守校正工具，而非强预测模型。",
            "- 成对排序辅助检验为中等支持，不能写成强验证。",
            "",
        ]
    )
    key_path.write_text("\n".join(key_lines), encoding="utf-8")

    warning_lines = [
        "# 第二问论文写作风险提醒",
        "",
        "以下提醒必须在论文写作时遵守，避免把弱监督小样本模型写得过强。",
        "",
    ]
    for row in warnings.itertuples(index=False):
        warning_lines.append(f"{row.warning_id}. {row.warning}")
    warning_lines.append("")
    warnings_path.write_text("\n".join(warning_lines), encoding="utf-8")
    return key_path, warnings_path


def _safe_excel(tables_dir: Path, name: str, sheet_name: str | int = 0) -> pd.DataFrame:
    """Read an Excel table if available; otherwise return an empty frame."""
    path = tables_dir / name
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_excel(path, sheet_name=sheet_name)
    except Exception:
        return pd.DataFrame()


def _fmt(value: Any, digits: int = 6) -> str:
    """Format a numeric value or return 待补充."""
    try:
        if pd.isna(value):
            return "待补充"
        return f"{float(value):.{digits}f}"
    except Exception:
        text = str(value)
        return text if text else "待补充"


def _top_items(df: pd.DataFrame, name_col: str, value_col: str, n: int = 8, ascending: bool = False) -> list[tuple[str, float]]:
    """Return top feature/value tuples from a table."""
    if df.empty or name_col not in df.columns or value_col not in df.columns:
        return []
    work = df.copy()
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work = work.dropna(subset=[value_col]).sort_values(value_col, ascending=ascending).head(n)
    return [(str(row[name_col]), float(row[value_col])) for _, row in work.iterrows()]


def _top_text(items: list[tuple[str, float]]) -> str:
    """Format top feature list."""
    if not items:
        return "待补充"
    return "、".join(f"{name}={value:.6f}" for name, value in items)


def _names_text(items: list[tuple[str, float]]) -> str:
    """Format feature-name list."""
    if not items:
        return "待补充"
    return "、".join(name for name, _ in items)


def _write_text(path: Path, lines: list[str]) -> None:
    """Write markdown lines as UTF-8."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _collect_problem2_results(tables_dir: Path) -> dict[str, Any]:
    """Collect Step 12-Step 20 real values for paper writing."""
    q = _safe_excel(tables_dir, "appendix2_q_labels.xlsx")
    corr = _safe_excel(tables_dir, "appendix2_correlation_analysis.xlsx")
    grey = _safe_excel(tables_dir, "appendix2_grey_relation.xlsx")
    kpre = _safe_excel(tables_dir, "appendix2_key_features_preliminary.xlsx")
    pls = _safe_excel(tables_dir, "pls_component_selection.xlsx")
    vip = _safe_excel(tables_dir, "pls_vip_scores.xlsx")
    pls_pred = _safe_excel(tables_dir, "pls_prediction_results.xlsx")
    qaf_eta = _safe_excel(tables_dir, "qaf_eta_selection.xlsx")
    qaf_pred = _safe_excel(tables_dir, "qaf_prediction_results.xlsx")
    bootstrap = _safe_excel(tables_dir, "bootstrap_vip_stability.xlsx")
    kfinal = _safe_excel(tables_dir, "key_feature_index_final.xlsx")
    delete_one = _safe_excel(tables_dir, "delete_one_sensitivity.xlsx")
    pairwise = _safe_excel(tables_dir, "pairwise_ranking_check.xlsx")
    group = _safe_excel(tables_dir, "pairwise_group_validation.xlsx")

    q_min = float(q["Q_label"].min()) if not q.empty and "Q_label" in q.columns else np.nan
    q_max = float(q["Q_label"].max()) if not q.empty and "Q_label" in q.columns else np.nan
    q_source = str(q["Q_source"].iloc[0]) if not q.empty and "Q_source" in q.columns else "待补充"
    label_note = str(q["label_note"].iloc[0]) if not q.empty and "label_note" in q.columns else "待补充"

    if not corr.empty and {"feature_name", "spearman_corr"}.issubset(corr.columns):
        spearman_values = corr.set_index("feature_name")["spearman_corr"].to_dict()
    else:
        spearman_values = {}
    corr_top = corr.loc[corr["use_in_model"].astype(str).ne("False")].copy() if not corr.empty and "use_in_model" in corr.columns else corr.copy()
    if not corr_top.empty and {"feature_name", "spearman_abs", "spearman_corr"}.issubset(corr_top.columns):
        corr_top["spearman_abs"] = pd.to_numeric(corr_top["spearman_abs"], errors="coerce")
        spearman_top8 = [
            (str(row["feature_name"]), float(row["spearman_corr"]))
            for _, row in corr_top.dropna(subset=["spearman_abs"]).sort_values("spearman_abs", ascending=False).head(8).iterrows()
        ]
    else:
        spearman_top8 = []

    grey_top8 = _top_items(
        grey.loc[grey["use_in_model"].astype(str).ne("False")] if not grey.empty and "use_in_model" in grey.columns else grey,
        "feature_name",
        "grey_relation_score",
        8,
        ascending=False,
    )
    kpre_top8 = _top_items(kpre.dropna(subset=["K_pre"]) if not kpre.empty and "K_pre" in kpre.columns else kpre, "feature_name", "K_pre", 8, ascending=False)

    pls_selected = pls.loc[pls["selected"].astype(bool)].iloc[0] if not pls.empty and "selected" in pls.columns and pls["selected"].any() else pd.Series(dtype=object)
    vip_top8 = _top_items(vip, "feature_name", "VIP", 8, ascending=False)
    vip_gt1 = vip.loc[pd.to_numeric(vip.get("VIP", pd.Series(dtype=float)), errors="coerce").gt(1), "feature_name"].astype(str).tolist() if not vip.empty and "VIP" in vip.columns else []
    paper_21_error = np.nan
    if not pls_pred.empty and {"paper_id", "abs_error"}.issubset(pls_pred.columns):
        row = pls_pred.loc[pls_pred["paper_id"].astype(str).eq("2-1")]
        if not row.empty:
            paper_21_error = float(row["abs_error"].iloc[0])

    qaf_selected = qaf_eta.loc[qaf_eta["selected"].astype(bool)].iloc[0] if not qaf_eta.empty and "selected" in qaf_eta.columns and qaf_eta["selected"].any() else pd.Series(dtype=object)
    phi_min = float(qaf_pred["phi_i"].min()) if not qaf_pred.empty and "phi_i" in qaf_pred.columns else np.nan
    phi_max = float(qaf_pred["phi_i"].max()) if not qaf_pred.empty and "phi_i" in qaf_pred.columns else np.nan

    stable_bootstrap = bootstrap.loc[bootstrap["bootstrap_selected"].astype(bool)] if not bootstrap.empty and "bootstrap_selected" in bootstrap.columns else pd.DataFrame()
    kfinal_top8 = kfinal.dropna(subset=["K_rank"]).sort_values("K_rank").head(8) if not kfinal.empty and "K_rank" in kfinal.columns else pd.DataFrame()
    final_key_features = kfinal.loc[kfinal["final_key_feature"].astype(bool), "feature_name"].astype(str).tolist() if not kfinal.empty and "final_key_feature" in kfinal.columns else []
    high_influence = delete_one.loc[delete_one["high_influence_sample"].astype(bool), "removed_paper_id"].astype(str).tolist() if not delete_one.empty and "high_influence_sample" in delete_one.columns else []
    delete_21 = delete_one.loc[delete_one["removed_paper_id"].astype(str).eq("2-1")].iloc[0] if not delete_one.empty and "removed_paper_id" in delete_one.columns and delete_one["removed_paper_id"].astype(str).eq("2-1").any() else pd.Series(dtype=object)

    pair = pairwise.iloc[0] if not pairwise.empty else pd.Series(dtype=object)
    group_21 = group.loc[group["heldout_paper"].astype(str).eq("2-1")].iloc[0] if not group.empty and "heldout_paper" in group.columns and group["heldout_paper"].astype(str).eq("2-1").any() else pd.Series(dtype=object)

    return {
        "q_min": q_min,
        "q_max": q_max,
        "q_source": q_source,
        "label_note": label_note,
        "spearman_top8": spearman_top8,
        "spearman_values": spearman_values,
        "grey_top8": grey_top8,
        "kpre_top8": kpre_top8,
        "pls_selected": pls_selected,
        "vip_top8": vip_top8,
        "vip_gt1": vip_gt1,
        "paper_21_error": paper_21_error,
        "qaf_selected": qaf_selected,
        "phi_min": phi_min,
        "phi_max": phi_max,
        "stable_bootstrap": stable_bootstrap,
        "kfinal_top8": kfinal_top8,
        "final_key_features": final_key_features,
        "high_influence": high_influence,
        "delete_21": delete_21,
        "pair": pair,
        "group_21": group_21,
    }


def _draft_lines(result: dict[str, Any]) -> list[str]:
    """Build Problem 2 draft markdown."""
    pls = result["pls_selected"]
    qaf = result["qaf_selected"]
    delete_21 = result["delete_21"]
    pair = result["pair"]
    group_21 = result["group_21"]
    spearman = result["spearman_values"]
    stacking_spearman = spearman.get("stacking_penalty", np.nan)

    stable_lines = []
    if isinstance(result["stable_bootstrap"], pd.DataFrame) and not result["stable_bootstrap"].empty:
        for row in result["stable_bootstrap"].sort_values("mean_VIP", ascending=False).itertuples(index=False):
            stable_lines.append(
                f"- `{row.feature_name}`：mean_VIP={row.mean_VIP:.6f}，P(VIP>1)={row.P_VIP_gt_1:.3f}，符号一致率={row.sign_consistency:.3f}"
            )
    else:
        stable_lines.append("- 待补充")

    kfinal_lines = []
    if isinstance(result["kfinal_top8"], pd.DataFrame) and not result["kfinal_top8"].empty:
        for row in result["kfinal_top8"].itertuples(index=False):
            kfinal_lines.append(f"- `{row.feature_name}`：K_final={row.K_final:.6f}")
    else:
        kfinal_lines.append("- 待补充")

    return [
        "# 5 问题2：同题论文可量化文本特征关联分析与小样本质量预测模型",
        "",
        "## 5.1 问题分析与建模目标",
        "",
        "第二问的目标不是重新建立一套完整评分体系，而是在第一问封版智能评价系统的基础上，研究附件2同一赛题论文中哪些可量化文本特征能够解释论文质量差异。附件2共包含10篇同一赛题论文，在赛题背景相同的条件下进行比较，可以相对削弱题目差异带来的影响，更集中地考察论文结构、模型方法、结果解释和写作规范等因素。",
        "",
        "由于样本量仅为 N=10，直接建立强预测模型存在明显小样本风险。因此，本问采用稳健标准化、Spearman相关分析、灰色关联度、PLS-VIP、Bootstrap稳定性、删除单样本敏感性分析以及成对排序辅助检验等方法，形成以“特征解释和稳定性审计”为主的分析框架。需要强调的是，本文使用的质量标签 `Q_i` 来自第一问封版智能评价系统，是弱监督质量标签，不是官方真实分数、专家分数或人工真值。",
        "",
        "## 5.2 弱监督质量标签 Q_i 与特征体系",
        "",
        f"附件2每篇论文的质量标签记为 `Q_i`，其来源为第一问封版智能评价系统输出，`Q_source = {result['q_source']}`，`label_note = {result['label_note']}`。由结果表可得，`Q_i` 的取值范围为 `{_fmt(result['q_min'])} ~ {_fmt(result['q_max'])}`。",
        "",
        "本文提取的表层文本特征 `X` 包括五类：篇幅结构类、数学表达类、结果展示类、逻辑表达类和学术规范类。深层质量校正特征 `H` 包括 `task_coverage`、`data_credibility`、`method_fit`、`formula_explanation`、`result_closure` 和 `stacking_penalty`。其中前五个指标为正向特征，`stacking_penalty` 为负向风险特征。",
        "",
        "在章节识别过程中，若某些章节只能作为候选段落识别，则按 0.5 的低置信权重参与特征计算，不将其提升为确定章节。Step12B 审计显示，`reference_norm_rate` 为常数列，后续从主相关排序和PLS建模中剔除；`appendix_code_presence` 为低方差特征，在主PLS模型中剔除或谨慎处理。",
        "",
        "## 5.3 稳健标准化与相关性分析",
        "",
        "为降低小样本和极端值对特征尺度的影响，本文采用基于中位数和MAD的稳健标准化：",
        "",
        "$$z_{ij}=\\frac{x_{ij}-\\operatorname{median}(x_j)}{\\operatorname{MAD}(x_j)+\\varepsilon},$$",
        "",
        "其中，$\\operatorname{MAD}(x_j)=\\operatorname{median}(|x_{ij}-\\operatorname{median}(x_j)|)$，$\\varepsilon$ 为防止分母为零的极小正数。由于样本量仅为10，本文主要采用Spearman秩相关衡量特征与弱监督质量标签之间的单变量关系，不强调显著性检验。",
        "",
        f"Spearman绝对值排名前8的特征为：{_top_text(result['spearman_top8'])}。其中 `method_fit`、`section_coverage`、`task_coverage`、`figure_table_explanation_rate` 和 `objective_constraint_completeness` 与质量标签关系较为明显。`total_chars` 只能解释为信息承载量和结构完整性相关，不能解释为篇幅越长越好。`stacking_penalty` 与 `Q_i` 的Spearman相关为 `{_fmt(stacking_spearman)}`，方向符合负向风险特征设定，但强度很弱，应谨慎表述。",
        "",
        "## 5.4 灰色关联度与初步关键性指数",
        "",
        "为从序列贴近程度角度补充相关性分析，本文以 `Q_i` 序列为参考序列，各文本特征为比较序列，计算灰色关联度。设第 $j$ 个特征在第 $i$ 篇论文上的标准化序列为 $x_j(i)$，参考序列为 $x_0(i)$，则灰色关联系数可写为：",
        "",
        "$$\\xi_j(i)=\\frac{\\Delta_{\\min}+\\rho\\Delta_{\\max}}{|x_0(i)-x_j(i)|+\\rho\\Delta_{\\max}},$$",
        "",
        "其中 $\\rho=0.5$，灰色关联度 $G_j$ 为各样本关联系数的均值。进一步构造初步关键性指数：",
        "",
        "$$K_{pre,j}=0.6|r^S_j|+0.4G_j,$$",
        "",
        f"灰色关联度Top 8为：{_names_text(result['grey_top8'])}。K_pre Top 8为：{_names_text(result['kpre_top8'])}。其中 `K_pre` 仅用于初步筛选，不作为最终关键特征结论。",
        "",
        "## 5.5 PLS质量预测模型与VIP关键特征识别",
        "",
        "为在多特征共线性条件下进一步识别关键特征，本文采用偏最小二乘回归（PLS）进行小样本建模，并用VIP指标衡量特征在潜变量中的贡献。候选潜变量数取 $A\\in\\{1,2\\}$，通过留一交叉验证（LOOCV）选择模型复杂度。",
        "",
        f"结果显示，最终选择 `A={int(pls.get('n_components', 0)) if len(pls) else '待补充'}`。PLS-LOOCV结果为：MAE={_fmt(pls.get('MAE', np.nan))}，RMSE={_fmt(pls.get('RMSE', np.nan))}，R2_LOO={_fmt(pls.get('R2_LOO', np.nan))}，Spearman={_fmt(pls.get('Spearman', np.nan))}。其中R2_LOO为负，说明PLS不适合作为强预测模型。本文主要使用PLS的VIP结果识别关键解释特征，而不是夸大其预测性能。",
        "",
        f"VIP Top 8为：{_top_text(result['vip_top8'])}。VIP大于1的特征包括：{', '.join(result['vip_gt1']) if result['vip_gt1'] else '待补充'}。其中 `page_count` 和 `total_chars` 只表示信息承载量和完整性相关。预测误差最大的样本为 `2-1`，其绝对误差为 `{_fmt(result['paper_21_error'])}`。",
        "",
        "## 5.6 质量调整因子 QAF",
        "",
        "考虑到PLS可能受表层统计特征影响，本文构造质量调整因子（QAF）进行保守校正。首先定义深层质量校正得分：",
        "",
        "$$u_i=\\operatorname{mean}(task\\_coverage_i,data\\_credibility_i,method\\_fit_i,formula\\_explanation_i,result\\_closure_i)-stacking\\_penalty_i.$$",
        "",
        "将其中心化为：",
        "",
        "$$u_i^c=u_i-\\overline{u}.$$",
        "",
        "质量调整因子为：",
        "",
        "$$\\phi_i=\\operatorname{clip}(1+\\eta u_i^c,0.90,1.10),$$",
        "",
        "最终预测为：",
        "",
        "$$\\hat Q^{QAF}_i=\\operatorname{clip}(\\phi_i\\tilde Q_i,Q_{\\min},Q_{\\max}).$$",
        "",
        f"结果表明，最优 `eta={_fmt(qaf.get('eta', np.nan), 2)}`，`phi_i` 范围为 `{_fmt(result['phi_min'])} ~ {_fmt(result['phi_max'])}`。PLS基线MAE={_fmt(qaf.get('PLS_base_MAE', np.nan))}，QAF后MAE={_fmt(qaf.get('MAE', np.nan))}；PLS基线RMSE={_fmt(qaf.get('PLS_base_RMSE', np.nan))}，QAF后RMSE={_fmt(qaf.get('RMSE', np.nan))}；PLS基线R2={_fmt(qaf.get('PLS_base_R2', np.nan))}，QAF后R2={_fmt(qaf.get('R2', np.nan))}；PLS基线Spearman={_fmt(qaf.get('PLS_base_Spearman', np.nan))}，QAF后Spearman={_fmt(qaf.get('Spearman', np.nan))}。",
        "",
        "QAF只带来小幅改善，应写作保守校正，不能写作显著提升。样本 `2-1` 的误差未被改善；样本 `2-10` 下调后误差改善。`2-1`、`2-2`、`2-10` 因 `stacking_penalty` 较高受到约束，但局部样本效果并不完全一致。",
        "",
        "## 5.7 小样本稳定性分析",
        "",
        "为检验关键特征结论在小样本下的稳定性，本文进行Bootstrap VIP稳定性分析。Bootstrap次数为 `B=1000`，有效次数为 `1000/1000`。稳定特征包括：",
        "",
        *stable_lines,
        "",
        "综合Spearman、灰色关联度、Bootstrap平均VIP和符号一致率，构造最终关键性指数：",
        "",
        "$$K_j=0.35|r^S_j|+0.25G_j+0.25VIP^{norm}_j+0.15SC_j,$$",
        "",
        "其中 $SC_j$ 表示Bootstrap系数符号一致率。K_final Top 8为：",
        "",
        *kfinal_lines,
        "",
        f"最终关键特征为：{', '.join(result['final_key_features']) if result['final_key_features'] else '待补充'}。删除单样本敏感性分析识别出的高影响论文包括：{', '.join(result['high_influence']) if result['high_influence'] else '待补充'}。特别地，删除 `2-1` 后，RMSE变化为 `{_fmt(delete_21.get('RMSE_delta_vs_full_LOOCV', np.nan))}`，Spearman变化为 `{_fmt(delete_21.get('Spearman_delta_vs_full_LOOCV', np.nan))}`，Top5特征Jaccard为 `{_fmt(delete_21.get('top5_jaccard_vs_full', np.nan), 3)}`，说明 `2-1` 是高影响样本。总体而言，小样本稳定性可审计但有限。",
        "",
        "## 5.8 成对排序辅助检验",
        "",
        "附件2共有10篇论文，可构造 $C(10,2)=45$ 个论文对。需要强调的是，这45对并非45个独立监督样本，不能作为独立样本验证模型性能。本文仅使用 `final_key_feature + K_final` 固定加权差分进行排序辅助检验，不进行复杂模型训练。",
        "",
        f"结果显示，成对样本数为 `{int(pair.get('total_pairs', 0)) if len(pair) else '待补充'}`，near_tie数量为 `{int(pair.get('near_tie_pairs', 0)) if len(pair) else '待补充'}`，总体pairwise accuracy为 `{_fmt(pair.get('overall_pairwise_accuracy', np.nan))}`，留一论文组平均accuracy为 `{_fmt(pair.get('group_validation_accuracy_mean', np.nan))}`，accuracy标准差为 `{_fmt(pair.get('group_validation_accuracy_std', np.nan))}`，去除near_tie后accuracy为 `{_fmt(pair.get('accuracy_without_near_tie', np.nan))}`，`2-1` 留一组accuracy为 `{_fmt(group_21.get('pairwise_accuracy', np.nan))}`。",
        "",
        "该结果对K_final关键特征结论提供中等支持，但不是强验证。小样本、弱监督标签和成对样本非独立性仍限制了排序结论的泛化。",
        "",
        "## 5.9 第二问结果总结",
        "",
        f"综合相关性分析、灰色关联度、PLS-VIP、Bootstrap稳定性、删除单样本敏感性和成对排序辅助检验，第二问最终识别出的关键特征为：{', '.join(result['final_key_features']) if result['final_key_features'] else '待补充'}。",
        "",
        "其中，`total_chars` 和 `page_count` 仅表示信息承载量和结构完整性相关，并不意味着篇幅越长越好；`method_fit`、`task_coverage`、`section_coverage`、`objective_constraint_completeness`、`figure_table_explanation_rate` 和 `conclusion_echo_rate` 更能体现论文的实质质量。由此可见，影响数学建模论文质量的关键因素不是单纯篇幅、公式数量或图表数量，而是方法是否贴题、任务是否覆盖、结构是否完整、目标函数与约束是否清晰、图表是否被解释以及结论是否形成闭环。",
        "",
        "本问的局限性也需要明确说明：`Q_i` 是弱监督标签而非官方真实分数；附件2样本量仅为10；PLS/QAF预测能力有限且R2为负；高影响样本较多；成对排序只作为辅助检验。若后续获得官方标签或专家评分，可进一步验证关键特征的稳健性。",
    ]


def _tables_lines() -> list[str]:
    """Build table insertion suggestions."""
    rows = [
        ("表X 弱监督质量标签 Q_i 表", "output/problem2_tables/appendix2_q_labels.xlsx", "5.2", "说明Q_i来源、范围和弱监督属性。"),
        ("表X 表层与深层特征体系表", "output/problem2_tables/appendix2_features_with_q.xlsx；output/problem2_tables/appendix2_deep_quality_features_auto.xlsx", "5.2", "展示表层特征X和深层特征H。"),
        ("表X Spearman Top特征表", "output/problem2_tables/appendix2_correlation_analysis.xlsx", "5.3", "展示相关性排名，强调不夸大显著性。"),
        ("表X K_pre初步关键性指数表", "output/problem2_tables/appendix2_key_features_preliminary.xlsx", "5.4", "展示灰色关联和K_pre初筛结果。"),
        ("表X PLS/VIP结果表", "output/problem2_tables/pls_component_selection.xlsx；output/problem2_tables/pls_vip_scores.xlsx", "5.5", "展示LOOCV指标和VIP Top特征。"),
        ("表X QAF结果表", "output/problem2_tables/qaf_eta_selection.xlsx；output/problem2_tables/qaf_prediction_results.xlsx", "5.6", "展示eta选择、phi_i和调整前后误差。"),
        ("表X Bootstrap稳定特征表", "output/problem2_tables/bootstrap_vip_stability.xlsx", "5.7", "展示mean_VIP、P(VIP>1)和符号一致率。"),
        ("表X K_final最终关键特征表", "output/problem2_tables/key_feature_index_final.xlsx", "5.7或5.9", "展示最终关键特征指数和final_key_feature。"),
        ("表X 成对排序辅助检验表", "output/problem2_tables/pairwise_ranking_check.xlsx；output/problem2_tables/pairwise_group_validation.xlsx", "5.8", "展示pairwise accuracy和留一论文组验证。"),
    ]
    lines = ["# 第二问建议插入表格", ""]
    for title, source, position, note in rows:
        lines.extend([f"## {title}", f"- 来源文件：`{source}`", f"- 建议位置：第{position}节", f"- 说明：{note}", ""])
    return lines


def _figures_lines() -> list[str]:
    """Build figure insertion suggestions."""
    rows = [
        ("图X 附件2特征热力图", "output/problem2_charts/appendix2_feature_heatmap.png", "5.3", "展示10篇论文在主要特征上的差异。"),
        ("图X Spearman相关性Top特征柱状图", "output/problem2_charts/appendix2_correlation_bar.png", "5.3", "展示主要特征与Q_i的秩相关强度。"),
        ("图X 灰色关联度Top特征图", "output/problem2_charts/grey_relation_bar.png", "5.4", "展示序列贴近程度较高的特征。"),
        ("图X K_pre初步关键特征图", "output/problem2_charts/key_features_preliminary_bar.png", "5.4", "展示初步关键性指数。"),
        ("图X PLS真实值与预测值散点图", "output/problem2_charts/pls_true_vs_pred.png", "5.5", "展示PLS预测能力有限，不能夸大。"),
        ("图X PLS VIP Top特征图", "output/problem2_charts/pls_vip_bar.png", "5.5", "展示VIP特征贡献。"),
        ("图X QAF调整前后误差对比图", "output/problem2_charts/qaf_before_after_error.png", "5.6", "展示QAF保守校正效果。"),
        ("图X Bootstrap VIP稳定性箱线图", "output/problem2_charts/bootstrap_vip_boxplot.png", "5.7", "展示Top特征的Bootstrap稳定性。"),
        ("图X K_final最终关键特征图", "output/problem2_charts/key_feature_index_final.png", "5.7", "展示最终关键特征指数。"),
        ("图X 删除单样本敏感性图", "output/problem2_charts/delete_one_sensitivity.png", "5.7", "展示高影响样本对模型稳定性的影响。"),
        ("图X 成对排序留一组准确率图", "output/problem2_charts/pairwise_ranking_accuracy_bar.png", "5.8", "展示成对排序辅助检验。"),
    ]
    lines = ["# 第二问建议插入图表", ""]
    for title, source, position, caption in rows:
        lines.extend([f"## {title}", f"- 来源文件：`{source}`", f"- 建议位置：第{position}节", f"- 图注建议：{caption}", ""])
    return lines


def _formula_lines() -> list[str]:
    """Build method formula collection."""
    return [
        "# 第二问方法公式整理",
        "",
        "## 1. 稳健标准化",
        "$$z_{ij}=\\frac{x_{ij}-\\operatorname{median}(x_j)}{\\operatorname{MAD}(x_j)+\\varepsilon},\\quad \\operatorname{MAD}(x_j)=\\operatorname{median}(|x_{ij}-\\operatorname{median}(x_j)|).$$",
        "",
        "## 2. Spearman秩相关",
        "$$r_s=1-\\frac{6\\sum_{i=1}^{n}d_i^2}{n(n^2-1)},$$",
        "其中 $d_i$ 为两个变量秩次之差。本文样本量为10，因此主要用于关系强弱描述，不强调显著性。",
        "",
        "## 3. 灰色关联度",
        "$$\\xi_j(i)=\\frac{\\Delta_{\\min}+\\rho\\Delta_{\\max}}{|x_0(i)-x_j(i)|+\\rho\\Delta_{\\max}},\\quad G_j=\\frac{1}{n}\\sum_{i=1}^{n}\\xi_j(i).$$",
        "本文取分辨系数 $\\rho=0.5$。",
        "",
        "## 4. 初步关键性指数 K_pre",
        "$$K_{pre,j}=0.6|r^S_j|+0.4G_j.$$",
        "",
        "## 5. PLS模型",
        "PLS通过提取潜变量同时解释特征矩阵 $X$ 和质量标签 $Q$，本文候选潜变量数为 $A\\in\\{1,2\\}$，并通过LOOCV选择。",
        "",
        "## 6. VIP指标",
        "$$VIP_j=\\sqrt{p\\frac{\\sum_{a=1}^{A}SS_a w_{ja}^2/\\|w_a\\|^2}{\\sum_{a=1}^{A}SS_a}},$$",
        "其中 $SS_a$ 表示第 $a$ 个潜变量对因变量的解释贡献。",
        "",
        "## 7. QAF质量调整因子",
        "$$u_i=\\operatorname{mean}(task\\_coverage_i,data\\_credibility_i,method\\_fit_i,formula\\_explanation_i,result\\_closure_i)-stacking\\_penalty_i,$$",
        "$$u_i^c=u_i-\\overline{u},$$",
        "$$\\phi_i=\\operatorname{clip}(1+\\eta u_i^c,0.90,1.10),$$",
        "$$\\hat Q^{QAF}_i=\\operatorname{clip}(\\phi_i\\tilde Q_i,Q_{\\min},Q_{\\max}).$$",
        "",
        "## 8. Bootstrap稳定性",
        "每次从10篇论文中有放回抽取10篇，重新拟合PLS(A=1)，记录VIP、VIP>1概率和系数符号一致率。",
        "",
        "## 9. 最终关键特征指数 K_final",
        "$$K_j=0.35|r^S_j|+0.25G_j+0.25VIP^{norm}_j+0.15SC_j.$$",
        "",
        "## 10. 成对排序加权差分",
        "$$pair\\_score_{ij}=\\sum_{k\\in\\mathcal{F}}K_k(x_{ik}-x_{jk}),$$",
        "若 $pair\\_score_{ij}>0$，则预测论文 $i$ 优于论文 $j$；反之预测论文 $j$ 优于论文 $i$。该检验只作为排序辅助验证。",
        "",
    ]


def _result_summary_lines(result: dict[str, Any]) -> list[str]:
    """Build compact result summary markdown."""
    pls = result["pls_selected"]
    qaf = result["qaf_selected"]
    pair = result["pair"]
    return [
        "# 第二问结果摘要",
        "",
        f"- Q_label范围：{_fmt(result['q_min'])} ~ {_fmt(result['q_max'])}",
        f"- Spearman Top 8：{_top_text(result['spearman_top8'])}",
        f"- K_pre Top 8：{_top_text(result['kpre_top8'])}",
        f"- PLS：A={int(pls.get('n_components', 0)) if len(pls) else '待补充'}，MAE={_fmt(pls.get('MAE', np.nan))}，RMSE={_fmt(pls.get('RMSE', np.nan))}，R2_LOO={_fmt(pls.get('R2_LOO', np.nan))}，Spearman={_fmt(pls.get('Spearman', np.nan))}",
        f"- VIP Top 8：{_top_text(result['vip_top8'])}",
        f"- QAF：eta={_fmt(qaf.get('eta', np.nan), 2)}，MAE={_fmt(qaf.get('MAE', np.nan))}，RMSE={_fmt(qaf.get('RMSE', np.nan))}，R2={_fmt(qaf.get('R2', np.nan))}，Spearman={_fmt(qaf.get('Spearman', np.nan))}",
        f"- K_final Top 8：{'; '.join(f'{row.feature_name}:{row.K_final:.6f}' for row in result['kfinal_top8'].itertuples(index=False)) if isinstance(result['kfinal_top8'], pd.DataFrame) and not result['kfinal_top8'].empty else '待补充'}",
        f"- final_key_feature：{', '.join(result['final_key_features']) if result['final_key_features'] else '待补充'}",
        f"- 成对排序：total_pairs={int(pair.get('total_pairs', 0)) if len(pair) else '待补充'}，near_tie={int(pair.get('near_tie_pairs', 0)) if len(pair) else '待补充'}，overall_accuracy={_fmt(pair.get('overall_pairwise_accuracy', np.nan))}，group_mean={_fmt(pair.get('group_validation_accuracy_mean', np.nan))}，accuracy_without_near_tie={_fmt(pair.get('accuracy_without_near_tie', np.nan))}",
        "",
        "最终可写结论：第二问识别出的关键因素集中在方法匹配、任务覆盖、结构完整性、目标函数与约束完整性、图表解释和结论闭环等方面。篇幅类特征只表示信息承载量和完整性相关，不能解释为篇幅越长越好。PLS/QAF预测能力有限，结论应以解释性和可审计性为主。",
        "",
    ]


def _limitations_lines() -> list[str]:
    """Build limitations markdown."""
    return [
        "# 第二问局限性与保守表述",
        "",
        "1. `Q_i` 为第一问封版智能评估系统输出的弱监督质量标签，不是官方真实质量分。",
        "2. 附件2样本量仅为 `N=10`，所有统计结论都应保守解释。",
        "3. PLS/QAF不是强预测模型，主要用于关键特征解释、初步预测和保守校正。",
        "4. PLS的 `R2_LOO` 为负，必须如实说明预测能力有限。",
        "5. QAF只带来小幅改善，不得写成显著提升。",
        "6. 成对排序辅助检验中的45对论文不是45个独立样本，不能作为独立监督验证。",
        "7. `total_chars` 和 `page_count` 只能解释为信息承载量/结构完整性相关，不能写成篇幅越长越好。",
        "8. 删除单样本敏感性分析显示高影响样本较多，特别是 `2-1`，因此最终结论应强调可审计但有限。",
        "9. 若后续获得官方标签或专家评分，可进一步验证当前弱监督结论。",
        "",
    ]


def generate_problem2_draft(tables_dir: Path, charts_dir: Path, output_dir: Path) -> dict[str, Path]:
    """Generate the Problem 2 paper draft section and writing materials."""
    tables_dir = Path(tables_dir)
    charts_dir = Path(charts_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = _collect_problem2_results(tables_dir)
    draft_path = output_dir / "problem2_draft.md"
    tables_path = output_dir / "problem2_tables_to_insert.md"
    figures_path = output_dir / "problem2_figures_to_insert.md"
    formulas_path = output_dir / "problem2_method_formulas.md"
    summary_path = output_dir / "problem2_result_summary.md"
    limitations_path = output_dir / "problem2_limitations.md"

    _write_text(draft_path, _draft_lines(result))
    _write_text(tables_path, _tables_lines())
    _write_text(figures_path, _figures_lines())
    _write_text(formulas_path, _formula_lines())
    _write_text(summary_path, _result_summary_lines(result))
    _write_text(limitations_path, _limitations_lines())

    return {
        "draft_path": draft_path,
        "tables_path": tables_path,
        "figures_path": figures_path,
        "formulas_path": formulas_path,
        "summary_path": summary_path,
        "limitations_path": limitations_path,
    }


def run_step21_problem2_draft(config_path: Path | None = None) -> dict[str, Any]:
    """Run Step 21 writing-material generation."""
    problem2_config = get_problem2_config(config_path)
    tables_dir = resolve_project_path(problem2_config["problem2_output_tables_dir"])
    charts_dir = resolve_project_path(problem2_config["problem2_output_charts_dir"])
    output_dir = resolve_project_path("paper_sections/problem2")
    paths = generate_problem2_draft(tables_dir, charts_dir, output_dir)
    return {"output_paths": paths, "tables_dir": tables_dir, "charts_dir": charts_dir, "output_dir": output_dir}


def run_step20_problem2_final_audit(config_path: Path | None = None) -> dict[str, Any]:
    """Run Step 20 final consistency audit for Problem 2."""
    problem2_config = get_problem2_config(config_path)
    tables_dir = resolve_project_path(problem2_config["problem2_output_tables_dir"])
    charts_dir = resolve_project_path(problem2_config["problem2_output_charts_dir"])
    logs_dir = resolve_project_path(problem2_config["problem2_output_logs_dir"])
    paper_dir = resolve_project_path("paper_sections/problem2")
    tables_dir.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    paper_dir.mkdir(parents=True, exist_ok=True)

    log_path = logs_dir / "problem2_final_audit.log"
    logger = _setup_logger(log_path)
    logger.info("Starting Step 20 final audit. No previous result files will be modified.")

    table_integrity, chart_integrity, audit_rows = _file_integrity(tables_dir, charts_dir)
    sample_consistency, sample_audit, _ = _sample_consistency(tables_dir)
    label_checks, label_audit = _label_checks(tables_dir)
    feature_usage, feature_audit = _feature_usage_checks(tables_dir)
    model_consistency, model_audit = _model_consistency_checks(tables_dir)
    warnings = _warnings_table()
    key_results = _key_results(tables_dir)
    key_md_path, warnings_md_path = _write_markdown(key_results, warnings, paper_dir)

    audit_rows.extend(sample_audit)
    audit_rows.extend(label_audit)
    audit_rows.extend(feature_audit)
    audit_rows.extend(model_audit)
    for row in warnings.itertuples(index=False):
        audit_rows.append(_make_audit_row("writing_warning", f"warning_{row.warning_id}", "PASS", row.warning))

    audit_summary = pd.DataFrame(audit_rows)
    has_fail = bool(audit_summary["status"].eq("FAIL").any())
    final_statement = (
        "第二问结果文件一致，可进入论文写作阶段。"
        if not has_fail
        else "第二问结果审计发现不一致，请先修复 FAIL 项后再进入论文写作阶段。"
    )
    audit_summary = pd.concat(
        [
            pd.DataFrame(
                [
                    {
                        "category": "final_conclusion",
                        "item": "can_enter_step21",
                        "status": _status(not has_fail),
                        "details": final_statement,
                    }
                ]
            ),
            audit_summary,
        ],
        ignore_index=True,
    )

    audit_path = tables_dir / "problem2_final_audit.xlsx"
    with pd.ExcelWriter(audit_path) as writer:
        audit_summary.to_excel(writer, sheet_name="audit_summary", index=False)
        table_integrity.to_excel(writer, sheet_name="file_integrity", index=False)
        chart_integrity.to_excel(writer, sheet_name="chart_integrity", index=False)
        sample_consistency.to_excel(writer, sheet_name="sample_consistency", index=False)
        label_checks.to_excel(writer, sheet_name="label_checks", index=False)
        feature_usage.to_excel(writer, sheet_name="feature_usage", index=False)
        model_consistency.to_excel(writer, sheet_name="model_consistency", index=False)
        key_results.to_excel(writer, sheet_name="key_results", index=False)
        warnings.to_excel(writer, sheet_name="writing_warnings", index=False)

    logger.info("Audit summary: %s", audit_summary.to_dict(orient="records"))
    logger.info("Final statement: %s", final_statement)
    logger.info("Saved audit workbook: %s", audit_path)
    logger.info("Saved key results markdown: %s", key_md_path)
    logger.info("Saved writing warnings markdown: %s", warnings_md_path)
    logger.info("Finished Step 20")

    return {
        "audit_summary": audit_summary,
        "table_integrity": table_integrity,
        "chart_integrity": chart_integrity,
        "sample_consistency": sample_consistency,
        "label_checks": label_checks,
        "feature_usage": feature_usage,
        "model_consistency": model_consistency,
        "key_results": key_results,
        "warnings": warnings,
        "has_fail": has_fail,
        "final_statement": final_statement,
        "audit_path": audit_path,
        "log_path": log_path,
        "key_md_path": key_md_path,
        "warnings_md_path": warnings_md_path,
    }
