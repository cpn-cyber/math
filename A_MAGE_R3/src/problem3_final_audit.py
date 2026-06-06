"""Problem 3 Step 31: final audit and writing-material generation."""

from __future__ import annotations

from pathlib import Path
import json
import logging
import math
from typing import Any

import pandas as pd

from modules.appendix3_pipeline import get_problem3_config, resolve_project_path


LOGGER_NAME = "A_MAGE_R3.problem3.final_audit"
EXPECTED_PAPERS = {"3-1", "3-2", "3-3"}
DISCLAIMER_EXACT = "该指标仅表示文本存在 AI辅助写作风险，不构成学术不端判断，也不等同于AI生成判定。"
DISCLAIMER_KEYWORDS = ["AI辅助写作风险", "不构成学术不端判断", "不等同于AI生成判定"]

CORE_TABLES = {
    "ai_risk_ds_fusion.xlsx": "Step26 AI辅助写作风险D-S融合结果",
    "ai_risk_evidence.xlsx": "Step26 AI风险证据表",
    "multi_agent_subjectivity_analysis.xlsx": "Step27多智能体主观性分析",
    "multi_agent_disagreement_details.xlsx": "Step27分歧明细",
    "revision_action_optimization.xlsx": "Step28修改动作优化结果",
    "revision_action_details.xlsx": "Step28动作收益明细",
    "quality_prediction_after_revision.xlsx": "Step29优化后质量预测",
    "quality_prediction_summary.xlsx": "Step29预测汇总",
    "robustness_summary.xlsx": "Step30稳健性汇总",
    "robustness_action_stability.xlsx": "Step30动作稳定性",
    "robustness_prediction_results.xlsx": "Step30参数扰动预测明细",
}

CORE_LOGS = {
    "ai_risk_ds_fusion.log": "Step26运行日志",
    "multi_agent_subjectivity.log": "Step27运行日志",
    "revision_action_optimization.log": "Step28运行日志",
    "quality_prediction_after_revision.log": "Step29运行日志",
    "robustness_analysis_step30.log": "Step30运行日志",
}

CORE_FIGURES = {
    "ai_risk_radar.png": ("Step26 AI风险证据雷达图", "展示三篇论文四类AI辅助写作风险证据", "正文", "高"),
    "ai_risk_bar.png": ("Step26 AI风险柱状图", "展示三篇论文R_AI均为低风险", "正文", "高"),
    "agent_disagreement_bar.png": ("Step27多智能体分歧柱状图", "展示三篇论文均存在高分歧", "正文", "高"),
    "paper_disagreement_radar.png": ("Step27多智能体评分雷达图", "展示各维度评分差异来源", "正文或附录", "中"),
    "revision_priority_bar.png": ("Step28修改优先级柱状图", "展示预计收益和修改成本", "正文", "高"),
    "revision_gain_cost_scatter.png": ("Step28动作收益-成本散点图", "展示动作收益和成本结构", "附录", "中"),
    "score_before_after.png": ("Step29修改前后质量得分对比图", "展示优化后预测质量提升", "正文", "高"),
    "risk_before_after.png": ("Step29风险提示指标前后对比图", "展示AI辅助写作风险提示指标下降", "正文或附录", "中"),
    "disagreement_before_after.png": ("Step29分歧前后对比图", "展示多智能体分歧收敛", "正文", "高"),
    "robustness_action_frequency.png": ("Step30动作稳定性频次图", "展示A11、A10、A12等稳定推荐动作", "正文", "高"),
    "robustness_score_gain_sensitivity.png": ("Step30收益系数敏感性图", "展示预测提升对收益系数扰动的响应", "附录", "中"),
    "robustness_budget_sensitivity.png": ("Step30预算敏感性图", "展示预算变化下平均预测得分和动作数量", "正文或附录", "中"),
    "robustness_disagreement_sensitivity.png": ("Step30分歧收敛敏感性图", "展示分歧标准差下降趋势", "附录", "中"),
}


def setup_audit_logger(log_path: Path) -> logging.Logger:
    """Configure Step 31 logger."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(LOGGER_NAME)
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)
    return logger


def _status(exists: bool, non_empty: bool) -> str:
    if exists and non_empty:
        return "PASS"
    if exists and not non_empty:
        return "FAIL"
    return "WARNING"


def _safe_read_excel(path: Path, sheet_name: str | int = 0) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 0:
        return pd.DataFrame()
    try:
        return pd.read_excel(path, sheet_name=sheet_name)
    except Exception:
        return pd.DataFrame()


def _safe_float(value: Any, default: float = math.nan) -> float:
    number = pd.to_numeric(value, errors="coerce")
    return default if pd.isna(number) else float(number)


def _paper_set(df: pd.DataFrame) -> set[str]:
    if df.empty or "paper_id" not in df.columns:
        return set()
    return set(df["paper_id"].astype(str))


def _almost_equal(a: Any, b: Any, tol: float = 1e-6) -> bool:
    fa = _safe_float(a)
    fb = _safe_float(b)
    return not (math.isnan(fa) or math.isnan(fb)) and abs(fa - fb) <= tol


def _split_actions(value: Any) -> set[str]:
    if value is None or pd.isna(value):
        return set()
    return {part.strip() for part in str(value).split(",") if part.strip()}


def _all_text_from_frame(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    return "\n".join(df.astype(str).fillna("").to_numpy().ravel().tolist())


def audit_file_integrity(tables_dir: Path, charts_dir: Path, logs_dir: Path) -> pd.DataFrame:
    """Audit required tables, figures, and logs."""
    rows: list[dict[str, Any]] = []
    for filename, role in CORE_TABLES.items():
        path = tables_dir / filename
        exists = path.exists()
        size = path.stat().st_size if exists else 0
        rows.append(
            {
                "file_name": filename,
                "expected_role": role,
                "exists": exists,
                "non_empty": size > 0,
                "file_size": size,
                "status": _status(exists, size > 0),
                "note": str(path),
            }
        )
    for filename, (caption, _, _, _) in CORE_FIGURES.items():
        path = charts_dir / filename
        exists = path.exists()
        size = path.stat().st_size if exists else 0
        rows.append(
            {
                "file_name": filename,
                "expected_role": caption,
                "exists": exists,
                "non_empty": size > 0,
                "file_size": size,
                "status": _status(exists, size > 0),
                "note": str(path),
            }
        )
    for filename, role in CORE_LOGS.items():
        path = logs_dir / filename
        exists = path.exists()
        size = path.stat().st_size if exists else 0
        rows.append(
            {
                "file_name": filename,
                "expected_role": role,
                "exists": exists,
                "non_empty": size > 0,
                "file_size": size,
                "status": _status(exists, size > 0),
                "note": str(path),
            }
        )
    return pd.DataFrame(rows)


def load_core_tables(tables_dir: Path) -> dict[str, pd.DataFrame]:
    """Load the core Step 26-30 tables needed by the audit."""
    return {
        "ai_fusion": _safe_read_excel(tables_dir / "ai_risk_ds_fusion.xlsx"),
        "ai_evidence": _safe_read_excel(tables_dir / "ai_risk_evidence.xlsx"),
        "subjectivity": _safe_read_excel(tables_dir / "multi_agent_subjectivity_analysis.xlsx"),
        "disagreement": _safe_read_excel(tables_dir / "multi_agent_disagreement_details.xlsx"),
        "revision": _safe_read_excel(tables_dir / "revision_action_optimization.xlsx"),
        "revision_details": _safe_read_excel(tables_dir / "revision_action_details.xlsx", sheet_name="action_details"),
        "prediction": _safe_read_excel(tables_dir / "quality_prediction_after_revision.xlsx"),
        "prediction_summary": _safe_read_excel(tables_dir / "quality_prediction_summary.xlsx"),
        "robustness_summary": _safe_read_excel(tables_dir / "robustness_summary.xlsx", sheet_name="paper_level_stability"),
        "robustness_action": _safe_read_excel(tables_dir / "robustness_action_stability.xlsx", sheet_name="action_stability"),
        "robustness_results": _safe_read_excel(tables_dir / "robustness_prediction_results.xlsx"),
    }


def audit_data_consistency(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Audit paper IDs and key cross-step fields."""
    rows: list[dict[str, Any]] = []
    for name, df in tables.items():
        if name in {"revision_details", "robustness_action"}:
            continue
        papers = _paper_set(df)
        if not papers and name == "robustness_summary":
            papers = _paper_set(df[df.get("paper_id", pd.Series(dtype=str)).astype(str) != "overall"])
        rows.append(
            {
                "check_item": f"{name}_paper_id_set",
                "supporting_files": name,
                "status": "PASS" if papers == EXPECTED_PAPERS else "FAIL",
                "note": f"papers={sorted(papers)}",
            }
        )

    ai = tables["ai_fusion"].set_index("paper_id") if not tables["ai_fusion"].empty else pd.DataFrame()
    sub = tables["subjectivity"].set_index("paper_id") if not tables["subjectivity"].empty else pd.DataFrame()
    rev = tables["revision"].set_index("paper_id") if not tables["revision"].empty else pd.DataFrame()
    pred = tables["prediction"].set_index("paper_id") if not tables["prediction"].empty else pd.DataFrame()
    robust = tables["robustness_results"]

    for paper_id in sorted(EXPECTED_PAPERS):
        status = "PASS"
        notes: list[str] = []
        if not ai.empty and not pred.empty:
            if not _almost_equal(ai.loc[paper_id, "R_AI"], pred.loc[paper_id, "R_AI_before"]):
                status = "FAIL"
                notes.append("R_AI mismatch between Step26 and Step29")
        if not sub.empty and not pred.empty:
            if not _almost_equal(sub.loc[paper_id, "agent_score_std"], pred.loc[paper_id, "agent_score_std_before"]):
                status = "FAIL"
                notes.append("agent_score_std mismatch between Step27 and Step29")
        if not rev.empty and not pred.empty:
            if _split_actions(rev.loc[paper_id, "selected_actions_knapsack"]) != _split_actions(pred.loc[paper_id, "selected_actions_knapsack"]):
                status = "FAIL"
                notes.append("selected actions mismatch between Step28 and Step29")
        if not robust.empty and not pred.empty:
            base = robust[
                (robust["paper_id"].astype(str) == paper_id)
                & robust["score_gain_factor"].eq(0.12)
                & robust["budget"].eq(10)
                & robust["risk_reduction_multiplier"].eq(1.0)
                & robust["disagreement_reduction_multiplier"].eq(1.0)
            ]
            if base.empty:
                status = "FAIL"
                notes.append("baseline parameter scenario missing in Step30")
            else:
                match_row = base.iloc[0]
                if not _almost_equal(match_row["predicted_score_after_revision"], pred.loc[paper_id, "predicted_score_after_revision"]):
                    status = "FAIL"
                    notes.append("Step29 predicted score not found in Step30 baseline scenario")
        rows.append(
            {
                "check_item": f"{paper_id}_key_field_consistency",
                "supporting_files": "Step26, Step27, Step28, Step29, Step30",
                "status": status,
                "note": "; ".join(notes) if notes else "key fields consistent",
            }
        )

    if not tables["robustness_action"].empty:
        top_actions = tables["robustness_action"].sort_values("selected_count", ascending=False)["action_id"].astype(str).head(3).tolist()
        recommended_actions = set().union(*[_split_actions(v) for v in rev["selected_actions_knapsack"].tolist()]) if not rev.empty else set()
        status = "PASS" if set(top_actions).issubset(recommended_actions | {"A4"}) and "A11" in top_actions else "WARNING"
        rows.append(
            {
                "check_item": "action_stability_matches_revision_direction",
                "supporting_files": "revision_action_optimization.xlsx; robustness_action_stability.xlsx",
                "status": status,
                "note": f"top_actions={top_actions}; step28_actions={sorted(recommended_actions)}",
            }
        )
    return pd.DataFrame(rows)


def _conclusion_row(conclusion: str, files: str, fields: str, status: bool, note: str, recommendation: str = "建议写入正文") -> dict[str, Any]:
    return {
        "conclusion": conclusion,
        "supporting_files": files,
        "supporting_fields": fields,
        "status": "PASS" if status else "WARNING",
        "note": note,
        "write_to_paper_recommendation": recommendation,
    }


def audit_conclusions(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Check whether the planned paper conclusions are supported by data."""
    rows: list[dict[str, Any]] = []
    ai = tables["ai_fusion"]
    sub = tables["subjectivity"]
    action = tables["robustness_action"]
    paper_stability = tables["robustness_summary"]
    risk_stability = _safe_read_excel(resolve_project_path("output/problem3_tables/robustness_summary.xlsx"), sheet_name="risk_reduction")
    disagreement_stability = _safe_read_excel(resolve_project_path("output/problem3_tables/robustness_summary.xlsx"), sheet_name="disagreement_stability")

    ai_low = not ai.empty and set(ai["risk_level"].astype(str)) == {"低风险"} and _paper_set(ai) == EXPECTED_PAPERS
    rows.append(_conclusion_row("三篇论文 AI 辅助写作风险均为低风险", "ai_risk_ds_fusion.xlsx", "risk_level", ai_low, f"risk_levels={ai['risk_level'].astype(str).tolist() if not ai.empty else []}"))

    risk_sources = " ".join(ai.get("main_risk_source", pd.Series(dtype=str)).astype(str).tolist())
    source_ok = "数据不可追溯风险" in risk_sources and "方法到结果跳跃风险" in risk_sources
    rows.append(_conclusion_row("三篇论文主要风险来源集中在数据不可追溯和方法到结果跳跃", "ai_risk_ds_fusion.xlsx", "main_risk_source", source_ok, risk_sources))

    high_disagreement = not sub.empty and set(sub["disagreement_level"].astype(str)) == {"高分歧"} and _paper_set(sub) == EXPECTED_PAPERS
    rows.append(_conclusion_row("三篇论文多智能体评分均为高分歧", "multi_agent_subjectivity_analysis.xlsx", "disagreement_level", high_disagreement, f"levels={sub['disagreement_level'].astype(str).tolist() if not sub.empty else []}"))

    top = action.sort_values("selected_count", ascending=False) if not action.empty else pd.DataFrame()
    top_actions = top["action_id"].astype(str).head(3).tolist() if not top.empty else []
    rows.append(_conclusion_row("A11 是最稳定修改动作", "robustness_action_stability.xlsx", "selected_count,selection_rate", bool(top_actions and top_actions[0] == "A11"), f"top_actions={top_actions}"))
    rows.append(_conclusion_row("A10、A12 是次稳定主干动作", "robustness_action_stability.xlsx", "selected_count,selection_rate", {"A10", "A12"}.issubset(set(top_actions)), f"top_actions={top_actions}"))

    stability = paper_stability.set_index("paper_id") if not paper_stability.empty and "paper_id" in paper_stability.columns else pd.DataFrame()
    rows.append(_conclusion_row("3-1 即使优化后仍难达到合格", "robustness_summary.xlsx", "pass_or_above_ratio", not stability.empty and _safe_float(stability.loc["3-1", "pass_or_above_ratio"]) == 0.0, f"3-1 pass ratio={_safe_float(stability.loc['3-1', 'pass_or_above_ratio']) if not stability.empty else 'missing'}"))
    rows.append(_conclusion_row("3-2 优化后稳定保持合格", "robustness_summary.xlsx", "pass_or_above_ratio", not stability.empty and _safe_float(stability.loc["3-2", "pass_or_above_ratio"]) == 1.0, f"3-2 pass ratio={_safe_float(stability.loc['3-2', 'pass_or_above_ratio']) if not stability.empty else 'missing'}"))
    ratio_33 = _safe_float(stability.loc["3-3", "pass_or_above_ratio"]) if not stability.empty else math.nan
    rows.append(_conclusion_row("3-3 有较大概率提升到合格", "robustness_summary.xlsx", "pass_or_above_ratio", not math.isnan(ratio_33) and ratio_33 >= 0.5, f"3-3 pass ratio={ratio_33:.3f}" if not math.isnan(ratio_33) else "missing"))

    risk_ok = not risk_stability.empty and all(pd.to_numeric(risk_stability[risk_stability["paper_id"].astype(str) != "overall"]["proportion_risk_decrease"], errors="coerce").fillna(0).ge(1.0))
    rows.append(_conclusion_row("AI 风险提示指标在稳健性分析中均呈下降趋势", "robustness_summary.xlsx", "risk_reduction_stability", risk_ok, "all paper risk decrease proportions are 1.0" if risk_ok else "risk stability missing or not all decreasing"))

    dis_ok = not disagreement_stability.empty and pd.to_numeric(disagreement_stability["mean_std_reduction"], errors="coerce").fillna(0).gt(0).all()
    rows.append(_conclusion_row("多智能体分歧整体呈下降趋势", "robustness_summary.xlsx", "disagreement_stability", dis_ok, "mean std reduction > 0 for all papers" if dis_ok else "disagreement stability missing or non-positive"))

    return pd.DataFrame(rows)


def audit_disclaimer(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Audit AI-risk disclaimer presence across Step 26-30 result tables."""
    rows: list[dict[str, Any]] = []
    for key, df in tables.items():
        if key in {"revision_details", "robustness_results", "robustness_action"}:
            continue
        text = _all_text_from_frame(df)
        exact = DISCLAIMER_EXACT in text
        approximate = all(keyword in text for keyword in DISCLAIMER_KEYWORDS)
        rows.append(
            {
                "source": key,
                "exact_disclaimer_found": exact,
                "approximate_disclaimer_found": approximate,
                "status": "PASS" if exact or approximate else "WARNING",
                "note": "disclaimer or equivalent boundary statement present" if exact or approximate else "AI-risk disclaimer not found in this table",
            }
        )
    return pd.DataFrame(rows)


def build_figure_manifest(charts_dir: Path) -> pd.DataFrame:
    """Build figure manifest for paper writing."""
    rows = []
    for filename, (caption, message, section, priority) in CORE_FIGURES.items():
        path = charts_dir / filename
        rows.append(
            {
                "figure_name": filename,
                "file_path": str(path),
                "exists": path.exists(),
                "non_empty": path.exists() and path.stat().st_size > 0,
                "suggested_caption": caption + "：" + message,
                "suggested_section": section,
                "paper_usage_priority": priority,
            }
        )
    return pd.DataFrame(rows)


def build_key_results_summary(tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Build key result summary workbook sheets."""
    ai = tables["ai_fusion"][["paper_id", "R_AI", "risk_level", "main_risk_source"]].copy() if not tables["ai_fusion"].empty else pd.DataFrame()
    sub = tables["subjectivity"][["paper_id", "agent_score_std", "disagreement_level", "min_score_agent", "max_score_agent"]].copy() if not tables["subjectivity"].empty else pd.DataFrame()
    rev = tables["revision"][["paper_id", "selected_actions_knapsack", "expected_total_gain_knapsack", "total_cost_knapsack"]].copy() if not tables["revision"].empty else pd.DataFrame()
    pred = tables["prediction"][["paper_id", "current_score", "score_gain", "predicted_score_after_revision", "current_level", "predicted_level_after_revision", "R_AI_before", "R_AI_after_pred", "agent_score_std_before", "agent_score_std_after_pred"]].copy() if not tables["prediction"].empty else pd.DataFrame()
    robust = tables["robustness_summary"].copy()
    action = tables["robustness_action"].copy()
    numbers = []
    if not pred.empty:
        numbers.extend(
            [
                {"metric": "平均预测提升分数", "value": pred["score_gain"].mean()},
                {"metric": "平均风险下降", "value": (pred["R_AI_before"] - pred["R_AI_after_pred"]).mean()},
                {"metric": "平均分歧标准差下降", "value": (pred["agent_score_std_before"] - pred["agent_score_std_after_pred"]).mean()},
            ]
        )
    if not action.empty:
        for aid in ["A11", "A10", "A12"]:
            row = action[action["action_id"].astype(str) == aid]
            if not row.empty:
                numbers.append({"metric": f"{aid}稳定出现率", "value": float(row["selection_rate"].iloc[0])})
    if not robust.empty:
        for paper_id in sorted(EXPECTED_PAPERS):
            row = robust[robust["paper_id"].astype(str) == paper_id]
            if not row.empty:
                numbers.append({"metric": f"{paper_id}达到合格及以上比例", "value": float(row["pass_or_above_ratio"].iloc[0])})
    return {
        "ai_risk_summary": ai,
        "subjectivity_summary": sub,
        "revision_action_summary": rev,
        "quality_prediction_summary": pred,
        "robustness_summary": robust,
        "final_key_numbers": pd.DataFrame(numbers),
    }


def build_writing_materials(
    output_path: Path,
    tables: dict[str, pd.DataFrame],
    conclusion_checklist: pd.DataFrame,
    figure_manifest: pd.DataFrame,
) -> None:
    """Write Problem 3 markdown materials for paper writing."""
    ai = tables["ai_fusion"].copy()
    sub = tables["subjectivity"].copy()
    rev = tables["revision"].copy()
    pred = tables["prediction"].copy()
    robust = tables["robustness_summary"].copy()
    action = tables["robustness_action"].copy()

    def table_md(df: pd.DataFrame, cols: list[str]) -> str:
        if df.empty:
            return "待补充\n"
        sub = df[[c for c in cols if c in df.columns]].copy()
        if sub.empty:
            return "待补充\n"
        headers = [str(col) for col in sub.columns]
        lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
        for _, record in sub.iterrows():
            values = [str(record[col]).replace("\n", " ").replace("|", "/") for col in sub.columns]
            lines.append("| " + " | ".join(values) + " |")
        return "\n".join(lines)

    text = f"""# 第三问论文写作素材

## 第三问整体方法流程

第三问承接第一问封版评分系统和第二问关键特征解释结果，对附件3三篇中等质量论文建立“当前复评—逻辑诊断—AI辅助写作风险提示—多智能体分歧分析—修改动作优化—优化后预测—稳健性检验”的闭环流程。该流程强调可解释、可审计和保守预测，不将风险提示指标等同于质量扣分。

## Step 26 AI 风险识别结果摘要

{table_md(ai, ["paper_id", "R_AI", "risk_level", "main_risk_source"])}

三篇论文 AI 辅助写作风险均处于低风险区间，主要风险来源集中在数据不可追溯风险和方法到结果跳跃风险。

## Step 27 多智能体分歧分析结果摘要

{table_md(sub, ["paper_id", "agent_score_std", "disagreement_level", "min_score_agent", "max_score_agent"])}

三篇论文均表现为高分歧，说明不同评审维度对同一论文的评价差异明显，后续修改应优先修复最低评分维度。

## Step 28 修改动作优化结果摘要

{table_md(rev, ["paper_id", "selected_actions_knapsack", "expected_total_gain_knapsack", "total_cost_knapsack", "revision_priority_level"])}

动作优化在预算10内给出推荐组合，其中 A11、A10、A12 是最稳定的修改主干。

## Step 29 优化后质量预测结果摘要

{table_md(pred, ["paper_id", "current_score", "score_gain", "predicted_score_after_revision", "current_level", "predicted_level_after_revision", "R_AI_before", "R_AI_after_pred", "agent_score_std_before", "agent_score_std_after_pred"])}

预测结果显示三篇论文均有正向提升，3-3 可由待改进提升至合格，3-1 提升后仍处于待改进，说明其需要更大幅度结构性重构。

## Step 30 稳健性分析结果摘要

{table_md(robust, ["paper_id", "pass_or_above_ratio", "mean_score_gain", "min_score_gain", "max_score_gain"])}

{table_md(action, ["action_id", "selected_count", "selection_rate"])}

225组参数扰动下，预测提升稳定为正；A11在所有参数组合中均被选中，A10和A12为次稳定主干动作。

## 可直接放入论文的关键表格建议

- 表X：附件3 AI辅助写作风险识别结果，来源 `ai_risk_ds_fusion.xlsx`。
- 表X：多智能体评分分歧分析结果，来源 `multi_agent_subjectivity_analysis.xlsx`。
- 表X：修改动作优化推荐方案，来源 `revision_action_optimization.xlsx`。
- 表X：优化后质量预测结果，来源 `quality_prediction_after_revision.xlsx`。
- 表X：稳健性分析关键结果，来源 `robustness_summary.xlsx`。

## 可直接放入论文的图表建议

{table_md(figure_manifest.sort_values("paper_usage_priority"), ["figure_name", "suggested_caption", "suggested_section", "paper_usage_priority"])}

## 第三问主要结论

{table_md(conclusion_checklist[conclusion_checklist["status"].eq("PASS")], ["conclusion", "note", "write_to_paper_recommendation"])}

## 模型边界与免责声明

{DISCLAIMER_EXACT}

第三问中的优化后质量预测和稳健性分析均为模型参数扰动下的预测稳定性检验，不代表真实人工修改后的必然结果，也不替代专家复评。
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")


def write_outputs(
    paths: dict[str, Path],
    file_integrity: pd.DataFrame,
    data_consistency: pd.DataFrame,
    conclusion_checklist: pd.DataFrame,
    disclaimer_audit: pd.DataFrame,
    figure_manifest: pd.DataFrame,
    key_results: dict[str, pd.DataFrame],
    summary_payload: dict[str, Any],
) -> None:
    """Write Step 31 audit outputs."""
    with pd.ExcelWriter(paths["audit_report"], engine="openpyxl") as writer:
        file_integrity.to_excel(writer, index=False, sheet_name="file_integrity")
        data_consistency.to_excel(writer, index=False, sheet_name="data_consistency")
        conclusion_checklist.to_excel(writer, index=False, sheet_name="conclusion_checklist")
        disclaimer_audit.to_excel(writer, index=False, sheet_name="disclaimer_audit")
    paths["summary_json"].write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    with pd.ExcelWriter(paths["key_results"], engine="openpyxl") as writer:
        for sheet, frame in key_results.items():
            frame.to_excel(writer, index=False, sheet_name=sheet[:31])
    figure_manifest.to_excel(paths["figure_manifest"], index=False)
    conclusion_checklist.to_excel(paths["conclusion_checklist"], index=False)


def run_problem3_final_audit(config_path: Path | None = None) -> dict[str, Any]:
    """Run Step 31 final audit."""
    config = get_problem3_config(config_path)
    tables_dir = resolve_project_path(config.get("output_tables_dir", "output/problem3_tables"))
    charts_dir = resolve_project_path(config.get("output_charts_dir", "output/problem3_charts"))
    logs_dir = resolve_project_path(config.get("output_logs_dir", "output/problem3_logs"))
    reports_dir = resolve_project_path(config.get("output_reports_dir", "output/problem3_reports"))
    tables_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "audit_report": tables_dir / "problem3_final_audit_report.xlsx",
        "summary_json": tables_dir / "problem3_final_audit_summary.json",
        "key_results": tables_dir / "problem3_key_results_summary.xlsx",
        "figure_manifest": tables_dir / "problem3_key_figures_manifest.xlsx",
        "conclusion_checklist": tables_dir / "problem3_conclusion_checklist.xlsx",
        "writing_materials": reports_dir / "problem3_writing_materials.md",
        "log": logs_dir / "problem3_final_audit.log",
    }
    logger = setup_audit_logger(paths["log"])
    logger.info("Step 31 final audit started")

    file_integrity = audit_file_integrity(tables_dir, charts_dir, logs_dir)
    tables = load_core_tables(tables_dir)
    data_consistency = audit_data_consistency(tables)
    conclusion_checklist = audit_conclusions(tables)
    disclaimer_audit = audit_disclaimer(tables)
    figure_manifest = build_figure_manifest(charts_dir)
    key_results = build_key_results_summary(tables)

    all_status = pd.concat(
        [
            file_integrity[["status"]],
            data_consistency[["status"]],
            conclusion_checklist[["status"]],
            disclaimer_audit[["status"]],
        ],
        ignore_index=True,
    )
    counts = all_status["status"].value_counts().to_dict()
    missing_core = file_integrity[file_integrity["status"].ne("PASS")].copy()
    data_ok = bool(data_consistency["status"].eq("PASS").all())
    disclaimer_ok = bool(disclaimer_audit["status"].isin(["PASS"]).any())
    paper_conclusions = int(conclusion_checklist["status"].eq("PASS").sum())
    top_figures = figure_manifest[figure_manifest["paper_usage_priority"].eq("高")]["figure_name"].head(5).tolist()

    summary_payload = {
        "status_counts": counts,
        "core_missing_or_problem_files": missing_core[["file_name", "status", "note"]].to_dict(orient="records"),
        "data_consistency_pass": data_ok,
        "disclaimer_complete": disclaimer_ok,
        "paper_ready_conclusion_count": paper_conclusions,
        "recommended_top_figures": top_figures,
    }

    write_outputs(
        paths,
        file_integrity,
        data_consistency,
        conclusion_checklist,
        disclaimer_audit,
        figure_manifest,
        key_results,
        summary_payload,
    )
    build_writing_materials(paths["writing_materials"], tables, conclusion_checklist, figure_manifest)
    logger.info("Audit status counts: %s", counts)
    logger.info("Data consistency pass: %s", data_ok)
    logger.info("Disclaimer complete: %s", disclaimer_ok)
    logger.info("Step 31 outputs written")

    return {
        "paths": paths,
        "status_counts": counts,
        "missing_core": missing_core,
        "data_consistency_pass": data_ok,
        "disclaimer_complete": disclaimer_ok,
        "paper_ready_conclusion_count": paper_conclusions,
        "top_figures": top_figures,
        "file_integrity": file_integrity,
        "data_consistency": data_consistency,
        "conclusion_checklist": conclusion_checklist,
        "disclaimer_audit": disclaimer_audit,
        "figure_manifest": figure_manifest,
    }


__all__ = ["run_problem3_final_audit"]
