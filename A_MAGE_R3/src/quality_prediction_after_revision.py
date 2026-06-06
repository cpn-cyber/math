"""Problem 3 Step 29: quality prediction after recommended revisions.

The prediction is conservative and rule-based. It uses Step 28 revision plans
to estimate score improvement, AI-assisted writing risk indicator reduction,
and multi-agent disagreement convergence. It does not modify Step 26-28 source
results and does not claim that revisions have actually been completed.
"""

from __future__ import annotations

from pathlib import Path
import logging
import math
import re
from typing import Any

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

from modules.appendix3_pipeline import get_problem3_config, resolve_project_path


LOGGER_NAME = "A_MAGE_R3.problem3.quality_prediction_after_revision"
DISCLAIMER = "该预测仅表示文本风险提示指标预计下降，不构成学术不端判断，也不等同于 AI 生成判定。"

ACTION_DIMENSIONS = {
    "A1": {"structure", "application_value"},
    "A2": {"logic", "modeling"},
    "A3": {"logic", "modeling"},
    "A4": {"structure", "modeling"},
    "A5": {"logic", "modeling"},
    "A6": {"modeling", "result_validation"},
    "A7": {"result_validation"},
    "A8": {"result_validation"},
    "A9": {"result_validation"},
    "A10": {"result_validation", "application_value"},
    "A11": {"logic", "result_validation"},
    "A12": {"logic", "modeling", "result_validation"},
    "A13": {"structure"},
    "A14": {"logic", "application_value"},
}

AGENT_TO_DIMENSION = {
    "structure_reviewer": "structure",
    "logic_reviewer": "logic",
    "modeling_reviewer": "modeling",
    "result_validation_reviewer": "result_validation",
    "application_value_reviewer": "application_value",
}


def setup_prediction_logger(log_path: Path) -> logging.Logger:
    """Configure Step 29 logger."""
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


def _set_chinese_font() -> None:
    """Use a Chinese-capable matplotlib font when available."""
    candidates = ["Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC", "Arial Unicode MS"]
    available = {font.name for font in font_manager.fontManager.ttflist}
    for candidate in candidates:
        if candidate in available:
            plt.rcParams["font.sans-serif"] = [candidate]
            break
    plt.rcParams["axes.unicode_minus"] = False


def _natural_sort_key(value: Any) -> list[Any]:
    """Natural sort key for paper IDs."""
    text = Path(str(value)).stem
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def _safe_numeric(value: Any, default: float = 0.0) -> float:
    """Convert scalar to float with fallback."""
    number = pd.to_numeric(value, errors="coerce")
    return default if pd.isna(number) else float(number)


def _split_actions(value: Any) -> list[str]:
    """Split selected action IDs."""
    if value is None or pd.isna(value):
        return []
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _score_level(score: float) -> str:
    """Map score to post-revision level."""
    if score >= 85:
        return "优秀"
    if score >= 75:
        return "良好"
    if score >= 60:
        return "合格"
    return "待改进"


def _disagreement_level(std: float) -> str:
    """Map multi-agent standard deviation to disagreement level."""
    if std < 3:
        return "低分歧"
    if std < 8:
        return "中分歧"
    return "高分歧"


def _improvement_level(score_gain: float) -> str:
    """Map predicted score gain to improvement level."""
    if score_gain >= 8:
        return "显著提升"
    if score_gain >= 4:
        return "中等提升"
    return "轻微提升"


def _read_excel_if_exists(path: Path, *, sheet_name: str | int = 0) -> pd.DataFrame:
    """Read Excel if present."""
    return pd.read_excel(path, sheet_name=sheet_name) if path.exists() else pd.DataFrame()


def load_step29_inputs(tables_dir: Path, logger: logging.Logger) -> tuple[dict[str, pd.DataFrame], list[Path]]:
    """Load all Step 29 input tables."""
    sources = {
        "revision_plan": tables_dir / "revision_action_optimization.xlsx",
        "revision_details": tables_dir / "revision_action_details.xlsx",
        "subjectivity": tables_dir / "multi_agent_subjectivity_analysis.xlsx",
        "ai_fusion": tables_dir / "ai_risk_ds_fusion.xlsx",
        "ai_evidence": tables_dir / "ai_risk_evidence.xlsx",
        "current_evaluation": tables_dir / "appendix3_current_evaluation.xlsx",
        "problem1_base": tables_dir / "appendix3_problem1_base_scores.xlsx",
        "features_normalized": tables_dir / "appendix3_features_normalized.xlsx",
    }
    data: dict[str, pd.DataFrame] = {}
    used: list[Path] = []
    for key, path in sources.items():
        if not path.exists():
            logger.warning("Optional Step 29 input missing: %s", path)
            data[key] = pd.DataFrame()
            continue
        try:
            sheet: str | int = "action_details" if key == "revision_details" else 0
            data[key] = pd.read_excel(path, sheet_name=sheet)
            used.append(path)
            logger.info("Loaded %s from %s", key, path)
        except Exception as exc:  # pragma: no cover - defensive I/O guard
            logger.exception("Failed to load %s from %s: %s", key, path, exc)
            data[key] = pd.DataFrame()
    return data, used


def _merge_inputs(data: dict[str, pd.DataFrame], logger: logging.Logger) -> pd.DataFrame:
    """Merge Step 29 inputs by paper_id."""
    plan = data.get("revision_plan", pd.DataFrame()).copy()
    if plan.empty:
        raise FileNotFoundError("revision_action_optimization.xlsx is required for Step 29.")
    subjectivity = data.get("subjectivity", pd.DataFrame()).copy()
    if not subjectivity.empty:
        value_cols = [
            col
            for col in [
                "agent_score_mean",
                "agent_score_std",
                "disagreement_level",
                "min_score_agent",
                "R_AI",
                "risk_level",
                "main_risk_source",
            ]
            if col in subjectivity.columns and col not in plan.columns
        ]
        cols = ["paper_id"] + value_cols if "paper_id" in subjectivity.columns else value_cols
        if cols:
            plan = plan.merge(subjectivity[cols], on="paper_id", how="left")

    current = data.get("current_evaluation", pd.DataFrame()).copy()
    if not current.empty:
        keep = [
            col
            for col in [
                "paper_id",
                "current_score",
                "final_score",
                "total_score",
                "Q_cur_baseline",
                "current_grade",
                "F1_score",
                "F2_key_score",
            ]
            if col in current.columns
        ]
        if keep:
            plan = plan.merge(current[keep], on="paper_id", how="left", suffixes=("", "_current_eval"))

    problem1 = data.get("problem1_base", pd.DataFrame()).copy()
    if not problem1.empty:
        keep = [col for col in ["paper_id", "F1_score"] if col in problem1.columns]
        if keep:
            plan = plan.merge(problem1[keep], on="paper_id", how="left", suffixes=("", "_problem1"))

    ai_fusion = data.get("ai_fusion", pd.DataFrame()).copy()
    if not ai_fusion.empty:
        keep = [col for col in ["paper_id", "R_AI", "risk_level", "main_risk_source"] if col in ai_fusion.columns]
        rename = {col: f"{col}_ai_fusion" for col in keep if col != "paper_id" and col in plan.columns}
        ai_fusion = ai_fusion[keep].rename(columns=rename)
        plan = plan.merge(ai_fusion, on="paper_id", how="left")

    logger.info("Merged Step 29 input columns: %s", list(plan.columns))
    return plan.sort_values("paper_id", key=lambda values: values.map(_natural_sort_key)).reset_index(drop=True)


def _choose_current_score(row: pd.Series) -> tuple[float, str]:
    """Choose the current score source with conservative fallbacks."""
    candidate_columns = [
        "current_score",
        "final_score",
        "total_score",
        "Q_cur_baseline",
        "agent_score_mean",
        "F1_score",
        "F1_score_problem1",
    ]
    for column in candidate_columns:
        if column in row.index and not pd.isna(row.get(column)):
            return _safe_numeric(row.get(column)), column
    return 0.0, "missing_default_0"


def _risk_reduction(actions: list[str]) -> tuple[float, dict[str, float]]:
    """Calculate predicted AI-risk indicator reduction from selected actions."""
    components = {
        "data_traceability_risk_reduction": 0.03 if "A11" in actions else 0.0,
        "method_jump_risk_reduction_A12": 0.03 if "A12" in actions else 0.0,
        "method_jump_risk_reduction_A7": 0.015 if "A7" in actions else 0.0,
        "writing_application_risk_reduction": 0.015 if ("A10" in actions or "A14" in actions) else 0.0,
    }
    return float(sum(components.values())), components


def _std_reduction(row: pd.Series, actions: list[str]) -> tuple[float, dict[str, float]]:
    """Calculate predicted disagreement standard deviation reduction."""
    std_before = _safe_numeric(row.get("agent_score_std"), default=0.0)
    weakest_agent = str(row.get("weakest_agent", row.get("min_score_agent", "")))
    weakest_dimension = AGENT_TO_DIMENSION.get(weakest_agent, "")
    selected_dimensions: set[str] = set()
    for action in actions:
        selected_dimensions.update(ACTION_DIMENSIONS.get(action, set()))

    components = {
        "weakest_dimension_targeted": 0.25 * std_before if weakest_dimension and weakest_dimension in selected_dimensions else 0.0,
        "three_or_more_actions": 0.10 * std_before if len(actions) >= 3 else 0.0,
        "evidence_chain_actions_A11_A12": 0.05 * std_before if ("A11" in actions or "A12" in actions) else 0.0,
    }
    return float(sum(components.values())), components


def predict_after_revision(merged: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build paper-level predictions and detailed calculation records."""
    rows: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []

    for _, row in merged.iterrows():
        paper_id = str(row["paper_id"])
        actions = _split_actions(row.get("selected_actions_knapsack"))
        current_score, current_score_source = _choose_current_score(row)
        expected_gain = _safe_numeric(row.get("expected_total_gain_knapsack"), default=0.0)
        raw_score_gain = expected_gain * 0.12
        score_gain = min(12.0, raw_score_gain)
        predicted_score = min(100.0, current_score + score_gain)

        r_ai_before = _safe_numeric(row.get("R_AI", row.get("R_AI_ai_fusion")), default=0.0)
        risk_total, risk_components = _risk_reduction(actions)
        r_ai_after = max(0.0, r_ai_before - risk_total)

        std_before = _safe_numeric(row.get("agent_score_std"), default=0.0)
        std_total, std_components = _std_reduction(row, actions)
        std_after = max(0.0, std_before - std_total)

        explanation = (
            f"{paper_id} 当前分数来源为 {current_score_source}。"
            f"Step28 预计收益 {expected_gain:.3f} 按 0.12 映射为 {raw_score_gain:.3f} 分，"
            f"受单篇最高 12 分上限约束后 score_gain={score_gain:.3f}。"
            f"推荐动作 {','.join(actions)} 预计使 AI风险提示指标下降 {risk_total:.3f}，"
            f"多智能体分歧标准差下降 {std_total:.3f}。"
        )

        rows.append(
            {
                "paper_id": paper_id,
                "current_score": current_score,
                "score_gain": score_gain,
                "predicted_score_after_revision": predicted_score,
                "current_level": _score_level(current_score),
                "predicted_level_after_revision": _score_level(predicted_score),
                "selected_actions_knapsack": ",".join(actions),
                "expected_total_gain_knapsack": expected_gain,
                "total_cost_knapsack": _safe_numeric(row.get("total_cost_knapsack"), default=0.0),
                "R_AI_before": r_ai_before,
                "R_AI_after_pred": r_ai_after,
                "risk_reduction_total": risk_total,
                "agent_score_std_before": std_before,
                "agent_score_std_after_pred": std_after,
                "disagreement_level_before": row.get("disagreement_level", _disagreement_level(std_before)),
                "disagreement_level_after_pred": _disagreement_level(std_after),
                "quality_improvement_level": _improvement_level(score_gain),
                "prediction_explanation": explanation,
                "disclaimer": DISCLAIMER,
                "current_score_source": current_score_source,
            }
        )

        detail = {
            "paper_id": paper_id,
            "selected_actions": ",".join(actions),
            "current_score_source": current_score_source,
            "expected_gain_to_score_mapping": "score_gain=min(expected_total_gain_knapsack*0.12, 12)",
            "raw_score_gain": raw_score_gain,
            "score_gain_capped": score_gain,
            "std_reduction_total": std_total,
        }
        detail.update(risk_components)
        detail.update(std_components)
        details.append(detail)

    return pd.DataFrame(rows), pd.DataFrame(details)


def build_summary(predictions: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build summary workbook sheets."""
    score_before_after = predictions[
        ["paper_id", "current_score", "predicted_score_after_revision", "score_gain", "current_level", "predicted_level_after_revision"]
    ].copy()
    risk_before_after = predictions[
        ["paper_id", "R_AI_before", "R_AI_after_pred", "risk_reduction_total"]
    ].copy()
    disagreement_before_after = predictions[
        ["paper_id", "agent_score_std_before", "agent_score_std_after_pred", "disagreement_level_before", "disagreement_level_after_pred"]
    ].copy()
    improvement_summary = (
        predictions.groupby("quality_improvement_level", as_index=False)
        .agg(paper_count=("paper_id", "count"), avg_score_gain=("score_gain", "mean"))
        .sort_values("quality_improvement_level")
    )
    interpretation = predictions[["paper_id", "prediction_explanation"]].copy()
    interpretation.loc[len(interpretation)] = {
        "paper_id": "method_boundary",
        "prediction_explanation": (
            "本步骤为基于修改动作的保守预测：分数提升、风险下降和分歧收敛均来自规则映射，"
            "不表示论文已经修改完成，也不替代真实复评。"
        ),
    }
    interpretation.loc[len(interpretation)] = {"paper_id": "risk_disclaimer", "prediction_explanation": DISCLAIMER}
    return {
        "prediction_overview": predictions,
        "score_before_after": score_before_after,
        "risk_before_after": risk_before_after,
        "disagreement_before_after": disagreement_before_after,
        "improvement_level_summary": improvement_summary,
        "interpretation_text": interpretation,
    }


def draw_quality_improvement_bar(predictions: pd.DataFrame, output_path: Path) -> None:
    """Draw predicted score gain by paper."""
    _set_chinese_font()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = predictions.sort_values("paper_id", key=lambda values: values.map(_natural_sort_key))
    fig, ax = plt.subplots(figsize=(7.2, 4.5), dpi=160)
    bars = ax.bar(data["paper_id"].astype(str), data["score_gain"], color="#4C78A8")
    ax.set_title("附件3论文修改后预测提升分数", fontsize=12, pad=12)
    ax.set_ylabel("预测提升分数")
    ax.set_ylim(0, max(12, float(data["score_gain"].max()) + 1))
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15, f"{bar.get_height():.2f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def draw_before_after(
    predictions: pd.DataFrame,
    before_col: str,
    after_col: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    """Draw grouped before/after bars."""
    _set_chinese_font()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = predictions.sort_values("paper_id", key=lambda values: values.map(_natural_sort_key))
    x = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(7.6, 4.8), dpi=160)
    ax.bar(x - 0.18, data[before_col], width=0.36, label="修改前", color="#4C78A8")
    ax.bar(x + 0.18, data[after_col], width=0.36, label="预测修改后", color="#F58518")
    ax.set_xticks(x)
    ax.set_xticklabels(data["paper_id"].astype(str).tolist())
    ax.set_title(title, fontsize=12, pad=12)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def write_outputs(
    paths: dict[str, Path],
    predictions: pd.DataFrame,
    details: pd.DataFrame,
    summaries: dict[str, pd.DataFrame],
) -> None:
    """Write Step 29 output tables."""
    paths["tables_dir"].mkdir(parents=True, exist_ok=True)
    predictions.to_excel(paths["prediction"], index=False)
    details.to_excel(paths["details"], index=False)
    with pd.ExcelWriter(paths["summary"], engine="openpyxl") as writer:
        for sheet_name, frame in summaries.items():
            frame.to_excel(writer, index=False, sheet_name=sheet_name[:31])


def run_quality_prediction_after_revision(config_path: Path | None = None) -> dict[str, Any]:
    """Run Step 29 quality prediction after revision."""
    config = get_problem3_config(config_path)
    tables_dir = resolve_project_path(config.get("output_tables_dir", "output/problem3_tables"))
    charts_dir = resolve_project_path(config.get("output_charts_dir", "output/problem3_charts"))
    logs_dir = resolve_project_path(config.get("output_logs_dir", "output/problem3_logs"))
    tables_dir.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "tables_dir": tables_dir,
        "charts_dir": charts_dir,
        "logs_dir": logs_dir,
        "prediction": tables_dir / "quality_prediction_after_revision.xlsx",
        "details": tables_dir / "quality_prediction_details.xlsx",
        "summary": tables_dir / "quality_prediction_summary.xlsx",
        "improvement_bar": charts_dir / "quality_improvement_bar.png",
        "score_before_after": charts_dir / "score_before_after.png",
        "risk_before_after": charts_dir / "risk_before_after.png",
        "disagreement_before_after": charts_dir / "disagreement_before_after.png",
        "log": logs_dir / "quality_prediction_after_revision.log",
    }

    logger = setup_prediction_logger(paths["log"])
    logger.info("Step 29 quality prediction after revision started")
    logger.info("Score gain mapping: score_gain=min(expected_total_gain_knapsack*0.12, 12)")
    logger.info("Risk reduction rules: A11=0.03 data; A12=0.03 method; A7=0.015 method; A10/A14=0.015 writing/application")
    logger.info("Disagreement reduction rules: weakest target=0.25 std; >=3 actions=0.10 std; A11/A12=0.05 std")

    data, used_sources = load_step29_inputs(tables_dir, logger)
    merged = _merge_inputs(data, logger)
    predictions, details = predict_after_revision(merged)
    summaries = build_summary(predictions)
    write_outputs(paths, predictions, details, summaries)

    draw_quality_improvement_bar(predictions, paths["improvement_bar"])
    draw_before_after(predictions, "current_score", "predicted_score_after_revision", "修改前后预测质量得分对比", "质量得分", paths["score_before_after"])
    draw_before_after(predictions, "R_AI_before", "R_AI_after_pred", "AI辅助写作风险提示指标修改前后预测", "R_AI", paths["risk_before_after"])
    draw_before_after(predictions, "agent_score_std_before", "agent_score_std_after_pred", "多智能体分歧标准差修改前后预测", "标准差", paths["disagreement_before_after"])

    logger.info("Used sources: %s", [str(path) for path in used_sources])
    logger.info("Average score gain: %.6f", float(predictions["score_gain"].mean()))
    logger.info("Average risk reduction: %.6f", float(predictions["risk_reduction_total"].mean()))
    logger.info(
        "Average disagreement std reduction: %.6f",
        float((predictions["agent_score_std_before"] - predictions["agent_score_std_after_pred"]).mean()),
    )
    logger.info("Step 29 outputs written")

    return {
        "paths": paths,
        "used_sources": used_sources,
        "predictions": predictions,
        "details": details,
        "summaries": summaries,
        "avg_score_gain": float(predictions["score_gain"].mean()),
        "avg_risk_reduction": float(predictions["risk_reduction_total"].mean()),
        "avg_disagreement_reduction": float((predictions["agent_score_std_before"] - predictions["agent_score_std_after_pred"]).mean()),
    }


__all__ = ["run_quality_prediction_after_revision", "predict_after_revision"]
