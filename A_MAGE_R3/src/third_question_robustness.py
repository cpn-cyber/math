"""Problem 3 Step 30: robustness analysis.

This step perturbs the score-gain mapping, revision budget, AI-risk reduction
multiplier, and disagreement-convergence multiplier. For every parameter
combination it re-solves the action-selection knapsack by paper, then recomputes
post-revision score, AI-risk indicator, and multi-agent disagreement estimates.
"""

from __future__ import annotations

from pathlib import Path
import itertools
import logging
import math
import re
from typing import Any

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

from modules.appendix3_pipeline import get_problem3_config, resolve_project_path


LOGGER_NAME = "A_MAGE_R3.problem3.robustness"

SCORE_GAIN_FACTORS = [0.08, 0.10, 0.12, 0.14, 0.16]
BUDGETS = [6, 8, 10, 12, 14]
RISK_MULTIPLIERS = [0.8, 1.0, 1.2]
DISAGREEMENT_MULTIPLIERS = [0.8, 1.0, 1.2]

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

DISCLAIMER = "稳健性分析不表示真实人工修改后的必然结果，而是模型参数扰动下的预测稳定性检验。"


def setup_robustness_logger(log_path: Path) -> logging.Logger:
    """Configure Step 30 logger."""
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
    """Use a Chinese-capable matplotlib font if available."""
    candidates = ["Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC", "Arial Unicode MS"]
    available = {font.name for font in font_manager.fontManager.ttflist}
    for candidate in candidates:
        if candidate in available:
            plt.rcParams["font.sans-serif"] = [candidate]
            break
    plt.rcParams["axes.unicode_minus"] = False


def _natural_sort_key(value: Any) -> list[Any]:
    """Natural sort key for paper IDs and action IDs."""
    text = Path(str(value)).stem
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def _safe_numeric(value: Any, default: float = 0.0) -> float:
    """Convert scalar to float with fallback."""
    number = pd.to_numeric(value, errors="coerce")
    return default if pd.isna(number) else float(number)


def _split_actions(value: Any) -> list[str]:
    """Split action IDs."""
    if value is None or pd.isna(value):
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _score_level(score: float) -> str:
    """Map quality score to four-level prediction label."""
    if score >= 85:
        return "优秀"
    if score >= 75:
        return "良好"
    if score >= 60:
        return "合格"
    return "待改进"


def _disagreement_level(std: float) -> str:
    """Map disagreement standard deviation to level."""
    if std < 3:
        return "低分歧"
    if std < 8:
        return "中分歧"
    return "高分歧"


def build_parameter_grid() -> pd.DataFrame:
    """Build the required 5 x 5 x 3 x 3 parameter grid."""
    rows: list[dict[str, Any]] = []
    scenario_id = 1
    for score_factor, budget, risk_multiplier, disagreement_multiplier in itertools.product(
        SCORE_GAIN_FACTORS,
        BUDGETS,
        RISK_MULTIPLIERS,
        DISAGREEMENT_MULTIPLIERS,
    ):
        rows.append(
            {
                "scenario_id": f"S{scenario_id:03d}",
                "score_gain_factor": score_factor,
                "budget": budget,
                "risk_reduction_multiplier": risk_multiplier,
                "disagreement_reduction_multiplier": disagreement_multiplier,
            }
        )
        scenario_id += 1
    return pd.DataFrame(rows)


def load_step30_inputs(tables_dir: Path, logger: logging.Logger) -> tuple[dict[str, pd.DataFrame], list[Path]]:
    """Load Step 30 input tables."""
    sources = {
        "prediction": tables_dir / "quality_prediction_after_revision.xlsx",
        "prediction_details": tables_dir / "quality_prediction_details.xlsx",
        "revision_plan": tables_dir / "revision_action_optimization.xlsx",
        "revision_details": tables_dir / "revision_action_details.xlsx",
        "action_library": tables_dir / "revision_action_library.xlsx",
        "subjectivity": tables_dir / "multi_agent_subjectivity_analysis.xlsx",
        "ai_fusion": tables_dir / "ai_risk_ds_fusion.xlsx",
        "ai_evidence": tables_dir / "ai_risk_evidence.xlsx",
    }
    data: dict[str, pd.DataFrame] = {}
    used: list[Path] = []
    for key, path in sources.items():
        if not path.exists():
            logger.warning("Optional Step 30 input missing: %s", path)
            data[key] = pd.DataFrame()
            continue
        sheet: str | int = "action_details" if key == "revision_details" else 0
        try:
            data[key] = pd.read_excel(path, sheet_name=sheet)
            used.append(path)
            logger.info("Loaded %s from %s", key, path)
        except Exception as exc:  # pragma: no cover - defensive I/O guard
            logger.exception("Failed to load %s from %s: %s", key, path, exc)
            data[key] = pd.DataFrame()
    return data, used


def _current_base_table(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Create base paper-level table for robustness simulations."""
    prediction = data.get("prediction", pd.DataFrame()).copy()
    if prediction.empty:
        raise FileNotFoundError("quality_prediction_after_revision.xlsx is required for Step 30.")

    base = prediction[
        [
            "paper_id",
            "current_score",
            "R_AI_before",
            "agent_score_std_before",
        ]
    ].copy()

    revision_plan = data.get("revision_plan", pd.DataFrame()).copy()
    if not revision_plan.empty:
        keep = [
            col
            for col in [
                "paper_id",
                "filename",
                "weakest_agent",
                "strongest_agent",
                "disagreement_level",
                "risk_level",
                "main_risk_sources",
            ]
            if col in revision_plan.columns
        ]
        base = base.merge(revision_plan[keep], on="paper_id", how="left")

    subjectivity = data.get("subjectivity", pd.DataFrame()).copy()
    if not subjectivity.empty:
        keep = [col for col in ["paper_id", "min_score_agent", "max_score_agent"] if col in subjectivity.columns]
        base = base.merge(subjectivity[keep], on="paper_id", how="left", suffixes=("", "_subjectivity"))

    base["weakest_agent"] = base.get("weakest_agent", pd.Series(index=base.index, dtype=object)).fillna(
        base.get("min_score_agent", pd.Series(index=base.index, dtype=object))
    )
    base["filename"] = base.get("filename", pd.Series(index=base.index, dtype=object)).fillna(base["paper_id"].astype(str) + ".txt")
    return base.sort_values("paper_id", key=lambda values: values.map(_natural_sort_key)).reset_index(drop=True)


def _knapsack_select(action_details: pd.DataFrame, budget: int) -> list[str]:
    """0-1 knapsack selection maximizing adjusted expected gain."""
    items = action_details.sort_values("action_id", key=lambda values: values.map(_natural_sort_key)).reset_index(drop=True)
    n = len(items)
    dp = np.zeros((n + 1, budget + 1), dtype=float)
    keep = np.zeros((n + 1, budget + 1), dtype=bool)
    for i in range(1, n + 1):
        cost = int(items.loc[i - 1, "cost"])
        gain = float(items.loc[i - 1, "adjusted_expected_gain"])
        for w in range(budget + 1):
            dp[i, w] = dp[i - 1, w]
            if cost <= w:
                candidate = dp[i - 1, w - cost] + gain
                if candidate > dp[i, w] + 1e-12:
                    dp[i, w] = candidate
                    keep[i, w] = True
    selected: list[str] = []
    w = budget
    for i in range(n, 0, -1):
        if keep[i, w]:
            action_id = str(items.loc[i - 1, "action_id"])
            selected.append(action_id)
            w -= int(items.loc[i - 1, "cost"])
    return list(reversed(selected))


def _risk_reduction(actions: list[str], multiplier: float) -> tuple[float, dict[str, float]]:
    """Calculate risk indicator reduction under multiplier."""
    components = {
        "data_traceability_risk_reduction": 0.03 if "A11" in actions else 0.0,
        "method_jump_risk_reduction_A12": 0.03 if "A12" in actions else 0.0,
        "method_jump_risk_reduction_A7": 0.015 if "A7" in actions else 0.0,
        "writing_application_risk_reduction": 0.015 if ("A10" in actions or "A14" in actions) else 0.0,
    }
    total = sum(components.values()) * multiplier
    return float(total), {key: value * multiplier for key, value in components.items()}


def _std_reduction(row: pd.Series, actions: list[str], multiplier: float) -> tuple[float, dict[str, float]]:
    """Calculate disagreement std reduction under multiplier."""
    std_before = _safe_numeric(row.get("agent_score_std_before"), default=0.0)
    weakest_agent = str(row.get("weakest_agent", row.get("min_score_agent", "")))
    weakest_dimension = AGENT_TO_DIMENSION.get(weakest_agent, "")
    selected_dimensions: set[str] = set()
    for action in actions:
        selected_dimensions.update(ACTION_DIMENSIONS.get(action, set()))
    base_components = {
        "weakest_dimension_targeted": 0.25 * std_before if weakest_dimension and weakest_dimension in selected_dimensions else 0.0,
        "three_or_more_actions": 0.10 * std_before if len(actions) >= 3 else 0.0,
        "evidence_chain_actions_A11_A12": 0.05 * std_before if ("A11" in actions or "A12" in actions) else 0.0,
    }
    components = {key: value * multiplier for key, value in base_components.items()}
    return float(sum(components.values())), components


def run_robustness_simulation(
    base: pd.DataFrame,
    action_details: pd.DataFrame,
    parameter_grid: pd.DataFrame,
) -> pd.DataFrame:
    """Run 225-parameter robustness scenarios for every paper."""
    if action_details.empty:
        raise FileNotFoundError("revision_action_details.xlsx/action_details is required for Step 30.")

    rows: list[dict[str, Any]] = []
    for _, scenario in parameter_grid.iterrows():
        budget = int(scenario["budget"])
        score_gain_factor = float(scenario["score_gain_factor"])
        risk_multiplier = float(scenario["risk_reduction_multiplier"])
        disagreement_multiplier = float(scenario["disagreement_reduction_multiplier"])

        for _, paper_row in base.iterrows():
            paper_id = str(paper_row["paper_id"])
            paper_actions = action_details.loc[action_details["paper_id"].astype(str) == paper_id].copy()
            selected_actions = _knapsack_select(paper_actions, budget)
            selected_df = paper_actions.loc[paper_actions["action_id"].isin(selected_actions)]
            expected_gain = float(selected_df["adjusted_expected_gain"].sum())
            total_cost = int(selected_df["cost"].sum())
            score_gain = min(12.0, expected_gain * score_gain_factor)
            current_score = _safe_numeric(paper_row.get("current_score"), default=0.0)
            predicted_score = min(100.0, current_score + score_gain)

            risk_reduction, risk_components = _risk_reduction(selected_actions, risk_multiplier)
            r_ai_before = _safe_numeric(paper_row.get("R_AI_before"), default=0.0)
            r_ai_after = max(0.0, r_ai_before - risk_reduction)

            std_reduction, std_components = _std_reduction(paper_row, selected_actions, disagreement_multiplier)
            std_before = _safe_numeric(paper_row.get("agent_score_std_before"), default=0.0)
            std_after = max(0.0, std_before - std_reduction)

            row = {
                "scenario_id": scenario["scenario_id"],
                "paper_id": paper_id,
                "filename": paper_row.get("filename", f"{paper_id}.txt"),
                "score_gain_factor": score_gain_factor,
                "budget": budget,
                "risk_reduction_multiplier": risk_multiplier,
                "disagreement_reduction_multiplier": disagreement_multiplier,
                "selected_actions": ",".join(selected_actions),
                "selected_action_count": len(selected_actions),
                "total_cost": total_cost,
                "expected_gain": expected_gain,
                "current_score": current_score,
                "score_gain": score_gain,
                "predicted_score_after_revision": predicted_score,
                "predicted_level_after_revision": _score_level(predicted_score),
                "R_AI_before": r_ai_before,
                "R_AI_after_pred": r_ai_after,
                "risk_reduction_total": risk_reduction,
                "agent_score_std_before": std_before,
                "agent_score_std_after_pred": std_after,
                "std_reduction": std_reduction,
                "disagreement_level_after_pred": _disagreement_level(std_after),
                "reaches_pass_or_above": predicted_score >= 60,
                "risk_decreased": risk_reduction > 0,
                "disagreement_decreased": std_reduction > 0,
            }
            row.update(risk_components)
            row.update(std_components)
            rows.append(row)
    return pd.DataFrame(rows)


def calculate_stability_tables(results: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Calculate Step 30 robustness summary tables."""
    scenario_overview = (
        results.groupby("scenario_id", as_index=False)
        .agg(
            mean_score_gain=("score_gain", "mean"),
            mean_predicted_score=("predicted_score_after_revision", "mean"),
            mean_risk_reduction=("risk_reduction_total", "mean"),
            mean_std_reduction=("std_reduction", "mean"),
            avg_action_count=("selected_action_count", "mean"),
        )
    )

    prediction_overview = pd.DataFrame(
        [
            {
                "metric": "mean_score_gain",
                "min": scenario_overview["mean_score_gain"].min(),
                "max": scenario_overview["mean_score_gain"].max(),
                "mean": scenario_overview["mean_score_gain"].mean(),
                "std": scenario_overview["mean_score_gain"].std(ddof=0),
            },
            {
                "metric": "proportion_positive_gain",
                "min": results.groupby("scenario_id")["score_gain"].apply(lambda s: float((s > 0).mean())).min(),
                "max": results.groupby("scenario_id")["score_gain"].apply(lambda s: float((s > 0).mean())).max(),
                "mean": float((results["score_gain"] > 0).mean()),
                "std": 0.0,
            },
        ]
    )

    paper_rows: list[dict[str, Any]] = []
    for paper_id, group in results.groupby("paper_id"):
        level_counts = group["predicted_level_after_revision"].value_counts(normalize=True)
        paper_rows.append(
            {
                "paper_id": paper_id,
                "mean_score_gain": group["score_gain"].mean(),
                "min_score_gain": group["score_gain"].min(),
                "max_score_gain": group["score_gain"].max(),
                "score_gain_std": group["score_gain"].std(ddof=0),
                "proportion_positive_gain": float((group["score_gain"] > 0).mean()),
                "pass_or_above_ratio": float(group["reaches_pass_or_above"].mean()),
                "level_ratio_优秀": float(level_counts.get("优秀", 0.0)),
                "level_ratio_良好": float(level_counts.get("良好", 0.0)),
                "level_ratio_合格": float(level_counts.get("合格", 0.0)),
                "level_ratio_待改进": float(level_counts.get("待改进", 0.0)),
                "mean_predicted_score": group["predicted_score_after_revision"].mean(),
                "min_predicted_score": group["predicted_score_after_revision"].min(),
                "max_predicted_score": group["predicted_score_after_revision"].max(),
            }
        )
    paper_stability = pd.DataFrame(paper_rows).sort_values("paper_id", key=lambda values: values.map(_natural_sort_key))

    action_rows: list[dict[str, Any]] = []
    for _, row in results.iterrows():
        for action in _split_actions(row["selected_actions"]):
            action_rows.append({"paper_id": row["paper_id"], "scenario_id": row["scenario_id"], "action_id": action})
    action_long = pd.DataFrame(action_rows)
    if action_long.empty:
        action_stability = pd.DataFrame(columns=["action_id", "selected_count", "selection_rate"])
        paper_action_stability = pd.DataFrame()
    else:
        total_paper_scenarios = len(results)
        action_stability = (
            action_long.groupby("action_id", as_index=False)
            .agg(selected_count=("scenario_id", "count"))
            .assign(selection_rate=lambda df: df["selected_count"] / total_paper_scenarios)
            .sort_values(["selected_count", "action_id"], ascending=[False, True])
        )
        paper_action_stability = (
            action_long.groupby(["paper_id", "action_id"], as_index=False)
            .agg(selected_count=("scenario_id", "count"))
        )
        scenario_count = results["scenario_id"].nunique()
        paper_action_stability["paper_selection_rate"] = paper_action_stability["selected_count"] / scenario_count
        paper_action_stability = paper_action_stability.sort_values(
            ["paper_id", "selected_count", "action_id"], ascending=[True, False, True]
        )
        paper_action_stability["paper_action_rank"] = paper_action_stability.groupby("paper_id")["selected_count"].rank(
            method="first", ascending=False
        )

    risk_stability = (
        results.groupby("paper_id", as_index=False)
        .agg(
            mean_risk_reduction=("risk_reduction_total", "mean"),
            min_risk_reduction=("risk_reduction_total", "min"),
            max_risk_reduction=("risk_reduction_total", "max"),
            proportion_risk_decrease=("risk_decreased", "mean"),
            mean_R_AI_after=("R_AI_after_pred", "mean"),
        )
        .sort_values("paper_id", key=lambda values: values.map(_natural_sort_key))
    )
    overall_risk = pd.DataFrame(
        [
            {
                "paper_id": "overall",
                "mean_risk_reduction": results["risk_reduction_total"].mean(),
                "min_risk_reduction": results["risk_reduction_total"].min(),
                "max_risk_reduction": results["risk_reduction_total"].max(),
                "proportion_risk_decrease": float(results["risk_decreased"].mean()),
                "mean_R_AI_after": results["R_AI_after_pred"].mean(),
            }
        ]
    )
    risk_stability = pd.concat([risk_stability, overall_risk], ignore_index=True)

    disagreement_rows: list[dict[str, Any]] = []
    for paper_id, group in results.groupby("paper_id"):
        level_improved = group["disagreement_level_after_pred"].isin(["中分歧", "低分歧"])
        disagreement_rows.append(
            {
                "paper_id": paper_id,
                "mean_std_reduction": group["std_reduction"].mean(),
                "min_std_reduction": group["std_reduction"].min(),
                "max_std_reduction": group["std_reduction"].max(),
                "proportion_disagreement_decrease": float(group["disagreement_decreased"].mean()),
                "ratio_high_to_mid_or_low": float(level_improved.mean()),
                "mean_std_after": group["agent_score_std_after_pred"].mean(),
                "min_std_after": group["agent_score_std_after_pred"].min(),
                "max_std_after": group["agent_score_std_after_pred"].max(),
            }
        )
    disagreement_stability = pd.DataFrame(disagreement_rows).sort_values("paper_id", key=lambda values: values.map(_natural_sort_key))

    key_conclusions = build_key_conclusions(
        results,
        paper_stability,
        action_stability,
        risk_stability,
        disagreement_stability,
    )
    return {
        "scenario_overview": scenario_overview,
        "prediction_robustness_overview": prediction_overview,
        "paper_level_stability": paper_stability,
        "action_stability": action_stability,
        "paper_action_stability": paper_action_stability,
        "risk_reduction_stability": risk_stability,
        "disagreement_stability": disagreement_stability,
        "key_conclusions": key_conclusions,
        "interpretation_text": key_conclusions[["item", "conclusion"]].copy(),
    }


def build_key_conclusions(
    results: pd.DataFrame,
    paper_stability: pd.DataFrame,
    action_stability: pd.DataFrame,
    risk_stability: pd.DataFrame,
    disagreement_stability: pd.DataFrame,
) -> pd.DataFrame:
    """Generate Chinese robustness conclusions."""
    rows: list[dict[str, Any]] = []
    positive_ratio = float((results["score_gain"] > 0).mean())
    score_gain_min = float(results.groupby("scenario_id")["score_gain"].mean().min())
    score_gain_max = float(results.groupby("scenario_id")["score_gain"].mean().max())
    rows.append(
        {
            "item": "score_gain_stability",
            "conclusion": f"225组参数扰动下，预测提升稳定为正，整体正向提升比例为{positive_ratio:.3f}，场景平均提升区间为{score_gain_min:.3f}~{score_gain_max:.3f}。",
        }
    )

    top_actions = action_stability.head(5)["action_id"].astype(str).tolist() if not action_stability.empty else []
    rows.append(
        {
            "item": "action_stability",
            "conclusion": f"稳定高频动作Top包括{','.join(top_actions)}；其中A11、A12、A10若进入高频，说明数据追溯、方法-结果链条和模型评价推广是稳健修复方向。",
        }
    )

    ratio_33 = paper_stability.loc[paper_stability["paper_id"].astype(str) == "3-3", "pass_or_above_ratio"]
    ratio_31 = paper_stability.loc[paper_stability["paper_id"].astype(str) == "3-1", "pass_or_above_ratio"]
    rows.append(
        {
            "item": "paper_3_3_pass_ratio",
            "conclusion": f"3-3达到合格及以上比例为{float(ratio_33.iloc[0]) if not ratio_33.empty else math.nan:.3f}，用于判断其是否在多数参数组合下可提升到合格。",
        }
    )
    rows.append(
        {
            "item": "paper_3_1_pass_ratio",
            "conclusion": f"3-1达到合格及以上比例为{float(ratio_31.iloc[0]) if not ratio_31.empty else math.nan:.3f}；若该比例较低，说明即使执行优化动作仍难以达到合格，需要更大幅度重构。",
        }
    )

    risk_overall = risk_stability.loc[risk_stability["paper_id"].astype(str) == "overall"]
    risk_prop = float(risk_overall["proportion_risk_decrease"].iloc[0]) if not risk_overall.empty else math.nan
    rows.append(
        {
            "item": "risk_decrease_stability",
            "conclusion": f"AI辅助写作风险提示指标在多数参数组合下下降，整体下降比例为{risk_prop:.3f}；该指标仅为风险提示，不构成作者身份或学术不端判断。",
        }
    )

    mean_disagreement_drop = float(disagreement_stability["mean_std_reduction"].mean()) if not disagreement_stability.empty else math.nan
    rows.append(
        {
            "item": "disagreement_convergence",
            "conclusion": f"多智能体评分分歧整体呈下降趋势，各论文平均标准差下降的均值为{mean_disagreement_drop:.3f}，说明推荐动作有助于收敛评审维度差异。",
        }
    )
    rows.append({"item": "method_boundary", "conclusion": DISCLAIMER})
    return pd.DataFrame(rows)


def draw_score_gain_sensitivity(results: pd.DataFrame, output_path: Path) -> None:
    """Draw mean score gain by score_gain_factor."""
    _set_chinese_font()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = results.groupby("score_gain_factor", as_index=False).agg(mean_score_gain=("score_gain", "mean"))
    fig, ax = plt.subplots(figsize=(7.2, 4.5), dpi=160)
    ax.plot(data["score_gain_factor"], data["mean_score_gain"], marker="o", linewidth=2, color="#4C78A8")
    ax.set_title("收益映射系数扰动下的平均提升分数", fontsize=12, pad=12)
    ax.set_xlabel("score_gain_factor")
    ax.set_ylabel("平均预测提升分数")
    ax.grid(linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def draw_budget_sensitivity(results: pd.DataFrame, output_path: Path) -> None:
    """Draw mean predicted score and action count by budget."""
    _set_chinese_font()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = results.groupby("budget", as_index=False).agg(
        mean_predicted_score=("predicted_score_after_revision", "mean"),
        avg_action_count=("selected_action_count", "mean"),
    )
    fig, ax1 = plt.subplots(figsize=(7.6, 4.8), dpi=160)
    ax1.bar(data["budget"] - 0.25, data["mean_predicted_score"], width=0.5, color="#4C78A8", label="平均预测得分")
    ax1.set_xlabel("修改预算")
    ax1.set_ylabel("平均预测得分")
    ax1.grid(axis="y", linestyle="--", alpha=0.35)
    ax2 = ax1.twinx()
    ax2.plot(data["budget"], data["avg_action_count"], marker="o", color="#F58518", label="平均动作数量")
    ax2.set_ylabel("平均动作数量")
    ax1.set_title("预算扰动下的平均预测得分和动作数量", fontsize=12, pad=12)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def draw_action_frequency(action_stability: pd.DataFrame, output_path: Path) -> None:
    """Draw stable action frequency."""
    _set_chinese_font()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = action_stability.sort_values(["selected_count", "action_id"], ascending=[False, True]).head(10)
    fig, ax = plt.subplots(figsize=(8.2, 4.8), dpi=160)
    ax.bar(data["action_id"].astype(str), data["selected_count"], color="#54A24B")
    ax.set_title("参数扰动下修改动作稳定出现频次", fontsize=12, pad=12)
    ax.set_xlabel("动作编号")
    ax.set_ylabel("选中次数")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    for idx, row in data.iterrows():
        ax.text(str(row["action_id"]), row["selected_count"] + 1, f"{int(row['selected_count'])}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def draw_disagreement_sensitivity(results: pd.DataFrame, output_path: Path) -> None:
    """Draw mean disagreement std reduction by multiplier."""
    _set_chinese_font()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = results.groupby("disagreement_reduction_multiplier", as_index=False).agg(mean_std_reduction=("std_reduction", "mean"))
    fig, ax = plt.subplots(figsize=(7.2, 4.5), dpi=160)
    ax.plot(data["disagreement_reduction_multiplier"], data["mean_std_reduction"], marker="o", color="#E45756", linewidth=2)
    ax.set_title("分歧收敛倍率扰动下的平均标准差下降", fontsize=12, pad=12)
    ax.set_xlabel("disagreement_reduction_multiplier")
    ax.set_ylabel("平均分歧标准差下降")
    ax.grid(linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def write_outputs(
    paths: dict[str, Path],
    parameter_grid: pd.DataFrame,
    results: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
) -> None:
    """Write Step 30 output tables."""
    paths["tables_dir"].mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(paths["analysis"], engine="openpyxl") as writer:
        tables["prediction_robustness_overview"].to_excel(writer, index=False, sheet_name="prediction_overview")
        tables["paper_level_stability"].to_excel(writer, index=False, sheet_name="paper_stability")
        tables["key_conclusions"].to_excel(writer, index=False, sheet_name="key_conclusions")
    parameter_grid.to_excel(paths["parameter_grid"], index=False)
    results.to_excel(paths["prediction_results"], index=False)
    with pd.ExcelWriter(paths["action_stability"], engine="openpyxl") as writer:
        tables["action_stability"].to_excel(writer, index=False, sheet_name="action_stability")
        tables["paper_action_stability"].to_excel(writer, index=False, sheet_name="paper_action_stability")
    with pd.ExcelWriter(paths["summary"], engine="openpyxl") as writer:
        parameter_grid.to_excel(writer, index=False, sheet_name="parameter_grid")
        tables["prediction_robustness_overview"].to_excel(writer, index=False, sheet_name="prediction_robustness")
        tables["paper_level_stability"].to_excel(writer, index=False, sheet_name="paper_level_stability")
        tables["action_stability"].to_excel(writer, index=False, sheet_name="action_stability")
        tables["risk_reduction_stability"].to_excel(writer, index=False, sheet_name="risk_reduction")
        tables["disagreement_stability"].to_excel(writer, index=False, sheet_name="disagreement_stability")
        tables["key_conclusions"].to_excel(writer, index=False, sheet_name="key_conclusions")
        tables["interpretation_text"].to_excel(writer, index=False, sheet_name="interpretation_text")


def run_third_question_robustness(config_path: Path | None = None) -> dict[str, Any]:
    """Run Step 30 robustness analysis."""
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
        "analysis": tables_dir / "robustness_analysis_step30.xlsx",
        "parameter_grid": tables_dir / "robustness_parameter_grid.xlsx",
        "prediction_results": tables_dir / "robustness_prediction_results.xlsx",
        "action_stability": tables_dir / "robustness_action_stability.xlsx",
        "summary": tables_dir / "robustness_summary.xlsx",
        "score_gain_chart": charts_dir / "robustness_score_gain_sensitivity.png",
        "budget_chart": charts_dir / "robustness_budget_sensitivity.png",
        "action_frequency_chart": charts_dir / "robustness_action_frequency.png",
        "disagreement_chart": charts_dir / "robustness_disagreement_sensitivity.png",
        "log": logs_dir / "robustness_analysis_step30.log",
    }

    logger = setup_robustness_logger(paths["log"])
    logger.info("Step 30 robustness analysis started")
    parameter_grid = build_parameter_grid()
    logger.info("Parameter grid size: %s", len(parameter_grid))
    logger.info("Score gain factors: %s", SCORE_GAIN_FACTORS)
    logger.info("Budgets: %s", BUDGETS)
    logger.info("Risk multipliers: %s", RISK_MULTIPLIERS)
    logger.info("Disagreement multipliers: %s", DISAGREEMENT_MULTIPLIERS)

    data, used_sources = load_step30_inputs(tables_dir, logger)
    base = _current_base_table(data)
    action_details = data["revision_details"]
    results = run_robustness_simulation(base, action_details, parameter_grid)
    tables = calculate_stability_tables(results)

    write_outputs(paths, parameter_grid, results, tables)
    draw_score_gain_sensitivity(results, paths["score_gain_chart"])
    draw_budget_sensitivity(results, paths["budget_chart"])
    draw_action_frequency(tables["action_stability"], paths["action_frequency_chart"])
    draw_disagreement_sensitivity(results, paths["disagreement_chart"])

    scenario_mean_gain = results.groupby("scenario_id")["score_gain"].mean()
    scenario_mean_risk = results.groupby("scenario_id")["risk_reduction_total"].mean()
    scenario_mean_std = results.groupby("scenario_id")["std_reduction"].mean()
    logger.info("Average score gain interval: %.6f ~ %.6f", scenario_mean_gain.min(), scenario_mean_gain.max())
    logger.info("Positive gain proportion: %.6f", float((results["score_gain"] > 0).mean()))
    logger.info("Average risk reduction interval: %.6f ~ %.6f", scenario_mean_risk.min(), scenario_mean_risk.max())
    logger.info("Average std reduction interval: %.6f ~ %.6f", scenario_mean_std.min(), scenario_mean_std.max())
    logger.info("Step 30 outputs written")

    return {
        "paths": paths,
        "used_sources": used_sources,
        "parameter_grid": parameter_grid,
        "results": results,
        "tables": tables,
        "parameter_combo_count": len(parameter_grid),
        "score_gain_interval": (float(scenario_mean_gain.min()), float(scenario_mean_gain.max())),
        "positive_gain_proportion": float((results["score_gain"] > 0).mean()),
        "risk_reduction_interval": (float(scenario_mean_risk.min()), float(scenario_mean_risk.max())),
        "std_reduction_interval": (float(scenario_mean_std.min()), float(scenario_mean_std.max())),
    }


__all__ = ["run_third_question_robustness", "build_parameter_grid", "run_robustness_simulation"]

