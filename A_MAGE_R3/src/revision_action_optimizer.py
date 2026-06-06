"""Problem 3 Step 28: revision action optimization.

This module transforms Step 26 AI-assisted writing risk evidence and Step 27
multi-agent reviewer disagreement into auditable, budget-constrained revision
plans. The output is a planning aid; it does not modify prior scores or claim
that revisions have actually been performed.
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


LOGGER_NAME = "A_MAGE_R3.problem3.revision_action_optimizer"

DIMENSIONS = [
    "structure",
    "logic",
    "modeling",
    "result_validation",
    "application_value",
]

DIMENSION_TO_AGENT = {
    "structure": "structure_reviewer",
    "logic": "logic_reviewer",
    "modeling": "modeling_reviewer",
    "result_validation": "result_validation_reviewer",
    "application_value": "application_value_reviewer",
}

AGENT_TO_DIMENSION = {agent: dimension for dimension, agent in DIMENSION_TO_AGENT.items()}

DIMENSION_LABELS = {
    "structure": "结构规范",
    "logic": "逻辑一致",
    "modeling": "数学建模",
    "result_validation": "结果验证",
    "application_value": "写作应用",
}

AGENT_LABELS = {
    "structure_reviewer": "结构规范智能体",
    "logic_reviewer": "逻辑一致智能体",
    "modeling_reviewer": "数学建模智能体",
    "result_validation_reviewer": "结果验证智能体",
    "application_value_reviewer": "写作应用智能体",
}

RISK_LABELS = {
    "data_traceability": "数据不可追溯风险",
    "method_jump": "方法到结果跳跃风险",
}


def setup_revision_logger(log_path: Path) -> logging.Logger:
    """Configure Step 28 logger."""
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
    """Convert a scalar value to float with a fallback."""
    number = pd.to_numeric(value, errors="coerce")
    return default if pd.isna(number) else float(number)


def build_revision_action_library() -> pd.DataFrame:
    """Construct the fixed, interpretable revision action library."""
    rows = [
        {
            "action_id": "A1",
            "action_name": "补充摘要中的研究目标、方法和结论",
            "target_dimensions": "structure,application_value",
            "expected_gain_structure": 5.0,
            "expected_gain_logic": 1.0,
            "expected_gain_modeling": 0.0,
            "expected_gain_result_validation": 1.0,
            "expected_gain_application_value": 3.0,
            "risk_reduction_data_traceability": 0.05,
            "risk_reduction_method_jump": 0.05,
            "cost": 3,
            "difficulty": "低",
            "description": "补全摘要中的目标、方法、结果和结论闭环，提升首部信息完整性。",
        },
        {
            "action_id": "A2",
            "action_name": "补充问题分析与建模思路",
            "target_dimensions": "logic,modeling",
            "expected_gain_structure": 1.0,
            "expected_gain_logic": 5.0,
            "expected_gain_modeling": 3.0,
            "expected_gain_result_validation": 0.0,
            "expected_gain_application_value": 0.5,
            "risk_reduction_data_traceability": 0.05,
            "risk_reduction_method_jump": 0.10,
            "cost": 4,
            "difficulty": "中",
            "description": "按小问明确任务拆解、变量关系和后续模型路线，减少逻辑跳跃。",
        },
        {
            "action_id": "A3",
            "action_name": "补充模型假设及合理性说明",
            "target_dimensions": "logic,modeling",
            "expected_gain_structure": 1.0,
            "expected_gain_logic": 3.5,
            "expected_gain_modeling": 4.5,
            "expected_gain_result_validation": 0.0,
            "expected_gain_application_value": 0.5,
            "risk_reduction_data_traceability": 0.05,
            "risk_reduction_method_jump": 0.05,
            "cost": 3,
            "difficulty": "中",
            "description": "补充假设、适用边界和合理性论证，使模型前提更清楚。",
        },
        {
            "action_id": "A4",
            "action_name": "完善符号说明表",
            "target_dimensions": "structure,modeling",
            "expected_gain_structure": 3.5,
            "expected_gain_logic": 0.5,
            "expected_gain_modeling": 3.5,
            "expected_gain_result_validation": 0.0,
            "expected_gain_application_value": 0.0,
            "risk_reduction_data_traceability": 0.00,
            "risk_reduction_method_jump": 0.05,
            "cost": 2,
            "difficulty": "低",
            "description": "统一变量、参数、单位和符号含义，提升公式可读性。",
        },
        {
            "action_id": "A5",
            "action_name": "增强模型推导过程",
            "target_dimensions": "logic,modeling",
            "expected_gain_structure": 0.0,
            "expected_gain_logic": 4.5,
            "expected_gain_modeling": 6.0,
            "expected_gain_result_validation": 1.5,
            "expected_gain_application_value": 0.0,
            "risk_reduction_data_traceability": 0.05,
            "risk_reduction_method_jump": 0.25,
            "cost": 5,
            "difficulty": "高",
            "description": "补充从目标函数、约束条件到求解过程的推导，强化方法到结果链条。",
        },
        {
            "action_id": "A6",
            "action_name": "补充算法流程或伪代码",
            "target_dimensions": "modeling,result_validation",
            "expected_gain_structure": 0.5,
            "expected_gain_logic": 1.0,
            "expected_gain_modeling": 5.0,
            "expected_gain_result_validation": 2.0,
            "expected_gain_application_value": 0.0,
            "risk_reduction_data_traceability": 0.00,
            "risk_reduction_method_jump": 0.18,
            "cost": 4,
            "difficulty": "中",
            "description": "给出求解流程、输入输出和关键参数，增强算法复现性。",
        },
        {
            "action_id": "A7",
            "action_name": "补充结果表格和可视化",
            "target_dimensions": "result_validation",
            "expected_gain_structure": 1.0,
            "expected_gain_logic": 1.0,
            "expected_gain_modeling": 0.5,
            "expected_gain_result_validation": 6.0,
            "expected_gain_application_value": 1.0,
            "risk_reduction_data_traceability": 0.05,
            "risk_reduction_method_jump": 0.18,
            "cost": 4,
            "difficulty": "中",
            "description": "补齐关键数值结果、图表和对比展示，使模型输出可检验。",
        },
        {
            "action_id": "A8",
            "action_name": "增加敏感性分析",
            "target_dimensions": "result_validation",
            "expected_gain_structure": 0.0,
            "expected_gain_logic": 1.0,
            "expected_gain_modeling": 1.0,
            "expected_gain_result_validation": 6.0,
            "expected_gain_application_value": 0.5,
            "risk_reduction_data_traceability": 0.00,
            "risk_reduction_method_jump": 0.10,
            "cost": 4,
            "difficulty": "中",
            "description": "对关键参数扰动进行稳健性检验，增强结果可信度。",
        },
        {
            "action_id": "A9",
            "action_name": "增加误差分析",
            "target_dimensions": "result_validation",
            "expected_gain_structure": 0.0,
            "expected_gain_logic": 1.0,
            "expected_gain_modeling": 0.5,
            "expected_gain_result_validation": 6.5,
            "expected_gain_application_value": 0.5,
            "risk_reduction_data_traceability": 0.05,
            "risk_reduction_method_jump": 0.10,
            "cost": 4,
            "difficulty": "中",
            "description": "补充误差来源、误差量化或局限性说明，避免结果无验证支撑。",
        },
        {
            "action_id": "A10",
            "action_name": "增加模型优缺点与推广",
            "target_dimensions": "result_validation,application_value",
            "expected_gain_structure": 0.5,
            "expected_gain_logic": 1.0,
            "expected_gain_modeling": 1.0,
            "expected_gain_result_validation": 2.5,
            "expected_gain_application_value": 5.5,
            "risk_reduction_data_traceability": 0.00,
            "risk_reduction_method_jump": 0.05,
            "cost": 3,
            "difficulty": "低",
            "description": "说明模型适用条件、优缺点、推广场景和改进方向。",
        },
        {
            "action_id": "A11",
            "action_name": "补充数据来源和数据处理说明",
            "target_dimensions": "logic,result_validation",
            "expected_gain_structure": 1.0,
            "expected_gain_logic": 3.0,
            "expected_gain_modeling": 1.5,
            "expected_gain_result_validation": 4.0,
            "expected_gain_application_value": 1.0,
            "risk_reduction_data_traceability": 0.30,
            "risk_reduction_method_jump": 0.05,
            "cost": 3,
            "difficulty": "中",
            "description": "补充数据来源、口径、预处理、缺失或异常处理，提升可追溯性。",
        },
        {
            "action_id": "A12",
            "action_name": "增强方法到结果的解释链条",
            "target_dimensions": "logic,modeling,result_validation",
            "expected_gain_structure": 0.0,
            "expected_gain_logic": 5.5,
            "expected_gain_modeling": 3.0,
            "expected_gain_result_validation": 4.0,
            "expected_gain_application_value": 0.5,
            "risk_reduction_data_traceability": 0.05,
            "risk_reduction_method_jump": 0.30,
            "cost": 4,
            "difficulty": "中",
            "description": "补足模型输出、结果解释和结论之间的证据链，降低方法跳跃。",
        },
        {
            "action_id": "A13",
            "action_name": "优化论文结构编号和格式规范",
            "target_dimensions": "structure",
            "expected_gain_structure": 5.0,
            "expected_gain_logic": 0.5,
            "expected_gain_modeling": 0.0,
            "expected_gain_result_validation": 0.5,
            "expected_gain_application_value": 1.0,
            "risk_reduction_data_traceability": 0.00,
            "risk_reduction_method_jump": 0.00,
            "cost": 2,
            "difficulty": "低",
            "description": "统一标题层级、图表编号、格式规范和章节顺序。",
        },
        {
            "action_id": "A14",
            "action_name": "优化结论与实际应用价值表达",
            "target_dimensions": "logic,application_value",
            "expected_gain_structure": 0.5,
            "expected_gain_logic": 3.0,
            "expected_gain_modeling": 0.0,
            "expected_gain_result_validation": 1.0,
            "expected_gain_application_value": 6.0,
            "risk_reduction_data_traceability": 0.00,
            "risk_reduction_method_jump": 0.05,
            "cost": 3,
            "difficulty": "低",
            "description": "让结论逐条回扣题目目标，并补充可落地的应用价值说明。",
        },
    ]
    return pd.DataFrame(rows)


def _read_excel_if_exists(path: Path, *, sheet_name: str | int = 0) -> pd.DataFrame:
    """Read an Excel file if it exists, otherwise return an empty frame."""
    return pd.read_excel(path, sheet_name=sheet_name) if Path(path).exists() else pd.DataFrame()


def load_step28_inputs(tables_dir: Path, logger: logging.Logger) -> tuple[dict[str, pd.DataFrame], list[Path]]:
    """Load all available Step 28 input tables."""
    sources = {
        "subjectivity": tables_dir / "multi_agent_subjectivity_analysis.xlsx",
        "agent_scores": tables_dir / "multi_agent_subjectivity_analysis.xlsx",
        "disagreement_details": tables_dir / "multi_agent_disagreement_details.xlsx",
        "ai_fusion": tables_dir / "ai_risk_ds_fusion.xlsx",
        "ai_evidence": tables_dir / "ai_risk_evidence.xlsx",
        "features_normalized": tables_dir / "appendix3_features_normalized.xlsx",
        "current_evaluation": tables_dir / "appendix3_current_evaluation.xlsx",
        "problem1_base": tables_dir / "appendix3_problem1_base_scores.xlsx",
        "agent_summary": tables_dir / "agent_subjectivity_summary.xlsx",
    }
    data: dict[str, pd.DataFrame] = {}
    used: list[Path] = []
    for key, path in sources.items():
        if not path.exists():
            logger.warning("Optional Step 28 input missing: %s", path)
            data[key] = pd.DataFrame()
            continue
        try:
            sheet = "agent_score_matrix" if key == "agent_scores" else 0
            data[key] = pd.read_excel(path, sheet_name=sheet)
            used.append(path)
            logger.info("Loaded %s from %s", key, path)
        except Exception as exc:  # pragma: no cover - defensive I/O guard
            logger.exception("Failed to load %s from %s: %s", key, path, exc)
            data[key] = pd.DataFrame()
    unique_used = list(dict.fromkeys(used))
    return data, unique_used


def _merge_inputs(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Merge subjectivity, AI risk, and optional current score inputs."""
    subjectivity = data.get("subjectivity", pd.DataFrame()).copy()
    if subjectivity.empty:
        raise FileNotFoundError("multi_agent_subjectivity_analysis.xlsx is required for Step 28.")
    agent_scores = data.get("agent_scores", pd.DataFrame()).copy()
    if not agent_scores.empty:
        keep = ["paper_id"] + [agent for agent in AGENT_TO_DIMENSION if agent in agent_scores.columns]
        subjectivity = subjectivity.merge(agent_scores[keep], on="paper_id", how="left", suffixes=("", "_agent_score"))

    ai_evidence = data.get("ai_evidence", pd.DataFrame()).copy()
    evidence_cols = [
        col
        for col in [
            "paper_id",
            "e1_template_expression",
            "e2_unsupported_conclusion",
            "e3_data_untraceable",
            "e4_method_result_jump",
            "top_risk_evidence",
        ]
        if col in ai_evidence.columns and col not in subjectivity.columns
    ]
    if evidence_cols:
        subjectivity = subjectivity.merge(ai_evidence[evidence_cols], on="paper_id", how="left")

    current_eval = data.get("current_evaluation", pd.DataFrame()).copy()
    if not current_eval.empty:
        keep = [col for col in ["paper_id", "Q_cur_baseline", "current_grade", "main_low_features", "key_feature_weaknesses"] if col in current_eval.columns]
        subjectivity = subjectivity.merge(current_eval[keep], on="paper_id", how="left")

    return subjectivity.sort_values("paper_id", key=lambda values: values.map(_natural_sort_key)).reset_index(drop=True)


def _dimension_scores(row: pd.Series) -> dict[str, float]:
    """Return the five agent dimension scores for a paper."""
    scores: dict[str, float] = {}
    for agent, dimension in AGENT_TO_DIMENSION.items():
        scores[dimension] = _safe_numeric(row.get(agent), default=np.nan)
    if any(pd.isna(value) for value in scores.values()):
        mean = _safe_numeric(row.get("agent_score_mean"), default=50.0)
        for dimension, value in list(scores.items()):
            if pd.isna(value):
                scores[dimension] = mean
    return scores


def _dimension_weights(row: pd.Series) -> dict[str, float]:
    """Calculate paper-specific boost weights for weak dimensions."""
    scores = _dimension_scores(row)
    mean_score = _safe_numeric(row.get("agent_score_mean"), default=float(np.mean(list(scores.values()))))
    weakest_agent = str(row.get("min_score_agent", ""))
    weakest_dimension = AGENT_TO_DIMENSION.get(weakest_agent, "")
    high_disagreement = str(row.get("disagreement_level", "")) == "高分歧"

    weights: dict[str, float] = {}
    for dimension, score in scores.items():
        gap = max(0.0, mean_score - score)
        weight = 1.0 + min(gap / 25.0, 1.0)
        if dimension == weakest_dimension:
            weight += 0.45
        if high_disagreement and gap > 0:
            weight += 0.20
        weights[dimension] = weight
    return weights


def _calculate_action_gain(
    paper_row: pd.Series,
    action_row: pd.Series,
) -> tuple[float, dict[str, float], str]:
    """Calculate paper-specific action gain and component notes."""
    weights = _dimension_weights(paper_row)
    dimension_components: dict[str, float] = {}
    quality_gain = 0.0
    for dimension in DIMENSIONS:
        base = _safe_numeric(action_row.get(f"expected_gain_{dimension}"), default=0.0)
        adjusted = base * weights[dimension]
        dimension_components[dimension] = adjusted
        quality_gain += adjusted

    e3 = _safe_numeric(paper_row.get("e3_data_untraceable"), default=0.0)
    e4 = _safe_numeric(paper_row.get("e4_method_result_jump"), default=0.0)
    data_risk_gain = 10.0 * _safe_numeric(action_row.get("risk_reduction_data_traceability"), default=0.0) * e3
    method_jump_gain = 10.0 * _safe_numeric(action_row.get("risk_reduction_method_jump"), default=0.0) * e4

    action_id = str(action_row["action_id"])
    explicit_bonus = 0.0
    weakest_agent = str(paper_row.get("min_score_agent", ""))
    if weakest_agent == "structure_reviewer" and action_id in {"A13", "A1", "A4"}:
        explicit_bonus += 1.5
    if weakest_agent == "logic_reviewer" and action_id in {"A2", "A5", "A12"}:
        explicit_bonus += 1.5
    if weakest_agent == "modeling_reviewer" and action_id in {"A3", "A5", "A6"}:
        explicit_bonus += 1.5
    if weakest_agent == "result_validation_reviewer" and action_id in {"A7", "A8", "A9", "A11"}:
        explicit_bonus += 1.5
    if weakest_agent == "application_value_reviewer" and action_id in {"A10", "A14", "A1"}:
        explicit_bonus += 1.5
    if e3 >= 0.60 and action_id == "A11":
        explicit_bonus += 1.5
    if e4 >= 0.40 and action_id in {"A12", "A5", "A7"}:
        explicit_bonus += 1.2

    total_gain = quality_gain + data_risk_gain + method_jump_gain + explicit_bonus
    note = (
        f"quality={quality_gain:.2f}; data_risk={data_risk_gain:.2f}; "
        f"method_jump={method_jump_gain:.2f}; explicit_bonus={explicit_bonus:.2f}"
    )
    return float(total_gain), dimension_components, note


def _greedy_select(action_details: pd.DataFrame, budget: int) -> list[str]:
    """Select actions by gain/cost ratio under budget."""
    chosen: list[str] = []
    spent = 0
    ordered = action_details.sort_values(["gain_cost_ratio", "adjusted_expected_gain"], ascending=False)
    for _, row in ordered.iterrows():
        cost = int(row["cost"])
        if spent + cost <= budget:
            chosen.append(str(row["action_id"]))
            spent += cost
    return chosen


def _knapsack_select(action_details: pd.DataFrame, budget: int) -> list[str]:
    """0-1 knapsack selection maximizing adjusted expected gain."""
    items = action_details.reset_index(drop=True)
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
            selected.append(str(items.loc[i - 1, "action_id"]))
            w -= int(items.loc[i - 1, "cost"])
    return list(reversed(selected))


def _selection_summary(details: pd.DataFrame, selected: list[str]) -> tuple[float, int]:
    """Return total gain and total cost for selected action IDs."""
    chosen = details.loc[details["action_id"].isin(selected)]
    return float(chosen["adjusted_expected_gain"].sum()), int(chosen["cost"].sum())


def _risk_reduction_summary(details: pd.DataFrame, selected: list[str]) -> str:
    """Summarize expected risk reduction from selected actions."""
    chosen = details.loc[details["action_id"].isin(selected)]
    data_reduction = float(chosen["risk_reduction_data_traceability"].sum())
    jump_reduction = float(chosen["risk_reduction_method_jump"].sum())
    parts: list[str] = []
    if data_reduction > 0:
        parts.append(f"数据追溯风险约束强度={data_reduction:.2f}")
    if jump_reduction > 0:
        parts.append(f"方法跳跃风险约束强度={jump_reduction:.2f}")
    return "；".join(parts) if parts else "以质量维度修复为主，风险约束收益较低"


def _priority_level(row: pd.Series, expected_gain: float) -> str:
    """Classify revision priority."""
    std = _safe_numeric(row.get("agent_score_std"), default=0.0)
    q_cur = _safe_numeric(row.get("Q_cur_baseline"), default=50.0)
    if expected_gain >= 45 or q_cur < 40 or std >= 16:
        return "高优先级"
    if expected_gain >= 30 or std >= 10:
        return "中优先级"
    return "低优先级"


def _recommendation_text(row: pd.Series, library: pd.DataFrame, selected: list[str]) -> str:
    """Generate a concise Chinese revision recommendation."""
    weakest_agent = str(row.get("min_score_agent", ""))
    strongest_agent = str(row.get("max_score_agent", ""))
    weak_label = AGENT_LABELS.get(weakest_agent, weakest_agent)
    strong_label = AGENT_LABELS.get(strongest_agent, strongest_agent)
    actions = library.loc[library["action_id"].isin(selected), "action_name"].tolist()
    action_text = "、".join(actions)
    risk_source = str(row.get("main_risk_source", "数据不可追溯风险、方法到结果跳跃风险"))
    return (
        f"该论文多智能体评分分歧为{row.get('disagreement_level', '未知')}，"
        f"相对优势维度为{strong_label}，主要弱项为{weak_label}。"
        f"建议优先执行：{action_text}。这些动作预计可修复弱评分维度，"
        f"并针对{risk_source}进行保守约束；AI辅助写作风险不直接等于质量扣分，"
        "本方案仅作为可执行修改路径。"
    )


def optimize_revision_actions(
    merged_inputs: pd.DataFrame,
    action_library: pd.DataFrame,
    budget: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Optimize revision action sets for each paper."""
    detail_rows: list[dict[str, Any]] = []
    plan_rows: list[dict[str, Any]] = []
    radar_rows: list[dict[str, Any]] = []

    for _, paper_row in merged_inputs.iterrows():
        paper_id = str(paper_row["paper_id"])
        paper_details: list[dict[str, Any]] = []
        for _, action_row in action_library.iterrows():
            gain, dimension_components, component_note = _calculate_action_gain(paper_row, action_row)
            cost = int(action_row["cost"])
            record = {
                "paper_id": paper_id,
                "filename": paper_row.get("filename", f"{paper_id}.txt"),
                "action_id": action_row["action_id"],
                "action_name": action_row["action_name"],
                "target_dimensions": action_row["target_dimensions"],
                "adjusted_expected_gain": gain,
                "cost": cost,
                "gain_cost_ratio": gain / cost if cost > 0 else math.inf,
                "difficulty": action_row["difficulty"],
                "risk_reduction_data_traceability": action_row["risk_reduction_data_traceability"],
                "risk_reduction_method_jump": action_row["risk_reduction_method_jump"],
                "gain_component_note": component_note,
            }
            for dimension, value in dimension_components.items():
                record[f"adjusted_gain_{dimension}"] = value
            paper_details.append(record)

        details_df = pd.DataFrame(paper_details)
        greedy = _greedy_select(details_df, budget)
        knapsack = _knapsack_select(details_df, budget)
        gain_greedy, cost_greedy = _selection_summary(details_df, greedy)
        gain_knapsack, cost_knapsack = _selection_summary(details_df, knapsack)
        details_df["selected_knapsack"] = details_df["action_id"].isin(knapsack)
        details_df["selected_greedy"] = details_df["action_id"].isin(greedy)
        detail_rows.extend(details_df.to_dict(orient="records"))

        scores_before = _dimension_scores(paper_row)
        selected_details = details_df.loc[details_df["selected_knapsack"]]
        for dimension in DIMENSIONS:
            estimated_gain = float(selected_details[f"adjusted_gain_{dimension}"].sum())
            before = scores_before[dimension]
            after = min(100.0, before + estimated_gain)
            radar_rows.append(
                {
                    "paper_id": paper_id,
                    "dimension": dimension,
                    "dimension_label": DIMENSION_LABELS[dimension],
                    "before_score": before,
                    "estimated_dimension_gain": estimated_gain,
                    "after_score_estimated": after,
                }
            )

        plan_rows.append(
            {
                "paper_id": paper_id,
                "filename": paper_row.get("filename", f"{paper_id}.txt"),
                "disagreement_level": paper_row.get("disagreement_level", ""),
                "weakest_agent": paper_row.get("min_score_agent", ""),
                "strongest_agent": paper_row.get("max_score_agent", ""),
                "agent_score_std": _safe_numeric(paper_row.get("agent_score_std"), default=0.0),
                "R_AI": _safe_numeric(paper_row.get("R_AI"), default=0.0),
                "risk_level": paper_row.get("risk_level", ""),
                "main_risk_sources": paper_row.get("main_risk_source", ""),
                "selected_actions_knapsack": ",".join(knapsack),
                "selected_actions_greedy": ",".join(greedy),
                "expected_total_gain_knapsack": gain_knapsack,
                "expected_total_gain_greedy": gain_greedy,
                "total_cost_knapsack": cost_knapsack,
                "total_cost_greedy": cost_greedy,
                "risk_reduction_summary": _risk_reduction_summary(details_df, knapsack),
                "revision_priority_level": _priority_level(paper_row, gain_knapsack),
                "final_recommendation_text": _recommendation_text(paper_row, action_library, knapsack),
            }
        )

    return pd.DataFrame(plan_rows), pd.DataFrame(detail_rows), pd.DataFrame(radar_rows)


def build_summary_tables(
    action_library: pd.DataFrame,
    plan: pd.DataFrame,
    details: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Build summary sheets for Step 28."""
    selected = details.loc[details["selected_knapsack"]].copy()
    frequency = (
        selected.groupby(["action_id", "action_name"], as_index=False)
        .agg(
            recommended_count=("paper_id", "nunique"),
            avg_adjusted_expected_gain=("adjusted_expected_gain", "mean"),
            avg_gain_cost_ratio=("gain_cost_ratio", "mean"),
        )
        .sort_values(["recommended_count", "avg_adjusted_expected_gain"], ascending=False)
    )
    gain_summary = plan[
        [
            "paper_id",
            "expected_total_gain_knapsack",
            "expected_total_gain_greedy",
            "total_cost_knapsack",
            "total_cost_greedy",
            "revision_priority_level",
        ]
    ].copy()
    risk_summary = plan[["paper_id", "R_AI", "risk_level", "main_risk_sources", "risk_reduction_summary"]].copy()
    optimization_compare = plan[
        [
            "paper_id",
            "selected_actions_knapsack",
            "selected_actions_greedy",
            "expected_total_gain_knapsack",
            "expected_total_gain_greedy",
            "total_cost_knapsack",
            "total_cost_greedy",
        ]
    ].copy()
    optimization_compare["gain_difference_knapsack_minus_greedy"] = (
        optimization_compare["expected_total_gain_knapsack"]
        - optimization_compare["expected_total_gain_greedy"]
    )
    interpretation = plan[["paper_id", "final_recommendation_text"]].copy()
    interpretation.loc[len(interpretation)] = {
        "paper_id": "method_boundary",
        "final_recommendation_text": "修改动作收益为规则化预估，用于有限预算下的优先级排序；后续 Step29 才进行优化后质量预测。",
    }
    return {
        "action_library": action_library,
        "paper_revision_plan": plan,
        "action_frequency": frequency,
        "expected_gain_summary": gain_summary,
        "risk_reduction_summary": risk_summary,
        "optimization_compare": optimization_compare,
        "interpretation_text": interpretation,
    }


def draw_revision_priority_bar(plan: pd.DataFrame, output_path: Path) -> None:
    """Draw expected total gain and cost by paper."""
    _set_chinese_font()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = plan.sort_values("paper_id", key=lambda values: values.map(_natural_sort_key))
    x = np.arange(len(ordered))
    fig, ax1 = plt.subplots(figsize=(8, 4.8), dpi=160)
    bars = ax1.bar(x - 0.18, ordered["expected_total_gain_knapsack"], width=0.36, color="#4C78A8", label="预计收益")
    ax1.set_ylabel("预计提升收益")
    ax1.set_xticks(x)
    ax1.set_xticklabels(ordered["paper_id"].astype(str))
    ax1.grid(axis="y", linestyle="--", alpha=0.35)
    ax2 = ax1.twinx()
    ax2.bar(x + 0.18, ordered["total_cost_knapsack"], width=0.36, color="#F58518", label="修改成本", alpha=0.85)
    ax2.set_ylabel("修改成本")
    ax1.set_title("附件3论文修改优先级：收益与成本", fontsize=12, pad=12)
    for bar in bars:
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f"{bar.get_height():.1f}", ha="center", fontsize=8)
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def draw_gain_cost_scatter(details: pd.DataFrame, output_path: Path) -> None:
    """Draw average action gain-cost scatter across papers."""
    _set_chinese_font()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = (
        details.groupby(["action_id", "action_name", "cost"], as_index=False)
        .agg(
            avg_gain=("adjusted_expected_gain", "mean"),
            selected_count=("selected_knapsack", "sum"),
        )
        .sort_values("action_id", key=lambda values: values.map(_natural_sort_key))
    )
    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=160)
    sizes = 70 + summary["selected_count"].astype(float) * 70
    ax.scatter(summary["cost"], summary["avg_gain"], s=sizes, c=summary["selected_count"], cmap="viridis", alpha=0.82)
    for _, row in summary.iterrows():
        ax.text(row["cost"] + 0.03, row["avg_gain"] + 0.15, row["action_id"], fontsize=8)
    ax.set_xlabel("动作成本")
    ax.set_ylabel("平均预计收益")
    ax.set_title("修改动作收益-成本散点图", fontsize=12, pad=12)
    ax.grid(linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def draw_paper_revision_plan_radar(radar_data: pd.DataFrame, output_path: Path) -> None:
    """Draw before/estimated-after radar charts for each paper."""
    _set_chinese_font()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    papers = sorted(radar_data["paper_id"].astype(str).unique(), key=_natural_sort_key)
    labels = [DIMENSION_LABELS[dimension] for dimension in DIMENSIONS]
    angles = np.linspace(0, 2 * np.pi, len(DIMENSIONS), endpoint=False).tolist()
    angles += angles[:1]
    fig, axes = plt.subplots(1, len(papers), figsize=(5 * len(papers), 5.2), subplot_kw={"polar": True}, dpi=160)
    if len(papers) == 1:
        axes = [axes]
    for ax, paper_id in zip(axes, papers):
        subset = radar_data.loc[radar_data["paper_id"].astype(str) == paper_id].set_index("dimension").loc[DIMENSIONS]
        before = subset["before_score"].astype(float).tolist()
        after = subset["after_score_estimated"].astype(float).tolist()
        before += before[:1]
        after += after[:1]
        ax.plot(angles, before, label="修改前", color="#4C78A8", linewidth=1.8)
        ax.fill(angles, before, color="#4C78A8", alpha=0.08)
        ax.plot(angles, after, label="预估修改后", color="#E45756", linewidth=1.8)
        ax.fill(angles, after, color="#E45756", alpha=0.08)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylim(0, 100)
        ax.set_title(str(paper_id), fontsize=11, pad=14)
        ax.grid(alpha=0.35)
    axes[0].legend(loc="upper left", bbox_to_anchor=(-0.18, 1.18), frameon=False)
    fig.suptitle("附件3论文修改前后维度预估雷达图", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def write_outputs(
    paths: dict[str, Path],
    action_library: pd.DataFrame,
    plan: pd.DataFrame,
    details: pd.DataFrame,
    radar_data: pd.DataFrame,
    summaries: dict[str, pd.DataFrame],
) -> None:
    """Write all Step 28 tables."""
    paths["tables_dir"].mkdir(parents=True, exist_ok=True)
    action_library.to_excel(paths["action_library"], index=False)
    plan.to_excel(paths["optimization"], index=False)
    with pd.ExcelWriter(paths["details"], engine="openpyxl") as writer:
        details.to_excel(writer, index=False, sheet_name="action_details")
        radar_data.to_excel(writer, index=False, sheet_name="dimension_before_after")
    with pd.ExcelWriter(paths["summary"], engine="openpyxl") as writer:
        for sheet_name, frame in summaries.items():
            frame.to_excel(writer, index=False, sheet_name=sheet_name[:31])


def run_revision_action_optimization(config_path: Path | None = None) -> dict[str, Any]:
    """Run Step 28 revision action optimization."""
    config = get_problem3_config(config_path)
    tables_dir = resolve_project_path(config.get("output_tables_dir", "output/problem3_tables"))
    charts_dir = resolve_project_path(config.get("output_charts_dir", "output/problem3_charts"))
    logs_dir = resolve_project_path(config.get("output_logs_dir", "output/problem3_logs"))
    tables_dir.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    budget = int(config.get("revision_budget_default", 10))

    paths = {
        "tables_dir": tables_dir,
        "charts_dir": charts_dir,
        "logs_dir": logs_dir,
        "action_library": tables_dir / "revision_action_library.xlsx",
        "optimization": tables_dir / "revision_action_optimization.xlsx",
        "details": tables_dir / "revision_action_details.xlsx",
        "summary": tables_dir / "revision_optimization_summary.xlsx",
        "priority_bar": charts_dir / "revision_priority_bar.png",
        "gain_cost_scatter": charts_dir / "revision_gain_cost_scatter.png",
        "plan_radar": charts_dir / "paper_revision_plan_radar.png",
        "log": logs_dir / "revision_action_optimization.log",
    }
    logger = setup_revision_logger(paths["log"])
    logger.info("Step 28 revision action optimization started")
    logger.info("Revision budget: %s", budget)

    data, used_sources = load_step28_inputs(tables_dir, logger)
    merged = _merge_inputs(data)
    action_library = build_revision_action_library()
    plan, details, radar_data = optimize_revision_actions(merged, action_library, budget=budget)
    summaries = build_summary_tables(action_library, plan, details)
    write_outputs(paths, action_library, plan, details, radar_data, summaries)

    draw_revision_priority_bar(plan, paths["priority_bar"])
    draw_gain_cost_scatter(details, paths["gain_cost_scatter"])
    draw_paper_revision_plan_radar(radar_data, paths["plan_radar"])

    logger.info("Used sources: %s", [str(path) for path in used_sources])
    logger.info("Paper count: %s", len(plan))
    logger.info("Top recommended action counts: %s", summaries["action_frequency"].head(5).to_dict(orient="records"))
    logger.info("Step 28 outputs written")

    return {
        "paths": paths,
        "used_sources": used_sources,
        "action_library": action_library,
        "plan": plan,
        "details": details,
        "summaries": summaries,
        "budget": budget,
    }


__all__ = [
    "run_revision_action_optimization",
    "build_revision_action_library",
    "optimize_revision_actions",
]
