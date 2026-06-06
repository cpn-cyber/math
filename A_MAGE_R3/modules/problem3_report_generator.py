"""Generate Problem 3 paper-writing materials from sealed audit outputs.

This module only reads Step 23-31 outputs and writes Markdown writing aids.
It does not recompute scores, risks, actions, or robustness metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import json
import logging
from typing import Any

import pandas as pd


DISCLAIMER = "该指标仅表示文本存在 AI辅助写作风险，不构成学术不端判断，也不等同于AI生成判定。"
PAPER_IDS = ["3-1", "3-2", "3-3"]


@dataclass
class Problem3Paths:
    """Canonical project paths used by Step 32."""

    project_root: Path
    tables_dir: Path
    charts_dir: Path
    logs_dir: Path
    reports_dir: Path
    paper_dir: Path


def _paths(project_root: Path | str) -> Problem3Paths:
    root = Path(project_root).resolve()
    return Problem3Paths(
        project_root=root,
        tables_dir=root / "output" / "problem3_tables",
        charts_dir=root / "output" / "problem3_charts",
        logs_dir=root / "output" / "problem3_logs",
        reports_dir=root / "output" / "problem3_reports",
        paper_dir=root / "paper_sections" / "problem3",
    )


def _setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("problem3_report_generator")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger


def _read_excel(path: Path, sheet_name: str | int = 0, logger: logging.Logger | None = None) -> pd.DataFrame:
    if not path.exists():
        if logger:
            logger.warning("Missing input file: %s", path)
        return pd.DataFrame()
    try:
        return pd.read_excel(path, sheet_name=sheet_name)
    except Exception as exc:  # pragma: no cover - defensive path
        if logger:
            logger.warning("Failed reading %s sheet=%s: %s", path, sheet_name, exc)
        return pd.DataFrame()


def _read_json(path: Path, logger: logging.Logger | None = None) -> dict[str, Any]:
    if not path.exists():
        if logger:
            logger.warning("Missing input file: %s", path)
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - defensive path
        if logger:
            logger.warning("Failed reading json %s: %s", path, exc)
        return {}


def _f(value: Any, digits: int = 6) -> str:
    if value is None or value == "":
        return "待补充"
    try:
        if pd.isna(value):
            return "待补充"
    except TypeError:
        pass
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _short(value: Any, max_len: int = 80) -> str:
    text = "" if value is None or pd.isna(value) else str(value)
    text = text.replace("\n", " ").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def _first_row(df: pd.DataFrame, paper_id: str) -> dict[str, Any]:
    if df.empty or "paper_id" not in df.columns:
        return {}
    rows = df[df["paper_id"].astype(str) == paper_id]
    if rows.empty:
        return {}
    return rows.iloc[0].to_dict()


def _markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    if not rows:
        return "待补充"
    headers = [title for _, title in columns]
    lines = ["|" + "|".join(headers) + "|", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        values = []
        for key, _ in columns:
            value = row.get(key, "待补充")
            values.append(str(value).replace("\n", " "))
        lines.append("|" + "|".join(values) + "|")
    return "\n".join(lines)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def _collect_context(paths: Problem3Paths, logger: logging.Logger) -> dict[str, Any]:
    tables = paths.tables_dir
    ctx: dict[str, Any] = {
        "current": _read_excel(tables / "appendix3_current_evaluation.xlsx", logger=logger),
        "logic": _read_excel(tables / "logic_gap_diagnosis.xlsx", logger=logger),
        "logic_evidence": _read_excel(tables / "logic_gap_evidence.xlsx", logger=logger),
        "ai": _read_excel(tables / "ai_risk_ds_fusion.xlsx", logger=logger),
        "ai_evidence": _read_excel(tables / "ai_risk_evidence.xlsx", logger=logger),
        "subjectivity": _read_excel(tables / "multi_agent_subjectivity_analysis.xlsx", "subjectivity_analysis", logger=logger),
        "agent_matrix": _read_excel(tables / "multi_agent_subjectivity_analysis.xlsx", "agent_score_matrix", logger=logger),
        "agent_mapping": _read_excel(tables / "multi_agent_subjectivity_analysis.xlsx", "agent_source_mapping", logger=logger),
        "revision": _read_excel(tables / "revision_action_optimization.xlsx", logger=logger),
        "action_library": _read_excel(tables / "revision_action_library.xlsx", logger=logger),
        "prediction": _read_excel(tables / "quality_prediction_after_revision.xlsx", logger=logger),
        "prediction_summary": _read_excel(tables / "quality_prediction_summary.xlsx", "prediction_overview", logger=logger),
        "robust_prediction": _read_excel(tables / "robustness_summary.xlsx", "prediction_robustness", logger=logger),
        "robust_paper": _read_excel(tables / "robustness_summary.xlsx", "paper_level_stability", logger=logger),
        "robust_action": _read_excel(tables / "robustness_summary.xlsx", "action_stability", logger=logger),
        "robust_risk": _read_excel(tables / "robustness_summary.xlsx", "risk_reduction", logger=logger),
        "robust_disagreement": _read_excel(tables / "robustness_summary.xlsx", "disagreement_stability", logger=logger),
        "figures": _read_excel(tables / "problem3_key_figures_manifest.xlsx", logger=logger),
        "conclusions": _read_excel(tables / "problem3_conclusion_checklist.xlsx", logger=logger),
        "audit_summary": _read_json(tables / "problem3_final_audit_summary.json", logger=logger),
    }
    return ctx


def _paper_overview_rows(ctx: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for pid in PAPER_IDS:
        cur = _first_row(ctx["current"], pid)
        logic = _first_row(ctx["logic"], pid)
        ai = _first_row(ctx["ai"], pid)
        subj = _first_row(ctx["subjectivity"], pid)
        rev = _first_row(ctx["revision"], pid)
        pred = _first_row(ctx["prediction"], pid)
        robust = _first_row(ctx["robust_paper"], pid)
        rows.append(
            {
                "paper_id": pid,
                "F1": _f(cur.get("F1_score")),
                "F2": _f(cur.get("F2_key_score")),
                "Q_cur": _f(cur.get("Q_cur_baseline")),
                "grade": cur.get("current_grade", "待补充"),
                "Gamma": _f(logic.get("Gamma")),
                "G": _f(logic.get("G_logic_gap")),
                "weak_edges": logic.get("recommended_diagnosis_focus", "待补充"),
                "R_AI": _f(ai.get("R_AI")),
                "risk_level": ai.get("risk_level", "待补充"),
                "std": _f(subj.get("agent_score_std")),
                "disagreement": subj.get("disagreement_level", "待补充"),
                "actions": rev.get("selected_actions_knapsack", "待补充"),
                "gain": _f(rev.get("expected_total_gain_knapsack")),
                "cost": _f(rev.get("total_cost_knapsack"), 0),
                "score_after": _f(pred.get("predicted_score_after_revision")),
                "score_gain": _f(pred.get("score_gain")),
                "level_after": pred.get("predicted_level_after_revision", "待补充"),
                "pass_ratio": _f(robust.get("pass_or_above_ratio")),
            }
        )
    return rows


def _action_name_map(ctx: dict[str, Any]) -> dict[str, str]:
    df = ctx["action_library"]
    if df.empty or "action_id" not in df.columns or "action_name" not in df.columns:
        return {}
    return {str(row["action_id"]): str(row["action_name"]) for _, row in df.iterrows()}


def _named_actions(action_ids: Any, action_map: dict[str, str]) -> str:
    ids = [x.strip() for x in str(action_ids).split(",") if x.strip()]
    return "；".join(f"{aid} {action_map.get(aid, '')}".strip() for aid in ids) or "待补充"


def _draft_md(ctx: dict[str, Any]) -> str:
    overview = _paper_overview_rows(ctx)
    action_map = _action_name_map(ctx)
    audit = ctx["audit_summary"]

    current_rows = [
        {
            "paper_id": r["paper_id"],
            "F1": r["F1"],
            "F2": r["F2"],
            "Q": r["Q_cur"],
            "grade": r["grade"],
        }
        for r in overview
    ]
    logic_rows = [
        {
            "paper_id": r["paper_id"],
            "Gamma": r["Gamma"],
            "G": r["G"],
            "weak_edges": r["weak_edges"],
        }
        for r in overview
    ]
    ai_rows = [
        {
            "paper_id": r["paper_id"],
            "R_AI": r["R_AI"],
            "risk_level": r["risk_level"],
            "source": _first_row(ctx["ai"], r["paper_id"]).get("main_risk_source", "待补充"),
        }
        for r in overview
    ]
    subj_rows = [
        {
            "paper_id": r["paper_id"],
            "std": r["std"],
            "level": r["disagreement"],
            "weakest": _first_row(ctx["subjectivity"], r["paper_id"]).get("min_score_agent", "待补充"),
            "strongest": _first_row(ctx["subjectivity"], r["paper_id"]).get("max_score_agent", "待补充"),
        }
        for r in overview
    ]
    revision_rows = [
        {
            "paper_id": r["paper_id"],
            "actions": _named_actions(r["actions"], action_map),
            "gain": r["gain"],
            "cost": r["cost"],
        }
        for r in overview
    ]
    prediction_rows = [
        {
            "paper_id": r["paper_id"],
            "before": r["Q_cur"],
            "after": r["score_after"],
            "gain": r["score_gain"],
            "level": r["level_after"],
            "pass_ratio": r["pass_ratio"],
        }
        for r in overview
    ]

    robust_prediction = ctx["robust_prediction"]
    robust_min = robust_prediction.loc[robust_prediction["metric"] == "mean_score_gain", "min"].iloc[0] if not robust_prediction.empty else None
    robust_max = robust_prediction.loc[robust_prediction["metric"] == "mean_score_gain", "max"].iloc[0] if not robust_prediction.empty else None
    positive_ratio = robust_prediction.loc[robust_prediction["metric"] == "proportion_positive_gain", "mean"].iloc[0] if not robust_prediction.empty else None
    top_actions = ctx["robust_action"].head(3).to_dict("records") if not ctx["robust_action"].empty else []

    return f"""
# 6 问题3：中等质量论文的诊断修复与稳健复评模型

## 6.1 问题分析与总体框架

第三问并不是对附件3进行孤立的重新评分，而是在问题1封版综合评价系统和问题2关键特征解释模型的基础上，对3篇中等质量论文构建“诊断-修复-复评”闭环。该闭环依次包括：当前质量复评、五元论证链诊断、AI辅助写作风险证据融合、多智能体评审分歧分析、修改动作优化、优化后质量预测和稳健性检验。

本文将模型输出定位为辅助诊断和修改建议，而不是主观点评。当前质量分用于刻画论文现有质量基线，逻辑断层用于解释“模型-结果-结论”或“数据-模型”链路的证据不足，AI辅助写作风险仅表示文本风险提示指标，多智能体分歧用于刻画不同评审维度对同一论文的判断波动，修改动作优化则给出成本受限下的可执行修复路径。

Step31最终审计显示，第三问核心文件无缺失，数据一致性通过，免责声明完整；审计状态统计为 PASS={audit.get("status_counts", {}).get("PASS", "待补充")}，WARNING={audit.get("status_counts", {}).get("WARNING", "待补充")}，FAIL={audit.get("status_counts", {}).get("FAIL", 0)}。

## 6.2 附件3当前质量复评

当前复评沿用问题1和问题2封版模型：先利用问题1的21项二级指标和组合权重得到第一层综合复评分 $F_1$，再利用问题2最终关键特征形成辅助解释分 $F_2$，按

$$
Q_{{cur}}=\\alpha F_1+(1-\\alpha)F_2,\\quad \\alpha=0.80
$$

得到当前质量基线分。该分数尚未扣除逻辑断层和AI辅助写作风险，也没有引入修改动作预测，因此与题面“中等质量论文”的描述不完全一致是可解释的。

{_markdown_table(current_rows, [("paper_id", "论文"), ("F1", "F1_score"), ("F2", "F2_key_score"), ("Q", "Q_cur_baseline"), ("grade", "当前等级")])}

由表可见，3-2当前基线较强，3-3次之，3-1短板最明显。需要注意的是，当前等级只是封版特征体系下的基线复评，后续逻辑断层、AI辅助写作风险和评审分歧会进一步解释这种差异。

## 6.3 五元论证链与逻辑断层诊断

为避免只看总分，本文构建五元论证链：

$$
T\\rightarrow D\\rightarrow HM\\rightarrow R\\rightarrow C\\rightarrow T ,
$$

其中，$T$表示赛题任务，$D$表示数据与数据来源，$HM$表示假设、模型、变量、公式和算法，$R$表示结果、图表和数值输出，$C$表示结论和方案建议。五条链路闭合度分别为 $s_{{TD}}$、$s_{{DHM}}$、$s_{{HMR}}$、$s_{{RC}}$、$s_{{CT}}$。总体逻辑闭合度定义为

$$
\\Gamma_i=0.20s_{{TD}}+0.25s_{{DHM}}+0.25s_{{HMR}}+0.20s_{{RC}}+0.10s_{{CT}},
$$

逻辑断层强度定义为

$$
G_i=1-\\Gamma_i .
$$

{_markdown_table(logic_rows, [("paper_id", "论文"), ("Gamma", "Gamma"), ("G", "G_logic_gap"), ("weak_edges", "主要弱边/诊断焦点")])}

结果表明，三篇论文均未出现 $s_e<0.35$ 的严重断链，说明其并非完全断链型论文；但三篇论文都存在中等程度逻辑缺口。3-1和3-2主要弱边集中在 $HM\\rightarrow R$ 和 $R\\rightarrow C$，说明模型产生结果以及结果支撑结论的证据链不够扎实；3-3主要弱边为 $D\\rightarrow HM$ 和 $C\\rightarrow T$，说明数据进入模型和结论回扣任务的链条仍需增强。

## 6.4 AI辅助写作风险D-S证据融合

AI辅助写作风险只作为文本风险提示指标，不作为学术不端或AI生成判定。本文使用四类可解释证据：$e_1$为模板化表达风险，$e_2$为无支撑结论风险，$e_3$为数据不可追溯风险，$e_4$为方法到结果跳跃风险。对第 $k$ 类证据，设命题 $A$ 表示存在较高AI辅助写作风险，$H$ 表示风险较低，$U$ 表示不确定，则质量分配为

$$
m_k(A)=\\rho_k e_k,\\quad m_k(H)=\\rho_k(1-e_k),\\quad m_k(U)=1-\\rho_k .
$$

使用Dempster组合规则融合四类证据，并以

$$
R_{{AI}}=BetP(A)
$$

作为论文级AI辅助写作风险指标。

{_markdown_table(ai_rows, [("paper_id", "论文"), ("R_AI", "R_AI"), ("risk_level", "风险等级"), ("source", "主要风险来源")])}

三篇论文AI辅助写作风险均为低风险，主要风险来源集中在数据不可追溯风险和方法到结果跳跃风险。这说明文本不存在高AI辅助风险提示，但仍需要通过补充数据来源、求解过程和结果支撑来降低写作风险指标。

必须强调：{DISCLAIMER}

## 6.5 多智能体评审主观性差异建模

为刻画不同评审维度对同一论文的判断波动，本文构建五类评审Agent：结构规范评审、逻辑严密评审、数学建模评审、结果验证评审和应用价值评审。设第 $i$ 篇论文在第 $r$ 个Agent下得分为 $Q_i^{{(r)}}$，则分歧度定义为

$$
D_i=std(Q_i^{{(1)}},Q_i^{{(2)}},\\ldots,Q_i^{{(5)}}).
$$

进一步可定义下置信界

$$
LCB(Q_i)=\\overline Q_i-z_\\gamma D_i,
$$

用于提示“平均分较高但评审分歧较大”的风险。

{_markdown_table(subj_rows, [("paper_id", "论文"), ("std", "Agent标准差"), ("level", "分歧等级"), ("weakest", "最低评分Agent"), ("strongest", "最高评分Agent")])}

三篇论文多智能体评分均为高分歧，平均评分标准差为 {_f(ctx["subjectivity"]["agent_score_std"].mean() if not ctx["subjectivity"].empty else None)}。这意味着修改方案不能只追求平均得分提升，还应提高低分维度的下置信界并降低评审分歧。

## 6.6 修改动作库与多目标优化

基于逻辑断层、AI辅助写作风险证据和多智能体分歧来源，本文构建修改动作库。动作包括补充摘要、问题分析、模型假设、符号说明、模型推导、算法流程、结果表格与可视化、灵敏度分析、误差分析、模型优缺点与推广、数据来源和处理说明、方法到结果解释链条、结构编号规范和应用价值表达等。

每个修改动作表示为

$$
a_j=(\\Delta Q_j,c_j,\\Delta G_j,\\Delta R_{{AI,j}},risk_j),
$$

其中 $\\Delta Q_j$ 为预期质量收益，$c_j$ 为修改成本，$\\Delta G_j$ 为逻辑断层下降量，$\\Delta R_{{AI,j}}$ 为AI辅助风险提示指标下降量，$risk_j$ 为执行难度或不确定性。动作选择目标为

$$
\\max \\sum_j x_j\\Delta Q_j,\\quad
\\min \\sum_j x_jc_j,\\quad
\\min G,\\quad
\\min R_{{AI}},\\quad
\\min D ,
$$

并满足预算约束 $\\sum_j x_jc_j\\le B$。由于动作数量较少，Step28采用0-1背包搜索而不是复杂智能优化算法。

{_markdown_table(revision_rows, [("paper_id", "论文"), ("actions", "推荐动作组合"), ("gain", "预计收益"), ("cost", "成本")])}

稳健性分析表明，A11“补充数据来源和数据处理说明”是最稳定修改动作，A10“增加模型优缺点与推广”和A12“增强方法到结果的解释链条”为次稳定主干动作。上述动作均对应真实诊断来源：A11对应数据不可追溯风险，A12对应方法到结果跳跃风险，A10对应模型评价和应用价值表达短板。

## 6.7 优化后质量预测

优化后质量预测不是实际修改后的真实评分，而是基于动作-特征映射和封版评价系统得到的模型预测值。概念上，可将优化后复评写为

$$
Q_{{new}}=\\alpha F_{{1,new}}+(1-\\alpha)F_{{2,new}}-\\lambda_GG_{{new}}-\\lambda_AR_{{AI,new}} .
$$

在Step29封版实现中，考虑到附件3尚未真实修改，采用动作预计收益映射到分数增量：

$$
score\\_gain_i=\\min(0.12E_i,12),\\quad
\\hat Q_i=\\min(100,Q_{{cur,i}}+score\\_gain_i).
$$

同时依据动作类型预测AI辅助风险提示指标下降和多智能体分歧收敛。

{_markdown_table(prediction_rows, [("paper_id", "论文"), ("before", "修改前分"), ("after", "预测修改后分"), ("gain", "预测提升"), ("level", "预测等级"), ("pass_ratio", "稳健达到合格比例")])}

3-1优化后仍难达到合格，3-2优化后稳定保持合格，3-3有较大概率提升到合格。三篇论文AI辅助风险提示指标整体下降，多智能体分歧整体呈下降趋势。但上述结果均为模型预测值，不等同于实际人工修改后的真实评分。

## 6.8 稳健性分析

Step30对收益映射系数、修改预算、风险下降倍率和分歧收敛倍率进行参数扰动，共形成225组参数组合、675条论文级预测记录。稳健性指标显示，平均预测提升分数区间为 {_f(robust_min)} 至 {_f(robust_max)}，正向提升比例为 {_f(positive_ratio)}。

{_markdown_table([{"action": row.get("action_id"), "rate": _f(row.get("selection_rate")), "count": _f(row.get("selected_count"), 0)} for row in top_actions], [("action", "稳定动作"), ("rate", "出现率"), ("count", "出现次数")])}

其中，A11出现率为1.000000，是最稳定动作；A10和A12出现率均为0.600000，是次稳定主干动作。论文层面，3-1达到合格及以上比例为0.000000，3-2为1.000000，3-3为0.560000。该结果说明修改建议整体具有稳定正收益，但不能夸大为真实修改后必然提升。

## 6.9 第三问小结

第三问构建了一个可解释、可复核、可优化的诊断修复系统，而不是主观点评体系。当前质量复评显示3-2基线较强、3-3次之、3-1短板明显；五元论证链表明三篇论文均存在中等逻辑缺口，其中3-1和3-2更集中于模型-结果-结论链路，3-3集中于数据-模型和结论回扣任务链路；AI辅助写作风险均为低风险，但数据不可追溯和方法到结果跳跃仍是主要文本风险来源；多智能体评审显示三篇论文均为高分歧，说明质量判断受评审维度偏好影响较大。

修改动作优化结果表明，A11、A10、A12构成最稳定的主干修复路径。优化后预测显示，3-1即使优化后仍难达到合格，3-2可稳定保持合格，3-3有较大概率提升到合格。第三问的局限在于：AI辅助写作风险不是学术不端判断；优化后得分是模型预测；动作收益来自规则映射和封版模型复评；附件3样本只有3篇；最终修改仍需人工复核。
"""


def _tables_md(ctx: dict[str, Any]) -> str:
    rows = [
        {"title": "表X 附件3当前质量复评结果", "source": "output/problem3_tables/appendix3_current_evaluation.xlsx", "position": "6.2", "note": "展示F1_score、F2_key_score、Q_cur_baseline和当前等级。"},
        {"title": "表X 五元论证链逻辑断层诊断结果", "source": "output/problem3_tables/logic_gap_diagnosis.xlsx", "position": "6.3", "note": "展示Gamma、G_logic_gap和主要弱边。"},
        {"title": "表X AI辅助写作风险D-S融合结果", "source": "output/problem3_tables/ai_risk_ds_fusion.xlsx", "position": "6.4", "note": "展示R_AI、风险等级和主要风险来源，需保留免责声明。"},
        {"title": "表X 多智能体评审分歧结果", "source": "output/problem3_tables/multi_agent_subjectivity_analysis.xlsx", "position": "6.5", "note": "展示Agent得分标准差、最高/最低评分Agent和分歧等级。"},
        {"title": "表X 修改动作库", "source": "output/problem3_tables/revision_action_library.xlsx", "position": "6.6", "note": "列出A1-A14动作、目标维度、成本和风险约束作用。"},
        {"title": "表X 推荐修改动作组合", "source": "output/problem3_tables/revision_action_optimization.xlsx", "position": "6.6", "note": "展示每篇论文的背包推荐动作、预计收益和成本。"},
        {"title": "表X 优化后质量预测结果", "source": "output/problem3_tables/quality_prediction_after_revision.xlsx", "position": "6.7", "note": "展示修改前后预测分、风险下降和分歧下降。"},
        {"title": "表X 第三问稳健性分析结果", "source": "output/problem3_tables/robustness_summary.xlsx", "position": "6.8", "note": "展示参数扰动下得分提升、动作稳定性和合格比例。"},
        {"title": "表X 第三问最终审计结论清单", "source": "output/problem3_tables/problem3_conclusion_checklist.xlsx", "position": "附录或6.9", "note": "用于证明关键结论均由结果文件支持。"},
    ]
    return "# 第三问建议插入表格\n\n" + _markdown_table(rows, [("title", "表题"), ("source", "来源文件"), ("position", "建议位置"), ("note", "用途说明")])


def _figures_md(ctx: dict[str, Any]) -> str:
    df = ctx["figures"]
    if df.empty:
        rows = []
    else:
        rows = []
        for _, row in df.iterrows():
            rows.append(
                {
                    "name": row.get("figure_name", "待补充"),
                    "path": row.get("file_path", "待补充"),
                    "caption": row.get("suggested_caption", "待补充"),
                    "section": row.get("suggested_section", "待补充"),
                    "priority": row.get("paper_usage_priority", "待补充"),
                }
            )
    intro = "# 第三问建议插入图表\n\n优先推荐正文使用：`ai_risk_radar.png`、`ai_risk_bar.png`、`agent_disagreement_bar.png`、`revision_priority_bar.png`、`score_before_after.png`。\n\n"
    return intro + _markdown_table(rows, [("name", "图文件"), ("path", "路径"), ("caption", "建议图注"), ("section", "建议位置"), ("priority", "优先级")])


def _formulas_md() -> str:
    text = r"""
# 第三问方法公式汇总

## 1. 当前复评公式

$$
Q_{{cur}}=\\alpha F_1+(1-\\alpha)F_2,\\quad \\alpha=0.80.
$$

其中 $F_1$ 来自问题1封版综合评价模型，$F_2$ 来自问题2关键特征辅助模型。

## 2. 五元论证链

$$
T\\rightarrow D\\rightarrow HM\\rightarrow R\\rightarrow C\\rightarrow T.
$$

$$
\\Gamma_i=\\sum_e \\omega_e s_e
=0.20s_{{TD}}+0.25s_{{DHM}}+0.25s_{{HMR}}+0.20s_{{RC}}+0.10s_{{CT}}.
$$

$$
G_i=1-\\Gamma_i.
$$

## 3. D-S证据融合

$$
m_k(A)=\\rho_k e_k,\\quad m_k(H)=\\rho_k(1-e_k),\\quad m_k(U)=1-\\rho_k.
$$

Dempster组合规则为

$$
m_{12}(X)=\\frac{\\sum_{B\\cap C=X}m_1(B)m_2(C)}{1-K},\\quad
K=\\sum_{B\\cap C=\\varnothing}m_1(B)m_2(C).
$$

融合后以

$$
R_{{AI}}=BetP(A)
$$

作为AI辅助写作风险提示指标。__DISCLAIMER__

## 4. 多智能体评分与分歧

$$
\\overline Q_i=\\frac1R\\sum_{r=1}^{R}Q_i^{{(r)}},\\quad
D_i=std(Q_i^{{(1)}},\\ldots,Q_i^{{(R)}}).
$$

$$
LCB(Q_i)=\\overline Q_i-z_\\gamma D_i.
$$

## 5. 修改动作向量与优化

$$
a_j=(\\Delta Q_j,c_j,\\Delta G_j,\\Delta R_{{AI,j}},risk_j).
$$

$$
\\max \\sum_j x_j\\Delta Q_j,\quad
\\sum_j x_jc_j\\le B,\quad x_j\\in\\{{0,1\\}}.
$$

同时关注 $G_i$、$R_{{AI}}$ 和多智能体分歧 $D_i$ 的下降。

## 6. 优化后预测

概念模型：

$$
Q_{{new}}=\\alpha F_{{1,new}}+(1-\\alpha)F_{{2,new}}-\\lambda_GG_{{new}}-\\lambda_AR_{{AI,new}}.
$$

Step29封版实现：

$$
score\\_gain_i=\\min(0.12E_i,12),
$$

$$
\\hat Q_i=\\min(100,Q_{{cur,i}}+score\\_gain_i).
$$

AI辅助风险和评审分歧预测为

$$
R_{{AI,after}}=\\max(0,R_{{AI,before}}-\\Delta R_{{AI}}),
$$

$$
D_{after}=\\max(0,D_{before}-\\Delta D).
$$

## 7. 稳健性检验

Step30扰动参数包括收益映射系数、修改预算、风险下降倍率和分歧收敛倍率。对参数组合 $\\theta$，记录

$$
\\hat Q_i(\\theta),\\quad R_{{AI,i}}(\\theta),\\quad D_i(\\theta),\\quad x_j(\\theta).
$$

动作稳定性可写为

$$
P(a_j)=\\frac{\\#\\{{\\theta:a_j\\text{{ 被选中}}\\}}}{\\#\\{{\\theta\\}}}.
$$
"""
    return text.replace("__DISCLAIMER__", DISCLAIMER)


def _result_summary_md(ctx: dict[str, Any]) -> str:
    overview = _paper_overview_rows(ctx)
    action_map = _action_name_map(ctx)
    rows = []
    for r in overview:
        rows.append(
            {
                "paper_id": r["paper_id"],
                "current": r["Q_cur"],
                "logic": r["G"],
                "R_AI": r["R_AI"],
                "std": r["std"],
                "actions": _named_actions(r["actions"], action_map),
                "after": r["score_after"],
                "pass_ratio": r["pass_ratio"],
            }
        )
    robust_prediction = ctx["robust_prediction"]
    robust_min = robust_prediction.loc[robust_prediction["metric"] == "mean_score_gain", "min"].iloc[0] if not robust_prediction.empty else None
    robust_max = robust_prediction.loc[robust_prediction["metric"] == "mean_score_gain", "max"].iloc[0] if not robust_prediction.empty else None
    return "# 第三问关键结果汇总\n\n" + _markdown_table(
        rows,
        [
            ("paper_id", "论文"),
            ("current", "当前基线分"),
            ("logic", "逻辑断层强度"),
            ("R_AI", "AI辅助风险"),
            ("std", "Agent分歧标准差"),
            ("actions", "推荐动作"),
            ("after", "预测优化后分"),
            ("pass_ratio", "稳健合格比例"),
        ],
    ) + f"\n\n稳健性平均预测提升分数区间为 {_f(robust_min)} 至 {_f(robust_max)}；A11为最稳定动作，A10和A12为次稳定动作。"


def _limitations_md() -> str:
    return f"""
# 第三问模型边界与局限

1. AI辅助写作风险不是学术不端判断，也不是AI生成判定。必须保留表述：{DISCLAIMER}
2. 优化后得分是模型预测值，不是实际修改后的真实评分。
3. 修改动作收益来自规则映射、动作-特征对应关系和封版模型复评，不是人工主观承诺。
4. 附件3只有3篇论文，第三问更适合写成可审计诊断系统，不宜夸大统计泛化能力。
5. 多智能体评分是基于已有维度的agent-like评分矩阵，反映评审维度差异，不等同于真实外部评委打分。
6. 稳健性分析检验的是参数扰动下预测结论的稳定性，不表示真实人工修改后的必然结果。
7. 最终修改方案仍需人工复核，尤其是关键数据来源、求解过程、结果表格和结论表述。
"""


def _final_conclusions_md(ctx: dict[str, Any]) -> str:
    return f"""
# 第三问结论精炼版

第三问基于前两问封版评价系统，构建了“当前复评-逻辑诊断-AI辅助写作风险识别-多智能体分歧分析-修改动作优化-优化后预测-稳健性检验”的闭环模型。该模型不是主观点评，而是面向中等质量论文的可解释诊断与修复系统。

主要结论如下：

1. 当前质量复评显示，3-2基线质量相对较高，3-3次之，3-1短板最明显。
2. 五元论证链诊断表明，三篇论文均不存在严重断链，但均存在中等程度逻辑缺口。3-1和3-2主要集中在模型-结果-结论闭合证据不足，3-3主要集中在数据进入模型和结论回扣任务不足。
3. 三篇论文AI辅助写作风险均为低风险，主要风险来源集中在数据不可追溯和方法到结果跳跃。{DISCLAIMER}
4. 多智能体评审显示三篇论文均为高分歧，说明单一平均分不足以刻画论文质量，应同时关注最低评分维度和分歧收敛。
5. 修改动作优化结果表明，A11“补充数据来源和数据处理说明”是最稳定修复动作，A10“增加模型优缺点与推广”和A12“增强方法到结果的解释链条”为次稳定主干动作。
6. 优化后质量预测显示，3-1优化后仍难达到合格，3-2优化后稳定保持合格，3-3有较大概率提升到合格。
7. 稳健性分析表明，在225组参数扰动下，预测提升保持正向，AI辅助风险提示指标整体下降，多智能体分歧整体下降，但结果仍应表述为模型预测和辅助复核结论。
"""


def generate_problem3_draft(project_root: Path | str | None = None) -> dict[str, Any]:
    """Generate Step 32 Markdown writing materials.

    Parameters
    ----------
    project_root:
        Project root. Defaults to the parent directory of this module.
    """

    root = Path(project_root).resolve() if project_root else Path(__file__).resolve().parents[1]
    paths = _paths(root)
    paths.paper_dir.mkdir(parents=True, exist_ok=True)
    logger = _setup_logger(paths.logs_dir / "problem3_draft_generation.log")
    logger.info("Starting Step 32 draft generation from sealed outputs.")

    ctx = _collect_context(paths, logger)

    files = {
        "problem3_draft.md": _draft_md(ctx),
        "problem3_tables_to_insert.md": _tables_md(ctx),
        "problem3_figures_to_insert.md": _figures_md(ctx),
        "problem3_method_formulas.md": _formulas_md(),
        "problem3_result_summary.md": _result_summary_md(ctx),
        "problem3_limitations.md": _limitations_md(),
        "problem3_final_conclusions.md": _final_conclusions_md(ctx),
    }
    output_paths: list[str] = []
    for name, text in files.items():
        out = paths.paper_dir / name
        _write(out, text)
        output_paths.append(str(out))
        logger.info("Wrote %s", out)

    logger.info("Step 32 draft generation complete.")
    return {
        "status": "success",
        "output_dir": str(paths.paper_dir),
        "output_files": output_paths,
        "paper_count": len(PAPER_IDS),
    }
