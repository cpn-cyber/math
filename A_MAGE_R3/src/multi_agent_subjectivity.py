"""Problem 3 Step 27: multi-agent reviewer subjectivity analysis.

This step does not call external reviewers or language models. When no explicit
multi-agent score table exists, it builds an auditable agent-like score matrix
from the sealed Problem 1 normalized indicators:

structure, logic, modeling, result validation, and writing/application.
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


LOGGER_NAME = "A_MAGE_R3.problem3.multi_agent_subjectivity"

SEARCH_KEYWORDS = [
    "agent",
    "multi_agent",
    "review",
    "score",
    "evaluation",
    "paper_quality",
]

AGENT_DEFINITIONS: dict[str, dict[str, Any]] = {
    "structure_reviewer": {
        "label": "结构规范智能体",
        "indicator_prefixes": ["I01", "I02", "I03", "I04"],
        "feature_group": "A1 结构规范性",
        "interpretation": "结构完整、摘要、图表编号和附录支撑",
    },
    "logic_reviewer": {
        "label": "逻辑一致智能体",
        "indicator_prefixes": ["I05", "I06", "I07", "I08"],
        "feature_group": "A2 问题理解与逻辑严密性",
        "interpretation": "问题重述、假设匹配、逻辑衔接和结果结论一致",
    },
    "modeling_reviewer": {
        "label": "数学建模智能体",
        "indicator_prefixes": ["I09", "I10", "I11", "I12", "I13"],
        "feature_group": "A3 方法合理性与数学建模质量",
        "interpretation": "模型数量、公式、变量、目标约束和方法合理性",
    },
    "result_validation_reviewer": {
        "label": "结果验证智能体",
        "indicator_prefixes": ["I14", "I15", "I16", "I17"],
        "feature_group": "A4 结果分析与验证",
        "interpretation": "结果完整、图表解释、灵敏度与误差分析",
    },
    "application_value_reviewer": {
        "label": "写作应用智能体",
        "indicator_prefixes": ["I18", "I19", "I20", "I21"],
        "feature_group": "A5 写作规范与应用价值",
        "interpretation": "参考文献、可读性、创新表达和推广价值",
    },
}

AI_RISK_DISCLAIMER = "AI 风险低不代表论文质量高，仅表示文本存在 AI 辅助写作风险较低。"


def setup_subjectivity_logger(log_path: Path) -> logging.Logger:
    """Configure Step 27 logger."""
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
    """Use a Chinese-capable matplotlib font when one is installed."""
    candidates = ["Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC", "Arial Unicode MS"]
    available = {font.name for font in font_manager.fontManager.ttflist}
    for candidate in candidates:
        if candidate in available:
            plt.rcParams["font.sans-serif"] = [candidate]
            break
    plt.rcParams["axes.unicode_minus"] = False


def _natural_sort_key(value: Any) -> list[Any]:
    """Natural sort key for paper IDs and paths."""
    text = Path(str(value)).stem
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def _indicator_prefix(column: str) -> str:
    """Return an indicator prefix such as I01."""
    match = re.match(r"(I\d+)", str(column))
    return match.group(1) if match else str(column)


def _safe_numeric(series: pd.Series) -> pd.Series:
    """Convert a series to numeric values."""
    return pd.to_numeric(series, errors="coerce")


def _score_to_100(series: pd.Series) -> pd.Series:
    """Normalize an arbitrary score-like series to a 0-100 scale."""
    values = _safe_numeric(series)
    if values.notna().sum() == 0:
        return values
    min_value = float(values.min())
    max_value = float(values.max())
    if min_value >= -1e-9 and max_value <= 1.2:
        return (values.clip(lower=0, upper=1) * 100).clip(lower=0, upper=100)
    if min_value >= -1e-9 and max_value <= 100 + 1e-9:
        return values.clip(lower=0, upper=100)
    if math.isclose(min_value, max_value):
        return pd.Series(np.full(len(values), 50.0), index=values.index)
    return ((values - min_value) / (max_value - min_value) * 100).clip(lower=0, upper=100)


def discover_score_sources(project_root: Path) -> list[Path]:
    """Find score or review candidate files without modifying them."""
    output_dir = project_root / "output"
    if not output_dir.exists():
        return []
    paths: list[Path] = []
    for path in output_dir.rglob("*"):
        if path.suffix.lower() not in {".xlsx", ".json", ".csv"}:
            continue
        name = path.name.lower()
        if any(keyword in name for keyword in SEARCH_KEYWORDS):
            paths.append(path)
    return sorted(paths, key=lambda item: str(item).lower())


def _find_indicator_columns(df: pd.DataFrame, prefixes: list[str]) -> list[str]:
    """Find normalized feature columns matching indicator prefixes."""
    columns: list[str] = []
    for prefix in prefixes:
        matches = [column for column in df.columns if _indicator_prefix(str(column)) == prefix]
        if matches:
            columns.append(matches[0])
    return columns


def build_agent_score_matrix(
    normalized_feature_path: Path,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build agent-like reviewer scores from sealed normalized indicators."""
    if not Path(normalized_feature_path).exists():
        raise FileNotFoundError(f"Normalized Appendix 3 feature table not found: {normalized_feature_path}")

    features = pd.read_excel(normalized_feature_path)
    required_id_cols = {"paper_id", "filename"}
    missing_ids = required_id_cols - set(features.columns)
    if missing_ids:
        raise ValueError(f"Appendix 3 normalized features missing ID columns: {sorted(missing_ids)}")

    score_rows = features[["paper_id", "filename"]].copy()
    mapping_rows: list[dict[str, Any]] = []

    for agent, definition in AGENT_DEFINITIONS.items():
        indicator_columns = _find_indicator_columns(features, definition["indicator_prefixes"])
        if not indicator_columns:
            logger.warning("%s has no matching indicator columns and will be omitted", agent)
            continue
        score_rows[agent] = (
            features[indicator_columns]
            .apply(_safe_numeric)
            .mean(axis=1, skipna=True)
            .pipe(_score_to_100)
        )
        mapping_rows.append(
            {
                "agent": agent,
                "agent_label": definition["label"],
                "feature_group": definition["feature_group"],
                "indicator_prefixes": ",".join(definition["indicator_prefixes"]),
                "source_columns": ",".join(indicator_columns),
                "source_file": str(Path(normalized_feature_path)),
                "score_scale": "0-100",
                "construction_note": "由第一问封版标准化二级指标按一级维度均值聚合，作为 agent-like 评分来源。",
            }
        )

    agent_columns = [agent for agent in AGENT_DEFINITIONS if agent in score_rows.columns]
    if len(agent_columns) < 2:
        raise ValueError("At least two agent-like score columns are required for subjectivity analysis.")

    score_rows = score_rows.sort_values("paper_id", key=lambda values: values.map(_natural_sort_key)).reset_index(drop=True)
    mapping = pd.DataFrame(mapping_rows)
    logger.info("Agent-like score matrix built from %s", normalized_feature_path)
    logger.info("Agent columns: %s", agent_columns)
    return score_rows, mapping


def calculate_disagreement(agent_scores: pd.DataFrame) -> pd.DataFrame:
    """Calculate paper-level reviewer disagreement statistics."""
    agent_columns = [column for column in AGENT_DEFINITIONS if column in agent_scores.columns]
    rows: list[dict[str, Any]] = []

    for _, row in agent_scores.iterrows():
        scores = row[agent_columns].astype(float)
        mean_score = float(scores.mean())
        std_score = float(scores.std(ddof=0))
        score_range = float(scores.max() - scores.min())
        cv = float(std_score / abs(mean_score)) if abs(mean_score) > 1e-9 else math.nan
        max_agent = str(scores.idxmax())
        min_agent = str(scores.idxmin())
        if std_score >= 8 or (not pd.isna(cv) and cv >= 0.15):
            level = "高分歧"
        elif std_score < 3 or (not pd.isna(cv) and cv < 0.05):
            level = "低分歧"
        else:
            level = "中分歧"

        max_label = AGENT_DEFINITIONS[max_agent]["label"]
        min_label = AGENT_DEFINITIONS[min_agent]["label"]
        explanation = (
            f"{max_label}评分最高，{min_label}评分最低，最大差异为{score_range:.2f}分；"
            f"说明该论文在{AGENT_DEFINITIONS[max_agent]['interpretation']}方面相对更强，"
            f"但在{AGENT_DEFINITIONS[min_agent]['interpretation']}方面存在相对短板。"
        )
        rows.append(
            {
                "paper_id": row["paper_id"],
                "filename": row["filename"],
                "agent_score_mean": mean_score,
                "agent_score_std": std_score,
                "agent_score_range": score_range,
                "agent_score_cv": cv,
                "max_score_agent": max_agent,
                "min_score_agent": min_agent,
                "max_score": float(scores.max()),
                "min_score": float(scores.min()),
                "disagreement_level": level,
                "max_difference_dimension": f"{max_agent} - {min_agent}",
                "difference_explanation": explanation,
            }
        )
    return pd.DataFrame(rows)


def calculate_agent_correlations(agent_scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Calculate Pearson and Spearman correlations among agent scores."""
    agent_columns = [column for column in AGENT_DEFINITIONS if column in agent_scores.columns]
    matrix = agent_scores[agent_columns].apply(pd.to_numeric, errors="coerce")
    pearson = matrix.corr(method="pearson")
    spearman = matrix.corr(method="spearman")

    long_rows: list[dict[str, Any]] = []
    for i, agent_i in enumerate(agent_columns):
        for agent_j in agent_columns[i + 1 :]:
            long_rows.append(
                {
                    "agent_i": agent_i,
                    "agent_j": agent_j,
                    "pearson_corr": pearson.loc[agent_i, agent_j],
                    "spearman_corr": spearman.loc[agent_i, agent_j],
                    "interpretation": _interpret_agent_pair(agent_i, agent_j, pearson.loc[agent_i, agent_j]),
                }
            )
    return pearson, spearman, pd.DataFrame(long_rows)


def _interpret_agent_pair(agent_i: str, agent_j: str, corr: float) -> str:
    """Provide a short pairwise correlation interpretation."""
    label_i = AGENT_DEFINITIONS.get(agent_i, {}).get("label", agent_i)
    label_j = AGENT_DEFINITIONS.get(agent_j, {}).get("label", agent_j)
    if pd.isna(corr):
        return f"{label_i}与{label_j}相关性无法稳定计算，样本量较小。"
    if corr >= 0.7:
        level = "高度同向"
    elif corr >= 0.3:
        level = "中等同向"
    elif corr <= -0.3:
        level = "存在反向差异"
    else:
        level = "相关较弱"
    return f"{label_i}与{label_j}{level}；N=3，仅用于一致性审计。"


def load_ai_risk(ai_fusion_path: Path, ai_evidence_path: Path, logger: logging.Logger) -> pd.DataFrame:
    """Load and merge Step 26 AI-writing assistance risk outputs."""
    if not Path(ai_fusion_path).exists():
        logger.warning("AI risk fusion table not found: %s", ai_fusion_path)
        return pd.DataFrame(columns=["paper_id"])

    fusion = pd.read_excel(ai_fusion_path)
    evidence = pd.read_excel(ai_evidence_path) if Path(ai_evidence_path).exists() else pd.DataFrame(columns=["paper_id"])
    needed_fusion = [
        column
        for column in [
            "paper_id",
            "R_AI",
            "risk_level",
            "main_risk_source",
            "risk_explanation",
            "conflict_K_max",
            "conflict_K_mean",
        ]
        if column in fusion.columns
    ]
    merged = fusion[needed_fusion].copy()
    evidence_cols = [
        column
        for column in [
            "paper_id",
            "e1_template_expression",
            "e2_unsupported_conclusion",
            "e3_data_untraceable",
            "e4_method_result_jump",
            "top_risk_evidence",
        ]
        if column in evidence.columns
    ]
    if evidence_cols:
        merged = merged.merge(evidence[evidence_cols], on="paper_id", how="left")
    logger.info("AI risk merged from %s and %s", ai_fusion_path, ai_evidence_path)
    return merged


def merge_ai_risk(
    disagreement: pd.DataFrame,
    agent_scores: pd.DataFrame,
    ai_risk: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Merge AI risk with paper disagreement and calculate risk-score relations."""
    merged = disagreement.merge(ai_risk, on="paper_id", how="left")
    merged["ai_risk_quality_boundary_note"] = AI_RISK_DISCLAIMER

    agent_columns = [column for column in AGENT_DEFINITIONS if column in agent_scores.columns]
    joined = agent_scores[["paper_id"] + agent_columns].merge(ai_risk, on="paper_id", how="left")
    rows: list[dict[str, Any]] = []
    for target in ["R_AI", "e1_template_expression", "e2_unsupported_conclusion", "e3_data_untraceable", "e4_method_result_jump"]:
        if target not in joined.columns:
            continue
        for agent in agent_columns:
            pair = joined[[agent, target]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(pair) < 2:
                pearson = math.nan
                spearman = math.nan
            else:
                pearson = float(pair[agent].corr(pair[target], method="pearson"))
                spearman = float(pair[agent].corr(pair[target], method="spearman"))
            rows.append(
                {
                    "risk_or_evidence": target,
                    "agent": agent,
                    "agent_label": AGENT_DEFINITIONS[agent]["label"],
                    "pearson_corr": pearson,
                    "spearman_corr": spearman,
                    "n": int(len(pair)),
                    "interpretation": _interpret_ai_agent_relation(agent, target, spearman),
                }
            )
    return merged, pd.DataFrame(rows)


def _interpret_ai_agent_relation(agent: str, target: str, corr: float) -> str:
    """Provide conservative text for AI-risk and score relations."""
    label = AGENT_DEFINITIONS.get(agent, {}).get("label", agent)
    target_text = {
        "R_AI": "综合 AI辅助写作风险",
        "e1_template_expression": "模板化表达风险",
        "e2_unsupported_conclusion": "无支撑结论风险",
        "e3_data_untraceable": "数据不可追溯风险",
        "e4_method_result_jump": "方法到结果跳跃风险",
    }.get(target, target)
    if pd.isna(corr):
        return f"{label}与{target_text}关系无法稳定估计；N=3，仅作审计提示。"
    if corr <= -0.3:
        direction = "呈反向关系"
    elif corr >= 0.3:
        direction = "呈同向波动"
    else:
        direction = "关系较弱"
    return f"{label}与{target_text}{direction}；N=3，不能作显著性或因果结论。"


def make_interpretation_text(
    merged: pd.DataFrame,
    ai_relation: pd.DataFrame,
) -> pd.DataFrame:
    """Build paper-level and overall interpretation text."""
    rows: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        rows.append(
            {
                "item": f"{row['paper_id']}_disagreement",
                "interpretation": (
                    f"{row['paper_id']} 的多维评分标准差为 {row['agent_score_std']:.2f}，"
                    f"极差为 {row['agent_score_range']:.2f}，分歧等级为 {row['disagreement_level']}。"
                    f"{row['difference_explanation']}"
                ),
            }
        )

    if not ai_relation.empty:
        writing = ai_relation[
            (ai_relation["agent"] == "application_value_reviewer")
            & (ai_relation["risk_or_evidence"].isin(["R_AI", "e1_template_expression"]))
        ]
        for record in writing.to_dict(orient="records"):
            rows.append(
                {
                    "item": f"ai_risk_relation_{record['risk_or_evidence']}",
                    "interpretation": record["interpretation"],
                }
            )

    rows.append(
        {
            "item": "method_boundary",
            "interpretation": (
                "本步骤的智能体评分来自第一问封版指标维度的可解释聚合，"
                "用于分析评分维度差异和主观波动，不等同于外部专家独立复评。"
            ),
        }
    )
    rows.append({"item": "ai_risk_boundary", "interpretation": AI_RISK_DISCLAIMER})
    return pd.DataFrame(rows)


def draw_correlation_heatmap(corr: pd.DataFrame, output_path: Path) -> None:
    """Draw agent correlation heatmap."""
    _set_chinese_font()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=160)
    data = corr.fillna(0).to_numpy(dtype=float)
    image = ax.imshow(data, vmin=-1, vmax=1, cmap="coolwarm")
    labels = [AGENT_DEFINITIONS.get(col, {}).get("label", col) for col in corr.columns]
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=8, color="black")
    ax.set_title("多智能体评分 Pearson 相关性热力图", fontsize=12, pad=12)
    fig.colorbar(image, ax=ax, shrink=0.8, label="Pearson r")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def draw_disagreement_bar(disagreement: pd.DataFrame, output_path: Path) -> None:
    """Draw paper-level standard deviation and range bars."""
    _set_chinese_font()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = disagreement.sort_values("paper_id", key=lambda values: values.map(_natural_sort_key))
    x = np.arange(len(ordered))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=160)
    ax.bar(x - width / 2, ordered["agent_score_std"], width=width, label="标准差", color="#4C78A8")
    ax.bar(x + width / 2, ordered["agent_score_range"], width=width, label="极差", color="#F58518", alpha=0.82)
    ax.set_xticks(x)
    ax.set_xticklabels(ordered["paper_id"].astype(str).tolist())
    ax.set_ylabel("分歧强度")
    ax.set_title("附件3论文多智能体评分分歧", fontsize=12, pad=12)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(frameon=False)
    for xpos, value in zip(x - width / 2, ordered["agent_score_std"]):
        ax.text(xpos, value + 0.5, f"{value:.1f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def draw_paper_radar(agent_scores: pd.DataFrame, output_path: Path) -> None:
    """Draw all Appendix 3 papers on an agent-score radar chart."""
    _set_chinese_font()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    agent_columns = [column for column in AGENT_DEFINITIONS if column in agent_scores.columns]
    labels = [AGENT_DEFINITIONS[col]["label"].replace("智能体", "") for col in agent_columns]
    angles = np.linspace(0, 2 * np.pi, len(agent_columns), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7.2, 6.2), subplot_kw={"polar": True}, dpi=160)
    for _, row in agent_scores.sort_values("paper_id", key=lambda values: values.map(_natural_sort_key)).iterrows():
        values = [float(row[col]) for col in agent_columns]
        values += values[:1]
        ax.plot(angles, values, linewidth=1.8, label=str(row["paper_id"]))
        ax.fill(angles, values, alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=8)
    ax.set_title("附件3论文多智能体评分雷达图", fontsize=12, pad=18)
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1), frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def write_outputs(
    paths: dict[str, Path],
    agent_scores: pd.DataFrame,
    mapping: pd.DataFrame,
    disagreement_ai: pd.DataFrame,
    pearson: pd.DataFrame,
    spearman: pd.DataFrame,
    corr_long: pd.DataFrame,
    ai_relation: pd.DataFrame,
    interpretation: pd.DataFrame,
    score_sources: list[Path],
) -> None:
    """Write all Step 27 tables."""
    tables_dir = paths["tables_dir"]
    tables_dir.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(paths["analysis_table"], engine="openpyxl") as writer:
        disagreement_ai.to_excel(writer, index=False, sheet_name="subjectivity_analysis")
        agent_scores.to_excel(writer, index=False, sheet_name="agent_score_matrix")
        mapping.to_excel(writer, index=False, sheet_name="agent_source_mapping")

    long_scores = agent_scores.melt(
        id_vars=["paper_id", "filename"],
        value_vars=[column for column in AGENT_DEFINITIONS if column in agent_scores.columns],
        var_name="agent",
        value_name="score_0_100",
    )
    long_scores["agent_label"] = long_scores["agent"].map(lambda agent: AGENT_DEFINITIONS[agent]["label"])
    details = disagreement_ai[
        [
            "paper_id",
            "filename",
            "max_score_agent",
            "min_score_agent",
            "max_score",
            "min_score",
            "agent_score_range",
            "disagreement_level",
            "max_difference_dimension",
            "difference_explanation",
        ]
    ].copy()
    with pd.ExcelWriter(paths["details_table"], engine="openpyxl") as writer:
        details.to_excel(writer, index=False, sheet_name="disagreement_details")
        long_scores.to_excel(writer, index=False, sheet_name="agent_scores_long")
        mapping.to_excel(writer, index=False, sheet_name="source_mapping")

    with pd.ExcelWriter(paths["correlation_table"], engine="openpyxl") as writer:
        pearson.to_excel(writer, sheet_name="pearson_matrix")
        spearman.to_excel(writer, sheet_name="spearman_matrix")
        corr_long.to_excel(writer, index=False, sheet_name="pairwise_correlation")

    source_df = pd.DataFrame({"identified_score_file": [str(path) for path in score_sources]})
    high_cases = disagreement_ai.loc[disagreement_ai["disagreement_level"].eq("高分歧")].copy()
    with pd.ExcelWriter(paths["summary_table"], engine="openpyxl") as writer:
        disagreement_ai.to_excel(writer, index=False, sheet_name="paper_disagreement_overview")
        pearson.to_excel(writer, sheet_name="agent_correlation_pearson")
        spearman.to_excel(writer, sheet_name="agent_correlation_spearman")
        ai_relation.to_excel(writer, index=False, sheet_name="ai_risk_merge_summary")
        high_cases.to_excel(writer, index=False, sheet_name="high_disagreement_cases")
        interpretation.to_excel(writer, index=False, sheet_name="interpretation_text")
        source_df.to_excel(writer, index=False, sheet_name="identified_score_sources")


def run_multi_agent_subjectivity(config_path: Path | None = None) -> dict[str, Any]:
    """Run Step 27 multi-agent reviewer subjectivity analysis."""
    project_root = Path(__file__).resolve().parents[1]
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
        "analysis_table": tables_dir / "multi_agent_subjectivity_analysis.xlsx",
        "details_table": tables_dir / "multi_agent_disagreement_details.xlsx",
        "correlation_table": tables_dir / "agent_score_correlation.xlsx",
        "summary_table": tables_dir / "agent_subjectivity_summary.xlsx",
        "correlation_heatmap": charts_dir / "agent_score_correlation_heatmap.png",
        "disagreement_bar": charts_dir / "agent_disagreement_bar.png",
        "paper_radar": charts_dir / "paper_disagreement_radar.png",
        "log": logs_dir / "multi_agent_subjectivity.log",
    }

    logger = setup_subjectivity_logger(paths["log"])
    logger.info("Step 27 multi-agent subjectivity analysis started")

    score_sources = discover_score_sources(project_root)
    logger.info("Identified score/review candidate files: %s", [str(path) for path in score_sources])

    normalized_feature_path = tables_dir / "appendix3_features_normalized.xlsx"
    agent_scores, mapping = build_agent_score_matrix(normalized_feature_path, logger)
    disagreement = calculate_disagreement(agent_scores)
    pearson, spearman, corr_long = calculate_agent_correlations(agent_scores)

    ai_risk = load_ai_risk(
        tables_dir / "ai_risk_ds_fusion.xlsx",
        tables_dir / "ai_risk_evidence.xlsx",
        logger,
    )
    disagreement_ai, ai_relation = merge_ai_risk(disagreement, agent_scores, ai_risk)
    interpretation = make_interpretation_text(disagreement_ai, ai_relation)

    write_outputs(
        paths=paths,
        agent_scores=agent_scores,
        mapping=mapping,
        disagreement_ai=disagreement_ai,
        pearson=pearson,
        spearman=spearman,
        corr_long=corr_long,
        ai_relation=ai_relation,
        interpretation=interpretation,
        score_sources=score_sources,
    )
    draw_correlation_heatmap(pearson, paths["correlation_heatmap"])
    draw_disagreement_bar(disagreement_ai, paths["disagreement_bar"])
    draw_paper_radar(agent_scores, paths["paper_radar"])

    mean_std = float(disagreement_ai["agent_score_std"].mean())
    high_count = int(disagreement_ai["disagreement_level"].eq("高分歧").sum())
    highest_row = disagreement_ai.sort_values("agent_score_std", ascending=False).iloc[0]
    logger.info("Average agent score std: %.6f", mean_std)
    logger.info("High disagreement papers: %s", high_count)
    logger.info("Highest disagreement paper: %s", highest_row["paper_id"])
    logger.info("Step 27 outputs written")

    return {
        "paths": paths,
        "score_sources": score_sources,
        "agent_scores": agent_scores,
        "mapping": mapping,
        "analysis": disagreement_ai,
        "pearson": pearson,
        "spearman": spearman,
        "ai_relation": ai_relation,
        "mean_std": mean_std,
        "high_disagreement_count": high_count,
        "highest_disagreement_paper": str(highest_row["paper_id"]),
    }


__all__ = [
    "run_multi_agent_subjectivity",
    "discover_score_sources",
    "build_agent_score_matrix",
    "calculate_disagreement",
    "calculate_agent_correlations",
]

