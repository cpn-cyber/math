"""AI-assisted writing risk evidence extraction and D-S fusion.

Problem 3 Step 26 estimates interpretable AI-assisted writing risk from local
textual evidence, section quality, and Step 25 argument-chain gaps. It does not
call external detectors and does not make authorship, misconduct, or generation
judgements.
"""

from __future__ import annotations

from pathlib import Path
import json
import logging
import math
import re
from typing import Any

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

from modules.appendix3_pipeline import get_problem3_config, load_config, resolve_project_path


LOGGER_NAME = "A_MAGE_R3.problem3.ai_risk_ds_fusion"
DISCLAIMER = "该指标仅表示文本存在 AI辅助写作风险，不构成学术不端判断，也不等同于AI生成判定。"

EVIDENCE_COLUMNS = [
    "template_expression",
    "unsupported_conclusion",
    "data_untraceable",
    "method_result_jump",
]

EVIDENCE_OUTPUT_COLUMNS = [
    "e1_template_expression",
    "e2_unsupported_conclusion",
    "e3_data_untraceable",
    "e4_method_result_jump",
]

DEFAULT_RELIABILITY = {
    "template_expression": 0.75,
    "unsupported_conclusion": 0.80,
    "data_untraceable": 0.85,
    "method_result_jump": 0.85,
}

BODY_SECTIONS = [
    "abstract",
    "problem_statement",
    "problem_analysis",
    "assumptions",
    "model_building",
    "model_solving",
    "results",
    "sensitivity_analysis",
    "error_analysis",
    "model_evaluation",
]

TEMPLATE_WORDS = [
    "本文首先",
    "首先",
    "其次",
    "然后",
    "最后",
    "综上",
    "综上所述",
    "由此可见",
    "可以看出",
    "本文通过",
    "本文建立",
    "本文提出",
    "具有重要意义",
    "具有一定",
    "较好的",
    "较高的",
    "有效提高",
    "合理有效",
    "科学合理",
]

VAGUE_WORDS = [
    "显著",
    "较好",
    "较差",
    "合理",
    "有效",
    "科学",
    "一定",
    "明显",
    "充分",
    "综合",
    "完善",
    "优化",
    "提升",
]

CONCLUSION_WORDS = [
    "结论",
    "说明",
    "表明",
    "可知",
    "得到",
    "最终",
    "因此",
    "建议",
    "方案",
    "策略",
    "综上",
]

SUPPORT_WORDS = [
    "数据",
    "如表",
    "如图",
    "表",
    "图",
    "结果",
    "计算",
    "模型",
    "公式",
    "指标",
    "评价值",
    "匹配度",
    "覆盖率",
    "成本",
    "误差",
]

DATA_WORDS = [
    "数据",
    "指标",
    "样本",
    "统计",
    "区域",
    "资源",
    "设施",
    "人口",
    "健康",
    "环境",
    "经济",
]

DATA_TRACE_WORDS = [
    "来源",
    "年鉴",
    "统计局",
    "政府",
    "附件",
    "单位",
    "年份",
    "预处理",
    "清洗",
    "缺失值",
    "异常值",
    "标准化",
    "归一化",
    "口径",
]

METHOD_WORDS = [
    "模型",
    "算法",
    "公式",
    "目标函数",
    "约束",
    "变量",
    "参数",
    "AHP",
    "TOPSIS",
    "熵权",
    "PSO",
    "DQN",
    "NSGA",
    "DE",
    "求解",
    "迭代",
    "优化",
]

RESULT_WORDS = [
    "结果",
    "得到",
    "输出",
    "求解结果",
    "优化结果",
    "评价结果",
    "数值",
    "方案",
    "最优",
    "预测值",
    "覆盖率",
    "成本",
]

PROCESS_WORDS = [
    "推导",
    "步骤",
    "过程",
    "迭代",
    "计算",
    "代入",
    "求解",
    "参数",
    "约束",
    "公式",
    "流程",
]


def setup_ai_risk_logger(log_path: Path) -> logging.Logger:
    """Configure Step 26 logger."""
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


def _set_chinese_font(logger: logging.Logger) -> None:
    """Use a Chinese-capable font if available."""
    candidates = ["Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC", "Arial Unicode MS"]
    available = {font.name for font in font_manager.fontManager.ttflist}
    for candidate in candidates:
        if candidate in available:
            plt.rcParams["font.sans-serif"] = [candidate]
            plt.rcParams["axes.unicode_minus"] = False
            logger.info("Chinese font selected: %s", candidate)
            return
    plt.rcParams["axes.unicode_minus"] = False
    logger.warning("No Chinese font found; charts may fall back to default glyph substitution.")


def _natural_sort_key(value: Any) -> list[Any]:
    """Natural sort key for paper IDs."""
    text = Path(str(value)).stem
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def _load_json(path: Path) -> dict[str, Any]:
    """Load JSON with UTF-8 tolerance."""
    return json.loads(Path(path).read_text(encoding="utf-8", errors="ignore"))


def _strip_page_markers(text: str) -> str:
    """Remove page markers."""
    return re.sub(r"\[PAGE\s+\d+\]", "", text or "", flags=re.IGNORECASE)


def _effective_chars(text: str) -> int:
    """Count non-whitespace characters without page markers."""
    return len(re.sub(r"\s+", "", _strip_page_markers(text or "")))


def _clip01(value: Any) -> float:
    """Clip numeric values to [0, 1]."""
    try:
        if value is None or pd.isna(value):
            return 0.0
        return float(np.clip(float(value), 0.0, 1.0))
    except (TypeError, ValueError):
        return 0.0


def _count_keywords(text: str, keywords: list[str]) -> int:
    """Count keyword occurrences by substring matching."""
    lower = (text or "").lower()
    return sum(lower.count(str(keyword).lower()) for keyword in keywords if keyword)


def _keyword_hits(text: str, keywords: list[str]) -> list[str]:
    """Return unique hit keywords."""
    lower = (text or "").lower()
    return [keyword for keyword in keywords if keyword and str(keyword).lower() in lower]


def _numeric_count(text: str) -> int:
    """Count numeric evidence tokens."""
    return len(re.findall(r"\d+(?:\.\d+)?%?", text or ""))


def _formula_density_flag(text: str) -> bool:
    """Detect formula-dense paragraphs that should not be scored as prose risk."""
    if _effective_chars(text) < 30:
        return True
    formula_hits = len(re.findall(r"[=<>≤≥∑√]|目标函数|约束|公式|s\.t\.|max|min", text or "", flags=re.IGNORECASE))
    line_count = max(1, len([line for line in (text or "").splitlines() if line.strip()]))
    return formula_hits >= 4 or formula_hits / line_count >= 0.8


def _short_excerpt(text: str, max_chars: int = 160) -> str:
    """Return a short, non-verbatim-heavy excerpt for evidence tracing."""
    compact = re.sub(r"\s+", " ", text or "").strip()
    return compact[:max_chars]


def _split_section_paragraphs(section_text: str, source_section: str) -> list[dict[str, Any]]:
    """Split section text into natural-ish paragraphs with source section tags."""
    section_text = _strip_page_markers(section_text or "")
    raw_blocks = re.split(r"\n\s*\n+", section_text)
    paragraphs: list[dict[str, Any]] = []
    for block in raw_blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        # PDF extraction often lacks blank lines; chunk long line blocks.
        chunks: list[str] = []
        current: list[str] = []
        current_chars = 0
        for line in lines:
            current.append(line)
            current_chars += _effective_chars(line)
            if current_chars >= 240 or len(current) >= 5:
                chunks.append("\n".join(current))
                current = []
                current_chars = 0
        if current:
            chunks.append("\n".join(current))
        for chunk in chunks:
            if _effective_chars(chunk) < 45:
                continue
            paragraphs.append({"source_section": source_section, "paragraph_text": chunk})
    return paragraphs


def split_paper_paragraphs(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Split one paper into scorable paragraphs, excluding references and appendix."""
    sections = payload.get("sections", {}) or {}
    paragraphs: list[dict[str, Any]] = []
    for section_name in BODY_SECTIONS:
        text = str(sections.get(section_name, "") or "")
        if not text.strip():
            continue
        paragraphs.extend(_split_section_paragraphs(text, section_name))
    return paragraphs


def score_paragraph(paragraph: str, source_section: str) -> dict[str, Any] | None:
    """Score one paragraph for the four AI-assisted writing risk evidence types."""
    if _formula_density_flag(paragraph):
        return None
    chars = max(1, _effective_chars(paragraph))
    template_hits = _count_keywords(paragraph, TEMPLATE_WORDS)
    vague_hits = _count_keywords(paragraph, VAGUE_WORDS)
    conclusion_hits = _count_keywords(paragraph, CONCLUSION_WORDS)
    support_hits = _count_keywords(paragraph, SUPPORT_WORDS) + min(_numeric_count(paragraph), 8)
    data_hits = _count_keywords(paragraph, DATA_WORDS)
    trace_hits = _count_keywords(paragraph, DATA_TRACE_WORDS) + min(_numeric_count(paragraph), 6)
    method_hits = _count_keywords(paragraph, METHOD_WORDS)
    result_hits = _count_keywords(paragraph, RESULT_WORDS)
    process_hits = _count_keywords(paragraph, PROCESS_WORDS) + min(_numeric_count(paragraph), 6)

    sentence_starts = re.findall(r"(?:^|[。！？；;]\s*)(本文|首先|其次|然后|最后|综上|因此)", paragraph)
    repeated_start_score = _clip01(len(sentence_starts) / 4)
    concrete_score = _clip01((support_hits + trace_hits + process_hits) / 10)

    template_expression = _clip01(
        0.42 * _clip01(template_hits / 5)
        + 0.28 * _clip01(vague_hits / 8)
        + 0.20 * repeated_start_score
        + 0.10 * (1.0 - concrete_score if vague_hits + template_hits >= 3 else 0.0)
    )

    conclusion_intensity = _clip01(conclusion_hits / 4)
    support_absence = 1.0 - _clip01(support_hits / 5)
    unsupported_conclusion = _clip01(conclusion_intensity * (0.65 * support_absence + 0.35))
    if source_section in {"abstract", "model_evaluation", "results"}:
        unsupported_conclusion = _clip01(unsupported_conclusion + 0.07 * conclusion_intensity)

    data_intensity = _clip01(data_hits / 5)
    trace_absence = 1.0 - _clip01(trace_hits / 4)
    data_untraceable = _clip01(data_intensity * (0.70 * trace_absence + 0.30))

    method_result_intensity = _clip01((method_hits + result_hits) / 7)
    process_absence = 1.0 - _clip01(process_hits / 5)
    method_result_jump = _clip01(method_result_intensity * (0.65 * process_absence + 0.35))
    if source_section == "results" and method_hits + result_hits > 0:
        method_result_jump = _clip01(method_result_jump + 0.06 * process_absence)

    hit_summary = {
        "template": ",".join(_keyword_hits(paragraph, TEMPLATE_WORDS)[:8]),
        "conclusion": ",".join(_keyword_hits(paragraph, CONCLUSION_WORDS)[:8]),
        "data": ",".join(_keyword_hits(paragraph, DATA_WORDS)[:8]),
        "method_result": ",".join((_keyword_hits(paragraph, METHOD_WORDS) + _keyword_hits(paragraph, RESULT_WORDS))[:8]),
    }
    return {
        "template_expression_score": template_expression,
        "unsupported_conclusion_score": unsupported_conclusion,
        "data_untraceable_score": data_untraceable,
        "method_result_jump_score": method_result_jump,
        "paragraph_chars": chars,
        "numeric_count": _numeric_count(paragraph),
        "template_hits": hit_summary["template"],
        "conclusion_hits": hit_summary["conclusion"],
        "data_hits": hit_summary["data"],
        "method_result_hits": hit_summary["method_result"],
        "evidence_excerpt": _short_excerpt(paragraph),
    }


def extract_paragraph_risk_details(sections_dir: Path, logger: logging.Logger) -> pd.DataFrame:
    """Extract paragraph-level risk evidence for all Appendix 3 papers."""
    rows: list[dict[str, Any]] = []
    for section_path in sorted(Path(sections_dir).glob("*.json"), key=_natural_sort_key):
        payload = _load_json(section_path)
        paper_id = str(payload.get("paper_id", section_path.stem))
        filename = str(payload.get("filename", f"{paper_id}.txt"))
        paragraphs = split_paper_paragraphs(payload)
        logger.info("%s raw paragraph candidates=%s", paper_id, len(paragraphs))
        paragraph_index = 0
        skipped_formula = 0
        for item in paragraphs:
            scores = score_paragraph(item["paragraph_text"], item["source_section"])
            if scores is None:
                skipped_formula += 1
                continue
            paragraph_index += 1
            rows.append(
                {
                    "paper_id": paper_id,
                    "filename": filename,
                    "paragraph_id": f"{paper_id}-P{paragraph_index:03d}",
                    "source_section": item["source_section"],
                    **scores,
                }
            )
        logger.info("%s scored_paragraphs=%s skipped_formula_or_short=%s", paper_id, paragraph_index, skipped_formula)
    return pd.DataFrame(rows)


def _load_edge_lookup(edges_path: Path) -> dict[str, dict[str, float]]:
    """Load Step 25 edge scores."""
    edges = pd.read_excel(edges_path)
    lookup: dict[str, dict[str, float]] = {}
    for paper_id, group in edges.groupby(edges["paper_id"].astype(str)):
        lookup[str(paper_id)] = dict(zip(group["edge"].astype(str), group["edge_score"].astype(float)))
    return lookup


def _load_logic_gap_lookup(diagnosis_path: Path) -> dict[str, dict[str, Any]]:
    """Load paper-level logic gap diagnosis."""
    table = pd.read_excel(diagnosis_path)
    return {str(row["paper_id"]): row.to_dict() for _, row in table.iterrows()}


def _aggregate_one(series: pd.Series) -> float:
    """Aggregate paragraph risks with mean and top-risk emphasis."""
    values = pd.to_numeric(series, errors="coerce").dropna().clip(lower=0, upper=1)
    if values.empty:
        return 0.0
    top_n = min(3, len(values))
    top_mean = float(values.sort_values(ascending=False).head(top_n).mean())
    return _clip01(0.62 * float(values.mean()) + 0.38 * top_mean)


def _edge_boost(edge_score: float, threshold: float = 0.70, cap: float = 0.12) -> float:
    """Convert a weak logic edge into a bounded risk boost."""
    if pd.isna(edge_score):
        return 0.0
    weakness = max(0.0, threshold - float(edge_score))
    return min(cap, weakness / threshold * cap)


def aggregate_paper_risk_evidence(
    paragraph_details: pd.DataFrame,
    diagnosis_path: Path,
    edges_path: Path,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Aggregate paragraph-level risk evidence to paper-level evidence."""
    edge_lookup = _load_edge_lookup(edges_path)
    diagnosis_lookup = _load_logic_gap_lookup(diagnosis_path)
    rows: list[dict[str, Any]] = []
    for paper_id, group in paragraph_details.groupby(paragraph_details["paper_id"].astype(str), sort=False):
        group = group.copy()
        filename = str(group["filename"].iloc[0])
        e1_base = _aggregate_one(group["template_expression_score"])
        e2_base = _aggregate_one(group["unsupported_conclusion_score"])
        e3_base = _aggregate_one(group["data_untraceable_score"])
        e4_base = _aggregate_one(group["method_result_jump_score"])

        edges = edge_lookup.get(str(paper_id), {})
        boost_e2 = _edge_boost(edges.get("R_to_C", math.nan), threshold=0.70, cap=0.12)
        boost_e3 = max(
            _edge_boost(edges.get("T_to_D", math.nan), threshold=0.70, cap=0.08),
            _edge_boost(edges.get("D_to_HM", math.nan), threshold=0.70, cap=0.10),
        )
        boost_e4 = _edge_boost(edges.get("HM_to_R", math.nan), threshold=0.70, cap=0.12)

        e1 = e1_base
        e2 = _clip01(e2_base + boost_e2)
        e3 = _clip01(e3_base + boost_e3)
        e4 = _clip01(e4_base + boost_e4)

        risk_cols = [
            "template_expression_score",
            "unsupported_conclusion_score",
            "data_untraceable_score",
            "method_result_jump_score",
        ]
        top = group.assign(overall_risk=group[risk_cols].mean(axis=1)).sort_values("overall_risk", ascending=False).head(2)
        top_evidence = " | ".join(
            f"{row.paragraph_id}[{row.source_section}]:{row.evidence_excerpt}" for row in top.itertuples(index=False)
        )
        diagnosis = diagnosis_lookup.get(str(paper_id), {})
        linked_logic_gap = (
            f"Gamma={float(diagnosis.get('Gamma', math.nan)):.3f}; "
            f"G_logic_gap={float(diagnosis.get('G_logic_gap', math.nan)):.3f}; "
            f"weak_edges R_to_C={edges.get('R_to_C', math.nan):.3f}, "
            f"T_to_D={edges.get('T_to_D', math.nan):.3f}, "
            f"D_to_HM={edges.get('D_to_HM', math.nan):.3f}, "
            f"HM_to_R={edges.get('HM_to_R', math.nan):.3f}"
        )
        note = (
            f"paragraph aggregation mean+top-risk; boosts bounded: "
            f"e2(+{boost_e2:.3f}) from R_to_C, "
            f"e3(+{boost_e3:.3f}) from T_to_D/D_to_HM, "
            f"e4(+{boost_e4:.3f}) from HM_to_R. "
            "Boosts are capped so logic gaps do not directly determine AI-assisted risk."
        )
        rows.append(
            {
                "paper_id": paper_id,
                "filename": filename,
                "e1_template_expression": e1,
                "e2_unsupported_conclusion": e2,
                "e3_data_untraceable": e3,
                "e4_method_result_jump": e4,
                "top_risk_evidence": top_evidence,
                "linked_logic_gap": linked_logic_gap,
                "evidence_note": note,
                "paragraph_count": int(len(group)),
            }
        )
        logger.info(
            "%s evidence aggregated: e1=%.6f e2=%.6f e3=%.6f e4=%.6f boosts=(%.3f,%.3f,%.3f)",
            paper_id,
            e1,
            e2,
            e3,
            e4,
            boost_e2,
            boost_e3,
            boost_e4,
        )
    return pd.DataFrame(rows).sort_values("paper_id", key=lambda s: s.map(_natural_sort_key)).reset_index(drop=True)


def _mass_from_evidence(evidence_value: float, reliability: float) -> dict[str, float]:
    """Build one evidence mass function on {A,H,U}."""
    e = _clip01(evidence_value)
    rho = _clip01(reliability)
    return {
        "A": rho * e,
        "H": rho * (1.0 - e),
        "U": 1.0 - rho,
    }


def _combine_two(m1: dict[str, float], m2: dict[str, float]) -> tuple[dict[str, float], float]:
    """Dempster combination for masses over A/H/U."""
    conflict = m1["A"] * m2["H"] + m1["H"] * m2["A"]
    denom = max(1e-12, 1.0 - conflict)
    combined = {
        "A": (m1["A"] * m2["A"] + m1["A"] * m2["U"] + m1["U"] * m2["A"]) / denom,
        "H": (m1["H"] * m2["H"] + m1["H"] * m2["U"] + m1["U"] * m2["H"]) / denom,
        "U": (m1["U"] * m2["U"]) / denom,
    }
    total = sum(combined.values())
    if total > 0:
        combined = {key: value / total for key, value in combined.items()}
    return combined, conflict


def ds_fuse_one(row: pd.Series, reliability: dict[str, float]) -> dict[str, Any]:
    """Fuse four evidence values with Dempster-Shafer combination."""
    evidence_values = {
        "template_expression": float(row["e1_template_expression"]),
        "unsupported_conclusion": float(row["e2_unsupported_conclusion"]),
        "data_untraceable": float(row["e3_data_untraceable"]),
        "method_result_jump": float(row["e4_method_result_jump"]),
    }
    masses = [
        _mass_from_evidence(evidence_values[name], reliability.get(name, DEFAULT_RELIABILITY[name]))
        for name in EVIDENCE_COLUMNS
    ]
    combined = masses[0]
    conflicts: list[float] = []
    for mass in masses[1:]:
        combined, conflict = _combine_two(combined, mass)
        conflicts.append(conflict)
    bel_a = combined["A"]
    pl_a = combined["A"] + combined["U"]
    betp_a = combined["A"] + 0.5 * combined["U"]
    return {
        "Bel_A": _clip01(bel_a),
        "Pl_A": _clip01(pl_a),
        "BetP_A": _clip01(betp_a),
        "R_AI": _clip01(betp_a),
        "conflict_K_max": max(conflicts) if conflicts else 0.0,
        "conflict_K_mean": float(np.mean(conflicts)) if conflicts else 0.0,
        "m_final_A": combined["A"],
        "m_final_H": combined["H"],
        "m_final_U": combined["U"],
    }


def _risk_level(value: float) -> str:
    """Map R_AI to risk level."""
    if value < 0.30:
        return "低风险"
    if value < 0.60:
        return "中风险"
    return "高风险"


def _main_risk_source(row: pd.Series) -> str:
    """Return dominant evidence source name."""
    values = {
        "模板化表达风险": row["e1_template_expression"],
        "无支撑结论风险": row["e2_unsupported_conclusion"],
        "数据不可追溯风险": row["e3_data_untraceable"],
        "方法到结果跳跃风险": row["e4_method_result_jump"],
    }
    ordered = sorted(values.items(), key=lambda item: float(item[1]), reverse=True)
    return "、".join(f"{name}({float(value):.3f})" for name, value in ordered[:2])


def fuse_ai_risk_table(
    evidence: pd.DataFrame,
    reliability: dict[str, float],
    logger: logging.Logger,
) -> pd.DataFrame:
    """Fuse paper-level evidence and build AI-assisted risk table."""
    rows: list[dict[str, Any]] = []
    for _, row in evidence.iterrows():
        fusion = ds_fuse_one(row, reliability)
        risk_level = _risk_level(fusion["R_AI"])
        source = _main_risk_source(row)
        explanation = (
            f"D-S fusion of four interpretable evidence types; main sources: {source}. "
            f"Final uncertainty mass U={fusion['m_final_U']:.3f}; conflict_K_max={fusion['conflict_K_max']:.3f}."
        )
        conflict_warning = "high_conflict_review" if fusion["conflict_K_max"] >= 0.60 else "normal_conflict"
        if fusion["conflict_K_max"] >= 0.60:
            logger.warning("%s D-S conflict is high: K_max=%.6f", row["paper_id"], fusion["conflict_K_max"])
        else:
            logger.info("%s D-S conflict normal: K_max=%.6f", row["paper_id"], fusion["conflict_K_max"])
        rows.append(
            {
                "paper_id": row["paper_id"],
                "filename": row["filename"],
                "Bel_A": fusion["Bel_A"],
                "Pl_A": fusion["Pl_A"],
                "BetP_A": fusion["BetP_A"],
                "R_AI": fusion["R_AI"],
                "risk_level": risk_level,
                "main_risk_source": source,
                "risk_explanation": explanation,
                "disclaimer": DISCLAIMER,
                "conflict_K_max": fusion["conflict_K_max"],
                "conflict_K_mean": fusion["conflict_K_mean"],
                "uncertainty_mass_U": fusion["m_final_U"],
                "conflict_warning": conflict_warning,
            }
        )
    return pd.DataFrame(rows).sort_values("paper_id", key=lambda s: s.map(_natural_sort_key)).reset_index(drop=True)


def build_disclaimer_table() -> pd.DataFrame:
    """Build required disclaimer table."""
    return pd.DataFrame(
        [
            {
                "item": "AI辅助写作风险边界",
                "required_statement": DISCLAIMER,
                "used_in_report": True,
            },
            {
                "item": "非质量扣分说明",
                "required_statement": "AI辅助写作风险不直接等于论文质量扣分，仅作为后续修改优先级和稳健复评风险项。",
                "used_in_report": True,
            },
            {
                "item": "非外部检测器说明",
                "required_statement": "本步骤未调用外部黑箱检测器，所有证据均来自文本、章节结构、逻辑断层和可解释规则。",
                "used_in_report": True,
            },
        ]
    )


def plot_ai_risk_radar(evidence: pd.DataFrame, output_path: Path, logger: logging.Logger) -> Path:
    """Plot four evidence dimensions as a radar chart."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = ["模板化表达", "无支撑结论", "数据不可追溯", "方法结果跳跃"]
    columns = EVIDENCE_OUTPUT_COLUMNS
    angles = np.linspace(0, 2 * np.pi, len(columns), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8.6, 8.2), subplot_kw={"polar": True})
    for _, row in evidence.iterrows():
        values = [float(row[column]) for column in columns]
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=str(row["paper_id"]))
        ax.fill(angles, values, alpha=0.08)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_title("附件3 AI辅助风险证据雷达图", y=1.08, fontsize=15, fontweight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.18, 1.08))
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    logger.info("Saved chart: %s", output_path)
    return output_path


def plot_ai_risk_bar(fusion: pd.DataFrame, output_path: Path, logger: logging.Logger) -> Path:
    """Plot R_AI by paper."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    colors = {"低风险": "#59A14F", "中风险": "#F28E2B", "高风险": "#E15759"}
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    bars = ax.bar(
        fusion["paper_id"].astype(str),
        fusion["R_AI"].astype(float),
        color=[colors.get(level, "#999999") for level in fusion["risk_level"]],
        edgecolor="#222222",
        linewidth=0.5,
    )
    ax.axhline(0.30, color="#777777", linestyle="--", linewidth=1)
    ax.axhline(0.60, color="#777777", linestyle="--", linewidth=1)
    ax.text(len(fusion) - 0.45, 0.305, "低/中阈值", fontsize=9, va="bottom")
    ax.text(len(fusion) - 0.45, 0.605, "中/高阈值", fontsize=9, va="bottom")
    ax.set_ylim(0, 1)
    ax.set_ylabel("R_AI = BetP_A")
    ax.set_title("附件3 AI辅助写作风险 R_AI", fontsize=15, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, fusion["R_AI"]):
        ax.text(bar.get_x() + bar.get_width() / 2, float(value) + 0.025, f"{float(value):.3f}", ha="center")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    logger.info("Saved chart: %s", output_path)
    return output_path


def plot_ai_evidence_stack(evidence: pd.DataFrame, output_path: Path, logger: logging.Logger) -> Path:
    """Plot stacked evidence contributions."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = ["模板化表达", "无支撑结论", "数据不可追溯", "方法结果跳跃"]
    columns = EVIDENCE_OUTPUT_COLUMNS
    x = np.arange(len(evidence))
    bottom = np.zeros(len(evidence))
    colors = ["#4E79A7", "#F28E2B", "#59A14F", "#E15759"]
    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    for label, column, color in zip(labels, columns, colors):
        values = evidence[column].astype(float).to_numpy()
        ax.bar(x, values, bottom=bottom, label=label, color=color, edgecolor="white", linewidth=0.4)
        bottom += values
    ax.set_xticks(x)
    ax.set_xticklabels(evidence["paper_id"].astype(str))
    ax.set_ylabel("Evidence score stack")
    ax.set_title("附件3 AI辅助风险四类证据堆叠图", fontsize=15, fontweight="bold")
    ax.legend(ncol=2)
    ax.grid(axis="y", alpha=0.20)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    logger.info("Saved chart: %s", output_path)
    return output_path


def run_ai_risk_ds_fusion(config_path: Path | None = None) -> dict[str, Any]:
    """Run Step 26 and save all required outputs."""
    try:
        _ = load_config(config_path)
    except RuntimeError:
        pass
    problem3_config = get_problem3_config(config_path)
    intermediate_dir = resolve_project_path(problem3_config["appendix3_intermediate_dir"])
    sections_dir = intermediate_dir / "sections_refined"
    if not sections_dir.is_dir() or not list(sections_dir.glob("*.json")):
        sections_dir = intermediate_dir / "sections"
    tables_dir = resolve_project_path(problem3_config["output_tables_dir"])
    charts_dir = resolve_project_path(problem3_config["output_charts_dir"])
    logs_dir = resolve_project_path(problem3_config["output_logs_dir"])
    tables_dir.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    log_path = logs_dir / "ai_risk_ds_fusion.log"
    logger = setup_ai_risk_logger(log_path)
    _set_chinese_font(logger)
    logger.info("Starting Step 26 AI-assisted writing risk D-S fusion.")
    logger.info("No external detector is used; outputs are not misconduct or authorship judgements.")
    logger.info("Using sections_dir=%s", sections_dir)

    reliability = dict(DEFAULT_RELIABILITY)
    reliability.update(dict(problem3_config.get("ai_risk_evidence_reliability", {}) or {}))
    logger.info("Evidence reliability: %s", reliability)

    paragraph_details = extract_paragraph_risk_details(sections_dir, logger)
    diagnosis_path = tables_dir / "logic_gap_diagnosis.xlsx"
    edges_path = tables_dir / "argument_chain_edges.xlsx"
    evidence = aggregate_paper_risk_evidence(paragraph_details, diagnosis_path, edges_path, logger)
    fusion = fuse_ai_risk_table(evidence, reliability, logger)
    disclaimer = build_disclaimer_table()

    paths = {
        "paragraph_details": tables_dir / "ai_risk_paragraph_details.xlsx",
        "evidence": tables_dir / "ai_risk_evidence.xlsx",
        "fusion": tables_dir / "ai_risk_ds_fusion.xlsx",
        "disclaimer": tables_dir / "ai_risk_disclaimer.xlsx",
        "radar": charts_dir / "ai_risk_radar.png",
        "bar": charts_dir / "ai_risk_bar.png",
        "stack": charts_dir / "ai_evidence_stack.png",
        "log": log_path,
    }
    paragraph_details.to_excel(paths["paragraph_details"], index=False)
    evidence.to_excel(paths["evidence"], index=False)
    fusion.to_excel(paths["fusion"], index=False)
    disclaimer.to_excel(paths["disclaimer"], index=False)
    plot_ai_risk_radar(evidence, paths["radar"], logger)
    plot_ai_risk_bar(fusion, paths["bar"], logger)
    plot_ai_evidence_stack(evidence, paths["stack"], logger)

    high_conflict = fusion.loc[fusion["conflict_K_max"].ge(0.60), ["paper_id", "conflict_K_max"]]
    if high_conflict.empty:
        logger.info("No high D-S conflict detected. max_conflict=%.6f", float(fusion["conflict_K_max"].max()))
    else:
        logger.warning("High D-S conflict detected: %s", high_conflict.to_dict(orient="records"))
    logger.info("Required disclaimer written to fusion and disclaimer tables: %s", DISCLAIMER)
    logger.info("Finished Step 26.")

    return {
        "paragraph_details": paragraph_details,
        "evidence": evidence,
        "fusion": fusion,
        "disclaimer": disclaimer,
        "paths": paths,
        "high_conflict": high_conflict,
    }


def extract_ai_risk_evidence(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Backward-compatible wrapper for Step 26 evidence extraction."""
    return run_ai_risk_ds_fusion(*args, **kwargs)


def fuse_ai_risk_evidence(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Backward-compatible wrapper for Step 26 D-S fusion."""
    return run_ai_risk_ds_fusion(*args, **kwargs)
