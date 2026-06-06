"""Five-element argument chain diagnostics for Problem 3 Step 25.

The diagnostic chain is T -> D -> HM -> R -> C -> T:
T  = task and subquestion goals,
D  = data, source, preprocessing, and measurement scope,
HM = hypotheses, model, variables, formulas, constraints, and algorithms,
R  = results, numerical outputs, figures/tables, and optimization outputs,
C  = conclusions, plans, and answers to task goals.

This step only diagnoses logic closure and evidence gaps. It does not evaluate
AI-writing risk, reviewer-agent disagreement, revision actions, or post-revision
scores.
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

try:  # pragma: no cover - fallback is exercised only when sklearn is absent.
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:  # pragma: no cover
    TfidfVectorizer = None
    cosine_similarity = None


LOGGER_NAME = "A_MAGE_R3.problem3.argument_chain"

NODE_TYPES = ["T", "D", "HM", "R", "C"]
EDGE_ORDER = ["T_to_D", "D_to_HM", "HM_to_R", "R_to_C", "C_to_T"]
EDGE_WEIGHTS = {
    "T_to_D": 0.20,
    "D_to_HM": 0.25,
    "HM_to_R": 0.25,
    "R_to_C": 0.20,
    "C_to_T": 0.10,
}

NODE_SECTION_MAP = {
    "T": ["problem_statement", "problem_analysis", "abstract"],
    "D": ["problem_statement", "problem_analysis", "model_building", "model_solving", "results", "appendix"],
    "HM": ["assumptions", "symbols", "model_building", "model_solving"],
    "R": ["results"],
    "C": ["model_evaluation", "results", "abstract"],
}

NODE_KEYWORDS = {
    "T": [
        "问题一",
        "问题二",
        "问题三",
        "任务",
        "目标",
        "要求",
        "评价",
        "优化",
        "预测",
        "选址",
        "资源",
        "设施",
        "公平性",
        "覆盖率",
        "成本",
    ],
    "D": [
        "数据",
        "数据来源",
        "附件",
        "指标",
        "样本",
        "预处理",
        "清洗",
        "缺失值",
        "异常值",
        "统计",
        "区域",
        "设施",
        "资源",
        "人口",
        "坐标",
    ],
    "HM": [
        "假设",
        "模型",
        "变量",
        "公式",
        "约束",
        "目标函数",
        "算法",
        "求解",
        "参数",
        "AHP",
        "TOPSIS",
        "PSO",
        "DQN",
        "NSGA",
        "DE",
    ],
    "R": [
        "结果",
        "得到",
        "计算结果",
        "优化结果",
        "评价结果",
        "如表",
        "如图",
        "数值",
        "方案",
        "覆盖率",
        "成本",
        "最优",
        "预测值",
    ],
    "C": [
        "结论",
        "总结",
        "建议",
        "方案",
        "回答",
        "表明",
        "说明",
        "可知",
        "最终",
        "策略",
        "推广",
        "改进",
    ],
}

EDGE_BRIDGE_KEYWORDS = {
    "T_to_D": ["数据", "指标", "来源", "附件", "样本", "资源", "设施", "人口", "区域"],
    "D_to_HM": ["变量", "指标", "参数", "模型", "权重", "矩阵", "约束", "输入", "数据"],
    "HM_to_R": ["求解", "结果", "迭代", "输出", "方案", "最优", "计算", "评价值", "覆盖率", "成本"],
    "R_to_C": ["结果", "结论", "表明", "说明", "可知", "方案", "建议", "最终"],
    "C_to_T": ["问题一", "问题二", "问题三", "任务", "目标", "评价", "优化", "预测", "回答"],
}

RELATED_LOW_FEATURES = {
    "T_to_D": ["I05", "section_coverage", "task_coverage"],
    "D_to_HM": ["I06", "I11", "method_fit"],
    "HM_to_R": ["I12", "I14", "method_fit", "objective_constraint_completeness"],
    "R_to_C": ["I08", "I15", "conclusion_echo_rate", "figure_table_explanation_rate"],
    "C_to_T": ["I05", "I08", "task_coverage", "conclusion_echo_rate"],
}

VALIDATION_SHORTFALL_SECTIONS = {
    "sensitivity_analysis": ("result_validation_shortfall", "缺少灵敏度/稳健性分析，后续应补充参数扰动检验"),
    "error_analysis": ("result_validation_shortfall", "缺少误差分析，后续应补充误差来源、误差量化或局限说明"),
    "model_evaluation": ("model_evaluation_shortfall", "缺少模型评价，后续应补充优缺点、适用条件和改进方向"),
}


def setup_argument_chain_logger(log_path: Path) -> logging.Logger:
    """Configure Step 25 logger."""
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


def _natural_sort_key(value: Any) -> list[Any]:
    """Natural sort key for paths and paper IDs."""
    text = Path(str(value)).stem
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def _set_chinese_font() -> None:
    """Use Chinese-capable fonts when available."""
    candidates = ["Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC", "Arial Unicode MS"]
    available = {font.name for font in font_manager.fontManager.ttflist}
    for candidate in candidates:
        if candidate in available:
            plt.rcParams["font.sans-serif"] = [candidate]
            break
    plt.rcParams["axes.unicode_minus"] = False


def _read_text(path: Path) -> str:
    """Read UTF-8 text defensively."""
    return Path(path).read_text(encoding="utf-8", errors="ignore") if Path(path).exists() else ""


def _load_json(path: Path) -> dict[str, Any]:
    """Load section JSON."""
    return json.loads(Path(path).read_text(encoding="utf-8", errors="ignore"))


def _strip_page_markers(text: str) -> str:
    """Remove page markers."""
    return re.sub(r"\[PAGE\s+\d+\]", "", text or "", flags=re.IGNORECASE)


def _effective_chars(text: str) -> int:
    """Count non-whitespace characters after page markers are removed."""
    return len(re.sub(r"\s+", "", _strip_page_markers(text or "")))


def _clip01(value: Any) -> float:
    """Clip a numeric value into [0, 1]."""
    if value is None or pd.isna(value):
        return 0.0
    try:
        return float(np.clip(float(value), 0.0, 1.0))
    except (TypeError, ValueError):
        return 0.0


def _section(payload: dict[str, Any], section_name: str) -> str:
    """Return confirmed section text."""
    sections = payload.get("sections", {}) or {}
    return str(sections.get(section_name, "") or "")


def _keyword_hits(text: str, keywords: list[str]) -> list[str]:
    """Return keywords found in text, preserving configured order."""
    lower = (text or "").lower()
    hits: list[str] = []
    for keyword in keywords:
        if keyword and str(keyword).lower() in lower:
            hits.append(keyword)
    return list(dict.fromkeys(hits))


def _keyword_score(text: str, keywords: list[str], target_hits: int = 6) -> float:
    """Calculate keyword coverage score."""
    if not keywords:
        return 0.0
    return _clip01(len(_keyword_hits(text, keywords)) / max(1, min(target_hits, len(keywords))))


def _snippet(text: str, keywords: list[str], max_chars: int = 260) -> str:
    """Extract a short evidence snippet around the first matched keyword."""
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return ""
    lower = text.lower()
    best_index = None
    for keyword in keywords:
        idx = lower.find(str(keyword).lower())
        if idx >= 0:
            best_index = idx
            break
    if best_index is None:
        return text[:max_chars]
    start = max(0, best_index - max_chars // 3)
    end = min(len(text), start + max_chars)
    return text[start:end]


def _page_marker_for_evidence(full_text: str, evidence: str) -> str:
    """Find the page marker nearest the evidence snippet if possible."""
    if not evidence:
        return ""
    marker_match = re.search(r"\[PAGE\s+\d+\]", evidence, flags=re.IGNORECASE)
    if marker_match:
        return marker_match.group(0)
    probe = re.sub(r"\s+", "", _strip_page_markers(evidence))[:40]
    if not probe:
        return ""
    compact_full = re.sub(r"\s+", "", full_text or "")
    compact_idx = compact_full.find(probe)
    if compact_idx < 0:
        return ""

    # Approximate by scanning original text prefix with a proportional index.
    ratio = compact_idx / max(1, len(compact_full))
    approx_original_idx = int(ratio * len(full_text))
    prefix = full_text[:approx_original_idx]
    markers = re.findall(r"\[PAGE\s+\d+\]", prefix, flags=re.IGNORECASE)
    return markers[-1] if markers else ""


def _node_source_text(payload: dict[str, Any], node_type: str, full_text: str) -> tuple[str, list[str]]:
    """Collect source text and section names for one node type."""
    source_sections = NODE_SECTION_MAP[node_type]
    parts: list[str] = []
    used_sections: list[str] = []
    for section_name in source_sections:
        text = _section(payload, section_name)
        if _effective_chars(text) > 0:
            parts.append(text)
            used_sections.append(section_name)

    if node_type == "C":
        conclusion_like = _extract_conclusion_like(full_text)
        if conclusion_like and "conclusion_like" not in used_sections:
            parts.append(conclusion_like)
            used_sections.append("conclusion_like")

    return "\n".join(parts), used_sections


def _extract_conclusion_like(full_text: str) -> str:
    """Extract conclusion-like tail evidence without creating a formal section."""
    lines = (full_text or "").splitlines()
    conclusion_keywords = ["结论", "总结", "建议", "策略", "回答", "最终方案", "主要结论"]
    for idx, line in enumerate(lines):
        compact = re.sub(r"\s+", "", line)
        if len(compact) <= 50 and any(keyword in compact for keyword in conclusion_keywords):
            chunk = "\n".join(lines[idx : min(len(lines), idx + 30)]).strip()
            if _effective_chars(chunk) >= 80:
                return chunk
    tail = "\n".join(lines[-45:]).strip()
    if any(keyword in tail for keyword in conclusion_keywords) and _effective_chars(tail) >= 80:
        return tail
    return ""


def _node_strength(
    node_type: str,
    text: str,
    used_sections: list[str],
    payload: dict[str, Any],
) -> tuple[float, list[str], str, str]:
    """Calculate node strength and textual notes."""
    keywords = NODE_KEYWORDS[node_type]
    hits = _keyword_hits(text, keywords)
    keyword_component = _keyword_score(text, keywords, target_hits=6)
    length_component = _clip01(_effective_chars(text) / 900)
    section_component = _clip01(len(used_sections) / max(1, len(NODE_SECTION_MAP[node_type])))

    if node_type == "R" and _effective_chars(_section(payload, "results")) > 0:
        section_component = max(section_component, 0.8)
    if node_type == "C" and "model_evaluation" in used_sections:
        section_component = max(section_component, 0.7)

    score = _clip01(0.45 * keyword_component + 0.25 * length_component + 0.30 * section_component)
    if _effective_chars(text) == 0:
        score = 0.0

    if score >= 0.65:
        confidence = "high"
    elif score >= 0.35:
        confidence = "medium"
    else:
        confidence = "low"

    note_parts: list[str] = []
    if _effective_chars(text) == 0:
        note_parts.append("no evidence text found")
    if score < 0.35:
        note_parts.append("weak_node")
    if node_type == "R":
        result_conf = (payload.get("confidence_flags") or {}).get("results", "")
        if result_conf:
            note_parts.append(f"results_confidence={result_conf}")
    missing_sections = set(payload.get("missing_sections") or [])
    if node_type == "C" and "model_evaluation" in missing_sections:
        note_parts.append("model_evaluation missing weakens conclusion node")
    if node_type == "R" and "results" in missing_sections:
        note_parts.append("results missing weakens result node")

    return score, hits, confidence, "; ".join(note_parts) or "evidence found"


def extract_nodes_for_paper(section_path: Path, text_dir: Path) -> list[dict[str, Any]]:
    """Extract T/D/HM/R/C nodes for one Appendix 3 paper."""
    payload = _load_json(section_path)
    paper_id = str(payload.get("paper_id", Path(section_path).stem))
    filename = str(payload.get("filename", f"{paper_id}.txt"))
    full_text = _read_text(Path(text_dir) / filename)

    rows: list[dict[str, Any]] = []
    for node_type in NODE_TYPES:
        source_text, source_sections = _node_source_text(payload, node_type, full_text)
        score, hits, confidence, note = _node_strength(node_type, source_text, source_sections, payload)
        evidence_text = _snippet(source_text, hits or NODE_KEYWORDS[node_type])
        rows.append(
            {
                "paper_id": paper_id,
                "filename": filename,
                "node_type": node_type,
                "node_found": score > 0,
                "node_strength": score,
                "source_section": ",".join(source_sections) if source_sections else "",
                "page_marker": _page_marker_for_evidence(full_text, evidence_text),
                "evidence_text": evidence_text,
                "keyword_hits": ",".join(hits),
                "confidence": confidence,
                "evidence_note": note,
            }
        )
    return rows


def _tfidf_similarity(text_a: str, text_b: str) -> float:
    """Calculate TF-IDF cosine similarity, with a token-overlap fallback."""
    text_a = _strip_page_markers(text_a or "")
    text_b = _strip_page_markers(text_b or "")
    if _effective_chars(text_a) == 0 or _effective_chars(text_b) == 0:
        return 0.0
    if TfidfVectorizer is not None and cosine_similarity is not None:
        try:
            vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=1)
            matrix = vectorizer.fit_transform([text_a, text_b])
            return _clip01(float(cosine_similarity(matrix[0], matrix[1])[0, 0]))
        except Exception:
            pass
    set_a = set(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]+", text_a))
    set_b = set(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]+", text_b))
    if not set_a or not set_b:
        return 0.0
    return _clip01(len(set_a & set_b) / len(set_a | set_b))


def _shared_keyword_score(source_text: str, target_text: str, keywords: list[str]) -> float:
    """Score whether bridge keywords appear in both sides or at least target."""
    source_hits = set(_keyword_hits(source_text, keywords))
    target_hits = set(_keyword_hits(target_text, keywords))
    if not keywords:
        return 0.0
    both = len(source_hits & target_hits)
    target_only = len(target_hits)
    return _clip01(0.65 * both / max(1, min(len(keywords), 5)) + 0.35 * target_only / max(1, min(len(keywords), 5)))


def _edge_rule_score(edge: str, nodes: dict[str, dict[str, Any]], payload: dict[str, Any]) -> tuple[float, str, str]:
    """Calculate edge-specific rule score, support text, and gap reason."""
    source_node, target_node = edge.split("_to_")
    source_text = str(nodes[source_node].get("evidence_text", ""))
    target_text = str(nodes[target_node].get("evidence_text", ""))
    combined_sections = payload.get("sections", {}) or {}
    all_confirmed = "\n".join(str(value) for value in combined_sections.values())

    bridge_score = _shared_keyword_score(source_text + "\n" + all_confirmed, target_text, EDGE_BRIDGE_KEYWORDS[edge])
    support = []
    reason = []

    if edge == "T_to_D":
        data_hits = _keyword_hits(all_confirmed, EDGE_BRIDGE_KEYWORDS[edge])
        task_hits = _keyword_hits(source_text, NODE_KEYWORDS["T"])
        rule = _clip01(0.55 * bridge_score + 0.25 * (len(data_hits) > 0) + 0.20 * _clip01(len(task_hits) / 4))
        support.append(f"task_hits={','.join(task_hits[:5]) or 'none'}; data_hits={','.join(data_hits[:5]) or 'none'}")
        if not data_hits:
            reason.append("task description has weak corresponding data/source evidence")
    elif edge == "D_to_HM":
        hm_text = _section(payload, "model_building") + "\n" + _section(payload, "model_solving")
        data_in_model = _keyword_hits(hm_text, EDGE_BRIDGE_KEYWORDS[edge])
        symbol_exists = _effective_chars(_section(payload, "symbols")) > 0
        rule = _clip01(0.50 * bridge_score + 0.30 * _clip01(len(data_in_model) / 5) + 0.20 * float(symbol_exists))
        support.append(f"data/model bridge={','.join(data_in_model[:6]) or 'none'}; symbols={symbol_exists}")
        if len(data_in_model) < 2:
            reason.append("data variables are weakly carried into model/hypothesis text")
    elif edge == "HM_to_R":
        hm_text = _section(payload, "model_building") + "\n" + _section(payload, "model_solving")
        result_text = _section(payload, "results")
        algorithm_hits = _keyword_hits(hm_text, ["算法", "求解", "PSO", "DQN", "NSGA", "DE", "迭代", "优化", "目标函数", "约束"])
        result_hits = _keyword_hits(result_text, EDGE_BRIDGE_KEYWORDS[edge])
        rule = _clip01(0.45 * bridge_score + 0.30 * _clip01(len(algorithm_hits) / 5) + 0.25 * _clip01(len(result_hits) / 5))
        support.append(f"algorithm_hits={','.join(algorithm_hits[:6]) or 'none'}; result_hits={','.join(result_hits[:6]) or 'none'}")
        if not result_hits:
            reason.append("model output is weakly traceable to numerical/result evidence")
    elif edge == "R_to_C":
        result_text = _section(payload, "results")
        conclusion_text = _section(payload, "model_evaluation") + "\n" + _extract_conclusion_like("\n".join(str(v) for v in combined_sections.values()))
        result_hits = _keyword_hits(result_text, NODE_KEYWORDS["R"])
        conclusion_hits = _keyword_hits(conclusion_text, EDGE_BRIDGE_KEYWORDS[edge])
        rule = _clip01(0.45 * bridge_score + 0.25 * _clip01(len(result_hits) / 6) + 0.30 * _clip01(len(conclusion_hits) / 5))
        support.append(f"result_hits={','.join(result_hits[:6]) or 'none'}; conclusion_hits={','.join(conclusion_hits[:6]) or 'none'}")
        if not conclusion_hits:
            reason.append("conclusion or recommendation does not strongly cite/result-support outputs")
    else:  # C_to_T
        conclusion_text = _section(payload, "model_evaluation") + "\n" + _extract_conclusion_like("\n".join(str(v) for v in combined_sections.values()))
        task_text = _section(payload, "problem_statement") + "\n" + _section(payload, "problem_analysis")
        conclusion_task_hits = _keyword_hits(conclusion_text, EDGE_BRIDGE_KEYWORDS[edge])
        task_hits = _keyword_hits(task_text, NODE_KEYWORDS["T"])
        rule = _clip01(0.45 * bridge_score + 0.30 * _clip01(len(conclusion_task_hits) / 4) + 0.25 * _clip01(len(task_hits) / 4))
        support.append(f"conclusion_task_hits={','.join(conclusion_task_hits[:5]) or 'none'}; task_hits={','.join(task_hits[:5]) or 'none'}")
        if len(conclusion_task_hits) < 2:
            reason.append("conclusion weakly returns to subquestion/task targets")

    return rule, " | ".join(support), "; ".join(reason) or "rule evidence available"


def calculate_edges_for_paper(
    payload: dict[str, Any],
    node_rows: list[dict[str, Any]],
    low_feature_lookup: dict[str, list[str]],
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Calculate five edge closure scores for one paper."""
    nodes = {row["node_type"]: row for row in node_rows}
    paper_id = str(node_rows[0]["paper_id"])
    filename = str(node_rows[0]["filename"])
    edge_rows: list[dict[str, Any]] = []
    scores: dict[str, float] = {}

    for edge in EDGE_ORDER:
        source_node, target_node = edge.split("_to_")
        source_text = str(nodes[source_node].get("evidence_text", ""))
        target_text = str(nodes[target_node].get("evidence_text", ""))
        node_component = (float(nodes[source_node]["node_strength"]) + float(nodes[target_node]["node_strength"])) / 2
        sim = _tfidf_similarity(source_text, target_text)
        rule, support, rule_reason = _edge_rule_score(edge, nodes, payload)
        edge_score = _clip01(0.35 * node_component + 0.25 * sim + 0.40 * rule)
        related_low = [
            feature
            for feature in low_feature_lookup.get(paper_id, [])
            if any(str(feature).startswith(prefix) or str(feature) == prefix for prefix in RELATED_LOW_FEATURES[edge])
        ]
        gap_reason = rule_reason
        if related_low:
            gap_reason += f"; related low features: {','.join(related_low[:5])}"
        if float(nodes[source_node]["node_strength"]) < 0.35 or float(nodes[target_node]["node_strength"]) < 0.35:
            gap_reason += "; source/target node is weak"
        edge_rows.append(
            {
                "paper_id": paper_id,
                "filename": filename,
                "edge": edge,
                "source_node": source_node,
                "target_node": target_node,
                "edge_score": edge_score,
                "edge_weight": EDGE_WEIGHTS[edge],
                "weighted_score": edge_score * EDGE_WEIGHTS[edge],
                "major_gap_flag": edge_score < 0.35,
                "evidence_support": f"node_component={node_component:.3f}; tfidf={sim:.3f}; rule={rule:.3f}; {support}",
                "gap_reason": gap_reason,
            }
        )
        scores[edge] = edge_score
    return edge_rows, scores


def _load_low_feature_lookup(low_feature_path: Path) -> dict[str, list[str]]:
    """Load Step 24 low-feature report by paper."""
    if not Path(low_feature_path).exists():
        return {}
    table = pd.read_excel(low_feature_path)
    lookup: dict[str, list[str]] = {}
    for paper_id, group in table.groupby(table["paper_id"].astype(str)):
        lookup[str(paper_id)] = group["feature_name"].astype(str).tolist()
    return lookup


def _load_section_audit(section_audit_path: Path) -> pd.DataFrame:
    """Load Step 23B section quality audit if available."""
    if not Path(section_audit_path).exists():
        return pd.DataFrame()
    return pd.read_excel(section_audit_path)


def _main_weak_nodes(node_rows: list[dict[str, Any]]) -> str:
    """Summarize weak nodes."""
    weak = [f"{row['node_type']}:{row['node_strength']:.3f}" for row in node_rows if float(row["node_strength"]) < 0.45]
    return ",".join(weak) if weak else "none"


def _diagnosis_summary(
    paper_id: str,
    gamma: float,
    gap: float,
    major_edges: list[str],
    weak_nodes: str,
    current_eval_lookup: dict[str, dict[str, Any]],
) -> str:
    """Write a concise paper-level logic diagnosis."""
    baseline = current_eval_lookup.get(paper_id, {})
    grade = baseline.get("current_grade", "unknown")
    q_cur = baseline.get("Q_cur_baseline", math.nan)
    if major_edges:
        gap_text = f"major gaps on {','.join(major_edges)}"
    else:
        gap_text = "no edge below 0.35; gaps are moderate or mild"
    return (
        f"Q_cur={q_cur:.3f}, grade={grade}; Gamma={gamma:.3f}, G_logic_gap={gap:.3f}; "
        f"{gap_text}; weak_nodes={weak_nodes}."
    )


def _recommended_focus(major_edges: list[str], weak_nodes: str, missing_evidence: list[str]) -> str:
    """Create diagnosis focus text without prescribing final revision actions."""
    focuses: list[str] = []
    if major_edges:
        focuses.append("priority edges: " + ",".join(major_edges))
    if weak_nodes != "none":
        focuses.append("weak nodes: " + weak_nodes)
    if missing_evidence:
        focuses.append("missing evidence: " + ",".join(missing_evidence[:4]))
    return "; ".join(focuses) if focuses else "maintain chain closure; later steps can inspect AI-risk and revision priorities"


def build_diagnosis_tables(
    nodes_table: pd.DataFrame,
    edges_table: pd.DataFrame,
    section_audit: pd.DataFrame,
    current_eval: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build paper-level diagnosis and evidence gap tables."""
    current_lookup = {
        str(row["paper_id"]): row.to_dict()
        for _, row in current_eval.iterrows()
    }
    audit_lookup = {
        str(row["paper_id"]): row.to_dict()
        for _, row in section_audit.iterrows()
    } if not section_audit.empty and "paper_id" in section_audit.columns else {}

    diagnosis_rows: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []

    for paper_id, group in edges_table.groupby(edges_table["paper_id"].astype(str), sort=False):
        group = group.copy()
        filename = str(group["filename"].iloc[0])
        score_lookup = dict(zip(group["edge"], group["edge_score"]))
        weighted_score = float(group["weighted_score"].sum())
        total_weight = float(group["edge_weight"].sum())
        gamma = weighted_score / total_weight if total_weight > 0 else 0.0
        gap = 1.0 - gamma
        major_edges = group.loc[group["major_gap_flag"], "edge"].astype(str).tolist()

        node_rows = nodes_table.loc[nodes_table["paper_id"].astype(str) == str(paper_id)].to_dict(orient="records")
        weak_nodes = _main_weak_nodes(node_rows)

        audit = audit_lookup.get(str(paper_id), {})
        missing_sections = [
            item.strip()
            for item in str(audit.get("missing_sections", "") or "").split(",")
            if item.strip() and item.strip().lower() != "nan"
        ]
        recognition_failed = [
            item.strip()
            for item in str(audit.get("recognition_failed_sections", "") or "").split(",")
            if item.strip() and item.strip().lower() != "nan"
        ]
        missing_evidence = list(dict.fromkeys(missing_sections + recognition_failed))

        for _, edge_row in group.iterrows():
            if bool(edge_row["major_gap_flag"]):
                evidence_rows.append(
                    {
                        "paper_id": paper_id,
                        "filename": filename,
                        "gap_edge": edge_row["edge"],
                        "evidence_missing_type": "major_logic_gap",
                        "evidence_text_or_missing_reason": edge_row["gap_reason"],
                        "related_low_feature": ",".join(
                            feature
                            for feature in RELATED_LOW_FEATURES.get(str(edge_row["edge"]), [])
                            if feature in str(edge_row["gap_reason"])
                        ),
                        "later_action_hint": "later Step28 may prioritize strengthening this argument-link evidence",
                    }
                )

        for section_name in missing_evidence:
            missing_type, hint = VALIDATION_SHORTFALL_SECTIONS.get(
                section_name,
                ("section_evidence_shortfall", "later Step28 may consider adding explicit section evidence"),
            )
            related = "I16" if section_name == "sensitivity_analysis" else "I17" if section_name == "error_analysis" else "I13"
            evidence_rows.append(
                {
                    "paper_id": paper_id,
                    "filename": filename,
                    "gap_edge": "HM_to_R/R_to_C",
                    "evidence_missing_type": missing_type,
                    "evidence_text_or_missing_reason": f"{section_name} missing or not independently identifiable",
                    "related_low_feature": related,
                    "later_action_hint": hint,
                }
            )

        diagnosis_rows.append(
            {
                "paper_id": paper_id,
                "filename": filename,
                "s_TD": score_lookup.get("T_to_D", 0.0),
                "s_DHM": score_lookup.get("D_to_HM", 0.0),
                "s_HMR": score_lookup.get("HM_to_R", 0.0),
                "s_RC": score_lookup.get("R_to_C", 0.0),
                "s_CT": score_lookup.get("C_to_T", 0.0),
                "Gamma": gamma,
                "G_logic_gap": gap,
                "major_gap_edges": ",".join(major_edges) if major_edges else "none",
                "main_weak_nodes": weak_nodes,
                "main_evidence_missing": ",".join(missing_evidence) if missing_evidence else "none",
                "diagnosis_summary": _diagnosis_summary(paper_id, gamma, gap, major_edges, weak_nodes, current_lookup),
                "recommended_diagnosis_focus": _recommended_focus(major_edges, weak_nodes, missing_evidence),
            }
        )

    diagnosis = pd.DataFrame(diagnosis_rows).sort_values("paper_id", key=lambda s: s.map(lambda v: "|".join(map(str, _natural_sort_key(v))))).reset_index(drop=True)
    evidence = pd.DataFrame(
        evidence_rows,
        columns=[
            "paper_id",
            "filename",
            "gap_edge",
            "evidence_missing_type",
            "evidence_text_or_missing_reason",
            "related_low_feature",
            "later_action_hint",
        ],
    )
    return diagnosis, evidence


def plot_argument_chain_heatmap(diagnosis: pd.DataFrame, chart_path: Path) -> Path:
    """Plot paper by edge closure heatmap."""
    _set_chinese_font()
    chart_path = Path(chart_path)
    chart_path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["s_TD", "s_DHM", "s_HMR", "s_RC", "s_CT"]
    data = diagnosis.set_index("paper_id")[columns].astype(float)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    image = ax.imshow(data.to_numpy(), cmap="YlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels(["T-D", "D-HM", "HM-R", "R-C", "C-T"])
    ax.set_yticks(range(len(data.index)))
    ax.set_yticklabels(data.index)
    ax.set_title("Appendix 3 Argument Chain Closure Heatmap")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(chart_path, dpi=180)
    plt.close(fig)
    return chart_path


def plot_logic_gap_bar(diagnosis: pd.DataFrame, chart_path: Path) -> Path:
    """Plot logic gap strength for each paper."""
    _set_chinese_font()
    chart_path = Path(chart_path)
    chart_path.parent.mkdir(parents=True, exist_ok=True)
    table = diagnosis.sort_values("G_logic_gap", ascending=False)

    fig, ax = plt.subplots(figsize=(8.5, 5))
    bars = ax.bar(table["paper_id"], table["G_logic_gap"], color="#E15759", edgecolor="#222222", linewidth=0.4)
    ax.set_ylim(0, 1)
    ax.set_ylabel("G_logic_gap")
    ax.set_title("Appendix 3 Logic Gap Strength")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, table["G_logic_gap"]):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.2f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(chart_path, dpi=180)
    plt.close(fig)
    return chart_path


def plot_logic_chain_radar(diagnosis: pd.DataFrame, chart_path: Path) -> Path:
    """Plot five edge scores as a radar chart."""
    _set_chinese_font()
    chart_path = Path(chart_path)
    chart_path.parent.mkdir(parents=True, exist_ok=True)
    labels = ["T-D", "D-HM", "HM-R", "R-C", "C-T"]
    columns = ["s_TD", "s_DHM", "s_HMR", "s_RC", "s_CT"]
    angles = np.linspace(0, 2 * np.pi, len(columns), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8.5, 8.5), subplot_kw={"polar": True})
    for _, row in diagnosis.iterrows():
        values = [float(row[column]) for column in columns]
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=str(row["paper_id"]))
        ax.fill(angles, values, alpha=0.08)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylim(0, 1)
    ax.set_title("Appendix 3 Logic Chain Radar", y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.18, 1.08))
    fig.tight_layout()
    fig.savefig(chart_path, dpi=180)
    plt.close(fig)
    return chart_path


def run_argument_chain_diagnosis(config_path: Path | None = None) -> dict[str, Any]:
    """Run Problem 3 Step 25 and save required outputs."""
    try:
        _ = load_config(config_path)
    except RuntimeError:
        pass
    problem3_config = get_problem3_config(config_path)

    text_dir = resolve_project_path(problem3_config["appendix3_extracted_text_dir"])
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

    log_path = logs_dir / "argument_chain_diagnosis.log"
    logger = setup_argument_chain_logger(log_path)
    logger.info("Starting Step 25 argument chain diagnosis")
    logger.info("Using sections_dir=%s", sections_dir)

    current_eval_path = tables_dir / "appendix3_current_evaluation.xlsx"
    low_feature_path = tables_dir / "appendix3_current_low_feature_report.xlsx"
    section_audit_path = tables_dir / "appendix3_section_quality_audit.xlsx"
    current_eval = pd.read_excel(current_eval_path)
    low_feature_lookup = _load_low_feature_lookup(low_feature_path)
    section_audit = _load_section_audit(section_audit_path)

    all_node_rows: list[dict[str, Any]] = []
    all_edge_rows: list[dict[str, Any]] = []
    for section_path in sorted(sections_dir.glob("*.json"), key=_natural_sort_key):
        payload = _load_json(section_path)
        node_rows = extract_nodes_for_paper(section_path, text_dir)
        edge_rows, edge_scores = calculate_edges_for_paper(payload, node_rows, low_feature_lookup)
        all_node_rows.extend(node_rows)
        all_edge_rows.extend(edge_rows)
        logger.info("%s edge scores: %s", section_path.stem, {key: round(value, 6) for key, value in edge_scores.items()})

    nodes_table = pd.DataFrame(all_node_rows)
    edges_table = pd.DataFrame(all_edge_rows)
    diagnosis, evidence = build_diagnosis_tables(nodes_table, edges_table, section_audit, current_eval)

    paths = {
        "nodes": tables_dir / "argument_chain_nodes.xlsx",
        "edges": tables_dir / "argument_chain_edges.xlsx",
        "diagnosis": tables_dir / "logic_gap_diagnosis.xlsx",
        "evidence": tables_dir / "logic_gap_evidence.xlsx",
        "heatmap": charts_dir / "argument_chain_heatmap.png",
        "gap_bar": charts_dir / "logic_gap_bar.png",
        "radar": charts_dir / "logic_chain_radar.png",
        "log": log_path,
    }
    nodes_table.to_excel(paths["nodes"], index=False)
    edges_table.to_excel(paths["edges"], index=False)
    diagnosis.to_excel(paths["diagnosis"], index=False)
    evidence.to_excel(paths["evidence"], index=False)

    plot_argument_chain_heatmap(diagnosis, paths["heatmap"])
    plot_logic_gap_bar(diagnosis, paths["gap_bar"])
    plot_logic_chain_radar(diagnosis, paths["radar"])

    logger.info("Nodes saved: %s rows=%s", paths["nodes"], len(nodes_table))
    logger.info("Edges saved: %s rows=%s", paths["edges"], len(edges_table))
    logger.info("Diagnosis saved: %s", paths["diagnosis"])
    logger.info("Evidence saved: %s rows=%s", paths["evidence"], len(evidence))
    logger.info("Finished Step 25")

    return {
        "nodes": nodes_table,
        "edges": edges_table,
        "diagnosis": diagnosis,
        "evidence": evidence,
        "paths": paths,
    }


def extract_argument_chain(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Backward-compatible wrapper for chain extraction."""
    return run_argument_chain_diagnosis(*args, **kwargs)


def calculate_logic_gap_score(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Backward-compatible wrapper for logic gap scoring."""
    return run_argument_chain_diagnosis(*args, **kwargs)
