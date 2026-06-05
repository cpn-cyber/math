"""Step 7C: audit Bradley-Terry effects and run fusion sensitivity analysis.

This step does not modify human pairwise winners, regenerate templates, or
classify grades. It audits rank changes caused by BT calibration and compares
fusion strategies using original S_BT and S_BT scaled to the TOPSIS score
range.
"""

from __future__ import annotations

from pathlib import Path
import logging
import re
from typing import Any

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LAMBDAS = [0.70, 0.80, 0.85, 0.90]
RANK_CHANGE_THRESHOLD = 8


def setup_logger(log_path: Path) -> logging.Logger:
    """Create a Step 7C logger."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("A_MAGE_R3.bt_audit_sensitivity")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.FileHandler(log_path, mode="w", encoding="utf-8-sig")
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(handler)
    return logger


def normalize_id(value: Any) -> str:
    """Normalize paper IDs to two-digit strings."""
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    match = re.search(r"(\d+)", text)
    return match.group(1).zfill(2) if match else text.zfill(2)


def spearman_rank(left: pd.Series, right: pd.Series) -> float:
    """Calculate Spearman correlation from two aligned rank columns."""
    left = pd.to_numeric(left, errors="coerce")
    right = pd.to_numeric(right, errors="coerce")
    valid = ~(left.isna() | right.isna())
    if valid.sum() < 2:
        return np.nan
    return float(np.corrcoef(left[valid], right[valid])[0, 1])


def set_chinese_font() -> None:
    """Use a Chinese font if available."""
    candidates = ["Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC", "Arial Unicode MS"]
    available = {font.name for font in font_manager.fontManager.ttflist}
    for candidate in candidates:
        if candidate in available:
            plt.rcParams["font.sans-serif"] = [candidate]
            break
    plt.rcParams["axes.unicode_minus"] = False


def load_inputs(root: Path) -> dict[str, pd.DataFrame]:
    """Load all Step 7C input files."""
    tables = root / "output/tables"
    topsis = pd.read_excel(tables / "appendix1_topsis_scores.xlsx", sheet_name="topsis_scores")
    bt = pd.read_excel(tables / "bradley_terry_scores.xlsx", sheet_name="bt_scores_tie_half")
    fusion = pd.read_excel(tables / "appendix1_rank_fusion.xlsx", sheet_name="rank_fusion")
    pairwise = pd.read_excel(tables / "pairwise_comparison_filled.xlsx", sheet_name="pairwise_template")
    tie_sensitivity = pd.read_excel(tables / "bradley_terry_tie_sensitivity.xlsx", sheet_name="tie_sensitivity")
    tie_summary = pd.read_excel(tables / "bradley_terry_tie_sensitivity.xlsx", sheet_name="summary")

    for frame in [topsis, bt, fusion, pairwise, tie_sensitivity]:
        for column in ["paper_id", "paper_i", "paper_j"]:
            if column in frame.columns:
                frame[column] = frame[column].apply(normalize_id)

    return {
        "topsis": topsis,
        "bt": bt,
        "fusion": fusion,
        "pairwise": pairwise,
        "tie_sensitivity": tie_sensitivity,
        "tie_summary": tie_summary,
    }


def outcome_for_paper(row: pd.Series, paper_id: str) -> str:
    """Return win/loss/tie for one paper in one pairwise row."""
    winner = str(row["winner"]).strip().lower()
    if winner == "tie":
        return "tie"
    if row["paper_i"] == paper_id:
        return "win" if winner == "i" else "loss"
    if row["paper_j"] == paper_id:
        return "win" if winner == "j" else "loss"
    return ""


def audit_rank_changes(fusion: pd.DataFrame, pairwise: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Audit papers with large rank changes and trace pairwise evidence."""
    fusion = fusion.copy()
    fusion["abs_rank_change"] = fusion["rank_change"].abs()
    abnormal = fusion.loc[fusion["abs_rank_change"] >= RANK_CHANGE_THRESHOLD].copy()
    abnormal = abnormal.sort_values(["abs_rank_change", "rank_fused"], ascending=[False, True])

    audit_rows = []
    trace_rows = []
    for _, paper in abnormal.iterrows():
        paper_id = normalize_id(paper["paper_id"])
        records = pairwise.loc[(pairwise["paper_i"].eq(paper_id)) | (pairwise["paper_j"].eq(paper_id))].copy()
        wins: list[str] = []
        losses: list[str] = []
        ties: list[str] = []
        for _, row in records.iterrows():
            opponent = row["paper_j"] if row["paper_i"] == paper_id else row["paper_i"]
            outcome = outcome_for_paper(row, paper_id)
            if outcome == "win":
                wins.append(opponent)
            elif outcome == "loss":
                losses.append(opponent)
            elif outcome == "tie":
                ties.append(opponent)
            trace_rows.append(
                {
                    "paper_id": paper_id,
                    "pair_id": row["pair_id"],
                    "opponent": opponent,
                    "paper_i": row["paper_i"],
                    "paper_j": row["paper_j"],
                    "winner": row["winner"],
                    "outcome_for_paper": outcome,
                    "reason": row.get("reason", ""),
                }
            )

        total = len(records)
        win_count = len(wins)
        loss_count = len(losses)
        tie_count = len(ties)
        decisive = win_count + loss_count
        win_rate = win_count / decisive if decisive else np.nan
        bias_flags = []
        if total < 2:
            bias_flags.append("comparison_count_lt_2")
        if total < 4:
            bias_flags.append("low_comparison_count")
        if decisive > 0 and (win_rate >= 0.80 or win_rate <= 0.20):
            bias_flags.append("one_sided_decisive_results")
        if total > 0 and tie_count / total >= 0.50:
            bias_flags.append("high_tie_ratio")

        audit_rows.append(
            {
                "paper_id": paper_id,
                "filename": paper["filename"],
                "rank_base": int(paper["rank_base"]),
                "rank_bt": int(paper["rank_bt"]),
                "rank_fused": int(paper["rank_fused"]),
                "rank_change": int(paper["rank_change"]),
                "abs_rank_change": int(abs(paper["rank_change"])),
                "comparison_count": total,
                "win_count": win_count,
                "loss_count": loss_count,
                "tie_count": tie_count,
                "win_rate_decisive": win_rate,
                "main_win_opponents": ",".join(wins) if wins else "",
                "main_loss_opponents": ",".join(losses) if losses else "",
                "tie_opponents": ",".join(ties) if ties else "",
                "has_insufficient_or_bias": bool(bias_flags),
                "bias_flags": ",".join(bias_flags) if bias_flags else "none",
            }
        )

    return pd.DataFrame(audit_rows), pd.DataFrame(trace_rows)


def add_scaled_bt(base: pd.DataFrame) -> pd.DataFrame:
    """Add S_BT scaled to the S_base min-max interval."""
    result = base.copy()
    sbase_min = float(result["S_base"].min())
    sbase_max = float(result["S_base"].max())
    sbt_min = float(result["S_BT"].min())
    sbt_max = float(result["S_BT"].max())
    if sbt_max == sbt_min:
        result["S_BT_scaled"] = (sbase_min + sbase_max) / 2
    else:
        result["S_BT_scaled"] = sbase_min + (result["S_BT"] - sbt_min) / (sbt_max - sbt_min) * (sbase_max - sbase_min)
    result["rank_bt_scaled"] = result["S_BT_scaled"].rank(ascending=False, method="first").astype(int)
    return result


def calculate_fusion_for_lambda(base: pd.DataFrame, score_column: str, lambda_value: float) -> pd.DataFrame:
    """Calculate one fusion variant."""
    result = base.copy()
    result["bt_score_version"] = score_column
    result["lambda"] = lambda_value
    result["S_rank"] = lambda_value * result["S_base"] + (1 - lambda_value) * result[score_column]
    result["rank_fused"] = result["S_rank"].rank(ascending=False, method="first").astype(int)
    result["rank_change"] = result["rank_base"].astype(int) - result["rank_fused"].astype(int)
    result["abs_rank_change"] = result["rank_change"].abs()
    return result


def lambda_sensitivity(base: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run lambda sensitivity for original and scaled BT scores."""
    all_rank_rows = []
    summary_rows = []
    for version in ["S_BT", "S_BT_scaled"]:
        for lambda_value in LAMBDAS:
            fused = calculate_fusion_for_lambda(base, version, lambda_value)
            all_rank_rows.append(fused)
            summary_rows.append(
                {
                    "bt_score_version": version,
                    "lambda": lambda_value,
                    "spearman_vs_topsis": spearman_rank(fused["rank_base"], fused["rank_fused"]),
                    "max_abs_rank_change": int(fused["abs_rank_change"].max()),
                    "mean_abs_rank_change": float(fused["abs_rank_change"].mean()),
                    "large_rank_change_count_gt_8": int((fused["abs_rank_change"] > RANK_CHANGE_THRESHOLD).sum()),
                    "large_rank_change_count_ge_8": int((fused["abs_rank_change"] >= RANK_CHANGE_THRESHOLD).sum()),
                    "S_rank_min": float(fused["S_rank"].min()),
                    "S_rank_max": float(fused["S_rank"].max()),
                }
            )
    return pd.DataFrame(summary_rows), pd.concat(all_rank_rows, ignore_index=True)


def recommend_scheme(summary: pd.DataFrame) -> dict[str, Any]:
    """Recommend a final BT fusion scheme from computed sensitivity metrics."""
    candidates = summary.copy()
    candidates["version_priority"] = candidates["bt_score_version"].map({"S_BT_scaled": 0, "S_BT": 1})
    feasible = candidates.loc[
        (candidates["spearman_vs_topsis"] >= 0.80)
        & (candidates["large_rank_change_count_gt_8"] == 0)
        & (candidates["max_abs_rank_change"] <= RANK_CHANGE_THRESHOLD)
    ].copy()
    if feasible.empty:
        feasible = candidates.loc[candidates["spearman_vs_topsis"] >= 0.80].copy()
    if feasible.empty:
        feasible = candidates.copy()
    feasible["lambda_distance_to_085"] = (feasible["lambda"] - 0.85).abs()
    feasible = feasible.sort_values(
        [
            "version_priority",
            "lambda_distance_to_085",
            "max_abs_rank_change",
            "mean_abs_rank_change",
        ],
        ascending=[True, True, True, True],
    )
    chosen = feasible.iloc[0].to_dict()
    return {
        "recommended_bt_score_version": chosen["bt_score_version"],
        "recommended_lambda": float(chosen["lambda"]),
        "spearman_vs_topsis": float(chosen["spearman_vs_topsis"]),
        "max_abs_rank_change": int(chosen["max_abs_rank_change"]),
        "mean_abs_rank_change": float(chosen["mean_abs_rank_change"]),
        "large_rank_change_count_gt_8": int(chosen["large_rank_change_count_gt_8"]),
        "reason": (
            "优先采用映射到 TOPSIS 分数区间的 S_BT_scaled；在 Spearman>=0.80、无超过8名的大幅变化前提下，"
            "lambda=0.85 保留适度 BT 校准，同时避免大幅推翻 TOPSIS。"
        ),
    }


def plot_lambda_spearman(summary: pd.DataFrame, chart_path: Path) -> Path:
    """Plot lambda-Spearman sensitivity."""
    chart_path.parent.mkdir(parents=True, exist_ok=True)
    set_chinese_font()
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for version, group in summary.groupby("bt_score_version"):
        label = "原始 S_BT" if version == "S_BT" else "S_BT_scaled"
        ax.plot(group["lambda"], group["spearman_vs_topsis"], marker="o", linewidth=2, label=label)
    ax.axhline(0.80, color="#C82423", linestyle="--", linewidth=1.3, label="0.80参考线")
    ax.set_title("lambda 与 TOPSIS 排名相关性", fontsize=14, fontweight="bold")
    ax.set_xlabel("lambda")
    ax.set_ylabel("Spearman")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(chart_path, dpi=220)
    plt.close(fig)
    return chart_path


def plot_lambda_max_change(summary: pd.DataFrame, chart_path: Path) -> Path:
    """Plot max absolute rank change by lambda."""
    chart_path.parent.mkdir(parents=True, exist_ok=True)
    set_chinese_font()
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for version, group in summary.groupby("bt_score_version"):
        label = "原始 S_BT" if version == "S_BT" else "S_BT_scaled"
        ax.plot(group["lambda"], group["max_abs_rank_change"], marker="o", linewidth=2, label=label)
    ax.axhline(RANK_CHANGE_THRESHOLD, color="#C82423", linestyle="--", linewidth=1.3, label="8名参考线")
    ax.set_title("lambda 与最大排名变化", fontsize=14, fontweight="bold")
    ax.set_xlabel("lambda")
    ax.set_ylabel("最大 |rank_change|")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(chart_path, dpi=220)
    plt.close(fig)
    return chart_path


def plot_rank_change_v2(final_fusion: pd.DataFrame, chart_path: Path) -> Path:
    """Plot final recommended rank changes."""
    chart_path.parent.mkdir(parents=True, exist_ok=True)
    set_chinese_font()
    plot_data = final_fusion.sort_values("rank_change")
    colors = np.where(plot_data["rank_change"] > 0, "#2E7D32", np.where(plot_data["rank_change"] < 0, "#C82423", "#7A7A7A"))
    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(plot_data["paper_id"].astype(str).str.zfill(2), plot_data["rank_change"], color=colors, alpha=0.86)
    ax.axvline(0, color="#333333", linewidth=1)
    ax.set_title("推荐融合方案排名变化", fontsize=14, fontweight="bold")
    ax.set_xlabel("rank_change = rank_base - rank_fused_v2")
    ax.set_ylabel("论文编号")
    ax.grid(axis="x", alpha=0.25)
    for bar in bars:
        width = bar.get_width()
        x = width + 0.12 if width >= 0 else width - 0.12
        ha = "left" if width >= 0 else "right"
        ax.text(x, bar.get_y() + bar.get_height() / 2, f"{width:.0f}", va="center", ha=ha, fontsize=8)
    fig.tight_layout()
    fig.savefig(chart_path, dpi=220)
    plt.close(fig)
    return chart_path


def main() -> None:
    """Run Step 7C."""
    tables = PROJECT_ROOT / "output/tables"
    charts = PROJECT_ROOT / "output/charts"
    logs = PROJECT_ROOT / "output/logs"
    logger = setup_logger(logs / "bt_audit_sensitivity.log")

    inputs = load_inputs(PROJECT_ROOT)
    fusion = inputs["fusion"].copy()
    pairwise = inputs["pairwise"].copy()
    tie_summary = inputs["tie_summary"].copy()

    logger.info("Starting Step 7C BT audit and lambda sensitivity.")
    abnormal_audit, abnormal_traces = audit_rank_changes(fusion, pairwise)
    logger.info("Large rank-change papers |rank_change| >= %s: %s", RANK_CHANGE_THRESHOLD, len(abnormal_audit))

    base = add_scaled_bt(fusion)
    sensitivity_summary, sensitivity_rank_details = lambda_sensitivity(base)
    recommendation = recommend_scheme(sensitivity_summary)
    final = calculate_fusion_for_lambda(
        base,
        recommendation["recommended_bt_score_version"],
        recommendation["recommended_lambda"],
    )
    final = final.rename(
        columns={
            "S_rank": "S_rank_v2",
            "rank_fused": "rank_fused_v2",
            "rank_change": "rank_change_v2",
            "abs_rank_change": "abs_rank_change_v2",
        }
    )
    final["recommended_bt_score_version"] = recommendation["recommended_bt_score_version"]
    final["recommended_lambda"] = recommendation["recommended_lambda"]
    final = final.sort_values("rank_fused_v2").reset_index(drop=True)

    recommendation_df = pd.DataFrame([recommendation])
    scale_summary = pd.DataFrame(
        [
            {"metric": "S_base_min", "value": float(base["S_base"].min())},
            {"metric": "S_base_max", "value": float(base["S_base"].max())},
            {"metric": "S_BT_min", "value": float(base["S_BT"].min())},
            {"metric": "S_BT_max", "value": float(base["S_BT"].max())},
            {"metric": "S_BT_scaled_min", "value": float(base["S_BT_scaled"].min())},
            {"metric": "S_BT_scaled_max", "value": float(base["S_BT_scaled"].max())},
        ]
    )

    with pd.ExcelWriter(tables / "bt_rank_change_audit.xlsx", engine="openpyxl") as writer:
        abnormal_audit.to_excel(writer, index=False, sheet_name="large_rank_changes")
        abnormal_traces.to_excel(writer, index=False, sheet_name="pairwise_trace")
        scale_summary.to_excel(writer, index=False, sheet_name="scale_summary")

    with pd.ExcelWriter(tables / "bt_lambda_sensitivity.xlsx", engine="openpyxl") as writer:
        sensitivity_summary.to_excel(writer, index=False, sheet_name="summary")
        sensitivity_rank_details.to_excel(writer, index=False, sheet_name="rank_details")
        recommendation_df.to_excel(writer, index=False, sheet_name="recommendation")
        tie_summary.to_excel(writer, index=False, sheet_name="tie_sensitivity_reference")

    with pd.ExcelWriter(tables / "appendix1_rank_fusion_v2.xlsx", engine="openpyxl") as writer:
        final.to_excel(writer, index=False, sheet_name="rank_fusion_v2")
        recommendation_df.to_excel(writer, index=False, sheet_name="recommendation")
        scale_summary.to_excel(writer, index=False, sheet_name="scale_summary")

    plot_lambda_spearman(sensitivity_summary, charts / "bt_lambda_spearman.png")
    plot_lambda_max_change(sensitivity_summary, charts / "bt_lambda_max_rank_change.png")
    plot_rank_change_v2(final.rename(columns={"rank_change_v2": "rank_change"}), charts / "rank_change_v2.png")

    logger.info("Recommendation: %s", recommendation)
    logger.info("Saved bt_rank_change_audit.xlsx, bt_lambda_sensitivity.xlsx, appendix1_rank_fusion_v2.xlsx")
    logger.info("Saved bt_lambda_spearman.png, bt_lambda_max_rank_change.png, rank_change_v2.png")
    logger.info("Finished Step 7C. Grade classification was not run.")

    print("Step 7C BT audit and sensitivity finished.")
    print(f"Large rank-change papers: {len(abnormal_audit)}")
    print("Lambda sensitivity summary:")
    print(sensitivity_summary.to_string(index=False))
    print("Recommendation:")
    print(recommendation_df.to_string(index=False))
    print("Output files saved.")


if __name__ == "__main__":
    main()
