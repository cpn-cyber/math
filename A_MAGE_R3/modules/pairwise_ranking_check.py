"""Pairwise ranking consistency check for Problem 2 Step 19."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
import logging
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from modules.appendix2_pipeline import get_problem2_config, resolve_project_path


LOGGER_NAME = "A_MAGE_R3.problem2.pairwise_ranking_check"
NEAR_TIE_THRESHOLD = 1.0
LENGTH_FEATURES = {"total_chars", "page_count"}


def setup_pairwise_logger(log_path: Path) -> logging.Logger:
    """Configure the Step 19 logger."""
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
    """Natural sort key for paper IDs."""
    import re

    text = str(value)
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def _as_bool(value: Any) -> bool:
    """Convert table values to bool."""
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def _load_key_features(key_feature_index: pd.DataFrame) -> pd.DataFrame:
    """Load final key features and K_final weights."""
    key_features = key_feature_index.loc[key_feature_index["final_key_feature"].apply(_as_bool)].copy()
    if key_features.empty:
        raise ValueError("key_feature_index_final.xlsx contains no final_key_feature=True rows.")
    required = {"feature_name", "K_final"}
    missing = required - set(key_features.columns)
    if missing:
        raise ValueError(f"key_feature_index_final.xlsx missing columns: {sorted(missing)}")
    key_features["feature_name"] = key_features["feature_name"].astype(str)
    key_features["K_final"] = pd.to_numeric(key_features["K_final"], errors="coerce")
    key_features = key_features.dropna(subset=["K_final"])
    if key_features.empty:
        raise ValueError("No valid K_final weights for final key features.")
    return key_features.sort_values("K_rank", na_position="last").reset_index(drop=True)


def _predict_winner(pair_score: float) -> str:
    """Return predicted winner from pair score."""
    if pair_score > 0:
        return "i"
    if pair_score < 0:
        return "j"
    return "tie"


def build_pairwise_differences(
    features: pd.DataFrame,
    quality_label: pd.Series | None = None,
    key_features: pd.DataFrame | None = None,
    near_tie_threshold: float = NEAR_TIE_THRESHOLD,
) -> pd.DataFrame:
    """Build C(n, 2) pairwise feature and quality differences."""
    matrix = features.copy()
    matrix["paper_id"] = matrix["paper_id"].astype(str)
    matrix = matrix.sort_values("paper_id", key=lambda series: series.map(_natural_sort_key)).reset_index(drop=True)
    if quality_label is not None:
        matrix["Q_label"] = pd.to_numeric(quality_label, errors="coerce")
    if "Q_label" not in matrix.columns:
        raise ValueError("Feature matrix must contain Q_label or quality_label must be provided.")
    if key_features is None:
        raise ValueError("key_features with feature_name and K_final is required.")

    feature_names = key_features["feature_name"].astype(str).tolist()
    missing_features = [feature for feature in feature_names if feature not in matrix.columns]
    if missing_features:
        raise ValueError(f"Feature matrix missing final key features: {missing_features}")
    weight_lookup = key_features.set_index("feature_name")["K_final"].to_dict()

    rows: list[dict[str, Any]] = []
    for pair_index, (idx_i, idx_j) in enumerate(combinations(range(len(matrix)), 2), start=1):
        row_i = matrix.iloc[idx_i]
        row_j = matrix.iloc[idx_j]
        delta_q = float(row_i["Q_label"] - row_j["Q_label"])
        true_winner = "i" if delta_q > 0 else "j" if delta_q < 0 else "tie"
        pair_score = 0.0
        contribution_parts: list[str] = []
        row: dict[str, Any] = {
            "pair_id": f"P{pair_index:03d}",
            "paper_i": row_i["paper_id"],
            "filename_i": row_i.get("filename", f"{row_i['paper_id']}.txt"),
            "Q_i": float(row_i["Q_label"]),
            "paper_j": row_j["paper_id"],
            "filename_j": row_j.get("filename", f"{row_j['paper_id']}.txt"),
            "Q_j": float(row_j["Q_label"]),
            "Delta_Q": delta_q,
            "true_winner": true_winner,
            "near_tie": abs(delta_q) < float(near_tie_threshold),
        }
        for feature in feature_names:
            delta_x = float(pd.to_numeric(pd.Series([row_i[feature] - row_j[feature]]), errors="coerce").iloc[0])
            weight = float(weight_lookup[feature])
            contribution = weight * delta_x
            pair_score += contribution
            row[f"Delta_{feature}"] = delta_x
            row[f"Contribution_{feature}"] = contribution
            contribution_parts.append(f"{feature}:{contribution:.6f}")
        predicted_winner = _predict_winner(pair_score)
        row["pair_score"] = pair_score
        row["predicted_winner"] = predicted_winner
        row["correct"] = predicted_winner == true_winner
        row["valid_pair"] = true_winner != "tie" and predicted_winner != "tie"
        row["feature_contributions"] = "; ".join(contribution_parts)
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_pairwise_ranking_consistency(pairwise_table: pd.DataFrame) -> pd.DataFrame:
    """Evaluate whether weighted pair scores agree with quality order."""
    valid = pairwise_table.loc[pairwise_table["valid_pair"].apply(_as_bool)]
    near_tie = pairwise_table.loc[pairwise_table["near_tie"].apply(_as_bool)]
    without_near_tie = valid.loc[~valid["near_tie"].apply(_as_bool)]

    total_pairs = int(len(pairwise_table))
    valid_pairs = int(len(valid))
    near_tie_pairs = int(len(near_tie))
    overall_accuracy = float(valid["correct"].mean()) if valid_pairs else np.nan
    accuracy_without_near_tie = float(without_near_tie["correct"].mean()) if len(without_near_tie) else np.nan
    if pd.notna(overall_accuracy) and overall_accuracy >= 0.70:
        conclusion = "Final key features provide supportive pairwise ranking consistency; pairs are dependent and used only as auxiliary validation."
    elif pd.notna(overall_accuracy) and overall_accuracy >= 0.60:
        conclusion = "Pairwise ranking consistency is moderate; small sample and weak-supervised labels limit generalization."
    else:
        conclusion = "Pairwise ranking consistency is limited; use only as audit evidence, not as a main model."
    return pd.DataFrame(
        [
            {
                "total_pairs": total_pairs,
                "valid_pairs": valid_pairs,
                "near_tie_pairs": near_tie_pairs,
                "overall_pairwise_accuracy": overall_accuracy,
                "group_validation_accuracy_mean": np.nan,
                "group_validation_accuracy_std": np.nan,
                "accuracy_without_near_tie": accuracy_without_near_tie,
                "conclusion": conclusion,
            }
        ]
    )


def _group_validation(pairwise_table: pd.DataFrame, paper_ids: list[str], high_influence_papers: set[str]) -> pd.DataFrame:
    """Run leave-one-paper group validation with fixed K_final weights."""
    rows: list[dict[str, Any]] = []
    for paper_id in paper_ids:
        test = pairwise_table.loc[pairwise_table["paper_i"].eq(paper_id) | pairwise_table["paper_j"].eq(paper_id)].copy()
        valid = test.loc[test["valid_pair"].apply(_as_bool)]
        non_tie = valid.loc[~valid["near_tie"].apply(_as_bool)]
        accuracy = float(valid["correct"].mean()) if len(valid) else np.nan
        accuracy_no_tie = float(non_tie["correct"].mean()) if len(non_tie) else np.nan
        near_tie_count = int(test["near_tie"].apply(_as_bool).sum())
        high_influence = paper_id in high_influence_papers
        if high_influence:
            note = "High-influence paper from Step18; group accuracy should be interpreted cautiously."
        else:
            note = "Fixed K_final weighted ranking check; no pairwise training performed."
        rows.append(
            {
                "heldout_paper": paper_id,
                "test_pair_count": int(len(test)),
                "pairwise_accuracy": accuracy,
                "accuracy_without_near_tie": accuracy_no_tie,
                "near_tie_count": near_tie_count,
                "high_influence_flag": high_influence,
                "note": note,
            }
        )
    return pd.DataFrame(rows).sort_values("heldout_paper", key=lambda series: series.map(_natural_sort_key)).reset_index(drop=True)


def _save_group_accuracy_chart(group: pd.DataFrame, chart_path: Path) -> None:
    """Save heldout-paper accuracy bar chart."""
    chart_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = group.sort_values("heldout_paper", key=lambda series: series.map(_natural_sort_key))
    colors = ["#b64b4b" if value else "#4f7cac" for value in ordered["high_influence_flag"]]
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=180)
    ax.bar(ordered["heldout_paper"], ordered["pairwise_accuracy"], color=colors)
    ax.axhline(float(ordered["pairwise_accuracy"].mean()), color="#333333", linestyle="--", linewidth=1, label="mean")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Pairwise accuracy")
    ax.set_title("Leave-One-Paper Pairwise Ranking Accuracy")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(chart_path)
    plt.close(fig)


def _save_weight_chart(key_features: pd.DataFrame, chart_path: Path) -> None:
    """Save K_final weights for final key features."""
    chart_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = key_features.sort_values("K_final").copy()
    colors = ["#8b6f47" if feature in LENGTH_FEATURES else "#4f7cac" for feature in ordered["feature_name"]]
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=180)
    ax.barh(ordered["feature_name"], ordered["K_final"], color=colors)
    ax.set_xlabel("K_final")
    ax.set_title("Final Key Feature Weights Used in Pairwise Check")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(chart_path)
    plt.close(fig)


def _feature_list_text(key_features: pd.DataFrame) -> str:
    """Return final key feature list text."""
    return ", ".join(key_features["feature_name"].astype(str).tolist())


def run_step19_pairwise_ranking_check(config_path: Path | None = None) -> dict[str, Any]:
    """Run Step 19 pairwise ranking consistency check."""
    problem2_config = get_problem2_config(config_path)
    tables_dir = resolve_project_path(problem2_config["problem2_output_tables_dir"])
    charts_dir = resolve_project_path(problem2_config["problem2_output_charts_dir"])
    logs_dir = resolve_project_path(problem2_config["problem2_output_logs_dir"])
    tables_dir.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    log_path = logs_dir / "pairwise_ranking_check.log"
    logger = setup_pairwise_logger(log_path)
    logger.info("Starting Step 19 pairwise ranking check; this is auxiliary validation, not a main model.")

    feature_matrix = pd.read_excel(tables_dir / "appendix2_feature_matrix_robust_scaled.xlsx")
    q_labels = pd.read_excel(tables_dir / "appendix2_q_labels.xlsx")
    key_index = pd.read_excel(tables_dir / "key_feature_index_final.xlsx")
    model_summary = pd.read_excel(tables_dir / "model_stability_summary.xlsx", sheet_name="summary")
    high_influence = pd.read_excel(tables_dir / "high_influence_samples.xlsx")

    for frame in [feature_matrix, q_labels, key_index, model_summary, high_influence]:
        if "paper_id" in frame.columns:
            frame["paper_id"] = frame["paper_id"].astype(str)
    feature_matrix["paper_id"] = feature_matrix["paper_id"].astype(str)
    if {"paper_id", "Q_label"}.issubset(q_labels.columns):
        q_check = feature_matrix[["paper_id", "Q_label"]].merge(
            q_labels[["paper_id", "Q_label"]].rename(columns={"Q_label": "Q_label_from_file"}),
            on="paper_id",
            how="left",
        )
        mismatch = q_check.loc[
            ~np.isclose(
                pd.to_numeric(q_check["Q_label"], errors="coerce"),
                pd.to_numeric(q_check["Q_label_from_file"], errors="coerce"),
                equal_nan=False,
            )
        ]
        if not mismatch.empty:
            raise ValueError(f"Q_label mismatch with appendix2_q_labels.xlsx: {mismatch.to_dict(orient='records')}")

    key_features = _load_key_features(key_index)
    pairwise = build_pairwise_differences(
        feature_matrix,
        key_features=key_features,
        near_tie_threshold=NEAR_TIE_THRESHOLD,
    )
    high_influence_papers = set(high_influence.get("removed_paper_id", pd.Series(dtype=str)).astype(str).tolist())
    paper_ids = feature_matrix.sort_values("paper_id", key=lambda series: series.map(_natural_sort_key))["paper_id"].astype(str).tolist()
    group = _group_validation(pairwise, paper_ids, high_influence_papers)
    check = evaluate_pairwise_ranking_consistency(pairwise)
    check.loc[0, "group_validation_accuracy_mean"] = float(group["pairwise_accuracy"].mean())
    check.loc[0, "group_validation_accuracy_std"] = float(group["pairwise_accuracy"].std(ddof=1))
    check.loc[0, "key_features_used"] = _feature_list_text(key_features)
    check.loc[0, "note"] = "The 45 pairs are dependent; group validation leaves out all pairs containing one paper."

    dataset_path = tables_dir / "pairwise_ranking_dataset.xlsx"
    check_path = tables_dir / "pairwise_ranking_check.xlsx"
    group_path = tables_dir / "pairwise_group_validation.xlsx"
    accuracy_chart_path = charts_dir / "pairwise_ranking_accuracy_bar.png"
    weight_chart_path = charts_dir / "pairwise_feature_weight_bar.png"

    pairwise.to_excel(dataset_path, index=False)
    check.to_excel(check_path, index=False)
    group.to_excel(group_path, index=False)
    _save_group_accuracy_chart(group, accuracy_chart_path)
    _save_weight_chart(key_features, weight_chart_path)

    paper_21 = group.loc[group["heldout_paper"].eq("2-1")]
    supports = bool(float(check.loc[0, "overall_pairwise_accuracy"]) >= 0.60)
    logger.info("Final key features used: %s", _feature_list_text(key_features))
    logger.info("Pairwise summary: %s", check.to_dict(orient="records"))
    logger.info("Group validation: %s", group.to_dict(orient="records"))
    logger.info("2-1 group validation: %s", paper_21.to_dict(orient="records"))
    logger.info("Pairwise supports K_final conclusion: %s", supports)
    logger.info("Saved outputs: %s, %s, %s, %s, %s", dataset_path, check_path, group_path, accuracy_chart_path, weight_chart_path)
    logger.info("Finished Step 19")

    return {
        "pairwise": pairwise,
        "check": check,
        "group": group,
        "key_features": key_features,
        "paper_21": paper_21,
        "supports": supports,
        "dataset_path": dataset_path,
        "check_path": check_path,
        "group_path": group_path,
        "accuracy_chart_path": accuracy_chart_path,
        "weight_chart_path": weight_chart_path,
        "log_path": log_path,
    }
