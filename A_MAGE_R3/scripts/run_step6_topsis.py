"""Step 6: run TOPSIS base scoring."""

from pathlib import Path
import sys

try:
    import yaml
except ImportError:  # pragma: no cover - fallback for partially installed envs.
    yaml = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.topsis import run_topsis  # noqa: E402


def load_config(config_path: Path) -> dict:
    """Load YAML configuration."""
    if yaml is None:
        return {
            "paths": {
                "output_tables_dir": "output/tables",
                "output_charts_dir": "output/charts",
                "output_logs_dir": "output/logs",
            },
            "feature_extraction": {
                "normalized_feature_filename": "appendix1_features_normalized.xlsx",
            },
            "weights": {
                "weight_table_filename": "appendix1_weights_ahp_entropy.xlsx",
            },
            "topsis": {
                "score_table_filename": "appendix1_topsis_scores.xlsx",
                "score_distribution_chart_filename": "topsis_score_distribution.png",
                "ranking_bar_chart_filename": "topsis_ranking_bar.png",
                "log_filename": "topsis.log",
            },
        }

    with Path(config_path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def resolve_project_path(relative_path: str) -> Path:
    """Resolve a project-relative path."""
    return PROJECT_ROOT / relative_path


def main() -> None:
    """Run Step 6 TOPSIS only."""
    config = load_config(PROJECT_ROOT / "config.yaml")
    paths = config["paths"]
    feature_config = config.get("feature_extraction", {})
    weight_config = config.get("weights", {})
    topsis_config = config.get("topsis", {})

    tables_dir = resolve_project_path(paths["output_tables_dir"])
    charts_dir = resolve_project_path(paths["output_charts_dir"])
    logs_dir = resolve_project_path(paths["output_logs_dir"])

    feature_table_path = tables_dir / feature_config.get(
        "normalized_feature_filename", "appendix1_features_normalized.xlsx"
    )
    weight_path = tables_dir / weight_config.get(
        "weight_table_filename", "appendix1_weights_ahp_entropy.xlsx"
    )
    audit_path = tables_dir / "feature_quality_audit.xlsx"
    output_path = tables_dir / topsis_config.get(
        "score_table_filename", "appendix1_topsis_scores.xlsx"
    )
    distribution_chart_path = charts_dir / topsis_config.get(
        "score_distribution_chart_filename", "topsis_score_distribution.png"
    )
    ranking_chart_path = charts_dir / topsis_config.get(
        "ranking_bar_chart_filename", "topsis_ranking_bar.png"
    )
    log_path = logs_dir / topsis_config.get("log_filename", "topsis.log")

    result = run_topsis(
        feature_table_path=feature_table_path,
        weight_path=weight_path,
        audit_path=audit_path,
        output_path=output_path,
        distribution_chart_path=distribution_chart_path,
        ranking_chart_path=ranking_chart_path,
        log_path=log_path,
    )

    scores = result["scores"]
    print("Step 6 TOPSIS base scoring finished.")
    print(f"Input normalized feature table: {feature_table_path}")
    print(f"Input combined weights: {weight_path}")
    print(f"Input feature quality audit: {audit_path}")
    print(f"TOPSIS score table: {output_path}")
    print(f"Score distribution chart: {distribution_chart_path}")
    print(f"Ranking bar chart: {ranking_chart_path}")
    print(f"Log: {log_path}")
    print(f"Papers scored: {len(scores)}")
    print(f"S_base range: {scores['S_base'].min():.6f} - {scores['S_base'].max():.6f}")
    print("Top 5 papers:")
    for _, row in scores.head(5).iterrows():
        print(f"  {row['rank_base']}. {row['paper_id']} {row['filename']}: {row['S_base']:.6f}")
    print("Bottom 5 papers:")
    for _, row in scores.tail(5).iterrows():
        print(f"  {row['rank_base']}. {row['paper_id']} {row['filename']}: {row['S_base']:.6f}")


if __name__ == "__main__":
    main()
