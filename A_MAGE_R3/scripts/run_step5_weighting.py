"""Step 5: calculate AHP-entropy combination weights."""

from pathlib import Path
import sys

try:
    import yaml
except ImportError:  # pragma: no cover - fallback for partially installed envs.
    yaml = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.weighting import calculate_weights  # noqa: E402


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
                "fusion_alpha": 0.6,
                "weight_table_filename": "appendix1_weights_ahp_entropy.xlsx",
                "weight_chart_filename": "appendix1_weights_ahp_entropy.png",
                "log_filename": "weighting.log",
            },
        }

    with Path(config_path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def resolve_project_path(relative_path: str) -> Path:
    """Resolve a project-relative path."""
    return PROJECT_ROOT / relative_path


def main() -> None:
    """Run Step 5 weighting only."""
    config = load_config(PROJECT_ROOT / "config.yaml")
    paths = config["paths"]
    feature_config = config.get("feature_extraction", {})
    weight_config = config.get("weights", {})

    tables_dir = resolve_project_path(paths["output_tables_dir"])
    charts_dir = resolve_project_path(paths["output_charts_dir"])
    logs_dir = resolve_project_path(paths["output_logs_dir"])

    feature_table_path = tables_dir / feature_config.get(
        "normalized_feature_filename", "appendix1_features_normalized.xlsx"
    )
    output_path = tables_dir / weight_config.get(
        "weight_table_filename", "appendix1_weights_ahp_entropy.xlsx"
    )
    chart_path = charts_dir / weight_config.get(
        "weight_chart_filename", "appendix1_weights_ahp_entropy.png"
    )
    log_path = logs_dir / weight_config.get("log_filename", "weighting.log")

    result = calculate_weights(
        feature_table_path=feature_table_path,
        output_path=output_path,
        chart_path=chart_path,
        log_path=log_path,
        alpha=float(weight_config.get("fusion_alpha", 0.6)),
    )

    combined_table = result["combined_table"]
    consistency_table = result["consistency_table"]
    imputation_report = result["imputation_report"]
    top5 = combined_table.sort_values("combined_weight", ascending=False).head(5)

    print("Step 5 AHP-entropy weighting finished.")
    print(f"Input normalized feature table: {feature_table_path}")
    print(f"Weight table: {output_path}")
    print(f"Weight chart: {chart_path}")
    print(f"Log: {log_path}")
    print(f"Indicators weighted: {len(combined_table)}")
    print(f"AHP max CR: {consistency_table['cr'].max():.6f}")
    print(f"Entropy missing cells imputed: {int(imputation_report['missing_imputed_count'].sum())}")
    print("Top 5 combined weights:")
    for _, row in top5.iterrows():
        print(f"  {row['rank']}. {row['indicator']}: {row['combined_weight']:.6f}")


if __name__ == "__main__":
    main()
