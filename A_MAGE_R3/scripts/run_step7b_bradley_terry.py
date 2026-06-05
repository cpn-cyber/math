"""Step 7B: fit Bradley-Terry ranking calibration model."""

from pathlib import Path
import sys

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.bradley_terry import run_bradley_terry_calibration  # noqa: E402


def load_config(config_path: Path) -> dict:
    """Load YAML config or return defaults."""
    if yaml is None:
        return {
            "paths": {
                "output_tables_dir": "output/tables",
                "output_charts_dir": "output/charts",
                "output_logs_dir": "output/logs",
            },
            "bradley_terry": {
                "fusion_lambda": 0.7,
                "ridge": 0.001,
                "log_filename": "bradley_terry.log",
            },
        }
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def resolve_project_path(relative_path: str) -> Path:
    """Resolve a project-relative path."""
    return PROJECT_ROOT / relative_path


def main() -> None:
    """Run Bradley-Terry fitting and rank fusion only."""
    config = load_config(PROJECT_ROOT / "config.yaml")
    paths = config["paths"]
    bt_config = config.get("bradley_terry", {})

    tables_dir = resolve_project_path(paths["output_tables_dir"])
    charts_dir = resolve_project_path(paths["output_charts_dir"])
    logs_dir = resolve_project_path(paths["output_logs_dir"])

    result = run_bradley_terry_calibration(
        pairwise_path=tables_dir / "pairwise_comparison_filled.xlsx",
        topsis_path=tables_dir / "appendix1_topsis_scores.xlsx",
        quality_check_path=tables_dir / "pairwise_quality_check.xlsx",
        bt_output_path=tables_dir / "bradley_terry_scores.xlsx",
        fusion_output_path=tables_dir / "appendix1_rank_fusion.xlsx",
        sensitivity_output_path=tables_dir / "bradley_terry_tie_sensitivity.xlsx",
        scatter_chart_path=charts_dir / "bt_vs_topsis_scatter.png",
        rank_change_chart_path=charts_dir / "rank_change.png",
        log_path=logs_dir / bt_config.get("log_filename", "bradley_terry.log"),
        fusion_lambda=float(bt_config.get("fusion_lambda", 0.7)),
        ridge=float(bt_config.get("ridge", 0.001)),
    )

    bt_scores = result["bt_scores"]
    rank_fusion = result["rank_fusion"]
    rank_change_top = result["rank_change_top"].head(5)

    print("Step 7B Bradley-Terry calibration finished.")
    print(f"Valid comparison count: {result['valid_count']}")
    print(f"Tie count: {result['tie_count']}")
    print("Tie strategy: half-win by default; skip-tie fitted for sensitivity.")
    print(f"theta range: {bt_scores['theta'].min():.6f} - {bt_scores['theta'].max():.6f}")
    print(f"S_BT range: {bt_scores['S_BT'].min():.6f} - {bt_scores['S_BT'].max():.6f}")
    print(f"S_rank range: {rank_fusion['S_rank'].min():.6f} - {rank_fusion['S_rank'].max():.6f}")
    print(f"TOPSIS vs fused Spearman: {result['topsis_fusion_spearman']:.6f}")
    print(f"Tie sensitivity Spearman: {result['tie_sensitivity_spearman']:.6f}")
    print("Top 5 largest rank changes:")
    for _, row in rank_change_top.iterrows():
        print(
            f"  {row['paper_id']} {row['filename']}: "
            f"rank_base={int(row['rank_base'])}, rank_fused={int(row['rank_fused'])}, "
            f"rank_change={int(row['rank_change'])}"
        )
    for label, path in result["outputs"].items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
