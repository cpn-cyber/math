"""Step 7A: generate a blank pairwise-comparison template."""

from pathlib import Path
import sys

try:
    import yaml
except ImportError:  # pragma: no cover - fallback for partially installed envs.
    yaml = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.bradley_terry import generate_pairwise_template  # noqa: E402


def load_config(config_path: Path) -> dict:
    """Load YAML configuration."""
    if yaml is None:
        return {
            "paths": {
                "output_tables_dir": "output/tables",
                "output_logs_dir": "output/logs",
            },
            "topsis": {
                "score_table_filename": "appendix1_topsis_scores.xlsx",
            },
            "pairwise_template": {
                "template_filename": "pairwise_comparison_template.xlsx",
                "log_filename": "pairwise_template.log",
                "close_gap_threshold": 3.0,
                "boundary_scores": [60, 50, 40],
                "min_appearances_per_paper": 2,
            },
        }

    with Path(config_path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def resolve_project_path(relative_path: str) -> Path:
    """Resolve a project-relative path."""
    return PROJECT_ROOT / relative_path


def main() -> None:
    """Run Step 7A template generation only."""
    config = load_config(PROJECT_ROOT / "config.yaml")
    paths = config["paths"]
    topsis_config = config.get("topsis", {})
    pairwise_config = config.get("pairwise_template", {})

    tables_dir = resolve_project_path(paths["output_tables_dir"])
    logs_dir = resolve_project_path(paths["output_logs_dir"])
    score_path = tables_dir / topsis_config.get("score_table_filename", "appendix1_topsis_scores.xlsx")
    output_path = tables_dir / pairwise_config.get("template_filename", "pairwise_comparison_template.xlsx")
    log_path = logs_dir / pairwise_config.get("log_filename", "pairwise_template.log")

    result = generate_pairwise_template(
        score_path=score_path,
        output_path=output_path,
        log_path=log_path,
        close_gap_threshold=float(pairwise_config.get("close_gap_threshold", 3.0)),
        boundary_scores=list(pairwise_config.get("boundary_scores", [60, 50, 40])),
        min_appearances_per_paper=int(pairwise_config.get("min_appearances_per_paper", 2)),
    )

    template = result["template"]
    coverage = result["coverage"]
    insufficient = result["insufficient_papers"]
    print("Step 7A pairwise comparison template generated.")
    print(f"Input TOPSIS score table: {score_path}")
    print(f"Template: {output_path}")
    print(f"Log: {log_path}")
    print(f"Pair count: {len(template)}")
    print(f"Average appearances per paper: {coverage['appearance_count'].mean():.4f}")
    print(f"Under-covered papers: {', '.join(insufficient) if insufficient else 'none'}")
    print("winner and reason columns are blank for human review.")


if __name__ == "__main__":
    main()
