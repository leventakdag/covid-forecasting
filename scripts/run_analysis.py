from __future__ import annotations

import argparse
from pathlib import Path

from covid_forecasting.analysis import ResultsAnalyzer, build_panel_analysis
from covid_forecasting.config import DateConfig, ProjectPaths
from covid_forecasting.data import PanelDataBuilder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate panel summary tables and forecasting result tables.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Directory containing raw CSV inputs.")
    parser.add_argument("--output-dir", type=Path, default=Path("preds/preds"), help="Directory containing saved prediction CSVs.")
    parser.add_argument("--report-dir", type=Path, default=Path("reports"), help="Directory for generated summary tables.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)

    paths = ProjectPaths(data_dir=args.data_dir, output_dir=args.output_dir)
    panel_bundle = PanelDataBuilder(paths=paths, date_config=DateConfig()).build()

    panel_analysis = build_panel_analysis(
        data_imputed=panel_bundle.data_imputed,
        good_countries=panel_bundle.good_countries,
    )
    panel_analysis.panel_summary.to_csv(args.report_dir / "panel_summary.csv", index=False)
    panel_analysis.country_target_summary.to_csv(args.report_dir / "country_target_summary.csv", index=False)
    panel_analysis.similarity_df.to_csv(args.report_dir / "country_similarity_to_nl.csv", index=False)
    panel_analysis.panel_desc_df.to_csv(args.report_dir / "panel_descriptive_statistics.csv", index=False)

    analyzer = ResultsAnalyzer(output_dir=args.output_dir)
    main_results = analyzer.build_main_results_table()
    ablation_results = analyzer.build_ablation_results_table()

    if not main_results.empty:
        main_results.to_csv(args.report_dir / "main_results_table.csv")
    if not ablation_results.empty:
        ablation_results.to_csv(args.report_dir / "ablation_results_table.csv")


if __name__ == "__main__":
    main()
