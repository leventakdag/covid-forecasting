from __future__ import annotations

import argparse
from pathlib import Path

from covid_forecasting.config import DateConfig, ProjectPaths, TCNSequenceConfig, TCNTrainingConfig
from covid_forecasting.data import PanelDataBuilder
from covid_forecasting.features import TCNSequenceBuilder
from covid_forecasting.models import TCNExperiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TCN thesis experiments.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Directory containing raw CSV inputs.")
    parser.add_argument("--output-dir", type=Path, default=Path("preds/preds"), help="Directory for saved prediction CSVs.")
    parser.add_argument("--test-country", type=str, default="NL", help="ISO2 code of the evaluation country.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = ProjectPaths(data_dir=args.data_dir, output_dir=args.output_dir)
    date_config = DateConfig()
    sequence_config = TCNSequenceConfig(test_country=args.test_country)

    panel_bundle = PanelDataBuilder(paths=paths, date_config=date_config).build()
    sequence_bundle = TCNSequenceBuilder(config=sequence_config).build(
        panel_bundle.data_imputed,
        panel_bundle.good_countries,
    )

    experiment = TCNExperiment(
        save_dir=paths.output_dir,
        sequence_config=sequence_config,
        training_config=TCNTrainingConfig(),
    )
    experiment.run_scope(sequence_bundle, scope="global")
    experiment.run_scope(sequence_bundle, scope="local")


if __name__ == "__main__":
    main()
