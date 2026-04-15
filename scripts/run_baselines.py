from __future__ import annotations

import argparse
from pathlib import Path

from covid_forecasting.config import DateConfig, HybridFeatureConfig, ProjectPaths, RollingExperimentConfig
from covid_forecasting.data import PanelDataBuilder
from covid_forecasting.features import HybridFeatureBuilder
from covid_forecasting.models import ARIMAExperiment, ElasticNetExperiment, XGBoostExperiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Elastic Net, XGBoost, and ARIMA thesis experiments.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Directory containing raw CSV inputs.")
    parser.add_argument("--output-dir", type=Path, default=Path("preds/preds"), help="Directory for saved prediction CSVs.")
    parser.add_argument("--test-country", type=str, default="NL", help="ISO2 code of the evaluation country.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = ProjectPaths(data_dir=args.data_dir, output_dir=args.output_dir)
    date_config = DateConfig()
    rolling_config = RollingExperimentConfig(test_country=args.test_country)

    panel_bundle = PanelDataBuilder(paths=paths, date_config=date_config).build()
    ml_bundle = HybridFeatureBuilder(config=HybridFeatureConfig()).build(panel_bundle.data_imputed)

    elastic_net = ElasticNetExperiment(save_dir=paths.output_dir, config=rolling_config)
    xgboost = XGBoostExperiment(save_dir=paths.output_dir, config=rolling_config)
    arima = ARIMAExperiment(save_dir=paths.output_dir, config=rolling_config)

    for scope in ["local", "global"]:
        elastic_net.run(
            df_data=ml_bundle.df_ml,
            lagged_cols=ml_bundle.lagged_features,
            cutoff_date=date_config.cutoff_date,
            end_date=date_config.end_date,
            train_scope=scope,
            feature_tag=ml_bundle.feature_tag,
        )
        xgboost.run(
            df_data=ml_bundle.df_ml,
            lagged_cols=ml_bundle.lagged_features,
            cutoff_date=date_config.cutoff_date,
            end_date=date_config.end_date,
            train_scope=scope,
            feature_tag=ml_bundle.feature_tag,
        )

    arima.run(
        df_data=ml_bundle.df_ml,
        cutoff_date=date_config.cutoff_date,
        end_date=date_config.end_date,
    )


if __name__ == "__main__":
    main()
