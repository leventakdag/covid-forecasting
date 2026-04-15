from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from covid_forecasting.config import RollingExperimentConfig
from covid_forecasting.utils import ensure_dir, rmse


@dataclass
class ElasticNetExperiment:
    save_dir: Path
    config: RollingExperimentConfig | None = None

    def __post_init__(self) -> None:
        self.config = self.config or RollingExperimentConfig()
        ensure_dir(self.save_dir)

    def run(
        self,
        df_data: pd.DataFrame,
        lagged_cols: list[str],
        cutoff_date: pd.Timestamp,
        end_date: pd.Timestamp,
        train_scope: str,
        feature_tag: str,
    ) -> pd.DataFrame:
        results_store = []
        current_anchor = cutoff_date

        while True:
            forecast_anchor_start = current_anchor
            forecast_anchor_end = forecast_anchor_start + pd.Timedelta(days=self.config.step_days)
            earliest_test_target = forecast_anchor_start + pd.Timedelta(days=self.config.horizon)

            if earliest_test_target > end_date:
                break

            latest_train_anchor = forecast_anchor_start - pd.Timedelta(days=self.config.horizon)
            earliest_train_anchor = latest_train_anchor - pd.Timedelta(days=self.config.training_window_days)

            train_time_mask = (
                (df_data["date"] >= earliest_train_anchor)
                & (df_data["date"] <= latest_train_anchor)
            )
            test_time_mask = (
                (df_data["date"] >= forecast_anchor_start)
                & (df_data["date"] < forecast_anchor_end)
            )

            if train_scope == "local":
                df_train = df_data[train_time_mask & (df_data["location_key"] == self.config.test_country)].copy()
            else:
                df_train = df_data[train_time_mask].copy()

            df_test = df_data[test_time_mask & (df_data["location_key"] == self.config.test_country)].copy()

            df_train_sc = df_train.copy()
            df_test_sc = df_test.copy()

            df_train_sc["target_diff"] = (
                df_train_sc["target_y_plus_7"] - np.log1p(df_train_sc["naive_pred_sum7"])
            )
            df_test_sc["target_diff"] = (
                df_test_sc["target_y_plus_7"] - np.log1p(df_test_sc["naive_pred_sum7"])
            )

            target_x_scaler = None
            target_y_scaler = None

            for loc in sorted(df_train_sc["location_key"].unique()):
                loc_mask = df_train_sc["location_key"] == loc

                x_scaler = StandardScaler()
                df_train_sc.loc[loc_mask, lagged_cols] = x_scaler.fit_transform(
                    df_train_sc.loc[loc_mask, lagged_cols]
                )

                y_scaler = StandardScaler()
                df_train_sc.loc[loc_mask, "target_diff"] = y_scaler.fit_transform(
                    df_train_sc.loc[loc_mask, ["target_diff"]]
                ).ravel()

                if loc == self.config.test_country:
                    target_x_scaler = x_scaler
                    target_y_scaler = y_scaler

            df_test_sc.loc[:, lagged_cols] = target_x_scaler.transform(df_test_sc.loc[:, lagged_cols])

            X_train = df_train_sc[lagged_cols].values
            y_train = df_train_sc["target_diff"].values
            X_test = df_test_sc[lagged_cols].values
            y_test_raw_absolute = df_test["target_y_plus_7"].values

            tscv = TimeSeriesSplit(n_splits=5)
            elastic_model = ElasticNetCV(
                l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0],
                cv=tscv,
                max_iter=5000,
                random_state=0,
                n_jobs=-1,
            )
            elastic_model.fit(X_train, y_train)

            pred_diff_sc = elastic_model.predict(X_test)
            pred_diff = target_y_scaler.inverse_transform(pred_diff_sc.reshape(-1, 1)).ravel()

            current_log_cases_test = np.log1p(df_test["naive_pred_sum7"].values)
            pred_log_reconstructed = current_log_cases_test + pred_diff

            pred_cases = np.clip(np.expm1(pred_log_reconstructed), 0, None)
            actual_cases = np.clip(np.expm1(y_test_raw_absolute), 0, None)

            for _, row in df_test.iterrows():
                idx = df_test.index.get_loc(row.name)
                results_store.append(
                    [
                        self.config.test_country,
                        row["date"] + pd.Timedelta(days=self.config.horizon),
                        row["date"],
                        actual_cases[idx],
                        pred_cases[idx],
                        row["naive_pred_sum7"],
                    ]
                )

            current_anchor += pd.Timedelta(days=self.config.step_days)

        res_df = pd.DataFrame(
            results_store,
            columns=["Location", "Target_Date", "Anchor_Date", "Actual", "ElasticNet", "Naive"],
        )
        res_df = res_df.sort_values(["Location", "Target_Date"]).dropna(subset=["Actual"])

        save_file = self.save_dir / (
            f"preds_elasticnet_train_{train_scope}_test_{self.config.test_country}_{feature_tag}.csv"
        )
        res_df.to_csv(save_file, index=False)
        return res_df

    @staticmethod
    def score(predictions: pd.DataFrame) -> dict[str, float]:
        act_vals = predictions["Actual"].values
        model_preds = predictions["ElasticNet"].values
        naive_preds = predictions["Naive"].values
        return {
            "rmse": rmse(act_vals, model_preds),
            "r2": float(r2_score(act_vals, model_preds)),
            "naive_rmse": rmse(act_vals, naive_preds),
            "naive_r2": float(r2_score(act_vals, naive_preds)),
        }
