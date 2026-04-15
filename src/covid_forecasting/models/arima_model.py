from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from covid_forecasting.config import RollingExperimentConfig
from covid_forecasting.utils import ensure_dir, rmse


@dataclass
class ARIMAExperiment:
    save_dir: Path
    config: RollingExperimentConfig | None = None

    def __post_init__(self) -> None:
        self.config = self.config or RollingExperimentConfig()
        ensure_dir(self.save_dir)

    @staticmethod
    def optimize_arima(y_train, p_values, d_value, q_values):
        best_aic = float("inf")
        best_model = None
        best_order = None

        for p_val in p_values:
            for q_val in q_values:
                try:
                    model = ARIMA(y_train, order=(p_val, d_value, q_val))
                    results = model.fit()

                    if results.aic < best_aic:
                        best_aic = results.aic
                        best_model = results
                        best_order = (p_val, d_value, q_val)
                except Exception:
                    continue

        if best_model is None:
            fallback = ARIMA(y_train, order=(0, 1, 0)).fit()
            return fallback, (0, 1, 0)

        return best_model, best_order

    def run(self, df_data: pd.DataFrame, cutoff_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
        results_store = []
        df_loc = df_data[df_data["location_key"] == self.config.test_country].copy()
        df_loc = df_loc.sort_values("date").set_index("date")

        current_anchor = pd.to_datetime(cutoff_date) - pd.Timedelta(days=self.config.horizon)
        p_grid = [0, 1, 2, 3, 7]
        d_val = 1
        q_grid = [0, 1, 2]

        while True:
            forecast_anchor_start = current_anchor
            forecast_anchor_end = forecast_anchor_start + pd.Timedelta(days=self.config.step_days)
            earliest_test_target = forecast_anchor_start + pd.Timedelta(days=self.config.horizon)

            if earliest_test_target > end_date:
                break

            latest_train_anchor = forecast_anchor_start - pd.Timedelta(days=self.config.horizon)
            earliest_train_anchor = latest_train_anchor - pd.Timedelta(days=self.config.training_window_days)

            train_mask = (df_loc.index >= earliest_train_anchor) & (df_loc.index <= latest_train_anchor)
            test_mask = (df_loc.index >= forecast_anchor_start) & (df_loc.index < forecast_anchor_end)

            df_train = df_loc[train_mask]
            df_test = df_loc[test_mask]

            if len(df_train) < 30 or len(df_test) == 0:
                current_anchor += pd.Timedelta(days=self.config.step_days)
                continue

            y_train = df_train["log_cases_sum7"].values
            best_model, _ = self.optimize_arima(y_train, p_grid, d_val, q_grid)

            for idx, row in df_test.iterrows():
                target_date = idx + pd.Timedelta(days=self.config.horizon)
                history_up_to_t = df_loc[df_loc.index <= idx]["log_cases_sum7"].values
                frozen_model = best_model.apply(history_up_to_t)
                forecast_logs = frozen_model.forecast(steps=self.config.horizon)
                pred_log_reconstructed = forecast_logs[-1]

                pred_cases = np.clip(np.expm1(pred_log_reconstructed), 0, None)
                actual_cases = np.clip(np.expm1(row["target_y_plus_7"]), 0, None)

                results_store.append(
                    [
                        self.config.test_country,
                        target_date,
                        idx,
                        actual_cases,
                        pred_cases,
                        row["naive_pred_sum7"],
                    ]
                )

            current_anchor += pd.Timedelta(days=self.config.step_days)

        res_df = pd.DataFrame(
            results_store,
            columns=["Location", "Target_Date", "Anchor_Date", "Actual", "ARIMA", "Naive"],
        )
        res_df = res_df.sort_values("Target_Date").dropna(subset=["Actual"])
        save_file = self.save_dir / f"preds_arima_test_{self.config.test_country}.csv"
        res_df.to_csv(save_file, index=False)
        return res_df

    @staticmethod
    def score(predictions: pd.DataFrame) -> dict[str, float]:
        act_vals = predictions["Actual"].values
        model_preds = predictions["ARIMA"].values
        naive_preds = predictions["Naive"].values
        return {
            "rmse": rmse(act_vals, model_preds),
            "naive_rmse": rmse(act_vals, naive_preds),
        }
