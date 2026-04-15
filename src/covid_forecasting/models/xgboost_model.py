from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from covid_forecasting.config import (
    RollingExperimentConfig,
    default_xgboost_global_param_grid,
    default_xgboost_local_param_grid,
)
from covid_forecasting.utils import ensure_dir, rmse


@dataclass
class XGBoostExperiment:
    save_dir: Path
    config: RollingExperimentConfig | None = None
    local_param_grid: list[dict[str, float]] = field(default_factory=default_xgboost_local_param_grid)
    global_param_grid: list[dict[str, float]] = field(default_factory=default_xgboost_global_param_grid)
    seed: int = 0

    def __post_init__(self) -> None:
        self.config = self.config or RollingExperimentConfig()
        ensure_dir(self.save_dir)

    @staticmethod
    def make_date_blocked_cv_splits(df_train: pd.DataFrame, n_splits: int = 5):
        unique_dates = np.array(sorted(df_train["date"].unique()))
        n_dates = len(unique_dates)

        if n_dates < (n_splits + 1) * 2:
            return None

        fold_size = n_dates // (n_splits + 1)
        if fold_size < 1:
            return None

        cv_splits = []
        pos_map = pd.Series(np.arange(len(df_train)), index=df_train.index)

        for i in range(1, n_splits + 1):
            val_start = i * fold_size
            val_end = (i + 1) * fold_size if i < n_splits else n_dates

            train_dates = unique_dates[:val_start]
            val_dates = unique_dates[val_start:val_end]

            if len(train_dates) == 0 or len(val_dates) == 0:
                continue

            train_idx_labels = df_train.index[df_train["date"].isin(train_dates)]
            val_idx_labels = df_train.index[df_train["date"].isin(val_dates)]

            if len(train_idx_labels) == 0 or len(val_idx_labels) == 0:
                continue

            train_pos = pos_map.loc[train_idx_labels].to_numpy()
            val_pos = pos_map.loc[val_idx_labels].to_numpy()

            if len(train_pos) > 0 and len(val_pos) > 0:
                cv_splits.append((train_pos, val_pos))

        if len(cv_splits) < 2:
            return None
        return cv_splits

    def _run_cv_early_stopping(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        cv_splits,
        param_grid: list[dict[str, float]],
    ):
        search_results = []
        max_estimators = 1000
        early_stopping_rounds = 100

        for params in param_grid:
            fold_rmses = []
            fold_best_iters = []

            for tr_idx, va_idx in cv_splits:
                X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
                X_va, y_va = X_train[va_idx], y_train[va_idx]

                model = XGBRegressor(
                    objective="reg:squarederror",
                    random_state=self.seed,
                    tree_method="hist",
                    device="cuda",
                    n_jobs=1,
                    verbosity=0,
                    subsample=0.8,
                    gamma=0.1,
                    n_estimators=max_estimators,
                    early_stopping_rounds=early_stopping_rounds,
                    eval_metric="rmse",
                    **params,
                )

                model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

                pred_va = model.predict(X_va)
                fold_rmses.append(np.sqrt(mean_squared_error(y_va, pred_va)))

                best_iter = getattr(model, "best_iteration", None)
                if best_iter is None:
                    best_iter = max_estimators - 1
                fold_best_iters.append(int(best_iter))

            search_results.append(
                {
                    "params": params,
                    "mean_cv_rmse": float(np.mean(fold_rmses)),
                    "std_cv_rmse": float(np.std(fold_rmses)),
                    "median_best_iteration": int(np.median(fold_best_iters)),
                }
            )

        return min(search_results, key=lambda d: d["mean_cv_rmse"])

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
        param_grid = self.local_param_grid if train_scope == "local" else self.global_param_grid

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

            nl_x_scaler = None
            nl_target_mean, nl_target_std = 0.0, 1.0

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
                    nl_x_scaler = x_scaler
                    nl_target_mean = y_scaler.mean_[0]
                    nl_target_std = y_scaler.scale_[0]

            df_test_sc.loc[:, lagged_cols] = nl_x_scaler.transform(df_test_sc.loc[:, lagged_cols])

            X_train = df_train_sc[lagged_cols].values
            y_train = df_train_sc["target_diff"].values
            X_test = df_test_sc[lagged_cols].values
            y_test_raw_absolute = df_test["target_y_plus_7"].values

            n_splits_local = 3 if len(df_train) < 80 else 5
            cv_splits = self.make_date_blocked_cv_splits(df_train_sc, n_splits=n_splits_local)
            if cv_splits is None:
                current_anchor += pd.Timedelta(days=self.config.step_days)
                continue

            best_cv_result = self._run_cv_early_stopping(
                X_train=X_train,
                y_train=y_train,
                cv_splits=cv_splits,
                param_grid=param_grid,
            )

            best_params = best_cv_result["params"]
            best_n_estimators = max(1, best_cv_result["median_best_iteration"] + 1)

            best_model = XGBRegressor(
                objective="reg:squarederror",
                random_state=self.seed,
                tree_method="hist",
                device="cuda",
                n_jobs=1,
                verbosity=0,
                subsample=0.8,
                gamma=0.1,
                n_estimators=best_n_estimators,
                eval_metric="rmse",
                **best_params,
            )
            best_model.fit(X_train, y_train, verbose=False)

            pred_diff_sc = best_model.predict(X_test)
            pred_diff = (pred_diff_sc * nl_target_std) + nl_target_mean

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
            columns=["Location", "Target_Date", "Anchor_Date", "Actual", "XGBoost", "Naive"],
        )
        res_df = res_df.sort_values(["Location", "Target_Date"]).dropna(subset=["Actual"])

        save_file = self.save_dir / (
            f"preds_xgboost_train_{train_scope}_test_{self.config.test_country}_{feature_tag}.csv"
        )
        res_df.to_csv(save_file, index=False)
        return res_df

    @staticmethod
    def score(predictions: pd.DataFrame) -> dict[str, float]:
        act_vals = predictions["Actual"].values
        model_preds = predictions["XGBoost"].values
        naive_preds = predictions["Naive"].values
        return {
            "rmse": rmse(act_vals, model_preds),
            "r2": float(r2_score(act_vals, model_preds)),
            "naive_rmse": rmse(act_vals, naive_preds),
            "naive_r2": float(r2_score(act_vals, naive_preds)),
        }
