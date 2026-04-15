from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from covid_forecasting.config import AnalysisConfig


def smape(y_true, y_pred) -> float:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denominator != 0
    result = np.zeros_like(numerator, dtype=float)
    result[mask] = numerator[mask] / denominator[mask]
    return float(np.mean(result) * 100)


@dataclass
class ResultsAnalyzer:
    output_dir: Path
    config: AnalysisConfig | None = None

    def __post_init__(self) -> None:
        self.config = self.config or AnalysisConfig()

    @staticmethod
    def _rmse(y_true, y_pred) -> float:
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))

    @staticmethod
    def _mae(y_true, y_pred) -> float:
        return float(mean_absolute_error(y_true, y_pred))

    def load_tcn_ensemble(self, scope: str, tag: str) -> pd.DataFrame | None:
        all_preds = []
        for run_i in range(self.config.n_runs):
            current_seed = self.config.base_seed + run_i

            if scope == "local":
                file_name = f"preds_local_{tag}_seed_{current_seed}.csv"
            elif scope in ["noweather", "nomobility", "nostringency"]:
                file_name = f"preds_{tag}_{scope}_seed_{current_seed}.csv"
            else:
                file_name = f"preds_{tag}_seed_{current_seed}.csv"

            file_path = self.output_dir / file_name
            if file_path.exists():
                df_run = pd.read_csv(file_path)
                df_run["Target_Date"] = pd.to_datetime(df_run["Target_Date"])
                all_preds.append(df_run)

        if not all_preds:
            return None

        df_all = pd.concat(all_preds, ignore_index=True)
        return (
            df_all.groupby("Target_Date")
            .agg({"Actual": "mean", "TCN": "mean", "Naive": "mean"})
            .reset_index()
        )

    def load_baseline(self, model_name: str, scope: str, tag: str) -> pd.DataFrame | None:
        if scope == "local":
            file_name = f"preds_{model_name.lower()}_train_local_test_NL_{tag}.csv"
        else:
            file_name = f"preds_{model_name.lower()}_train_global_test_NL_{tag}.csv"

        file_path = self.output_dir / file_name
        if not file_path.exists():
            return None

        df = pd.read_csv(file_path)
        df["Target_Date"] = pd.to_datetime(df["Target_Date"])
        ignore_cols = [
            "Target_Date",
            "Anchor_Date",
            "Actual",
            "Naive",
            "Loc",
            "Location",
            "Run",
            "iso_code",
            "Unnamed: 0",
        ]
        pred_cols = [c for c in df.columns if c not in ignore_cols]
        if len(pred_cols) == 0:
            return None

        df = df.rename(columns={pred_cols[0]: "Pred"})
        df["Pred"] = df["Pred"].astype(float)
        return df

    def build_main_results_table(self) -> pd.DataFrame:
        table_rows = []
        features = ["ar", "full"]

        base_df = self.load_tcn_ensemble("global", "ar")
        if base_df is not None:
            for regime in self.config.core_regimes:
                mask = (
                    (base_df["Target_Date"] >= pd.to_datetime(regime["start"]))
                    & (base_df["Target_Date"] <= pd.to_datetime(regime["end"]))
                )
                df_reg = base_df[mask].dropna()
                if len(df_reg) > 0:
                    y_act = df_reg["Actual"].values
                    y_pred = df_reg["Naive"].values
                    table_rows.append(
                        {
                            "Model": "Naive Baseline",
                            "Features": "-",
                            "Wave": regime["name"],
                            "RMSE": self._rmse(y_act, y_pred),
                            "MAE": self._mae(y_act, y_pred),
                            "sMAPE": smape(y_act, y_pred),
                        }
                    )

        arima_file = self.output_dir / "preds_arima_test_NL.csv"
        if arima_file.exists():
            df_arima = pd.read_csv(arima_file)
            df_arima["Target_Date"] = pd.to_datetime(df_arima["Target_Date"])

            for regime in self.config.core_regimes:
                mask = (
                    (df_arima["Target_Date"] >= pd.to_datetime(regime["start"]))
                    & (df_arima["Target_Date"] <= pd.to_datetime(regime["end"]))
                )
                df_reg = df_arima[mask].dropna(subset=["Actual", "ARIMA"])
                if len(df_reg) > 0:
                    y_act = df_reg["Actual"].values
                    y_pred = df_reg["ARIMA"].values
                    table_rows.append(
                        {
                            "Model": "ARIMA",
                            "Features": "AR",
                            "Wave": regime["name"],
                            "RMSE": self._rmse(y_act, y_pred),
                            "MAE": self._mae(y_act, y_pred),
                            "sMAPE": smape(y_act, y_pred),
                        }
                    )

        for tag in features:
            for scope in ["local", "global"]:
                tcn_df = self.load_tcn_ensemble(scope, tag)
                if tcn_df is not None:
                    for regime in self.config.core_regimes:
                        mask = (
                            (tcn_df["Target_Date"] >= pd.to_datetime(regime["start"]))
                            & (tcn_df["Target_Date"] <= pd.to_datetime(regime["end"]))
                        )
                        df_reg = tcn_df[mask].dropna()
                        if len(df_reg) > 0:
                            y_act = df_reg["Actual"].values
                            y_pred = df_reg["TCN"].values
                            table_rows.append(
                                {
                                    "Model": f"TCN ({scope.capitalize()})",
                                    "Features": tag.upper(),
                                    "Wave": regime["name"],
                                    "RMSE": self._rmse(y_act, y_pred),
                                    "MAE": self._mae(y_act, y_pred),
                                    "sMAPE": smape(y_act, y_pred),
                                }
                            )

                enet_df = self.load_baseline("elasticnet", scope, tag)
                if enet_df is not None:
                    for regime in self.config.core_regimes:
                        mask = (
                            (enet_df["Target_Date"] >= pd.to_datetime(regime["start"]))
                            & (enet_df["Target_Date"] <= pd.to_datetime(regime["end"]))
                        )
                        df_reg = enet_df[mask].dropna()
                        if len(df_reg) > 0:
                            y_act = df_reg["Actual"].values
                            y_pred = df_reg["Pred"].values
                            table_rows.append(
                                {
                                    "Model": f"ElasticNet ({scope.capitalize()})",
                                    "Features": tag.upper(),
                                    "Wave": regime["name"],
                                    "RMSE": self._rmse(y_act, y_pred),
                                    "MAE": self._mae(y_act, y_pred),
                                    "sMAPE": smape(y_act, y_pred),
                                }
                            )

                xgb_df = self.load_baseline("xgboost", scope, tag)
                if xgb_df is not None:
                    for regime in self.config.core_regimes:
                        mask = (
                            (xgb_df["Target_Date"] >= pd.to_datetime(regime["start"]))
                            & (xgb_df["Target_Date"] <= pd.to_datetime(regime["end"]))
                        )
                        df_reg = xgb_df[mask].dropna()
                        if len(df_reg) > 0:
                            y_act = df_reg["Actual"].values
                            y_pred = df_reg["Pred"].values
                            table_rows.append(
                                {
                                    "Model": f"XGBoost ({scope.capitalize()})",
                                    "Features": tag.upper(),
                                    "Wave": regime["name"],
                                    "RMSE": self._rmse(y_act, y_pred),
                                    "MAE": self._mae(y_act, y_pred),
                                    "sMAPE": smape(y_act, y_pred),
                                }
                            )

        df_table = pd.DataFrame(table_rows)
        if df_table.empty:
            return df_table

        df_table = df_table.set_index(["Model", "Features", "Wave"])
        main_results = df_table[["RMSE", "MAE", "sMAPE"]].unstack("Wave").round(2)
        main_results = main_results.swaplevel(0, 1, axis=1)

        col_order = [
            (wave, metric)
            for wave in ["Alpha", "Delta", "Omicron"]
            for metric in ["RMSE", "MAE", "sMAPE"]
        ]
        col_order = [col for col in col_order if col in main_results.columns]
        return main_results[col_order]

    def build_ablation_results_table(self) -> pd.DataFrame:
        table_rows = []
        ablation_scopes = ["global", "noweather", "nomobility", "nostringency"]

        for scope in ablation_scopes:
            tcn_df = self.load_tcn_ensemble(scope, "full")
            if tcn_df is None:
                continue

            for regime in self.config.core_regimes:
                mask = (
                    (tcn_df["Target_Date"] >= pd.to_datetime(regime["start"]))
                    & (tcn_df["Target_Date"] <= pd.to_datetime(regime["end"]))
                )
                df_reg = tcn_df[mask].dropna()
                if len(df_reg) == 0:
                    continue

                if scope == "global":
                    name = "TCN Global (All Features)"
                elif scope == "noweather":
                    name = "TCN Global (- Weather)"
                elif scope == "nomobility":
                    name = "TCN Global (- Mobility)"
                else:
                    name = "TCN Global (- Policy)"

                y_act = df_reg["Actual"].values
                y_pred = df_reg["TCN"].values
                table_rows.append(
                    {
                        "Model": name,
                        "Wave": regime["name"],
                        "RMSE": self._rmse(y_act, y_pred),
                        "MAE": self._mae(y_act, y_pred),
                        "sMAPE": smape(y_act, y_pred),
                    }
                )

        df_table = pd.DataFrame(table_rows)
        if df_table.empty:
            return df_table

        df_table = df_table.set_index(["Model", "Wave"])
        ablation_results = df_table[["RMSE", "MAE", "sMAPE"]].unstack("Wave").round(2)
        ablation_results = ablation_results.swaplevel(0, 1, axis=1)

        col_order = [
            (wave, metric)
            for wave in ["Alpha", "Delta", "Omicron"]
            for metric in ["RMSE", "MAE", "sMAPE"]
        ]
        col_order = [col for col in col_order if col in ablation_results.columns]
        return ablation_results[col_order]
