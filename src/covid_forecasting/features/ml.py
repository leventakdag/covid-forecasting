from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from covid_forecasting.config import HybridFeatureConfig


@dataclass
class MLFeatureBundle:
    df_ml: pd.DataFrame
    lagged_features: list[str]
    feature_tag: str


class HybridFeatureBuilder:
    def __init__(self, config: HybridFeatureConfig | None = None) -> None:
        self.config = config or HybridFeatureConfig()

    @staticmethod
    def _make_panel_daily(df: pd.DataFrame) -> pd.DataFrame:
        out = []
        for loc, group in df.groupby("location_key", sort=False):
            group = group.sort_values("date").set_index("date")
            group = group.asfreq("D").ffill()
            group["location_key"] = loc
            out.append(group.reset_index())

        out = pd.concat(out, axis=0, ignore_index=True)
        out = out.sort_values(["location_key", "date"]).reset_index(drop=True)
        return out

    @staticmethod
    def _add_country_features(group: pd.DataFrame) -> pd.DataFrame:
        group = group.sort_values("date").copy()

        group["new_confirmed_clipped"] = group["new_confirmed"].clip(lower=0)
        group["cases_sum7"] = group["new_confirmed_clipped"].rolling(7, min_periods=1).sum()
        group["log_cases_sum7"] = np.log1p(group["cases_sum7"])

        group["momentum_7"] = group["log_cases_sum7"].diff(7)
        group["acceleration"] = group["momentum_7"].diff(1)

        group["RMT"] = group["average_temperature_celsius"].rolling(7, min_periods=1).mean()
        group["RMH"] = group["relative_humidity"].rolling(7, min_periods=1).mean()
        group["Interaction"] = group["RMT"] * group["RMH"]

        if "stringency_index" in group.columns:
            group["SI"] = group["stringency_index"]

        group["MOB_TRANSIT"] = group["mobility_transit_stations"].rolling(7, min_periods=1).mean()
        group["MOB_RETAIL"] = group["mobility_retail_and_recreation"].rolling(7, min_periods=1).mean()
        group["MOB_WORKPLACES"] = group["mobility_workplaces"].rolling(7, min_periods=1).mean()
        return group

    @staticmethod
    def _add_hybrid_lags_and_target(
        group: pd.DataFrame,
        dense_features: list[str],
        snapshot_features: list[str],
        dense_lags: tuple[int, ...],
        snapshot_lags: tuple[int, ...],
        horizon: int,
    ) -> pd.DataFrame:
        group = group.sort_values("date").copy()

        for lag in dense_lags:
            suffix = "_today" if lag == 0 else f"_lag_{lag}"
            for feature in dense_features:
                group[f"{feature}{suffix}"] = group[feature].shift(lag)

        for lag in snapshot_lags:
            suffix = "_today" if lag == 0 else f"_lag_{lag}"
            for feature in snapshot_features:
                group[f"{feature}{suffix}"] = group[feature].shift(lag)

        group["target_y_plus_7"] = group["log_cases_sum7"].shift(-horizon)
        group["naive_pred_sum7"] = group["cases_sum7"]
        return group

    def build(self, data_imputed: pd.DataFrame) -> MLFeatureBundle:
        df = data_imputed.copy()
        df["date"] = pd.to_datetime(df["date"])

        needed_cols = [
            "location_key",
            "date",
            "new_confirmed",
            "average_temperature_celsius",
            "relative_humidity",
            "mobility_transit_stations",
            "mobility_retail_and_recreation",
            "mobility_workplaces",
        ]
        if "stringency_index" in df.columns:
            needed_cols.append("stringency_index")

        df = df[needed_cols].copy()
        df = df.sort_values(["location_key", "date"]).reset_index(drop=True)
        df = self._make_panel_daily(df)
        df = (
            df.groupby("location_key", group_keys=False)
            .apply(self._add_country_features)
            .reset_index(drop=True)
        )

        if self.config.base_config == "momentum":
            all_features = ["momentum_7"]
            feature_tag = "momentum"
        elif self.config.base_config == "ar":
            all_features = ["log_cases_sum7", "momentum_7", "acceleration"]
            feature_tag = "ar"
        else:
            all_features = ["log_cases_sum7"]
            feature_tag = "custom"

        if self.config.use_exog:
            all_features += [
                "RMT",
                "RMH",
                "Interaction",
                "SI",
                "MOB_TRANSIT",
                "MOB_WORKPLACES",
                "MOB_RETAIL",
            ]
            feature_tag = "full"

        all_features = [column for column in all_features if column in df.columns]
        dense_features = ["momentum_7"] if "momentum_7" in all_features else []
        snapshot_features = [feature for feature in all_features if feature not in dense_features]

        df = (
            df.groupby("location_key", group_keys=False)
            .apply(
                lambda group: self._add_hybrid_lags_and_target(
                    group,
                    dense_features=dense_features,
                    snapshot_features=snapshot_features,
                    dense_lags=self.config.dense_lags,
                    snapshot_lags=self.config.snapshot_lags,
                    horizon=self.config.horizon,
                )
            )
            .reset_index(drop=True)
        )

        lagged_features: list[str] = []

        for lag in self.config.dense_lags:
            suffix = "_today" if lag == 0 else f"_lag_{lag}"
            for feature in dense_features:
                lagged_features.append(f"{feature}{suffix}")

        for lag in self.config.snapshot_lags:
            suffix = "_today" if lag == 0 else f"_lag_{lag}"
            for feature in snapshot_features:
                lagged_features.append(f"{feature}{suffix}")

        df_ml = df.dropna(
            subset=lagged_features + ["target_y_plus_7", "naive_pred_sum7"]
        ).copy()
        df_ml["date"] = pd.to_datetime(df_ml["date"])
        df_ml = df_ml.sort_values(["location_key", "date"]).reset_index(drop=True)

        return MLFeatureBundle(
            df_ml=df_ml,
            lagged_features=lagged_features,
            feature_tag=feature_tag,
        )
