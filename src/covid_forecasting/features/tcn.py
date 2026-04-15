from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from covid_forecasting.config import TCNSequenceConfig


@dataclass
class TCNSequenceBundle:
    X: np.ndarray
    y: np.ndarray
    dates_arr: np.ndarray
    locs_arr: np.ndarray
    anchor_dates_arr: np.ndarray
    baseline_lookup: dict
    feature_cols: list[str]
    feature_tag: str
    num_features: int


class TCNSequenceBuilder:
    def __init__(self, config: TCNSequenceConfig | None = None) -> None:
        self.config = config or TCNSequenceConfig()

    @staticmethod
    def _add_tcn_features(group: pd.DataFrame) -> pd.DataFrame:
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

    def build(self, data_imputed: pd.DataFrame, good_countries: list[str]) -> TCNSequenceBundle:
        df = data_imputed.copy()
        df = df[df["location_key"].isin(good_countries)].copy()
        df["date"] = pd.to_datetime(df["date"])

        rename_map = {
            "mobility_workplaces_mob": "mobility_workplaces",
            "stringency_index_gov": "stringency_index",
            "average_temperature_celsius_weather": "average_temperature_celsius",
            "relative_humidity_weather": "relative_humidity",
        }
        df = df.rename(columns=rename_map)

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
        df = (
            df.groupby("location_key", group_keys=False)
            .apply(self._add_tcn_features)
            .reset_index(drop=True)
        )

        feature_cols = ["log_cases_sum7", "momentum_7", "acceleration"]
        feature_tag = "ar"

        if self.config.use_exog:
            feature_cols += [
                "RMT",
                "RMH",
                "Interaction",
                "SI",
                "MOB_TRANSIT",
                "MOB_WORKPLACES",
                "MOB_RETAIL",
            ]
            feature_tag = "full"

        target_col = "log_cases_sum7"
        num_features = len(feature_cols)
        df_clean = df.dropna(subset=feature_cols).copy()

        df_nl_full = df[df["location_key"] == self.config.test_country].copy()
        if not df_nl_full.empty:
            df_nl_full = df_nl_full.sort_values("date").set_index("date").asfreq("D")
            df_nl_full["new_confirmed_clipped"] = df_nl_full["new_confirmed_clipped"].ffill()
            df_nl_full["cases_sum7"] = df_nl_full["new_confirmed_clipped"].rolling(7, min_periods=1).sum()
            df_nl_full["naive_pred"] = df_nl_full["cases_sum7"].shift(7)
            baseline_lookup = df_nl_full[["naive_pred"]].to_dict("index")
        else:
            baseline_lookup = {}

        X_list, y_list, loc_list, dates_list = [], [], [], []
        target_idx = feature_cols.index(target_col)

        for loc, sub in df_clean.groupby("location_key"):
            sub = sub.sort_values("date")
            sub_vals = sub[feature_cols].values
            dates = sub["date"].values

            n_rows = len(sub)
            if n_rows < self.config.lookback_lags + self.config.horizon:
                continue

            for t in range(self.config.lookback_lags - 1, n_rows - self.config.horizon):
                window_seq = sub_vals[t - self.config.lookback_lags + 1 : t + 1, :]
                target_log = sub_vals[t + self.config.horizon, target_idx]
                naive_log = sub_vals[t, target_idx]
                y_diff = target_log - naive_log

                X_list.append(window_seq)
                y_list.append(y_diff)
                loc_list.append(loc)
                dates_list.append(dates[t + self.config.horizon])

        if len(X_list) == 0:
            raise ValueError("No sequences generated. Check LOOKBACK_LAGS, HORIZON, and date coverage.")

        X = np.stack(X_list)
        y = np.array(y_list).reshape(-1, 1)
        dates_arr = np.array(dates_list)
        locs_arr = np.array(loc_list)
        anchor_dates_arr = dates_arr - np.timedelta64(self.config.horizon, "D")

        return TCNSequenceBundle(
            X=X,
            y=y,
            dates_arr=dates_arr,
            locs_arr=locs_arr,
            anchor_dates_arr=anchor_dates_arr,
            baseline_lookup=baseline_lookup,
            feature_cols=feature_cols,
            feature_tag=feature_tag,
            num_features=num_features,
        )
