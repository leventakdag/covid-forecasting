from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PanelAnalysisArtifacts:
    panel_summary: pd.DataFrame
    country_target_summary: pd.DataFrame
    similarity_df: pd.DataFrame
    panel_desc_df: pd.DataFrame


def build_panel_analysis(data_imputed: pd.DataFrame, good_countries: list[str]) -> PanelAnalysisArtifacts:
    df_panel = data_imputed.copy()
    df_panel["date"] = pd.to_datetime(df_panel["date"])
    df_panel["new_confirmed_clipped"] = df_panel["new_confirmed"].clip(lower=0)
    df_panel["cases_sum7"] = df_panel.groupby("location_key")["new_confirmed_clipped"].transform(
        lambda x: x.rolling(7, min_periods=1).sum()
    )
    df_panel["log_cases_sum7"] = np.log1p(df_panel["cases_sum7"])

    for col in [
        "mobility_retail_and_recreation",
        "mobility_grocery_and_pharmacy",
        "mobility_parks",
        "mobility_transit_stations",
        "mobility_workplaces",
        "mobility_residential",
        "average_temperature_celsius",
        "relative_humidity",
    ]:
        if col in df_panel.columns:
            df_panel[f"{col}_ma7"] = df_panel.groupby("location_key")[col].transform(
                lambda x: x.rolling(7, min_periods=1).mean()
            )

    df_panel = df_panel[df_panel["location_key"].isin(good_countries)].copy()

    panel_summary = pd.DataFrame(
        {
            "Metric": [
                "Number of countries",
                "Total country-day observations",
                "Start date",
                "End date",
                "Median days per country",
                "Mean days per country",
                "Netherlands observations",
            ],
            "Value": [
                df_panel["location_key"].nunique(),
                len(df_panel),
                df_panel["date"].min().date(),
                df_panel["date"].max().date(),
                int(df_panel.groupby("location_key").size().median()),
                round(df_panel.groupby("location_key").size().mean(), 1),
                int((df_panel["location_key"] == "NL").sum()) if "NL" in set(df_panel["location_key"]) else np.nan,
            ],
        }
    )

    country_target_summary = (
        df_panel.groupby("location_key")
        .agg(
            N=("cases_sum7", "size"),
            Mean_Cases7=("cases_sum7", "mean"),
            SD_Cases7=("cases_sum7", "std"),
            Median_Cases7=("cases_sum7", "median"),
            Min_Cases7=("cases_sum7", "min"),
            Max_Cases7=("cases_sum7", "max"),
        )
        .reset_index()
        .sort_values("Mean_Cases7", ascending=False)
    )

    similarity_rows = []
    if "NL" in set(df_panel["location_key"]):
        nl_series = (
            df_panel[df_panel["location_key"] == "NL"][["date", "log_cases_sum7"]]
            .rename(columns={"log_cases_sum7": "log_cases_sum7_nl"})
            .sort_values("date")
        )

        for country in sorted(df_panel["location_key"].unique()):
            if country == "NL":
                continue

            other = (
                df_panel[df_panel["location_key"] == country][["date", "log_cases_sum7"]]
                .rename(columns={"log_cases_sum7": "log_cases_sum7_other"})
                .sort_values("date")
            )

            merged = nl_series.merge(other, on="date", how="inner").dropna()
            if len(merged) >= 30:
                similarity_rows.append(
                    {
                        "location_key": country,
                        "Overlap_Days": len(merged),
                        "Corr_with_NL_log_cases7": merged["log_cases_sum7_nl"].corr(merged["log_cases_sum7_other"]),
                    }
                )

    similarity_df = pd.DataFrame(similarity_rows).sort_values(
        "Corr_with_NL_log_cases7",
        ascending=False,
    )

    desc_vars = [
        "cases_sum7",
        "mobility_retail_and_recreation_ma7",
        "mobility_transit_stations_ma7",
        "mobility_workplaces_ma7",
        "stringency_index",
        "average_temperature_celsius_ma7",
        "relative_humidity_ma7",
    ]

    desc_rows = []
    for variable in desc_vars:
        if variable in df_panel.columns:
            desc_rows.append(
                {
                    "Variable": variable,
                    "Mean": df_panel[variable].mean(),
                    "SD": df_panel[variable].std(),
                    "Median": df_panel[variable].median(),
                    "Min": df_panel[variable].min(),
                    "Max": df_panel[variable].max(),
                }
            )

    panel_desc_df = pd.DataFrame(desc_rows)

    return PanelAnalysisArtifacts(
        panel_summary=panel_summary,
        country_target_summary=country_target_summary,
        similarity_df=similarity_df,
        panel_desc_df=panel_desc_df,
    )
