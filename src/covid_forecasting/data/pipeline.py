from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from covid_forecasting.config import DateConfig, PanelConfig, ProjectPaths


@dataclass
class PanelDataBundle:
    merged_clean: pd.DataFrame
    data_imputed: pd.DataFrame
    good_countries: list[str]
    countries_found: list[str]
    missing_before: pd.DataFrame
    missing_after: pd.DataFrame
    mob_cols: list[str]
    gov_cols: list[str]
    weather_cols: list[str]


class PanelDataBuilder:
    def __init__(
        self,
        paths: ProjectPaths,
        date_config: DateConfig | None = None,
        panel_config: PanelConfig | None = None,
    ) -> None:
        self.paths = paths
        self.date_config = date_config or DateConfig()
        self.panel_config = panel_config or PanelConfig()

    def _load_data(self, path, iso_list: list[str] | tuple[str, ...]) -> pd.DataFrame:
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"])

        mask_date = (
            (df["date"] >= self.date_config.start_date)
            & (df["date"] <= self.date_config.end_date)
        )
        mask_country = df["location_key"].isin(iso_list) & (df["location_key"].str.len() == 2)
        return df.loc[mask_date & mask_country].copy()

    def _build_merged_panel(self) -> tuple[pd.DataFrame, list[str], list[str], list[str], list[str], pd.DataFrame, pd.DataFrame]:
        df_epi = self._load_data(self.paths.epidemiology_path, self.panel_config.iso_codes)
        countries_found = sorted(df_epi["location_key"].unique())

        df_mob = self._load_data(self.paths.mobility_path, countries_found)
        df_gov = self._load_data(self.paths.government_path, countries_found)
        df_weather = self._load_data(self.paths.weather_path, countries_found)

        weather_keep = [
            "location_key",
            "date",
            "average_temperature_celsius",
            "rainfall_mm",
            "relative_humidity",
        ]
        df_weather = df_weather[[col for col in weather_keep if col in df_weather.columns]]

        all_dates = pd.date_range(self.date_config.start_date, self.date_config.end_date)
        skeleton = pd.MultiIndex.from_product(
            [countries_found, all_dates],
            names=["location_key", "date"],
        ).to_frame(index=False)

        merged = (
            skeleton
            .merge(df_epi, on=["location_key", "date"], how="left")
            .merge(df_mob, on=["location_key", "date"], how="left", suffixes=("", "_mob"))
            .merge(df_gov, on=["location_key", "date"], how="left", suffixes=("", "_gov"))
            .merge(df_weather, on=["location_key", "date"], how="left", suffixes=("", "_weather"))
        )

        mob_cols = [c for c in merged.columns if c in df_mob.columns and c not in ["location_key", "date"]]
        gov_cols = [c for c in merged.columns if c in df_gov.columns and c not in ["location_key", "date"]]
        weather_cols = [c for c in merged.columns if c in df_weather.columns and c not in ["location_key", "date"]]

        miss_mob = merged.groupby("location_key")[mob_cols].apply(lambda x: x.isna().mean().mean() * 100)
        miss_gov = merged.groupby("location_key")[gov_cols].apply(lambda x: x.isna().mean().mean() * 100)
        miss_weather = merged.groupby("location_key")[weather_cols].apply(lambda x: x.isna().mean().mean() * 100)

        good_countries = (
            miss_mob[miss_mob <= self.panel_config.missing_threshold].index
            .intersection(miss_gov[miss_gov <= self.panel_config.missing_threshold].index)
            .intersection(miss_weather[miss_weather <= self.panel_config.missing_threshold].index)
        )

        stats_before = pd.DataFrame(
            [
                ["Mobility", miss_mob.mean(), miss_mob.median()],
                ["Government response", miss_gov.mean(), miss_gov.median()],
                ["Weather", miss_weather.mean(), miss_weather.median()],
            ],
            columns=["Block", "Mean_Before", "Median_Before"],
        )
        stats_after = pd.DataFrame(
            [
                ["Mobility", miss_mob[good_countries].mean(), miss_mob[good_countries].median()],
                ["Government response", miss_gov[good_countries].mean(), miss_gov[good_countries].median()],
                ["Weather", miss_weather[good_countries].mean(), miss_weather[good_countries].median()],
            ],
            columns=["Block", "Mean_After", "Median_After"],
        )

        merged_clean = merged[merged["location_key"].isin(good_countries)].copy()
        return (
            merged_clean,
            sorted(good_countries),
            countries_found,
            mob_cols,
            gov_cols,
            weather_cols,
            stats_before,
            stats_after,
        )

    def _inject_rivm_ground_truth(self, merged_clean: pd.DataFrame) -> pd.DataFrame:
        rivm_raw_1 = pd.read_csv(self.paths.rivm_cases_path, sep=";")
        rivm_raw_2 = pd.read_csv(self.paths.rivm_cases_2_path, sep=";")
        rivm_raw = pd.concat([rivm_raw_1, rivm_raw_2], ignore_index=True)

        rivm_raw["Date_of_publication"] = pd.to_datetime(
            rivm_raw["Date_of_publication"],
            errors="coerce",
        )
        rivm_raw["Total_reported"] = (
            pd.to_numeric(rivm_raw["Total_reported"], errors="coerce").fillna(0)
        )

        rivm_daily_nl = (
            rivm_raw.groupby("Date_of_publication", as_index=False)["Total_reported"]
            .sum()
            .rename(
                columns={
                    "Date_of_publication": "date",
                    "Total_reported": "new_confirmed_rivm",
                }
            )
        )

        rivm_daily_nl = rivm_daily_nl[
            (rivm_daily_nl["date"] >= self.date_config.start_date)
            & (rivm_daily_nl["date"] <= self.date_config.end_date)
        ].copy()
        rivm_daily_nl["location_key"] = "NL"

        merged_clean = merged_clean.merge(
            rivm_daily_nl[["location_key", "date", "new_confirmed_rivm"]],
            on=["location_key", "date"],
            how="left",
        )

        nl_replace_mask = (
            (merged_clean["location_key"] == "NL")
            & (merged_clean["new_confirmed_rivm"].notna())
        )
        merged_clean.loc[nl_replace_mask, "new_confirmed"] = merged_clean.loc[
            nl_replace_mask,
            "new_confirmed_rivm",
        ]
        return merged_clean.drop(columns=["new_confirmed_rivm"])

    def _impute_panel(self, merged_clean: pd.DataFrame) -> pd.DataFrame:
        data = merged_clean[list(self.panel_config.cols_to_keep)].copy()
        data["date"] = pd.to_datetime(data["date"])

        all_dates = pd.date_range(
            start=self.date_config.start_date,
            end=self.date_config.end_date,
            freq="D",
        )
        countries = data["location_key"].unique()
        full_index = pd.MultiIndex.from_product(
            [countries, all_dates],
            names=["location_key", "date"],
        )

        df_imp = (
            data
            .set_index(["location_key", "date"])
            .reindex(full_index)
            .sort_index()
        )
        df_imp = df_imp.groupby(level=0).ffill()
        return df_imp.reset_index()

    def build(self) -> PanelDataBundle:
        (
            merged_clean,
            good_countries,
            countries_found,
            mob_cols,
            gov_cols,
            weather_cols,
            stats_before,
            stats_after,
        ) = self._build_merged_panel()
        merged_clean = self._inject_rivm_ground_truth(merged_clean)
        data_imputed = self._impute_panel(merged_clean)

        return PanelDataBundle(
            merged_clean=merged_clean,
            data_imputed=data_imputed,
            good_countries=good_countries,
            countries_found=countries_found,
            missing_before=stats_before,
            missing_after=stats_after,
            mob_cols=mob_cols,
            gov_cols=gov_cols,
            weather_cols=weather_cols,
        )
