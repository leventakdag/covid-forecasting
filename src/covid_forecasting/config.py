from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


EUROPE_ISO2 = (
    "AL", "AD", "AM", "AT", "AZ", "BY", "BE", "BA", "BG", "HR", "CY", "CZ",
    "DK", "EE", "FI", "FR", "GE", "DE", "GR", "HU", "IS", "IE", "IT", "LV",
    "LI", "LT", "LU", "MT", "MD", "MC", "ME", "NL", "MK", "NO", "PL", "PT",
    "RO", "SM", "RS", "SK", "SI", "ES", "SE", "CH", "UA", "GB", "KZ", "XK",
)


@dataclass(frozen=True)
class DateConfig:
    start: str = "2020-03-01"
    cutoff: str = "2021-02-15"
    end: str = "2022-04-24"

    @property
    def start_date(self) -> pd.Timestamp:
        return pd.Timestamp(self.start)

    @property
    def cutoff_date(self) -> pd.Timestamp:
        return pd.Timestamp(self.cutoff)

    @property
    def end_date(self) -> pd.Timestamp:
        return pd.Timestamp(self.end)


@dataclass(frozen=True)
class ProjectPaths:
    data_dir: Path = Path("data")
    output_dir: Path = Path("preds/preds")

    @property
    def epidemiology_path(self) -> Path:
        return self.data_dir / "epidemiology.csv"

    @property
    def mobility_path(self) -> Path:
        return self.data_dir / "mobility.csv"

    @property
    def government_path(self) -> Path:
        return self.data_dir / "oxford-government-response.csv"

    @property
    def weather_path(self) -> Path:
        return self.data_dir / "weather.csv"

    @property
    def rivm_cases_path(self) -> Path:
        return self.data_dir / "RIVM_cases.csv"

    @property
    def rivm_cases_2_path(self) -> Path:
        return self.data_dir / "RIVM_cases2.csv"


@dataclass(frozen=True)
class PanelConfig:
    iso_codes: tuple[str, ...] = EUROPE_ISO2
    missing_threshold: float = 10.0
    cols_to_keep: tuple[str, ...] = (
        "location_key",
        "date",
        "new_confirmed",
        "mobility_retail_and_recreation",
        "mobility_grocery_and_pharmacy",
        "mobility_parks",
        "mobility_transit_stations",
        "mobility_workplaces",
        "mobility_residential",
        "stringency_index",
        "average_temperature_celsius",
        "relative_humidity",
    )


@dataclass(frozen=True)
class RollingExperimentConfig:
    test_country: str = "NL"
    horizon: int = 7
    step_days: int = 7
    training_window_days: int = 365


@dataclass(frozen=True)
class HybridFeatureConfig:
    base_config: str = "ar"
    use_exog: bool = True
    dense_lags: tuple[int, ...] = tuple(range(0, 21))
    snapshot_lags: tuple[int, ...] = (0, 7, 14)
    horizon: int = 7


def default_xgboost_local_param_grid() -> list[dict[str, float]]:
    return [
        {"max_depth": 2, "min_child_weight": 4, "learning_rate": 0.05, "reg_alpha": 1.0, "reg_lambda": 10.0, "colsample_bytree": 0.6},
        {"max_depth": 3, "min_child_weight": 4, "learning_rate": 0.05, "reg_alpha": 1.0, "reg_lambda": 10.0, "colsample_bytree": 0.6},
        {"max_depth": 2, "min_child_weight": 6, "learning_rate": 0.03, "reg_alpha": 2.0, "reg_lambda": 15.0, "colsample_bytree": 0.5},
        {"max_depth": 3, "min_child_weight": 6, "learning_rate": 0.03, "reg_alpha": 2.0, "reg_lambda": 15.0, "colsample_bytree": 0.5},
        {"max_depth": 4, "min_child_weight": 2, "learning_rate": 0.05, "reg_alpha": 0.5, "reg_lambda": 5.0, "colsample_bytree": 0.8},
        {"max_depth": 3, "min_child_weight": 2, "learning_rate": 0.05, "reg_alpha": 0.5, "reg_lambda": 5.0, "colsample_bytree": 0.8},
    ]


def default_xgboost_global_param_grid() -> list[dict[str, float]]:
    return [
        {"max_depth": 3, "min_child_weight": 10, "learning_rate": 0.05, "reg_alpha": 0.1, "reg_lambda": 1.0, "colsample_bytree": 0.8},
        {"max_depth": 3, "min_child_weight": 20, "learning_rate": 0.05, "reg_alpha": 0.5, "reg_lambda": 3.0, "colsample_bytree": 0.8},
        {"max_depth": 4, "min_child_weight": 10, "learning_rate": 0.05, "reg_alpha": 0.1, "reg_lambda": 1.0, "colsample_bytree": 0.8},
        {"max_depth": 4, "min_child_weight": 20, "learning_rate": 0.03, "reg_alpha": 0.5, "reg_lambda": 3.0, "colsample_bytree": 0.7},
        {"max_depth": 5, "min_child_weight": 10, "learning_rate": 0.03, "reg_alpha": 0.1, "reg_lambda": 1.0, "colsample_bytree": 0.7},
        {"max_depth": 5, "min_child_weight": 20, "learning_rate": 0.05, "reg_alpha": 0.5, "reg_lambda": 5.0, "colsample_bytree": 0.8},
    ]


@dataclass(frozen=True)
class TCNSequenceConfig:
    use_exog: bool = True
    horizon: int = 7
    step_days: int = 7
    lookback_lags: int = 21
    test_country: str = "NL"


@dataclass(frozen=True)
class TCNTrainingConfig:
    n_runs: int = 10
    base_seed: int = 0
    local_batch_size: int = 16
    local_max_epochs: int = 100
    local_lr: float = 1e-4
    local_patience: int = 10
    local_weight_decay: float = 1e-3
    local_channels: tuple[int, ...] = (16, 16)
    local_kernel_size: int = 3
    local_dropout: float = 0.4
    global_batch_size: int = 128
    global_max_epochs: int = 100
    global_lr: float = 1e-3
    global_patience: int = 10
    global_weight_decay: float = 0.0
    global_channels: tuple[int, ...] = (32, 32, 32)
    global_kernel_size: int = 3
    global_dropout: float = 0.2


@dataclass(frozen=True)
class AnalysisConfig:
    base_seed: int = 0
    n_runs: int = 10
    core_regimes: tuple[dict[str, str], ...] = field(
        default_factory=lambda: (
            {"name": "Alpha", "start": "2021-02-22", "end": "2021-06-13"},
            {"name": "Delta", "start": "2021-07-05", "end": "2021-12-19"},
            {"name": "Omicron", "start": "2022-01-03", "end": "2022-04-24"},
        )
    )
