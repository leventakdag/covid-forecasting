from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from covid_forecasting.config import TCNSequenceConfig, TCNTrainingConfig
from covid_forecasting.features.tcn import TCNSequenceBundle
from covid_forecasting.utils import ensure_dir, rmse, set_seed


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, dilation, dropout=0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.chomp2,
            self.relu2,
            self.dropout2,
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNRegressor(nn.Module):
    def __init__(self, num_features, num_channels=(32, 32, 32), kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        for i in range(len(num_channels)):
            dilation_size = 2 ** i
            in_channels = num_features if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    dilation=dilation_size,
                    dropout=dropout,
                )
            )
        self.network = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], 1)

    def forward(self, x):
        y = self.network(x)
        return self.linear(y[:, :, -1])


@dataclass
class TCNExperiment:
    save_dir: Path
    sequence_config: TCNSequenceConfig | None = None
    training_config: TCNTrainingConfig | None = None

    def __post_init__(self) -> None:
        self.sequence_config = self.sequence_config or TCNSequenceConfig()
        self.training_config = self.training_config or TCNTrainingConfig()
        ensure_dir(self.save_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _scope_settings(self, scope: str) -> dict[str, object]:
        if scope == "local":
            return {
                "batch_size": self.training_config.local_batch_size,
                "max_epochs": self.training_config.local_max_epochs,
                "lr": self.training_config.local_lr,
                "patience": self.training_config.local_patience,
                "weight_decay": self.training_config.local_weight_decay,
                "channels": self.training_config.local_channels,
                "kernel_size": self.training_config.local_kernel_size,
                "dropout": self.training_config.local_dropout,
            }
        return {
            "batch_size": self.training_config.global_batch_size,
            "max_epochs": self.training_config.global_max_epochs,
            "lr": self.training_config.global_lr,
            "patience": self.training_config.global_patience,
            "weight_decay": self.training_config.global_weight_decay,
            "channels": self.training_config.global_channels,
            "kernel_size": self.training_config.global_kernel_size,
            "dropout": self.training_config.global_dropout,
        }

    def _build_model(self, num_features: int, settings: dict[str, object]) -> TCNRegressor:
        return TCNRegressor(
            num_features=num_features,
            num_channels=settings["channels"],
            kernel_size=settings["kernel_size"],
            dropout=settings["dropout"],
        ).to(self.device)

    def _scale_sequences(
        self,
        X_train_raw: np.ndarray,
        y_train_raw: np.ndarray,
        X_test_raw: np.ndarray,
        locs_train: np.ndarray,
        test_locs: np.ndarray,
    ):
        X_train_sc = np.zeros_like(X_train_raw)
        X_test_sc = np.zeros_like(X_test_raw)
        y_train_sc = np.zeros_like(y_train_raw)

        nl_target_mean, nl_target_std = 0.0, 1.0

        for loc in np.unique(locs_train):
            l_mask = locs_train == loc
            X_loc = X_train_raw[l_mask]
            n_rows, lookback, n_features = X_loc.shape

            scaler_x = StandardScaler()
            X_train_sc[l_mask] = scaler_x.fit_transform(X_loc.reshape(-1, n_features)).reshape(
                n_rows,
                lookback,
                n_features,
            )

            scaler_y = StandardScaler()
            y_train_sc[l_mask] = scaler_y.fit_transform(y_train_raw[l_mask].reshape(-1, 1))

            if loc == self.sequence_config.test_country:
                nl_target_mean = scaler_y.mean_[0]
                nl_target_std = scaler_y.scale_[0]

                t_mask = test_locs == loc
                if t_mask.any():
                    X_test_loc = X_test_raw[t_mask]
                    n_test, _, _ = X_test_loc.shape
                    X_test_sc[t_mask] = scaler_x.transform(
                        X_test_loc.reshape(-1, n_features)
                    ).reshape(n_test, lookback, n_features)

        return X_train_sc, y_train_sc, X_test_sc, nl_target_mean, nl_target_std

    def _find_optimal_epochs(
        self,
        X_train_t: torch.Tensor,
        y_train_t: torch.Tensor,
        anchor_dates: np.ndarray,
        settings: dict[str, object],
        current_seed: int,
    ) -> int:
        batch_size = int(settings["batch_size"])
        n_train = len(X_train_t)
        if n_train <= batch_size:
            return 30

        order = np.argsort(pd.to_datetime(anchor_dates).values)
        X_train_ord = X_train_t[order]
        y_train_ord = y_train_t[order]
        split_idx = min(max(batch_size, int(n_train * 0.8)), n_train - batch_size)

        X_fit = X_train_ord[:split_idx]
        y_fit = y_train_ord[:split_idx]
        X_val = X_train_ord[split_idx:]
        y_val = y_train_ord[split_idx:]

        set_seed(current_seed)
        model = self._build_model(X_train_t.shape[1], settings)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(settings["lr"]),
            weight_decay=float(settings["weight_decay"]),
        )

        generator = torch.Generator(device="cpu")
        generator.manual_seed(current_seed)
        train_loader = DataLoader(
            TensorDataset(X_fit, y_fit),
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
        )

        best_loss = float("inf")
        patience_counter = 0
        optimal_epochs = 0

        for epoch in range(1, int(settings["max_epochs"]) + 1):
            model.train()
            for xb, yb in train_loader:
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_loss = criterion(model(X_val), y_val).item()

            if val_loss < best_loss:
                best_loss = val_loss
                optimal_epochs = epoch
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= int(settings["patience"]):
                break

        return 1 if optimal_epochs == 0 else optimal_epochs

    def _train_full_model(
        self,
        X_train_t: torch.Tensor,
        y_train_t: torch.Tensor,
        settings: dict[str, object],
        current_seed: int,
        optimal_epochs: int,
    ) -> TCNRegressor:
        set_seed(current_seed)
        model = self._build_model(X_train_t.shape[1], settings)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(settings["lr"]),
            weight_decay=float(settings["weight_decay"]),
        )
        full_loader = DataLoader(
            TensorDataset(X_train_t, y_train_t),
            batch_size=int(settings["batch_size"]),
            shuffle=True,
        )

        model.train()
        for _ in range(1, optimal_epochs + 1):
            for xb, yb in full_loader:
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()

        return model

    def run_scope(self, sequences: TCNSequenceBundle, scope: str) -> list[pd.DataFrame]:
        settings = self._scope_settings(scope)
        outputs: list[pd.DataFrame] = []

        for run_i in range(self.training_config.n_runs):
            current_seed = self.training_config.base_seed + run_i
            results_store = []
            current_train_end = np.datetime64(pd.Timestamp("2021-02-15"))
            final_date = np.datetime64(pd.Timestamp("2022-04-24"))

            while True:
                anchor_start = current_train_end
                anchor_end = current_train_end + np.timedelta64(self.sequence_config.step_days, "D")
                earliest_target = anchor_start + np.timedelta64(self.sequence_config.horizon, "D")

                if earliest_target > final_date:
                    break

                train_start_limit = anchor_start - np.timedelta64(365, "D")

                if scope == "local":
                    train_mask = (
                        (sequences.dates_arr <= anchor_start)
                        & (sequences.dates_arr >= train_start_limit)
                        & (sequences.locs_arr == self.sequence_config.test_country)
                    )
                else:
                    train_mask = (
                        (sequences.dates_arr <= anchor_start)
                        & (sequences.dates_arr >= train_start_limit)
                    )

                test_mask = (
                    (sequences.anchor_dates_arr >= anchor_start)
                    & (sequences.anchor_dates_arr < anchor_end)
                    & (sequences.dates_arr <= final_date)
                    & (sequences.locs_arr == self.sequence_config.test_country)
                )

                if train_mask.sum() < int(settings["batch_size"]) or test_mask.sum() == 0:
                    current_train_end += np.timedelta64(self.sequence_config.step_days, "D")
                    continue

                if train_mask.any() and test_mask.any():
                    max_train_anchor = sequences.anchor_dates_arr[train_mask].max()
                    min_test_anchor = sequences.anchor_dates_arr[test_mask].min()
                    if not (max_train_anchor < min_test_anchor):
                        raise ValueError(
                            f"Leakage risk: max_train_anchor={max_train_anchor} >= min_test_anchor={min_test_anchor}"
                        )

                set_seed(current_seed)

                X_train_raw = sequences.X[train_mask]
                y_train_raw = sequences.y[train_mask]
                locs_train = sequences.locs_arr[train_mask]

                X_test_raw = sequences.X[test_mask]
                y_test_raw = sequences.y[test_mask]
                test_locs = sequences.locs_arr[test_mask]
                test_dates = sequences.dates_arr[test_mask]

                (
                    X_train_sc,
                    y_train_sc,
                    X_test_sc,
                    nl_target_mean,
                    nl_target_std,
                ) = self._scale_sequences(
                    X_train_raw=X_train_raw,
                    y_train_raw=y_train_raw,
                    X_test_raw=X_test_raw,
                    locs_train=locs_train,
                    test_locs=test_locs,
                )

                X_train_t = torch.tensor(X_train_sc, dtype=torch.float32).permute(0, 2, 1).to(self.device)
                y_train_t = torch.tensor(y_train_sc, dtype=torch.float32).to(self.device)
                X_test_t = torch.tensor(X_test_sc, dtype=torch.float32).permute(0, 2, 1).to(self.device)

                optimal_epochs = self._find_optimal_epochs(
                    X_train_t=X_train_t,
                    y_train_t=y_train_t,
                    anchor_dates=sequences.anchor_dates_arr[train_mask],
                    settings=settings,
                    current_seed=current_seed,
                )

                model = self._train_full_model(
                    X_train_t=X_train_t,
                    y_train_t=y_train_t,
                    settings=settings,
                    current_seed=current_seed,
                    optimal_epochs=optimal_epochs,
                )

                model.eval()
                with torch.no_grad():
                    y_pred_scaled = model(X_test_t).cpu().numpy().flatten()

                y_pred_diff = (y_pred_scaled * nl_target_std) + nl_target_mean
                y_actual_diff = y_test_raw.flatten()

                for i, target_date in enumerate(test_dates):
                    ts = pd.Timestamp(target_date)
                    anchor_ts = ts - pd.Timedelta(days=self.sequence_config.horizon)
                    naive_val = sequences.baseline_lookup.get(ts, {}).get("naive_pred", np.nan)

                    if pd.isna(naive_val):
                        continue

                    naive_log = np.log1p(naive_val)
                    pred_log_recon = naive_log + y_pred_diff[i]
                    act_log_recon = naive_log + y_actual_diff[i]

                    pred_cases = np.clip(np.expm1(pred_log_recon), 0, None)
                    actual_cases = np.clip(np.expm1(act_log_recon), 0, None)

                    existing_target_dates = {row[1] for row in results_store}
                    if ts in existing_target_dates:
                        raise ValueError(f"Batch overlap detected: duplicate Target_Date in current step: {ts}")

                    results_store.append(
                        [run_i + 1, ts, anchor_ts, actual_cases, pred_cases, naive_val]
                    )

                current_train_end += np.timedelta64(self.sequence_config.step_days, "D")

            if len(results_store) == 0:
                continue

            res_df = pd.DataFrame(
                results_store,
                columns=["Run", "Target_Date", "Anchor_Date", "Actual", "TCN", "Naive"],
            )
            res_df = res_df.sort_values("Target_Date").dropna()

            if scope == "local":
                file_name = f"preds_local_{sequences.feature_tag}_seed_{current_seed}.csv"
            else:
                file_name = f"preds_{sequences.feature_tag}_seed_{current_seed}.csv"

            res_df.to_csv(self.save_dir / file_name, index=False)
            outputs.append(res_df)

        return outputs

    @staticmethod
    def score(predictions: pd.DataFrame) -> dict[str, float]:
        act_vals = predictions["Actual"].values
        model_preds = predictions["TCN"].values
        naive_preds = predictions["Naive"].values
        return {
            "rmse": rmse(act_vals, model_preds),
            "r2": float(r2_score(act_vals, model_preds)),
            "naive_rmse": rmse(act_vals, naive_preds),
            "naive_r2": float(r2_score(act_vals, naive_preds)),
        }
