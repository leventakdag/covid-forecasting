from __future__ import annotations

import numpy as np

from covid_forecasting.analysis.reporting import smape
from covid_forecasting.utils import rmse


def test_rmse_is_zero_for_identical_arrays() -> None:
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    assert rmse(y_true, y_pred) == 0.0


def test_smape_is_zero_for_identical_arrays() -> None:
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([10.0, 20.0, 30.0])
    assert smape(y_true, y_pred) == 0.0


def test_smape_handles_nonzero_error() -> None:
    y_true = np.array([10.0, 10.0])
    y_pred = np.array([20.0, 10.0])
    assert smape(y_true, y_pred) > 0.0
