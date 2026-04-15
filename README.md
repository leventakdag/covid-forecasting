# covid-forecasting

This repository contains the codebase for my thesis on 7-day-ahead forecasting of COVID-19 cases in the Netherlands using European panel data. The project compares statistical, machine learning, and deep learning models in a walk-forward forecasting setup.

## Project Summary

- Forecast target: Dutch COVID-19 case counts on a 7-day horizon
- Data: European country-level panel data used for cross-country learning
- Model families: ARIMA, Elastic Net, XGBoost, and a Temporal Convolutional Network (TCN)
- Features: epidemiology, mobility, government response, weather, and RIVM case data

## Repository Structure

```text
.
|-- pyproject.toml
|-- environment.yml
|-- README.md
|-- scripts/
|   |-- run_baselines.py
|   |-- run_tcn.py
|   `-- run_analysis.py
|-- src/
|   `-- covid_forecasting/
|       |-- config.py
|       |-- utils.py
|       |-- data/
|       |-- features/
|       |-- models/
|       `-- analysis/
|-- tests/
|   `-- test_metrics.py
|-- notebooks/
|   |-- README.md
|   `-- legacy/
|-- data/
|   `-- README.md
`-- preds/
    |-- README.md
    `-- preds/
        `-- *.csv
```

## Package Design

The project is split into reusable components:

- `data/`: panel loading, filtering, RIVM integration, and imputation
- `features/`: supervised lag features for classical models and 3D sequence generation for the TCN
- `models/`: object-oriented experiment runners for Elastic Net, XGBoost, ARIMA, and TCN
- `analysis/`: panel summary tables and result-table generation from saved prediction files
- `scripts/`: thin entry points for running the main workflows

## Workflow Entry Points

### Run the classical baselines

```bash
python scripts/run_baselines.py --data-dir data --output-dir preds/preds
```

### Run the TCN experiments

```bash
python scripts/run_tcn.py --data-dir data --output-dir preds/preds
```

### Generate summary tables

```bash
python scripts/run_analysis.py --data-dir data --output-dir preds/preds --report-dir reports
```

## Environment Setup

This project uses Conda.

```bash
conda env create -f environment.yml
conda activate covid-forecasting
```

The environment installs the local package in editable mode, so the `src/` modules and `scripts/` entry points work immediately after setup.

## Data Notes

Raw input datasets are not included in the repository. The expected files are documented in `data/README.md`.

To rerun the project locally, place the following files inside the configured data directory:

- `epidemiology.csv`
- `mobility.csv`
- `oxford-government-response.csv`
- `weather.csv`
- `RIVM_cases.csv`
- `RIVM_cases2.csv`
