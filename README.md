# covid-forecasting

This repository contains the codebase for my thesis on one-week-ahead forecasting of the 7-day rolling sum of newly confirmed COVID-19 cases in the Netherlands. The main research question is whether Dutch forecasts improve when models are trained on a multi-country panel from the WHO European Region and when external information is added beyond past case counts.

## Project Summary

- Forecast target: one-week-ahead forecasts of the 7-day rolling sum of newly confirmed Dutch COVID-19 cases
- Main comparison 1: Dutch-only (`local`) training versus multi-country (`global`) training
- Main comparison 2: autoregressive (`AR`) specifications versus feature-rich (`full`) specifications with mobility, weather, and government response data
- Evaluation design: weekly walk-forward forecasting over a daily panel spanning 1 March 2020 to 24 April 2022
- Geographic scope: countries in the WHO European Region, with final evaluation on the Netherlands
- Model families used in the study: Elastic Net, XGBoost, and a Temporal Convolutional Network (TCN), with ARIMA retained as a baseline
- Data sources: epidemiology, mobility, government response, weather, and RIVM case data

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
|-- data/
|   `-- README.md
`-- preds/
    |-- README.md
    `-- preds/
        `-- *.csv
```

## Package Design

- `data/`: panel loading, filtering, RIVM integration, and imputation
- `features/`: autoregressive and exogenous feature construction for classical models, plus 3D sequence generation for the TCN
- `models/`: object-oriented experiment runners for Dutch-only versus multi-country forecasting with Elastic Net, XGBoost, ARIMA, and TCN
- `analysis/`: panel summary tables and result-table generation for local/global and AR/full comparisons
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


```bash
conda env create -f environment.yml
conda activate covid-forecasting
```


## Data Notes

Raw input datasets are not included in the repository. The expected files are documented in `data/README.md`.

To rerun the project locally, place the following files inside the configured data directory:

- `epidemiology.csv`
- `mobility.csv`
- `oxford-government-response.csv`
- `weather.csv`
- `RIVM_cases.csv`
- `RIVM_cases2.csv`

Sources of data: 

- `https://health.google.com/covid-19/open-data/`
- `https://data.rivm.nl/covid-19/`
