# An Open, Operational Probabilistic Forecasting Pipeline for Day-Ahead Electricity Prices

This repository is the code base for the paper *An Open, Operational
Probabilistic Forecasting Pipeline for Day-Ahead Electricity Prices*. It
produces 15-minute point and probabilistic day-ahead electricity-price
forecasts for the EPEX DE-LU bidding zone using LEAR point models and SQRA
probabilistic post-processing.

The repository supports two separate workflows:

- `run_pipeline.py` is the fully open, automated, operational pipeline. It
  maintains the required market and ICON-D2 inputs, estimates the point and
  probabilistic models, generates the next day-ahead forecasts, and can submit
  both forecasts to Energy Arena.
- `run_full_experiment.py` and `run_full_evaluation.py` reproduce the paper's
  experimental analysis. The first script generates the historical point and
  SQRA forecasts; the second evaluates the completed forecasts and creates the
  paper's tables and figures.

All automated computation is implemented in ordinary Python modules. The
notebooks are optional interfaces for interactive inspection and do not contain
an alternative pipeline implementation.

## Repository structure

```text
DA_Price_Forecasting_Pipeline_DE_LU/
├── run_pipeline.py                 # daily operational entry point
├── prepare_dwd_data.py             # daily ICON-D2 preparation
├── run_full_experiment.py          # paper point and SQRA experiments
├── run_full_evaluation.py          # paper evaluation, tables, and figures
├── pipeline/
│   ├── operational/                # downloads, validation, submission, locking
│   ├── lear/
│   │   ├── lear_model.py           # reusable LEAR and feature functions
│   │   ├── run_lear.py             # LEAR/ANC command-line entry point
│   │   └── lear_pipeline.ipynb     # optional interactive interface
│   ├── sqra/
│   │   ├── sqra_model.py           # reusable SQRA functions
│   │   ├── run_sqra.py             # SQRA command-line entry point
│   │   └── sqra_pipeline.ipynb     # optional interactive interface
│   ├── delivery_index.py           # canonical delivery-day/MTU indexing
│   └── dwd_history.py              # appendable ICON-D2 history format
├── evaluation/
│   ├── evaluation_core.py          # metrics and statistical tests
│   ├── run_evaluation.py           # evaluation command-line entry point
│   ├── evaluation.ipynb            # optional interactive interface
│   └── ...                         # coverage checks and Excel exports
├── preprocessing/                  # historical ERA5 and ICON-D2 preprocessing
├── visualization/                  # publication figures and tables
├── data/
│   ├── clustering/                 # spatial ICON-D2 cluster definitions
│   ├── icon/                       # consolidated operational weather histories
│   └── shapefile/                  # map assets
├── experiment_config.py            # shared paper-experiment configuration
├── experiment_manifest.py          # canonical LEAR and SQRA configurations
├── .env.example                    # configuration template
└── requirements.txt                # one environment for the complete repository
```

## Setup

### Python environment

Python 3.9 is recommended. In PowerShell, from the repository root:

```powershell
py -3.9 -m venv forecast
.\forecast\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r .\requirements.txt
```

The single `requirements.txt` covers the operational pipeline, paper
experiments, evaluation, visualizations, and optional notebooks. Registering a
Jupyter kernel is optional:

```powershell
python -m ipykernel install --user --name forecast --display-name "forecast (Python 3.9)"
```

### Environment configuration

Create the local configuration file:

```powershell
Copy-Item .env.example .env
```

At minimum, configure:

```dotenv
ENTSOE_API_KEY=...
ENERGY_ARENA_API_KEY=...

ICON_DATA_ROOT=data/icon
DWD_OPERATIONAL_RAW_ROOT=data/dwd_raw
OPERATIONAL_MARKET_DATA_ROOT=data/market
OPERATIONAL_RESULTS_ROOT=results/operational
OPERATIONAL_OUTPUT_ROOT=output/operational
```

`ENTSOE_API_KEY` is required for DE-LU prices, EXAA prices, and load forecasts.
`ENERGY_ARENA_API_KEY` is required only when forecasts are submitted. Input,
cache, result, and output paths can be absolute or relative to the repository
root. 

Before the first operational run, inspect the resolved configuration and live
Energy Arena target:

```powershell
python run_pipeline.py --check-setup
python run_pipeline.py --dry-run
```

## Daily operational pipeline

### Operational data

The operational pipeline uses only the data required for the next daily
forecast and its rolling calibration:

| Input | Purpose | Acquisition |
|---|---|---|
| EPEX DE-LU day-ahead prices | LEAR targets and lagged price features | ENTSO-E API |
| ENTSO-E DE-LU load forecast | Fundamental-model feature | ENTSO-E API |
| EXAA day-ahead prices | EXAA-Enriched and EXAA-Only features | ENTSO-E API, sequence 2 |
| DWD ICON-D2 forecasts | Wind and solar features | DWD Open Data |
| ICON-D2 cluster definitions | Spatial aggregation for `C=1,5,25` | Versioned under `data/clustering` |

The market caches are updated automatically whenever coverage is missing:

```text
OPERATIONAL_MARKET_DATA_ROOT/
├── entsoe/
│   ├── prices_da.csv
│   └── load_forecast.csv
└── exaa/
    └── prices_exaa.csv
```

ENTSO-E prices are always maintained. Load forecasts are required by
weather-based models, while EXAA prices are loaded only when the selected model
uses them or `DOWNLOAD_EXAA=true`. Optional variables
`OPERATIONAL_ENTSOE_PRICE_CACHE_DIR`,
`OPERATIONAL_ENTSOE_LOAD_CACHE_DIR`, and `OPERATIONAL_EXAA_CACHE_DIR` can place
the three caches elsewhere. When an ENTSO-E or EXAA request fails, the pipeline
waits `MARKET_DATA_RETRY_SECONDS` (300 seconds by default) before its next
attempt. EXAA-Only makes eight total EXAA attempts. The Fundamental pipeline
makes four total load-forecast attempts and, if the fourth still fails,
continues with a consistent feature set that omits the load block.

### Maintaining the ICON-D2 calibration history

The DWD operational archive does not provide the complete sequence of past
ICON-D2 forecasts needed to reconstruct a rolling model calibration later.
Consequently, the relevant forecast run must be collected every day. The
repository downloads the preceding 06 UTC ICON-D2 run, verifies it, aggregates
it to the spatial resolutions `C=1,5,25`, and appends the new delivery day to
the persistent weather histories.

The consolidated cluster histories under the default `data/icon` path are
updated and published to this GitHub repository every day. A fresh clone
therefore contains the collected historical ICON-D2 cluster information up to
the repository's latest update. Users do not need to obtain all past raw
forecasts themselves before calibrating the model; they only need to continue
the daily download and preparation process for new delivery days. The large raw
GRIB/ZIP downloads are not versioned—only the compact, preprocessed Parquet
histories required by the forecasting models are maintained in the repository.

This produces four appendable Parquet histories for every spatial resolution:

```text
ICON_DATA_ROOT/
├── c1/
│   ├── u10.parquet
│   ├── v10.parquet
│   ├── ASWDIR_S.parquet
│   ├── ASWDIFD_S.parquet
│   └── dwd_history.json
├── c5/
│   ├── u10.parquet
│   ├── v10.parquet
│   ├── ASWDIR_S.parquet
│   ├── ASWDIFD_S.parquet
│   └── dwd_history.json
└── c25/
    ├── u10.parquet
    ├── v10.parquet
    ├── ASWDIR_S.parquet
    ├── ASWDIFD_S.parquet
    └── dwd_history.json
```

`u10` and `v10` contain the wind components. `ASWDIR_S` and `ASWDIFD_S`
contain direct and diffuse short-wave radiation. Every history records the
delivery date, issue date, run hour, valid timestamp, and spatial cluster
values. `dwd_history.json` records and validates the history coverage.

Prepare tomorrow's weather data independently before model execution:

```powershell
python .\prepare_dwd_data.py
```

The preparation script performs the following steps:

1. download or locate only the four required ICON-D2 variables for the 06 UTC
   run;
2. validate the ZIP, BZip2, and GRIB content;
3. retry failed or incomplete downloads;
4. preprocess all required fields for `C=1,5,25`;
5. deduplicate repeated valid timestamps;
6. append or replace the target delivery day in each Parquet history;
7. verify all twelve updated histories; and
8. delete the consumed raw data only after successful validation.

The default Fundamental/SQRA system needs approximately 116 preceding delivery
days for its first complete run: 56 days for LEAR plus 60 days for SQRA. After
that bootstrap, one new weather day is appended on every operational day.

### Running the complete daily pipeline

The default configuration follows the paper's selected operational setup:

- SQRA inputs: Fundamental LEAR forecasts with ICON-D2, `D_LEAR=56`, and
  `C={1,5,25}`;
- probabilistic forecast: one pooled SQRA fit per requested quantile using the
  preceding 60 delivery days and all valid MTUs;
- point submission: the SQRA median forecast (`q=0.5`); and
- probabilistic submission: all quantiles requested by Energy Arena.

The `C=5` Fundamental LEAR forecast is therefore no longer submitted directly.
It remains one of the three point forecasts used to estimate SQRA.

Run the complete workflow and submit both forecasts:

```powershell
python run_pipeline.py
```

The runner then:

1. resolves and validates the current DE-LU Energy Arena point and quantile
   challenges;
2. verifies that the consolidated ICON-D2 histories contain the target day,
   invoking the DWD preparation step if necessary;
3. updates the ENTSO-E price/load caches and, when required, the EXAA cache;
4. fits any missing rolling LEAR point forecasts and updates their histories;
5. fits the target-day SQRA models and generates non-crossing quantiles;
6. uses SQRA `q=0.5` for the point payload and all requested SQRA quantiles for
   the probabilistic payload;
7. creates DST-aware Energy Arena payloads with 92, 96, or 100 physical MTUs;
   and
8. submits the point and quantile payloads with `ENERGY_ARENA_API_KEY`.

Generate and validate everything without submitting:

```powershell
python run_pipeline.py --no-submit
```

After the live cutoff, Energy Arena may already advertise the day after
tomorrow. For a local test only, shift that target back by one delivery day:

```powershell
python run_pipeline.py --no-submit --d-1
```

`--d-1` is intentionally restricted to `--no-submit`.

### Operational model variants

Select the model on the command line or through `.env`:

```powershell
# Default ICON-D2 Fundamental model
python run_pipeline.py --fundamental

# ICON-D2 Fundamental model enriched with EXAA prices
python run_pipeline.py --exaa-enriched

# EXAA-only model without weather or load features
python run_pipeline.py --exaa-only
```

The EXAA-Only SQRA forecast combines the `D_LEAR={56,112,364}` EXAA-Only point
models and submits its `q=0.5` median to the point challenge. The former
`D_LEAR=364` point forecast remains an SQRA member but is no longer submitted
directly. Weather-based SQRA likewise combines all three cluster resolutions.
`--cluster 1|5|25` and `POINT_WEATHER_CLUSTERS` retain the reference LEAR run
reported in logs and metadata; they do not change the submitted SQRA median.

The equivalent `.env` settings are:

```dotenv
POINT_FORECAST_VARIANT=fundamental
POINT_WEATHER_CLUSTERS=5
SQRA_FORECAST_VARIANT=auto
SQRA_TRAIN_DAYS=60
SQRA_MTU_SPECIFIC=false
```
A practical local-time schedule is:

- approximately 10:00: `prepare_dwd_data.py`;
- approximately 10:30: Fundamental `run_pipeline.py`;
- approximately 11:15: `run_pipeline.py --exaa-only`.

## Paper experimental analysis

The historical paper workflow is intentionally separate from the daily
operational pipeline. `run_full_experiment.py` performs the expensive point and
SQRA backtests. `run_full_evaluation.py` reads those completed forecasts and
runs the cheaper evaluation, table, and figure stages.

### Experimental data

The paper analysis uses:

- ENTSO-E DE-LU day-ahead prices and load forecasts;
- EXAA prices from ENTSO-E sequence 2;
- historical DWD ICON-D2 forecasts;
- ERA5 reanalysis fields; and
- the spatial cluster and map files under `data/`.

Market inputs are downloaded into the configured paper caches when required
coverage is missing. ERA5 and historical ICON-D2 inputs cannot be reconstructed
by the experiment runner and must be supplied or preprocessed first.

Expected paths are configured in `.env`:

```dotenv
ERA5_RAW_ARCHIVE=/path/to/raw/ERA5
ERA5_DATA_ROOT=data/era5
DWD_RAW_ARCHIVE=/path/to/raw/ICON-D2
ICON_DATA_ROOT=data/icon

MARKET_DATA_CACHE_DIR=data/cache/entsoe
ENTSOE_DE_LU_CACHE_DIR=data/cache/entsoe
EXAA_CACHE_DIR=data/cache/entsoe
ENTSOE_LOAD_FORECAST_CACHE_DIR=data/cache/entsoe

RESULTS_ROOT=results_extended_2025-12-01_2026-07-31
OUTPUT_ROOT=output_extended_2025-12-01_2026-07-31
```

### Historical weather preprocessing

```powershell
python .\preprocessing\preprocess_historic.py --icon
python .\preprocessing\preprocess_historic.py --era5 --clusters 1,5,25
```

### Experiment configuration

The evaluation horizon and rolling calibration share one configuration:

```dotenv
EVALUATION_START_DATE=
EVALUATION_END_DATE=
EVALUATION_SKIP_DATES=
FORECAST_SKIP_DATES=
SQRA_TRAIN_DAYS=60
SQRA_MTU_SPECIFIC=false
LEAR_USE_VST=true
MAX_LEAR_TRAIN_DAYS=364
```

### Point-forecast experiments

With no selector, the runner creates all 28 point forecasts: 27 fitted LEAR
configurations and the EXAA-naive benchmark. It does not continue into SQRA or
evaluation automatically:

```powershell
python .\run_full_experiment.py
```

The expensive runs can be split into independent batches:

```powershell
python .\run_full_experiment.py --dwd
python .\run_full_experiment.py --exaa
python .\run_full_experiment.py --era5 --d 56
python .\run_full_experiment.py --era5 --d 112
python .\run_full_experiment.py --era5 --d 364
```

- `--dwd` selects the six ICON-D2 point models.
- `--exaa` selects the three EXAA-Only LEAR models and EXAA-naive.
- `--era5` selects the 18 ERA5 models.
- `--d 56|112|364` restricts the LEAR calibration window.
- `--force` recomputes only the selected batch; otherwise complete outputs are
  resumed.

### SQRA experiments

After the required point forecasts exist:

```powershell
python .\evaluation\check_point_forecast_coverage.py --show-ok
python .\run_full_experiment.py --sqra
```

`--sqra` by itself runs no point models. The coverage checker reports missing
files, missing delivery days, partial 96-MTU days, extra dates, and whether a
gap affects the evaluation period.

The six SQRA configurations use:

- ERA5 Fundamental: `D_LEAR=364`, `C={1,5,25}`;
- ICON-D2 Fundamental: `D_LEAR=56`, `C={1,5,25}`;
- ERA5 EXAA-Enriched: `D_LEAR=364`, `C={1,5,25}`;
- ICON-D2 EXAA-Enriched: `D_LEAR=56`, `C={1,5,25}`;
- EXAA-Only: `D_LEAR={56,112,364}`; and
- EXAA-Naive: the raw EXAA point forecast.


### Evaluation, tables, and figures

Once all point and SQRA forecasts exist, run the independent post-processing
pipeline:

```powershell
python .\run_full_evaluation.py
```

The evaluation stage calculates point and probabilistic metrics, Kupiec tests,
and Giacomini-White tests. It also appends or refreshes the
`period=evaluation` row in every point model's `metrics.csv` without refitting a
model. The plotting stage generates the configured manuscript tables and
figures using the completed result files.

Absolute Normalized Contribution calculations are optional because they are
memory-intensive:

```powershell
python .\run_full_evaluation.py --anc
```

### Additional analyses and direct entry points

Computational stages and supplemental analyses can also be run directly:

```powershell
python -m pipeline.lear.run_lear exaa-naive
python -m pipeline.lear.run_lear operational --config era5_d56_c1_fundamental
python -m pipeline.lear.run_lear anc --variant fundamental
python -m pipeline.sqra.run_sqra --config era5_fundamental
python -m evaluation.run_evaluation
python -m evaluation.run_benchmarks
python .\evaluation\export_sqra_evaluation_metrics.py
python .\evaluation\export_sqra_kupiec_excel.py
python .\evaluation\plot_monthly_point_mae.py
python .\visualization\plot_exaa_epex_correlation.py
python .\visualization\plot_cluster_heatmap.py
```

The benchmark runner creates persistence `d-1` and `d-7` point forecasts and
their SQRA post-processed variants below `RESULTS_ROOT/benchmarks`. The two
Excel exporters provide the SQRA evaluation metrics and raw Kupiec MTU counts.
The monthly script creates the point-model monthly MAE and monthly negative
EPEX-price figures. The correlation script reports Pearson and Spearman EXAA–
EPEX correlations for the configured delivery-day groups.

## Tests

Run the synthetic parity, schema, validation, and orchestration tests from the
activated environment:

```powershell
python -m unittest discover -s tests -v
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
