# Day-Ahead Price Forecasting Pipeline DE-LU

This repository contains the forecasting pipeline accompanying the bachelor
thesis *An Open-Source Probabilistic Forecasting Pipeline for German Day-Ahead
Prices*. It produces 15-minute day-ahead forecasts for the EPEX DE-LU bidding
zone with LEAR point models and SQRA probabilistic post-processing.

The automated pipeline is implemented entirely in Python modules. The three
notebooks are optional interactive views. `run_pipeline.py` is the daily
operational entry point through Energy Arena submission. Paper backtests are
handled separately by `run_full_experiment.py` and `run_full_evaluation.py`.

## Repository structure

```text
DA_Price_Forecasting_Pipeline_DE_LU/
├── pipeline/
│   ├── lear/
│   │   ├── lear_model.py
│   │   ├── run_lear.py
│   │   └── lear_pipeline.ipynb
│   ├── sqra/
│   │   ├── sqra_model.py
│   │   ├── run_sqra.py
│   │   └── sqra_pipeline.ipynb
│   └── operational/
│       ├── config.py
│       ├── dwd.py
│       ├── energy_arena.py
│       └── runner.py
├── evaluation/
│   ├── evaluation_core.py
│   ├── run_evaluation.py
│   └── evaluation.ipynb
├── preprocessing/
├── visualization/
├── experiment_config.py
├── experiment_manifest.py
├── requirements.txt
├── run_pipeline.py
├── run_full_experiment.py
└── run_full_evaluation.py
```

## Data

The pipeline uses:

- ENTSO-E DE-LU day-ahead prices and load forecasts;
- EXAA day-ahead prices obtained through the existing ENTSO-E client logic;
- preprocessed DWD ICON-D2 forecasts;
- preprocessed ERA5 reanalysis data.

Market inputs are downloaded into the configured caches when their requested
coverage is missing. The daily runner downloads and verifies the current
ICON-D2 run when `DOWNLOAD_DWD=true`; raw ERA5 and historical ICON-D2 archives
cannot be reconstructed from that operational endpoint and must be supplied
for paper backtests or initial rolling-history bootstrap. All paths are
configured in `.env`; notebook settings are not used by automated execution.

## Setup with one Python 3.9 environment

In PowerShell from the repository root:

```powershell
py -3.9 -m venv forecast
.\forecast\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r .\requirements.txt
```

The same requirements file also contains the optional Jupyter packages, so the
retained notebooks work in this one environment. Registering a notebook kernel
is optional:

```powershell
python -m ipykernel install --user --name forecast --display-name "forecast (Python 3.9)"
```

Copy `.env.example` to `.env`, add the ENTSO-E and Energy Arena API keys, and
set the raw, processed, cache, and output paths.

## Daily operational pipeline

The default daily configuration follows the paper's selected operational
model:

- point submission: DWD/ICON-D2 Fundamental LEAR, `D=56`, `C=5`;
- SQRA inputs: the matching DWD Fundamental point forecasts for `C=1,5,25`;
- quantile submission: one pooled SQRA fit per required quantile, using the
  preceding 60 delivery days and all 96 MTUs per day.

Thus, one default run generates all three DWD point forecasts needed by SQRA,
submits only the `C=5` forecast to the point challenge, and combines `C=1,5,25`
for the quantile challenge. The Energy Arena challenge metadata determines the
required quantile levels and the correct 92/96/100-value DST payload length.

First inspect the resolved live target and model plan without downloading or
fitting:

```powershell
python run_pipeline --check-setup
python run_pipeline --dry-run
```

Run the complete workflow and submit both forecasts:

```powershell
python run_pipeline
```

Use `--no-submit` for a complete local run that saves and validates both JSON
payloads but does not call the submission endpoint:

```powershell
python run_pipeline --no-submit
```

After the daily submission cutoff, Energy Arena may already advertise the day
after tomorrow as its next target. To test tomorrow's complete forecast locally,
shift that live target back by one delivery day:

```powershell
python run_pipeline --no-submit --d-1
```

`--d-1` is deliberately restricted to `--no-submit`. It affects DWD download,
preprocessing, point forecasts, SQRA, and the locally saved payloads consistently;
it never posts the back-shifted payloads to Energy Arena.

For the EXAA-only account/model variant, the submitted point forecast uses
`D=364`, while SQRA combines the `D=56`, `D=112`, and `D=364` EXAA-only point
forecasts:

```powershell
python run_pipeline --exaa_only --arena-profile exaa_only
python run_pipeline --exaa_only --energy-arena YOUR_OTHER_ACCOUNT_API_KEY
```

The profile form reads `ENERGY_ARENA_API_KEY_EXAA_ONLY` from `.env`. The raw
key override is convenient for a one-off run, but command-line values can be
visible in terminal history. `--exaa-enriched` selects the DWD EXAA-enriched
configuration. `--cluster 1|5|25` changes the point model submitted for an
explicit non-default run; weather-based SQRA still generates all three
cluster members.

Fundamental and EXAA-only runs may overlap when they use different Arena
profiles. Pipeline locks are scoped to the LEAR/SQRA histories and Arena
payload namespace actually being updated, while each shared ENTSO-E/EXAA
cache has its own inter-process lock. A second process therefore waits for a
short cache update without blocking the independent model fitting. Runs which
would write the same model history or the same Arena account remain blocked.
The lock files below `OPERATIONAL_OUTPUT_ROOT/locks` are persistent OS-lock
handles; their presence alone does not mean that a process is still active.

A practical local-time daily sequence is to run `prepare_dwd_data.py` around
10:00, start the Fundamental pipeline once the target-day ENTSO-E load forecast
is available (for example 10:30), and start EXAA-only after the 10:15 EXAA
auction results are visible (for example 11:15). Availability checks and the
live Energy Arena deadline returned by the API remain authoritative.

The daily runner performs these operations in order:

1. resolve and validate the live DE-LU point and quantile challenges;
2. ensure the separate DWD preparation step has updated the consolidated
   C=1,5,25 histories (running it inline only when still necessary);
3. load the prepared `u_10m`, `v_10m`, `aswdir_s`, and `aswdifd_s` histories;
4. extend the ENTSO-E price/load and optional EXAA caches;
5. fit missing rolling LEAR point forecasts and persist their history;
6. fit target-day SQRA quantiles, construct DST-aware payloads, and submit.

Operational downloads are staged under `DWD_OPERATIONAL_RAW_ROOT` (by default
`data/dwd_raw`). `DOWNLOAD_DWD=false` means that the current raw GRIB files are
already present there; integrity verification and preprocessing still run.
`DWD_RAW_ARCHIVE` remains the separate source for historical preprocessing.
Both the legacy
`dwd_icon_daily_YYYYMMDD/icon-d2/HH/VARIABLE/*.grib2.bz2` layout and the newer
`dwd_icon_archived_YYYYMMDD/icon-d2__HH__VARIABLE.zip` layout are accepted.
ZIP sets placed directly inside a `dwd_icon_daily_YYYYMMDD` folder are also
accepted, matching the transitional archive layout.
New downloads are written in the latter layout and contain the 49 required
regular-lat-lon files for each of the four LEAR weather variables. Downloads
and ZIP files are written atomically, every nested BZip2 stream is fully
decompressed to verify its CRC and GRIB signature, and failed downloads are
retried according to the `.env` retry settings. With
`DELETE_DWD_RAW_AFTER_PREPROCESS=true`, the operational raw ZIPs are deleted
only after all requested consolidated cluster histories have been updated and
successfully verified.

If ENTSO-E leaves a historical EPEX delivery day unpublished, the operational
pipeline retains that day as missing after attempting a refresh. The missing
day is excluded from LEAR and SQRA calibration targets. For an operational
forecast affected by an unavailable EPEX lag, only the unavailable price-lag
columns are suppressed for that rolling fit; all available weather, load,
calendar, EXAA, and other price-lag inputs continue to be used. Missing
non-price inputs remain fatal rather than being silently discarded.

Operational market caches are stored under `OPERATIONAL_MARKET_DATA_ROOT`
(default `data/market`): ENTSO-E prices and load forecasts are written below
`data/market/entsoe`, and EXAA prices below `data/market/exaa`. Each pipeline
run verifies the complete interval required by its rolling models and fetches
the smallest date span covering missing or newly required timestamps instead
of downloading the complete calibration history again. ENTSO-E prices are
always maintained; load forecasts are maintained for weather-based models;
EXAA is maintained when the selected point/SQRA model uses it or when
`DOWNLOAD_EXAA=true`. The three cache directories can be overridden separately
with `OPERATIONAL_ENTSOE_PRICE_CACHE_DIR`,
`OPERATIONAL_ENTSOE_LOAD_CACHE_DIR`, and `OPERATIONAL_EXAA_CACHE_DIR`.

For a first weather-based run, the consolidated histories must cover the
60-day SQRA window plus the 56-day LEAR window (about 116 preceding delivery
days). `prepare_dwd_data.py` bootstraps them from existing processed daily
outputs; otherwise sufficient historical raw runs are required. Existing 09
UTC histories may seed calibration, while new operational targets use 06 UTC.
Subsequent daily runs append one delivery day and reuse persisted forecasts.

The operational format contains four appendable histories directly in every
cluster directory:

```text
ICON_DATA_ROOT/
  c1/{u10,v10,ASWDIR_S,ASWDIFD_S}.parquet
  c5/{u10,v10,ASWDIR_S,ASWDIFD_S}.parquet
  c25/{u10,v10,ASWDIR_S,ASWDIFD_S}.parquet
```

Each row records the delivery date, ICON issue date, run hour, timestamp, and
cluster values. Existing daily CSV, XLSX, and `icon_d2_aggregated.parquet`
folders are used to bootstrap these histories without GRIB reprocessing. The
forecast loaders prefer the consolidated format once its manifest is complete.

The weather preparation can be scheduled before the main pipeline:

```powershell
python .\prepare_dwd_data.py --target-date 2026-09-17
```

With no target date it prepares tomorrow. It downloads or validates only the
preceding 06 UTC ICON-D2 run and the four required variables, aggregates C=1,
5, and 25, appends/replaces that delivery day, and verifies all histories.
After successful verification it removes only the consumed raw 06 UTC inputs;
use `--keep-raw` to retain them. The later `run_pipeline` call sees the complete
histories and skips weather preparation. To build histories only from existing
processed data, run:

```powershell
python .\prepare_dwd_data.py --migrate-only
```

Payloads, submission receipts, and logs are stored below
`OPERATIONAL_OUTPUT_ROOT`. Payload JSON files remain account-, challenge-, and
target-specific. Logs are replace-on-each-run files named `fundamental.log`,
`exaa_enriched.log`, or `exaa_only.log`. Submission receipts retain only
`submissions/<account>/<challenge>/latest.json`, which is replaced after the
next successful submission. An identical immediate rerun is not posted twice
unless `--force-submit` is used. Point/SQRA forecast histories are stored below
`OPERATIONAL_RESULTS_ROOT` using the same result layout as the paper pipeline.

## Historical weather preprocessing

Inspect the automatically derived plan without writing data:

```powershell
python .\preprocessing\preprocess_historic.py --dry-run
```

With no source flag, both sources and clusters 1, 5, and 25 are processed.
They can also be selected explicitly:

```powershell
python .\preprocessing\preprocess_historic.py --icon
python .\preprocessing\preprocess_historic.py --era5 --clusters 1,5,25
```

To recreate one or several ICON raw-folder dates without scanning every daily
folder, use `--date` together with `--force`:

```powershell
python .\preprocessing\preprocess_historic.py --icon --date 2026-06-12 2026-06-18 --clusters 1,5,25 --force
```

Each date selects its `dwd_icon_daily_YYYYMMDD` source (nested files or a ZIP
set) or its `dwd_icon_archived_YYYYMMDD` ZIP set. These are ICON initialization
dates, not the following delivery dates.

ERA5 years and the required weather history are derived from the evaluation
dates in `.env`. Existing complete outputs are reused unless `--force` is
specified.

## Full paper rerun

The evaluation horizon and calibration behavior have one source of truth:

```dotenv
EVALUATION_START_DATE=2025-12-01
EVALUATION_END_DATE=2026-07-31
EVALUATION_SKIP_DATES=2026-01-22,2026-06-12
FORECAST_SKIP_DATES=2026-01-22,2026-06-12
SQRA_TRAIN_DAYS=60
SQRA_MTU_SPECIFIC=false
LEAR_USE_VST=true
```

Forecast result schema version 2 uses the explicit index
`(delivery_date, mtu)`, where every retained delivery day has MTUs 1 through
96. This implements the manuscript's DST normalization: the four missing MTUs
on a 23-hour spring day are interpolated, and repeated MTUs on a 25-hour autumn
day are averaged. Consequently, 29 March 2026 is included in the evaluation.
The two remaining skip dates lack usable common ICON-D2 inputs. Legacy
timestamp-indexed forecasts are not resumed and must be regenerated.

### Point-forecast runs

With no source selector, the forecast runner processes all 28 point forecasts
(27 fitted LEAR configurations plus EXAA-naive). It does not automatically
continue into SQRA, evaluation, or plots:

```powershell
python .\run_full_experiment.py --dry-run
python .\run_full_experiment.py --preflight
python .\run_full_experiment.py
```

Long point-forecast computation can be split into independent batches:

```powershell
python .\run_full_experiment.py --dwd
python .\run_full_experiment.py --exaa
python .\run_full_experiment.py --era5 --d 56
python .\run_full_experiment.py --era5 --d 112
python .\run_full_experiment.py --era5 --d 364
```

`--dwd` selects all six DWD/ICON-D2 point models. `--exaa` selects the three
EXAA-only LEAR models and EXAA-naive. `--era5` selects all 18 ERA5 point
models; `--d` restricts fitted models to one of the configured training
windows 56, 112, or 364. Source flags can be combined. With no source flag,
`--d` filters the otherwise complete point-model selection. EXAA-naive remains
part of an EXAA batch because it has no fitted training-window parameter.

Complete selected point results are resumed. Use `--force` to deliberately
recompute only the selected batch.

Long weather batches can be split further by cluster and model variant. For
example, the six ERA5 D=364 configurations can be run one at a time:

```powershell
python .\run_full_experiment.py --era5 --d 364 --cluster 1  --fundamental true
python .\run_full_experiment.py --era5 --d 364 --cluster 1  --fundamental false
python .\run_full_experiment.py --era5 --d 364 --cluster 5  --fundamental true
python .\run_full_experiment.py --era5 --d 364 --cluster 5  --fundamental false
python .\run_full_experiment.py --era5 --d 364 --cluster 25 --fundamental true
python .\run_full_experiment.py --era5 --d 364 --cluster 25 --fundamental false
```

`--fundamental true` selects the fundamental model and `false` selects the
EXAA-enriched model. If `--fundamental` is omitted, both variants for the
selected cluster are run. The cluster and variant filters also work with DWD,
and require an explicit `--era5` or `--dwd` source.

### SQRA runs

After the required point forecasts exist, validate and run all six SQRA
configurations separately:

```powershell
python .\evaluation\check_point_forecast_coverage.py --show-ok
python .\run_full_experiment.py --sqra --preflight
python .\run_full_experiment.py --sqra
```

The coverage checker inspects all 28 point outputs and reports missing files,
entirely missing delivery days, partial 96-MTU days, extra dates, and whether a
gap affects the configured evaluation period. Configured skip dates are not
reported as missing.

`--sqra` by itself runs no point models. It can also be combined with source
flags to run the selected point batch first and SQRA second, provided all
other SQRA member forecasts already exist.

The six SQRA configurations use these point-forecast members:

- ERA5 Fundamental: `d364/c1`, `d364/c5`, and `d364/c25`.
- ICON-D2 Fundamental: `d56/c1`, `d56/c5`, and `d56/c25`.
- ERA5 EXAA-Enriched: `d364/c1`, `d364/c5`, and `d364/c25`.
- ICON-D2 EXAA-Enriched: `d56/c1`, `d56/c5`, and `d56/c25`.
- EXAA-Only: `d56`, `d112`, and `d364`.
- EXAA-Naive: the raw EXAA point forecast.

Their existing configuration names and `RESULTS_ROOT/sqra_results/<name>`
output directories are preserved when SQRA is rerun.

### Evaluation and figures

Once all 28 point forecasts and six SQRA forecasts exist, use the independent
post-processing runner:

```powershell
python .\run_full_evaluation.py --preflight
python .\run_full_evaluation.py
```

It runs evaluation first and then the manuscript tables and figures. These can
also be selected individually:

```powershell
python .\run_full_evaluation.py --stages evaluation
python .\run_full_evaluation.py --stages plots
```

During the evaluation stage, the already-generated point forecast files are
used to append or refresh a final `period=evaluation` row in each point
model's `metrics.csv`. This covers only the configured evaluation dates and
skip dates and does not refit any forecasting model.

ANC remains optional because it is memory-intensive. When requested, its two
variants run before evaluation and plots:

```powershell
python .\run_full_evaluation.py --anc
```

Every executed subprocess streams to the terminal and to:

```text
OUTPUT_ROOT/logs/<stage>/<configuration>.log
```

No executed notebook copies are produced.

## Direct Python entry points

Each computational stage can also be run independently:

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
```

The notebooks import these same modules and contain no separate model or
evaluation implementation.

`evaluation.run_benchmarks` is independent of the full experiment and main
evaluation registry. It creates the d-1 and d-7 persistence point forecasts
and their two SQRA post-processed variants under
`RESULTS_ROOT/benchmarks/`. Each of the four folders contains `forecast.csv`,
`runtime.csv`, `metrics.csv`, and `config.json`.

The SQRA metrics exporter reads all six probabilistic forecast outputs and
writes evaluation-period median MAE, median RMSE, and APS to
`OUTPUT_ROOT/sqra_evaluation_metrics.xlsx`. The workbook also records the
evaluation dates, excluded delivery dates, quantiles, and metric definitions.

The SQRA Kupiec exporter recreates the four raw MTU-count columns underlying
manuscript Table D.5 (50% and 80% prediction intervals, each at the 1% and 5%
significance levels). It saves the six configurations and calculation metadata
to `OUTPUT_ROOT/sqra_kupiec_mtu_counts.xlsx`.

The monthly point-forecast MAE script compares four representative models
from December 2025 through July 2026. It prints the monthly values and saves
`monthly_point_forecast_mae.csv` and `monthly_point_forecast_mae.pdf` under
`OUTPUT_ROOT`. Each monthly value is the MAE pooled across all included
15-minute MTUs in that calendar month; configured evaluation skip dates are
excluded. The same script also counts all physical EPEX DE-LU quarter-hours
with prices below zero and saves `monthly_epex_negative_price_mtus.csv` and
`monthly_epex_negative_price_mtus.pdf`. This market count includes every
available delivery interval, including the spring daylight-saving day.

The price-correlation script matches EXAA and EPEX DE-LU prices by local
delivery date and 15-minute MTU over the configured evaluation period. It
prints and saves Pearson and Spearman results for all days, Tuesday–Saturday,
Sunday, Monday, and MTUs for which the EPEX price is below zero. Its outputs are
`exaa_epex_price_correlations.csv` and
`exaa_epex_price_correlations.pdf` under `OUTPUT_ROOT`.

## Tests

Run the synthetic parity, schema, validation, and orchestration tests from the
activated `forecast` environment:

```powershell
python -m unittest discover -s tests -v
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
