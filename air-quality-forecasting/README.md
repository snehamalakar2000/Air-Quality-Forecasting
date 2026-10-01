# Air Quality Forecasting with a Reproducible ML Pipeline

Predict Beijing PM2.5 exactly one hour ahead using current and historical pollution, current weather, and the known target calendar. This is a retrospective rolling forecast, not a full-year forecast made in advance.

The approved Architect document is preserved unchanged in [docs/plan.md](docs/plan.md). Its statements about implementation not being authorized describe the earlier planning stage; this project implements the subsequent Builder request.

## Quick start

Run from the folder containing `config.json`. Use Python **3.11**; the delivered project was checked with Python 3.11.16.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-deps --no-build-isolation -e .
python -m air_quality validate --config config.json
python -m pytest -q
python -m air_quality smoke
python -m air_quality select --config config.json
python -m air_quality evaluate --config config.json --selection artifacts/selected_config.json
```

Installation needs internet or cached wheels. Validation, tests, smoke, selection, and evaluation use local files. There is no training-time downloader or live API. The original static CSV is included. Allow roughly a minute or a few minutes for selection depending on your computer.

See [the Mac and VS Code instructions](docs/mac_setup.md) for each setup step. [docs/verification.md](docs/verification.md) separates checks performed by the Builder from checks you still need to run. The ZIP includes generated `artifacts/` for review; `.gitignore` excludes them when you put the source on GitHub. Keep `data/raw/` tracked.

## Prediction contract and implementation choices

- `t` is the latest completed hour, interpreted as Beijing local time, `Asia/Shanghai`. Its pollution and weather readings are assumed available immediately; actual publication delays are not established by this dataset.
- Predict observed PM2.5 at exactly `t + 1 hour`. Missing current PM2.5 prevents a forecast. Missing target PM2.5 prevents training/scoring, but does not prevent inference.
- `data.py` verifies the byte checksum, size, row count, date range, schema, values, and timestamps, then creates a complete hourly grid. It logs absent hours separately from present records with missing readings. Negative temperature/dew point and genuinely high pollution are retained.
- `features.py` constructs features before filtering rows. Lags mean elapsed hours, not previous available readings. Trailing means include the current hour and require 3 of 6 or 12 of 24 observed readings. Counts and fixed missing flags identify incomplete history. No interpolation, backfill, target imputation, or centered windows is used.
- Measured weather comes only from `t`. Target hour/month sine and cosine and target weekday are already known at `t`; no measured next-hour weather is used. Identifier `No` and raw year are not predictors.
- Wind missingness becomes the explicit category `missing`. Training-only one-hot categories safely handle unseen values. Numeric medians and Ridge scaling are learned inside scikit-learn pipelines. An entirely missing numeric training column fails explicitly.
- Splits use **target** timestamps: 2010–2012 training, 2013 validation, 2014 test. A year-end origin belongs to the partition of the next-hour label. History crosses split boundaries because earlier readings are available during rolling forecasting.
- `select` tries three Ridge alphas and two forest depths on the same validation hours. Selection uses MAE, then RMSE, then Ridge. Predictions are clipped below at zero for selection, scoring, and inference. There is no upper clipping.
- Selected Ridge and forest pipelines remain fitted on training only. The validation manifest freezes settings, the training 90th-percentile threshold, configuration hash, and model hashes. `evaluate` verifies these and **never fits**. It rejects changed configuration, data, or model files. The test suite explicitly forbids fitting during evaluation.
- The default forest has 200 trees, minimum leaf size 5, two worker threads, seed 42, and depths 10/20. Only synthetic smoke/tests use a smaller forest.

Configuration paths are relative to the repository root. Editing any config value invalidates the existing selection manifest; it is deliberately strict. Do not change model choices after reviewing the 2014 holdout and describe the same holdout as untouched.

## Measured results

The included artifacts were generated from the included CSV and default config, using the pinned dependencies and Python 3.11.16 on Linux. All methods use identical scored timestamps.

| Partition | Eligible scored hours | Possible target hours | Scoring coverage |
|---|---:|---:|---:|
| Training | 24,274 | 26,304 | 92.28% |
| Validation | 8,643 | 8,760 | 98.66% |
| Test | 8,626 | 8,760 | 98.47% |

The source has 43,824 records, no absent timestamp rows, and 2,067 missing PM2.5 observations. Its longest run of missing pollution readings is 155 hours. Weather columns have no missing source values. Historical feature missingness still occurs near the start and near pollution outages. Forecast coverage and overlapping exclusion reasons are recorded separately in `artifacts/data_quality.json`.

| Method | Validation MAE | Validation RMSE | Test MAE | Test RMSE |
|---|---:|---:|---:|---:|
| Persistence | 13.169 | 24.806 | 11.925 | 22.061 |
| Ridge, alpha 0.1 **(validation-selected)** | 13.001 | 23.793 | 11.817 | 21.400 |
| Random Forest, depth 10 | 13.022 | 25.591 | 11.634 | 21.389 |

Errors are in µg/m³. Validation recommends Ridge over persistence. Its final test MAE improvement is **0.90%** over persistence. The forest's test improvement is **2.44%**, but that does not change the validation-selected model. This is a small overall gain over a strong baseline.

High pollution means target PM2.5 **≥ 218 µg/m³**, the 90th percentile of eligible training targets. This is an error-analysis definition, not a health category. There are 883 high-pollution scored test hours and 7,743 remaining hours.

| Method, high-pollution hours | MAE | RMSE | Mean signed error | Predictions below actual |
|---|---:|---:|---:|---:|
| Persistence | 22.878 | 38.221 | −5.458 | 56.17% |
| Ridge | 24.120 | 38.984 | −14.917 | 74.97% |
| Random Forest | 24.245 | 40.095 | −12.078 | 67.61% |

Both learned models underpredict high concentrations more often than persistence and have worse high-pollution MAE, despite better overall errors. Their gains are concentrated in the remaining periods. These associations do not establish that any weather variable causes an error. Full slice statistics, including signed errors and counts, are saved in `test_metrics.json`.

## Three required figures

![Training pollution and missingness](artifacts/plots/training_overview.png)

![Fixed-period predictions](artifacts/plots/test_predictions_fixed_period.png)

The predefined January 1–7, 2014 period has 168 of 168 eligible scored hours. The plotting code inserts NaNs at excluded hours to break lines; synthetic tests verify gaps and an empty window without choosing another week.

![Final model errors](artifacts/plots/model_error_comparison.png)

Only these three figures are implemented. Additional seasonal diagnostics and feature importance are future work.

## Saved outputs

| File in `artifacts/` | Meaning |
|---|---|
| `data_quality.json` | Provenance, missingness/runs, exclusions, and coverage per split |
| `validation_metrics.csv` | Persistence and all five validation configurations |
| `selected_config.json` | Frozen decision, thresholds, parameters, and artifact/config/data hashes |
| `ridge.joblib`, `forest.joblib` | Training-fitted, validation-tuned pipelines |
| `model.joblib` | The selected learned pipeline, Ridge for this run |
| `run_metadata.json` | Python/package versions, seed, effective config, counts, parameters, Git commit if available |
| `test_metrics.json` | Overall/high/remaining metrics, relative improvements, and coverage |
| `test_predictions.csv` | Auditable origin/target times, observed target/current reading, all predictions, signed errors, history counts |
| `plots/` | Exactly the three required PNGs |

`joblib` files should be loaded only from this project's trusted generated outputs. Evaluation uses the separately frozen family models; `model.joblib` is the convenience copy for inference. Model bytes may differ across machines; compare predictions and metrics with a small numerical tolerance, not binary equality. Parallel forest accumulation can vary in its final floating-point bits.

## Inference without a future label

Reusable inference is exposed through `air_quality.models.forecast`. This illustrates the selected learned model; if your validation run recommends persistence, use the observed current reading instead.

```python
from air_quality.pipeline import read_config, load_selection
from air_quality.data import load_data
from air_quality.features import build_features
from air_quality.models import forecast

config = read_config("config.json")
selection, models = load_selection(config, "artifacts/selected_config.json")
frame, _ = load_data(config["data_path"], config["source"])
# Simulate readings available only through the chosen origin.
history = frame.loc[:"2014-01-02 10:00"]
inputs = build_features(history, config["features"])
origins, predictions = forecast(models[selection["selected_learned_model"]], inputs.tail(1))
print(origins, predictions)  # Targets are these origins + one hour.
```

The inference helper filters missing current readings and requires no label. The static file loader still verifies the original source; use a validated historical prefix for this example rather than changing the source file.

## Docker and CI

The Dockerfile uses `/app` and pins the multi-platform Python 3.11 slim Bookworm image by digest. Requirements, code, tests, config, and static CSV are copied into the image. Caches, virtual environments, Git metadata, and generated artifacts are excluded.

```bash
docker build -t air-quality-forecasting:local .
mkdir -p artifacts
docker run --rm --network none air-quality-forecasting:local python -m pytest -q
docker run --rm --network none air-quality-forecasting:local python -m air_quality smoke
docker run --rm --network none -v "$PWD/artifacts:/app/artifacts" air-quality-forecasting:local python -m air_quality select --config config.json
docker run --rm --network none -v "$PWD/artifacts:/app/artifacts" air-quality-forecasting:local python -m air_quality evaluate --config config.json --selection artifacts/selected_config.json
```

The Builder did not build or run Docker in its Linux environment. The digest was resolved against Docker Hub on October 1, 2026. The manual macOS verification below records the actual local Docker results.

`.github/workflows/ci.yml` runs on push and pull request, installs Python 3.11 and pinned dependencies, validates the real static CSV, runs tests, builds Docker, and runs a deterministic synthetic smoke command with `--network none`. CI never selects or evaluates models on the real 2014 holdout. GitHub Actions execution has not occurred here; upload to your repository and check the Actions tab.

### Manual Docker smoke test completed on macOS

On October 1, 2026, the project was built and run locally on macOS using the `air-quality-forecasting:local` image. The container test command completed with **37 passed**. The selection and evaluation commands below also completed without errors, with `--network none` and the Mac `artifacts/` directory mounted into `/app/artifacts`:

```bash
docker run --rm --network none -v "$PWD/artifacts:/app/artifacts" air-quality-forecasting:local python -m air_quality select --config config.json
docker run --rm --network none -v "$PWD/artifacts:/app/artifacts" air-quality-forecasting:local python -m air_quality evaluate --config config.json --selection artifacts/selected_config.json
```

The validation-selected learned model was Ridge, and the recommended method was Ridge. Evaluation scored 8,626 eligible 2014 test hours. The test results were:

| Method | Test MAE | Test RMSE |
|---|---:|---:|
| Persistence | 11.925 | 22.061 |
| Ridge | 11.817 | 21.400 |
| Random Forest | 11.634 | 21.389 |

The high-pollution threshold was 218 µg/m³, with 883 high-pollution test hours. The Docker run saved the expected artifacts to the Mac `artifacts/` directory, including `selected_config.json`, the three model files, `run_metadata.json`, `test_metrics.json`, `test_predictions.csv`, `validation_metrics.csv`, and `data_quality.json`. The three required figures—`training_overview.png`, `test_predictions_fixed_period.png`, and `model_error_comparison.png`—were opened and rendered correctly.

This manual check verifies local Docker build, container tests, offline model selection, offline frozen evaluation, output persistence, and figure rendering. GitHub Actions has not been run yet, and the independent Tester stage remains pending.

## Limitations and student-owned material

These historical 2010–2014 readings from one pollution location do not establish present-day or other-city performance. Airport weather is spatially mismatched with the monitor. Arrival times are assumed, accumulated wind/snow/rain reset conventions need investigation, and missing readings reduce coverage. A single validation year, no uncertainty intervals, and one-step-only predictions limit conclusions. This is not an official warning system.

Complete your own [manual smoke-test record](docs/manual_smoke_test.md), [personal reflection](docs/reflection.md), and [conversation transcript record](docs/conversations.md). They contain prompts and empty fields, not invented student results or statements.

## Data attribution

Chen, S. (2015). *Beijing PM2.5* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5JS49. Dataset license: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Source details and checksum: [data/README.md](data/README.md). The original CSV is preserved unchanged; derived features and reports are project transformations.
