# Air Quality Forecasting with a Reproducible ML Pipeline

**Status:** Finalized plan for Option 3: Create a New Project, reviewed and approved on October 1, 2026. Implementation has not been authorized in this turn. No pipeline, model, tests, or Docker image has been implemented.

## 1. Purpose and requirements

Predict Beijing's PM2.5 concentration exactly one hour ahead. Demonstrate a small, reproducible data engineering workflow: validate a static CSV, construct time-safe features, train models, compare them fairly, test important behavior, and reproduce the workflow in Docker.

Use Python 3.11, pandas, NumPy, scikit-learn, pytest, GitHub Actions, Docker, matplotlib, and joblib. NumPy supports calendar calculations; matplotlib creates plots; joblib saves fitted pipelines. No deep learning, live API, web app, workflow orchestrator, or cloud deployment is needed for version 1.

### Dataset and provenance

- Source: [UCI Beijing PM2.5 dataset](https://archive.ics.uci.edu/dataset/381/beijing+pm2+5+data).
- Source file: `PRSA_data_2010.1.1-2014.12.31.csv`.
- UCI describes 43,824 hourly records covering 2010–2014, with `NA` missing values. Pollution observations come from the US Embassy in Beijing; weather observations come from Beijing Capital International Airport.
- Download once during setup. Commit the approximately 1.9 MB CSV under `data/raw/`, with CC BY 4.0 attribution. Training, testing, and evaluation must read this local file without downloading anything.
- Record source URL, citation, download date, filename, byte size, and SHA-256 checksum in `data/README.md`. Record actual validation results during implementation; do not assume the CSV has every timestamp merely because its advertised row count matches a complete grid.
- Dataset citation: Chen, S. (2015). Beijing PM2.5 [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5JS49.

### Prediction-time contract

Let **t** be the latest completed hour whose readings are available. Immediately after receiving those readings, predict **PM2.5 at t + 1 hour**, in micrograms per cubic meter (µg/m³).

Example: after observing the 10:00 record, predict the 11:00 record. Features may use observations at 10:00 or earlier. They may use the known calendar for 11:00. They may not use pollution or measured weather from 11:00.

Interpret timestamps as Beijing local time (`Asia/Shanghai`). The CSV does not establish publication delays, so immediate availability of its hourly readings is a project assumption, not a verified operational guarantee.

This is **rolling one-hour forecasting**: each successive forecast can use newly observed readings. It is not a forecast of a whole year made at the start of that year.

Require an observed PM2.5 value at t. If it is missing, skip that forecast and record why. For supervised training or scoring, also require an observed PM2.5 value at exactly t + 1 hour. Inference itself does not require that future value.

### Acceptance criteria

- A clean clone can be installed and run using documented commands.
- After setup or image build, the pipeline runs without a network connection.
- Every scored row has `target_time - origin_time == 1 hour`.
- All models and the baseline are scored on exactly the same eligible rows.
- Learned preprocessing and model fitting use only the training partition.
- Model selection uses validation results; final test outcomes do not influence selection.
- Runs save counts, configuration, metrics, predictions, exactly the three required figures in Section 8, and the fitted pipeline. Additional plots are future improvements, not version 1 requirements.
- pytest checks the time contract, leakage prevention, and meaningful failure cases; CI runs those checks and an offline container smoke test.
- Beating persistence is a research outcome, not a condition for passing the assignment. Report honestly if persistence wins.

## 2. Repository structure

```text
air-quality-forecasting/
  README.md
  pyproject.toml
  requirements.txt
  config.json
  .gitignore
  .dockerignore
  Dockerfile
  .github/workflows/ci.yml
  docs/plan.md
  data/
    README.md
    raw/PRSA_data_2010.1.1-2014.12.31.csv
  src/air_quality/
    __init__.py
    __main__.py
    data.py
    features.py
    models.py
    evaluation.py
    pipeline.py
  tests/
    conftest.py
    test_data.py
    test_features.py
    test_models.py
    test_pipeline.py
  artifacts/                 # generated; ignored by Git
```

File responsibilities:

- `README.md`: project question, quick start, prediction assumptions, results summary, and limitations.
- `pyproject.toml`: package installation and pytest configuration. The CLI is `python -m air_quality`.
- `requirements.txt`: exact tested versions of direct and transitive dependencies; Builder resolves and pins a compatible environment.
- `config.json`: data path, date boundaries, feature settings, small model grid, random seed, and output directory.
- `data.py`: local loading, checksum verification, schema checks, timestamp construction, hourly reindexing, and data-quality counts.
- `features.py`: causal feature construction, exact-hour target alignment, eligibility flags, and chronological splitting.
- `models.py`: persistence prediction and the two scikit-learn pipelines.
- `evaluation.py`: metrics, common prediction table, error slices, and plots.
- `pipeline.py` / `__main__.py`: coordinate validation, selection, final evaluation, and smoke-test commands.
- `artifacts/`: `data_quality.json`, `validation_metrics.csv`, `selected_config.json`, `model.joblib`, `run_metadata.json`, `test_metrics.json`, `test_predictions.csv`, and `plots/`.

Keep reusable behavior in Python modules. An optional notebook can explain results later, but the workflow must not depend on running notebook cells in order.

## 3. Data validation and missing values

### Validate before modeling

Require columns `No`, `year`, `month`, `day`, `hour`, `pm2.5`, `DEWP`, `TEMP`, `PRES`, `cbwd`, `Iws`, `Is`, and `Ir`. Rename `pm2.5` to `pm25` internally.

1. Parse `NA` as missing. Reject malformed numeric values rather than silently converting them to missing.
2. Reject invalid dates, noninteger calendar fields, hours outside 0–23, duplicate timestamps, infinite numeric values, and negative observed PM2.5 values.
3. Reject negative wind/snow/rain accumulation values and nonpositive pressure. Negative temperature and dew point are valid. Preserve genuine high pollution values; do not cap or discard them as outliers.
4. Sort by timestamp and log whether reordering was necessary. `No` is an identifier, not a time index or model feature.
5. Report raw rows, date range, per-column missingness, absent hours, longest missing runs, and counts of unusual wind categories. Unexpected wind categories are logged and safely encoded rather than crashing inference.
6. Construct a complete hourly index between the first and last timestamp. Reindex onto it and mark inserted hours with `row_present=False`. Inserted rows remain missing; never invent sensor observations.
7. A checksum mismatch or unusable training partition should raise a clear error before fitting. Record the expected source size and date range as provenance checks; synthetic fixtures use their own documented checksum and range.

### Separate unknown labels from missing inputs

- **Current PM2.5 missing:** no forecast in this version; persistence would otherwise have no valid current value.
- **Target PM2.5 missing or target hour absent:** exclude the row from training/scoring. Never impute a target.
- **Historical lags, rolling features, or numeric weather missing:** retain eligible rows and use training medians in the fitted pipeline. Include fixed missingness flags for numeric feature columns so the model can recognize incomplete inputs.
- **Wind direction missing:** substitute the explicit category `missing`; one-hot encode categories learned from training, with unknown-category handling enabled.
- **Entire numeric feature missing in training:** fail with a clear message rather than silently dropping the feature.

Do not backfill, interpolate between past and future observations, or use global means. Do not forward-fill PM2.5 before constructing the target or baseline. Medians, scaling values, and category vocabularies are learned inside the training pipeline and reused unchanged for validation and test data.

Save exclusions separately by reason and partition, plus total unique excluded origins; reasons can overlap. Report forecast coverage as well as accuracy, because missing current readings can remove difficult periods.

## 4. Features and exact-hour alignment

Construct lags on the complete hourly grid **before dropping any rows**. A row-based shift is safe only after this reindexing and an explicit timestamp check. Alternatively, look up values by exact timestamp. Keep `origin_time` and `target_time` as audit columns, not model inputs.

| Feature group | Proposed inputs | Information available at t |
|---|---|---|
| Current pollution | `pm25(t)` | Latest observed concentration |
| Historical pollution | Values at t−1, t−2, t−3, t−6, t−12, t−24 hours | Exact elapsed-hour lookups |
| Recent summaries | Trailing 6-hour and 24-hour PM2.5 means, and observed-value counts | End at t; include current hour |
| Recent change | `pm25(t) − pm25(t−1)` | Missing if either value is unavailable |
| Weather | `DEWP`, `TEMP`, `PRES`, `cbwd`, `Iws`, `Is`, `Ir` at t | Current measurements only |
| Calendar | Sine/cosine of target hour and target month; target day of week | The future clock/calendar is known |
| Missingness | Fixed numeric missing flags and history counts | Describes input quality |

For the 6-hour mean use hours t−5 through t and require at least 3 observed values. For the 24-hour mean use t−23 through t and require at least 12. If fewer are available, leave the mean missing for training-median imputation. Counts explicitly describe how much history supports each mean.

Calendar sine/cosine pairs put adjacent clock times close together, including 23:00 and 00:00. Do not include the row number or raw year as a predictor in version 1.

UCI describes `Iws`, `Is`, and `Ir` as accumulated variables. Use their recorded current values without treating them as standalone next-hour wind or precipitation. Document that their accumulation/reset conventions need further investigation.

### The gap rule

Suppose records exist at 10:00 and 12:00 but 11:00 is absent:

- The 10:00 target is **11:00**, whose value is missing. This row is not supervised or scored.
- The 12:00 reading cannot become the 10:00 target.
- For origin 12:00, the one-hour lag is the missing 11:00 reading; the two-hour lag is 10:00.
- There must be no two-hour jump disguised as a one-hour forecast.

Do not remove missing PM2.5 rows and then call `shift(-1)`. Do not compute centered rolling windows. Changing observations after t must not change features or a forecast made at t, although changing the t+1 label will naturally change its scoring error.

## 5. Chronological splits

Assign partitions by **target timestamp**, using half-open intervals:

| Partition | Target timestamps in Beijing local time | Purpose |
|---|---|---|
| Training | 2010-01-01 00:00 ≤ target < 2013-01-01 00:00 | Fit preprocessing and models |
| Validation | 2013-01-01 00:00 ≤ target < 2014-01-01 00:00 | Choose model/settings |
| Test | 2014-01-01 00:00 ≤ target < 2015-01-01 00:00 | Final assessment |

Use only targets whose required observations actually exist. The first grid hour has no preceding forecast origin, and the last origin has no observed next-hour label; neither creates a fabricated training example.

Splitting by target keeps the label at a year boundary in the correct partition. For example, an origin of 2012-12-31 23:00 targets the first validation hour and must not become a training row.

Past history may cross a partition boundary. At the start of validation, past training readings are legitimate inputs. Later test forecasts may use earlier observed test readings, because those readings have arrived by that origin. Do not reset feature history at each split, and do not refit the model as test observations arrive.

No random shuffling, random train/test split, or ordinary shuffled cross-validation. A fixed validation year keeps this project understandable; rolling cross-validation is an optional future extension.

## 6. Persistence baseline

For each eligible origin, predict `prediction(t+1) = observed_pm25(t)`.

This baseline asks whether a model adds useful information beyond assuming pollution stays unchanged for one hour. Score it on the same validation/test target timestamps as the learned models. Do not substitute an imputed current reading for an observed baseline value.

## 7. Models, selection, and metrics

### Model A: Ridge regression

A regularized linear model: easy to explain and a useful check on whether recent readings plus weather improve the baseline.

- Numeric branch: training-median imputation, then `StandardScaler`; pass fixed missing flags without scaling if desired.
- Categorical branch: missing-category replacement and `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`.
- Use `ColumnTransformer` and `Pipeline` so fitting preprocessing is part of fitting the model.
- Try `alpha` values 0.1, 1, and 10.

### Model B: Random forest regression

An average of decision trees that can learn nonlinear patterns and weather/pollution interactions without deep learning.

- Training-median imputation, the same missing flags, and one-hot encoding; numeric scaling is unnecessary.
- Fix `n_estimators=200`, `min_samples_leaf=5`, `random_state=42`, and a modest `n_jobs=2`.
- Try `max_depth=10` and `max_depth=20` only.

Five configurations total are sufficient. Use identical features and eligible rows. Define nonnegative prediction clipping (`max(0, prediction)`) before model selection and apply it consistently during validation, final scoring, and inference. Do not clip high predictions or observed targets.

### Evaluation and decision rule

- **MAE:** average absolute error in µg/m³. This is the main selection metric and reads as a typical error size.
- **RMSE:** square root of average squared error, also in µg/m³. Large mistakes receive greater weight.
- Use validation MAE to select each model's settings and the best learned model. Break exact ties with lower RMSE, then prefer Ridge.
- Compare the chosen learned model with persistence. If persistence has lower or equal validation MAE, explicitly recommend persistence while still retaining the best learned model for the assignment comparison.
- Save the selected settings, selection rule, and training-fitted pipelines before opening final test results.
- **Do not refit on training + validation in this version.** Keep medians, scales, categories, and models fitted on 2010–2012 only, including final test evaluation.
- Final evaluation may report persistence, selected Ridge, and selected forest for an honest comparison, with the validation-selected model clearly identified. Do not switch the selected model because another method happens to win on test.

Save a prediction table with origin time, target time, observed target, current PM2.5, each prediction, signed errors, and input-history counts. Record sample counts and relative MAE improvement over persistence; guard against division by zero. Never claim achieved improvements before running the project.

## 8. Three required figures and high-pollution errors

Use training data for exploratory feature decisions. Validation diagnostics may inform decisions before freezing the plan. Test plots are final reporting only.

Produce only these three required figures. Multiple panels within a figure are acceptable; no separate diagnostic plots are required.

1. **Training-data pollution and missingness overview** (`training_overview.png`). Use two aligned panels covering 2010–2012: monthly mean observed PM2.5, and monthly missing-hour counts. Distinguish absent timestamp rows from present rows with missing PM2.5 so the same missing hour is not counted twice. Compute summaries from observed values before imputation. Preserve detailed per-column missingness, missing runs, exclusions, and coverage in the data-quality report.
2. **Actual versus predicted values for a fixed test period** (`test_predictions_fixed_period.png`). Plot actual PM2.5, persistence, and the validation-selected learned model for target timestamps from **2014-01-01 00:00 inclusive to 2014-01-08 00:00 exclusive**, in Beijing local time. Choose this period before viewing results; do not replace it with a better-looking week. Plot forecasts only at eligible scored timestamps, retain gaps without drawing connecting lines across missing hours, and display the scored-hour count. If coverage is poor, explain it rather than changing the window. The full test prediction CSV still contains both learned models.
3. **Model error comparison, including high-pollution periods** (`model_error_comparison.png`). Use separate MAE and RMSE panels comparing persistence, validation-tuned Ridge, and validation-tuned Random Forest on the final test set. Within each panel, show overall, high-pollution, and remaining-period errors. Clearly label the validation-selected model, units, subgroup counts, and training-derived high-pollution threshold. Keep validation selection metrics in the saved comparison table; do not select a model using this test figure.

Define **high pollution** as target PM2.5 at or above the training target's 90th percentile. Save that threshold before test evaluation. It is a project error-analysis definition, not a medical or regulatory category.

For high and remaining target concentrations, report sample counts, MAE, RMSE, mean signed error (`prediction − actual`), and the fraction of predictions below the actual value. Compare with persistence on the same slices. Show counts so a tiny subgroup cannot support a broad conclusion.

Save these error-slice statistics in the final metrics report alongside the third figure. If a subgroup is empty, report count 0 and null metrics; omit its bars with an explanatory note. Briefly discuss whether high concentrations are systematically underpredicted and whether the learned model improves on persistence. Explain patterns as associations, not proof that a weather variable caused an error.

### Future improvements — outside version 1

- Actual-versus-predicted scatterplots.
- Signed error versus concentration and MAE by target hour or month.
- A separate high-pollution episode plot.
- Rapid-change error analysis using a threshold learned from training data.
- Additional seasonal diagnostics or feature-importance plots.

These may be added in a later version; the Builder should not implement them as part of the first version. Any later test-set analysis must remain reporting only and must not feed back into model selection on the same holdout.

## 9. Meaningful pytest coverage

Use small hand-built hourly fixtures with known values and deliberate faults. Include one small end-to-end fixture spanning the configured partitions. Parameterize dates for fixtures so tests do not require five years of artificial data.

| Test | Expected behavior |
|---|---|
| Ordinary consecutive hours | Known features and exact next-hour target match manual calculations |
| Missing timestamp between two readings | No target bridges the gap; lag distances still represent real hours |
| Missing current PM2.5 | Origin excluded with an explicit reason |
| Missing next-hour PM2.5 | Label never imputed; no supervised/scored row |
| Partial history/start of dataset | Counts and minimum-observation rolling rules are correct; eligible rows can be imputed |
| Future values changed | Features and forecast at t stay unchanged; target is tested separately |
| Year/month/leap-day boundary | Target advances exactly one hour and calendar/split assignment is correct |
| Year-end origin targeting validation/test | Label cannot enter the preceding partition |
| Unsorted input | Sorting yields the same chronological features |
| Duplicate timestamp/invalid date/malformed value | Clear validation failure rather than silent correction |
| Negative PM2.5/infinite number | Validation fails; real high readings remain intact |
| Missing numeric weather | Training median is used and missing flag is set |
| Extreme validation/test inputs | Stored imputer statistics, scale parameters, categories, and fitted model remain unchanged |
| Unseen wind category | Inference succeeds with stable output feature shape |
| Entire training feature missing/empty eligible split | Clear early error |
| Persistence and metric arithmetic | Manually calculated predictions, MAE, and RMSE match |
| Common scoring population | Baseline and both models have identical target timestamps and counts |
| High-pollution slice | Threshold comes only from training; empty slice produces count 0 and null metrics |
| Fixed-period reporting | Uses the configured target-time window, retains missing-hour gaps, and handles an empty window without choosing another period |
| End-to-end smoke run | Expected files exist, the three required figures are produced (or explicitly annotated for empty fixture windows), predictions are finite/nonnegative, and metadata records seed/hash/splits |
| Repeat synthetic run | Same seed/config produces equivalent metrics and predictions within numeric tolerance |

Use a smaller forest only for the smoke fixture. Tests should catch incorrect behavior, not enforce a desired accuracy result. Separate feature construction from label construction so the future-mutation test can distinguish legitimate label changes from feature leakage.

## 10. Reproducible commands and CI

**These are proposed commands for the Builder to implement, not commands that work yet.** Run from the repository root.

### Local setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-deps --no-build-isolation -e .
```

The pinned dependency file must include required packaging/build dependencies as well as runtime and test dependencies. Installation needs internet or an existing wheel cache; ordinary execution does not.

### Validate, test, select, and evaluate

```bash
python -m air_quality validate --config config.json
python -m pytest -q
python -m air_quality select --config config.json
python -m air_quality evaluate --config config.json --selection artifacts/selected_config.json
```

`select` fits on training and scores validation only. `evaluate` loads the frozen selection and training-fitted models to produce final test reports; it does not fit or tune anything. The selection manifest references model artifacts and records their configuration/hash. Final evaluation rejects a mismatched dataset or configuration.

Rerunning the unchanged final evaluation for reproducibility is acceptable. Changing the model after inspecting test performance makes the old test period development data; document that and obtain a new untouched holdout for any renewed unbiased assessment.

### Docker

```bash
docker build -t air-quality-forecasting:local .
mkdir -p artifacts
docker run --rm --network none air-quality-forecasting:local python -m pytest -q
docker run --rm --network none -v "$PWD/artifacts:/app/artifacts" air-quality-forecasting:local python -m air_quality select --config config.json
docker run --rm --network none -v "$PWD/artifacts:/app/artifacts" air-quality-forecasting:local python -m air_quality evaluate --config config.json --selection artifacts/selected_config.json
```

Use `/app` as the image working directory. Copy code, tests, configuration, and the checked-in static CSV into the image. Install pinned dependencies during build. Exclude `.venv`, `.git`, caches, and generated artifacts from the build context. Pin the chosen Python 3.11 image by digest during implementation and record it.

GitHub Actions should run on pushes and pull requests: install Python 3.11 dependencies, install the package, validate the checked-in CSV, run pytest, build the image, and execute `python -m air_quality smoke` inside the container with `--network none`. That smoke command uses deterministic synthetic data and reduced model settings, runs the complete workflow, and verifies outputs. CI must not repeatedly select models using or score the real final test year.

Record Python/package versions, seed, effective configuration, source checksum, eligible counts per split, selected parameters, and Git commit when available in `run_metadata.json`. Exact byte-for-byte model files across different computers are not promised; prediction/metric agreement within a small numeric tolerance is the reproducibility target.

## 11. Risks and limitations

- **Strong baseline:** one-hour pollution often changes slowly. Added model complexity may not beat persistence.
- **Selective coverage:** missing current or target readings are excluded. Report what fraction of hours is represented; scores do not cover all outages.
- **Historical/local scope:** 2010–2014 readings from one pollution location do not establish performance in current Beijing or other cities.
- **Spatial mismatch:** airport weather may not match conditions at the pollution monitor.
- **Publication-delay assumption:** real deployment would need verified arrival times and possibly older input readings.
- **Accumulated weather fields:** resetting conventions may complicate interpretation.
- **Extreme episodes:** limited examples and regression toward typical values can cause underprediction at high concentrations. Analyze this explicitly.
- **Validation uncertainty:** choosing among five configurations on one year is manageable but less robust than repeated chronological validation.
- **Operational limits:** this is retrospective one-step regression, not an official air-quality warning system. No uncertainty intervals or multi-hour forecasts are included.
- **Reproducibility limits:** training randomness, package versions, hardware, and changed input files can affect results; pins, hashes, and metadata reduce these differences.

## 12. Implementation sequence for the Builder

1. Follow the approved prediction-time contract, training-only fitting rule, and three-figure scope in this finalized plan when implementation is separately requested.
2. Scaffold the package/configuration; download and preserve the static CSV with attribution and checksum.
3. Implement loading, validation, gap reporting, and hourly reindexing. Add validation/gap tests first.
4. Build causal features and separate exact-hour labels; implement eligibility reports and target-time splits. Verify with tiny manual examples before fitting anything.
5. Implement persistence and MAE/RMSE; verify identical scoring populations and metric arithmetic.
6. Implement Ridge and forest preprocessing pipelines and the five-config validation comparison.
7. Freeze validation selection and training-fitted artifacts. Implement held-out evaluation, high-pollution error reports, and only the three required figures in Section 8.
8. Complete the end-to-end tests, pinned setup, Docker image, offline smoke command, and GitHub Actions.
9. Run the full local/container workflow, review outputs, and write README results and limitations.

### Approved decisions and scope

- Current-hour pollution and weather are assumed available when the one-hour forecast is issued.
- Skip forecasts when current PM2.5 is missing; never impute supervised targets.
- Use persistence, Ridge regression, and a small Random Forest.
- Use 2010–2012 training, 2013 validation, and 2014 test, with no training-plus-validation refit.
- Keep timestamp-gap checks, leakage prevention, missing-data reports, and meaningful typical-case and edge-case tests.
- Require only the three figures in Section 8; defer other plots and optional analysis.
- This document finalizes the design only. Wait for a separate implementation request before building the project.

**Interview explanation:** “I forecast the next hour's pollution from information already available. I check timestamps so missing hours never become incorrect targets, compare two simple models with persistence, fit preprocessing only on historical training data, and make the workflow reproducible with tests, Docker, and CI.”

## References

- [UCI dataset and variable documentation](https://archive.ics.uci.edu/dataset/381/beijing+pm2+5+data).
- [scikit-learn pipelines and composite estimators](https://scikit-learn.org/stable/modules/compose.html).
- [scikit-learn SimpleImputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html).
- [scikit-learn OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html).
