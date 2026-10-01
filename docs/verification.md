# Project verification — October 1, 2026

This document separates the checks run by the Builder in Linux from the completed macOS manual run, the successful GitHub Actions workflow, and the independent Tester review. It does not replace the student's manual smoke-test record.

## Historical Builder environment and dependency resolution

- CPython **3.11.16**, Linux x86_64, isolated project virtual environment.
- Pinned direct and transitive dependencies plus pip, setuptools, wheel, and the test runner in `requirements.txt`.
- `pytest` is intentionally a development/CI dependency in `requirements.txt`, not a runtime dependency in `pyproject.toml`; the installed package does not need pytest to run the forecasting CLI.
- Dependencies installed in this environment, editable package installed using `python -m pip install --no-deps --no-build-isolation -e .`.
- `python -m pip check`: **passed**, no broken requirements.
- Initial strict tests exposed a matplotlib/pyparsing deprecation warning; `pyparsing==3.2.3` is the tested compatible pin. The suite treats warnings as errors.

## Historical Builder checks before the final targeted fixes

| Check | Actual outcome |
|---|---|
| Original UCI archive downloaded and CSV extracted | Original static CSV present; SHA-256 matches config and data provenance |
| Approved plan preservation | Byte comparison passed; `docs/plan.md` unchanged |
| `python -m air_quality validate --config config.json` | Passed: 43,824 raw rows, 0 absent hours, 2,067 missing pollution readings |
| `python -m pytest -q` | **37 tests passed**; includes numerical regression, parametrized failure cases, and synthetic end-to-end/repeat runs |
| `python -m air_quality smoke` | Passed, 48 synthetic test rows; required files and exactly three PNGs verified |
| `python -m air_quality select --config config.json` | Completed full five-configuration comparison; Ridge alpha 0.1 selected on validation |
| `python -m air_quality evaluate --config config.json --selection artifacts/selected_config.json` | Completed frozen evaluation on 8,626 real test rows |
| Python offline execution check | Synthetic smoke and real frozen evaluation passed with `socket.socket`, `socket.create_connection`, and `socket.getaddrinfo` replaced by failing functions |
| `python -m compileall -q src tests` | Passed |
| Figure review | All three real-data PNGs visually reviewed; fixed window has 168/168 scored hours; gaps tested synthetically |
| Frozen artifacts during evaluation | Tests verify unchanged model/selection hashes and explicitly prohibit `Pipeline.fit` |
| Output population/values | Finite nonnegative predictions and exact one-hour origin/target differences verified |

The Python network check blocks ordinary Python networking. It is **not** OS-level network isolation or a Docker `--network none` run. No container claim is based on that check.

Synthetic reproducibility uses floating-point tolerances. Forest worker scheduling can alter final summation bits; metrics are not required to have exact dictionary/byte equality.

## Historical manual Docker checks and initial external review

| Check | Actual outcome |
|---|---|
| macOS Docker build | Passed on October 1, 2026 with the `air-quality-forecasting:local` image |
| macOS container tests | **37 passed** with `--network none` |
| macOS Docker synthetic smoke | Passed with `--network none` |
| macOS Docker selection and evaluation | Passed offline; Ridge selected/recommended and 8,626 eligible 2014 test rows scored |
| macOS generated outputs and figures | Expected outputs persisted locally; all three required PNGs opened correctly |
| GitHub Actions | Successful workflow run: static-data validation, pytest, Docker build, and offline synthetic smoke passed |
| Initial independent Tester review of `3852c14` | No confirmed algorithmic defect found through source/test inspection and existing CI evidence. Setup instructions were mostly satisfactory; documentation and reproducibility issues were identified. No Tester-owned fresh clone, pytest run, or Docker build was completed because of an environment approval-limit error. |

The generated `artifacts/` directory is intentionally ignored by Git. The JSON, CSV, model, and PNG outputs listed above are produced locally by selection/evaluation and are not expected to be present in a clean GitHub checkout.

## Checks not run directly by the Builder

- **Docker build and container runs in this Linux environment:** not run because the Docker executable/engine was absent. The Dockerfile base digest was retrieved from Docker Hub, not built here. The successful macOS and CI results above are the relevant Docker evidence.
- **GitHub Actions from this local process:** not triggered by the Builder; the successful workflow result was observed in the repository/Tester stage.
- **Student-owned reflection and transcript exports:** not generated or reconstructed by the Builder. Keep those genuine and complete them separately for submission.

The original Builder directory had no accessible Git metadata, so its local `run_metadata.json` recorded `git_commit: null`. Historical reference: the initial independent review covered commit `3852c14` and cited a [successful CI run](https://github.com/snehamalakar2000/Air-Quality-Forecasting/actions/runs/36893672345). Neither reference identifies the later targeted-fix revision.

## Follow-up metadata and configuration fixes — October 1, 2026

The Builder prepared targeted changes against base commit `78a145e`; this is the historical base, not the identifier of the resulting fix commit. Changes were limited to `src/air_quality/pipeline.py` and `tests/test_pipeline.py`:

- Pytest version reporting is optional when pytest is absent; required runtime dependencies remain required.
- Configuration validation rejects empty model grids, invalid tree counts, invalid minimum leaf sizes, invalid worker counts, and missing required source metadata keys.
- Regression tests cover invalid settings and valid boundary cases. Forecasting methodology and feature logic were unchanged.

| Evidence source | Reported or observed result |
|---|---|
| Builder execution, Python 3.12 | **60 tests passed**; compileall, pip check, dataset validation, and git diff whitespace check passed |
| Builder smoke without pytest metadata | Passed with 48 synthetic test rows; pytest metadata was confirmed absent |
| Student macOS execution after applying fixes | Dataset validation passed: 43,824 rows, 0 absent hours; synthetic smoke passed with 48 test rows; git diff whitespace check produced no errors |
| Student observation after pushing fixes | GitHub Actions showed a successful run |
| Independent Tester follow-up | Remaining findings concerned stale documentation/references, CI maintenance warnings, and deeper alpha/depth grid validation |

The Builder used Python 3.12 for the follow-up test run because its bundled Python 3.11 interpreter was unavailable. It did not rerun Docker. The 37-test macOS Docker record above describes the earlier manual run; it is not evidence of a new local Docker run after these fixes. No new student-local pytest count is asserted here.

The exact follow-up commit hash and run URL were not supplied for this documentation update. See the [repository Actions history](https://github.com/snehamalakar2000/Air-Quality-Forecasting/actions) for the run corresponding to the pushed metadata/configuration fixes. The successful CI observation predates this documentation-only update.

## Remaining maintenance and robustness work

- The Tester reported Node.js 20 action-runtime and upcoming `ubuntu-latest` migration warnings. These were maintenance notices, not failures of the reported CI run.
- Model grids are checked for non-emptiness, but not every alpha/depth element has comprehensive early type and numeric-range validation. This remains a lower-priority robustness limitation.
- This documentation update removes stale claims that CI has not run and labels earlier evidence as historical. It does not claim a new test execution or a new independent review of these edited documents.
