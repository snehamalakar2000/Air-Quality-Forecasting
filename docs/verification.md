# Builder verification — October 1, 2026

These are checks performed by the Builder in the Linux execution environment. They are not the student's manual smoke-test record and do not establish Mac, Docker, or GitHub Actions success.

## Environment and dependency resolution

- CPython **3.11.16**, Linux x86_64, isolated project virtual environment.
- Pinned direct and transitive dependencies plus pip, setuptools, and wheel in `requirements.txt`.
- Dependencies installed in this environment, editable package installed using `python -m pip install --no-deps --no-build-isolation -e .`.
- `python -m pip check`: **passed**, no broken requirements.
- Initial strict tests exposed a matplotlib/pyparsing deprecation warning; `pyparsing==3.2.3` is the tested compatible pin. The suite treats warnings as errors.

## Checks actually run

| Check | Actual outcome |
|---|---|
| Original UCI archive downloaded and CSV extracted | Original static CSV present; SHA-256 matches config and data provenance |
| Approved plan preservation | Byte comparison passed; `docs/plan.md` unchanged |
| `python -m air_quality validate --config config.json` | Passed: 43,824 raw rows, 0 absent hours, 2,067 missing pollution readings |
| `python -m pytest -q` | **35 tests passed**; includes parametrized failure cases and synthetic end-to-end/repeat runs |
| `python -m air_quality smoke` | Passed, 48 synthetic test rows; required files and exactly three PNGs verified |
| `python -m air_quality select --config config.json` | Completed full five-configuration comparison; Ridge alpha 0.1 selected on validation |
| `python -m air_quality evaluate --config config.json --selection artifacts/selected_config.json` | Completed frozen evaluation on 8,626 real test rows |
| Python offline execution check | Synthetic smoke and real frozen evaluation passed with `socket.socket`, `socket.create_connection`, and `socket.getaddrinfo` replaced by failing functions |
| `python -m compileall -q src tests` | Passed |
| Figure review | All three real-data PNGs visually reviewed; fixed window has 168/168 scored hours; gaps tested synthetically |
| Frozen artifacts during evaluation | Tests verify unchanged model/selection hashes and explicitly prohibit `Pipeline.fit` |
| Output population/values | Finite nonnegative predictions and exact one-hour origin/target differences verified |

The Python network check blocks ordinary Python networking. It is **not** OS-level network isolation or a Docker `--network none` run. No claim of container success follows from it.

Synthetic reproducibility uses floating-point tolerances. Forest worker scheduling can alter final summation bits; metrics are not required to have exact dictionary/byte equality.

## Not run here — complete locally or in your repository

- **Docker build and container runs:** not run because the Docker executable/engine is absent. The Dockerfile base digest was retrieved from the Docker Hub registry, not built. Pinned multi-platform image: `python:3.11-slim-bookworm@sha256:a36c24f9cbdf4fd0f52d67f0823eeac19c2028c637cecc392d97f980d4fec56b`.
- **GitHub Actions:** workflow supplied but not executed here. It validates the static source, runs pytest, builds Docker, and runs synthetic smoke offline; it does not select or score the real holdout.
- **Mac/VS Code execution:** instructions supplied in `mac_setup.md`; not verified on your Mac.
- **Your manual smoke-test record, reflection, and transcripts:** intentionally left for you. The TODO templates are not completed evidence.

There was no accessible Git repository/commit in the Builder's project directory; `run_metadata.json` records `git_commit: null`. After you run the pipeline in a committed repository it records the accessible commit automatically.
