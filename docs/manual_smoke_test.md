# My manual smoke-test record

The Builder's automated checks are recorded separately in `verification.md`. The results below are my own macOS checks from October 1, 2026.

| Check I perform | Date/time | Command or action | Actual result/evidence |
|---|---|---|---|
| Mac setup and interpreter | October 1, 2026 | Conda environment with Python 3.11.16 | Completed |
| Dependency installation | October 1, 2026 | Installed the updated pinned dependencies, including NumPy 2.3.5 | Completed; the earlier Ridge warning was resolved |
| Dataset validation | October 1, 2026 | Completed as part of the project workflow | Completed |
| pytest in the Docker container | October 1, 2026 | `docker run --rm --network none air-quality-forecasting:local python -m pytest -q` | **37 passed** |
| Synthetic smoke in the Docker container | October 1, 2026 | `docker run --rm --network none air-quality-forecasting:local python -m air_quality smoke` | Completed without error |
| Docker build | October 1, 2026 | `docker build --progress=plain -t air-quality-forecasting:local .` | Completed without error |
| Model selection in Docker | October 1, 2026 | `docker run --rm --network none -v "$PWD/artifacts:/app/artifacts" air-quality-forecasting:local python -m air_quality select --config config.json` | Ridge selected; Ridge recommended |
| Frozen evaluation in Docker | October 1, 2026 | `docker run --rm --network none -v "$PWD/artifacts:/app/artifacts" air-quality-forecasting:local python -m air_quality evaluate --config config.json --selection artifacts/selected_config.json` | Completed; 8,626 eligible test rows |
| Saved-output inspection | October 1, 2026 | Inspected `artifacts/` and `artifacts/plots/` with `ls -lh` | Expected artifacts were present and nonzero |
| Figure inspection | October 1, 2026 | Opened `training_overview.png`, `test_predictions_fixed_period.png`, and `model_error_comparison.png` | All three opened correctly |
| GitHub Actions | TODO | TODO | Not tested yet |

Problems I encountered and how I resolved them: The original Conda environment produced nine Ridge failures with `RuntimeWarning: divide by zero encountered in matmul`. Updating NumPy from 2.2.6 to 2.3.5 resolved the issue. After the update, 37 tests passed.

Still pending: GitHub Actions and the independent Tester stage. Include terminal output or screenshots if my assignment requires them. An unavailable or failed check should be recorded honestly.
