# Run the project on your Mac in VS Code

## 1. Download, unzip, and open the folder

Download `air-quality-forecasting.zip` and double-click it in Finder. Move the extracted `air-quality-forecasting` folder wherever you keep course projects. In VS Code choose **File → Open Folder** and open that folder, not its parent. Open **Terminal → New Terminal**.

Confirm that your terminal is at the project root:

```bash
pwd
ls
```

You should see `README.md`, `config.json`, `src`, `tests`, and `data`. Hidden files are present in the ZIP; VS Code's Explorer shows `.github`, `.gitignore`, and `.dockerignore`. Finder toggles hidden files with Command + Shift + period.

## 2. Check Python 3.11

```bash
python3.11 --version
```

Use Python 3.11 for the pinned environment, even if another `python3` version is installed. If `python3.11` is unavailable and you already use Homebrew:

```bash
brew install python@3.11
```

If that does not put it on PATH, use `$(brew --prefix python@3.11)/bin/python3.11` in place of `python3.11` in the next step. Homebrew information: https://brew.sh/ and https://formulae.brew.sh/formula/python@3.11. Do not install dependencies into your Mac's system Python.

## 3. Create and activate a virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python --version
```

The terminal prompt normally shows `(.venv)`. If you open another terminal later, run `source .venv/bin/activate` again from the project root.

## 4. Install the pinned dependencies and local package

```bash
python -m pip install -r requirements.txt
python -m pip install --no-deps --no-build-isolation -e .
python -m pip check
```

This step needs internet the first time. All normal pipeline commands then use the included CSV offline. Exact versions, including packaging and test dependencies, are in `requirements.txt`. `pytest` is included there for local development and CI, but is intentionally not a runtime dependency in `pyproject.toml`. Use these commands rather than updating scikit-learn or pandas independently.

Install the Microsoft **Python** extension in VS Code if needed. Press Command + Shift + P, choose **Python: Select Interpreter**, and select the `.venv/bin/python` interpreter in this project. If it is not listed, choose **Enter interpreter path** and browse to it.

## 5. Check data and tests before training

```bash
python -m air_quality validate --config config.json
python -m pytest -q
python -m air_quality smoke
```

Expected outcomes: data validation succeeds, all tests pass, and the synthetic smoke command prints `status: passed`. Smoke is a temporary synthetic run; it does not retrain on or score the real 2014 test year.

## 6. Train and select using validation data

```bash
python -m air_quality select --config config.json
```

This fits preprocessing and models on 2010–2012 and chooses settings using 2013. It writes `artifacts/selected_config.json`, `validation_metrics.csv`, and fitted models. The included reference run chose Ridge, alpha 0.1. Reproduction should agree within small numerical tolerance; model-file bytes are not guaranteed identical on a different platform.

## 7. Evaluate the frozen choice

```bash
python -m air_quality evaluate --config config.json --selection artifacts/selected_config.json
```

This loads the fitted pipelines and reports the 2014 holdout without fitting. Open these files in VS Code:

- `artifacts/test_metrics.json` for errors and coverage.
- `artifacts/test_predictions.csv` for each scored hour.
- The three PNGs in `artifacts/plots/`.

The `artifacts/` directory is generated locally and ignored by Git; a clean GitHub checkout does not include reference artifacts. Running steps 6–7 creates or replaces the local outputs. Keep model choices unchanged when reproducing the already viewed holdout. For model redesign, use a new untouched holdout before claiming fresh out-of-sample performance.

If you see a configuration or dataset mismatch, restore the original config/CSV to reproduce this run. If you intentionally make a new development experiment, run selection again and document the change rather than editing the frozen manifest.

## 8. Check Docker separately

Install and open [Docker Desktop for Mac](https://docs.docker.com/desktop/setup/install/mac-install/) if needed. Wait until its engine is running. Check `docker version`; both client and server should appear.

From the project root, with Docker Desktop running:

```bash
docker build -t air-quality-forecasting:local .
mkdir -p artifacts
docker run --rm --network none air-quality-forecasting:local python -m pytest -q
docker run --rm --network none air-quality-forecasting:local python -m air_quality smoke
docker run --rm --network none -v "$PWD/artifacts:/app/artifacts" air-quality-forecasting:local python -m air_quality select --config config.json
docker run --rm --network none -v "$PWD/artifacts:/app/artifacts" air-quality-forecasting:local python -m air_quality evaluate --config config.json --selection artifacts/selected_config.json
```

Build needs internet; the runs explicitly disable networking. The quoted `$PWD` mount works when your folder path contains spaces. Container selection replaces the model artifacts on your Mac; evaluate with that same selection afterward. The image digest is multi-platform for Intel and Apple Silicon. This workflow was successfully verified on macOS with 37 container tests, offline selection/evaluation, and all three figures opening correctly.

If `docker: command not found` appears, Docker's command-line tools are not installed/available. If the engine connection fails, launch Docker Desktop and wait. If a shared-directory permission dialog appears, allow Docker to access your project/artifacts folder.

## 9. Complete your own assignment evidence

`docs/manual_smoke_test.md` records the completed macOS commands, dates, and results. Update it only if your own reproduction differs. Write your own reflection in `docs/reflection.md`, and put genuine conversation records in `docs/conversations.md` or link to your exported files; those student-owned materials are not generated by the project.

## 10. Put the source on GitHub

Keep `.github/workflows/ci.yml`, `.gitignore`, `.dockerignore`, config, tests, package, and the raw CSV. Generated `artifacts/` and `.venv/` are ignored. The repository's GitHub Actions workflow has a successful run covering static-data validation, pytest, Docker build, and offline smoke. After future pushes, check the **Actions** tab for the new workflow result.

A simple first commit from the project root, after creating your own empty GitHub repository:

```bash
git init
git add .
git status
git commit -m "Build reproducible exact-hour air quality pipeline"
```

Review `git status` before the commit: it should include the static dataset and hidden configuration but exclude `.venv` and `artifacts`. Then follow GitHub's displayed commands to add your own remote and push. Do not guess the repository URL.
