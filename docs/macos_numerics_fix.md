# Targeted fix for Ridge matmul warnings on macOS

## Finding and scope

The supplied Mac output reports 26 passes and 9 failures. Every failure enters Ridge's default Cholesky path and stops at `safe_sparse_dot(X.T, X)`, which invokes NumPy matrix multiplication. This happens before the Cholesky solve. pytest correctly treats the RuntimeWarning as an error.

With the original Python 3.11.16 / NumPy 2.2.6 pins on Linux, inspected float64 transformed training inputs were:

| Source | Matrix shape | All inputs / targets finite | Largest absolute input |
|---|---|---|---|
| Synthetic test fixture | 139 × 48 | Yes / yes | 7.30595184 |
| Original real CSV training partition | 24,274 × 50 | Yes / yes | 31.80927306 |

Synthetic centered features have rank 30 out of 48 because some columns are constant or dependent. That is valid for positive-alpha Ridge and does not explain division by zero during matrix multiplication. The observed finite magnitudes are far below overflow territory. Missing raw lags are intentional and are imputed inside the training-fitted preprocessing pipeline.

NumPy has documented spurious matrix-multiplication warnings on Apple ARM systems, including M4 with Accelerate. NumPy 2.3.1 introduced an initial fix, and 2.3.5 includes a broader runtime floating-point-error check for Apple ARM builds. The original 2.2.6 pin predates both fixes. Matching package version numbers across Linux and macOS do not guarantee the same BLAS/LAPACK implementation.

This is the leading explanation for the supplied trace, not proof of the Mac's CPU/backend. The log does not identify its architecture or loaded numerical libraries. Run the supplied diagnostic to establish those details locally. A failed identity-matrix multiplication independently of project data is strong evidence of a numerical-stack issue.

## Exact changes

The dependency fix changes only these two existing files:

1. `requirements.txt`: replace `numpy==2.2.6` with `numpy==2.3.5`.
2. `pyproject.toml`: replace `"numpy==2.2.6"` with `"numpy==2.3.5"` in project dependencies.

Keep both pins consistent. Other dependency pins, model code, feature definitions, configuration, alphas, tests, and warning settings are unchanged. SciPy 1.15.3 declares NumPy `>=1.23.5,<2.5`, which includes 2.3.5. There is no environment-variable change required by this fix.

The patch also adds:

- `scripts/diagnose_ridge.py`: prints Python/OS/architecture, NumPy's build configuration and detected thread pools, then checks transformed inputs, an identity matrix, the centered Gram matrix against direct summation, and Ridge fitting/prediction. It converts numerical warnings to failures and exits nonzero if a probe fails. It does not modify project artifacts.
- `tests/test_numerics.py`: two additional tests for finite preprocessed inputs and matrix products versus direct sums. All 35 original tests remain.
- This document, `docs/macos_numerics_fix.md`.

SVD is not the primary fix: it still calls matrix multiplication internally and would not repair a broken numerical backend. The solver remains `auto`, choosing Cholesky for these dense inputs.

## Apply on the Mac

Activate the environment and work from the folder containing `config.json`:

```bash
conda activate air-quality
pwd
python --version
```

Merge the five patch files into their matching paths in your existing project. If the patch ZIP is in Downloads, run from the project root:

```bash
unzip -o "$HOME/Downloads/air-quality-macos-numpy-fix.zip" -d .
```

Only the five listed files are included; data, config, model outputs, plan, and student-written documents are not in this patch. Before installing the new dependency, you can record the original environment's behavior:

```bash
python scripts/diagnose_ridge.py > ridge-diagnostic-before.txt 2>&1
cat ridge-diagnostic-before.txt
```

That diagnostic may fail in the affected environment; it is intended to expose the failure. `inputs_finite=True` and `targets_finite=True` check the values on your actual Mac. The NumPy build configuration's BLAS section helps identify Accelerate or OpenBLAS; threadpoolctl does not always list Accelerate.

Install the new NumPy pin and refreshed editable project metadata together:

```bash
python -m pip install --no-build-isolation -r requirements.txt -e .
python -m pip check
python -c "import numpy; print(numpy.__version__, numpy.__file__)"
```

The version should be `2.3.5`; the path should point inside your `air-quality` Conda environment. Using one install command avoids a temporary conflict between the old package metadata's NumPy pin and the new requirements pin. A Conda environment can host pip-installed wheels; the relevant fact is what NumPy actually imports, not the environment's name.

Restart any running notebook kernel or Python console so it imports the updated NumPy. Each terminal command below starts a fresh Python process:

```bash
python scripts/diagnose_ridge.py > ridge-diagnostic-after.txt 2>&1
cat ridge-diagnostic-after.txt
python -m pytest -q
python -W error::RuntimeWarning -m air_quality smoke
```

Expected: diagnostic probes pass, 37 tests pass if the two new tests were copied, and synthetic smoke passes with 48 scored test rows. If you make only the two dependency edits, the original suite still contains 35 tests.

After those checks pass, regenerate model artifacts with the corrected environment:

```bash
python -m air_quality select --config config.json
python -m air_quality evaluate --config config.json --selection artifacts/selected_config.json
```

This repeats the same experiment using the repaired dependency; it is not a reason to change model choices after seeing test results. The original downloaded artifacts were produced with NumPy 2.2.6; this patch deliberately does not replace them or claim they were generated with the new version. Your rerun records the new version in run metadata.

If a diagnostic or test still fails, keep its output and run:

```bash
python -m pip check
conda list
```

Provide the new diagnostic and first traceback. A remaining failure needs the actual OS, architecture, backend, and imported package paths before another change is justified. Do not disable warnings, remove tests, set `np.seterr(...='ignore')`, or change model settings to make the error disappear. This fix does not require thread-count or CPU-feature environment variables.

## What was actually verified by the Builder

On Linux x86_64, Python 3.11.16:

- Original NumPy 2.2.6 transformed matrices and labels were finite; the original failing operation succeeded on Linux/OpenBLAS.
- NumPy 2.3.5 installed alongside the original other pins; refreshed package metadata and `pip check` passed.
- All 35 original tests plus 2 new tests passed (37 total), with the original warnings-as-errors configuration.
- Diagnostic identity, finite-input, direct-sum/Gram agreement, Ridge fit, and validation-prediction checks passed for synthetic and real training inputs.
- Synthetic smoke passed with 48 test rows.
- Compared real validation predictions before/after the dependency update for all three Ridge alphas. Maximum absolute differences were approximately 1.48e-12 (alpha 0.1), 8.53e-13 (alpha 1), and 3.41e-13 (alpha 10) µg/m³.

The Mac error was not reproduced in the Builder environment because it uses Linux/OpenBLAS, not the user's Mac numerical stack. The subsequent macOS verification reported that NumPy 2.3.5 resolved the warning and that all 37 tests passed. Docker and GitHub Actions were verified afterward and are recorded in `docs/manual_smoke_test.md` and `docs/verification.md`; those checks were not part of the original Linux patch investigation. Final real test evaluation was not rerun as part of that investigation; the real-data comparison there used validation only.

## Primary references

- NumPy's original issue: https://github.com/numpy/numpy/issues/28687
- Additional affected builds: https://github.com/numpy/numpy/issues/29820
- Initial fix: https://numpy.org/devdocs/release/2.3.1-notes.html
- Broader Apple ARM fix, PR #30237 in the release list: https://numpy.org/doc/stable/release/2.3.5-notes.html
- Ridge solver documentation: https://scikit-learn.org/1.6/modules/generated/sklearn.linear_model.Ridge.html
