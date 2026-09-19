# EPFL CS-433 Project 1 — MICHD

Binary classification on CDC BRFSS 2015: predict myocardial infarction / coronary heart disease (`_MICHD` in `{+1, -1}`).

Competition: [AIcrowd — EPFL Machine Learning Project 1](https://www.aicrowd.com/challenges/epfl-machine-learning-project-1).

NumPy only in the pipeline (no pandas, no scikit-learn). matplotlib/seaborn are for plots in notebooks.

## Layout

```text
implementations.py   # six required methods
helpers.py           # load_csv_data, create_csv_submission
run.py               # baseline: preprocess → least squares / logistic → CSV
src/preprocess.py    # drop columns, sentinels, median impute, standardize
data/                # CSVs (gitignored) — see data/README.md
doc/                 # tutorials, EDA, project plan (not graded)
```

## Data

Download the three AIcrowd files and put them in `data/` **without renaming**:

- `x_train.csv`, `y_train.csv`, `x_test.csv`

Optional: `sample_submission.csv`. Do not commit CSVs.

`run.py` uses `data/cache_eda.npz` when it exists (faster reload). Delete that file to force a full `np.genfromtxt` load.

## Environment

Coding env (example): conda env `ml` with Python 3.12 + NumPy.

Public tests: conda env from `grading_tests/environment.yml` (name `project1-grading`). On Apple Silicon the pinned `scikit-learn=1.1.2` may be missing; any recent NumPy + pytest + gitpython + black is enough for the public checks.

```bash
conda activate ml   # or project1-grading
```

## Run the baseline

From the **repository root**:

```bash
python run.py
```

This:

1. Loads train/test (cache or CSVs).
2. Shuffles and holds out 20% of train (`seed=0`) **before** computing medians.
3. Fits `src/preprocess.py` on the train fold only (drop admin/leakage/>90% missing, BRFSS sentinels, median impute, standardize).
4. Trains `least_squares` (labels `{+1,-1}`) and `logistic_regression` (labels `{0,1}`, 50 GD steps, `gamma=0.1`).
5. Prints validation accuracy / recall / F1 of the **+1** class (accuracy alone is misleading: ~91% of people are `-1`).
6. Writes `submission_baseline.csv` (`Id,Prediction`) using the model with higher val F1+.

First CSV load can take tens of seconds; logistic GD on ~260k rows also takes tens of seconds.

Upload `submission_baseline.csv` to AIcrowd to check the format. It is gitignored (`*.csv`).

## The six methods

`implementations.py` (each returns `(w, loss)`):

- `mean_squared_error_gd` / `mean_squared_error_sgd` (SGD batch size 1)
- `least_squares` / `ridge_regression` (no `numpy.linalg.lstsq`; ridge loss is **unregularized** MSE)
- `logistic_regression` / `reg_logistic_regression` (`y ∈ {0,1}`; regularized logistic returns **unregularized** NLL)

MSE uses the lecture factor `1/(2N)`.

## Public tests

```bash
conda activate project1-grading
cd grading_tests
pytest --github_link "https://github.com/mia-san-king/ML_project_1/tree/<FULL_COMMIT_HASH>" .
```

While iterating you may pass a local folder as `--github_link` and ignore `test_github_link_format`.

## Notes

- Do not treat BRFSS codes 7/9/88 as missing on every column (`_AGEG5YR` 7 and 9 are real age groups; `PHYSHLTH` 88 means zero days).
- `HAREHAB1` and related rehab/aspirin items are dropped as leakage.
- Team tutorials: `doc/helpers.ipynb`, `doc/eda.ipynb`, `doc/preprocess_tutorial.ipynb`, `doc/run_tutorial.ipynb`.
