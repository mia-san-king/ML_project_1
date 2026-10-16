# EPFL CS-433 Project 1 — MICHD

Binary classification on CDC BRFSS 2015: predict myocardial infarction / coronary heart disease (`_MICHD` in `{+1, -1}`).

Competition: [AIcrowd — EPFL Machine Learning Project 1](https://www.aicrowd.com/challenges/epfl-machine-learning-project-1).

NumPy only in the pipeline (no pandas, no scikit-learn). matplotlib/seaborn are for plots in notebooks.

## Layout

```text
implementations.py   # six required methods
helpers.py           # load_csv_data, create_csv_submission
run.py               # preprocess → method from config.yaml → CSV
config.yaml          # phase 1 tune / phase 2 submit; method and hyperparameters
src/preprocess.py    # drop columns, sentinels, median impute, standardize
data/                # competition CSVs (gitignored)
outputs/             # submission CSV from phase 2
doc/                 # tutorials, EDA, project plan (not graded)
report/              # 2-page LaTeX report → `cd report && make`
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

## Run training (`config.yaml`)

From the **repository root**:

```bash
python run.py
```

Optional: `python run.py path/to/other.yaml`

Training is split into two phases (set `phase` in [`config.yaml`](config.yaml)):

| Phase | What it does |
| --- | --- |
| **1** | Train on 80% of labeled rows; grid-search threshold (and `lambda` for `ridge`) on the 20% validation split. Prints a YAML snippet to paste back into the config. **Does not** write a submission CSV. |
| **2** | Uses the hyperparameters already in the config (defaults reproduce the best logistic run). Refits preprocess + model on **all** training rows, predicts test, writes `outputs/submission_baseline.csv`. |

**Default (shipped config):** `phase: 2`, `method: logistic`, `max_iters: 200`, `gamma: 0.1`, `threshold: 0.23` — matches validation F1+ ≈ 0.418 and AIcrowd public F1+ **0.432**.

**Methods** (`method:` in config — one at a time):

- `logistic` — plain logistic regression (labels `{0,1}`; threshold on probability)
- `reg_logistic` — L2-regularized logistic; `lambda` is fixed in the config (default `2.85e-4`), and phase 1 searches only the threshold
- `least_squares` — normal equations (labels `{+1,-1}`; threshold on score `xᵀw`, default `0`)
- `ridge` — ridge regression (`lambda` in config)
- `mse_gd` — linear regression by batch gradient descent
- `mse_sgd` — linear regression by SGD (one random row per step; slow for large `max_iters`)

Iterative methods show a progress bar on **stderr**. `least_squares` and `ridge` are closed-form (no bar).

Typical workflow:

1. Set `phase: 1` and pick a `method`.
2. Run `python run.py`, copy the printed block into `config.yaml`, set `phase: 2`.
3. Run again and upload the CSV to AIcrowd.

Preprocess: 20% validation split uses `seed` and `val_ratio` from config; medians/scales are fit on the training fold only in phase 1, on all train in phase 2.

First CSV load can take tens of seconds; logistic GD on ~260k rows takes on the order of minutes per fit.

Upload `outputs/submission_baseline.csv` to AIcrowd to check the format. Competition CSVs stay gitignored; that submission file is the exception.

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
