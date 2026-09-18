# Project 1 — Plan

**Deadline:** Thursday 29 Oct 2026, 16:00  
**Submit at:** [http://mlcourse.epfl.ch](http://mlcourse.epfl.ch)  
**Competition:** [AIcrowd — EPFL Machine Learning Project 1](https://www.aicrowd.com/challenges/epfl-machine-learning-project-1)

Official spec: `project1_description.pdf` (also on branch `main`).

---

## Status (18 Sep 2026)

**Done**

- [x] Public GitHub repo + SSH push (`Yijun_Liu`)
- [x] CSVs kept local under `data/` and gitignored
- [x] Six methods in root `implementations.py` (docstrings + public-test numbers)
- [x] Tutorial: `doc/six_functions/six_functions_tutorial.ipynb`

**Not done yet (do these next, in order)**

1. **Run the official public tests** from `doc/grading_tests/` against a *local* path to this repo (not only the toy notebook checks).
2. **Add** `helpers.py` at the repo root: `load_csv_data` and `create_csv_submission` (from the course project helpers, NumPy only — no pandas).
3. **Load** `data/*.csv` **once** and print shapes, a few column names, and class counts of `y` (`+1` vs `-1`).
4. **AIcrowd:** `@epfl.ch` account, **one team** of 3, join the challenge (5 submissions/day shared).
5. **EDA notebook** (NumPy + matplotlib/seaborn only): missing codes (`77/88/99`/blank), leakage columns, 5–10 bullet findings.
6. **Baseline pipeline:** standardize features → train/val split → `least_squares` and `logistic_regression` → predictions in `{+1,-1}` → one AIcrowd CSV to check the format.
7. **Fill** `README.md` (still empty) and later `run.py` / `run.ipynb`.

Do **not** start fancy features or many AIcrowd submissions until 1–3 are done.

---



## 1. What this project is

Binary classification: given lifestyle / health survey features, predict whether a person has **MICHD** (myocardial infarct / coronary heart disease).


| File                            | Role                                    |
| ------------------------------- | --------------------------------------- |
| `dataset/x_train.csv`           | 328 135 people × 321 features           |
| `dataset/y_train.csv`           | labels in `{+1, -1}` (`_MICHD`)         |
| `dataset/x_test.csv`            | 109 379 people, **no labels**           |
| `dataset/sample_submission.csv` | format: `Id,Prediction` with `{+1, -1}` |


Data comes from CDC BRFSS (2015). Medical expertise is **not** required.

You must:

1. Implement the 6 lecture methods yourself (NumPy only).
2. Build a full pipeline on this dataset (clean → features → train → evaluate → predict).
3. Write a 2-page report.
4. Optionally submit predictions to AIcrowd for feedback (rank is **not** graded).

**Grading (this year’s PDF):** Project 1 is **not** part of the course grade. It is practice + feedback for Project 2 (30%). The AIcrowd page still says “10%”; treat the PDF as the source of truth.

---



## 2. Hard constraints (do not break these)

Allowed:

- Python standard library
- NumPy
- matplotlib / seaborn **only for plots**

Forbidden:

- Pandas, scikit-learn, PyTorch, TensorFlow, …
- Other people’s code
- Extra datasets

Weights are **1D arrays** shape `(D,)`, not `(D, 1)`.

Iterative methods return **only the last** `(w, loss)`.  
For ridge / regularized logistic, the returned **loss must not include** the penalty term.

MSE uses a factor **0.5** (as in lecture).  
SGD mini-batch size is **1**.  
`least_squares`: `numpy.linalg` is OK, `numpy.linalg.lstsq` **is not**.  
Logistic labels for those two functions must be `y ∈ {0, 1}` (convert from `{+1, -1}`).

---



## 3. Repo layout (required for grading tests)

Public GitHub repo of the team. Root must contain:

```text
README.md              # how to run, data path, what the pipeline does  (empty — fill this)
implementations.py     # the 6 required functions                       (done)
run.py  or  run.ipynb  # reproduces your best AIcrowd CSV              (missing)
helpers.py             # load_csv_data, create_csv_submission          (missing)
```

This repo currently:

```text
data/                  # local CSVs (gitignored); see data/README.md
doc/PROJECT_PLAN.md
doc/six_functions/     # tutorial (not graded)
doc/grading_tests/     # public tests (run from here; do not rely on them being in the submission root)
```

Staff tests (`doc/grading_tests/`):

```bash
pytest --github_link <GITHUB-REPO-URL> .
```

They check: files exist, function names/signatures, docstrings, a GitHub URL to a **specific commit** (`.../tree/<commit>`). Passing public tests ≠ full credit.

---



## 4. Six functions you must implement first

Put all of these in `implementations.py`. Each returns `(w, loss)`.


| Function                                                               | Method                           |
| ---------------------------------------------------------------------- | -------------------------------- |
| `mean_squared_error_gd(y, tx, initial_w, max_iters, gamma)`            | Linear regression, GD            |
| `mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma)`           | Linear regression, SGD (batch 1) |
| `least_squares(y, tx)`                                                 | Normal equations                 |
| `ridge_regression(y, tx, lambda_)`                                     | Ridge, normal equations          |
| `logistic_regression(y, tx, initial_w, max_iters, gamma)`              | Logistic, GD, `y ∈ {0,1}`        |
| `reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma)` | L2-regularized logistic, GD      |


**Status:** implemented in root `implementations.py`. Still run `pytest` from `doc/grading_tests/` with `--github_link` pointing at this folder.

---



## 5. Work plan (weeks)

Assume a team of 3. Deadline is **29 Oct**.

### Phase 0 — Setup

- [x] GitHub repo (public), branch `Yijun_Liu`, SSH
- [x] Implement the 6 functions + docstrings
- [x] Invite teammates as **collaborators** (write access)
- [x] AIcrowd account (`@epfl.ch`) + **one** team
- [x] Public `pytest` on `implementations.py`
- [x] `helpers.py` + confirm CSVs load without pandas



### Phase 1 — Understand the data (EDA)

Do this **before** fancy models.

- Class balance: how many `+1` vs `-1`? (CVD is usually rare → accuracy can look high while you always predict healthy.)
- Which columns are categorical vs continuous.
- Missing / “refused / don’t know” codes.
- Leakage risk: drop any feature that is the disease itself or a near-duplicate of the label (e.g. rehab after heart attack). If unsure, document the decision.
- Simple plots: histograms, correlation of a few obvious risk factors (age, BMI, smoking, blood pressure).

Write 5–10 bullets of findings. These become report Section 1.

### Phase 2 — Baseline pipeline (must exist early)

Goal: one **reproducible** submission that is better than “always −1”.

1. Load with `load_csv_data` (keep Ids).
2. Clean: impute or drop missing; encode categoricals; **standardize** continuous features (your Task A code).
3. Train/validation split or **k-fold CV** on `x_train` only. Never tune on AIcrowd test.
4. Fit **least squares** and **logistic regression** as baselines.
5. Threshold predictions to `{+1, -1}` (for logistic: `0.5` on probabilities; for least squares: `0`).
6. Write CSV with `create_csv_submission`.
7. Submit once to AIcrowd to verify the format.

Track a **local** metric that matches the competition (check the AIcrowd “Overview / Rules” page for the official score; do not assume accuracy).

### Phase 3 — Improve (the scientific part of the report)

Change **one thing at a time**. Log: method, λ / γ / features, CV score.

Prioritize in this order (highest typical impact first):

1. **Cleaning & missing values** (sentinels, rare categories).
2. **Class imbalance** (reweight loss, subsample majority, or tune threshold).
3. **Feature selection / grouping** (drop IDs, survey metadata, constant columns).
4. **Regularization**: ridge / `reg_logistic_regression` + λ via CV.
5. **Simple feature engineering**: logs, bins, a few interactions (age × smoking, BMI × activity).
6. **Hyperparameters**: step size, iterations, polynomial degree if you add it.

Required for the report: an **ablation table** — which change moved the metric the most?

Also diagnose over/underfitting (train vs val gap). Use Ng’s ML advice if stuck.

### Phase 4 — Report (start a draft by mid-October)

Max **2 pages PDF** + optional 3rd page of references. No appendix.  
Template: `projects/project1/latex-example-paper`.

Write for an ML beginner. Include:

- Problem and data (1 short paragraph).
- Preprocessing (exact rules, so someone can reproduce).
- Methods tried (the 6 implementations + what you actually used for the final model).
- How you evaluated (folds, metric, why not only AIcrowd).
- Baselines vs final model, **with numbers**.
- Ablation: top 2–3 changes.
- Honest failure cases if you have them.

Figures: labeled axes, units, readable in print.

### Phase 5 — Freeze and submit (last 3–4 days)

- `run.py` / `run.ipynb` produces **exactly** the CSV of your best (or documented) submission.
- If training is slow, `run.py` may only load saved weights; document where training lives.
- README: environment, commands, data paths, how to train vs infer.
- Push a commit; copy the GitHub URL **with** `/tree/<commit-hash>`.
- Upload PDF + GitHub link on [mlcourse.epfl.ch](http://mlcourse.epfl.ch).

---



## 6. Suggested team split


| Person | Owns                                                            |
| ------ | --------------------------------------------------------------- |
| A      | `implementations.py` (GD / SGD / least squares / ridge) + tests |
| B      | Logistic + regularized logistic, preprocessing, CV loop         |
| C      | EDA, features, `run.py`, AIcrowd submissions, LaTeX             |


Everyone reviews the report. Sync twice a week: what was the last CV number, what we try next.

---



## 7. Competition hygiene

- 5 AIcrowd submissions **per team per day**. Failed format does not count (per AIcrowd).
- Rank is **for fun** this project. Do not overfit the public leaderboard.
- Always decide with **local CV**, then submit.

Submission CSV:

```text
Id,Prediction
328135,-1
328136,1
...
```

Ids must match `x_test.csv`.

---



## 8. Done when…

- [x] Six functions in `implementations.py`, signatures + docstrings (pytest still to run)
- [ ] Public tests pass from `doc/grading_tests/`
- [ ] Local CV for the chosen model, plus at least two baselines
- [ ] Ablation of the main improvements
- [ ] `run.py` / `run.ipynb` reproduces the submitted CSV
- [ ] README is enough for a TA to run it
- [ ] 2-page PDF from the LaTeX template
- [ ] GitHub commit URL + PDF uploaded by 29 Oct 16:00

---



## 9. Sources

- Course PDF: `ML_course/projects/project1/project1_description.pdf`
- [AIcrowd challenge page](https://www.aicrowd.com/challenges/epfl-machine-learning-project-1)
- Dataset background: [CDC BRFSS 2015](https://www.cdc.gov/brfss/annual_data/annual_2015.html)
- Public grading tests: `ML_course/projects/project1/grading_tests/`

