import numpy as np

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from helpers import load_csv_data, create_csv_submission
from implementations import least_squares, logistic_regression
from preprocess import load_feature_names, fit_preprocess, transform_preprocess, add_bias, train_val_split

VAL_RATIO = 0.2
SEED = 0
LOGISTIC_ITERS = 50
LOGISTIC_GAMMA = 0.1
OUT_CSV = ROOT / "submission_baseline.csv"

def load_raw():
    cache = DATA / "cache_eda.npz"
    names = load_feature_names(DATA)
    if cache.exists():
        z = np.load(cache)
        print("loaded", cache)
        return z["y"], z["x"], z["ids_tr"], z["x_te"], z["ids_te"], names
    y, x, ids_tr, x_te, ids_te = load_csv_data(str(DATA))
    print("loaded CSVs")
    return y, x, ids_tr, x_te, ids_te, names

def to_pm1(yhat):
    """Map scores or {0,1} to {+1,-1}. Zeros (rare) → -1."""
    yhat = np.sign(np.asarray(yhat).reshape(-1))
    yhat[yhat == 0] = -1
    return yhat.astype(int)


def metrics(y_true, y_pred, title=""):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = to_pm1(y_pred)
    acc = np.mean(y_true == y_pred)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == -1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == -1))
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    print(
        f"{title:22s}  acc={acc:.4f}  rec+={rec:.4f}  prec+={prec:.4f}  "
        f"F1+={f1:.4f}  (tp={tp} fp={fp} fn={fn})"
    )
    return f1

def main():
    y, x, ids_tr, x_te, ids_te, names = load_raw()

    # 1) split before any median / std
    x_tr, y_tr, _, x_val, y_val, _ = train_val_split(
        x, y, ids_tr, val_ratio=VAL_RATIO, seed=SEED
    )

    # 2) fit on train only, then val + test
    x_tr_z, stats = fit_preprocess(x_tr, names)
    x_val_z = transform_preprocess(x_val, names, stats)
    x_te_z = transform_preprocess(x_te, names, stats)
    tx_tr, tx_val, tx_te = add_bias(x_tr_z), add_bias(x_val_z), add_bias(x_te_z)

    metrics(y_val, np.full_like(y_val, -1), "always -1")

    # 3) least squares, labels stay {+1,-1}
    w_ls, loss_ls = least_squares(y_tr, tx_tr)
    print("least_squares train MSE", float(loss_ls))
    f1_ls = metrics(y_val, tx_val @ w_ls, "least_squares val")
    metrics(y_tr, tx_tr @ w_ls, "least_squares train")

    # 4) logistic, labels {0,1} only here
    y_tr01 = (y_tr + 1) / 2.0
    w_log, loss_log = logistic_regression(
        y_tr01,
        tx_tr,
        np.zeros(tx_tr.shape[1]),
        LOGISTIC_ITERS,
        LOGISTIC_GAMMA,
    )
    print("logistic train NLL", float(loss_log))
    p_val = 1.0 / (1.0 + np.exp(-np.clip(tx_val @ w_log, -30, 30)))
    f1_log = metrics(y_val, np.where(p_val >= 0.5, 1, -1), "logistic val")

    
    # 5) pick higher val F1+ for the AIcrowd file
    if f1_log > f1_ls:
        print("submission model: logistic")
        y_te_hat = to_pm1(
            np.where(
                1.0 / (1.0 + np.exp(-np.clip(tx_te @ w_log, -30, 30))) >= 0.5, 1, -1
            )
        )
    else:
        print("submission model: least_squares")
        y_te_hat = to_pm1(tx_te @ w_ls)

    print("test P(+1)", float(np.mean(y_te_hat == 1)))
    create_csv_submission(ids_te, y_te_hat, str(OUT_CSV))
    print("wrote", OUT_CSV)
    
if __name__ == "__main__":
    main()