import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CONFIG_PATH = ROOT / "config.yaml"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from helpers import load_csv_data, create_csv_submission
from implementations import (
    least_squares,
    logistic_gradient,
    logistic_loss,
    mse,
    mse_gradient,
    ridge_regression,
    sigmoid,
)
from preprocess import add_bias, fit_preprocess, load_feature_names, train_val_split, transform_preprocess

METHODS = (
    "logistic",
    "reg_logistic",
    "least_squares",
    "ridge",
    "mse_gd",
    "mse_sgd",
)
LOGISTIC_METHODS = {"logistic", "reg_logistic"}
LAMBDA_GRID_METHODS = {"ridge"}
PROGRESS_WIDTH = 40


def load_yaml(path):
    """Minimal YAML subset: nested maps, scalars, no lists."""
    text = Path(path).read_text()
    lines = []
    for raw in text.splitlines():
        if "#" in raw:
            raw = raw[: raw.index("#")]
        if raw.strip():
            lines.append(raw.rstrip())

    def parse_block(start, base_indent):
        out = {}
        i = start
        while i < len(lines):
            line = lines[i]
            stripped = line.lstrip()
            if not stripped:
                i += 1
                continue
            indent = len(line) - len(stripped)
            if indent < base_indent:
                break
            if indent > base_indent:
                raise ValueError(f"unexpected indent at line {i + 1}: {line}")
            if ":" not in stripped:
                raise ValueError(f"expected key: value at line {i + 1}")
            key, rest = stripped.split(":", 1)
            key = key.strip()
            rest = rest.strip()
            if rest:
                out[key] = _scalar(rest)
                i += 1
            else:
                i += 1
                child, i = parse_block(i, base_indent + 2)
                out[key] = child
        return out, i

    def _scalar(s):
        s = s.strip().strip("'\"")
        if re.match(r"^-?\d+$", s):
            return int(s)
        try:
            return float(s)
        except ValueError:
            return s

    root, _ = parse_block(0, 0)
    return root


def progress_bar(step, total, prefix="", file=None):
    if total <= 0:
        return
    file = sys.stderr if file is None else file
    step = min(step, total)
    frac = step / total
    filled = int(PROGRESS_WIDTH * frac)
    bar = "#" * filled + "-" * (PROGRESS_WIDTH - filled)
    file.write(f"\r{prefix} [{bar}] {step}/{total} ({100 * frac:3.0f}%)")
    file.flush()
    if step >= total:
        file.write("\n")
        file.flush()


def logistic_train(y, tx, initial_w, max_iters, gamma, lambda_=None, label="GD"):
    y = np.asarray(y, dtype=float).reshape(-1)
    w = np.asarray(initial_w, dtype=float).reshape(-1)
    for i in range(1, max_iters + 1):
        grad = logistic_gradient(y, tx, w)
        if lambda_ is not None:
            grad = grad + 2.0 * lambda_ * w
        w = w - gamma * grad
        progress_bar(i, max_iters, prefix=label)
    return w, logistic_loss(y, tx, w)


def mse_gd_train(y, tx, initial_w, max_iters, gamma, label="mse_gd"):
    y = np.asarray(y, dtype=float).reshape(-1)
    w = np.asarray(initial_w, dtype=float).reshape(-1)
    for i in range(1, max_iters + 1):
        w = w - gamma * mse_gradient(y, tx, w)
        progress_bar(i, max_iters, prefix=label)
    return w, mse(y, tx, w)


def mse_sgd_train(y, tx, initial_w, max_iters, gamma, label="mse_sgd"):
    y = np.asarray(y, dtype=float).reshape(-1)
    w = np.asarray(initial_w, dtype=float).reshape(-1)
    n = y.shape[0]
    for i in range(1, max_iters + 1):
        j = np.random.randint(n)
        w = w - gamma * mse_gradient(y[j : j + 1], tx[j : j + 1], w)
        progress_bar(i, max_iters, prefix=label)
    return w, mse(y, tx, w)


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
    yhat = np.sign(np.asarray(yhat).reshape(-1))
    yhat[yhat == 0] = -1
    return yhat.astype(int)


def f1_positive(y_true, y_pred_pm1):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = to_pm1(y_pred_pm1)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == -1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == -1))
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    if prec + rec == 0:
        return 0.0, rec, prec
    return 2 * prec * rec / (prec + rec), rec, prec


def metrics(y_true, y_pred, title=""):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = to_pm1(y_pred)
    acc = np.mean(y_true == y_pred)
    f1, rec, prec = f1_positive(y_true, y_pred)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == -1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == -1))
    print(
        f"{title:28s}  acc={acc:.4f}  rec+={rec:.4f}  prec+={prec:.4f}  "
        f"F1+={f1:.4f}  (tp={tp} fp={fp} fn={fn})"
    )
    return f1


def threshold_grid(cfg):
    p1 = cfg["phase1"]
    return np.linspace(
        float(p1["threshold_min"]),
        float(p1["threshold_max"]),
        int(p1["threshold_n"]),
    )


def lambda_grid(cfg):
    p1 = cfg["phase1"]
    return np.logspace(
        np.log10(float(p1["lambda_min"])),
        np.log10(float(p1["lambda_max"])),
        int(p1["lambda_n"]),
    )


def best_threshold_scores(scores, y_true, grid, logistic=False):
    best = (0.0, float(grid[0]), 0.0, 0.0)
    for t in grid:
        if logistic:
            pred = np.where(scores >= t, 1, -1)
        else:
            pred = np.where(scores >= t, 1, -1)
        f1, rec, prec = f1_positive(y_true, pred)
        if f1 > best[0]:
            best = (f1, float(t), rec, prec)
    return best


def val_scores(method, w, tx_val, logistic=False):
    s = tx_val @ w
    if logistic:
        return sigmoid(s)
    return s


def train_method(method, y_train, tx_train, block, lambda_=None, label="train"):
    w0 = np.zeros(tx_train.shape[1])
    if method == "logistic":
        return logistic_train(
            y_train,
            tx_train,
            w0,
            int(block["max_iters"]),
            float(block["gamma"]),
            lambda_=None,
            label=label,
        )
    if method == "reg_logistic":
        lam = lambda_ if lambda_ is not None else float(block["lambda"])
        return logistic_train(
            y_train,
            tx_train,
            w0,
            int(block["max_iters"]),
            float(block["gamma"]),
            lambda_=lam,
            label=label,
        )
    if method == "least_squares":
        print("least_squares (closed form)", file=sys.stderr)
        return least_squares(y_train, tx_train)
    if method == "ridge":
        lam = lambda_ if lambda_ is not None else float(block["lambda"])
        print(f"ridge (closed form, lambda={lam:.2e})", file=sys.stderr)
        return ridge_regression(y_train, tx_train, lam)
    if method == "mse_gd":
        return mse_gd_train(
            y_train,
            tx_train,
            w0,
            int(block["max_iters"]),
            float(block["gamma"]),
            label=label,
        )
    if method == "mse_sgd":
        return mse_sgd_train(
            y_train,
            tx_train,
            w0,
            int(block["max_iters"]),
            float(block["gamma"]),
            label=label,
        )
    raise ValueError(f"unknown method: {method}")


def labels_for_method(method, y_pm1):
    if method in LOGISTIC_METHODS:
        return (y_pm1 + 1) / 2.0
    return y_pm1


def predict_pm1(method, w, tx, threshold, logistic=False):
    scores = val_scores(method, w, tx, logistic=logistic)
    if logistic:
        return to_pm1(np.where(scores >= threshold, 1, -1))
    return to_pm1(np.where(scores >= threshold, 1, -1))


def print_phase1_snippet(method, block, best_f1, t_star, lambda_star=None):
    print("\n=== Paste into config.yaml (then set phase: 2) ===")
    print(f"method: {method}")
    if method == "ridge" and lambda_star is not None:
        print("ridge:")
        print(f"  lambda: {lambda_star:.6g}")
        print(f"  threshold: {t_star:.4f}")
    elif method == "logistic":
        print("logistic:")
        print(f"  max_iters: {int(block['max_iters'])}")
        print(f"  gamma: {block['gamma']}")
        print(f"  threshold: {t_star:.4f}")
    elif method == "reg_logistic":
        print("reg_logistic:")
        print(f"  lambda: {float(block['lambda']):.6g}")
        print(f"  max_iters: {int(block['max_iters'])}")
        print(f"  gamma: {block['gamma']}")
        print(f"  threshold: {t_star:.4f}")
    elif method == "least_squares":
        print("least_squares:")
        print(f"  threshold: {t_star:.4f}")
    elif method in ("mse_gd", "mse_sgd"):
        print(f"{method}:")
        print(f"  max_iters: {int(block['max_iters'])}")
        print(f"  gamma: {block['gamma']}")
        print(f"  threshold: {t_star:.4f}")
    print(f"# val F1+={best_f1:.4f}")


def run_phase1(cfg):
    method = cfg["method"]
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    block = cfg[method]
    seed = int(cfg["seed"])
    np.random.seed(seed)
    val_ratio = float(cfg["val_ratio"])
    grid_t = threshold_grid(cfg)
    logistic = method in LOGISTIC_METHODS

    y, x, ids_tr, _, _, names = load_raw()
    x_tr, y_tr, _, x_val, y_val, _ = train_val_split(
        x, y, ids_tr, val_ratio=val_ratio, seed=seed
    )
    x_tr_z, stats = fit_preprocess(x_tr, names)
    x_val_z = transform_preprocess(x_val, names, stats)
    tx_tr, tx_val = add_bias(x_tr_z), add_bias(x_val_z)
    y_tr_fit = labels_for_method(method, y_tr)

    print(f"=== Phase 1: tune {method} (seed={seed}, val_ratio={val_ratio}) ===")

    if method in LAMBDA_GRID_METHODS:
        grid_l = lambda_grid(cfg)
        best = (0.0, float(grid_t[0]), float(grid_l[0]))
        for idx, lam in enumerate(grid_l, start=1):
            label = f"{method} λ={lam:.1e} ({idx}/{len(grid_l)}) "
            w, loss = train_method(
                method, y_tr_fit, tx_tr, block, lambda_=float(lam), label=label
            )
            print(f"train loss={float(loss):.6f}")
            scores = val_scores(method, w, tx_val, logistic=logistic)
            f1, t, _, _ = best_threshold_scores(scores, y_val, grid_t, logistic=logistic)
            print(f"  val F1+={f1:.4f}  t*={t:.3f}  lambda={lam:.2e}")
            if f1 > best[0]:
                best = (f1, t, float(lam))
        print_phase1_snippet(method, block, best[0], best[1], lambda_star=best[2])
        return

    w, loss = train_method(method, y_tr_fit, tx_tr, block, label=f"{method} val ")
    print(f"train loss={float(loss):.6f}")
    scores = val_scores(method, w, tx_val, logistic=logistic)
    f1, t_star, _, _ = best_threshold_scores(scores, y_val, grid_t, logistic=logistic)
    metrics(
        y_val,
        predict_pm1(method, w, tx_val, t_star, logistic=logistic),
        f"val @t*={t_star:.3f}",
    )
    print_phase1_snippet(method, block, f1, t_star)


def run_phase2(cfg):
    method = cfg["method"]
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    block = cfg[method]
    threshold = float(block["threshold"])
    logistic = method in LOGISTIC_METHODS
    out_csv = ROOT / cfg.get("out_csv", "outputs/submission_baseline.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    np.random.seed(int(cfg["seed"]))

    y, x, ids_tr, x_te, ids_te, names = load_raw()
    print(f"=== Phase 2: {method} on full train (threshold={threshold}) ===")

    x_full_z, stats = fit_preprocess(x, names)
    x_te_z = transform_preprocess(x_te, names, stats)
    tx_full, tx_te = add_bias(x_full_z), add_bias(x_te_z)
    y_full = labels_for_method(method, y)

    lam = float(block["lambda"]) if method in LAMBDA_GRID_METHODS or method == "reg_logistic" else None
    w, loss = train_method(
        method,
        y_full,
        tx_full,
        block,
        lambda_=lam,
        label=f"{method} full ",
    )
    print(f"full train loss={float(loss):.6f}")
    y_te_hat = predict_pm1(method, w, tx_te, threshold, logistic=logistic)
    print("test P(+1)", float(np.mean(y_te_hat == 1)))
    create_csv_submission(ids_te, y_te_hat, str(out_csv))
    print("wrote", out_csv)


def main():
    cfg_path = CONFIG_PATH
    if len(sys.argv) > 1:
        cfg_path = Path(sys.argv[1])
    cfg = load_yaml(cfg_path)
    phase = int(cfg.get("phase", 2))
    print("config:", cfg_path, "| phase:", phase, "| method:", cfg["method"])
    if phase == 1:
        run_phase1(cfg)
    elif phase == 2:
        run_phase2(cfg)
    else:
        raise ValueError("phase must be 1 or 2")


if __name__ == "__main__":
    main()
