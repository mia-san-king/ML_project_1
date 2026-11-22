"""Compare the frozen T1, T3 and T6 v2 predictions on identical held-out photos."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


MODELS = ("T1", "T3", "T6")
UPPER_KM = tuple(range(25, 251, 25))


def summarize(frame):
    result = {"n": len(frame), "models": {}, "T6_vs": {}}
    for model in MODELS:
        err = frame[f"{model}_error_km"].to_numpy(dtype=float)
        if not len(err) or not np.isfinite(err).all() or (err < 0).any():
            raise ValueError(f"Invalid distances for {model}")
        counts = np.bincount(np.searchsorted(UPPER_KM, err, side="left"), minlength=len(UPPER_KM) + 1)
        result["models"][model] = {
            "counts": counts.tolist(), "frequency_pct": (100 * counts / len(err)).tolist(),
            "median_km": float(np.median(err)), "mean_km": float(np.mean(err)),
            "p90_km": float(np.quantile(err, 0.9)), "hits_25km": int(counts[0]),
            "within_25km_pct": float(100 * counts[0] / len(err)),
        }
    hits = frame.T6_error_km <= 25
    for model in ("T1", "T3"):
        other = frame[f"{model}_error_km"] <= 25
        saved, lost = int((hits & ~other).sum()), int((~hits & other).sum())
        result["T6_vs"][model] = {"rescued_25km": saved, "lost_25km": lost, "net_25km": saved - lost}
    return result


def evaluate(workspace, model_path, output, per_photo=None):
    from models.t6_router.bundle import load_bundle
    from models.t6_router.error_distribution_data import load_test_comparison
    from models.t6_router.router import select_expert
    from scripts.confidence.evaluate import sha256

    assets = load_bundle(model_path)
    pred = workspace / "data/local_runs/confidence_models/predictions.parquet"
    frame = load_test_comparison(pred, workspace=workspace)
    use4, _ = select_expert(frame, assets["gate"])
    frame["T6_error_km"] = np.where(use4, frame.T4_error_km, frame.T5_error_km)
    frame["T6_expert"] = np.where(use4, "T4", "T5")
    report = {
        "test_population": "Real held-out photos only. All three methods use the +8192 training condition.",
        "distance": "Great-circle error in km. Bins include their upper edge; last bin exceeds 250 km.",
        "upper_edges_km": UPPER_KM, "pooled_weight": "One vote per photo.",
        "predictions_sha256": sha256(pred), "gate_sha256": sha256(assets["gate"]),
        "bundle_manifest_sha256": sha256(model_path),
        "pooled": summarize(frame),
        "splits": {split: summarize(part) for split, part in frame.groupby("split")},
    }
    if per_photo is not None:
        per_photo.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(per_photo, index=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def plot(report, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator, PercentFormatter

    plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti TC", "Microsoft YaHei", "Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    group = report["pooled"]
    upper = report["upper_edges_km"]
    labels = [f"{lo}–{hi}" for lo, hi in zip([0, *upper[:-1]], upper)] + [f">{upper[-1]}"]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(11.6, 5.5), layout="constrained")
    colors = {"T1": "#4877a8", "T3": "#c47b24", "T6": "#167767"}
    fig.suptitle(f"瑞士历史照片定位误差分布 · {group['n']:,} 张测试照片",
                 x=0.065, ha="left", fontsize=19)
    ax.set_title("合并常规测试、伯尔尼市和提契诺州，每张照片权重相同", loc="left", fontsize=11, pad=14, color="#455460")
    for offset, model in enumerate(MODELS):
        values = group["models"][model]["frequency_pct"]
        ax.bar(x + (offset - 1) * .25, values, .23, color=colors[model], label=model)
    ax.set(xlabel="预测位置与实际拍摄位置的距离（公里）", ylabel="测试照片占比",
           xticks=x, xticklabels=labels, xlim=(-.6, len(labels) - .4))
    ax.xaxis.labelpad = 12
    ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.grid(axis="y", color="#d8dfe4", linewidth=.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#a3afb6")
    ax.tick_params(axis="both", length=0, pad=5)
    ax.legend(ncols=3, frameon=False, loc="upper right", fontsize=11)
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="action", required=True)
    ev = commands.add_parser("evaluate")
    for name in ("workspace", "model", "output"):
        ev.add_argument("--" + name, type=Path, required=True)
    ev.add_argument("--per-photo", type=Path)
    figure = commands.add_parser("plot")
    figure.add_argument("--summary", type=Path, required=True)
    figure.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    if args.action == "evaluate":
        outcome = evaluate(args.workspace, args.model, args.output, args.per_photo)
        print("Verified test photos:", outcome["pooled"]["n"])
    else:
        plot(json.loads(args.summary.read_text(encoding="utf-8")), args.output)


if __name__ == "__main__":
    main()
