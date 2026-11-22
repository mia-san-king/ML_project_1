"""Recompute paired T4 geographic-error distributions and plot empirical CDFs."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import PercentFormatter
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from models.t5_reranker.evaluate import original_rows, read_predictions


def cdf(sorted_error, thresholds):
    return np.searchsorted(sorted_error, thresholds, side="right") * (100.0 / len(sorted_error))


def summarize(values):
    quantiles = np.quantile(values, [.1, .25, .5, .75, .9, .95, .99])
    return {"mean": float(values.mean()), **dict(zip(("p10", "p25", "p50", "p75", "p90", "p95", "p99"), quantiles))}


def main():
    truth, train, _ = original_rows(ROOT / "data/t1_original", ROOT / "data/splits/main.csv.gz")
    errors = [read_predictions(OUT / "models" / arm / "predictions.parquet", "T4", truth, train)[1]
              for arm in ("random", "sparse")]
    masks = {"all_tests": truth.split.ne("val"), **{name: truth.split.eq(name)
             for name in ("val", "test_in", "test_city", "test_region")}}
    pairs, results = {}, {}
    for key, mask in masks.items():
        a, b = [e[mask.to_numpy()] for e in errors]
        difference = b - a
        old, new = np.sort(a), np.sort(b)
        pairs[key] = old, new
        thresholds = np.union1d(old, new)
        gap = cdf(new, thresholds) - cdf(old, thresholds)
        results[key] = {"n":len(a), "random_km":summarize(a), "sparse_km":summarize(b),
                        "fraction_closer":float(np.mean(difference < 0)), "fraction_farther":float(np.mean(difference > 0)),
                        "fraction_same":float(np.mean(difference == 0)), "median_absolute_change_km":float(np.median(abs(difference))),
                        "fraction_absolute_change_at_most_5km":float(np.mean(abs(difference)<=5)),
                        "max_absolute_hit_rate_gap_pp":float(np.max(np.abs(gap)))}
    (OUT / "error_distribution.json").write_text(json.dumps(results, indent=2, allow_nan=False)+"\n",encoding="utf-8")

    for name in ("Microsoft YaHei", "Noto Sans CJK SC", "PingFang SC", "SimHei"):
        try:
            font_manager.findfont(name, fallback_to_default=False)
        except ValueError:
            continue
        plt.rcParams["font.family"] = name
        break
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 2, figsize=(12.3, 4.75))
    fig.subplots_adjust(left=.071, right=.98, bottom=.155, top=.72, wspace=.24)
    ink, muted = "#243747", "#536777"
    fig.text(.05, .97, "随机增强与稀疏地区优先增强的 T4 定位误差分布", fontsize=18, color=ink, weight="bold", va="top")
    fig.text(.05, .875, "每组 4,096 张扩散图  ·  相同的 64,009 张测试照片  ·  各距离阈值均按照片计数", color=muted, fontsize=10)
    left, right = axes
    left.set_title("三个测试集合并的累计命中率", loc="left", fontsize=12, color=ink, pad=10)
    right.set_title("各测试集的命中率差（稀疏优先 − 随机）", loc="left", fontsize=12, color=ink, pad=10)
    old, new = pairs["all_tests"]
    for values, label, color, style, width in ((old, "随机", "#2D6987", "-", 2.9), (new, "稀疏优先", "#CF6D28", "--", 1.55)):
        left.step(np.r_[0, values], 100*np.arange(len(values)+1)/len(values), where="post",
                  label=label, color=color, linestyle=style, linewidth=width)
    left.set_ylim(-1, 102)
    left.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    left.legend(frameon=False, loc="lower right", fontsize=9)
    right.axhline(0, color="#758695", linewidth=1)
    right.set_ylim(-.55, 1.55)
    right.set_yticks([-.5, 0, .5, 1., 1.5], ["−0.5", "0", "+0.5", "+1.0", "+1.5"])
    right.set_ylabel("百分点", color=muted, fontsize=9)
    grid = np.arange(0, 241, 1)
    for key, label, color in (("test_city", "伯尔尼市", "#B66529"), ("test_region", "提契诺州", "#277F77"), ("test_in", "常规测试", "#526BA0")):
        old, new = pairs[key]
        right.plot(grid, cdf(new, grid)-cdf(old, grid), label=label, color=color, lw=1.45)
    right.legend(frameon=False, loc="upper right", fontsize=9)
    for ax in axes:
        ax.set_xlim(0, 240)
        ax.set_xticks([0, 25, 50, 100, 150, 200, 240])
        ax.set_xlabel("定位误差阈值（km）", fontsize=10, color=muted)
        ax.tick_params(labelsize=9, colors=muted, length=2)
        ax.spines[["right", "top"]].set_visible(False)
        for side in ("bottom", "left"):
            ax.spines[side].set_color("#B8C4CB")
        ax.grid(axis="y", color="#DCE3E7", linewidth=.7)
        ax.set_axisbelow(True)
    fig.savefig(OUT / "error_distribution.png", dpi=175, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
