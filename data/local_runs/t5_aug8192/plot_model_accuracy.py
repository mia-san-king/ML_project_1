"""Plot the augmented models on matching evaluation IDs from the prediction files."""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data/local_runs/tools"))
from experiment_paths import T1_AUGMENTED
from models.t5_reranker.evaluate import original_rows, read_predictions


OUT = Path(__file__).resolve().parent
RUNS = OUT.parent
COLORS = {"T1": "#4978A2", "T3": "#B66F24", "T4": "#278678", "T5": "#865B99"}
SPLITS = {"test_in": "常规测试集", "test_city": "伯尔尼市", "test_region": "提契诺州", "val": "验证集"}
THRESHOLDS = np.array([1, 5, 25, 50])


def collect(t1_predictions):
    truth, train, _ = original_rows(ROOT / "data/t1_original", ROOT / "data/splits/main.csv.gz")
    sources = {
        "T1": t1_predictions,
        "T3": RUNS / "t3_t4_8192/augmented/T3/predictions.parquet",
        "T4": RUNS / "t3_t4_8192/augmented/T4/predictions.parquet",
        "T5": OUT / "results/predictions.parquet",
    }
    stats = {split: {} for split in SPLITS}
    counts = truth.split.value_counts().to_dict()
    for name, path in sources.items():
        _, errors = read_predictions(path, name, truth, train)
        for split in SPLITS:
            subset = errors[truth.split.eq(split).to_numpy()]
            hits = (subset[:, None] <= THRESHOLDS).mean(axis=0) * 100
            stats[split][name] = (hits, np.quantile(subset, [0.5, 0.9]))
    return counts, stats


def use_chinese_font():
    families = ["Microsoft YaHei", "Noto Sans CJK SC", "Source Han Sans SC", "PingFang SC", "SimHei"]
    for family in families:
        try:
            font_manager.findfont(family, fallback_to_default=False)
        except ValueError:
            continue
        plt.rcParams.update({"font.family": family, "axes.unicode_minus": False})
        return
    raise RuntimeError("A Chinese sans-serif font is required for this figure")


def plot(counts, stats, output):
    use_chinese_font()
    ink, muted, line = "#213544", "#536774", "#DAE1E5"
    fig = plt.figure(figsize=(13.8, 10.4), facecolor="white")
    fig.text(.045, .961, "瑞士历史照片定位：四个模型的误差分布", fontsize=22,
             color=ink, weight="bold", va="top")
    fig.text(.045, .912, f"本地 8,192 张扩散图增强版  ·  {sum(counts[s] for s in SPLITS):,} 张真实评估照片",
             fontsize=11, color=muted)
    fig.legend(handles=[Patch(color=color, label=name) for name, color in COLORS.items()],
               ncol=4, frameon=False, loc="upper right", bbox_to_anchor=(.97, .961),
               handlelength=1, handletextpad=.45, columnspacing=1.3, fontsize=11)
    layout = fig.add_gridspec(4, 3, left=.045, right=.965, top=.819, bottom=.074,
                              width_ratios=(.93, 4.4, 3.1), hspace=.58, wspace=.18)
    for row, (split, title) in enumerate(SPLITS.items()):
        label = fig.add_subplot(layout[row, 0])
        label.set_axis_off()
        label.text(0, .66, title, fontsize=13, weight="bold", color=ink, transform=label.transAxes)
        label.text(0, .42, f"{counts[split]:,} 张", fontsize=10, color=muted, transform=label.transAxes)
        hits_ax = fig.add_subplot(layout[row, 1])
        km_ax = fig.add_subplot(layout[row, 2])
        if row == 0:
            hits_ax.set_title("累计命中率（越高越好）", loc="left", fontsize=14,
                              weight="bold", color=ink, pad=15)
            km_ax.set_title("定位误差（越低越好）", loc="left", fontsize=14,
                            weight="bold", color=ink, pad=15)
        for ax, groups, limit, ticks in (
            (hits_ax, ["≤1 km", "≤5 km", "≤25 km", "≤50 km"], 80, [0, 20, 40, 60, 80]),
            (km_ax, ["中位误差", "90% 分位误差"], 180, [0, 50, 100, 150]),
        ):
            ax.set_ylim(0, limit)
            ax.set_xlim(-.53, len(groups)-.47)
            ax.set_yticks(ticks)
            ax.set_xticks(np.arange(len(groups)), groups)
            ax.tick_params(axis="both", colors=muted, labelsize=9.5, length=0, pad=4)
            ax.set_axisbelow(True)
            ax.grid(axis="y", color=line, linewidth=.75)
            ax.spines[["left", "right", "top"]].set_visible(False)
            ax.spines["bottom"].set_color(line)
        hits_ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
        km_ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}"))
        hits_ax.get_xticklabels()[2].set_weight("bold")
        hits_ax.get_xticklabels()[2].set_color(ink)
        for col, (name, color) in enumerate(COLORS.items()):
            hits, distance = stats[split][name]
            offset = (col - 1.5) * .195
            hits_ax.bar(np.arange(4) + offset, hits, color=color, width=.17)
            km_ax.bar(np.arange(2) + offset, distance, color=color, width=.17)
            # Label the project's primary hit threshold, plus both distance summaries.
            hits_ax.text(2 + offset, hits[2] + 1.2, f"{hits[2]:.1f}",
                         ha="center", color=ink, fontsize=8.3)
            for j, d in enumerate(distance):
                km_ax.text(j + offset, d + 2, f"{d:.1f}", ha="center", color=ink, fontsize=8.3)
    fig.text(.045, .019, "命中率按距离阈值累计，柱上标注 25 km 命中率。右图单位为 km，90% 分位误差表示 90% 的预测误差不超过该距离。",
             color=muted, fontsize=9.2)
    fig.savefig(output, dpi=165, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--t1-predictions", type=Path,
                     default=T1_AUGMENTED / "results/preds/T1_aug8192_s0.parquet")
    cli.add_argument("--output", type=Path, default=OUT / "model_accuracy.png")
    args = cli.parse_args()
    plot(*collect(args.t1_predictions), args.output)
