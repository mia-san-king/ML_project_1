#!/usr/bin/env python3
"""Fig. 1: spatial split map (non-nadir labels)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

from lib.geo import load_boundaries, to_lv95
from lib.paths import META

COLORS = {
    "train": "#4C78A8",
    "val": "#F58518",
    "test_in": "#54A24B",
    "test_city": "#E45756",
    "test_region": "#B279A2",
    "buffer": "#BAB0AC",
}
ORDER = ["train", "val", "test_in", "test_city", "test_region", "buffer"]
LABELS = {
    "train": "train",
    "val": "val",
    "test_in": "test_in",
    "test_city": "Bern (test_city)",
    "test_region": "Ticino (test_region)",
    "buffer": "buffer",
}


def main():
    meta = pd.read_parquet(META / "smapshot.parquet", columns=["id", "latitude", "longitude"])
    splits = pd.read_csv("data/splits/main.csv.gz")
    frame = splits.merge(meta, on="id")
    frame = frame[frame.split.isin(ORDER)].copy()
    easting, northing = to_lv95(frame.latitude.values, frame.longitude.values)
    frame["e"], frame["n"] = easting, northing

    switzerland, cantons = load_boundaries()
    out = Path("report/figures/fig1_split_map.png")
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.2, 7.2))
    cantons.boundary.plot(ax=ax, linewidth=0.35, color="#444444")
    gpd.GeoSeries([switzerland], crs=2056).boundary.plot(ax=ax, linewidth=0.7, color="black")
    for name in ORDER:
        part = frame[frame.split == name]
        ax.scatter(
            part.e,
            part.n,
            s=0.2,
            c=COLORS[name],
            label=f"{LABELS[name]} ({len(part):,})",
            rasterized=True,
            linewidths=0,
        )
    ax.set_aspect("equal")
    ax.set_xlabel("LV95 easting (m)")
    ax.set_ylabel("LV95 northing (m)")
    ax.set_title("Fig. 1  SMAPshot split (Ticino + Bern, nadir excluded)")
    ax.legend(markerscale=12, loc="upper left", framealpha=0.92, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
