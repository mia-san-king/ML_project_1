#!/usr/bin/env python3
"""Build 01_eda.ipynb source cells (executed separately)."""
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
NB = PROJECT / "notebooks" / "01_eda.ipynb"

CELLS = [
    (
        "markdown",
        """# SMAPshot 探索性分析（EDA）

**目的**：在固定 train/val/test 划分之前，用元数据决定留出州和留出城市。

**选择标准**
- 每个留出集 ≥ 2,000 张，且同时有 terrestrial 与航拍类（非 nadir）。
- 留出州尽量与训练区在语言、地形上不同（建议 **Ticino**）。
- 留出城市不能是数据占比过大的枢纽（排除 Zürich）。""",
    ),
    (
        "code",
        """import json
import os
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from matplotlib import colormaps
from pyproj import Transformer

# 项目根目录
PROJECT = Path.cwd()
if not (PROJECT / "lib").is_dir():
    PROJECT = PROJECT.parent
import sys
sys.path.insert(0, str(PROJECT))

from lib.geo import haversine_km, load_boundaries, to_lv95
from lib.paths import META, GEO

DATA = Path(os.environ.get("P2_DATA", "/rcp-scratch/students/yijliu/ml-project2/data"))
META = DATA / "meta"
GEO = DATA / "geo"

plt.rcParams.update({
    "figure.figsize": (10, 6),
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "font.size": 10,
})
UA = {"User-Agent": "EPFL CS-433 student project (contact: yijun.liu@epfl.ch)"}

df = pd.read_parquet(META / "smapshot.parquet")
print(f"validated 图像: {len(df):,} 行, id 唯一: {df.id.is_unique}")
print(f"缺坐标: {(df.latitude.isna() | df.longitude.isna()).sum():,}")
""",
    ),
    (
        "markdown",
        "## 1. `view_type` 与拍摄年代",
    ),
    (
        "code",
        """vt = df["view_type"].fillna("(empty)")
vt_counts = vt.value_counts()
display(vt_counts.to_frame("count"))

fig, ax = plt.subplots()
vt_counts.plot(kind="barh", ax=ax, color="#4C72B0")
ax.set_xlabel("图像数量")
ax.set_title("view_type 分布")
ax.invert_yaxis()
plt.tight_layout()
plt.show()

# 年代
raw_dates = pd.to_datetime(df["date_shot_min"], errors="coerce", utc=True)
years = raw_dates.dt.year
valid_mask = years.between(1800, 2026)
n_bad = years.notna().sum() - valid_mask.sum()
n_missing = raw_dates.isna().sum()
print(f"date_shot_min 非空: {raw_dates.notna().sum():,}; 缺失: {n_missing:,}")
print(f"明显非法年份 (不在 1800–2026): {n_bad:,}")

fig, ax = plt.subplots()
years[valid_mask].hist(bins=60, ax=ax, color="#55A868", edgecolor="white")
ax.set_xlabel("拍摄年份")
ax.set_ylabel("图像数量")
ax.set_title("拍摄年代分布（1800–2026）")
plt.tight_layout()
plt.show()
""",
    ),
    (
        "markdown",
        "## 2. 收藏（collection）",
    ),
    (
        "code",
        """coll_path = META / "collections.json"
if not coll_path.exists():
    r = requests.get("https://smapshot.heig-vd.ch/api/v1/collections", headers=UA, timeout=120)
    r.raise_for_status()
    coll_path.write_text(json.dumps(r.json(), ensure_ascii=False, indent=2))
    print("已缓存 collections.json")

collections = pd.DataFrame(json.loads(coll_path.read_text()))
collections = collections.rename(columns={"id": "collection_id", "name": "collection_name"})
merged = df.merge(collections[["collection_id", "collection_name"]], on="collection_id", how="left")

top = (
    merged.groupby(["collection_id", "collection_name"], dropna=False)
    .size()
    .sort_values(ascending=False)
    .head(15)
    .reset_index(name="n_images")
)
display(top)

tessin = merged[merged.collection_name.str.contains("Perrochet|Tessin", case=False, na=False)]
zurich = merged[merged.collection_name.str.contains("Zürich|Zurich", case=False, na=False)]
print(f"Fonds Perrochet / Tessin 相关: {len(tessin):,} 张")
print(f"Views of Zürich 等相关: {len(zurich):,} 张")
""",
    ),
    (
        "markdown",
        "## 3. 瑞士空间分布（LV95）",
    ),
    (
        "code",
        """ch, cantons = load_boundaries()
e, n = to_lv95(df.latitude.values, df.longitude.values)
pts = gpd.GeoSeries(gpd.points_from_xy(e, n), crs=2056)
inside_ch = pts.within(ch.buffer(5_000))
print(f"瑞士国界 +5 km 内: {inside_ch.sum():,} 张")
print(f"境外 (含 Rio / Vorarlberg 等): {(~inside_ch).sum():,} 张")

plot_df = df.loc[inside_ch].copy()
plot_df["e"], plot_df["n"] = e[inside_ch], n[inside_ch]
plot_df["view_type_plot"] = plot_df["view_type"].fillna("empty")

palette = {
    "terrestrial": "#2ca02c",
    "lowOblique": "#1f77b4",
    "highOblique": "#ff7f0e",
    "nadir": "#9467bd",
    "empty": "#7f7f7f",
}

fig, ax = plt.subplots(figsize=(11, 9))
cantons.boundary.plot(ax=ax, linewidth=0.4, color="#333333", alpha=0.6)
gpd.GeoSeries(ch.buffer(5_000), crs=2056).boundary.plot(ax=ax, linewidth=0.8, color="black")

for vt, sub in plot_df.groupby("view_type_plot"):
    ax.scatter(
        sub["e"],
        sub["n"],
        s=0.15,
        alpha=0.35,
        c=palette.get(vt, "#aaaaaa"),
        label=vt,
        rasterized=True,
    )
ax.set_aspect("equal")
ax.set_xlabel("LV95 Easting (m)")
ax.set_ylabel("LV95 Northing (m)")
ax.set_title("瑞士范围内图像位置（按 view_type）")
ax.legend(markerscale=8, loc="upper left", framealpha=0.9)
plt.tight_layout()
plt.show()
""",
    ),
    (
        "markdown",
        "## 4. 候选留出州与城市",
    ),
    (
        "code",
        """located = df.dropna(subset=["latitude", "longitude"]).copy()
e_all, n_all = to_lv95(located.latitude.values, located.longitude.values)
pts_df = gpd.GeoDataFrame(
    located[["id", "view_type"]],
    geometry=gpd.points_from_xy(e_all, n_all),
    crs=2056,
)
joined = gpd.sjoin(pts_df, cantons[["name", "geometry"]], predicate="within", how="inner")
canton_counts = joined.groupby("name").size().sort_values(ascending=False)
canton_tbl = canton_counts.reset_index(name="n_images")
display(canton_tbl)

CITIES = {
    "Basel": (47.5596, 7.5886),
    "Bern": (46.9480, 7.4474),
    "Lausanne": (46.5197, 6.6323),
    "Luzern": (47.0502, 8.3093),
    "Zürich": (47.3769, 8.5417),
    "Genève": (46.2044, 6.1432),
    "St. Gallen": (47.4245, 9.3767),
}

rows = []
for city, (la, lo) in CITIES.items():
    d = haversine_km(located.latitude.values, located.longitude.values, la, lo)
    mask = d <= 10
    sub = located.loc[mask]
    rows.append(
        {
            "city": city,
            "within_10km": int(mask.sum()),
            "terrestrial": int((sub.view_type == "terrestrial").sum()),
            "nadir": int((sub.view_type == "nadir").sum()),
            "aerial_other": int(mask.sum() - (sub.view_type == "terrestrial").sum() - (sub.view_type == "nadir").sum()),
        }
    )
city_tbl = pd.DataFrame(rows).sort_values("within_10km", ascending=False)
display(city_tbl)
""",
    ),
    (
        "markdown",
        """## 5. 建议（供 S4 划分）

| 角色 | 建议 | 理由 |
| --- | --- | --- |
| **留出州** | **Ticino** | 约 1.4 万张，terrestrial 约 9k；阿尔卑斯以南、意大利语区，且有 Fonds Perrochet (Tessin) 收藏，适合 RQ3「没见过地区」。 |
| **留出城市** | **Bern** | 10 km 内约 4k 张，terrestrial 约 1,455；体量适中，不像 Zürich（约 3.2 万）会掏空训练集。 |
| **排除** | Zürich | 专用收藏体量大，作为留出城市损失过多训练数据。 |
| **不优先** | Basel | 10 km 内总数够，但 terrestrial 仅约 176，地面视角泛化样本偏弱。 |
| **不用作瑞士留出城市** | 4Dbrowser | 覆盖点几乎在欧洲中北部，瑞士框内仅约 685 点；仅可作可选跨国压力测试（见教程 S12d）。 |

> **说明**：`data/splits/main.csv.gz` 若已按 Basel 生成，尚未按本 notebook 建议改为 Bern；提交划分前应重跑 `scripts/make_splits.py`。""",
    ),
]


def main():
    import nbformat
    from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

    nb = new_notebook(cells=[])
    for kind, source in CELLS:
        if kind == "markdown":
            nb.cells.append(new_markdown_cell(source))
        else:
            nb.cells.append(new_code_cell(source))
    NB.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, NB)
    print(f"wrote {NB} ({len(nb.cells)} cells)")


if __name__ == "__main__":
    main()