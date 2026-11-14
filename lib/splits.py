import numpy as np
import pandas as pd
import geopandas as gpd

from lib.geo import load_boundaries, to_lv95


def make_splits(
    df,
    region="Ticino",
    city_center=(46.9480, 7.4474),
    city_r_km=10,
    buffer_km=10,
    block_km=10,
    frac=(0.7, 0.1, 0.2),
    seed=0,
):
    """Assign every row a split. Missing coordinates stay labeled ``missing``."""
    missing = df["latitude"].isna() | df["longitude"].isna()
    located = df.loc[~missing].copy()

    switzerland, cantons = load_boundaries()
    easting, northing = to_lv95(located.latitude.values, located.longitude.values)
    points = gpd.GeoSeries(gpd.points_from_xy(easting, northing), crs=2056)

    inside = points.within(switzerland.buffer(5_000)).values
    canton = cantons.loc[cantons.name == region]
    if canton.empty:
        names = ", ".join(sorted(cantons.name.astype(str)))
        raise ValueError(f"canton {region!r} not found. Natural Earth names: {names}")
    region_geom = canton.geometry.union_all()
    in_region = points.within(region_geom).values
    near_region = points.within(region_geom.buffer(buffer_km * 1000)).values & ~in_region

    city_e, city_n = to_lv95(np.array([city_center[0]]), np.array([city_center[1]]))
    distance_km = np.hypot(easting - city_e[0], northing - city_n[0]) / 1000
    in_city = distance_km <= city_r_km
    near_city = (distance_km > city_r_km) & (distance_km <= city_r_km + buffer_km)

    split = np.full(len(located), "", dtype=object)
    split[~inside] = "outside"
    split[inside & (near_region | near_city)] = "buffer"
    split[inside & in_city] = "test_city"
    split[inside & in_region] = "test_region"

    rest = inside & (split == "")
    block_x = np.floor(easting / (block_km * 1000)).astype(np.int64)
    block_y = np.floor(northing / (block_km * 1000)).astype(np.int64)
    block = block_x * 100_000 + block_y
    blocks = np.unique(block[rest])
    rng = np.random.default_rng(seed)
    rng.shuffle(blocks)
    n_train, n_val = int(frac[0] * len(blocks)), int(frac[1] * len(blocks))
    lookup = {
        block_id: name
        for name, chosen in (
            ("train", blocks[:n_train]),
            ("val", blocks[n_train : n_train + n_val]),
            ("test_in", blocks[n_train + n_val :]),
        )
        for block_id in chosen
    }
    split[rest] = [lookup[block_id] for block_id in block[rest]]

    random_split = np.full(len(located), "", dtype=object)
    draw = rng.random(int(rest.sum()))
    random_split[rest] = np.where(
        draw < frac[0],
        "train",
        np.where(draw < frac[0] + frac[1], "val", "test_in"),
    )

    out = pd.DataFrame(
        {
            "id": located.id.values,
            "split": split,
            "random_split": random_split,
            "block": block,
        }
    )
    missing_rows = pd.DataFrame(
        {
            "id": df.loc[missing, "id"].values,
            "split": "missing",
            "random_split": "",
            "block": -1,
        }
    )
    out = pd.concat([out, missing_rows], ignore_index=True)
    view_type = df.set_index("id").loc[out.id.values, "view_type"]
    nadir = view_type.astype(str).values == "nadir"
    out.loc[nadir, "split"] = "nadir_" + out.loc[nadir, "split"].astype(str)
    return out
