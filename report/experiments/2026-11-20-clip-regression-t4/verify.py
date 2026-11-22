"""Reload the T4 checkpoint, reproduce sampled predictions, and quantify test spread."""
import hashlib
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
import torch

run = Path(__file__).resolve().parent
sys.path.insert(0, str(run / "source"))
from models.clip_regression.lib.regression import LinearTimeRegressor, clip_to_switzerland
from models.geoclip_time.lib.temporal_data import encode_date_intervals

project = run.parent.parent
data = project / "data"
results = run / "results"
pred = pd.read_parquet(results / "predictions.parquet").set_index("id", verify_integrity=True)
reference = pd.read_parquet(data / "results/preds/zhang7_T1_dual_time_available_s0_20261118.parquet").set_index("id", verify_integrity=True).loc[pred.index]
splits = pd.read_csv(project / "ML_project_1/data/splits/main.csv.gz").set_index("id", verify_integrity=True)
cache_ids = pd.Index(np.load(data / "features/clip_vitl14_ids.npy"))
train_ids = np.sort(cache_ids.intersection(splits.index[splits.split == "train"]).to_numpy(dtype="<i8"))
summary = json.loads((results / "summary.json").read_text())
assert hashlib.sha256(train_ids.tobytes()).hexdigest() == summary["train_id_sha256"]
assert len(train_ids) == summary["train_n"] == 176308
assert len(pred) == 93538 and not pred.index.isin(train_ids).any()
assert (pred.split == reference.split).all() and (pred.split == splits.loc[pred.index, "split"]).all()

sample = pd.concat([pred.groupby("split").sample(n=128, random_state=4),
                    pred[pred.was_projected].head(64)])
sample = sample.loc[~sample.index.duplicated()]
meta = pd.read_parquet(data / "meta/smapshot.parquet", columns=["id", "date_shot_min", "date_shot_max"]).set_index("id").loc[sample.index]
checkpoint = torch.load(results / "T4.pt", map_location="cpu", weights_only=True)
features = np.load(data / "features/clip_vitl14.npy", mmap_mode="r")[cache_ids.get_indexer(sample.index)]
dates, _ = encode_date_intervals(meta, **checkpoint["time_config"])
model = LinearTimeRegressor(checkpoint["feature_dim"]).double().eval()
model.load_state_dict(checkpoint["model_state"])
with torch.inference_mode():
    offsets = model(torch.tensor(features).double(), torch.tensor(dates).double()).numpy()
raw = offsets * checkpoint["scale_m"] + np.array(checkpoint["origin_m"])
border = gpd.read_file(data / "geo/ne_admin0.gpkg").to_crs(2056).geometry.union_all()
bounded = clip_to_switzerland(raw, border)
lon, lat = Transformer.from_crs(2056, 4326, always_xy=True).transform(bounded[:, 0], bounded[:, 1])
np.testing.assert_allclose(np.column_stack((lon, lat)), sample[["lon_pred", "lat_pred"]], atol=1e-9, rtol=0)
assert np.array_equal(sample.was_projected, np.any(bounded != raw, axis=1))
assert np.array_equal(sample.has_date, dates[:, 2].astype(bool))

to_swiss = Transformer.from_crs(4326, 2056, always_xy=True)
def spread_km(latitudes, longitudes):
    easting, northing = to_swiss.transform(longitudes.to_numpy(), latitudes.to_numpy())
    return float(np.sqrt(np.var(easting) + np.var(northing)) / 1000)

test = pred[pred.split == "test_in"]
t1 = reference.loc[test.index]
year_change = model.linear.weight.detach().numpy()[:, -3] * 20 / checkpoint["time_config"]["year_scale"]
checks = {
    "reloaded_predictions_checked": len(sample),
    "test_in_rms_distance_to_each_clouds_own_centroid_km": {
        "ground_truth": spread_km(test.latitude, test.longitude),
        "T4": spread_km(test.lat_pred, test.lon_pred),
        "T1": spread_km(t1.lat_pred, t1.lon_pred),
    },
    "T4_raw_shift_1930_to_1950_at_fixed_interval_width_km": {
        "east": float(year_change[0]), "north": float(year_change[1]),
        "length": float(np.linalg.norm(year_change)),
    },
}
(results / "verification.json").write_text(json.dumps(checks, indent=2) + "\n")
print(json.dumps(checks, indent=2))
