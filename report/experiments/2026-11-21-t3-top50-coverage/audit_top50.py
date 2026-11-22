"""Reconstruct the original T3 visual shortlist and compare to its archived winner."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

base = Path("/rcp-scratch/students/zhang7/ml-project2")
original = base / "runs/2026-11-18-geoclip-t0-photo-rerank"
sys.path.insert(0, str(original / "source"))
from models.geoclip_time.lib.temporal_data import load_temporal_data
from models.geoclip_time.lib.temporal_training import load_temporal_checkpoint
from models.geoclip_time.lib.temporal_retrieval import exact_cosine_topk, rerank_by_date
from models.geoclip_time.scripts.rerank_references import (
    EVAL_SPLITS, _errors_km, _ids_hash, _progress, _sha256,
    _validate_checkpoint_data, dates_to_retrieval_years,
)

out = Path.cwd()
summary = json.loads((original / "results/summary.json").read_text())
manifest = json.loads((original / "input_manifest.json").read_text())
for entry in manifest["inputs"]:
    assert _sha256(entry["path"]) == entry["sha256"], entry["path"]
checkpoint = Path(summary["checkpoint"]["path"])
assert _sha256(checkpoint) == manifest["T0_sha256"] == summary["checkpoint"]["sha256"]
assert summary["top_k_effective"] == 50 and summary["selected_gamma"] == 1.0
torch.set_num_threads(summary["threads"])
model, payload = load_temporal_checkpoint(checkpoint, device="cpu")
model.eval().requires_grad_(False)
data = load_temporal_data(base / "data", Path(summary["inputs"]["split"]["path"]),
                          feature_name=payload["config"]["features"], time_config=payload["time_config"])
_validate_checkpoint_data(model, payload, data)
rows = data.table.loc[data.table.split.isin(("train", *EVAL_SPLITS))].copy().reset_index(drop=True)
print("Original inputs verified; encoding images and parsing dates.", flush=True)
years, known = dates_to_retrieval_years(rows.date_shot_min.to_numpy(), rows.date_shot_max.to_numpy())
embeddings = np.empty((len(rows), 512), dtype=np.float32)
with torch.inference_mode():
    batch = summary["encode_batch"]
    for start in range(0, len(rows), batch):
        ids = rows.feat_idx.iloc[start:start + batch].to_numpy(dtype=np.int64)
        features = torch.as_tensor(np.asarray(data.features[ids], dtype=np.float32), device="cpu")
        times = torch.zeros((len(ids), model.time_dim), dtype=torch.float32)
        embeddings[start:start + len(ids)] = model.encode_image(features, times).cpu().numpy()
reference = rows.split.eq("train").to_numpy()
references = rows.loc[reference].reset_index(drop=True)
gps_fields = ["latitude", "longitude"]
rgps = references[gps_fields].to_numpy(dtype=float)
rids = references.id.to_numpy()
assert _ids_hash(rids) == summary["references"]["ordered_ids_sha256"]
rfeatures = embeddings[reference]
predictions = pd.read_parquet(original / "results/predictions.parquet")
report = {"radius_km": 25, "k": 50, "training_references": len(rids), "gamma": 1.0,
          "winner_ids_checked_against_original_predictions": True, "splits": {}}


def fixed_list(values):
    return pa.FixedSizeListArray.from_arrays(pa.array(values.reshape(-1)), values.shape[1])


for split in ("test_in", "test_city", "test_region"):
    mask = rows.split.eq(split).to_numpy()
    queries = rows.loc[mask]
    ids = queries.id.to_numpy()
    assert _ids_hash(ids) == summary["queries"][split]["ordered_ids_sha256"]
    indices, scores = exact_cosine_topk(embeddings[mask], rfeatures, top_k=50,
                                        query_chunk=summary["query_chunk"], progress_callback=_progress(split))
    assert indices.shape == (len(ids), 50)
    assert indices.min() >= 0 and (np.diff(np.sort(indices, axis=1), axis=1) > 0).all()
    winner = rerank_by_date(indices, scores, years[mask], years[reference], rgps,
                            gamma=summary["selected_gamma"], query_known=known[mask], reference_known=known[reference])
    qgps = queries[gps_fields].to_numpy(dtype=float)
    dist = _errors_km(np.repeat(qgps, 50, axis=0), rgps[indices.reshape(-1)]).reshape(-1, 50)
    eligible = (dist <= 25).any(axis=1)
    in_list = indices == winner["neighbor_indices"][:, None]
    assert (in_list.sum(axis=1) == 1).all()
    year_rank = in_list.argmax(axis=1)
    for method, refs, errors in (
        ("visual", indices[:, 0], dist[:, 0]),
        ("year", winner["neighbor_indices"], dist[np.arange(len(ids)), year_rank]),
    ):
        recorded = predictions.loc[(predictions.split == split) & (predictions.method == method)].set_index("id")
        assert len(recorded) == len(ids) and recorded.index.is_unique
        assert np.array_equal(rids[refs], recorded.loc[ids, "neighbor_id"].to_numpy()), (split, method)
        np.testing.assert_allclose(errors, recorded.loc[ids, "error_km"], rtol=0, atol=1e-7)
    visual_hits = dist[:, 0] <= 25
    year_hits = dist[np.arange(len(ids)), year_rank] <= 25
    row = {
        "queries": len(ids), "top50_within_25km": int(eligible.sum()),
        "year_winner_within_25km": int(year_hits.sum()),
        "top50_hit_year_miss": int((eligible & ~year_hits).sum()),
        "no_top50_hit": int((~eligible).sum()),
        "visual_first_within_25km": int(visual_hits.sum()),
        "change_from_visual_miss_to_year_hit": int((~visual_hits & year_hits).sum()),
        "change_from_visual_hit_to_year_miss": int((visual_hits & ~year_hits).sum()),
    }
    report["splits"][split] = row
    print(split, row, flush=True)
    table = pa.table({"query_id": ids, "ranked_reference_ids": fixed_list(rids[indices]),
                      "ranked_distance_km": fixed_list(dist), "year_winner_rank_1_based": year_rank + 1})
    pq.write_table(table, out / (split + ".parquet"), compression="zstd")
report["total"] = {key: sum(r[key] for r in report["splits"].values()) for key in row}
(out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
print(report["total"], flush=True)
