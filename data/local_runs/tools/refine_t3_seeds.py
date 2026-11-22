"""Evaluate frozen GeoCLIP coordinate gradients starting from logged T3 year-winners."""
import hashlib
import json
import os
import sys
from pathlib import Path
from experiment_paths import T1_ORIGINAL, T1_AUGMENTED

os.environ["HF_HUB_OFFLINE"] = "1"
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import torch
from torch.nn.functional import normalize

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from models.geoclip_time.lib.temporal_data import encode_date_intervals
from models.geoclip_time.lib.temporal_training import load_temporal_checkpoint
from models.geoclip_time.scripts.rerank_references import _errors_km, _sha256, _write_json

RUNS = ROOT / "data/local_runs"
PAIRED = RUNS / "t3_t4_8192"
DATA = ROOT / "data/t1_original"
DEST = RUNS / "t3_refine_analysis"
T1_PATHS = {
    "original": T1_ORIGINAL / "ckpt/zhang7_T1_dual_time_available_s0_20261118.pt",
    "augmented": T1_AUGMENTED / "ckpt/T1_aug8192_s0.pt",
}
STEPS = (0.25, 0.25, 0.25, 0.25, 0.125, 0.0625, 0.03125, 0.015625)
TOTALS = {"test_in": 47593, "test_city": 3377, "test_region": 13039}


def optimize(model, features, times, seed, border, radius, iterations):
    device = features.device
    with torch.no_grad():
        image = model.encode_image(features, times)
    origin = torch.as_tensor(seed, dtype=torch.float64, device=device)
    km_per_degree = np.pi / 180 * 6371.0088
    scale = km_per_degree * torch.stack((torch.ones_like(origin[:, 0]), torch.cos(torch.deg2rad(origin[:, 0]))), dim=1)

    def score(index, offset):
        gps = origin[index] + offset / scale[index]
        loc = model.encode_location_base(gps.float())
        if model.location_time_adapter is not None:
            loc = model.location_time_adapter(loc, times[index])
        return gps, (normalize(loc, dim=-1) * image[index]).sum(dim=1)

    active = torch.arange(len(seed), device=device)
    offset = torch.zeros_like(origin)
    with torch.no_grad():
        _, initial_score = score(active, offset)
    best = initial_score.clone()
    moves = torch.zeros_like(active)
    for step in range(iterations):
        if not len(active):
            break
        if step % len(STEPS) == 0:
            cycle_start = moves[active].clone()
        here = offset[active].detach().requires_grad_(True)
        _, current = score(active, here)
        gradient, = torch.autograd.grad(current.sum(), here)
        if not torch.isfinite(gradient).all():
            raise ValueError("Nonfinite coordinate gradient")
        with torch.no_grad():
            proposal = here + STEPS[step % len(STEPS)] * normalize(gradient, dim=1)
            proposal = proposal / (proposal.norm(dim=1, keepdim=True) / radius).clamp(min=1.)
            gps, proposed_score = score(active, proposal)
            xy = gps.cpu().numpy()
            inside = torch.from_numpy(shapely.intersects_xy(border, xy[:, 1], xy[:, 0])).to(device)
            take = inside & (proposed_score > current) & (proposed_score > best[active])
            changed = active[take]
            offset[changed] = proposal[take]
            best[changed] = proposed_score[take]
            moves[changed] += 1
        if (step + 1) % len(STEPS) == 0:
            active = active[moves[active] != cycle_start]
    return (origin + offset / scale).cpu().numpy(), (best - initial_score).cpu().numpy(), moves.cpu().numpy()


def metrics(frame, radius):
    a, b = frame.seed_error_km, frame.final_error_km
    old, new = a <= 25, b <= 25
    changed = frame.accepted_moves > 0
    return {
        "n": len(frame), "hits_before": int(old.sum()), "hits_after": int(new.sum()),
        "pct_before": 100 * float(old.mean()), "pct_after": 100 * float(new.mean()),
        "gained": int((~old & new).sum()), "lost": int((old & ~new).sum()),
        "median_before_km": float(a.median()), "median_after_km": float(b.median()),
        "misses_within_radius_of_threshold": int(((a > 25) & (a <= 25 + radius * 1.002)).sum()),
        "closer": int((a - b > 1e-8).sum()), "farther": int((b - a > 1e-8).sum()),
        "score_increased": int((frame.score_gain > 0).sum()), "moved": int(changed.sum()),
        "median_displacement_km": float(frame.displacement_km.median()), "max_displacement_km": float(frame.displacement_km.max()),
    }


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(3)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    coords = ["latitude", "longitude"]
    pred_coords = ["lat_pred", "lon_pred"]
    meta = pd.read_parquet(DATA / "meta/smapshot.parquet", columns=["id", *coords, "date_shot_min", "date_shot_max"]).set_index("id", verify_integrity=True)
    split = pd.read_csv(ROOT / "data/splits/main.csv.gz", usecols=["id", "split"]).set_index("id", verify_integrity=True).split
    feat_ids = pd.Index(np.load(DATA / "features/clip_vitl14_ids.npy"))
    features = np.load(DATA / "features/clip_vitl14.npy", mmap_mode="r")
    border_path = PAIRED / "t4_source/ne_admin0.gpkg"
    border = gpd.read_file(border_path).to_crs(4326).geometry.union_all()
    shapely.prepare(border)
    report = {"gpu": torch.cuda.get_device_name(), "seed_method": "fixed T3 year winner",
              "steps_km": STEPS, "earth_radius_km": 6371.0088, "border_sha256": _sha256(border_path),
              "rule": "positive matched-encoder query-to-coordinate cosine improvement, in Switzerland; weights frozen",
              "runs": {}}
    all_frames = []
    for group in ("augmented", "original"):
        preds = PAIRED / group / "T3/predictions.parquet"
        seeds = pd.read_parquet(preds, filters=[("method", "=", "year"), ("split", "in", list(TOTALS))]).set_index("id", verify_integrity=True).sort_index()
        if seeds.groupby("split").size().to_dict() != TOTALS or not seeds.split.equals(split.loc[seeds.index]):
            raise ValueError("Original test ID counts/splits differ")
        np.testing.assert_array_equal(seeds[pred_coords], meta.loc[seeds.neighbor_id, coords])
        if not split.loc[seeds.neighbor_id].eq("train").all() or min(seeds.index) <= 0:
            raise ValueError("Seed provenance mismatch")
        t0 = PAIRED / group / "T0/ckpt" / f"T0_t3_{group}_s0.pt"
        for tower, path in (("T1", T1_PATHS[group]), ("T0", t0)):
            model, payload = load_temporal_checkpoint(path, device="cuda")
            model.eval().requires_grad_(False)
            policies = [(1, 8, list(TOTALS))]
            if group == "augmented" and tower == "T1":
                policies.append((5, 128, ["test_city", "test_region"]))
            for radius, steps, use_splits in policies:
                records = seeds.loc[seeds.split.isin(use_splits)]
                metadata = meta.loc[records.index]
                dates = encode_date_intervals(metadata, **payload["time_config"])[0]
                positions = feat_ids.get_indexer(records.index)
                if min(positions) < 0:
                    raise ValueError("Missing image feature")
                initial = records[pred_coords].to_numpy(dtype=float)
                parts = []
                for start in range(0, len(records), 512):
                    block = slice(start, start + 512)
                    image = torch.as_tensor(features[positions[block]].astype(np.float32), device="cuda")
                    times = torch.as_tensor(dates[block], dtype=torch.float32, device="cuda")
                    parts.append(optimize(model, image, times, initial[block], border, radius, steps))
                final, gains, moves = [np.concatenate(parts_for_field) for parts_for_field in zip(*parts)]
                # Access query labels only for evaluation after inference.
                truth = metadata[coords].to_numpy(dtype=float)
                before = _errors_km(truth, initial)
                after = _errors_km(truth, final)
                travel = _errors_km(initial, final)
                np.testing.assert_allclose(before, records.error_km.to_numpy(), atol=1e-7, rtol=0)
                if max(travel) > radius * 1.002 or not np.isfinite(after).all() or np.any(gains < 0):
                    raise ValueError("Invalid optimized result")
                frames = records[["split", "neighbor_id", *pred_coords]].copy()
                frames = frames.assign(group=group,tower=tower,radius_km=radius,steps=steps,final_lat=final[:,0],final_lon=final[:,1],
                    seed_error_km=before, final_error_km=after, score_gain=gains, accepted_moves=moves, displacement_km=travel)
                key = f"{group}_{tower}_{radius}km_{steps}steps"
                report["runs"][key] = {"checkpoint": str(path), "epoch": payload["epoch"], "checkpoint_sha256": _sha256(path),
                    "seed_predictions_sha256": _sha256(preds), "splits": {k:metrics(g,radius) for k,g in frames.groupby("split")}}
                all_frames.append(frames.reset_index())
                print(key, json.dumps({k:[v["gained"],v["lost"]] for k,v in report["runs"][key]["splits"].items()}),flush=True)
    parquet = DEST / "refined_predictions.parquet"
    temporary = parquet.with_suffix(".tmp")
    pd.concat(all_frames,ignore_index=True).to_parquet(temporary,index=False)
    temporary.replace(parquet)
    report["predictions_sha256"] = _sha256(parquet)
    _write_json(DEST / "refinement.json", report)


if __name__ == "__main__":
    main()
