"""Predict T4/T5 locations and confidence features in one complete excluded region."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import geopandas as gpd
import numpy as np
import pandas as pd
import torch
from pyproj import Transformer

from lib.confidence_features import GalleryFeatureExtractor
from models.clip_regression.lib.regression import clip_to_switzerland
from models.clip_regression.scripts.compare_t1 import _errors
from models.clip_regression.scripts.train import CHUNK_SIZE, LV95_ORIGIN_M, fit_linear
from models.geoclip_time.lib.temporal_data import load_temporal_data
from models.geoclip_time.lib.temporal_training import load_temporal_checkpoint
from models.geoclip_time.scripts.rerank_references import (
    DEFAULT_GAMMAS, _ids_hash, _sha256, dates_to_retrieval_years, select_gamma_on_validation,
)
from models.t5_reranker.candidates import encode_images, year_scores
from models.t5_reranker.rerank import WEIGHTS, _encode_gallery, _restore_section, _score, choose_candidates, select_weight
from models.t6_router.region_towers import fit_towers
from scripts.augmentation.parent_retrieval import exact_parent_cosine_topk, parent_reference_layout


RERANK_SELECTION = {"protocol": "full_fold_val_sequential_v1", "gamma_grid": list(DEFAULT_GAMMAS),
                    "weight_grid": list(WEIGHTS), "criterion": "minimum median km; ties prefer smaller coefficient"}


def _partition(rows):
    if not np.array_equal(rows.feat_idx, np.arange(len(rows))):
        raise ValueError("Expected time rows and feature rows in the same cache order")
    refs = rows.loc[rows.split.eq("train")].reset_index(drop=True)
    originals, groups = parent_reference_layout(refs)
    queries = rows.loc[rows.split.eq("query")].reset_index(drop=True)
    if not len(queries) or queries.id.le(0).any():
        raise ValueError("Queries must be positive original IDs")
    return refs, refs.iloc[originals].reset_index(drop=True), groups, queries


def _predict_t4(data, refs, parents, queries, border, output):
    source = refs.feat_idx.to_numpy()
    to_lv95 = Transformer.from_crs(4326, 2056, always_xy=True)
    to_wgs = Transformer.from_crs(2056, 4326, always_xy=True)
    target = np.column_stack(to_lv95.transform(refs.longitude.to_numpy(), refs.latitude.to_numpy()))
    model, rank = fit_linear(data.features, data.times, source, (target - LV95_ORIGIN_M) / 1000.)
    torch.save({"model": "T4", "feature_name": "clip_vitl14", "feature_dim": data.features.shape[1],
                "model_state": model.state_dict(), "time_config": data.audit["dates"]["config"],
                "output_crs": "EPSG:2056", "origin_m": LV95_ORIGIN_M.tolist(), "scale_m": 1000.}, output)
    idx = queries.feat_idx.to_numpy()
    parts = []
    with torch.inference_mode():
        for start in range(0, len(idx), CHUNK_SIZE):
            batch = idx[start:start + CHUNK_SIZE]
            features = torch.as_tensor(data.features[batch], dtype=torch.float64)
            dates = torch.as_tensor(data.times[batch], dtype=torch.float64)
            parts.append(model(features, dates).numpy())
    raw = np.concatenate(parts) * 1000. + LV95_ORIGIN_M
    projected = clip_to_switzerland(raw, border)
    lon, lat = to_wgs.transform(projected[:, 0], projected[:, 1])
    coords = np.column_stack([lat, lon])
    geo = GalleryFeatureExtractor(parents[["latitude", "longitude"]].to_numpy())
    features = geo.transform(coords, data.times[idx, 2], workers=4)
    features["t4_was_projected"] = (projected != raw).any(axis=1)
    features["T4_lat_pred"], features["T4_lon_pred"] = lat, lon
    features["T4_error_km"] = _errors(queries[["latitude", "longitude"]].to_numpy(), coords)
    return features, rank


@torch.inference_mode()
def _predict_t5(data, refs, parents, groups, queries, t0, t1, *, device, top_k=50,
                gallery_path=None, selection_path=None):
    vectors = encode_images(t0, data.features, refs.feat_idx.to_numpy(), 2048)
    coords = parents[["latitude", "longitude"]].to_numpy(dtype=np.float64)
    years, _ = dates_to_retrieval_years(parents.date_shot_min.to_numpy(), parents.date_shot_max.to_numpy())
    bases = _encode_gallery(t1, coords, device)

    def shortlist(rows):
        idx = rows.feat_idx.to_numpy()
        images = encode_images(t0, data.features, idx, 2048)
        candidates, visual = exact_parent_cosine_topk(images, vectors, groups, top_k=top_k, device=device)
        query_years, _ = dates_to_retrieval_years(rows.date_shot_min.to_numpy(), rows.date_shot_max.to_numpy())
        return candidates, visual, query_years, _score(t1, data, idx, bases, candidates, device)

    gamma, weight = 1., 1.
    if selection_path is not None:
        val = data.table.loc[data.table.split.eq("val")]
        if val.empty or val.id.le(0).any() or val.id.duplicated().any() or val.id.isin(queries.id).any():
            raise ValueError("Rerank fitting requires distinct positive fold-val IDs disjoint from queries")
        candidates, visual, val_years, location = shortlist(val)
        truth = val[["latitude", "longitude"]].to_numpy()
        gamma, gamma_curve = select_gamma_on_validation(candidates, visual, val_years, years, coords, truth, val.split)
        base = year_scores(visual, val_years, years[candidates], gamma)
        weight, weight_curve = select_weight(base, location, coords[candidates], truth, val.split)
        audit = {**RERANK_SELECTION, "split": "val", "n": len(val), "ids": val.id.to_list(),
                 "ids_sha256": _ids_hash(val.id), "gallery_ids_sha256": _ids_hash(parents.id),
                 "gamma": gamma, "weight": weight, "gamma_curve": gamma_curve, "weight_curve": weight_curve}
        selection_path.write_text(json.dumps(audit, allow_nan=False) + "\n", encoding="utf-8")

    parent_idx, visual, query_years, location = shortlist(queries)
    base = year_scores(visual, query_years, years[parent_idx], gamma)
    winner = choose_candidates(base, location, weight)
    row = np.arange(len(queries))
    selected = parent_idx[row, winner]
    if gallery_path is not None:
        np.savez(gallery_path, vectors=vectors, gps=coords, groups=groups,
                 ids=parents.id.to_numpy(dtype=np.int64), years=years)
    return pd.DataFrame({
        "T5_neighbor_id": parents.id.to_numpy()[selected], "T5_lat_pred": coords[selected, 0],
        "T5_lon_pred": coords[selected, 1], "t5_visual_rank": winner + 1,
        "t5_t3_score": base[row, winner], "t5_location_cosine": location[row, winner],
        "T5_error_km": _errors(queries[["latitude", "longitude"]].to_numpy(), coords[selected]),
    })


def run_experts(workspace, augmented_root, assigned_split_csv, output_dir, *, device="cuda", refit_rerank=False):
    workspace, augmented_root, split, output = map(Path, (workspace, augmented_root, assigned_split_csv, output_dir))
    torch.set_num_threads(4)
    if torch.device(device).type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
    checkpoints = fit_towers(augmented_root, split, output / "towers", device=device)
    border_path = workspace / "data/local_runs/t6_aug8192/bundle_v2/ne_admin0.gpkg"
    artifacts = output / "experts"
    source = {**{name: _sha256(path) for name, path in checkpoints.items()},
              "split_sha256": _sha256(split), "border_sha256": _sha256(border_path)}
    if refit_rerank:
        source["rerank_selection"] = RERANK_SELECTION
    if artifacts.exists():
        receipt = json.loads((artifacts / "summary.json").read_text(encoding="utf-8"))
        if receipt["inputs"] != source or receipt["predictions_sha256"] != _sha256(artifacts / "predictions.parquet"):
            raise ValueError("Existing regional expert predictions differ from this protocol")
        if refit_rerank and receipt["rerank_validation_sha256"] != _sha256(artifacts / "rerank_validation.json"):
            raise ValueError("Existing rerank validation evidence differs from this protocol")
        return artifacts / "predictions.parquet"
    data = load_temporal_data(augmented_root, split)
    refs, parents, groups, queries = _partition(data.table)
    border = gpd.read_file(border_path).to_crs(2056).geometry.union_all()
    with TemporaryDirectory(dir=output) as temp:
        stage = Path(temp) / "experts"
        stage.mkdir()
        t4, rank = _predict_t4(data, refs, parents, queries, border, stage / "T4.pt")
        t0, p0 = load_temporal_checkpoint(checkpoints["T0"])
        t1, p1 = load_temporal_checkpoint(checkpoints["T1"], device=device)
        selection_path = stage / "rerank_validation.json" if refit_rerank else None
        t5 = _predict_t5(data, refs, parents, groups, queries, t0, t1, device=device,
                         gallery_path=stage / "t5_gallery.npz", selection_path=selection_path)
        coefficients = {"gamma": 1., "weight": 1.}
        selection_receipt = {}
        if selection_path is not None:
            selected = json.loads(selection_path.read_text(encoding="utf-8"))
            coefficients = {key: selected[key] for key in coefficients}
            selection_receipt = {"rerank_validation_sha256": _sha256(selection_path),
                                 "rerank_validation_count": selected["n"],
                                 "rerank_validation_ids_sha256": selected["ids_sha256"]}
        inputs = json.loads((checkpoints["T0"].parents[1] / "provenance.json").read_text(encoding="utf-8"))["inputs"]
        torch.save({
            "model": "T5", "format_version": 1, **coefficients, **selection_receipt, "top_k": 50, "parent_grouping": True,
            "gallery_ids_sha256": _ids_hash(parents.id), "t0_source_sha256": source["T0"], "t1_source_sha256": source["T1"],
            "input_sha256": {"metadata": inputs["meta/smapshot.parquet"], "features": inputs["features/clip_vitl14.npy"],
                             "feature_ids": inputs["features/clip_vitl14_ids.npy"], "split": inputs["assigned_split"]},
            "t0": _restore_section(p0), "t1": _restore_section(p1),
        }, stage / "T5.pt")
        frame = pd.concat([queries[["id", "block", "latitude", "longitude"]], t4, t5], axis=1)
        assignments = pd.read_csv(split, usecols=lambda c: c in ("id", "region"))
        if "region" in assignments:
            frame["region"] = frame.id.map(assignments.set_index("id", verify_integrity=True).region)
        frame.to_parquet(stage / "predictions.parquet", index=False)
        summary = {"inputs": source, "query_count": len(frame), "query_ids_sha256": _ids_hash(frame.id),
                   "train_count": len(refs), "unique_train_parents": len(parents),
                   "gallery_ids_sha256": _ids_hash(parents.id), "train_ids_sha256": _ids_hash(refs.id),
                   "t4_design_rank": rank, "top_k": 50, **coefficients, **selection_receipt,
                   "predictions_sha256": _sha256(stage / "predictions.parquet")}
        (stage / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        stage.rename(artifacts)
    return artifacts / "predictions.parquet"


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("workspace", "augmented-root", "assigned-split-csv", "output-dir"):
        cli.add_argument("--" + name, type=Path, required=True)
    cli.add_argument("--device", default="cuda")
    cli.add_argument("--refit-rerank", action="store_true",
                     help="Fit the T3 year penalty and T5 location weight on all allowed fold validation photos")
    print(run_experts(**vars(cli.parse_args())), flush=True)
