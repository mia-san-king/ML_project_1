"""Locate one new photo from its raw CLIP ViT-L/14 feature and optional capture dates."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from models.geoclip_time.lib.temporal_data import encode_date_intervals, load_temporal_data
from models.geoclip_time.lib.temporal_retrieval import exact_cosine_topk
from models.geoclip_time.scripts.rerank_references import _ids_hash, _sha256, dates_to_retrieval_years
from models.t5_reranker.candidates import encode_images, year_scores
from models.t5_reranker.rerank import choose_candidates, location_cosines, restore_tower
from scripts.augmentation.parent_retrieval import exact_parent_cosine_topk, parent_reference_layout


def _load_query(feature):
    query = np.load(feature, allow_pickle=False)
    if query.shape not in {(768,), (1, 768)} or query.dtype.kind not in "fiu":
        raise ValueError("Expected a real raw CLIP ViT-L/14 feature of shape [768] or [1,768]")
    query = query.reshape(1, 768).astype(np.float32, copy=False)
    if not np.isfinite(query).all():
        raise ValueError("Query feature must contain only finite values")
    return query


def _load_bundle(model_path):
    bundle = torch.load(model_path, map_location="cpu", weights_only=True)
    if (bundle["model"], bundle["format_version"]) != ("T5", 1):
        raise ValueError("Expected a format version 1 T5 checkpoint")
    return bundle


def prepare_gallery(bundle, data_root, split_csv):
    """Precompute visual references, years and GPS from the checkpoint's training split."""
    data_root = Path(data_root)
    files = {"metadata": data_root / "meta/smapshot.parquet", "features": data_root / "features/clip_vitl14.npy",
             "feature_ids": data_root / "features/clip_vitl14_ids.npy", "split": Path(split_csv)}
    for name, path in files.items():
        if _sha256(path) != bundle["input_sha256"][name]:
            raise ValueError(f"Training gallery {name} differs from the fitted T5 checkpoint")
    data = load_temporal_data(data_root, split_csv, time_config=bundle["t0"]["time_config"])
    references = data.table.loc[data.table.split.eq("train")].reset_index(drop=True)
    parents = references
    groups = np.empty(0, dtype=np.int64)
    if bundle["parent_grouping"]:
        originals, groups = parent_reference_layout(references)
        parents = references.iloc[originals]
    if (parents.id <= 0).any() or _ids_hash(parents.id) != bundle["gallery_ids_sha256"]:
        raise ValueError("Training parent IDs or their order differ from the fitted T5 checkpoint")

    t0 = restore_tower(bundle["t0"], "cpu").eval()
    vectors = encode_images(t0, data.features, references.feat_idx.to_numpy(), 2048)
    intervals = pd.MultiIndex.from_frame(parents[["date_shot_min", "date_shot_max"]])
    codes, unique = intervals.factorize(sort=False)
    years, _ = dates_to_retrieval_years(unique.get_level_values(0), unique.get_level_values(1))
    return {"ids": parents.id.to_numpy(dtype=np.int64), "gps": parents[["latitude", "longitude"]].to_numpy(),
            "years": years[codes], "vectors": vectors, "groups": groups}


def _score_photo(bundle, gallery, query, date_min, date_max, device):
    ids, gps, years = gallery["ids"], gallery["gps"], gallery["years"]
    groups, reference_vectors = gallery["groups"], gallery["vectors"]
    if (ids.ndim != 1 or gps.shape != (len(ids), 2) or years.shape != (len(ids),)
            or _ids_hash(ids) != bundle["gallery_ids_sha256"]):
        raise ValueError("Indexed training parents do not match the T5 checkpoint")
    t0 = restore_tower(bundle["t0"], "cpu").eval()
    query_vector = encode_images(t0, query, np.array([0]), 1)
    k = int(bundle["top_k"])
    if bundle["parent_grouping"]:
        shortlist, visual = exact_parent_cosine_topk(query_vector, reference_vectors, groups, top_k=k, device="cpu")
    else:
        if len(reference_vectors) != len(ids) or groups.size:
            raise ValueError("Ungrouped T5 index requires one vector per gallery parent")
        shortlist, visual = exact_cosine_topk(query_vector, reference_vectors, top_k=k)
    candidates = shortlist[0]
    query_year, _ = dates_to_retrieval_years([date_min], [date_max])
    base = year_scores(visual, query_year, years[shortlist], bundle["gamma"])

    interval = pd.DataFrame({"date_shot_min": [date_min], "date_shot_max": [date_max]})
    query_time, _ = encode_date_intervals(interval, **bundle["t1"]["time_config"])
    t1 = restore_tower(bundle["t1"], device).eval()
    gps = gps[candidates]
    with torch.inference_mode():
        bases = t1.encode_location_base(torch.as_tensor(gps, dtype=torch.float32, device=device))
        location = location_cosines(t1, torch.as_tensor(query, device=device),
                                    torch.as_tensor(query_time, device=device), bases[None]).cpu().numpy()
    weight = bundle["weight"]
    position = int(choose_candidates(base, location, weight)[0])
    return {"neighbor_id": int(ids[candidates[position]]),
            "lat_pred": float(gps[position, 0]), "lon_pred": float(gps[position, 1]),
            "visual_rank": position + 1, "visual_score": float(visual[0, position]),
            "t3_score": float(base[0, position]), "location_cosine": float(location[0, position]),
            "score": float(base[0, position] + weight * float(location[0, position])),
            "gamma": float(bundle["gamma"]), "weight": float(weight)}


def predict_photo(*, model_path, data_root, split_csv, feature, date_min=None, date_max=None, device="cpu"):
    """Return the winning real training photo and its GPS using the fitted T5 bundle."""
    query = _load_query(feature)
    bundle = _load_bundle(model_path)
    gallery = prepare_gallery(bundle, data_root, split_csv)
    return _score_photo(bundle, gallery, query, date_min, date_max, device)


def predict_indexed_photo(*, model_path, gallery_path, feature, date_min=None, date_max=None, device="cpu"):
    """Run T5 from its frozen model and an NPZ produced by prepare_gallery."""
    query = _load_query(feature)
    bundle = _load_bundle(model_path)
    with np.load(gallery_path, allow_pickle=False) as gallery:
        return _score_photo(bundle, gallery, query, date_min, date_max, device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", dest="model_path", required=True, type=Path, help="Fitted T5.pt bundle")
    parser.add_argument("--data-root", required=True, type=Path, help="Input gallery used to fit T5")
    parser.add_argument("--split-csv", required=True, type=Path)
    parser.add_argument("--feature", required=True, type=Path, help="Raw 768-D, pre-projection CLIP feature (.npy)")
    parser.add_argument("--date-min", help="Lower endpoint of the query capture interval (ISO date or year)")
    parser.add_argument("--date-max", help="Upper endpoint of the query capture interval (ISO date or year)")
    parser.add_argument("--device", default="cpu", help="T1 scoring device, e.g. cpu or cuda")
    args = parser.parse_args()
    torch.set_num_threads(4)
    print(json.dumps(predict_photo(**vars(args)), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
