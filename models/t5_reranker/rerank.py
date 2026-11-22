"""Fit the T5 fusion weight on validation and export predictions with it fixed."""
import argparse
import json
import tempfile
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from torch.nn.functional import normalize

from models.geoclip_time.lib.temporal_data import load_temporal_data
from models.geoclip_time.lib.temporal_factory import build_temporal_model
from models.geoclip_time.lib.temporal_training import load_temporal_checkpoint
from models.geoclip_time.scripts.rerank_references import EVAL_SPLITS, _errors_km, _ids_hash, _metrics, _sha256


# Fixed before looking at any T5 results. Zero exactly reproduces T3.
WEIGHTS = (0., 0.03125, 0.0625, 0.125, 0.25, 0.5, 0.75, 1., 1.5, 2., 3., 4., 8., 16.)


def choose_candidates(base_scores, location_scores, weight):
    """Return visual-rank indices, retaining the first visual rank on exact ties."""
    base = np.asarray(base_scores, dtype=np.float64)
    location = np.asarray(location_scores, dtype=np.float64)
    if base.ndim != 2 or base.shape[1] == 0 or location.shape != base.shape:
        raise ValueError("Expected matching query-by-candidate score matrices")
    if not np.isfinite(weight) or weight < 0 or not np.isfinite(base).all() or not np.isfinite(location).all():
        raise ValueError("Scores must be finite and the location weight finite and nonnegative")
    return np.argmax(base + weight * location, axis=1)


def select_weight(base_scores, location_scores, candidate_gps, query_gps, query_splits, weights=WEIGHTS):
    """Fit a single coefficient by minimum full-validation median geographic error."""
    base = np.asarray(base_scores)
    splits = np.asarray(query_splits)
    gps = np.asarray(candidate_gps)
    if base.ndim != 2 or not len(base) or splits.shape != (len(base),) or not (splits == "val").all():
        raise ValueError("Weight selection requires exclusively validation queries")
    if gps.shape != (*base.shape, 2) or np.shape(query_gps) != (len(base), 2):
        raise ValueError("GPS and score rows must align")
    grid = sorted(set(weights))
    if not grid:
        raise ValueError("Supply a nonempty weight grid")
    rows = np.arange(len(base))
    curve = []
    for weight in grid:
        winners = choose_candidates(base, location_scores, weight)
        errors = _errors_km(query_gps, gps[rows, winners])
        curve.append({"weight": float(weight), **_metrics(errors)})
    best = min(curve, key=lambda row: (row["median_km"], row["weight"]))
    return best["weight"], curve


@torch.inference_mode()
def location_cosines(model, query_features, query_times, candidate_bases):
    """Score the K GPS bases in each query's own row, using its date in both towers."""
    if model.mode != "both" or candidate_bases.ndim != 3 or len(candidate_bases) != len(query_features):
        raise ValueError("Expected the dual-time T1 and one candidate list per query")
    images = model.encode_image(query_features, query_times)
    locations = normalize(model.location_time_adapter(candidate_bases, query_times[:, None, :]), dim=-1)
    return (images[:, None, :] * locations).sum(dim=-1)


def restore_tower(section, device):
    model = build_temporal_model(section["architecture"], pretrained=False)
    model.load_state_dict(section["model_state"], strict=True)
    return model.eval().requires_grad_(False).to(device)


def _restore_section(payload):
    return {key: payload[key] for key in ("architecture", "time_config", "model_state")}


@torch.inference_mode()
def _encode_gallery(model, coords, device):
    parts = []
    for start in range(0, len(coords), 2048):
        gps = torch.as_tensor(coords[start:start+2048], dtype=torch.float32, device=device)
        parts.append(model.encode_location_base(gps))
    return torch.cat(parts)


def _load_shortlist(directory, split, provenance, index, train_ids):
    path = directory / f"{split}.npz"
    expected = provenance["splits"][split]
    if _sha256(path) != expected["sha256"]:
        raise ValueError(f"{split} candidates differ from manifest")
    with np.load(path, allow_pickle=False) as cache:
        qids, parents, base = (cache[key] for key in ("query_ids", "parent_ids", "base_scores"))
    if (len(qids) != expected["n"] or not pd.Index(qids).is_unique or (qids <= 0).any()
            or _ids_hash(qids) != expected["ordered_ids_sha256"]
            or not index.loc[qids, "split"].eq(split).all()):
        raise ValueError(f"Query IDs do not match {split}")
    if parents.shape != (len(qids), 50) or base.shape != parents.shape:
        raise ValueError("Expected exactly 50 T3 parents per query")
    refs = train_ids.get_indexer(parents.ravel()).reshape(parents.shape)
    if (refs < 0).any() or (np.diff(np.sort(refs, axis=1), axis=1) == 0).any():
        raise ValueError("Shortlist must contain distinct positive training parents")
    return qids, refs, base


def _score(model, data, positions, bases, refs, device):
    scores = np.empty(refs.shape, dtype=np.float32)
    for start in range(0, len(refs), 128):
        block = slice(start, start + 128)
        rows = positions[block]
        feats = torch.as_tensor(data.features[rows], dtype=torch.float32, device=device)
        times = torch.as_tensor(data.times[rows], dtype=torch.float32, device=device)
        idx = torch.as_tensor(refs[block], dtype=torch.long, device=device)
        scores[block] = location_cosines(model, feats, times, bases[idx]).cpu().numpy()
    if not np.isfinite(scores).all() or np.abs(scores).max() > 1.00001:
        raise ValueError("Location tower produced invalid cosine scores")
    return scores


def run(data_root, split_csv, candidates, t0, t1, output, device):
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    started = perf_counter()
    torch.set_num_threads(4)
    if torch.device(device).type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
    manifest = json.loads((candidates / "manifest.json").read_text(encoding="utf-8"))
    inputs = {"features": data_root / "features/clip_vitl14.npy",
              "feature_ids": data_root / "features/clip_vitl14_ids.npy",
              "metadata": data_root / "meta/smapshot.parquet", "split": split_csv}
    if any(_sha256(path) != manifest["input_sha256"][key] for key, path in inputs.items()):
        raise ValueError("Candidate gallery and T5 inputs differ")
    if _sha256(t0) != manifest["checkpoint_sha256"]:
        raise ValueError("Wrong T0 for these shortlists")
    t0_payload = torch.load(t0, map_location="cpu", weights_only=True)
    model, t1_payload = load_temporal_checkpoint(t1, device=device)
    model.requires_grad_(False)
    if model.mode != "both" or t0_payload["architecture"]["mode"] != "none":
        raise ValueError("Expected the T0 visual tower and the T1 dual-time checkpoint")
    data = load_temporal_data(data_root, split_csv, time_config=t1_payload["time_config"])
    for field in ("coverage", "metadata_rows", "feature_rows", "split_rows", "feature_dim"):
        if any(payload["audit"][field] != data.audit[field] for payload in (t0_payload, t1_payload)):
            raise ValueError(f"Checkpoint and dataset differ at {field}")
    index = data.table.set_index("id", verify_integrity=True)
    if not np.array_equal(index.feat_idx, np.arange(len(index))):
        raise ValueError("Expected time rows and feature rows to share cache order")
    gallery = index.loc[(index.index > 0) & index.split.eq("train")]
    coords = gallery[["latitude", "longitude"]].to_numpy(dtype=np.float64)
    gallery_digest = _ids_hash(gallery.index)
    if gallery_digest != manifest["training_reference_ids_sha256"]:
        raise ValueError("Real training parent order differs from T3")
    bases = _encode_gallery(model, coords, device)
    output.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        stage = Path(temporary) / "results"
        stage.mkdir()
        weight = None
        for split in EVAL_SPLITS:
            qids, refs, base = _load_shortlist(candidates, split, manifest, index, gallery.index)
            positions = index.loc[qids, "feat_idx"].to_numpy()
            location = _score(model, data, positions, bases, refs, device)
            if split == "val":
                validation = index.loc[qids]
                truth = validation[["latitude", "longitude"]].to_numpy()
                weight, curve = select_weight(base, location, coords[refs], truth, validation.split)
                selection = {"split": split, "n": len(qids), "query_ids_sha256": _ids_hash(qids),
                             "criterion": "lowest median_km, ties prefer smaller weight", "weight": weight, "curve": curve}
                (stage / "validation.json").write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
                bundle = {"model": "T5", "format_version": 1, "weight": weight, "gamma": manifest["gamma"],
                          "top_k": 50, "parent_grouping": manifest["parent_grouping"], "gallery_ids_sha256": gallery_digest,
                          "input_sha256": manifest["input_sha256"], "t0_source_sha256": manifest["checkpoint_sha256"],
                          "t1_source_sha256": _sha256(t1), "t0": _restore_section(t0_payload), "t1": _restore_section(t1_payload)}
                torch.save(bundle, stage / "T5.pt")
                print(f"T5 validation selected weight={weight} across {len(qids)} real photos", flush=True)
            if weight is None:
                raise RuntimeError("Fit on validation before evaluating test splits")
            winner = choose_candidates(base, location, weight)
            row = np.arange(len(refs))
            gallery_rows = refs[row, winner]
            score = base[row, winner] + weight * location[row, winner].astype(np.float64)
            frames.append(pd.DataFrame({"id": qids, "split": split, "neighbor_id": gallery.index.to_numpy()[gallery_rows],
                                        "lat_pred": coords[gallery_rows, 0], "lon_pred": coords[gallery_rows, 1],
                                        "visual_rank": winner + 1, "t3_score": base[row, winner],
                                        "location_cosine": location[row, winner], "score": score}))
            print(f"Scored {split}: {len(qids)} x 50", flush=True)
        predictions = stage / "predictions.parquet"
        pd.concat(frames, ignore_index=True).to_parquet(predictions, index=False)
        receipt = {"model": "T5", "weight": weight, "gamma": manifest["gamma"], "device": device,
                   "hardware": torch.cuda.get_device_name(device) if torch.device(device).type == "cuda" else "CPU",
                   "torch": str(torch.__version__), "fit_and_predict_seconds": perf_counter() - started,
                   "candidates_sha256": _sha256(candidates / "manifest.json"), "checkpoint_sha256": _sha256(stage / "T5.pt"),
                   "predictions_sha256": _sha256(predictions), "query_count": sum(map(len, frames))}
        (stage / "run.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        stage.rename(output)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data-root", "split-csv", "candidates", "t0", "t1", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    print(json.dumps(run(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
