#!/usr/bin/env python3
"""T3 (Yang_Zhang): T0 image retrieval with capture-year reranking.

The frozen, trained T0 image projection maps cached CLIP features to 512-D.
Only training photos are references; their actual coordinates are predictions.
This does not use the location tower or the geographic candidate grid.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time


DEFAULT_GAMMAS = (0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0)
EVAL_SPLITS = ("val", "test_in", "test_city", "test_region")
TOP_K = 50


def dates_to_retrieval_years(date_mins, date_maxs):
    """Match f019e23:lib/time_feats.py:dates_to_features scalar semantics.

    Parse each endpoint separately, including mixed pandas object columns.
    Keep the historical midpoint/day-of-year rule, numeric nanosecond parsing,
    reversed-interval handling and midpoint-only [1800,2030] validity check.
    """
    import numpy as np
    import pandas as pd

    if len(date_mins) != len(date_maxs):
        raise ValueError("date endpoint arrays must have equal lengths")
    years = np.full(len(date_mins), np.nan, dtype=np.float64)
    for i, (lower, upper) in enumerate(zip(date_mins, date_maxs)):
        a = pd.to_datetime(lower, errors="coerce", utc=True)
        b = pd.to_datetime(upper, errors="coerce", utc=True)
        if pd.isna(a) or pd.isna(b):
            continue
        midpoint = a + (b - a) / 2
        year = float(midpoint.year) + (float(midpoint.day_of_year) - 1) / 366.0
        if 1800 <= year <= 2030:
            years[i] = year
    return years, np.isfinite(years)


def _errors_km(truth, predicted):
    import numpy as np

    truth, predicted = np.asarray(truth, dtype=float), np.asarray(predicted, dtype=float)
    if truth.ndim != 2 or truth.shape[1] != 2 or predicted.shape != truth.shape:
        raise ValueError("truth and predictions must have matching [N,2] shapes")
    if not len(truth) or not np.isfinite(truth).all() or not np.isfinite(predicted).all():
        raise ValueError("truth and predictions must be nonempty and finite")
    truth, predicted = np.radians(truth), np.radians(predicted)
    delta = predicted - truth
    h = np.sin(delta[:, 0] / 2) ** 2 + np.cos(truth[:, 0]) * np.cos(predicted[:, 0]) * np.sin(delta[:, 1] / 2) ** 2
    return 2 * 6371.0088 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def _metrics(errors):
    import numpy as np

    return {"n": int(len(errors)), "median_km": float(np.median(errors)),
            "mean_km": float(np.mean(errors)), "acc@25km": float(np.mean(errors <= 25) * 100)}


def select_gamma_on_validation(
    indices, scores, query_years, reference_years, reference_gps,
    query_truth, query_splits, *, gammas=DEFAULT_GAMMAS,
    query_known=None, reference_known=None,
):
    """Select minimum full-validation median; exact ties keep the first gamma."""
    import numpy as np
    from models.geoclip_time.lib.temporal_retrieval import rerank_by_date

    splits = np.asarray(query_splits)
    if splits.shape != (len(indices),) or not len(splits) or not (splits == "val").all():
        raise ValueError("gamma selection requires exclusively nonempty val rows")
    gammas = tuple(float(gamma) for gamma in gammas)
    if not gammas or not np.isfinite(gammas).all() or min(gammas) < 0:
        raise ValueError("gammas must be nonempty, finite and nonnegative")
    curve = []
    for gamma in gammas:
        result = rerank_by_date(indices, scores, query_years, reference_years, reference_gps,
                                gamma=gamma, query_known=query_known, reference_known=reference_known)
        curve.append({"gamma": gamma, **_metrics(_errors_km(query_truth, result["pred_gps"]))})
    return min(curve, key=lambda row: row["median_km"])["gamma"], curve


def build_prediction_rows(query_rows, reference_rows, reranked, *, method, gamma):
    """Join the actual winning reference index to its ID and coordinates."""
    import numpy as np

    if not len(reference_rows) or not reference_rows["split"].eq("train").all():
        raise ValueError("reference rows must contain train only")
    if query_rows.id.duplicated().any() or reference_rows.id.duplicated().any():
        raise ValueError("query and reference IDs must be unique")
    if query_rows.id.isin(reference_rows.id).any():
        raise ValueError("query IDs must not overlap training references")
    winners = np.asarray(reranked["neighbor_indices"])
    if winners.shape != (len(query_rows),) or not np.issubdtype(winners.dtype, np.integer):
        raise ValueError("neighbor_indices must be an integer array aligned with query rows")
    if (winners < 0).any() or (winners >= len(reference_rows)).any():
        raise ValueError("neighbor_indices must index reference rows")
    neighbors = reference_rows.iloc[winners]
    gps = neighbors[["latitude", "longitude"]].to_numpy(dtype=float)
    if not np.array_equal(gps, np.asarray(reranked["pred_gps"])):
        raise ValueError("predicted GPS does not match the winning reference ID")
    out = query_rows[["id", "split"]].copy().reset_index(drop=True)
    out["method"], out["gamma"] = method, float(gamma)
    out["neighbor_id"], out["neighbor_split"] = neighbors.id.to_numpy(), "train"
    out["lat_pred"], out["lon_pred"] = gps[:, 0], gps[:, 1]
    out["visual_score"] = reranked["visual_scores"]
    out["year_penalty"] = reranked["year_penalties"]
    out["score"] = reranked["scores"]
    return out


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ids_hash(ids):
    import numpy as np

    return hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest()


def _write_json(path, payload):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _progress(stage):
    started, last = time.monotonic(), [0.0]

    def report(done, total):
        now = time.monotonic()
        if done == total or now - last[0] >= 30:
            print(f"{stage}: {done}/{total}, elapsed={now - started:.1f}s", flush=True)
            last[0] = now
    return report


def _validate_checkpoint_data(model, payload, data):
    if payload["architecture"].get("mode") != "none" or model.mode != "none":
        raise ValueError("reference retrieval requires a trained T0 checkpoint with mode=none")
    if payload["config"].get("model", {}).get("mode") != "none":
        raise ValueError("checkpoint configuration must also specify mode=none")
    if model.embed_dim != 512 or data.features.shape[1] != 768:
        raise ValueError("this experiment requires cached 768-D CLIP features and 512-D T0 embeddings")
    if payload.get("epoch", 0) < 1 or not payload.get("history"):
        raise ValueError("checkpoint must include completed training history")
    initialization = payload["config"].get("initialization")
    if initialization not in (None, "pretrained"):
        raise ValueError("this experiment requires the pretrained-then-trained T0 checkpoint")
    original = payload.get("audit", {})
    for key in ("coverage", "feature_rows", "feature_dim", "metadata_rows", "split_rows"):
        if key not in original or original[key] != data.audit[key]:
            raise ValueError(f"input audit differs from the T0 checkpoint: {key}")
    for split in ("train", *EVAL_SPLITS):
        coverage = data.audit["coverage"].get(split, {})
        if not coverage.get("available", 0):
            raise ValueError(f"no available feature rows for {split}")
        if coverage["missing"] and not payload["config"].get("allow_partial_features", False):
            raise ValueError(f"partial feature coverage was not authorized by T0: {split}")
    validation_ids = payload.get("validation_ids", [])
    available_val = set(data.table.loc[data.table.split.eq("val"), "id"].tolist())
    if not validation_ids or len(set(validation_ids)) != len(validation_ids) or not set(validation_ids) <= available_val:
        raise ValueError("checkpoint validation IDs must be unique and present in the available val split")


def run_retrieval(*, data_root, split_csv, checkpoint, output_dir, threads=4,
                  query_chunk=64, encode_batch=2048, parent_grouping=False, device="cpu"):
    """Run retrieval; optionally max over original/variant parent groups first."""
    for key, value in (("threads", threads), ("query_chunk", query_chunk), ("encode_batch", encode_batch)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{key} must be a positive integer")
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    if not isinstance(parent_grouping, bool):
        raise ValueError("parent_grouping must be boolean")
    if not parent_grouping and device != "cpu":
        raise ValueError("GPU retrieval is only supported with parent_grouping; default retrieval is CPU")
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[variable] = str(threads)
    import numpy as np
    import pandas as pd
    import torch
    from models.geoclip_time.lib.temporal_data import load_temporal_data
    from models.geoclip_time.lib.temporal_training import load_temporal_checkpoint
    from models.geoclip_time.lib.temporal_retrieval import exact_cosine_topk, rerank_by_date

    if torch.device(device).type not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    torch.set_num_threads(threads)
    started = time.monotonic()
    data_root, split_csv, checkpoint = (Path(path).expanduser().resolve() for path in (data_root, split_csv, checkpoint))
    print(f"retrieval: device={device}, threads={threads}, query_chunk={query_chunk}, top_k={TOP_K}, parent_grouping={parent_grouping}", flush=True)
    print(f"loading frozen T0 checkpoint: {checkpoint}", flush=True)
    model, payload = load_temporal_checkpoint(checkpoint, device=device)
    model.eval().requires_grad_(False)
    print("validating feature cache, metadata, split IDs and T0 coverage", flush=True)
    feature_name = payload["config"]["features"]
    data = load_temporal_data(data_root, split_csv, feature_name=feature_name, time_config=payload["time_config"])
    _validate_checkpoint_data(model, payload, data)
    print("coverage=" + json.dumps(data.audit["coverage"], sort_keys=True), flush=True)
    rows = data.table.loc[data.table.split.isin(("train", *EVAL_SPLITS))].copy().reset_index(drop=True)
    if parent_grouping:
        from scripts.augmentation.parent_retrieval import (
            exact_parent_cosine_topk, parent_reference_layout, validate_augmented_rows,
        )
        augmentation_audit = validate_augmented_rows(rows)
    print(f"parsing scalar capture dates once for {len(rows)} used image IDs", flush=True)
    years, known = dates_to_retrieval_years(rows.date_shot_min.to_numpy(), rows.date_shot_max.to_numpy())
    embeddings = np.empty((len(rows), 512), dtype=np.float32)
    report_encoding = _progress("encode frozen T0 image projection")
    with torch.inference_mode():
        for start in range(0, len(rows), encode_batch):
            positions = rows.feat_idx.iloc[start:start + encode_batch].to_numpy(dtype=np.int64)
            features = torch.as_tensor(np.asarray(data.features[positions], dtype=np.float32), device=device)
            times = torch.zeros((len(positions), model.time_dim), dtype=torch.float32, device=device)
            encoded = model.encode_image(features, times).cpu().numpy()
            if not np.isfinite(encoded).all() or not np.allclose(np.linalg.norm(encoded, axis=1), 1, atol=1e-5):
                raise ValueError("T0 produced nonfinite or nonunit image embeddings")
            embeddings[start:start + len(positions)] = encoded
            report_encoding(start + len(positions), len(rows))
    reference_mask = rows.split.eq("train").to_numpy()
    reference_rows = rows.loc[reference_mask].reset_index(drop=True)
    reference_embeddings, reference_years = embeddings[reference_mask], years[reference_mask]
    reference_known = known[reference_mask]
    reference_embedding_count = len(reference_rows)
    if parent_grouping:
        original_positions, parent_indices = parent_reference_layout(reference_rows)
        reference_rows = reference_rows.iloc[original_positions].reset_index(drop=True)
        reference_years, reference_known = reference_years[original_positions], reference_known[original_positions]
    reference_gps = reference_rows[["latitude", "longitude"]].to_numpy(dtype=float)
    top_k = min(TOP_K, len(reference_rows))
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs, metrics, split_provenance = [], [], {}
    selected_gamma, curve = None, None
    for split in EVAL_SPLITS:  # val is necessarily first; test labels never enter selection.
        mask = rows.split.eq(split).to_numpy()
        query_rows = rows.loc[mask].reset_index(drop=True)
        query_years, query_known = years[mask], known[mask]
        print(f"retrieving {split}: queries={len(query_rows)}, train_references={len(reference_rows)}", flush=True)
        if parent_grouping:
            indices, scores = exact_parent_cosine_topk(
                embeddings[mask], reference_embeddings, parent_indices, top_k=top_k,
                query_chunk=query_chunk, device=device, progress_callback=_progress(split + " retrieval"),
            )
        else:
            indices, scores = exact_cosine_topk(embeddings[mask], reference_embeddings, top_k=top_k,
                                                query_chunk=query_chunk, progress_callback=_progress(split + " retrieval"))
        if split == "val":
            selected_gamma, curve = select_gamma_on_validation(
                indices, scores, query_years, reference_years, reference_gps,
                query_rows[["latitude", "longitude"]].to_numpy(), query_rows.split.to_numpy(),
                query_known=query_known, reference_known=reference_known,
            )
            _write_json(output_dir / "val_gamma_search.json", {
                "selection_split": "val", "selection_rows": len(query_rows),
                "criterion": "minimum median_km; exact ties prefer first gamma",
                "gammas": list(DEFAULT_GAMMAS), "selected_gamma": selected_gamma, "curve": curve,
            })
            print(f"selected gamma={selected_gamma} using all {len(query_rows)} available val rows", flush=True)
        for method, gamma in (("visual", 0.0), ("year", selected_gamma)):
            result = rerank_by_date(indices, scores, query_years, reference_years, reference_gps,
                                    gamma=gamma, query_known=query_known, reference_known=reference_known)
            prediction = build_prediction_rows(query_rows, reference_rows, result, method=method, gamma=gamma)
            prediction["error_km"] = _errors_km(query_rows[["latitude", "longitude"]].to_numpy(), result["pred_gps"])
            prediction["query_year"] = query_years
            prediction["neighbor_year"] = reference_years[result["neighbor_indices"]]
            outputs.append(prediction)
            row_metrics = {"split": split, "method": method, "gamma": gamma, **_metrics(prediction.error_km.to_numpy())}
            metrics.append(row_metrics)
            print(json.dumps(row_metrics, sort_keys=True), flush=True)
        split_provenance[split] = {"n": len(query_rows), "ordered_ids_sha256": _ids_hash(query_rows.id),
                                   "known_dates": int(query_known.sum())}
    predictions = pd.concat(outputs, ignore_index=True)
    temporary = output_dir / "predictions.parquet.tmp"
    predictions.to_parquet(temporary, index=False)
    temporary.replace(output_dir / "predictions.parquet")
    input_paths = {"metadata": data_root / "meta/smapshot.parquet", "split": split_csv,
                   "features": data_root / "features" / f"{feature_name}.npy",
                   "feature_ids": data_root / "features" / f"{feature_name}_ids.npy"}
    summary = {
        "experiment": "trained T0 image-reference retrieval with validation-selected year reranking",
        "method_labels": {"year": "T3 (Yang_Zhang)", "visual": "visual-only control"},
        "device": device, "threads": threads, "query_chunk": query_chunk, "encode_batch": encode_batch,
        "top_k_requested": TOP_K, "top_k_effective": top_k, "selected_gamma": selected_gamma,
        "gammas": list(DEFAULT_GAMMAS), "selection_split": "full available val", "test_gamma_search": False,
        "score": "cosine - gamma * min(abs(query_year-reference_year)/50, 1); zero penalty if either date missing",
        "visual_ties": "smaller reference index in feature-cache order; reranked ties keep visual rank",
        "date_policy": {"source": "f019e23:lib/time_feats.py:dates_to_features",
                        "parser": "scalar pandas.to_datetime(errors=coerce, utc=True) for each endpoint",
                        "year": "midpoint.year + (midpoint.dayofyear-1)/366", "valid_midpoint_year_range": [1800, 2030],
                        "both_endpoints_required": True, "reject_reversed_intervals": False},
        "checkpoint": {"path": str(checkpoint), "sha256": _sha256(checkpoint),
                       "run_name": payload["config"].get("run_name"), "epoch": payload["epoch"],
                       "architecture": payload["architecture"], "training_time_config": payload["time_config"],
                       "training_validation_n": len(payload["validation_ids"]),
                       "training_validation_ids_sha256": _ids_hash(payload["validation_ids"])},
        "inputs": {name: {"path": str(path), "bytes": path.stat().st_size} for name, path in input_paths.items()},
        "input_alignment": {"checkpoint_audit_match": True, "full_input_sha256_verification": "required separately from experiment launcher; not checked by this CLI"},
        "id_hash_encoding": "ordered little-endian signed int64 bytes, SHA256",
        "references": {"split": "train", "n": len(reference_rows), "ordered_ids_sha256": _ids_hash(reference_rows.id),
                       "known_dates": int(reference_known.sum())},
        "queries": split_provenance, "audit": data.audit, "metrics": metrics,
        "outputs": {"predictions": "predictions.parquet", "validation_curve": "val_gamma_search.json"},
        "source_sha256": _sha256(__file__), "elapsed_seconds": time.monotonic() - started,
    }
    if parent_grouping:
        summary["parent_grouping"] = {
            "enabled": True, "rule": "parent_id=abs(image_id); negative IDs are train-only variants",
            "score": "maximum cosine over original and variant before unique-parent top-K",
            "tie_policy": "original parent feature-cache order; reranked ties retain visual rank",
            "reference_embedding_count": reference_embedding_count,
            "reference_parent_count": len(reference_rows), "augmentation_audit": augmentation_audit,
            "helper_source_sha256": _sha256(Path(__file__).resolve().parents[3] / "scripts/augmentation/parent_retrieval.py"),
        }
    _write_json(output_dir / "summary.json", summary)
    print(f"wrote predictions.parquet, val_gamma_search.json and summary.json in {output_dir}", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("data-root", "split-csv", "checkpoint", "output-dir"):
        parser.add_argument("--" + argument, required=True, type=Path)
    parser.add_argument("--threads", type=int, default=4, help="CPU Torch/BLAS threads (default: 4)")
    parser.add_argument("--query-chunk", type=int, default=64, help="query rows per exact cosine matrix product")
    parser.add_argument("--encode-batch", type=int, default=2048, help="cached feature rows per frozen T0 projection batch")
    parser.add_argument("--parent-grouping", action="store_true", help="max cosine over original positive ID and its negative-ID variant before top-K")
    parser.add_argument("--device", default="cpu", help="cpu (default), or cuda for parent-grouped retrieval")
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    try:
        run_retrieval(**vars(args))
    except (ValueError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
