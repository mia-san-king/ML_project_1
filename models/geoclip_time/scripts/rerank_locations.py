#!/usr/bin/env python3
"""Frozen T1 query-time location retrieval with capture-year reranking.

Every training photo supplies one reference GPS, including duplicate GPS.
All evaluation queries share one retrieval pass. Only validation labels select
the capped year penalty; test predictions use that fixed selected gamma.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time


def candidate_spread_summary(indices, reference_gps, *, chunk=4096):
    """Summarize each query's maximum distance from top-1 to its top-K GPS."""
    import numpy as np
    from models.geoclip_time.scripts.rerank_references import _errors_km

    indices, gps = np.asarray(indices), np.asarray(reference_gps)
    if isinstance(chunk, bool) or not isinstance(chunk, int) or chunk < 1:
        raise ValueError("candidate spread chunk must be a positive integer")
    if indices.ndim != 2 or not indices.size or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("candidate indices must be a nonempty integer [N,K] matrix")
    if gps.ndim != 2 or gps.shape[1] != 2 or (indices < 0).any() or (indices >= len(gps)).any():
        raise ValueError("candidate indices must address reference GPS rows")
    radii = np.empty(len(indices), dtype=np.float64)
    for start in range(0, len(indices), chunk):
        block = indices[start:start + chunk]
        anchors = np.repeat(gps[block[:, 0]], block.shape[1], axis=0)
        distances = _errors_km(anchors, gps[block].reshape(-1, 2)).reshape(block.shape)
        radii[start:start + len(block)] = distances.max(axis=1)
    return {"n": len(indices), "candidate_radius_median_km": float(np.median(radii)),
            "candidate_radius_le5km_percent": float(np.mean(radii <= 5) * 100)}


def _validate_checkpoint_data(model, payload, data):
    from models.geoclip_time.scripts.rerank_references import EVAL_SPLITS

    if payload["architecture"].get("mode") != "both" or model.mode != "both":
        raise ValueError("location retrieval requires a trained T1 checkpoint with mode=both")
    if payload["config"].get("model", {}).get("mode") != "both":
        raise ValueError("checkpoint configuration must also specify mode=both")
    if model.embed_dim != 512 or model.time_dim != 3 or data.features.shape[1] != 768:
        raise ValueError("this experiment requires cached 768-D CLIP, 512-D T1 embeddings and 3-D time")
    if payload.get("epoch", 0) < 1 or not payload.get("history"):
        raise ValueError("checkpoint must include completed training history")
    if payload["config"].get("initialization") not in (None, "pretrained"):
        raise ValueError("this experiment requires the pretrained-then-trained T1 checkpoint")
    expected_time_keys = {"year_center", "year_scale", "min_year", "max_year"}
    time_config = payload["time_config"]
    if set(time_config) != expected_time_keys or payload["config"].get("time") != time_config:
        raise ValueError("checkpoint time_config must be complete and match its training time configuration")
    original = payload.get("audit", {})
    if original.get("dates", {}).get("config") != time_config or data.audit["dates"]["config"] != time_config:
        raise ValueError("date preprocessing configuration differs from the T1 checkpoint time_config")
    for key in ("coverage", "feature_rows", "feature_dim", "metadata_rows", "split_rows", "dates"):
        if key not in original or original[key] != data.audit[key]:
            raise ValueError(f"input audit differs from the T1 checkpoint: {key}")
    for split in ("train", *EVAL_SPLITS):
        coverage = data.audit["coverage"].get(split, {})
        if not coverage.get("available", 0):
            raise ValueError(f"no available feature rows for {split}")
        if coverage["missing"] and not payload["config"].get("allow_partial_features", False):
            raise ValueError(f"partial feature coverage was not authorized by T1: {split}")
    validation_ids = payload.get("validation_ids", [])
    available_val = set(data.table.loc[data.table.split.eq("val"), "id"].tolist())
    if not validation_ids or len(set(validation_ids)) != len(validation_ids) or not set(validation_ids) <= available_val:
        raise ValueError("checkpoint validation IDs must be unique and present in the available val split")


def run_retrieval(*, data_root, split_csv, checkpoint, output_dir, threads=4,
                  query_chunk=256, reference_chunk=8192, device="cpu"):
    """Restore T1, retrieve all query-time locations once and evaluate reranking."""
    for key, value in (("threads", threads), ("query_chunk", query_chunk), ("reference_chunk", reference_chunk)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{key} must be a positive integer")
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[variable] = str(threads)
    import numpy as np
    import pandas as pd
    import torch
    from models.geoclip_time.lib.temporal_data import load_temporal_data
    from models.geoclip_time.lib.temporal_training import _FeatureRows, load_temporal_checkpoint
    from models.geoclip_time.lib.temporal_location_retrieval import query_time_location_topk
    from models.geoclip_time.lib.temporal_retrieval import rerank_by_date
    from models.geoclip_time.scripts.rerank_references import (
        DEFAULT_GAMMAS, EVAL_SPLITS, TOP_K, _errors_km, _ids_hash, _metrics,
        _progress, _sha256, _write_json, build_prediction_rows,
        dates_to_retrieval_years, select_gamma_on_validation,
    )

    torch.set_num_threads(threads)
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    started = time.monotonic()
    data_root, split_csv, checkpoint = (Path(path).expanduser().resolve() for path in (data_root, split_csv, checkpoint))
    print(f"T1 location retrieval: device={device}, threads={threads}, query_chunk={query_chunk}, reference_chunk={reference_chunk}", flush=True)
    print(f"loading frozen T1 checkpoint: {checkpoint}", flush=True)
    model, payload = load_temporal_checkpoint(checkpoint, device=device)
    model.eval().requires_grad_(False)
    print("validating feature IDs, coverage and checkpoint-bound T1 date preprocessing", flush=True)
    feature_name = payload["config"]["features"]
    data = load_temporal_data(data_root, split_csv, feature_name=feature_name, time_config=payload["time_config"])
    _validate_checkpoint_data(model, payload, data)
    print("coverage=" + json.dumps(data.audit["coverage"], sort_keys=True), flush=True)
    reference_rows = data.table.loc[data.table.split.eq("train")].copy().reset_index(drop=True)
    query_mask = data.table.split.isin(EVAL_SPLITS).to_numpy()
    query_rows = data.table.loc[query_mask].copy().reset_index(drop=True)
    reference_gps = reference_rows[["latitude", "longitude"]].to_numpy(dtype=np.float64)
    query_times = data.times[query_mask]
    query_features = _FeatureRows(data.features, query_rows.feat_idx.to_numpy(dtype=np.int64))
    print(f"parsing scalar archive years: references={len(reference_rows)}, queries={len(query_rows)}", flush=True)
    reference_years, reference_known = dates_to_retrieval_years(reference_rows.date_shot_min, reference_rows.date_shot_max)
    query_years, query_known = dates_to_retrieval_years(query_rows.date_shot_min, query_rows.date_shot_max)
    unique_times = len(np.unique(query_times, axis=0))
    top_k = min(TOP_K, len(reference_rows))
    print(f"single retrieval pass: queries={len(query_rows)}, train_photos={len(reference_rows)}, unique_query_times={unique_times}, top_k={top_k}", flush=True)
    with torch.inference_mode():
        indices, scores = query_time_location_topk(
            model, query_features, query_times, reference_gps, top_k=top_k,
            query_chunk=query_chunk, reference_chunk=reference_chunk,
            progress_callback=_progress("all queries"), date_progress_callback=_progress("query-time groups"),
        )
    validation = query_rows.split.eq("val").to_numpy()
    selected_gamma, curve = select_gamma_on_validation(
        indices[validation], scores[validation], query_years[validation], reference_years, reference_gps,
        query_rows.loc[validation, ["latitude", "longitude"]].to_numpy(), query_rows.loc[validation, "split"].to_numpy(),
        query_known=query_known[validation], reference_known=reference_known,
    )
    print(f"selected gamma={selected_gamma} using all {int(validation.sum())} available val rows", flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "val_gamma_search.json", {
        "selection_split": "val", "selection_rows": int(validation.sum()),
        "criterion": "minimum median_km; exact ties prefer first gamma",
        "gammas": list(DEFAULT_GAMMAS), "selected_gamma": selected_gamma, "curve": curve,
    })
    outputs, metrics, split_provenance, spread = [], [], {}, {}
    for split in EVAL_SPLITS:
        mask = query_rows.split.eq(split).to_numpy()
        rows = query_rows.loc[mask].reset_index(drop=True)
        candidates, cosines = indices[mask], scores[mask]
        # Diagnostics are computed only after gamma is fixed, for every query.
        spread[split] = candidate_spread_summary(candidates, reference_gps)
        for method, gamma in (("location", 0.0), ("year", selected_gamma)):
            result = rerank_by_date(candidates, cosines, query_years[mask], reference_years, reference_gps,
                                    gamma=gamma, query_known=query_known[mask], reference_known=reference_known)
            prediction = build_prediction_rows(rows, reference_rows, result, method=method, gamma=gamma)
            prediction = prediction.rename(columns={"visual_score": "spatial_cosine"})
            prediction["error_km"] = _errors_km(rows[["latitude", "longitude"]].to_numpy(), result["pred_gps"])
            prediction["query_year"] = query_years[mask]
            prediction["neighbor_year"] = reference_years[result["neighbor_indices"]]
            outputs.append(prediction)
            row_metrics = {"split": split, "method": method, "gamma": gamma, **_metrics(prediction.error_km.to_numpy())}
            metrics.append(row_metrics)
            print(json.dumps(row_metrics, sort_keys=True), flush=True)
        print(f"{split} candidate spread=" + json.dumps(spread[split], sort_keys=True), flush=True)
        split_provenance[split] = {"n": len(rows), "ordered_ids_sha256": _ids_hash(rows.id),
                                   "known_dates": int(query_known[mask].sum()),
                                   "known_T1_times": int((query_times[mask, 2] == 1).sum())}
    predictions = pd.concat(outputs, ignore_index=True)
    temporary = output_dir / "predictions.parquet.tmp"
    predictions.to_parquet(temporary, index=False)
    temporary.replace(output_dir / "predictions.parquet")
    input_paths = {"metadata": data_root / "meta/smapshot.parquet", "split": split_csv,
                   "features": data_root / "features" / f"{feature_name}.npy",
                   "feature_ids": data_root / "features" / f"{feature_name}_ids.npy"}
    summary = {
        "experiment": "trained T1 query-time location retrieval with validation-selected archive-year reranking",
        "device": device, "threads": threads, "query_chunk": query_chunk, "reference_chunk": reference_chunk,
        "top_k_requested": TOP_K, "top_k_effective": top_k, "selected_gamma": selected_gamma,
        "gammas": list(DEFAULT_GAMMAS), "selection_split": "full available val", "test_gamma_search": False,
        "retrieval_passes": 1, "unique_query_times": unique_times,
        "first_stage": "unscaled cosine between frozen T1 image(query features,query time) and location(reference GPS,query time)",
        "score": "spatial_cosine - gamma * min(abs(query_year-reference_year)/50, 1); zero penalty if either archive year missing",
        "spatial_ties": "smaller reference index in feature-cache order; reranked ties keep spatial rank",
        "date_policy": {"source": "f019e23:lib/time_feats.py:dates_to_features (same as T3)",
                        "parser": "scalar pandas.to_datetime(errors=coerce, utc=True) for each endpoint",
                        "year": "midpoint.year + (midpoint.dayofyear-1)/366", "valid_midpoint_year_range": [1800, 2030],
                        "both_endpoints_required": True, "reject_reversed_intervals": False,
                        "first_stage_time": "checkpoint time_config normalized midpoint, half-width, known mask; distinct from archive-year penalty"},
        "checkpoint": {"path": str(checkpoint), "sha256": _sha256(checkpoint),
                       "run_name": payload["config"].get("run_name"), "epoch": payload["epoch"],
                       "architecture": payload["architecture"], "training_time_config": payload["time_config"],
                       "training_validation_n": len(payload["validation_ids"]),
                       "training_validation_ids_sha256": _ids_hash(payload["validation_ids"])},
        "inputs": {name: {"path": str(path), "bytes": path.stat().st_size} for name, path in input_paths.items()},
        "input_alignment": {"checkpoint_audit_match": True, "checkpoint_time_config_match": True,
                            "full_input_sha256_verification": "required separately from experiment launcher; not checked by this CLI"},
        "id_hash_encoding": "ordered little-endian signed int64 bytes, SHA256",
        "references": {"split": "train", "n": len(reference_rows), "ordered_ids_sha256": _ids_hash(reference_rows.id),
                       "known_dates": int(reference_known.sum()), "duplicate_gps_retained": True},
        "queries": split_provenance, "audit": data.audit, "metrics": metrics,
        "candidate_spread": spread,
        "candidate_spread_definition": "maximum Haversine distance from top-1 candidate GPS to any top-K candidate GPS per query; all available queries, evaluated after gamma selection",
        "outputs": {"predictions": "predictions.parquet", "validation_curve": "val_gamma_search.json"},
        "source_sha256": _sha256(__file__), "elapsed_seconds": time.monotonic() - started,
    }
    _write_json(output_dir / "summary.json", summary)
    print(f"wrote predictions.parquet, val_gamma_search.json and summary.json in {output_dir}", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("data-root", "split-csv", "checkpoint", "output-dir"):
        parser.add_argument("--" + argument, required=True, type=Path)
    parser.add_argument("--threads", type=int, default=4, help="CPU Torch/BLAS threads")
    parser.add_argument("--query-chunk", type=int, default=256, help="query rows per cosine matrix product")
    parser.add_argument("--reference-chunk", type=int, default=8192, help="reference rows per location encoding batch")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    try:
        run_retrieval(**vars(args))
    except (ValueError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
