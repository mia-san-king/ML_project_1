"""Rebuild the exact train-photo shortlist and year scores of a recorded CPU T3 run.

Run with python -m models.t5_reranker.candidates. Each split's NPZ stores
query_ids, parent_ids, visual_scores and base_scores in visual rank order.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import torch

from models.geoclip_time.lib.temporal_data import load_temporal_data
from models.geoclip_time.lib.temporal_retrieval import exact_cosine_topk, rerank_by_date
from models.geoclip_time.lib.temporal_training import load_temporal_checkpoint
from models.geoclip_time.scripts.rerank_references import (
    EVAL_SPLITS, TOP_K, _ids_hash, _progress, _sha256, _validate_checkpoint_data,
    dates_to_retrieval_years,
)
from scripts.augmentation.parent_retrieval import exact_parent_cosine_topk, parent_reference_layout


def year_scores(visual_scores, query_years, candidate_years, gamma):
    """T3 scores for all candidates; invalid/missing years carry zero penalty."""
    q = np.asarray(query_years, dtype=np.float64)[:, None]
    ref = np.asarray(candidate_years, dtype=np.float64)
    gaps = np.zeros(visual_scores.shape, dtype=np.float64)
    np.subtract(q, ref, out=gaps, where=np.isfinite(q) & np.isfinite(ref))
    return visual_scores - gamma * np.minimum(np.abs(gaps) / 50.0, 1.0)


def encode_images(model, features, feature_rows, batch_size):
    """Project cached CLIP rows with T3's frozen CPU T0 image encoder."""
    embeddings = np.empty((len(feature_rows), model.embed_dim), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(feature_rows), batch_size):
            block = slice(start, start + batch_size)
            inputs = torch.from_numpy(features[feature_rows[block]].astype(np.float32, copy=False))
            dates = torch.zeros((len(inputs), model.time_dim), dtype=torch.float32)
            embeddings[block] = model.encode_image(inputs, dates).numpy()
    return embeddings


def _verify_winners(recorded, method, ids, neighbors, gps, scores, gamma):
    frame = recorded.loc[recorded.method.eq(method)].set_index("id", verify_integrity=True)
    if len(frame) != len(ids):
        raise ValueError(f"Unexpected number of {method} predictions")
    frame = frame.loc[ids]
    np.testing.assert_array_equal(frame.neighbor_id, neighbors)
    np.testing.assert_array_equal(frame[["lat_pred", "lon_pred"]], gps)
    np.testing.assert_array_equal(frame.gamma, np.full(len(ids), gamma))
    np.testing.assert_allclose(frame.score, scores, rtol=0, atol=1e-6)


def prepare(*, t3_dir, data_root, split_csv, checkpoint, output_dir):
    """Export verified T3 candidates as one atomic directory; no fitting occurs here."""
    t3_dir, data_root, split_csv, checkpoint, output_dir = (
        Path(p).resolve() for p in (t3_dir, data_root, split_csv, checkpoint, output_dir)
    )
    if output_dir.exists():
        raise FileExistsError(f"Choose a new candidate directory: {output_dir}")
    summary_path = t3_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (summary["device"] != "cpu" or summary["top_k_effective"] != TOP_K
            or summary["test_gamma_search"] or summary["selection_split"] != "full available val"):
        raise ValueError("Expected an exhaustive CPU T3 with 50 candidates and validation-selected gamma")
    checkpoint_hash = _sha256(checkpoint)
    if checkpoint_hash != summary["checkpoint"]["sha256"]:
        raise ValueError("Checkpoint differs from recorded T3")
    gamma = float(summary["selected_gamma"])
    torch.set_num_threads(summary["threads"])
    model, payload = load_temporal_checkpoint(checkpoint, device="cpu")
    model.eval().requires_grad_(False)
    data = load_temporal_data(data_root, split_csv, feature_name=payload["config"]["features"],
                              time_config=payload["time_config"])
    _validate_checkpoint_data(model, payload, data)
    rows = data.table.loc[data.table.split.isin(("train", *EVAL_SPLITS))].reset_index(drop=True)
    years, _ = dates_to_retrieval_years(rows.date_shot_min.to_numpy(), rows.date_shot_max.to_numpy())
    image = encode_images(model, data.features, rows.feat_idx.to_numpy(dtype=np.int64), summary["encode_batch"])
    training = rows.split.eq("train").to_numpy()
    ref = rows.loc[training].reset_index(drop=True)
    ref_image, ref_years = image[training], years[training]
    groups = None
    if summary.get("parent_grouping", {}).get("enabled"):
        originals, groups = parent_reference_layout(ref)
        ref, ref_years = ref.iloc[originals], ref_years[originals]
    ref_ids = ref.id.to_numpy(dtype=np.int64)
    ref_gps = ref[["latitude", "longitude"]].to_numpy(dtype=np.float64)
    if (ref_ids <= 0).any() or _ids_hash(ref_ids) != summary["references"]["ordered_ids_sha256"]:
        raise ValueError("Training photo IDs differ from recorded T3")
    pred_path = t3_dir / summary["outputs"]["predictions"]
    predictions = pd.read_parquet(pred_path, columns=["id", "split", "method", "gamma", "score",
                                                       "neighbor_id", "lat_pred", "lon_pred"])
    feature_name = payload["config"]["features"]
    inputs = {"features": data_root / "features" / f"{feature_name}.npy",
              "feature_ids": data_root / "features" / f"{feature_name}_ids.npy",
              "metadata": data_root / "meta/smapshot.parquet", "split": split_csv}
    manifest = {
        "t3_summary_sha256": _sha256(summary_path), "t3_predictions_sha256": _sha256(pred_path),
        "checkpoint_sha256": checkpoint_hash, "gamma": gamma, "top_k": TOP_K,
        "parent_grouping": groups is not None, "reference_parent_count": len(ref_ids),
        "training_reference_ids_sha256": _ids_hash(ref_ids),
        "input_sha256": {name: _sha256(path) for name, path in inputs.items()},
        "splits": {},
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output_dir.parent) as staging:
        export = Path(staging) / "candidates"
        export.mkdir()
        for split in EVAL_SPLITS:
            use = rows.split.eq(split).to_numpy()
            ids = rows.loc[use, "id"].to_numpy(dtype=np.int64)
            if (ids <= 0).any() or _ids_hash(ids) != summary["queries"][split]["ordered_ids_sha256"]:
                raise ValueError(f"Real query IDs differ from recorded T3 for {split}")
            options = dict(top_k=TOP_K, query_chunk=summary["query_chunk"], progress_callback=_progress(split))
            if groups is None:
                indices, visual = exact_cosine_topk(image[use], ref_image, **options)
            else:
                indices, visual = exact_parent_cosine_topk(image[use], ref_image, groups, device="cpu", **options)
            parents = ref_ids[indices]
            if np.any(np.diff(np.sort(parents, axis=1), axis=1) == 0):
                raise ValueError(f"Repeated parent in shortlist for {split}")
            query_years = years[use]
            base = year_scores(visual, query_years, ref_years[indices], gamma)
            archived = predictions.loc[predictions.split.eq(split)]
            _verify_winners(archived, "visual", ids, parents[:, 0], ref_gps[indices[:, 0]], visual[:, 0], 0.)
            winners = base.argmax(axis=1)
            rank = np.arange(len(ids))
            source = rerank_by_date(indices, visual, query_years, ref_years, ref_gps, gamma=gamma)
            np.testing.assert_array_equal(source["neighbor_indices"], indices[rank, winners])
            np.testing.assert_array_equal(source["scores"], base[rank, winners])
            _verify_winners(archived, "year", ids, parents[rank, winners], source["pred_gps"], source["scores"], gamma)
            dest = export / f"{split}.npz"
            np.savez_compressed(dest, query_ids=ids, parent_ids=parents, visual_scores=visual, base_scores=base)
            manifest["splits"][split] = {"n": len(ids), "sha256": _sha256(dest), "ordered_ids_sha256": _ids_hash(ids)}
        (export / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        export.rename(output_dir)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("t3-dir", "data-root", "split-csv", "checkpoint", "output-dir"):
        parser.add_argument("--" + name, required=True, type=Path)
    print(json.dumps(prepare(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
