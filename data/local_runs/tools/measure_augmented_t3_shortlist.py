"""Audit the augmented T3 top-50 parent shortlist against saved visual and year winners."""
from pathlib import Path
from experiment_paths import VOLUME
import json
import os
import sys

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[name] = "4"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from models.geoclip_time.lib.temporal_data import load_temporal_data
from models.geoclip_time.lib.temporal_training import load_temporal_checkpoint
from models.geoclip_time.lib.temporal_retrieval import rerank_by_date
from models.geoclip_time.scripts.rerank_references import (
    EVAL_SPLITS, _errors_km, _ids_hash, _progress, _sha256, _validate_checkpoint_data,
    dates_to_retrieval_years,
)
from scripts.augmentation.parent_retrieval import exact_parent_cosine_topk, parent_reference_layout


def main():
    run = ROOT / "data/local_runs/t3_t4_8192/augmented"
    summary = json.loads((run / "T3/summary.json").read_text())
    receipt = json.loads((run / "T0/training.json").read_text())
    torch.set_num_threads(summary["threads"])
    dataset = VOLUME / "aug8192/t1_dataset"
    paths = {"metadata": dataset / "meta/smapshot.parquet", "split": dataset / "splits/main.csv.gz",
             "features": dataset / "features/clip_vitl14.npy", "feature_ids": dataset / "features/clip_vitl14_ids.npy"}
    for path in paths.values():
        assert _sha256(path) == receipt["input_sha256"][path.relative_to(dataset).as_posix()]
    ckpt = run / "T0/ckpt" / (summary["checkpoint"]["run_name"] + ".pt")
    assert _sha256(ckpt) == summary["checkpoint"]["sha256"] == receipt["checkpoint_sha256"]
    assert summary["device"] == "cpu" and summary["parent_grouping"]["enabled"]
    k, gamma = summary["top_k_effective"], summary["selected_gamma"]
    assert (k, gamma) == (50, 1.0)
    model, state = load_temporal_checkpoint(ckpt, device="cpu")
    model.eval().requires_grad_(False)
    data = load_temporal_data(dataset, paths["split"], feature_name=state["config"]["features"],
                              time_config=state["time_config"])
    _validate_checkpoint_data(model, state, data)
    rows = data.table.loc[data.table.split.isin(("train", *EVAL_SPLITS))].reset_index(drop=True)
    encoded = np.empty((len(rows), 512), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(rows), summary["encode_batch"]):
            idx = rows.feat_idx.iloc[start:start+summary["encode_batch"]].to_numpy(dtype=np.int64)
            features = torch.as_tensor(np.asarray(data.features[idx], dtype=np.float32))
            times = torch.zeros((len(idx), model.time_dim), dtype=torch.float32)
            encoded[start:start+len(idx)] = model.encode_image(features, times).numpy()
    years, known = dates_to_retrieval_years(rows.date_shot_min.to_numpy(), rows.date_shot_max.to_numpy())
    train = rows.split.eq("train").to_numpy()
    reference = rows.loc[train].reset_index(drop=True)
    original_pos, parent = parent_reference_layout(reference)
    original = reference.iloc[original_pos]
    parents = original.id.to_numpy()
    assert _ids_hash(parents) == summary["references"]["ordered_ids_sha256"]
    assert (len(reference),len(original)) == (184500,176308)
    gps_columns = ["latitude", "longitude"]
    reference_gps = original[gps_columns].to_numpy(dtype=float)
    reference_encoded = encoded[train]
    reference_years, reference_known = years[train][original_pos], known[train][original_pos]
    saved_path = run / "T3/predictions.parquet"
    saved = pd.read_parquet(saved_path, columns=["id","split","method","neighbor_id","error_km"])
    report = {"experiment": "Local T3 with 8192 diffusion variants, unique-parent top-50 coverage",
              "radius_km":25, "k":k, "gamma":gamma, "reference_embeddings":len(reference),
              "reference_parents":len(original), "checkpoint_sha256":_sha256(ckpt),
              "archived_local_predictions_sha256":_sha256(saved_path), "splits":{}}
    for split in ("test_city","test_region","test_in"):
        mask = rows.split.eq(split).to_numpy()
        query = rows.loc[mask]
        ids = query.id.to_numpy()
        assert _ids_hash(ids) == summary["queries"][split]["ordered_ids_sha256"]
        idx, scores = exact_parent_cosine_topk(
            encoded[mask], reference_encoded, parent, top_k=k, device="cpu",
            query_chunk=summary["query_chunk"], progress_callback=_progress(split))
        assert idx.shape == (len(ids), k) and (np.diff(np.sort(idx,axis=1),axis=1)>0).all()
        rerank = rerank_by_date(idx, scores, years[mask], reference_years, reference_gps,
                                gamma=gamma, query_known=known[mask], reference_known=reference_known)
        truth = query[gps_columns].to_numpy(dtype=float)
        distances = _errors_km(np.repeat(truth, k, axis=0), reference_gps[idx.ravel()]).reshape(-1, k)
        found = (distances <= 25).any(axis=1)
        hits = {}
        for method, winners in (("visual",idx[:,0]), ("year",rerank["neighbor_indices"])):
            archive = saved[saved.split.eq(split) & saved.method.eq(method)].set_index("id",verify_integrity=True)
            assert len(archive) == len(ids)
            np.testing.assert_array_equal(archive.loc[ids, "neighbor_id"], parents[winners])
            errors = _errors_km(truth, reference_gps[winners])
            np.testing.assert_allclose(archive.loc[ids, "error_km"], errors, atol=1e-7, rtol=0)
            hits[method] = errors <= 25
            assert not (hits[method] & ~found).any()
        stats = {"n":len(ids), "top50_hits":int(found.sum()), "top50_hit_pct":100*found.mean(),
                 "visual_top1_hits":int(hits["visual"].sum()), "year_top1_hits":int(hits["year"].sum()),
                 "year_top1_hit_pct":100*hits["year"].mean(), "no_top50_hit":int((~found).sum()),
                 "top50_hit_year_miss":int((found & ~hits["year"]).sum()),
                 "year_winner_hit_pct_given_top50_hit":100*hits["year"].sum()/found.sum(),
                 "candidate_list_unique_parents":True,"both_winner_ids_and_distances_verified":True}
        report["splits"][split] = stats
        print(split, json.dumps(stats), flush=True)
    output = ROOT / "data/local_runs/t3_refine_analysis/top50_aug.json"
    output.parent.mkdir(exist_ok=True, parents=True)
    temp = output.with_suffix(".tmp")
    temp.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    temp.replace(output)


if __name__ == "__main__":
    main()
