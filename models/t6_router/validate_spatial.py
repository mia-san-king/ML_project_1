"""Evaluate a two-distance T4/T5 selection rule on validation blocks."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import sklearn

from models.clip_regression.scripts.compare_t1 import ERROR_ATOL_KM, _errors
from models.t5_reranker.evaluate import GPS, PREDICTED, original_rows
from models.t6_router.spatial_validation import evaluate
from scripts.confidence.evaluate import FEATURES, sha256


ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "report/experiments/2026-11-22-local-models/manifests/assets.tsv.gz"
CANDIDATES = "data/local_runs/t5_aug8192/candidates/val.npz"
CONFIDENCE = "data/local_runs/confidence_models"


def shortlist_spread(points):
    """Median great-circle distance over all unordered pairs, per shortlist."""
    if points.ndim != 3 or points.shape[1] < 2 or points.shape[2] != 2:
        raise ValueError("Expected a batch of shortlists of latitude/longitude pairs")
    left, right = np.triu_indices(points.shape[1], k=1)
    km = _errors(points[:, left].reshape(-1, 2), points[:, right].reshape(-1, 2))
    return np.median(km.reshape(len(points), len(left)), axis=1)


def candidate_spreads(path, ids, train, winner_gps):
    with np.load(path, allow_pickle=False) as cache:
        queries, parents = cache["query_ids"], cache["parent_ids"]
    if (queries.shape != (len(ids),) or queries.dtype.kind not in "iu"
            or parents.shape != (len(ids), 50) or parents.dtype.kind not in "iu"):
        raise ValueError("Expected 50 integer parent IDs per validation photo")
    order = np.argsort(queries)
    if not np.array_equal(queries[order], ids):
        raise ValueError("Shortlist query IDs differ from the real validation IDs")
    coords = train[list(GPS)].to_numpy(dtype=float)
    spread = np.empty(len(ids))
    for start in range(0, len(ids), 256):
        batch = slice(start, start + 256)
        candidate_ids = parents[order[batch]]
        if (np.diff(np.sort(candidate_ids, axis=1), axis=1) == 0).any():
            raise ValueError("Repeated original training photo within a shortlist")
        indices = train.index.get_indexer(candidate_ids.ravel())
        if (indices < 0).any():
            raise ValueError("A candidate is outside the original training gallery")
        points = coords[indices.reshape(candidate_ids.shape)]
        is_winner = (points == winner_gps[batch, None, :]).all(axis=2)
        if not is_winner.any(axis=1).all():
            raise ValueError("A frozen T5 winner is outside its shortlist")
        spread[batch] = shortlist_spread(points)
    return spread


def load_expert_validation(workspace, *, extra_paths=()):
    """Return frozen predictions, geography, and raw inputs for cross-fitting their confidence."""
    files = [f"{CONFIDENCE}/inputs.parquet", f"{CONFIDENCE}/predictions.parquet",
             "data/t1_original/meta/smapshot.parquet", "data/splits/main.csv.gz",
             "data/t1_original/features/clip_vitl14_ids.npy", *extra_paths]
    archive = pd.read_csv(ARCHIVE, sep="\t").query("root == 'workspace'").set_index("path")
    hashes = {}
    for rel in files:
        hashes[rel] = sha256(workspace / rel)
        if hashes[rel] != archive.at[rel, "sha256"]:
            raise ValueError(f"Input differs from the frozen experiment: {rel}")

    truth, train, _ = original_rows(workspace / "data/t1_original", workspace / "data/splits/main.csv.gz")
    truth = truth.loc[truth.split.eq("val")]
    if len(truth) != 29529 or truth.block.nunique() != 47:
        raise ValueError("Unexpected validation coverage")
    frame = truth[["split", "block"]].copy()
    coords = {}
    for model in ("T4", "T5"):
        filters = [("split", "=", "val"), ("model", "=", model)]
        confidence_dir = workspace / CONFIDENCE
        features = pd.read_parquet(confidence_dir / "inputs.parquet", filters=filters,
                                   columns=["id", "block", *FEATURES[model]]).set_index("id", verify_integrity=True).sort_index()
        preds = pd.read_parquet(confidence_dir / "predictions.parquet", filters=filters,
                                columns=["id", "block", *PREDICTED, "error_km", "p_within_25km"]
                                ).set_index("id", verify_integrity=True).sort_index()
        for source in (features, preds):
            if not source.index.equals(truth.index) or not source.block.eq(truth.block).all():
                raise ValueError(f"{model}: validation IDs or blocks differ")
        coords[model] = preds[list(PREDICTED)].to_numpy(dtype=float)
        frame[[f"{model}_{col}" for col in PREDICTED]] = coords[model]
        errors = _errors(truth[list(GPS)].to_numpy(dtype=float), coords[model])
        np.testing.assert_allclose(errors, preds.error_km, atol=ERROR_ATOL_KM, rtol=0)
        frame[FEATURES[model]] = features[FEATURES[model]]
        frame[f"{model}_error_km"] = errors
        frame[f"{model}_p25"] = preds.p_within_25km
    return frame.reset_index(), train, coords, hashes


def load_validation(workspace):
    frame, train, coords, hashes = load_expert_validation(workspace, extra_paths=(CANDIDATES,))
    gaps = _errors(coords["T4"], coords["T5"])
    spreads = candidate_spreads(workspace / CANDIDATES, frame.id.to_numpy(), train, coords["T5"])
    return frame, gaps, spreads, hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=ROOT, help="workspace containing the archived Windows inputs")
    parser.add_argument("--output-dir", type=Path, required=True, help="new directory for the report and per-photo decisions")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        parser.error(f"Output already exists: {output}")
    frame, gaps, spreads, hashes = load_validation(args.workspace)
    summary, decisions = evaluate(frame, gaps, spreads)
    summary.update({
        "input_sha256": hashes, "sklearn_version": sklearn.__version__,
        "distance_km": "Great-circle distance between the frozen T4 and T5 predictions.",
        "spread_km": "Median of the 1,225 pairwise great-circle distances between 50 distinct real parent photos. Co-located parents retain their zero distance.",
        "scope": "Validation only. Expert weights and T5 reranking weight are fixed; T5 previously used this validation set to select its weight.",
    })
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temp:
        stage = Path(temp) / "results"
        stage.mkdir()
        decisions.to_parquet(stage / "decisions.parquet", index=False)
        summary["decisions_sha256"] = sha256(stage / "decisions.parquet")
        (stage / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        stage.rename(output)
    print(f"RESULT {len(decisions)} validation decisions written to {output}")
    print(json.dumps(summary["comparisons"], indent=2))


if __name__ == "__main__":
    main()
