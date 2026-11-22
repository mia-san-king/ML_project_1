"""Freeze sampling, held-out endpoints and replacement order before sparse generation."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data/local_runs/t4_geo4096"
COUNT = 4096
RESERVE = 384


def digest(a):
    return hashlib.sha256(np.asarray(a, dtype="<i8").tobytes()).hexdigest()


def describe(ids, train):
    counts = train.block.value_counts()
    chosen = train.loc[ids].block.value_counts()
    return {"blocks_with_parents": len(chosen), "largest_allocation": int(chosen.max()),
            "at_or_below_training_block_count_q25": int(chosen.reindex(counts[counts <= counts.quantile(.25)].index, fill_value=0).sum()),
            "min_photos_in_selected_block": int(counts.loc[chosen.index].min())}


def main():
    ids = np.load(ROOT / "data/t1_original/features/clip_vitl14_ids.npy")
    split = pd.read_csv(ROOT / "data/splits/main.csv.gz", usecols=["id", "split", "block"]).set_index("id", verify_integrity=True)
    cached = split.loc[ids]
    train = cached.loc[cached.split.eq("train")].sort_index()
    if len(train) != 176308 or train.block.isna().any() or train.block.nunique() != 325 or train.index.min() <= 0:
        raise ValueError("Unexpected training population")
    random = np.random.default_rng(0).permutation(train.index.to_numpy())[:COUNT]
    rng = np.random.default_rng(0)
    # One photo per active block per round, with no systematic preference in the partial last round.
    queues = [rng.permutation(g.index.to_numpy()) for _,g in train.groupby("block", sort=True)]
    sparse = []
    for i in range(max(map(len, queues))):
        active = [j for j,q in enumerate(queues) if i < len(q)]
        sparse.extend(int(queues[j][i]) for j in rng.permutation(active))
        if len(sparse) >= COUNT + RESERVE:
            break
    sparse = np.asarray(sparse[:COUNT + RESERVE], dtype=np.int64)
    assert len(np.unique(sparse)) == COUNT + RESERVE
    endpoints = {}
    for label in ("val", "test_in", "test_city", "test_region"):
        rows = cached.loc[cached.split.eq(label)].sort_index()
        shared = int(len(set(rows.block.dropna()) & set(train.block)))
        if label in ("val", "test_in") and shared:
            raise ValueError("Spatial block leakage")
        endpoints[label] = {"n":len(rows),"blocks":rows.block.nunique(),"block_ids_shared_with_train":shared,
                            "sorted_query_ids_sha256":digest(rows.index)}
    protocol = {"fixed_utc": datetime.now(timezone.utc).isoformat(), "train_photos":len(train),"train_blocks":325,
        "block_system":"project 10 km LV95 spatial grid", "budget_per_arm":COUNT, "reserves":RESERVE,
        "random":"Uniform permutation of all real cached train IDs sorted ascending, NumPy default_rng(0), first 4096",
        "sparse":"Group the same sorted real training IDs by sorted block, shuffle each block's queue using NumPy default_rng(0), take one from every nonempty block per round in shuffled block order, stop after 4096 accepted",
        "replacement":"Ignore permanent source failures and QA rejects, taking the following predetermined parent until target",
        "generator":"Same existing frozen SDXL img2img recipe, one image per distinct parent, same parent-specific seed, reuse verified features on overlap",
        "trainer":"Official frozen-CLIP/date T4, double-precision pivoted QR with border projection. All 176308 real training images used in both arms.",
        "primary":"Difference in <=25 km hit rate in percentage points, sparse minus random, on all cached eval photos of the existing validation spatial blocks",
        "secondary":"Unweighted mean within-block hit-rate differences; independent test_in spatial blocks; Bern and Ticino regional results are descriptive",
        "interval":"Paired bootstrap of geographic blocks; interpretations conditional on the chosen training sets",
        "train_block_count_q25":float(train.block.value_counts().quantile(.25)),
        "random_ids_sha256":digest(random),"sparse_priority_ids_sha256":digest(sparse),"endpoints":endpoints,
        "planned": {"random":describe(random, train),"sparse":describe(sparse[:COUNT],train),
                    "overlap":len(np.intersect1d(random, sparse[:COUNT]))}}
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "protocol.json").exists():
        existing = json.loads((OUT / "protocol.json").read_text())
        for key in ("random_ids_sha256","sparse_priority_ids_sha256"):
            assert existing[key] == protocol[key]
        return
    np.save(OUT / "sparse_parents.npy", sparse)
    tmp = OUT / "protocol.tmp"
    tmp.write_text(json.dumps(protocol,indent=2)+"\n")
    tmp.replace(OUT / "protocol.json")
    print(json.dumps(protocol["planned"]), flush=True)


if __name__ == "__main__":
    main()
