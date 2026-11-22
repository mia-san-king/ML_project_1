"""Paired spatial block bootstrap for the fixed gamma, evaluated on test_in."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

run_dir = Path(__file__).resolve().parent
split_path = run_dir.parents[1] / 'ML_project_1_time/data/splits/main.csv.gz'
assert hashlib.sha256(split_path.read_bytes()).hexdigest() == '67d2a56bcad5ba826d73a63174315779827456fd9011e40583f743021e72a0a3'
pred = pd.read_parquet(run_dir / 'results/predictions.parquet')
test = pred.loc[pred['split'].eq('test_in')]
wide = test.pivot(index='id', columns='method', values='error_km')[['location','year']]
assert wide.shape == (47593, 2) and np.isfinite(wide).all().all()
splits = pd.read_csv(split_path, usecols=['id','block','split']).set_index('id').loc[wide.index]
assert splits['block'].notna().all() and splits['split'].eq('test_in').all()
blocks, inverse = np.unique(splits['block'].to_numpy(), return_inverse=True)
groups = [np.flatnonzero(inverse == j) for j in range(len(blocks))]
data = wide.to_numpy()

def changes(rows):
    return np.array([np.median(rows[:, 1]) - np.median(rows[:, 0]), 100*((rows[:,1] <= 25).mean() - (rows[:, 0] <= 25).mean())])

rng = np.random.default_rng(0)
samples=[]
for _ in range(1000):
    chosen = rng.integers(len(groups), size=len(groups))
    samples.append(changes(data[np.concatenate([groups[j] for j in chosen])]))
ci = np.percentile(samples, [2.5,97.5], axis=0).T
out={'split':'test_in', 'n':len(wide), 'spatial_blocks':len(blocks), 'replicates':len(samples), 'seed':0,'direction':'year minus location','delta_median_km':changes(data)[0], 'delta_median_km_paired_block_ci95':ci[0].tolist(), 'delta_acc25_percentage_points':changes(data)[1], 'delta_acc25_pp_paired_block_ci95':ci[1].tolist()}
(run_dir/'paired_test_in.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
