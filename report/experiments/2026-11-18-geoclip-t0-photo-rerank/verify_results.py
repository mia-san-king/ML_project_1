"""Independently check retrieval predictions against original SMAPshot metadata."""
from pathlib import Path
import json

import numpy as np
import pandas as pd

run = Path(__file__).resolve().parent
base = run.parents[1]
team = Path('/rcp-scratch/students/yijliu/ml-project2/data/results/preds')
data = base / 'data'
meta = pd.read_parquet(data / 'meta/smapshot.parquet', columns=['id', 'latitude', 'longitude']).set_index('id', verify_integrity=True)
split = pd.read_csv(base / 'ML_project_1/data/splits/main.csv.gz').set_index('id', verify_integrity=True)['split']
cache_ids = np.load(data / 'features/clip_vitl14_ids.npy')
train_refs = set(split.index[split.eq('train')]).intersection(cache_ids)
assert len(train_refs) == 176308
expected = {'val': 29529, 'test_in': 47593, 'test_city': 3377, 'test_region': 13039}

def errs(table):
    truth = meta.loc[table.index].to_numpy()
    guess = table[['lat_pred', 'lon_pred']].to_numpy()
    assert np.isfinite(guess).all()
    a, b = np.radians(truth).T
    c, d = np.radians(guess).T
    half_chord_sq = np.sin((c-a)/2)**2 + np.cos(a)*np.cos(c)*np.sin((d-b)/2)**2
    return 12742.0176 * np.arctan2(np.sqrt(half_chord_sq), np.sqrt(np.maximum(0,1-half_chord_sq)))

def describe(table):
    by_id = table.set_index('id', verify_integrity=True).sort_index()
    by_id['error'] = errs(by_id)
    assert expected == by_id['split'].value_counts().to_dict()
    assert np.array_equal(split.loc[by_id.index], by_id['split'])
    out = {s: {'n': len(x), 'median_km': float(x.error.median()), 'acc25_pct': float(100*x.error.le(25).mean())} for s,x in by_id.groupby('split')}
    return by_id, out

p = pd.read_parquet(run/'results/predictions.parquet')
assert p.method.nunique() == 2
selection = json.loads((run/'results/val_gamma_search.json').read_text())
assert selection['selection_split'] == 'val' and selection['selection_rows'] == expected['val']
gamma = min(selection['curve'], key=lambda item: item['median_km'])['gamma']
assert gamma == selection['selected_gamma'] and p.loc[p.method.eq('year'), 'gamma'].eq(gamma).all()
year_gap = (p.query_year - p.neighbor_year).abs().clip(upper=50).fillna(0) / 50
np.testing.assert_allclose(year_gap, p.year_penalty)
np.testing.assert_allclose(p.visual_score - p.gamma * year_gap, p.score)
stats, full = {}, {}
for method, group in p.groupby('method', sort=False):
    assert set(group.neighbor_id).issubset(train_refs)
    assert np.array_equal(meta.loc[group.neighbor_id].to_numpy(), group[['lat_pred','lon_pred']].to_numpy())
    full[method], stats[method] = describe(group)
    np.testing.assert_allclose(full[method]['error'], full[method].error_km, atol=1e-8)
methods = list(full)
for method, path in {'grid_T0':data/'results/preds/zhang7_T0_no_time_available_s0_20261118.parquet', 'grid_T1':data/'results/preds/zhang7_T1_dual_time_available_s0_20261118.parquet','teammate_CLIP_visual':team/'B2.parquet','teammate_CLIP_year':team/'T0.parquet'}.items():
    full[method], stats[method] = describe(pd.read_parquet(path))
for frame in full.values():
    assert frame.index.equals(full[methods[0]].index)
base_method = next(m for m in methods if p.loc[p.method.eq(m),'gamma'].eq(0).all())
other = next(m for m in methods if m != base_method)
changes = {}
for name in expected:
    a = full[base_method].query('split == @name')
    b = full[other].loc[a.index]
    changed = (a.lat_pred != b.lat_pred) | (a.lon_pred != b.lon_pred)
    changes[name] = {'changed_gps': int(changed.sum()), 'closer': int(b.error.lt(a.error).sum()), 'farther': int(b.error.gt(a.error).sum()), 'changed_reference': int(a.neighbor_id.ne(b.neighbor_id).sum())}
out = {'selected_gamma': gamma, 'cached_training_references':len(train_refs), 'counts': expected, 'metrics': stats, 'paired_changes': changes}
(run/'verified_comparison.json').write_text(json.dumps(out, indent=2, allow_nan=False)+'\n')
print(json.dumps(out, indent=2))
