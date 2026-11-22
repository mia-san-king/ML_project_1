"""Refine T1's top-1 archive GPS with the query date fixed. Validation only."""
from pathlib import Path
import hashlib
import json
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import torch

BASE = Path('/rcp-scratch/students/zhang7/ml-project2')
DATA = BASE / 'data'
PREVIOUS = BASE / 'runs/2026-11-18-geoclip-t1-location-rerank'
STEPS_KM = (0.25, 0.25, 0.25, 0.25, 0.125, 0.0625, 0.03125, 0.015625)
sys.path.insert(0, str(PREVIOUS / 'source'))
from models.geoclip_time.lib.temporal_data import encode_date_intervals
from models.geoclip_time.lib.temporal_training import load_temporal_checkpoint


def distances(a, b):
    a, b = np.radians(a), np.radians(b)
    delta = b - a
    h = np.sin(delta[:, 0]/2)**2 + np.cos(a[:, 0])*np.cos(b[:, 0])*np.sin(delta[:, 1]/2)**2
    return 12742.0176*np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def refine(model, features, times, seed_latlon, boundary):
    x0 = torch.tensor(seed_latlon, dtype=torch.float64)
    km_per_degree = np.pi/180 * 6371.0088
    scale = torch.stack((torch.full_like(x0[:, 0], km_per_degree), km_per_degree*torch.cos(torch.deg2rad(x0[:, 0]))), dim=1)
    with torch.no_grad():
        image = model.encode_image(features, times)

    def score(offset):
        gps = x0 + offset/scale
        base = model.encode_location_base(gps.float())
        vectors = torch.nn.functional.normalize(model.location_time_adapter(base, times), dim=-1)
        return gps, (image*vectors).sum(dim=1)

    offset = torch.zeros_like(x0)
    with torch.no_grad():
        _, initial_score = score(offset)
    for step_km in STEPS_KM:
        offset = offset.detach().requires_grad_(True)
        _, current = score(offset)
        grad, = torch.autograd.grad(current.sum(), offset)
        assert torch.isfinite(grad).all()
        with torch.no_grad():
            proposal = offset + step_km*torch.nn.functional.normalize(grad, dim=1)
            proposal = proposal / proposal.norm(dim=1, keepdim=True).clamp(min=1.)
            gps, proposed_score = score(proposal)
            inside = torch.from_numpy(shapely.intersects_xy(boundary, gps[:, 1].numpy(), gps[:, 0].numpy()))
            accept = inside & (proposed_score > current)
            offset = torch.where(accept[:, None], proposal, offset).detach()
    with torch.no_grad():
        final_xy, final_score = score(offset)
    assert (final_score >= initial_score - 1e-6).all()
    return final_xy.numpy(), initial_score.numpy(), final_score.numpy()


def metrics(errors):
    return {'median_km': float(np.median(errors)), 'mean_km': float(errors.mean()),
            'within_25km': int((errors <= 25).sum()), 'acc25_percent': float(100*np.mean(errors <= 25))}


def describe(rows):
    before, after = rows.seed_error.to_numpy(), rows.final_error.to_numpy()
    return {'n': len(rows), 'known_query_time_n': int(rows.known_time.sum()),
            'before': metrics(before), 'after': metrics(after),
            'closer': int((after < before-1e-8).sum()), 'farther': int((after > before+1e-8).sum()),
            'unchanged': int((np.abs(after-before) <= 1e-8).sum()),
            'score_improved': int((rows.score_gain > 0).sum()), 'median_score_gain': float(rows.score_gain.median()),
            'median_displacement_km': float(rows.travel.median()), 'maximum_displacement_km': float(rows.travel.max()),
            'id_sha256': hashlib.sha256(np.asarray(rows.index, dtype='<i8').tobytes()).hexdigest()}


torch.set_num_threads(2)
checkpoint = DATA/'ckpt/zhang7_T1_dual_time_available_s0_20261118.pt'
model, ckpt = load_temporal_checkpoint(checkpoint, device='cpu')
model.eval().requires_grad_(False)
# Only the best location from the first stage is read. No reference dates enter refinement.
cols = ['id', 'gamma', 'lat_pred', 'lon_pred', 'error_km', 'spatial_cosine']
frame = pd.read_parquet(PREVIOUS/'results/predictions.parquet', columns=cols,
                        filters=[('split', '=', 'val'), ('method', '=', 'location')]).set_index('id').sort_index()
assert len(frame) == 29529 and frame.index.is_unique and frame.gamma.eq(0).all()
matched = frame.loc[~frame.index.isin(ckpt['validation_ids'])].sample(2048, random_state=0).sort_index()
meta = pd.read_parquet(DATA/'meta/smapshot.parquet', columns=['id', 'latitude', 'longitude', 'date_shot_min', 'date_shot_max']).set_index('id').loc[frame.index]
enc_dates = torch.tensor(encode_date_intervals(meta, **ckpt['time_config'])[0], dtype=torch.float32)
indices = pd.Index(np.load(DATA/'features/clip_vitl14_ids.npy')).get_indexer(frame.index)
assert (indices >= 0).all()
feats = np.load(DATA/'features/clip_vitl14.npy', mmap_mode='r')
boundary = gpd.read_file(DATA/'geo/ne_admin0.gpkg').to_crs(4326).geometry.union_all()
shapely.prepare(boundary)
# Match the previous pilot's row order and batch size exactly on its 2,048 photos.
blocks = [frame.index.get_indexer(matched.index)]
remainder = np.flatnonzero(~frame.index.isin(matched.index))
blocks.extend(remainder[i:i+1024] for i in range(0, len(remainder), 1024))
results = []
max_score_difference = 0.
for i, positions in enumerate(blocks):
    rows = frame.iloc[positions]
    initial = rows[['lat_pred', 'lon_pred']].to_numpy(dtype='float64')
    fs = torch.tensor(feats[indices[positions]], dtype=torch.float32)
    times = enc_dates[positions]
    final, s0, s1 = refine(model, fs, times, initial, boundary)
    truth = meta.iloc[positions][['latitude', 'longitude']].to_numpy()
    e0, e1 = distances(truth, initial), distances(truth, final)
    np.testing.assert_allclose(e0, rows.error_km.to_numpy(), rtol=0, atol=1e-7)
    score_difference = float(np.abs(s0 - rows.spatial_cosine.to_numpy()).max())
    assert score_difference < .001
    max_score_difference = max(max_score_difference, score_difference)
    travel = distances(initial, final)
    assert travel.max() < 1.001
    result = pd.DataFrame({'seed_error': e0, 'final_error': e1, 'score_gain': s1-s0, 'travel': travel,
                           'known_time': times[:,2].numpy().astype(bool)}, index=rows.index)
    results.append(result)
    if i in (0, len(blocks)-1):
        print(f'validation block {i+1}/{len(blocks)}, median {np.median(e0):.4f} -> {np.median(e1):.4f} km',file=sys.stderr,flush=True)
all_rows = pd.concat(results).sort_index()
assert all_rows.index.equals(frame.index)
out = {'checkpoint': checkpoint.name, 'epoch': ckpt['epoch'], 'seed_method': 'T1 spatial top-1 from train photo coordinates; gamma=0',
       'fixed_time': 'query midpoint, interval half-width, known mask from original T1 preprocessing',
       'max_offset_km': 1, 'proposed_steps_km': STEPS_KM, 'scope': 'validation only',
       'max_initial_cosine_difference_vs_previous_run': max_score_difference,
       'matched_pilot': describe(all_rows.loc[matched.index]), 'full_validation': describe(all_rows),
       'full_validation_known_query_time': describe(all_rows.loc[all_rows.known_time])}
print(json.dumps(out, indent=2, allow_nan=False))
