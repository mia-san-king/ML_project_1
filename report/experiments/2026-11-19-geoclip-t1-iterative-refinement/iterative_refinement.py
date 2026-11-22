"""Follow T1's input-coordinate gradient at fixed query time on the paired validation pilot."""
from pathlib import Path
import hashlib
import json
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import torch
import torch.nn.functional as F

RUN = Path(__file__).resolve().parent
BASE = Path('/rcp-scratch/students/zhang7/ml-project2')
DATA = BASE / 'data'
ORIGINAL = BASE / 'runs/2026-11-18-geoclip-t1-location-rerank'
sys.path.insert(0, str(ORIGINAL / 'source'))
from models.geoclip_time.lib.temporal_training import load_temporal_checkpoint
from models.geoclip_time.lib.temporal_data import encode_date_intervals

STEPS = (.25, .25, .25, .25, .125, .0625, .03125, .015625)
CHECK_STEPS = (0, 1, 8, 32, 128)


def geo_distance_km(a, b):
    a, b = np.radians(a), np.radians(b)
    d = b-a
    h = np.sin(d[:, 0]/2)**2 + np.cos(a[:, 0])*np.cos(b[:, 0])*np.sin(d[:, 1]/2)**2
    return 12742.0176*np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def track(radius_km, model, images, times, starting_points, border):
    origin = torch.tensor(starting_points, dtype=torch.float64)
    km_per_degree = np.pi/180*6371.0088
    scale = torch.stack((torch.full_like(origin[:, 0], km_per_degree), km_per_degree*torch.cos(torch.deg2rad(origin[:, 0]))), dim=1)

    def encode_and_score(offset, rows):
        points = origin[rows] + offset / scale[rows]
        locations = model.encode_location_base(points.float())
        conditioned = F.normalize(model.location_time_adapter(locations, times[rows]), dim=-1)
        return points, (images[rows]*conditioned).sum(dim=1)

    n = len(origin)
    offset = torch.zeros_like(origin)
    all_ids = torch.arange(n)
    with torch.no_grad():
        _, best = encode_and_score(offset, all_ids)
    accepted = torch.zeros(n, dtype=torch.int64)
    active = all_ids
    snapshots = {}
    def record(step):
        snapshots[step] = ((origin+offset/scale).numpy().copy(), best.numpy().copy(), accepted.numpy().copy())
        print(f'radius {radius_km} km, step {step}: active {len(active)}/{n}, median accepted moves {np.median(accepted.numpy()):g}',flush=True)

    record(0)
    for k in range(1, max(CHECK_STEPS)+1):
        if (k-1) % len(STEPS) == 0:
            cycle_start_counts = accepted[active].clone()
        if len(active):
            here = offset[active].detach().requires_grad_(True)
            _, score = encode_and_score(here, active)
            grad, = torch.autograd.grad(score.sum(), here)
            assert torch.isfinite(grad).all()
            with torch.no_grad():
                trial = here + STEPS[(k-1) % len(STEPS)]*F.normalize(grad, dim=1)
                trial = trial/(trial.norm(dim=1, keepdim=True)/radius_km).clamp(min=1.)
                gps, trial_score = encode_and_score(trial, active)
                inside = torch.from_numpy(shapely.intersects_xy(border, gps[:, 1].numpy(), gps[:, 0].numpy()))
                use = inside & (trial_score > score) & (trial_score > best[active])
                winners = active[use]
                offset[winners] = trial[use]
                best[winners] = trial_score[use]
                accepted[winners] += 1
        if k % len(STEPS) == 0:
            # Identical successive cycles at a fixed point give identical proposals.
            active = active[accepted[active] != cycle_start_counts]
        if k in CHECK_STEPS:
            record(k)
    return snapshots


def summarize(frame):
    return {'n': len(frame),'median_km': float(frame.error_km.median()), 'mean_km': float(frame.error_km.mean()),
            'within25': int(frame.error_km.le(25).sum()), 'acc25_percent': float(frame.error_km.le(25).mean()*100),
            'median_displacement_km': float(frame.travel_km.median()), 'max_displacement_km': float(frame.travel_km.max()),
            'median_accepted_moves': float(frame.accepted.median()),'mean_accepted_moves': float(frame.accepted.mean()),
            'median_score_gain': float(frame.score_gain.median()), 'score_improved':int(frame.score_gain.gt(0).sum()),
            'closer':int((frame.error_km < frame.initial_error_km-1e-8).sum()),'farther':int((frame.error_km > frame.initial_error_km+1e-8).sum())}


def main():
    torch.set_num_threads(2)
    path = DATA/'ckpt/zhang7_T1_dual_time_available_s0_20261118.pt'
    model, ckpt = load_temporal_checkpoint(path, device='cpu')
    model.eval().requires_grad_(False)
    cols = ['id', 'lat_pred', 'lon_pred', 'gamma', 'error_km', 'spatial_cosine']
    records = pd.read_parquet(ORIGINAL/'results/predictions.parquet', columns=cols, filters=[('split','=','val'),('method','=','location')]).set_index('id').sort_index()
    records = records.loc[~records.index.isin(ckpt['validation_ids'])].sample(2048, random_state=0).sort_index()
    ids = records.index.to_numpy(); digest = hashlib.sha256(ids.astype('<i8').tobytes()).hexdigest()
    assert digest == 'ccc083af685c0a6bc4e0b0e4487f1bae4fde9c682191187273695d387d6dee4c' and records.gamma.eq(0).all()
    meta = pd.read_parquet(DATA/'meta/smapshot.parquet',columns=['id','latitude','longitude','date_shot_min','date_shot_max']).set_index('id').loc[ids]
    times = torch.tensor(encode_date_intervals(meta, **ckpt['time_config'])[0],dtype=torch.float32)
    indices = pd.Index(np.load(DATA/'features/clip_vitl14_ids.npy')).get_indexer(ids)
    assert (indices >= 0).all()
    features = torch.tensor(np.load(DATA/'features/clip_vitl14.npy',mmap_mode='r')[indices],dtype=torch.float32)
    border = gpd.read_file(DATA/'geo/ne_admin0.gpkg').to_crs(4326).geometry.union_all(); shapely.prepare(border)
    initial = records[['lat_pred','lon_pred']].to_numpy(dtype='float64')
    with torch.no_grad():
        images = model.encode_image(features,times)
    outputs, all_metrics = [], {}
    for radius in (1, 5):
        snap = track(radius, model, images, times, initial, border)
        # Labels are used for reporting only, after all coordinates have been selected.
        truth = meta[['latitude','longitude']].to_numpy()
        reference_error = geo_distance_km(initial,truth)
        np.testing.assert_allclose(reference_error,records.error_km,rtol=0,atol=1e-7)
        assert np.abs(snap[0][1]-records.spatial_cosine.to_numpy()).max() < .001
        stats = {}
        previous_score = snap[0][1]
        for step, (location, score, accepted) in snap.items():
            errors = geo_distance_km(location, truth)
            travel = geo_distance_km(location, initial)
            assert (score >= previous_score).all() and travel.max() < radius*1.001 and (accepted<=step).all()
            previous_score = score
            output = pd.DataFrame({'id':ids,'radius_km':radius,'step':step,'lat':location[:,0],'lon':location[:,1],
                         'error_km':errors,'initial_error_km':reference_error,'score':score,'score_gain':score-snap[0][1],
                         'travel_km':travel,'accepted':accepted})
            stats[str(step)] = summarize(output)
            outputs.append(output)
        all_metrics[str(radius)] = stats
    target = RUN/'snapshots.parquet'
    temp = target.with_suffix('.tmp'); pd.concat(outputs,ignore_index=True).to_parquet(temp,index=False);temp.replace(target)
    report = {'sample':'matched validation sample','id_sha256':digest,'checkpoint':path.name,'epoch':ckpt['epoch'], 'known_dates':int(times[:,2].sum()),
              'seed_source':'T1 top spatial score without archive-year reranking','fixed_query_time':True,'steps_km_repeated':STEPS,'recorded_steps':CHECK_STEPS,
              'stopping':'Eight consecutive rejected proposals spanning one full repeated cycle, at unchanged coordinates. Subsequent identical cycles skipped.','results':all_metrics}
    target = RUN/'summary.json'; temp=target.with_suffix('.tmp');temp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');temp.replace(target)


if __name__ == '__main__':
    main()
