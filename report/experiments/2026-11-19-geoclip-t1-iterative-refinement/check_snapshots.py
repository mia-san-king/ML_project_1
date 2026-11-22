"""Cross-check the paired step checkpoints and produce a compact comparison."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parent
previous = root.parent/'2026-11-19-geoclip-t1-coordinate-refinement-pilot/validation_no_rerank.json'
prior = json.loads(previous.read_text())['matched_pilot']
summary = json.loads((root/'summary.json').read_text())
frames = pd.read_parquet(root/'snapshots.parquet')
assert len(frames)==2*5*2048 and not frames[['id','radius_km','step']].duplicated().any()
base = frames.query('radius_km == 1 and step == 0').set_index('id').sort_index()
assert hashlib.sha256(base.index.to_numpy(dtype='<i8').tobytes()).hexdigest()==prior['id_sha256']==summary['id_sha256']
checks = {}
for radius in (1,5):
    label = str(radius)
    track = {}
    previous_frame = base
    for k in (0,1,8,32,128):
        f = frames.query('radius_km == @radius and step == @k').set_index('id').sort_index()
        pd.testing.assert_index_equal(f.index,base.index)
        np.testing.assert_allclose(f.initial_error_km,base.error_km,atol=1e-7,rtol=0.)
        assert (f.score>=previous_frame.score).all() and (f.accepted>=previous_frame.accepted).all() and f.accepted.max()<=k
        assert f.travel_km.max()<=1.001*radius and np.isfinite(f[['lat','lon','error_km','score']]).all().all()
        st=summary['results'][label][str(k)];np.testing.assert_allclose([f.error_km.median(), f.error_km.mean(),f.travel_km.median()], [st['median_km'],st['mean_km'],st['median_displacement_km']],rtol=0.,atol=1e-8);assert f.error_km.le(25).sum()==st['within25']
        if radius==1 and k in (0,8):
            benchmark=prior['before' if k==0 else 'after'];np.testing.assert_allclose([st['median_km'],st['mean_km']], [benchmark['median_km'],benchmark['mean_km']],rtol=0.,atol=1e-5);assert benchmark['within_25km']==st['within25']
        track[k]=f
        previous_frame=f
    x,y=track[8],track[128]
    diff=y.error_km-x.error_km
    checks[label]={'median_reduction_km_8_to_128':float(x.error_km.median()-y.error_km.median()),'mean_reduction_km_8_to_128':float(-diff.mean()),'closer_8_to_128':int((diff< -1e-8).sum()),'farther_8_to_128':int((diff>1e-8).sum()),'extra_moves_after_8':int((y.accepted>x.accepted).sum()),'extra_moves_after_32':int((y.accepted>track[32].accepted).sum()),'at_radius_at_128':int((y.travel_km>=radius*.999).sum()),'outside_1km_at_128':int((y.travel_km>1.001).sum()),'max_path_independent_distance_km':float(y.travel_km.max())}
report = {'same_ids':True,'last_experiment_8_step_reproduced':True,'counts':checks}
(root/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
for r,rows in summary['results'].items():
 print('radius',r)
 for k,m in rows.items():print(k,*(round(m[n],5) for n in ('median_km','mean_km','acc25_percent','within25','median_displacement_km','median_accepted_moves','median_score_gain','closer','farther')))
