"""Fixed local gradient-ascent probe on validation, with no parameter updates."""
from pathlib import Path
import json
import sys
import warnings
import numpy as np
import pandas as pd
import torch
import geopandas as gpd
import shapely

base = Path('/rcp-scratch/students/zhang7/ml-project2')
data = base / 'data'
run = base / 'runs/2026-11-18-geoclip-t1-location-rerank'
sys.path.insert(0, str(run / 'source'))
from models.geoclip_time.lib.temporal_data import encode_date_intervals
from models.geoclip_time.lib.temporal_training import load_temporal_checkpoint

model, ckpt = load_temporal_checkpoint(data/'ckpt/zhang7_T1_dual_time_available_s0_20261118.pt',device='cpu')
model.eval().requires_grad_(False)
torch.set_num_threads(2)
frame = pd.read_parquet(run/'results/predictions.parquet',filters=[('split','=','val'),('method','=','year')]).set_index('id').sort_index()
frame = frame.loc[~frame.index.isin(ckpt['validation_ids'])].sample(2048,random_state=0).sort_index()
meta = pd.read_parquet(data/'meta/smapshot.parquet', columns=['id','date_shot_min','date_shot_max','latitude','longitude']).set_index('id').loc[frame.index]
feat_rows = pd.Index(np.load(data/'features/clip_vitl14_ids.npy')).get_indexer(frame.index); assert (feat_rows>=0).all()
f = torch.tensor(np.load(data/'features/clip_vitl14.npy',mmap_mode='r')[feat_rows],dtype=torch.float32)
times = torch.tensor(encode_date_intervals(meta, **ckpt['time_config'])[0],dtype=torch.float32)
initial = frame[['lat_pred','lon_pred']].to_numpy(dtype='float64'); x0 = torch.tensor(initial,dtype=torch.float64)
# Local north/east in km. Limit excursions to 1 km, where tangent-plane distances approximate arc length closely.
km_per_degree = np.pi/180*6371.0088
scale = torch.stack((torch.full_like(x0[:,0],km_per_degree), km_per_degree*torch.cos(torch.deg2rad(x0[:,0]))),dim=1)
boundary = gpd.read_file(data/'geo/ne_admin0.gpkg').to_crs(4326).geometry.union_all()
shapely.prepare(boundary)
with torch.no_grad():
    v = model.encode_image(f,times)
def parts(offset):
    xy = x0+offset/scale
    loc = model.encode_location_base(xy.float())
    pair = torch.nn.functional.normalize(model.location_time_adapter(loc,times),dim=-1)
    return xy,(v*pair).sum(dim=1)
offset = torch.zeros_like(x0)
with torch.no_grad(): _,initial_scores = parts(offset)
score_error = float(np.abs(initial_scores.numpy()-frame.spatial_cosine.to_numpy()).max()); assert score_error < 0.001
for step_km in (0.25,0.25,0.25,0.25,0.125,0.0625,0.03125,0.015625):
    offset = offset.detach().requires_grad_(True)
    _,scores = parts(offset)
    grad, = torch.autograd.grad(scores.sum(),offset)
    assert torch.isfinite(grad).all()
    with torch.no_grad():
        proposed = offset + step_km*torch.nn.functional.normalize(grad,dim=1)
        proposed = proposed / proposed.norm(dim=1,keepdim=True).clamp(min=1.)
        gps, trial = parts(proposed)
        interior = torch.from_numpy(shapely.intersects_xy(boundary, gps[:,1].numpy(),gps[:,0].numpy()))
        accept = interior & (trial > scores)
        offset = torch.where(accept[:,None],proposed,offset).detach()
with torch.no_grad():
    final_xy, final_scores = parts(offset)
def distance(a,b):
    a,b = np.radians(a), np.radians(b)
    h=np.sin((a[:,0]-b[:,0])/2)**2+np.cos(a[:,0])*np.cos(b[:,0])*np.sin((a[:,1]-b[:,1])/2)**2
    return 12742.0176*np.arcsin(np.sqrt(np.clip(h,0,1)))
truth = meta[['latitude','longitude']].to_numpy(); old,new = distance(truth,initial), distance(truth,final_xy.numpy()); np.testing.assert_allclose(old,frame.error_km.to_numpy(),rtol=0,atol=1e-7)
travel=distance(initial,final_xy.numpy());assert travel.max() < 1.001 and (final_scores >= initial_scores-1e-6).all()
def metrics(x):return {'median_km':float(np.median(x)), 'mean_km':float(x.mean()), 'acc25_percent':float(100*np.mean(x<=25))}
out={'split':'validation sample excluding checkpoint selection subset','n':len(frame),'seed':0,'checkpoint_epoch':ckpt['epoch'],'seed_method':'T1 location retrieval with archive-year reranking','frozen_query_time':True,'trust_radius_km':1,'proposed_steps_km':[.25,.25,.25,.25,.125,.0625,.03125,.015625],'score_reproduction_max_abs_diff':score_error,'before':metrics(old),'after':metrics(new),'closer':int((new<old-1e-8).sum()),'farther':int((new>old+1e-8).sum()),'max_travel_km':float(travel.max()),'median_travel_km':float(np.median(travel)),'increased_similarity':int((final_scores>initial_scores).sum()),'median_cosine_gain':float((final_scores-initial_scores).median())};print(json.dumps(out,indent=2,allow_nan=False))
