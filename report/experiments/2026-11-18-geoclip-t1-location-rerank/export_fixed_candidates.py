"""Reproduce the original retrieval pass and retain the fixed test_in candidates."""
from pathlib import Path
import sys
import os
import subprocess
import numpy as np
import pandas as pd
import torch

run=Path(__file__).resolve().parent
sys.path.insert(0,str(run/'source'))
from models.geoclip_time.lib.temporal_data import load_temporal_data
from models.geoclip_time.lib.temporal_training import load_temporal_checkpoint, _FeatureRows
from models.geoclip_time.lib.temporal_location_retrieval import query_time_location_topk
from models.geoclip_time.scripts.rerank_references import EVAL_SPLITS, dates_to_retrieval_years, _progress

base=run.parents[1]
device = 'cpu'
usage = subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used','--format=csv,noheader,nounits'],text=True)
for item in usage.splitlines():
    gpu, uuid, mib = (field.strip() for field in item.split(','))
    if gpu in ('0','1','2') and int(mib) < 32:
        os.environ['CUDA_VISIBLE_DEVICES']=uuid
        device='cuda'
        break
torch.set_num_threads(6 if device=='cpu' else 2)
print('Fixed candidate diagnostic device',device,'threads',torch.get_num_threads(),flush=True)
model, payload=load_temporal_checkpoint(base/'data/ckpt/zhang7_T1_dual_time_available_s0_20261118.pt',device=device)
model.eval().requires_grad_(False)
data=load_temporal_data(base/'data',base/'ML_project_1/data/splits/main.csv.gz',feature_name=payload['config']['features'],time_config=payload['time_config'])
refs=data.table.loc[data.table.split.eq('train')]
qmask=data.table.split.isin(EVAL_SPLITS).to_numpy()
queries=data.table.loc[qmask]
test=queries.split.eq('test_in').to_numpy()
def gps(frame):
    return frame[['latitude','longitude']].to_numpy(dtype=np.float64)
idx, scores=query_time_location_topk(model,_FeatureRows(data.features,queries.feat_idx.to_numpy(dtype=np.int64)),data.times[qmask],gps(refs),top_k=50,query_chunk=128,reference_chunk=8192,date_progress_callback=_progress('date groups'))
def archive_years(df):
    pairs=pd.MultiIndex.from_frame(df[['date_shot_min','date_shot_max']]); codes, unique=pairs.factorize(sort=False)
    assert (codes>=0).all()
    years,_=dates_to_retrieval_years(unique.get_level_values(0),unique.get_level_values(1))
    return years[codes]
q=queries.loc[test]
np.savez(run/'candidate_diagnostic_tmp.npz',device=np.asarray(device),indices=idx[test].astype('int32'), scores=scores[test],query_ids=q.id.to_numpy(), reference_ids=refs.id.to_numpy(),query_years=archive_years(q),reference_years=archive_years(refs),query_gps=gps(q),reference_gps=gps(refs))
(run/'candidate_diagnostic_tmp.npz').rename(run/'candidate_diagnostic.npz')
print('Exported',len(q),'test_in x',idx.shape[1], 'candidates',flush=True)
