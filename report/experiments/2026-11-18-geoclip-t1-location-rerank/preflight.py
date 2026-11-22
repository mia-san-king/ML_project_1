"""Verify the fixed experiment inputs against the earlier T1 training run."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

run=Path(__file__).resolve().parent
base=run.parents[1]
original=json.loads((base/'runs/2026-11-18-geoclip-time/input_manifest.json').read_text())
checkpoint=base/'data/ckpt/zhang7_T1_dual_time_available_s0_20261118.pt'
items=[*original['inputs'], {'path':str(checkpoint), 'sha256':'9789dc6a862b355e7d35ef03709d4c4f844c15f58bfa2907f9b2244941be7578'}]
for item in items:
    with open(item['path'], 'rb') as f:
        assert hashlib.file_digest(f,'sha256').hexdigest()==item['sha256'], item['path']
meta=pd.read_parquet(base/'data/meta/smapshot.parquet',columns=['id','date_shot_min','date_shot_max'])
splits=pd.read_csv(base/'ML_project_1/data/splits/main.csv.gz',usecols=['id','split'])
ids=np.load(base/'data/features/clip_vitl14_ids.npy'); rows=meta.set_index('id').loc[ids].join(splits.set_index('id'))
q=rows[rows.split.isin(['val','test_in','test_city','test_region'])]
# Upper bound estimate only; exact time features for the run use T1's checkpoint preprocessing.
pairs=q[['date_shot_min','date_shot_max']].astype('string').apply(lambda c: c.str.slice(0,4)).fillna('unknown')
manifest={'source_training_run':original['code_commit'],'inputs':items,'eligible_queries':len(q),'cached_train_refs':int(rows.split.eq('train').sum()),'approx_query_year_interval_groups':len(pairs.drop_duplicates())}
(run/'input_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print({k:v for k,v in manifest.items() if k!='inputs'})
