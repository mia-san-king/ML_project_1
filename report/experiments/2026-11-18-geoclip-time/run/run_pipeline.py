import hashlib,json,sys,time,traceback
from pathlib import Path
from datetime import datetime,timezone
import torch,yaml
base=Path('/rcp-scratch/students/zhang7/ml-project2')
repo=base/'ML_project_1'
run=base/'runs/2026-11-18-geoclip-time'
sys.path.insert(0,str(repo))
from models.geoclip_time.lib.temporal_training import run_training,run_prediction
manifest=json.loads((run/'input_manifest.json').read_text())
status=run/'pipeline_status.json'
results={}
started=time.monotonic()
def record(state,stage,**extra):
    payload={'state':state,'stage':stage,'updated_utc':datetime.now(timezone.utc).isoformat(),'elapsed_seconds':time.monotonic()-started,'code_commit':manifest['code_commit'],'results':results,**extra}
    temp=status.with_suffix('.tmp')
    temp.write_text(json.dumps(payload,indent=2)+'\n')
    temp.replace(status)
    print(json.dumps({'state':state,'stage':stage,'elapsed_seconds':payload['elapsed_seconds']}),flush=True)
def verify_inputs():
    for entry in manifest['inputs']:
        p=Path(entry['path'])
        before=p.stat()
        h=hashlib.sha256()
        with p.open('rb') as f:
            for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
        after=p.stat()
        if h.hexdigest()!=entry['sha256'] or (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
            raise RuntimeError(f'Input changed since preflight: {p}')
try:
    record('running','verify inputs')
    configs={label:yaml.safe_load((run/f'{label}_full.yaml').read_text()) for label in ('T0_no_time','T1_dual_time')}
    validation_ids=None
    for label,cfg in configs.items():
        verify_inputs()
        record('running',f'{label}: training')
        torch.cuda.reset_peak_memory_stats()
        phase=time.monotonic()
        result=run_training(cfg,device='cuda')
        torch.cuda.synchronize()
        if validation_ids is None: validation_ids=result['validation_ids']
        elif validation_ids!=result['validation_ids']: raise RuntimeError('T0/T1 validation subsets differ')
        results[label]={'checkpoint':str(result['checkpoint']),'training_log':str(result['log']),'best_val_subset_median_km':result['best_val_median_km'],'training_seconds':time.monotonic()-phase,'training_peak_mib':torch.cuda.max_memory_allocated()/2**20}
        del result
        torch.cuda.empty_cache()
        verify_inputs()
        record('running',f'{label}: training complete')
    for label,cfg in configs.items():
        verify_inputs()
        record('running',f'{label}: evaluation')
        phase=time.monotonic()
        pred=run_prediction(results[label]['checkpoint'],device='cuda',splits=('val','test_in','test_city','test_region'),allow_partial_features=True)
        torch.cuda.synchronize()
        results[label].update(predictions=str(pred['predictions']),metrics=str(pred['metrics']),evaluation_seconds=time.monotonic()-phase)
        verify_inputs()
        record('running',f'{label}: evaluation complete')
        torch.cuda.empty_cache()
    record('completed','all training and evaluation complete')
except Exception as exc:
    record('failed','exception',error=str(exc))
    traceback.print_exc()
    raise
