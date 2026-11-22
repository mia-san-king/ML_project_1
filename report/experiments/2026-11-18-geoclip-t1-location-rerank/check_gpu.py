from pathlib import Path
import numpy as np
import torch
from models.geoclip_time.lib.temporal_location_retrieval import query_time_location_topk
from models.geoclip_time.lib.temporal_training import load_temporal_checkpoint

checkpoint = Path('/rcp-scratch/students/zhang7/ml-project2/data/ckpt/zhang7_T1_dual_time_available_s0_20261118.pt')
model, _ = load_temporal_checkpoint(checkpoint, device='cuda')
rng = np.random.default_rng(827)
f = rng.normal(size=(16, 768)).astype('float32')
gps = np.column_stack((rng.uniform(46, 47.8, 128), rng.uniform(6.2, 9.8, 128))).astype('float32')
t = np.array([[-.5,0,1],[0,0,0],[.2,.06,1],[.5,0,1]], dtype='float32').repeat(4, axis=0)
with torch.inference_mode():
    expected = (model.score_candidates(*(torch.as_tensor(a, device='cuda') for a in (f,t,gps))) / model.logit_scale.exp()).cpu().numpy()
indices, scores = query_time_location_topk(model, f, t, gps, top_k=50, query_chunk=4, reference_chunk=32)
np.testing.assert_allclose(scores,np.take_along_axis(expected,indices,axis=1),atol=4e-5,rtol=0)
# A small numerical tolerance covers backend rounding at the K-boundary.
np.testing.assert_allclose(scores, np.sort(expected,axis=1)[:, ::-1][:,:50], atol=4e-5,rtol=0)
print('GPU native T1 top-50 scores match:',torch.cuda.get_device_name(0),'max abs diff',float(np.abs(scores-np.take_along_axis(expected,indices,axis=1)).max()),flush=True)
