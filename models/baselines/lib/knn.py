"""Image-feature nearest-neighbor baseline."""

import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def knn_predict(f_query, f_train, gps_train, k=1, chunk=4096, device="cuda"):
    train = F.normalize(torch.as_tensor(f_train, device=device).float(), dim=1)
    similarities, indices = [], []
    for start in range(0, len(f_query), chunk):
        query = F.normalize(torch.as_tensor(f_query[start : start + chunk], device=device).float(), dim=1)
        sim, idx = (query @ train.T).topk(k, dim=1)
        similarities.append(sim.cpu())
        indices.append(idx.cpu())
    sim = torch.cat(similarities).numpy()
    idx = torch.cat(indices).numpy()
    gps_train = np.asarray(gps_train, dtype=float)
    return gps_train[idx[:, 0]], sim, idx
