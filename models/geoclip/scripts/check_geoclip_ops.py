#!/usr/bin/env python3
"""CPU smoke test for models/geoclip/lib/geoclip_ops without images or GeoCLIP weights."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import torch
import torch.nn as nn

from models.baselines.lib.knn import knn_predict
from models.geoclip.lib.geoclip_ops import predict_feats, regional_penalty, train, train_step_loss


class TinyLoc(nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        self.net = nn.Linear(2, dim)

    def forward(self, gps):
        return self.net(gps)


class TinyModel(nn.Module):
    def __init__(self, feat_dim=16, embed_dim=8):
        super().__init__()
        self.image_encoder = nn.Module()
        self.image_encoder.mlp = nn.Linear(feat_dim, embed_dim)
        self.location_encoder = TinyLoc(embed_dim)
        self.logit_scale = nn.Parameter(torch.tensor(0.0))

    def to(self, device):
        super().to(device)
        return self


def main():
    device = "cpu"
    model = TinyModel().to(device)
    gallery = np.array([[46.9, 7.4], [47.0, 8.5], [46.5, 6.6]], dtype=float)
    feats = np.random.randn(4, 16).astype(np.float32)
    penalty = regional_penalty(gallery)
    assert penalty.shape == (3,)
    preds, _, _, lse = predict_feats(model, feats, gallery, penalty=penalty, lam=1.0, top_k=2)
    assert preds.shape == (4, 2)
    assert lse.shape == (4,)

    f_train = np.random.randn(10, 16).astype(np.float32)
    gps_train = np.tile(gallery, (4, 1))[:10]
    pred_knn, sim, _ = knn_predict(feats, f_train, gps_train, device=device)
    assert pred_knn.shape == (4, 2)
    assert sim.shape[0] == 4

    f_tr = torch.randn(64, 16, device=device)
    g_tr = torch.tensor([[46.9, 7.4], [47.0, 8.5]] * 32, device=device)
    neg = torch.tensor([[46.2, 8.0], [47.3, 8.2], [46.8, 7.0]], device=device)
    loss = train_step_loss(model, f_tr[:8], g_tr[:8], neg, soft_tau_km=5.0)
    assert loss.ndim == 0 and torch.isfinite(loss)

    model, score = train(model, f_tr, g_tr, neg, epochs=1, batch_size=8, train_loc=True, soft_tau_km=None)
    assert np.isfinite(score) or score == score  # allow nan if no val_fn
    print("geoclip_ops smoke ok")


if __name__ == "__main__":
    main()