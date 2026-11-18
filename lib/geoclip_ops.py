import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.geo import R_EARTH, haversine_km, haversine_torch

CH_CENTER = (46.80, 8.23)


def regional_penalty(gallery, center=CH_CENTER, radius_km=179.0, scale_km=50.0):
    gallery = np.asarray(gallery, dtype=float)
    dist = haversine_km(gallery[:, 0], gallery[:, 1], center[0], center[1])
    return np.clip((dist - radius_km) / scale_km, 0, None) ** 2


@torch.no_grad()
def embed_gallery(model, gallery, batch_size=20_000):
    device = model.logit_scale.device
    gallery_t = torch.as_tensor(gallery, dtype=torch.float32, device=device)
    chunks = []
    for start in range(0, len(gallery_t), batch_size):
        batch = gallery_t[start : start + batch_size]
        chunks.append(F.normalize(model.location_encoder(batch), dim=1))
    return torch.cat(chunks, dim=0)


@torch.no_grad()
def predict_feats(model, feats, gallery, penalty=None, lam=0.0, top_k=1, chunk=1024):
    """Return top-1 GPS, top indices, top logits, logsumexp per row."""
    device = model.logit_scale.device
    gallery = np.asarray(gallery, dtype=float)
    gallery_emb = embed_gallery(model, gallery)
    penalty_t = None
    if penalty is not None:
        penalty_t = lam * torch.as_tensor(penalty, device=device, dtype=torch.float32)
    scale = model.logit_scale.exp()
    top_indices, top_logits, logsumexp = [], [], []
    for start in range(0, len(feats), chunk):
        batch = torch.as_tensor(feats[start : start + chunk], device=device).float()
        image_emb = F.normalize(model.image_encoder.mlp(batch), dim=1)
        logits = scale * image_emb @ gallery_emb.T
        if penalty_t is not None:
            logits = logits - penalty_t
        values, indices = logits.topk(top_k, dim=1)
        top_indices.append(indices.cpu())
        top_logits.append(values.cpu())
        logsumexp.append(torch.logsumexp(logits, dim=1).cpu())
    top_indices = torch.cat(top_indices).numpy()
    top_logits = torch.cat(top_logits).numpy()
    logsumexp = torch.cat(logsumexp).numpy()
    predictions = gallery[top_indices[:, 0]]
    return predictions, top_indices, top_logits, logsumexp


@torch.no_grad()
def predict_mlp512(
    model,
    mlp_feats,
    gallery,
    gallery_emb=None,
    penalty=None,
    lam=0.0,
    top_k=20,
    chunk=1024,
):
    """mlp_feats: (N,512) already L2-normalized or raw (will normalize)."""
    device = model.logit_scale.device
    gallery = np.asarray(gallery, dtype=float)
    if gallery_emb is None:
        gallery_emb = embed_gallery(model, gallery)
    else:
        gallery_emb = gallery_emb.to(device)
    penalty_t = None
    if penalty is not None:
        penalty_t = lam * torch.as_tensor(penalty, device=device, dtype=torch.float32)
    scale = model.logit_scale.exp()
    top_indices, top_logits, logsumexp = [], [], []
    for start in range(0, len(mlp_feats), chunk):
        batch = torch.as_tensor(mlp_feats[start : start + chunk], device=device).float()
        image_emb = F.normalize(batch, dim=1) if batch.shape[1] == 512 else F.normalize(
            model.image_encoder.mlp(batch), dim=1
        )
        logits = scale * image_emb @ gallery_emb.T
        if penalty_t is not None:
            logits = logits - penalty_t
        values, indices = logits.topk(top_k, dim=1)
        top_indices.append(indices.cpu())
        top_logits.append(values.cpu())
        logsumexp.append(torch.logsumexp(logits, dim=1).cpu())
    top_indices = torch.cat(top_indices).numpy()
    top_logits = torch.cat(top_logits).numpy()
    logsumexp = torch.cat(logsumexp).numpy()
    predictions = gallery[top_indices[:, 0]]
    return predictions, top_indices, top_logits, logsumexp


@torch.no_grad()
def compute_mlp512(model, clip768, chunk=2048):
    device = model.logit_scale.device
    mlp = model.image_encoder.mlp
    parts = []
    for start in range(0, len(clip768), chunk):
        batch = torch.as_tensor(np.asarray(clip768[start : start + chunk]), device=device).float()
        parts.append(F.normalize(mlp(batch), dim=1).half().cpu())
    return torch.cat(parts, dim=0).numpy()


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


def jitter(gps, sigma_km):
    lat = gps[:, 0:1]
    noise = torch.randn_like(gps) * sigma_km
    lat_scale = 111.2
    lon_scale = 111.2 * torch.cos(torch.deg2rad(lat)).clamp(min=0.05)
    delta = torch.cat([noise[:, :1] / lat_scale, noise[:, 1:] / lon_scale], dim=1)
    return gps + delta


def train_step_loss(model, image_feats, gps_batch, neg_pool, *, soft_tau_km=None, sigma_km=1.0, n_neg=256):
    """One optimization step on a single batch (for tests and training loop)."""
    device = gps_batch.device
    mlp = model.image_encoder.mlp
    loc = model.location_encoder
    neg = neg_pool[torch.randint(len(neg_pool), (n_neg,), device=device)]
    locations = torch.cat([jitter(gps_batch, sigma_km), neg])
    image_emb = F.normalize(mlp(image_feats.float()), dim=1)
    loc_emb = F.normalize(loc(locations), dim=1)
    logits = model.logit_scale.exp() * image_emb @ loc_emb.T
    if soft_tau_km is None:
        targets = torch.arange(len(gps_batch), device=device)
        return F.cross_entropy(logits, targets)
    weights = torch.softmax(-haversine_torch(gps_batch, locations) / soft_tau_km, dim=1)
    return -(weights * F.log_softmax(logits, dim=1)).sum(1).mean()


def train(
    model,
    f_tr,
    g_tr,
    neg_pool,
    *,
    train_loc=True,
    soft_tau_km=None,
    epochs=1,
    batch_size=32,
    n_neg=256,
    sigma_km=1.0,
    lr_mlp=1e-4,
    lr_loc=3e-5,
    val_fn=None,
    seed=0,
):
    torch.manual_seed(seed)
    mlp = model.image_encoder.mlp
    loc = model.location_encoder
    groups = [{"params": mlp.parameters(), "lr": lr_mlp}, {"params": [model.logit_scale], "lr": lr_mlp}]
    if train_loc:
        groups.append({"params": loc.parameters(), "lr": lr_loc})
    else:
        for param in loc.parameters():
            param.requires_grad_(False)
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
    best = float("inf")
    best_state = None
    n = len(f_tr)
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, device=f_tr.device)
        last_loss = None
        for start in range(0, n - batch_size + 1, batch_size):
            idx = perm[start : start + batch_size]
            loss = train_step_loss(
                model,
                f_tr[idx],
                g_tr[idx],
                neg_pool,
                soft_tau_km=soft_tau_km,
                sigma_km=sigma_km,
                n_neg=n_neg,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                model.logit_scale.clamp_(0, np.log(100))
            last_loss = float(loss.item())
        if val_fn is not None:
            model.eval()
            val_score = val_fn(model)
            if val_score < best:
                best = val_score
                best_state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
            print(f"epoch {epoch} loss={last_loss:.4f} val={val_score:.4f}", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
        return model, best
    return model, last_loss if last_loss is not None else float("nan")