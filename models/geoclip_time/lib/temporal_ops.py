"""Temporal contrastive training and bounded-memory gallery retrieval."""

import math
from numbers import Integral

import numpy as np
import torch
from torch.nn import functional as F

from models.geoclip_time.lib.temporal_model import _matrix


def _distances_km(queries, candidates):
    """Pairwise great-circle distances without importing dataset path helpers."""
    query = torch.deg2rad(queries)
    candidate = torch.deg2rad(candidates)
    delta = query[:, None, :] - candidate[None, :, :]
    a = (
        torch.sin(delta[..., 0] / 2).square()
        + torch.cos(query[:, None, 0]) * torch.cos(candidate[None, :, 0])
        * torch.sin(delta[..., 1] / 2).square()
    )
    return 2 * 6371.0 * torch.asin(a.clamp(0, 1).sqrt())


def temporal_loss(model, feats, gps, times, negative_gps, *, soft_tau_km=None):
    """Single-direction image-to-GPS objective with shared query times.

    Candidates are ``[gps; negative_gps]``. The hard objective maximizes the
    total probability of all coordinates exactly equal to the query GPS,
    including duplicates in the negative pool. With ``soft_tau_km``, use
    distance-softmax targets and cross entropy instead.
    """
    _matrix(feats, "feats")
    _matrix(gps, "gps", width=2, rows=len(feats))
    _matrix(times, "times", width=model.time_dim, rows=len(feats))
    _matrix(negative_gps, "negative_gps", width=2, allow_empty=True)
    if soft_tau_km is not None and (not math.isfinite(soft_tau_km) or soft_tau_km <= 0):
        raise ValueError("soft_tau_km must be finite and positive")
    candidates = torch.cat([gps, negative_gps], dim=0)
    logits = model.score_candidates(feats, times, candidates)
    if soft_tau_km is not None:
        weights = torch.softmax(-_distances_km(gps, candidates) / soft_tau_km, dim=1)
        return -(weights * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
    positive = (gps[:, None, :] == candidates[None, :, :]).all(dim=-1)
    positive_logits = logits.masked_fill(~positive, -torch.inf)
    return (logits.logsumexp(dim=1) - positive_logits.logsumexp(dim=1)).mean()


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _batch_tensor(values, device, dtype):
    if isinstance(values, torch.Tensor):
        return values.to(device=device, dtype=dtype)
    # Copy the selected slice so read-only NumPy memmaps remain safe and do not
    # produce PyTorch's non-writable-array warning.
    return torch.tensor(np.asarray(values), device=device, dtype=dtype)


def _array_shape(values, name, *, width=None, rows=None):
    shape = getattr(values, "shape", ())
    if len(shape) != 2 or shape[0] == 0 or shape[1] == 0:
        raise ValueError(f"{name} must be a nonempty two-dimensional array")
    if width is not None and shape[1] != width:
        raise ValueError(f"{name} must have {width} columns")
    if rows is not None and shape[0] != rows:
        raise ValueError(f"{name} must have {rows} rows")


def _rank(scores, indices, k):
    # Stable descending sort resolves equal scores by gallery index. Both the
    # retained prefix and each incoming chunk already use ascending indices
    # for ties, so merging in gallery order preserves that rule across chunks.
    order = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :k]
    return scores.gather(1, order), indices.gather(1, order)


@torch.no_grad()
def predict_temporal(
    model, feats, times, gallery, *, top_k=1, query_chunk=16, gallery_chunk=256,
):
    """Return NumPy ``(pred_gps, top_indices, top_logits, logsumexp)``.

    Every GPS base is encoded exactly once per call and cached on CPU. Only
    ``query_chunk * gallery_chunk`` pairs are conditioned at a time. Top-k and
    logsumexp are merged across gallery chunks; ties favor the lower gallery
    index. Evaluation mode is temporary, preserving each module's prior mode.
    Empty queries/gallery and invalid chunk sizes/top-k raise ``ValueError``.
    """
    for name, value in (("top_k", top_k), ("query_chunk", query_chunk), ("gallery_chunk", gallery_chunk)):
        _positive_integer(value, name)
    _array_shape(feats, "feats")
    _array_shape(times, "times", width=model.time_dim, rows=len(feats))
    _array_shape(gallery, "gallery", width=2)
    if top_k > len(gallery):
        raise ValueError("top_k must not exceed the gallery size")
    device = model.logit_scale.device
    dtype = model.logit_scale.dtype
    gallery_np = (
        gallery.detach().cpu().numpy() if isinstance(gallery, torch.Tensor)
        else np.asarray(gallery)
    )
    previous_modes = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        base_cache = torch.empty((len(gallery), model.embed_dim), dtype=dtype)
        for start in range(0, len(gallery), gallery_chunk):
            coordinates = _batch_tensor(gallery[start : start + gallery_chunk], device, dtype)
            base = model.encode_location_base(coordinates)
            base_cache[start : start + len(base)] = base.cpu()

        all_indices, all_logits, all_normalizers = [], [], []
        for start in range(0, len(feats), query_chunk):
            image_input = _batch_tensor(feats[start : start + query_chunk], device, dtype)
            query_times = _batch_tensor(times[start : start + query_chunk], device, dtype)
            image = model.encode_image(image_input, query_times)
            batch_size = len(image)
            best_scores = torch.empty((batch_size, 0), device=device, dtype=dtype)
            best_indices = torch.empty((batch_size, 0), device=device, dtype=torch.long)
            normalizer = torch.full((batch_size,), -torch.inf, device=device, dtype=dtype)
            for offset in range(0, len(gallery), gallery_chunk):
                base = base_cache[offset : offset + gallery_chunk].to(device)
                locations = model.condition_locations(base, query_times)
                scores = model.logit_scale.exp() * torch.einsum("bd,bcd->bc", image, locations)
                normalizer = torch.logaddexp(normalizer, scores.logsumexp(dim=1))
                indices = torch.arange(offset, offset + len(base), device=device).expand(batch_size, -1)
                best_scores, best_indices = _rank(
                    torch.cat([best_scores, scores], dim=1),
                    torch.cat([best_indices, indices], dim=1),
                    top_k,
                )
            all_indices.append(best_indices.cpu())
            all_logits.append(best_scores.cpu())
            all_normalizers.append(normalizer.cpu())
        indices = torch.cat(all_indices).cpu().numpy()
        logits = torch.cat(all_logits).cpu().numpy()
        normalizer = torch.cat(all_normalizers).cpu().numpy()
        return gallery_np[indices[:, 0]], indices, logits, normalizer
    finally:
        for module, training in previous_modes:
            module.training = training
