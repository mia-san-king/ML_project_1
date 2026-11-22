"""Exact retrieval of training GPS rows conditioned on each query's time."""
from __future__ import annotations

import numpy as np
import torch

from models.geoclip_time.lib.temporal_ops import _batch_tensor
from models.geoclip_time.lib.temporal_retrieval import (
    _positive_integer,
    _topk_by_score_then_index,
)


def _shape(values, name, *, width=None, rows=None):
    shape = getattr(values, "shape", ())
    if len(shape) != 2 or shape[1] == 0:
        raise ValueError(f"{name} must be a two-dimensional array with nonzero width")
    if width is not None and shape[1] != width:
        raise ValueError(f"{name} must have {width} columns")
    if rows is not None and shape[0] != rows:
        raise ValueError(f"{name} must have {rows} rows")
    return shape


@torch.inference_mode()
def query_time_location_topk(
    model, query_features, query_times, reference_gps, *, top_k=50,
    query_chunk=256, reference_chunk=8192, progress_callback=None,
    date_progress_callback=None,
):
    """Return reference indices and unscaled cosine scores as NumPy ``[N,K]``.

    The frozen ``mode='both'`` model encodes each image once and each reference
    GPS base once. All queries sharing an identical normalized time vector use
    one conditioned reference matrix, assembled in ``reference_chunk`` slices.
    No reference date enters this computation. Matrix multiplication exhausts
    the supplied reference set; exact ties prefer the lower reference index.

    Caches stay on the model's device in its dtype. Peak cache storage is
    approximately ``(N + 2*R)*D`` numbers plus one ``query_chunk * R`` score
    matrix and bounded encoder intermediates. No whole-input conversion of
    ``query_features`` occurs: lazy objects with shape/length/slicing work.
    Image and GPS embeddings returned by the model are already normalized and
    are not renormalized or recopied for each date group.

    Query groups may run out of row order; outputs are restored to original
    query order. Optional callbacks receive ``(completed, total)`` after each
    scored query chunk or completed unique date, respectively. Empty queries
    return empty arrays without model work. Every module's prior train/eval
    mode is restored even if a callback raises.
    """
    if getattr(model, "mode", None) != "both":
        raise ValueError("query-time location retrieval requires model.mode='both'")
    for name, value in (("top_k", top_k), ("query_chunk", query_chunk), ("reference_chunk", reference_chunk)):
        _positive_integer(value, name)
    for name, callback in (("progress_callback", progress_callback), ("date_progress_callback", date_progress_callback)):
        if callback is not None and not callable(callback):
            raise ValueError(f"{name} must be callable")
    n_queries = _shape(query_features, "query_features")[0]
    _shape(query_times, "query_times", width=model.time_dim, rows=n_queries)
    n_references = _shape(reference_gps, "reference_gps", width=2)[0]
    if n_references == 0 or top_k > n_references:
        raise ValueError("references must be nonempty and top_k must not exceed their count")
    all_indices = np.empty((n_queries, top_k), dtype=np.int64)
    all_scores = np.empty((n_queries, top_k), dtype=np.float32)
    if n_queries == 0:
        return all_indices, all_scores

    device, dtype = model.logit_scale.device, model.logit_scale.dtype
    # Group exactly the values seen by the model, after its dtype conversion.
    time_cache = _batch_tensor(query_times, "cpu", dtype)
    if not torch.isfinite(time_cache).all():
        raise ValueError("query_times must contain only finite values")
    unique_times, groups, counts = np.unique(
        time_cache.to(dtype=torch.float64).numpy(), axis=0,
        return_inverse=True, return_counts=True,
    )
    query_order = np.argsort(groups, kind="stable")
    previous_modes = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        image_cache = torch.empty((n_queries, model.embed_dim), dtype=dtype, device=device)
        for start in range(0, n_queries, query_chunk):
            feats = _batch_tensor(query_features[start : start + query_chunk], device, dtype)
            times = time_cache[start : start + len(feats)].to(device)
            image_cache[start : start + len(feats)] = model.encode_image(feats, times)

        base_cache = torch.empty((n_references, model.embed_dim), dtype=dtype, device=device)
        for start in range(0, n_references, reference_chunk):
            gps = _batch_tensor(reference_gps[start : start + reference_chunk], device, dtype)
            base_cache[start : start + len(gps)] = model.encode_location_base(gps)

        # Reuse this storage across dates; no concatenation duplicates a full
        # conditioned gallery and no normalization is repeated while scoring.
        conditioned = torch.empty_like(base_cache)
        group_start = 0
        completed_queries = 0
        for date_index, (time, count) in enumerate(zip(unique_times, counts)):
            single_time = _batch_tensor(time[None, :], device, dtype)
            for start in range(0, n_references, reference_chunk):
                base = base_cache[start : start + reference_chunk]
                conditioned[start : start + len(base)] = model.condition_locations(base, single_time)[0]
            group_end = group_start + int(count)
            for start in range(group_start, group_end, query_chunk):
                rows = query_order[start : min(start + query_chunk, group_end)]
                row_indices = torch.as_tensor(rows, dtype=torch.long, device=device)
                scores = image_cache.index_select(0, row_indices) @ conditioned.T
                indices, values = _topk_by_score_then_index(scores, top_k)
                del scores
                all_indices[rows] = indices.cpu().numpy()
                all_scores[rows] = values.float().cpu().numpy()
                completed_queries += len(rows)
                if progress_callback is not None:
                    progress_callback(completed_queries, n_queries)
            group_start = group_end
            if date_progress_callback is not None:
                date_progress_callback(date_index + 1, len(unique_times))
        return all_indices, all_scores
    finally:
        for module, training in previous_modes:
            module.training = training
