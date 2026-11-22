"""Exact CPU image-neighbor retrieval and capped calendar-year reranking.

Callers select the training reference rows before calling these helpers. Row
indices in every result refer to that supplied reference array, never to the
metadata table or a candidate's position within top-K.
"""
from __future__ import annotations

import math
from numbers import Integral

import numpy as np
import torch


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _embedding_source(values, name):
    source = values if isinstance(values, torch.Tensor) else np.asarray(values)
    if source.ndim != 2 or source.shape[1] == 0:
        raise ValueError(f"{name} must be a two-dimensional matrix with nonzero width")
    if isinstance(source, torch.Tensor):
        if source.is_complex():
            raise ValueError(f"{name} must contain real numbers")
    elif not np.issubdtype(source.dtype, np.number) or np.iscomplexobj(source):
        raise ValueError(f"{name} must contain real numbers")
    return source


def _normalized_batch(source, name):
    if isinstance(source, torch.Tensor):
        batch = source.detach().to(device="cpu", dtype=torch.float32)
    else:
        # Copy only this slice; read-only memory maps remain untouched.
        batch = torch.tensor(np.asarray(source), dtype=torch.float32, device="cpu")
    if not torch.isfinite(batch).all():
        raise ValueError(f"{name} must contain only finite values")
    norms = torch.linalg.vector_norm(batch, dim=1, keepdim=True)
    if not torch.isfinite(norms).all() or (norms <= 0).any():
        raise ValueError(f"{name} must contain finite nonzero-norm vectors")
    return batch / norms


def _topk_by_score_then_index(scores, top_k):
    """Sort K entries, scanning a full row only for a tie across the K cut."""
    count = min(top_k + 1, scores.shape[1])
    values, indices = torch.topk(scores, count, dim=1, sorted=True)
    selected_values = values[:, :top_k].clone()
    selected_indices = indices[:, :top_k].clone()
    if top_k < scores.shape[1]:
        boundary_rows = torch.nonzero(
            values[:, top_k - 1] == values[:, top_k], as_tuple=False,
        ).flatten()
        for row in boundary_rows.tolist():
            threshold = values[row, top_k - 1]
            strict_count = int((selected_values[row] > threshold).sum())
            # nonzero visits reference indices in ascending order. This also
            # handles an all-equal gallery without sorting the entire gallery.
            tied = torch.nonzero(scores[row] == threshold, as_tuple=False).flatten()
            selected_indices[row, strict_count:] = tied[: top_k - strict_count]
            selected_values[row, strict_count:] = threshold
    by_index = selected_indices.argsort(dim=1)
    selected_indices = selected_indices.gather(1, by_index)
    selected_values = selected_values.gather(1, by_index)
    by_score = selected_values.argsort(dim=1, descending=True, stable=True)
    return selected_indices.gather(1, by_score), selected_values.gather(1, by_score)


@torch.no_grad()
def exact_cosine_topk(
    query_embeddings, reference_embeddings, *, top_k=50, query_chunk=256,
    progress_callback=None,
):
    """Return ``(reference_indices, cosine_scores)`` as NumPy ``[N,K]`` arrays.

    Cosines use normalized CPU float32 embeddings and an exhaustive matrix
    product, without approximate search. The normalized reference matrix stays
    resident; query and score matrices are bounded by ``query_chunk``. Inputs
    may be NumPy arrays, memory maps, or tensors and are never modified.

    Scores descend; exact ties prefer the lower reference index, including
    ties at the K boundary. Near ties remain ordered by their computed float32
    scores. Empty queries return empty ``[0,K]`` outputs. Empty references,
    zero/nonfinite vectors, invalid chunk sizes, and K above the reference
    count raise ``ValueError``. If supplied, ``progress_callback(done, total)``
    runs after each completed query chunk, including the final chunk.
    """
    _positive_integer(top_k, "top_k")
    _positive_integer(query_chunk, "query_chunk")
    if progress_callback is not None and not callable(progress_callback):
        raise ValueError("progress_callback must be callable")
    queries = _embedding_source(query_embeddings, "query_embeddings")
    references = _embedding_source(reference_embeddings, "reference_embeddings")
    if len(references) == 0:
        raise ValueError("reference_embeddings must not be empty")
    if queries.shape[1] != references.shape[1]:
        raise ValueError("query and reference embedding dimensions must match")
    if top_k > len(references):
        raise ValueError("top_k must not exceed the reference count")
    all_indices = np.empty((len(queries), top_k), dtype=np.int64)
    all_scores = np.empty((len(queries), top_k), dtype=np.float32)
    # Avoid a second full reference-sized tensor while reading/normalizing.
    reference_matrix = torch.empty(references.shape, dtype=torch.float32, device="cpu")
    for start in range(0, len(references), 8192):
        batch = _normalized_batch(references[start : start + 8192], "reference_embeddings")
        reference_matrix[start : start + len(batch)] = batch
    for start in range(0, len(queries), query_chunk):
        batch = _normalized_batch(queries[start : start + query_chunk], "query_embeddings")
        scores = batch @ reference_matrix.T
        indices, values = _topk_by_score_then_index(scores, top_k)
        all_indices[start : start + len(batch)] = indices.numpy()
        all_scores[start : start + len(batch)] = values.numpy()
        if progress_callback is not None:
            progress_callback(start + len(batch), len(queries))
    return all_indices, all_scores


def _numpy(values):
    return values.detach().cpu().numpy() if isinstance(values, torch.Tensor) else np.asarray(values)


def _years_and_mask(years, known, size, name):
    years = np.asarray(_numpy(years), dtype=np.float64)
    if years.shape != (size,):
        raise ValueError(f"{name} must have shape [{size}]")
    valid = np.isfinite(years)
    if known is not None:
        mask = _numpy(known)
        if mask.shape != (size,) or not np.isin(mask, [0, 1]).all():
            raise ValueError(f"{name} known mask must have shape [{size}] and contain booleans")
        valid &= mask.astype(bool)
    return years, valid


def rerank_by_date(
    indices, scores, query_years, reference_years, reference_gps, *, gamma,
    query_known=None, reference_known=None, cap_years=50.0,
):
    """Choose one supplied visual candidate per query using calendar years.

    The score is ``cosine - gamma * min(abs(query_year-ref_year)/cap_years, 1)``.
    A penalty applies only when both years are finite and both optional known
    masks are true. Missing dates never gain an invented year. Inputs must be
    in descending visual-score order; exact reranked ties keep that order.
    Thus gamma zero reproduces the visual top-1, including its tie policy.

    Return aligned arrays ``neighbor_indices``, ``pred_gps``, ``scores``,
    ``visual_scores``, and ``year_penalties``. The last array contains the
    normalized capped gap before multiplication by gamma. No reference outside
    the supplied top-K can be selected, and no input is modified.
    """
    if not math.isfinite(gamma) or gamma < 0:
        raise ValueError("gamma must be finite and nonnegative")
    if not math.isfinite(cap_years) or cap_years <= 0:
        raise ValueError("cap_years must be finite and positive")
    candidates = _numpy(indices)
    visual = np.asarray(_numpy(scores), dtype=np.float64)
    gps = _numpy(reference_gps)
    if candidates.ndim != 2 or candidates.shape[1] == 0:
        raise ValueError("indices must have shape [N,K] with K positive")
    if not np.issubdtype(candidates.dtype, np.integer):
        raise ValueError("indices must contain integers")
    if visual.shape != candidates.shape or not np.isfinite(visual).all():
        raise ValueError("scores must be finite and have the same shape as indices")
    if (np.diff(visual, axis=1) > 0).any():
        raise ValueError("scores must be in descending visual order")
    if gps.ndim != 2 or gps.shape[1] != 2 or len(gps) == 0 or not np.isfinite(gps).all():
        raise ValueError("reference_gps must have finite, nonempty shape [R,2]")
    if (candidates < 0).any() or (candidates >= len(gps)).any():
        raise ValueError("indices must refer to valid reference rows")
    query_years, query_valid = _years_and_mask(
        query_years, query_known, len(candidates), "query_years",
    )
    reference_years, reference_valid = _years_and_mask(
        reference_years, reference_known, len(gps), "reference_years",
    )
    valid_pairs = query_valid[:, None] & reference_valid[candidates]
    gaps = np.zeros(visual.shape, dtype=np.float64)
    np.subtract(
        query_years[:, None], reference_years[candidates],
        out=gaps, where=valid_pairs,
    )
    penalties = np.minimum(np.abs(gaps), cap_years) / cap_years
    reranked = visual - gamma * penalties
    winner_columns = reranked.argmax(axis=1)
    rows = np.arange(len(candidates))
    winners = candidates[rows, winner_columns].astype(np.int64, copy=True)
    return {
        "neighbor_indices": winners,
        "pred_gps": gps[winners].copy(),
        "scores": reranked[rows, winner_columns],
        "visual_scores": visual[rows, winner_columns],
        "year_penalties": penalties[rows, winner_columns],
    }
