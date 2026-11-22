"""Exact image retrieval with one maximum-cosine score per original photo."""
from __future__ import annotations

from numbers import Integral

import numpy as np


def parent_reference_layout(rows):
    """Return original row positions and each reference's compact parent index.

    Positive IDs are originals; negative IDs identify the sole variant of their
    positive parent. Groups retain original feature-cache order for exact ties.
    Dates and GPS must be inherited exactly, including missing dates.
    """
    required = {"id", "split", "latitude", "longitude", "date_shot_min", "date_shot_max"}
    if not required.issubset(rows.columns) or rows.empty:
        raise ValueError("parent references require nonempty IDs, split, GPS and date columns")
    ids = rows.id.to_numpy()
    if (not np.issubdtype(ids.dtype, np.integer) or (ids == 0).any()
            or (ids == np.iinfo(np.int64).min).any() or len(np.unique(ids)) != len(ids)):
        raise ValueError("parent references need unique nonzero signed integer IDs")
    if not rows.split.eq("train").all():
        raise ValueError("parent references must contain train only")
    originals = np.flatnonzero(ids > 0)
    lookup = {int(ids[position]): group for group, position in enumerate(originals)}
    parents = np.abs(ids)
    if any(int(parent) not in lookup for parent in parents):
        raise ValueError("every variant must have its positive parent in training references")
    groups = np.array([lookup[int(parent)] for parent in parents], dtype=np.int64)
    expected = rows.iloc[originals[groups]].reset_index(drop=True)
    actual = rows.reset_index(drop=True)
    for column in ("latitude", "longitude", "date_shot_min", "date_shot_max"):
        equal = actual[column].eq(expected[column]) | (actual[column].isna() & expected[column].isna())
        if not equal.all():
            raise ValueError(f"variant {column} differs from its original parent")
    return originals, groups


def validate_augmented_rows(rows):
    """Reject synthetic evaluation queries and report train augmentation counts."""
    if rows.id.duplicated().any() or rows.id.isna().any() or (rows.id == 0).any():
        raise ValueError("augmented rows need unique nonzero IDs")
    if ((rows.id < 0) & ~rows.split.eq("train")).any():
        raise ValueError("synthetic IDs are forbidden outside the training split")
    eval_rows = rows.split.isin(("val", "test_in", "test_city", "test_region"))
    train = rows.loc[rows.split.eq("train")]
    originals, _ = parent_reference_layout(train)
    variants = int((train.id < 0).sum())
    if not variants:
        raise ValueError("augmented training requires at least one cached synthetic variant")
    return {"train_originals": len(originals), "train_variants": variants,
            "train_rows": len(train), "parents_with_variant": variants,
            "evaluation_originals": int(eval_rows.sum())}


def exact_parent_cosine_topk(query_embeddings, reference_embeddings, parent_indices, *,
                            top_k=50, query_chunk=128, device="cpu", progress_callback=None):
    """Exhaustively score all variants, max per parent, then take unique top-K.

    Outputs index the compact parent array, not individual augmented references.
    All computation is float32; exact ties prefer smaller parent indices. CUDA
    uses full-precision float32 matmul (TF32 disabled), but arithmetic can still
    differ from CPU near floating-point ties. No approximate search is used.
    """
    import torch
    from models.geoclip_time.lib.temporal_retrieval import _topk_by_score_then_index

    for name, value in (("top_k", top_k), ("query_chunk", query_chunk)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if progress_callback is not None and not callable(progress_callback):
        raise ValueError("progress_callback must be callable")
    target = torch.device(device)
    if target.type not in ("cpu", "cuda"):
        raise ValueError("parent retrieval supports cpu or cuda")
    queries = query_embeddings if isinstance(query_embeddings, torch.Tensor) else np.asarray(query_embeddings)
    refs = reference_embeddings if isinstance(reference_embeddings, torch.Tensor) else np.asarray(reference_embeddings)
    if (queries.ndim != 2 or refs.ndim != 2 or not len(refs) or refs.shape[1] == 0
            or queries.shape[1] != refs.shape[1]):
        raise ValueError("embeddings must have matching nonzero widths and nonempty references")
    groups = np.asarray(parent_indices)
    if (groups.shape != (len(refs),) or not np.issubdtype(groups.dtype, np.integer)
            or (groups < 0).any()):
        raise ValueError("parent_indices must be one nonnegative integer per reference")
    unique = np.unique(groups)
    if not np.array_equal(unique, np.arange(len(unique))) or top_k > len(unique):
        raise ValueError("parent indices must be compact 0..P-1 and top_k must not exceed parent count")

    def normalized(values):
        if isinstance(values, torch.Tensor):
            if values.is_complex():
                raise ValueError("embeddings must be real")
            batch = values.detach().to(device=target, dtype=torch.float32)
        else:
            if not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values):
                raise ValueError("embeddings must be real")
            batch = torch.tensor(np.asarray(values), dtype=torch.float32, device=target)
        norms = torch.linalg.vector_norm(batch, dim=1, keepdim=True)
        if not torch.isfinite(batch).all() or not torch.isfinite(norms).all() or (norms <= 0).any():
            raise ValueError("embeddings must contain finite nonzero vectors")
        return batch / norms

    all_indices = np.empty((len(queries), top_k), dtype=np.int64)
    all_scores = np.empty((len(queries), top_k), dtype=np.float32)
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        if target.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False
        with torch.inference_mode():
            reference_matrix = torch.empty(refs.shape, dtype=torch.float32, device=target)
            for start in range(0, len(refs), 8192):
                batch = normalized(refs[start:start + 8192])
                reference_matrix[start:start + len(batch)] = batch
            group_tensor = torch.as_tensor(groups, dtype=torch.long, device=target)
            for start in range(0, len(queries), query_chunk):
                query = normalized(queries[start:start + query_chunk])
                raw_scores = query @ reference_matrix.T
                parent_scores = torch.full((len(query), len(unique)), -torch.inf, dtype=torch.float32, device=target)
                parent_scores.scatter_reduce_(1, group_tensor.expand(len(query), -1), raw_scores,
                                              reduce="amax", include_self=True)
                indices, scores = _topk_by_score_then_index(parent_scores, top_k)
                all_indices[start:start + len(query)] = indices.cpu().numpy()
                all_scores[start:start + len(query)] = scores.cpu().numpy()
                if progress_callback is not None:
                    progress_callback(start + len(query), len(queries))
    finally:
        if target.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    return all_indices, all_scores
