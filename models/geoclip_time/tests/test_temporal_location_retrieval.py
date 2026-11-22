"""Query-time GPS retrieval matches dense scoring and reuses shared work."""

from collections import Counter
import math

import numpy as np
import pytest
import torch
from torch import nn

from models.geoclip_time.lib.temporal_model import TemporalGeoCLIP


class CountingModel(TemporalGeoCLIP):
    def __init__(self):
        torch.manual_seed(23)
        super().__init__(
            nn.Sequential(nn.Linear(3, 4), nn.Dropout(0.8)),
            nn.Sequential(nn.Linear(2, 4), nn.Dropout(0.8)),
            embed_dim=4, hidden_dim=7, logit_scale=math.log(7),
        )
        with torch.no_grad():
            for adapter in (self.image_time_adapter, self.location_time_adapter):
                adapter.output_projection.weight.normal_(std=0.3)
        self.reset_counts()

    def reset_counts(self):
        self.image_rows = 0
        self.base_rows = 0
        self.conditioned_batches = []

    def encode_image(self, feats, times):
        self.image_rows += len(feats)
        return super().encode_image(feats, times)

    def encode_location_base(self, gps):
        self.base_rows += len(gps)
        return super().encode_location_base(gps)

    def condition_locations(self, base, times):
        self.conditioned_batches.append((len(base), times.detach().cpu().clone()))
        return super().condition_locations(base, times)


class SliceOnlyFeatures:
    def __init__(self, values):
        self.values = values
        self.shape = values.shape
        self.reads = []

    def __len__(self):
        return len(self.values)

    def __getitem__(self, key):
        assert isinstance(key, slice), "Features must be read in bounded slices"
        self.reads.append((key.start, min(key.stop, len(self))))
        return self.values[key]


def inputs():
    feats = torch.tensor([
        [0.3, -0.2, 1.1], [1.0, 0.7, -0.2], [-0.6, 0.4, 0.5],
        [0.2, 0.9, -0.7], [0.8, -0.3, 0.1],
    ])
    times = torch.tensor([
        [-0.5, 0.1, 1.0], [0.0, 0.0, 0.0], [0.4, 0.05, 1.0],
        [-0.5, 0.1, 1.0], [0.0, 0.0, 0.0],
    ])
    gps = torch.tensor([[-0.3, 0.2], [0.5, 0.1], [0.7, -0.4], [0.0, 1.0], [-1.0, 0.0]])
    return feats, times, gps


@pytest.mark.parametrize("query_chunk,reference_chunk", [(1, 1), (2, 2), (8, 4)])
def test_grouped_retrieval_matches_unscaled_dense_scores_and_reads_each_input_once(
    query_chunk, reference_chunk,
):
    from models.geoclip_time.lib.temporal_location_retrieval import query_time_location_topk

    model = CountingModel().eval()
    feats, times, gps = inputs()
    with torch.no_grad():
        expected = model.score_candidates(feats, times, gps) / model.logit_scale.exp()
    expected_indices = expected.argsort(dim=1, descending=True, stable=True)[:, :3]
    model.reset_counts()
    model.train()
    model.location_encoder.eval()
    lazy_features = SliceOnlyFeatures(feats.numpy())
    progress = []
    date_progress = []
    indices, scores = query_time_location_topk(
        model, lazy_features, times.numpy(), gps.numpy(), top_k=3,
        query_chunk=query_chunk, reference_chunk=reference_chunk,
        progress_callback=lambda done, total: progress.append((done, total)),
        date_progress_callback=lambda done, total: date_progress.append((done, total)),
    )
    np.testing.assert_array_equal(indices, expected_indices.numpy())
    np.testing.assert_allclose(scores, expected.gather(1, expected_indices).numpy(), atol=2e-6, rtol=2e-6)
    assert np.abs(scores).max() <= 1.000001
    assert model.image_rows == len(feats)
    assert model.base_rows == len(gps)
    assert lazy_features.reads == [
        (start, min(start + query_chunk, len(feats))) for start in range(0, len(feats), query_chunk)
    ]
    by_time = Counter()
    for count, query_time in model.conditioned_batches:
        assert count <= reference_chunk
        assert query_time.shape == (1, 3)
        by_time[tuple(query_time[0].tolist())] += count
    assert by_time == {tuple(time.tolist()): len(gps) for time in times.unique(dim=0)}
    assert progress[-1] == (len(feats), len(feats))
    assert all(0 < item[0] <= len(feats) and item[1] == len(feats) for item in progress)
    assert all(left[0] < right[0] for left, right in zip(progress, progress[1:]))
    assert date_progress == [(1, 3), (2, 3), (3, 3)]
    assert model.training and not model.location_encoder.training
    assert all(parameter.grad is None for parameter in model.parameters())


def test_duplicate_gps_and_exact_boundary_ties_prefer_lower_reference_index():
    from models.geoclip_time.lib.temporal_location_retrieval import query_time_location_topk

    model = TemporalGeoCLIP(nn.Identity(), nn.Identity(), embed_dim=2, hidden_dim=4, logit_scale=math.log(9))
    with torch.no_grad():
        for adapter in (model.image_time_adapter, model.location_time_adapter):
            adapter.output_projection.weight.zero_()
            adapter.output_projection.bias.zero_()
    feats = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    times = torch.tensor([[-0.5, 0.1, 1.0], [0.0, 0.0, 0.0], [-0.5, 0.1, 1.0]])
    gps = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0], [0.0, -1.0]])
    indices, scores = query_time_location_topk(
        model, feats, times, gps, top_k=2, query_chunk=1, reference_chunk=2,
    )
    np.testing.assert_array_equal(indices, [[0, 2], [1, 0], [0, 2]])
    np.testing.assert_array_equal(scores, [[1, 1], [1, 0], [1, 1]])


def test_empty_queries_return_empty_arrays_without_encoding_references():
    from models.geoclip_time.lib.temporal_location_retrieval import query_time_location_topk

    model = CountingModel()
    feats, times, gps = inputs()
    indices, scores = query_time_location_topk(model, feats[:0], times[:0], gps, top_k=2)
    assert indices.shape == scores.shape == (0, 2)
    assert indices.dtype == np.int64 and scores.dtype == np.float32
    assert model.image_rows == model.base_rows == 0


@pytest.mark.parametrize("invalid", [
    "mode", "top_k", "query_chunk", "reference_chunk", "too_many", "empty_refs",
    "times_shape", "bad_features", "bad_times", "bad_gps", "callback",
])
def test_invalid_inputs_fail_clearly(invalid):
    from models.geoclip_time.lib.temporal_location_retrieval import query_time_location_topk

    model = CountingModel()
    feats, times, gps = inputs()
    kwargs = {"top_k": 2}
    if invalid == "mode":
        model.mode = "none"
    elif invalid in ("top_k", "query_chunk", "reference_chunk"):
        kwargs[invalid] = 0
    elif invalid == "too_many":
        kwargs["top_k"] = 6
    elif invalid == "empty_refs":
        gps = gps[:0]
    elif invalid == "times_shape":
        times = times[:1]
    elif invalid == "bad_features":
        feats[0, 0] = float("nan")
    elif invalid == "bad_times":
        times[0, 0] = float("nan")
    elif invalid == "bad_gps":
        gps[0, 0] = float("nan")
    else:
        kwargs["progress_callback"] = True
    with pytest.raises(ValueError):
        query_time_location_topk(model, feats, times, gps, **kwargs)
    assert model.training


def test_callback_failure_restores_each_original_module_mode():
    from models.geoclip_time.lib.temporal_location_retrieval import query_time_location_topk

    model = CountingModel().train()
    model.location_encoder.eval()

    def fail(*_args):
        raise RuntimeError("stop")

    with pytest.raises(RuntimeError, match="stop"):
        query_time_location_topk(model, *inputs(), top_k=2, progress_callback=fail)
    assert model.training and not model.location_encoder.training
