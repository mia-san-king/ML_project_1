"""Temporal scoring invariants, independent of downloaded GeoCLIP weights."""

import math

import numpy as np
import pytest
import torch
from torch import nn


def make_model(*, mode="both", embed_dim=4):
    from models.geoclip_time.lib.temporal_model import TemporalGeoCLIP

    torch.manual_seed(13)
    return TemporalGeoCLIP(
        nn.Linear(3, embed_dim),
        nn.Linear(2, embed_dim),
        embed_dim=embed_dim,
        hidden_dim=7,
        mode=mode,
        logit_scale=torch.tensor(0.0),
    )


def inputs():
    feats = torch.tensor([[0.3, -0.2, 1.1], [1.0, 0.7, -0.2]])
    gps = torch.tensor([[46.0, 7.0], [47.0, 9.0]])
    times = torch.tensor([[-0.5, 0.1, 1.0], [0.4, 0.05, 1.0]])
    negatives = torch.tensor([[45.0, 6.0], [48.0, 10.0]])
    return feats, gps, times, negatives


def test_baseline_is_normalized_cosine_and_independent_of_time():
    model = make_model(mode="none")
    feats, gps, times, _ = inputs()
    image = model.image_mlp(feats)
    location = model.location_encoder(gps)
    expected = (image / image.norm(dim=-1, keepdim=True)) @ (
        location / location.norm(dim=-1, keepdim=True)
    ).T
    torch.testing.assert_close(model.score_candidates(feats, times, gps), expected)
    torch.testing.assert_close(
        model.score_candidates(feats, times, gps),
        model.score_candidates(feats, times * 100 + 9, gps),
        rtol=0,
        atol=0,
    )


def test_each_candidate_uses_its_query_time_and_gps_base_runs_once():
    from models.geoclip_time.lib.temporal_model import TemporalGeoCLIP

    model = TemporalGeoCLIP(
        nn.Identity(), nn.Identity(), embed_dim=2, hidden_dim=2, logit_scale=0.0,
    )
    # Set an analytic location interaction: g + GELU(g + [time, 0]).
    with torch.no_grad():
        model.image_time_adapter.output_projection.weight.zero_()
        model.image_time_adapter.output_projection.bias.zero_()
        adapter = model.location_time_adapter
        adapter.base_projection.weight.copy_(torch.eye(2))
        adapter.base_projection.bias.zero_()
        adapter.time_projection.weight.copy_(torch.tensor([[1.0, 0, 0], [0, 0, 0]]))
        adapter.output_projection.weight.copy_(torch.eye(2))
        adapter.output_projection.bias.zero_()
    feats = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    times = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]])
    candidates = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    seen = []
    handle = model.location_encoder.register_forward_hook(
        lambda _module, args, _output: seen.append(args[0].clone())
    )
    scores = model.score_candidates(feats, times, candidates)
    handle.remove()
    assert len(seen) == 1
    torch.testing.assert_close(seen[0], candidates)
    # GELU(1) = 0.841344746..., GELU(-1) = -0.1586552539....
    # These are hand-derived scores, independent of the scoring helpers.
    a = 0.8413447460685429
    expected = torch.tensor([
        [1.0, 0.0, -1.0, 0.0],
        [1.0, a / math.hypot(a, 1 + a), -1.0, a / math.hypot(a, 2 - a)],
    ])
    torch.testing.assert_close(scores, expected)


def test_timestamp_only_representations_cannot_identify_candidates():
    from models.geoclip_time.lib.temporal_model import TemporalGeoCLIP
    from models.geoclip_time.lib.temporal_ops import temporal_loss

    class ZeroTower(nn.Module):
        def forward(self, values):
            return values.new_zeros((len(values), 4))

    model = TemporalGeoCLIP(
        ZeroTower(), ZeroTower(), embed_dim=4, hidden_dim=7,
        logit_scale=torch.tensor(0.0),
    )
    feats, gps, times, negatives = inputs()
    scores = model.score_candidates(feats, times, torch.cat([gps, negatives]))
    torch.testing.assert_close(scores, scores[:, :1].expand_as(scores))
    assert temporal_loss(model, feats, gps, times, negatives).item() == pytest.approx(
        math.log(4), abs=1e-6
    )


def test_joint_loss_reaches_both_towers_and_temporal_parameters():
    from models.geoclip_time.lib.temporal_ops import temporal_loss

    model = make_model()
    feats, gps, times, negatives = inputs()
    loss = temporal_loss(model, feats, gps, times, negatives)
    loss.backward()
    assert torch.isfinite(loss)
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name


def test_constructor_preserves_frozen_location_parameters():
    from models.geoclip_time.lib.temporal_model import TemporalGeoCLIP

    location = nn.Sequential(nn.Linear(2, 5), nn.GELU(), nn.Linear(5, 4))
    location[0].weight.requires_grad_(False)
    model = TemporalGeoCLIP(nn.Linear(3, 4), location, embed_dim=4)
    assert not model.location_encoder[0].weight.requires_grad
    assert model.location_encoder[2].weight.requires_grad


def test_identical_gps_are_positive_even_when_in_negative_pool():
    from models.geoclip_time.lib.temporal_ops import temporal_loss

    model = make_model(mode="none")
    feats, gps, times, _ = inputs()
    gps[:] = torch.tensor([46.0, 7.0])
    loss = temporal_loss(model, feats, gps, times, gps[:1])
    assert loss.item() == pytest.approx(0.0, abs=1e-7)
    loss.backward()
    assert model.image_mlp.weight.grad.abs().max().item() < 1e-6


def test_hard_loss_marginalizes_all_matching_coordinate_probabilities():
    from models.geoclip_time.lib.temporal_model import TemporalGeoCLIP
    from models.geoclip_time.lib.temporal_ops import temporal_loss

    model = TemporalGeoCLIP(
        nn.Identity(), nn.Identity(), embed_dim=2, mode="none", logit_scale=0.0,
    )
    # Query 0 scores [1, 0, 1]; query 1 scores [0, 1, 0]. Its target is
    # exactly the sum of the matching columns, not one diagonal column.
    gps = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    times = torch.zeros(2, 3)
    actual = temporal_loss(model, gps, gps, times, gps[:1])
    expected = (
        math.log(2 * math.e + 1) - math.log(2 * math.e)
        + math.log(math.e + 2) - 1
    ) / 2
    assert actual.item() == pytest.approx(expected, abs=1e-6)


def test_distance_soft_target_matches_hand_computed_equator_example():
    from models.geoclip_time.lib.temporal_model import TemporalGeoCLIP
    from models.geoclip_time.lib.temporal_ops import temporal_loss

    location = nn.Linear(2, 2)
    with torch.no_grad():
        location.weight.copy_(torch.tensor([[0.0, -1.0], [0.0, 1.0]]))
        location.bias.copy_(torch.tensor([2.0, -1.0]))
    model = TemporalGeoCLIP(
        nn.Identity(), location, embed_dim=2, mode="none", logit_scale=0.0,
    )
    # From (0, 1) to (0, 2) is one degree along the equator. With tau
    # equal to that distance, target weights are [e/(e+1), 1/(e+1)].
    gps = torch.tensor([[0.0, 1.0]])
    negatives = torch.tensor([[0.0, 2.0]])
    actual = temporal_loss(
        model, torch.tensor([[1.0, 0.0]]), gps, torch.zeros(1, 3), negatives,
        soft_tau_km=6371.0 * math.pi / 180,
    )
    # Location outputs [1, 0], [0, 1] give logits [1, 0].
    expected = math.log(math.e + 1) - math.e / (math.e + 1)
    assert actual.item() == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize("query_chunk,gallery_chunk", [(1, 1), (2, 2), (9, 9)])
def test_chunked_retrieval_matches_hand_scores_and_preserves_model_mode(
    query_chunk, gallery_chunk,
):
    from models.geoclip_time.lib.temporal_model import TemporalGeoCLIP
    from models.geoclip_time.lib.temporal_ops import predict_temporal

    model = TemporalGeoCLIP(
        nn.Identity(), nn.Identity(), embed_dim=2, mode="none", logit_scale=0.0,
    )
    model.train()
    model.location_encoder.eval()
    feats = np.array([[1, 0], [0, 1], [-1, 0]], dtype=np.float32)
    gallery = np.array([[1, 0], [0, 1], [-1, 0], [3, 4], [-3, -4]], dtype=np.float32)
    times = np.zeros((3, 3), dtype=np.float32)
    scores = np.array([[1, 0, -1, 0.6, -0.6], [0, 1, 0, 0.8, -0.8], [-1, 0, 1, -0.6, 0.6]])
    expected_indices = np.array([[0, 3, 1], [1, 3, 0], [2, 4, 1]])
    seen = []
    handle = model.location_encoder.register_forward_hook(
        lambda _module, args, _output: seen.append(args[0].clone())
    )
    pred, indices, logits, logsumexp = predict_temporal(
        model, feats, times, gallery, top_k=3,
        query_chunk=query_chunk, gallery_chunk=gallery_chunk,
    )
    handle.remove()
    assert all(isinstance(x, np.ndarray) for x in (pred, indices, logits, logsumexp))
    np.testing.assert_array_equal(indices, expected_indices)
    np.testing.assert_array_equal(pred, gallery[expected_indices[:, 0]])
    np.testing.assert_allclose(logits, np.take_along_axis(scores, expected_indices, axis=1), atol=1e-6)
    np.testing.assert_allclose(logsumexp, np.log(np.exp(scores).sum(axis=1)), atol=1e-6)
    torch.testing.assert_close(torch.cat(seen), torch.from_numpy(gallery))
    assert all(len(chunk) <= gallery_chunk for chunk in seen)
    assert model.training
    assert not model.location_encoder.training


def test_chunked_temporal_retrieval_matches_full_query_conditioned_scores():
    from models.geoclip_time.lib.temporal_ops import predict_temporal

    model = make_model()
    feats, gps, times, negatives = inputs()
    gallery = torch.cat([gps, negatives])
    full_scores = model.score_candidates(feats, times, gallery).detach()
    pred, indices, values, normalizer = predict_temporal(
        model, feats, times, gallery, top_k=3, query_chunk=1, gallery_chunk=2,
    )
    expected_values, expected_indices = full_scores.topk(3, dim=1)
    np.testing.assert_array_equal(indices, expected_indices.numpy())
    np.testing.assert_allclose(values, expected_values.numpy(), atol=1e-6)
    np.testing.assert_allclose(normalizer, full_scores.logsumexp(dim=1).numpy(), atol=1e-6)
    np.testing.assert_array_equal(pred, gallery[expected_indices[:, 0]].numpy())


def test_retrieval_disables_dropout_then_restores_training():
    from models.geoclip_time.lib.temporal_model import TemporalGeoCLIP
    from models.geoclip_time.lib.temporal_ops import predict_temporal

    model = TemporalGeoCLIP(
        nn.Dropout(0.95), nn.Dropout(0.95), embed_dim=2,
        mode="none", logit_scale=0.0,
    )
    feats = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    times = torch.zeros(2, 3)
    gallery = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    _, indices, logits, normalizer = predict_temporal(model, feats, times, gallery)
    np.testing.assert_array_equal(indices, [[0], [1]])
    np.testing.assert_allclose(logits, [[1.0], [1.0]])
    np.testing.assert_allclose(normalizer, [math.log(1 + math.e)] * 2, atol=1e-6)
    assert model.training and model.image_mlp.training and model.location_encoder.training


def test_missing_dates_still_train_residual_adapters():
    from models.geoclip_time.lib.temporal_ops import temporal_loss

    model = make_model()
    feats, gps, times, negatives = inputs()
    temporal_loss(model, feats, gps, torch.zeros_like(times), negatives).backward()
    for adapter in (model.image_time_adapter, model.location_time_adapter):
        gradient = adapter.output_projection.weight.grad
        assert gradient is not None and gradient.abs().sum() > 0


@pytest.mark.parametrize("kwargs", [
    {"top_k": 0}, {"top_k": 3}, {"query_chunk": 0}, {"gallery_chunk": -1},
    {"top_k": 1.5}, {"query_chunk": True},
])
def test_retrieval_rejects_invalid_chunk_or_topk(kwargs):
    from models.geoclip_time.lib.temporal_ops import predict_temporal

    model = make_model()
    feats, gps, times, _ = inputs()
    with pytest.raises(ValueError):
        predict_temporal(model, feats, times, gps, **kwargs)


@pytest.mark.parametrize("which", ["queries", "gallery", "times", "gps_shape", "nonfinite"])
def test_retrieval_rejects_empty_or_misaligned_inputs(which):
    from models.geoclip_time.lib.temporal_ops import predict_temporal

    model = make_model()
    feats, gps, times, _ = inputs()
    if which == "queries":
        feats, times = feats[:0], times[:0]
    elif which == "gallery":
        gps = gps[:0]
    elif which == "times":
        times = times[:1]
    elif which == "gps_shape":
        gps = gps[:, :1]
    else:
        feats[0, 0] = float("nan")
    with pytest.raises(ValueError):
        predict_temporal(model, feats, times, gps)


@pytest.mark.parametrize("tau", [0.0, -1.0, float("inf"), float("nan")])
def test_soft_target_requires_positive_finite_distance_scale(tau):
    from models.geoclip_time.lib.temporal_ops import temporal_loss

    model = make_model()
    feats, gps, times, negatives = inputs()
    with pytest.raises(ValueError):
        temporal_loss(model, feats, gps, times, negatives, soft_tau_km=tau)


def test_model_rejects_unsupported_mode_and_misaligned_time():
    with pytest.raises(ValueError):
        make_model(mode="image")
    model = make_model()
    feats, gps, times, _ = inputs()
    with pytest.raises(ValueError):
        model.score_candidates(feats, times[:1], gps)


def test_loss_rejects_misaligned_gps_and_accepts_empty_negative_pool():
    from models.geoclip_time.lib.temporal_ops import temporal_loss

    model = make_model()
    feats, gps, times, negatives = inputs()
    with pytest.raises(ValueError):
        temporal_loss(model, feats, gps[:1], times, negatives)
    assert torch.isfinite(temporal_loss(model, feats, gps, times, negatives[:0]))
