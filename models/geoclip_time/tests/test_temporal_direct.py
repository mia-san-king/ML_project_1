"""Direct time fusion: initialization, gradients, retrieval and restoration."""

from copy import deepcopy
from importlib.resources import files
import math

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F


class TinyRFF(nn.Module):
    def __init__(self):
        super().__init__()
        self.b = nn.Parameter(torch.randn(3, 2), requires_grad=False)

    def forward(self, coordinates):
        angle = 2 * math.pi * (coordinates @ self.b.T)
        return torch.cat((angle.cos(), angle.sin()), dim=-1)


class TinyCapsule(nn.Module):
    def __init__(self):
        super().__init__()
        self.capsule = nn.Sequential(
            TinyRFF(), nn.Linear(6, 7), nn.ReLU(),
            nn.Linear(7, 7), nn.ReLU(), nn.Linear(7, 7), nn.ReLU(),
        )
        self.head = nn.Sequential(nn.Linear(7, 4))

    def forward(self, coordinates):
        return self.head(self.capsule(coordinates))


class TinyLocation(nn.Module):
    def __init__(self):
        super().__init__()
        self.n = 2
        for index in range(self.n):
            self.add_module(f"LocEnc{index}", TinyCapsule())

    def forward(self, coordinates):
        return sum(capsule(coordinates) for capsule in self.children())


def tiny_towers():
    return nn.Sequential(nn.Linear(3, 7), nn.ReLU(), nn.Linear(7, 4)), TinyLocation()


def make_direct(*, seed=21):
    from models.geoclip_time.lib.temporal_direct import DirectTemporalGeoCLIP

    torch.manual_seed(seed)
    image, location = tiny_towers()
    return DirectTemporalGeoCLIP(
        image, location, embed_dim=4, logit_scale=0.0,
        coordinate_projection=lambda coordinates: coordinates,
    )


def inputs():
    feats = torch.tensor([
        [0.3, -0.2, 1.1], [1.0, 0.7, -0.2], [-0.6, 0.4, 0.5],
        [0.2, 0.9, -0.7], [0.8, -0.3, 0.1],
    ])
    times = torch.tensor([
        [-0.5, 0.1, 1.0], [0.4, 0.05, 1.0], [-0.5, 0.1, 1.0],
        [0.0, 0.0, 0.0], [0.4, 0.05, 1.0],
    ])
    gallery = torch.tensor([[46.0, 7.0], [47.0, 9.0], [46.0, 7.0], [45.0, 6.0], [47.0, 9.0]])
    return feats, times, gallery


def activate_time_columns(model):
    generator = torch.Generator().manual_seed(8)
    first_layers = [model.image_mlp[0]] + [
        capsule.capsule[1] for capsule in model.location_encoder.children()
    ]
    with torch.no_grad():
        for layer in first_layers:
            layer.weight[:, -3:].copy_(torch.randn(layer.out_features, 3, generator=generator) * 0.3)


def test_zero_time_columns_preserve_the_original_tower_functions():
    from models.geoclip_time.lib.temporal_direct import DirectTemporalGeoCLIP

    torch.manual_seed(21)
    image, location = tiny_towers()
    original_image, original_location = deepcopy(image), deepcopy(location)
    model = DirectTemporalGeoCLIP(
        image, location, embed_dim=4, logit_scale=0.0,
        coordinate_projection=lambda coordinates: coordinates,
    )
    feats, times, _ = inputs()
    # The location contract consumes concatenated RFF values; no coordinate
    # projection is needed to check preservation of the original MLP function.
    coordinates = torch.tensor([[-0.3, 0.2], [0.5, 0.1], [0.7, -0.4]])
    base = torch.cat([capsule.capsule[0](coordinates) for capsule in original_location.children()], dim=1)
    expected_image = F.normalize(original_image(feats), dim=-1)
    expected_location = F.normalize(original_location(coordinates), dim=-1)
    torch.testing.assert_close(model.encode_image(feats, times), expected_image)
    actual = model.condition_locations(base, times)
    torch.testing.assert_close(actual, expected_location[None].expand(len(times), -1, -1))
    torch.testing.assert_close(model.locations_at_time(base, times[0]), expected_location)
    for old, new in zip(original_location.children(), model.location_encoder.children()):
        torch.testing.assert_close(old.capsule[0].b, new.capsule[0].b, rtol=0, atol=0)
        assert not new.capsule[0].b.requires_grad
    tower_pairs = ((original_image, model.image_mlp), (original_location, model.location_encoder))
    for old_tower, new_tower in tower_pairs:
        new_parameters = dict(new_tower.named_parameters())
        for name, parameter in old_tower.named_parameters():
            expanded = new_parameters[name]
            common = expanded[:, :parameter.shape[1]] if expanded.shape != parameter.shape else expanded
            torch.testing.assert_close(common, parameter, rtol=0, atol=0)
            assert expanded.requires_grad == parameter.requires_grad
    # Three time columns in one 7-output image layer and two 7-output GPS layers.
    old_count = sum(p.numel() for tower in (original_image, original_location) for p in tower.parameters())
    new_count = sum(p.numel() for tower in (model.image_mlp, model.location_encoder) for p in tower.parameters())
    assert new_count - old_count == 63
    (model.encode_image(feats, times).sum() + actual.sum()).backward()
    (expected_image.sum() + expected_location.sum() * len(times)).backward()
    for old_tower, new_tower in tower_pairs:
        new_parameters = dict(new_tower.named_parameters())
        for name, parameter in old_tower.named_parameters():
            if not parameter.requires_grad:
                continue
            gradient = new_parameters[name].grad
            common = gradient[:, :parameter.shape[1]] if gradient.shape != parameter.shape else gradient
            torch.testing.assert_close(common, parameter.grad, rtol=1e-4, atol=2e-6)


def test_first_contrastive_backward_reaches_time_columns_in_both_towers():
    from models.geoclip_time.lib.temporal_ops import temporal_loss

    model = make_direct()
    feats, times, gallery = inputs()
    loss = temporal_loss(model, feats[:2], gallery[:2], times[:2], gallery[3:])
    loss.backward()
    assert torch.isfinite(loss)
    first_layers = [model.image_mlp[0]] + [
        capsule.capsule[1] for capsule in model.location_encoder.children()
    ]
    for layer in first_layers:
        assert torch.count_nonzero(layer.weight[:, -3:]) == 0
        assert layer.weight.grad is not None
        assert torch.isfinite(layer.weight.grad).all()
        assert layer.weight.grad[:, -3:].abs().sum() > 0
        assert layer.weight.grad[:, :-3].abs().sum() > 0
    for capsule in model.location_encoder.children():
        assert capsule.capsule[0].b.grad is None
        assert capsule.head[0].weight.grad.abs().sum() > 0


def test_time_columns_condition_both_images_and_every_candidate():
    model = make_direct()
    activate_time_columns(model)
    feats, times, gallery = inputs()
    repeated_images = feats[:1].expand(2, -1)
    pair_times = times[:2]
    images = model.encode_image(repeated_images, pair_times)
    base = model.encode_location_base(gallery)
    locations = model.condition_locations(base, pair_times)
    assert not torch.allclose(images[0], images[1])
    assert not torch.allclose(locations[0], locations[1])
    for index, time in enumerate(pair_times):
        torch.testing.assert_close(locations[index], model.locations_at_time(base, time))
    scores = model.score_candidates(repeated_images, pair_times, gallery)
    for index in range(2):
        single = model.score_candidates(repeated_images[index:index + 1], pair_times[index:index + 1], gallery)
        torch.testing.assert_close(scores[index], single[0])


@pytest.mark.parametrize("require_time_grad", [False, True])
def test_location_conditioning_batches_dates_without_losing_input_gradients(require_time_grad):
    model = make_direct()
    activate_time_columns(model)
    _, times, gallery = inputs()
    times.requires_grad_(require_time_grad)
    base = model.encode_location_base(gallery)
    seen = [[] for _ in model.location_encoder.children()]
    handles = []
    for capsule, shapes in zip(model.location_encoder.children(), seen):
        handles.append(capsule.capsule[1].register_forward_hook(
            lambda _module, arguments, _output, shapes=shapes: shapes.append(tuple(arguments[0].shape))
        ))
    try:
        locations = model.condition_locations(base, times)
    finally:
        for handle in handles:
            handle.remove()
    expected_dates = 5 if require_time_grad else 3
    assert all(shapes == [(expected_dates, len(gallery), 9)] for shapes in seen)
    if require_time_grad:
        direction = torch.arange(1, 5, dtype=locations.dtype)
        (locations * direction).sum().backward()
        assert torch.isfinite(times.grad).all()
        assert (times.grad.abs().sum(dim=1) > 0).all()


@pytest.mark.parametrize("query_chunk,gallery_chunk", [(1, 1), (2, 2), (4, 3), (8, 20)])
def test_chunked_direct_retrieval_preserves_interleaved_dates(query_chunk, gallery_chunk):
    from models.geoclip_time.lib.temporal_ops import predict_temporal

    model = make_direct().eval()
    activate_time_columns(model)
    feats, times, gallery = inputs()
    # Use separated candidates for numerical equivalence; equal coordinates
    # can acquire rounding differences in BLAS kernels of different batch sizes.
    gallery[2] = torch.tensor([46.1, 7.8])
    gallery[4] = torch.tensor([47.2, 9.5])
    with torch.no_grad():
        dense = model.score_candidates(feats, times, gallery)
    expected_indices = torch.argsort(dense, dim=1, descending=True, stable=True)[:, :4]
    expected_scores = dense.gather(1, expected_indices)
    model.train()
    model.location_encoder.eval()
    pred, indices, logits, logsumexp = predict_temporal(
        model, feats.numpy(), times.numpy(), gallery.numpy(), top_k=4,
        query_chunk=query_chunk, gallery_chunk=gallery_chunk,
    )
    np.testing.assert_array_equal(indices, expected_indices.numpy())
    np.testing.assert_array_equal(pred, gallery.numpy()[indices[:, 0]])
    np.testing.assert_allclose(logits, expected_scores.numpy(), atol=2e-6, rtol=2e-6)
    np.testing.assert_allclose(logsumexp, dense.logsumexp(dim=1).numpy(), atol=2e-6, rtol=2e-6)
    assert model.training
    assert not model.location_encoder.training


@pytest.mark.parametrize("query_chunk,gallery_chunk", [(1, 1), (2, 2), (4, 3), (8, 20)])
def test_direct_exact_ties_favor_low_gallery_indices(query_chunk, gallery_chunk):
    from models.geoclip_time.lib.temporal_ops import predict_temporal

    model = make_direct().eval()
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.zero_()
        # Both embeddings are exactly the first unit vector. Scores equal 1
        # independently of BLAS rounding, query date, and gallery chunk size.
        model.image_mlp[-1].bias[0] = 1
        for capsule in model.location_encoder.children():
            capsule.head[0].bias[0] = 1
    feats, times, gallery = inputs()
    pred, indices, logits, logsumexp = predict_temporal(
        model, feats.numpy(), times.numpy(), gallery.numpy(), top_k=4,
        query_chunk=query_chunk, gallery_chunk=gallery_chunk,
    )
    np.testing.assert_array_equal(indices, np.tile(np.arange(4), (len(feats), 1)))
    np.testing.assert_array_equal(pred, np.tile(gallery[0].numpy(), (len(feats), 1)))
    np.testing.assert_array_equal(logits, np.ones((len(feats), 4)))
    np.testing.assert_allclose(logsumexp, 1 + math.log(len(gallery)), atol=1e-6)


def test_direct_checkpoint_roundtrip_retains_time_conditioning(tmp_path):
    from models.geoclip_time.lib.temporal_training import load_temporal_checkpoint

    model = make_direct().eval()
    activate_time_columns(model)
    feats, times, gallery = inputs()
    with torch.no_grad():
        expected = model.score_candidates(feats, times, gallery)
    path = tmp_path / "direct.pt"
    torch.save({
        "format_version": 1, "architecture": {"mode": "direct"},
        "time_config": {"year_center": 1950.0, "year_scale": 100.0},
        "config": {"model": {"mode": "direct"}}, "model_state": model.state_dict(),
    }, path)

    def factory(architecture, *, pretrained):
        assert architecture["mode"] == "direct"
        assert pretrained is False
        return make_direct(seed=99)

    restored, payload = load_temporal_checkpoint(path, model_factory=factory)
    with torch.no_grad():
        actual = restored.score_candidates(feats, times, gallery)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert payload["architecture"]["mode"] == "direct"
    assert not restored.training


def test_packaged_pretrained_direct_matches_no_time_geoclip():
    pytest.importorskip("geoclip")
    root = files("geoclip").joinpath("model", "weights")
    required = ("image_encoder_mlp_weights.pth", "location_encoder_weights.pth", "logit_scale_weights.pth")
    if not all(root.joinpath(name).is_file() for name in required):
        pytest.skip("GeoCLIP packaged weights are unavailable; no download is needed")
    from models.geoclip_time.lib.temporal_factory import build_temporal_model

    baseline = build_temporal_model({"mode": "none"}, pretrained=True).eval()
    direct = build_temporal_model({"mode": "direct"}, pretrained=True).eval()
    generator = torch.Generator().manual_seed(2)
    feats = torch.randn(3, 768, generator=generator)
    times = torch.tensor([[-0.5, 0.1, 1.0], [0.4, 0.05, 1.0], [0.0, 0.0, 0.0]])
    gps = torch.tensor([[46.95, 7.45], [46.20, 6.14], [47.38, 8.54], [46.00, 9.00]])
    with torch.no_grad():
        expected = baseline.score_candidates(feats, times, gps)
        actual = direct.score_candidates(feats, times, gps)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    for name, parameter in baseline.location_encoder.named_parameters():
        if not parameter.requires_grad:
            counterpart = dict(direct.location_encoder.named_parameters())[name]
            torch.testing.assert_close(parameter, counterpart, rtol=0, atol=0)
            assert not counterpart.requires_grad


def test_real_direct_checkpoint_restores_with_default_factory(tmp_path):
    pytest.importorskip("geoclip")
    from models.geoclip_time.lib.temporal_factory import build_temporal_model
    from models.geoclip_time.lib.temporal_training import load_temporal_checkpoint

    model = build_temporal_model({"mode": "direct"}, pretrained=False).eval()
    activate_time_columns(model)
    generator = torch.Generator().manual_seed(2)
    feats = torch.randn(2, 768, generator=generator)
    times = torch.tensor([[-0.5, 0.1, 1.0], [0.4, 0.05, 1.0]])
    gps = torch.tensor([[46.95, 7.45], [46.20, 6.14], [47.38, 8.54]])
    with torch.no_grad():
        expected = model.score_candidates(feats, times, gps)
    path = tmp_path / "real_direct.pt"
    torch.save({
        "format_version": 1, "architecture": {"mode": "direct"},
        "time_config": {"year_center": 1950.0, "year_scale": 100.0},
        "config": {"model": {"mode": "direct"}}, "model_state": model.state_dict(),
    }, path)
    restored, _ = load_temporal_checkpoint(path)
    with torch.no_grad():
        actual = restored.score_candidates(feats, times, gps)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
