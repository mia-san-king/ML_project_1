"""Image-only time fusion keeps the jointly trained GPS tower date-independent."""

from importlib.resources import files

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from models.geoclip_time.tests.test_temporal_training import fixture_data, tiny_factory


def make_model(mode="image_only"):
    from models.geoclip_time.lib.temporal_model import TemporalGeoCLIP

    torch.manual_seed(13)
    return TemporalGeoCLIP(
        nn.Linear(3, 4), nn.Linear(2, 4), embed_dim=4,
        hidden_dim=7, mode=mode, logit_scale=0.0,
    )


def inputs():
    feats = torch.tensor([[0.3, -0.2, 1.1], [1.0, 0.7, -0.2], [-0.5, 0.2, 0.9]])
    times = torch.tensor([[-0.5, 0.1, 1.0], [0.4, 0.05, 1.0], [0.0, 0.0, 0.0]])
    gps = torch.tensor([[46.0, 7.0], [47.0, 9.0], [45.0, 6.0], [48.0, 10.0]])
    return feats, times, gps


def activate_image_time(model):
    generator = torch.Generator().manual_seed(2)
    with torch.no_grad():
        weight = model.image_time_adapter.output_projection.weight
        weight.copy_(torch.randn(weight.shape, generator=generator) * 0.3)


def assert_shared_initial_state(image_only, both):
    image_state, both_state = image_only.state_dict(), both.state_dict()
    assert image_state.keys() <= both_state.keys()
    assert both_state.keys() - image_state.keys()
    assert all(name.startswith("location_time_adapter.") for name in both_state.keys() - image_state.keys())
    for name, tensor in image_state.items():
        torch.testing.assert_close(tensor, both_state[name], rtol=0, atol=0)


def test_same_seed_preserves_all_shared_initial_tensors():
    image_only, both = make_model("image_only"), make_model("both")
    assert_shared_initial_state(image_only, both)
    assert image_only.image_time_adapter is not None
    assert image_only.location_time_adapter is None
    assert all(parameter.requires_grad for parameter in image_only.location_encoder.parameters())


def test_only_image_embeddings_depend_on_query_time():
    model = make_model()
    activate_image_time(model)
    feats, times, gps = inputs()
    repeated_image = feats[:1].expand(len(times), -1)
    images = model.encode_image(repeated_image, times)
    assert not torch.allclose(images[0], images[1])
    base = model.encode_location_base(gps)
    expected = F.normalize(base, dim=-1)[None].expand(len(times), -1, -1)
    torch.testing.assert_close(model.condition_locations(base, times), expected, rtol=0, atol=0)
    torch.testing.assert_close(model.condition_locations(base, times * 10 + 3), expected, rtol=0, atol=0)
    scores = model.score_candidates(repeated_image, times, gps)
    torch.testing.assert_close(scores, images @ expected[0].T)
    assert not torch.allclose(scores[0], scores[1])


def test_contrastive_loss_still_trains_both_towers_and_the_image_time_adapter():
    from models.geoclip_time.lib.temporal_ops import temporal_loss

    model = make_model()
    feats, times, gps = inputs()
    loss = temporal_loss(model, feats[:2], gps[:2], times[:2], gps[2:])
    loss.backward()
    assert torch.isfinite(loss)
    for name, parameter in model.named_parameters():
        assert parameter.requires_grad, name
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name


@pytest.mark.parametrize("query_chunk,gallery_chunk", [(1, 1), (2, 3)])
def test_image_only_retrieval_matches_dense_scores(query_chunk, gallery_chunk):
    from models.geoclip_time.lib.temporal_ops import predict_temporal

    model = make_model().eval()
    activate_image_time(model)
    feats, times, gps = inputs()
    with torch.no_grad():
        scores = model.score_candidates(feats, times, gps)
    expected_indices = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :2]
    pred, indices, logits, logsumexp = predict_temporal(
        model, feats.numpy(), times.numpy(), gps.numpy(), top_k=2,
        query_chunk=query_chunk, gallery_chunk=gallery_chunk,
    )
    np.testing.assert_array_equal(indices, expected_indices.numpy())
    np.testing.assert_array_equal(pred, gps.numpy()[indices[:, 0]])
    np.testing.assert_allclose(logits, scores.gather(1, expected_indices).numpy(), rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(logsumexp, scores.logsumexp(dim=1).numpy(), rtol=2e-6, atol=2e-6)


def test_image_only_training_checkpoint_and_prediction(fixture_data):
    from models.geoclip_time.lib.temporal_training import run_training, run_prediction, load_temporal_checkpoint

    root, split_csv, cfg = fixture_data
    cfg["run_name"] = "tiny_image_only"
    cfg["model"]["mode"] = "image_only"
    result = run_training(cfg, model_factory=tiny_factory)
    restored, payload = load_temporal_checkpoint(result["checkpoint"], model_factory=tiny_factory)
    assert payload["architecture"]["mode"] == "image_only"
    assert payload["config"]["model"]["mode"] == "image_only"
    assert restored.location_time_adapter is None
    for name, tensor in result["model"].state_dict().items():
        torch.testing.assert_close(tensor, restored.state_dict()[name], rtol=0, atol=0)
    predictions = run_prediction(
        result["checkpoint"], data_root=root, split_csv=split_csv,
        splits=("val", "test_in", "test_city"), model_factory=tiny_factory,
    )
    frame = pd.read_parquet(predictions["predictions"])
    assert frame.id.tolist() == [40, 31, 30, 50]
    assert frame.split.tolist() == ["test_in", "val", "val", "test_city"]
    assert np.isfinite(frame[["lat_pred", "lon_pred"]].to_numpy()).all()


def test_packaged_pretrained_image_only_shares_initial_state_with_both():
    pytest.importorskip("geoclip")
    root = files("geoclip").joinpath("model", "weights")
    required = ("image_encoder_mlp_weights.pth", "location_encoder_weights.pth", "logit_scale_weights.pth")
    if not all(root.joinpath(name).is_file() for name in required):
        pytest.skip("GeoCLIP packaged weights are unavailable; no download is needed")
    from models.geoclip_time.lib.temporal_factory import build_temporal_model

    torch.manual_seed(0)
    both = build_temporal_model({"mode": "both"}, pretrained=True)
    torch.manual_seed(0)
    image_only = build_temporal_model({"mode": "image_only"}, pretrained=True)
    assert_shared_initial_state(image_only, both)
    for name, parameter in image_only.location_encoder.named_parameters():
        assert parameter.requires_grad == dict(both.location_encoder.named_parameters())[name].requires_grad
