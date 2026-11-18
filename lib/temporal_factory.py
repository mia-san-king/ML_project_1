"""Construct cached-feature GeoCLIP towers without creating a CLIP backbone."""
from __future__ import annotations

from importlib.resources import as_file, files
import math

import torch
from torch import nn

from lib.temporal_model import TemporalGeoCLIP


DEFAULT_ARCHITECTURE = {
    "backbone": "geoclip", "feature_dim": 768, "embed_dim": 512,
    "time_dim": 3, "hidden_dim": 128, "mode": "both", "sigma": [1, 16, 256],
}


def build_temporal_model(architecture: dict, *, pretrained: bool = True) -> TemporalGeoCLIP:
    """Load packaged small-tower weights only; restore needs no pretrained files.

    GeoCLIP itself is deliberately never instantiated: even its non-pretrained
    constructor creates the full CLIP network. LocationEncoder has no backbone.
    """
    arch = {**DEFAULT_ARCHITECTURE, **architecture}
    if arch["backbone"] != "geoclip":
        raise ValueError("unsupported backbone; expected geoclip")
    if (arch["feature_dim"], arch["embed_dim"], arch["time_dim"]) != (768, 512, 3):
        raise ValueError("GeoCLIP requires feature_dim=768, embed_dim=512, time_dim=3")
    if pretrained and arch["sigma"] != [1, 16, 256]:
        raise ValueError("packaged pretrained weights require sigma=[1, 16, 256]")
    # Lazy import keeps data validation and CLI help independent of GeoCLIP.
    from geoclip import LocationEncoder

    projection = nn.Sequential(nn.Linear(768, 768), nn.ReLU(), nn.Linear(768, 512))
    location = LocationEncoder(sigma=arch["sigma"], from_pretrained=False)
    scale = torch.tensor(math.log(1 / 0.07), dtype=torch.float32)
    if pretrained:
        root = files("geoclip").joinpath("model", "weights")

        def read(name):
            with as_file(root.joinpath(name)) as path:
                return torch.load(path, map_location="cpu", weights_only=True)

        projection.load_state_dict(read("image_encoder_mlp_weights.pth"), strict=True)
        location.load_state_dict(read("location_encoder_weights.pth"), strict=True)
        scale = read("logit_scale_weights.pth")
    return TemporalGeoCLIP(
        projection, location, embed_dim=arch["embed_dim"], time_dim=arch["time_dim"],
        hidden_dim=arch["hidden_dim"], mode=arch["mode"], logit_scale=scale,
    )
