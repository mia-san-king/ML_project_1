"""Fuse query dates directly into GeoCLIP's pretrained projection layers."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from models.geoclip_time.lib.temporal_model import _matrix


def _append_zero_columns(layer, count):
    """Expand a pretrained input layer while preserving its initial function."""
    expanded = nn.Linear(
        layer.in_features + count, layer.out_features,
        bias=layer.bias is not None, device=layer.weight.device,
        dtype=layer.weight.dtype,
    )
    with torch.no_grad():
        expanded.weight.zero_()
        expanded.weight[:, :layer.in_features].copy_(layer.weight)
        if layer.bias is not None:
            expanded.bias.copy_(layer.bias)
    expanded.weight.requires_grad_(layer.weight.requires_grad)
    if layer.bias is not None:
        expanded.bias.requires_grad_(layer.bias.requires_grad)
    expanded.train(layer.training)
    return expanded


class DirectTemporalGeoCLIP(nn.Module):
    """Append dates to image features and to each capsule's fixed GPS RFF.

    The supplied towers are expanded in place. Only their first input layers
    change shape; existing columns, biases, downstream weights, and parameter
    trainability are preserved. Added columns start at zero, so initialization
    reproduces the supplied pretrained model for every date.

    ``coordinate_projection`` defaults to GeoCLIP's equal-Earth projection,
    imported only when needed. A caller may supply the same function directly
    or a small projection fixture when testing without the GeoCLIP package.
    """

    def __init__(
        self, image_mlp, location_encoder, *, embed_dim=512, time_dim=3,
        logit_scale=None, coordinate_projection=None,
    ):
        super().__init__()
        for name, value in (("embed_dim", embed_dim), ("time_dim", time_dim)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(image_mlp, nn.Sequential) or not len(image_mlp) or not isinstance(image_mlp[0], nn.Linear):
            raise ValueError("image_mlp must start with a Linear layer")
        count = getattr(location_encoder, "n", 0)
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("location_encoder must contain GeoCLIP capsules")
        capsules = []
        for index in range(count):
            capsule = getattr(location_encoder, f"LocEnc{index}", None)
            layers = getattr(capsule, "capsule", None)
            if (not isinstance(layers, nn.Sequential) or len(layers) < 2
                    or not isinstance(layers[1], nn.Linear)
                    or not isinstance(getattr(capsule, "head", None), nn.Module)):
                raise ValueError("each location capsule must contain RFF, Linear, and head modules")
            capsules.append(capsule)
        scale = torch.as_tensor(
            math.log(1 / 0.07) if logit_scale is None else logit_scale,
            dtype=torch.float32,
        ).detach().clone()
        if scale.numel() != 1 or not torch.isfinite(scale).all():
            raise ValueError("logit_scale must be one finite scalar in log space")
        if coordinate_projection is not None and not callable(coordinate_projection):
            raise ValueError("coordinate_projection must be callable")

        self.image_mlp = image_mlp
        self.location_encoder = location_encoder
        self.embed_dim = embed_dim
        self.time_dim = time_dim
        self.mode = "direct"
        self.logit_scale = nn.Parameter(scale.reshape(()))
        self.coordinate_projection = coordinate_projection
        self.feature_dim = image_mlp[0].in_features
        self._rff_widths = tuple(capsule.capsule[1].in_features for capsule in capsules)
        self.location_base_dim = sum(self._rff_widths)
        image_mlp[0] = _append_zero_columns(image_mlp[0], time_dim)
        for capsule in capsules:
            capsule.capsule[1] = _append_zero_columns(capsule.capsule[1], time_dim)

    def encode_image(self, feats, times):
        """Normalized image/date embeddings, shaped ``[B, embed_dim]``."""
        _matrix(feats, "feats", width=self.feature_dim)
        _matrix(times, "times", width=self.time_dim, rows=len(feats))
        image = self.image_mlp(torch.cat((feats, times.to(feats)), dim=-1))
        _matrix(image, "image embeddings", width=self.embed_dim, rows=len(feats))
        return F.normalize(image, dim=-1)

    def encode_location_base(self, gps):
        """Cache concatenated fixed RFF features, shaped ``[C, base_dim]``."""
        _matrix(gps, "gps", width=2)
        projection = self.coordinate_projection
        if projection is None:
            from geoclip.model.location_encoder import equal_earth_projection

            projection = equal_earth_projection
        coordinates = projection(gps)
        bases = [
            getattr(self.location_encoder, f"LocEnc{index}").capsule[0](coordinates)
            for index in range(self.location_encoder.n)
        ]
        base = torch.cat(bases, dim=-1)
        _matrix(base, "location bases", width=self.location_base_dim, rows=len(gps))
        return base

    def _condition_times(self, base, times):
        """Evaluate every supplied date in one batched traversal per capsule."""
        query_times = times[:, None, :].expand(-1, len(base), -1)
        features = None
        offset = 0
        for index, width in enumerate(self._rff_widths):
            capsule = getattr(self.location_encoder, f"LocEnc{index}")
            rff = base[None, :, offset : offset + width].expand(len(times), -1, -1)
            values = torch.cat((rff, query_times), dim=-1)
            for layer_index in range(1, len(capsule.capsule)):
                values = capsule.capsule[layer_index](values)
            contribution = capsule.head(values)
            features = contribution if features is None else features + contribution
            offset += width
        if features.shape != (len(times), len(base), self.embed_dim):
            raise ValueError("location embeddings have an unexpected shape")
        if not torch.isfinite(features).all():
            raise ValueError("location embeddings must contain only finite values")
        return F.normalize(features, dim=-1)

    def locations_at_time(self, base, time):
        """Normalized candidate locations at one query date, shaped ``[C,D]``."""
        _matrix(base, "location bases", width=self.location_base_dim)
        if not isinstance(time, torch.Tensor) or time.ndim != 1:
            raise ValueError("time must be a one-dimensional tensor")
        _matrix(time.unsqueeze(0), "time", width=self.time_dim)
        return self._condition_times(base, time.to(base).unsqueeze(0))[0]

    def condition_locations(self, base, times):
        """Return ``[B,C,D]`` embeddings, reusing each distinct query date."""
        _matrix(base, "location bases", width=self.location_base_dim)
        _matrix(times, "times", width=self.time_dim)
        times = times.to(base)
        if times.requires_grad:
            # Training inputs normally have no gradients. Preserve independent
            # date gradients if a caller explicitly requests them.
            return self._condition_times(base, times)
        unique_times, inverse = torch.unique(times, dim=0, return_inverse=True)
        conditioned = self._condition_times(base, unique_times)
        return conditioned.index_select(0, inverse)

    def score_candidates(self, feats, times, gps):
        """Scaled cosine scores ``[B,C]``, with query time for all candidates."""
        image = self.encode_image(feats, times)
        base = self.encode_location_base(gps)
        locations = self.condition_locations(base, times)
        return self.logit_scale.exp() * torch.einsum("bd,bcd->bc", image, locations)
