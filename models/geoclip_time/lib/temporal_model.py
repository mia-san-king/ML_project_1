"""Lightweight time-conditioned towers over cached CLIP features.

This module does not load pretrained weights, CLIP, or dataset paths. The
caller supplies the image projection and location encoder, including their
existing ``requires_grad`` flags (notably the fixed RFF matrices).
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


def _matrix(values, name, *, width=None, rows=None, allow_empty=False):
    if not isinstance(values, torch.Tensor) or values.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional tensor")
    if width is not None and values.shape[1] != width:
        raise ValueError(f"{name} must have {width} columns")
    if rows is not None and len(values) != rows:
        raise ValueError(f"{name} must have {rows} rows")
    if (not allow_empty and len(values) == 0) or values.shape[1] == 0:
        raise ValueError(f"{name} must not be empty")
    if not torch.isfinite(values).all():
        raise ValueError(f"{name} must contain only finite values")


class _TimeResidual(nn.Module):
    """A bottleneck with nonlinear base/time interaction and a small residual."""

    def __init__(self, embed_dim, time_dim, hidden_dim):
        super().__init__()
        self.base_projection = nn.Linear(embed_dim, hidden_dim)
        self.time_projection = nn.Linear(time_dim, hidden_dim, bias=False)
        self.output_projection = nn.Linear(hidden_dim, embed_dim)
        # A small, nonzero start preserves the pretrained representation while
        # allowing gradients through both projections on the first update.
        nn.init.normal_(self.output_projection.weight, std=1e-3)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, base, times):
        hidden = self.base_projection(base) + self.time_projection(times)
        return base + self.output_projection(F.gelu(hidden))


class TemporalGeoCLIP(nn.Module):
    """Jointly trainable image/time and GPS/time towers.

    ``times`` contains the query's normalized midpoint, interval half-width,
    and known-date mask. For each query, the same time vector conditions every
    candidate. Missing dates can learn a residual via their zero-valued time
    vector; they are not discarded. ``mode='none'`` omits temporal adapters.
    """

    def __init__(
        self, image_mlp, location_encoder, *, embed_dim=512, time_dim=3,
        hidden_dim=128, mode="both", logit_scale=None,
    ):
        super().__init__()
        if mode not in {"both", "none"}:
            raise ValueError("mode must be 'both' or 'none'")
        for name, value in (("embed_dim", embed_dim), ("time_dim", time_dim), ("hidden_dim", hidden_dim)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.image_mlp = image_mlp
        self.location_encoder = location_encoder
        self.embed_dim = embed_dim
        self.time_dim = time_dim
        self.hidden_dim = hidden_dim
        self.mode = mode
        scale = torch.as_tensor(
            math.log(1 / 0.07) if logit_scale is None else logit_scale,
            dtype=torch.float32,
        ).detach().clone()
        if scale.numel() != 1 or not torch.isfinite(scale).all():
            raise ValueError("logit_scale must be one finite scalar in log space")
        self.logit_scale = nn.Parameter(scale.reshape(()))
        self.image_time_adapter = None
        self.location_time_adapter = None
        if mode == "both":
            self.image_time_adapter = _TimeResidual(embed_dim, time_dim, hidden_dim)
            self.location_time_adapter = _TimeResidual(embed_dim, time_dim, hidden_dim)

    def encode_image(self, feats, times):
        """Return normalized query image embeddings, shaped ``[B, D]``."""
        _matrix(feats, "feats")
        _matrix(times, "times", width=self.time_dim, rows=len(feats))
        base = self.image_mlp(feats)
        _matrix(base, "image embeddings", width=self.embed_dim, rows=len(feats))
        if self.mode == "both":
            base = self.image_time_adapter(base, times.to(base))
        return F.normalize(base, dim=-1)

    def encode_location_base(self, gps):
        """Compute time-independent GPS features once, shaped ``[C, D]``."""
        _matrix(gps, "gps", width=2)
        base = self.location_encoder(gps)
        _matrix(base, "location embeddings", width=self.embed_dim, rows=len(gps))
        return base

    def condition_locations(self, base, times):
        """Return normalized GPS/query-time pairs, shaped ``[B, C, D]``."""
        _matrix(base, "location bases", width=self.embed_dim)
        _matrix(times, "times", width=self.time_dim)
        if self.mode == "none":
            return F.normalize(base, dim=-1).unsqueeze(0).expand(len(times), -1, -1)
        # The singleton candidate axis on times is intentional: no candidate's
        # own date can enter this scoring path.
        conditioned = self.location_time_adapter(base[None, :, :], times.to(base)[:, None, :])
        return F.normalize(conditioned, dim=-1)

    def score_candidates(self, feats, times, gps):
        """Scaled cosine scores ``[B, C]`` using each row's query time."""
        image = self.encode_image(feats, times)
        base = self.encode_location_base(gps)
        locations = self.condition_locations(base, times)
        return self.logit_scale.exp() * torch.einsum("bd,bcd->bc", image, locations)
