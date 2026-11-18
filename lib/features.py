"""Cache CLIP image features (768-d, before GeoCLIP image MLP)."""
from __future__ import annotations

import torch
from geoclip.model.image_encoder import _extract_image_features


@torch.no_grad()
def encode_clip_pre_mlp(image_encoder, pixel_values: torch.Tensor) -> torch.Tensor:
    """768-d CLIP image embedding, matching ImageEncoder input to mlp."""
    return _extract_image_features(
        image_encoder.CLIP.get_image_features(pixel_values=pixel_values)
    )


@torch.no_grad()
def verify_pre_mlp_matches_forward(
    image_encoder, pixel_values: torch.Tensor, atol: float = 1e-3
) -> torch.Tensor:
    pre = encode_clip_pre_mlp(image_encoder, pixel_values)
    via_mlp = image_encoder.mlp(pre)
    full = image_encoder(pixel_values)
    if not torch.allclose(via_mlp, full, atol=atol):
        diff = (via_mlp - full).abs().max().item()
        raise AssertionError(f"pre_mlp vs forward max diff {diff:.6f} (atol={atol})")
    return pre