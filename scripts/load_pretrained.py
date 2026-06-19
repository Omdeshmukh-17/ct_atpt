"""Load pretrained ViT-B/16 (ImageNet) weights into CT-ATPT.

The key insight: ViT-B/16 has the EXACT same architecture dimensions as our
CT-ATPT model (embed_dim=768, depth=12, heads=12, mlp_ratio=4.0). So we can
transfer all transformer block weights directly. Only two things need adaptation:

1. Patch embedding: 2D Conv2d (16×16) → 3D Conv3d (8×16×16)
   Strategy: Replicate the 2D kernel along Z and divide by patch_z
   (central frame initialization — averages the 2D feature across Z slices)

2. Positional embeddings: 14×14 = 196 positions → 12×6×6 = 432 positions
   Strategy: 3D interpolation of the 2D grid to our 3D grid

Everything else (CLS token, detection head, pruning parameters, CLS+GAP head)
remains randomly initialized — these are our novel additions.

Usage:
    from scripts.load_pretrained import load_imagenet_vit_into_ctatpt
    model = CTATPT(config)
    loaded_keys = load_imagenet_vit_into_ctatpt(model)
"""
from __future__ import annotations

import math
from collections import OrderedDict

import torch
import torch.nn.functional as F


def _download_vit_b16_weights() -> OrderedDict:
    """Download ViT-B/16 pretrained on ImageNet-21k, fine-tuned on ImageNet-1k.
    
    Uses torchvision's built-in weights (no extra dependencies needed).
    Returns the state_dict with original key names.
    """
    try:
        from torchvision.models import vit_b_16, ViT_B_16_Weights
        print("Loading ViT-B/16 ImageNet weights from torchvision...")
        vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        return vit.state_dict()
    except ImportError:
        raise ImportError(
            "torchvision is required. Install with: pip install torchvision"
        )


def _inflate_2d_to_3d(weight_2d: torch.Tensor, patch_z: int) -> torch.Tensor:
    """Inflate a 2D conv kernel [C_out, C_in, H, W] to 3D [C_out, C_in, Z, H, W].
    
    Central frame initialization: repeat the 2D kernel along Z and divide by
    patch_z so the average output magnitude is preserved.
    """
    # weight_2d: [C_out, C_in, H, W]
    # Expand to: [C_out, C_in, Z, H, W]
    weight_3d = weight_2d.unsqueeze(2).repeat(1, 1, patch_z, 1, 1)
    weight_3d = weight_3d / patch_z  # preserve output scale
    return weight_3d


def _interpolate_pos_embed_3d(
    pos_embed_2d: torch.Tensor,
    grid_2d: tuple[int, int],
    grid_3d: tuple[int, int, int],
) -> torch.Tensor:
    """Interpolate 2D positional embeddings to 3D grid.
    
    pos_embed_2d: [1, 1 + H*W, D] (includes CLS token at position 0)
    grid_2d: (H, W) of the original 2D ViT (e.g., 14, 14)
    grid_3d: (Z, Y, X) of our 3D model (e.g., 12, 6, 6)
    
    Returns: [1, 1 + Z*Y*X, D]
    """
    D = pos_embed_2d.shape[-1]
    cls_embed = pos_embed_2d[:, :1, :]  # [1, 1, D] — CLS token
    patch_embed_2d = pos_embed_2d[:, 1:, :]  # [1, H*W, D]
    
    H, W = grid_2d
    Z, Y, X = grid_3d
    
    # Reshape to spatial grid: [1, D, H, W]
    patch_embed_2d = patch_embed_2d.reshape(1, H, W, D).permute(0, 3, 1, 2)
    
    # First interpolate 2D: [1, D, H, W] → [1, D, Y, X]
    patch_embed_yx = F.interpolate(
        patch_embed_2d.float(),
        size=(Y, X),
        mode="bilinear",
        align_corners=False,
    )  # [1, D, Y, X]
    
    # Expand along Z by repeating and interpolating: [1, D, Z, Y, X]
    # Treat as a 3D volume: unsqueeze Z, repeat, then interpolate
    patch_embed_3d = patch_embed_yx.unsqueeze(2).repeat(1, 1, Z, 1, 1)
    
    # Flatten Z, Y, X → tokens: [1, D, Z*Y*X] → [1, Z*Y*X, D]
    num_tokens = Z * Y * X
    patch_embed_3d = patch_embed_3d.reshape(1, D, num_tokens).permute(0, 2, 1)
    
    # Concatenate CLS + patches
    return torch.cat([cls_embed.float(), patch_embed_3d], dim=1)


def _map_torchvision_to_ctatpt(vit_sd: OrderedDict) -> dict[str, torch.Tensor]:
    """Map torchvision ViT-B/16 key names to our CT-ATPT key names.
    
    Torchvision ViT structure:
        conv_proj.weight                        → patch_embed.weight
        conv_proj.bias                          → patch_embed.bias
        class_token                             → cls_token
        encoder.pos_embedding                   → pos_embed
        encoder.layers.encoder_layer_N.ln_1.*   → blocks.N.norm1.*
        encoder.layers.encoder_layer_N.self_attention.in_proj_weight  → split into q,k,v
        encoder.layers.encoder_layer_N.self_attention.in_proj_bias    → split into q,k,v
        encoder.layers.encoder_layer_N.self_attention.out_proj.*      → blocks.N.attn.out_proj.*
        encoder.layers.encoder_layer_N.ln_2.*   → blocks.N.norm2.*
        encoder.layers.encoder_layer_N.mlp.0.*  → blocks.N.ffn.net.0.*  (Linear1)
        encoder.layers.encoder_layer_N.mlp.3.*  → blocks.N.ffn.net.3.*  (Linear2)
    
    Returns a dict of CT-ATPT keys → tensors.
    """
    mapped = {}
    
    for key, val in vit_sd.items():
        # Patch embedding
        if key == "conv_proj.weight":
            mapped["patch_embed.weight"] = val  # Will be inflated 2D→3D later
        elif key == "conv_proj.bias":
            mapped["patch_embed.bias"] = val
        
        # CLS token
        elif key == "class_token":
            mapped["cls_token"] = val
        
        # Positional embedding
        elif key == "encoder.pos_embedding":
            mapped["pos_embed"] = val  # Will be interpolated later
        
        # Transformer blocks
        elif key.startswith("encoder.layers.encoder_layer_"):
            # Extract layer number
            parts = key.split(".")
            layer_idx = int(parts[2].replace("encoder_layer_", ""))
            rest = ".".join(parts[3:])
            
            # LayerNorm 1 (pre-attention)
            if rest.startswith("ln_1."):
                param = rest.replace("ln_1.", "")
                mapped[f"blocks.{layer_idx}.norm1.{param}"] = val
            
            # Self-attention
            elif rest == "self_attention.in_proj_weight":
                # Combined QKV weight [3*D, D] → split into individual
                D = val.shape[1]
                q, k, v = val.chunk(3, dim=0)
                mapped[f"blocks.{layer_idx}.attn.in_proj_weight"] = val
            
            elif rest == "self_attention.in_proj_bias":
                mapped[f"blocks.{layer_idx}.attn.in_proj_bias"] = val
            
            elif rest.startswith("self_attention.out_proj."):
                param = rest.replace("self_attention.out_proj.", "")
                mapped[f"blocks.{layer_idx}.attn.out_proj.{param}"] = val
            
            # LayerNorm 2 (pre-FFN)
            elif rest.startswith("ln_2."):
                param = rest.replace("ln_2.", "")
                mapped[f"blocks.{layer_idx}.norm2.{param}"] = val
            
            # MLP (FFN)
            # torchvision mlp: Linear(0) → GELU(1) → Dropout(2) → Linear(3) → Dropout(4)
            # Our FFN:         Linear(0) → GELU(1) → Dropout(2) → Linear(3) → Dropout(4)
            elif rest.startswith("mlp."):
                # mlp.0.weight → ffn.net.0.weight (first linear)
                # mlp.3.weight → ffn.net.3.weight (second linear)
                param = rest.replace("mlp.", "ffn.net.")
                mapped[f"blocks.{layer_idx}.{param}"] = val
        
        # Skip: heads.weight, heads.bias (ImageNet classifier — not needed)
        # Skip: encoder.ln.* (final LayerNorm — we use our own in cls_head)
    
    return mapped


def load_imagenet_vit_into_ctatpt(
    model,  # CTATPT instance
    strict: bool = False,
    verbose: bool = True,
) -> list[str]:
    """Load pretrained ViT-B/16 ImageNet weights into a CT-ATPT model.
    
    Loads transformer block weights (attention, FFN, layer norms) directly.
    Adapts patch embedding (2D→3D) and positional embeddings (interpolation).
    Leaves our novel components randomly initialized:
        - CLS+GAP classification head
        - Detection head
        - Pruning parameters (importance_logits, lambda_raw, temperature_raw)
    
    Args:
        model: CTATPT model instance (must have embed_dim=768, depth=12, heads=12)
        strict: If True, raise error on missing keys. Default False.
        verbose: Print loading summary.
    
    Returns:
        List of keys that were successfully loaded.
    """
    config = model.config
    
    # Verify compatibility
    if config.embed_dim != 768 or config.depth != 12 or config.num_heads != 12:
        raise ValueError(
            f"Pretrained ViT-B/16 requires embed_dim=768, depth=12, num_heads=12. "
            f"Got embed_dim={config.embed_dim}, depth={config.depth}, num_heads={config.num_heads}"
        )
    
    # Download weights
    vit_sd = _download_vit_b16_weights()
    
    # Map key names
    mapped = _map_torchvision_to_ctatpt(vit_sd)
    
    # Adapt patch embedding: 2D [768, 3, 16, 16] → 3D [768, 1, 8, 16, 16]
    if "patch_embed.weight" in mapped:
        w2d = mapped["patch_embed.weight"]  # [768, 3, 16, 16]
        # Average RGB channels to get grayscale: [768, 1, 16, 16]
        w2d_gray = w2d.mean(dim=1, keepdim=True)
        pz = config.patch_size[0]
        py, px = config.patch_size[1], config.patch_size[2]
        # Interpolate spatial dims if needed (16×16 → py×px)
        if w2d_gray.shape[2] != py or w2d_gray.shape[3] != px:
            w2d_gray = F.interpolate(
                w2d_gray.float(), size=(py, px), mode="bilinear", align_corners=False
            )
        # Inflate to 3D
        mapped["patch_embed.weight"] = _inflate_2d_to_3d(w2d_gray, pz)
        if verbose:
            print(f"  Patch embed: [768,3,16,16] → [768,1,{pz},{py},{px}] (RGB→gray, 2D→3D)")
    
    # Adapt positional embeddings: 2D grid → 3D grid
    if "pos_embed" in mapped:
        pe2d = mapped["pos_embed"]  # [1, 197, 768] (1 CLS + 14×14 patches)
        grid_2d = (14, 14)  # ViT-B/16 default
        grid_3d = model.grid_shape
        mapped["pos_embed"] = _interpolate_pos_embed_3d(pe2d, grid_2d, grid_3d)
        n_tokens = grid_3d[0] * grid_3d[1] * grid_3d[2]
        if verbose:
            print(f"  Pos embed: [1,197,768] → [1,{1+n_tokens},768] "
                  f"(14×14 → {grid_3d[0]}×{grid_3d[1]}×{grid_3d[2]})")
    
    # Load into model
    model_sd = model.state_dict()
    loaded_keys = []
    skipped_keys = []
    
    for key, val in mapped.items():
        if key in model_sd:
            if model_sd[key].shape == val.shape:
                model_sd[key] = val
                loaded_keys.append(key)
            else:
                skipped_keys.append(
                    f"{key}: shape mismatch (pretrained={val.shape}, model={model_sd[key].shape})"
                )
        else:
            skipped_keys.append(f"{key}: not found in model")
    
    model.load_state_dict(model_sd, strict=False)
    
    # Report
    not_loaded = [k for k in model_sd if k not in loaded_keys]
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Pretrained ViT-B/16 → CT-ATPT Loading Summary")
        print(f"{'='*60}")
        print(f"  Loaded:          {len(loaded_keys)} parameters")
        print(f"  Skipped:         {len(skipped_keys)} parameters")
        print(f"  Random init:     {len(not_loaded)} parameters (our novel components)")
        print(f"\n  Novel components (randomly initialized):")
        # Group by prefix for cleaner output
        novel_prefixes = set()
        for k in not_loaded:
            prefix = k.split(".")[0]
            novel_prefixes.add(prefix)
        for prefix in sorted(novel_prefixes):
            keys_in_prefix = [k for k in not_loaded if k.startswith(prefix)]
            print(f"    {prefix}: {len(keys_in_prefix)} params")
        print(f"{'='*60}")
    
    if strict and skipped_keys:
        raise RuntimeError(f"Failed to load keys:\n" + "\n".join(skipped_keys))
    
    return loaded_keys
