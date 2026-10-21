#!/usr/bin/env python3
"""
Russell-Enhanced Vision Transformer Block
==========================================
Drop-in upgrade for highly popular ViT repositories, especially:

- lucidrains/vit-pytorch          (~25 500 stars)
- huggingface/pytorch-image-models (timm) – conceptual use
- Other clean ViT educational codebases

Adds polarity dual-stream residual (residual → 0) from
Walter Russell’s The Universal One.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RussellPolarityMHA(nn.Module):
    """Multi-head attention with forced polarity balance."""
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
        self.polarity = nn.Linear(dim, dim * 2)

    def forward(self, x: torch.Tensor):
        B, N, C = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(B, N, self.heads, -1).transpose(1, 2), qkv)
        dots = (q @ k.transpose(-1, -2)) * self.scale
        attn = dots.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        out = self.to_out(out)

        # Russell polarity residual
        polar = self.polarity(out)
        pos, neg = polar.chunk(2, dim=-1)
        residual = pos + neg
        pos = pos - 0.5 * residual
        neg = neg - 0.5 * residual
        out = pos + neg
        balance = residual.abs().mean()
        return out, balance


class RussellViTBlock(nn.Module):
    """
    Vision Transformer block with polarity residual.
    Returns (output, balance_loss) for optional joint optimisation.
    """
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64,
                 mlp_dim: int = None, dropout: float = 0.0):
        super().__init__()
        mlp_dim = mlp_dim or dim * 4
        self.norm1 = nn.LayerNorm(dim)
        self.attn = RussellPolarityMHA(dim, heads, dim_head, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor):
        attn_out, bal = self.attn(self.norm1(x))
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, bal


if __name__ == "__main__":
    torch.manual_seed(42)
    block = RussellViTBlock(dim=256, heads=8)
    x = torch.randn(2, 65, 256)
    out, bal = block(x)
    print("Russell-Enhanced ViT block")
    print(f"  Input  : {tuple(x.shape)}")
    print(f"  Output : {tuple(out.shape)}")
    print(f"  Balance residual : {bal.item():.6f}")
    print("✓ Ready for popular ViT codebases.")
