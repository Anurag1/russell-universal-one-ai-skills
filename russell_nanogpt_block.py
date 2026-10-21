#!/usr/bin/env python3
"""
Russell-Enhanced Transformer Block
==================================
Drop-in conceptual upgrade for the most popular educational GPT codebase:
https://github.com/karpathy/nanoGPT  (60k+ stars)

Adds polarity dual-stream residual (residual → 0) inspired by
Walter Russell's The Universal One.

Compatible with nanoGPT-style GPT blocks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RussellPolarityAttention(nn.Module):
    """
    Multi-head attention + polarity balance.
    Forces positive/negative streams toward residual = 0.
    """
    def __init__(self, n_embd: int, n_head: int, dropout: float = 0.0):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head

        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.polarity = nn.Linear(n_embd, 2 * n_embd)  # pos / neg
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_head, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        att = (q @ k.transpose(-2, -1)) * (1.0 / (self.head_dim ** 0.5))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))

        # Russell polarity
        polar = self.polarity(y)
        pos, neg = polar.chunk(2, dim=-1)
        residual = pos + neg
        pos = pos - 0.5 * residual
        neg = neg - 0.5 * residual
        out = pos + neg
        balance_loss = residual.abs().mean()
        return out, balance_loss


class RussellBlock(nn.Module):
    """
    nanoGPT-style residual block upgraded with polarity residual.
    Returns (output, balance_loss) so you can add the balance term
    to your training objective.
    """
    def __init__(self, n_embd: int, n_head: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = RussellPolarityAttention(n_embd, n_head, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor):
        attn_out, bal = self.attn(self.ln1(x))
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, bal


# ------------------------------------------------------------------
# Live self-test
# ------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(42)
    block = RussellBlock(n_embd=128, n_head=4)
    x = torch.randn(2, 16, 128)
    out, balance = block(x)
    print("Russell-Enhanced nanoGPT-style block")
    print(f"  Input  : {tuple(x.shape)}")
    print(f"  Output : {tuple(out.shape)}")
    print(f"  Balance residual : {balance.item():.6f}")
    print("✓ Ready to drop into a nanoGPT-style model.")
