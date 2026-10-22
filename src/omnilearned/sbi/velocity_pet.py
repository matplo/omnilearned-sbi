"""PET2-style velocity with smooth locality (density-compatible).

Upstream ``PET_body`` combines (i) a local embedding over the K nearest
neighbours chosen by a hard ``topk``, (ii) pairwise interaction features
(log m², log ΔR, log kT) as attention biases and (iii) class, condition and
time tokens in a global transformer. The hard ``topk`` makes the velocity
only piecewise smooth, and the log features are singular at coincident
particles and use a hard min; the divergence along a density trajectory then
jumps (``smoothness.py``). ``SmoothPETVelocity`` keeps the PET2 structure but
makes every part smooth in the particle coordinates:

* local embedding: messages from every other real particle, weighted by a
  softmax of a Gaussian distance kernel −ΔR²/(2 s²) (learnable s) instead of
  the K nearest;
* interaction features: Δη, Δφ, ½ log(ΔR² + ε²), the sum of log pT and a soft
  minimum of log pT (−logaddexp(−a, −b)), mapped by an MLP to one attention
  bias per head, plus a Gaussian locality bias −ΔR²/(2 s_h²) per head;
* class tokens, a condition token and a time token as in PET2, and a context
  vector (time ⊕ condition) added to every particle token;
* pre-norm transformer blocks with explicit attention biases; padded keys get
  a constant −1e9, so padded slots never influence real ones.

Coordinates are the model coordinates (standardized atanh η, φ and log pT),
so "ΔR" is a smooth distance in those units; the network learns the scale.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from omnilearned.sbi.velocity import FourierTime

NEG = -1e9
PAIR_FEATURES = 5


def pair_features(x, eps):
    """Smooth pairwise features [B, P, P, 5] and squared distance [B, P, P]."""
    d_eta = x[:, :, None, 0] - x[:, None, :, 0]
    d_phi = x[:, :, None, 1] - x[:, None, :, 1]
    d2 = d_eta.square() + d_phi.square()
    lpt_i, lpt_j = x[:, :, None, 2], x[:, None, :, 2]
    soft_min = -torch.logaddexp(-lpt_i, -lpt_j)
    feats = torch.stack([d_eta, d_phi, 0.5 * torch.log(d2 + eps * eps),
                         (lpt_i + lpt_j).expand_as(d2), soft_min.expand_as(d2)], -1)
    return feats, d2


class BiasedAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, h, bias):
        B, T, D = h.shape
        q, k, v = self.qkv(h).reshape(B, T, 3, self.heads, D // self.heads).unbind(2)
        q, k, v = (a.transpose(1, 2) for a in (q, k, v))
        scores = q @ k.transpose(-1, -2) / math.sqrt(D // self.heads) + bias
        out = (torch.softmax(scores, -1) @ v).transpose(1, 2).reshape(B, T, D)
        return self.proj(out)


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=2):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = BiasedAttention(dim, heads)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_ratio * dim), nn.GELU(),
                                 nn.Linear(mlp_ratio * dim, dim))
        self.gamma1 = nn.Parameter(1e-3 * torch.ones(dim))  # LayerScale as in PET2
        self.gamma2 = nn.Parameter(1e-3 * torch.ones(dim))

    def forward(self, h, bias, mask_f):
        h = h + self.gamma1 * self.attn(self.norm1(h), bias) * mask_f
        return h + self.gamma2 * self.mlp(self.norm2(h)) * mask_f


class SmoothLocalEmbedding(nn.Module):
    """Gaussian-kernel weighted messages from the other real particles."""

    def __init__(self, num_features, dim, local_dim, scale):
        super().__init__()
        self.msg = nn.Sequential(nn.Linear(num_features + PAIR_FEATURES, local_dim), nn.GELU(),
                                 nn.Linear(local_dim, local_dim), nn.GELU())
        self.out = nn.Linear(local_dim, dim)
        self.log_scale = nn.Parameter(torch.tensor(math.log(scale)))

    def forward(self, x, feats, d2, key_bias):
        B, P, _ = x.shape
        rel = x[:, :, None, :] - x[:, None, :, :]
        m = self.msg(torch.cat([rel, feats], -1))  # [B, P, P, local]
        logits = -d2 / (2 * torch.exp(2 * self.log_scale)) + key_bias
        eye = torch.eye(P, dtype=torch.bool, device=x.device)
        logits = logits.masked_fill(eye, NEG)  # exclude self (constant)
        w = torch.softmax(logits, -1)
        return self.out((w.unsqueeze(-1) * m).sum(2))


class SmoothPETVelocity(nn.Module):
    """Smooth PET2-style conditional velocity (see module docstring)."""

    def __init__(self, num_features, cond_dim, dim=256, depth=6, heads=8, mlp_ratio=2,
                 num_tokens=4, local_dim=64, local_scale=0.5, pair_eps=1e-2, time_dim=64, time_max_freq=64.0):
        super().__init__()
        self.config = dict(architecture="pet_smooth", num_features=num_features,
                           cond_dim=cond_dim, dim=dim, depth=depth, heads=heads,
                           mlp_ratio=mlp_ratio, num_tokens=num_tokens, local_dim=local_dim,
                           local_scale=local_scale, pair_eps=pair_eps, time_dim=time_dim,
                           time_max_freq=time_max_freq)
        self.heads, self.num_tokens, self.pair_eps = heads, num_tokens, pair_eps
        self.embed = nn.Sequential(nn.Linear(num_features, mlp_ratio * dim), nn.GELU(),
                                   nn.Linear(mlp_ratio * dim, dim))
        self.local = SmoothLocalEmbedding(num_features, dim, local_dim, local_scale)
        self.interaction = nn.Sequential(nn.Linear(PAIR_FEATURES, 2 * dim // heads), nn.GELU(),
                                         nn.Linear(2 * dim // heads, heads))
        self.log_head_scale = nn.Parameter(
            torch.log(torch.linspace(0.25, 2.0, heads)))  # Gaussian locality per head
        self.time = FourierTime(time_dim, time_max_freq)
        self.time_token = nn.Sequential(nn.Linear(time_dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.cond_token = nn.Sequential(nn.Linear(cond_dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.context = nn.Sequential(nn.Linear(time_dim + cond_dim, dim), nn.GELU(),
                                     nn.Linear(dim, dim))
        self.tokens = nn.Parameter(0.02 * torch.randn(1, num_tokens, dim))
        self.blocks = nn.ModuleList([Block(dim, heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, num_features)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, t, cond, mask):
        mask = mask.bool()
        B, P, _ = x.shape
        mask_f = mask.to(x.dtype).unsqueeze(-1)
        x = x * mask_f
        if t.dim() == 0:
            t = t.expand(B)
        tf = self.time(t).to(x.dtype)
        cond = cond.to(x.dtype)
        key_bias = torch.where(mask[:, None, :], 0.0, NEG).to(x.dtype)  # [B, 1, P]
        feats, d2 = pair_features(x, self.pair_eps)

        h = self.embed(x) + self.local(x, feats, d2, key_bias) + self.context(
            torch.cat([tf, cond], -1))[:, None]
        h = h * mask_f
        extra = torch.cat([self.tokens.expand(B, -1, -1).to(x.dtype),
                           self.cond_token(cond)[:, None], self.time_token(tf)[:, None]], 1)
        h = torch.cat([extra, h], 1)
        E = extra.shape[1]
        full_mask_f = torch.cat([torch.ones(B, E, 1, dtype=x.dtype, device=x.device), mask_f], 1)

        # Pair bias on particle-particle entries; tokens see and are seen by all real slots.
        pb = self.interaction(feats).permute(0, 3, 1, 2)  # [B, H, P, P]
        scale2 = torch.exp(2 * self.log_head_scale).to(x.dtype)[None, :, None, None]
        pb = pb - d2[:, None] / (2 * scale2)
        bias = torch.zeros(B, self.heads, E + P, E + P, dtype=x.dtype, device=x.device)
        bias[:, :, E:, E:] = pb
        bias = bias + torch.cat([torch.zeros(B, 1, E, dtype=x.dtype, device=x.device),
                                 key_bias], -1)[:, None]

        for blk in self.blocks:
            h = blk(h, bias, full_mask_f)
        return self.out(self.norm(h[:, E:])) * mask_f
