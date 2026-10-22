"""Constituent multiplicity model p(N | theta) for variable-size clouds.

With variable multiplicity the per-event likelihood factorizes as

    q(x, N | theta) = p(N | theta) q(x | N, theta),

where the flow gives q(x | N, theta) over the N real (masked-in) constituents
and N is also a known condition of the flow (column ``n_constituents``).
``MultiplicityModel`` is a categorical distribution over N = 0..max_n whose
logits are a smooth (SiLU) MLP of the normalized inferred parameters, so
log p(N | theta) is a smooth function of theta, as the grid likelihood needs.
It conditions on the inferred parameters only, never on N or other context.
"""

import copy
import math
import time

import numpy as np
import torch
import torch.nn as nn

from omnilearned.sbi.parameters import ConditionLayout

MULTIPLICITY_COLUMN = "n_constituents"


def layout_with_multiplicity(unknown, lower, upper, scale=100.0, units=None):
    """Flow condition layout: unknowns in [-1, 1] plus N / ``scale`` as known context."""
    lay = ConditionLayout.from_bounds(unknown, lower, upper, known=(MULTIPLICITY_COLUMN,),
                                      units=units)
    lay.scale[MULTIPLICITY_COLUMN] = float(scale)
    return lay


def multiplicity_context(mask):
    """Known-context column [N, 1] (real constituent count) from a mask."""
    m = np.asarray(mask, dtype=bool)
    return m.sum(1, keepdims=True).astype(np.float64)


class MultiplicityModel(nn.Module):
    """Categorical p(N | theta) over N = 0..max_n with smooth logits."""

    def __init__(self, cond_dim, max_n, hidden=128, depth=3):
        super().__init__()
        self.config = dict(cond_dim=cond_dim, max_n=max_n, hidden=hidden, depth=depth)
        layers, d = [], cond_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.SiLU()]
            d = hidden
        layers.append(nn.Linear(d, max_n + 1))
        self.net = nn.Sequential(*layers)
        self.layout = None  # ConditionLayout over the unknown parameters (set by the owner)

    @property
    def max_n(self):
        return self.config["max_n"]

    def logits(self, cond):
        return self.net(cond)

    def log_prob(self, n, cond):
        """log p(N = n | cond) per row; n outside 0..max_n gives -inf."""
        n = torch.as_tensor(n, device=cond.device).long()
        lp = torch.log_softmax(self.logits(cond).double(), -1)
        inside = (n >= 0) & (n <= self.max_n)
        out = lp.gather(1, n.clamp(0, self.max_n)[:, None])[:, 0]
        return torch.where(inside, out, torch.full_like(out, -math.inf))

    @torch.no_grad()
    def sample(self, cond, generator=None):
        p = torch.softmax(self.logits(cond).double(), -1).cpu()
        return torch.multinomial(p, 1, generator=generator)[:, 0]

    # ---- persistence (inside a FlowSurrogate checkpoint) ------------------
    def state(self):
        return {"config": dict(self.config),
                "model_state": {k: v.detach().cpu() for k, v in self.state_dict().items()},
                "layout": None if self.layout is None else self.layout.to_dict()}

    @classmethod
    def from_state(cls, state, device="cpu"):
        m = cls(**state["config"])
        m.load_state_dict(state["model_state"])
        if state.get("layout") is not None:
            m.layout = ConditionLayout.from_dict(state["layout"])
        m.to(device).eval()
        for p in m.parameters():
            p.requires_grad_(False)
        return m


def train_multiplicity(model, cond, n, val_cond, val_n, steps=4000, batch_size=4096,
                       lr=1e-3, weight_decay=1e-5, seed=0, device="cpu", val_every=250,
                       log=print):
    """Fit by maximum likelihood (cross entropy); keep the best validation state.

    Args:
        cond, val_cond: [N, C] float tensors of normalized unknown parameters.
        n, val_n: [N] integer multiplicities.
    Returns:
        (best model on CPU, history)
    """
    torch.manual_seed(seed)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    dev = torch.device(device)
    model = model.to(dev).float()
    cond, n = cond.float().to(dev), torch.as_tensor(n).long().to(dev)
    val_cond, val_n = val_cond.float().to(dev), torch.as_tensor(val_n).long().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    hist = {"step": [], "train_nll": [], "val_nll": []}
    best, best_state, running, start = (math.inf, -1), None, 0.0, time.time()
    for step in range(steps):
        model.train()
        idx = torch.randint(0, n.shape[0], (min(batch_size, n.shape[0]),), generator=gen).to(dev)
        loss = -model.log_prob(n[idx], cond[idx]).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        running += float(loss.detach())
        if (step + 1) % val_every == 0 or step + 1 == steps:
            model.eval()
            with torch.no_grad():
                vl = float(-model.log_prob(val_n, val_cond).mean())
            k = (step + 1) % val_every or val_every
            hist["step"].append(step + 1)
            hist["train_nll"].append(running / k)
            hist["val_nll"].append(vl)
            running = 0.0
            log(f"multiplicity step {step + 1:6d} train {hist['train_nll'][-1]:.4f} "
                f"val {vl:.4f} {time.time() - start:.0f}s")
            if vl < best[0]:
                best = (vl, step + 1)
                best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
    hist["best_val_nll"], hist["best_step"] = best
    out = copy.deepcopy(model).cpu()
    if best_state is not None:
        out.load_state_dict(best_state)
    out.layout = model.layout
    out.eval()
    for p in out.parameters():
        p.requires_grad_(False)
    return out, hist


def poisson_reference(mean, max_n):
    """log Poisson pmf over 0..max_n (test helper and sanity reference)."""
    k = np.arange(max_n + 1)
    return k * math.log(mean) - mean - np.array([math.lgamma(v + 1) for v in k])
