"""Jet transverse momentum density p(log pT_J | theta).

The jet pT is a known observation (measured context) for likelihood evaluation,
and its distribution over theta (inferred parameters) carries much of the
quenching signal. We model it as a conditional mixture of Gaussians over
log(pT/GeV), conditioned on the normalized unknown parameters.
"""

import copy
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from omnilearned.sbi.parameters import ConditionLayout

# Recommended number of mixture components
DEFAULT_N_COMPONENTS = 5


def layout_with_jet_pt(unknown, lower, upper, units=None):
    """Condition layout for p(log pT | theta).

    Args:
        unknown: list of unknown parameter names.
        lower, upper: bounds for unknown parameters.
        units: optional dict of column units.

    Returns:
        ConditionLayout with normalized unknown parameters.
    """
    lay = ConditionLayout.from_bounds(unknown, lower, upper, units=units)
    return lay


class JetPtModel(nn.Module):
    """Conditional mixture of Gaussians for log(pT_J / GeV).

    The model maps normalized theta to mixture parameters (logits, means, log-std),
    then evaluates the likelihood under a mixture of Gaussians. It supports
    weighted maximum likelihood fitting to capture weighted empirical distributions.
    """

    def __init__(self, cond_dim, n_components=DEFAULT_N_COMPONENTS, hidden=128, depth=3):
        """Initialize the jet pT model.

        Args:
            cond_dim: dimension of the normalized condition (unknown parameters).
            n_components: number of Gaussian mixture components (default 5).
            hidden: hidden layer width (default 128).
            depth: number of hidden layers (default 3).
        """
        super().__init__()
        self.config = dict(cond_dim=cond_dim, n_components=n_components,
                          hidden=hidden, depth=depth)
        self.n_components = n_components

        # Build MLP: cond_dim -> hidden -> hidden -> (n_components, 2*n_components)
        # Outputs: mixture logits [n_components], means [n_components],
        # log standard deviations [n_components]
        layers, d = [], cond_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.SiLU()]
            d = hidden
        # Final layer outputs: logits (n_comp), means (n_comp), log_stds (n_comp)
        layers.append(nn.Linear(d, 3 * n_components))
        self.net = nn.Sequential(*layers)
        self.layout = None  # ConditionLayout over the unknown parameters (set by owner)

    def forward(self, cond):
        """Compute mixture parameters from condition.

        Args:
            cond: [B, cond_dim] normalized condition.

        Returns:
            Tuple of:
            - logits: [B, n_components] mixture weights (before softmax).
            - means: [B, n_components] component means in log pT space.
            - log_stds: [B, n_components] component log standard deviations.
        """
        out = self.net(cond)
        logits = out[:, :self.n_components]
        means = out[:, self.n_components:2*self.n_components]
        log_stds = out[:, 2*self.n_components:]
        return logits, means, log_stds

    def log_prob(self, log_pt, cond):
        """Evaluate log p(log_pt | cond) under mixture of Gaussians.

        Args:
            log_pt: [B] or [B, 1] log transverse momenta (log(pT/GeV)).
            cond: [B, cond_dim] normalized condition.

        Returns:
            [B] log probabilities.
        """
        log_pt = torch.as_tensor(log_pt, dtype=torch.float32, device=cond.device).squeeze(-1)
        logits, means, log_stds = self.forward(cond.float())

        # Ensure broadcasting-compatible shapes
        # logits: [B, K], means: [B, K], log_stds: [B, K]
        # log_pt: [B] -> [B, 1] for broadcasting
        log_pt = log_pt.unsqueeze(-1)  # [B, 1]

        # Gaussian log likelihood: -0.5 * ((x - mu) / sigma)^2 - log(sigma) - 0.5 * log(2*pi)
        stds = torch.exp(log_stds)  # [B, K]
        gaussian_ll = -0.5 * ((log_pt - means) / stds) ** 2 - log_stds - 0.5 * math.log(2 * math.pi)
        # [B, K]

        # Mixture: log sum_k exp(log pi_k + log N(x | mu_k, sigma_k))
        log_weights = F.log_softmax(logits.double(), -1)  # [B, K]
        log_ll = torch.logsumexp(log_weights + gaussian_ll.double(), -1)  # [B]
        return log_ll.float()

    @torch.no_grad()
    def sample(self, cond, generator=None):
        """Sample from p(log pT | cond).

        Args:
            cond: [B, cond_dim] normalized condition.
            generator: optional torch.Generator for reproducibility.

        Returns:
            [B] sampled log(pT/GeV) values.
        """
        logits, means, log_stds = self.forward(cond.float())
        batch_size = logits.shape[0]

        # Sample mixture component index
        weights = F.softmax(logits.float(), -1).cpu()  # [B, K]
        comp_idx = torch.multinomial(weights, 1, generator=generator)[:, 0]  # [B]

        # Sample from the selected Gaussian component
        means_cpu = means.float().cpu()
        stds_cpu = torch.exp(log_stds).float().cpu()
        z = torch.randn(batch_size, generator=generator)
        samples = means_cpu[torch.arange(batch_size), comp_idx] + stds_cpu[torch.arange(batch_size), comp_idx] * z
        return samples

    # ---- persistence (inside a checkpoint) ------------------------------------
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


def train_jet_pt(model, cond, log_pt, val_cond, val_log_pt, steps=4000,
                batch_size=4096, lr=1e-3, weight_decay=1e-5, seed=0, device="cpu",
                val_every=250, log=print, weights=None, val_weights=None):
    """Fit p(log pT | theta) by maximum likelihood.

    Args:
        model: JetPtModel instance.
        cond, val_cond: [N, C] float tensors of normalized unknown parameters.
        log_pt, val_log_pt: [N] log(pT/GeV) values.
        weights, val_weights: optional [N] per-event sampling weights; the NLL is then
            the weighted mean sum(w nll) / sum(w). Required for accurate fitting
            when the sample is biased (e.g. hard-pT weighted JEWEL).
        steps: number of training steps.
        batch_size: minibatch size.
        lr: learning rate.
        weight_decay: L2 regularization.
        seed: random seed for reproducibility.
        device: computation device ("cpu" or "cuda").
        val_every: validation frequency (steps between reports).
        log: logging function.

    Returns:
        Tuple of:
        - best model on CPU with evaluation mode set.
        - history dict with keys "step", "train_nll", "val_nll", "best_val_nll", "best_step".
    """
    torch.manual_seed(seed)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    dev = torch.device(device)
    model = model.to(dev).float()
    cond = cond.float().to(dev)
    log_pt = torch.as_tensor(log_pt, dtype=torch.float32).to(dev)
    val_cond = val_cond.float().to(dev)
    val_log_pt = torch.as_tensor(val_log_pt, dtype=torch.float32).to(dev)
    w = None if weights is None else torch.as_tensor(weights, dtype=torch.float32).to(dev)
    vw = None if val_weights is None else torch.as_tensor(val_weights, dtype=torch.float32).to(dev)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    hist = {"step": [], "train_nll": [], "val_nll": []}
    best, best_state, running, start = (math.inf, -1), None, 0.0, time.time()

    for step in range(steps):
        model.train()
        idx = torch.randint(0, log_pt.shape[0], (min(batch_size, log_pt.shape[0]),),
                           generator=gen).to(dev)
        lp = model.log_prob(log_pt[idx], cond[idx])
        loss = -lp.mean() if w is None else -(lp * w[idx]).sum() / w[idx].sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        running += float(loss.detach())

        if (step + 1) % val_every == 0 or step + 1 == steps:
            model.eval()
            with torch.no_grad():
                vlp = model.log_prob(val_log_pt, val_cond)
                vl = float(-vlp.mean() if vw is None else -(vlp * vw).sum() / vw.sum())
            k = (step + 1) % val_every or val_every
            hist["step"].append(step + 1)
            hist["train_nll"].append(running / k)
            hist["val_nll"].append(vl)
            running = 0.0
            log(f"jet_pt step {step + 1:6d} train {hist['train_nll'][-1]:.4f} "
                f"val {vl:.4f} {time.time() - start:.0f}s")
            if vl < best[0]:
                best = (vl, step + 1)
                best_state = copy.deepcopy({k: v.detach().cpu()
                                           for k, v in model.state_dict().items()})

    hist["best_val_nll"], hist["best_step"] = best
    out = copy.deepcopy(model).cpu()
    if best_state is not None:
        out.load_state_dict(best_state)
    out.layout = model.layout
    out.eval()
    for p in out.parameters():
        p.requires_grad_(False)
    return out, hist
