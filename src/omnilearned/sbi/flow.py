"""Single-process conditional flow-matching training and checkpoint bundle.

Training, generation and likelihood evaluation share the checkpoint's base
specification, feature transform, condition layout and explicit masks. The
trainer runs on one explicitly selected device (CPU, CUDA or MPS) without
distributed setup; model tensors are float32.

Variable multiplicity: clouds are padded (zeros, explicit mask) to a fixed
width such as 256. Every computation is cut to the active width (the last
real slot in the batch), which changes no value because padded slots never
influence real ones. With a multiplicity model the checkpoint represents
q(x, N | theta) = p(N | theta) q(x | N, theta): ``log_prob`` adds
log p(N | theta), the known condition column ``n_constituents`` is filled from
the mask, and ``sample`` draws N first.
"""

import copy
import hashlib
import io
import json
import math
import time
from dataclasses import dataclass, asdict

import numpy as np
import torch

from omnilearned.sbi.base import base_from_dict
from omnilearned.sbi.device import ieee_float32, resolve_device, synchronize
from omnilearned.sbi.likelihood import (
    frozen_for_inference,
    log_prob,
    sample,
)
from omnilearned.sbi.multiplicity import (
    MULTIPLICITY_COLUMN,
    MultiplicityModel,
    multiplicity_context,
)
from omnilearned.sbi.parameters import ConditionLayout
from omnilearned.sbi.transforms import FeatureTransform
from omnilearned.sbi.velocity import build_velocity, velocity_config


CHECKPOINT_SCHEMA = "omnilearned-sbi-flow-v1"


@dataclass
class TrainOptions:
    steps: int = 2000
    batch_size: int = 256
    lr: float = 3e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 200
    ema_decay: float = 0.999
    grad_clip: float = 1.0
    val_every: int = 250
    val_batches: int = 8
    seed: int = 0
    val_seed: int = 1
    device: str = "auto"
    allow_device_fallback: bool = False
    # 0: uniform minibatches. k > 1: each minibatch is drawn uniformly from one
    # of k equal-count buckets of the real multiplicity (bucket chosen with
    # probability proportional to its size, so every event keeps the same
    # overall probability); batches are cut to the active width, so short
    # clouds no longer pay for the longest one in the set.
    length_buckets: int = 0

    def to_dict(self):
        return asdict(self)


def active_width(mask):
    """Number of leading slots that hold any real particle in the batch."""
    cols = mask.reshape(-1, mask.shape[-1]).any(0)
    idx = torch.nonzero(cols)
    return int(idx.max()) + 1 if idx.numel() else 1


def flow_matching_loss(model, x, mask, cond, base, generator=None, t=None, eps=None):
    """Masked conditional flow-matching loss (mean over real coordinates).

    Padded slots beyond the batch's active width are cut first (no value changes).
    """
    w = active_width(mask)
    x, mask = x[:, :w], mask[:, :w]
    if eps is not None:
        eps = eps[:, :w]
    mask_f = mask.to(x.dtype).unsqueeze(-1)
    if t is None:
        t = torch.rand(x.shape[0], device=x.device, generator=generator)
    if eps is None:
        eps = base.sample_like(x, mask, generator)
    tt = t.view(-1, 1, 1)
    z = ((1 - tt) * x + tt * eps) * mask_f
    target = (eps - x) * mask_f
    v = model(z, t, cond, mask)
    err = ((v - target) ** 2 * mask_f).reshape(x.shape[0], -1).sum(-1)
    ncoord = mask_f.reshape(x.shape[0], -1).sum(-1) * x.shape[-1]
    return (err / ncoord.clamp(min=1)).mean()


class FlowSurrogate:
    """A conditional generator O with everything needed to use it."""

    def __init__(self, model, base, transform, layout, metadata=None, device="cpu",
                 multiplicity=None):
        self.model = model
        self.base = base
        self.transform = transform
        self.layout = layout
        self.metadata = dict(metadata or {})
        self.device = torch.device(device)
        self.multiplicity = multiplicity
        if multiplicity is not None:
            multiplicity.to(self.device)

    @property
    def uses_multiplicity_context(self):
        return MULTIPLICITY_COLUMN in self.layout.known

    def _context(self, mask, context, context_names):
        """Known context rows, with ``n_constituents`` derived from the mask if needed."""
        if not self.uses_multiplicity_context:
            return context, context_names
        n = multiplicity_context(mask)
        if context is None:
            return n, [MULTIPLICITY_COLUMN]
        names = list(context_names or [c for c in self.layout.known if c != MULTIPLICITY_COLUMN])
        return np.concatenate([np.asarray(context, np.float64), n], 1), names + [MULTIPLICITY_COLUMN]

    def log_multiplicity(self, counts, candidates, candidate_names=None):
        """log p(N | theta) [M, len(counts)] for candidate rows (0 without a model)."""
        candidates = np.atleast_2d(candidates)
        counts = np.asarray(counts)
        if self.multiplicity is None:
            return np.zeros((candidates.shape[0], counts.size))
        lay = self.multiplicity.layout
        cond = torch.from_numpy(lay.assemble(candidates, candidate_names)).to(self.device)
        with torch.no_grad():
            lp = torch.log_softmax(self.multiplicity.logits(cond).double(), -1).cpu().numpy()
        n = np.clip(counts, 0, self.multiplicity.max_n)
        out = lp[:, n]
        out[:, (counts < 0) | (counts > self.multiplicity.max_n)] = -np.inf
        return out

    @staticmethod
    def _trim(x, mask):
        m = np.asarray(mask, dtype=bool)
        w = active_width(torch.from_numpy(m))
        return np.asarray(x)[:, :w], m[:, :w]

    # ---- density --------------------------------------------------------
    @ieee_float32()
    def log_prob(self, x_phys, mask, candidates, event_ids, options,
                 candidate_names=None, context=None):
        """Per-(event, candidate) physical-coordinate log densities.

        Args:
            x_phys: [N, P, F] float numpy/tensor observations.
            candidates: [M, n_unknown] candidate parameter values.
        Returns:
            (float64 array [M, N], diagnostics)
        """
        x_phys, mask = self._trim(x_phys, mask)
        context, context_names = self._context(mask, context, None)
        x = torch.as_tensor(np.asarray(x_phys), dtype=torch.float32)
        mask_t = torch.as_tensor(np.asarray(mask), dtype=torch.bool)
        u, logj = self.transform.forward(x.double(), mask_t)
        u = u.float()
        candidates = np.atleast_2d(candidates)
        M, N = candidates.shape[0], x.shape[0]
        ctx = None
        if context is not None:
            ctx = np.repeat(np.asarray(context)[None], M, 0).reshape(M * N, -1)
        cond = self.layout.assemble(
            np.repeat(candidates, N, axis=0), candidate_names, context=ctx,
            context_names=context_names,
        )
        xs = u.repeat(M, 1, 1).to(self.device)
        ms = mask_t.repeat(M, 1).to(self.device)
        ids = np.tile(np.asarray(event_ids), M)
        with frozen_for_inference(self.model):
            lp, diag = log_prob(
                self.model,
                xs,
                ms,
                self.base,
                cond=torch.from_numpy(cond).to(self.device),
                options=options,
                event_ids=ids,
            )
        lp = lp + logj.repeat(M)
        out = lp.numpy().reshape(M, N)
        if self.multiplicity is not None:
            lm = self.log_multiplicity(mask_t.sum(1).numpy(), candidates, candidate_names)
            diag["log_multiplicity"] = lm
            out = out + lm
        return out, diag

    @ieee_float32()
    def log_prob_paired(self, x_phys, mask, rows, event_ids, options,
                        candidate_names=None, context=None):
        """Log density of event i at its own parameter row ``rows[i]`` -> [N]."""
        x_phys, mask = self._trim(x_phys, mask)
        context, context_names = self._context(mask, context, None)
        x = torch.as_tensor(np.asarray(x_phys), dtype=torch.float32)
        mask_t = torch.as_tensor(np.asarray(mask), dtype=torch.bool)
        u, logj = self.transform.forward(x.double(), mask_t)
        cond = self.layout.assemble(np.asarray(rows), candidate_names, context=context,
                                    context_names=context_names)
        with frozen_for_inference(self.model):
            lp, diag = log_prob(self.model, u.float().to(self.device),
                                mask_t.to(self.device), self.base,
                                cond=torch.from_numpy(cond).to(self.device),
                                options=options, event_ids=np.asarray(event_ids))
        out = (lp + logj).numpy()
        if self.multiplicity is not None:
            counts = mask_t.sum(1).numpy()
            rows = np.atleast_2d(rows)
            lm = np.array([self.log_multiplicity(counts[i:i + 1], rows[i], candidate_names)[0, 0]
                           for i in range(len(counts))])
            diag["log_multiplicity"] = lm
            out = out + lm
        return out, diag

    # ---- generation -----------------------------------------------------
    @ieee_float32()
    def sample(self, candidate, n, num_particles=None, seed=0, nsteps=64, solver="midpoint",
               candidate_names=None, batch=4096, pad_to=None, return_mask=False,
               sort_by_length=False):
        """Physical-coordinate samples at one candidate.

        ``num_particles`` fixes the multiplicity (returns ``x`` [n, P, F], or
        ``(x, mask)`` with ``return_mask``); an array of n counts fixes N per cloud
        (returns ``(x, mask)``, as for drawn N). With ``num_particles=None`` N is
        drawn from the multiplicity model and ``(x, mask)`` is returned:
        left-packed, zero-padded to ``pad_to`` (default: the largest N drawn).
        ``sort_by_length`` integrates the clouds in order of N (each batch is cut
        to its largest N, so this saves most of the padded work) and returns them
        in draw order; the base noise is then assigned in sorted order, so the
        samples differ from the unsorted ones (same distribution, same N).
        """
        F = self.base.num_features
        gen = torch.Generator().manual_seed(int(seed))
        cand = np.atleast_2d(candidate)
        if num_particles is None:
            if self.multiplicity is None:
                raise ValueError("variable multiplicity needs a multiplicity model")
            mcond = torch.from_numpy(
                self.multiplicity.layout.assemble(np.repeat(cand, n, 0), candidate_names)
            ).to(self.device)
            counts = self.multiplicity.sample(mcond, gen).numpy()
        elif np.ndim(num_particles):
            counts = np.asarray(num_particles, dtype=np.int64)
            if counts.shape != (n,) or counts.min(initial=1) < 1:
                raise ValueError("num_particles needs n positive counts")
        else:
            counts = np.full(n, int(num_particles))
        variable = num_particles is None or np.ndim(num_particles) > 0
        width = int(pad_to or max(1, counts.max() if n else 1))
        if counts.max(initial=0) > width:
            raise ValueError("pad_to is smaller than a drawn multiplicity")
        order = None
        if sort_by_length and variable:
            order = np.argsort(counts, kind="stable")
            counts = counts[order]
        xs, ms = [], []
        with frozen_for_inference(self.model):
            for s in range(0, n, batch):
                m = min(batch, n - s)
                cnt = torch.as_tensor(counts[s:s + m])
                w = max(1, int(cnt.max()))
                mask = torch.arange(w)[None, :] < cnt[:, None]
                noise = self.base.sample_like(torch.zeros(m, w, F), mask, gen)
                context, context_names = self._context(mask.numpy(), None, None)
                cond = self.layout.assemble_torch(
                    np.repeat(cand, m, 0), self.device,
                    candidate_names=candidate_names, context=context,
                    context_names=context_names,
                )
                u = sample(self.model, self.base, (m, w, F), mask,
                           cond=cond, nsteps=nsteps, solver=solver,
                           device=self.device, noise=noise)
                x = self.transform.inverse(u.cpu().double(), mask).float().numpy()
                xs.append(np.pad(x, ((0, 0), (0, width - w), (0, 0))))
                ms.append(np.pad(mask.numpy(), ((0, 0), (0, width - w))))
        x, mask = np.concatenate(xs), np.concatenate(ms)
        if order is not None:
            inv = np.empty_like(order)
            inv[order] = np.arange(order.size)
            x, mask = x[inv], mask[inv]
        if not variable and not return_mask:
            return x
        return x, mask

    # ---- persistence ----------------------------------------------------
    def state(self):
        return {
            "schema": CHECKPOINT_SCHEMA,
            "architecture": velocity_config(self.model),
            "model_state": {k: v.detach().cpu() for k, v in self.model.state_dict().items()},
            "base": self.base.to_dict(),
            "transform": self.transform.to_dict(),
            "condition_layout": self.layout.to_dict(),
            "metadata": self.metadata,
            "multiplicity": None if self.multiplicity is None else self.multiplicity.state(),
        }

    def save(self, path):
        buf = io.BytesIO()
        torch.save(self.state(), buf)
        data = buf.getvalue()
        with open(path, "wb") as f:
            f.write(data)
        return hashlib.sha256(data).hexdigest()

    @classmethod
    def load(cls, path, device="cpu"):
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state.get("schema") != CHECKPOINT_SCHEMA:
            raise ValueError(f"{path} is not a {CHECKPOINT_SCHEMA} checkpoint")
        model = build_velocity(state["architecture"])
        model.load_state_dict(state["model_state"])
        model.to(device)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        with open(path, "rb") as f:
            sha = hashlib.sha256(f.read()).hexdigest()
        meta = dict(state["metadata"])
        meta["checkpoint_sha256"] = sha
        return cls(
            model,
            base_from_dict(state["base"]),
            FeatureTransform.from_dict(state["transform"]),
            ConditionLayout.from_dict(state["condition_layout"]),
            meta,
            device,
            multiplicity=(None if state.get("multiplicity") is None
                          else MultiplicityModel.from_state(state["multiplicity"], device)),
        )


def _lr_lambda(warmup, total):
    def f(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        prog = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))

    return f


def minibatch_sampler(lengths, batch_size, buckets=0, generator=None):
    """Callable returning minibatch indices (CPU) over events with real ``lengths``.

    ``buckets`` <= 1: uniform over all events (the original stream). Otherwise the
    events are split into ``buckets`` equal-count groups by length; a group is
    chosen with probability proportional to its size and the batch is drawn
    uniformly inside it, so each event keeps probability 1/n per draw while a
    batch spans a narrow range of lengths. ``.edges`` lists the largest length
    per group.
    """
    lengths = torch.as_tensor(lengths)
    n = lengths.numel()
    if buckets <= 1:
        def draw():
            return torch.randint(0, n, (batch_size,), generator=generator)
        draw.edges = None
        return draw
    order = torch.argsort(lengths, stable=True)
    groups = [g for g in torch.tensor_split(order, buckets) if g.numel()]
    weights = torch.tensor([float(g.numel()) for g in groups], dtype=torch.float64)

    def draw():
        g = groups[int(torch.multinomial(weights, 1, generator=generator))]
        return g[torch.randint(0, g.numel(), (batch_size,), generator=generator)]

    draw.edges = [int(lengths[g[-1]]) for g in groups]
    return draw


def train_flow(model, base, train, val, options, log=print, on_best=None):
    """Train a velocity model by flow matching; select by validation loss only.

    Args:
        model: velocity module ``(x, t, cond, mask) -> v``.
        train, val: dicts with tensors ``u`` [N, P, F] (model coordinates),
            ``mask`` [N, P] and ``cond`` [N, C].
        on_best: callback ``(ema_model, step, val_loss)`` when validation
            improves (e.g. to save a checkpoint).
    Returns:
        (best EMA model on CPU, history dict)
    """
    info = resolve_device(options.device, options.allow_device_fallback)
    device = info.torch_device
    torch.manual_seed(options.seed)
    gen = torch.Generator(device="cpu").manual_seed(options.seed)
    model = model.to(device).float()
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)

    tr = {k: v.to(device) for k, v in train.items()}
    va = {k: v.to(device) for k, v in val.items()}
    ntr, nva = tr["u"].shape[0], va["u"].shape[0]

    # Fixed validation noise/time so the selection metric is comparable.
    vgen = torch.Generator(device="cpu").manual_seed(options.val_seed)
    vb = min(options.batch_size * 4, nva)
    val_sets = []
    for _ in range(options.val_batches):
        idx = torch.randint(0, nva, (vb,), generator=vgen)
        t = torch.rand(vb, generator=vgen)
        eps = base.sample_like(va["u"][idx].cpu(), va["mask"][idx].cpu(), vgen)
        val_sets.append((idx.to(device), t.to(device), eps.to(device)))

    opt = torch.optim.AdamW(model.parameters(), lr=options.lr,
                            weight_decay=options.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, _lr_lambda(options.warmup_steps, options.steps)
    )
    history = {"step": [], "train_loss": [], "val_loss": [], "seconds": [],
               "device": info.to_dict()}
    draw = minibatch_sampler(tr["mask"].sum(1).cpu(), options.batch_size,
                             options.length_buckets, gen)
    history["length_buckets"] = draw.edges
    best = (float("inf"), -1)
    best_state = None
    running = 0.0
    start = time.time()
    for step in range(options.steps):
        model.train()
        idx = draw().to(device)
        t = torch.rand(options.batch_size, generator=gen).to(device)
        # Noise is drawn on CPU for device-independent streams; padded
        # entries are masked inside the loss.
        eps = base.sample_like(
            torch.zeros((options.batch_size,) + tuple(tr["u"].shape[1:])), None, gen
        ).to(device)
        loss = flow_matching_loss(model, tr["u"][idx], tr["mask"][idx],
                                  tr["cond"][idx], base, t=t, eps=eps)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), options.grad_clip)
        opt.step()
        sched.step()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.mul_(options.ema_decay).add_(pm.detach(), alpha=1 - options.ema_decay)
        running += float(loss.detach())
        if (step + 1) % options.val_every == 0 or step + 1 == options.steps:
            with torch.no_grad():
                vl = np.mean([
                    float(flow_matching_loss(ema, va["u"][i], va["mask"][i],
                                             va["cond"][i], base, t=t_, eps=e_))
                    for i, t_, e_ in val_sets
                ])
            synchronize(device)
            ntrain = (step + 1) % options.val_every or options.val_every
            history["step"].append(step + 1)
            history["train_loss"].append(running / ntrain)
            history["val_loss"].append(float(vl))
            history["seconds"].append(time.time() - start)
            running = 0.0
            log(f"step {step + 1:6d} train {history['train_loss'][-1]:.5f} "
                f"val(ema) {vl:.5f} lr {sched.get_last_lr()[0]:.2e} "
                f"{time.time() - start:.0f}s")
            if vl < best[0]:
                best = (float(vl), step + 1)
                best_state = {k: v.detach().cpu().clone() for k, v in ema.state_dict().items()}
                if on_best is not None:
                    on_best(ema, step + 1, float(vl))
    history["best_val_loss"], history["best_step"] = best
    model_cpu = copy.deepcopy(ema).cpu()
    if best_state is not None:
        model_cpu.load_state_dict(best_state)
    return model_cpu, history


def json_dumps(obj):
    return json.dumps(obj, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
