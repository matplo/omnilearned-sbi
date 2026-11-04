"""train_flow resume: a killed run must be continuable to the same target step, not just
restartable from its last best-val checkpoint (which carries no optimizer/RNG state).

Motivation: P3 (JEWEL v3 surrogate training) runs on interactive-QOS GPU allocations with a
hard 4 h walltime, and a single run (400k steps) is close to that limit. Without this,
a run killed at e.g. step 350k would either be silently mislabeled "400k" at a worse step, or
have to restart its ~4 h from scratch -- losing the GPU time already spent either way.

Coverage gap, left open rather than faked: everything here runs on device="cpu", so it
cannot catch a CPU/GPU boundary bug in the resume path (map_location on torch.load applying
to every tensor in the checkpoint, including the minibatch generator's CPU-only ByteTensor
state -- hit for real on both seed0 and seed1's first --resume attempt, 2026-11-03, after
~3h of training each). Confirmed directly on a real GPU resume instead of simulated here.
"""

import torch
from torch import nn

from omnilearned.sbi.base import GaussianBase
from omnilearned.sbi.flow import TrainOptions, train_flow


class TinyVelocity(nn.Module):
    def __init__(self, f=3, c=2, h=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(f + 1 + c, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, f))

    def forward(self, x, t, cond, mask):
        b, p, _ = x.shape
        z = torch.cat([x, t.view(-1, 1, 1).expand(b, p, 1), cond[:, None, :].expand(b, p, cond.shape[-1])], -1)
        return self.net(z) * mask[..., None].to(x.dtype)


def _clouds(n, p=4, f=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, p, f, generator=g)
    mask = torch.arange(p)[None] < torch.randint(1, p + 1, (n,), generator=g)[:, None]
    return {"u": x * mask[..., None], "mask": mask, "cond": torch.randn(n, 2, generator=g)}


def test_resume_continues_bit_identically_to_an_uninterrupted_run(tmp_path):
    data = _clouds(200)
    opts = TrainOptions(steps=40, batch_size=16, val_every=10, val_batches=2, warmup_steps=5,
                        device="cpu")

    torch.manual_seed(7)
    direct_model = TinyVelocity()
    direct_best, direct_hist = train_flow(direct_model, GaussianBase.standard(3), data, data,
                                          opts, log=lambda m: None)

    # the same run, but "killed" after step 20 and resumed by a second train_flow call --
    # a fresh model/optimizer each time, exactly as a second CLI invocation would build them.
    # stop_after_step, not a shorter options.steps: the LR schedule is a function of
    # options.steps, so a genuinely shorter run would not be the same schedule at all.
    torch.manual_seed(7)
    first_model = TinyVelocity()
    rp = tmp_path / "resume.pt"
    train_flow(first_model, GaussianBase.standard(3), data, data, opts,
              log=lambda m: None, resume_path=rp, stop_after_step=20)
    assert rp.exists()
    ck = torch.load(rp, weights_only=False)
    assert ck["step"] == 20

    torch.manual_seed(999)  # must be irrelevant: resume restores the real RNG stream
    second_model = TinyVelocity()  # fresh, random weights -- resume must overwrite them
    resumed_best, resumed_hist = train_flow(second_model, GaussianBase.standard(3), data, data,
                                            opts, log=lambda m: None, resume_from=rp, resume_path=rp)

    assert resumed_hist["step"] == direct_hist["step"] == [10, 20, 30, 40]
    assert resumed_hist["train_loss"] == direct_hist["train_loss"]
    assert resumed_hist["val_loss"] == direct_hist["val_loss"]
    assert resumed_hist["best_step"] == direct_hist["best_step"]
    for k in direct_best.state_dict():
        assert torch.equal(direct_best.state_dict()[k], resumed_best.state_dict()[k])


def test_resume_file_is_written_at_every_validation_point_not_only_on_improvement(tmp_path):
    """The on_best checkpoint only ever reflects the best step; resume_path must not skip a
    validation point just because that step's loss did not improve on the previous best."""
    data = _clouds(200)
    opts = TrainOptions(steps=30, batch_size=16, val_every=10, val_batches=2, warmup_steps=2,
                        device="cpu")
    rp = tmp_path / "resume.pt"
    seen_steps = []
    torch.manual_seed(1)
    train_flow(TinyVelocity(), GaussianBase.standard(3), data, data, opts,
              log=lambda m: None, resume_path=rp,
              on_best=lambda m, step, vl: seen_steps.append(step))
    final = torch.load(rp, weights_only=False)
    assert final["step"] == 30  # the LAST validation point's snapshot, not just an improving one
