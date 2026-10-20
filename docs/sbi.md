# Conditional flow likelihoods

The `omnilearned.sbi` API evaluates one density per independent event, then fits
all named parameters jointly across the full observation sample. It does not
require simulation truth or a class-label column in measured inputs.

`SmoothVelocity` is a separately trained global-attention field with external
fixed validity masks. It is permutation equivariant, uses smooth activations,
and has no hard nearest-neighbor selection. Existing PET checkpoints are not
converted into this architecture. Their changing neighbor graphs and inferred
masks remain a separate density-correctness question; their generation paths are
preserved.

```python
from omnilearned.sbi import (
    ConditionSpec, GaussianBase, LikelihoodOptions, log_prob, resolve_device,
)

device = resolve_device("cpu")  # select mps/cuda explicitly when available
spec = ConditionSpec(("parameter_a", "parameter_b"))
base = GaussianBase((0., 0., 0.), (1., 1., 1.))
values, diagnostics = log_prob(
    model, features.to(device), {"parameter_a": 0.2, "parameter_b": 0.7},
    mask=mask.to(device), event_ids=event_ids, base=base,
    condition_spec=spec, options=LikelihoodOptions(steps=64, probes=4),
)
```

The model contract is `model(coordinates, time_vector, condition, fixed_mask)`.
Models must be deterministic in evaluation mode and independent across the
event dimension. The evaluator restores each module's previous training state
and leaves parameter gradient flags and existing parameter gradients unchanged.
Input derivatives remain enabled for divergence evaluation even inside a caller's
`no_grad` context. `inference_mode` must not wrap density evaluation.

The data-to-base convention is time 0 to time 1:

```text
log p_data = log p_base + integral_0^1 divergence(velocity) dt
```

Sampling uses exactly the same velocity and `GaussianBase`, in reverse time.
Base means/scales, condition normalization, architecture and measure metadata are
stored in `omnilearned-density-v1` checkpoints. Runtime-smoke and legacy artifacts
are rejected by the density loader. The shared base also supplies training noise.

Density values are in model coordinates unless a per-event transform log
Jacobian is provided. `transform_features` supports finite intervals via atanh
and a training-only affine transformation; its Jacobian returns the density to
the declared independent physical-coordinate measure. Discrete counts, labels,
order conventions and selection normalizations remain explicit model metadata
and application responsibilities. A selected shape likelihood does not include
an event-yield factor.

`LikelihoodOptions` exposes midpoint/RK4 steps, exact or repeated Hutchinson
divergence, probe distribution/count/seed and maximum forward evaluations.
Stable event IDs determine each event's probes and allow common randomness
across candidates and batches. Padding coordinates do not contribute to the
velocity, divergence or base. Exact divergence is useful for permutation tests;
finite stochastic trace estimates have finite permutation/probe noise unless
particle identities and probes are permuted together.

`trace_chunk_size` batches vector-Jacobian products on CPU (default 1 preserves
the sequential path). Larger chunks require explicit CPU execution; compare
their memory and runtime for the actual cloud size. Both exact and Hutchinson
traces have sequential-versus-batched regression tests. `transport` evolves
externally supplied coordinates using the same field and solver as sampling,
supporting shared-latent Monte Carlo and forward/reverse trajectory tests.

An optional `collective_log_vector_norm` transform bounds the magnitude of a
summed planar vector while retaining its continuous spectrum. Its configuration
names the log-magnitude and angle feature indices and the open log-norm interval.
It shifts all valid log magnitudes together, mapping the collective log norm
through atanh; the inverse uses tanh. The physical log Jacobian includes this
single collective determinant in addition to the coordinate and affine terms.
Tests check the complete finite-difference Jacobian, inversion, masks, permutation
equivariance and generated support. Applications must train a new checkpoint
with this transform and still account for any other selection requirements.

`scan` takes an evaluator returning per-event values and an `Observations`
source. It retains the complete joint log-likelihood grid. CPU float64 `math.fsum`
accumulation makes event chunks computational only. `normalize_posterior` applies
a scientific prior once and uses trapezoidal integration cell volumes, including
on nonuniform axes. Results distinguish conditional slices at the MLE, profile
likelihoods and posterior marginal densities. Boundary MLEs are flagged.

`scan_local` integrates an independent local parameter for each event using
normalized supplied quadrature weights, then sums event logs. It never integrates
one shared local coordinate after multiplying all event densities. The
application must estimate the selected-sample local distribution independently,
including its selection and simulation uncertainty.

MPS is single-process and float32. Device selection fails if the requested device
is unavailable or automatic MPS CPU fallback is enabled. Per-event and ensemble
log-density reductions use CPU float64. Transfer MPS tensors to CPU before
casting to float64: the combined device/dtype conversion silently produced zeros
in the inspected PyTorch 2.14.1 execution path. The CPU/MPS density regression
test checks this using real attention, optimizer and divergence operations.

The lightweight entry point is:

```text
python -m omnilearned.sbi FIT.json --device cpu --output NEW_OUTPUT_DIRECTORY
```

The main CLI also exposes `omnilearned likelihood`. JSON inputs contain
observations, immutable checkpoint identity, named candidate axes, an explicit
uniform scientific prior, numerical options and scan budgets. The generic API
also accepts nonuniform priors. The application must provide its own fit driver
when checkpoint metadata requires selection normalization. Scan-budget failures
retain a partial grid and never fabricate a posterior.

Use the assigned application's prepared environment wrapper for execution.
The paired application owns simulation, training designs, studies and reports.
Numerical tests establish evaluator correctness, not physical simulator closure.
