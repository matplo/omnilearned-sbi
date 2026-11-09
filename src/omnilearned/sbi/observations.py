"""Observation input path shared by closure pseudo data and measured data.

An observation file is HDF5 with only these datasets:

* ``x``        [N, P, F] float constituent features in physical coordinates
* ``mask``     [N, P] bool validity mask (explicit; no feature is a sentinel)
* ``event_id`` [N] int64 stable identifiers (select Hutchinson probes)
* ``context``  optional [N, C] known measured context, attribute ``names``

Attributes ``feature_names`` and ``record_schema`` describe the content.
Truth parameters, generator histories and diagnostics must live elsewhere;
the reader refuses files that carry them so measured-data and closure fits
use an identical path.
"""

import hashlib
from dataclasses import dataclass

import h5py
import numpy as np


ALLOWED = ("x", "mask", "event_id", "group_id", "context")
FORBIDDEN_PREFIXES = ("truth", "theta", "param", "label", "diagnostic", "history")
RECORD_SCHEMA = "omnilearned-sbi-observations-v1"


@dataclass
class Observations:
    x: np.ndarray
    mask: np.ndarray
    event_id: np.ndarray
    context: np.ndarray = None
    context_names: tuple = ()
    feature_names: tuple = ()
    sha256: str = ""
    path: str = ""
    # appended last on purpose: several callers build Observations positionally, so a new
    # field inserted earlier would silently shift context/feature_names.
    group_id: np.ndarray = None   # originating event; jets of one event share it

    def __len__(self):
        return self.x.shape[0]

    def subset(self, index):
        return Observations(
            self.x[index],
            self.mask[index],
            self.event_id[index],
            None if self.context is None else self.context[index],
            self.context_names,
            self.feature_names,
            self.sha256,
            self.path,
            None if self.group_id is None else self.group_id[index],
        )


def restrict_to_core(obs, r_max=0.4, require_min=1):
    """New ``Observations`` with constituents beyond ``r_max`` of the jet axis dropped.

    Mitigation for the P4 finding that the surrogate under-produces constituents beyond
    the jet's R=0.4 clustering cone (up to 39x deficit near dR~0.5), and that this
    deficiency's severity correlates with T_ref directly, not only through multiplicity
    (FINDINGS_JEWEL_V3.md, "the beyond-R=0.4 halo deficit correlates with T_ref directly").
    `explorer_core.html` showed core-jet (dR<0.4) direct-vs-surrogate agreement is excellent
    (1-3%); this applies that same restriction at the likelihood level, not just as a
    diagnostic, so inference is not exposed to the known, theta-correlated bias. This does
    not fix the surrogate -- it avoids scoring the region where it is known to be wrong, at
    the cost of discarding whatever real quenching signal is in that region.

    Needs ``deta``/``dphi`` in ``obs.feature_names`` (the v3 representation). Padded
    entries stay exactly zero (the project's convention everywhere else); a constituent cut
    by this function is masked out and its features zeroed, not merely masked, so it cannot
    leak through a caller that reads ``x`` without ``mask``.

    ``require_min``: a jet left with fewer than this many real constituents raises --
    silently scoring a near-empty cloud is a worse failure mode than stopping to look at it.
    """
    if "deta" not in obs.feature_names or "dphi" not in obs.feature_names:
        raise ValueError(f"restrict_to_core needs deta/dphi in feature_names, got {obs.feature_names}")
    deta = obs.x[..., obs.feature_names.index("deta")]
    dphi = obs.x[..., obs.feature_names.index("dphi")]
    dr = np.hypot(deta, dphi)
    core = obs.mask & (dr < r_max)
    counts = core.sum(1)
    if require_min > 0 and np.any(counts < require_min):
        bad = int((counts < require_min).sum())
        raise ValueError(f"{bad} of {len(obs)} jets have fewer than {require_min} core "
                         f"(dR<{r_max}) constituents after the cut")
    x = np.where(core[..., None], obs.x, 0.0)
    return Observations(x, core, obs.event_id, obs.context, obs.context_names,
                        obs.feature_names, obs.sha256, obs.path, obs.group_id)


def file_sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def write_observations(path, x, mask, event_id, feature_names, context=None,
                       context_names=(), attrs=None, group_id=None):
    """Write observations.

    ``event_id`` must be unique per row: it seeds the Hutchinson probes
    (:func:`likelihood.stable_probes`), so two rows sharing an id would be evaluated with
    identical probe vectors and their divergence errors would be correlated -- unbiased per
    row, but it inflates the variance of the summed log-likelihood. For an inclusive jet
    sample, where several jets come from one event, pass the originating event in
    ``group_id`` instead; it is carried for diagnostics (the composite-likelihood coverage
    check) and never touches the likelihood.
    """
    x = np.asarray(x)
    if x.dtype != np.float64:  # float64 is kept (no rounding onto bounds)
        x = x.astype(np.float32)
    mask = np.asarray(mask, dtype=bool)
    event_id = np.asarray(event_id, dtype=np.int64)
    if x.ndim != 3 or mask.shape != x.shape[:2] or event_id.shape != x.shape[:1]:
        raise ValueError("inconsistent observation shapes")
    if np.unique(event_id).size != event_id.size:
        raise ValueError("event ids must be unique (they seed the probes); "
                         "for several jets of one event use group_id")
    if group_id is not None:
        group_id = np.asarray(group_id, dtype=np.int64)
        if group_id.shape != event_id.shape:
            raise ValueError("group_id must have one entry per row")
    if np.any(x[~mask] != 0):
        raise ValueError("padded entries must be zero")
    with h5py.File(path, "w") as f:
        f.create_dataset("x", data=x, compression="gzip")
        f.create_dataset("mask", data=mask, compression="gzip")
        f.create_dataset("event_id", data=event_id)
        if group_id is not None:
            f.create_dataset("group_id", data=group_id)
        if context is not None:
            ds = f.create_dataset("context", data=np.asarray(context, np.float64))
            ds.attrs["names"] = list(context_names)
        f.attrs["feature_names"] = list(feature_names)
        f.attrs["record_schema"] = RECORD_SCHEMA
        for k, v in (attrs or {}).items():
            if k.lower().startswith(FORBIDDEN_PREFIXES):
                raise ValueError(f"attribute '{k}' not allowed in observations")
            f.attrs[k] = v
    return file_sha256(path)


def read_observations(path):
    with h5py.File(path, "r") as f:
        keys = list(f.keys())
        bad = [k for k in keys if k.lower().startswith(FORBIDDEN_PREFIXES)]
        bad += [k for k in f.attrs if k.lower().startswith(FORBIDDEN_PREFIXES)]
        if bad:
            raise ValueError(
                f"observation file {path} carries truth/diagnostic fields {bad}; "
                "fitting inputs must contain observations only"
            )
        extra = [k for k in keys if k not in ALLOWED]
        if extra:
            raise ValueError(f"unexpected datasets {extra} in {path}")
        if f.attrs.get("record_schema") != RECORD_SCHEMA:
            raise ValueError(f"{path} is not a {RECORD_SCHEMA} file")
        x = f["x"][...]
        mask = f["mask"][...].astype(bool)
        eid = f["event_id"][...].astype(np.int64)
        gid = f["group_id"][...].astype(np.int64) if "group_id" in f else None
        context, cnames = None, ()
        if "context" in f:
            context = f["context"][...]
            cnames = tuple(f["context"].attrs.get("names", []))
        fnames = tuple(f.attrs.get("feature_names", []))
    return Observations(x, mask, eid, context, cnames, fnames, file_sha256(path),
                        str(path), gid)
