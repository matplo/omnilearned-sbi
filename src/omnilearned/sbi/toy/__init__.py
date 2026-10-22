"""Invented jet-quenching toy simulators for Route A closure tests.

``toy-quenching-v1`` (``simulator``) is ported unchanged from the omni-sbi-mc
application (a/src/jet_quenching_inference): 32 particles per event, all in
one jet. ``toy-quenching-v2`` (``simulator_v2``) adds Poisson multiplicities
and keeps the leading jet's actual constituents, padded to 256 with a mask.
See PLAN_NLE_PERLMUTTER.md. Modules:

* ``simulator``: the v1 generator (``sample_toy``, ``ToyConstants``, ``Selection``).
* ``simulator_v2``: the v2 generator and ``select_sets``, the constituent-level
  selection shared by toy jets and surrogate samples.
* ``reco``: anti-kt reconstruction with FastJet through hepyy/cppyy.
* ``selection``: the same selection applied to generated clouds, and the
  surrogate selection-acceptance table.
* ``observables``: jet observables and a KS statistic.
* ``design``: event-id streams, seeded generators, reserved settings.
* ``records``: HDF5 record writers.
* ``config``: load ``configs/toy-quenching-v1.toml`` or ``-v2.toml`` (by
  ``backend``); ``sample_setting`` and ``array_sha256`` for reproducible draws.
"""

from omnilearned.sbi.toy.config import array_sha256, load_toy_config, sample_setting  # noqa: F401
