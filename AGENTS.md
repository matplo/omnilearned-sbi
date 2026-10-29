# OmniLearned SBI: branch dev/nle-jewel

**Read `docs/RESTART_JEWEL.md` first** — it holds the current state and the exact next
command after any interruption. Then `docs/PLAN_JEWEL_NLE.md`, the standing brief for this
track: infer JEWEL's (b, T_ref, tau_init) from ensembles of jets as point clouds, jets
20-500 GeV. The running log is `FINDINGS_JEWEL.md` at the repository root.

`PLAN_NLE_PERLMUTTER.md` remains authoritative for the **shared ground rules and lessons**
(its sections 1-6) and is the record of the toy track, which is **closed**: do not start new
toy generation, training or closure work.

The toy simulator still lives in the repository (`src/omnilearned/sbi/toy/`) because the NLE
machinery, the gate and `ptbins` are shared code, not because the toy track is active.
Changing a backend's `[jewel]`, `[subtraction]` or `[selection]` section defines a new
backend id (JEWEL v2 = `jewel-2.6.0-cs-v2`, `configs/jewel-2.6.0-cs-v2.toml`).

- Do not create or modify environments; the user provides an `henv` environment.
- Do not switch branches while training or fitting runs import this checkout.
- Record source revision, dirty status and actual device with every result.
- Do not commit data, checkpoints, run trees, figures or binaries; never force-push.
- Add tests (`tests/sbi`) for every new generator, record format, driver and numerical method.
