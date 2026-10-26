import numpy as np


def _sample(rng, n, shift_bin=None, shift=0.0):
    from omnilearned.sbi.toy.ptbins import KEYS, PT_EDGES

    pt = rng.uniform(PT_EDGES[0], PT_EDGES[-1], n)
    obs = {k: rng.normal(1.0, 0.5, n) for k in KEYS}
    if shift_bin is not None:
        b = np.digitize(pt, PT_EDGES) - 1
        obs["width"][b == shift_bin] += shift
    return pt, obs


def test_pt_binned_locates_shift_in_one_bin():
    from omnilearned.sbi.toy.ptbins import PT_EDGES, pt_binned

    rng = np.random.default_rng(0)
    pd_, od = _sample(rng, 80000)
    ps, os_ = _sample(rng, 80000, shift_bin=6, shift=0.25)
    doc = pt_binned(pd_, od, ps, os_)
    assert sum(doc["n_direct"]) == 80000 and len(doc["n_direct"]) == len(PT_EDGES) - 1
    w = np.array(doc["observables"]["width"]["effect"])
    assert abs(w[6] - 0.5) < 0.05  # 0.25 / σ 0.5
    assert np.all(np.abs(np.delete(w, 6)) < 0.05)
    p = np.array(doc["observables"]["width"]["pull"])
    assert p[6] > 10 and np.all(np.abs(np.delete(p, 6)) < 5)
    assert np.allclose(doc["observables"]["ptd"]["sd_ratio"], 1.0, atol=0.06)


def test_pt_binned_sparse_bin_and_combine():
    from omnilearned.sbi.toy.ptbins import combine, pt_binned

    rng = np.random.default_rng(1)
    pd_, od = _sample(rng, 4000)
    ps, os_ = _sample(rng, 4000)
    ps[ps > 170] = 65.0  # surrogate has no jets in the last bin
    doc = pt_binned(pd_, od, ps, os_)
    assert doc["n_surrogate"][-1] == 0 and doc["observables"]["width"]["effect"][-1] is None
    c = combine([doc, doc, doc, doc])
    # identical points: Stouffer pull = √P × pull, mean effect = effect, se 0
    assert np.isclose(c["observables"]["width"]["pull_combined"][0], 2 * doc["observables"]["width"]["pull"][0])
    assert np.isclose(c["observables"]["width"]["effect"][0], doc["observables"]["width"]["effect"][0])
    assert np.isnan(c["observables"]["width"]["effect"][-1]) and c["n_direct"][0] == 4 * doc["n_direct"][0]
