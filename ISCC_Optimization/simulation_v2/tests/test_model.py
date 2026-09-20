"""Sanity checks for the signal model and the optimisation framework.

Run with ``python -m pytest tests`` or directly: ``python tests/test_model.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stochastic_mec import (  # noqa: E402
    AlgorithmParams,
    HybridSolver,
    SystemModel,
    SystemParams,
    count_cases,
    evaluate_solution,
    exhaustive_search,
    make_tpc,
    sample_topology_with_cells,
)

FAST = AlgorithmParams(
    n_agents_bwoa=8, max_iter_bwoa=15, patience_bwoa=5,
    n_agents_tpc=8, max_iter_tpc=30, patience_tpc=5, pop_min=6, pop_max=12,
)


def _model(seed=0, ul_cells=2, dl_cells=2, ues=6, **kw):
    rng = np.random.default_rng(seed)
    params = SystemParams(**kw)
    topo = sample_topology_with_cells(rng, params, ul_cells, dl_cells, ues)
    return SystemModel(topo, params, rng), rng


def _full_assoc(model, rng):
    a = np.zeros(model.assoc_shape, dtype=np.int8)
    for n in range(model.n_ul):
        a[n, int(rng.integers(model.K))] = 1
    for m in range(model.m_dl):
        a[model.n_ul + m, int(rng.integers(model.K))] = 1
    return a


def test_zero_forcing_removes_intra_cell_interference():
    """h_{i}^H w_{j} == delta_ij inside every downlink cell."""
    model, _ = _model(seed=3)
    for m in range(model.m_dl):
        idx = model.dl_ue_of_cell[m]
        if idx.size < 2 or idx.size > model.L:
            continue
        for k in range(model.K):
            gain = model.dl_gain[m][idx, k, :]          # (N_m, N_m)
            np.testing.assert_allclose(np.diag(gain), 1.0, rtol=1e-6, atol=1e-6)
            off = gain - np.diag(np.diag(gain))
            assert np.max(off) < 1e-12, "ZF left intra-cell interference"


def test_smca_matches_numerical_optimum():
    """The closed-form F* beats any random feasible computation allocation."""
    model, rng = _model(seed=5)
    ul_sub = np.where(model.admissible, np.arange(model.n_ul) % model.K, -1)
    v_star, f_star = model.smca(ul_sub)
    rho, chi = model.default_split(ul_sub)
    weight = model.beta_t * rho * chi * model.c_raw / model.t_ref

    def v_of(f):
        return sum(weight[n] / f[n] for n in range(model.n_ul) if f[n] > 0)

    assert abs(v_of(f_star) - v_star) < 1e-6 * max(v_star, 1.0)
    for _ in range(200):
        f = np.zeros(model.n_ul)
        for m in range(model.m_ul):
            members = np.flatnonzero((model.ul_serving == m) & (ul_sub >= 0))
            if members.size == 0:
                continue
            share = rng.dirichlet(np.ones(members.size))
            f[members] = share * model.server_capacity[m]
        assert v_of(f) >= v_star - 1e-9


def test_lower_bound_is_a_lower_bound():
    """Theorem 2/3: W~(A, p~*) <= W(A, p) for every feasible p."""
    model, rng = _model(seed=7)
    assoc = _full_assoc(model, rng)
    ul_sub, dl_sub = model.decode(assoc)
    xi = model.cochannel_at_sbs(dl_sub, model.equal_split_dl_power(dl_sub))
    lower = model.mpc_lower_bound(ul_sub)
    for _ in range(50):
        p = rng.uniform(model.p.p_min, model.p.p_max, size=model.n_ul)
        assert model.mpc_objective(ul_sub, p, xi) >= lower - 1e-9


def test_dl_utility_upper_bound_dominates():
    """The bound used by Lemmas 1 & 2 really dominates the achievable U^dl."""
    model, rng = _model(seed=11)
    assoc = _full_assoc(model, rng)
    ul_sub, dl_sub = model.decode(assoc)
    bound = model.dl_utility_upper_bound(dl_sub)
    lo, hi = model.dl_power_bounds(dl_sub)
    for _ in range(50):
        q = rng.uniform(lo, hi)
        p = rng.uniform(model.p.p_min, model.p.p_max, size=model.n_ul)
        assert model.dl_utility(dl_sub, q, ul_sub, p) <= bound + 1e-9


def test_power_optimisers_agree():
    """WOA, IWOA and PSO reach a comparable MPC optimum."""
    model, rng = _model(seed=13)
    assoc = _full_assoc(model, rng)
    ul_sub, dl_sub = model.decode(assoc)
    xi = model.cochannel_at_sbs(dl_sub, model.equal_split_dl_power(dl_sub))
    lb = np.full(model.n_ul, model.p.p_min)
    ub = np.full(model.n_ul, model.p.p_max)
    scores = {}
    for name in ("WOA", "IWOA", "PSO"):
        opt = make_tpc(name, FAST, np.random.default_rng(21))
        scores[name] = opt.minimize(lambda p: model.mpc_objective(ul_sub, p, xi), lb, ub).score
    best = min(scores.values())
    for name, s in scores.items():
        assert s <= best * 1.5 + 1e-9, f"{name} is far from the best MPC value: {scores}"


def test_solution_is_feasible():
    """The returned solution respects every hard constraint of (11)."""
    model, rng = _model(seed=17, ues=8)
    sol = HybridSolver(model, rng, algo=FAST, tpc="WOA").solve()
    a_ul, a_dl = model.split(sol.assoc)
    assert np.all(a_ul.sum(axis=1) <= 1)                 # (11c)
    assert np.all(a_dl.sum(axis=1) == 1) or model.m_dl == 0   # (11i)
    assert np.all(sol.ul_power <= model.p.p_max + 1e-12)      # (11d)
    ul_sub, dl_sub = model.decode(sol.assoc)
    assert model.sic_violation(ul_sub, sol.ul_power) < 1e-12  # (11e)
    for m in range(model.m_dl):                                # (11l)
        k, served = dl_sub[m], model.dl_ue_of_cell[m]
        if k < 0 or served.size == 0:
            continue
        used = float(sol.dl_power[served] @ model.w_norm2[served, k])
        assert used <= model.sbs_budget[m] * (1 + 1e-6)
    _, f_alloc = model.smca(ul_sub)
    for m in range(model.m_ul):                                # (11g)
        members = np.flatnonzero((ul_sub >= 0) & (model.ul_serving == m))
        assert f_alloc[members].sum() <= model.server_capacity[m] * (1 + 1e-9)


def test_bwoa_is_near_the_exhaustive_optimum():
    """On a tiny instance the hybrid solver stays close to the true optimum.

    Mirrors the "BWOA vs. EX" experiment of the paper, which reports an
    optimality gap of a couple of per cent.
    """
    model, _ = _model(seed=23, ul_cells=1, dl_cells=1, ues=4, n_subchannels=3)
    assert count_cases(model) <= 5000
    algo = AlgorithmParams(n_agents_bwoa=20, max_iter_bwoa=60, patience_bwoa=20,
                           n_agents_tpc=15, max_iter_tpc=60, patience_tpc=10)
    bw = HybridSolver(model, np.random.default_rng(1), algo=algo, tpc="WOA").solve()
    ex = exhaustive_search(model, np.random.default_rng(1), tpc="WOA", algo=algo)
    assert bw.utility <= ex.utility + 1e-6, "BWOA beat the exhaustive optimum"
    gap = (ex.utility - bw.utility) / max(abs(ex.utility), 1.0)
    assert gap < 0.05, f"optimality gap of {100 * gap:.1f} % is too large"


def test_positions_are_3d_and_jammers_exist():
    model, _ = _model(seed=2)
    assert model.topo.sbs_pos.shape[1] == 3
    assert model.topo.ue_pos.shape[1] == 3
    assert np.all(model.topo.ue_pos[:, 2] == model.p.z_min)
    assert np.all(model.topo.sbs_pos[:, 2] >= model.p.z_uav_min)


def test_jamming_lowers_ul_sijnr():
    model, rng = _model(seed=4)
    if model.n_jam == 0 or model.n_ul == 0:
        return
    assoc = _full_assoc(model, rng)
    ul_sub, dl_sub = model.decode(assoc)
    p = np.full(model.n_ul, model.p.p_max)
    xi = model.cochannel_at_sbs(dl_sub, model.equal_split_dl_power(dl_sub))
    g_j = model.ul_sinr(ul_sub, p, xi, dl_sub)
    saved = model.n_jam
    model.n_jam = 0
    g_0 = model.ul_sinr(ul_sub, p, xi, dl_sub)
    model.n_jam = saved
    assert np.all(g_j <= g_0 + 1e-12)


def test_alca_never_offloads():
    """All-local: rho = 0, and the UL utility is the closed-form local value.

    Local UEs are not worthless under ISCC -- shrinking chi also shrinks the
    local workload -- so the UL utility is no longer zero as in v1.
    """
    model, rng = _model(seed=29)
    sol = HybridSolver(model, rng, algo=FAST, tpc="WOA", scheme="ALCA").solve()
    a_ul, _ = model.split(sol.assoc)
    assert a_ul.sum() == 0
    met = evaluate_solution(model, sol)
    assert met.offload_ratio == 0.0
    assert np.all(sol.rho == 0.0)
    omega = model._omega(model.t_cmp, model.e_cmp)             # rho = 0 branches
    chi = model.optimal_chi(omega)
    acc = model.accuracy(chi)
    local = model.u_const - chi * omega + model.beta_a * (acc - model.lam_th) / (1 - model.lam_th)
    expect = float(np.sum(np.where(model.admissible, local, 0.0)))
    assert abs(met.ul_utility - expect) < 1e-9, (met.ul_utility, expect)


# --------------------------------------------------------------------- #
# ISCC (manuscript_v2)
# --------------------------------------------------------------------- #
def _iscc_state(seed):
    model, rng = _model(seed=seed, ues=10)
    ul_sub = np.where(model.admissible & (rng.random(model.n_ul) < 0.7),
                      rng.integers(0, model.K, model.n_ul), -1)
    dl = np.zeros(model.m_dl, dtype=int)
    xi = model.cochannel_at_sbs(dl, model.equal_split_dl_power(dl))
    p = rng.uniform(model.p.p_min, model.p.p_max, model.n_ul)
    rates = model.ul_rates(ul_sub, p, xi, dl)
    return model, rng, ul_sub, dl, xi, p, rates


def test_split_and_retention_are_jointly_optimal():
    """Props. 2-3: closed-form (rho, chi) beats a dense grid for every UE."""
    for seed in (1, 2, 3):
        model, rng, ul_sub, _, _, p, rates = _iscc_state(seed)
        f = rng.uniform(0.2e9, 3e9, model.n_ul)
        rho, chi = model.optimal_split(ul_sub, rates, f, p)
        u = model.ul_utilities(rho, chi, rates, f, p)
        grid = np.linspace(0.0, 1.0, 201)
        for n in np.flatnonzero(model.admissible):
            best = -np.inf
            for r in (grid if ul_sub[n] >= 0 else [0.0]):
                for c in grid[grid >= min(model.chi_min[n], 1.0) - 1e-12]:
                    rr, cc = rho.copy(), chi.copy()
                    rr[n], cc[n] = r, c
                    best = max(best, model.ul_utilities(rr, cc, rates, f, p)[n])
            assert best <= u[n] + 1e-9, (seed, n, best, u[n])


def test_clipped_water_filling_is_optimal():
    """Prop. 1: no budget-preserving transfer lowers the weighted delay."""
    model, rng, ul_sub, _, _, p, rates = _iscc_state(4)
    rho = np.where(ul_sub >= 0, rng.uniform(0.1, 1.0, model.n_ul), 0.0)
    chi = rng.uniform(0.7, 1.0, model.n_ul)
    _, f_star = model.smca(ul_sub, rho, chi, rates)

    def cost(f):
        d, _ = model._branches(rho, rates, f, p)
        return float(np.sum(model.beta_t * chi * d / model.t_ref))

    base = cost(f_star)
    for m in range(model.m_ul):
        members = np.flatnonzero((ul_sub >= 0) & (rho > 0) & (model.ul_serving == m))
        assert f_star[members].sum() <= model.server_capacity[m] * (1 + 1e-9)
        for i in members:
            for j in members:
                if i != j:
                    f = f_star.copy()
                    f[i], f[j] = 0.9 * f[i], f[j] + 0.1 * f[i]
                    assert cost(f) >= base * (1 - 1e-9)


def test_uniform_bound_dominates_every_split():
    """The pruning bound holds for any powers and the optimised (F, rho, chi)."""
    for seed in (5, 6, 7):
        model, rng, ul_sub, dl, xi, _, _ = _iscc_state(seed)
        bound = model.ul_utility_upper_bound(ul_sub)
        for _ in range(20):
            p = rng.uniform(model.p.p_min, model.p.p_max, model.n_ul)
            rates = model.ul_rates(ul_sub, p, xi, dl)
            rho, chi, f = model.iscc_allocate(ul_sub, rates, p)
            assert bound >= float(np.sum(model.ul_utilities(rho, chi, rates, f, p))) - 1e-9


def test_iscc_reduces_to_proportional_rule():
    """rho = chi = 1 and no sensing overhead recover the v1 allocation."""
    model, _ = _model(seed=8, sensing_time=0.0, sensing_power=0.0, extract_cycles_per_bit=0.0)
    ul_sub = np.where(model.admissible, np.arange(model.n_ul) % model.K, -1)
    _, f = model.smca(ul_sub)
    root = np.sqrt(model.beta_t * model.f_local)
    for m in range(model.m_ul):
        idx = np.flatnonzero((ul_sub >= 0) & (model.ul_serving == m))
        if idx.size:
            np.testing.assert_allclose(f[idx], root[idx] / root[idx].sum() * model.server_capacity[m])


def test_mf_noise_fix_restores_a_usable_uplink():
    """v1's L B N0 scaling makes the UL SNR hopeless; the fix restores it."""
    snr = {}
    for fix in (False, True):
        model, _ = _model(seed=9, mf_noise_fix=fix, lambda_jammer=0.0)
        n = 0
        ul_sub = np.full(model.n_ul, -1)
        ul_sub[n] = 0
        g = model.ul_sinr(ul_sub, np.full(model.n_ul, model.p.p_max),
                          np.zeros((model.K, model.m_ul)), np.full(model.m_dl, -1))
        snr[fix] = 10 * np.log10(g[n])
    assert snr[False] < 0 < snr[True], snr


def test_inadmissible_ues_are_forced_local():
    model, rng = _model(seed=10, sensing_snr_db=0.0)
    assert not model.admissible.all()
    assoc = _full_assoc(model, rng)
    ul_sub, _ = model.decode(assoc)
    assert np.all(ul_sub[~model.admissible] == -1)


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {name}: {exc}")
    raise SystemExit(1 if failures else 0)
