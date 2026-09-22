"""Algorithm 3: BWOA over the joint sub-channel association ``A_net``.

For every candidate association the solver

1. tests the pruning bound ``F(A)`` (Lemma 2(i)), here the bound of
   :meth:`SystemModel.ul_utility_upper_bound`, which holds uniformly over
   ``(p, F, rho, chi)``.  The quasi-convex Lemma 2(ii) test of v1 is not
   used: it bounds ``W`` at a fixed ``(rho, chi)``, which is unknown before
   the inner solve, and the uniform bound already folds in the best power,
2. only then pays for the two metaheuristic inner problems: MPC for the UL
   powers and DL-TPC for the broadcast powers,
3. and finally applies Props. 1-3 in closed form for ``(F, rho, chi)``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .config import AlgorithmParams
from .optimizers import bwoa_step, make_tpc
from .schemes import Scheme, make_scheme
from .system import Solution, SystemModel


@dataclass
class _Inner:
    """Result of the two inner power problems for one association."""

    utility: float
    p_ul: np.ndarray
    q_dl: np.ndarray
    pruned: bool = False


class HybridSolver:
    """Hybrid ``<TPC>-BWOA`` solver (``TPC`` in {WOA, IWOA, PSO})."""

    def __init__(
        self,
        model: SystemModel,
        rng: np.random.Generator,
        algo: AlgorithmParams | None = None,
        tpc: str = "WOA",
        scheme: str | Scheme = "MF-SIC",
    ):
        self.model = model
        self.rng = rng
        self.algo = algo or AlgorithmParams()
        self.tpc_name = tpc.upper()
        self.tpc = make_tpc(tpc, self.algo, rng)
        self.scheme = scheme if isinstance(scheme, Scheme) else make_scheme(scheme, model, rng)
        self.n_inner_calls = 0

    # ------------------------------------------------------------------ #
    # inner problems
    # ------------------------------------------------------------------ #
    def solve_mpc(
        self,
        ul_sub: np.ndarray,
        xi: np.ndarray,
        rho: np.ndarray | None = None,
        chi: np.ndarray | None = None,
    ) -> tuple[float, np.ndarray]:
        """Algorithm 1 -- multi-task power control at a fixed split ``(rho, chi)``."""
        m = self.model
        p_full = np.full(m.n_ul, m.p.p_min)
        active = np.flatnonzero(ul_sub >= 0)
        if active.size == 0:
            return 0.0, p_full

        lb = np.full(active.size, m.p.p_min)
        ub = np.full(active.size, m.p.p_max)

        def fitness(x: np.ndarray) -> float:
            p = p_full.copy()
            p[active] = x
            return m.mpc_objective(ul_sub, p, xi, rho, chi)

        res = self.tpc.minimize(fitness, lb, ub)
        p_full[active] = res.position
        return res.score, p_full

    def solve_dl_tpc(
        self, dl_sub: np.ndarray, ul_sub: np.ndarray, p_ul: np.ndarray
    ) -> tuple[float, np.ndarray]:
        """Algorithm 2 -- downlink transmit-power control."""
        m = self.model
        if m.n_dl == 0:
            return 0.0, np.zeros(0)
        lb, ub = m.dl_power_bounds(dl_sub)

        def fitness(q: np.ndarray) -> float:            # minimise the negative
            return -m.dl_objective(dl_sub, q, ul_sub, p_ul)

        res = self.tpc.minimize(fitness, lb, ub)
        # never return something worse than the trivial equal-split allocation
        fallback = m.equal_split_dl_power(dl_sub)
        if -m.dl_objective(dl_sub, fallback, ul_sub, p_ul) < res.score:
            return -m.dl_objective(dl_sub, fallback, ul_sub, p_ul), fallback
        return res.score, res.position

    # ------------------------------------------------------------------ #
    # fitness of one binary whale
    # ------------------------------------------------------------------ #
    def evaluate(self, assoc: np.ndarray, incumbent: float = -np.inf) -> _Inner:
        m, cfg = self.model, self.algo
        ul_sub, dl_sub = m.decode(assoc)
        penalty = m.association_penalty(assoc) + self.scheme.penalty(m, assoc)

        fixed = m.ul_utility_upper_bound(ul_sub) + m.dl_utility_upper_bound(dl_sub) - penalty

        zero_p = np.full(m.n_ul, m.p.p_min)
        zero_q = np.zeros(m.n_dl)

        # ---- Lemma 2(i) with the uniform bound ------------------------- #
        if cfg.use_early_stop and fixed <= incumbent:
            return _Inner(-np.inf, zero_p, zero_q, pruned=True)

        self.n_inner_calls += 1

        # ---- power control at the atomic split, then DL ---------------- #
        xi_ref = m.cochannel_at_sbs(dl_sub, m.equal_split_dl_power(dl_sub))
        rho0, chi0 = m.default_split(ul_sub)
        _, p_ul = self.solve_mpc(ul_sub, xi_ref, rho0, chi0)
        _, q_dl = self.solve_dl_tpc(dl_sub, ul_sub, p_ul)
        # (F, rho, chi) by Props. 1-3 happen inside m.utility

        utility = m.utility(assoc, p_ul, q_dl) - penalty
        return _Inner(utility, p_ul, q_dl)

    # ------------------------------------------------------------------ #
    # Algorithm 3
    # ------------------------------------------------------------------ #
    def solve(self) -> Solution:
        m, cfg, rng = self.model, self.algo, self.rng
        start = time.perf_counter()

        rows, k = m.assoc_shape
        if rows == 0:
            return Solution(0.0, np.zeros((0, k), dtype=np.int8), np.zeros(0),
                            np.zeros(0), np.zeros(0), np.zeros(0))

        pop = self.scheme.seed(m, rng, cfg.n_agents_bwoa).astype(float)
        best_score = -np.inf
        best = _Inner(-np.inf, np.full(m.n_ul, m.p.p_min), np.zeros(m.n_dl))
        best_assoc = pop[0].astype(np.int8)
        curve, stalled = [], 0

        for it in range(cfg.max_iter_bwoa):
            for s in range(pop.shape[0]):
                assoc = pop[s].astype(np.int8)
                inner = self.evaluate(assoc, best_score)
                if inner.pruned:
                    continue
                if inner.utility > best_score:
                    best_score, best, best_assoc = inner.utility, inner, assoc.copy()

            prev = curve[-1] if curve else -np.inf
            curve.append(best_score)
            stalled = stalled + 1 if abs(best_score - prev) < cfg.tol_bwoa else 0
            if stalled >= cfg.patience_bwoa:
                break

            leader = best_assoc.astype(float)
            pop = bwoa_step(pop, leader, it, cfg.max_iter_bwoa, rng, cfg.sigmoid_slope)

        _, rho, chi, f_alloc = m.ul_state(best_assoc, best.p_ul, best.q_dl)
        return Solution(
            utility=float(best_score),
            assoc=best_assoc,
            ul_power=best.p_ul,
            dl_power=best.q_dl,
            server_alloc=f_alloc,
            curve=np.asarray(curve),
            runtime=time.perf_counter() - start,
            n_inner_calls=self.n_inner_calls,
            rho=rho,
            chi=chi,
        )


def solve_block(
    model: SystemModel,
    rng: np.random.Generator,
    tpc: str = "WOA",
    scheme: str = "MF-SIC",
    algo: AlgorithmParams | None = None,
) -> Solution:
    """Convenience wrapper: optimise one time block end to end."""
    return HybridSolver(model, rng, algo=algo, tpc=tpc, scheme=scheme).solve()
