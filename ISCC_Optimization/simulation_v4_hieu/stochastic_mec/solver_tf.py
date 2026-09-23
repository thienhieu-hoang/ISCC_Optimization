"""Algorithm 3: Batched BWOA Solver with TensorFlow."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import tensorflow as tf

from .config import AlgorithmParams
from .optimizers_tf import bwoa_step_tf, make_tpc_tf
from .schemes import Scheme, make_scheme
from .system_tf import SolutionTF, SystemModelTF


@dataclass
class _InnerTF:
    utility: float
    p_ul: np.ndarray
    q_dl: np.ndarray
    pruned: bool = False


class HybridSolverTF:
    """Hybrid <TPC>-BWOA solver accelerated with TensorFlow."""

    def __init__(
        self,
        model: SystemModelTF,
        rng: np.random.Generator,
        algo: AlgorithmParams | None = None,
        tpc: str = "WOA",
        scheme: str | Scheme = "MF-SIC",
    ):
        self.model = model
        self.rng = rng
        self.algo = algo or AlgorithmParams()
        self.tpc_name = tpc.upper()
        self.tpc = make_tpc_tf(tpc, self.algo)
        self.scheme = scheme if isinstance(scheme, Scheme) else make_scheme(scheme, model.n_ul, rng)
        self.n_inner_calls = 0

    # ------------------------------------------------------------------ #
    # Inner Power Control Problems (Batched Swarm)
    # ------------------------------------------------------------------ #
    def solve_mpc(
        self,
        ul_sub: tf.Tensor,
        xi: tf.Tensor,
        rho: tf.Tensor,
        chi: tf.Tensor,
    ) -> tuple[float, tf.Tensor]:
        m = self.model
        n_ul = m.n_ul
        if n_ul == 0:
            return 0.0, tf.zeros(0, dtype=tf.float32)

        active_mask = (ul_sub >= 0)
        active_indices = tf.cast(tf.where(active_mask)[:, 0], tf.int32)

        if tf.size(active_indices) == 0:
            p_full = tf.fill([n_ul], float(m.p.p_min))
            return 0.0, p_full

        num_active = int(tf.size(active_indices))
        lb = tf.fill([num_active], float(m.p.p_min))
        ub = tf.fill([num_active], float(m.p.p_max))

        # Batched fitness closure: x has shape (S, num_active)
        def batched_fitness(x: tf.Tensor) -> tf.Tensor:
            s_size = tf.shape(x)[0]
            # Construct (S, n_ul) power tensor
            p_batch = tf.fill([s_size, n_ul], float(m.p.p_min))
            # Place x into active indices across batch
            # scatter nd
            batch_idx = tf.repeat(tf.range(s_size), num_active)
            active_rep = tf.tile(active_indices, [s_size])
            indices = tf.stack([batch_idx, active_rep], axis=-1)
            updates = tf.reshape(x, [-1])
            p_batch = tf.tensor_scatter_nd_update(p_batch, indices, updates)

            ul_sub_expanded = tf.broadcast_to(ul_sub, [s_size, n_ul])
            xi_expanded = tf.broadcast_to(xi, [s_size, m.K, m.m_ul])
            rho_expanded = tf.broadcast_to(rho, [s_size, n_ul])
            chi_expanded = tf.broadcast_to(chi, [s_size, n_ul])

            return m.mpc_objective_tf(ul_sub_expanded, p_batch, xi_expanded, rho_expanded, chi_expanded)

        res = self.tpc.minimize(batched_fitness, lb, ub)
        
        p_opt = np.full(n_ul, m.p.p_min, dtype=np.float32)
        p_opt[active_indices.numpy()] = res.position
        return res.score, tf.constant(p_opt, dtype=tf.float32)

    def solve_dl_tpc(
        self,
        dl_sub: tf.Tensor,
        ul_sub: tf.Tensor,
        p_ul: tf.Tensor,
    ) -> tuple[float, tf.Tensor]:
        m = self.model
        if m.n_dl == 0:
            return 0.0, tf.zeros(0, dtype=tf.float32)

        lb, ub = m.dl_power_bounds_tf(dl_sub)

        def batched_fitness(q: tf.Tensor) -> tf.Tensor:
            s_size = tf.shape(q)[0]
            dl_sub_exp = tf.broadcast_to(dl_sub, [s_size, m.m_dl])
            ul_sub_exp = tf.broadcast_to(ul_sub, [s_size, m.n_ul])
            p_ul_exp = tf.broadcast_to(p_ul, [s_size, m.n_ul])
            return -m.dl_objective_tf(dl_sub_exp, q, ul_sub_exp, p_ul_exp)

        res = self.tpc.minimize(batched_fitness, lb, ub)
        fallback = m.equal_split_dl_power_tf(dl_sub)

        # Compare with equal-split fallback
        score_fallback = -float(m.dl_objective_tf(dl_sub, fallback, ul_sub, p_ul))
        if score_fallback < res.score:
            return score_fallback, fallback
        return res.score, tf.constant(res.position, dtype=tf.float32)

    # ------------------------------------------------------------------ #
    # Fitness Evaluation of Candidates
    # ------------------------------------------------------------------ #
    def evaluate(self, assoc: tf.Tensor) -> _InnerTF:
        m = self.model
        ul_sub, dl_sub = m.decode_tf(assoc)
        penalty = float(m.association_penalty_tf(assoc)) + self.scheme.penalty(assoc.numpy(), m.n_ul)

        self.n_inner_calls += 1

        xi_ref = m.cochannel_at_sbs_tf(dl_sub, m.equal_split_dl_power_tf(dl_sub))
        rho0 = tf.cast(ul_sub >= 0, tf.float32)
        chi0 = tf.ones_like(rho0)

        _, p_ul = self.solve_mpc(ul_sub, xi_ref, rho0, chi0)
        _, q_dl = self.solve_dl_tpc(dl_sub, ul_sub, p_ul)

        util = float(m.utility_tf(assoc, p_ul, q_dl)) - penalty
        return _InnerTF(util, p_ul.numpy(), q_dl.numpy())

    # ------------------------------------------------------------------ #
    # Algorithm 3 (Outer BWOA Search)
    # ------------------------------------------------------------------ #
    def solve(self) -> SolutionTF:
        m, cfg, rng = self.model, self.algo, self.rng
        start = time.perf_counter()

        rows, k = m.assoc_shape
        if rows == 0:
            return SolutionTF(0.0, np.zeros((0, k), dtype=np.int8), np.zeros(0),
                              np.zeros(0), np.zeros(0), np.zeros(0))

        pop_np = self.scheme.seed(m.assoc_shape, m.n_ul, m.m_dl, rng, cfg.n_agents_bwoa).astype(np.float32)
        pop = tf.constant(pop_np, dtype=tf.float32)

        best_score = -np.inf
        best = _InnerTF(-np.inf, np.full(m.n_ul, m.p.p_min, dtype=np.float32), np.zeros(m.n_dl, dtype=np.float32))
        best_assoc = pop_np[0].astype(np.int8)
        curve, stalled = [], 0

        for it in range(cfg.max_iter_bwoa):
            for s in range(pop.shape[0]):
                assoc_s = pop[s]
                inner = self.evaluate(assoc_s)
                if inner.utility > best_score:
                    best_score = inner.utility
                    best = inner
                    best_assoc = assoc_s.numpy().astype(np.int8)

            prev = curve[-1] if curve else -np.inf
            curve.append(best_score)
            stalled = stalled + 1 if abs(best_score - prev) < cfg.tol_bwoa else 0
            if stalled >= cfg.patience_bwoa:
                break

            leader = tf.constant(best_assoc, dtype=tf.float32)
            pop = bwoa_step_tf(pop, leader, it, cfg.max_iter_bwoa, slope=cfg.sigmoid_slope)

        # Compute final states
        assoc_tf = tf.constant(best_assoc, dtype=tf.float32)
        p_ul_tf = tf.constant(best.p_ul, dtype=tf.float32)
        q_dl_tf = tf.constant(best.q_dl, dtype=tf.float32)
        ul_sub, dl_sub = m.decode_tf(assoc_tf)
        xi = m.cochannel_at_sbs_tf(dl_sub, q_dl_tf)
        rates = m.ul_rates_tf(ul_sub, p_ul_tf, xi, dl_sub)
        rho, chi, f_alloc = m.iscc_allocate_tf(ul_sub, rates, p_ul_tf)

        return SolutionTF(
            utility=float(best_score),
            assoc=best_assoc,
            ul_power=best.p_ul,
            dl_power=best.q_dl,
            server_alloc=f_alloc.numpy(),
            curve=np.asarray(curve, dtype=np.float32),
            runtime=time.perf_counter() - start,
            n_inner_calls=self.n_inner_calls,
            rho=rho.numpy(),
            chi=chi.numpy(),
        )


def solve_block_tf(
    model: SystemModelTF,
    rng: np.random.Generator,
    tpc: str = "WOA",
    scheme: str = "MF-SIC",
    algo: AlgorithmParams | None = None,
) -> SolutionTF:
    return HybridSolverTF(model, rng, algo=algo, tpc=tpc, scheme=scheme).solve()
