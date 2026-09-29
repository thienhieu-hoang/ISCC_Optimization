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


def _repair_valid(assoc_np: np.ndarray, n_ul: int, m_dl: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """Ensure every row satisfies association constraints (UL: <=1, DL: ==1)."""
    repaired = assoc_np.copy()
    for row in range(n_ul):
        chans = np.flatnonzero(repaired[row, :])
        if chans.size > 1:
            repaired[row, :] = 0
            repaired[row, int(rng.choice(chans))] = 1
    for m_idx in range(m_dl):
        row = n_ul + m_idx
        chans = np.flatnonzero(repaired[row, :])
        if chans.size != 1:
            repaired[row, :] = 0
            repaired[row, int(rng.choice(chans)) if chans.size > 1 else int(rng.integers(k))] = 1
    return repaired


def _perturb_valid(assoc_np: np.ndarray, n_ul: int, m_dl: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """Make a valid 1-step move: reassign 1 random UE or DL UAV to a different subchannel."""
    new_assoc = _repair_valid(assoc_np, n_ul, m_dl, k, rng)
    rows = n_ul + m_dl
    if rows == 0 or k == 0:
        return new_assoc

    row = int(rng.integers(rows))
    if row < n_ul:
        current_chans = np.flatnonzero(new_assoc[row, :])
        curr = int(current_chans[0]) if current_chans.size > 0 else -1
        choices = [c for c in range(-1, k) if c != curr]
        choice = int(rng.choice(choices))
        new_assoc[row, :] = 0
        if choice >= 0:
            new_assoc[row, choice] = 1
    else:
        current_chans = np.flatnonzero(new_assoc[row, :])
        curr = int(current_chans[0]) if current_chans.size > 0 else -1
        choices = [c for c in range(k) if c != curr] if k > 1 else [0]
        choice = int(rng.choice(choices))
        new_assoc[row, :] = 0
        new_assoc[row, choice] = 1

    return new_assoc


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
        self.eval_cache: dict[bytes, _InnerTF] = {}
        self.n_cache_hits = 0

    # ------------------------------------------------------------------ #
    # Inner Power Control Problems (Batched Swarm)
    # ------------------------------------------------------------------ #
    def solve_mpc(
        self,
        ul_sub: tf.Tensor,
        xi: tf.Tensor,
        rho: tf.Tensor,
        chi: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        m = self.model
        n_ul = m.n_ul
        if n_ul == 0:
            return tf.constant(0.0, dtype=tf.float32), tf.zeros(0, dtype=tf.float32)

        active_mask = (ul_sub >= 0)
        active_indices = tf.cast(tf.where(active_mask)[:, 0], tf.int32)

        if tf.size(active_indices) == 0:
            p_full = tf.fill([n_ul], float(m.p.p_min))
            return tf.constant(0.0, dtype=tf.float32), p_full

        num_active = int(tf.size(active_indices))
        lb = tf.fill([num_active], float(m.p.p_min))
        ub = tf.fill([num_active], float(m.p.p_max))
        s_size = self.algo.n_agents_tpc

        # Precompute static broadcasted tensors outside the inner optimization loop
        batch_idx = tf.repeat(tf.range(s_size), num_active)
        active_rep = tf.tile(active_indices, [s_size])
        scatter_indices = tf.stack([batch_idx, active_rep], axis=-1)
        p_base = tf.fill([s_size, n_ul], float(m.p.p_min))

        ul_sub_expanded = tf.broadcast_to(ul_sub, [s_size, n_ul])
        xi_expanded = tf.broadcast_to(xi, [s_size, m.K, m.m_ul])
        rho_expanded = tf.broadcast_to(rho, [s_size, n_ul])
        chi_expanded = tf.broadcast_to(chi, [s_size, n_ul])

        # Batched fitness closure: x has shape (S, num_active)
        def batched_fitness(x: tf.Tensor) -> tf.Tensor:
            updates = tf.reshape(x, [-1])
            p_batch = tf.tensor_scatter_nd_update(p_base, scatter_indices, updates)
            return m.mpc_objective_tf(ul_sub_expanded, p_batch, xi_expanded, rho_expanded, chi_expanded)

        res = self.tpc.minimize(batched_fitness, lb, ub)
        
        p_full = tf.fill([n_ul], float(m.p.p_min))
        pos_tf = res.position_tf if res.position_tf is not None else tf.constant(res.position, dtype=tf.float32)
        p_opt_tf = tf.tensor_scatter_nd_update(
            p_full,
            tf.expand_dims(active_indices, -1),
            pos_tf,
        )
        score_tf = res.score_tf if res.score_tf is not None else tf.constant(res.score, dtype=tf.float32)
        return score_tf, p_opt_tf

    def solve_dl_tpc(
        self,
        dl_sub: tf.Tensor,
        ul_sub: tf.Tensor,
        p_ul: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        m = self.model
        if m.n_dl == 0:
            return tf.constant(0.0, dtype=tf.float32), tf.zeros(0, dtype=tf.float32)

        lb, ub = m.dl_power_bounds_tf(dl_sub)
        s_size = self.algo.n_agents_tpc

        # Precompute static broadcasted tensors outside the loop
        dl_sub_exp = tf.broadcast_to(dl_sub, [s_size, m.m_dl])
        ul_sub_exp = tf.broadcast_to(ul_sub, [s_size, m.n_ul])
        p_ul_exp = tf.broadcast_to(p_ul, [s_size, m.n_ul])

        def batched_fitness(q: tf.Tensor) -> tf.Tensor:
            return -m.dl_objective_tf(dl_sub_exp, q, ul_sub_exp, p_ul_exp)

        res = self.tpc.minimize(batched_fitness, lb, ub)
        fallback = m.equal_split_dl_power_tf(dl_sub)

        pos_tf = res.position_tf if res.position_tf is not None else tf.constant(res.position, dtype=tf.float32)
        score_res_tf = res.score_tf if res.score_tf is not None else tf.constant(res.score, dtype=tf.float32)

        # Compare with equal-split fallback on GPU:
        score_fallback_tf = -m.dl_objective_tf(dl_sub, fallback, ul_sub, p_ul)
        use_fallback = score_fallback_tf < score_res_tf
        q_opt = tf.where(use_fallback, fallback, pos_tf)
        best_score_tf = tf.where(use_fallback, score_fallback_tf, score_res_tf)
        return best_score_tf, q_opt

    # ------------------------------------------------------------------ #
    # Fitness Evaluation of Candidates
    # ------------------------------------------------------------------ #
    def evaluate(self, assoc: tf.Tensor, assoc_np: np.ndarray | None = None) -> _InnerTF:
        m = self.model
        if assoc_np is None:
            assoc_np = assoc.numpy().astype(np.int8)

        # Fast CPU scheme check before invoking inner GPU solvers
        scheme_pen = self.scheme.penalty(assoc_np, m.n_ul)
        if scheme_pen > 0:
            return _InnerTF(-scheme_pen, np.full(m.n_ul, m.p.p_min, dtype=np.float32), np.zeros(m.n_dl, dtype=np.float32))

        self.n_inner_calls += 1

        ul_sub, dl_sub = m.decode_tf(assoc)
        xi_ref = m.cochannel_at_sbs_tf(dl_sub, m.equal_split_dl_power_tf(dl_sub))
        rho0 = tf.cast(ul_sub >= 0, tf.float32)
        chi0 = tf.ones_like(rho0)

        _, p_ul = self.solve_mpc(ul_sub, xi_ref, rho0, chi0)
        _, q_dl = self.solve_dl_tpc(dl_sub, ul_sub, p_ul)

        util_tf = m.utility_tf(assoc, p_ul, q_dl)
        return _InnerTF(float(util_tf), p_ul.numpy(), q_dl.numpy())

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

        self.eval_cache.clear()
        self.n_cache_hits = 0
        self.n_flips = 0
        use_cache = getattr(cfg, "enable_cache", True)
        max_retries = getattr(cfg, "cache_max_retries", 0)

        pop_np = self.scheme.seed(m.assoc_shape, m.n_ul, m.m_dl, rng, cfg.n_agents_bwoa).astype(np.float32)
        pop = tf.constant(pop_np, dtype=tf.float32)

        best_score = -np.inf
        best = _InnerTF(-np.inf, np.full(m.n_ul, m.p.p_min, dtype=np.float32), np.zeros(m.n_dl, dtype=np.float32))
        best_assoc = pop_np[0].astype(np.int8)
        curve, stalled = [], 0

        for it in range(cfg.max_iter_bwoa):
            new_pop_np = pop.numpy().astype(np.int8)
            for s in range(pop.shape[0]):
                assoc_s_np = _repair_valid(new_pop_np[s], m.n_ul, m.m_dl, k, rng)

                if use_cache:
                    key = assoc_s_np.tobytes()
                    retries = 0
                    while key in self.eval_cache and retries < max_retries:
                        assoc_s_np = _perturb_valid(assoc_s_np, m.n_ul, m.m_dl, k, rng)
                        key = assoc_s_np.tobytes()
                        retries += 1

                    if retries > 0:
                        self.n_flips += 1

                    new_pop_np[s] = assoc_s_np

                    if key in self.eval_cache:
                        self.n_cache_hits += 1
                        inner = self.eval_cache[key]
                    else:
                        assoc_s_tf = tf.constant(assoc_s_np, dtype=tf.float32)
                        inner = self.evaluate(assoc_s_tf, assoc_s_np)
                        self.eval_cache[key] = inner
                else:
                    new_pop_np[s] = assoc_s_np
                    assoc_s_tf = tf.constant(assoc_s_np, dtype=tf.float32)
                    inner = self.evaluate(assoc_s_tf, assoc_s_np)

                if inner.utility > best_score:
                    best_score = inner.utility
                    best = inner
                    best_assoc = assoc_s_np.copy()

            prev = curve[-1] if curve else -np.inf
            curve.append(best_score)
            stalled = stalled + 1 if abs(best_score - prev) < cfg.tol_bwoa else 0
            if stalled >= cfg.patience_bwoa:
                break

            pop = tf.constant(new_pop_np, dtype=tf.float32)
            leader = tf.constant(best_assoc, dtype=tf.float32)
            t_tf = tf.constant(it, dtype=tf.int32)
            max_iter_tf = tf.constant(cfg.max_iter_bwoa, dtype=tf.int32)
            pop = bwoa_step_tf(pop, leader, t_tf, max_iter_tf, slope=cfg.sigmoid_slope)

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
            n_cache_hits=self.n_cache_hits,
            n_flips=self.n_flips,
        )


def solve_block_tf(
    model: SystemModelTF,
    rng: np.random.Generator,
    tpc: str = "WOA",
    scheme: str = "MF-SIC",
    algo: AlgorithmParams | None = None,
) -> SolutionTF:
    return HybridSolverTF(model, rng, algo=algo, tpc=tpc, scheme=scheme).solve()
