"""Algorithm 3: Batched BWOA Solver with Multi-Start PGD on GPU."""

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
    p_ul_tf: tf.Tensor
    q_dl_tf: tf.Tensor
    utility_tf: tf.Tensor
    p_ul_np: np.ndarray | None = None
    q_dl_np: np.ndarray | None = None
    pruned: bool = False

    @property
    def p_ul(self) -> np.ndarray:
        if self.p_ul_np is None:
            self.p_ul_np = self.p_ul_tf.numpy()
        return self.p_ul_np

    @property
    def q_dl(self) -> np.ndarray:
        if self.q_dl_np is None:
            self.q_dl_np = self.q_dl_tf.numpy()
        return self.q_dl_np


@tf.function
def repair_valid_tf(assoc: tf.Tensor, n_ul: int, m_dl: int, k: int) -> tf.Tensor:
    """Enforce physical association constraints on GPU tensor:
    - UL UEs (rows 0 to n_ul-1): <= 1 subchannel (local computing if 0)
    - DL UAVs (rows n_ul to n_ul+m_dl-1): == 1 subchannel
    """
    if k == 0:
        return assoc
    rows = n_ul + m_dl
    if rows == 0:
        return assoc

    parts = []
    if n_ul > 0:
        ul_part = assoc[:n_ul]
        ul_sum = tf.reduce_sum(ul_part, axis=-1, keepdims=True)
        ul_noise = tf.random.uniform(tf.shape(ul_part), 0.0, 1e-4, dtype=tf.float32)
        ul_choice = tf.argmax(ul_part * 10.0 + ul_noise, axis=-1)
        ul_one_hot = tf.one_hot(ul_choice, depth=k, dtype=tf.float32)
        repaired_ul = tf.where(ul_sum > 1.0, ul_one_hot, ul_part)
        repaired_ul = tf.where(repaired_ul >= 0.5, 1.0, 0.0)
        parts.append(repaired_ul)

    if m_dl > 0:
        dl_part = assoc[n_ul:n_ul + m_dl]
        dl_noise = tf.random.uniform(tf.shape(dl_part), 0.0, 1e-4, dtype=tf.float32)
        dl_choice = tf.argmax(dl_part * 10.0 + dl_noise, axis=-1)
        repaired_dl = tf.one_hot(dl_choice, depth=k, dtype=tf.float32)
        parts.append(repaired_dl)

    return tf.concat(parts, axis=0) if len(parts) > 1 else parts[0]


@tf.function
def repair_population_tf(pop: tf.Tensor, n_ul: int, m_dl: int, k: int) -> tf.Tensor:
    """Batch repair of an entire population tensor (S, rows, k) on GPU."""
    if k == 0:
        return pop
    parts = []
    if n_ul > 0:
        ul_part = pop[:, :n_ul, :]
        ul_sum = tf.reduce_sum(ul_part, axis=-1, keepdims=True)
        ul_noise = tf.random.uniform(tf.shape(ul_part), 0.0, 1e-4, dtype=tf.float32)
        ul_choice = tf.argmax(ul_part * 10.0 + ul_noise, axis=-1)
        ul_one_hot = tf.one_hot(ul_choice, depth=k, dtype=tf.float32)
        repaired_ul = tf.where(ul_sum > 1.0, ul_one_hot, ul_part)
        repaired_ul = tf.where(repaired_ul >= 0.5, 1.0, 0.0)
        parts.append(repaired_ul)

    if m_dl > 0:
        dl_part = pop[:, n_ul:n_ul + m_dl, :]
        dl_noise = tf.random.uniform(tf.shape(dl_part), 0.0, 1e-4, dtype=tf.float32)
        dl_choice = tf.argmax(dl_part * 10.0 + dl_noise, axis=-1)
        repaired_dl = tf.one_hot(dl_choice, depth=k, dtype=tf.float32)
        parts.append(repaired_dl)

    return tf.concat(parts, axis=1) if len(parts) > 1 else parts[0]


@tf.function
def perturb_valid_tf(assoc: tf.Tensor, n_ul: int, m_dl: int, k: int) -> tf.Tensor:
    """Make a valid 1-step perturbation directly on GPU:
    reassign 1 random UE or DL UAV to a different valid channel.
    """
    rows = n_ul + m_dl
    if rows == 0 or k <= 1:
        return repair_valid_tf(assoc, n_ul, m_dl, k)

    row = tf.random.uniform([], minval=0, maxval=rows, dtype=tf.int32)
    current_chan = tf.argmax(assoc[row], axis=-1, output_type=tf.int32)
    has_chan = tf.reduce_sum(assoc[row]) > 0.5

    shift = tf.random.uniform([], minval=1, maxval=k, dtype=tf.int32)
    new_chan = (current_chan + shift) % k

    is_ul = row < n_ul
    turn_off = is_ul & has_chan & (tf.random.uniform([]) < 0.25)

    new_row = tf.cond(
        turn_off,
        lambda: tf.zeros([k], dtype=tf.float32),
        lambda: tf.one_hot(new_chan, depth=k, dtype=tf.float32),
    )

    indices = tf.reshape(row, [1, 1])
    updates = tf.reshape(new_row, [1, k])
    perturbed = tf.tensor_scatter_nd_update(assoc, indices, updates)
    return repair_valid_tf(perturbed, n_ul, m_dl, k)


def _repair_valid(assoc_np: np.ndarray, n_ul: int, m_dl: int, k: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Compatibility wrapper around repair_valid_tf."""
    assoc_tf = tf.constant(assoc_np, dtype=tf.float32)
    return repair_valid_tf(assoc_tf, n_ul, m_dl, k).numpy().astype(assoc_np.dtype)


def _perturb_valid(assoc_np: np.ndarray, n_ul: int, m_dl: int, k: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Compatibility wrapper around perturb_valid_tf."""
    assoc_tf = tf.constant(assoc_np, dtype=tf.float32)
    return perturb_valid_tf(assoc_tf, n_ul, m_dl, k).numpy().astype(assoc_np.dtype)


def scheme_penalty_tf(scheme: Scheme, assoc: tf.Tensor, n_ul: int) -> tf.Tensor:
    """Evaluate access scheme penalty on GPU tensor."""
    name = scheme.name.upper()
    if name == "MF-SIC":
        return tf.constant(0.0, dtype=tf.float32)
    a_ul = assoc[:n_ul]
    a_dl = assoc[n_ul:]
    big = tf.constant(1e14, dtype=tf.float32)
    if name == "ARJOA":
        g = tf.reduce_sum(a_ul, axis=-1) - 1.0
        return big * tf.reduce_sum(tf.square(g))
    if name == "IOJOA":
        dec = tf.constant(scheme.offload_decision, dtype=tf.float32) if scheme.offload_decision is not None else tf.ones(n_ul, dtype=tf.float32)
        g = tf.reduce_sum(a_ul, axis=-1) - dec
        return big * tf.reduce_sum(tf.square(g))
    if name == "ALCA":
        return big * tf.reduce_sum(a_ul)
    if name == "FDMA":
        load = tf.reduce_sum(a_ul, axis=0) + tf.reduce_sum(a_dl, axis=0)
        g = tf.maximum(load - 1.0, 0.0)
        return big * tf.reduce_sum(tf.square(g))
    return tf.constant(0.0, dtype=tf.float32)


def get_tabu_key(assoc: tf.Tensor) -> bytes:
    """Extract a fast, unique byte key for the association matrix for Tabu table lookup."""
    return bytes(tf.cast(assoc >= 0.5, tf.int8).numpy().tobytes())


class HybridSolverTF:
    """Hybrid <TPC>-BWOA solver accelerated with TensorFlow on GPU."""

    def __init__(
        self,
        model: SystemModelTF,
        rng: np.random.Generator,
        algo: AlgorithmParams | None = None,
        tpc: str = "PGD",
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
        self.n_flips = 0

    # ------------------------------------------------------------------ #
    # Inner Power Control Problems (Multi-Start PGD / Swarm on GPU)
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

        # Build structured multi-start seeds:
        # 1. p_max (full power budget)
        p_max_active = ub
        # 2. p_dagger (Lemma 1 analytical bisection root)
        p_dag_full = m.compute_p_dagger_tf(ul_sub, rho, chi)
        p_dag_active = tf.gather(p_dag_full, active_indices)
        # 3. p_half (mid-range power)
        p_half_active = 0.5 * (lb + ub)
        # 4. p_low (low power budget)
        p_low_active = lb + 0.1 * (ub - lb)

        seeds = tf.stack([p_max_active, p_dag_active, p_half_active, p_low_active], axis=0)
        s_size = int(tf.shape(seeds)[0]) if self.tpc_name in ("PGD", "MULTI_PGD") else self.algo.n_agents_tpc

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

        res = self.tpc.minimize(batched_fitness, lb, ub, seeds=seeds)
        
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

        # Multi-start seeds for DL powers:
        q_seeds = tf.stack([ub, 0.5 * (lb + ub), lb + 0.1 * (ub - lb), lb], axis=0)
        s_size = int(tf.shape(q_seeds)[0]) if self.tpc_name in ("PGD", "MULTI_PGD") else self.algo.n_agents_tpc

        # Precompute static broadcasted tensors outside the loop
        dl_sub_exp = tf.broadcast_to(dl_sub, [s_size, m.m_dl])
        ul_sub_exp = tf.broadcast_to(ul_sub, [s_size, m.n_ul])
        p_ul_exp = tf.broadcast_to(p_ul, [s_size, m.n_ul])

        def batched_fitness(q: tf.Tensor) -> tf.Tensor:
            return -m.dl_objective_tf(dl_sub_exp, q, ul_sub_exp, p_ul_exp)

        res = self.tpc.minimize(batched_fitness, lb, ub, seeds=q_seeds)
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
        scheme_pen_tf = scheme_penalty_tf(self.scheme, assoc, m.n_ul)
        if float(scheme_pen_tf) > 0.0:
            return _InnerTF(
                utility=float(-scheme_pen_tf),
                p_ul_tf=tf.fill([m.n_ul], float(m.p.p_min)),
                q_dl_tf=tf.zeros(m.n_dl, dtype=tf.float32),
                utility_tf=-scheme_pen_tf,
            )

        self.n_inner_calls += 1

        ul_sub, dl_sub = m.decode_tf(assoc)
        xi_ref = m.cochannel_at_sbs_tf(dl_sub, m.equal_split_dl_power_tf(dl_sub))
        rho0 = tf.cast(ul_sub >= 0, tf.float32)
        chi0 = tf.ones_like(rho0)

        _, p_ul = self.solve_mpc(ul_sub, xi_ref, rho0, chi0)
        _, q_dl = self.solve_dl_tpc(dl_sub, ul_sub, p_ul)

        util_tf = m.utility_tf(assoc, p_ul, q_dl)
        return _InnerTF(
            utility=float(util_tf),
            p_ul_tf=p_ul,
            q_dl_tf=q_dl,
            utility_tf=util_tf,
        )

    # ------------------------------------------------------------------ #
    # Algorithm 3 (Outer BWOA Search on GPU)
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

        # Initialize population on GPU
        pop_np = self.scheme.seed(m.assoc_shape, m.n_ul, m.m_dl, rng, cfg.n_agents_bwoa).astype(np.float32)
        pop = tf.constant(pop_np, dtype=tf.float32)
        pop = repair_population_tf(pop, m.n_ul, m.m_dl, k)

        best_score_tf = tf.constant(-np.inf, dtype=tf.float32)
        best_assoc_tf = pop[0]
        best_p_ul_tf = tf.fill([m.n_ul], float(m.p.p_min))
        best_q_dl_tf = tf.zeros(m.n_dl, dtype=tf.float32)
        curve, stalled = [], 0

        for it in range(cfg.max_iter_bwoa):
            new_pop_list = []

            # Sequential BWOA agent evaluation on GPU with Tabu table memoization
            for s in range(pop.shape[0]):
                assoc_s = pop[s]
                assoc_s = repair_valid_tf(assoc_s, m.n_ul, m.m_dl, k)

                if use_cache:
                    key = get_tabu_key(assoc_s)
                    retries = 0
                    while key in self.eval_cache and retries < max_retries:
                        assoc_s = perturb_valid_tf(assoc_s, m.n_ul, m.m_dl, k)
                        key = get_tabu_key(assoc_s)
                        retries += 1

                    if retries > 0:
                        self.n_flips += 1

                    if key in self.eval_cache:
                        self.n_cache_hits += 1
                        inner = self.eval_cache[key]
                    else:
                        inner = self.evaluate(assoc_s)
                        self.eval_cache[key] = inner
                else:
                    inner = self.evaluate(assoc_s)

                new_pop_list.append(assoc_s)

                if inner.utility > float(best_score_tf):
                    best_score_tf = inner.utility_tf
                    best_assoc_tf = assoc_s
                    best_p_ul_tf = inner.p_ul_tf
                    best_q_dl_tf = inner.q_dl_tf

            # Re-stack population on GPU
            pop = tf.stack(new_pop_list, axis=0)

            cur_best = float(best_score_tf)
            prev = curve[-1] if curve else -np.inf
            curve.append(cur_best)
            stalled = stalled + 1 if abs(cur_best - prev) < cfg.tol_bwoa else 0
            if stalled >= cfg.patience_bwoa:
                break

            # Vectorized BWOA position update step ON GPU
            leader = best_assoc_tf
            t_tf = tf.constant(it, dtype=tf.int32)
            max_iter_tf = tf.constant(cfg.max_iter_bwoa, dtype=tf.int32)
            pop = bwoa_step_tf(pop, leader, t_tf, max_iter_tf, slope=cfg.sigmoid_slope)

        # Compute final states on GPU
        ul_sub, dl_sub = m.decode_tf(best_assoc_tf)
        xi = m.cochannel_at_sbs_tf(dl_sub, best_q_dl_tf)
        rates = m.ul_rates_tf(ul_sub, best_p_ul_tf, xi, dl_sub)
        rho, chi, f_alloc = m.iscc_allocate_tf(ul_sub, rates, best_p_ul_tf)

        return SolutionTF(
            utility=float(best_score_tf),
            assoc=best_assoc_tf.numpy().astype(np.int8),
            ul_power=best_p_ul_tf.numpy(),
            dl_power=best_q_dl_tf.numpy(),
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
    tpc: str = "PGD",
    scheme: str = "MF-SIC",
    algo: AlgorithmParams | None = None,
) -> SolutionTF:
    return HybridSolverTF(model, rng, algo=algo, tpc=tpc, scheme=scheme).solve()
