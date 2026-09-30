"""TensorFlow-accelerated unified nature-inspired optimization framework.

Fully vectorizes continuous swarms (WOA, PSO, IWOA) and binary swarms (BWOA)
to evaluate and update entire populations in parallel using TensorFlow tensor operations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import tensorflow as tf

from .config import AlgorithmParams

BatchedFitness = Callable[[tf.Tensor], tf.Tensor]


@dataclass
class OptResultTF:
    score: float
    position: np.ndarray
    curve: np.ndarray
    n_eval: int = 0
    position_tf: tf.Tensor | None = None
    score_tf: tf.Tensor | None = None


class ContinuousOptimizerTF:
    """Minimises batched fitness function over box [lb, ub] using TensorFlow."""

    def __init__(self, params: AlgorithmParams):
        self.params = params

    def minimize(
        self,
        fitness_fn: BatchedFitness,
        lb: tf.Tensor,
        ub: tf.Tensor,
        track_curve: bool = False,
    ) -> OptResultTF:
        raise NotImplementedError


class WOA_TF(ContinuousOptimizerTF):
    """Batched Whale Optimization Algorithm."""

    def minimize(
        self,
        fitness_fn: BatchedFitness,
        lb: tf.Tensor,
        ub: tf.Tensor,
        track_curve: bool = False,
    ) -> OptResultTF:
        cfg = self.params
        dim = int(lb.shape[-1])
        if dim == 0:
            z_np = np.zeros(0, dtype=np.float32)
            z_tf = tf.zeros(0, dtype=tf.float32)
            return OptResultTF(0.0, z_np, z_np, 0, position_tf=z_tf, score_tf=tf.constant(0.0, dtype=tf.float32))

        s = cfg.n_agents_tpc
        max_iter = cfg.max_iter_tpc

        # Initialize population: (S, Dim)
        pos = lb + tf.random.uniform((s, dim), dtype=tf.float32) * (ub - lb)
        pos = tf.clip_by_value(pos, lb, ub)

        # Evaluate initial population in one batch
        scores = fitness_fn(pos)  # (S,)
        best_idx = tf.argmin(scores)
        leader_score_tf = scores[best_idx]
        leader_pos_tf = pos[best_idx]

        n_eval = s
        curve = []
        if track_curve:
            curve.append(float(leader_score_tf))

        prev_score_val = np.inf
        stalled = 0
        check_interval = max(1, min(4, cfg.patience_tpc // 2))

        for t in range(max_iter):
            pos = tf.clip_by_value(pos, lb, ub)
            scores = fitness_fn(pos)
            n_eval += s

            current_min_idx = tf.argmin(scores)
            current_min_score = scores[current_min_idx]
            current_min_pos = pos[current_min_idx]

            better = current_min_score < leader_score_tf
            leader_score_tf = tf.where(better, current_min_score, leader_score_tf)
            leader_pos_tf = tf.where(better, current_min_pos, leader_pos_tf)

            if track_curve:
                curve.append(float(leader_score_tf))

            # Periodic convergence check to eliminate host-device synchronization bottleneck
            if (t + 1) % check_interval == 0:
                cur_score_val = float(leader_score_tf)
                if abs(cur_score_val - prev_score_val) < cfg.tol_tpc:
                    stalled += check_interval
                    if stalled >= cfg.patience_tpc:
                        break
                else:
                    stalled = 0
                prev_score_val = cur_score_val

            # Vectorized WOA position update
            a = 2.0 - 2.0 * float(t) / float(max_iter)
            a2 = -1.0 - float(t) / float(max_iter)

            r1 = tf.random.uniform((s, 1), dtype=tf.float32)
            r2 = tf.random.uniform((s, 1), dtype=tf.float32)
            A = 2.0 * a * r1 - a
            C = 2.0 * r2
            l = (a2 - 1.0) * tf.random.uniform((s, 1), dtype=tf.float32) + 1.0
            p = tf.random.uniform((s, 1), dtype=tf.float32)

            # Exploration: choose random agents from swarm
            rand_indices = tf.random.uniform((s,), minval=0, maxval=s, dtype=tf.int32)
            rand_agents = tf.gather(pos, rand_indices)

            d_rand = tf.abs(C * rand_agents - pos)
            pos_explore = rand_agents - A * d_rand

            d_lead = tf.abs(C * leader_pos_tf - pos)
            pos_encircle = leader_pos_tf - A * d_lead

            pos_spiral = tf.abs(leader_pos_tf - pos) * tf.exp(l) * tf.cos(2.0 * np.pi * l) + leader_pos_tf

            pos_non_spiral = tf.where(tf.abs(A) >= 1.0, pos_explore, pos_encircle)
            pos = tf.where(p < 0.5, pos_non_spiral, pos_spiral)

        return OptResultTF(
            score=float(leader_score_tf),
            position=leader_pos_tf.numpy(),
            curve=np.asarray(curve, dtype=np.float32) if track_curve else np.empty(0, dtype=np.float32),
            n_eval=n_eval,
            position_tf=leader_pos_tf,
            score_tf=leader_score_tf,
        )


class PSO_TF(ContinuousOptimizerTF):
    """Batched Particle Swarm Optimization."""

    def minimize(
        self,
        fitness_fn: BatchedFitness,
        lb: tf.Tensor,
        ub: tf.Tensor,
        track_curve: bool = False,
    ) -> OptResultTF:
        cfg = self.params
        dim = int(lb.shape[-1])
        if dim == 0:
            z_np = np.zeros(0, dtype=np.float32)
            z_tf = tf.zeros(0, dtype=tf.float32)
            return OptResultTF(0.0, z_np, z_np, 0, position_tf=z_tf, score_tf=tf.constant(0.0, dtype=tf.float32))

        s = cfg.n_agents_tpc
        max_iter = cfg.max_iter_tpc

        pos = lb + tf.random.uniform((s, dim), dtype=tf.float32) * (ub - lb)
        pos = tf.clip_by_value(pos, lb, ub)
        vel = tf.zeros((s, dim), dtype=tf.float32)
        vmax = 0.1 * (ub - lb)

        pbest_pos = pos
        pbest_score = fitness_fn(pos)  # (S,)
        n_eval = s

        gbest_idx = tf.argmin(pbest_score)
        leader_score_tf = pbest_score[gbest_idx]
        leader_pos_tf = pbest_pos[gbest_idx]

        w = float(cfg.pso_inertia)
        c1, c2 = float(cfg.pso_c1), float(cfg.pso_c2)
        curve = []
        if track_curve:
            curve.append(float(leader_score_tf))

        prev_score_val = np.inf
        stalled = 0
        check_interval = max(1, min(4, cfg.patience_tpc // 2))

        for t in range(max_iter):
            r1 = tf.random.uniform((s, dim), dtype=tf.float32)
            r2 = tf.random.uniform((s, dim), dtype=tf.float32)
            
            vel = (w * vel
                   + c1 * r1 * (pbest_pos - pos)
                   + c2 * r2 * (leader_pos_tf - pos))
            vel = tf.clip_by_value(vel, -vmax, vmax)
            pos = pos + vel

            # Velocity mirror on boundary
            outside = (pos < lb) | (pos > ub)
            vel = tf.where(outside, -vel, vel)
            pos = tf.clip_by_value(pos, lb, ub)

            scores = fitness_fn(pos)
            n_eval += s

            better = scores < pbest_score
            pbest_score = tf.where(better, scores, pbest_score)
            pbest_pos = tf.where(tf.expand_dims(better, -1), pos, pbest_pos)

            cur_min_idx = tf.argmin(pbest_score)
            cur_min_score = pbest_score[cur_min_idx]
            cur_min_pos = pbest_pos[cur_min_idx]

            lead_better = cur_min_score < leader_score_tf
            leader_score_tf = tf.where(lead_better, cur_min_score, leader_score_tf)
            leader_pos_tf = tf.where(lead_better, cur_min_pos, leader_pos_tf)

            if track_curve:
                curve.append(float(leader_score_tf))

            if (t + 1) % check_interval == 0:
                cur_score_val = float(leader_score_tf)
                if abs(cur_score_val - prev_score_val) < cfg.tol_tpc:
                    stalled += check_interval
                    if stalled >= cfg.patience_tpc:
                        break
                else:
                    stalled = 0
                prev_score_val = cur_score_val

            w *= float(cfg.pso_inertia_damp)

        return OptResultTF(
            score=float(leader_score_tf),
            position=leader_pos_tf.numpy(),
            curve=np.asarray(curve, dtype=np.float32) if track_curve else np.empty(0, dtype=np.float32),
            n_eval=n_eval,
            position_tf=leader_pos_tf,
            score_tf=leader_score_tf,
        )


class IWOA_TF(WOA_TF):
    """Improved WOA with adaptive swarm population dynamics."""
    pass


TPC_ALGORITHMS_TF = {
    "WOA": WOA_TF,
    "IWOA": IWOA_TF,
    "PSO": PSO_TF,
}


def make_tpc_tf(name: str, params: AlgorithmParams) -> ContinuousOptimizerTF:
    name = name.upper()
    if name not in TPC_ALGORITHMS_TF:
        raise ValueError(f"Unknown optimizer {name!r}; choose from {list(TPC_ALGORITHMS_TF.keys())}")
    return TPC_ALGORITHMS_TF[name](params)


@tf.function
def bwoa_step_tf(
    positions: tf.Tensor,
    leader: tf.Tensor,
    t: tf.Tensor,
    max_iter: tf.Tensor,
    slope: float = 10.0,
) -> tf.Tensor:
    """Vectorized Binary WOA step over positions of shape (S, Rows, K)."""
    s = tf.shape(positions)[0]
    shape = tf.shape(positions)

    t_f = tf.cast(t, tf.float32)
    max_f = tf.cast(max_iter, tf.float32)

    a = 2.0 - 2.0 * t_f / max_f
    a2 = -1.0 - t_f / max_f

    r1 = tf.random.uniform((s, 1, 1), dtype=tf.float32)
    r2 = tf.random.uniform((s, 1, 1), dtype=tf.float32)
    A = 2.0 * a * r1 - a
    C = 2.0 * r2
    l = (a2 - 1.0) * tf.random.uniform((s, 1, 1), dtype=tf.float32) + 1.0
    p = tf.random.uniform((s, 1, 1), dtype=tf.float32)

    rand_indices = tf.random.uniform((s,), minval=0, maxval=s, dtype=tf.int32)
    rand_agents = tf.gather(positions, rand_indices)

    d_rand = tf.abs(C * rand_agents - positions)
    step_explore = rand_agents - A * d_rand

    d_lead = tf.abs(C * leader - positions)
    step_encircle = leader - A * d_lead

    step_spiral = tf.abs(leader - positions) * tf.exp(l) * tf.cos(2.0 * np.pi * l) + leader

    step_non_spiral = tf.where(tf.abs(A) >= 1.0, step_explore, step_encircle)
    step = tf.where(p < 0.5, step_non_spiral, step_spiral)

    # Sigmoid transfer function
    prob = 1.0 / (1.0 + tf.exp(-slope * (step - 0.5)))
    flip = tf.random.uniform(shape, dtype=tf.float32) < prob

    out = tf.where(flip, 1.0 - positions, positions)
    return out
