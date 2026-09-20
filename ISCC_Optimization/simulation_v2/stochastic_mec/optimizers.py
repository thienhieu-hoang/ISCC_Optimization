"""The unified nature-inspired optimisation framework.

Three interchangeable continuous minimisers (WOA, IWOA, PSO) share the loop
skeleton of Algorithms 1 and 2 of the paper -- evaluate every search agent,
track the leader, then move the agents -- and differ only in the update rule.
A binary variant (BWOA) drives the sub-channel-allocation search of
Algorithm 3.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .config import AlgorithmParams

Fitness = Callable[[np.ndarray], float]


@dataclass
class OptResult:
    score: float
    position: np.ndarray
    curve: np.ndarray
    n_eval: int = 0


class ContinuousOptimizer(ABC):
    """Minimise ``fitness`` over the box ``[lb, ub]``."""

    name = "base"

    def __init__(self, params: AlgorithmParams, rng: np.random.Generator):
        self.params = params
        self.rng = rng

    # -- to be provided by each algorithm ------------------------------- #
    @abstractmethod
    def _init_population(self, lb: np.ndarray, ub: np.ndarray, n_agents: int) -> np.ndarray:
        ...

    @abstractmethod
    def _move(self, pos: np.ndarray, leader: np.ndarray, t: int, max_iter: int) -> np.ndarray:
        ...

    # -- shared driver --------------------------------------------------- #
    def minimize(self, fitness: Fitness, lb: np.ndarray, ub: np.ndarray) -> OptResult:
        cfg = self.params
        dim = lb.size
        if dim == 0:
            return OptResult(0.0, np.zeros(0), np.zeros(0))

        pos = self._init_population(lb, ub, cfg.n_agents_tpc)
        leader, leader_score = pos[0].copy(), np.inf
        curve, stalled, n_eval = [], 0, 0

        for t in range(cfg.max_iter_tpc):
            pos = np.clip(pos, lb, ub)
            for agent in pos:
                score = fitness(agent)
                n_eval += 1
                if score < leader_score:
                    leader_score, leader = score, agent.copy()

            prev = curve[-1] if curve else np.inf
            curve.append(leader_score)
            stalled = stalled + 1 if abs(leader_score - prev) < cfg.tol_tpc else 0
            if stalled >= cfg.patience_tpc:
                break

            pos = self._move(pos, leader, t, cfg.max_iter_tpc)

        return OptResult(float(leader_score), leader, np.asarray(curve), n_eval)


# --------------------------------------------------------------------- #
# Whale Optimization Algorithm
# --------------------------------------------------------------------- #
class WOA(ContinuousOptimizer):
    name = "WOA"

    def _init_population(self, lb, ub, n_agents):
        return lb + self.rng.random((n_agents, lb.size)) * (ub - lb)

    def _woa_step(self, pos: np.ndarray, leader: np.ndarray, t: int, max_iter: int) -> np.ndarray:
        rng = self.rng
        n_agents, dim = pos.shape
        a = 2.0 - 2.0 * t / max_iter                     # 2 -> 0
        a2 = -1.0 - t / max_iter                         # -1 -> -2
        new = pos.copy()
        for i in range(n_agents):
            A = 2.0 * a * rng.random() - a
            C = 2.0 * rng.random()
            b, l = 1.0, (a2 - 1.0) * rng.random() + 1.0
            if rng.random() < 0.5:
                if abs(A) >= 1.0:                        # exploration
                    rand_agent = pos[rng.integers(n_agents)]
                    d = np.abs(C * rand_agent - pos[i])
                    new[i] = rand_agent - A * d
                else:                                    # shrinking encircling
                    d = np.abs(C * leader - pos[i])
                    new[i] = leader - A * d
            else:                                        # spiral update
                d = np.abs(leader - pos[i])
                new[i] = d * np.exp(b * l) * np.cos(2.0 * np.pi * l) + leader
        return new

    def _move(self, pos, leader, t, max_iter):
        return self._woa_step(pos, leader, t, max_iter)


# --------------------------------------------------------------------- #
# Improved WOA: the population grows / shrinks with the search progress
# --------------------------------------------------------------------- #
class IWOA(WOA):
    name = "IWOA"

    def minimize(self, fitness: Fitness, lb: np.ndarray, ub: np.ndarray) -> OptResult:
        cfg = self.params
        rng = self.rng
        dim = lb.size
        if dim == 0:
            return OptResult(0.0, np.zeros(0), np.zeros(0))

        pos = self._init_population(lb, ub, cfg.n_agents_tpc)
        leader, leader_score = pos[0].copy(), np.inf
        curve, stalled, n_eval = [], 0, 0
        leader_history: list[np.ndarray] = []

        for t in range(cfg.max_iter_tpc):
            pos = np.clip(pos, lb, ub)
            for agent in pos:
                score = fitness(agent)
                n_eval += 1
                if score < leader_score:
                    leader_score, leader = score, agent.copy()

            prev = curve[-1] if curve else np.inf
            curve.append(leader_score)
            stalled = stalled + 1 if abs(leader_score - prev) < cfg.tol_tpc else 0
            if stalled >= cfg.patience_tpc:
                break

            leader_history.append(leader.copy())
            pos = self._resize(pos, leader, leader_history)
            pos = self._woa_step(pos, leader, t, cfg.max_iter_tpc)

        return OptResult(float(leader_score), leader, np.asarray(curve), n_eval)

    def _resize(self, pos: np.ndarray, leader: np.ndarray, history: list[np.ndarray]) -> np.ndarray:
        """Shrink when the leader keeps improving, grow when the search stalls."""
        cfg, rng = self.params, self.rng
        if len(history) < 3:
            return pos
        improving = not np.allclose(history[-1], history[-2]) and not np.allclose(history[-2], history[-3])
        stalling = np.allclose(history[-1], history[-2])
        n = pos.shape[0]

        dist = np.linalg.norm(pos - leader, axis=1)
        if improving and n > cfg.pop_min:
            n_drop = max(1, int(round(n * (cfg.pop_max - n) ** 2 / cfg.pop_max ** 2)))
            n_drop = min(n_drop, n - cfg.pop_min)
            keep = np.argsort(dist)[: n - n_drop]        # drop the most distant agents
            return pos[keep]
        if stalling and n < cfg.pop_max:
            n_add = max(1, int(round(n * (cfg.pop_max - n) ** 2 / cfg.pop_max ** 2)))
            n_add = min(n_add, cfg.pop_max - n)
            order = np.argsort(dist)                     # best agents of the swarm
            pool = order[: max(2, min(n, 2 * n_add))]
            alpha = cfg.iwoa_alpha
            children = []
            for _ in range(n_add):
                i, j = rng.choice(pool, size=2, replace=False)
                children.append(np.sqrt(alpha) * pos[i] + np.sqrt(1.0 - alpha) * pos[j])
            return np.vstack([pos, np.asarray(children)])
        return pos


# --------------------------------------------------------------------- #
# Particle Swarm Optimization
# --------------------------------------------------------------------- #
class PSO(ContinuousOptimizer):
    name = "PSO"

    def _init_population(self, lb, ub, n_agents):
        self._vel = np.zeros((n_agents, lb.size))
        self._vmax = 0.1 * (ub - lb)
        self._best_pos = None
        self._best_cost = None
        self._w = self.params.pso_inertia
        self._lb, self._ub = lb, ub
        return lb + self.rng.random((n_agents, lb.size)) * (ub - lb)

    def minimize(self, fitness: Fitness, lb: np.ndarray, ub: np.ndarray) -> OptResult:
        cfg, rng = self.params, self.rng
        dim = lb.size
        if dim == 0:
            return OptResult(0.0, np.zeros(0), np.zeros(0))

        pos = self._init_population(lb, ub, cfg.n_agents_tpc)
        cost = np.array([fitness(x) for x in pos])
        n_eval = pos.shape[0]
        self._best_pos, self._best_cost = pos.copy(), cost.copy()
        g = int(np.argmin(cost))
        leader, leader_score = pos[g].copy(), float(cost[g])

        curve, stalled = [], 0
        for _ in range(cfg.max_iter_tpc):
            r1 = rng.random(pos.shape)
            r2 = rng.random(pos.shape)
            self._vel = (
                self._w * self._vel
                + cfg.pso_c1 * r1 * (self._best_pos - pos)
                + cfg.pso_c2 * r2 * (leader - pos)
            )
            self._vel = np.clip(self._vel, -self._vmax, self._vmax)
            pos = pos + self._vel
            outside = (pos < lb) | (pos > ub)
            self._vel[outside] *= -1.0                    # velocity mirror effect
            pos = np.clip(pos, lb, ub)

            for i, x in enumerate(pos):
                c = fitness(x)
                n_eval += 1
                if c < self._best_cost[i]:
                    self._best_cost[i], self._best_pos[i] = c, x.copy()
                    if c < leader_score:
                        leader_score, leader = c, x.copy()

            prev = curve[-1] if curve else np.inf
            curve.append(leader_score)
            stalled = stalled + 1 if abs(leader_score - prev) < cfg.tol_tpc else 0
            if stalled >= cfg.patience_tpc:
                break
            self._w *= cfg.pso_inertia_damp

        return OptResult(float(leader_score), leader, np.asarray(curve), n_eval)

    def _move(self, pos, leader, t, max_iter):   # pragma: no cover - unused
        raise NotImplementedError


TPC_ALGORITHMS: dict[str, type[ContinuousOptimizer]] = {
    "WOA": WOA,
    "IWOA": IWOA,
    "PSO": PSO,
}


def make_tpc(name: str, params: AlgorithmParams, rng: np.random.Generator) -> ContinuousOptimizer:
    try:
        return TPC_ALGORITHMS[name.upper()](params, rng)
    except KeyError as exc:  # pragma: no cover
        raise ValueError(f"unknown power-control algorithm {name!r}") from exc


# --------------------------------------------------------------------- #
# Binary WOA position update (used by the SMSA search of Algorithm 3)
# --------------------------------------------------------------------- #
def bwoa_step(
    positions: np.ndarray,
    leader: np.ndarray,
    t: int,
    max_iter: int,
    rng: np.random.Generator,
    slope: float = 10.0,
) -> np.ndarray:
    """Flip the bits of every binary whale with a sigmoid transfer function.

    ``positions`` has shape ``(S, *assoc_shape)`` and holds 0/1 entries.
    """
    n_agents = positions.shape[0]
    a = 2.0 - 2.0 * t / max_iter
    a2 = -1.0 - t / max_iter
    out = positions.copy()
    for s in range(n_agents):
        A = 2.0 * a * rng.random() - a
        C = 2.0 * rng.random()
        b, l = 1.0, (a2 - 1.0) * rng.random() + 1.0
        if rng.random() < 0.5:
            if abs(A) >= 1.0:
                other = positions[rng.integers(n_agents)]
                step = other - A * np.abs(C * other - positions[s])
            else:
                step = leader - A * np.abs(C * leader - positions[s])
        else:
            step = np.abs(leader - positions[s]) * np.exp(b * l) * np.cos(2.0 * np.pi * l) + leader
        flip = rng.random(step.shape) < 1.0 / (1.0 + np.exp(-slope * (step - 0.5)))
        out[s] = np.where(flip, 1 - positions[s], positions[s])
    return out
