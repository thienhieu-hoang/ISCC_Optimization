"""Exhaustive search over the association matrix (the WOA-EX benchmark).

Enumerates all ``(K + 1)^{N_ul} * K^{M_dl}`` sub-channel assignments and reuses
the very same inner power/computation solvers as Algorithm 3, so the two curves
differ only in how ``A_net`` is searched.  Only usable for tiny instances --
the caller is warned above ``max_cases``.
"""

from __future__ import annotations

import itertools
import time

import numpy as np

from .config import AlgorithmParams
from .solver import HybridSolver
from .system import Solution, SystemModel


def count_cases(model: SystemModel) -> int:
    return (model.K + 1) ** model.n_ul * model.K ** model.m_dl


def exhaustive_search(
    model: SystemModel,
    rng: np.random.Generator,
    tpc: str = "WOA",
    algo: AlgorithmParams | None = None,
    max_cases: int = 2_000_000,
) -> Solution:
    n_cases = count_cases(model)
    if n_cases > max_cases:
        raise ValueError(
            f"exhaustive search would visit {n_cases:,} associations "
            f"(limit {max_cases:,}); shrink N_ul, M_dl or K"
        )

    solver = HybridSolver(model, rng, algo=algo, tpc=tpc, scheme="MF-SIC")
    rows, k = model.assoc_shape
    start = time.perf_counter()

    best_score = -np.inf
    best_assoc = np.zeros((rows, k), dtype=np.int8)
    best_p = np.full(model.n_ul, model.p.p_min)
    best_q = np.zeros(model.n_dl)

    ul_choices = itertools.product(range(-1, k), repeat=model.n_ul)   # -1 == local
    for ul in ul_choices:
        for dl in itertools.product(range(k), repeat=model.m_dl):
            assoc = np.zeros((rows, k), dtype=np.int8)
            for n, c in enumerate(ul):
                if c >= 0:
                    assoc[n, c] = 1
            for m, c in enumerate(dl):
                assoc[model.n_ul + m, c] = 1

            inner = solver.evaluate(assoc, best_score)
            if inner.pruned or inner.utility <= best_score:
                continue
            best_score, best_assoc = inner.utility, assoc
            best_p, best_q = inner.p_ul, inner.q_dl

    _, rho, chi, f_alloc = model.ul_state(best_assoc, best_p, best_q)
    return Solution(
        utility=float(best_score),
        assoc=best_assoc,
        ul_power=best_p,
        dl_power=best_q,
        server_alloc=f_alloc,
        curve=np.asarray([best_score]),
        runtime=time.perf_counter() - start,
        n_inner_calls=solver.n_inner_calls,
        rho=rho,
        chi=chi,
    )
