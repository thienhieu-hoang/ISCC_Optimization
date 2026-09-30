"""Exhaustive search over the association matrix (the WOA-EX benchmark with TensorFlow).

Enumerates all (K + 1)^{N_ul} * K^{M_dl} sub-channel assignments and reuses
the very same inner power/computation solvers as Algorithm 3, so the two curves
differ only in how A_net is searched.
"""

from __future__ import annotations

import itertools
import time

import numpy as np
import tensorflow as tf

from .config import AlgorithmParams
from .solver_tf import HybridSolverTF
from .system_tf import SolutionTF, SystemModelTF


def count_cases(model: SystemModelTF) -> int:
    """Number of valid association matrices for this network."""
    return (model.K + 1) ** model.n_ul * model.K ** model.m_dl


def exhaustive_search(
    model: SystemModelTF,
    rng: np.random.Generator,
    tpc: str = "WOA",
    algo: AlgorithmParams | None = None,
    max_cases: int = 2_000_000,
) -> SolutionTF:
    n_cases = count_cases(model)
    if n_cases > max_cases:
        raise ValueError(
            f"exhaustive search would visit {n_cases:,} associations "
            f"(limit {max_cases:,}); shrink N_ul, M_dl or K"
        )

    solver = HybridSolverTF(model, rng, algo=algo, tpc=tpc, scheme="MF-SIC")
    rows, k = model.assoc_shape
    start = time.perf_counter()

    best_score = -np.inf
    best_assoc = np.zeros((rows, k), dtype=np.int8)
    best_p = np.full(model.n_ul, model.p.p_min, dtype=np.float32)
    best_q = np.zeros(model.n_dl, dtype=np.float32)

    ul_choices = itertools.product(range(-1, k), repeat=model.n_ul)  # -1 == local
    for ul in ul_choices:
        for dl in itertools.product(range(k), repeat=model.m_dl):
            assoc = np.zeros((rows, k), dtype=np.int8)
            for n, c in enumerate(ul):
                if c >= 0:
                    assoc[n, c] = 1
            for m, c in enumerate(dl):
                assoc[model.n_ul + m, c] = 1

            assoc_tf = tf.constant(assoc, dtype=tf.float32)
            inner = solver.evaluate(assoc_tf, assoc)
            if inner.utility <= best_score:
                continue
            best_score = inner.utility
            best_assoc = assoc
            best_p = inner.p_ul
            best_q = inner.q_dl

    # Compute final states
    assoc_tf = tf.constant(best_assoc, dtype=tf.float32)
    p_ul_tf = tf.constant(best_p, dtype=tf.float32)
    q_dl_tf = tf.constant(best_q, dtype=tf.float32)
    ul_sub, dl_sub = model.decode_tf(assoc_tf)
    xi = model.cochannel_at_sbs_tf(dl_sub, q_dl_tf)
    rates = model.ul_rates_tf(ul_sub, p_ul_tf, xi, dl_sub)
    rho, chi, f_alloc = model.iscc_allocate_tf(ul_sub, rates, p_ul_tf)

    return SolutionTF(
        utility=float(best_score),
        assoc=best_assoc,
        ul_power=best_p,
        dl_power=best_q,
        server_alloc=f_alloc.numpy(),
        curve=np.asarray([best_score], dtype=np.float32),
        runtime=time.perf_counter() - start,
        n_inner_calls=solver.n_inner_calls,
        rho=rho.numpy(),
        chi=chi.numpy(),
    )
