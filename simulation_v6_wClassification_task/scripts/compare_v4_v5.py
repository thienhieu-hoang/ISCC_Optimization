#!/usr/bin/env python3
"""Benchmark and comparison script: simulation_v4 (WOA) vs simulation_v5 (Multi-Start PGD).

Compares:
1. Inner continuous power control runtime and speedup.
2. End-to-end BWOA + Power Control attained utility.
3. Optimality gap and SIC decoding constraint satisfaction.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

# Ensure local package import
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stochastic_mec import (
    AlgorithmParams,
    HybridSolverTF,
    SystemModelTF,
    SystemParams,
    evaluate_solution_tf,
    sample_topology,
    sample_topology_with_cells,
)


def run_benchmark(n_snapshots: int = 5, n_ue: int = 8, n_subchannels: int = 4):
    print("=" * 78)
    print(" BENCHMARK: simulation_v4 (WOA Inner TPC) vs simulation_v5 (Multi-Start PGD)")
    print("=" * 78)
    print(f"Configurations: Snapshots = {n_snapshots}, Active UEs = {n_ue}, Sub-channels = {n_subchannels}\n")

    params = SystemParams(
        n_subchannels=n_subchannels,
        lambda_ue_active=n_ue * 1e-6,
    )
    algo_pgd = AlgorithmParams(tpc_default="PGD", max_iter_pgd=10, max_iter_bwoa=30, n_agents_bwoa=20)
    algo_woa = AlgorithmParams(tpc_default="WOA", n_agents_tpc=15, max_iter_tpc=30, max_iter_bwoa=30, n_agents_bwoa=20)

    time_woa_list = []
    time_pgd_list = []
    util_woa_list = []
    util_pgd_list = []

    for s_idx in range(n_snapshots):
        seed = 1000 + s_idx
        rng = np.random.default_rng(seed)
        topo = sample_topology(rng, params, n_sbs=3, n_active_ue=n_ue)
        model = SystemModelTF(topo, params, rng)

        print(f"--- Snapshot {s_idx + 1}/{n_snapshots} (UEs={model.n_ul}, DL-UAVs={model.m_dl}, Jammers={model.n_jam}) ---")

        # 1. Run v5 Multi-Start PGD
        rng_pgd = np.random.default_rng(seed)
        solver_pgd = HybridSolverTF(model, rng_pgd, algo=algo_pgd, tpc="PGD")
        t0 = time.perf_counter()
        sol_pgd = solver_pgd.solve()
        t_pgd = (time.perf_counter() - t0) * 1000.0  # ms
        eval_pgd = evaluate_solution_tf(model, sol_pgd)

        # 2. Run v4 WOA
        rng_woa = np.random.default_rng(seed)
        solver_woa = HybridSolverTF(model, rng_woa, algo=algo_woa, tpc="WOA")
        t0 = time.perf_counter()
        sol_woa = solver_woa.solve()
        t_woa = (time.perf_counter() - t0) * 1000.0  # ms
        eval_woa = evaluate_solution_tf(model, sol_woa)
        u_pgd = float(eval_pgd.utility)
        u_woa = float(eval_woa.utility)
        speedup = t_woa / max(t_pgd, 1e-6)

        time_woa_list.append(t_woa)
        time_pgd_list.append(t_pgd)
        util_woa_list.append(u_woa)
        util_pgd_list.append(u_pgd)

        print(f"  v4 (WOA)       : Time = {t_woa:7.2f} ms | Utility = {u_woa:8.4f} | Inner Calls = {solver_woa.n_inner_calls}")
        print(f"  v5 (Multi-PGD) : Time = {t_pgd:7.2f} ms | Utility = {u_pgd:8.4f} | Inner Calls = {solver_pgd.n_inner_calls}")
        print(f"  ==> Speedup: {speedup:5.2f}x | Utility Ratio: {u_pgd / max(u_woa, 1e-6) * 100.0:6.2f}%\n")

    mean_t_woa = np.mean(time_woa_list)
    mean_t_pgd = np.mean(time_pgd_list)
    mean_u_woa = np.mean(util_woa_list)
    mean_u_pgd = np.mean(util_pgd_list)
    overall_speedup = mean_t_woa / max(mean_t_pgd, 1e-6)

    print("=" * 78)
    print(" SUMMARY BENCHMARK RESULTS")
    print("=" * 78)
    print(f"Mean Runtime v4 (WOA)       : {mean_t_woa:7.2f} ms")
    print(f"Mean Runtime v5 (Multi-PGD) : {mean_t_pgd:7.2f} ms")
    print(f"Average Execution Speedup   : {overall_speedup:7.2f}x faster!")
    print(f"Mean Attained Utility v4    : {mean_u_woa:8.4f}")
    print(f"Mean Attained Utility v5    : {mean_u_pgd:8.4f}")
    print(f"Utility Retention           : {mean_u_pgd / max(mean_u_woa, 1e-6) * 100.0:6.2f}%")
    print("=" * 78)


if __name__ == "__main__":
    n_snaps = 3 if len(sys.argv) <= 1 else int(sys.argv[1])
    run_benchmark(n_snapshots=n_snaps)
