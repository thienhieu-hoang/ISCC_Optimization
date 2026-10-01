#!/usr/bin/env python3
"""Accuracy-Delay Frontier sweep over beta_n^a (PGD-BWOA: ISCC vs. Atomic Task).

Settings:
- 2 UL UAV inference servers, 1 DL UAV SBS, 20 Active UEs.
- Fixed 3D topology snapshot & channel realization across ALL sweep points and both schemes.
- Multiprocessing: sweep points beta_n^a in XS are distributed independently across worker threads (jobs).
- Algorithm: PGD-BWOA (Multi-Start PGD with Adam on GPU for inner TPC, BWOA for association).
- Compares ISCC (partial offloading + optimal retention) vs Atomic Task ablation (rho=a, chi=1).
"""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Pure relative path: find simulation directory by traversing upwards
SIM_DIR = None
for p in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]:
    if (p / "stochastic_mec").exists():
        SIM_DIR = p
        break
if SIM_DIR is None:
    SIM_DIR = Path(__file__).resolve().parents[3]

SCRIPTS_DIR = SIM_DIR / "scripts"
if str(SIM_DIR) not in sys.path:
    sys.path.insert(0, str(SIM_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import numpy as np
from _common import base_parser, load_plotting
from stochastic_mec import (
    AlgorithmParams,
    HybridSolverTF,
    SweepResult,
    SystemModelTF,
    SystemParams,
    evaluate_solution_tf,
    sample_topology_with_cells,
)

# ==============================================================================
# CONFIGURATION
# ==============================================================================
XS = [0.0, 0.02, 0.05, 0.10, 0.17, 0.25, 0.35, 0.50, 0.80]  # Accuracy weight beta_n^a
N_UL = 2                           # Number of UL UAV servers (default: 2)
N_DL = 1                           # Number of DL UAV SBSs (default: 1)
N_UE = 10                          # Number of Active UEs (default: 20)
REALIZATIONS = 1                   # Single shared realization
SEED = 2025                        # Random seed
MAX_ITER = 200                     # Maximum iterations for BWOA
N_AGENTS = 30                      # Number of agents for BWOA
MAX_ITER_PGD = 10                  # Maximum iterations for inner continuous PGD (default: 10)
LR_PGD = 0.005                     # Learning rate for PGD with Adam
CACHE_MAX_RETRIES = 0              # 0 = Pure Memoization (no perturbation flips)
JOBS = 4                           # Number of parallel worker threads / processes
# ==============================================================================

ISCC_LABEL = "ISCC (partial + retention)"
ATOMIC_LABEL = "Atomic task ($\\rho=a$, $\\chi=1$)"


def _next_result_dir(parent_dir: Path, prefix: str = "result_") -> Path:
    """Find the next available result folder: result_1, result_2, result_3, ..."""
    idx = 1
    while (parent_dir / f"{prefix}{idx}").exists():
        idx += 1
    out_dir = parent_dir / f"{prefix}{idx}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _worker_simulate_accuracy_point(
    task: tuple,
) -> tuple[float, dict[str, float], dict[str, float]]:
    """Worker process evaluating one beta_n^a point for both ISCC and Atomic schemes on the shared topology."""
    x, base_params, algo, topo, seed = task

    import tensorflow as tf

    tf.random.set_seed(seed + 10007)

    # 1. Run ISCC (partial offloading + adaptive retention)
    channel_rng_iscc = np.random.default_rng(seed)
    params_iscc = base_params.with_(
        beta_acc=x,
        beta_time=(1.0 - x) / 2.0,
        enable_iscc=True,
    )
    model_iscc = SystemModelTF(topo, params_iscc, channel_rng_iscc, scheme="MF-SIC")
    solver_rng_iscc = np.random.default_rng(seed + 10007)
    solver_iscc = HybridSolverTF(model_iscc, solver_rng_iscc, algo=algo, tpc="PGD", scheme="MF-SIC")
    sol_iscc = solver_iscc.solve()
    met_iscc = evaluate_solution_tf(model_iscc, sol_iscc)

    # 2. Run Atomic Task ablation (rho = a, chi = 1) on the EXACT same channels and topology
    channel_rng_atom = np.random.default_rng(seed)
    params_atom = base_params.with_(
        beta_acc=x,
        beta_time=(1.0 - x) / 2.0,
        enable_iscc=False,
    )
    model_atom = SystemModelTF(topo, params_atom, channel_rng_atom, scheme="MF-SIC")
    solver_rng_atom = np.random.default_rng(seed + 10007)
    solver_atom = HybridSolverTF(model_atom, solver_rng_atom, algo=algo, tpc="PGD", scheme="MF-SIC")
    sol_atom = solver_atom.solve()
    met_atom = evaluate_solution_tf(model_atom, sol_atom)

    return x, met_iscc.as_dict(), met_atom.as_dict()


def main() -> None:
    ap = base_parser(__doc__)
    jobs = globals().get("JOBS", 4)
    ap.set_defaults(realizations=REALIZATIONS, seed=SEED, jobs=jobs)
    ap.add_argument(
        "--xs",
        type=float,
        nargs="+",
        default=XS,
        help="Accuracy weight beta_n^a sweep points",
    )
    ap.add_argument(
        "--n-ul",
        type=int,
        default=globals().get("N_UL", 2) or 2,
        help="Number of UL UAVs (default: 2)",
    )
    ap.add_argument(
        "--n-dl",
        type=int,
        default=globals().get("N_DL", 1) or 1,
        help="Number of DL UAVs (default: 1)",
    )
    ap.add_argument(
        "--n-ue",
        type=int,
        default=globals().get("N_UE", 20) or 20,
        help="Number of Active UEs (default: 20)",
    )
    args = ap.parse_args()

    # Algorithm configuration
    max_iter_bwoa = globals().get("MAX_ITER_BWOA", globals().get("MAX_ITER", 200))
    max_iter_pgd = globals().get("MAX_ITER_PGD", 10)
    lr_pgd = globals().get("LR_PGD", 0.005)
    n_agents_bwoa = globals().get("N_AGENTS_BWOA", globals().get("N_AGENTS", 30))
    cache_retries = globals().get("CACHE_MAX_RETRIES", 0)

    algo = AlgorithmParams(
        max_iter_bwoa=20 if args.quick else max_iter_bwoa,
        n_agents_bwoa=10 if args.quick else n_agents_bwoa,
        max_iter_pgd=5 if args.quick else max_iter_pgd,
        lr_pgd=lr_pgd,
        cache_max_retries=cache_retries,
        tpc_default="PGD",
    )

    # Base system parameters
    base_params = SystemParams(
        lambda_sbs_ul=args.n_ul * 1e-6,
        lambda_sbs_dl=args.n_dl * 1e-6,
        lambda_ue_active=args.n_ue * 1e-6,
    )

    # Output directory
    script_dir = Path(__file__).resolve().parent
    if args.out is None:
        args.out = _next_result_dir(script_dir)
    else:
        args.out.mkdir(parents=True, exist_ok=True)

    # 1. SAMPLE SHARED TOPOLOGY ONCE (Fixed across ALL beta_a sweep points and both schemes)
    rng_topo = np.random.default_rng(args.seed)
    min_dl = 0 if args.n_ue < (args.n_ul + args.n_dl) else 1
    shared_topo = sample_topology_with_cells(
        rng_topo,
        base_params,
        args.n_ul,
        args.n_dl,
        args.n_ue,
        min_dl_per_cell=min_dl,
    )
    topo_hash = abs(hash(shared_topo.ue_pos.tobytes())) % 100000

    print("====================================================================")
    print(" ACCURACY-DELAY FRONTIER SWEEP (PGD-BWOA: ISCC vs. ATOMIC)")
    print("====================================================================")
    print(f"Fixed Topology: Topo-ID = {topo_hash:05d} (shared across all points & schemes)")
    print(f"Network: {args.n_ul} UL UAVs, {args.n_dl} DL UAVs, {args.n_ue} Active UEs")
    print(f"Accuracy weight points beta_a: {args.xs}")
    print(f"Algorithm: PGD-BWOA (BWOA agents={algo.n_agents_bwoa}, iter={algo.max_iter_bwoa} | PGD iter={algo.max_iter_pgd})")
    print(f"Cache mode: Pure Memoization (cache_max_retries={algo.cache_max_retries})")
    print(f"Parallel worker threads (jobs): {args.jobs}")
    print(f"Results will be saved to: {args.out}")
    print("====================================================================\n", flush=True)

    # 2. PREPARE INDEPENDENT TASKS PER SWEEP POINT
    xs_sorted = sorted([float(x) for x in args.xs])
    tasks = [
        (x, base_params, algo, shared_topo, args.seed)
        for x in xs_sorted
    ]

    results_by_x: dict[float, tuple[dict[str, float], dict[str, float]]] = {}
    start_total = time.perf_counter()

    # 3. DISPATCH ACROSS PARALLEL WORKER THREADS (JOBS)
    if args.jobs > 1 and len(tasks) > 1:
        workers = min(int(args.jobs), len(tasks))
        print(f"Dispatching {len(tasks)} beta_a points across {workers} parallel worker processes...", flush=True)
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_x = {
                executor.submit(_worker_simulate_accuracy_point, task): task[0]
                for task in tasks
            }
            for fut in as_completed(future_to_x):
                x_val, met_iscc, met_atom = fut.result()
                results_by_x[x_val] = (met_iscc, met_atom)
                print(
                    f"  [done] beta_a = {x_val:0.1f} | "
                    f"ISCC: U={met_iscc['utility']:+6.4f}, Acc={met_iscc['mean_accuracy']:.4f}, Delay={met_iscc['mean_norm_delay']:.4f}, Chi={met_iscc['mean_retention']:.4f} | "
                    f"Atomic: U={met_atom['utility']:+6.4f}, Acc={met_atom['mean_accuracy']:.4f}, Delay={met_atom['mean_norm_delay']:.4f}",
                    flush=True,
                )
    else:
        print(f"Running {len(tasks)} beta_a points sequentially...", flush=True)
        for task in tasks:
            x_val, met_iscc, met_atom = _worker_simulate_accuracy_point(task)
            results_by_x[x_val] = (met_iscc, met_atom)
            print(
                f"  [done] beta_a = {x_val:0.1f} | "
                f"ISCC: U={met_iscc['utility']:+6.4f}, Acc={met_iscc['mean_accuracy']:.4f}, Delay={met_iscc['mean_norm_delay']:.4f}, Chi={met_iscc['mean_retention']:.4f} | "
                f"Atomic: U={met_atom['utility']:+6.4f}, Acc={met_atom['mean_accuracy']:.4f}, Delay={met_atom['mean_norm_delay']:.4f}",
                flush=True,
            )

    elapsed_total = time.perf_counter() - start_total
    print(f"\nAll {len(tasks)} points completed in {elapsed_total:.2f} s.\n", flush=True)

    # 4. ASSEMBLE SWEEPRESULT OBJECT
    result = SweepResult(
        r"accuracy weight $\beta_n^{\tt a}$",
        xs_sorted,
        {},
    )

    summary_records = {}
    for x in xs_sorted:
        met_iscc, met_atom = results_by_x[x]
        result.add(ISCC_LABEL, met_iscc)
        result.add(ATOMIC_LABEL, met_atom)

        summary_records[f"{x:.1f}"] = {
            "beta_a": float(x),
            "ISCC": {
                "utility": round(met_iscc["utility"], 4),
                "accuracy": round(met_iscc["mean_accuracy"], 4),
                "norm_delay": round(met_iscc["mean_norm_delay"], 4),
                "retention": round(met_iscc["mean_retention"], 4),
                "offload_ratio": round(met_iscc["offload_ratio"], 4),
                "runtime_s": round(met_iscc["runtime"], 2),
            },
            "Atomic": {
                "utility": round(met_atom["utility"], 4),
                "accuracy": round(met_atom["mean_accuracy"], 4),
                "norm_delay": round(met_atom["mean_norm_delay"], 4),
                "retention": round(met_atom["mean_retention"], 4),
                "offload_ratio": round(met_atom["offload_ratio"], 4),
                "runtime_s": round(met_atom["runtime"], 2),
            },
        }

    # 5. SAVE STRUCTURED RESULTS (.json and .mat)
    res_path = args.out / "accuracy_weight.json"
    result.save(res_path)
    print(f"Saved sweep results to: {res_path} (and .mat)")

    summary_file = args.out / "summary.json"
    summary_payload = {
        "job": "accuracy_tradeoff",
        "algorithm": "PGD-BWOA",
        "n_ul": args.n_ul,
        "n_dl": args.n_dl,
        "n_ue": args.n_ue,
        "seed": args.seed,
        "topo_id": f"{topo_hash:05d}",
        "elapsed_total_s": round(elapsed_total, 2),
        "points": summary_records,
    }
    summary_file.write_text(json.dumps(summary_payload, indent=2))
    print(f"Saved detailed summary to: {summary_file}")

    # 6. RENDER FIGURES
    plotting = load_plotting()
    if plotting is not None:
        fig_frontier_png = args.out / "accuracy_tradeoff.png"
        fig_frontier_pdf = args.out / "accuracy_tradeoff.pdf"
        plotting.plot_frontier(
            result,
            "mean_norm_delay",
            "mean_accuracy",
            r"Mean normalised delay $T_n/T_n^{\rm ref}$",
            r"Mean inference accuracy $\Lambda_n$",
            fig_frontier_png,
            annotate=r"$\beta^{\tt a}$",
        )
        plotting.plot_frontier(
            result,
            "mean_norm_delay",
            "mean_accuracy",
            r"Mean normalised delay $T_n/T_n^{\rm ref}$",
            r"Mean inference accuracy $\Lambda_n$",
            fig_frontier_pdf,
            annotate=r"$\beta^{\tt a}$",
        )
        print(f"Saved accuracy-delay Pareto frontier plots to:\n  {fig_frontier_png}\n  {fig_frontier_pdf}")

        # Additional metric plots
        for metric, ylabel, fname in [
            ("utility", "System Utility", "accuracy_weight_su.pdf"),
            ("mean_accuracy", r"Mean inference accuracy $\Lambda_n$", "accuracy_weight_acc.pdf"),
            ("mean_norm_delay", r"Mean normalised delay $T_n/T_n^{\rm ref}$", "accuracy_weight_delay.pdf"),
            ("mean_retention", r"Mean retention $\chi_n^\star$", "accuracy_weight_chi.pdf"),
        ]:
            out_f = args.out / fname
            plotting.plot_sweep(result, metric, ylabel, out_f)
            print(f"Saved plot: {out_f}")


if __name__ == "__main__":
    main()
