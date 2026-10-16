#!/usr/bin/env python3
"""Benchmark: Proposed PGD-BWOA Only (3 Subchannels, 2 UL UAVs, 1 DL UAV)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Pure relative path: find simulation_v4 by traversing upwards (no git needed)
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
from _common import base_parser, load_plotting, outputs
from stochastic_mec import (
    AlgorithmParams,
    HybridSolverTF,
    SweepResult,
    SystemModelTF,
    SystemParams,
    count_cases,
    evaluate_solution_tf,
    sample_topology_with_cells,
)

# ==============================================================================
# CONFIGURATION
# ==============================================================================
DENSITIES = [2, 3, 4, 5, 6]              
SUBCHANNELS = 3                    # Number of subcarriers (K = 3)
UL_CELLS = 2                       # Number of uplink UAV cells (M_ul = 2)
DL_CELLS = 1                       # Number of downlink UAV cells (M_dl = 1)
MAX_CASES = 200_000                # Maximum combinations limit (matches exhaustive skip threshold)
REALIZATIONS = 1
SEED = 2025                        # Random seed
MAX_ITER = 200                     # Maximum iterations for BWOA (default: 120; TPC defaults to 60)
N_AGENTS = 30                      # Number of agents for BWOA (default: 30; TPC defaults to 15)
MAX_ITER_PGD = 10                  # Maximum iterations for inner continuous PGD (default: 10)
LR_PGD = 0.005                     # Learning rate for PGD with Adam
CACHE_MAX_RETRIES = 0              # 0 = Pure Memoization (no perturbation flips)
# ==============================================================================


def _next_result_dir(parent_dir: Path, prefix: str = "result_") -> Path:
    """Find the next available result folder: result_1, result_2, result_3, ..."""
    idx = 1
    while (parent_dir / f"{prefix}{idx}").exists():
        idx += 1
    out_dir = parent_dir / f"{prefix}{idx}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def main() -> None:
    ap = base_parser(__doc__)
    ap.set_defaults(realizations=REALIZATIONS, seed=SEED)
    ap.add_argument("--densities", type=int, nargs="+", default=DENSITIES)
    ap.add_argument("--subchannels", type=int, default=SUBCHANNELS)
    ap.add_argument("--ul-cells", type=int, default=UL_CELLS)
    ap.add_argument("--dl-cells", type=int, default=DL_CELLS)
    ap.add_argument("--max-cases", type=int, default=MAX_CASES)
    args = ap.parse_args()

    max_iter_bwoa = globals().get("MAX_ITER_BWOA", globals().get("MAX_ITER", 200))
    max_iter_pgd = globals().get("MAX_ITER_PGD", 10)
    lr_pgd = globals().get("LR_PGD", 0.005)
    n_agents_bwoa = globals().get("N_AGENTS_BWOA", globals().get("N_AGENTS", 30))
    cache_retries = globals().get("CACHE_MAX_RETRIES", 0)

    algo = AlgorithmParams(
        max_iter_bwoa=max_iter_bwoa,
        max_iter_pgd=max_iter_pgd,
        lr_pgd=lr_pgd,
        n_agents_bwoa=n_agents_bwoa,
        cache_max_retries=cache_retries,
        tpc_default="PGD",
    )

    script_dir = Path(__file__).resolve().parent
    if args.out is None:
        args.out = _next_result_dir(script_dir)
    else:
        args.out.mkdir(parents=True, exist_ok=True)

    print("====================================================================")
    print(" PROPOSED PGD-BWOA (STANDALONE)")
    print("====================================================================")
    print(f"Densities (Active UEs): {args.densities}")
    print(f"Subchannels: {args.subchannels}, UL UAVs: {args.ul_cells}, DL UAVs: {args.dl_cells}")
    print(f"Realizations per point: {args.realizations}, Seed: {args.seed}")
    print(f"BWOA: {algo.n_agents_bwoa} agents, {algo.max_iter_bwoa} max iters (Cache mode: {algo.cache_max_retries})")
    print(f"TPC:  PGD ({algo.n_pgd_seeds} seeds, {algo.max_iter_pgd} steps, lr={algo.lr_pgd})")
    print(f"Results will be saved to: {args.out}")
    print("====================================================================\n")

    result = SweepResult(
        r"active-UE density [$\times 10^{-6}/m^2$]",
        [float(d) for d in args.densities],
        {},
    )
    records = []

    for i, density in enumerate(args.densities):
        params = SystemParams(n_subchannels=args.subchannels, lambda_ue_active=density * 1e-6)
        rows = []
        print(f"[Density = {density} Active UEs]")

        for r in range(args.realizations):
            seed = args.seed + 1000 * i + r
            rng = np.random.default_rng(seed)
            min_dl = 0 if density < (args.ul_cells + args.dl_cells) else 1
            topo = sample_topology_with_cells(rng, params, args.ul_cells, args.dl_cells, density, min_dl_per_cell=min_dl)
            model = SystemModelTF(topo, params, rng)
            n_cases = count_cases(model)

            topo_hash = abs(hash(topo.ue_pos.tobytes())) % 100000
            chan_hash = abs(hash(model.hnorm2_np.tobytes())) % 100000

            if n_cases > args.max_cases:
                print(f"  R{r + 1}/{args.realizations} (Seed {seed}): skipped ({n_cases:,} cases > limit)")
                continue

            sol_b = HybridSolverTF(model, np.random.default_rng(seed), algo=algo, tpc="PGD").solve()
            mb = evaluate_solution_tf(model, sol_b)
            rows.append(mb)

            rec = {
                "density": int(density),
                "realization": int(r + 1),
                "seed": int(seed),
                "topo_id": f"{topo_hash:05d}",
                "chan_id": f"{chan_hash:05d}",
                "utility": round(float(sol_b.utility), 4),
                "runtime_s": round(float(sol_b.runtime), 2),
                "offload_ratio": round(float(mb.offload_ratio), 4),
                "n_cases": int(n_cases),
            }
            records.append(rec)

            print(f"  R{r + 1}/{args.realizations} (Seed {seed} | Topo-ID: {topo_hash:05d} | Chan-ID: {chan_hash:05d}): "
                  f"PGD-BWOA U = {sol_b.utility:+.4f} ({sol_b.runtime:.2f} s) | {n_cases:,} cases", flush=True)

        result.add("PGD-BWOA", {
            "utility": float(np.mean([m.utility for m in rows])) if rows else float("nan"),
            "runtime": float(np.mean([m.runtime for m in rows])) if rows else float("nan"),
            "offload_ratio": float(np.mean([m.offload_ratio for m in rows])) if rows else float("nan"),
        })

    # Save primary sweep files
    res_path, _ = outputs(args, "compare_with_ex")
    result.save(res_path)
    print(f"\nSaved sweep result to: {res_path}")

    # Save detailed summary json
    summary_path = args.out / "summary.json"
    summary_data = {
        "job": "PGD-BWOA",
        "densities": args.densities,
        "subchannels": args.subchannels,
        "ul_cells": args.ul_cells,
        "dl_cells": args.dl_cells,
        "realizations": args.realizations,
        "seed": args.seed,
        "records": records,
    }
    summary_path.write_text(json.dumps(summary_data, indent=2))
    print(f"Saved detailed summary to: {summary_path}")

    # Plot curve if plotting is available
    plotting = load_plotting()
    if plotting is not None:
        plot_data = SweepResult(
            result.x_label,
            result.x,
            {k: v for k, v in result.data.items() if not k.startswith("_")},
        )
        _, f1 = outputs(args, "compare_with_ex_su")
        plotting.plot_sweep(plot_data, "utility", "System utility", f1)
        _, f2 = outputs(args, "compare_with_ex_time")
        plotting.plot_sweep(plot_data, "runtime", "Algorithm runtime [s]", f2)
        print(f"Saved plots to:\n  {f1}\n  {f2}")


if __name__ == "__main__":
    main()
