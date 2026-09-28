#!/usr/bin/env python3
"""Benchmark: Proposed BWOA vs. Exhaustive Search."""

from __future__ import annotations

import sys
from pathlib import Path

# Pure relative path: find simulation_v4 by traversing upwards (no git needed)
SIM_DIR = None
for p in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]:
    if p.name == "simulation_v4":
        SIM_DIR = p
        break
if SIM_DIR is None:
    SIM_DIR = Path(__file__).resolve().parents[2]

SCRIPTS_DIR = SIM_DIR / "scripts"
if str(SIM_DIR) not in sys.path:
    sys.path.insert(0, str(SIM_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from _common import base_parser  # noqa: E402
from compare_exhaustive import main as compare_exhaustive_main  # noqa: E402
from stochastic_mec import AlgorithmParams  # noqa: E402

# ==============================================================================
# CONFIGURATION
# ==============================================================================
DENSITIES = [2, 3, 4, 5, 6, 7]      # Active-UE density: [2, 3, 4, 5, 6, 7]
SUBCHANNELS = 3                    # Number of subcarriers (K = 3)
UL_CELLS = 2                       # Number of uplink UAV cells (M_ul = 2)
DL_CELLS = 1                       # Number of downlink UAV cells (M_dl = 1)
MAX_CASES = 200_000                # Maximum combinations limit
REALIZATIONS = 5                   # Number of Monte-Carlo realizations
SEED = 2025                        # Random seed
MAX_ITER = 120                     # Maximum iterations for BWOA and TPC (default: 120)
N_AGENTS = 30                      # Number of agents for BWOA and TPC (default: 30)
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
    # Run compare_exhaustive with configured defaults
    import argparse
    ap = base_parser(__doc__)
    ap.set_defaults(realizations=REALIZATIONS, seed=SEED)
    ap.add_argument("--densities", type=int, nargs="+", default=DENSITIES)
    ap.add_argument("--subchannels", type=int, default=SUBCHANNELS)
    ap.add_argument("--ul-cells", type=int, default=UL_CELLS)
    ap.add_argument("--dl-cells", type=int, default=DL_CELLS)
    ap.add_argument("--max-cases", type=int, default=MAX_CASES)
    args = ap.parse_args()

    max_iter = globals().get("MAX_ITER", None)
    n_agents = globals().get("N_AGENTS", None)
    algo_kwargs = {}
    if max_iter is not None:
        algo_kwargs.update(max_iter_bwoa=max_iter, max_iter_tpc=max_iter)
    if n_agents is not None:
        algo_kwargs.update(n_agents_bwoa=n_agents, n_agents_tpc=n_agents)

    if not args.quick and algo_kwargs:
        args.algo = AlgorithmParams(**algo_kwargs)

    script_dir = Path(__file__).resolve().parent
    if args.out is None:
        args.out = _next_result_dir(script_dir)
    else:
        args.out.mkdir(parents=True, exist_ok=True)

    print(f"Densities: {args.densities}")
    print(f"Subchannels: {args.subchannels}, UL UAVs: {args.ul_cells}, DL UAVs: {args.dl_cells}")
    print(f"Max cases: {args.max_cases:,}")
    print(f"Realizations per point: {args.realizations}")
    print(f"Results will be saved to: {args.out}")

    # Invoke the exhaustive comparison solver
    import compare_exhaustive
    # Temporarily set sys.argv or call core logic
    from _common import outputs, load_plotting
    from stochastic_mec import (
        HybridSolverTF,
        SweepResult,
        SystemModelTF,
        SystemParams,
        count_cases,
        evaluate_solution_tf,
        exhaustive_search,
        sample_topology_with_cells,
        algorithm_params,
    )
    import numpy as np

    algo = algorithm_params(args)
    if args.quick:
        args.densities = [d for d in args.densities if d <= 4]
        args.max_cases = min(args.max_cases, 5_000)

    result = SweepResult(
        r"active-UE density [$\times 10^{-6}/m^2$]",
        [float(d) for d in args.densities],
        {},
    )

    for i, density in enumerate(args.densities):
        params = SystemParams(n_subchannels=args.subchannels, lambda_ue_active=density * 1e-6)
        rows = {"BWOA": [], "EX": []}
        gaps = []
        print(f"[density = {density}]")
        for r in range(args.realizations):
            seed = args.seed + 1000 * i + r
            rng = np.random.default_rng(seed)
            topo = sample_topology_with_cells(rng, params, args.ul_cells, args.dl_cells, density)
            model = SystemModelTF(topo, params, rng)
            n_cases = count_cases(model)
            if n_cases > args.max_cases:
                print(f"  realisation {r + 1}: skipped ({n_cases:,} cases > limit)")
                continue

            sol_b = HybridSolverTF(model, np.random.default_rng(seed), algo=algo, tpc="WOA").solve()
            sol_e = exhaustive_search(model, np.random.default_rng(seed), tpc="WOA", algo=algo, max_cases=args.max_cases)
            mb = evaluate_solution_tf(model, sol_b)
            me = evaluate_solution_tf(model, sol_e)
            rows["BWOA"].append(mb)
            rows["EX"].append(me)

            denom = abs(sol_e.utility) if abs(sol_e.utility) > 1e-9 else 1.0
            gap_pct = 100.0 * (sol_e.utility - sol_b.utility) / denom
            gaps.append(gap_pct)
            print(f"  realisation {r + 1}: BWOA {sol_b.utility:+.4f} ({sol_b.runtime:.2f} s)  |  "
                  f"EX {sol_e.utility:+.4f} ({sol_e.runtime:.2f} s)  |  "
                  f"{n_cases:,} cases  |  gap {gap_pct:+.2f} %")

        for label in ("BWOA", "EX"):
            met = rows[label]
            result.add(label, {
                "utility": float(np.mean([m.utility for m in met])) if met else float("nan"),
                "runtime": float(np.mean([m.runtime for m in met])) if met else float("nan"),
                "offload_ratio": float(np.mean([m.offload_ratio for m in met])) if met else float("nan"),
            })
        result.data.setdefault("_gap_percent", {}).setdefault("gap", []).append(
            float(np.mean(gaps)) if gaps else float("nan")
        )

    res_path, _ = outputs(args, "compare_with_ex")
    result.save(res_path)
    print(f"saved {res_path}")
    print("mean optimality gap [%]:",
          [round(g, 2) for g in result.data["_gap_percent"]["gap"]])

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
        print(f"saved {f1}\nsaved {f2}")


if __name__ == "__main__":
    main()
