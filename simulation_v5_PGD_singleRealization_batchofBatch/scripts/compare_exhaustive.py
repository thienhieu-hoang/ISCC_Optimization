#!/usr/bin/env python3
"""Fig. 2 -- proposed BWOA vs. exhaustive search (utility and runtime).

Both solvers share the same inner problems (SMCA in closed form, MPC by WOA,
DL-TPC by WOA); they differ only in how the association matrix A_net is
explored. Because exhaustive search visits (K+1)^{N_ul} * K^{M_dl}
associations, the comparison is restricted -- as in the paper -- to a small
system with K = 3 sub-channels and a UE density in [2, 7] x 1e-6 /m^2.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from _common import algorithm_params, base_parser, load_plotting, outputs
from stochastic_mec import (
    HybridSolverTF,
    SweepResult,
    SystemModelTF,
    SystemParams,
    count_cases,
    evaluate_solution_tf,
    exhaustive_search,
    sample_topology_with_cells,
)


def main() -> None:
    ap = base_parser(__doc__)
    ap.add_argument("--densities", type=int, nargs="+", default=[2, 3, 4, 5, 6, 7])
    ap.add_argument("--subchannels", type=int, default=3)
    ap.add_argument("--ul-cells", type=int, default=2)
    ap.add_argument("--dl-cells", type=int, default=1)
    ap.add_argument("--max-cases", type=int, default=200_000)
    args = ap.parse_args()
    if args.realizations == 20:  # Cheaper default for this heavy combinatorial figure
        args.realizations = 5
    if args.quick:
        # Keep smoke run short
        args.densities = [d for d in args.densities if d <= 4]
        args.max_cases = min(args.max_cases, 5_000)
        print("[--quick] reduced BWOA budget: the optimality gap below is NOT "
              "representative; drop --quick for paper figures.\n")

    algo = algorithm_params(args)
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
    if plotting is None:
        return
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
