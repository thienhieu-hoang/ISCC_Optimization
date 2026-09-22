#!/usr/bin/env python3
"""Fig. 3 -- convergence of the proposed BWOA solution, and of the three
power-control engines (WOA / IWOA / PSO) on the same MPC instance."""

from __future__ import annotations

import numpy as np

from _common import algorithm_params, base_parser, load_plotting, outputs
from stochastic_mec import (
    HybridSolver,
    SystemModel,
    SystemParams,
    make_tpc,
    sample_topology_with_cells,
)


def main() -> None:
    ap = base_parser(__doc__)
    ap.add_argument("--ues", type=int, default=6)
    ap.add_argument("--ul-cells", type=int, default=3)
    ap.add_argument("--dl-cells", type=int, default=3)
    args = ap.parse_args()

    params = SystemParams()
    algo = algorithm_params(args)

    # ---- outer loop: BWOA over A_net, one curve per hybrid ---------------
    outer: dict[str, np.ndarray] = {}
    for tpc in ("WOA", "IWOA", "PSO"):
        rng = np.random.default_rng(args.seed)
        topo = sample_topology_with_cells(rng, params, args.ul_cells, args.dl_cells, args.ues)
        model = SystemModel(topo, params, rng)
        sol = HybridSolver(model, rng, algo=algo, tpc=tpc).solve()
        outer[f"{tpc}-BWOA"] = sol.curve
        print(f"{tpc:5s}-BWOA  U = {sol.utility:+.4f}  "
              f"inner solves = {sol.n_inner_calls}  time = {sol.runtime:.2f} s")

    plotting = load_plotting()
    if plotting is not None:
        _, fig_path = outputs(args, "convergence_bwoa")
        plotting.plot_curves(outer, fig_path, ylabel="System utility")
        print(f"saved {fig_path}")

    # ---- inner loop: the MPC problem alone -------------------------------
    rng = np.random.default_rng(args.seed)
    topo = sample_topology_with_cells(rng, params, args.ul_cells, args.dl_cells, args.ues)
    model = SystemModel(topo, params, rng)
    assoc = np.zeros(model.assoc_shape, dtype=np.int8)
    for n in range(model.n_ul):
        assoc[n, n % model.K] = 1
    for m in range(model.m_dl):
        assoc[model.n_ul + m, (m + 1) % model.K] = 1
    ul_sub, dl_sub = model.decode(assoc)
    xi = model.cochannel_at_sbs(dl_sub, model.equal_split_dl_power(dl_sub))
    lb = np.full(model.n_ul, params.p_min)
    ub = np.full(model.n_ul, params.p_max)

    inner: dict[str, np.ndarray] = {}
    for tpc in ("WOA", "IWOA", "PSO"):
        opt = make_tpc(tpc, algo, np.random.default_rng(args.seed + 7))
        res = opt.minimize(lambda p: model.mpc_objective(ul_sub, p, xi), lb, ub)
        inner[tpc] = res.curve
        print(f"{tpc:5s} MPC  W* = {res.score:.6f}  ({res.n_eval} evaluations)")

    if plotting is not None:
        _, fig2 = outputs(args, "convergence_mpc")
        plotting.plot_curves(inner, fig2, ylabel=r"MPC objective $\mathbb{W}$")
        print(f"saved {fig2}")


if __name__ == "__main__":
    main()
