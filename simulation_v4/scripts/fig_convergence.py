#!/usr/bin/env python3
"""Fig. 3 -- convergence of BWOA and inner power control (WOA / IWOA / PSO)."""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from _common import algorithm_params, base_parser, load_plotting, outputs
from stochastic_mec import (
    HybridSolverTF,
    SystemModelTF,
    SystemParams,
    make_tpc_tf,
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

    outer: dict[str, np.ndarray] = {}
    for tpc in ("WOA", "IWOA", "PSO"):
        rng = np.random.default_rng(args.seed)
        topo = sample_topology_with_cells(rng, params, args.ul_cells, args.dl_cells, args.ues)
        model = SystemModelTF(topo, params, rng)
        sol = HybridSolverTF(model, rng, algo=algo, tpc=tpc).solve()
        outer[f"{tpc}-BWOA"] = sol.curve
        print(f"{tpc:5s}-BWOA  U = {sol.utility:+.4f}  "
              f"inner solves = {sol.n_inner_calls}  time = {sol.runtime:.2f} s")

    plotting = load_plotting()
    if plotting is not None:
        _, fig_path = outputs(args, "convergence_bwoa")
        plotting.plot_curves(outer, fig_path, ylabel="System utility")
        print(f"saved {fig_path}")

    # Inner loop: MPC problem alone
    rng = np.random.default_rng(args.seed)
    topo = sample_topology_with_cells(rng, params, args.ul_cells, args.dl_cells, args.ues)
    model = SystemModelTF(topo, params, rng)
    assoc = np.zeros(model.assoc_shape, dtype=np.float32)
    for n in range(model.n_ul):
        assoc[n, n % model.K] = 1.0
    for m in range(model.m_dl):
        assoc[model.n_ul + m, (m + 1) % model.K] = 1.0
    
    assoc_tf = tf.constant(assoc, dtype=tf.float32)
    ul_sub, dl_sub = model.decode_tf(assoc_tf)
    xi = model.cochannel_at_sbs_tf(dl_sub, model.equal_split_dl_power_tf(dl_sub))
    lb = tf.fill([model.n_ul], float(params.p_min))
    ub = tf.fill([model.n_ul], float(params.p_max))
    rho0 = tf.cast(ul_sub >= 0, tf.float32)
    chi0 = tf.ones_like(rho0)

    inner: dict[str, np.ndarray] = {}
    for tpc in ("WOA", "IWOA", "PSO"):
        opt = make_tpc_tf(tpc, algo)
        def batched_fit(p):
            s_size = tf.shape(p)[0]
            ul_exp = tf.broadcast_to(ul_sub, [s_size, model.n_ul])
            xi_exp = tf.broadcast_to(xi, [s_size, model.K, model.m_ul])
            rho_exp = tf.broadcast_to(rho0, [s_size, model.n_ul])
            chi_exp = tf.broadcast_to(chi0, [s_size, model.n_ul])
            return model.mpc_objective_tf(ul_exp, p, xi_exp, rho_exp, chi_exp)

        res = opt.minimize(batched_fit, lb, ub)
        inner[tpc] = res.curve
        print(f"{tpc:5s} MPC  W* = {res.score:.6f}  ({res.n_eval} evaluations)")

    if plotting is not None:
        _, fig2 = outputs(args, "convergence_mpc")
        plotting.plot_curves(inner, fig2, ylabel=r"MPC objective $\mathbb{W}$")
        print(f"saved {fig2}")


if __name__ == "__main__":
    main()
