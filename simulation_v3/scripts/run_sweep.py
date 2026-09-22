#!/usr/bin/env python3
"""Parameter sweeps of the paper (Figs. 5-11).

Usage
-----
    python run_sweep.py ue-density        # offloading ratio + system utility
    python run_sweep.py sbs-density       # utility vs. the density of UL SBSs
    python run_sweep.py datasize          # utility vs. D_n
    python run_sweep.py taskload          # utility vs. C_n
    python run_sweep.py capacity          # utility vs. F_m^max
    python run_sweep.py power             # utility vs. p_n^max
    python run_sweep.py preference        # delay / energy vs. beta_t
    python run_sweep.py accuracy-tradeoff # ISCC: accuracy vs. delay over beta_a
    python run_sweep.py sensing-snr       # ISCC: chi*, rho*, admissibility vs. gamma^sen
"""

from __future__ import annotations

import numpy as np

from _common import algorithm_params, base_parser, load_plotting, outputs
from stochastic_mec import DEFAULT_CURVES, SystemParams, SweepResult, fixed_topology, run_sweep

AREA = SystemParams().area   # 1 km^2 -> a density of x*1e-6 /m^2 means x UEs


def _plot(result, name, args, panels):
    plotting = load_plotting()
    res_path, _ = outputs(args, name)
    result.save(res_path)
    print(f"saved {res_path}")
    if plotting is None:
        return
    for suffix, metric, ylabel in panels:
        _, fig = outputs(args, f"{name}_{suffix}")
        plotting.plot_sweep(result, metric, ylabel, fig)
        print(f"saved {fig}")


# --------------------------------------------------------------------- #
def sweep_ue_density(args):
    """Fig. 5 -- density of active UEs from 4 to 28 (x 1e-6 /m^2)."""
    xs = list(range(4, 29, 6))
    result = run_sweep(
        r"active-UE density [$\times 10^{-6}/m^2$]",
        xs,
        lambda x: SystemParams(lambda_ue_active=x * 1e-6),
        DEFAULT_CURVES,
        n_realizations=args.realizations,
        base_seed=args.seed,
        algo=algorithm_params(args),
        jobs=args.jobs,
        make_topology=lambda x: fixed_topology(3, 3, int(x)),
    )
    _plot(result, "ue_density", args,
          [("po", "offload_ratio", "Offloading percentage"),
           ("su", "utility", "System utility")])


def sweep_uav_density(args):
    xs = [1, 2, 3, 4, 5, 6]
    result = run_sweep(
        r"UL-UAV density [$\times 10^{-6}/m^2$]",
        xs,
        lambda x: SystemParams(lambda_sbs_ul=x * 1e-6),
        DEFAULT_CURVES,
        n_realizations=args.realizations,
        base_seed=args.seed,
        algo=algorithm_params(args),
        jobs=args.jobs,
        make_topology=lambda x: fixed_topology(int(x), 3, 10),
    )
    _plot(result, "uav_density", args, [("su", "utility", "System utility")])


def sweep_jammer_density(args):
    xs = [0, 1, 2, 3, 4]
    result = run_sweep(
        r"jammer density [$\times 10^{-6}/m^2$]",
        xs,
        lambda x: SystemParams(lambda_jammer=x * 1e-6),
        DEFAULT_CURVES,
        n_realizations=args.realizations,
        base_seed=args.seed,
        algo=algorithm_params(args),
        jobs=args.jobs,
        topology=fixed_topology(3, 3, 10),
    )
    _plot(result, "jammer_density", args, [("su", "utility", "System utility")])


def sweep_datasize(args):
    """Fig. 7 -- input data size D_n in [0.2, 1] MB."""
    xs = [0.2, 0.4, 0.6, 0.8, 1.0]
    result = run_sweep(
        r"input data size $D_n$ [MB]",
        xs,
        lambda x: SystemParams(data_size=x * 1e6),
        DEFAULT_CURVES,
        n_realizations=args.realizations,
        base_seed=args.seed,
        algo=algorithm_params(args),
        jobs=args.jobs,
        topology=fixed_topology(3, 3, 10),
    )
    _plot(result, "datasize", args, [("su", "utility", "System utility")])


def sweep_taskload(args):
    """Fig. 8 -- required computation C_n in [0.5, 2.5] Gcycles."""
    xs = [0.5, 1.0, 1.5, 2.0, 2.5]
    result = run_sweep(
        r"task load $C_n$ [Gcycles]",
        xs,
        lambda x: SystemParams(cpu_cycles=x * 1e9),
        DEFAULT_CURVES,
        n_realizations=args.realizations,
        base_seed=args.seed,
        algo=algorithm_params(args),
        jobs=args.jobs,
        topology=fixed_topology(3, 3, 10),
    )
    _plot(result, "taskload", args, [("su", "utility", "System utility")])


def sweep_capacity(args):
    """Fig. 9 -- MEC server capacity F_m^max in [8, 16] GHz."""
    xs = [8, 10, 12, 14, 16]
    result = run_sweep(
        r"server capacity $F_m^{\max}$ [GHz]",
        xs,
        lambda x: SystemParams(server_capacity=x * 1e9),
        DEFAULT_CURVES,
        n_realizations=args.realizations,
        base_seed=args.seed,
        algo=algorithm_params(args),
        jobs=args.jobs,
        topology=fixed_topology(3, 3, 10),
    )
    _plot(result, "capacity", args, [("su", "utility", "System utility")])


def sweep_power(args):
    """Fig. 10 -- UE transmit-power budget p_n^max in [17, 25] mW.

    (The manuscript quotes 0.25 W elsewhere; the sweep follows the range that
    is written in the "Influence of system parameters" paragraph.)
    """
    xs = [17, 19, 21, 23, 25]
    result = run_sweep(
        r"UE power budget $p_n^{\max}$ [mW]",
        xs,
        lambda x: SystemParams(p_max=x * 1e-3),
        DEFAULT_CURVES,
        n_realizations=args.realizations,
        base_seed=args.seed,
        algo=algorithm_params(args),
        jobs=args.jobs,
        topology=fixed_topology(3, 3, 10),
    )
    _plot(result, "power_budget", args, [("su", "utility", "System utility")])


def sweep_preference(args):
    """Fig. 11 -- time/energy preference at the default accuracy weight.

    ``beta_a`` keeps its default share, so ``beta_t`` spans ``(0, 1 - beta_a)``
    and ``beta_e = 1 - beta_t - beta_a``.
    """
    share = 1.0 - SystemParams().beta_acc
    xs = [round(share * f, 3) for f in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)]
    result = run_sweep(
        r"time preference $\beta_n^{\tt t}$",
        xs,
        lambda x: SystemParams(beta_time=x),
        [("MF-SIC (WOA-BWOA)", "WOA", "MF-SIC")],
        n_realizations=args.realizations,
        base_seed=args.seed,
        algo=algorithm_params(args),
        jobs=args.jobs,
        topology=fixed_topology(3, 3, 10),
    )
    _plot(result, "preference", args,
          [("time", "total_delay", "Total task execution time [s]"),
           ("energy", "total_energy", "Total energy consumption [J]"),
           ("su", "utility", "System utility")])


# --------------------------------------------------------------------- #
# ISCC sweeps (manuscript_v2, Sec. V)
# --------------------------------------------------------------------- #
ISCC_CURVE = ("WOA-BWOA", "WOA", "MF-SIC")


def _iscc_vs_atomic(args, x_label, xs, make_params, topology):
    """Run the ISCC solver and the atomic-task ablation on identical topologies."""
    merged = SweepResult(x_label, [float(v) for v in xs], {})
    for label, iscc in (("ISCC (partial + retention)", True), ("Atomic task ($\\rho=a$, $\\chi=1$)", False)):
        res = run_sweep(
            x_label, xs,
            lambda x, iscc=iscc: make_params(x).with_(enable_iscc=iscc),
            [ISCC_CURVE],
            n_realizations=args.realizations,
            base_seed=args.seed,
            algo=algorithm_params(args),
            jobs=args.jobs,
            topology=topology,
        )
        merged.data[label] = res.data[ISCC_CURVE[0]]
    return merged


def sweep_accuracy_tradeoff(args):
    """Accuracy--delay frontier traced by the accuracy weight beta_a (Prop. 3).

    ``beta_t = beta_e = (1 - beta_a) / 2`` so only the accuracy share moves.
    """
    xs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    result = _iscc_vs_atomic(
        args, r"accuracy weight $\beta_n^{\tt a}$", xs,
        lambda x: SystemParams(beta_acc=x, beta_time=(1.0 - x) / 2.0),
        fixed_topology(3, 3, 10),
    )
    _plot(result, "accuracy_weight", args,
          [("acc", "mean_accuracy", r"Mean inference accuracy $\Lambda_n$"),
           ("delay", "mean_norm_delay", r"Mean normalised delay $T_n/T_n^{\rm ref}$"),
           ("chi", "mean_retention", r"Mean retention $\chi_n^\star$"),
           ("su", "utility", "System utility")])
    plotting = load_plotting()
    if plotting is not None:
        _, fig = outputs(args, "accuracy_tradeoff")
        plotting.plot_frontier(result, "mean_norm_delay", "mean_accuracy",
                               r"Mean normalised delay $T_n/T_n^{\rm ref}$",
                               r"Mean inference accuracy $\Lambda_n$", fig,
                               annotate=r"$\beta^{\tt a}$")
        print(f"saved {fig}")


def sweep_sensing_snr(args):
    """Retention and split versus sensing quality (Props. 2-3, eq. chimin)."""
    xs = [2.0, 4.0, 6.0, 8.0, 10.0, 14.0, 18.0, 22.0]
    result = _iscc_vs_atomic(
        args, r"mean sensing SNR $\bar\gamma^{\tt sen}$ [dB]", xs,
        lambda x: SystemParams(sensing_snr_db=x),
        fixed_topology(3, 3, 10),
    )
    _plot(result, "sensing_snr", args,
          [("chi", "mean_retention", r"Mean retention $\chi_n^\star$"),
           ("rho", "mean_split", r"Mean offloaded fraction $\rho_n^\star$"),
           ("adm", "admissible_ratio", r"Admissible UEs ($\chi_n^{\min}\leq 1$)"),
           ("acc", "mean_accuracy", r"Mean inference accuracy $\Lambda_n$"),
           ("su", "utility", "System utility")])
    plotting = load_plotting()
    if plotting is not None:
        _, fig = outputs(args, "retention_vs_snr")
        iscc = {"x": result.x, **result.data["ISCC (partial + retention)"]}
        plotting.plot_dual(iscc, "mean_retention", "mean_split", result.x_label,
                           r"retention $\chi_n^\star$", r"offloaded fraction $\rho_n^\star$", fig,
                           floor=[min(1.0, SystemParams(sensing_snr_db=x).retention_floor) for x in xs])
        print(f"saved {fig}")


SWEEPS = {
    "ue-density": sweep_ue_density,
    "uav-density": sweep_uav_density,
    "sbs-density": sweep_uav_density,
    "jammer-density": sweep_jammer_density,
    "datasize": sweep_datasize,
    "taskload": sweep_taskload,
    "capacity": sweep_capacity,
    "power": sweep_power,
    "preference": sweep_preference,
    "accuracy-tradeoff": sweep_accuracy_tradeoff,
    "sensing-snr": sweep_sensing_snr,
}


def main() -> None:
    ap = base_parser(__doc__)
    ap.add_argument("sweep", choices=sorted(SWEEPS), help="which figure to reproduce")
    args = ap.parse_args()
    SWEEPS[args.sweep](args)


if __name__ == "__main__":
    main()
