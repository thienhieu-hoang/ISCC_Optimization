#!/usr/bin/env python3
"""Half-column panels for the manuscript's side-by-side figures (Figs. 2 and 4).

A 6.0 x 3.6 in canvas placed at 0.49 of a column is shrunk ~3.5x, so its 9-10 pt
text prints at 2-3 pt. These panels are drawn on a 2.5 x 1.5 in canvas (still
5:3) with 7 pt text, which prints at ~5 pt -- the practical floor for IEEE
two-panel subfigures. Data come from the cached results/*.json; nothing is
re-simulated. Colours are the CVD-validated subset used in Figs. 1 and 4.
"""

from __future__ import annotations

import json

import numpy as np

from _common import FIGURES, base_parser
from stochastic_mec import SystemParams

RESULTS = FIGURES.parent / "results"
ISCC, ATOMIC = "ISCC (partial + retention)", "Atomic task ($\\rho=a$, $\\chi=1$)"
BLUE, VERM, PURP = "#0072B2", "#D55E00", "#AA4499"
SIZE = (2.5, 1.5)          # 5:3


def main() -> None:
    args = base_parser(__doc__).parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5, "legend.fontsize": 6, "lines.linewidth": 1.2,
        "lines.markersize": 3.5, "axes.linewidth": 0.6, "grid.linewidth": 0.4,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    })
    out = args.out or FIGURES
    params = SystemParams()

    def save(fig, name):
        fig.tight_layout(pad=0.25)
        fig.savefig(out / name, dpi=400)
        plt.close(fig)
        print("saved", out / name)

    # ---- Fig. 2: BWOA vs exhaustive search ---------------------------------- #
    ex = json.loads((RESULTS / "compare_with_ex.json").read_text())
    x = ex["x"]
    for metric, ylabel, name in [("utility", "system utility", "compare_with_ex_su.png"),
                                 ("runtime", "runtime [s]", "compare_with_ex_time.png")]:
        fig, ax = plt.subplots(figsize=SIZE)
        ax.plot(x, ex["data"]["BWOA"][metric], color=BLUE, marker="o", label="BWOA")
        ax.plot(x, ex["data"]["EX"][metric], color=VERM, marker="v", label="exhaustive")
        ax.set_xlabel("active-UE density [$\\times10^{-6}$/m$^2$]")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.35)
        ax.legend(loc="upper left", handlelength=1.6, borderpad=0.3)
        save(fig, name)

    # ---- Fig. 4(a): accuracy--delay frontier -------------------------------- #
    acc = json.loads((RESULTS / "accuracy_weight.json").read_text())
    fig, ax = plt.subplots(figsize=SIZE)
    s = acc["data"][ISCC]
    ax.plot(s["mean_norm_delay"], s["mean_accuracy"], color=BLUE, marker="o", label="ISCC")
    a = acc["data"][ATOMIC]
    ax.plot(a["mean_norm_delay"], a["mean_accuracy"], color=VERM, marker="s",
            ls="none", label="atomic task")
    ax.axhline(params.accuracy_threshold, color="0.45", ls="--", lw=0.9,
               label="$\\Lambda^{\\tt th}$")
    ax.annotate("$\\beta^{\\tt a}{=}0$", (s["mean_norm_delay"][0], s["mean_accuracy"][0]),
                textcoords="offset points", xytext=(4, 3), fontsize=6)
    ax.set_xlabel("normalized delay")
    ax.set_ylabel("accuracy $\\Lambda_n$")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="lower right", handlelength=1.6, borderpad=0.3)
    save(fig, "accuracy_tradeoff.png")

    # ---- Fig. 4(b): retention and offloading vs sensing SNR ----------------- #
    snr = json.loads((RESULTS / "sensing_snr.json").read_text())
    x = np.asarray(snr["x"], dtype=float)
    s = snr["data"][ISCC]
    floor = np.minimum([SystemParams(sensing_snr_db=v).retention_floor for v in x], 1.0)
    g_star = 10 * np.log10(2 ** (-np.log(1 - params.accuracy_threshold)
                                 / params.accuracy_sensitivity) - 1)
    fig, ax = plt.subplots(figsize=SIZE)
    ax.axvspan(x.min() - 1, g_star, color="0.92", zorder=0)
    ax.text(g_star - 0.4, 0.04, "inadmissible", ha="right", va="bottom",
            fontsize=5.5, color="0.35")
    ax.plot(x, s["mean_retention"], color=BLUE, marker="o", label="$\\chi_n^\\star$")
    ax.plot(x, floor, color=BLUE, ls="--", lw=0.9, label="$\\chi_n^{\\min}$")
    ax.plot(x, s["offload_ratio"], color=PURP, marker="D", ms=3, label="offloading")
    ax.set_xlim(x.min() - 1, x.max() + 1)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("mean sensing SNR [dB]")
    ax.set_ylabel("fraction")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="center right", handlelength=1.6, borderpad=0.3)
    save(fig, "retention_vs_snr.png")


if __name__ == "__main__":
    main()
