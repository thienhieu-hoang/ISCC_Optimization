#!/usr/bin/env python3
"""Fig. 5 -- ISCC-specific effects, re-plotted from cached sweep results."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from _common import FIGURES, base_parser
from stochastic_mec import SystemParams

RESULTS = FIGURES.parent / "results"
ISCC, ATOMIC = "ISCC (partial + retention)", "Atomic task ($\\rho=a$, $\\chi=1$)"
C_ISCC, C_ATOM, C_OFFL = "#0072B2", "#D55E00", "#AA4499"


def main() -> None:
    ap = base_parser(__doc__)
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = args.out or FIGURES
    params = SystemParams()

    acc_path = RESULTS / "accuracy_weight.json"
    if acc_path.exists():
        acc = json.loads(acc_path.read_text())
        beta = acc["x"]
        fig, ax = plt.subplots(figsize=(6.0, 3.6))
        s = acc["data"][ISCC]
        ax.plot(s["mean_norm_delay"], s["mean_accuracy"], color=C_ISCC, marker="o",
                lw=1.8, ms=6, label="ISCC (partial offloading + retention)")
        s = acc["data"][ATOMIC]
        ax.plot(s["mean_norm_delay"], s["mean_accuracy"], color=C_ATOM, marker="s",
                ls="none", ms=6, label="atomic task ($\\rho_n{=}a_n$, $\\chi_n{=}1$)")
        s = acc["data"][ISCC]
        for j, dx, dy in [(0, 6, 4), (len(beta) - 1, 6, -12)]:
            ax.annotate(f"$\\beta^{{\\tt a}}={beta[j]:g}$",
                        (s["mean_norm_delay"][j], s["mean_accuracy"][j]),
                        textcoords="offset points", xytext=(dx, dy), fontsize=9)
        ax.axhline(params.accuracy_threshold, color="0.45", ls="--", lw=1.1,
                   label="accuracy floor $\\Lambda^{\\tt th}$")
        ax.set_xlabel("mean normalized delay $\\mathsf{T}_n/\\mathsf{T}_n^{\\tt ref}$")
        ax.set_ylabel("mean inference accuracy $\\Lambda_n$")
        ax.grid(True, alpha=0.35)
        ax.legend(loc="lower right", fontsize=9)
        fig.tight_layout()
        fig.savefig(out / "accuracy_tradeoff.png", dpi=200)
        plt.close(fig)
        print("saved", out / "accuracy_tradeoff.png")

    snr_path = RESULTS / "sensing_snr.json"
    if snr_path.exists():
        snr = json.loads(snr_path.read_text())
        x = np.asarray(snr["x"], dtype=float)
        s = snr["data"][ISCC]
        floor = np.array([SystemParams(sensing_snr_db=v).retention_floor for v in x])
        g_star = 10 * np.log10(2 ** (-np.log(1 - params.accuracy_threshold)
                                     / params.accuracy_sensitivity) - 1)

        fig, ax = plt.subplots(figsize=(6.0, 3.6))
        ax.axvspan(x.min() - 1, g_star, color="0.92", zorder=0)
        ax.text(g_star - 0.3, 0.06, "$\\chi_n^{\\min}>1$:\ninadmissible", ha="right",
                va="bottom", fontsize=8.5, color="0.35")
        ax.plot(x, s["mean_retention"], color=C_ISCC, marker="o", lw=1.8, ms=6,
                label="optimal retention $\\chi_n^\\star$")
        ax.plot(x, np.minimum(floor, 1.0), color=C_ISCC, ls="--", lw=1.2,
                label="retention floor $\\chi_n^{\\min}$")
        ax.plot(x, s["offload_ratio"], color=C_OFFL, marker="D", lw=1.8, ms=5.5,
                label="fraction of UEs offloading")
        ax.set_xlim(x.min() - 1, x.max() + 1)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("mean sensing SNR $\\bar\\gamma^{\\tt sen}$ [dB]")
        ax.set_ylabel("fraction")
        ax.grid(True, alpha=0.35)
        ax.legend(loc="center right", fontsize=9)
        fig.tight_layout()
        fig.savefig(out / "retention_vs_snr.png", dpi=200)
        plt.close(fig)
        print("saved", out / "retention_vs_snr.png", f"(chi_min=1 at {g_star:.2f} dB)")


if __name__ == "__main__":
    main()
