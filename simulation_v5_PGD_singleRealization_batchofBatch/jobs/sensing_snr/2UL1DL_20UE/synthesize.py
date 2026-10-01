#!/usr/bin/env python3
"""Synthesize and replot publication-quality retention_vs_snr from sensing_snr.json.

Settings:
- Reads from the latest result_* folder (e.g. result_1) in jobs/sensing_snr/2UL1DL_20UE.
- Plots optimal retention chi_n^*, retention floor chi_n^min, and offloading ratio vs. mean sensing SNR.
- Colors:
  * Blue (#0072B2): Optimal retention chi_n^* (solid) and Retention floor chi_n^min (dashed)
  * Purple (#AA4499): Offloading ratio (solid diamond)
- Shaded gray area for the inadmissible regime where chi_n^min > 1.
- Outputs retention_vs_snr.pdf and retention_vs_snr.png in syn/ directory.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

# Setup paths relative to script location
SCRIPT_DIR = Path(__file__).resolve().parent

import numpy as np


def _find_latest_result_dir(parent_dir: Path) -> Path | None:
    """Find the latest result_X folder inside the job directory."""
    if not parent_dir.exists():
        return None
    res_dirs = [
        d for d in parent_dir.iterdir()
        if d.is_dir() and d.name.startswith("result_")
    ]
    if not res_dirs:
        return None
    def _idx(d: Path) -> int:
        parts = d.name.split("_")
        return int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
    return sorted(res_dirs, key=_idx)[-1]


def compute_retention_floor(
    snr_db_vals: list[float] | np.ndarray,
    accuracy_threshold: float = 0.90,
    accuracy_sensitivity: float = 1.0,
) -> tuple[np.ndarray, float]:
    """Compute theoretical retention floor chi_n^min and the threshold SNR g_star (in dB) where chi^min = 1."""
    x = np.asarray(snr_db_vals, dtype=float)
    snr_lin = 10.0 ** (x / 10.0)
    sensing_info = np.log2(1.0 + snr_lin)
    floor = -np.log(1.0 - accuracy_threshold) / (accuracy_sensitivity * sensing_info)
    
    # Boundary where chi^min == 1
    # 1.0 = -ln(1 - Lambda^th) / (vartheta * log2(1 + gamma))
    # log2(1 + gamma) = -ln(1 - Lambda^th) / vartheta
    g_star = 10.0 * np.log10(2.0 ** (-np.log(1.0 - accuracy_threshold) / accuracy_sensitivity) - 1.0)
    return floor, float(g_star)


def plot_retention_vs_snr(
    data: dict[str, dict[str, list[float]]],
    snr_vals: list[float],
    out_pdf: Path,
    out_png: Path | None = None,
    color_blue: str = "#0072B2",
    color_purple: str = "#AA4499",
    accuracy_threshold: float = 0.90,
    accuracy_sensitivity: float = 1.0,
    label_fontsize: float = 16.0,
    tick_labelsize: float = 15.0,
    legend_fontsize: float = 12.0,
    tick_length: float = 4.0,
) -> None:
    """Render publication-quality Retention and Offloading vs. Sensing SNR plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matched_key = None
    for key in data.keys():
        if "iscc" in key.lower():
            matched_key = key
            break
    if matched_key is None:
        raise ValueError("Could not find ISCC series in data keys: " + str(list(data.keys())))

    s = data[matched_key]
    x = np.asarray(snr_vals, dtype=float)

    # Compute theoretical retention floor and threshold
    raw_floor, g_star = compute_retention_floor(x, accuracy_threshold, accuracy_sensitivity)
    floor_capped = np.minimum(raw_floor, 1.0)

    fig, ax = plt.subplots(figsize=(6.4, 4.0))

    # Shaded gray area for the inadmissible regime
    x_min_bound = x[0] - 0.8
    ax.axvspan(x_min_bound, g_star, color="0.92", zorder=0)
    ax.text(
        (x_min_bound + g_star) / 2.0,
        0.52,
        "inadmissible\n($\\chi_n^{\\min} > 1$)",
        ha="center",
        va="center",
        fontsize=11.0,
        color="0.40",
        fontweight="normal",
        zorder=1,
    )

    # Line 1: Optimal retention chi_n^*
    if "mean_retention" in s:
        ret_vals = np.asarray(s["mean_retention"][:len(x)], dtype=float)
        ax.plot(
            x,
            ret_vals,
            color=color_blue,
            linestyle="-",
            marker="o",
            markersize=8.5,
            fillstyle="none",
            markeredgewidth=1.3,
            linewidth=1.8,
            label=r"Optimal retention $\chi_n^\star$",
            zorder=3,
        )

    # Line 2: Retention floor chi_n^min
    ax.plot(
        x,
        floor_capped,
        color=color_blue,
        linestyle="--",
        linewidth=1.6,
        label=r"Retention floor $\chi_n^{\min}$",
        zorder=3,
    )

    # Line 3: Fraction of offloading UEs (offload_ratio)
    if "offload_ratio" in s:
        offload_vals = np.asarray(s["offload_ratio"][:len(x)], dtype=float)
        ax.plot(
            x,
            offload_vals,
            color=color_purple,
            linestyle="-",
            marker="D",
            markersize=8.0,
            fillstyle="none",
            markeredgewidth=1.3,
            linewidth=1.8,
            label=r"Offloading ratio",
            zorder=3,
        )

    ax.set_xlabel(r"Mean Sensing SNR $\bar\gamma^{\tt sen}$ [dB]", fontsize=label_fontsize)
    ax.set_ylabel(r"Ratio", fontsize=label_fontsize)

    ax.set_xlim(x[0] - 0.6, x[-1] + 0.6)
    ax.set_ylim(-0.02, 1.05)

    ax.tick_params(axis="both", which="major", labelsize=tick_labelsize, length=tick_length, width=1.0)
    ax.grid(True, alpha=0.35)
    ax.legend(loc="center right", fontsize=legend_fontsize, framealpha=0.9)

    fig.tight_layout()

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, dpi=200, bbox_inches="tight", pad_inches=0.2)
    if out_png is not None:
        fig.savefig(out_png, dpi=200, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    print(f"[OK] Saved plot: {out_pdf.name} (and .png)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--res-dir",
        type=Path,
        default=None,
        help="Path to result directory containing sensing_snr.json (default: latest in target job dir)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Destination directory for output plots (default: syn/ folder next to script)",
    )
    parser.add_argument(
        "--color-blue",
        type=str,
        default="#0072B2",
        help="Color for retention curves (default: #0072B2)",
    )
    parser.add_argument(
        "--color-purple",
        type=str,
        default="#AA4499",
        help="Color for offloading ratio curve (default: #AA4499)",
    )
    parser.add_argument(
        "--label-fontsize",
        type=float,
        default=16.0,
        help="Font size for axis labels (default: 16)",
    )
    parser.add_argument(
        "--tick-labelsize",
        type=float,
        default=15.0,
        help="Font size for tick numbers (default: 15)",
    )
    parser.add_argument(
        "--legend-fontsize",
        type=float,
        default=12.0,
        help="Font size for legend (default: 12)",
    )
    parser.add_argument(
        "--tick-length",
        type=float,
        default=4.0,
        help="Length of tick markers in points (default: 4.0)",
    )
    parser.add_argument(
        "--copy-to-paper",
        action="store_true",
        help="Also copy retention_vs_snr.pdf to ISCC_Optimization/figures/",
    )
    args = parser.parse_args()

    # Find target directory
    res_dir = args.res_dir or _find_latest_result_dir(SCRIPT_DIR)
    if res_dir is None or not res_dir.exists():
        print(f"[Error] No result directory found in: {SCRIPT_DIR}")
        sys.exit(1)

    json_file = res_dir / "sensing_snr.json"
    if not json_file.exists():
        print(f"[Error] File not found: {json_file}")
        sys.exit(1)

    out_dir = args.out or (SCRIPT_DIR / "syn")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("====================================================================")
    print(" SYNTHESIZE RETENTION & OFFLOADING VS. SENSING SNR")
    print("====================================================================")
    print(f"Reading from: {json_file.resolve()}")
    print(f"Saving to:    {out_dir.resolve()}")
    print("====================================================================\n")

    payload = json.loads(json_file.read_text())
    snr_vals = payload.get("x", [])
    data = payload.get("data", {})

    pdf_out = out_dir / "retention_vs_snr.pdf"
    png_out = out_dir / "retention_vs_snr.png"

    plot_retention_vs_snr(
        data=data,
        snr_vals=snr_vals,
        out_pdf=pdf_out,
        out_png=png_out,
        color_blue=args.color_blue,
        color_purple=args.color_purple,
        label_fontsize=args.label_fontsize,
        tick_labelsize=args.tick_labelsize,
        legend_fontsize=args.legend_fontsize,
        tick_length=args.tick_length,
    )

    # Optional: Only copy to paper figures directory if explicitly requested via --copy-to-paper
    if args.copy_to_paper:
        paper_fig_dir = Path(__file__).resolve().parents[4] / "ISCC_Optimization" / "figures"
        if paper_fig_dir.exists():
            dest_pdf = paper_fig_dir / "retention_vs_snr.pdf"
            dest_png = paper_fig_dir / "retention_vs_snr.png"
            try:
                shutil.copy2(pdf_out, dest_pdf)
                shutil.copy2(png_out, dest_png)
                print(f"[OK] Synced to paper figures:\n  {dest_pdf}\n  {dest_png}")
            except Exception as e:
                print(f"[Warning] Could not copy to paper figures: {e}")

    print(f"\nSuccessfully generated plots in:\n  {out_dir.resolve()}")


if __name__ == "__main__":
    main()
