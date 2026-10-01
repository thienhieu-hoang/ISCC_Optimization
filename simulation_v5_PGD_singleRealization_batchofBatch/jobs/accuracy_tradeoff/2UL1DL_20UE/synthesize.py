#!/usr/bin/env python3
"""Replot publication-quality accuracy_tradeoff.pdf from accuracy_weight.json.

Settings:
- Reads from the latest result_* folder (e.g. result_1) in 2UL1DL_20UE.
- Plots the Pareto frontier: Mean Inference Accuracy Lambda_n vs. Mean Normalised Delay T_n/T_n^ref.
- Outputs accuracy_tradeoff.pdf and accuracy_tradeoff.png with high-resolution publication styling.
"""

from __future__ import annotations

import argparse
import json
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


def plot_accuracy_tradeoff(
    data: dict[str, dict[str, list[float]]],
    beta_vals: list[float],
    out_pdf: Path,
    out_png: Path | None = None,
    label_fontsize: float = 16.0,
    tick_labelsize: float = 15.0,
    legend_fontsize: float = 11.0,
    tick_length: float = 4.0,
    annotate_fontsize: float = 11.0,
) -> None:
    """Render publication-quality Accuracy-Delay Tradeoff Pareto frontier."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.2, 3.8))

    # Curve styling mapping
    styles = {
        "ISCC": {
            "color": "C0",
            "marker": "o",
            "linestyle": "-",
            "label": "ISCC (partial + retention)",
            "markersize": 10,
            "fillstyle": "none",
            "markeredgewidth": 1.2,
        },
        "Atomic": {
            "color": "C1",
            "marker": "v",
            "linestyle": "-",
            "label": r"Atomic task ($\rho=a$, $\chi=1$)",
            "markersize": 10,
            "fillstyle": "none",
            "markeredgewidth": 1.2,
        },
    }

    # Match keys from JSON data
    matched_keys = {}
    for key in data.keys():
        if "iscc" in key.lower():
            matched_keys["ISCC"] = key
        elif "atomic" in key.lower():
            matched_keys["Atomic"] = key

    # Plot curves
    iscc_points = None
    for scheme in ["ISCC", "Atomic"]:
        if scheme not in matched_keys:
            continue
        full_key = matched_keys[scheme]
        series = data[full_key]

        x = np.asarray(series["mean_norm_delay"], dtype=float)
        y = np.asarray(series["mean_accuracy"], dtype=float)

        st = styles[scheme]
        ax.plot(
            x,
            y,
            marker=st["marker"],
            color=st["color"],
            linestyle=st["linestyle"],
            linewidth=1.8,
            markersize=st["markersize"],
            fillstyle=st["fillstyle"],
            markeredgewidth=st["markeredgewidth"],
            label=st["label"],
        )

        if scheme == "ISCC":
            iscc_points = (x, y)

    # Annotate beta^a endpoints on the ISCC curve
    if iscc_points is not None and len(beta_vals) > 0:
        x_iscc, y_iscc = iscc_points
        n_pts = len(beta_vals)

        # First endpoint (beta^a = 0)
        ax.annotate(
            r"$\beta^{\tt a} = 0$",
            (x_iscc[0], y_iscc[0]),
            textcoords="offset points",
            xytext=(10, -3),
            fontsize=annotate_fontsize,
            fontweight="normal",
        )

        # Last endpoint (beta^a = max)
        last_idx = min(n_pts - 1, len(x_iscc) - 1)
        ax.annotate(
            rf"$\beta^{{\tt a}} = {beta_vals[last_idx]:g}$",
            (x_iscc[last_idx], y_iscc[last_idx]),
            textcoords="offset points",
            xytext=(-10, -18),
            fontsize=annotate_fontsize,
            fontweight="normal",
        )

    ax.set_xlabel(r"Norm. Delay $T_n/T_n^{\rm ref}$", fontsize=label_fontsize)
    ax.set_ylabel(r"Accuracy $\Lambda_n$", fontsize=label_fontsize)
    ax.tick_params(axis="both", which="major", labelsize=tick_labelsize, length=tick_length, width=1.0)
    ax.grid(True, alpha=0.35)
    ax.legend(loc="upper left", fontsize=legend_fontsize)
    fig.tight_layout()

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, dpi=200, bbox_inches="tight", pad_inches=0.2)
    if out_png is not None:
        fig.savefig(out_png, dpi=200, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    print(f"[OK] Saved plot: {out_pdf.name} (and .png)")


def plot_accuracy_delay_dual(
    data: dict[str, dict[str, list[float]]],
    beta_vals: list[float],
    out_pdf: Path,
    out_png: Path | None = None,
    label_fontsize: float = 16.0,
    tick_labelsize: float = 15.0,
    legend_fontsize: float = 10.5,
    tick_length: float = 4.0,
) -> None:
    """Render dual y-axis plot: Accuracy (left) and Norm. Delay (right) vs. Accuracy Weight beta^a."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(6.4, 4.0))
    ax2 = ax1.twinx()

    matched_keys = {}
    for key in data.keys():
        if "iscc" in key.lower():
            matched_keys["ISCC"] = key
        elif "atomic" in key.lower():
            matched_keys["Atomic"] = key

    # 4 lines configuration:
    # Scheme -> Color: ISCC -> C0, Atomic -> C1
    # Metric -> Linestyle/Marker:
    #   Accuracy (left) -> solid '-', marker 'o' (ISCC) or 'v' (Atomic)
    #   Norm. Delay (right) -> dashed '--', marker 's' (ISCC) or '^' (Atomic)
    line_configs = [
        ("ISCC", "mean_accuracy", ax1, "-", "o", "C0", r"ISCC: Accuracy (left)"),
        ("Atomic", "mean_accuracy", ax1, "-", "v", "C1", r"Atomic: Accuracy (left)"),
        ("ISCC", "mean_norm_delay", ax2, "--", "s", "C0", r"ISCC: Norm. Delay (right)"),
        ("Atomic", "mean_norm_delay", ax2, "--", "^", "C1", r"Atomic: Norm. Delay (right)"),
    ]

    for scheme, metric, ax, ls, mk, col, lbl in line_configs:
        if scheme not in matched_keys:
            continue
        series = data[matched_keys[scheme]]
        if metric not in series:
            continue
        y = np.asarray(series[metric], dtype=float)
        x = np.asarray(beta_vals[:len(y)], dtype=float)

        ax.plot(
            x,
            y,
            linestyle=ls,
            color=col,
            marker=mk,
            markersize=9,
            fillstyle="none",
            markeredgewidth=1.2,
            linewidth=1.8,
            label=lbl,
        )

    ax1.set_xlabel(r"Accuracy Weight $\beta_n^{\tt a}$", fontsize=label_fontsize)
    ax1.set_ylabel(r"Accuracy $\Lambda_n$", fontsize=label_fontsize)
    ax2.set_ylabel(r"Norm. Delay $T_n/T_n^{\rm ref}$", fontsize=label_fontsize)

    # Set distinct ranges so Atomic Accuracy and Atomic Delay curves do not collide visually
    ax1.set_ylim(0.88, 0.985)
    ax2.set_ylim(0.35, 0.88)

    ax1.tick_params(axis="both", which="major", labelsize=tick_labelsize, length=tick_length, width=1.0)
    ax2.tick_params(axis="y", which="major", labelsize=tick_labelsize, length=tick_length, width=1.0)

    ax1.grid(True, alpha=0.35)

    # Combine legends from both axes and place in clean lower-right area
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right", fontsize=legend_fontsize)

    fig.tight_layout()

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, dpi=200, bbox_inches="tight", pad_inches=0.2)
    if out_png is not None:
        fig.savefig(out_png, dpi=200, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    print(f"[OK] Saved plot: {out_pdf.name} (and .png)")


def plot_accuracy_delay_dual_colored(
    data: dict[str, dict[str, list[float]]],
    beta_vals: list[float],
    out_pdf: Path,
    out_png: Path | None = None,
    label_fontsize: float = 16.0,
    tick_labelsize: float = 15.0,
    legend_fontsize: float = 12.0,
    tick_length: float = 4.0,
) -> None:
    """Render dual y-axis plot color-coded by axis:
    - Left y-axis (Accuracy): Darker Blue (#08519c)
    - Right y-axis (Norm. Delay): Darker Purple (#6a1b9a)
    - Lines match the color of their respective y-axis.
    - Linestyle/marker differentiates scheme:
        * ISCC: Solid line with circle marker
        * Atomic: Dashed line with triangle-down marker
    - Legend contains only 2 entries (ISCC and Atomic).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, ax1 = plt.subplots(figsize=(6.4, 4.0))
    ax2 = ax1.twinx()

    matched_keys = {}
    for key in data.keys():
        if "iscc" in key.lower():
            matched_keys["ISCC"] = key
        elif "atomic" in key.lower():
            matched_keys["Atomic"] = key

    color_acc = "#0072b2" #"#08519c"   # Darker Blue for Accuracy (left axis)
    color_del = "#aa4499" # "#6a1b9a"   # Darker Purple for Delay (right axis)

    # Scheme definitions: (scheme_key, linestyle, marker)
    schemes = [
        ("ISCC", "-", "o"),
        ("Atomic", "--", "v"),
    ]

    for scheme, ls, mk in schemes:
        if scheme not in matched_keys:
            continue
        series = data[matched_keys[scheme]]

        # 1. Plot Accuracy on left axis (Blue)
        if "mean_accuracy" in series:
            y_acc = np.asarray(series["mean_accuracy"], dtype=float)
            x_acc = np.asarray(beta_vals[:len(y_acc)], dtype=float)
            ax1.plot(
                x_acc,
                y_acc,
                linestyle=ls,
                color=color_acc,
                marker=mk,
                markersize=8.5,
                fillstyle="none",
                markeredgewidth=1.3,
                linewidth=1.8,
            )

        # 2. Plot Norm. Delay on right axis (Dark Orange)
        if "mean_norm_delay" in series:
            y_del = np.asarray(series["mean_norm_delay"], dtype=float)
            x_del = np.asarray(beta_vals[:len(y_del)], dtype=float)
            ax2.plot(
                x_del,
                y_del,
                linestyle=ls,
                color=color_del,
                marker=mk,
                markersize=8.5,
                fillstyle="none",
                markeredgewidth=1.3,
                linewidth=1.8,
            )

    # Left axis styling (Accuracy - Blue)
    ax1.set_xlabel(r"Accuracy Weight $\beta_n^{\tt a}$", fontsize=label_fontsize)
    ax1.set_ylabel(r"Accuracy $\Lambda_n$", fontsize=label_fontsize, color=color_acc)
    ax1.tick_params(axis="y", labelcolor=color_acc, colors=color_acc, labelsize=tick_labelsize, length=tick_length, width=1.0)
    ax1.tick_params(axis="x", labelsize=tick_labelsize, length=tick_length, width=1.0)
    ax1.spines["left"].set_color(color_acc)
    ax1.spines["left"].set_linewidth(1.3)
    ax1.set_ylim(0.88, 0.985)
    ax1.grid(True, alpha=0.35)

    # Accuracy threshold line at Lambda^th (value when beta^a = 0)
    if "ISCC" in matched_keys and "mean_accuracy" in data[matched_keys["ISCC"]]:
        lambda_th = float(data[matched_keys["ISCC"]]["mean_accuracy"][0])
        ax1.axhline(
            y=lambda_th,
            color=color_acc,
            linestyle="--",
            linewidth=1.2,
            alpha=0.75,
            zorder=1,
        )
        ax1.text(
            0.32,
            lambda_th + 0.002,
            r"$\Lambda^{\tt th}$ (threshold)",
            color=color_acc,
            fontsize=11.5,
            fontweight="normal",
            zorder=5,
            bbox=dict(boxstyle="square,pad=0.15", facecolor="white", edgecolor="none", alpha=0.8),
        )

    # Right axis styling (Norm. Delay - Dark Orange)
    ax2.set_ylabel(r"Norm. Delay $T_n/T_n^{\rm ref}$", fontsize=label_fontsize, color=color_del)
    ax2.tick_params(axis="y", labelcolor=color_del, colors=color_del, labelsize=tick_labelsize, length=tick_length, width=1.0)
    ax2.spines["right"].set_color(color_del)
    ax2.spines["right"].set_linewidth(1.3)
    ax2.set_ylim(0.35, 0.88)

    # 2 Legend entries: ISCC and Atomic
    legend_elements = [
        Line2D(
            [0], [0],
            color="black",
            linestyle="-",
            marker="o",
            fillstyle="none",
            markeredgewidth=1.3,
            markersize=8.5,
            linewidth=1.8,
            label="ISCC",
        ),
        Line2D(
            [0], [0],
            color="black",
            linestyle="--",
            marker="v",
            fillstyle="none",
            markeredgewidth=1.3,
            markersize=8.5,
            linewidth=1.8,
            label="Atomic",
        ),
    ]
    ax1.legend(
        handles=legend_elements,
        loc="lower right",
        fontsize=legend_fontsize,
        framealpha=0.9,
    )

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
        help="Path to result directory containing accuracy_weight.json (default: latest in target job dir)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Destination directory for output plots (default: syn/ folder next to script)",
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
        default=11.0,
        help="Font size for legend (default: 11)",
    )
    parser.add_argument(
        "--tick-length",
        type=float,
        default=4.0,
        help="Length of tick markers in points (default: 4.0)",
    )
    parser.add_argument(
        "--all-plots",
        action="store_true",
        help="Also replot individual metric curves (accuracy_weight_su, _acc, _delay, _chi)",
    )
    args = parser.parse_args()

    # Find target directory
    res_dir = args.res_dir or _find_latest_result_dir(SCRIPT_DIR)
    if res_dir is None or not res_dir.exists():
        print(f"[Error] No result directory found in: {SCRIPT_DIR}")
        sys.exit(1)

    json_file = res_dir / "accuracy_weight.json"
    if not json_file.exists():
        print(f"[Error] File not found: {json_file}")
        sys.exit(1)

    out_dir = args.out or (SCRIPT_DIR / "syn")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("====================================================================")
    print(" REPLOT ACCURACY-DELAY TRADEOFF PARETO FRONTIER")
    print("====================================================================")
    print(f"Reading from: {json_file.resolve()}")
    print(f"Saving to:    {out_dir.resolve()}")
    print("====================================================================\n")

    payload = json.loads(json_file.read_text())
    beta_vals = payload.get("x", [])
    data = payload.get("data", {})

    # Replot accuracy_tradeoff
    pdf_out = out_dir / "accuracy_tradeoff.pdf"
    png_out = out_dir / "accuracy_tradeoff.png"
    plot_accuracy_tradeoff(
        data=data,
        beta_vals=beta_vals,
        out_pdf=pdf_out,
        out_png=png_out,
        label_fontsize=args.label_fontsize,
        tick_labelsize=args.tick_labelsize,
        legend_fontsize=args.legend_fontsize,
        tick_length=args.tick_length,
    )

    # Plot dual y-axis comparison: Accuracy (left) & Norm. Delay (right) vs. beta^a (4 legend entries)
    pdf_dual = out_dir / "accuracy_delay_dual.pdf"
    png_dual = out_dir / "accuracy_delay_dual.png"
    plot_accuracy_delay_dual(
        data=data,
        beta_vals=beta_vals,
        out_pdf=pdf_dual,
        out_png=png_dual,
        label_fontsize=args.label_fontsize,
        tick_labelsize=args.tick_labelsize,
        legend_fontsize=args.legend_fontsize - 1.0,
        tick_length=args.tick_length,
    )

    # Plot dual y-axis comparison color-coded by axis (2 legend entries: ISCC and Atomic)
    pdf_dual_col = out_dir / "accuracy_delay_dual_colored.pdf"
    png_dual_col = out_dir / "accuracy_delay_dual_colored.png"
    plot_accuracy_delay_dual_colored(
        data=data,
        beta_vals=beta_vals,
        out_pdf=pdf_dual_col,
        out_png=png_dual_col,
        label_fontsize=args.label_fontsize,
        tick_labelsize=args.tick_labelsize,
        legend_fontsize=args.legend_fontsize,
        tick_length=args.tick_length,
    )

    if args.all_plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        extra_specs = [
            ("utility", "System Utility", "accuracy_weight_su"),
            ("mean_accuracy", r"Accuracy $\Lambda_n$", "accuracy_weight_acc"),
            ("mean_norm_delay", r"Norm. Delay $T_n/T_n^{\rm ref}$", "accuracy_weight_delay"),
            ("mean_retention", r"Retention $\chi_n^\star$", "accuracy_weight_chi"),
        ]
        markers = {"ISCC": "o", "Atomic": "v"}
        colors = {"ISCC": "C0", "Atomic": "C1"}
        labels = {"ISCC": "ISCC (partial + retention)", "Atomic": r"Atomic task ($\rho=a$, $\chi=1$)"}

        for metric, ylabel, fname in extra_specs:
            fig, ax = plt.subplots(figsize=(6.0, 3.6))
            for k, series in data.items():
                if metric not in series:
                    continue
                scheme = "ISCC" if "iscc" in k.lower() else "Atomic"
                vals = np.asarray(series[metric], dtype=float)
                ax.plot(
                    beta_vals,
                    vals,
                    marker=markers[scheme],
                    color=colors[scheme],
                    linewidth=1.8,
                    markersize=10,
                    fillstyle="none",
                    markeredgewidth=1.2,
                    label=labels[scheme],
                )
            ax.set_xlabel(r"Accuracy Weight $\beta_n^{\tt a}$", fontsize=args.label_fontsize)
            ax.set_ylabel(ylabel, fontsize=args.label_fontsize)
            ax.tick_params(axis="both", which="major", labelsize=args.tick_labelsize, length=args.tick_length)
            ax.grid(True, alpha=0.35)
            ax.legend(loc="best", fontsize=args.legend_fontsize)
            fig.tight_layout()
            p_pdf = out_dir / f"{fname}.pdf"
            p_png = out_dir / f"{fname}.png"
            fig.savefig(p_pdf, dpi=200, bbox_inches="tight")
            fig.savefig(p_png, dpi=200, bbox_inches="tight")
            plt.close(fig)
            print(f"[OK] Saved plot: {p_pdf.name} (and .png)")

    print(f"\nSuccessfully generated plots in:\n  {out_dir.resolve()}")


if __name__ == "__main__":
    main()
