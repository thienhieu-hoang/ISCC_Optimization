#!/usr/bin/env python3
"""Synthesize UAV Density sweep results for 1DL-20UE with UL UAVs from 1 to 6.

Sources merged:
- 1DL20UE_1_2 (UL UAVs = 1, 2)
- 1DL20UE_3_4 (UL UAVs = 3, 4)
- 1DL20UE_5_6 (UL UAVs = 5, 6)

Outputs generated in jobs/UAV_density/syn/syn_x (auto-incrementing syn_1, syn_2, ...):
- uav_density.json (Unified multi-point sweep JSON)
- uav_density.mat  (MATLAB format)
- summary.json    (Synthesized summary table)
- Publication-quality PDF and PNG plots for Utility, Offloading, Delay, Energy, Accuracy, and Runtime
  matching the exact default color and line style of run_job.py (PGD-BWOA inheriting IWOA-BWOA styles).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Setup paths relative to script location
SCRIPT_DIR = Path(__file__).resolve().parent
SIM_DIR = None
for p in [SCRIPT_DIR, *SCRIPT_DIR.parents]:
    if (p / "stochastic_mec").exists():
        SIM_DIR = p
        break
if SIM_DIR is None:
    SIM_DIR = SCRIPT_DIR.parents[1]

SCRIPTS_DIR = SIM_DIR / "scripts"
if str(SIM_DIR) not in sys.path:
    sys.path.insert(0, str(SIM_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import numpy as np

class SweepResult:
    def __init__(self, x_label: str, x: list[float], data: dict[str, dict[str, list[float]]]):
        self.x_label = x_label
        self.x = [float(v) for v in x]
        self.data = data

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"x_label": self.x_label, "x": self.x, "data": self.data}
        path.write_text(json.dumps(payload, indent=2))
        try:
            from scipy.io import savemat
            safe_data = {
                "".join(c if c.isalnum() else "_" for c in k)[:31].strip("_"): v
                for k, v in self.data.items()
            }
            savemat(path.with_suffix(".mat"), {"x_label": self.x_label, "x": self.x, "data": safe_data})
        except Exception as exc:
            print(f"[warning] could not save .mat file to {path.with_suffix('.mat')}: {exc}")
        return path

# Source folders in UAV density (1DL, 20UE: 1 to 6 UL UAVs)
FOLDERS = [
    "1DL20UE_1_2",
    "1DL20UE_3_4",
    "1DL20UE_5_6",
]

# Exact default curve ordering matching run_job.py (DEFAULT_CURVES)
# PGD-BWOA is index 0 (inheriting the primary color C0 / tab:blue and marker 'o' from IWOA-BWOA)
ALGO_ORDER = [
    "PGD-BWOA",
    "WOA-BWOA",
    "PSO-BWOA",
    "ARJOA",
    "IOJOA",
    "FDMA",
    "ALCA"
]

# Markers matching stochastic_mec/plotting.py: MARKERS = ["o", "v", "s", "^", "x", "d", "*", "p"]
MARKERS = ["o", "v", "s", "^", "x", "d", "*", "p"]


def _next_syn_dir(parent_dir: Path, prefix: str = "syn_") -> Path:
    """Find the next available syn folder: syn_1, syn_2, syn_3, ..."""
    parent_dir.mkdir(parents=True, exist_ok=True)
    idx = 1
    while (parent_dir / f"{prefix}{idx}").exists():
        idx += 1
    out_dir = parent_dir / f"{prefix}{idx}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _find_latest_result_dir(parent_dir: Path) -> Path | None:
    """Find the latest result_X folder inside a job directory."""
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


def collect_sweep_data(
    base_dir: Path,
    folder_names: list[str],
) -> tuple[str, list[float], dict[str, dict[str, list[float]]], list[dict]]:
    """Read .json files from each subfolder and merge them into a sorted dataset."""
    x_label = r"UL-UAV Density [$\times 10^{-6}/m^2$]"
    raw_points_by_x: dict[float, dict[str, dict[str, float]]] = {}
    found_folders: list[dict] = []

    for name in folder_names:
        folder = base_dir / name
        res_dir = _find_latest_result_dir(folder)
        if res_dir is None:
            print(f"[warning] Folder {folder} has no result_* directories. Skipping.")
            continue

        json_file = res_dir / "uav_density.json"
        if not json_file.exists():
            json_file = res_dir / "ue_density.json"
        if not json_file.exists():
            print(f"[warning] No json result found in {res_dir}. Skipping.")
            continue

        print(f"Reading from: {json_file.relative_to(base_dir)}")
        data_block = json.loads(json_file.read_text())
        raw_x_label = data_block.get("x_label")
        if raw_x_label:
            if "ul-uav density" in raw_x_label.lower():
                x_label = r"UL-UAV Density [$\times 10^{-6}/m^2$]"
            else:
                x_label = raw_x_label
        xs = data_block.get("x", [])
        data = data_block.get("data", {})

        found_folders.append({
            "folder": name,
            "result_dir": res_dir.name,
            "x_points": xs,
        })

        for i, x_val in enumerate(xs):
            x_f = float(x_val)
            if x_f not in raw_points_by_x:
                raw_points_by_x[x_f] = {}
            for algo, metrics in data.items():
                if algo not in raw_points_by_x[x_f]:
                    raw_points_by_x[x_f][algo] = {}
                for m_key, m_vals in metrics.items():
                    if i < len(m_vals):
                        raw_points_by_x[x_f][algo][m_key] = m_vals[i]

    if not raw_points_by_x:
        raise RuntimeError("No valid uav_density.json / ue_density.json results found to synthesize!")

    # Sort all points ascending by x
    sorted_xs = sorted(raw_points_by_x.keys())
    print(f"\nSynthesizing {len(sorted_xs)} unique UL-UAV density points: {sorted_xs}")

    # Discover all algorithms present across all points
    all_algos = set()
    for x_f in sorted_xs:
        all_algos.update(raw_points_by_x[x_f].keys())

    # Order algorithms strictly by ALGO_ORDER to match default plot style
    ordered_algos = [a for a in ALGO_ORDER if a in all_algos]
    for a in sorted(all_algos):
        if a not in ordered_algos:
            ordered_algos.append(a)

    # Determine all metric keys available
    all_metrics = set()
    for x_f in sorted_xs:
        for algo in ordered_algos:
            all_metrics.update(raw_points_by_x[x_f].get(algo, {}).keys())

    # Assemble unified data dictionary: data[algo][metric] = [val_x1, val_x2, ...]
    unified_data: dict[str, dict[str, list[float]]] = {}
    for algo in ordered_algos:
        unified_data[algo] = {}
        for m in all_metrics:
            series = []
            for x_f in sorted_xs:
                val = raw_points_by_x[x_f].get(algo, {}).get(m, float("nan"))
                series.append(val)
            unified_data[algo][m] = series

    return x_label, sorted_xs, unified_data, found_folders


def plot_metric(
    xs: list[float],
    data: dict[str, dict[str, list[float]]],
    metric: str,
    xlabel: str,
    ylabel: str,
    out_pdf: Path,
    out_png: Path | None = None,
    title: str | None = None,
    log_y: bool = False,
    filter_algos: list[str] | None = None,
    label_fontsize: float = 16,
    tick_labelsize: float = 15,
    tick_length: float = 4.0,
    legend_fontsize: float = 9,
) -> None:
    """Render plot matching the exact default style and colors of run_job.py / plotting.py."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.0, 3.6))

    algos_to_plot = filter_algos if filter_algos is not None else [a for a in ALGO_ORDER if a in data]
    for a in data:
        if a not in algos_to_plot and (filter_algos is None):
            algos_to_plot.append(a)

    for i, algo in enumerate(algos_to_plot):
        if algo not in data or metric not in data[algo]:
            continue
        vals = np.asarray(data[algo][metric], dtype=float)
        # Use consistent color and marker index based on ALGO_ORDER
        algo_idx = ALGO_ORDER.index(algo) if algo in ALGO_ORDER else i
        color = f"C{algo_idx % 10}"
        marker = MARKERS[algo_idx % len(MARKERS)]

        ax.plot(
            xs,
            vals,
            marker=marker,
            color=color,
            linewidth=1.8,
            markersize=10,
            fillstyle="none",
            markeredgewidth=1.2,
            label=algo,
        )

    ax.set_xlabel(xlabel, fontsize=label_fontsize)
    ax.set_ylabel(ylabel, fontsize=label_fontsize)
    ax.tick_params(axis="both", which="major", labelsize=tick_labelsize, length=tick_length, width=1.0)
    ax.grid(True, alpha=0.35)
    if log_y:
        ax.set_yscale("log")
    if title:
        ax.set_title(title)

    ax.legend(loc="best", fontsize=legend_fontsize)
    fig.tight_layout()

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, dpi=200, bbox_inches="tight")
    if out_png is not None:
        fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot: {out_pdf.name} (and .png)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--folders",
        nargs="+",
        default=FOLDERS,
        help="Subfolder names in jobs/UAV_density to merge (default: FOLDERS)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Destination directory (default: auto-incrementing syn/syn_x)",
    )
    parser.add_argument(
        "--xlabel",
        type=str,
        default=None,
        help="Custom label for x-axis (default: UL-UAV Density [$\\times 10^{-6}/m^2$])",
    )
    parser.add_argument(
        "--label-fontsize",
        type=float,
        default=16.0,
        help="Font size for x and y axis labels (default: 16)",
    )
    parser.add_argument(
        "--tick-labelsize",
        type=float,
        default=15.0,
        help="Font size for axis tick numbers (default: 15)",
    )
    parser.add_argument(
        "--legend-fontsize",
        type=float,
        default=9.0,
        help="Font size for plot legend (default: 9)",
    )
    parser.add_argument(
        "--tick-length",
        type=float,
        default=4.0,
        help="Length of tick markers in points (default: 4.0)",
    )
    args = parser.parse_args()

    # Automatically find next syn_x directory if --out is not explicitly specified
    syn_parent = SCRIPT_DIR / "syn"
    if args.out is None:
        out_dir = _next_syn_dir(syn_parent, prefix="syn_")
    else:
        out_dir = args.out
        out_dir.mkdir(parents=True, exist_ok=True)

    print("====================================================================")
    print(" SYNTHESIZE UAV DENSITY SWEEP (1DL-20UE: 1 to 6 UL UAVs)")
    print("====================================================================")
    print(f"Base Directory: {SCRIPT_DIR}")
    print(f"Target Output:  {out_dir}")
    print(f"Folders to scan: {len(args.folders)}")
    print("====================================================================\n")

    # 1. READ AND MERGE JSON DATA
    x_label, sorted_xs, unified_data, found_folders = collect_sweep_data(
        SCRIPT_DIR,
        args.folders,
    )
    if args.xlabel is not None:
        x_label = args.xlabel

    sweep_res = SweepResult(x_label, sorted_xs, unified_data)

    # 2. SAVE STRUCTURED SYNTHESIZED RESULTS (.json and .mat)
    json_path = out_dir / "uav_density.json"
    sweep_res.save(json_path)
    print(f"\n[OK] Saved merged sweep to: {json_path} (and .mat)")

    # 3. BUILD AND SAVE SYNTHESIS SUMMARY JSON
    summary_table = {}
    for i, x in enumerate(sorted_xs):
        x_key = f"{x:g}"
        summary_table[x_key] = {}
        for algo in unified_data:
            summary_table[x_key][algo] = {
                "utility": round(unified_data[algo]["utility"][i], 4) if "utility" in unified_data[algo] and i < len(unified_data[algo]["utility"]) else None,
                "offload_ratio": round(unified_data[algo]["offload_ratio"][i], 4) if "offload_ratio" in unified_data[algo] and i < len(unified_data[algo]["offload_ratio"]) else None,
                "mean_accuracy": round(unified_data[algo]["mean_accuracy"][i], 4) if "mean_accuracy" in unified_data[algo] and i < len(unified_data[algo]["mean_accuracy"]) else None,
                "mean_norm_delay": round(unified_data[algo]["mean_norm_delay"][i], 4) if "mean_norm_delay" in unified_data[algo] and i < len(unified_data[algo]["mean_norm_delay"]) else None,
                "total_energy": round(unified_data[algo]["total_energy"][i], 4) if "total_energy" in unified_data[algo] and i < len(unified_data[algo]["total_energy"]) else None,
                "runtime_s": round(unified_data[algo]["runtime"][i], 2) if "runtime" in unified_data[algo] and i < len(unified_data[algo]["runtime"]) else None,
            }

    summary_payload = {
        "synthesized_job": "UAV_density_1DL20UE_1to6",
        "n_dl": 1,
        "n_ue": 20,
        "source_folders": found_folders,
        "x": sorted_xs,
        "algorithms": list(unified_data.keys()),
        "summary": summary_table,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2))
    print(f"[OK] Saved summary table to: {summary_path}")

    # 4. PRINT TERMINAL COMPARISON TABLE FOR SYSTEM UTILITY
    algos_to_show = [a for a in ALGO_ORDER if a in unified_data]
    for a in unified_data:
        if a not in algos_to_show:
            algos_to_show.append(a)
    header_cols = [f"{algo:<12}" for algo in algos_to_show]
    header_str = f"{'UL-UAV Dens':<12} | " + " | ".join(header_cols)
    sep_line = "=" * len(header_str)
    print("\n" + sep_line)
    print(header_str)
    print(sep_line)
    for i, x in enumerate(sorted_xs):
        row_cols = []
        for algo in algos_to_show:
            vals = unified_data.get(algo, {}).get("utility", [])
            val_str = f"{vals[i]:6.3f}" if i < len(vals) and not np.isnan(vals[i]) else "N/A"
            row_cols.append(f"{val_str:<12}")
        print(f"M_ul = {x:<6g} | " + " | ".join(row_cols))
    print(sep_line)

    # 5. GENERATE PUBLICATION-QUALITY PLOTS WITH EXACT RUN_JOB.PY DEFAULT STYLING
    print("\nGenerating publication figures matching run_job.py default styling (PDF & PNG)...")
    style_kwargs = dict(
        label_fontsize=args.label_fontsize,
        tick_labelsize=args.tick_labelsize,
        tick_length=args.tick_length,
        legend_fontsize=args.legend_fontsize,
    )
    plots_spec = [
        ("utility", "System Utility", "uav_density_su", False, None),
        ("offload_ratio", "Offloading Percentage", "uav_density_po", False, None),
        ("mean_norm_delay", r"Mean Normalised Delay $T_n/T_n^{\rm ref}$", "uav_density_delay", False, None),
        ("total_energy", "Total Energy Consumption [J]", "uav_density_energy", False, None),
        ("mean_accuracy", r"Mean Inference Accuracy $\Lambda_n$", "uav_density_acc", False, None),
        ("runtime", "Execution Runtime [s]", "uav_density_runtime", True, None),
        ("runtime", "Execution Runtime [s]", "uav_density_runtime_bwoa", True, ["PGD-BWOA", "WOA-BWOA", "PSO-BWOA"]),
        ("runtime", "Execution Runtime [s]", "uav_density_runtime_bwoa_linear", False, ["PGD-BWOA", "WOA-BWOA", "PSO-BWOA"]),
    ]

    for metric, ylabel, fname, log_y, filter_algos in plots_spec:
        pdf_f = out_dir / f"{fname}.pdf"
        png_f = out_dir / f"{fname}.png"
        plot_metric(
            sorted_xs,
            unified_data,
            metric,
            x_label,
            ylabel,
            pdf_f,
            png_f,
            log_y=log_y,
            filter_algos=filter_algos,
            **style_kwargs,
        )

    print(f"\nAll synthesized files successfully written to:\n  {out_dir.resolve()}")


if __name__ == "__main__":
    main()
