#!/usr/bin/env python3
"""Synthesize benchmark results comparing BWOA, PGD-BWOA, and Exhaustive Search (EX) from 2 to 8 UEs.

Sources merged:
- BWOA (WOA-BWOA):
  - BWOA_3subc_2UL1DL_2to4UE (UEs = 2, 3, 4)
  - BWOA_3subc_2UL1DL_5UE     (UEs = 5)
  - BWOA_3subc_2UL1DL_6UE     (UEs = 6)
  - BWOA_3subc_2UL1DL_7UE     (UEs = 7)
  - BWOA_3subc_2UL1DL_8UE     (UEs = 8)
- Exhaustive Search (EX):
  - exhaustive_3sub_2UL1DL_2to4UE (UEs = 2, 3, 4)
  - exhaustive_3sub_2UL1DL_5UE     (UEs = 5)
  - exhaustive_3sub_2UL1DL_6UE     (UEs = 6)
  - exhaustive_3sub_2UL1DL_7UE     (UEs = 7)
  - exhaustive_3sub_2UL1DL_8UE     (UEs = 8)
- PGD-BWOA:
  - PGD_BWOA_3subc_2UL1DL_2to4UE (UEs = 2, 3, 4)
  - PGD_BWOA_3subc_2UL1DL_5UE     (UEs = 5)
  - PGD_BWOA_3subc_2UL1DL_6UE     (UEs = 6)
  - PGD_BWOA_3subc_2UL1DL_7UE     (UEs = 7)
  - PGD_BWOA_3subc_2UL1DL_8UE     (UEs = 8)

Outputs generated in jobs/compare_exhaustive/syn/syn_x (auto-incrementing syn_1, syn_2, ...):
- compare_exhaustive.json (Unified multi-point sweep JSON)
- compare_exhaustive.mat  (MATLAB format)
- summary.json            (Synthesized summary and optimality gap table)
- Two types of publication figures (both PDF and PNG at 200 dpi):
  1. WOA-BWOA vs. EX
     - compare_woa_vs_ex_su
     - compare_woa_vs_ex_time (log & linear)
     - compare_woa_vs_ex_po
     - compare_woa_vs_ex_gap
  2. 3 approaches: PGD-BWOA, WOA-BWOA, and EX
     - compare_3approaches_su (and compare_with_ex_su)
     - compare_3approaches_time (log & linear, and compare_with_ex_time)
     - compare_3approaches_po
     - compare_3approaches_gap
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

# Source subfolders in jobs/compare_exhaustive
FOLDERS = [
    # BWOA (WOA-BWOA)
    "BWOA_3subc_2UL1DL_2to4UE",
    "BWOA_3subc_2UL1DL_5UE",
    "BWOA_3subc_2UL1DL_6UE",
    "BWOA_3subc_2UL1DL_7UE",
    "BWOA_3subc_2UL1DL_8UE",
    # Exhaustive search (EX)
    "exhaustive_3sub_2UL1DL_2to4UE",
    "exhaustive_3sub_2UL1DL_5UE",
    "exhaustive_3sub_2UL1DL_6UE",
    "exhaustive_3sub_2UL1DL_7UE",
    "exhaustive_3sub_2UL1DL_8UE",
    # PGD-BWOA
    "PGD_BWOA_3subc_2UL1DL_2to4UE",
    "PGD_BWOA_3subc_2UL1DL_5UE",
    "PGD_BWOA_3subc_2UL1DL_6UE",
    "PGD_BWOA_3subc_2UL1DL_7UE",
    "PGD_BWOA_3subc_2UL1DL_8UE",
]

# Canonical algorithm order
ALGO_ORDER = [
    "PGD-BWOA",
    "WOA-BWOA",
    "EX",
]

# Marker list consistent with stochastic_mec/plotting.py
MARKERS = ["o", "v", "s", "^", "x", "d", "*", "p"]

# Fixed styling matching UE_density/synthesize.py
# PGD-BWOA: C0 (blue), circle 'o'
# WOA-BWOA: C1 (orange), triangle-down 'v'
# EX: dark red dashed line, square 's'
ALGO_STYLE_MAP = {
    "PGD-BWOA": {
        "color": "C0",
        "marker": "o",
        "linestyle": "-",
        "label": "PGD-BWOA",
    },
    "WOA-BWOA": {
        "color": "C1",
        "marker": "v",
        "linestyle": "-",
        "label": "WOA-BWOA",
    },
    "EX": {
        "color": "#c10000",
        "marker": "s",
        "linestyle": "--",
        "label": "Exhaustive Search",
    },
}


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
    """Find the latest result_X folder inside a subfolder."""
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
) -> tuple[str, list[float], dict[str, dict[str, list[float]]], list[dict], dict[float, dict[str, dict]]]:
    """Read results from each folder and merge into unified dataset."""
    x_label = r"Active-UE Density [$\times 10^{-6}/m^2$]"
    raw_points_by_x: dict[float, dict[str, dict[str, float]]] = {}
    found_folders: list[dict] = []
    metadata_by_x: dict[float, dict[str, dict]] = {}

    for name in folder_names:
        folder = base_dir / name
        res_dir = _find_latest_result_dir(folder)
        if res_dir is None:
            print(f"[warning] Folder {folder.name} has no result_* directories. Skipping.")
            continue

        # Prefer compare_with_ex.json
        json_file = res_dir / "compare_with_ex.json"
        if not json_file.exists():
            json_file = res_dir / "ue_density.json"
        if not json_file.exists():
            print(f"[warning] No json result found in {res_dir.relative_to(base_dir)}. Skipping.")
            continue

        print(f"Reading: {res_dir.relative_to(base_dir)} / {json_file.name}")
        data_block = json.loads(json_file.read_text())
        raw_x_label = data_block.get("x_label")
        if raw_x_label:
            if "active-ue density" in raw_x_label.lower():
                x_label = r"Active-UE Density [$\times 10^{-6}/m^2$]"
            else:
                x_label = raw_x_label
        xs = data_block.get("x", [])
        data = data_block.get("data", {})

        # Optional metadata from summary.json if available
        sum_file = res_dir / "summary.json"
        records = []
        if sum_file.exists():
            try:
                sum_data = json.loads(sum_file.read_text())
                records = sum_data.get("records", [])
            except Exception:
                pass

        # For PGD_BWOA_3subc_2UL1DL_2to4UE, only take densities 2, 3, 4 (discard 5 and 6)
        valid_xs = [
            x_val for x_val in xs
            if not ("PGD_BWOA" in name and "2to4UE" in name and float(x_val) > 4.0)
        ]
        found_folders.append({
            "folder": name,
            "result_dir": res_dir.name,
            "x_points": valid_xs,
        })

        for i, x_val in enumerate(xs):
            x_f = float(x_val)
            # Skip densities 5 and 6 from the 2to4UE PGD folder (taken from dedicated 5UE & 6UE folders)
            if "PGD_BWOA" in name and "2to4UE" in name and x_f > 4.0:
                continue

            if x_f not in raw_points_by_x:
                raw_points_by_x[x_f] = {}
            if x_f not in metadata_by_x:
                metadata_by_x[x_f] = {}

            # Associate record if found
            rec_match = next((r for r in records if float(r.get("density", -1)) == x_f), None)

            for raw_algo, metrics in data.items():
                # Canonicalize algorithm name: BWOA -> WOA-BWOA
                algo = "WOA-BWOA" if raw_algo == "BWOA" else raw_algo
                if algo not in raw_points_by_x[x_f]:
                    raw_points_by_x[x_f][algo] = {}
                for m_key, m_vals in metrics.items():
                    if i < len(m_vals):
                        raw_points_by_x[x_f][algo][m_key] = m_vals[i]

                if rec_match:
                    metadata_by_x[x_f][algo] = rec_match

    if not raw_points_by_x:
        raise RuntimeError("No valid compare_with_ex.json results found to synthesize!")

    sorted_xs = sorted(raw_points_by_x.keys())
    print(f"\nSynthesizing {len(sorted_xs)} unique active-UE points: {sorted_xs}")

    # Determine algorithms present
    all_algos = set()
    for x_f in sorted_xs:
        all_algos.update(raw_points_by_x[x_f].keys())

    ordered_algos = [a for a in ALGO_ORDER if a in all_algos]
    for a in sorted(all_algos):
        if a not in ordered_algos:
            ordered_algos.append(a)

    # Determine metric keys available
    all_metrics = set()
    for x_f in sorted_xs:
        for algo in ordered_algos:
            all_metrics.update(raw_points_by_x[x_f].get(algo, {}).keys())

    # Build unified data dictionary: data[algo][metric] = [val_x1, val_x2, ...]
    unified_data: dict[str, dict[str, list[float]]] = {}
    for algo in ordered_algos:
        unified_data[algo] = {}
        for m in all_metrics:
            series = []
            for x_f in sorted_xs:
                val = raw_points_by_x[x_f].get(algo, {}).get(m, float("nan"))
                series.append(val)
            unified_data[algo][m] = series

    # Calculate optimality gap vs EX if EX and utility exist
    if "EX" in unified_data and "utility" in unified_data["EX"]:
        ex_utils = unified_data["EX"]["utility"]
        for algo in ordered_algos:
            gaps = []
            algo_utils = unified_data[algo].get("utility", [])
            for i, ue in enumerate(ex_utils):
                if i < len(algo_utils) and not np.isnan(algo_utils[i]) and not np.isnan(ue):
                    denom = abs(ue) if abs(ue) > 1e-9 else 1.0
                    gap = 100.0 * (ue - algo_utils[i]) / denom
                    gaps.append(max(0.0, gap))
                else:
                    gaps.append(float("nan"))
            unified_data[algo]["gap"] = gaps

    return x_label, sorted_xs, unified_data, found_folders, metadata_by_x


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
    algo_label_map: dict[str, str] | None = None,
    label_fontsize: float = 16,
    tick_labelsize: float = 15,
    tick_length: float = 4.0,
    legend_fontsize: float = 12,
    y_min: float | None = None,
    y_max: float | None = None,
) -> None:
    """Render publication-quality plot matching the exact style of synthesize.py."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.0, 3.6))

    algos_to_plot = filter_algos if filter_algos is not None else list(data.keys())
    active_algos = [a for a in algos_to_plot if a in data and metric in data[a]]

    for i, algo in enumerate(algos_to_plot):
        if algo not in data or metric not in data[algo]:
            continue
        vals = np.asarray(data[algo][metric], dtype=float)

        style = ALGO_STYLE_MAP.get(algo, {})
        color = style.get("color", f"C{i % 10}")
        marker = style.get("marker", MARKERS[i % len(MARKERS)])
        linestyle = style.get("linestyle", "-")
        label = style.get("label", algo)

        if algo_label_map and algo in algo_label_map:
            label = algo_label_map[algo]
        elif algo == "WOA-BWOA" and len(active_algos) == 2 and "EX" in active_algos:
            label = "BWOA"

        ax.plot(
            xs,
            vals,
            marker=marker,
            color=color,
            linestyle=linestyle,
            linewidth=1.8,
            markersize=10,
            fillstyle="none",
            markeredgewidth=1.2,
            label=label,
        )

    ax.set_xlabel(xlabel, fontsize=label_fontsize)
    ax.set_ylabel(ylabel, fontsize=label_fontsize)
    ax.tick_params(axis="both", which="major", labelsize=tick_labelsize, length=tick_length, width=1.0)
    ax.grid(True, alpha=0.35)

    if log_y:
        ax.set_yscale("log")
    if y_min is not None or y_max is not None:
        ax.set_ylim(bottom=y_min, top=y_max)
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
        help="Subfolder names in jobs/compare_exhaustive to merge",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Destination directory (default: auto-incrementing syn/syn_x)",
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
        default=12.0,
        help="Font size for plot legend (default: 12)",
    )
    parser.add_argument(
        "--tick-length",
        type=float,
        default=4.0,
        help="Length of tick markers in points (default: 4.0)",
    )
    parser.add_argument(
        "--ex-color",
        type=str,
        default=None,
        help="Line and marker color for Exhaustive search (default: from ALGO_STYLE_MAP)",
    )
    parser.add_argument(
        "--ex-linestyle",
        type=str,
        default=None,
        help="Line style for Exhaustive search (default: from ALGO_STYLE_MAP)",
    )
    parser.add_argument(
        "--ex-marker",
        type=str,
        default=None,
        help="Marker for Exhaustive search (default: from ALGO_STYLE_MAP)",
    )
    parser.add_argument(
        "--xlabel",
        type=str,
        default=None,
        help="Custom label for x-axis (default: Active-UE Density [$\\times 10^{-6}/m^2$])",
    )
    args = parser.parse_args()

    # Update EX styling only if custom command-line flags were explicitly passed
    if args.ex_color is not None:
        ALGO_STYLE_MAP["EX"]["color"] = args.ex_color
    if args.ex_linestyle is not None:
        ALGO_STYLE_MAP["EX"]["linestyle"] = args.ex_linestyle
    if args.ex_marker is not None:
        ALGO_STYLE_MAP["EX"]["marker"] = args.ex_marker

    ex_color = ALGO_STYLE_MAP["EX"]["color"]
    ex_linestyle = ALGO_STYLE_MAP["EX"]["linestyle"]
    ex_marker = ALGO_STYLE_MAP["EX"]["marker"]

    # Find next incremental syn_x directory
    syn_parent = SCRIPT_DIR / "syn"
    if args.out is None:
        out_dir = _next_syn_dir(syn_parent, prefix="syn_")
    else:
        out_dir = args.out
        out_dir.mkdir(parents=True, exist_ok=True)

    print("====================================================================")
    print(" SYNTHESIZE EXHAUSTIVE BENCHMARK (2 to 8 Active UEs)")
    print("====================================================================")
    print(f"Base Directory:   {SCRIPT_DIR}")
    print(f"Target Output:    {out_dir}")
    print(f"Folders to scan:  {len(args.folders)}")
    print(f"Font settings:    label={args.label_fontsize}pt, tick={args.tick_labelsize}pt, legend={args.legend_fontsize}pt")
    print(f"Styles:           PGD-BWOA=C0 (circle), WOA-BWOA=C1 (tri-down), EX={ex_color} ({ex_marker}, {ex_linestyle})")
    print("====================================================================\n")

    # 1. READ AND MERGE JSON DATA
    x_label, sorted_xs, unified_data, found_folders, metadata_by_x = collect_sweep_data(
        SCRIPT_DIR,
        args.folders,
    )
    if args.xlabel is not None:
        x_label = args.xlabel

    sweep_res = SweepResult(x_label, sorted_xs, unified_data)

    # 2. SAVE STRUCTURED SYNTHESIZED RESULTS (.json and .mat)
    json_path = out_dir / "compare_exhaustive.json"
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
                "runtime_s": round(unified_data[algo]["runtime"][i], 4) if "runtime" in unified_data[algo] and i < len(unified_data[algo]["runtime"]) else None,
                "offload_ratio": round(unified_data[algo]["offload_ratio"][i], 4) if "offload_ratio" in unified_data[algo] and i < len(unified_data[algo]["offload_ratio"]) else None,
                "gap_pct": round(unified_data[algo]["gap"][i], 3) if "gap" in unified_data[algo] and i < len(unified_data[algo]["gap"]) else None,
            }

    gap_table = {}
    if "EX" in unified_data and "utility" in unified_data["EX"]:
        for i, x in enumerate(sorted_xs):
            x_key = f"{x:g}"
            u_ex = unified_data["EX"]["utility"][i]
            gap_table[x_key] = {}
            for algo in ["PGD-BWOA", "WOA-BWOA"]:
                if algo in unified_data and "utility" in unified_data[algo]:
                    u_algo = unified_data[algo]["utility"][i]
                    denom = abs(u_ex) if abs(u_ex) > 1e-9 else 1.0
                    gap_pct = 100.0 * (u_ex - u_algo) / denom
                    gap_table[x_key][algo] = round(gap_pct, 3)

    summary_payload = {
        "synthesized_job": "compare_exhaustive_2to8UE",
        "subchannels": 3,
        "n_ul": 2,
        "n_dl": 1,
        "source_folders": found_folders,
        "x": sorted_xs,
        "algorithms": list(unified_data.keys()),
        "summary": summary_table,
        "optimality_gap_pct": gap_table,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2))
    print(f"[OK] Saved summary table to: {summary_path}")

    # 4. PRINT TERMINAL COMPARISON TABLES
    algos_to_show = [a for a in ALGO_ORDER if a in unified_data]

    # --- System Utility Table ---
    print("\n" + "=" * 65)
    print(" SYSTEM UTILITY COMPARISON")
    print("=" * 65)
    header_cols = [f"{algo:<12}" for algo in algos_to_show]
    print(f"{'Active UEs':<12} | " + " | ".join(header_cols))
    print("-" * 65)
    for i, x in enumerate(sorted_xs):
        row = []
        for algo in algos_to_show:
            vals = unified_data.get(algo, {}).get("utility", [])
            v_str = f"{vals[i]:6.4f}" if i < len(vals) and not np.isnan(vals[i]) else "N/A"
            row.append(f"{v_str:<12}")
        print(f"N = {x:<8g} | " + " | ".join(row))
    print("=" * 65)

    # --- Execution Runtime Table ---
    print("\n" + "=" * 65)
    print(" EXECUTION RUNTIME [s] COMPARISON")
    print("=" * 65)
    print(f"{'Active UEs':<12} | " + " | ".join(header_cols))
    print("-" * 65)
    for i, x in enumerate(sorted_xs):
        row = []
        for algo in algos_to_show:
            vals = unified_data.get(algo, {}).get("runtime", [])
            v_str = f"{vals[i]:8.2f}s" if i < len(vals) and not np.isnan(vals[i]) else "N/A"
            row.append(f"{v_str:<12}")
        print(f"N = {x:<8g} | " + " | ".join(row))
    print("=" * 65)

    # --- Optimality Gap Table ---
    if "EX" in unified_data:
        print("\n" + "=" * 65)
        print(" OPTIMALITY GAP [%] VS. EXHAUSTIVE SEARCH (EX)")
        print("=" * 65)
        compare_algos = [a for a in ["PGD-BWOA", "WOA-BWOA"] if a in unified_data]
        gap_cols = [f"{algo:<12}" for algo in compare_algos]
        print(f"{'Active UEs':<12} | " + " | ".join(gap_cols))
        print("-" * 65)
        for i, x in enumerate(sorted_xs):
            row = []
            for algo in compare_algos:
                gaps = unified_data.get(algo, {}).get("gap", [])
                g_str = f"{gaps[i]:6.2f}%" if i < len(gaps) and not np.isnan(gaps[i]) else "N/A"
                row.append(f"{g_str:<12}")
            print(f"N = {x:<8g} | " + " | ".join(row))
        print("=" * 65)

    # 5. GENERATE PUBLICATION-QUALITY FIGURES (PDF & PNG)
    print("\nGenerating publication figures (PDF & PNG)...")

    # Common font keyword args
    style_kwargs = dict(
        label_fontsize=args.label_fontsize,
        tick_labelsize=args.tick_labelsize,
        tick_length=args.tick_length,
        legend_fontsize=args.legend_fontsize,
    )

    # TYPE 1: WOA-BWOA vs. EX (shows legend "BWOA" when comparing 2 lines)
    print("\n--- Type 1: WOA-BWOA vs. EX ---")
    woa_vs_ex_algos = ["WOA-BWOA", "EX"]
    plot_specs_type1 = [
        ("utility", "System Utility", "compare_woa_vs_ex_su", False),
        ("runtime", "Execution Runtime [s]", "compare_woa_vs_ex_time", True),
        ("runtime", "Execution Runtime [s]", "compare_woa_vs_ex_time_linear", False),
        ("offload_ratio", "Offloading Percentage", "compare_woa_vs_ex_po", False),
    ]
    for metric, ylabel, fname, log_y in plot_specs_type1:
        plot_metric(
            sorted_xs,
            unified_data,
            metric,
            x_label,
            ylabel,
            out_dir / f"{fname}.pdf",
            out_dir / f"{fname}.png",
            log_y=log_y,
            filter_algos=woa_vs_ex_algos,
            algo_label_map={"WOA-BWOA": "BWOA"},
            **style_kwargs,
        )

    # TYPE 2: 3 Approaches (PGD-BWOA, WOA-BWOA, and EX)
    print("\n--- Type 2: 3 Approaches (PGD-BWOA, WOA-BWOA, and EX) ---")
    three_algos = ["PGD-BWOA", "WOA-BWOA", "EX"]
    plot_specs_type2 = [
        ("utility", "System Utility", "compare_3approaches_su", False),
        ("runtime", "Execution Runtime [s]", "compare_3approaches_time", True),
        ("runtime", "Execution Runtime [s]", "compare_3approaches_time_linear", False),
        ("offload_ratio", "Offloading Percentage", "compare_3approaches_po", False),
        # Also save standard aliases matching run_job.py / compare_results.py convention
        ("utility", "System Utility", "compare_with_ex_su", False),
        ("runtime", "Execution Runtime [s]", "compare_with_ex_time", True),
    ]
    for metric, ylabel, fname, log_y in plot_specs_type2:
        plot_metric(
            sorted_xs,
            unified_data,
            metric,
            x_label,
            ylabel,
            out_dir / f"{fname}.pdf",
            out_dir / f"{fname}.png",
            log_y=log_y,
            filter_algos=three_algos,
            **style_kwargs,
        )

    # OPTIONAL: Optimality Gap Plots
    if "gap" in unified_data.get("WOA-BWOA", {}):
        print("\n--- Optimality Gap Figures ---")
        plot_metric(
            sorted_xs,
            unified_data,
            "gap",
            x_label,
            "Optimality Gap [%] vs. EX",
            out_dir / "compare_woa_vs_ex_gap.pdf",
            out_dir / "compare_woa_vs_ex_gap.png",
            log_y=False,
            filter_algos=["WOA-BWOA"],
            algo_label_map={"WOA-BWOA": "BWOA"},
            **style_kwargs,
        )
        plot_metric(
            sorted_xs,
            unified_data,
            "gap",
            x_label,
            "Optimality Gap [%] vs. EX",
            out_dir / "compare_3approaches_gap.pdf",
            out_dir / "compare_3approaches_gap.png",
            log_y=False,
            filter_algos=["PGD-BWOA", "WOA-BWOA"],
            **style_kwargs,
        )

    print(f"\n====================================================================")
    print(f"All synthesized files successfully written to:\n  {out_dir.resolve()}")
    print(f"====================================================================")


if __name__ == "__main__":
    main()
