#!/usr/bin/env python3
"""Compare and plot results from independent BWOA and Exhaustive search runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Pure relative path: find simulation_v4 by traversing upwards (no git needed)
SIM_DIR = None
for p in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]:
    if p.name == "simulation_v4":
        SIM_DIR = p
        break
if SIM_DIR is None:
    SIM_DIR = Path(__file__).resolve().parents[2]

SCRIPTS_DIR = SIM_DIR / "scripts"
if str(SIM_DIR) not in sys.path:
    sys.path.insert(0, str(SIM_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import numpy as np
from _common import load_plotting
from stochastic_mec import SweepResult


def _find_latest_result_dir(parent_dir: Path) -> Path | None:
    if not parent_dir.exists():
        return None
    res_dirs = [d for d in parent_dir.iterdir() if d.is_dir() and d.name.startswith("result_")]
    if not res_dirs:
        return None
    res_dirs.sort(key=lambda d: int(d.name.split("_")[1]) if d.name.split("_")[1].isdigit() else 0)
    return res_dirs[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bwoa-dir",
        type=Path,
        default=None,
        help="Path to BWOA result directory (default: latest in BWOA_3subc_2UL1DL_2to4UE)",
    )
    parser.add_argument(
        "--ex-dir",
        type=Path,
        default=None,
        help="Path to Exhaustive search result directory (default: latest in exhaustive_3sub_2UL1DL_2to4UE)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Directory to save combined comparison plots (default: current directory)",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    bwoa_dir = args.bwoa_dir or _find_latest_result_dir(base_dir / "BWOA_3subc_2UL1DL_2to4UE")
    ex_dir = args.ex_dir or _find_latest_result_dir(base_dir / "exhaustive_3sub_2UL1DL_2to4UE")

    if bwoa_dir is None or not bwoa_dir.exists():
        print(f"[Error] BWOA result directory not found: {bwoa_dir}")
        sys.exit(1)
    if ex_dir is None or not ex_dir.exists():
        print(f"[Error] Exhaustive result directory not found: {ex_dir}")
        sys.exit(1)

    print(f"Comparing BWOA results: {bwoa_dir}")
    print(f"Against EX results:     {ex_dir}\n")

    bwoa_summary_file = bwoa_dir / "summary.json"
    ex_summary_file = ex_dir / "summary.json"

    if not bwoa_summary_file.exists() or not ex_summary_file.exists():
        print("[Error] Missing summary.json in one or both directories.")
        sys.exit(1)

    bwoa_data = json.loads(bwoa_summary_file.read_text())
    ex_data = json.loads(ex_summary_file.read_text())

    bwoa_records = {(r["density"], r["realization"]): r for r in bwoa_data.get("records", [])}
    ex_records = {(r["density"], r["realization"]): r for r in ex_data.get("records", [])}

    densities = sorted(list({k[0] for k in bwoa_records.keys() | ex_records.keys()}))

    print(f"{'Density':>7} | {'R':>3} | {'Seed':>6} | {'Topo-ID':>7} | {'Chan-ID':>7} | {'BWOA Utility':>12} | {'EX Utility':>12} | {'Gap (%)':>8} | {'BWOA (s)':>8} | {'EX (s)':>8}")
    print("-" * 105)

    gaps_per_density: dict[int, list[float]] = {d: [] for d in densities}
    bwoa_util_per_density: dict[int, list[float]] = {d: [] for d in densities}
    ex_util_per_density: dict[int, list[float]] = {d: [] for d in densities}
    bwoa_time_per_density: dict[int, list[float]] = {d: [] for d in densities}
    ex_time_per_density: dict[int, list[float]] = {d: [] for d in densities}

    for d in densities:
        for r_idx in range(1, max(bwoa_data.get("realizations", 5), ex_data.get("realizations", 5)) + 1):
            key = (d, r_idx)
            if key not in bwoa_records or key not in ex_records:
                continue
            rb = bwoa_records[key]
            re = ex_records[key]

            # Verification of topology and channel match
            matched = (rb.get("topo_id") == re.get("topo_id")) and (rb.get("chan_id") == re.get("chan_id"))
            match_mark = "" if matched else " [MISMATCH!]"

            ub, ue = rb["utility"], re["utility"]
            tb, te = rb["runtime_s"], re["runtime_s"]
            denom = abs(ue) if abs(ue) > 1e-9 else 1.0
            gap = 100.0 * (ue - ub) / denom

            gaps_per_density[d].append(gap)
            bwoa_util_per_density[d].append(ub)
            ex_util_per_density[d].append(ue)
            bwoa_time_per_density[d].append(tb)
            ex_time_per_density[d].append(te)

            print(f"{d:>7} | {r_idx:>3} | {rb['seed']:>6} | {rb.get('topo_id', '-'):>7} | {rb.get('chan_id', '-'):>7} | "
                  f"{ub:>+12.4f} | {ue:>+12.4f} | {gap:>+7.2f}% | {tb:>7.2f}s | {te:>7.2f}s{match_mark}")

    print("-" * 105)
    print("\nSummary per Density:")
    print(f"{'Density':>7} | {'Mean BWOA U':>12} | {'Mean EX U':>12} | {'Mean Gap (%)':>12} | {'BWOA Mean (s)':>14} | {'EX Mean (s)':>14}")
    print("-" * 85)
    for d in densities:
        mean_b = np.mean(bwoa_util_per_density[d]) if bwoa_util_per_density[d] else np.nan
        mean_e = np.mean(ex_util_per_density[d]) if ex_util_per_density[d] else np.nan
        mean_gap = np.mean(gaps_per_density[d]) if gaps_per_density[d] else np.nan
        mean_tb = np.mean(bwoa_time_per_density[d]) if bwoa_time_per_density[d] else np.nan
        mean_te = np.mean(ex_time_per_density[d]) if ex_time_per_density[d] else np.nan
        print(f"{d:>7} | {mean_b:>+12.4f} | {mean_e:>+12.4f} | {mean_gap:>+11.2f}% | {mean_tb:>13.2f}s | {mean_te:>13.2f}s")
    print("-" * 85)

    out_dir = args.out or base_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Plot comparison
    plotting = load_plotting()
    if plotting is not None:
        plot_result = SweepResult(
            r"Active-UE Density [$\times 10^{-6}/m^2$]",
            [float(d) for d in densities],
            {
                "BWOA": {
                    "utility": [float(np.mean(bwoa_util_per_density[d])) for d in densities],
                    "runtime": [float(np.mean(bwoa_time_per_density[d])) for d in densities],
                },
                "EX": {
                    "utility": [float(np.mean(ex_util_per_density[d])) for d in densities],
                    "runtime": [float(np.mean(ex_time_per_density[d])) for d in densities],
                },
            },
        )
        f1 = out_dir / "compare_with_ex_su.png"
        f2 = out_dir / "compare_with_ex_time.png"
        plotting.plot_sweep(plot_result, "utility", "System Utility", f1)
        plotting.plot_sweep(plot_result, "runtime", "Algorithm Runtime [s]", f2)
        print(f"\nSaved combined comparison plots to:\n  {f1}\n  {f2}")


if __name__ == "__main__":
    main()
