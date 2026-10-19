#!/usr/bin/env python3
"""Convergence simulation with Pure Memoization (3 UL, 3 DL, 10 UEs) and decoupled swarms."""

from __future__ import annotations

import sys
from pathlib import Path

# Pure relative path: find simulation_v4 by traversing upwards (no git needed)
SIM_DIR = None
for p in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]:
    if (p / "stochastic_mec").exists():
        SIM_DIR = p
        break
if SIM_DIR is None:
    SIM_DIR = Path(__file__).resolve().parents[3]

SCRIPTS_DIR = SIM_DIR / "scripts"
if str(SIM_DIR) not in sys.path:
    sys.path.insert(0, str(SIM_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from _common import base_parser  # noqa: E402
from fig_convergence import run_convergence  # noqa: E402
from stochastic_mec import AlgorithmParams  # noqa: E402

# ==============================================================================
# CONFIGURATION
# ==============================================================================
UES = 10                            # Number of active UEs
UL_CELLS = 3                       # Number of uplink UAV cells
DL_CELLS = 3                       # Number of downlink UAV cells
SEED = 2025                        # Random seed for topology generation
MAX_ITER = 500                     # Maximum iterations for BWOA (TPC defaults to 60)
N_AGENTS = 30                      # Number of agents for BWOA (TPC defaults to 15)
N_AGENTS_TPC = 15                  # Number of agents for inner continuous TPC
MAX_ITER_TPC = 60                  # Maximum iterations for inner continuous TPC
CACHE_MAX_RETRIES = 0              # 0 = Pure Memoization (no perturbation flips)
JOBS = 4
# ==============================================================================


def _next_result_dir(parent_dir: Path, prefix: str = "result_") -> Path:
    """Find the next available result folder: result_1, result_2, result_3, ..."""
    idx = 1
    while (parent_dir / f"{prefix}{idx}").exists():
        idx += 1
    out_dir = parent_dir / f"{prefix}{idx}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def main() -> None:
    ap = base_parser(__doc__)
    jobs = globals().get("JOBS", 1)
    ap.set_defaults(seed=SEED, jobs=jobs)
    ap.add_argument(
        "--ues",
        type=int,
        default=UES,
        help="Number of active UEs (default: 10)",
    )
    ap.add_argument(
        "--ul-cells",
        type=int,
        default=UL_CELLS,
        help="Number of uplink UAV cells (default: 3)",
    )
    ap.add_argument(
        "--dl-cells",
        type=int,
        default=DL_CELLS,
        help="Number of downlink UAV cells (default: 3)",
    )
    args = ap.parse_args()

    # Apply custom MAX_ITER and N_AGENTS if defined; otherwise falls back to defaults
    max_iter_bwoa = globals().get("MAX_ITER_BWOA", globals().get("MAX_ITER", None))
    max_iter_tpc = globals().get("MAX_ITER_TPC", 60)
    n_agents_bwoa = globals().get("N_AGENTS_BWOA", globals().get("N_AGENTS", None))
    n_agents_tpc = globals().get("N_AGENTS_TPC", 15)
    cache_retries = globals().get("CACHE_MAX_RETRIES", 0)

    algo_kwargs = {"cache_max_retries": cache_retries}
    if max_iter_bwoa is not None:
        algo_kwargs["max_iter_bwoa"] = max_iter_bwoa
    if max_iter_tpc is not None:
        algo_kwargs["max_iter_tpc"] = max_iter_tpc
    if n_agents_bwoa is not None:
        algo_kwargs["n_agents_bwoa"] = n_agents_bwoa
    if n_agents_tpc is not None:
        algo_kwargs["n_agents_tpc"] = n_agents_tpc

    if not args.quick and algo_kwargs:
        args.algo = AlgorithmParams(**algo_kwargs)

    # Automatically find the next result_x folder right next to this run_job.py file
    script_dir = Path(__file__).resolve().parent
    if args.out is None:
        args.out = _next_result_dir(script_dir)
    else:
        args.out.mkdir(parents=True, exist_ok=True)

    algo = args.algo if getattr(args, "algo", None) is not None else AlgorithmParams()
    print(f"Topology: {args.ues} UEs, {args.ul_cells} UL UAVs, {args.dl_cells} DL UAVs", flush=True)
    print(f"Random seed: {args.seed}", flush=True)
    print(f"Max iterations: BWOA={algo.max_iter_bwoa}, TPC={algo.max_iter_tpc}", flush=True)
    print(f"Number of agents: BWOA={algo.n_agents_bwoa}, TPC={algo.n_agents_tpc}", flush=True)
    print(f"Cache mode: Pure Memoization (cache_max_retries={algo.cache_max_retries})", flush=True)
    print(f"Parallel jobs: {args.jobs}", flush=True)
    print(f"Results will be saved to: {args.out}", flush=True)
    run_convergence(args)


if __name__ == "__main__":
    main()
