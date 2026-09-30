#!/usr/bin/env python3
"""UL-UAV Density sweep (1 to 4 with 1 DL UAV and 20 UEs)."""

from __future__ import annotations

import sys
from pathlib import Path

# Pure relative path: find simulation_v4 by traversing upwards (no git needed)
SIM_DIR = None
for p in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]:
    if p.name == "simulation_v4":
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
from run_sweep import sweep_uav_density  # noqa: E402
from stochastic_mec import AlgorithmParams  # noqa: E402

# ==============================================================================
# CONFIGURATION
# ==============================================================================
XS = list(range(1, 5, 1))          # UL-UAV density: [1, 2, 3, 4] in 1e-6/m^2
REALIZATIONS = 100                 # Number of Monte-Carlo realizations
SEED = 2025                        # Random seed
MAX_ITER = 200                     # Maximum iterations for BWOA (default: 120; TPC defaults to 60)
N_AGENTS = 30                      # Number of agents for BWOA (default: 30; TPC defaults to 15)
N_DL = 1                           # Number of DL UAVs (default: 3)
N_UE = 20                          # Number of active UEs (default: 10)
JOBS = 4                           # Number of parallel workers (default: 4)
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
    jobs = globals().get("JOBS", 4)
    ap.set_defaults(realizations=REALIZATIONS, seed=SEED, jobs=jobs)
    ap.add_argument(
        "--xs",
        type=float,
        nargs="+",
        default=XS,
        help="UL-UAV density values in 1e-6/m^2",
    )
    ap.add_argument(
        "--n-dl",
        type=int,
        default=globals().get("N_DL", 3) or 3,
        help="Number of DL UAVs (default: 3 if not set)",
    )
    ap.add_argument(
        "--n-ue",
        type=int,
        default=globals().get("N_UE", 10) or 10,
        help="Number of active UEs (default: 10)",
    )
    args = ap.parse_args()

    # Apply custom MAX_ITER and N_AGENTS if defined; otherwise falls back to defaults
    max_iter_bwoa = globals().get("MAX_ITER_BWOA", globals().get("MAX_ITER", None))
    max_iter_tpc = globals().get("MAX_ITER_TPC", None)
    n_agents_bwoa = globals().get("N_AGENTS_BWOA", globals().get("N_AGENTS", None))
    n_agents_tpc = globals().get("N_AGENTS_TPC", None)

    algo_kwargs = {}
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
    print(f"Sweep points (UL UAVs): {args.xs}", flush=True)
    print(f"DL UAVs: {args.n_dl}, Active UEs: {args.n_ue}", flush=True)
    print(f"Max iterations: BWOA={algo.max_iter_bwoa}, TPC={algo.max_iter_tpc}", flush=True)
    print(f"Number of agents: BWOA={algo.n_agents_bwoa}, TPC={algo.n_agents_tpc}", flush=True)
    print(f"Parallel jobs: {args.jobs}", flush=True)
    print(f"Results will be saved to: {args.out}", flush=True)
    sweep_uav_density(args)


if __name__ == "__main__":
    main()

