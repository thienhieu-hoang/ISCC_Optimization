#!/usr/bin/env python3
"""Convergence simulation (BWOA and inner WOA / IWOA / PSO) reproducing Fig. 3."""

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
    SIM_DIR = Path(__file__).resolve().parents[2]

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
MAX_ITER = 500                     # Maximum iterations for BWOA and TPC (default: 120)
N_AGENTS = 50                      # Number of agents for BWOA and TPC (default: 30)
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
    ap.set_defaults(seed=SEED)
    ap.add_argument(
        "--ues",
        type=int,
        default=UES,
        help="Number of active UEs (default: 6)",
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

    # Apply custom MAX_ITER and N_AGENTS if defined; otherwise falls back to defaults (120 and 30)
    max_iter = globals().get("MAX_ITER", None)
    n_agents = globals().get("N_AGENTS", None)
    algo_kwargs = {}
    if max_iter is not None:
        algo_kwargs.update(max_iter_bwoa=max_iter, max_iter_tpc=max_iter)
    if n_agents is not None:
        algo_kwargs.update(n_agents_bwoa=n_agents, n_agents_tpc=n_agents)

    if not args.quick and algo_kwargs:
        args.algo = AlgorithmParams(**algo_kwargs)

    # Automatically find the next result_x folder right next to this run_job.py file
    script_dir = Path(__file__).resolve().parent
    if args.out is None:
        args.out = _next_result_dir(script_dir)
    else:
        args.out.mkdir(parents=True, exist_ok=True)

    print(f"Topology: {args.ues} UEs, {args.ul_cells} UL UAVs, {args.dl_cells} DL UAVs")
    print(f"Random seed: {args.seed}")
    print(f"Max iterations: {max_iter if (not args.quick and max_iter) else ('quick mode (20/40)' if args.quick else 'default (120)')}")
    print(f"Number of agents: {n_agents if (not args.quick and n_agents) else ('quick mode (10)' if args.quick else 'default (30)')}")
    print(f"Results will be saved to: {args.out}")
    run_convergence(args)


if __name__ == "__main__":
    main()
