"""Shared CLI plumbing for the simulation scripts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stochastic_mec import AlgorithmParams  # noqa: E402

RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"


def base_parser(description: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("-r", "--realizations", type=int, default=20,
                    help="number of HPPP time blocks averaged per point")
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--quick", action="store_true",
                    help="small populations / few iterations, for a fast smoke run")
    ap.add_argument("-j", "--jobs", type=int, default=1,
                    help="run the Monte-Carlo realisations on this many processes")
    ap.add_argument("--out", type=Path, default=None, help="output directory")
    return ap


def algorithm_params(args: argparse.Namespace) -> AlgorithmParams:
    if getattr(args, "algo", None) is not None:
        return args.algo
    if args.quick:
        return AlgorithmParams(
            n_agents_bwoa=10, max_iter_bwoa=20, patience_bwoa=6,
            n_agents_tpc=10, max_iter_tpc=40, patience_tpc=6,
            pop_min=8, pop_max=16,
        )
    return AlgorithmParams()


def load_plotting():
    try:
        from stochastic_mec import plotting
    except ImportError as exc:
        print(f"[figures skipped] {exc}. Install matplotlib to render the plots.")
        return None
    return plotting


def outputs(args: argparse.Namespace, name: str) -> tuple[Path, Path]:
    root = args.out or ROOT
    res = (root / "results") if args.out is None else root
    fig = (root / "figures") if args.out is None else root
    res.mkdir(parents=True, exist_ok=True)
    fig.mkdir(parents=True, exist_ok=True)
    return res / f"{name}.json", fig / f"{name}.pdf"
