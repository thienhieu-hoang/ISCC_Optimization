#!/usr/bin/env python3
"""Reproduce every figure of the paper in one go.

    python run_all.py --quick -r 3          # ~2 minutes, smoke run
    python run_all.py -r 30 -j 8            # full campaign
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from _common import base_parser

HERE = Path(__file__).resolve().parent

JOBS = [
    ("Fig. 2  network snapshot", ["fig_topology.py"]),
    ("Fig. 3  convergence", ["fig_convergence.py"]),
    ("Fig. 4  BWOA vs exhaustive search", ["compare_exhaustive.py"]),
    ("Fig. 5  UE density", ["run_sweep.py", "ue-density"]),
    ("Fig. 6  UL-UAV density", ["run_sweep.py", "uav-density"]),
    ("Fig. 6b jammer density", ["run_sweep.py", "jammer-density"]),
    ("Fig. 7  input data size", ["run_sweep.py", "datasize"]),
    ("Fig. 8  task load", ["run_sweep.py", "taskload"]),
    ("Fig. 9  server capacity", ["run_sweep.py", "capacity"]),
    ("Fig. 10 UE power budget", ["run_sweep.py", "power"]),
    ("Fig. 11 time/energy preference", ["run_sweep.py", "preference"]),
    ("ISCC   accuracy-delay frontier", ["run_sweep.py", "accuracy-tradeoff"]),
    ("ISCC   retention vs sensing SNR", ["run_sweep.py", "sensing-snr"]),
]


def main() -> None:
    ap = base_parser(__doc__)
    ap.add_argument("--only", nargs="*", default=None,
                    help="run only the scripts whose name contains one of these strings")
    args = ap.parse_args()

    passthrough = ["-r", str(args.realizations), "--seed", str(args.seed), "-j", str(args.jobs)]
    if args.quick:
        passthrough.append("--quick")
    if args.out:
        passthrough += ["--out", str(args.out)]

    for title, cmd in JOBS:
        if args.only and not any(tok in " ".join(cmd) for tok in args.only):
            continue
        print(f"\n=== {title} " + "=" * (60 - len(title)))
        start = time.perf_counter()
        rc = subprocess.call([sys.executable, str(HERE / cmd[0]), *cmd[1:], *passthrough])
        print(f"--- {title}: {'ok' if rc == 0 else f'FAILED ({rc})'} "
              f"in {time.perf_counter() - start:.1f} s")


if __name__ == "__main__":
    main()
