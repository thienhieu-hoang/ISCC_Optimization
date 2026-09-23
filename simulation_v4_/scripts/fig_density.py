#!/usr/bin/env python3
"""Fig. 3 -- UE- and UAV-density sweeps, re-plotted from cached results."""

from __future__ import annotations

import json

from _common import FIGURES, base_parser
from stochastic_mec import SweepResult, plotting

RESULTS = FIGURES.parent / "results"
RENAME = {"ARJOA": "force-all-offload"}


def main() -> None:
    out = base_parser(__doc__).parse_args().out or FIGURES
    for src, name in [("ue_density.json", "ue_density_su.pdf"),
                      ("uav_density.json", "uav_density_su.pdf")]:
        fpath = RESULTS / src
        if not fpath.exists():
            continue
        d = json.loads(fpath.read_text())
        data = {RENAME.get(k, k): v for k, v in d["data"].items()}
        saved = plotting.plot_sweep(SweepResult(d["x_label"], d["x"], data), "utility",
                                    "System utility", out / name)
        print("saved", saved, "series:", list(data))


if __name__ == "__main__":
    main()
