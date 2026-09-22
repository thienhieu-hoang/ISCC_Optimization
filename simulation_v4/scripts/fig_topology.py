#!/usr/bin/env python3
"""Fig. 2 -- a snapshot of the simulated stochastic MEC network."""

from __future__ import annotations

import numpy as np

from _common import FIGURES, base_parser, load_plotting
from stochastic_mec import SystemParams, sample_topology


def main() -> None:
    ap = base_parser(__doc__)
    ap.add_argument("--sbs", "--uavs", type=int, default=6)
    ap.add_argument("--ues", type=int, default=10)
    ap.add_argument("--jammers", type=int, default=3)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    params = SystemParams()
    for _ in range(200):
        topo = sample_topology(
            rng, params, n_sbs=args.sbs, n_active_ue=args.ues,
            n_inactive_ue=4, n_jammers=args.jammers,
        )
        if topo.ul_cells.size and topo.dl_cells.size:
            break

    print(f"UL UAVs: {topo.ul_cells.tolist()}   DL UAVs: {topo.dl_cells.tolist()}   "
          f"null: {topo.null_cells.tolist()}   jammers: {topo.n_jammer}")
    plotting = load_plotting()
    if plotting is not None:
        out = (args.out or FIGURES) / "topology_snapshot.png"
        plotting.plot_topology(topo, out)
        print(f"saved {out}")


if __name__ == "__main__":
    main()
