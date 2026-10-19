"""Post-processing of solved time blocks."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class BlockMetrics:
    utility: float                # U   (system utility)
    offload_ratio: float          # fraction of active UL UEs that offloaded
    total_delay: float            # sum_n T_n  [s]
    total_energy: float           # sum_n E_n  [J]
    ul_utility: float             # sum_n U^ul_n
    dl_utility: float             # sum_n' U^dl_n'
    sum_dl_rate: float            # sum_n' R^dl_n'  [bit/s]
    runtime: float                # wall-clock time of the solver [s]
    n_offloading: int
    n_ul_ues: int
    n_dl_ues: int
    # ---- ISCC (Sec. II-C) -------------------------------------------------
    mean_accuracy: float          # mean Lambda_n over admissible UL UEs
    mean_retention: float         # mean chi_n over admissible UL UEs
    mean_split: float             # mean rho_n over offloading UEs
    mean_delay: float             # mean T_n over admissible UL UEs [s]
    mean_norm_delay: float        # mean T_n / T_n^ref over admissible UL UEs
    admissible_ratio: float       # fraction of UL UEs with chi_n^min <= 1

    def as_dict(self) -> dict:
        return asdict(self)


def average(metrics: list[BlockMetrics]) -> dict[str, float]:
    if not metrics:
        return {}
    keys = metrics[0].as_dict().keys()
    out = {}
    for k in keys:
        vals = np.asarray([getattr(mm, k) for mm in metrics], dtype=float)
        out[k] = float(np.nanmean(vals)) if np.any(np.isfinite(vals)) else float("nan")
    return out
