"""Post-processing of one solved time block into the figures' quantities."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .system import Solution, SystemModel


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


def _mean(x: np.ndarray) -> float:
    return float(np.mean(x)) if x.size else float("nan")


def evaluate_solution(model: SystemModel, sol: Solution) -> BlockMetrics:
    ul_sub, dl_sub = model.decode(sol.assoc)
    offloading = np.flatnonzero(ul_sub >= 0)

    if model.n_ul:
        rates, rho, chi, f_alloc = model.ul_state(sol.assoc, sol.ul_power, sol.dl_power)
        t, e = model.per_ue_time_energy(sol.assoc, sol.ul_power, sol.dl_power)
        ul_util = float(np.sum(model.ul_utilities(rho, chi, rates, f_alloc, sol.ul_power)))
        adm = model.admissible
        acc = model.accuracy(chi)
    else:
        t = e = rho = chi = acc = np.zeros(0)
        adm = np.zeros(0, dtype=bool)
        ul_util = 0.0

    dl_util = model.dl_utility(dl_sub, sol.dl_power, ul_sub, sol.ul_power)

    return BlockMetrics(
        utility=ul_util + dl_util,
        offload_ratio=offloading.size / model.n_ul if model.n_ul else 0.0,
        total_delay=float(t.sum()),
        total_energy=float(e.sum()),
        ul_utility=ul_util,
        dl_utility=dl_util,
        sum_dl_rate=dl_util * model.p.rate_scaling,
        runtime=sol.runtime,
        n_offloading=int(offloading.size),
        n_ul_ues=model.n_ul,
        n_dl_ues=model.n_dl,
        mean_accuracy=_mean(acc[adm]),
        mean_retention=_mean(chi[adm]),
        mean_split=_mean(rho[offloading]) if offloading.size else float("nan"),
        mean_delay=_mean(t[adm]),
        mean_norm_delay=_mean(t[adm] / model.t_ref[adm]) if model.n_ul else float("nan"),
        admissible_ratio=float(np.mean(adm)) if adm.size else float("nan"),
    )


def average(metrics: list[BlockMetrics]) -> dict[str, float]:
    """Mean over realisations -- the Monte-Carlo estimate of Upsilon_sys.

    ISCC averages ignore realisations where the quantity is undefined (no
    admissible or no offloading UE), hence ``nanmean``.
    """
    if not metrics:
        return {}
    keys = metrics[0].as_dict().keys()
    out = {}
    for k in keys:
        vals = np.asarray([getattr(mm, k) for mm in metrics], dtype=float)
        out[k] = float(np.nanmean(vals)) if np.any(np.isfinite(vals)) else float("nan")
    return out
