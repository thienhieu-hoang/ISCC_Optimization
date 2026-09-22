"""Monte-Carlo drivers shared by the simulation scripts.

The comprehensive system utility of the paper,

    Upsilon_sys = E_t[ Upsilon_t ],

is estimated by averaging the per-time-block optimum over ``n_realizations``
independent HPPP snapshots.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

from .config import AlgorithmParams, SystemParams
from .metrics import BlockMetrics, average, evaluate_solution
from .network import Topology, sample_topology, sample_topology_with_cells
from .schemes import SCHEMES
from .solver import HybridSolver
from .system import SystemModel

# Topology factories are small picklable objects (not closures) so that the
# Monte-Carlo loop can be spread over several processes.
@dataclass(frozen=True)
class FixedTopology:
    """Controlled number of UL/DL cells and active UEs (used by most sweeps)."""

    n_ul_cells: int = 3
    n_dl_cells: int = 3
    n_active_ue: int | None = None

    def __call__(self, rng: np.random.Generator, params: SystemParams) -> Topology:
        n_ue = self.n_active_ue
        if n_ue is None:
            n_ue = max(1, int(round(params.lambda_ue_active * params.area)))
        return sample_topology_with_cells(rng, params, self.n_ul_cells, self.n_dl_cells, n_ue)


@dataclass(frozen=True)
class PoissonTopology:
    """Fully stochastic: every count is drawn from its own Poisson law."""

    def __call__(self, rng: np.random.Generator, params: SystemParams) -> Topology:
        return sample_topology(rng, params)


TopologyFactory = Callable[[np.random.Generator, SystemParams], Topology]


def fixed_topology(
    n_ul_cells: int = 3, n_dl_cells: int = 3, n_active_ue: int | None = None
) -> TopologyFactory:
    return FixedTopology(n_ul_cells, n_dl_cells, n_active_ue)


def poisson_topology() -> TopologyFactory:
    return PoissonTopology()


def simulate_block(
    seed: int,
    params: SystemParams,
    algo: AlgorithmParams,
    tpc: str,
    scheme: str,
    topology: TopologyFactory,
) -> tuple[BlockMetrics, np.ndarray]:
    """Optimise one time block; returns its metrics and the BWOA curve."""
    rng = np.random.default_rng(seed)
    topo = topology(rng, params)
    model = SystemModel(topo, params, rng)
    solver = HybridSolver(model, rng, algo=algo, tpc=tpc, scheme=scheme)
    sol = solver.solve()
    return evaluate_solution(model, sol), sol.curve


def _worker(job) -> BlockMetrics:
    return simulate_block(*job)[0]


def monte_carlo(
    n_realizations: int,
    base_seed: int,
    params: SystemParams,
    algo: AlgorithmParams,
    tpc: str,
    scheme: str,
    topology: TopologyFactory,
    verbose: bool = False,
    jobs: int = 1,
) -> dict[str, float]:
    args = [(base_seed + r, params, algo, tpc, scheme, topology) for r in range(n_realizations)]
    if jobs and jobs > 1:
        with ProcessPoolExecutor(max_workers=min(jobs, os.cpu_count() or 1)) as pool:
            out = list(pool.map(_worker, args))
    else:
        out = []
        for r, job in enumerate(args):
            met = _worker(job)
            out.append(met)
            if verbose:
                print(f"    realisation {r + 1}/{n_realizations}: U = {met.utility:+.4f}")
    return average(out)


@dataclass
class SweepResult:
    """Tidy container: ``values[curve_label][metric] -> list over x``."""

    x_label: str
    x: list[float]
    data: dict[str, dict[str, list[float]]]

    def add(self, label: str, metrics: dict[str, float]) -> None:
        slot = self.data.setdefault(label, {})
        for k, v in metrics.items():
            slot.setdefault(k, []).append(v)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"x_label": self.x_label, "x": self.x, "data": self.data}, indent=2))
        return path

    @staticmethod
    def load(path: str | Path) -> "SweepResult":
        raw = json.loads(Path(path).read_text())
        return SweepResult(raw["x_label"], raw["x"], raw["data"])


def run_sweep(
    x_label: str,
    x_values: Sequence[float],
    make_params: Callable[[float], SystemParams],
    curves: Iterable[tuple[str, str, str]],
    *,
    n_realizations: int = 20,
    base_seed: int = 2025,
    algo: AlgorithmParams | None = None,
    topology: TopologyFactory | None = None,
    make_topology: Callable[[float], TopologyFactory] | None = None,
    verbose: bool = True,
    jobs: int = 1,
) -> SweepResult:
    """Run every ``(label, tpc, scheme)`` curve over ``x_values``.

    ``make_params(x)`` returns the system parameters for abscissa ``x``;
    ``make_topology(x)`` (optional) lets the topology depend on ``x`` too.
    """
    algo = algo or AlgorithmParams()
    curves = list(curves)
    result = SweepResult(x_label, [float(v) for v in x_values], {})

    for xi, x in enumerate(x_values):
        params = make_params(x)
        topo = make_topology(x) if make_topology is not None else (topology or fixed_topology())
        if verbose:
            print(f"[{x_label} = {x}]")
        for label, tpc, scheme in curves:
            if verbose:
                print(f"  {label}")
            metrics = monte_carlo(
                n_realizations,
                base_seed + 1000 * xi,          # same topologies across curves
                params,
                algo,
                tpc,
                scheme,
                topo,
                jobs=jobs,
            )
            result.add(label, metrics)
            if verbose:
                print(f"    U = {metrics['utility']:+.4f}   "
                      f"offloaded = {100 * metrics['offload_ratio']:.1f} %   "
                      f"t = {metrics['runtime']:.2f} s")
    return result


DEFAULT_CURVES = [
    ("WOA-BWOA", "WOA", "MF-SIC"),
    ("IWOA-BWOA", "IWOA", "MF-SIC"),
    ("PSO-BWOA", "PSO", "MF-SIC"),
    ("ARJOA", "WOA", "ARJOA"),
    ("IOJOA", "WOA", "IOJOA"),
    ("FDMA", "WOA", "FDMA"),
    ("ALCA", "WOA", "ALCA"),
]

__all__ = [
    "FixedTopology",
    "PoissonTopology",
    "SweepResult",
    "SCHEMES",
    "DEFAULT_CURVES",
    "fixed_topology",
    "poisson_topology",
    "simulate_block",
    "monte_carlo",
    "run_sweep",
]
