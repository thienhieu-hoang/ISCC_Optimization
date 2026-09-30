"""Monte-Carlo drivers for the TensorFlow parallel simulation engine."""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import numpy as np
import tensorflow as tf

from .config import AlgorithmParams, SystemParams
from .metrics import BlockMetrics, average
from .network import Topology, sample_topology, sample_topology_with_cells
from .schemes import SCHEMES
from .solver_tf import HybridSolverTF
from .system_tf import SolutionTF, SystemModelTF


@dataclass(frozen=True)
class FixedTopology:
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
    def __call__(self, rng: np.random.Generator, params: SystemParams) -> Topology:
        return sample_topology(rng, params)


TopologyFactory = Callable[[np.random.Generator, SystemParams], Topology]


def fixed_topology(
    n_ul_cells: int = 3, n_dl_cells: int = 3, n_active_ue: int | None = None
) -> TopologyFactory:
    return FixedTopology(n_ul_cells, n_dl_cells, n_active_ue)


def poisson_topology() -> TopologyFactory:
    return PoissonTopology()


def evaluate_solution_tf(model: SystemModelTF, sol: SolutionTF) -> BlockMetrics:
    assoc_tf = tf.constant(sol.assoc, dtype=tf.float32)
    p_ul_tf = tf.constant(sol.ul_power, dtype=tf.float32)
    q_dl_tf = tf.constant(sol.dl_power, dtype=tf.float32)

    ul_sub, dl_sub = model.decode_tf(assoc_tf)
    offloading = np.flatnonzero(ul_sub.numpy() >= 0)

    if model.n_ul:
        xi = model.cochannel_at_sbs_tf(dl_sub, q_dl_tf)
        rates = model.ul_rates_tf(ul_sub, p_ul_tf, xi, dl_sub)
        rho, chi, f_alloc = model.iscc_allocate_tf(ul_sub, rates, p_ul_tf)
        delta, theta = model._branches_tf(rho, rates, f_alloc, p_ul_tf)
        
        t = (model.t_sen_np + model.t_ext_np + chi.numpy() * delta.numpy())
        e = (model.e_sen_np + model.e_ext_np + chi.numpy() * theta.numpy())
        
        ul_util_per_ue = model.ul_utilities_tf(rho, chi, rates, f_alloc, p_ul_tf)
        ul_util = float(tf.reduce_sum(ul_util_per_ue))
        adm = model.admissible_np
        acc = model.accuracy_tf(chi).numpy()
        rho_np = rho.numpy()
        chi_np = chi.numpy()
    else:
        t = e = rho_np = chi_np = acc = np.zeros(0)
        adm = np.zeros(0, dtype=bool)
        ul_util = 0.0

    dl_gamma = model.dl_sinr_tf(dl_sub, q_dl_tf, ul_sub, p_ul_tf)
    dl_rates = float(model.p.subchannel_bw) * tf.experimental.numpy.log2(1.0 + dl_gamma)
    dl_util = float(tf.reduce_sum(dl_rates) / float(model.p.rate_scaling))

    def _mean(x: np.ndarray) -> float:
        return float(np.mean(x)) if x.size else float("nan")

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
        mean_retention=_mean(chi_np[adm]),
        mean_split=_mean(rho_np[offloading]) if offloading.size else float("nan"),
        mean_delay=_mean(t[adm]),
        mean_norm_delay=_mean(t[adm] / model.t_ref_np[adm]) if model.n_ul else float("nan"),
        admissible_ratio=float(np.mean(adm)) if adm.size else float("nan"),
    )


def simulate_block_tf(
    seed: int,
    params: SystemParams,
    algo: AlgorithmParams,
    tpc: str,
    scheme: str,
    topology: TopologyFactory | Topology,
) -> tuple[BlockMetrics, np.ndarray]:
    rng = np.random.default_rng(seed)
    topo = topology(rng, params) if callable(topology) else topology
    channel_rng = np.random.default_rng(seed)
    model = SystemModelTF(topo, params, channel_rng, scheme=scheme)
    solver_rng = np.random.default_rng(seed + 10007)
    solver = HybridSolverTF(model, solver_rng, algo=algo, tpc=tpc, scheme=scheme)
    sol = solver.solve()
    return evaluate_solution_tf(model, sol), sol.curve


def partition_curves_for_threads(
    curves: Iterable[tuple[str, str, str]]
) -> list[list[tuple[str, str, str]]]:
    """Group curves into 4 balanced thread groups for single-realization runs:
    Thread 1: WOA-BWOA (heavy outer swarm metaheuristic)
    Thread 2: IWOA-BWOA (heavy outer swarm metaheuristic)
    Thread 3: PSO-BWOA (heavy outer swarm metaheuristic)
    Thread 4: ARJOA, IOJOA, FDMA, ALCA (fast heuristics)
    """
    c_list = list(curves)
    g1, g2, g3, g4 = [], [], [], []
    for c in c_list:
        label = c[0].upper()
        if "IWOA" in label:
            g2.append(c)
        elif "PSO" in label:
            g3.append(c)
        elif "WOA" in label and ("MF-SIC" in c[2].upper() or "BWOA" in label):
            g1.append(c)
        else:
            g4.append(c)

    groups = [g for g in [g1, g2, g3, g4] if len(g) > 0]
    return groups if groups else [c_list]


def _worker_simulate_scheme_group(
    args: tuple,
) -> list[tuple[str, BlockMetrics, np.ndarray]]:
    """Worker process: runs assigned algorithms on the shared topology & channel state."""
    group_idx, curve_list, seed, params, algo, topo = args
    results = []
    for label, tpc, scheme in curve_list:
        channel_rng = np.random.default_rng(seed)
        model = SystemModelTF(topo, params, channel_rng, scheme=scheme)
        solver_rng = np.random.default_rng(seed + 10007)
        solver = HybridSolverTF(model, solver_rng, algo=algo, tpc=tpc, scheme=scheme)
        sol = solver.solve()
        met = evaluate_solution_tf(model, sol)
        results.append((label, met, sol.curve))
    return results


def _worker_simulate_block(args: tuple) -> tuple[int, BlockMetrics]:
    seed, params, algo, tpc, scheme, topology = args
    met, _ = simulate_block_tf(seed, params, algo, tpc, scheme, topology)
    return seed, met


def monte_carlo_tf(
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
    out = []
    for r in range(n_realizations):
        seed = base_seed + r
        met, _ = simulate_block_tf(seed, params, algo, tpc, scheme, topology)
        out.append(met)
        if verbose:
            print(f"    realisation {r + 1}/{n_realizations}: U = {met.utility:+.4f}", flush=True)
    return average(out)


@dataclass
class SweepResult:
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
        payload = {"x_label": self.x_label, "x": self.x, "data": self.data}
        path.write_text(json.dumps(payload, indent=2))
        try:
            from scipy.io import savemat
            savemat(path.with_suffix(".mat"), payload)
        except Exception as exc:
            print(f"[warning] could not save .mat file to {path.with_suffix('.mat')}: {exc}")
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
    n_realizations: int = 1,
    base_seed: int = 2025,
    algo: AlgorithmParams | None = None,
    topology: TopologyFactory | None = None,
    make_topology: Callable[[float], TopologyFactory] | None = None,
    verbose: bool = True,
    jobs: int = 4,
    summary_path: str | Path | Sequence[str | Path] | None = None,
) -> SweepResult:
    algo = algo or AlgorithmParams()
    curves = list(curves)
    result = SweepResult(x_label, [float(v) for v in x_values], {})

    summary_files: list[Path] = []
    if summary_path is not None:
        if isinstance(summary_path, (list, tuple)):
            summary_files = [Path(p) for p in summary_path]
        else:
            summary_files = [Path(summary_path)]

    summary_data: dict = {
        "parameter": x_label,
        "n_realizations": n_realizations,
        "points": {},
    }
    for sf in summary_files:
        if sf.exists():
            try:
                loaded = json.loads(sf.read_text())
                if isinstance(loaded, dict) and "points" in loaded:
                    for pk, pv in loaded["points"].items():
                        summary_data["points"].setdefault(pk, {}).update(pv)
                break
            except Exception:
                pass

    thread_groups = partition_curves_for_threads(curves)

    for xi, x in enumerate(x_values):
        params = make_params(x)
        seed = base_seed + 1000 * xi
        x_key = str(x)
        if verbose:
            print(f"[{x_label} = {x}]", flush=True)

        # 1. Sample outer network topology snapshot ONCE for this sweep point
        rng_topo = np.random.default_rng(seed)
        topo_factory = make_topology(x) if make_topology is not None else (topology or fixed_topology())
        sampled_topo = topo_factory(rng_topo, params)

        if verbose and len(thread_groups) > 1:
            print(f"  Branching {len(curves)} schemes across {len(thread_groups)} thread groups (jobs={jobs}):", flush=True)
            for gi, grp in enumerate(thread_groups, 1):
                grp_names = [c[0] for c in grp]
                print(f"    Thread {gi}: {grp_names}", flush=True)

        summary_data["points"].setdefault(x_key, {})

        # 2. Dispatch thread tasks branched from identical topology/channels condition
        tasks = [
            (group_idx, group_curves, seed, params, algo, sampled_topo)
            for group_idx, group_curves in enumerate(thread_groups)
        ]

        curve_metrics: dict[str, BlockMetrics] = {}

        if jobs > 1 and len(thread_groups) > 1:
            workers = min(int(jobs), len(thread_groups))
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(_worker_simulate_scheme_group, task) for task in tasks]
                for fut in as_completed(futures):
                    for label, met, _ in fut.result():
                        curve_metrics[label] = met
        else:
            for task in tasks:
                for label, met, _ in _worker_simulate_scheme_group(task):
                    curve_metrics[label] = met

        # 3. Record and log results in the original curves order
        for label, tpc, scheme in curves:
            if label not in curve_metrics:
                continue
            metrics = curve_metrics[label].as_dict()
            result.add(label, metrics)
            summary_str = (
                f"U = {metrics['utility']:+.4f}   "
                f"offloaded = {100 * metrics['offload_ratio']:.1f} %   "
                f"t = {metrics['runtime']:.2f} s"
            )
            if verbose:
                print(f"    {label:<12} | {summary_str}", flush=True)

            summary_data["points"][x_key][label] = {
                "utility": round(float(metrics["utility"]), 4),
                "offloaded_pct": round(float(100.0 * metrics["offload_ratio"]), 1),
                "runtime_s": round(float(metrics["runtime"]), 2),
                "summary": summary_str,
            }

            for sf in summary_files:
                try:
                    sf.parent.mkdir(parents=True, exist_ok=True)
                    sf.write_text(json.dumps(summary_data, indent=2))
                except Exception:
                    pass

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
