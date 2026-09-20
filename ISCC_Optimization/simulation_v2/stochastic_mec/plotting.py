"""Matplotlib helpers reproducing the figure style of the manuscript."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .experiments import SweepResult  # noqa: E402
from .network import Topology  # noqa: E402

MARKERS = ["o", "v", "s", "^", "x", "d", "*", "p"]


def _style(ax, xlabel: str, ylabel: str) -> None:
    ax.grid(True, alpha=0.35)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="best", fontsize=9)


def plot_sweep(
    result: SweepResult,
    metric: str,
    ylabel: str,
    path: str | Path,
    xlabel: str | None = None,
    title: str | None = None,
) -> Path:
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    for i, (label, series) in enumerate(result.data.items()):
        if metric not in series:
            continue
        ax.plot(result.x, series[metric], marker=MARKERS[i % len(MARKERS)],
                linewidth=1.8, markersize=6, label=label)
    _style(ax, xlabel or result.x_label, ylabel)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_curves(
    curves: dict[str, np.ndarray],
    path: str | Path,
    xlabel: str = "Iteration",
    ylabel: str = "System utility",
) -> Path:
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    for i, (label, y) in enumerate(curves.items()):
        y = np.asarray(y, dtype=float)
        ax.plot(np.arange(1, y.size + 1), y, linewidth=1.8,
                marker=MARKERS[i % len(MARKERS)], markevery=max(1, y.size // 15), label=label)
    _style(ax, xlabel, ylabel)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_topology(topo: Topology, path: str | Path, side: float = 500.0) -> Path:
    """Snapshot of one realisation: SBSs, their Voronoi cells and the UEs.

    The Voronoi tessellation is drawn from a nearest-SBS raster, which keeps
    the module free of any dependency beyond NumPy and Matplotlib.
    """
    fig, ax = plt.subplots(figsize=(6.0, 5.6))
    uav_xy = topo.sbs_pos[:, :2]
    grid = np.linspace(-side, side, 600)
    gx, gy = np.meshgrid(grid, grid)
    pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
    nearest = np.argmin(
        np.linalg.norm(pts[:, None, :] - uav_xy[None, :, :], axis=-1), axis=1
    ).reshape(gx.shape)
    ax.contour(gx, gy, nearest, levels=np.arange(topo.n_sbs) + 0.5,
               colors="0.65", linewidths=0.9)

    ul = topo.sbs_mode == 0
    ax.scatter(*uav_xy[ul].T, marker="s", s=90, c="tab:green", label="UL UAV")
    ax.scatter(*uav_xy[~ul].T, marker="^", s=90, c="tab:olive", label="DL UAV")
    for i, (x, y) in enumerate(uav_xy):
        ax.annotate(f"UAV{i + 1}", (x + 12, y + 8), fontsize=8)

    ue_ul, ue_dl = topo.ul_ues, topo.dl_ues
    if ue_ul.size:
        ax.scatter(*topo.ue_pos[ue_ul, :2].T, marker="o", s=34, c="tab:blue", label="active UE (UL)")
    if ue_dl.size:
        ax.scatter(*topo.ue_pos[ue_dl, :2].T, marker="o", s=34, c="tab:cyan", label="active UE (DL)")
    if topo.ue_inactive_pos.size:
        ax.scatter(*topo.ue_inactive_pos[:, :2].T, marker="o", s=26, c="tab:red", label="inactive UE")
    if topo.jammer_pos.size:
        ax.scatter(*topo.jammer_pos[:, :2].T, marker="x", s=50, c="k", label="jammer")

    ax.set_xlim(-side, side)
    ax.set_ylim(-side, side)
    ax.set_aspect("equal")
    _style(ax, r"$x$ [m]", r"$y$ [m]")
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------- #
# ISCC figures (manuscript_v2)
# ---------------------------------------------------------------------- #
def plot_frontier(
    result: SweepResult,
    x_metric: str,
    y_metric: str,
    xlabel: str,
    ylabel: str,
    path: str | Path,
    annotate: str | None = None,
) -> Path:
    """Parametric curve ``(x_metric, y_metric)`` traced by the sweep variable."""
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    for i, (label, series) in enumerate(result.data.items()):
        x = np.asarray(series[x_metric], dtype=float)
        y = np.asarray(series[y_metric], dtype=float)
        ax.plot(x, y, marker=MARKERS[i % len(MARKERS)], linewidth=1.8, markersize=6, label=label)
        if annotate and i == 0:
            for j in (0, len(result.x) - 1):
                ax.annotate(f"{annotate}$={result.x[j]:g}$", (x[j], y[j]),
                            textcoords="offset points", xytext=(6, -12), fontsize=8)
    _style(ax, xlabel, ylabel)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_dual(
    series: dict,
    left_metric: str,
    right_metric: str,
    xlabel: str,
    left_label: str,
    right_label: str,
    path: str | Path,
    floor: list[float] | None = None,
) -> Path:
    """Two metrics of one curve on twin y-axes, with an optional floor line."""
    x = np.asarray(series["x"], dtype=float)
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    c0, c1 = "tab:blue", "tab:red"
    ax.plot(x, series[left_metric], marker="o", color=c0, linewidth=1.8, label=left_label)
    if floor is not None:
        ax.plot(x, floor, linestyle="--", color=c0, alpha=0.6, linewidth=1.2,
                label=r"floor $\chi^{\min}$ at $\bar\gamma^{\tt sen}$")
    ax.set_ylabel(left_label, color=c0)
    ax.tick_params(axis="y", labelcolor=c0)
    ax2 = ax.twinx()
    ax2.plot(x, series[right_metric], marker="s", color=c1, linewidth=1.8, label=right_label)
    ax2.set_ylabel(right_label, color=c1)
    ax2.tick_params(axis="y", labelcolor=c1)
    ax.set_xlabel(xlabel)
    ax.grid(True, alpha=0.35)
    h0, l0 = ax.get_legend_handles_labels()
    h1, l1 = ax2.get_legend_handles_labels()
    ax.legend(h0 + h1, l0 + l1, loc="best", fontsize=9)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path
