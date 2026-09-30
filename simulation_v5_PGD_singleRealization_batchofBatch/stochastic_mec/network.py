"""3D HPPP topology for SI-TNTN UAV-MEC."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import SystemParams

UL, DL = 0, 1


@dataclass
class Topology:
    """One realisation (time block) of the SI-TNTN."""

    sbs_pos: np.ndarray          # (M, 3) UAV positions
    sbs_mode: np.ndarray         # (M,)  UL or DL
    ue_pos: np.ndarray           # (N, 3) active UEs
    ue_cell: np.ndarray          # (N,)  serving UAV
    ue_inactive_pos: np.ndarray  # (N_inactive, 3)
    local_cpu: np.ndarray        # (N,)
    jammer_pos: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))

    @property
    def uav_pos(self) -> np.ndarray:
        return self.sbs_pos

    @property
    def n_sbs(self) -> int:
        return self.sbs_pos.shape[0]

    @property
    def n_ue(self) -> int:
        return self.ue_pos.shape[0]

    @property
    def n_jammer(self) -> int:
        return self.jammer_pos.shape[0]

    def _busy(self) -> np.ndarray:
        counts = np.bincount(self.ue_cell, minlength=self.n_sbs)
        return counts > 0

    @property
    def ul_cells(self) -> np.ndarray:
        return np.flatnonzero((self.sbs_mode == UL) & self._busy())

    @property
    def dl_cells(self) -> np.ndarray:
        return np.flatnonzero((self.sbs_mode == DL) & self._busy())

    @property
    def null_cells(self) -> np.ndarray:
        return np.flatnonzero(~self._busy())

    @property
    def ul_ues(self) -> np.ndarray:
        return np.flatnonzero(self.sbs_mode[self.ue_cell] == UL)

    @property
    def dl_ues(self) -> np.ndarray:
        return np.flatnonzero(self.sbs_mode[self.ue_cell] == DL)


def _xy(rng: np.random.Generator, n: int, side: float) -> np.ndarray:
    return side * (rng.random((n, 2)) - 0.5)


def _ground(rng: np.random.Generator, n: int, side: float, z: float) -> np.ndarray:
    if n <= 0:
        return np.zeros((0, 3))
    return np.column_stack([_xy(rng, n, side), np.full(n, z)])


def _aerial(rng: np.random.Generator, n: int, side: float, z0: float, z1: float) -> np.ndarray:
    if n <= 0:
        return np.zeros((0, 3))
    return np.column_stack([_xy(rng, n, side), rng.uniform(z0, z1, size=n)])


def sample_topology(
    rng: np.random.Generator,
    params: SystemParams,
    n_sbs: int | None = None,
    n_active_ue: int | None = None,
    n_inactive_ue: int | None = None,
    n_ul_cells: int | None = None,
    n_jammers: int | None = None,
) -> Topology:
    """Draw one 3D HPPP snapshot."""
    side, area = params.area_side, params.area
    lam_uav = params.lambda_sbs_ul + params.lambda_sbs_dl

    if n_sbs is None:
        n_sbs = max(1, int(rng.poisson(lam_uav * area)))
    if n_active_ue is None:
        n_active_ue = max(1, int(rng.poisson(params.lambda_ue_active * area)))
    if n_inactive_ue is None:
        n_inactive_ue = int(rng.poisson(params.lambda_ue_inactive * area))
    if n_jammers is None:
        n_jammers = int(rng.poisson(params.lambda_jammer * area))

    sbs_pos = _aerial(rng, n_sbs, side, params.z_uav_min, params.z_uav_max)

    if n_ul_cells is None:
        p_ul = params.lambda_sbs_ul / lam_uav
        sbs_mode = np.where(rng.random(n_sbs) < p_ul, UL, DL)
    else:
        sbs_mode = np.full(n_sbs, DL)
        sbs_mode[rng.permutation(n_sbs)[:n_ul_cells]] = UL

    ue_pos = _ground(rng, n_active_ue, side, params.z_min)
    ue_inactive_pos = _ground(rng, n_inactive_ue, side, params.z_min)
    jammer_pos = _ground(rng, n_jammers, side, params.z_min)

    d = np.linalg.norm(ue_pos[:, None, :] - sbs_pos[None, :, :], axis=-1)
    ue_cell = np.argmin(d, axis=1)

    local_cpu = rng.choice(np.asarray(params.local_cpu_choices), size=n_active_ue)

    return Topology(
        sbs_pos=sbs_pos,
        sbs_mode=sbs_mode,
        ue_pos=ue_pos,
        ue_cell=ue_cell,
        ue_inactive_pos=ue_inactive_pos,
        local_cpu=local_cpu,
        jammer_pos=jammer_pos,
    )


def _sample_point_in_cell(
    rng: np.random.Generator,
    sbs_pos: np.ndarray,
    target_cell: int,
    side: float,
    z_min: float,
    max_tries: int = 500,
) -> np.ndarray:
    center = sbs_pos[target_cell, :2]
    pt = np.array([center[0], center[1], z_min], dtype=np.float32)
    if np.argmin(np.linalg.norm(pt - sbs_pos, axis=-1)) == target_cell:
        for radius in [20.0, 50.0, 100.0, 150.0]:
            for _ in range(25):
                r = radius * np.sqrt(rng.random())
                theta = rng.uniform(0, 2 * np.pi)
                cand = np.array([
                    np.clip(center[0] + r * np.cos(theta), -side / 2, side / 2),
                    np.clip(center[1] + r * np.sin(theta), -side / 2, side / 2),
                    z_min,
                ], dtype=np.float32)
                if np.argmin(np.linalg.norm(cand - sbs_pos, axis=-1)) == target_cell:
                    return cand
        return pt

    for _ in range(max_tries):
        cand = np.array([
            rng.uniform(-side / 2, side / 2),
            rng.uniform(-side / 2, side / 2),
            z_min,
        ], dtype=np.float32)
        if np.argmin(np.linalg.norm(cand - sbs_pos, axis=-1)) == target_cell:
            return cand
    return pt


def sample_topology_with_cells(
    rng: np.random.Generator,
    params: SystemParams,
    n_ul_cells: int,
    n_dl_cells: int,
    n_active_ue: int,
    *,
    max_tries: int = 200,
    ensure_more_ul: bool = True,
    min_ul_per_cell: int = 1,
    min_dl_per_cell: int = 1,
) -> Topology:
    """Draw a topology ensuring all UL and DL cells are non-empty, favoring UL cells."""
    n_sbs = n_ul_cells + n_dl_cells
    min_dl_total = n_dl_cells * min_dl_per_cell
    min_ul_total = max(n_ul_cells * min_ul_per_cell, (min_dl_total + 1) if ensure_more_ul else (n_ul_cells * min_ul_per_cell))
    min_total = min_dl_total + min_ul_total
    effective_n_ue = max(int(n_active_ue), min_total)

    # 1. Fast rejection sampling attempt with standard uniform scattering
    for _ in range(max_tries):
        topo = sample_topology(
            rng,
            params,
            n_sbs=n_sbs,
            n_active_ue=effective_n_ue,
            n_ul_cells=n_ul_cells,
        )
        cond_cells = (topo.ul_cells.size == n_ul_cells and topo.dl_cells.size == n_dl_cells)
        cond_more_ul = (topo.ul_ues.size > topo.dl_ues.size) if ensure_more_ul else True
        if cond_cells and cond_more_ul:
            return topo

    # 2. Fallback to guaranteed cell-controlled placement (100% guarantee)
    side = params.area_side
    for _ in range(100):
        sbs_pos = _aerial(rng, n_sbs, side, params.z_uav_min, params.z_uav_max)
        test_pts = np.column_stack([sbs_pos[:, :2], np.full(n_sbs, params.z_min)])
        d_test = np.linalg.norm(test_pts[:, None, :] - sbs_pos[None, :, :], axis=-1)
        if len(np.unique(np.argmin(d_test, axis=1))) == n_sbs:
            break

    sbs_mode = np.full(n_sbs, DL)
    ul_indices = rng.choice(n_sbs, size=n_ul_cells, replace=False)
    sbs_mode[ul_indices] = UL
    dl_indices = np.setdiff1d(np.arange(n_sbs), ul_indices)

    # Calculate target UEs for UL vs DL (~65% UL bias)
    target_n_ul = max(min_ul_total, int(round(effective_n_ue * 0.65)))
    target_n_dl = effective_n_ue - target_n_ul
    if target_n_dl < min_dl_total:
        target_n_dl = min_dl_total
        target_n_ul = max(min_ul_total, effective_n_ue - target_n_dl)
    if ensure_more_ul and target_n_ul <= target_n_dl:
        target_n_ul = target_n_dl + 1

    dl_counts = np.full(n_dl_cells, min_dl_per_cell, dtype=int)
    rem_dl = target_n_dl - np.sum(dl_counts)
    if rem_dl > 0:
        dl_counts += rng.multinomial(rem_dl, np.ones(n_dl_cells) / n_dl_cells)

    ul_counts = np.full(n_ul_cells, min_ul_per_cell, dtype=int)
    rem_ul = target_n_ul - np.sum(ul_counts)
    if rem_ul > 0:
        ul_counts += rng.multinomial(rem_ul, np.ones(n_ul_cells) / n_ul_cells)

    cell_counts = {}
    for idx, c in zip(ul_indices, ul_counts):
        cell_counts[idx] = c
    for idx, c in zip(dl_indices, dl_counts):
        cell_counts[idx] = c

    ue_positions = []
    for m in range(n_sbs):
        for _ in range(cell_counts[m]):
            pt = _sample_point_in_cell(rng, sbs_pos, m, side, params.z_min)
            ue_positions.append(pt)

    ue_pos = np.array(ue_positions, dtype=np.float32)
    d = np.linalg.norm(ue_pos[:, None, :] - sbs_pos[None, :, :], axis=-1)
    ue_cell = np.argmin(d, axis=1)

    local_cpu = rng.choice(np.asarray(params.local_cpu_choices), size=len(ue_pos))
    ue_inactive_pos = _ground(rng, int(rng.poisson(params.lambda_ue_inactive * params.area)), side, params.z_min)
    n_jam = int(rng.poisson(params.lambda_jammer * params.area)) if params.lambda_jammer > 0 else 0
    jammer_pos = _ground(rng, n_jam, side, params.z_min)

    return Topology(
        sbs_pos=sbs_pos,
        sbs_mode=sbs_mode,
        ue_pos=ue_pos,
        ue_cell=ue_cell,
        ue_inactive_pos=ue_inactive_pos,
        local_cpu=local_cpu,
        jammer_pos=jammer_pos,
    )
