"""3D HPPP topology for SI-TNTN UAV-MEC (manuscript_v1).

UAVs sit in [z_uav_min, z_uav_max]; UEs and jammers lie on the ground plane.
Association is nearest-UAV in 3D Euclidean distance.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import SystemParams

UL, DL = 0, 1


@dataclass
class Topology:
    """One realisation (time block) of the SI-TNTN."""

    sbs_pos: np.ndarray          # (M, 3) UAV positions (API name kept)
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


def sample_topology_with_cells(
    rng: np.random.Generator,
    params: SystemParams,
    n_ul_cells: int,
    n_dl_cells: int,
    n_active_ue: int,
    *,
    max_tries: int = 200,
) -> Topology:
    """Draw a topology whose UL/DL UAV cells are all non-null."""
    for _ in range(max_tries):
        topo = sample_topology(
            rng,
            params,
            n_sbs=n_ul_cells + n_dl_cells,
            n_active_ue=n_active_ue,
            n_ul_cells=n_ul_cells,
        )
        if topo.ul_cells.size == n_ul_cells and topo.dl_cells.size == n_dl_cells:
            return topo
    return topo
