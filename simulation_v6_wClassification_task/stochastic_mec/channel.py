"""G2A/A2G path loss, Nakagami-m fading, inverse-Gamma shadowing, ZF beamforming."""

from __future__ import annotations

import numpy as np

from .config import SystemParams

MIN_DISTANCE = 1.0


def pathloss_g2a_db(distance_m: np.ndarray, params: SystemParams, alpha: float | None = None) -> np.ndarray:
    d = np.maximum(np.asarray(distance_m, dtype=float), params.g2a_d0)
    a = params.g2a_alpha if alpha is None else alpha
    return 10.0 * a * np.log10(d / params.g2a_d0) + params.g2a_pl0


def pathloss_a2g_db(distance_m: np.ndarray, params: SystemParams) -> np.ndarray:
    d = np.maximum(np.asarray(distance_m, dtype=float), MIN_DISTANCE)
    fspl = 20.0 * np.log10(d) + 20.0 * np.log10(params.carrier_mhz) - 27.55
    return fspl + params.a2g_eta_db


def pathloss_db(distance_m: np.ndarray, params: SystemParams, link: str = "g2a") -> np.ndarray:
    if link == "a2g":
        return pathloss_a2g_db(distance_m, params)
    if link == "g2g":
        return pathloss_g2a_db(distance_m, params, alpha=params.g2g_alpha)
    return pathloss_g2a_db(distance_m, params)


def _invgamma(rng: np.random.Generator, shape: tuple[int, ...], alpha: float) -> np.ndarray:
    alpha = max(float(alpha), 1.01)
    return 1.0 / rng.gamma(alpha, 1.0 / (alpha - 1.0), size=shape)


def _nakagami(rng: np.random.Generator, shape: tuple[int, ...], m: float) -> np.ndarray:
    m = max(float(m), 0.5)
    power = rng.gamma(m, 1.0 / m, size=shape)
    phase = rng.uniform(0.0, 2.0 * np.pi, size=shape)
    return np.sqrt(power) * np.exp(1j * phase)


def large_scale_gain(
    rng: np.random.Generator,
    tx_pos: np.ndarray,
    rx_pos: np.ndarray,
    params: SystemParams,
    link: str = "g2a",
) -> np.ndarray:
    dist = np.linalg.norm(tx_pos[:, None, :] - rx_pos[None, :, :], axis=-1)
    gain_db = -pathloss_db(dist, params, link=link)
    zeta = _invgamma(rng, gain_db.shape, params.invgamma_shape)
    return (10.0 ** (gain_db / 10.0)) * zeta


def miso_channel(
    rng: np.random.Generator,
    ue_pos: np.ndarray,
    sbs_pos: np.ndarray,
    params: SystemParams,
    link: str = "g2a",
) -> np.ndarray:
    n, m = ue_pos.shape[0], sbs_pos.shape[0]
    k, ell = params.n_subchannels, params.n_antennas
    beta = large_scale_gain(rng, ue_pos, sbs_pos, params, link=link)
    if params.small_scale_fading:
        small = _nakagami(rng, (n, m, k, ell), params.nakagami_m)
    else:
        small = np.ones((n, m, k, ell), dtype=complex)
    return np.sqrt(beta)[:, :, None, None] * small


def siso_channel(
    rng: np.random.Generator,
    tx_pos: np.ndarray,
    rx_pos: np.ndarray,
    params: SystemParams,
    link: str = "g2g",
) -> np.ndarray:
    beta = large_scale_gain(rng, tx_pos, rx_pos, params, link=link)
    shape = (tx_pos.shape[0], rx_pos.shape[0], params.n_subchannels)
    small = _nakagami(rng, shape, params.nakagami_m) if params.small_scale_fading else np.ones(shape, complex)
    return np.sqrt(beta)[:, :, None] * small


def mimo_channel(
    rng: np.random.Generator,
    tx_pos: np.ndarray,
    rx_pos: np.ndarray,
    params: SystemParams,
    link: str = "a2g",
) -> np.ndarray:
    beta = large_scale_gain(rng, tx_pos, rx_pos, params, link=link)
    ell = params.n_antennas
    shape = (tx_pos.shape[0], rx_pos.shape[0], params.n_subchannels, ell, ell)
    small = _nakagami(rng, shape, params.nakagami_m) if params.small_scale_fading else np.ones(shape, complex)
    return np.sqrt(beta)[:, :, None, None, None] * small


def gram_matrices(h: np.ndarray) -> np.ndarray:
    return np.einsum("nmkl,imkl->nimk", h.conj(), h)


def zero_forcing_beamformers(h_dl: np.ndarray, ue_of_cell: list[np.ndarray]) -> list[np.ndarray]:
    n_dl, n_cells, n_sub, ell = h_dl.shape
    beamformers: list[np.ndarray] = []
    for m in range(n_cells):
        idx = ue_of_cell[m]
        if idx.size == 0:
            beamformers.append(np.zeros((n_sub, ell, 0), dtype=complex))
            continue
        w = np.empty((n_sub, ell, idx.size), dtype=complex)
        for k in range(n_sub):
            hm = h_dl[idx, m, k, :].T
            gram = hm.conj().T @ hm
            reg = 1e-12 * np.trace(gram).real / max(idx.size, 1)
            w[k] = hm @ np.linalg.pinv(gram + reg * np.eye(idx.size))
        beamformers.append(w)
    return beamformers
