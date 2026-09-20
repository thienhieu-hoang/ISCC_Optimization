"""System model: signal model, sub-problem objectives and penalty functions.

The joint association variable used everywhere below is

    A_net = [A_ul ; A_dl]  of shape (N_ul + M_dl, K),

i.e. the first ``N_ul`` rows say which sub-channel (if any) an uplink UE uses
to offload, and the remaining ``M_dl`` rows say which sub-channel a downlink
SBS uses to broadcast.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import channel
from .config import SystemParams
from .network import Topology


@dataclass
class Solution:
    """Everything Algorithm 3 returns for one time block."""

    utility: float
    assoc: np.ndarray            # (N_ul + M_dl, K) binary
    ul_power: np.ndarray         # (N_ul,)
    dl_power: np.ndarray         # (N_dl,)
    server_alloc: np.ndarray     # (N_ul,) F_nm of the offloading UEs, 0 otherwise
    curve: np.ndarray            # BWOA convergence curve
    runtime: float = 0.0
    n_inner_calls: int = 0
    rho: np.ndarray | None = None    # (N_ul,) offloading split (Prop. 2)
    chi: np.ndarray | None = None    # (N_ul,) feature retention (Prop. 3)


class SystemModel:
    """Signal model + objective functions of one network realisation."""

    def __init__(self, topo: Topology, params: SystemParams, rng: np.random.Generator):
        self.topo = topo
        self.p = params
        self.rng = rng

        self.K = params.n_subchannels
        self.L = params.n_antennas
        self.noise = params.noise_power                       # B_k * N_0
        self.ul_noise = self.L * self.noise                   # L * B_k * N_0

        # ---- index bookkeeping -----------------------------------------
        self.ul_cells = topo.ul_cells
        self.dl_cells = topo.dl_cells
        self.ul_ues = topo.ul_ues
        self.dl_ues = topo.dl_ues
        self.n_ul, self.n_dl = self.ul_ues.size, self.dl_ues.size
        self.m_ul, self.m_dl = self.ul_cells.size, self.dl_cells.size

        # serving cell of each UE, expressed in *local* (UL-cell / DL-cell) indices
        cell_of = topo.ue_cell
        ul_lookup = {int(c): i for i, c in enumerate(self.ul_cells)}
        dl_lookup = {int(c): i for i, c in enumerate(self.dl_cells)}
        self.ul_serving = np.array([ul_lookup[int(cell_of[n])] for n in self.ul_ues], dtype=int)
        self.dl_serving = np.array([dl_lookup[int(cell_of[n])] for n in self.dl_ues], dtype=int)
        self.dl_ue_of_cell = [np.flatnonzero(self.dl_serving == m) for m in range(self.m_dl)]
        self.ul_ue_of_cell = [np.flatnonzero(self.ul_serving == m) for m in range(self.m_ul)]

        self._build_channels()
        self._build_tasks()

    # ------------------------------------------------------------------ #
    # construction
    # ------------------------------------------------------------------ #
    def _build_channels(self) -> None:
        p, rng, topo = self.p, self.rng, self.topo
        ul_pos = topo.ue_pos[self.ul_ues]
        dl_pos = topo.ue_pos[self.dl_ues]
        ul_bs = topo.sbs_pos[self.ul_cells]
        dl_bs = topo.sbs_pos[self.dl_cells]

        h_ul = channel.miso_channel(rng, ul_pos, ul_bs, p, link="g2a") if self.n_ul and self.m_ul \
            else np.zeros((self.n_ul, self.m_ul, self.K, self.L), dtype=complex)
        gram = channel.gram_matrices(h_ul) if h_ul.size else np.zeros((self.n_ul, self.n_ul, self.m_ul, self.K), complex)
        self.cross2 = np.abs(gram) ** 2
        self.hnorm2 = np.real(np.einsum("nnmk->nmk", gram)) if h_ul.size \
            else np.zeros((self.n_ul, self.m_ul, self.K))

        h_dl = channel.miso_channel(rng, dl_pos, dl_bs, p, link="a2g") if self.n_dl and self.m_dl \
            else np.zeros((self.n_dl, self.m_dl, self.K, self.L), dtype=complex)
        self.beamformers = channel.zero_forcing_beamformers(h_dl, self.dl_ue_of_cell)

        # |h_{n',k,j}^H w_{j,k,i}|^2 for every DL UE n', DL cell j, sub-channel k
        self.dl_gain = []
        for j in range(self.m_dl):
            w = self.beamformers[j]                            # (K, L, N_j)
            if w.shape[2] == 0:
                self.dl_gain.append(np.zeros((self.n_dl, self.K, 0)))
                continue
            g = np.einsum("nkl,kli->nki", h_dl[:, j, :, :].conj(), w)
            self.dl_gain.append(np.abs(g) ** 2)                # (N_dl, K, N_j)

        # ||w_{m,n'}||^2 of the serving beam of each DL UE, per sub-channel
        self.w_norm2 = np.zeros((self.n_dl, self.K))
        for j in range(self.m_dl):
            idx = self.dl_ue_of_cell[j]
            if idx.size == 0:
                continue
            w = self.beamformers[j]                            # (K, L, N_j)
            self.w_norm2[idx] = np.sum(np.abs(w) ** 2, axis=1).T   # (N_j, K)

        self.ue2ue2 = np.abs(channel.siso_channel(rng, ul_pos, dl_pos, p, link="g2g")) ** 2 \
            if self.n_ul and self.n_dl else np.zeros((self.n_ul, self.n_dl, self.K))
        if self.m_dl and self.m_ul:
            g_bs = channel.mimo_channel(rng, dl_bs, ul_bs, p, link="a2g")
            self.bs2bs2 = np.sum(np.abs(g_bs) ** 2, axis=-1)
        else:
            self.bs2bs2 = np.zeros((self.m_dl, self.m_ul, self.K, self.L))

        jam_pos = topo.jammer_pos
        self.n_jam = jam_pos.shape[0]
        if self.n_jam and self.n_ul and self.m_ul:
            h_j = channel.miso_channel(rng, jam_pos, ul_bs, p, link="g2a")
            self.jam_ul_cross2 = np.abs(np.einsum("nmkl,qmkl->nqmk", h_ul.conj(), h_j)) ** 2
        else:
            self.jam_ul_cross2 = np.zeros((self.n_ul, self.n_jam, self.m_ul, self.K))
        if self.n_jam and self.n_dl:
            self.jam_dl2 = np.abs(channel.siso_channel(rng, jam_pos, dl_pos, p, link="g2g")) ** 2
        else:
            self.jam_dl2 = np.zeros((self.n_jam, self.n_dl, self.K))
        self.jam_active = rng.random((self.n_jam, self.K)) < p.jam_prob if self.n_jam \
            else np.zeros((0, self.K), dtype=bool)

    def _build_tasks(self) -> None:
        """Sensed tasks of Sec. II-C and the all-local reference costs (eq. ref)."""
        p = self.p
        f_loc = self.topo.local_cpu[self.ul_ues]
        kappa = p.energy_coefficient
        self.f_local = f_loc
        self.d_raw = p.data_size                                       # D_n^raw
        self.c_raw = p.cpu_cycles                                      # C_n^raw
        self.t_cmp = p.cpu_cycles / f_loc                              # C^raw / F^loc
        self.e_cmp = kappa * p.cpu_cycles * f_loc ** 2                 # kappa C^raw (F^loc)^2

        self.t_sen = np.full(self.n_ul, p.sensing_time)
        self.e_sen = np.full(self.n_ul, p.sensing_power * p.sensing_time)
        ext_cycles = p.extract_cycles_per_bit * p.data_size
        self.t_ext = ext_cycles / f_loc
        self.e_ext = kappa * ext_cycles * f_loc ** 2
        self.t_ref = self.t_sen + self.t_ext + self.t_cmp              # T_n^ref
        self.e_ref = self.e_sen + self.e_ext + self.e_cmp              # E_n^ref

        # per-UE sensing SNR -> L_n and the retention floor chi_n^min (eq. chimin)
        snr_db = p.sensing_snr_db + p.sensing_snr_std_db * self.rng.standard_normal(self.n_ul)
        self.sens_info = np.log2(1.0 + 10.0 ** (snr_db / 10.0))        # L_n
        self.vartheta = np.full(self.n_ul, p.accuracy_sensitivity)
        self.lam_th = p.accuracy_threshold
        self.chi_min = -np.log(1.0 - self.lam_th) / (self.vartheta * self.sens_info)
        self.admissible = self.chi_min <= 1.0

        self.beta_t = np.full(self.n_ul, p.beta_time)
        self.beta_e = np.full(self.n_ul, p.beta_energy)
        self.beta_a = np.full(self.n_ul, p.beta_acc)
        # constant part of U_n^ul (the chi-free terms of eq. Uul)
        self.u_const = (self.beta_t * (self.t_ref - self.t_sen - self.t_ext) / self.t_ref
                        + self.beta_e * (self.e_ref - self.e_sen - self.e_ext) / self.e_ref)

        self.server_capacity = np.full(self.m_ul, p.server_capacity)
        self.sbs_budget = np.full(self.m_dl, p.sbs_power_budget)

    # ------------------------------------------------------------------ #
    # ISCC: accuracy, split and retention (Props. 1-3)
    # ------------------------------------------------------------------ #
    def default_split(self, ul_sub: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``rho_n = a_n^ul``, ``chi_n = 1`` -- the atomic task of v1."""
        return (ul_sub >= 0).astype(float), np.ones(self.n_ul)

    def accuracy(self, chi: np.ndarray) -> np.ndarray:
        """``Lambda_n(chi_n)`` of eq. (acc)."""
        return 1.0 - np.exp(-self.vartheta * chi * self.sens_info)

    def eta_theta(self, rho: np.ndarray, chi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(eta_n, theta_n)`` of eq. (W): the v1 constants scaled by ``chi_n rho_n``."""
        scale = chi * rho * self.d_raw
        eta = self.beta_t * scale / self.t_ref
        theta = self.beta_e * scale / (self.p.amplifier_efficiency * self.e_ref)
        return eta, theta

    def _branches(
        self, rho: np.ndarray, rates: np.ndarray, f_alloc: np.ndarray, p_ul: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Per-unit-retention delay ``Delta_n`` (eq. Tn) and energy ``Theta_n`` (eq. Theta)."""
        off = rho > 0
        r = np.where(off, np.maximum(rates, 1e-12), np.inf)
        f = np.where(off, np.maximum(f_alloc, 1e-12), np.inf)
        d_loc = (1.0 - rho) * self.t_cmp
        d_ofl = np.where(off, rho * (self.d_raw / r + self.c_raw / f), 0.0)
        theta = (1.0 - rho) * self.e_cmp + np.where(
            off, rho * p_ul * self.d_raw / (self.p.amplifier_efficiency * r), 0.0)
        return np.maximum(d_loc, d_ofl), theta

    def _omega(self, delta: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Marginal delay-energy price ``Omega_n`` of one retained feature unit."""
        return self.beta_t * delta / self.t_ref + self.beta_e * theta / self.e_ref

    def optimal_chi(self, omega: np.ndarray) -> np.ndarray:
        """Prop. 3: ``chi_n^* = clip(-ln(Psi_n) / (vartheta_n L_n), chi_n^min, 1)``."""
        vl = self.vartheta * self.sens_info
        lo = np.minimum(self.chi_min, 1.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            psi = (1.0 - self.lam_th) * omega / (self.beta_a * vl)
            chi = np.where(psi > 0, -np.log(psi) / vl, 1.0)
        chi = np.where(self.beta_a > 0, chi, lo)
        return np.clip(chi, lo, 1.0)

    def rho_star(self, rates: np.ndarray, f_alloc: np.ndarray) -> np.ndarray:
        """Branch-balancing split of eq. (rhostar)."""
        r = np.maximum(rates, 1e-12)
        f = np.maximum(f_alloc, 1e-12)
        return self.t_cmp / (self.t_cmp + self.d_raw / r + self.c_raw / f)

    def optimal_split(
        self, ul_sub: np.ndarray, rates: np.ndarray, f_alloc: np.ndarray, p_ul: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Props. 2 & 3 jointly for fixed ``(A, p, F)``.

        ``U_n = const - chi_n Omega_n(rho_n) + accuracy(chi_n)`` and ``Omega_n`` is
        convex piecewise linear in ``rho_n`` with its kink at ``rho_n^*``, so the
        maximising split minimises ``Omega_n`` over ``{0, rho_n^*, 1}`` for *every*
        ``chi_n > 0``; ``chi_n`` then follows in closed form.  The pair is therefore
        jointly optimal, not merely a coordinate step.
        """
        offloading = ul_sub >= 0
        rs = self.rho_star(rates, f_alloc)
        cands = np.stack([np.zeros(self.n_ul), rs, np.ones(self.n_ul)])
        omegas = np.stack([self._omega(*self._branches(c, rates, f_alloc, p_ul)) for c in cands])
        pick = np.argmin(omegas, axis=0)
        rho = np.where(offloading, cands[pick, np.arange(self.n_ul)], 0.0)
        omega = self._omega(*self._branches(rho, rates, f_alloc, p_ul))
        return rho, self.optimal_chi(omega)

    def ul_utilities(
        self,
        rho: np.ndarray,
        chi: np.ndarray,
        rates: np.ndarray,
        f_alloc: np.ndarray,
        p_ul: np.ndarray,
    ) -> np.ndarray:
        """Per-UE ``U_n^ul`` of eq. (Uul); inadmissible UEs are dropped (zero)."""
        delta, theta = self._branches(rho, rates, f_alloc, p_ul)
        acc = self.accuracy(chi)
        u = (self.u_const - chi * self._omega(delta, theta)
             + self.beta_a * (acc - self.lam_th) / (1.0 - self.lam_th))
        return np.where(self.admissible, u, 0.0)

    def iscc_allocate(
        self, ul_sub: np.ndarray, rates: np.ndarray, p_ul: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Block-coordinate rounds of Prop. 1 (``F``) and Props. 2-3 (``rho, chi``)."""
        rho, chi = self.default_split(ul_sub)
        if not self.p.enable_iscc:
            # atomic-task ablation: same objective, (rho, chi) frozen at (a, 1)
            _, f = self.smca(ul_sub, rho, chi, rates)
            return rho, chi, f
        f = np.zeros(self.n_ul)
        for _ in range(max(1, self.p.iscc_rounds)):
            _, f = self.smca(ul_sub, rho, chi, rates)
            rho, chi = self.optimal_split(ul_sub, rates, f, p_ul)
        _, f = self.smca(ul_sub, rho, chi, rates)
        return rho, chi, f

    def ul_utility_upper_bound(self, ul_sub: np.ndarray) -> float:
        """Upper bound on ``sum_n U_n^ul`` valid uniformly over ``(p, F, rho, chi)``.

        Pruning must hold before ``rho, chi`` are known, so the v1 bound
        ``N^ofl - V(A, F*)`` is no longer valid (``V`` depends on ``rho chi``).
        Instead each offloading UE is given an interference-free rate at
        ``p_max``, the whole server, and the offload-energy floor
        ``p / R(p) >= ln 2 / (B a)``.  Each lowers ``Delta_n`` and ``Theta_n``
        pointwise, and the maximum of the resulting bound over ``(rho, chi)``
        is exact by the argument of :meth:`optimal_split`.  Local UEs are exact.
        """
        bw, xi_amp = self.p.subchannel_bw, self.p.amplifier_efficiency
        offloading = ul_sub >= 0
        r_bar = np.full(self.n_ul, 1e-12)
        f_bar = np.full(self.n_ul, 1e-12)
        e_floor = np.zeros(self.n_ul)                      # lower bound on p D / (xi R)
        for n in np.flatnonzero(offloading):
            k, m = ul_sub[n], self.ul_serving[n]
            a = self.ifree_gain(n, m, k)
            if a <= 0:
                continue
            r_bar[n] = bw * np.log2(1.0 + a * self.p.p_max)
            f_bar[n] = self.server_capacity[m]
            e_floor[n] = self.d_raw * np.log(2.0) / (xi_amp * bw * a)

        rs = self.rho_star(r_bar, f_bar)
        best = None
        for c in (np.zeros(self.n_ul), rs, np.ones(self.n_ul)):
            rho = np.where(offloading, c, 0.0)
            delta, _ = self._branches(rho, r_bar, f_bar, np.zeros(self.n_ul))
            theta = (1.0 - rho) * self.e_cmp + rho * e_floor
            om = self._omega(delta, theta)
            best = om if best is None else np.minimum(best, om)
        chi = self.optimal_chi(best)
        acc = self.accuracy(chi)
        u = (self.u_const - chi * best
             + self.beta_a * (acc - self.lam_th) / (1.0 - self.lam_th))
        return float(np.sum(np.where(self.admissible, u, 0.0)))

    def ifree_gain(self, n: int, m: int, k: int) -> float:
        """Slope ``a`` of the interference-free UL SNR ``a p`` (Lemma 1, bound)."""
        own = self.hnorm2[n, m, k]
        if self.p.mf_noise_fix:
            return own / self.noise                       # p ||h||^2 / (B N0)
        return own ** 2 / self.ul_noise                   # v1: p ||h||^4 / (L B N0)

    # ------------------------------------------------------------------ #
    # association helpers
    # ------------------------------------------------------------------ #
    @property
    def assoc_shape(self) -> tuple[int, int]:
        return (self.n_ul + self.m_dl, self.K)

    def split(self, assoc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Split ``A_net`` into ``(A_ul, A_dl)``."""
        return assoc[: self.n_ul], assoc[self.n_ul:]

    @staticmethod
    def _first_selected(rows: np.ndarray) -> np.ndarray:
        """Index of the first '1' in each row, ``-1`` if the row is all zeros."""
        if rows.size == 0:
            return np.zeros(rows.shape[0], dtype=int) - 1
        any_ = rows.any(axis=1)
        return np.where(any_, rows.argmax(axis=1), -1)

    def decode(self, assoc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Sub-channel chosen by each UL UE (``-1`` = local) and each DL SBS.

        UEs failing the admissibility test ``chi_n^min <= 1`` are discarded
        (Algorithm 1, line 1): their rows are projected to "local" here rather
        than penalised, since BWOA rarely finds an exactly-zero row on its own.
        """
        a_ul, a_dl = self.split(assoc)
        ul_sub = self._first_selected(a_ul)
        if ul_sub.size:
            ul_sub = np.where(self.admissible, ul_sub, -1)
        return ul_sub, self._first_selected(a_dl)

    def association_penalty(self, assoc: np.ndarray) -> float:
        """``U_pen(A_net)`` -- one sub-channel per UL UE, exactly one per DL SBS."""
        a_ul, a_dl = self.split(assoc)
        pen = 0.0
        if self.n_ul:
            g2 = a_ul.sum(axis=1) - 1.0                       # <= 0 required
            pen += self.p.penalty_assoc * np.sum(np.where(g2 > 0, g2 ** 2, 0.0))
        if self.m_dl:
            g3 = np.abs(a_dl.sum(axis=1) - 1.0)               # == 0 required
            pen += self.p.penalty_assoc * np.sum(np.where(g3 > 0, g3 ** 2, 0.0))
        return float(pen)

    # ------------------------------------------------------------------ #
    # SMCA: computation-resource allocation (closed form, convex)
    # ------------------------------------------------------------------ #
    def smca(
        self,
        ul_sub: np.ndarray,
        rho: np.ndarray | None = None,
        chi: np.ndarray | None = None,
        rates: np.ndarray | None = None,
    ) -> tuple[float, np.ndarray]:
        """Prop. 1: clipped water-filling ``F*`` and the compute-delay cost ``V``.

        ``F*_{nm} = min{ t_m sqrt(beta_t rho chi C^raw / T^ref), Fbar_{nm} }``, with
        ``t_m`` set so the server budget binds (or every UE clipped at ``Fbar``).
        With ``rho = chi = 1``, no sensing overhead and no ``rates`` this is the
        proportional rule of v1.  ``V = sum_n (beta_t / T^ref) chi rho C^raw / F``.
        """
        if rho is None or chi is None:
            rho, chi = self.default_split(ul_sub)
        f_alloc = np.zeros(self.n_ul)
        v = 0.0
        offloading = (ul_sub >= 0) & (rho > 0)
        weight = self.beta_t * rho * chi * self.c_raw / self.t_ref
        root = np.sqrt(np.maximum(weight, 0.0))

        # Fbar of eq. (Fbar): beyond it the local branch is the active one
        f_bar = np.full(self.n_ul, np.inf)
        if rates is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                slack = (1.0 - rho) * self.t_cmp - rho * self.d_raw / np.maximum(rates, 1e-12)
                f_bar = np.where(slack > 0, rho * self.c_raw / slack, np.inf)

        for m in range(self.m_ul):
            members = np.flatnonzero(offloading & (self.ul_serving == m))
            if members.size == 0:
                continue
            cap = self.server_capacity[m]
            s_n, fb = root[members], f_bar[members]
            if s_n.sum() <= 0.0:
                # beta_t == 0: V does not depend on F, share the server evenly
                f_alloc[members] = np.minimum(cap / members.size, fb)
                continue
            if np.all(np.isfinite(fb)) and fb.sum() <= cap:
                alloc = fb.copy()
            else:
                lo, hi = 0.0, cap / s_n[s_n > 0].min()
                for _ in range(100):
                    t = 0.5 * (lo + hi)
                    if np.minimum(s_n * t, fb).sum() < cap:
                        lo = t
                    else:
                        hi = t
                alloc = np.minimum(s_n * hi, fb)
            f_alloc[members] = alloc
            pos = alloc > 0
            v += float(np.sum(weight[members][pos] / alloc[pos]))
        return float(v), f_alloc

    # ------------------------------------------------------------------ #
    # uplink signal model (MF-SIC)
    # ------------------------------------------------------------------ #
    def dl_antenna_power(self, dl_sub: np.ndarray, dl_power: np.ndarray) -> np.ndarray:
        """Per-antenna transmit power ``P^SBS_{m',k}[l']`` of the DL SBSs."""
        out = np.zeros((self.m_dl, self.K, self.L))
        for m in range(self.m_dl):
            k = dl_sub[m]
            idx = self.dl_ue_of_cell[m]
            if k < 0 or idx.size == 0:
                continue
            w = self.beamformers[m][k]                        # (L, N_m)
            out[m, k] = np.abs(w) ** 2 @ dl_power[idx]
        return out

    def cochannel_at_sbs(self, dl_sub: np.ndarray, dl_power: np.ndarray) -> np.ndarray:
        """``Xi^SBS_{k,m}``: DL-to-UL co-channel interference, shape ``(K, M_ul)``."""
        if self.m_dl == 0 or self.m_ul == 0:
            return np.zeros((self.K, self.m_ul))
        p_ant = self.dl_antenna_power(dl_sub, dl_power)        # (M_dl, K, L)
        # bs2bs2: (M_dl, M_ul, K, L) already summed over the transmit antenna index
        return np.einsum("dkl,dukl->ku", p_ant, self.bs2bs2)

    def _jam_ul(self, n: int, m: int, k: int, ul_sub: np.ndarray, dl_sub: np.ndarray) -> float:
        if self.n_jam == 0:
            return 0.0
        busy = bool(np.any(ul_sub == k) or np.any(dl_sub == k))
        if not busy:
            return 0.0
        mask = self.jam_active[:, k]
        if not np.any(mask):
            return 0.0
        return float(np.sum(self.jam_ul_cross2[n, mask, m, k]) * self.p.jammer_power)

    def ul_sinr(
        self,
        ul_sub: np.ndarray,
        p_ul: np.ndarray,
        xi: np.ndarray,
        dl_sub: np.ndarray | None = None,
    ) -> np.ndarray:
        if dl_sub is None:
            dl_sub = np.full(self.m_dl, -1, dtype=int)
        gamma = np.zeros(self.n_ul)
        for k in range(self.K):
            users = np.flatnonzero(ul_sub == k)
            if users.size == 0:
                continue
            for n in users:
                m = self.ul_serving[n]
                own = self.hnorm2[n, m, k]
                if own <= 0.0:
                    continue
                others = users[users != n]
                same = others[self.ul_serving[others] == m]
                weaker = same[self.hnorm2[same, m, k] <= own]
                inter = others[self.ul_serving[others] != m]
                i_w = float(np.sum(p_ul[weaker] * self.cross2[n, weaker, m, k]))
                i_o = float(np.sum(p_ul[inter] * self.cross2[n, inter, m, k]))
                jam = self._jam_ul(n, m, k, ul_sub, dl_sub)
                if self.p.mf_noise_fix:
                    den = i_w + i_o + own * xi[k, m] / self.L + jam + own * self.noise
                else:
                    den = i_w + i_o + xi[k, m] + jam + self.ul_noise
                gamma[n] = p_ul[n] * own ** 2 / den
        return gamma

    def ul_rates(
        self,
        ul_sub: np.ndarray,
        p_ul: np.ndarray,
        xi: np.ndarray,
        dl_sub: np.ndarray | None = None,
    ) -> np.ndarray:
        return self.p.subchannel_bw * np.log2(1.0 + self.ul_sinr(ul_sub, p_ul, xi, dl_sub))

    def sic_violation(self, ul_sub: np.ndarray, p_ul: np.ndarray) -> float:
        """Sum of squared violations of the SIC decodability constraint."""
        total = 0.0
        eps = self.p.sic_threshold
        for k in range(self.K):
            users = np.flatnonzero(ul_sub == k)
            if users.size < 2:
                continue
            for n in users:
                m = self.ul_serving[n]
                own = self.hnorm2[n, m, k]
                others = users[users != n]
                same = others[self.ul_serving[others] == m]
                weaker = same[self.hnorm2[same, m, k] <= own]
                if weaker.size == 0:
                    continue
                g5 = eps * float(np.sum(p_ul[weaker] * self.hnorm2[weaker, m, k])) - p_ul[n] * own
                if g5 > 0:
                    total += g5 ** 2
        return total

    # ---- MPC sub-problem (Algorithm 1) -------------------------------- #
    def mpc_objective(
        self,
        ul_sub: np.ndarray,
        p_ul: np.ndarray,
        xi: np.ndarray,
        rho: np.ndarray | None = None,
        chi: np.ndarray | None = None,
    ) -> float:
        """``W(A_ul, p) + W_pen(A_ul, p)`` -- to be *minimised*."""
        offloading = np.flatnonzero(ul_sub >= 0)
        if offloading.size == 0:
            return 0.0
        if rho is None or chi is None:
            rho, chi = self.default_split(ul_sub)
        eta, theta = self.eta_theta(rho, chi)
        rates = self.ul_rates(ul_sub, p_ul, xi)
        rate = np.maximum(rates[offloading], 1e-12)
        w = float(np.sum((eta[offloading] + theta[offloading] * p_ul[offloading]) / rate))

        g4 = p_ul - self.p.p_max
        pen = self.p.penalty_power * float(np.sum(np.where(g4 > 0, g4 ** 2, 0.0)))
        pen += self.p.penalty_sic * self.sic_violation(ul_sub, p_ul)
        return w + pen

    def mpc_lower_bound(
        self,
        ul_sub: np.ndarray,
        rho: np.ndarray | None = None,
        chi: np.ndarray | None = None,
    ) -> float:
        """``min_p W~(A_ul, p)`` -- Theorems 2 & 3 (interference-free, quasi-convex).

        Each UE's sub-problem is solved by bisection on ``phi(p) = 0``.
        """
        offloading = np.flatnonzero(ul_sub >= 0)
        if offloading.size == 0:
            return 0.0
        if rho is None or chi is None:
            rho, chi = self.default_split(ul_sub)
        etas, thetas = self.eta_theta(rho, chi)
        bw, p_min, p_max = self.p.subchannel_bw, self.p.p_min, self.p.p_max
        tol = 1e-9
        total = 0.0
        for n in offloading:
            k, m = ul_sub[n], self.ul_serving[n]
            a = self.ifree_gain(n, m, k)
            eta, theta = etas[n], thetas[n]
            if eta <= 0 and theta <= 0:
                continue

            def phi(x: float) -> float:
                return theta * np.log2(1 + a * x) - (a / np.log(2)) * (eta + theta * x) / (1 + a * x)

            if a <= 0:
                continue
            if phi(p_max) <= 0:
                x_star = p_max
            else:
                lo, hi = p_min, p_max
                while hi - lo > tol:
                    mid = 0.5 * (lo + hi)
                    if phi(mid) <= 0:
                        lo = mid
                    else:
                        hi = mid
                x_star = 0.5 * (lo + hi)
            total += (eta + theta * x_star) / (bw * np.log2(1 + a * x_star))
        return float(total)

    # ------------------------------------------------------------------ #
    # downlink signal model (ZF beamforming)
    # ------------------------------------------------------------------ #
    def dl_sinr(
        self,
        dl_sub: np.ndarray,
        dl_power: np.ndarray,
        ul_sub: np.ndarray,
        p_ul: np.ndarray,
    ) -> np.ndarray:
        """SINR of every DL UE; ZF cancels the intra-cell terms."""
        gamma = np.zeros(self.n_dl)
        for m in range(self.m_dl):
            k = dl_sub[m]
            served = self.dl_ue_of_cell[m]
            if k < 0 or served.size == 0:
                continue
            interferers = [j for j in range(self.m_dl) if j != m and dl_sub[j] == k]
            for n in served:
                inter = 0.0
                for j in interferers:
                    idx = self.dl_ue_of_cell[j]
                    if idx.size:
                        inter += float(self.dl_gain[j][n, k] @ dl_power[idx])
                ul_on_k = np.flatnonzero(ul_sub == k)
                cci = float(np.sum(p_ul[ul_on_k] * self.ue2ue2[ul_on_k, n, k])) if ul_on_k.size else 0.0
                jam = 0.0
                if self.n_jam:
                    mask = self.jam_active[:, k]
                    if np.any(mask):
                        jam = float(np.sum(self.jam_dl2[mask, n, k]) * self.p.jammer_power)
                gamma[n] = dl_power[n] / (inter + cci + jam + self.noise)
        return gamma

    def dl_utility(
        self,
        dl_sub: np.ndarray,
        dl_power: np.ndarray,
        ul_sub: np.ndarray,
        p_ul: np.ndarray,
    ) -> float:
        """``U^dl = sum_n' R^dl_n' / R_const``."""
        if self.n_dl == 0:
            return 0.0
        gamma = self.dl_sinr(dl_sub, dl_power, ul_sub, p_ul)
        rates = self.p.subchannel_bw * np.log2(1.0 + gamma)
        return float(np.sum(rates) / self.p.rate_scaling)

    def dl_objective(
        self,
        dl_sub: np.ndarray,
        dl_power: np.ndarray,
        ul_sub: np.ndarray,
        p_ul: np.ndarray,
    ) -> float:
        """``U^dl - U^dl_pen`` -- to be *maximised* (Algorithm 2)."""
        if self.n_dl == 0:
            return 0.0
        util = self.dl_utility(dl_sub, dl_power, ul_sub, p_ul)
        pen = 0.0
        nu = self.p.penalty_dl

        # g_8: per-SBS power budget
        for m in range(self.m_dl):
            k, served = dl_sub[m], self.dl_ue_of_cell[m]
            if k < 0 or served.size == 0:
                continue
            g8 = float(dl_power[served] @ self.w_norm2[served, k]) - self.sbs_budget[m]
            if g8 > 0:
                pen += nu * g8 ** 2
        # g_7: non-negative powers
        g7 = -dl_power
        pen += nu * float(np.sum(np.where(g7 > 0, g7 ** 2, 0.0)))
        # g_9: SINR threshold for successful broadcasting
        gamma = self.dl_sinr(dl_sub, dl_power, ul_sub, p_ul)
        g9 = self.p.dl_sinr_threshold - gamma
        pen += nu * float(np.sum(np.where(g9 > 0, g9 ** 2, 0.0)))
        return util - pen

    def dl_utility_upper_bound(self, dl_sub: np.ndarray) -> float:
        """Interference-free, full-budget bound on ``U^dl`` used by Lemma 1/2.

        Allocating the whole budget of a cell to a single UE and dropping every
        interference term upper-bounds each individual rate, hence their sum.
        """
        if self.n_dl == 0:
            return 0.0
        total = 0.0
        for m in range(self.m_dl):
            k, served = dl_sub[m], self.dl_ue_of_cell[m]
            if k < 0 or served.size == 0:
                continue
            w2 = np.maximum(self.w_norm2[served, k], 1e-30)
            p_top = self.sbs_budget[m] / w2
            total += float(np.sum(self.p.subchannel_bw * np.log2(1.0 + p_top / self.noise)))
        return total / self.p.rate_scaling

    def dl_power_bounds(self, dl_sub: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Box constraints for the DL-TPC search agents (one variable per DL UE)."""
        lo = np.zeros(self.n_dl)
        hi = np.zeros(self.n_dl)
        for m in range(self.m_dl):
            k, served = dl_sub[m], self.dl_ue_of_cell[m]
            if k < 0 or served.size == 0:
                continue
            w2 = np.maximum(self.w_norm2[served, k], 1e-30)
            hi[served] = self.sbs_budget[m] / w2
            lo[served] = np.minimum(self.p.sbs_power_min, 1e-3 * hi[served])
        return lo, np.maximum(hi, lo + 1e-18)

    def equal_split_dl_power(self, dl_sub: np.ndarray) -> np.ndarray:
        """Feasible reference allocation: budget shared equally inside each cell."""
        q = np.zeros(self.n_dl)
        for m in range(self.m_dl):
            k, served = dl_sub[m], self.dl_ue_of_cell[m]
            if k < 0 or served.size == 0:
                continue
            w2 = np.maximum(self.w_norm2[served, k], 1e-30)
            q[served] = self.sbs_budget[m] / (served.size * w2)
        return q

    # ------------------------------------------------------------------ #
    # overall system utility
    # ------------------------------------------------------------------ #
    def fixed_part(self, assoc: np.ndarray, dl_util: float) -> float:
        """``F(A_net)``: an upper bound on ``U`` for every ``(p, F, rho, chi)``."""
        ul_sub, _ = self.decode(assoc)
        return self.ul_utility_upper_bound(ul_sub) + dl_util - self.association_penalty(assoc)

    def ul_state(
        self, assoc: np.ndarray, p_ul: np.ndarray, dl_power: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """``(rates, rho, chi, F)`` implied by an association and the powers."""
        ul_sub, dl_sub = self.decode(assoc)
        xi = self.cochannel_at_sbs(dl_sub, dl_power)
        rates = self.ul_rates(ul_sub, p_ul, xi, dl_sub)
        rho, chi, f_alloc = self.iscc_allocate(ul_sub, rates, p_ul)
        return rates, rho, chi, f_alloc

    def utility(
        self,
        assoc: np.ndarray,
        p_ul: np.ndarray,
        dl_power: np.ndarray,
    ) -> float:
        """Total system utility ``U = sum_n U^ul_n + sum_n' U^dl_n'``.

        Local UEs contribute too: with ``rho_n = 0`` they still trade accuracy
        against local compute through ``chi_n``.
        """
        ul_sub, dl_sub = self.decode(assoc)
        u_ul = 0.0
        if self.n_ul:
            rates, rho, chi, f_alloc = self.ul_state(assoc, p_ul, dl_power)
            u_ul = float(np.sum(self.ul_utilities(rho, chi, rates, f_alloc, p_ul)))
        return u_ul + self.dl_utility(dl_sub, dl_power, ul_sub, p_ul)

    # ---- diagnostics used by the figures ------------------------------ #
    def per_ue_time_energy(
        self, assoc: np.ndarray, p_ul: np.ndarray, dl_power: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Actual delay ``T_n`` (eq. Tn) and energy ``E_n`` (eq. En) of every UL UE."""
        rates, rho, chi, f_alloc = self.ul_state(assoc, p_ul, dl_power)
        delta, theta = self._branches(rho, rates, f_alloc, p_ul)
        return self.t_sen + self.t_ext + chi * delta, self.e_sen + self.e_ext + chi * theta
