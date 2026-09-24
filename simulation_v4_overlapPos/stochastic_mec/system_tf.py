"""TensorFlow batched implementation of the SI-TNTN UAV-MEC ISCC system model.

Vectorizes signal calculations, SINR, ISCC closed-form coordinate updates,
and objective functions across batches of search agents and realizations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import tensorflow as tf

from . import channel
from .config import SystemParams
from .network import Topology


@dataclass
class SolutionTF:
    """Everything Algorithm 3 returns for one or more time blocks."""

    utility: float
    assoc: np.ndarray            # (N_ul + M_dl, K) binary
    ul_power: np.ndarray         # (N_ul,)
    dl_power: np.ndarray         # (N_dl,)
    server_alloc: np.ndarray     # (N_ul,) F_nm
    curve: np.ndarray            # BWOA convergence curve
    runtime: float = 0.0
    n_inner_calls: int = 0
    rho: np.ndarray | None = None
    chi: np.ndarray | None = None


class SystemModelTF:
    """TensorFlow-accelerated signal model + objective functions."""

    def __init__(self, topo: Topology, params: SystemParams, rng: np.random.Generator):
        self.topo = topo
        self.p = params
        self.rng = rng

        self.K = params.n_subchannels
        self.L = params.n_antennas
        self.noise = float(params.noise_power)
        self.ul_noise = float(self.L * self.noise)

        # Index bookkeeping
        self.ul_cells = topo.ul_cells
        self.dl_cells = topo.dl_cells
        self.ul_ues = topo.ul_ues
        self.dl_ues = topo.dl_ues
        self.n_ul, self.n_dl = int(self.ul_ues.size), int(self.dl_ues.size)
        self.m_ul, self.m_dl = int(self.ul_cells.size), int(self.dl_cells.size)

        cell_of = topo.ue_cell
        ul_lookup = {int(c): i for i, c in enumerate(self.ul_cells)}
        dl_lookup = {int(c): i for i, c in enumerate(self.dl_cells)}
        self.ul_serving_np = np.array([ul_lookup[int(cell_of[n])] for n in self.ul_ues], dtype=np.int32)
        self.dl_serving_np = np.array([dl_lookup[int(cell_of[n])] for n in self.dl_ues], dtype=np.int32)
        self.dl_ue_of_cell = [np.flatnonzero(self.dl_serving_np == m) for m in range(self.m_dl)]
        self.ul_ue_of_cell = [np.flatnonzero(self.ul_serving_np == m) for m in range(self.m_ul)]

        self.ul_serving_tf = tf.constant(self.ul_serving_np, dtype=tf.int32)
        self.dl_serving_tf = tf.constant(self.dl_serving_np, dtype=tf.int32)

        self._build_channels()
        self._build_tasks()

    # ------------------------------------------------------------------ #
    # Channel & Task Setup
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
        self.cross2_np = np.abs(gram) ** 2
        self.hnorm2_np = np.real(np.einsum("nnmk->nmk", gram)) if h_ul.size \
            else np.zeros((self.n_ul, self.m_ul, self.K))

        h_dl = channel.miso_channel(rng, dl_pos, dl_bs, p, link="a2g") if self.n_dl and self.m_dl \
            else np.zeros((self.n_dl, self.m_dl, self.K, self.L), dtype=complex)
        self.beamformers = channel.zero_forcing_beamformers(h_dl, self.dl_ue_of_cell)

        # Precompute beamformer gains and norm squared
        self.dl_gain_list = []
        for j in range(self.m_dl):
            w = self.beamformers[j]                            # (K, L, N_j)
            if w.shape[2] == 0:
                self.dl_gain_list.append(np.zeros((self.n_dl, self.K, 0), dtype=np.float32))
                continue
            g = np.einsum("nkl,kli->nki", h_dl[:, j, :, :].conj(), w)
            self.dl_gain_list.append(np.abs(g).astype(np.float32) ** 2)

        self.w_norm2_np = np.zeros((self.n_dl, self.K), dtype=np.float32)
        for j in range(self.m_dl):
            idx = self.dl_ue_of_cell[j]
            if idx.size == 0:
                continue
            w = self.beamformers[j]                            # (K, L, N_j)
            self.w_norm2_np[idx] = np.sum(np.abs(w) ** 2, axis=1).T

        self.ue2ue2_np = (np.abs(channel.siso_channel(rng, ul_pos, dl_pos, p, link="g2g")) ** 2).astype(np.float32) \
            if self.n_ul and self.n_dl else np.zeros((self.n_ul, self.n_dl, self.K), dtype=np.float32)
        
        if self.m_dl and self.m_ul:
            g_bs = channel.mimo_channel(rng, dl_bs, ul_bs, p, link="a2g")
            self.bs2bs2_np = np.sum(np.abs(g_bs) ** 2, axis=-1).astype(np.float32)
        else:
            self.bs2bs2_np = np.zeros((self.m_dl, self.m_ul, self.K, self.L), dtype=np.float32)

        jam_pos = topo.jammer_pos
        self.n_jam = int(jam_pos.shape[0])
        if self.n_jam and self.n_ul and self.m_ul:
            h_j = channel.miso_channel(rng, jam_pos, ul_bs, p, link="g2a")
            self.jam_ul_cross2_np = np.abs(np.einsum("nmkl,qmkl->nqmk", h_ul.conj(), h_j)) ** 2
        else:
            self.jam_ul_cross2_np = np.zeros((self.n_ul, self.n_jam, self.m_ul, self.K), dtype=np.float32)
        if self.n_jam and self.n_dl:
            self.jam_dl2_np = (np.abs(channel.siso_channel(rng, jam_pos, dl_pos, p, link="g2g")) ** 2).astype(np.float32)
        else:
            self.jam_dl2_np = np.zeros((self.n_jam, self.n_dl, self.K), dtype=np.float32)
        self.jam_active_np = rng.random((self.n_jam, self.K)) < p.jam_prob if self.n_jam \
            else np.zeros((0, self.K), dtype=bool)

        # Convert static channel matrices to TF Constants (float32 / float64)
        self.cross2_tf = tf.constant(self.cross2_np, dtype=tf.float32)
        self.hnorm2_tf = tf.constant(self.hnorm2_np, dtype=tf.float32)
        self.w_norm2_tf = tf.constant(self.w_norm2_np, dtype=tf.float32)
        self.ue2ue2_tf = tf.constant(self.ue2ue2_np, dtype=tf.float32)
        self.bs2bs2_tf = tf.constant(self.bs2bs2_np, dtype=tf.float32)
        self.jam_ul_cross2_tf = tf.constant(self.jam_ul_cross2_np, dtype=tf.float32)
        self.jam_dl2_tf = tf.constant(self.jam_dl2_np, dtype=tf.float32)
        self.jam_active_tf = tf.constant(self.jam_active_np, dtype=tf.bool)

    def _build_tasks(self) -> None:
        p = self.p
        f_loc = self.topo.local_cpu[self.ul_ues].astype(np.float32)
        kappa = float(p.energy_coefficient)
        self.f_local_np = f_loc
        self.d_raw = float(p.data_size)
        self.c_raw = float(p.cpu_cycles)
        self.t_cmp_np = (p.cpu_cycles / f_loc).astype(np.float32)
        self.e_cmp_np = (kappa * p.cpu_cycles * f_loc ** 2).astype(np.float32)

        self.t_sen_np = np.full(self.n_ul, p.sensing_time, dtype=np.float32)
        self.e_sen_np = np.full(self.n_ul, p.sensing_power * p.sensing_time, dtype=np.float32)
        ext_cycles = float(p.extract_cycles_per_bit * p.data_size)
        self.t_ext_np = (ext_cycles / f_loc).astype(np.float32)
        self.e_ext_np = (kappa * ext_cycles * f_loc ** 2).astype(np.float32)
        self.t_ref_np = self.t_sen_np + self.t_ext_np + self.t_cmp_np
        self.e_ref_np = self.e_sen_np + self.e_ext_np + self.e_cmp_np

        snr_db = p.sensing_snr_db + p.sensing_snr_std_db * self.rng.standard_normal(self.n_ul)
        self.sens_info_np = np.log2(1.0 + 10.0 ** (snr_db / 10.0)).astype(np.float32)
        self.vartheta_np = np.full(self.n_ul, p.accuracy_sensitivity, dtype=np.float32)
        self.lam_th = float(p.accuracy_threshold)
        self.chi_min_np = (-np.log(1.0 - self.lam_th) / (self.vartheta_np * self.sens_info_np)).astype(np.float32)
        self.admissible_np = self.chi_min_np <= 1.0

        self.beta_t_np = np.full(self.n_ul, p.beta_time, dtype=np.float32)
        self.beta_e_np = np.full(self.n_ul, p.beta_energy, dtype=np.float32)
        self.beta_a_np = np.full(self.n_ul, p.beta_acc, dtype=np.float32)
        self.u_const_np = (self.beta_t_np * (self.t_ref_np - self.t_sen_np - self.t_ext_np) / self.t_ref_np
                           + self.beta_e_np * (self.e_ref_np - self.e_sen_np - self.e_ext_np) / self.e_ref_np).astype(np.float32)

        self.server_capacity_np = np.full(self.m_ul, p.server_capacity, dtype=np.float32)
        self.sbs_budget_np = np.full(self.m_dl, p.sbs_power_budget, dtype=np.float32)

        # TF constants
        self.t_cmp_tf = tf.constant(self.t_cmp_np, dtype=tf.float32)
        self.e_cmp_tf = tf.constant(self.e_cmp_np, dtype=tf.float32)
        self.t_ref_tf = tf.constant(self.t_ref_np, dtype=tf.float32)
        self.e_ref_tf = tf.constant(self.e_ref_np, dtype=tf.float32)
        self.sens_info_tf = tf.constant(self.sens_info_np, dtype=tf.float32)
        self.vartheta_tf = tf.constant(self.vartheta_np, dtype=tf.float32)
        self.chi_min_tf = tf.constant(self.chi_min_np, dtype=tf.float32)
        self.admissible_tf = tf.constant(self.admissible_np, dtype=tf.bool)
        self.beta_t_tf = tf.constant(self.beta_t_np, dtype=tf.float32)
        self.beta_e_tf = tf.constant(self.beta_e_np, dtype=tf.float32)
        self.beta_a_tf = tf.constant(self.beta_a_np, dtype=tf.float32)
        self.u_const_tf = tf.constant(self.u_const_np, dtype=tf.float32)
        self.server_capacity_tf = tf.constant(self.server_capacity_np, dtype=tf.float32)
        self.sbs_budget_tf = tf.constant(self.sbs_budget_np, dtype=tf.float32)

    @property
    def assoc_shape(self) -> tuple[int, int]:
        return (self.n_ul + self.m_dl, self.K)

    # ------------------------------------------------------------------ #
    # Association helpers (Batched)
    # ------------------------------------------------------------------ #
    def split_tf(self, assoc: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        """assoc shape (..., n_ul + m_dl, K) -> (assoc_ul, assoc_dl)."""
        return assoc[..., :self.n_ul, :], assoc[..., self.n_ul:, :]

    def decode_tf(self, assoc: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        """Returns ul_sub (..., n_ul) and dl_sub (..., m_dl) as int32."""
        a_ul, a_dl = self.split_tf(assoc)
        if self.n_ul > 0:
            any_ul = tf.reduce_any(a_ul > 0, axis=-1)
            argmax_ul = tf.cast(tf.argmax(a_ul, axis=-1), tf.int32)
            ul_sub = tf.where(any_ul, argmax_ul, -1)
            # Inadmissible mask
            ul_sub = tf.where(self.admissible_tf, ul_sub, -1)
        else:
            ul_sub = tf.zeros(tf.concat([tf.shape(assoc)[:-2], [0]], axis=0), dtype=tf.int32)

        if self.m_dl > 0:
            any_dl = tf.reduce_any(a_dl > 0, axis=-1)
            argmax_dl = tf.cast(tf.argmax(a_dl, axis=-1), tf.int32)
            dl_sub = tf.where(any_dl, argmax_dl, -1)
        else:
            dl_sub = tf.zeros(tf.concat([tf.shape(assoc)[:-2], [0]], axis=0), dtype=tf.int32)

        return ul_sub, dl_sub

    def association_penalty_tf(self, assoc: tf.Tensor) -> tf.Tensor:
        """assoc shape (..., n_ul + m_dl, K) -> penalty (...,)."""
        a_ul, a_dl = self.split_tf(assoc)
        pen = tf.zeros(tf.shape(assoc)[:-2], dtype=tf.float32)
        if self.n_ul > 0:
            g2 = tf.reduce_sum(tf.cast(a_ul, tf.float32), axis=-1) - 1.0
            pen_ul = self.p.penalty_assoc * tf.reduce_sum(tf.square(tf.maximum(g2, 0.0)), axis=-1)
            pen = pen + pen_ul
        if self.m_dl > 0:
            g3 = tf.abs(tf.reduce_sum(tf.cast(a_dl, tf.float32), axis=-1) - 1.0)
            pen_dl = self.p.penalty_assoc * tf.reduce_sum(tf.square(g3), axis=-1)
            pen = pen + pen_dl
        return pen

    # ------------------------------------------------------------------ #
    # Uplink Signal Model (Batched TF)
    # ------------------------------------------------------------------ #
    def cochannel_at_sbs_tf(self, dl_sub: tf.Tensor, dl_power: tf.Tensor) -> tf.Tensor:
        """dl_sub: (..., M_dl), dl_power: (..., N_dl) -> Xi: (..., K, M_ul)."""
        if self.m_dl == 0 or self.m_ul == 0:
            batch_shape = tf.shape(dl_power)[:-1]
            return tf.zeros(tf.concat([batch_shape, [self.K, self.m_ul]], axis=0), dtype=tf.float32)

        # Compute per-antenna power at DL SBSs: (..., M_dl, K, L)
        p_ant_list = []
        for m in range(self.m_dl):
            served = self.dl_ue_of_cell[m]
            if served.size == 0:
                p_ant_list.append(tf.zeros(tf.concat([tf.shape(dl_power)[:-1], [self.K, self.L]], axis=0), dtype=tf.float32))
                continue
            w = self.beamformers[m]  # (K, L, N_m)
            w_sq = tf.constant(np.abs(w) ** 2, dtype=tf.float32)  # (K, L, N_m)
            # dl_power subset: (..., N_m)
            q_m = tf.gather(dl_power, served, axis=-1)
            # p_m_k_l = einsum('...i, kli -> ...kl', q_m, w_sq)
            p_ant_m = tf.einsum("...i,kli->...kl", q_m, w_sq)
            
            # Mask out subchannels not matching dl_sub[..., m]
            k_m = dl_sub[..., m]  # (...,)
            k_mask = tf.one_hot(k_m, depth=self.K, dtype=tf.float32)  # (..., K)
            k_mask = tf.expand_dims(k_mask, axis=-1)  # (..., K, 1)
            p_ant_list.append(p_ant_m * k_mask)

        p_ant = tf.stack(p_ant_list, axis=-3)  # (..., M_dl, K, L)
        # bs2bs2: (M_dl, M_ul, K, L)
        # xi: (..., K, M_ul) = einsum('...dkl, dukl -> ...ku', p_ant, bs2bs2)
        xi = tf.einsum("...dkl,dukl->...ku", p_ant, self.bs2bs2_tf)
        return xi

    def ul_sinr_tf(
        self,
        ul_sub: tf.Tensor,
        p_ul: tf.Tensor,
        xi: tf.Tensor,
        dl_sub: tf.Tensor | None = None,
    ) -> tf.Tensor:
        """Batched UL SINR calculation: (..., N_ul)."""
        if self.n_ul == 0 or self.m_ul == 0:
            return tf.zeros(tf.shape(p_ul), dtype=tf.float32)

        batch_shape = tf.shape(p_ul)[:-1]
        n_ul = self.n_ul
        k_dim = self.K
        
        # Expand channel matrices for batched broadcasting
        # ul_sub: (..., N_ul)
        # own_gain: ||h_{n, m(n), k}||^2 for each user and subchannel
        # cross2_tf: (N, N, M, K)
        # hnorm2_tf: (N, M, K)
        
        # For each user n:
        sinr_list = []
        for n in range(n_ul):
            m = int(self.ul_serving_np[n])
            # ul_sub for user n: (...,)
            k_n = ul_sub[..., n]
            is_active = (k_n >= 0)
            k_safe = tf.maximum(k_n, 0)
            
            # own gain: (K,)
            own_all_k = self.hnorm2_tf[n, m, :]  # (K,)
            own = tf.gather(own_all_k, k_safe)   # (...,)
            
            # Interference from other users:
            # Weaker same-cell users on same k:
            i_w = tf.zeros(batch_shape, dtype=tf.float32)
            i_o = tf.zeros(batch_shape, dtype=tf.float32)
            
            for other in range(n_ul):
                if other == n:
                    continue
                m_other = int(self.ul_serving_np[other])
                same_k = (ul_sub[..., other] == k_n) & is_active
                p_other = p_ul[..., other]
                
                cross_all_k = self.cross2_tf[n, other, m, :]  # (K,)
                cross = tf.gather(cross_all_k, k_safe)        # (...,)
                inter_val = tf.where(same_k, p_other * cross, 0.0)
                
                if m_other == m:
                    # check if weaker:
                    weaker_mask = (self.hnorm2_np[other, m, :] <= self.hnorm2_np[n, m, :])
                    weaker_k = tf.gather(tf.constant(weaker_mask, dtype=tf.bool), k_safe)
                    i_w = i_w + tf.where(weaker_k, inter_val, 0.0)
                else:
                    i_o = i_o + inter_val

            # Jammer interference
            jam = tf.zeros(batch_shape, dtype=tf.float32)
            if self.n_jam > 0:
                # jam_ul_cross2_tf: (N, Q, M, K)
                # jam_active_tf: (Q, K)
                jam_p = float(self.p.jammer_power)
                jam_cross_k = tf.gather(self.jam_ul_cross2_tf[n, :, m, :], k_safe, axis=-1)  # (Q, ...)
                jam_act_k = tf.gather(self.jam_active_tf[:, :], k_safe, axis=-1)            # (Q, ...)
                jam_val = tf.reduce_sum(tf.where(jam_act_k, jam_cross_k * jam_p, 0.0), axis=0) # (...,)
                jam = tf.where(is_active, jam_val, 0.0)

            # Xi: (..., K, M_ul) -> gather k_safe at cell m
            xi_km = tf.gather(xi[..., :, m], k_safe, batch_dims=tf.rank(k_safe))

            if self.p.mf_noise_fix:
                den = i_w + i_o + own * xi_km / float(self.L) + jam + own * self.noise
            else:
                den = i_w + i_o + xi_km + jam + self.ul_noise

            p_n = p_ul[..., n]
            gamma_n = tf.where(is_active & (own > 0.0), p_n * tf.square(own) / tf.maximum(den, 1e-30), 0.0)
            sinr_list.append(gamma_n)

        return tf.stack(sinr_list, axis=-1)

    def ul_rates_tf(
        self,
        ul_sub: tf.Tensor,
        p_ul: tf.Tensor,
        xi: tf.Tensor,
        dl_sub: tf.Tensor | None = None,
    ) -> tf.Tensor:
        gamma = self.ul_sinr_tf(ul_sub, p_ul, xi, dl_sub)
        bw = float(self.p.subchannel_bw)
        return bw * tf.experimental.numpy.log2(1.0 + gamma)

    def sic_violation_tf(self, ul_sub: tf.Tensor, p_ul: tf.Tensor) -> tf.Tensor:
        """Batched SIC decodability violations: (...,)."""
        if self.n_ul < 2:
            return tf.zeros(tf.shape(p_ul)[:-1], dtype=tf.float32)

        eps = float(self.p.sic_threshold)
        total = tf.zeros(tf.shape(p_ul)[:-1], dtype=tf.float32)
        
        for n in range(self.n_ul):
            m = int(self.ul_serving_np[n])
            k_n = ul_sub[..., n]
            is_active = (k_n >= 0)
            k_safe = tf.maximum(k_n, 0)
            own = tf.gather(self.hnorm2_tf[n, m, :], k_safe)

            lhs = tf.zeros(tf.shape(total), dtype=tf.float32)
            for w in range(self.n_ul):
                if w == n or self.ul_serving_np[w] != m:
                    continue
                same_k = (ul_sub[..., w] == k_n) & is_active
                weaker_mask = (self.hnorm2_np[w, m, :] <= self.hnorm2_np[n, m, :])
                weaker_k = tf.gather(tf.constant(weaker_mask, dtype=tf.bool), k_safe)
                h_w = tf.gather(self.hnorm2_tf[w, m, :], k_safe)
                lhs = lhs + tf.where(same_k & weaker_k, p_ul[..., w] * h_w, 0.0)

            g5 = eps * lhs - p_ul[..., n] * own
            total = total + tf.where(is_active & (lhs > 0.0), tf.square(tf.maximum(g5, 0.0)), 0.0)

        return total

    # ------------------------------------------------------------------ #
    # Downlink Signal Model (Batched TF)
    # ------------------------------------------------------------------ #
    def dl_sinr_tf(
        self,
        dl_sub: tf.Tensor,
        dl_power: tf.Tensor,
        ul_sub: tf.Tensor,
        p_ul: tf.Tensor,
    ) -> tf.Tensor:
        """Batched DL SINR: (..., N_dl)."""
        if self.n_dl == 0:
            return tf.zeros(tf.shape(dl_power), dtype=tf.float32)

        batch_shape = tf.shape(dl_power)[:-1]
        sinr_list = []

        for n in range(self.n_dl):
            m = int(self.dl_serving_np[n])
            k_m = dl_sub[..., m]
            is_active = (k_m >= 0)
            k_safe = tf.maximum(k_m, 0)

            # Inter-cell interference from other DL SBSs j != m broadcasting on k_m
            inter = tf.zeros(batch_shape, dtype=tf.float32)
            for j in range(self.m_dl):
                if j == m:
                    continue
                served_j = self.dl_ue_of_cell[j]
                if served_j.size == 0:
                    continue
                same_k = (dl_sub[..., j] == k_m) & is_active
                # dl_gain: (N_dl, K, N_j)
                gain_j = tf.gather(self.dl_gain_list[j][n], k_safe) # (..., N_j)
                q_j = tf.gather(dl_power, served_j, axis=-1)        # (..., N_j)
                inter_j = tf.reduce_sum(gain_j * q_j, axis=-1)
                inter = inter + tf.where(same_k, inter_j, 0.0)

            # CCI from UL UEs transmitting on k_m
            cci = tf.zeros(batch_shape, dtype=tf.float32)
            if self.n_ul > 0:
                for u in range(self.n_ul):
                    ul_on_k = (ul_sub[..., u] == k_m) & is_active
                    g_u = tf.gather(self.ue2ue2_tf[u, n, :], k_safe)
                    cci = cci + tf.where(ul_on_k, p_ul[..., u] * g_u, 0.0)

            # Jammer
            jam = tf.zeros(batch_shape, dtype=tf.float32)
            if self.n_jam > 0:
                jam_p = float(self.p.jammer_power)
                jam_g = tf.gather(self.jam_dl2_tf[:, n, :], k_safe, axis=-1) # (Q, ...)
                jam_act = tf.gather(self.jam_active_tf[:, :], k_safe, axis=-1) # (Q, ...)
                jam_val = tf.reduce_sum(tf.where(jam_act, jam_g * jam_p, 0.0), axis=0)
                jam = tf.where(is_active, jam_val, 0.0)

            den = inter + cci + jam + self.noise
            q_n = dl_power[..., n]
            gamma_n = tf.where(is_active, q_n / tf.maximum(den, 1e-30), 0.0)
            sinr_list.append(gamma_n)

        return tf.stack(sinr_list, axis=-1)

    def dl_objective_tf(
        self,
        dl_sub: tf.Tensor,
        dl_power: tf.Tensor,
        ul_sub: tf.Tensor,
        p_ul: tf.Tensor,
    ) -> tf.Tensor:
        """dl_objective to be MAXIMISED: (...,)."""
        if self.n_dl == 0:
            return tf.zeros(tf.shape(dl_power)[:-1], dtype=tf.float32)

        gamma = self.dl_sinr_tf(dl_sub, dl_power, ul_sub, p_ul)
        rates = float(self.p.subchannel_bw) * tf.experimental.numpy.log2(1.0 + gamma)
        util = tf.reduce_sum(rates, axis=-1) / float(self.p.rate_scaling)

        pen = tf.zeros(tf.shape(util), dtype=tf.float32)
        nu = float(self.p.penalty_dl)

        # g8: SBS power budget
        for m in range(self.m_dl):
            served = self.dl_ue_of_cell[m]
            if served.size == 0:
                continue
            k_m = dl_sub[..., m]
            is_active = (k_m >= 0)
            k_safe = tf.maximum(k_m, 0)
            # w_norm2: (N_dl, K)
            w2_served = tf.gather(self.w_norm2_tf, served, axis=0) # (N_m, K)
            # gather k_safe:
            w2_k = tf.gather(w2_served, k_safe, axis=-1) # (N_m, ...)
            w2_k = tf.transpose(w2_k, perm=tf.concat([tf.range(1, tf.rank(w2_k)), [0]], axis=0)) # (..., N_m)
            q_served = tf.gather(dl_power, served, axis=-1) # (..., N_m)
            p_tot = tf.reduce_sum(q_served * w2_k, axis=-1)
            g8 = p_tot - float(self.sbs_budget_np[m])
            pen = pen + tf.where(is_active, nu * tf.square(tf.maximum(g8, 0.0)), 0.0)

        # g7: non-negative powers
        pen = pen + nu * tf.reduce_sum(tf.square(tf.maximum(-dl_power, 0.0)), axis=-1)

        # g9: SINR threshold
        g9 = float(self.p.dl_sinr_threshold) - gamma
        pen = pen + nu * tf.reduce_sum(tf.square(tf.maximum(g9, 0.0)), axis=-1)

        return util - pen

    # ------------------------------------------------------------------ #
    # ISCC: Closed Form & Utility (Props 1-3)
    # ------------------------------------------------------------------ #
    def accuracy_tf(self, chi: tf.Tensor) -> tf.Tensor:
        return 1.0 - tf.exp(-self.vartheta_tf * chi * self.sens_info_tf)

    def eta_theta_tf(self, rho: tf.Tensor, chi: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        scale = chi * rho * self.d_raw
        eta = self.beta_t_tf * scale / self.t_ref_tf
        theta = self.beta_e_tf * scale / (float(self.p.amplifier_efficiency) * self.e_ref_tf)
        return eta, theta

    def _branches_tf(
        self, rho: tf.Tensor, rates: tf.Tensor, f_alloc: tf.Tensor, p_ul: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor]:
        off = rho > 0.0
        r = tf.where(off, tf.maximum(rates, 1e-12), 1e30)
        f = tf.where(off, tf.maximum(f_alloc, 1e-12), 1e30)
        d_loc = (1.0 - rho) * self.t_cmp_tf
        d_ofl = tf.where(off, rho * (self.d_raw / r + self.c_raw / f), 0.0)
        theta = (1.0 - rho) * self.e_cmp_tf + tf.where(
            off, rho * p_ul * self.d_raw / (float(self.p.amplifier_efficiency) * r), 0.0)
        return tf.maximum(d_loc, d_ofl), theta

    def _omega_tf(self, delta: tf.Tensor, theta: tf.Tensor) -> tf.Tensor:
        return self.beta_t_tf * delta / self.t_ref_tf + self.beta_e_tf * theta / self.e_ref_tf

    def optimal_chi_tf(self, omega: tf.Tensor) -> tf.Tensor:
        vl = self.vartheta_tf * self.sens_info_tf
        lo = tf.minimum(self.chi_min_tf, 1.0)
        psi = (1.0 - self.lam_th) * omega / (self.beta_a_tf * vl + 1e-30)
        chi = tf.where(psi > 0.0, -tf.math.log(tf.maximum(psi, 1e-30)) / vl, 1.0)
        chi = tf.where(self.beta_a_tf > 0.0, chi, lo)
        return tf.clip_by_value(chi, lo, 1.0)

    def rho_star_tf(self, rates: tf.Tensor, f_alloc: tf.Tensor) -> tf.Tensor:
        r = tf.maximum(rates, 1e-12)
        f = tf.maximum(f_alloc, 1e-12)
        return self.t_cmp_tf / (self.t_cmp_tf + self.d_raw / r + self.c_raw / f)

    def optimal_split_tf(
        self, ul_sub: tf.Tensor, rates: tf.Tensor, f_alloc: tf.Tensor, p_ul: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor]:
        offloading = (ul_sub >= 0)
        rs = self.rho_star_tf(rates, f_alloc)
        
        # 3 candidate splits: 0, rs, 1
        c0 = tf.zeros_like(rs)
        c1 = rs
        c2 = tf.ones_like(rs)
        
        d0, t0 = self._branches_tf(c0, rates, f_alloc, p_ul)
        om0 = self._omega_tf(d0, t0)
        
        d1, t1 = self._branches_tf(c1, rates, f_alloc, p_ul)
        om1 = self._omega_tf(d1, t1)
        
        d2, t2 = self._branches_tf(c2, rates, f_alloc, p_ul)
        om2 = self._omega_tf(d2, t2)
        
        om_stack = tf.stack([om0, om1, om2], axis=-1)
        c_stack = tf.stack([c0, c1, c2], axis=-1)
        
        best_idx = tf.argmin(om_stack, axis=-1) # (..., N_ul)
        
        # Gather best rho
        rho_best = tf.gather(c_stack, tf.expand_dims(best_idx, -1), batch_dims=tf.rank(best_idx))
        rho_best = tf.squeeze(rho_best, axis=-1)
        rho = tf.where(offloading, rho_best, 0.0)
        
        delta, theta = self._branches_tf(rho, rates, f_alloc, p_ul)
        omega = self._omega_tf(delta, theta)
        chi = self.optimal_chi_tf(omega)
        return rho, chi

    def smca_tf(
        self,
        ul_sub: tf.Tensor,
        rho: tf.Tensor,
        chi: tf.Tensor,
        rates: tf.Tensor,
    ) -> tf.Tensor:
        """Prop 1 water-filling compute allocation: returns f_alloc (..., N_ul)."""
        if self.n_ul == 0:
            return tf.zeros(tf.concat([tf.shape(rho)[:-1], [0]], axis=0), dtype=tf.float32)

        offloading = (ul_sub >= 0) & (rho > 0.0)
        weight = self.beta_t_tf * rho * chi * self.c_raw / self.t_ref_tf
        root = tf.sqrt(tf.maximum(weight, 0.0))
        
        slack = (1.0 - rho) * self.t_cmp_tf - rho * self.d_raw / tf.maximum(rates, 1e-12)
        f_bar = tf.where(slack > 0.0, rho * self.c_raw / tf.maximum(slack, 1e-12), 1e30)

        f_alloc_list = [tf.zeros(tf.shape(rho)[:-1], dtype=tf.float32) for _ in range(self.n_ul)]

        for m in range(self.m_ul):
            members = self.ul_ue_of_cell[m]
            if members.size == 0:
                continue
            cap = float(self.server_capacity_np[m])
            s_n = tf.gather(root, members, axis=-1)
            fb = tf.gather(f_bar, members, axis=-1)
            off_m = tf.gather(offloading, members, axis=-1)
            
            s_n_eff = tf.where(off_m, s_n, 0.0)
            fb_eff = tf.where(off_m, fb, 0.0)

            # Bisection for water-filling multiplier t
            lo = tf.zeros(tf.shape(rho)[:-1], dtype=tf.float32)
            hi = tf.fill(tf.shape(rho)[:-1], cap / tf.maximum(tf.reduce_min(tf.where(s_n_eff > 0.0, s_n_eff, 1e30), axis=-1), 1e-12))
            
            for _ in range(30):
                mid = 0.5 * (lo + hi)
                alloc = tf.minimum(s_n_eff * tf.expand_dims(mid, -1), fb_eff)
                tot = tf.reduce_sum(alloc, axis=-1)
                lo = tf.where(tot < cap, mid, lo)
                hi = tf.where(tot >= cap, mid, hi)
            
            alloc_final = tf.minimum(s_n_eff * tf.expand_dims(hi, -1), fb_eff)
            for idx, user_idx in enumerate(members):
                f_alloc_list[user_idx] = alloc_final[..., idx]

        return tf.stack(f_alloc_list, axis=-1)

    def iscc_allocate_tf(
        self, ul_sub: tf.Tensor, rates: tf.Tensor, p_ul: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Runs coordinate updates for (rho, chi, F)."""
        rho = tf.cast(ul_sub >= 0, tf.float32)
        chi = tf.ones_like(rho)
        if self.n_ul == 0:
            empty = tf.zeros(tf.concat([tf.shape(rho)[:-1], [0]], axis=0), dtype=tf.float32)
            return rho, chi, empty
        
        f = self.smca_tf(ul_sub, rho, chi, rates)
        if not self.p.enable_iscc:
            return rho, chi, f

        for _ in range(max(1, self.p.iscc_rounds)):
            rho, chi = self.optimal_split_tf(ul_sub, rates, f, p_ul)
            f = self.smca_tf(ul_sub, rho, chi, rates)
            
        return rho, chi, f

    def ul_utilities_tf(
        self,
        rho: tf.Tensor,
        chi: tf.Tensor,
        rates: tf.Tensor,
        f_alloc: tf.Tensor,
        p_ul: tf.Tensor,
    ) -> tf.Tensor:
        delta, theta = self._branches_tf(rho, rates, f_alloc, p_ul)
        acc = self.accuracy_tf(chi)
        om = self._omega_tf(delta, theta)
        u = (self.u_const_tf - chi * om
             + self.beta_a_tf * (acc - self.lam_th) / (1.0 - self.lam_th))
        return tf.where(self.admissible_tf, u, 0.0)

    def utility_tf(
        self,
        assoc: tf.Tensor,
        p_ul: tf.Tensor,
        dl_power: tf.Tensor,
    ) -> tf.Tensor:
        """Total system utility: (...,)."""
        ul_sub, dl_sub = self.decode_tf(assoc)
        u_ul = tf.zeros(tf.shape(assoc)[:-2], dtype=tf.float32)

        if self.n_ul > 0:
            xi = self.cochannel_at_sbs_tf(dl_sub, dl_power)
            rates = self.ul_rates_tf(ul_sub, p_ul, xi, dl_sub)
            rho, chi, f_alloc = self.iscc_allocate_tf(ul_sub, rates, p_ul)
            u_ul_per_ue = self.ul_utilities_tf(rho, chi, rates, f_alloc, p_ul)
            u_ul = tf.reduce_sum(u_ul_per_ue, axis=-1)

        dl_gamma = self.dl_sinr_tf(dl_sub, dl_power, ul_sub, p_ul)
        dl_rates = float(self.p.subchannel_bw) * tf.experimental.numpy.log2(1.0 + dl_gamma)
        u_dl = tf.reduce_sum(dl_rates, axis=-1) / float(self.p.rate_scaling)

        return u_ul + u_dl

    def mpc_objective_tf(
        self,
        ul_sub: tf.Tensor,
        p_ul: tf.Tensor,
        xi: tf.Tensor,
        rho: tf.Tensor,
        chi: tf.Tensor,
    ) -> tf.Tensor:
        """MPC power control objective to be MINIMISED: (...,)."""
        offloading = (ul_sub >= 0)
        eta, theta = self.eta_theta_tf(rho, chi)
        rates = self.ul_rates_tf(ul_sub, p_ul, xi)
        rate_safe = tf.maximum(rates, 1e-12)

        w_terms = tf.where(offloading, (eta + theta * p_ul) / rate_safe, 0.0)
        w = tf.reduce_sum(w_terms, axis=-1)

        g4 = p_ul - float(self.p.p_max)
        pen_p = float(self.p.penalty_power) * tf.reduce_sum(tf.square(tf.maximum(g4, 0.0)), axis=-1)
        pen_sic = float(self.p.penalty_sic) * self.sic_violation_tf(ul_sub, p_ul)
        return w + pen_p + pen_sic

    def equal_split_dl_power_tf(self, dl_sub: tf.Tensor) -> tf.Tensor:
        """Equal split fallback DL power: (..., N_dl)."""
        if self.n_dl == 0:
            return tf.zeros(tf.concat([tf.shape(dl_sub)[:-1], [0]], axis=0), dtype=tf.float32)

        q_list = []
        for m in range(self.m_dl):
            served = self.dl_ue_of_cell[m]
            if served.size == 0:
                continue
            k_m = dl_sub[..., m]
            k_safe = tf.maximum(k_m, 0)
            # w_norm2: (N_dl, K)
            w2_served = tf.gather(self.w_norm2_tf, served, axis=0) # (N_m, K)
            w2_k = tf.gather(w2_served, k_safe, axis=-1) # (N_m, ...)
            w2_k = tf.transpose(w2_k, perm=tf.concat([tf.range(1, tf.rank(w2_k)), [0]], axis=0)) # (..., N_m)
            w2_k = tf.maximum(w2_k, 1e-30)
            q_m = float(self.sbs_budget_np[m]) / (float(served.size) * w2_k)
            q_list.append(q_m)

        return tf.concat(q_list, axis=-1) if q_list else tf.zeros(tf.concat([tf.shape(dl_sub)[:-1], [self.n_dl]], axis=0), dtype=tf.float32)

    def dl_power_bounds_tf(self, dl_sub: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        """Upper and lower bounds for DL powers: (..., N_dl)."""
        if self.n_dl == 0:
            z = tf.zeros(tf.concat([tf.shape(dl_sub)[:-1], [0]], axis=0), dtype=tf.float32)
            return z, z

        lo_list, hi_list = [], []
        for m in range(self.m_dl):
            served = self.dl_ue_of_cell[m]
            if served.size == 0:
                continue
            k_m = dl_sub[..., m]
            k_safe = tf.maximum(k_m, 0)
            w2_served = tf.gather(self.w_norm2_tf, served, axis=0)
            w2_k = tf.gather(w2_served, k_safe, axis=-1)
            w2_k = tf.transpose(w2_k, perm=tf.concat([tf.range(1, tf.rank(w2_k)), [0]], axis=0))
            w2_k = tf.maximum(w2_k, 1e-30)
            hi_m = float(self.sbs_budget_np[m]) / w2_k
            lo_m = tf.minimum(float(self.p.sbs_power_min), 1e-3 * hi_m)
            lo_list.append(lo_m)
            hi_list.append(hi_m)

        lo = tf.concat(lo_list, axis=-1) if lo_list else tf.zeros(tf.concat([tf.shape(dl_sub)[:-1], [self.n_dl]], axis=0), dtype=tf.float32)
        hi = tf.concat(hi_list, axis=-1) if hi_list else tf.zeros(tf.concat([tf.shape(dl_sub)[:-1], [self.n_dl]], axis=0), dtype=tf.float32)
        return lo, tf.maximum(hi, lo + 1e-18)
