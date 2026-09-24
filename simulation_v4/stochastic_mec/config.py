"""Configuration objects for the SI-TNTN UAV-MEC ISCC simulator (TensorFlow Parallel edition)."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Sequence

import numpy as np

DBM_TO_DB = 30.0


def db2lin(x: float | np.ndarray) -> float | np.ndarray:
    """Convert a dB quantity to linear scale."""
    return 10.0 ** (np.asarray(x, dtype=float) / 10.0)


@dataclass(frozen=True)
class SystemParams:
    """Physical / network parameters of the SI-TNTN UAV-MEC system."""

    area_side: float = 1000.0                      # [m]
    z_min: float = 0.0
    z_uav_min: float = 80.0
    z_uav_max: float = 120.0

    lambda_sbs_ul: float = 3e-6                    # [UAV / m^2]
    lambda_sbs_dl: float = 3e-6
    lambda_ue_active: float = 10e-6
    lambda_ue_inactive: float = 5e-6
    lambda_jammer: float = 2e-6

    n_subchannels: int = 5
    subchannel_bw: float = 1e6
    n_antennas: int = 4
    noise_psd_dbm_per_hz: float = -174.0
    carrier_ghz: float = 2.4

    g2a_alpha: float = 2.5
    g2a_d0: float = 1.0
    g2a_pl0: float = 40.0
    g2g_alpha: float = 3.5
    a2g_eta_db: float = 20.0
    nakagami_m: float = 2.0
    invgamma_shape: float = 3.0
    shadowing_std_db: float = 8.0
    small_scale_fading: bool = True
    pathloss_distance_in_km: bool = False
    pl_const: float = 22.7
    pl_freq_coeff: float = 26.0
    pl_dist_coeff: float = 36.7

    jammer_power: float = 0.10
    jam_prob: float = 0.8

    # ---- uplink UEs -----------------------------------------------------
    p_max: float = 0.25                            # [W]  (~24 dBm)
    p_min: float = 1e-8                            # [W]
    sic_threshold: float = 1.001                   # eps_th
    amplifier_efficiency: float = 1.0              # xi_n
    energy_coefficient: float = 5e-27              # kappa_n
    local_cpu_choices: Sequence[float] = (0.5e9, 0.8e9, 1.0e9)   # F_n^loc

    # ---- sensed tasks (Sec. II-C) -------------------------------------
    data_size: float = 420e3                       # D_n^raw [bit]
    cpu_cycles: float = 1.0e9                      # C_n^raw [cycles]
    beta_time: float = 1.0 / 3.0                   # beta_n^t
    beta_acc: float = 1.0 / 3.0                    # beta_n^a (beta_e = 1 - beta_t - beta_a)

    sensing_snr_db: float = 10.0                   # gamma_n^sen [dB]
    sensing_snr_std_db: float = 3.0                # per-UE spread of gamma_n^sen [dB]
    sensing_time: float = 5e-3                     # T_n^sen [s]
    sensing_power: float = 0.1                     # p_n^sen [W]
    accuracy_sensitivity: float = 1.0              # vartheta_n
    accuracy_threshold: float = 0.9                # Lambda^th
    extract_cycles_per_bit: float = 50.0           # varpi_n [cycles/bit]
    iscc_rounds: int = 3                           # block-coordinate rounds of Props. 1-3
    enable_iscc: bool = True                       # False -> ablation: same objective, rho = a_n^ul, chi = 1
    mf_noise_fix: bool = True

    # ---- MEC servers ----------------------------------------------------
    server_capacity: float = 8e9                   # F_m^max [cycles/s]

    # ---- downlink SBSs --------------------------------------------------
    sbs_power_budget: float = 39.81                # P_m^max [W] (46 dBm)
    sbs_power_min: float = 0.25e-3                 # [W]
    dl_sinr_threshold: float = 1.001               # gamma_th
    rate_scaling: float = 1e8                      # R_const [bit/s]

    # ---- penalty factors ------------------------------------------------
    penalty_power: float = 1e14                    # nu_n          (C: p <= p_max)
    penalty_sic: float = 1e14                      # lambda_mkn    (C: SIC)
    penalty_dl: float = 1e18                       # DL power / SINR constraints
    penalty_assoc: float = 1e14                    # BWOA association constraints

    def __post_init__(self) -> None:
        if not 0.0 <= self.beta_time <= 1.0:
            raise ValueError("beta_time must lie in [0, 1]")
        if not 0.0 <= self.beta_acc <= 1.0 - self.beta_time + 1e-12:
            raise ValueError("beta_acc must lie in [0, 1 - beta_time]")
        if not 0.0 <= self.accuracy_threshold < 1.0:
            raise ValueError("accuracy_threshold must lie in [0, 1)")

    # ---- derived quantities --------------------------------------------
    @property
    def area(self) -> float:
        return self.area_side ** 2

    @property
    def volume(self) -> float:
        return self.area_side ** 2 * (self.z_uav_max - self.z_min)

    @property
    def carrier_mhz(self) -> float:
        return self.carrier_ghz * 1e3

    @property
    def beta_energy(self) -> float:
        return max(0.0, 1.0 - self.beta_time - self.beta_acc)

    @property
    def sensing_info(self) -> float:
        """``L_n = log2(1 + gamma_n^sen)`` -- sensed information per retained feature."""
        return float(np.log2(1.0 + db2lin(self.sensing_snr_db)))

    @property
    def retention_floor(self) -> float:
        """``chi_n^min`` of eq. (chimin); > 1 means the UE is inadmissible."""
        return float(-np.log(1.0 - self.accuracy_threshold)
                     / (self.accuracy_sensitivity * self.sensing_info))

    @property
    def noise_power(self) -> float:
        """Thermal noise power in one sub-channel, B_k * N_0 [W]."""
        return db2lin(self.noise_psd_dbm_per_hz - DBM_TO_DB) * self.subchannel_bw

    def with_(self, **changes) -> "SystemParams":
        """Return a copy with the given fields replaced (for parameter sweeps)."""
        return replace(self, **changes)


@dataclass(frozen=True)
class AlgorithmParams:
    """Tuning knobs of the unified nature-inspired optimisation framework."""

    # Algorithm 3 -- BWOA over the joint association matrix A^net
    n_agents_bwoa: int = 30                        # S_3
    max_iter_bwoa: int = 120                       # I_3^max
    patience_bwoa: int = 25                        # I_c
    tol_bwoa: float = 1e-5                         # varepsilon
    cache_max_retries: int = 10                    # Max retries to perturb duplicate positions
    enable_cache: bool = True                      # Enable memoization / evaluation cache

    # Algorithms 1 & 2 -- continuous power control (WOA / IWOA / PSO)
    n_agents_tpc: int = 30                         # S_1, S_2
    max_iter_tpc: int = 120                        # I_1^max, I_2^max
    patience_tpc: int = 15
    tol_tpc: float = 1e-6

    # IWOA adaptive population
    pop_min: int = 20
    pop_max: int = 40
    iwoa_alpha: float = 0.25

    # PSO
    pso_inertia: float = 1.0
    pso_inertia_damp: float = 0.99
    pso_c1: float = 1.5
    pso_c2: float = 2.0

    # BWOA transfer function steepness (sigmoid slope)
    sigmoid_slope: float = 10.0

    # Lemma 1 / Lemma 2 early-stop pruning of the inner power problems
    use_early_stop: bool = True
    bisection_tol: float = 1e-6

    # Batching settings for TensorFlow execution
    batch_size: int = 16

    def with_(self, **changes) -> "AlgorithmParams":
        return replace(self, **changes)


DEFAULT_SYSTEM = SystemParams()
DEFAULT_ALGORITHM = AlgorithmParams()
