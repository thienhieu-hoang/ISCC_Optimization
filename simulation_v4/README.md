# SI-TNTN UAV-MEC ISCC Optimization Simulator (`simulation_v4`)

`simulation_v4` is a high-performance, **batched parallel TensorFlow implementation** of the Integrated Sensing, Communication, and Computation (ISCC) optimization framework for Space-Air-Ground Non-Terrestrial Networks (SI-TNTN UAV-MEC) under malicious jamming.

---

## Table of Contents
1. [Environment Setup & Prerequisites](#1-environment-setup--prerequisites)
2. [Quick Start & Verification](#2-quick-start--verification)
3. [Figure Reproduction Guide (Manuscript v4)](#3-figure-reproduction-guide-manuscript-v4)
4. [Running Custom Parameter Sweeps](#4-running-custom-parameter-sweeps)
5. [Architecture & Helper Module Guide (`stochastic_mec/`)](#5-architecture--helper-module-guide-stochastic_mec)
6. [Workflow & Module Interaction](#6-workflow--module-interaction)

---

## 1. Environment Setup & Prerequisites

All scripts are executed within the dedicated Conda environment **`TF_GPU-py3_11`** with TensorFlow (GPU/CPU accelerated), NumPy, SciPy, and Matplotlib.

Activate the environment or run commands directly using `conda run`:

```powershell
# Activate environment
conda activate TF_GPU-py3_11

# OR run directly via conda run:
conda run -n TF_GPU-py3_11 python <script_path>
```

---

## 2. Quick Start & Verification

### A. Quick Numerical Equivalence & Performance Test
Verify that the TensorFlow engine matches the reference mathematical model and benchmark execution speed:

```powershell
conda run -n TF_GPU-py3_11 python simulation_v4/scripts/compare_v3_v4.py
```

### B. Quick Smoke Test Sweep
Run a fast 3-realization sweep across active-UE density to verify the full optimization pipeline:

```powershell
conda run -n TF_GPU-py3_11 python simulation_v4/scripts/run_sweep.py ue-density --quick -r 3
```

---

## 3. Figure Reproduction Guide (Manuscript v4)

To reproduce the exact figures in the manuscript, execute the corresponding command below:

| Figure in Paper | Description | Execution Command | Helper / Output Scripts | Output Files |
| :--- | :--- | :--- | :--- | :--- |
| **Fig. 1** | **3D Network Topology Snapshot** (Spatial distribution of UEs, UL-UAVs, DL-UAVs, Jammers, Voronoi cells, G2A/A2G links) | `conda run -n TF_GPU-py3_11 python simulation_v4/scripts/fig_topology3d.py` | `network.py`, `channel.py` | `figures/topology_3d.pdf` |
| **Fig. 2** | **BWOA vs. Exhaustive Search**<br>• (a) Attained Utility vs. Active UEs<br>• (b) Execution Runtime vs. Active UEs | `conda run -n TF_GPU-py3_11 python simulation_v4/scripts/run_sweep.py exhaustive -r 100` | `solver_tf.py`, `optimizers_tf.py`, `plotting.py` | `figures/exhaustive_utility.pdf`, `figures/exhaustive_time.pdf` |
| **Fig. 3(a)** | **System Utility vs. Active-UE Density** (Comparing Proposed MF-SIC vs. Force-All-Offload, IOJOA, FDMA, ALCA) | `conda run -n TF_GPU-py3_11 python simulation_v4/scripts/run_sweep.py ue-density -r 100` | `experiments_tf.py`, `schemes.py`, `plotting.py` | `results/ue_density.json`, `figures/ue_density_su.pdf` |
| **Fig. 3(b)** | **System Utility vs. UL-UAV Server Density** (Comparing Proposed MF-SIC vs. baselines across server densities) | `conda run -n TF_GPU-py3_11 python simulation_v4/scripts/run_sweep.py uav-density -r 100` | `experiments_tf.py`, `schemes.py`, `plotting.py` | `results/uav_density.json`, `figures/uav_density_su.pdf` |
| **Fig. 4(a)** | **Accuracy–Delay Pareto Frontier** (Mean accuracy $\Lambda_n$ vs. Normalized delay $T_n/T_n^{\text{ref}}$ over preference weight $\beta^{\text{a}}$) | `conda run -n TF_GPU-py3_11 python simulation_v4/scripts/run_sweep.py accuracy-tradeoff -r 100` | `system_tf.py`, `fig_iscc.py`, `plotting.py` | `results/accuracy_weight.json`, `figures/accuracy_tradeoff.pdf` |
| **Fig. 4(b)** | **ISCC Retention & Offloading vs. Sensing SNR** (Optimal retention $\chi_n^\star$, floor $\chi_n^{\min}$, and offload split $\rho_n^\star$ vs. $\bar{\gamma}^{\text{sen}}$) | `conda run -n TF_GPU-py3_11 python simulation_v4/scripts/run_sweep.py sensing-snr -r 100` | `system_tf.py`, `fig_iscc.py`, `plotting.py` | `results/sensing_snr.json`, `figures/retention_vs_snr.pdf` |

### Automated Batch Run (All Figures in One Command)
To execute all parameter sweeps and render all paper figures sequentially:

```powershell
# Full run (100 realizations per point, parallel execution)
conda run -n TF_GPU-py3_11 python simulation_v4/scripts/run_all.py -r 100 -j 4

# Fast run (fewer realizations for quick verification)
conda run -n TF_GPU-py3_11 python simulation_v4/scripts/run_all.py --quick
```

---

## 4. Running Custom Parameter Sweeps

The CLI `scripts/run_sweep.py` provides independent subcommands for granular sensitivity studies:

```powershell
# Sweep jammer density:
conda run -n TF_GPU-py3_11 python simulation_v4/scripts/run_sweep.py jammer-density -r 50

# Sweep task raw data size Dn:
conda run -n TF_GPU-py3_11 python simulation_v4/scripts/run_sweep.py datasize -r 50

# Sweep computation task workload Cn:
conda run -n TF_GPU-py3_11 python simulation_v4/scripts/run_sweep.py taskload -r 50

# Sweep UAV MEC server computational capacity Fm^max:
conda run -n TF_GPU-py3_11 python simulation_v4/scripts/run_sweep.py capacity -r 50

# Sweep UE maximum transmit power budget Pn^max:
conda run -n TF_GPU-py3_11 python simulation_v4/scripts/run_sweep.py power -r 50

# Sweep time vs. energy preference weights (beta_t vs beta_e):
conda run -n TF_GPU-py3_11 python simulation_v4/scripts/run_sweep.py preference -r 50
```

---

## 5. Architecture & Helper Module Guide (`stochastic_mec/`)

The simulation backend is located in `stochastic_mec/`. Below is the role and purpose of each core engine module:

```
stochastic_mec/
├── config.py           <- Physical, network, and algorithmic parameters dataclasses
├── network.py          <- 3D Homogeneous Poisson Point Process (HPPP) node generator
├── channel.py          <- 3D Pathloss, Nakagami-m fading, and ZFBF precoding matrices
├── system_tf.py        <- Core TensorFlow vectorized system model & closed-form ISCC updates
├── optimizers_tf.py    <- Batched continuous (WOA/PSO/IWOA) & binary (BWOA) swarm metaheuristics
├── solver_tf.py        <- Hybrid TPC-BWOA Algorithm 1 with Lemma 2 feasibility pruning
├── experiments_tf.py   <- Monte Carlo parallel sweep runner & snapshot evaluator
├── metrics.py          <- Post-simulation metrics aggregation & statistics
├── schemes.py          <- Baseline access schemes (MF-SIC, Force-All-Offload, IOJOA, FDMA, ALCA)
└── plotting.py         <- Publication-grade Matplotlib/Seaborn IEEE styling utilities
```

### Module Breakdown

#### 1. `config.py` — Parameter & Configuration Management
- **Role:** Defines system dataclasses (`SystemParams`, `AlgorithmParams`, `SweepResult`).
- **Purpose:** Centralizes all physical parameters (carrier frequencies $f_c$, bandwidths $B_k$, noise power $\sigma_0^2$, self-interference leakage $\chi_{\text{SI}}$, HPPP intensities $\lambda_0, \lambda_{\text{UE}}, \lambda_{\text{Q}}$, UAV altitude $H$, pathloss exponents, and ISCC parameters $\beta_n^{\text{t}}, \beta_n^{\text{e}}, \beta_n^{\text{a}}, \xi_0, \kappa_a, \alpha_a$).

#### 2. `network.py` — Spatial 3D HPPP Topology Generation
- **Role:** Generates spatial coordinates for all nodes in the SI-TNTN network.
- **Purpose:** Samples 3D Poisson point processes for Ground UEs $\vec{\ell}_n$, Uplink UAV MEC Servers $\vec{\ell}_m$, Downlink UAVs $\vec{\ell}_{m'}$, and Malicious Jammers $\vec{\ell}_q$. Establishes nearest-server Voronoi associations and Euclidean distance matrices.

#### 3. `channel.py` — Channel Propagation & Beamforming
- **Role:** Computes 3D wireless propagation channels and Zero-Forcing Beamforming (ZFBF).
- **Purpose:** Models elevation-dependent Line-of-Sight (LoS) probabilities, pathloss, small-scale Nakagami-$m$ fading, and inverse-Gamma shadowing for G2A (uplink), A2G (downlink), and G2G (jamming) links. Computes pseudoinverse Zero-Forcing transmit precoding matrices $\vec{W}$.

#### 4. `system_tf.py` — TensorFlow Vectorized System Model & Closed-Form Solvers
- **Role:** The core computational engine running on TensorFlow.
- **Purpose:**
  - Evaluates batched Signal-to-Interference-plus-Jamming-and-Noise Ratios ($\text{SIJNR}$) across all sub-channels and UEs in parallel using tensor operations (`tf.einsum`, `tf.gather`).
  - Implements **Proposition 1** (Optimal split $\rho_n^\star$ over $\{0, \rho_n^{\text{bal}}, 1\}$), **Proposition 2** (Optimal feature retention $\chi_n^\star = \max(\chi_n^{\min}, \chi_n^{\text{unc}})$), and **Proposition 3** (Optimal local/UAV computing clock frequencies $F_n^{\text{loc}}, F_{nm}$).
  - Evaluates the total weighted ISCC system utility objective function.

#### 5. `optimizers_tf.py` — Batched Swarm Metaheuristics
- **Role:** Vectorized optimization algorithms.
- **Purpose:**
  - **Continuous Metaheuristics:** Batched Whale Optimization Algorithm (WOA), Improved WOA (IWOA), and Particle Swarm Optimization (PSO) to optimize continuous UE transmit powers $\vec{p} = [p_{n,k}]$.
  - **Discrete Metaheuristics:** Batched Binary Whale Optimization Algorithm (BWOA) with V-shaped/sigmoid transfer functions to optimize the binary sub-channel assignment matrix $\vec{A} = [a_{n,k}]$.

#### 6. `solver_tf.py` — Joint Hybrid Optimization Solver (Algorithm 1)
- **Role:** Coordinates the two-tier joint optimization algorithm.
- **Purpose:**
  - Implements **Algorithm 1** (`solve_iscc_tf`).
  - Incorporates the **Lemma 2 Feasibility Pruning Operator $\mathbb{F}(\vec{A})$**, which filters out sub-channel assignments violating NOMA/SIC multiplexing limits before fitness evaluation, speeding up convergence.
  - Alternates between outer BWOA (sub-channel association $\vec{A}$), inner batched TPC (transmit power $\vec{p}$), and exact closed-form coordinate updates for ISCC variables $(\vec{\rho}^\star, \vec{\chi}^\star, \vec{F}^\star)$.

#### 7. `experiments_tf.py` — Monte Carlo Simulation Engine
- **Role:** Orchestrates large-scale batch simulation runs.
- **Purpose:** Drives multi-seed random HPPP snapshot realizations across parameter grids, dispatches parallel worker processes, manages caching (`results/*.json`), and computes mean/confidence interval statistics.

#### 8. `schemes.py` — Baseline Access & Resource Allocation Schemes
- **Role:** Implements benchmark algorithms for performance comparison.
- **Purpose:** Contains comparative algorithms evaluated against the proposed MF-SIC:
  - `MF-SIC (WOA-BWOA)`: Proposed matched-filter SIC with joint ISCC optimization.
  - `Force-All-Offload` (`ARJOA`): Benchmark forcing complete offloading ($\rho_n = 1$).
  - `IOJOA`: Independent Orthogonal Jamming/Offloading Allocation.
  - `FDMA`: Orthogonal frequency division without NOMA/SIC multiplexing.
  - `ALCA`: Association and Local Computing Allocation baseline.

#### 9. `metrics.py` & `plotting.py` — Metrics & Publication Plotting
- **Role:** Visualization and performance reporting.
- **Purpose:** Formats simulation outputs into publication-quality vector figures (`.pdf`) styled according to IEEE formatting standards with custom color palettes, markers, and LaTeX labels.

---

## 6. Workflow & Module Interaction

The diagram below illustrates how the helper and execution modules interact during a simulation run:

```
[scripts/run_sweep.py / run_all.py]
               │
               ▼
   [stochastic_mec/experiments_tf.py] ──(Monte Carlo Snapshot Sampling)
               │
               ├──► [stochastic_mec/network.py]   (Generate 3D HPPP UEs, UAVs, Jammers)
               ├──► [stochastic_mec/channel.py]   (Compute 3D Pathloss, Nakagami, ZFBF W)
               │
               ▼
   [stochastic_mec/solver_tf.py] (Algorithm 1: Hybrid TPC-BWOA)
         │                       │
         ▼                       ▼
  [optimizers_tf.py]      [system_tf.py]
  • Outer: BWOA (A)       • Batched SIJNR & Jamming
  • Inner: WOA/PSO (p)    • Closed-form ISCC Updates (Props 1-3: rho*, chi*, F*)
  • Lemma 2 Filter F(A)   • System Utility Objective Evaluation
               │
               ▼
   [stochastic_mec/metrics.py] & [stochastic_mec/plotting.py]
               │
               ▼
       Output: figures/*.pdf & results/*.json
```
