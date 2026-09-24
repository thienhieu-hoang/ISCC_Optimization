# SI-TNTN UAV-MEC ISCC Optimization Simulator (`simulation_v4`)

`simulation_v4` is a high-performance, **batched parallel TensorFlow implementation** of the Integrated Sensing, Communication, and Computation (ISCC) optimization framework for Space-Air-Ground Non-Terrestrial Networks (SI-TNTN UAV-MEC) under malicious jamming.

---

## Table of Contents
1. [Environment Setup & Prerequisites](#1-environment-setup--prerequisites)
2. [Quick Start & Verification](#2-quick-start--verification)
3. [Figure Reproduction Guide (Manuscript v4)](#3-figure-reproduction-guide-manuscript-v4)
4. [Running Custom Parameter Sweeps](#4-running-custom-parameter-sweeps)
5. [Codebase Architecture & Directory Guide](#5-codebase-architecture--directory-guide)
   - [Overview: `scripts/` vs `stochastic_mec/`](#overview-scripts-vs-stochastic_mec)
   - [A. `scripts/` — Executable Simulation & Figure Drivers](#a-scripts--executable-simulation--figure-drivers)
   - [B. `stochastic_mec/` — Core Simulation Engine & Library Modules](#b-stochastic_mec--core-simulation-engine--library-modules)
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

### A. Quick Convergence & Numerical Verification Test
Run a fast convergence test of the BWOA solver and inner power control algorithms with `--quick`:

```powershell
conda run -n TF_GPU-py3_11 python simulation_v4/scripts/fig_convergence.py --quick
```

### B. Quick Smoke Test Sweep
Run a fast single-realization sweep across active-UE density to verify the full optimization pipeline:

```powershell
conda run -n TF_GPU-py3_11 python simulation_v4/scripts/run_sweep.py ue-density --quick -r 1
```

---

## 3. Figure Reproduction Guide (Manuscript v4)

To reproduce the exact figures in the manuscript, execute the corresponding command below:

| Figure in Paper | Description | Execution Command | Helper / Output Scripts | Output Files |
| :--- | :--- | :--- | :--- | :--- |
| **Fig. 1** | **3D Network Topology Snapshot** (Spatial distribution of UEs, UL-UAVs, DL-UAVs, Jammers, Voronoi cells, G2A/A2G links) | `conda run -n TF_GPU-py3_11 python simulation_v4/scripts/fig_topology3d.py` | `network.py`, `channel.py` | `figures/topology3d.pdf` |
| **Fig. 2** | **Algorithm Convergence Curves**<br>• (a) Outer BWOA convergence (WOA/IWOA/PSO)<br>• (b) Inner TPC subproblem convergence | `conda run -n TF_GPU-py3_11 python simulation_v4/scripts/fig_convergence.py` | `solver_tf.py`, `optimizers_tf.py`, `plotting.py` | `figures/convergence_bwoa.pdf`, `figures/convergence_mpc.pdf` |
| **Fig. 3(a)** | **System Utility vs. Active-UE Density** (Comparing Proposed MF-SIC vs. Force-All-Offload, IOJOA, FDMA, ALCA) | `conda run -n TF_GPU-py3_11 python simulation_v4/scripts/run_sweep.py ue-density -r 100` | `experiments_tf.py`, `schemes.py`, `plotting.py` | `results/ue_density.json`, `figures/ue_density_su.pdf` |
| **Fig. 3(b)** | **System Utility vs. UL-UAV Server Density** (Comparing Proposed MF-SIC vs. baselines across server densities) | `conda run -n TF_GPU-py3_11 python simulation_v4/scripts/run_sweep.py uav-density -r 100` | `experiments_tf.py`, `schemes.py`, `plotting.py` | `results/uav_density.json`, `figures/uav_density_su.pdf` |
| **Fig. 4(a)** | **Accuracy–Delay Pareto Frontier** (Mean accuracy $\Lambda_n$ vs. Normalized delay $T_n/T_n^{\text{ref}}$ over preference weight $\beta^{\text{a}}$) | `conda run -n TF_GPU-py3_11 python simulation_v4/scripts/run_sweep.py accuracy-tradeoff -r 100` | `system_tf.py`, `fig_iscc.py`, `plotting.py` | `results/accuracy_weight.json`, `figures/accuracy_tradeoff.pdf` |
| **Fig. 4(b)** | **ISCC Retention & Offloading vs. Sensing SNR** (Optimal retention $\chi_n^\star$, floor $\chi_n^{\min}$, and offload split $\rho_n^\star$ vs. $\bar{\gamma}^{\text{sen}}$) | `conda run -n TF_GPU-py3_11 python simulation_v4/scripts/run_sweep.py sensing-snr -r 100` | `system_tf.py`, `fig_iscc.py`, `plotting.py` | `results/sensing_snr.json`, `figures/retention_vs_snr.pdf` |

### Automated Batch Run (All Figures in One Command)
To execute all parameter sweeps and render all paper figures sequentially:

```powershell
# Full run (100 realizations per point, multi-process execution)
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

## 5. Codebase Architecture & Directory Guide

### Overview: `scripts/` vs `stochastic_mec/`

> [!NOTE]
> **Architectural Separation of Concerns:**
> - **`scripts/` (Execution Layer & Entry Points):** Contains the **main runnable simulation scripts**, CLI sweep drivers, and figure generation routines that you execute directly to run experiments and produce manuscript plots.
> - **`stochastic_mec/` (Engine & Library Layer):** Contains the **reusable core library modules**, physical channel models, batched TensorFlow system algorithms, metaheuristic optimizers, mathematical solvers, and plotting utilities imported by the scripts.

```
simulation_v4/
├── scripts/                    <- [EXECUTION LAYER] Main entry points, CLI drivers & plot generators
│   ├── run_sweep.py            <- Primary CLI driver for individual parameter sweeps (Figs. 3-4)
│   ├── run_all.py              <- Master orchestrator executing the full simulation pipeline
│   ├── fig_topology3d.py       <- Fig. 1: 3D spatial network topology visualization (saved as PDF)
│   ├── fig_topology3d_pdf.py   <- Fig. 1 variant: High-res PDF 3D plot with custom viewpoint
│   ├── fig_topology3d_pdf_wGround.py <- Fig. 1 variant: 3D plot with ground grid & Voronoi projection
│   ├── fig_topology.py         <- 2D planar snapshot & Voronoi cell projections
│   ├── fig_convergence.py      <- Fig. 2: BWOA & inner TPC (WOA/IWOA/PSO) convergence curves
│   ├── fig_iscc.py             <- Fig. 4: ISCC Pareto frontier & retention vs. sensing SNR from cached data
│   ├── fig_density.py          <- Fig. 3: UE & UAV density sensitivity plots from cached data
│   ├── fig_compact.py          <- Compact half-column subfigure generator for double-column layout
│   └── _common.py              <- Shared CLI argument parsing, directory resolution, and config utilities
│
└── stochastic_mec/             <- [LIBRARY LAYER] Core simulation engine & helper packages
    ├── __init__.py             <- Package initialization and public API export
    ├── config.py               <- Dataclasses for physical parameters, algorithm configs, & sweep results
    ├── network.py              <- 3D Homogeneous Poisson Point Process (HPPP) node & Voronoi cell sampler
    ├── channel.py              <- 3D LoS/NLoS pathloss, Nakagami-m fading, and ZFBF precoding matrices
    ├── system_tf.py            <- Core TensorFlow vectorized system model & closed-form ISCC updates
    ├── optimizers_tf.py        <- Batched continuous (WOA/PSO/IWOA) & discrete (BWOA) swarm metaheuristics
    ├── solver_tf.py            <- Joint Hybrid TPC-BWOA Algorithm 1 with Lemma 2 feasibility pruning
    ├── experiments_tf.py       <- Monte Carlo parallel sweep runner & snapshot evaluator
    ├── schemes.py              <- Baseline access schemes (MF-SIC, Force-All-Offload, IOJOA, FDMA, ALCA)
    ├── metrics.py              <- Post-simulation metrics aggregation, statistics, & averaging
    └── plotting.py             <- Publication-grade Matplotlib IEEE styling & plotting routines
```

---

### A. `scripts/` — Executable Simulation & Figure Drivers

Each file in `scripts/` serves as a user-facing command-line interface (CLI) or plotting script:

1. **`run_sweep.py`**
   - **Role:** Main simulation driver for individual parameter sweeps.
   - **Usage:** Executes Monte Carlo sweeps across active-UE density, UAV density, jammer density, task data size, computation workload, UAV server capacity, transmit power budget, and sensing SNR. Saves JSON metrics in `results/` and vector PDF plots in `figures/`.

2. **`run_all.py`**
   - **Role:** Master batch execution script.
   - **Usage:** Runs all parameter sweeps and generates all figures sequentially or across multiple worker processes with a single command (`python scripts/run_all.py`).

3. **`fig_topology3d.py`**
   - **Role:** Renders Fig. 1 of the manuscript (3D SI-TNTN network snapshot).
   - **Usage:** Samples a real HPPP realization and draws 3D node coordinates (Uplink UAVs, Downlink UAVs, active/inactive UEs, jammers), 3D communication/jamming links, altitude drop lines, and places the legend inside the plot canvas, saving directly to `figures/topology3d.pdf`.

4. **`fig_topology3d_pdf.py` & `fig_topology3d_pdf_wGround.py`**
   - **Role:** Specialized variants of Fig. 1.
   - **Usage:** Provide fine-tuned 3D camera angles, optional ground-plane Voronoi cell grid projections, and publication-ready aspect ratios.

5. **`fig_topology.py`**
   - **Role:** 2D planar topology generator.
   - **Usage:** Generates 2D bird's-eye views of Voronoi cell partitions, user distributions, and server associations.

6. **`fig_convergence.py`**
   - **Role:** Convergence analysis driver (Fig. 2).
   - **Usage:** Runs outer BWOA and inner TPC algorithms (WOA, IWOA, PSO) on a single snapshot and plots convergence rate vs. iteration index (`figures/convergence_bwoa.pdf` and `figures/convergence_mpc.pdf`).

7. **`fig_iscc.py`**
   - **Role:** ISCC feature trade-off plotter.
   - **Usage:** Reads pre-computed results from `results/` and generates Fig. 4(a) (accuracy–delay Pareto frontier over $\beta^{\text{a}}$) and Fig. 4(b) (feature retention $\chi_n^\star$ and offloading split $\rho_n^\star$ vs. sensing SNR $\bar{\gamma}^{\text{sen}}$).

8. **`fig_density.py`**
   - **Role:** Density sweep visualizer.
   - **Usage:** Quick re-plotting script to regenerate Fig. 3(a) and 3(b) from cached JSON results without re-running long simulations.

9. **`fig_compact.py`**
   - **Role:** Compact layout formatter.
   - **Usage:** Generates condensed half-column subfigures for side-by-side inclusion in double-column IEEE Transactions templates.

10. **`_common.py`**
    - **Role:** Shared execution helper.
    - **Usage:** Contains common CLI argument parsers (`base_parser`), algorithm parameter builders (`algorithm_params`), output path resolvers (`outputs`), and dynamic plotting loaders.

---

### B. `stochastic_mec/` — Core Simulation Engine & Library Modules

The `stochastic_mec/` folder contains the backend libraries and mathematical models:

1. **`__init__.py`**
   - **Role:** Package root and public namespace definition.
   - **Usage:** Exposes key classes and functions (`SystemParams`, `AlgorithmParams`, `HybridSolverTF`, `run_sweep`, `sample_topology`) for clean imports.

2. **`config.py` — Configuration & Parameter Management**
   - **Role:** Centralized repository of physical and algorithmic constants.
   - **Details:** Encapsulates `SystemParams` (carrier frequencies, bandwidths, noise powers, HPPP intensities $\lambda_0, \lambda_{\text{UE}}, \lambda_{\text{Q}}$, UAV altitude bounds, ISCC preference weights $\beta_n^{\text{t}}, \beta_n^{\text{e}}, \beta_n^{\text{a}}$, sensing SNR, accuracy thresholds) and `AlgorithmParams` (population sizes, iteration limits, early-stopping patience).

3. **`network.py` — Spatial 3D HPPP Topology Generator**
   - **Role:** Spatial point process and geometric modeling.
   - **Details:** Generates 3D coordinates for Ground UEs $\vec{\ell}_n$, Uplink UAV MEC Servers $\vec{\ell}_m$, Downlink UAVs $\vec{\ell}_{m'}$, and Malicious Jammers $\vec{\ell}_q$. Computes nearest-server Voronoi cell partitions and 3D Euclidean distance matrices.

4. **`channel.py` — Wireless Channel Propagation & Beamforming**
   - **Role:** Physical layer wireless channel calculations.
   - **Details:** Implements elevation-angle-dependent Line-of-Sight (LoS) probabilities, pathloss models, Nakagami-$m$ small-scale fading, and inverse-Gamma shadowing for G2A, A2G, and G2G links. Computes Zero-Forcing Beamforming (ZFBF) precoding matrices $\vec{W}$.

5. **`system_tf.py` — Vectorized System Model & Closed-Form ISCC Solvers**
   - **Role:** Core TensorFlow computational engine.
   - **Details:**
     - Computes batched Signal-to-Interference-plus-Jamming-and-Noise Ratios ($\text{SIJNR}$) and achievable data rates across all sub-channels and swarm candidates simultaneously using `tf.gather` and broadcasted matrix operations.
     - Implements **Proposition 1** (optimal task splitting $\rho_n^\star \in \{0, \rho_n^{\text{bal}}, 1\}$), **Proposition 2** (optimal feature retention $\chi_n^\star = \max(\chi_n^{\min}, \chi_n^{\text{unc}})$), and **Proposition 3** (optimal local/UAV computing allocations $F_n^{\text{loc}}, F_{nm}$).
     - Evaluates the total weighted ISCC system utility and verifies SIC decodability constraints.

6. **`optimizers_tf.py` — Batched Swarm Metaheuristics**
   - **Role:** Vectorized optimization algorithms.
   - **Details:**
     - **Continuous Optimizers (TPC):** Batched Whale Optimization Algorithm (WOA), Improved WOA (IWOA), and Particle Swarm Optimization (PSO) to optimize continuous UE transmit powers $\vec{p} = [p_{n,k}]$ across all $S$ swarm agents simultaneously.
     - **Discrete Optimizer (BWOA):** Batched Binary Whale Optimization Algorithm with sigmoid transfer functions to optimize the binary sub-channel assignment matrix $\vec{A} = [a_{n,k}]$.

7. **`solver_tf.py` — Joint Hybrid Optimization Solver (Algorithm 1)**
   - **Role:** Coordinates the two-tier joint optimization algorithm.
   - **Details:**
     - Implements **Algorithm 1** (`HybridSolverTF`).
     - Incorporates the **Lemma 2 Feasibility Pruning Operator $\mathbb{F}(\vec{A})$**, which instantly filters out invalid sub-channel assignments violating NOMA/SIC multiplexing limits before fitness evaluation.
     - Coordinates the outer BWOA (association $\vec{A}$), inner batched TPC (powers $\vec{p}$), and exact closed-form coordinate updates for ISCC variables $(\vec{\rho}^\star, \vec{\chi}^\star, \vec{F}^\star)$.

8. **`experiments_tf.py` — Monte Carlo Sweep Runner**
   - **Role:** Experiment orchestration and parallel dispatch.
   - **Details:** Executes multi-seed Monte Carlo realization loops across parameter grids, manages parallel worker processes, handles result serialization (`results/*.json`), and aggregates statistical metrics.

9. **`schemes.py` — Baseline Access & Resource Allocation Schemes**
   - **Role:** Benchmark algorithms for performance comparisons.
   - **Details:** Defines comparative algorithms evaluated against the proposed MF-SIC:
     - `MF-SIC (WOA-BWOA)`: Proposed matched-filter SIC with joint ISCC optimization.
     - `Force-All-Offload` (`ARJOA`): Complete offloading benchmark ($\rho_n = 1$).
     - `IOJOA`: Independent Orthogonal Jamming/Offloading Allocation.
     - `FDMA`: Orthogonal frequency division without NOMA multiplexing.
     - `ALCA`: Association and Local Computing Allocation baseline.

10. **`metrics.py` — Simulation Metrics & Statistics**
    - **Role:** Aggregation and evaluation helper.
    - **Details:** Extracts and averages key Key Performance Indicators (KPIs) across realizations: system utility, offloading percentage, mean normalized delay $T_n/T_n^{\text{ref}}$, mean feature retention $\chi_n^\star$, inference accuracy $\Lambda_n$, energy consumption, and solver execution times.

11. **`plotting.py` — Publication Plotting Utilities**
    - **Role:** Figure formatting and visualization library.
    - **Details:** Standardizes figure styling for IEEE publications (color-blind safe Okabe-Ito / Tol palettes, custom markers, LaTeX axis labels, tight bounding boxes, and PDF vector export).

---

## 6. Workflow & Module Interaction

The diagram below illustrates how the execution scripts in `scripts/` invoke the library engine in `stochastic_mec/`:

```
[scripts/run_sweep.py / run_all.py / fig_*.py]
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
