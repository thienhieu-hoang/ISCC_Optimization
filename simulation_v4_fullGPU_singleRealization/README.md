# SI-TNTN UAV-MEC ISCC Optimization Simulator (`simulation_v4_fullGPU_singleRealization`)

`simulation_v4_fullGPU_singleRealization` is a dedicated **single-realization ($R=1$) parallel execution engine** of the Integrated Sensing, Communication, and Computation (ISCC) optimization framework for Space-Air-Ground Non-Terrestrial Networks (SI-TNTN UAV-MEC) under malicious jamming.

### 4-Thread Balanced Scheme Parallelism
Instead of realization-level multiprocessing (which cannot parallelize when $R=1$), this engine evaluates a **single shared network topology & channel snapshot** across **4 parallel worker threads**:
- **Thread 1:** `WOA-BWOA`
- **Thread 2:** `IWOA-BWOA`
- **Thread 3:** `PSO-BWOA`
- **Thread 4:** Heuristics executed sequentially: `ARJOA` + `IOJOA` + `FDMA` + `ALCA`

All 4 threads branch from the **exact same geometry and channel realization**, providing a **~3.3× wall-clock speedup** with 100% fair baseline comparison.
See [README_batch_run.md](file:///c:/Users/AT30890/Hoctap/3_ISCC_Optimization/simulation_v4_fullGPU_singleRealization/README_batch_run.md) for full architectural details.

---

## Key Architecture & Modifications in `simulation_v4_fullGPU_singleRealization`

This version is specifically adapted for fast, reproducible, and balanced **single-snapshot ($R = 1$) simulations**:

1. **4-Thread Balanced Scheme Parallelism ([experiments_tf.py](file:///c:/Users/AT30890/Hoctap/3_ISCC_Optimization/simulation_v4_fullGPU_singleRealization/stochastic_mec/experiments_tf.py#L129-L150))**:
   - Rather than multiprocessing across realizations (which leaves cores idle when $R=1$), the 7 baseline schemes are partitioned into 4 balanced thread groups:
     - **Thread 1:** `WOA-BWOA` (~30–40s)
     - **Thread 2:** `IWOA-BWOA` (~30–45s)
     - **Thread 3:** `PSO-BWOA` (~30–40s)
     - **Thread 4:** Fast heuristics (`ARJOA` + `IOJOA` + `FDMA` + `ALCA` ~10–30s total)
   - Dispatched via `ProcessPoolExecutor(max_workers=jobs)` defaulting to `jobs = 4`, yielding a **~3.3× wall-clock speedup** over sequential execution.

2. **Branching from Identical Geometry & Channel Conditions ([experiments_tf.py](file:///c:/Users/AT30890/Hoctap/3_ISCC_Optimization/simulation_v4_fullGPU_singleRealization/stochastic_mec/experiments_tf.py#L280-L300))**:
   - The 3D spatial network topology snapshot (`sampled_topo`) is generated **once** per sweep point $x$ using `base_seed + 1000 * xi`.
   - All 4 threads receive this identical `sampled_topo` and re-seed their channel RNG with `seed`.
   - Every scheme operates under 100% identical channel matrices, zero-forcing beamformers, jammer interference, and task requirements.

3. **Guaranteed Non-Empty Cells & UL Bias in Topology Generation ([network.py](file:///c:/Users/AT30890/Hoctap/3_ISCC_Optimization/simulation_v4_fullGPU_singleRealization/stochastic_mec/network.py#L169-L270))**:
   - Enhanced `sample_topology_with_cells` guarantees that **every UL cell has $\ge 1$ UE** and **every DL cell has $\ge 1$ UE** (no dropped/null cells).
   - Biases user allocation such that approximately **65% of UEs are assigned to UL cells** ($N_{\text{UL}} > N_{\text{DL}}$), ensuring rich multi-user ISCC offloading and interference dynamics.
   - Safe Voronoi coordinate placement prevents 3D altitude differences from starving lower UAV cells.

4. **GPU-Accelerated BWOA Swarm with Tabu Memoization ([solver_tf.py](file:///c:/Users/AT30890/Hoctap/3_ISCC_Optimization/simulation_v4_fullGPU_singleRealization/stochastic_mec/solver_tf.py#L380-L415))**:
   - Binary swarm position updates (`bwoa_step_tf`), constraint repairs (`repair_valid_tf`), and perturbation flips (`perturb_valid_tf`) execute as GPU tensor operations.
   - Exact Tabu table caching (`eval_cache`) memoizes solved association matrices, eliminating redundant inner power evaluations.
   - Inner power allocation subproblem runs on GPU in batches of $S = 15$ continuous whale agents simultaneously (`ContinuousWOA`).

---

## Table of Contents
1. [Key Architecture & Modifications](#key-architecture--modifications-in-simulation_v4_fullgpu_singlerealization)
2. [Environment Setup & Prerequisites](#1-environment-setup--prerequisites)
3. [Quick Start & Verification](#2-quick-start--verification)
4. [Figure Reproduction Guide (Manuscript v4)](#3-figure-reproduction-guide-manuscript-v4)
5. [Running Custom Parameter Sweeps](#4-running-custom-parameter-sweeps)
6. [Codebase Architecture & Directory Guide](#5-codebase-architecture--directory-guide)
7. [Workflow & Module Interaction](#6-workflow--module-interaction)

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
conda run -n TF_GPU-py3_11 python simulation_v4_fullGPU_singleRealization/scripts/fig_convergence.py --quick
```

### B. Quick Smoke Test Sweep
Run a fast single-realization sweep across active-UE density to verify the full optimization pipeline:

```powershell
conda run -n TF_GPU-py3_11 python simulation_v4_fullGPU_singleRealization/scripts/run_sweep.py ue-density --quick -r 1 -j 4
```

---

## 3. Figure Reproduction Guide (Manuscript v4)

To reproduce the exact figures in the manuscript, execute the corresponding command below:

| Figure in Paper | Description | Execution Command | Helper / Output Scripts | Output Files |
| :--- | :--- | :--- | :--- | :--- |
| **Fig. 1** | **3D Network Topology Snapshot** (Spatial distribution of UEs, UL-UAVs, DL-UAVs, Jammers, Voronoi cells, G2A/A2G links) | `conda run -n TF_GPU-py3_11 python simulation_v4_fullGPU_singleRealization/scripts/fig_topology3d.py` | `network.py`, `channel.py` | `figures/topology3d.pdf` |
| **Fig. 2** | **Algorithm Convergence Curves**<br>• (a) Outer BWOA convergence (WOA/IWOA/PSO)<br>• (b) Inner TPC subproblem convergence | `conda run -n TF_GPU-py3_11 python simulation_v4_fullGPU_singleRealization/scripts/fig_convergence.py` | `solver_tf.py`, `optimizers_tf.py`, `plotting.py` | `figures/convergence_bwoa.pdf`, `figures/convergence_mpc.pdf` |
| **Fig. 3(a)** | **System Utility vs. Active-UE Density** (Comparing Proposed MF-SIC vs. Force-All-Offload, IOJOA, FDMA, ALCA) | `conda run -n TF_GPU-py3_11 python simulation_v4_fullGPU_singleRealization/scripts/run_sweep.py ue-density -r 1 -j 4` | `experiments_tf.py`, `schemes.py`, `plotting.py` | `results/ue_density.json`, `figures/ue_density_su.pdf` |
| **Fig. 3(b)** | **System Utility vs. UL-UAV Server Density** (Comparing Proposed MF-SIC vs. baselines across server densities) | `conda run -n TF_GPU-py3_11 python simulation_v4_fullGPU_singleRealization/scripts/run_sweep.py uav-density -r 1 -j 4` | `experiments_tf.py`, `schemes.py`, `plotting.py` | `results/uav_density.json`, `figures/uav_density_su.pdf` |
| **Fig. 4(a)** | **Accuracy–Delay Pareto Frontier** (Mean accuracy $\Lambda_n$ vs. Normalized delay $T_n/T_n^{\text{ref}}$ over preference weight $\beta^{\text{a}}$) | `conda run -n TF_GPU-py3_11 python simulation_v4_fullGPU_singleRealization/scripts/run_sweep.py accuracy-tradeoff -r 1 -j 4` | `system_tf.py`, `fig_iscc.py`, `plotting.py` | `results/accuracy_weight.json`, `figures/accuracy_tradeoff.pdf` |
| **Fig. 4(b)** | **ISCC Retention & Offloading vs. Sensing SNR** (Optimal retention $\chi_n^\star$, floor $\chi_n^{\min}$, and offload split $\rho_n^\star$ vs. $\bar{\gamma}^{\text{sen}}$) | `conda run -n TF_GPU-py3_11 python simulation_v4_fullGPU_singleRealization/scripts/run_sweep.py sensing-snr -r 1 -j 4` | `system_tf.py`, `fig_iscc.py`, `plotting.py` | `results/sensing_snr.json`, `figures/retention_vs_snr.pdf` |

### Automated Batch Run (All Figures in One Command)
To execute all parameter sweeps and render all paper figures sequentially:

```powershell
# Single-realization 4-thread execution
conda run -n TF_GPU-py3_11 python simulation_v4_fullGPU_singleRealization/scripts/run_all.py -r 1 -j 4

# Fast verification run
conda run -n TF_GPU-py3_11 python simulation_v4_fullGPU_singleRealization/scripts/run_all.py --quick -r 1 -j 4
```

---

## 4. Running Custom Parameter Sweeps

The CLI `scripts/run_sweep.py` provides independent subcommands for granular sensitivity studies (all running 1 shared realization across 4 parallel scheme threads):

```powershell
# Sweep active-UE density (Fig. 3a):
conda run -n TF_GPU-py3_11 python simulation_v4_fullGPU_singleRealization/scripts/run_sweep.py ue-density -r 1 -j 4

# Sweep UL-UAV MEC server density (Fig. 3b):
conda run -n TF_GPU-py3_11 python simulation_v4_fullGPU_singleRealization/scripts/run_sweep.py uav-density -r 1 -j 4

# Sweep malicious jammer density:
conda run -n TF_GPU-py3_11 python simulation_v4_fullGPU_singleRealization/scripts/run_sweep.py jammer-density -r 1 -j 4

# Sweep task raw data size Dn:
conda run -n TF_GPU-py3_11 python simulation_v4_fullGPU_singleRealization/scripts/run_sweep.py datasize -r 1 -j 4

# Sweep computation task workload Cn:
conda run -n TF_GPU-py3_11 python simulation_v4_fullGPU_singleRealization/scripts/run_sweep.py taskload -r 1 -j 4

# Sweep UAV MEC server computational capacity Fm^max:
conda run -n TF_GPU-py3_11 python simulation_v4_fullGPU_singleRealization/scripts/run_sweep.py capacity -r 1 -j 4

# Sweep UE maximum transmit power budget Pn^max:
conda run -n TF_GPU-py3_11 python simulation_v4_fullGPU_singleRealization/scripts/run_sweep.py power -r 1 -j 4

# Sweep time vs. energy preference weights (beta_t vs beta_e):
conda run -n TF_GPU-py3_11 python simulation_v4_fullGPU_singleRealization/scripts/run_sweep.py preference -r 1 -j 4
```

---

## 5. Codebase Architecture & Directory Guide

### Overview: `scripts/` vs `stochastic_mec/`

> [!NOTE]
> **Architectural Separation of Concerns:**
> - **`scripts/` (Execution Layer & Entry Points):** Contains the **main runnable simulation scripts**, CLI sweep drivers, and figure generation routines that you execute directly to run experiments and produce manuscript plots.
> - **`stochastic_mec/` (Engine & Library Layer):** Contains the **reusable core library modules**, physical channel models, batched TensorFlow system algorithms, metaheuristic optimizers, mathematical solvers, and plotting utilities imported by the scripts.

```
simulation_v4_fullGPU_singleRealization/
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
    ├── experiments_tf.py       <- Single-realization 4-thread scheme-parallel sweep runner & snapshot evaluator
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

8. **`experiments_tf.py` — Single-Realization Scheme-Parallel Sweep Runner**
   - **Role:** Experiment orchestration and 4-thread scheme parallel dispatch.
   - **Details:** Generates the single 3D network topology and channel snapshot once per parameter point, dispatches the 4 scheme groups (`WOA-BWOA`, `IWOA-BWOA`, `PSO-BWOA`, Heuristics) across parallel CPU worker processes with isolated GPU contexts, handles result serialization (`results/*.json`), and aggregates KPI metrics.

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
   [stochastic_mec/experiments_tf.py] ──(Single-Realization Snapshot Sampling)
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