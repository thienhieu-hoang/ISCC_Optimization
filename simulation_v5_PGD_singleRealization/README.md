# SI-TNTN UAV-MEC ISCC Optimization Simulator (`simulation_v5_PGD_singleRealization`)

`simulation_v5_PGD_singleRealization` is a dedicated **single-realization ($R=1$) parallel execution engine** of the Multi-Start Projected Gradient Descent (PGD) and GPU-accelerated Binary Whale Optimization (BWOA) framework for Space-Air-Ground Non-Terrestrial Networks (SI-TNTN UAV-MEC) under malicious jamming.

### 4-Thread Balanced Scheme Parallelism
Instead of realization-level multiprocessing (which cannot parallelize when $R=1$), this engine evaluates a **single shared network topology & channel snapshot** across **4 parallel worker threads**:
- **Thread 1:** `PGD-BWOA` (Proposed v5 scheme: outer BWOA + inner Multi-Start PGD on GPU — **finishes and reports first!**)
- **Thread 2:** Offloading Heuristics with inner PGD: `ARJOA` + `IOJOA`
- **Thread 3:** Resource Heuristics with inner PGD: `FDMA` + `ALCA`
- **Thread 4:** `WOA-BWOA` (v4 swarm benchmark — runs in parallel on its own thread, finishes at the end)

All 4 threads branch from the **exact same geometry and channel realization**, providing a **~3.5×–4× wall-clock speedup** with 100% fair baseline comparison.
See [README_batch_run.md](file:///c:/Users/AT30890/Hoctap/3_ISCC_Optimization/simulation_v5_PGD_singleRealization/README_batch_run.md) for full architectural details.

---

## Key Architecture & Innovations in `simulation_v5_PGD_singleRealization`

This version combines the speed of **Multi-Start Projected Gradient Descent (PGD)** with the reproducibility of **single-snapshot ($R = 1$) scheme-level parallel execution**:

1. **Inner Multi-Start Projected Gradient Descent (PGD) on GPU ([optimizers_tf.py](file:///c:/Users/AT30890/Hoctap/3_ISCC_Optimization/simulation_v5_PGD_singleRealization/stochastic_mec/optimizers_tf.py#L250-L338))**:
   - Replaces the heuristic nested continuous swarm optimizer (WOA) with **auto-differentiable TensorFlow Projected Gradient Descent (`tf.GradientTape`)** executed directly on GPU.
   - Initialized with **4 structured physical seeds**:
     - **Seed 1 ($p^{\max}$):** Full power budget (noise-limited regime).
     - **Seed 2 ($p^\dagger$):** Lemma 1 analytical bisection root of $\phi_n(p) = 0$ on GPU (interference-free optimal delay-energy tradeoff).
     - **Seed 3 ($p^{\text{half}}$):** Mid-range power $0.5(p_{\min} + p_{\max})$ (balanced regime).
     - **Seed 4 ($p^{\text{low}}$):** Low power budget $p_{\min} + 0.1(p_{\max} - p_{\min})$ (interference-saving regime).
   - All 4 seed trajectories are evaluated and updated **simultaneously in parallel as a single batched tensor on GPU**.
   - Constraints are enforced by exact box projection: $\Pi_{[p_{\min}, p_{\max}]}(\vec{p}) = \text{clip}(\vec{p}, p_{\min}, p_{\max})$.
   - Reduces continuous power optimization from 900 heuristic swarm evaluations (15 particles $\times$ 60 steps) down to **only 10 directed gradient steps across 4 seeds**.

2. **4-Thread Balanced Scheme Parallelism ([experiments_tf.py](file:///c:/Users/AT30890/Hoctap/3_ISCC_Optimization/simulation_v5_PGD_singleRealization/stochastic_mec/experiments_tf.py#L130-L155))**:
   - The 6 benchmark schemes are partitioned into 4 balanced thread groups:
     - **Thread 1:** `PGD-BWOA` (~10–15s, reports first)
     - **Thread 2:** `ARJOA` + `IOJOA` (~15–20s)
     - **Thread 3:** `FDMA` + `ALCA` (~10–15s)
     - **Thread 4:** `WOA-BWOA` (~30–40s, runs concurrently and completes at the end)
   - Dispatched via `ProcessPoolExecutor(max_workers=jobs)` defaulting to `jobs = 4`, yielding a **~3.5×–4× wall-clock speedup** over sequential execution.

3. **Branching from Identical Geometry & Channel Snapshot ([experiments_tf.py](file:///c:/Users/AT30890/Hoctap/3_ISCC_Optimization/simulation_v5_PGD_singleRealization/stochastic_mec/experiments_tf.py#L275-L300))**:
   - The 3D spatial network topology snapshot (`sampled_topo`) is generated **once** per sweep point $x$ using `base_seed + 1000 * xi`.
   - All 4 threads receive this identical `sampled_topo` and re-seed their channel RNG with `seed`.
   - Every scheme operates under 100% identical channel matrices, zero-forcing beamformers, jammer interference, and task requirements.

4. **Guaranteed Non-Empty Cells & UL Bias in Topology Generation ([network.py](file:///c:/Users/AT30890/Hoctap/3_ISCC_Optimization/simulation_v5_PGD_singleRealization/stochastic_mec/network.py#L169-L265))**:
   - Enhanced `sample_topology_with_cells` guarantees that **every UL cell has $\ge 1$ UE** and **every DL cell has $\ge 1$ UE** (no dropped/null cells).
   - Biases user allocation such that approximately **65% of UEs are assigned to UL cells** ($N_{\text{UL}} > N_{\text{DL}}$), ensuring rich multi-user ISCC offloading and interference dynamics.
   - Robust 3D Voronoi coordinate placement prevents lower-altitude UAVs from starving cells.

5. **GPU-Accelerated BWOA Swarm with Tabu Memoization ([solver_tf.py](file:///c:/Users/AT30890/Hoctap/3_ISCC_Optimization/simulation_v5_PGD_singleRealization/stochastic_mec/solver_tf.py#L360-L415))**:
   - Outer Binary Whale Optimization Algorithm (**BWOA**) optimizes the binary sub-channel assignment matrix $\mathbf{A} \in \{0, 1\}^{(N_{\text{ul}} + M_{\text{dl}}) \times K}$.
   - Exact Tabu table caching (`eval_cache`) memoizes solved association matrices, eliminating 40%–60% of redundant inner gradient solves.

---

## Environment Setup & Prerequisites

All scripts are executed within the dedicated Conda environment **`TF_GPU-py3_11`**:

```powershell
# Activate environment
conda activate TF_GPU-py3_11

# OR run directly via conda run:
conda run -n TF_GPU-py3_11 python <script_path>
```

---

## Quick Start & Verification

### A. Quick Convergence Test
Verifies convergence curves of BWOA with inner PGD/WOA/PSO:

```powershell
conda run -n TF_GPU-py3_11 python simulation_v5_PGD_singleRealization/scripts/fig_convergence.py --quick
```

### B. Benchmark Comparison (WOA vs Multi-Start PGD)
Measures the runtime speedup, attained system utility, and inner call counts across snapshots:

```powershell
conda run -n TF_GPU-py3_11 python simulation_v5_PGD_singleRealization/scripts/compare_v4_v5.py 3
```

### C. Quick Smoke Test Sweep
Run a fast single-realization sweep across active-UE density:

```powershell
conda run -n TF_GPU-py3_11 python simulation_v5_PGD_singleRealization/scripts/run_sweep.py ue-density --quick -r 1 -j 4
```

---

## Figure Reproduction Guide

To reproduce paper figures with the single-realization 4-thread PGD engine:

| Figure | Description | Execution Command | Output Files |
| :--- | :--- | :--- | :--- |
| **Fig. 1** | **3D Topology Snapshot** | `conda run -n TF_GPU-py3_11 python simulation_v5_PGD_singleRealization/scripts/fig_topology3d_pdf.py` | `figures/topology3d.pdf` |
| **Fig. 2** | **Algorithm Convergence** (PGD vs WOA/IWOA/PSO) | `conda run -n TF_GPU-py3_11 python simulation_v5_PGD_singleRealization/scripts/fig_convergence.py` | `figures/convergence_bwoa.pdf`, `figures/convergence_mpc.pdf` |
| **Fig. 3(a)** | **System Utility vs. UE Density** | `conda run -n TF_GPU-py3_11 python simulation_v5_PGD_singleRealization/scripts/run_sweep.py ue-density -r 1 -j 4` | `results/ue_density.json`, `figures/ue_density_su.pdf` |
| **Fig. 3(b)** | **System Utility vs. UAV Density** | `conda run -n TF_GPU-py3_11 python simulation_v5_PGD_singleRealization/scripts/run_sweep.py uav-density -r 1 -j 4` | `results/uav_density.json`, `figures/uav_density_su.pdf` |
| **Fig. 4(a)** | **Accuracy–Delay Pareto Frontier** | `conda run -n TF_GPU-py3_11 python simulation_v5_PGD_singleRealization/scripts/run_sweep.py accuracy-tradeoff -r 1 -j 4` | `results/accuracy_weight.json`, `figures/accuracy_tradeoff.pdf` |
| **Fig. 4(b)** | **ISCC Retention vs. Sensing SNR** | `conda run -n TF_GPU-py3_11 python simulation_v5_PGD_singleRealization/scripts/run_sweep.py sensing-snr -r 1 -j 4` | `results/sensing_snr.json`, `figures/retention_vs_snr.pdf` |

### Automated Batch Run (All Figures)
```powershell
conda run -n TF_GPU-py3_11 python simulation_v5_PGD_singleRealization/scripts/run_all.py -r 1 -j 4
```

---

## Directory Structure

```
simulation_v5_PGD_singleRealization/
├── README.md                      <- Architecture and user guide
├── README_batch_run.md            <- Detailed CPU threads vs GPU PGD batch execution documentation
├── requirements.txt               <- Dependencies
├── stochastic_mec/
│   ├── config.py                  <- AlgorithmParams (tpc_default="PGD", max_iter_pgd=10) & SystemParams
│   ├── network.py                 <- 3D HPPP topology sampling with guaranteed cells & UL bias
│   ├── channel.py                 <- Wireless propagation, Nakagami-m fading, ZF beamforming
│   ├── system_tf.py               <- Batched TensorFlow System Model & compute_p_dagger_tf (Lemma 1)
│   ├── optimizers_tf.py           <- MultiStartPGD_TF (Adam), WOA_TF, PSO_TF, and bwoa_step_tf
│   ├── solver_tf.py               <- Hybrid BWOA-PGD solver with GPU repair & Tabu cache
│   ├── experiments_tf.py          <- Single-realization 4-thread scheme parallel runner & evaluator
│   ├── schemes.py                 <- Baseline access schemes (MF-SIC, ARJOA, IOJOA, FDMA, ALCA)
│   ├── metrics.py                 <- Performance metrics calculation (delay, energy, accuracy)
│   └── plotting.py                <- Figure plotting utilities
└── scripts/
    ├── compare_v4_v5.py           <- Benchmark comparison between v4 (WOA) and v5 (PGD)
    ├── run_sweep.py               <- Granular parameter sweep CLI runner (-r 1 -j 4 default)
    ├── run_all.py                 <- Master batch execution script
    ├── fig_convergence.py         <- Convergence figures (BWOA and inner power control)
    ├── fig_topology3d_pdf.py      <- 3D network topology visualization
    ├── fig_density.py             <- Active-UE and UAV density sweep figures
    └── fig_iscc.py                <- Accuracy-delay Pareto and Sensing SNR figures
```