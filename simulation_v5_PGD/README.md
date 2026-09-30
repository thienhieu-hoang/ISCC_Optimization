# SI-TNTN UAV-MEC ISCC Optimization Simulator (`simulation_v5_PGD`)

`simulation_v5_PGD` is the next-generation **Multi-Start Projected Gradient Descent (Multi-Start PGD)** and **GPU-accelerated Binary Whale Optimization (BWOA)** simulator for Space-Air-Ground Non-Terrestrial Networks (SI-TNTN UAV-MEC) under malicious jamming.

---

## 1. Key Innovations in `simulation_v5_PGD` vs `simulation_v4`

1. **Multi-Start Projected Gradient Descent (PGD) Inner Power Solver**:
   - Replaces the heuristic nested continuous swarm optimizer (WOA) with **auto-differentiable TensorFlow Projected Gradient Descent (`tf.GradientTape`)** directly on GPU.
   - Initialized with **4 structured physical seeds**:
     - **Seed 1 ($p^{\max}$):** Full power budget (noise-limited regime).
     - **Seed 2 ($p^\dagger$):** Lemma 1 analytical bisection root of $\phi_n(p) = 0$ on GPU (interference-free optimal delay-energy tradeoff).
     - **Seed 3 ($p^{\text{half}}$):** Mid-range power $0.5(p_{\min} + p_{\max})$ (balanced regime).
     - **Seed 4 ($p^{\text{low}}$):** Low power budget $p_{\min} + 0.1(p_{\max} - p_{\min})$ (interference-saving regime).
   - All 4 seed trajectories are evaluated and updated **simultaneously in parallel as a single batched tensor on GPU**.
   - Constraints are enforced by exact box projection: $\Pi_{[p_{\min}, p_{\max}]}(\vec{p}) = \text{clip}(\vec{p}, p_{\min}, p_{\max})$.

2. **Sequential Outer BWOA with GPU Tensor Acceleration & Tabu Memoization**:
   - Outer Binary Whale Optimization Algorithm (**BWOA**) optimizes the joint binary association matrix $\mathbf{A}^{\text{net}} \in \{0, 1\}^{(N_{\text{ul}} + M_{\text{dl}}) \times K}$.
   - **GPU Tensor Constraint Repair (`repair_valid_tf` & `repair_population_tf`)**: Row constraints (UL UE $\le 1$ subchannel, DL UAV $= 1$ subchannel) are strictly enforced directly on GPU tensors via `@tf.function` (`tf.one_hot`, `tf.argmax`, `tf.where`), eliminating slow CPU loops.
   - **GPU Neighborhood Perturbation (`perturb_valid_tf`)**: 1-step moves are executed directly on GPU tensors.
   - **Tabu Table Memoization (`eval_cache`)**: Fast hash lookup skips already-visited association matrices in $0\text{ ms}$, preventing redundant inner gradient solves during whale exploitation.

3. **Massive Computational Speedup**:
   - Reduces continuous power optimization from 900 heuristic swarm evaluations (15 particles $\times$ 60 steps) down to **only 10 directed gradient steps across 4 seeds**.
   - Achieves a **$\approx 5\times - 10\times$ overall speedup** per realization while maintaining $>98\%$ utility optimality.

---

## 2. Environment Setup & Prerequisites

All scripts are executed within the Conda environment **`TF_GPU-py3_11`**:

```powershell
# Activate environment
conda activate TF_GPU-py3_11

# Run directly:
conda run -n TF_GPU-py3_11 python <script_path>
```

> [!NOTE]
> On Linux / High-Performance Computing clusters (e.g., **Calcul Québec** with SLURM and NVIDIA A100/V100 GPUs), TensorFlow natively allocates all tensors and `@tf.function` graphs directly onto the physical GPU. On native Windows, TensorFlow $\ge 2.11$ executes these tensor operations on CPU instructions (oneDNN) unless run inside WSL2 Ubuntu.

---

## 3. Quick Start & Benchmark

### A. Run Benchmark Comparison (v4 WOA vs v5 Multi-Start PGD)
Measures the runtime speedup, attained system utility, and inner call counts across snapshots:

```powershell
conda run -n TF_GPU-py3_11 python simulation_v5_PGD/scripts/compare_v4_v5.py 3
```

### B. Run Quick Convergence Test
Verifies convergence curves of BWOA with inner PGD/WOA/PSO:

```powershell
conda run -n TF_GPU-py3_11 python simulation_v5_PGD/scripts/fig_convergence.py --quick
```

### C. Run Parameter Sweep Smoke Test
```powershell
conda run -n TF_GPU-py3_11 python simulation_v5_PGD/scripts/run_sweep.py ue-density --quick -r 3
```

---

## 4. Directory Structure

```
simulation_v5_PGD/
├── README.md                      <- Architecture and user guide of v5_PGD
├── README_batch_run.md            <- Detailed batching hierarchy & GPU execution documentation
├── requirements.txt               <- Dependencies
├── stochastic_mec/
│   ├── config.py                  <- AlgorithmParams (tpc_default="PGD", max_iter_pgd=10) & SystemParams
│   ├── network.py                 <- 3D HPPP topology sampling (UAVs, UEs, Jammers)
│   ├── channel.py                 <- Wireless propagation, Nakagami-m fading, ZF beamforming
│   ├── system_tf.py               <- Batched TensorFlow System Model & compute_p_dagger_tf (Lemma 1)
│   ├── optimizers_tf.py           <- MultiStartPGD_TF (Adam), WOA_TF, PSO_TF, and bwoa_step_tf
│   ├── solver_tf.py               <- Hybrid BWOA-PGD solver with GPU repair & Tabu cache
│   ├── experiments_tf.py          <- Monte Carlo simulation driver & multi-processing
│   ├── schemes.py                 <- Baseline access schemes (MF-SIC, ARJOA, IOJOA, FDMA, ALCA)
│   ├── metrics.py                 <- Performance metrics calculation (delay, energy, accuracy)
│   └── plotting.py                <- Figure plotting utilities
└── scripts/
    ├── compare_v4_v5.py           <- Benchmark comparison between v4 (WOA) and v5 (PGD)
    ├── run_sweep.py               <- Granular parameter sweep CLI runner
    ├── run_all.py                 <- Sequential batch execution of all paper figures
    ├── fig_convergence.py         <- Convergence figures (BWOA and inner power control)
    ├── fig_topology3d.py          <- 3D network topology visualization
    ├── fig_density.py             <- Active-UE and UAV density sweep figures
    └── fig_iscc.py                <- Accuracy-delay Pareto and Sensing SNR figures
```

---

## 5. Inner Optimization Pipeline: Multi-Start PGD Architecture

```
[Candidate Association A from BWOA Swarm]
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. Check Tabu Memoization Table:                            │
│    • Key = get_tabu_key(assoc)                              │
│    • In cache?  ──Yes──> Instant reuse (0 ms, zero re-solve)│
│    • Not in cache? ──No──> Proceed to Multi-Start PGD       │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Initialize 4 Structured Multi-Start Seeds:               │
│    • Seed 1: p_max     (Full power budget)                  │
│    • Seed 2: p_dagger  (Lemma 1 bisection root of phi_n = 0)│
│    • Seed 3: p_half    (0.5 * (p_min + p_max))              │
│    • Seed 4: p_low     (p_min + 0.1 * (p_max - p_min))      │
│    Tensor Shape: (4, num_active_ues) on GPU                 │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Batched Auto-Diff PGD (tf.GradientTape on GPU):          │
│    For step = 1 to max_iter_pgd (10 iterations):            │
│      • Loss = mpc_objective(p)                              │
│      • Exact Analytical Gradient: g = dLoss / dp            │
│      • Adam Momentum Update: m = beta1*m + (1-beta1)*g      │
│      • Box Projection: p = clip_by_value(p, p_min, p_max)   │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. Select Best Trajectory across 4 Seeds:                   │
│    • Evaluate objective for all 4 trajectories in parallel  │
│    • Best Power: p* = argmin(Loss_1, Loss_2, Loss_3, Loss_4)│
│    • Store in Tabu table: eval_cache[key] = (util, p*, q*)  │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. Closed-Form ISCC Coordinate Updates:                     │
│    • Props 1-3: Exact (rho*, chi*, F*) in O(N) zero loops   │
└─────────────────────────────────────────────────────────────┘
```

---

## 6. Algorithm Parameter Reference

| Parameter | Default | Meaning |
| :--- | :--- | :--- |
| `tpc_default` | `"PGD"` | Default inner continuous power solver (`"PGD"`, `"WOA"`, `"PSO"`) |
| `n_pgd_seeds` | `4` | Structured seeds: $[p^{\max}, p^\dagger, p^{\text{half}}, p^{\text{low}}]$ |
| `max_iter_pgd` | `10` | Number of projected gradient descent steps (Adam) |
| `lr_pgd` | `0.005` | Learning rate for PGD / Adam |
| `pgd_optimizer` | `"adam"` | First-order update rule: `"adam"` or `"sgd"` |
| `n_agents_bwoa` | `30` | Number of binary search agents in outer BWOA ($S_3$) |
| `max_iter_bwoa` | `120` | Maximum outer BWOA iterations ($I_3^{\max}$) |
| `patience_bwoa` | `25` | Early stopping patience for outer BWOA |
| `enable_cache` | `True` | Tabu evaluation cache memoization enabled |
