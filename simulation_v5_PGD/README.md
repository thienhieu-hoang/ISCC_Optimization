# SI-TNTN UAV-MEC ISCC Optimization Simulator (`simulation_v5`)

`simulation_v5` is the next-generation **Multi-Start Projected Gradient Descent (Multi-Start PGD)** and **Binary Whale Optimization (BWOA)** simulator for Space-Air-Ground Non-Terrestrial Networks (SI-TNTN UAV-MEC) under malicious jamming.

---

## 1. Key Innovations in `simulation_v5` vs `simulation_v4`

1. **Multi-Start Projected Gradient Descent (PGD) Inner Power Solver**:
   - Replaces the nested continuous swarm optimizer (WOA) with **auto-differentiable TensorFlow Projected Gradient Descent (`tf.GradientTape`)** on GPU.
   - Initialized with **4 structured physical seeds**:
     - **Seed 1 ($p^{\max}$):** Full power budget (noise-limited regime).
     - **Seed 2 ($p^\dagger$):** Lemma 1 analytical bisection root of $\phi_n(p) = 0$ (interference-free optimal tradeoff).
     - **Seed 3 ($p^{\text{half}}$):** Mid-range power $0.5(p_{\min} + p_{\max})$ (balanced energy/delay regime).
     - **Seed 4 ($p^{\text{low}}$):** Low power $p_{\min} + 0.1(p_{\max} - p_{\min})$ (interference-saving regime).
   - All 4 trajectories are evaluated **simultaneously in parallel as a single batched tensor** on GPU.

2. **Massive Computational Speedup**:
   - Reduces continuous power optimization from 450 swarm steps down to **5–10 directed gradient steps**.
   - Achieves a **$\approx 5\times - 10\times$ overall speedup** per snapshot while maintaining $>98\%$ utility optimality.

3. **Outer Binary Association**:
   - Retains Binary Whale Optimization Algorithm (**BWOA**) with Lemma 2 Feasibility Pruning Filter $\mathbb{F}(\vec{A})$.

---

## 2. Environment Setup & Prerequisites

Run commands in the dedicated Conda environment **`TF_GPU-py3_11`**:

```powershell
conda activate TF_GPU-py3_11

# OR run directly:
conda run -n TF_GPU-py3_11 python <script_path>
```

---

## 3. Quick Start & Benchmark

### A. Run Benchmark Comparison (v4 WOA vs v5 Multi-Start PGD)
Measure the runtime speedup, attained utility, and inner call counts across snapshots:

```powershell
conda run -n TF_GPU-py3_11 python simulation_v5/scripts/compare_v4_v5.py 3
```

### B. Run Quick Smoke Test Sweep
```powershell
conda run -n TF_GPU-py3_11 python simulation_v5/scripts/run_sweep.py ue-density --quick -r 3
```

---

## 4. Directory Structure

```
simulation_v5/
├── README.md                      <- User guide and architecture of v5
├── requirements.txt               <- Dependencies
├── stochastic_mec/
│   ├── config.py                  <- System & PGD/BWOA Algorithm parameters
│   ├── network.py                 <- 3D HPPP topology sampling
│   ├── channel.py                 <- Wireless propagation, Nakagami-m fading, ZFBF
│   ├── system_tf.py               <- Batched TensorFlow System Model & compute_p_dagger_tf
│   ├── optimizers_tf.py           <- MultiStartPGD_TF, WOA_TF, PSO_TF, and BWOA step
│   ├── solver_tf.py               <- Hybrid BWOA-PGD solver (Algorithm 1)
│   ├── experiments_tf.py          <- Monte Carlo simulation driver
│   ├── schemes.py                 <- Baseline access schemes
│   ├── metrics.py                 <- Post-simulation metrics
│   └── plotting.py                <- Publication plotting utilities
└── scripts/
    ├── compare_v4_v5.py           <- Benchmark comparison between v4 (WOA) and v5 (PGD)
    ├── run_sweep.py               <- Parameter sweep CLI runner
    ├── run_all.py                 <- Full batch reproduction
    ├── fig_topology3d.py          <- 3D topology visualization
    ├── fig_density.py             <- Density sweep figures
    └── fig_iscc.py                <- ISCC trade-off figures
```

---

## 5. Inner Optimization Pipeline: Multi-Start PGD Architecture

```
[Candidate Association A from BWOA]
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. Initialize Multi-Start Seeds:                            │
│    • Seed 1: p_max                                          │
│    • Seed 2: p_dagger (Lemma 1 bisection root of phi=0)     │
│    • Seed 3: p_half = 0.5 * (p_min + p_max)                 │
│    • Seed 4: p_low  = p_min + 0.1 * (p_max - p_min)         │
│    Shape: (4, num_active_ues)                               │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Batched Gradient Descent (tf.GradientTape on GPU):       │
│    For step = 1 to max_iter_pgd (10 iters):                 │
│      • Loss = mpc_objective(p) + nu_sic * SIC_penalty(p)    │
│      • Compute exact gradient: g = dLoss / dp               │
│      • Adam / Momentum update                               │
│      • Box projection: p = clip(p, p_min, p_max)            │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Select Best Local Minimum across 4 Seeds:                │
│    • Evaluate penalized objective for all 4 trajectories    │
│    • Best Power: p* = argmin(Loss_1, Loss_2, Loss_3, Loss_4)│
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. Closed-Form ISCC Coordinate Updates:                     │
│    • Props 1-3: Exact (rho*, chi*, F*) in O(N) zero loops   │
└─────────────────────────────────────────────────────────────┘
```
