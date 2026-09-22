# SI-TNTN UAV-MEC ISCC Optimization Simulator (TensorFlow Edition - `simulation_v4`)

`simulation_v4` is a **batched, parallel TensorFlow implementation** of the Integrated Sensing, Communication, and Computation (ISCC) framework for space-air-ground non-terrestrial networks (SI-TNTN UAV-MEC).

## Key Differences from `simulation_v3`

1. **Fully Tensorized Swarms**:
   - Continuous power control optimizers (WOA, PSO, IWOA) evaluate and update all $S$ swarm agents simultaneously as multi-dimensional tensors.
   - Discrete Binary Whale Optimization (BWOA) computes transfer functions and bit flips in parallel across candidate association matrices.
2. **Vectorized Signal & ISCC Computations**:
   - Batched SINR, ZF beamforming projections, SIC violation checks, and closed-form ISCC coordinate updates (Props 1–3) are computed using broadcasted TensorFlow tensor operations (`tf.einsum`, `tf.gather`, `tf.reduce_sum`) eliminating nested Python loops.
3. **Execution**:
   - Runs seamlessly within the `TF_GPU-py3_11` conda environment.

## Directory Structure

```
simulation_v4/
├── requirements.txt
├── README.md
├── stochastic_mec/
│   ├── __init__.py
│   ├── config.py           # System and Algorithm parameters
│   ├── network.py          # 3D HPPP topology sampling
│   ├── channel.py          # Channel modeling (G2A/A2G/G2G, Nakagami, ZF beamforming)
│   ├── system_tf.py        # Batched TensorFlow System Model & ISCC Closed-Form Solver
│   ├── optimizers_tf.py    # Batched TensorFlow continuous and binary optimizers (WOA/PSO/BWOA)
│   ├── solver_tf.py        # Batched Hybrid <TPC>-BWOA solver
│   ├── experiments_tf.py   # Monte Carlo driver and sweep runners
│   ├── metrics.py          # Post-processing metrics
│   ├── schemes.py          # Access schemes (MF-SIC, ARJOA, IOJOA, FDMA, ALCA)
│   └── plotting.py         # Publication-quality plotting utilities
└── scripts/
    ├── compare_v3_v4.py    # Verification & benchmark script
    ├── run_sweep.py        # Parameter sweep runner
    ├── run_all.py          # Reproduce all figures in one go
    ├── fig_convergence.py  # Convergence curves (Fig. 3)
    ├── fig_topology.py     # 2D network snapshot (Fig. 2)
    ├── fig_topology3d.py   # 3D spatial network visualization (Fig. 1)
    ├── fig_iscc.py         # ISCC frontier & retention vs SNR (Fig. 5)
    ├── fig_density.py      # UE & UAV density sweeps
    └── fig_compact.py      # Half-column subfigures
```

## Quick Start

Run inside the `TF_GPU-py3_11` conda environment:

```powershell
# 1. Run quick verification test on a single snapshot
conda run -n TF_GPU-py3_11 python simulation_v4/scripts/compare_v3_v4.py

# 2. Run a fast smoke test sweep
conda run -n TF_GPU-py3_11 python simulation_v4/scripts/run_sweep.py ue-density --quick -r 3

# 3. Generate 3D network topology visualization
conda run -n TF_GPU-py3_11 python simulation_v4/scripts/fig_topology3d.py
```
