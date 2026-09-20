# SI-TNTN UAV-MEC ISCC simulator (`manuscript_v2`)

Extends `simulation_v1` with the ISCC model of `manuscript_v2.tex`: sensing-derived
tasks, partial offloading, and an inference-accuracy term in the utility.

```bash
/opt/miniconda3/bin/python tests/test_model.py            # 16 checks
cd scripts && /opt/miniconda3/bin/python run_all.py --quick -r 2
```

(The `research` conda env named in v1's README does not exist on this machine;
the base miniconda Python has numpy and matplotlib.)

## What changed from v1

| Paper | Code |
|---|---|
| Sensed task, eq. (task)–(chimin) | `SystemModel._build_tasks`, `accuracy`, `chi_min`, `admissible` |
| Partial offloading, eqs. (Tn)–(Theta) | `SystemModel._branches` |
| Utility with accuracy, eq. (Uul) | `SystemModel.ul_utilities` |
| Prop. 1, clipped water-filling | `SystemModel.smca(ul_sub, rho, chi, rates)` |
| Props. 2–3, split and retention | `SystemModel.optimal_split` (jointly optimal, see docstring) |
| Pruning bound, eq. (Fbound) | `SystemModel.ul_utility_upper_bound` |
| Algorithm 1 | `HybridSolver.evaluate`, then `SystemModel.iscc_allocate` |

New sweeps (`scripts/run_sweep.py`):

- `accuracy-tradeoff`: sweeps β^a with β^t = β^e = (1 − β^a)/2 and writes
  `figures/accuracy_tradeoff.png` (accuracy against normalized delay).
- `sensing-snr`: sweeps the mean γ^sen and writes `figures/retention_vs_snr.png`
  (χ*, ρ*, and the χ^min floor).

Both sweeps run the ISCC solver and an **atomic-task ablation**
(`enable_iscc=False`: the same objective with ρ = a, χ = 1) on the same topologies.

## Fixes that change v1 results

1. **The MF-SIC noise scaling in v1 is wrong** (`mf_noise_fix`, default `True`).
   The MF output signal is p‖h‖⁴, so the noise after combining is ‖h‖² B N₀, not
   L B N₀. The DL→UL term Ξ needs the same ‖h‖² projection, taken here as ‖h‖² Ξ / L
   (isotropic). With v1's scaling the UL SNR is about −50 dB, so v1 almost never
   offloads. In `simulation_v1/results/*.json`, `offload_ratio` is 0 at nearly every
   point and `ul_utility` is 0, so v1's utility curves are DL-only. Setting
   `mf_noise_fix=False` reproduces the v1 SINR. The same error is in eq. (ul_sijnr)
   of both manuscripts.
2. **v1's Lemma 2 pruning is not valid under ISCC.** V and W scale with ρχ, which is
   not known when an association is pruned. The replacement bound holds for every
   (p, F, ρ, χ), and `test_uniform_bound_dominates_every_split` checks it. The v1
   quasiconvex test (ii) is no longer used for pruning.
3. **Inadmissible UEs (χ^min > 1) are projected to local in `decode()`** rather than
   penalized. BWOA almost never drives a row exactly to zero, so with the penalty
   whole runs stalled at −1e14.
4. **The preference sweep range is rescaled.** β^a takes a share of the weights, so
   β^t now spans (0, 1 − β^a).

## New parameters (`SystemParams`)

`beta_acc`, `sensing_snr_db` and `sensing_snr_std_db` (per-UE spread),
`sensing_time`, `sensing_power`, `accuracy_sensitivity` (ϑ), `accuracy_threshold`
(Λ^th), `extract_cycles_per_bit` (ϖ), `iscc_rounds`, `enable_iscc`, `mf_noise_fix`.
