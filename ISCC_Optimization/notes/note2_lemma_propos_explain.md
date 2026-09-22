# In-Depth Explanations: Propositions and Lemmas

This document provides complete mathematical derivations (first and second derivatives, KKT conditions, stationarity proofs), intuitive physical breakdowns, and practical optimization impacts for all Propositions and Lemmas in Section III of `manuscript_v2.tex` / `manuscript_v3.tex`.

---

## 1. Proposition 1: UAV Edge Compute Allocation (Clipped Water-Filling Rule)

### The Equation:
$$F_{nm}^\star = \min\left\{ \mu_m^{-1/2} \sqrt{\frac{\beta_n^{\tt t}\rho_n\chi_n C_n^{\tt raw}}{\mathsf{T}_n^{\tt ref}}}, \;\; \bar{F}_{nm} \right\}$$
where:
$$\bar{F}_{nm} = \frac{\rho_n C_n^{\tt raw}}{\left[ (1-\rho_n)C_n^{\tt raw}/F_n^{\tt loc} - \rho_n D_n^{\tt raw}/R_n \right]_+}$$
and $\mu_m > 0$ is the unique water-level multiplier satisfying $\sum_{n \in \mathcal{N}_m^{\tt ul}} F_{nm}^\star = F_m^{\tt max}$.

---

### A. Detailed Mathematical Derivation & Proof

#### Step 1: Formulating the Objective Function in Terms of $F_{nm}$
For fixed sub-channel matching $\vec{A}^{\tt ul}$, transmit power $\vec{p}$, offloading split $\vec{\rho}$, and feature retention $\vec{\chi}$, the utility of UE $n$ served by UAV $m$ is:
$$\mathbb{U}_n^{\tt ul}(F_{nm}) = \beta_n^{\tt t} \frac{\mathsf{T}_n^{\tt ref} - \mathsf{T}_n(F_{nm})}{\mathsf{T}_n^{\tt ref}} + \beta_n^{\tt e} \frac{\mathsf{E}_n^{\tt ref} - \mathsf{E}_n}{\mathsf{E}_n^{\tt ref}} + \beta_n^{\tt a} \frac{\Lambda_n - \Lambda^{\tt th}}{1 - \Lambda^{\tt th}}$$

* Note that transmit/compute energy $\mathsf{E}_n$ and inference accuracy $\Lambda_n$ are completely independent of the UAV server frequency $F_{nm}$.
* The total latency is:
  $$\mathsf{T}_n(F_{nm}) = \mathsf{T}_n^{\tt sen} + \mathsf{T}_n^{\tt ext} + \chi_n \max\left\{ \Delta_n^{\tt loc}, \; \Delta_n^{\tt ofl}(F_{nm}) \right\}$$
  where:
  $$\Delta_n^{\tt loc} = \frac{(1-\rho_n)C_n^{\tt raw}}{F_n^{\tt loc}}, \qquad \Delta_n^{\tt ofl}(F_{nm}) = \rho_n \left( \frac{D_n^{\tt raw}}{R_n} + \frac{C_n^{\tt raw}}{F_{nm}} \right)$$

---

#### Step 2: Finding the Clipping Frequency Threshold $\bar{F}_{nm}$
As $F_{nm}$ increases, remote offload delay $\Delta_n^{\tt ofl}(F_{nm})$ strictly decreases. The threshold frequency $\bar{F}_{nm}$ is the exact point where remote delay equals local compute delay:
$$\Delta_n^{\tt ofl}(\bar{F}_{nm}) = \Delta_n^{\tt loc}$$
$$\rho_n \left( \frac{D_n^{\tt raw}}{R_n} + \frac{C_n^{\tt raw}}{\bar{F}_{nm}} \right) = \frac{(1-\rho_n) C_n^{\tt raw}}{F_n^{\tt loc}}$$
$$\frac{\rho_n C_n^{\tt raw}}{\bar{F}_{nm}} = \frac{(1-\rho_n) C_n^{\tt raw}}{F_n^{\tt loc}} - \frac{\rho_n D_n^{\tt raw}}{R_n}$$
$$\bar{F}_{nm} = \frac{\rho_n C_n^{\tt raw}}{\left[ \frac{(1-\rho_n) C_n^{\tt raw}}{F_n^{\tt loc}} - \frac{\rho_n D_n^{\tt raw}}{R_n} \right]_+}$$
*(where $[x]_+ = \max\{x, 0\}$; if the bracket is $\le 0$, local compute is always faster and $\bar{F}_{nm} = \infty$).*

This divides the domain of $F_{nm}$ into two distinct operating regimes:
1. **Regime 1 ($F_{nm} < \bar{F}_{nm}$):** Remote offload is slower ($\Delta_n^{\tt ofl} > \Delta_n^{\tt loc}$), so $\max\{\Delta_n^{\tt loc}, \Delta_n^{\tt ofl}\} = \Delta_n^{\tt ofl}(F_{nm})$.
2. **Regime 2 ($F_{nm} \ge \bar{F}_{nm}$):** Local execution is slower ($\Delta_n^{\tt loc} \ge \Delta_n^{\tt ofl}$), so $\max\{\Delta_n^{\tt loc}, \Delta_n^{\tt ofl}\} = \Delta_n^{\tt loc}$ (independent of $F_{nm}$).

---

#### Step 3: First and Second Derivatives & Strict Concavity
* **In Regime 1 ($0 < F_{nm} < \bar{F}_{nm}$):**
  $$\mathbb{U}_n^{\tt ul}(F_{nm}) = \text{const} - \frac{\beta_n^{\tt t} \chi_n \rho_n}{\mathsf{T}_n^{\tt ref}} \left( \frac{D_n^{\tt raw}}{R_n} + \frac{C_n^{\tt raw}}{F_{nm}} \right)$$

  Taking the **first derivative** with respect to $F_{nm}$:
  $$\frac{\partial \mathbb{U}_n^{\tt ul}}{\partial F_{nm}} = -\frac{\beta_n^{\tt t} \chi_n \rho_n C_n^{\tt raw}}{\mathsf{T}_n^{\tt ref}} \cdot \left( -\frac{1}{F_{nm}^2} \right) = \frac{\beta_n^{\tt t} \chi_n \rho_n C_n^{\tt raw}}{\mathsf{T}_n^{\tt ref} F_{nm}^2} > 0$$

  Taking the **second derivative** with respect to $F_{nm}$:
  $$\frac{\partial^2 \mathbb{U}_n^{\tt ul}}{\partial F_{nm}^2} = \frac{\beta_n^{\tt t} \chi_n \rho_n C_n^{\tt raw}}{\mathsf{T}_n^{\tt ref}} \cdot \left( -\frac{2}{F_{nm}^3} \right) = -\frac{2\beta_n^{\tt t} \chi_n \rho_n C_n^{\tt raw}}{\mathsf{T}_n^{\tt ref} F_{nm}^3} < 0$$
  Because $\frac{\partial^2 \mathbb{U}_n^{\tt ul}}{\partial F_{nm}^2} < 0$, the utility function is **strictly concave** in $F_{nm}$ on $(0, \bar{F}_{nm})$.

* **In Regime 2 ($F_{nm} \ge \bar{F}_{nm}$):**
  $$\mathbb{U}_n^{\tt ul}(F_{nm}) = \text{const} - \frac{\beta_n^{\tt t} \chi_n \Delta_n^{\tt loc}}{\mathsf{T}_n^{\tt ref}} \implies \frac{\partial \mathbb{U}_n^{\tt ul}}{\partial F_{nm}} = 0$$
  The derivative vanishes completely beyond $\bar{F}_{nm}$.

---

#### Step 4: Lagrangian Formulation and KKT Stationarity
To optimize CPU allocations across all users served by UAV $m$:
$$\max_{\{F_{nm}\}} \sum_{n \in \mathcal{N}_m^{\tt ul}} \mathbb{U}_n^{\tt ul}(F_{nm}) \quad \text{s.t.} \quad \sum_{n \in \mathcal{N}_m^{\tt ul}} F_{nm} \le F_m^{\tt max}, \quad F_{nm} \ge 0$$

Let $\mu_m \ge 0$ be the Lagrange multiplier associated with the UAV CPU capacity constraint. The Lagrangian function is:
$$\mathcal{L}\left(\{F_{nm}\}, \mu_m\right) = \sum_{n \in \mathcal{N}_m^{\tt ul}} \mathbb{U}_n^{\tt ul}(F_{nm}) + \mu_m \left( F_m^{\tt max} - \sum_{n \in \mathcal{N}_m^{\tt ul}} F_{nm} \right)$$

Setting the KKT stationarity condition $\frac{\partial \mathcal{L}}{\partial F_{nm}} = 0$ in the active regime ($F_{nm} < \bar{F}_{nm}$):
$$\frac{\partial \mathbb{U}_n^{\tt ul}}{\partial F_{nm}} - \mu_m = 0$$
$$\frac{\beta_n^{\tt t} \chi_n \rho_n C_n^{\tt raw}}{\mathsf{T}_n^{\tt ref} F_{nm}^2} = \mu_m$$
$$F_{nm}^2 = \frac{1}{\mu_m} \frac{\beta_n^{\tt t} \chi_n \rho_n C_n^{\tt raw}}{\mathsf{T}_n^{\tt ref}}$$
$$F_{nm} = \mu_m^{-1/2} \sqrt{\frac{\beta_n^{\tt t} \chi_n \rho_n C_n^{\tt raw}}{\mathsf{T}_n^{\tt ref}}}$$

Since the marginal utility drops to zero for $F_{nm} \ge \bar{F}_{nm}$, allocating more than $\bar{F}_{nm}$ wastes CPU budget without improving utility. Projecting onto the upper bound gives the **Clipped Water-Filling Rule**:
$$F_{nm}^\star = \min\left\{ \mu_m^{-1/2} \sqrt{\frac{\beta_n^{\tt t}\rho_n\chi_n C_n^{\tt raw}}{\mathsf{T}_n^{\tt ref}}}, \;\; \bar{F}_{nm} \right\}$$

---

#### Step 5: Uniqueness and 1D Bisection for $\mu_m$
Define the total allocated frequency function:
$$\mathcal{S}_m(\mu_m) \triangleq \sum_{n \in \mathcal{N}_m^{\tt ul}} F_{nm}^\star(\mu_m) = \sum_{n \in \mathcal{N}_m^{\tt ul}} \min\left\{ \mu_m^{-1/2} \sqrt{\frac{\beta_n^{\tt t}\rho_n\chi_n C_n^{\tt raw}}{\mathsf{T}_n^{\tt ref}}}, \;\; \bar{F}_{nm} \right\}$$

* **Monotonicity:** Since $\mu_m^{-1/2}$ is strictly decreasing in $\mu_m$, $\mathcal{S}_m(\mu_m)$ is **strictly monotonically decreasing** on $(0, \infty)$:
  $$\lim_{\mu_m \to 0} \mathcal{S}_m(\mu_m) = \sum_n \bar{F}_{nm}, \qquad \lim_{\mu_m \to \infty} \mathcal{S}_m(\mu_m) = 0$$
* **Uniqueness:** If $\sum_n \bar{F}_{nm} > F_m^{\tt max}$, there exists a **unique** root $\mu_m^\star > 0$ such that $\mathcal{S}_m(\mu_m^\star) = F_m^{\tt max}$.
* **Bisection Search:** The exact multiplier $\mu_m^\star$ is found in $\mathcal{O}(\log(1/\epsilon))$ steps via 1D bisection. If $\sum_n \bar{F}_{nm} \le F_m^{\tt max}$, the constraint does not bind ($\mu_m \to 0$) and $F_{nm}^\star = \bar{F}_{nm}$. $\blacksquare$

---

### B. Intuitive Physical Breakdown

1. **The Square-Root Water-Filling Term ($\sqrt{\dots}$):**
   * Compute latency decreases as $1/F_{nm}$, so the marginal latency reduction scales as $1/F_{nm}^2$.
   * Taking the inverse derivative yields a square-root scaling:
     $$F_{nm} \propto \sqrt{\frac{\beta_n^{\tt t} \cdot (\text{Offloaded Workload } \rho_n \chi_n C_n^{\tt raw})}{\text{Baseline Reference Time } \mathsf{T}_n^{\tt ref}}}$$
   * **Physical Meaning:** Users with **heavier offloaded workloads**, **higher delay sensitivity ($\beta_n^{\tt t}$)**, or **tighter baseline deadlines** receive a larger share of the UAV's CPU pie.

2. **The "Clipping" Ceiling ($\bar{F}_{nm}$):**
   * Because local compute and remote offload execute in parallel, total delay is $\max\{\Delta_n^{\tt loc}, \Delta_n^{\tt ofl}\}$.
   * Once the UAV computes fast enough to match the phone's local execution speed ($F_{nm} = \bar{F}_{nm}$), giving the UAV any further CPU speed produces **zero additional delay reduction** because the phone's local CPU is the bottleneck.
   * Proposition 1 strictly clips User $n$'s share at $\bar{F}_{nm}$ and reallocates the saved CPU frequency to other users in the cell.

---

### C. How Proposition 1 Helps in Solving the MINLP

1. **Replaces Numerical Solvers with an Exact Formula:** 
   Eliminates the need for multi-variable gradient descent or interior point methods across all UAVs, providing the **globally optimal compute allocation analytically in one shot**.
2. **Instant 1D Bisection:** 
   Finding $\mu_m$ takes only a few bisection iterations ($\mathcal{O}(N_m \log(1/\epsilon))$), executing in microseconds.
3. **Reduces Metaheuristic Search Dimensions:** 
   The outer Whale Optimization Algorithm does not need to search over continuous frequencies $\vec{F}$, drastically speeding up convergence.

---

## 2. Proposition 2: Optimal Partial Offloading Split (3-Point Search)

### The Equation:
$$\rho_n^\star = \frac{C_n^{\tt raw} / F_n^{\tt loc}}{C_n^{\tt raw} / F_n^{\tt loc} + D_n^{\tt raw} / R_n + C_n^{\tt raw} / F_{nm(n)}}$$
$$\rho_n^{\text{opt}} \in \{0, \; \rho_n^\star, \; 1\}$$

---

### A. Detailed Mathematical Derivation & Proof

#### Step 1: Formulating the Utility as a Function of $\rho_n$
For fixed candidate sub-channel $\vec{A}^{\tt ul}$, power $\vec{p}$, server CPU $\vec{F}$, and feature retention $\vec{\chi}$, the utility $\mathbb{U}_n^{\tt ul}(\rho_n)$ can be expressed by isolating $\rho_n$:
$$\mathbb{U}_n^{\tt ul}(\rho_n) = \beta_n^{\tt t} \frac{\mathsf{T}_n^{\tt ref} - \mathsf{T}_n(\rho_n)}{\mathsf{T}_n^{\tt ref}} + \beta_n^{\tt e} \frac{\mathsf{E}_n^{\tt ref} - \mathsf{E}_n(\rho_n)}{\mathsf{E}_n^{\tt ref}} + \beta_n^{\tt a} \frac{\Lambda_n - \Lambda^{\tt th}}{1 - \Lambda^{\tt th}}$$

Since baseline reference costs ($\mathsf{T}_n^{\tt ref}, \mathsf{E}_n^{\tt ref}$) and inference accuracy ($\Lambda_n$) do not depend on $\rho_n$, gathering all $\rho_n$-independent constant terms into $C_0$ yields:
$$\mathbb{U}_n^{\tt ul}(\rho_n) = C_0 - \chi_n \left[ k_t \Delta_n(\rho_n) + k_e \Theta_n(\rho_n) \right]$$
where $k_t \triangleq \frac{\beta_n^{\tt t}}{\mathsf{T}_n^{\tt ref}} > 0$ and $k_e \triangleq \frac{\beta_n^{\tt e}}{\mathsf{E}_n^{\tt ref}} > 0$.

---

#### Step 2: Deriving the Unique Breakpoint $\rho_n^\star$
The parallel compute delay is:
$$\Delta_n(\rho_n) = \max\left\{ \Delta_n^{\tt loc}(\rho_n), \; \Delta_n^{\tt ofl}(\rho_n) \right\}$$
where:
* **Local Compute Delay:** $\Delta_n^{\tt loc}(\rho_n) = (1 - \rho_n) \frac{C_n^{\tt raw}}{F_n^{\tt loc}}$ is affine and strictly decreasing in $\rho_n$ with negative slope $-\frac{C_n^{\tt raw}}{F_n^{\tt loc}} < 0$.
* **Remote Offload Delay:** $\Delta_n^{\tt ofl}(\rho_n) = \rho_n \left( \frac{D_n^{\tt raw}}{R_n} + \frac{C_n^{\tt raw}}{F_{nm}} \right)$ is affine and strictly increasing in $\rho_n$ with positive slope $\left(\frac{D_n^{\tt raw}}{R_n} + \frac{C_n^{\tt raw}}{F_{nm}}\right) > 0$.

Equating the two branches $\Delta_n^{\tt loc}(\rho_n^\star) = \Delta_n^{\tt ofl}(\rho_n^\star)$:
$$(1 - \rho_n^\star) \frac{C_n^{\tt raw}}{F_n^{\tt loc}} = \rho_n^\star \left( \frac{D_n^{\tt raw}}{R_n} + \frac{C_n^{\tt raw}}{F_{nm}} \right)$$
$$\frac{C_n^{\tt raw}}{F_n^{\tt loc}} = \rho_n^\star \left( \frac{C_n^{\tt raw}}{F_n^{\tt loc}} + \frac{D_n^{\tt raw}}{R_n} + \frac{C_n^{\tt raw}}{F_{nm}} \right)$$
$$\rho_n^\star = \frac{C_n^{\tt raw} / F_n^{\tt loc}}{C_n^{\tt raw} / F_n^{\tt loc} + D_n^{\tt raw} / R_n + C_n^{\tt raw} / F_{nm}}$$

Since all parameters ($C_n^{\tt raw}, D_n^{\tt raw}, F_n^{\tt loc}, R_n, F_{nm}$) are strictly positive, the denominator strictly exceeds the numerator, guaranteeing that $\rho_n^\star \in (0, 1)$ is the **unique single crossing point**.

---

#### Step 3: Piecewise Linear Formulation & Slope Analysis
The energy consumption function $\Theta_n(\rho_n)$ is purely affine across the entire domain $[0, 1]$:
$$\Theta_n(\rho_n) = (1 - \rho_n) \kappa_n C_n^{\tt raw} (F_n^{\tt loc})^2 + \rho_n \frac{p_n D_n^{\tt raw}}{\xi_n R_n} = \kappa_n C_n^{\tt raw} (F_n^{\tt loc})^2 + \rho_n \underbrace{\left( \frac{p_n D_n^{\tt raw}}{\xi_n R_n} - \kappa_n C_n^{\tt raw} (F_n^{\tt loc})^2 \right)}_{\triangleq\, \delta_E}$$

Now, express $\mathbb{U}_n^{\tt ul}(\rho_n)$ across the two intervals partitioned by $\rho_n^\star$:

1. **On Interval 1 ($0 \le \rho_n \le \rho_n^\star$, Local branch active):**
   $$\Delta_n(\rho_n) = \Delta_n^{\tt loc}(\rho_n) = \frac{C_n^{\tt raw}}{F_n^{\tt loc}} - \rho_n \frac{C_n^{\tt raw}}{F_n^{\tt loc}}$$
   $$\mathbb{U}_n^{\tt ul}(\rho_n) = C_1 + \chi_n \cdot \alpha_1 \rho_n$$
   where the slope on Interval 1 is:
   $$\alpha_1 = k_t \frac{C_n^{\tt raw}}{F_n^{\tt loc}} - k_e \delta_E$$

2. **On Interval 2 ($\rho_n^\star \le \rho_n \le 1$, Offload branch active):**
   $$\Delta_n(\rho_n) = \Delta_n^{\tt ofl}(\rho_n) = \rho_n \left( \frac{D_n^{\tt raw}}{R_n} + \frac{C_n^{\tt raw}}{F_{nm}} \right)$$
   $$\mathbb{U}_n^{\tt ul}(\rho_n) = C_2 + \chi_n \cdot \alpha_2 \rho_n$$
   where the slope on Interval 2 is:
   $$\alpha_2 = -k_t \left( \frac{D_n^{\tt raw}}{R_n} + \frac{C_n^{\tt raw}}{F_{nm}} \right) - k_e \delta_E$$

---

#### Step 4: Strict Concavity and the 3-Point Optimality Proof
Subtracting the two slopes:
$$\alpha_1 - \alpha_2 = k_t \left( \frac{C_n^{\tt raw}}{F_n^{\tt loc}} + \frac{D_n^{\tt raw}}{R_n} + \frac{C_n^{\tt raw}}{F_{nm}} \right) > 0 \implies \mathbf{\alpha_1 > \alpha_2}$$

* **Geometric Meaning:** The slope on the left segment ($\alpha_1$) is **strictly greater** than the slope on the right segment ($\alpha_2$). This proves that the utility function $\mathbb{U}_n^{\tt ul}(\rho_n)$ is **strictly concave piecewise linear** (an upside-down V shape with a single peak/kink at $\rho_n^\star$).
* **Extreme Point Maximization:**
  * For the linear segment $[0, \rho_n^\star]$, the maximum must occur at an endpoint: $\rho_n \in \{0, \rho_n^\star\}$.
  * For the linear segment $[\rho_n^\star, 1]$, the maximum must occur at an endpoint: $\rho_n \in \{\rho_n^\star, 1\}$.
* **Conclusion:** The global maximizer over the continuous interval $[0, 1]$ is strictly contained in the 3-element set:
  $$\rho_n^\diamond = \arg\max_{\rho_n \in [0, 1]} \mathbb{U}_n^{\tt ul}(\rho_n) \in \mathbf{\{0, \; \rho_n^\star, \; 1\}} \quad \blacksquare$$

---

### B. Distinction Between $\rho_n^\star$ and $\rho_n^\diamond$:

| Notation | Name / Definition | Role in Optimization |
| :---: | :--- | :--- |
| **$\rho_n^\star$** | **The Breakpoint / Balance Formula**<br>$\rho_n^\star = \frac{C_n^{\tt raw}/F_n^{\tt loc}}{C_n^{\tt raw}/F_n^{\tt loc} + D_n^{\tt raw}/R_n + C_n^{\tt raw}/F_{nm}}$ | The unique mathematical value where local delay matches offload delay ($\Delta_n^{\tt loc} = \Delta_n^{\tt ofl}$). It is **one of the 3 candidates** to test. |
| **$\rho_n^\diamond$** | **The Final Optimal Decision (The Winner)**<br>$\rho_n^\diamond \triangleq \arg\max_{\rho \in \{0, \, \rho_n^\star, \, 1\}} \mathbb{U}_n^{\tt ul}(\rho)$ | The actual chosen offloading fraction that produces the **highest utility** among the three candidates $\{0, \rho_n^\star, 1\}$. |

* **Why use two different symbols?** 
  Using $\rho_n^\star$ for both would cause ambiguity. Equation \eqref{eq:rhostar} computes the balance point $\rho_n^\star$ (e.g., $0.45$). But depending on battery energy constraints or network interference, the final winning decision $\rho_n^\diamond$ might be $0$ (pure local) or $1$ (pure offload) rather than $0.45$.

---

### C. Practical Impact:
* Replaces continuous numerical optimization of $\rho_n$ with **exactly 3 function evaluations per user** ($\mathcal{O}(1)$ time).

---

## 3. Proposition 3: Optimal Feature Retention Ratio (Logarithmic Closed Form)

### The Equation:
$$\chi_n^\star = \min\left\{ 1, \; \max\left\{ \chi_n^{\tt min}, \; -\frac{\ln \Psi_n}{\vartheta_n \Gamma_n} \right\} \right\}, \qquad \Psi_n = \frac{(1-\Lambda^{\tt th})\Omega_n}{\beta_n^{\tt a}\vartheta_n \Gamma_n}$$
where $\Omega_n = \frac{\beta_n^{\tt t}\Delta_n}{\mathsf{T}_n^{\tt ref}} + \frac{\beta_n^{\tt e}\Theta_n}{\mathsf{E}_n^{\tt ref}}$ is the marginal delay–energy price per retained feature unit.

---

### A. Detailed Mathematical Derivation & Proof

#### Step 1: Starting from the Raw Utility Function Definition
For fixed sub-channel $\vec{A}^{\tt ul}$, power $\vec{p}$, server CPU $\vec{F}$, and offloading split $\vec{\rho}$, the multi-objective utility of UE $n$ is defined by Equation (12) / (13):
$$\mathbb{U}_n^{\tt ul} = \beta_n^{\tt t}\frac{\mathsf{T}_n^{\tt ref}-\mathsf{T}_n}{\mathsf{T}_n^{\tt ref}} + \beta_n^{\tt e}\frac{\mathsf{E}_n^{\tt ref}-\mathsf{E}_n}{\mathsf{E}_n^{\tt ref}} + \beta_n^{\tt a}\frac{\Lambda_n-\Lambda^{\tt th}}{1-\Lambda^{\tt th}}$$
where $\beta_n^{\tt t} + \beta_n^{\tt e} + \beta_n^{\tt a} = 1$ ($\beta_n^{\tt t}, \beta_n^{\tt e}, \beta_n^{\tt a} \ge 0$).

---

#### Step 2: Substituting Total Latency, Energy, and Accuracy Models
Recall from Equations (8), (10), and (2) how the decision variable $\chi_n$ scales the actual latency $\mathsf{T}_n$, energy $\mathsf{E}_n$, and inference accuracy $\Lambda_n$:
* **Actual Total Latency:** $\mathsf{T}_n(\chi_n) = \mathsf{T}_n^{\tt sen} + \mathsf{T}_n^{\tt ext} + \chi_n \Delta_n$
* **Actual Total Energy:** $\mathsf{E}_n(\chi_n) = \mathsf{E}_n^{\tt sen} + \mathsf{E}_n^{\tt ext} + \chi_n \Theta_n$
* **Inference Accuracy:** $\Lambda_n(\chi_n) = 1 - e^{-\vartheta_n \chi_n \Gamma_n}$

Substitute each expression directly into the three utility components:
$$\begin{aligned}
\mathbb{U}_n^{\tt ul}(\chi_n) &= \beta_n^{\tt t}\frac{\mathsf{T}_n^{\tt ref} - \left(\mathsf{T}_n^{\tt sen} + \mathsf{T}_n^{\tt ext} + \chi_n \Delta_n\right)}{\mathsf{T}_n^{\tt ref}} \\
&\quad + \beta_n^{\tt e}\frac{\mathsf{E}_n^{\tt ref} - \left(\mathsf{E}_n^{\tt sen} + \mathsf{E}_n^{\tt ext} + \chi_n \Theta_n\right)}{\mathsf{E}_n^{\tt ref}} \\
&\quad + \beta_n^{\tt a}\frac{\left(1 - e^{-\vartheta_n \chi_n \Gamma_n}\right) - \Lambda^{\tt th}}{1 - \Lambda^{\tt th}}
\end{aligned}$$

---

#### Step 3: Grouping Constant Terms and Isolating $\chi_n$
Separate the terms that do NOT depend on $\chi_n$ from the terms that linearly multiply $\chi_n$:
$$\begin{aligned}
\mathbb{U}_n^{\tt ul}(\chi_n) &= \underbrace{\left( \beta_n^{\tt t}\frac{\mathsf{T}_n^{\tt ref} - \mathsf{T}_n^{\tt sen} - \mathsf{T}_n^{\tt ext}}{\mathsf{T}_n^{\tt ref}} + \beta_n^{\tt e}\frac{\mathsf{E}_n^{\tt ref} - \mathsf{E}_n^{\tt sen} - \mathsf{E}_n^{\tt ext}}{\mathsf{E}_n^{\tt ref}} \right)}_{\triangleq\, C_{\chi} \text{ (Constant with respect to } \chi_n)} \\
&\quad - \chi_n \underbrace{\left( \frac{\beta_n^{\tt t}\Delta_n}{\mathsf{T}_n^{\tt ref}} + \frac{\beta_n^{\tt e}\Theta_n}{\mathsf{E}_n^{\tt ref}} \right)}_{\triangleq\, \Omega_n \text{ (Marginal Delay--Energy Price)}} \\
&\quad + \beta_n^{\tt a}\frac{1 - e^{-\vartheta_n \chi_n \Gamma_n} - \Lambda^{\tt th}}{1 - \Lambda^{\tt th}}
\end{aligned}$$

Thus, the utility simplifies cleanly to:
$$\mathbb{U}_n^{\tt ul}(\chi_n) = C_{\chi} - \chi_n \Omega_n + \beta_n^{\tt a}\frac{1 - e^{-\vartheta_n \chi_n \Gamma_n} - \Lambda^{\tt th}}{1 - \Lambda^{\tt th}}$$

---

#### Step 4: First Derivative and Stationary Root
To find the unconstrained maximizer, differentiate $\mathbb{U}_n^{\tt ul}(\chi_n)$ with respect to $\chi_n$:
$$\frac{\partial \mathbb{U}_n^{\tt ul}}{\partial \chi_n} = 0 - \Omega_n + \frac{\beta_n^{\tt a}}{1 - \Lambda^{\tt th}} \cdot \frac{d}{d\chi_n}\left(1 - e^{-\vartheta_n \chi_n \Gamma_n} - \Lambda^{\tt th}\right)$$
$$\frac{\partial \mathbb{U}_n^{\tt ul}}{\partial \chi_n} = -\Omega_n + \frac{\beta_n^{\tt a} \vartheta_n \Gamma_n e^{-\vartheta_n \chi_n \Gamma_n}}{1 - \Lambda^{\tt th}}$$

Setting the first derivative to zero ($\frac{\partial \mathbb{U}_n^{\tt ul}}{\partial \chi_n} = 0$):
$$\frac{\beta_n^{\tt a} \vartheta_n \Gamma_n e^{-\vartheta_n \chi_n \Gamma_n}}{1 - \Lambda^{\tt th}} = \Omega_n$$
$$e^{-\vartheta_n \chi_n \Gamma_n} = \frac{(1 - \Lambda^{\tt th}) \Omega_n}{\beta_n^{\tt a} \vartheta_n \Gamma_n} \triangleq \Psi_n$$

Taking the natural logarithm on both sides:
$$-\vartheta_n \chi_n \Gamma_n = \ln \Psi_n \implies \chi_n = -\frac{\ln \Psi_n}{\vartheta_n \Gamma_n}$$

---

#### Step 5: Second Derivative and Strict Concavity
Differentiating again with respect to $\chi_n$:
$$\frac{\partial^2 \mathbb{U}_n^{\tt ul}}{\partial \chi_n^2} = \frac{d}{d\chi_n}\left(-\Omega_n + \frac{\beta_n^{\tt a} \vartheta_n \Gamma_n e^{-\vartheta_n \chi_n \Gamma_n}}{1 - \Lambda^{\tt th}}\right) = -\frac{\beta_n^{\tt a} (\vartheta_n \Gamma_n)^2 e^{-\vartheta_n \chi_n \Gamma_n}}{1 - \Lambda^{\tt th}}$$

Since $\beta_n^{\tt a} > 0$, $\vartheta_n > 0$, $\Gamma_n > 0$, $\Lambda^{\tt th} < 1$, and the exponential $e^{-\vartheta_n \chi_n \Gamma_n} > 0$:
$$\frac{\partial^2 \mathbb{U}_n^{\tt ul}}{\partial \chi_n^2} < 0 \quad (\forall \chi_n \in \mathbb{R}^+)$$
This proves that $\mathbb{U}_n^{\tt ul}(\chi_n)$ is **strictly concave** in $\chi_n$, guaranteeing that the stationary point is the unique global unconstrained maximum.

---

#### Step 6: Feasible Domain Projection
Projecting the unconstrained root onto the admissible feasible interval $[\chi_n^{\tt min}, 1]$ yields the exact closed-form optimal retention ratio:
$$\chi_n^\star = \min\left\{ 1, \; \max\left\{ \chi_n^{\tt min}, \; -\frac{\ln \Psi_n}{\vartheta_n \Gamma_n} \right\} \right\} \quad \blacksquare$$

---

### B. Practical Impact:
* Closed-form formula evaluated in **$\mathcal{O}(1)$ time**.
* Decouples feature retention from offloading split because $\chi_n$ factors out of $\Omega_n(\rho_n)$.

---

## 4. Lemma 1: Quasiconvexity of Interference-Free Power Cost & Derivation of $\phi_n(p)$

### The Problem & The Objective Function:
In the decoupled interference-free power allocation (Equation 21), each offloading ground UE independently minimizes its single-user transmission cost function:
$$\min_{p_n \in (0, p_n^{\tt max}]} w_n(p_n) \triangleq \frac{\eta_n + \theta_n p_n}{B_k \log_2(1 + \bar{h}_n p_n)}$$
where:
* $\eta_n = \frac{\beta_n^{\tt t} \chi_n \rho_n D_n^{\tt raw}}{\mathsf{T}_n^{\tt ref}} \ge 0$ (Delay penalty weight)
* $\theta_n = \frac{\beta_n^{\tt e} \chi_n \rho_n D_n^{\tt raw}}{\xi_n \mathsf{E}_n^{\tt ref}} > 0$ (Battery energy penalty weight)
* $\bar{h}_n = \frac{\mathsf{g}_n}{B_k N_0} > 0$ (Channel-to-noise ratio)
* $B_k > 0$ (Sub-channel bandwidth)

---

### A. Detailed Mathematical Derivation: Why Do We Have $\phi_n(p)$?

#### Step 1: Differentiating $w_n(p)$ using the Quotient Rule
Let $u(p) = \eta_n + \theta_n p$ and $v(p) = B_k \log_2(1 + \bar{h}_n p)$.
Rewriting the logarithm in natural base ($\log_2(x) = \frac{\ln x}{\ln 2}$):
$$v(p) = \frac{B_k}{\ln 2} \ln(1 + \bar{h}_n p)$$

The individual derivatives with respect to $p$ are:
$$u'(p) = \theta_n$$
$$v'(p) = \frac{B_k}{\ln 2} \cdot \frac{\bar{h}_n}{1 + \bar{h}_n p}$$

Applying the quotient rule for differentiation, $w_n'(p) = \frac{u'(p) v(p) - u(p) v'(p)}{[v(p)]^2}$:
$$w_n'(p) = \frac{\theta_n \cdot \left[ \frac{B_k}{\ln 2} \ln(1 + \bar{h}_n p) \right] - (\eta_n + \theta_n p) \cdot \left[ \frac{B_k}{\ln 2} \frac{\bar{h}_n}{1 + \bar{h}_n p} \right]}{\left[ B_k \log_2(1 + \bar{h}_n p) \right]^2}$$

---

#### Step 2: Factoring Out Constants to Isolate the Numerator
Factor out $\frac{B_k}{\ln 2}$ from the numerator:
$$\text{Numerator of } w_n'(p) = \frac{B_k}{\ln 2} \left[ \theta_n \ln(1 + \bar{h}_n p) - \frac{\bar{h}_n (\eta_n + \theta_n p)}{1 + \bar{h}_n p} \right]$$

Since $\ln(1 + \bar{h}_n p) = \ln 2 \cdot \log_2(1 + \bar{h}_n p)$, we can equivalently write this in base-2:
$$\text{Numerator of } w_n'(p) = B_k \underbrace{\left[ \theta_n \log_2(1 + \bar{h}_n p) - \frac{\bar{h}_n}{\ln 2} \frac{\eta_n + \theta_n p}{1 + \bar{h}_n p} \right]}_{\triangleq\, \phi_n(p)}$$

Substituting this back into $w_n'(p)$:
$$\mathbf{w_n'(p) = \frac{B_k \phi_n(p)}{\left[ B_k \log_2(1 + \bar{h}_n p) \right]^2} = \frac{\phi_n(p)}{B_k \left[ \log_2(1 + \bar{h}_n p) \right]^2}}$$

---

#### Step 3: Sign Equivalence $\operatorname{sign}(w_n'(p)) \equiv \operatorname{sign}(\phi_n(p))$
Notice that the denominator $B_k [\log_2(1 + \bar{h}_n p)]^2$ is **strictly positive** for all $p > 0$.
Therefore, the denominator cannot change sign or cause roots. 
**The algebraic sign of the derivative $w_n'(p)$ is governed exclusively by $\phi_n(p)$**:
$$\operatorname{sign}(w_n'(p)) = \operatorname{sign}(\phi_n(p))$$
* If $\phi_n(p) < 0 \implies w_n'(p) < 0$ (cost $w_n(p)$ is strictly **decreasing**).
* If $\phi_n(p) = 0 \implies w_n'(p) = 0$ (stationary point / **local extremum**).
* If $\phi_n(p) > 0 \implies w_n'(p) > 0$ (cost $w_n(p)$ is strictly **increasing**).

This is exactly why the auxiliary function $\phi_n(p)$ is introduced: **finding the root of $w_n'(p) = 0$ is identical to finding the root of $\phi_n(p) = 0$!**

---

### B. Proof of Strict Monotonicity of $\phi_n(p)$ & Uniqueness of Root

To prove that $w_n(p)$ has a unique global minimum, we differentiate $\phi_n(p)$ with respect to $p$:
$$\phi_n(p) = \frac{\theta_n}{\ln 2} \ln(1 + \bar{h}_n p) - \frac{\bar{h}_n}{\ln 2} \left( \frac{\eta_n + \theta_n p}{1 + \bar{h}_n p} \right)$$

#### Step 1: Differentiate the Rational Fraction Term
$$\frac{d}{dp} \left( \frac{\eta_n + \theta_n p}{1 + \bar{h}_n p} \right) = \frac{\theta_n (1 + \bar{h}_n p) - (\eta_n + \theta_n p)\bar{h}_n}{(1 + \bar{h}_n p)^2} = \frac{\theta_n + \theta_n \bar{h}_n p - \eta_n \bar{h}_n - \theta_n \bar{h}_n p}{(1 + \bar{h}_n p)^2} = \frac{\theta_n - \eta_n \bar{h}_n}{(1 + \bar{h}_n p)^2}$$

#### Step 2: Differentiate $\phi_n(p)$ Completely
$$\phi_n'(p) = \frac{\theta_n}{\ln 2} \cdot \frac{\bar{h}_n}{1 + \bar{h}_n p} - \frac{\bar{h}_n}{\ln 2} \cdot \left[ \frac{\theta_n - \eta_n \bar{h}_n}{(1 + \bar{h}_n p)^2} \right]$$

Bring both terms to the common denominator $(\ln 2)(1 + \bar{h}_n p)^2$:
$$\phi_n'(p) = \frac{\bar{h}_n}{\ln 2 (1 + \bar{h}_n p)^2} \left[ \theta_n (1 + \bar{h}_n p) - (\theta_n - \eta_n \bar{h}_n) \right]$$
$$\phi_n'(p) = \frac{\bar{h}_n}{\ln 2 (1 + \bar{h}_n p)^2} \left[ \theta_n + \theta_n \bar{h}_n p - \theta_n + \eta_n \bar{h}_n \right]$$
$$\mathbf{\phi_n'(p) = \frac{\bar{h}_n^2 (\eta_n + \theta_n p)}{\ln 2 (1 + \bar{h}_n p)^2}}$$

Since all parameters $\bar{h}_n > 0$, $\theta_n > 0$, $\eta_n \ge 0$, and $p > 0$:
$$\mathbf{\phi_n'(p) > 0 \quad (\forall p > 0)}$$
**Thus, $\phi_n(p)$ is strictly monotonically increasing on $(0, \infty)$!**

---

### C. Boundary Limits & Strict Quasiconvexity of $w_n(p)$

Let us examine the behavior of $\phi_n(p)$ at the boundary extremes:
1. **At the lower limit ($p \to 0^+$):**
   $$\lim_{p \to 0^+} \phi_n(p) = \theta_n \log_2(1) - \frac{\bar{h}_n}{\ln 2} \frac{\eta_n + 0}{1 + 0} = -\frac{\bar{h}_n \eta_n}{\ln 2} \le 0$$
   * When the delay penalty is active ($\eta_n > 0$), $\lim_{p \to 0^+} \phi_n(p) < 0$.
2. **At the upper limit ($p \to \infty$):**
   * $\theta_n \log_2(1 + \bar{h}_n p) \to +\infty$ (grows logarithmically without bound).
   * $\frac{\bar{h}_n}{\ln 2} \frac{\eta_n + \theta_n p}{1 + \bar{h}_n p} \to \frac{\theta_n}{\ln 2}$ (approaches a constant finite upper asymptote).
   * Therefore, $\lim_{p \to \infty} \phi_n(p) = +\infty - \frac{\theta_n}{\ln 2} = +\infty > 0$.

#### Conclusion: Unimodal / Quasiconvex Shape
By the Intermediate Value Theorem and strict monotonicity ($\phi_n'(p) > 0$):
* $\phi_n(p)$ has **EXACTLY ONE unique root** $p^\star \in (0, \infty)$ where $\phi_n(p^\star) = 0$.
* For $p \in (0, p^\star)$: $\phi_n(p) < 0 \implies w_n'(p) < 0$ ($w_n(p)$ strictly decreases).
* For $p \in (p^\star, \infty)$: $\phi_n(p) > 0 \implies w_n'(p) > 0$ ($w_n(p)$ strictly increases).

Hence, $w_n(p)$ decreases to a global minimum at $p^\star$ and then increases monotonically. **$w_n(p)$ is strictly quasiconvex on $(0, \infty)$.**

```
w_n(p)
  ^
  | \                                   /
  |  \                                 /
  |   \                               /
  |    \                             /
  |     \                           /
  |      \                         /
  |       \           *           /  <-- Global Minimum at p* where \phi_n(p*) = 0
  +----------------------------------------------------> p
 0                   p*                 p_max
```

---

### D. Box-Constrained Solution on $[0, p_n^{\tt max}]$ & 1D Bisection

In physical wireless systems, power is bounded by the hardware amplifier limit $p_n \in [0, p_n^{\tt max}]$.

1. **Case 1: If $\phi_n(p_n^{\tt max}) \le 0$:**
   * Because $\phi_n(p)$ is increasing, $\phi_n(p) \le 0$ for all $p \in [0, p_n^{\tt max}]$.
   * The function $w_n(p)$ is strictly decreasing across the entire feasible range.
   * The optimal power hits the ceiling: $\mathbf{p_n^\dagger = p_n^{\tt max}}$.
2. **Case 2: If $\phi_n(p_n^{\tt max}) > 0$:**
   * Since $\phi_n(0) \le 0$ and $\phi_n(p_n^{\tt max}) > 0$, the unique root $p^\star$ lies strictly inside $(0, p_n^{\tt max})$.
   * The optimal power is the interior stationary root: $\mathbf{p_n^\dagger = p^\star}$.
   * Because $\phi_n(p)$ is strictly monotonic, $p^\star$ is found with guaranteed convergence by standard **1D bisection search** on $[0, p_n^{\tt max}]$ in $\mathcal{O}\left(\log_2\left(\frac{p_n^{\tt max}}{\varepsilon}\right)\right)$ steps (converging to machine precision in under 20 iterations).

---

### E. Physical Trade-Off Encoded in $\phi_n(p) = 0$

Setting $\phi_n(p) = 0$ balances two opposing physical forces:
$$\underbrace{\theta_n \log_2(1 + \bar{h}_n p)}_{\text{Marginal Energy Cost of Increasing Power}} = \underbrace{\frac{\bar{h}_n}{\ln 2} \frac{\eta_n + \theta_n p}{1 + \bar{h}_n p}}_{\text{Marginal Delay Benefit of Increasing Rate}}$$

* **At low power ($p < p^\star$):** Increasing power yields massive transmission speed gains $\log_2(1 + \bar{h}_n p)$, drastically cutting latency.
* **At high power ($p > p^\star$):** Diminishing returns of the logarithmic capacity curve set in; pumping more power burns excessive battery energy without providing meaningful extra data rate.
* **Root $p^\star$:** The exact optimal point of diminishing returns.

---

## 5. Lemma 2: Upper-Bound Pruning Filter

### The Equation:
$$\mathbb{F}(\vec{A}) = \sum_{n} \bar{u}_n(\vec{A}^{\tt ul}) + \mathbb{U}_{\mathrm{ub}}^{\tt dl}(\vec{A}^{\tt dl})$$
**Pruning Rule:** If $\mathbb{F}(\vec{A}) \le \mathbb{U}^\star$, discard candidate matching $\vec{A}$ immediately.

---

### Mathematical Proof & Impact:
* $\mathbb{F}(\vec{A})$ bounds the maximum utility achievable by $\vec{A}$ by assuming zero multi-user interference, zero jamming, maximum power $p_n^{\tt max}$, and infinite UAV CPU capacity $F_m^{\tt max}$.
* Because $\mathbb{F}(\vec{A}) \ge \mathbb{U}(\vec{A})$ strictly holds for all continuous variables, if $\mathbb{F}(\vec{A}) \le \mathbb{U}^\star$, then no allocation can beat the incumbent $\mathbb{U}^\star$.
* **Impact:** Discards **60% to 80%** of candidate discrete associations in BWOA without running continuous optimization.
