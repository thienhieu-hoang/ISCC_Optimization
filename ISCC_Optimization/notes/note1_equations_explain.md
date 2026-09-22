# Detailed Mathematical & Physical Explanation of Equations & Constraints

---

## After Equation 6 (Uplink SIJNR)

$$\gamma_{n,k,m} = \frac{\mathsf{g}_n^2\, p_n}{\mathcal{I}_{n,k,m}^{\tt w} + \mathcal{I}_{n,k,m}^{\tt o} + \mathcal{J}_{n,k,m} + \mathsf{g}_n(\Xi_{k,m}/L + B_kN_0)}$$

### 1. Step-by-Step Proof: Why the Power of $\vec{h}_{n,k,m}^{\sf H}\vec{i}_{\text{leak}}$ is $\mathsf{g}_n\frac{\Xi_{k,m}}{L}$

#### Step 1: Definition of the Raw Interference Vector
The total DL interference vector hitting the $L$ receive antennas of UAV $m$ is:
$$\vec{i}_{\text{leak}} = \sum_{m'\in\mathcal{M}_k^{\tt sub}} \vec{G}_{m',k,m} \vec{P}_{m',k}^{\tt UAV} \quad \in \mathbb{C}^{L \times 1}$$

The total power of this vector across all $L$ antennas is defined as:
$$\Xi_{k,m} \triangleq \mathbb{E}\left[\|\vec{i}_{\text{leak}}\|^2\right] = \mathbb{E}\left[\text{Tr}\left(\vec{i}_{\text{leak}}\vec{i}_{\text{leak}}^{\sf H}\right)\right]$$

#### Step 2: Spatial Covariance Matrix (Isotropic / Uncorrelated Property)
Because the DL UAVs broadcast towards randomly distributed ground UEs with independent fading phases, the interference arriving across the $L$ receive antennas of UAV $m$ is **spatially uncorrelated and evenly distributed** (isotropic):

$$\mathbf{R}_{\text{leak}} \triangleq \mathbb{E}\left[\vec{i}_{\text{leak}}\vec{i}_{\text{leak}}^{\sf H}\right] = \sigma_{\text{leak}}^2 \vec{I}_L$$
where $\sigma_{\text{leak}}^2$ is the average power per antenna, and $\vec{I}_L$ is the $L \times L$ identity matrix.

Taking the trace of both sides:
$$\text{Tr}\left(\mathbf{R}_{\text{leak}}\right) = \text{Tr}\left(\sigma_{\text{leak}}^2 \vec{I}_L\right) = L \sigma_{\text{leak}}^2$$

Since $\text{Tr}(\mathbf{R}_{\text{leak}}) = \Xi_{k,m}$, we solve for $\sigma_{\text{leak}}^2$:
$$L \sigma_{\text{leak}}^2 = \Xi_{k,m} \implies \sigma_{\text{leak}}^2 = \frac{\Xi_{k,m}}{L}$$

Therefore, the spatial covariance matrix is:
$$\mathbb{E}\left[\vec{i}_{\text{leak}}\vec{i}_{\text{leak}}^{\sf H}\right] = \frac{\Xi_{k,m}}{L} \vec{I}_L$$

#### Step 3: Applying the Receive Filter $\vec{h}_{n,k,m}^{\sf H}$
The receiver inside UAV $m$ projects the vector $\vec{i}_{\text{leak}}$ onto its matched filter $\vec{h}_{n,k,m}^{\sf H}$:
$$\tilde{i}_{\text{leak}} = \vec{h}_{n,k,m}^{\sf H} \vec{i}_{\text{leak}} \quad \in \mathbb{C}$$

#### Step 4: Calculating the Filtered Power (Linear Algebra Derivation)
The expected power of the scalar filtered signal $\tilde{i}_{\text{leak}}$ is:
$$\mathbb{E}\left[|\tilde{i}_{\text{leak}}|^2\right] = \mathbb{E}\left[\tilde{i}_{\text{leak}} \tilde{i}_{\text{leak}}^*\right] = \mathbb{E}\left[\vec{h}_{n,k,m}^{\sf H}\vec{i}_{\text{leak}}\vec{i}_{\text{leak}}^{\sf H}\vec{h}_{n,k,m}\right]$$
$$= \vec{h}_{n,k,m}^{\sf H} \mathbb{E}\left[\vec{i}_{\text{leak}}\vec{i}_{\text{leak}}^{\sf H}\right] \vec{h}_{n,k,m} = \vec{h}_{n,k,m}^{\sf H} \left(\frac{\Xi_{k,m}}{L}\vec{I}_L\right) \vec{h}_{n,k,m}$$
$$= \frac{\Xi_{k,m}}{L} \left(\vec{h}_{n,k,m}^{\sf H}\vec{h}_{n,k,m}\right) = \frac{\Xi_{k,m}}{L} \|\vec{h}_{n,k,m}\|^2 = \mathbf{\mathsf{g}_n \left(\frac{\Xi_{k,m}}{L}\right)}$$

---

### 2. Distinction Between Physical Radiation ($\Xi_{k,m}$) and Receiver Filtering ($\tilde{i}_{\text{leak}}$)

* **$\Xi_{k,m}$ (In the Air):** Total raw power from DL UAVs hitting the antennas of UAV $m$. Does not contain $\vec{h}_{n,k,m}$ because DL UAVs transmit independently of Ground UE $n$.
* **$\tilde{i}_{\text{leak}}$ (Inside Receiver DSP):** Result of applying filter $\vec{h}_{n,k,m}^{\sf H}$ to extract user $n$. Contains $\vec{h}_{n,k,m}$ because it is the receiver's mathematical filter.

---

## After Equation 7 (SIC Decoding Margin & Offload Coupling)

$$p_n\mathsf{g}_n\big/\sum_{i\in\mathcal{W}_{n,k,m}}p_i\mathsf{g}_i \ge \epsilon_m^{\tt th}$$

### Explanation of $a_n^{\tt ul} = \sum_k a_{n,k}^{\tt ul}$ and the Coupling Constraint $\rho_n \le a_n^{\tt ul}$:

#### 1. Mathematical Definitions:
1. **$a_{n,k}^{\tt ul} \in \{0, 1\}$:** Binary sub-channel allocation indicator ($a_{n,k}^{\tt ul} = 1$ if UE $n$ uses sub-channel $k$).
2. **$a_n^{\tt ul} \triangleq \sum_{k=1}^K a_{n,k}^{\tt ul} \in \{0, 1\}$:** Overall wireless channel access indicator.
   * $a_n^{\tt ul} = 1$: UE $n$ is allocated a sub-channel.
   * $a_n^{\tt ul} = 0$: UE $n$ has no sub-channel.
3. **$\rho_n \in [0, 1]$:** Partial offloading split ratio ($\rho_n$ computed on UAV, $1-\rho_n$ computed locally).

#### 2. Physical Role of $\rho_n \le a_n^{\tt ul}$:
* **If $a_n^{\tt ul} = 0$ (No Channel):** Enforces $\rho_n \le 0 \implies \mathbf{\rho_n = 0}$. 
  The UE executes 100% locally ($1 - \rho_n = 1$) and does not attempt wireless transmission, preventing a division by zero ($\frac{\dots}{R_n = 0} = \infty$).
* **If $a_n^{\tt ul} = 1$ (Channel Granted):** $\rho_n \le 1$, allowing the UE to optimize its partial offloading split $\rho_n \in [0, 1]$ freely.

---

## After Equation 12 (Downlink SIJNR Under Zero-Forcing Beamforming)

$$\gamma_{m',k,n'}^{\tt dl} = \frac{q_{m',n'}}{\mathcal{I}_{n'}^{\tt inter} + \mathcal{I}_{n'}^{\tt CCI} + \mathcal{J}_{n'}^{\tt dl} + \sigma_{k,n'}^2}$$

### 1. Physical Meaning of the Equation
This equation models the Signal-to-Interference-plus-Jamming-and-Noise Ratio (SIJNR) of a downlink ground UE $n'$ served by Downlink UAV $m'$ broadcasting over sub-channel $k$.

---

### 2. Why the Numerator is Simply $q_{m',n'}$ (Zero-Forcing Precoding)

1. **Downlink Transmit Signal from UAV $m'$:**
   DL UAV $m'$ has $L$ antennas and transmits the multi-user precoded vector:
   $$\vec{x}_{m',k} = \sum_{i \in \mathcal{N}_{m'}^{\tt dl}} \sqrt{q_{m',i}} \vec{w}_{m',i} s_i$$
   where $\vec{w}_{m',i} \in \mathbb{C}^{L \times 1}$ is the beamforming vector (column of $\vec{W}_{m',k}$) and $q_{m',i}$ is the DL transmit power allocated to UE $i$.

2. **Signal Received at DL UE $n'$:**
   The signal arrives at ground UE $n'$ through its channel $\vec{h}_{n',k,m'}^{\sf H} \in \mathbb{C}^{1 \times L}$:
   $$y_{n'}^{\tt dl} = \underbrace{\vec{h}_{n',k,m'}^{\sf H} \vec{w}_{m',n'} \sqrt{q_{m',n'}} s_{n'}}_{\text{Desired Signal for UE } n'} + \underbrace{\sum_{i \neq n'} \vec{h}_{n',k,m'}^{\sf H} \vec{w}_{m',i} \sqrt{q_{m',i}} s_i}_{\text{Intra-Cell Interference from other UEs in same cell}} + \text{Interference} + \text{Noise}$$

3. **The Zero-Forcing Beamforming (ZFBF) Inversion:**
   The precoder $\vec{W}_{m',k} = \vec{H}^{\sf H} (\vec{H}\vec{H}^{\sf H})^{-1}$ is designed to satisfy:
   $$\vec{h}_{n',k,m'}^{\sf H} \vec{w}_{m',i} = \begin{cases} 1, & \text{if } i = n' \text{ (desired user)} \\ 0, & \text{if } i \neq n' \text{ (intra-cell co-channel users)} \end{cases}$$

4. **Consequences:**
   * **Desired Signal Power (Numerator):** 
     $$\mathbb{E}\left[\left|\vec{h}_{n',k,m'}^{\sf H} \vec{w}_{m',n'} \sqrt{q_{m',n'}} s_{n'}\right|^2\right] = |1 \cdot \sqrt{q_{m',n'}}|^2 = \mathbf{q_{m',n'}}$$
   * **Intra-Cell Interference:** 
     $$\sum_{i \neq n'} \left|\vec{h}_{n',k,m'}^{\sf H} \vec{w}_{m',i}\right|^2 q_{m',i} = \sum_{i \neq n'} |0|^2 q_{m',i} = \mathbf{0}$$
     ZFBF completely cancels all intra-cell interference among UEs served by the same UAV $m'$.

---

### 3. Breakdown of the Denominator Terms

| Denominator Term | Name | Physical Mechanism & Origin |
| :--- | :--- | :--- |
| **$\mathcal{I}_{n'}^{\tt inter}$** | **Inter-Cell DL Interference** | $\sum_{j\neq m'}\sum_{i\in\mathcal{N}_j^{\tt dl}}q_{j,i}|\vec{h}_{n',k,j}^{\sf H}\vec{w}_{j,k,i}|^2$. Other DL UAVs $j \neq m'$ broadcast on sub-channel $k$ to their own UEs. Their beamformers $\vec{w}_{j,k,i}$ do not null interference at UE $n'$ in neighboring cells. |
| **$\mathcal{I}_{n'}^{\tt CCI}$** | **UL-to-DL Co-Channel Interference** | $\sum_{i\in\mathcal{N}_k^{\tt sub}}p_i|\ell_{i,k,n'}|^2$. Because Uplink and Downlink share the same sub-channel $k$, ground UEs $i$ offloading on UL leak power into DL ground UE $n'$ over Ground-to-Ground (G2G) channels $\ell_{i,k,n'}$. |
| **$\mathcal{J}_{n'}^{\tt dl}$** | **Hostile Ground Jamming** | $\sum_{q}a_{q,k}|h_{q,k,n'}|^2 p_q$. Malicious ground jammers $q$ radiating jamming power $p_q$ on sub-channel $k$ to corrupt the downlink broadcast. |
| **$\sigma_{k,n'}^2$** | **Thermal Noise** | $B_k N_0$. Ambient AWGN background noise across sub-channel bandwidth $B_k$. |

---

### 4. Downlink Power Budget & Minimum Rate Constraints

* **Total DL UAV Transmit Power:**
  $$\sum_{n' \in \mathcal{N}_{m'}^{\tt dl}} q_{m',n'} \|\vec{w}_{m',n'}\|^2 \le P_{m'}^{\tt max}$$
  where $\|\vec{w}_{m',n'}\|^2$ is the beamforming weight penalty required by ZF to null interference.
* **Achievable Downlink Rate:**
  $$R_{n'}^{\tt dl} = \sum_{m',k} a_{m',k}^{\tt dl} B_k \log_2(1 + \gamma_{m',k,n'}^{\tt dl}) \quad \text{[bits/s]}$$
* **Quality-of-Service (QoS) Threshold:** $\gamma_{m',k,n'}^{\tt dl} \ge \gamma_{\tt th}$ guarantees reliable broadcast reception.

---

## After Equation 13 (Uplink Multi-Objective Task Utility)

$$\mathbb{U}_n^{\tt ul} = \beta_n^{\tt t}\frac{\mathsf{T}_n^{\tt ref}-\mathsf{T}_n}{\mathsf{T}_n^{\tt ref}} + \beta_n^{\tt e}\frac{\mathsf{E}_n^{\tt ref}-\mathsf{E}_n}{\mathsf{E}_n^{\tt ref}} + \beta_n^{\tt a}\frac{\Lambda_n-\Lambda^{\tt th}}{1-\Lambda^{\tt th}}$$

$$\text{with } \beta_n^{\tt t} + \beta_n^{\tt e} + \beta_n^{\tt a} = 1, \quad \beta_n^{\tt t}, \beta_n^{\tt e}, \beta_n^{\tt a} \ge 0$$

---

### 1. What are $\mathsf{T}_n^{\tt ref}$ and $\mathsf{E}_n^{\tt ref}$?

From Equation 4 (Section II-B):
$$\mathsf{T}_n^{\tt ref} = \mathsf{T}_n^{\tt sen} + \mathsf{T}_n^{\tt ext} + \frac{C_n^{\tt raw}}{F_n^{\tt loc}}$$
$$\mathsf{E}_n^{\tt ref} = \mathsf{E}_n^{\tt sen} + \mathsf{E}_n^{\tt ext} + \kappa_n C_n^{\tt raw}(F_n^{\tt loc})^2$$

* **Physical Meaning:** $\mathsf{T}_n^{\tt ref}$ and $\mathsf{E}_n^{\tt ref}$ represent the **Reference Baseline Costs** of UE $n$ under the default standalone execution scenario:
  1. **All-Local Compute ($\rho_n = 0$):** The UE does not offload any task to the UAV server; 100% of the workload is executed locally on the UE CPU ($F_n^{\tt loc}$).
  2. **Full Feature Retention ($\chi_n = 1$):** The UE does not prune any features; it computes on the full raw workload $C_n^{\tt raw}$.
* **Normalization Role:**
  * **$\frac{\mathsf{T}_n^{\tt ref} - \mathsf{T}_n}{\mathsf{T}_n^{\tt ref}}$:** The **normalized fractional delay reduction (latency savings)** relative to baseline. When $\mathsf{T}_n < \mathsf{T}_n^{\tt ref}$, this term is positive in $[0, 1)$, rewarding the system for executing faster.
  * **$\frac{\mathsf{E}_n^{\tt ref} - \mathsf{E}_n}{\mathsf{E}_n^{\tt ref}}$:** The **normalized fractional energy savings** relative to baseline. When $\mathsf{E}_n < \mathsf{E}_n^{\tt ref}$, this term is positive in $[0, 1)$, rewarding the system for saving device battery energy.

---

### 2. What does $\frac{\Lambda_n - \Lambda^{\tt th}}{1 - \Lambda^{\tt th}}$ mean?

* **Components:**
  * **$\Lambda_n(\chi_n) = 1 - e^{-\vartheta_n \chi_n \Gamma_n} \in [\Lambda^{\tt th}, 1)$:** The actual inference accuracy achieved at retention ratio $\chi_n$.
  * **$\Lambda^{\tt th} \in (0, 1)$:** The minimum required inference accuracy threshold (QoS requirement, e.g., 0.85).
  * **$1$:** The theoretical maximum achievable accuracy bound (100% classification accuracy).

* **Physical Meaning:**
  * $\Lambda_n - \Lambda^{\tt th}$ represents the **excess accuracy surplus** achieved beyond the mandatory threshold.
  * $1 - \Lambda^{\tt th}$ represents the **maximum possible accuracy headroom** above the threshold.
  * Therefore, $\frac{\Lambda_n - \Lambda^{\tt th}}{1 - \Lambda^{\tt th}}$ is the **normalized accuracy gain / accuracy satisfaction metric**, scaling strictly into $[0, 1)$:
    * **At $\Lambda_n = \Lambda^{\tt th}$ (boundary minimum):** $\frac{\Lambda^{\tt th} - \Lambda^{\tt th}}{1 - \Lambda^{\tt th}} = \mathbf{0}$ (zero bonus utility).
    * **As $\Lambda_n \to 1$ (near-perfect accuracy):** $\frac{1 - \Lambda^{\tt th}}{1 - \Lambda^{\tt th}} = \mathbf{1}$ (maximum bonus utility of 1).

---

### 3. Why This Normalization Unifies ISCC Optimization

1. **Uniform Dimensionless Scale:** Delay savings, energy savings, and accuracy gain are all mapped to the **identical dimensionless range $[0, 1)$**.
2. **Convex Combination ($\sum \beta = 1$):** The weighting coefficients $(\beta_n^{\tt t}, \beta_n^{\tt e}, \beta_n^{\tt a})$ act as pure preference weights without requiring artificial unit scaling factors:
   * Latency-sensitive tasks $\rightarrow$ increase $\beta_n^{\tt t}$.
   * Battery-limited IoT devices $\rightarrow$ increase $\beta_n^{\tt e}$.
   * High-precision AI inference $\rightarrow$ increase $\beta_n^{\tt a}$.

---

## After Equation 20 (The Power Control Objective Function $\mathbb{W}$)

$$\mathbb{W} = \sum_{n: a_n^{\tt ul} = 1} \frac{\eta_n + \theta_n p_n}{R_{n,k,m}}$$
$$\eta_n = \frac{\beta_n^{\tt t} \chi_n \rho_n D_n^{\tt raw}}{\mathsf{T}_n^{\tt ref}}, \qquad \theta_n = \frac{\beta_n^{\tt e} \chi_n \rho_n D_n^{\tt raw}}{\xi_n \mathsf{E}_n^{\tt ref}}$$

---

### 1. Why Do We Consider $-\sum_n \mathbb{U}_n^{\tt ul}$? (Is It Incorrect?)

**No, it is 100% mathematically correct.**

* **Maximizing Utility $\equiv$ Minimizing Negative Utility (Cost):**
  In mathematical optimization, maximizing a utility function is identical to minimizing its negative:
  $$\max_{\vec{p}} \sum_{n} \mathbb{U}_n^{\tt ul} \iff \min_{\vec{p}} \left( -\sum_{n} \mathbb{U}_n^{\tt ul} \right)$$
* **Standard Canonical Form:** Optimization algorithms (e.g., fractional programming, bisection, and metaheuristics like WOA/PSO) are conventionally defined in standard **minimization form**: $\min_{\vec{p}} \mathbb{W}(\vec{p})$.
* By taking $-\mathbb{U}_n^{\tt ul}$, the reward function is converted into an **aggregate transmission cost function $\mathbb{W}$** (measuring the combined delay and battery energy penalty), which we seek to minimize.

---

### 2. Step-by-Step Derivation of $\mathbb{W}$

#### Step 1: Write Out $-\mathbb{U}_n^{\tt ul}$
Starting from the definition of $\mathbb{U}_n^{\tt ul}$ (Equation 13):
$$\mathbb{U}_n^{\tt ul} = \beta_n^{\tt t}\frac{\mathsf{T}_n^{\tt ref}-\mathsf{T}_n}{\mathsf{T}_n^{\tt ref}} + \beta_n^{\tt e}\frac{\mathsf{E}_n^{\tt ref}-\mathsf{E}_n}{\mathsf{E}_n^{\tt ref}} + \beta_n^{\tt a}\frac{\Lambda_n-\Lambda^{\tt th}}{1-\Lambda^{\tt th}}$$

Taking the negative $-\mathbb{U}_n^{\tt ul}$:
$$-\mathbb{U}_n^{\tt ul} = -\beta_n^{\tt t}\frac{\mathsf{T}_n^{\tt ref}-\mathsf{T}_n}{\mathsf{T}_n^{\tt ref}} - \beta_n^{\tt e}\frac{\mathsf{E}_n^{\tt ref}-\mathsf{E}_n}{\mathsf{E}_n^{\tt ref}} - \beta_n^{\tt a}\frac{\Lambda_n-\Lambda^{\tt th}}{1-\Lambda^{\tt th}}$$
$$-\mathbb{U}_n^{\tt ul} = \frac{\beta_n^{\tt t}}{\mathsf{T}_n^{\tt ref}}\mathsf{T}_n + \frac{\beta_n^{\tt e}}{\mathsf{E}_n^{\tt ref}}\mathsf{E}_n \underbrace{- \beta_n^{\tt t} - \beta_n^{\tt e} - \beta_n^{\tt a}\frac{\Lambda_n-\Lambda^{\tt th}}{1-\Lambda^{\tt th}}}_{\text{Constant with respect to } \vec{p}}$$

---

#### Step 2: Identify What Depends on Transmit Power $\vec{p}$
For fixed sub-channels $\vec{A}^{\tt ul}$, compute allocation $\vec{F}$, offload split $\vec{\rho}$, and feature retention $\vec{\chi}$:
* Sensing costs ($\mathsf{T}_n^{\tt sen}, \mathsf{E}_n^{\tt sen}$), feature extraction ($\mathsf{T}_n^{\tt ext}, \mathsf{E}_n^{\tt ext}$), local compute delay ($\Delta_n^{\tt loc}$), local compute energy, and accuracy ($\Lambda_n$) do **NOT** depend on $p_n$.
* The transmit power $p_n$ and the achievable uplink rate $R_{n,k,m}(\vec{p})$ appear **exclusively in two places**:
  1. **In Delay $\mathsf{T}_n$ (via the wireless offload airtime):**
     $$\mathsf{T}_n = \text{const} + \chi_n \rho_n \frac{D_n^{\tt raw}}{R_{n,k,m}}$$
  2. **In Energy $\mathsf{E}_n$ (via the wireless transmit energy):**
     $$\mathsf{E}_n = \text{const} + \chi_n \frac{\rho_n p_n D_n^{\tt raw}}{\xi_n R_{n,k,m}}$$

---

#### Step 3: Substitute Delay and Energy Terms into $-\mathbb{U}_n^{\tt ul}$
Substitute the $p$-dependent terms into the weighted cost:
$$-\mathbb{U}_n^{\tt ul} = \text{const} + \frac{\beta_n^{\tt t}}{\mathsf{T}_n^{\tt ref}}\left( \frac{\chi_n \rho_n D_n^{\tt raw}}{R_{n,k,m}} \right) + \frac{\beta_n^{\tt e}}{\mathsf{E}_n^{\tt ref}}\left( \frac{\chi_n \rho_n p_n D_n^{\tt raw}}{\xi_n R_{n,k,m}} \right)$$

Factor out $1 / R_{n,k,m}$:
$$-\mathbb{U}_n^{\tt ul} = \text{const} + \frac{1}{R_{n,k,m}} \left[ \underbrace{\left(\frac{\beta_n^{\tt t}\chi_n\rho_n D_n^{\tt raw}}{\mathsf{T}_n^{\tt ref}}\right)}_{\triangleq\, \eta_n} + \underbrace{\left(\frac{\beta_n^{\tt e}\chi_n\rho_n D_n^{\tt raw}}{\xi_n\mathsf{E}_n^{\tt ref}}\right)}_{\triangleq\, \theta_n} p_n \right]$$
$$-\mathbb{U}_n^{\tt ul} = \text{const} + \frac{\eta_n + \theta_n p_n}{R_{n,k,m}}$$

---

#### Step 4: Sum Across All Active Offloading Users ($a_n^{\tt ul} = 1$)
Summing over all UEs that are granted a wireless sub-channel:
$$-\sum_{n} \mathbb{U}_n^{\tt ul} = \text{Total Constants} + \sum_{n: a_n^{\tt ul} = 1} \frac{\eta_n + \theta_n p_n}{R_{n,k,m}}$$

Dropping the additive constants (which do not affect the minimizer $\vec{p}$), we obtain the exact power control objective function $\mathbb{W}$:
$$\mathbf{\mathbb{W}(\vec{p}) \triangleq \sum_{n: a_n^{\tt ul} = 1} \frac{\eta_n + \theta_n p_n}{R_{n,k,m}}} \quad \blacksquare$$

---

### 3. Physical Intuition of the Terms in $\mathbb{W}$

* **$\frac{\eta_n}{R_{n,k,m}}$ (Delay Cost):** 
  Measures the weighted wireless transmission airtime. As transmit power $p_n$ increases, the uplink rate $R_{n,k,m}$ increases, driving this delay penalty **down**.
* **$\frac{\theta_n p_n}{R_{n,k,m}}$ (Battery Energy Cost):** 
  Measures the weighted battery energy consumption ($p_n \times \text{airtime}$). Since both $p_n$ and $R_{n,k,m}$ grow with power, this term drives the power **down** to conserve battery.
* **The Trade-Off in $\mathbb{W}$:** 
  Minimizing $\mathbb{W}$ finds the exact sweet spot for $p_n$ that achieves high transmission speed (small delay penalty) without wasting excessive battery energy or causing overwhelming interference to other UEs.

---

### 4. In-Depth Explanation of the Commentary After Equation 20

> *"The ISCC variables enter only as the scaling $\chi_n\rho_n$ of $(\eta_n,\theta_n)$, so $\mathbb{W}$ retains its atomic-task structure and Lemma 1 applies at any fixed $(\rho_n,\chi_n)$. Equation (20) assumes the offload branch active ($\rho_n\ge\rho_n^\star$); on the local branch $\eta_n=0$ and all statements below still hold."*

This commentary contains two important theoretical insights:

#### Insight A: Why ISCC Retains the "Atomic-Task Structure"
* **What is the Atomic-Task Baseline?**
  In conventional mobile edge computing without feature compression ($\chi_n = 1$) and without partial splitting ($\rho_n = 1$, full all-or-nothing task offloading), the coefficients are fixed constants:
  $$\eta_n^{\text{atomic}} = \frac{\beta_n^{\tt t} D_n^{\tt raw}}{\mathsf{T}_n^{\tt ref}}, \qquad \theta_n^{\text{atomic}} = \frac{\beta_n^{\tt e} D_n^{\tt raw}}{\xi_n \mathsf{E}_n^{\tt ref}}$$
* **How ISCC Modifies the Problem:**
  Introducing feature retention $\chi_n$ and partial offloading $\rho_n$ simply multiplies both coefficients by the scalar factor $(\chi_n \rho_n)$:
  $$\eta_n = (\chi_n \rho_n) \cdot \eta_n^{\text{atomic}}, \qquad \theta_n = (\chi_n \rho_n) \cdot \theta_n^{\text{atomic}}$$
* **Mathematical Consequence:**
  The functional form of the power control objective $\mathbb{W}(\vec{p}) = \sum_n \frac{\eta_n + \theta_n p_n}{R_{n,k,m}(\vec{p})}$ is **completely unchanged** in the $p$-domain. Adding ISCC compression does not introduce any new non-linearities or cross-coupling into the power optimization. Therefore, the **quasiconvexity proved in Lemma 1 and the 1D bisection root-finding method remain 100% valid** for any fixed $(\chi_n, \rho_n)$.

---

#### Insight B: Why $\rho_n \ge \rho_n^\star$ Means "Offload Branch Active" & Why $\eta_n = 0$ on the Local Branch

Total compute latency is determined by the parallel execution of the local device CPU and the remote UAV:
$$\Delta_n(\rho_n) = \max\left\{ \Delta_n^{\tt loc}(\rho_n), \; \Delta_n^{\tt ofl}(\rho_n) \right\}$$

##### 1. The Monotonicity of the Two Parallel Delay Branches:
* **Local Delay:** $\Delta_n^{\tt loc}(\rho_n) = (1 - \rho_n) \frac{C_n^{\tt raw}}{F_n^{\tt loc}}$
  * **Strictly decreasing** with respect to $\rho_n$.
  * At $\rho_n = 0$ (all-local), $\Delta_n^{\tt loc}$ is at its maximum $\frac{C_n^{\tt raw}}{F_n^{\tt loc}}$.
  * As $\rho_n$ increases, the phone retains less workload ($1-\rho_n$), so local compute delay drops linearly down to $0$ at $\rho_n = 1$.
* **Remote Offload Delay:** $\Delta_n^{\tt ofl}(\rho_n) = \rho_n \left( \frac{D_n^{\tt raw}}{R_n} + \frac{C_n^{\tt raw}}{F_{nm}} \right)$
  * **Strictly increasing** with respect to $\rho_n$.
  * At $\rho_n = 0$, $\Delta_n^{\tt ofl} = 0$ (no data transmitted or computed remotely).
  * As $\rho_n$ increases, more data bits are transmitted across the wireless channel and more cycles are processed on the UAV, so remote delay grows linearly from $0$ up to its maximum at $\rho_n = 1$.

##### 2. The Balance Point $\rho_n^\star$:
Because $\Delta_n^{\tt loc}(\rho_n)$ starts high and decreases to $0$, while $\Delta_n^{\tt ofl}(\rho_n)$ starts at $0$ and increases, their curves cross at a unique equilibrium point $\rho_n^\star \in (0, 1)$ where both branches take equal time:
$$\Delta_n^{\tt loc}(\rho_n^\star) = \Delta_n^{\tt ofl}(\rho_n^\star) \iff (1 - \rho_n^\star)\frac{C_n^{\tt raw}}{F_n^{\tt loc}} = \rho_n^\star \left(\frac{D_n^{\tt raw}}{R_n} + \frac{C_n^{\tt raw}}{F_{nm}}\right)$$
$$\implies \rho_n^\star = \frac{\frac{C_n^{\tt raw}}{F_n^{\tt loc}}}{\frac{C_n^{\tt raw}}{F_n^{\tt loc}} + \frac{D_n^{\tt raw}}{R_n} + \frac{C_n^{\tt raw}}{F_{nm}}}$$

```
Latency
  ^
  |                   \                     /  <-- \Delta_n^{ofl}(\rho_n) [Increasing]
  |                    \                   /
  |                     \                 /
  |                      \   Intersection/
  |                       \     \rho_n* /
  |                        \     .     /
  |                         \    .    /
  |                          \   .   /
  |                           \  .  /
  |                            \ . /
  |                   ----------\./-----------
  |                              .
  |         Local Branch Active  |  Offload Branch Active
  |  (\Delta_n^{loc} dominates)  |  (\Delta_n^{ofl} dominates)
  +----------------------------------------------------> \rho_n
 0                            \rho_n*                     1
```

##### 3. The Two Operational Regimes:

1. **Regime 1: $\rho_n \ge \rho_n^\star$ $\implies$ "Offload Branch Active / Bottleneck":**
   * **Physical Mechanism:** When the offloading ratio $\rho_n$ is large (greater than or equal to $\rho_n^\star$), the UE sends a heavy workload to the UAV. The local phone has very little work left ($1-\rho_n$ is small) and finishes early. Meanwhile, transmitting and computing the large workload remotely takes longer.
   * **Mathematical Condition:**
     $$\Delta_n^{\tt ofl}(\rho_n) \ge \Delta_n^{\tt loc}(\rho_n) \implies \Delta_n = \max\{\Delta_n^{\tt loc}, \Delta_n^{\tt ofl}\} = \mathbf{\Delta_n^{\tt ofl}(\rho_n)}$$
   * **Impact on Power Control:** The total delay $\mathsf{T}_n$ directly depends on the wireless transmission time $\frac{\rho_n D_n^{\tt raw}}{R_{n,k,m}}$. Speeding up the wireless link $R_{n,k,m}$ via power $p_n$ directly reduces total mission delay.
   * **Active Delay Coefficient:**
     $$\eta_n = \frac{\beta_n^{\tt t} \chi_n \rho_n D_n^{\tt raw}}{\mathsf{T}_n^{\tt ref}} > 0$$

2. **Regime 2: $\rho_n < \rho_n^\star$ $\implies$ "Local Branch Active / Bottleneck":**
   * **Physical Mechanism:** When $\rho_n$ is small (less than $\rho_n^\star$), the UE offloads only a tiny fraction. The UAV finishes almost immediately, but the local device's CPU is overloaded with the remaining $(1-\rho_n)$ bulk workload.
   * **Mathematical Condition:**
     $$\Delta_n^{\tt loc}(\rho_n) > \Delta_n^{\tt ofl}(\rho_n) \implies \Delta_n = \max\{\Delta_n^{\tt loc}, \Delta_n^{\tt ofl}\} = \mathbf{\Delta_n^{\tt loc}(\rho_n)}$$
   * **Impact on Power Control:** Total execution delay is dictated **entirely by the local phone CPU**: $\Delta_n = (1-\rho_n)\frac{C_n^{\tt raw}}{F_n^{\tt loc}}$.
   * In this regime, even if we increase transmit power to make wireless transmission infinitely fast ($R_{n,k,m} \to \infty$), it produces **zero total delay savings** because the overall system is idling and waiting for the local phone CPU to complete!
   * **Latency Sensitivity Drops to Zero ($\eta_n = 0$):** Since wireless rate does not affect the bottleneck delay, the delay sensitivity weight drops out ($\mathbf{\eta_n = 0}$).
   * In this regime, transmit power $p_n$ only affects battery energy consumption $\frac{\theta_n p_n}{R_{n,k,m}}$.

##### 4. Why All Analytical Proofs Hold Unconditionally:
Setting $\eta_n = 0$ on the local branch is simply a non-negative boundary case ($\eta_n \ge 0$). The function:
$$w_n(p) = \frac{0 + \theta_n p}{B_k \log_2(1 + \bar{h}_n p)}$$
remains strictly quasiconvex, and all analytical properties (Lemma 1, bisection convergence, and global optimality) hold unconditionally across both branches.

---

## After Equation 21 (The Separable Lower Bound $\widetilde{\mathbb{W}} \le \mathbb{W}$)

$$\widetilde{\mathbb{W}}(\vec{A}^{\tt ul},\vec{p}) = \sum_{n:a_n^{\tt ul}=1} \frac{\eta_n+\theta_n p_n}{B_k\log_2(1+\bar h_n p_n)} \le \mathbb{W}(\vec{A}^{\tt ul},\vec{p})$$
$$\text{with } \bar{h}_n = \frac{\mathsf{g}_n}{B_k N_0}$$

---

### 1. Step-by-Step Mathematical Derivation of the Inequality

#### Step 1: Physical Uplink SIJNR vs. Interference-Free SNR
From Equation 6, the actual received Signal-to-Interference-plus-Jamming-and-Noise Ratio (SIJNR) is:
$$\gamma_{n,k,m} = \frac{\mathsf{g}_n^2 p_n}{\mathcal{I}_{n,k,m}^{\tt w} + \mathcal{I}_{n,k,m}^{\tt o} + \mathcal{J}_{n,k,m} + \mathsf{g}_n(\Xi_{k,m}/L + B_k N_0)}$$

All interference, leakage, and jamming power terms are strictly non-negative physical quantities:
$$\mathcal{I}_{n,k,m}^{\tt w} \ge 0, \quad \mathcal{I}_{n,k,m}^{\tt o} \ge 0, \quad \mathcal{J}_{n,k,m} \ge 0, \quad \Xi_{k,m} \ge 0$$

If we drop all interference terms ($\mathcal{I}^{\tt w} = 0, \mathcal{I}^{\tt o} = 0$), jamming ($\mathcal{J} = 0$), and DL leakage ($\Xi = 0$), the denominator achieves its absolute physical minimum (thermal noise floor only):
$$\text{Denominator} \ge \mathsf{g}_n B_k N_0$$

Therefore, the actual SIJNR is upper-bounded by the idealized interference-free Signal-to-Noise Ratio (SNR):
$$\gamma_{n,k,m} \le \frac{\mathsf{g}_n^2 p_n}{\mathsf{g}_n B_k N_0} = \frac{\mathsf{g}_n p_n}{B_k N_0} = \bar{h}_n p_n \triangleq \tilde{\gamma}_n$$
where $\bar{h}_n \triangleq \frac{\mathsf{g}_n}{B_k N_0}$ is the channel-to-noise ratio.

---

#### Step 2: Monotonicity of the Achievable Rate
Because the Shannon capacity function $B_k \log_2(1 + x)$ is **strictly monotonically increasing** with respect to SINR $x$:
$$\gamma_{n,k,m} \le \bar{h}_n p_n \implies B_k \log_2(1 + \gamma_{n,k,m}) \le B_k \log_2(1 + \bar{h}_n p_n)$$
$$\mathbf{R_{n,k,m} \le \widetilde{R}_n \triangleq B_k \log_2(1 + \bar{h}_n p_n)}$$
That is, the true achievable data rate $R_{n,k,m}$ in an interference/jamming environment is **always smaller than or equal to** the interference-free rate $\widetilde{R}_n$.

---

#### Step 3: Reciprocal Inversion in the Objective Function
In the power cost function $\mathbb{W}$, the transmission rate appears in the **denominator**:
$$\mathbb{W} = \sum_{n:a_n^{\tt ul}=1} \frac{\eta_n + \theta_n p_n}{R_{n,k,m}}$$

Since the numerator $(\eta_n + \theta_n p_n) \ge 0$ is non-negative and identical in both expressions, taking the reciprocal reverses the inequality:
$$R_{n,k,m} \le \widetilde{R}_n \implies \frac{1}{\widetilde{R}_n} \le \frac{1}{R_{n,k,m}}$$
$$\implies \frac{\eta_n + \theta_n p_n}{B_k \log_2(1 + \bar{h}_n p_n)} \le \frac{\eta_n + \theta_n p_n}{R_{n,k,m}}$$

---

#### Step 4: Summing Across All Active Offloading UEs ($a_n^{\tt ul} = 1$)
Summing both sides over all active offloading UEs yields the lower-bound inequality:
$$\mathbf{\widetilde{\mathbb{W}}(\vec{A}^{\tt ul},\vec{p}) = \sum_{n:a_n^{\tt ul}=1} \frac{\eta_n+\theta_n p_n}{B_k\log_2(1+\bar h_n p_n)} \le \sum_{n:a_n^{\tt ul}=1} \frac{\eta_n+\theta_n p_n}{R_{n,k,m}} = \mathbb{W}(\vec{A}^{\tt ul},\vec{p})} \quad \blacksquare$$

---

### 2. Physical and Algorithmic Significance: Why Do We Need $\widetilde{\mathbb{W}}$?

| Property | Original Objective $\mathbb{W}(\vec{p})$ | Separable Lower Bound $\widetilde{\mathbb{W}}(\vec{p})$ |
| :--- | :--- | :--- |
| **Coupling** | **Coupled:** The rate $R_{n,k,m}(\vec{p})$ depends on all co-channel powers $(p_1, \dots, p_N)$ through interference $\mathcal{I}^{\tt w}, \mathcal{I}^{\tt o}$. | **Decoupled (Separable):** Each term depends **only** on user $n$'s own transmit power $p_n$. |
| **Convexity / Tractability** | **Non-convex:** Joint multi-user power control in interference networks is NP-hard. | **Sum of 1D Quasiconvex Functions:** Each component $w_n(p_n)$ is strictly quasiconvex (Lemma 1). |
| **Solvability** | Requires complex iterative heuristic search (e.g. WOA/PSO). | Can be solved to **exact global optimality in microseconds** via 1D bisection root-finding on each UE independently. |
| **Role in the Paper** | Represents the true physical system performance under full interference and jamming. | 1. Provides a rigorous theoretical benchmark / lower bound on transmission cost.<br>2. Used for channel-aware heuristic pruning (Lemma 2) and high-quality initializations for swarm solvers. |

---

## Before Equation 23 (Upper-Bound Utility Filter $\mathbb{F}(\vec{A})$ for Association Pruning)

$$\mathbb{F}(\vec{A}) = \sum_{n}\bar u_n(\vec{A}^{\tt ul}) + \mathbb{U}^{\tt dl}_{\mathrm{ub}}(\vec{A}^{\tt dl})$$

---

### 1. In-Depth Explanation of the Text Before Equation 22

> *"Pruning must act before $(\vec{\rho},\vec{\chi})$ are known, and both the compute and the power cost scale with $\chi_n\rho_n$, so the conventional atomic-task bound is no longer valid. Instead, for each offloading UE replace $R_n$ by the interference-free rate $\bar R_n=B_k\log_2(1+\bar h_np_n^{\tt max})$, $F_{nm}$ by $F_m^{\tt max}$, and the offload energy $p_nD_n^{\tt raw}/(\xi_nR_n)$ by its floor $D_n^{\tt raw}\ln2/(\xi_nB_k\bar h_n)$, which follows from $\log_2(1+x)\le x/\ln2$. Each replacement lowers $\Delta_n$ and $\Theta_n$ pointwise, so $\bar\Omega_n(\rho_n)$ lower-bounds $\Omega_n(\rho_n)$ for every $(\vec{p},\vec{F})$. With $\bar u_n$ the maximum of $\mathbb{U}_n^{\tt ul}$ under $\bar\Omega_n$, obtained exactly by Propositions~\ref{prop:rho}--\ref{prop:chi}, let $\mathbb{F}(\vec{A})=\sum_{n}\bar u_n(\vec{A}^{\tt ul})+\mathbb{U}^{\tt dl}_{\mathrm{ub}}(\vec{A}^{\tt dl})$."*

This paragraph addresses a fundamental challenge in ISCC optimization: **How can we evaluate whether a discrete sub-channel association $\vec{A}$ is worth optimizing before actually spending computational time optimizing the continuous variables $(\vec{p}, \vec{F}, \vec{\rho}, \vec{\chi})$?**

---

### 2. Breakdown of the 3 Key Concepts

#### Concept A: Why the Atomic-Task Bound Fails in ISCC
* In conventional MEC systems with atomic tasks ($\chi_n = 1, \rho_n = 1$), task size is static and fixed.
* In ISCC, task payload and computation shrink dynamically by the product $(\chi_n \rho_n)$. Because $\chi_n$ and $\rho_n$ adaptively trade off delay, battery energy, and inference accuracy, their optimal values depend on the marginal delay-energy price $\Omega_n(\rho_n)$.
* Therefore, a simple heuristic bound cannot assume $(\chi_n = 1, \rho_n = 1)$. The bound must hold **universally across all continuous configurations $(\vec{p}, \vec{F}, \vec{\rho}, \vec{\chi})$**.

---

#### Concept B: The 3 Physical Resource Limit Substitutions (Pointwise Lower Bounds)
To construct an upper bound on utility $\mathbb{U}_n^{\tt ul}$, we must construct a **pointwise lower bound** on delay $\Delta_n$ and energy $\Theta_n$. We replace each bottleneck by its absolute best-case physical capacity:

1. **Limit 1: Maximum Interference-Free Data Rate ($\bar{R}_n$):**
   * Real data rate $R_{n,k,m}(\vec{p})$ suffers from multi-user interference, jamming, and finite transmit power.
   * In the best possible universe (zero interference, zero jammers, full power $p_n^{\tt max}$):
     $$R_{n,k,m}(\vec{p}) \le \bar{R}_n \triangleq B_k \log_2(1 + \bar{h}_n p_n^{\tt max})$$
   * Replacing $R_n$ by $\bar{R}_n$ strictly lowers transmission delay: $\frac{D_n^{\tt raw}}{\bar{R}_n} \le \frac{D_n^{\tt raw}}{R_n}$.

2. **Limit 2: Maximum Dedicated UAV Server Frequency ($F_m^{\tt max}$):**
   * In reality, multiple users share the UAV server capacity: $\sum_n F_{nm} \le F_m^{\tt max} \implies F_{nm} \le F_m^{\tt max}$.
   * In the ideal case, the entire server capacity is dedicated 100% to user $n$:
     $$\frac{C_n^{\tt raw}}{F_m^{\tt max}} \le \frac{C_n^{\tt raw}}{F_{nm}}$$

3. **Limit 3: Fundamental Shannon Energy Floor per Bit:**
   * Ground UE transmission energy is:
     $$E_n^{\tt ofl} = \frac{p_n D_n^{\tt raw}}{\xi_n R_n(p_n)} \ge \frac{p_n D_n^{\tt raw}}{\xi_n B_k \log_2(1 + \bar{h}_n p_n)}$$
   * Using the standard natural logarithm inequality $\ln(1 + x) \le x$, we have:
     $$\log_2(1 + x) = \frac{\ln(1 + x)}{\ln 2} \le \frac{x}{\ln 2}$$
     $$\implies B_k \log_2(1 + \bar{h}_n p_n) \le B_k \frac{\bar{h}_n p_n}{\ln 2} = \frac{B_k \bar{h}_n}{\ln 2} p_n$$
   * Substituting this upper bound of capacity into the denominator:
     $$\frac{p_n D_n^{\tt raw}}{\xi_n B_k \log_2(1 + \bar{h}_n p_n)} \ge \frac{p_n D_n^{\tt raw}}{\xi_n \left( \frac{B_k \bar{h}_n}{\ln 2} p_n \right)} = \mathbf{\frac{D_n^{\tt raw} \ln 2}{\xi_n B_k \bar{h}_n}}$$
   * Notice that transmit power $p_n$ cancels out completely!
   * **Physical Meaning:** $\frac{D_n^{\tt raw} \ln 2}{\xi_n B_k \bar{h}_n}$ is the **theoretical minimum thermodynamic energy** required to transmit $D_n^{\tt raw}$ bits over channel $\bar{h}_n$.

---

#### Concept C: Bounding the Marginal Price $\bar{\Omega}_n(\rho_n) \le \Omega_n(\rho_n)$

With these 3 replacements:
$$\bar{\Delta}_n(\rho_n) = \max\left\{ (1-\rho_n)\frac{C_n^{\tt raw}}{F_n^{\tt loc}}, \; \rho_n\left(\frac{D_n^{\tt raw}}{\bar{R}_n} + \frac{C_n^{\tt raw}}{F_m^{\tt max}}\right) \right\} \le \Delta_n$$
$$\bar{\Theta}_n(\rho_n) = (1-\rho_n)\kappa_n C_n^{\tt raw}(F_n^{\tt loc})^2 + \rho_n \frac{D_n^{\tt raw} \ln 2}{\xi_n B_k \bar{h}_n} \le \Theta_n$$

Thus, the marginal delay-energy price satisfies:
$$\mathbf{\bar{\Omega}_n(\rho_n) \triangleq \frac{\beta_n^{\tt t}\bar{\Delta}_n(\rho_n)}{\mathsf{T}_n^{\tt ref}} + \frac{\beta_n^{\tt e}\bar{\Theta}_n(\rho_n)}{\mathsf{E}_n^{\tt ref}} \le \Omega_n(\rho_n) \quad (\forall \vec{p}, \vec{F})}$$

---

### 3. Closed-Form Evaluation of $\bar{u}_n$ in $\mathcal{O}(1)$ Microsecond Time

Recall from Proposition 3 that the uplink utility is:
$$\mathbb{U}_n^{\tt ul} = \text{const} - \chi_n \Omega_n(\rho_n) + \beta_n^{\tt a}\frac{1 - e^{-\vartheta_n \chi_n \Gamma_n} - \Lambda^{\tt th}}{1 - \Lambda^{\tt th}}$$

Because $\Omega_n$ appears with a negative sign ($-\chi_n \Omega_n$), replacing $\Omega_n$ by its lower bound $\bar{\Omega}_n$ produces an **upper bound on utility**:
$$\mathbb{U}_n^{\tt ul}(\vec{p}, \vec{F}, \rho_n, \chi_n) \le \mathbb{U}_n^{\tt ul}(\bar{\Omega}_n) \le \bar{u}_n$$

To compute $\bar{u}_n$:
1. Evaluate the piecewise-linear function $\bar{\Omega}_n(\rho_n)$ at the 3 candidate split points $\rho_n \in \{0, \bar{\rho}_n^\star, 1\}$ (by Proposition 2).
2. Take the minimum $\bar{\Omega}_n^\star$.
3. Plug $\bar{\Omega}_n^\star$ into the closed-form formula of Proposition 3 to obtain $\bar{\chi}_n^\star$ and the maximum bound $\bar{u}_n$.
4. **Time Complexity:** Computed in **$\mathcal{O}(1)$ basic arithmetic operations** without running any iterative optimization!

---

### 4. How the Filter $\mathbb{F}(\vec{A})$ Works in the Pruning Algorithm (Lemma 2)

$$\mathbb{F}(\vec{A}) = \sum_{n} \bar{u}_n(\vec{A}^{\tt ul}) + \mathbb{U}_{\mathrm{ub}}^{\tt dl}(\vec{A}^{\tt dl})$$

* **Incumbent Champion $\mathbb{U}^\star$:** The best real utility found so far.
* **Pruning Decision:**
  $$\text{If } \mathbf{\mathbb{F}(\vec{A}) \le \mathbb{U}^\star} \implies \mathbf{\text{Discard } \vec{A} \text{ immediately!}}$$
* **Guarantee:** If the ideal upper bound cannot even match the current champion $\mathbb{U}^\star$, the candidate $\vec{A}$ can never produce a better real solution in harsh interference conditions.
* **Algorithmic Benefit:** Prunes **60% to 80%** of discrete association combinations instantly, reducing overall solver execution time by several folds.
