[← Return to Main Index](../index.md)

# 8.1 Parameter Space

## Introduction

The Noisy Storage Model (NSM) operates within a **multidimensional parameter space** that simultaneously captures quantum channel imperfections, detection inefficiencies, and adversarial storage capabilities. Understanding this parameter space is critical for:

1. **Security Analysis**: Determining which parameter regimes enable information-theoretic security
2. **Experimental Design**: Translating abstract security requirements into physical hardware specifications
3. **Protocol Optimization**: Balancing extraction efficiency against security margins

This section provides a comprehensive taxonomy of NSM parameters, their physical interpretations, and the constraints that govern their admissible ranges.

## Parameter Taxonomy

### Core NSM Parameters

The fundamental NSM security assumption rests on four key parameters:

| Parameter | Symbol | Domain | Physical Meaning |
|-----------|--------|--------|------------------|
| **Storage Noise** | $r$ | $[0, 1]$ | Adversary's qubit preservation probability during $\Delta t$ |
| **Storage Rate** | $\nu$ | $[0, 1]$ | Fraction of qubits adversary can store |
| **Wait Time** | $\Delta t$ | $(0, \infty)$ | Time between measurement and basis revelation (ns) |
| **Storage Dimension** | $d$ | $\{2, 3, \ldots\}$ | Hilbert space dimension (qubits: $d=2$) |

**Storage Noise $r$**: Models a depolarizing channel $\mathcal{N}(\rho) = r\rho + (1-r)\frac{\mathbb{I}}{2}$ acting on the adversary's quantum memory. At $r=1$, storage is perfect (no security); at $r=0$, complete depolarization (maximal security).

**Storage Rate $\nu$**: Quantifies the adversary's **quantum memory capacity**. For $n$ transmitted qubits, the adversary can store at most $\lfloor \nu n \rfloor$ coherently. The remainder must be measured immediately or discarded.

**Wait Time $\Delta t$**: Physical enforcement mechanism. Honest parties delay basis revelation by $\Delta t$, ensuring the adversary's storage experiences decoherence. Typical values: $\Delta t \sim 10^6$ ns ($1$ ms) to $10^9$ ns ($1$ s).

**Storage Dimension $d$**: For qubits, $d=2$ is fixed. Generalized NSM protocols may use qutrits ($d=3$) or higher dimensions [1].

### Channel Parameters

Physical imperfections in quantum communication:

| Parameter | Symbol | Domain | Physical Meaning |
|-----------|--------|--------|------------------|
| **Channel Fidelity** | $F$ | $(0.5, 1]$ | Bell state fidelity of generated EPR pairs |
| **Detection Efficiency** | $\eta$ | $(0, 1]$ | Probability a photon triggers detector |
| **Detector Error** | $e_{\text{det}}$ | $[0, 1]$ | Intrinsic measurement error rate |
| **Dark Count Prob.** | $P_{\text{dark}}$ | $[0, 1]$ | Spontaneous detector firing probability |

**Channel Fidelity $F$**: For a maximally entangled state $|\Phi^+\rangle = (|00\rangle + |11\rangle)/\sqrt{2}$, the prepared state $\rho$ satisfies $F = \langle \Phi^+ | \rho | \Phi^+ \rangle$. Depolarizing noise: $F = 1 - \frac{3}{4}p_{\text{depol}}$.

**Detection Efficiency $\eta$**: Combines:
- **Fiber transmission**: $\eta_{\text{fiber}} = 10^{-\alpha L / 10}$ (where $\alpha \approx 0.2$ dB/km at 1550 nm)
- **Detector quantum efficiency**: $\eta_{\text{QE}} \sim 0.1$–$0.7$ (silicon APDs at 800 nm)
- **Coupling losses**: $\eta_{\text{coupling}} \sim 0.5$–$0.9$

Total: $\eta = \eta_{\text{fiber}} \times \eta_{\text{QE}} \times \eta_{\text{coupling}}$.

**Detector Error $e_{\text{det}}$**: Polarization misalignment, imperfect beamsplitters, or waveguide crosstalk contribute to basis-independent errors. Typical: $e_{\text{det}} \sim 0.01$–$0.05$.

**Dark Counts**: Thermal noise or detector afterpulsing. Characterized by rate $R_{\text{dark}}$ (counts/s) and detection window $\tau_{\text{window}}$: $P_{\text{dark}} = R_{\text{dark}} \tau_{\text{window}}$. Silicon APDs: $R_{\text{dark}} \sim 10^2$–$10^4$ Hz.

### Derived Quantities

These are computed from the core parameters:

| Quantity | Formula | Significance |
|----------|---------|--------------|
| **Channel QBER** | $Q_{\text{ch}} = \frac{1-F}{2} + e_{\text{det}} + \frac{(1-\eta)P_{\text{dark}}}{2}$ | Measured error rate (Phase II) |
| **Storage QBER** | $Q_{\text{storage}} = \frac{1-r}{2}$ | Security threshold |
| **Storage Capacity** | $C_\mathcal{N} = 1 - H_2(r)$ | Classical capacity of storage channel |
| **Min-Entropy Rate** | $h_{\min}(r)$ | Extractable entropy per bit (Phase IV) |

**Security Condition**: The fundamental NSM requirement is [2]:

$$
Q_{\text{ch}} < Q_{\text{storage}} \quad \text{and} \quad C_\mathcal{N} \cdot \nu < \frac{1}{2}
$$

**Physical Interpretation**: The channel must be **noisier** than the adversary's storage, and the adversary's storage capacity must be **bounded** below $1/2$ bit per qubit.

## Parameter Constraints

### Physical Feasibility

**Hard Constraints** (enforced by physics):

1. **Fidelity Lower Bound**: $F > 0.5$ (distinguishability from maximally mixed state)
2. **Efficiency Upper Bound**: $\eta \leq 1$ (cannot exceed 100% detection)
3. **Probability Constraints**: $0 \leq e_{\text{det}}, P_{\text{dark}}, r, \nu \leq 1$
4. **Time Positivity**: $\Delta t > 0$

### Security Constraints

**NSM Security Requirements**:

1. **Strictly Less Condition** [3]:
   $$
   Q_{\text{ch}} < Q_{\text{storage}} = \frac{1 - r}{2}
   $$

2. **Storage Capacity Bound** [2]:
   $$
   C_\mathcal{N} \cdot \nu < \frac{1}{2}
   $$
   where $C_\mathcal{N} = 1 - H_2(r)$ and $H_2(x) = -x \log_2 x - (1-x) \log_2 (1-x)$.

3. **Conservative QBER Threshold** [3]:
   $$
   Q_{\text{ch}} < 0.11
   $$
   (11% threshold from Schaffner's analysis of depolarizing attacks)

4. **Hard QBER Limit** [2]:
   $$
   Q_{\text{ch}} < 0.22
   $$
   (22% limit beyond which no NSM protocol can be secure)

**Example**: For $r = 0.75$:
- $Q_{\text{storage}} = 0.125$
- $C_\mathcal{N} = 1 - H_2(0.75) \approx 0.189$
- Storage capacity constraint: $\nu < 0.5 / 0.189 \approx 2.65$ (always satisfied for $\nu \leq 1$)

### Operational Constraints

**Protocol Viability**:

1. **Min-Entropy Positivity**:
   $$
   n \cdot h_{\min}(r) > |\Sigma| + \Delta_{\text{sec}}
   $$
   Avoids "Death Valley" where no key can be extracted.

2. **QBER Reconciliation Bound**:
   $$
   Q_{\text{ch}} < R_{\text{code}} / 2
   $$
   For error correction to succeed, QBER must be below the code's error-correcting capacity.

3. **Statistical Significance**:
   $$
   n_{\text{test}} \geq \frac{\log(2/\varepsilon_{\text{est}})}{2\delta^2}
   $$
   Sample size for QBER estimation within $\pm \delta$ with confidence $1 - \varepsilon_{\text{est}}$.

## Experimental Parameter Regimes

### Erven et al. (2014) Configuration [4]

The canonical NSM experiment:

| Parameter | Value | Justification |
|-----------|-------|---------------|
| $r$ | $0.75$ | Achievable with $\Delta t = 1$ ms, room-temperature storage |
| $\nu$ | $0.002$ | Memory capacity $\sim 2$ qubits per 1000 transmitted |
| $\Delta t$ | $10^6$ ns ($1$ ms) | Photon time-of-flight + processing delay |
| $F$ | $0.975$ | PDC source with spatial mode filtering |
| $\eta$ | $0.06$ | 100 km fiber + silicon APD ($\eta_{\text{QE}} = 0.6$) |
| $e_{\text{det}}$ | $0.015$ | Polarization analyzer imperfections |
| $P_{\text{dark}}$ | $5 \times 10^{-6}$ | $R_{\text{dark}} = 500$ Hz, $\tau = 10$ ns |

**Resulting QBER**: $Q_{\text{ch}} \approx 0.029$ (well below $Q_{\text{storage}} = 0.125$).

### Simulation-Optimized Regime

For SquidASM/NetSquid simulations (Caligo default):

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| $r$ | $0.75$ | Standard NSM assumption |
| $\nu$ | $0.002$ | Minimal storage threat |
| $\Delta t$ | $10^6$ ns | Computationally tractable |
| $F$ | $0.95$ | Moderate depolarizing noise |
| $\eta$ | $1.0$ | Perfect detection (isolate fidelity effects) |
| $e_{\text{det}}$ | $0.0$ | No detector errors |
| $P_{\text{dark}}$ | $0.0$ | No dark counts |

**Simplified QBER**: $Q_{\text{ch}} = \frac{1-F}{2} = 0.025$.

### High-Noise Regime (Stress Testing)

Exploring protocol robustness:

| Parameter | Value | Stress Factor |
|-----------|-------|---------------|
| $r$ | $0.80$ | Higher storage fidelity (adversary advantage) |
| $\Delta t$ | $10^5$ ns ($100$ μs) | Reduced wait time |
| $F$ | $0.85$ | Significant depolarization |
| $\eta$ | $0.10$ | Long-distance fiber (loss-dominated) |
| $P_{\text{dark}}$ | $10^{-4}$ | High dark count rate |

**Resulting QBER**: $Q_{\text{ch}} \approx 0.095$ (near conservative threshold).

## Parameter Selection Methodology

### Step 1: Define Security Target

Choose security parameter $\varepsilon_{\text{sec}}$ (e.g., $10^{-10}$) and target key length $\ell_{\text{target}}$ (e.g., 256 bits).

### Step 2: Fix NSM Core Parameters

Select $r$ and $\nu$ based on adversary model:

- **Conservative**: $r = 0.75$, $\nu = 0.002$ (Erven regime)
- **Optimistic**: $r = 0.50$, $\nu = 0.01$ (higher entropy)
- **Pessimistic**: $r = 0.90$, $\nu = 0.1$ (stronger adversary)

### Step 3: Determine Channel Parameters

**If using physical hardware**: Measure $F$, $\eta$, $e_{\text{det}}$, $P_{\text{dark}}$ via quantum state tomography and detector characterization.

**If simulating**: Choose parameters to match target QBER:

$$
Q_{\text{target}} = 0.05 \implies F \approx 0.90 \text{ (for $\eta=1$, $e_{\text{det}}=0$)}
$$

### Step 4: Compute Minimum $n$

Use the minimum input length formula:

$$
n \geq \frac{\ell_{\text{target}} + \Delta_{\text{sec}}}{h_{\min}(r) - f_{\text{leak}}}
$$

where $f_{\text{leak}} \approx H_2(Q_{\text{ch}})$ (Shannon entropy of binary channel).

**Example**: For $\ell = 256$, $r = 0.75$ ($h_{\min} = 0.25$), $Q_{\text{ch}} = 0.05$ ($f_{\text{leak}} \approx 0.29$):

$$
n \geq \frac{256 + 64}{0.25 - 0.29} = \text{undefined (Death Valley!)}
$$

**Adjustment**: Reduce $Q_{\text{ch}}$ to 0.03 ($f_{\text{leak}} \approx 0.20$):

$$
n \geq \frac{320}{0.05} = 6400 \text{ bits}
$$

### Step 5: Validate Security Conditions

Check:

1. $Q_{\text{ch}} < \frac{1-r}{2}$ ✓
2. $C_\mathcal{N} \cdot \nu < 0.5$ ✓
3. $Q_{\text{ch}} < 0.11$ ✓
4. $n \cdot h_{\min}(r) > |\Sigma| + \Delta_{\text{sec}}$ ✓

If any fails, iterate parameter selection.

## Parameter Sensitivity Analysis

### Fidelity Sensitivity

**QBER vs. Fidelity** (for $\eta = 1$, $e_{\text{det}} = 0$):

$$
\frac{\partial Q_{\text{ch}}}{\partial F} = -\frac{1}{2}
$$

A 1% decrease in fidelity increases QBER by 0.5%.

**Impact on Extractable Length**: For $n = 1000$, $r = 0.75$:

| $F$ | $Q_{\text{ch}}$ | $\ell$ (bits) | Efficiency |
|-----|-----------------|---------------|------------|
| 0.98 | 0.010 | 82 | 8.2% |
| 0.95 | 0.025 | 71 | 7.1% |
| 0.90 | 0.050 | 54 | 5.4% |
| 0.85 | 0.075 | 38 | 3.8% |

### Storage Noise Sensitivity

**Min-Entropy vs. $r$**:

- **Dupuis-König regime** ($r < 0.25$): $h_{\min} \propto -\log_2(1 + 3r^2)$ (slowly varying)
- **Lupo regime** ($r \geq 0.25$): $h_{\min} = 1 - r$ (linear)

| $r$ | $h_{\min}$ | Δ$h_{\min}$ per 0.05 Δ$r$ |
|-----|-----------|-------------------------|
| 0.50 | 0.50 | -0.05 |
| 0.75 | 0.25 | -0.05 |
| 0.90 | 0.10 | -0.05 |

**Implication**: In Lupo regime, entropy degrades linearly with storage noise.

## Caligo Implementation

### NSMParameters Dataclass

```python
@dataclass(frozen=True)
class NSMParameters:
    storage_noise_r: float      # ∈ [0, 1]
    storage_rate_nu: float      # ∈ [0, 1]
    delta_t_ns: float           # > 0
    channel_fidelity: float     # ∈ (0.5, 1]
    detection_eff_eta: float = 1.0
    detector_error: float = 0.0
    dark_count_prob: float = 0.0
    storage_dimension_d: int = 2
    
    def __post_init__(self):
        # Validate all constraints (INV-NSM-001 through INV-NSM-006)
        validate_nsm_invariants(self)
```

### Parameter Factory Methods

```python
# Erven et al. configuration
params_erven = NSMParameters.from_erven_2014()

# Simulation-optimized
params_sim = NSMParameters.for_simulation(
    storage_noise_r=0.75,
    channel_fidelity=0.95
)

# Custom
params_custom = NSMParameters(
    storage_noise_r=0.70,
    storage_rate_nu=0.005,
    delta_t_ns=5e5,
    channel_fidelity=0.92,
    detection_eff_eta=0.15
)
```

## References

[1] Damgård, I., Fehr, S., Renner, R., Salvail, L., & Schaffner, C. (2007). A tight high-order entropic quantum uncertainty relation with applications. In *Advances in Cryptology—CRYPTO 2007* (pp. 360-378). Springer.

[2] König, R., Wehner, S., & Wullschleger, J. (2012). Unconditional security from noisy quantum storage. *IEEE Transactions on Information Theory*, 58(3), 1962-1984.\

[3] Schaffner, C., Terhal, B. M., & Wehner, S. (2009). Robust cryptography in the noisy-quantum-storage model. *Quantum Information & Computation*, 9(11-12), 963-996.

[4] Erven, C., et al. (2014). An experimental implementation of oblivious transfer in the noisy storage model. *Nature Communications*, 5, 3418.

---

[← Return to Main Index](../index.md) | [Next: NSM-to-Physical Mapping](./physical_mapping.md)
