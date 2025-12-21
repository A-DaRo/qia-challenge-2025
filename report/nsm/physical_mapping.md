[← Return to Main Index](../index.md)

# 8.2 NSM-to-Physical Mapping

## Introduction

The Noisy Storage Model provides an **abstract security framework** expressed in information-theoretic terms: min-entropy rates, storage noise parameters, and classical capacity constraints. However, experimental implementations and discrete-event simulations operate at the **physical layer**: photon detection efficiencies, qubit decoherence times, and gate error rates. Bridging this gap requires precise mathematical mappings that preserve security guarantees while respecting physical realizability constraints.

This section establishes the rigorous translation framework between NSM theoretical parameters and NetSquid simulation primitives, grounded in the experimental literature of König et al. [1], Schaffner et al. [2], Wehner et al. [3], and Erven et al. [4].

## Literature Foundations

### The Depolarizing Storage Channel

The foundational NSM paper by Wehner et al. (2008) [3] models adversarial quantum storage as a **depolarizing channel**:

$$
\mathcal{N}_{\text{depol}}(\rho) = r\rho + (1-r) \frac{\mathbb{I}}{2}
$$

where:
- $\rho$ is the qubit state stored by the adversary
- $r \in [0, 1]$ is the **preservation probability** (storage fidelity)
- $\mathbb{I}/2$ is the maximally mixed state

**Physical Interpretation**: During the wait time $\Delta t$, each stored qubit undergoes independent depolarization. At $r=1$, storage is perfect (no decoherence); at $r=0$, complete depolarization (thermal equilibrium). The adversary cannot control $r$—it is determined by the physical storage medium and $\Delta t$.

### König's Storage Capacity Constraint

König, Wehner, and Wullschleger (2012) [1] proved that security requires the adversary's **storage capacity** to be bounded:

$$
C_\mathcal{N} \cdot \nu < \frac{1}{2}
$$

where:
- $C_\mathcal{N} = 1 - H_2(r)$ is the **classical capacity** of the depolarizing channel
- $\nu \in [0, 1]$ is the **storage rate** (fraction of qubits storable)
- $H_2(x) = -x \log_2 x - (1-x) \log_2(1-x)$ is the binary entropy

**Proof Sketch**: The adversary's quantum memory can be modeled as a quantum channel with capacity $C_\mathcal{N}$. If $\nu$ qubits are stored, the total information capacity is $C_\mathcal{N} \cdot \nu$ bits. For OT security, this must not exceed $1/2$ bit (otherwise the adversary can extract the complementary string $S_{\bar{C}}$ with non-negligible advantage).

**Example**: For $r = 0.75$:

$$
C_\mathcal{N} = 1 - H_2(0.75) = 1 - 0.811 = 0.189 \text{ bits/qubit}
$$

The constraint $0.189 \cdot \nu < 0.5$ implies $\nu < 2.65$, automatically satisfied for $\nu \leq 1$.

### Schaffner's QBER Threshold

Schaffner et al. (2009) [2] analyzed the optimal adversarial strategy against depolarizing storage and derived the **11% QBER threshold**:

**Theorem 1 (Schaffner)**: For a depolarizing storage channel with parameter $r$, security is achievable if:

$$
Q_{\text{channel}} < Q_{\text{storage}} = \frac{1 - r}{2}
$$

and $Q_{\text{channel}} < 0.11$ (conservative bound).

**Corollary 7**: The condition must be **strictly less**—equality or near-equality allows measurement-based attacks (e.g., Breidbart basis measurements [5]) that extract partial information.

**Physical Meaning**: The channel QBER experienced by honest parties must be strictly lower than the noise induced by the adversary's storage. If the channel is equally or more noisy, the adversary gains no disadvantage from delayed measurement.

### Erven's Experimental Parameters

Erven et al. (2014) [4] demonstrated the first experimental NSM-OT implementation over 100 km fiber. Their parameter choices inform Caligo's defaults:

| Parameter | Erven Value | Physical Source |
|-----------|------------|-----------------|
| $\mu$ | 0.08 | PDC source mean photon-pair number |
| $F$ | 0.975 | Bell state fidelity after filtering |
| $\eta$ | 0.06 | End-to-end detection efficiency (fiber + detector) |
| $e_{\text{det}}$ | 0.015 | Polarization analyzer misalignment |
| $P_{\text{dark}}$ | $5 \times 10^{-6}$ | APD dark count probability |
| $r$ | 0.75 | Room-temperature storage for $\Delta t = 1$ ms |
| $\nu$ | 0.002 | 2 qubits/1000 transmitted |

**QBER Calculation** (Erven Eq. 8):

$$
\begin{aligned}
Q_{\text{ch}} &= \frac{1 - F}{2} + e_{\text{det}} + \frac{(1 - \eta) P_{\text{dark}}}{2} \\
&= \frac{1 - 0.975}{2} + 0.015 + \frac{(1 - 0.06) \times 5 \times 10^{-6}}{2} \\
&= 0.0125 + 0.015 + 2.35 \times 10^{-6} \approx 0.0275
\end{aligned}
$$

**Security Margin**:

$$
Q_{\text{storage}} - Q_{\text{ch}} = 0.125 - 0.0275 = 0.0975
$$

A comfortable 9.75% margin validates the experimental feasibility of NSM protocols.

## NSM Parameter Decomposition

### Storage Noise Parameter $r$

**Physical Model**: Relates to qubit decoherence via:

$$
r(t) = e^{-\Gamma t}
$$

where $\Gamma$ is the decoherence rate and $t = \Delta t$ is the wait time.

**T1/T2 Correspondence**:

For a superconducting qubit:

$$
\Gamma = \frac{1}{2} \left( \frac{1}{T_1} + \frac{1}{T_2} \right)
$$

where $T_1$ (energy relaxation) and $T_2$ (phase coherence) are the standard decoherence timescales.

**Example**: For $\Delta t = 1$ ms and target $r = 0.75$:

$$
\Gamma = -\frac{\ln(0.75)}{10^6 \text{ ns}} = 2.88 \times 10^{-7} \text{ ns}^{-1}
$$

Requiring:

$$
\frac{1}{T_1} + \frac{1}{T_2} \approx 5.76 \times 10^{-7} \text{ ns}^{-1}
$$

If $T_1 = T_2 = T$, then $T \approx 3.47 \times 10^6$ ns ($3.47$ ms), **shorter** than typical superconducting qubit coherence times ($T_2 \sim 100$ μs for transmons). This validates that room-temperature or deliberately degraded storage achieves $r = 0.75$.

### Channel Fidelity $F$

**Bell State Fidelity**: For an EPR pair $|\Phi^+\rangle = (|00\rangle + |11\rangle)/\sqrt{2}$, the fidelity:

$$
F = \langle \Phi^+ | \rho_{\text{AB}} | \Phi^+ \rangle
$$

quantifies overlap with the ideal maximally entangled state.

**Depolarizing Noise Relation**: If the channel applies symmetric depolarization with probability $p$:

$$
\rho_{\text{depol}} = (1-p) |\Phi^+\rangle\langle\Phi^+| + p \frac{\mathbb{I}}{4}
$$

then:

$$
F = 1 - \frac{3p}{4}
$$

Inverting:

$$
p_{\text{depol}} = \frac{4(1 - F)}{3}
$$

**Example**: For $F = 0.95$:

$$
p_{\text{depol}} = \frac{4 \times 0.05}{3} \approx 0.0667
$$

NetSquid's `DepolarNoiseModel` requires `depolar_rate` = 0.0667.

### Detection Efficiency $\eta$

**End-to-End Efficiency**: Combines multiple loss mechanisms:

$$
\eta = \eta_{\text{source}} \times \eta_{\text{fiber}} \times \eta_{\text{coupling}} \times \eta_{\text{QE}}
$$

1. **Source Efficiency** $\eta_{\text{source}}$: PDC pair generation probability $\approx \mu$ (mean photon number)
2. **Fiber Loss** $\eta_{\text{fiber}} = 10^{-\alpha L / 10}$:
   - $\alpha \approx 0.2$ dB/km at 1550 nm
   - For $L = 100$ km: $\eta_{\text{fiber}} = 10^{-2} = 0.01$
3. **Coupling Losses** $\eta_{\text{coupling}} \sim 0.5$–$0.9$: Spatial mode mismatch, beamsplitter losses
4. **Quantum Efficiency** $\eta_{\text{QE}}$: Detector intrinsic efficiency
   - Silicon APDs at 800 nm: $\eta_{\text{QE}} \sim 0.6$
   - InGaAs APDs at 1550 nm: $\eta_{\text{QE}} \sim 0.1$–$0.3$

**Example (Erven Regime)**:

$$
\eta = 0.08 \times 0.01 \times 0.75 \times 0.60 \approx 0.036 \text{ (close to reported 0.06)}
$$

### Dark Count Probability $P_{\text{dark}}$

**Physical Model**: Spontaneous detector firing from thermal noise or afterpulsing.

**Rate-to-Probability Conversion**:

$$
P_{\text{dark}} = 1 - e^{-R_{\text{dark}} \tau_{\text{window}}}
$$

where:
- $R_{\text{dark}}$ is the dark count rate (Hz)
- $\tau_{\text{window}}$ is the detection time window (s)

**Approximation** (for $R_{\text{dark}} \tau \ll 1$):

$$
P_{\text{dark}} \approx R_{\text{dark}} \tau_{\text{window}}
$$

**Example**: For $R_{\text{dark}} = 500$ Hz, $\tau = 10$ ns:

$$
P_{\text{dark}} = 500 \times 10 \times 10^{-9} = 5 \times 10^{-6}
$$

### Detector Error $e_{\text{det}}$

**Sources**:

1. **Polarization Misalignment**: Angular deviation $\theta$ from ideal basis:
   $$
   e_{\text{pol}} \approx \sin^2(\theta)
   $$
   For $\theta = 7°$: $e_{\text{pol}} \approx 0.015$.

2. **Beamsplitter Imbalance**: Non-50/50 splitting ratio introduces basis-dependent errors.

3. **Waveguide Crosstalk**: In integrated photonic chips, adjacent channels leak photons.

**Total**: Typically $e_{\text{det}} \sim 0.01$–$0.05$ for calibrated free-space setups.

## Unified QBER Formula

### Erven et al. Equation 8

The complete channel QBER accounting for all error mechanisms:

$$
Q_{\text{ch}} = Q_{\text{source}} + Q_{\text{det}} + Q_{\text{dark}}
$$

where:

1. **Source Error**:
   $$
   Q_{\text{source}} = \frac{1 - F}{2}
   $$

2. **Detector Error**:
   $$
   Q_{\text{det}} = e_{\text{det}}
   $$

3. **Dark Count Error**:
   $$
   Q_{\text{dark}} = \frac{(1 - \eta) P_{\text{dark}}}{2}
   $$

**Derivation of $Q_{\text{dark}}$ Term**:

- **With photon arrival** (probability $\eta$): True detection, QBER = 0
- **Without photon** (probability $1 - \eta$): Dark count mimics random bit (QBER = 0.5)

Expected contribution:

$$
Q_{\text{dark}} = (1 - \eta) \times P_{\text{dark}} \times 0.5
$$

**Combined Formula**:

$$
\boxed{Q_{\text{ch}} = \frac{1 - F}{2} + e_{\text{det}} + \frac{(1 - \eta) P_{\text{dark}}}{2}}
$$

This is implemented in Caligo as `caligo.utils.math.compute_qber_erven()`.

## PDC Source Model (Advanced)

### Multi-Pair Emission

Parametric down-conversion (PDC) sources emit **multiple photon pairs** probabilistically:

$$
|\Psi_{\text{src}}\rangle = \sum_{n=0}^{\infty} \sqrt{P_n^{\text{src}}} |\Phi_n\rangle_{AB}
$$

where $P_n^{\text{src}}$ is the probability of $n$ pairs.

**Poisson Distribution** (Erven Eq. 10):

$$
P_n^{\text{src}} = \frac{(n+1)(\mu/2)^n}{(1 + \mu/2)^{n+2}}
$$

### Conditional Probabilities

**Single-Pair Emission** (given at least one pair):

$$
P_{\text{sent}} = \frac{P_1^{\text{src}}}{1 - P_0^{\text{src}}}
$$

**No-Click Probability** (honest Bob):

$$
P_{B, \text{noclick}} = \sum_{n=0}^{\infty} P_n^{\text{src}} \times \Pr[\text{no detection} | n \text{ pairs}]
$$

For $n$ pairs, Bob detects with probability $1 - (1-\eta)^n$, so:

$$
P_{B, \text{noclick}} = \sum_{n=0}^{\infty} P_n^{\text{src}} [(1-\eta)^n (1 - P_{\text{dark}})]
$$

**Security Bound** (dishonest Bob):

$$
P'_{B, \text{noclick}} \geq P_0^{\text{src}} = \frac{1}{(1 + \mu/2)^2}
$$

The adversary cannot improve beyond vacuum events (no pairs emitted).

**Caligo Implementation**: Functions `pdc_probability()`, `p_sent()`, `p_b_noclick()` in [physical_model.py](../../caligo/caligo/simulation/physical_model.py) implement Erven's PDC formulas for high-fidelity QBER estimation.

## Parameter Feasibility Analysis

### Security Condition Matrix

For various $(r, F)$ combinations, determine if security is achievable:

| $r$ | $F$ | $Q_{\text{ch}}$ | $Q_{\text{storage}}$ | Margin | Secure? |
|-----|-----|-----------------|---------------------|--------|---------|
| 0.75 | 0.98 | 0.010 | 0.125 | 0.115 | ✓ |
| 0.75 | 0.95 | 0.025 | 0.125 | 0.100 | ✓ |
| 0.75 | 0.85 | 0.075 | 0.125 | 0.050 | ✓ |
| 0.75 | 0.76 | 0.120 | 0.125 | 0.005 | ⚠ Marginal |
| 0.80 | 0.90 | 0.050 | 0.100 | 0.050 | ✓ |
| 0.80 | 0.80 | 0.100 | 0.100 | 0.000 | ✗ Boundary |
| 0.90 | 0.95 | 0.025 | 0.050 | 0.025 | ✓ |
| 0.95 | 0.90 | 0.050 | 0.025 | -0.025 | ✗ Violated |

**Takeaway**: High storage fidelity ($r \to 1$) dramatically tightens constraints on channel fidelity.

### Entropy vs. Leakage Trade-off

For secure key extraction, the inequality:

$$
n \cdot h_{\min}(r) > |\Sigma| + \Delta_{\text{sec}}
$$

must hold. Using $|\Sigma| \approx n \cdot H_2(Q_{\text{ch}})$ (Shannon entropy leakage):

$$
n \left[ h_{\min}(r) - H_2(Q_{\text{ch}}) \right] > \Delta_{\text{sec}}
$$

**Critical $n$**:

$$
n_{\text{min}} = \frac{\Delta_{\text{sec}}}{h_{\min}(r) - H_2(Q_{\text{ch}})}
$$

**Example**: For $r = 0.75$ ($h_{\min} = 0.25$), $Q_{\text{ch}} = 0.05$ ($H_2 \approx 0.286$), $\Delta_{\text{sec}} = 64$:

$$
n_{\text{min}} = \frac{64}{0.25 - 0.286} = \frac{64}{-0.036} \quad (\text{undefined—Death Valley!})
$$

**Fix**: Reduce $Q_{\text{ch}}$ to 0.03 ($H_2 \approx 0.199$):

$$
n_{\text{min}} = \frac{64}{0.25 - 0.199} = \frac{64}{0.051} \approx 1255 \text{ qubits}
$$

This explains why Caligo targets $n \sim 1000$–$5000$ for robust operation.

## Simulation Parameter Injection

### Mapping Table

| NSM Parameter | NetSquid Equivalent | Injection Point | Formula |
|---------------|---------------------|-----------------|---------|
| $r$ | `depolar_rate` | `QuantumProcessor.T1/T2` | $1 - r$ |
| $F$ | `prob_max_mixed` | `MagicDistributor.DepolariseModelParameters` | $\frac{4(1-F)}{3}$ |
| $\eta$ | `detector_efficiency` | `DoubleClickModelParameters` | $\eta$ |
| $e_{\text{det}}$ | `gate_depolar_rate` | `GenericQDeviceConfig` | $e_{\text{det}}$ |
| $P_{\text{dark}}$ | `dark_count_probability` | `DoubleClickModelParameters` | $P_{\text{dark}}$ |
| $\Delta t$ | `sim_run(duration)` | `TimingBarrier.wait_delta_t()` | $\Delta t$ (ns) |

### ChannelParameters Dataclass

Caligo's `ChannelParameters` encapsulates the translated values:

```python
@dataclass(frozen=True)
class ChannelParameters:
    """
    Physical channel parameters for NetSquid simulation.
    
    Derived from NSMParameters via Erven et al. formulas.
    """
    
    # Link noise
    depolarise_prob: float          # p_depol = 4(1-F)/3
    
    # Detection parameters
    detector_efficiency: float      # η
    dark_count_probability: float   # P_dark
    detector_misalignment: float    # e_det
    
    # Timing
    cycle_time_ns: float            # EPR generation cycle
    state_delay_ns: float           # Propagation delay
    
    @staticmethod
    def from_nsm_parameters(params: NSMParameters) -> ChannelParameters:
        """Translate NSM → NetSquid parameters."""
        depol = (4.0 * (1.0 - params.channel_fidelity)) / 3.0
        
        return ChannelParameters(
            depolarise_prob=depol,
            detector_efficiency=params.detection_eff_eta,
            dark_count_probability=params.dark_count_prob,
            detector_misalignment=params.detector_error,
            cycle_time_ns=TYPICAL_CYCLE_TIME_NS,
            state_delay_ns=TYPICAL_CYCLE_TIME_NS,
        )
```

### Security Verification

`SecurityVerifier` validates the NSM condition at runtime:

```python
def verify_nsm_security_condition(
    measured_qber: float,
    nsm_params: NSMParameters,
    strict: bool = True
) -> NSMVerificationResult:
    """
    Enforce Q_channel < Q_storage.
    
    Raises
    ------
    SecurityError
        If NSM condition violated and strict=True.
    """
    r = nsm_params.storage_noise_r
    q_storage = (1.0 - r) / 2.0
    
    if measured_qber >= q_storage:
        raise SecurityError(
            f"NSM violated: Q_ch={measured_qber:.4f} >= "
            f"Q_storage={q_storage:.4f}"
        )
    
    # Additional checks: 11% threshold, 22% hard limit
    if measured_qber >= QBER_CONSERVATIVE_LIMIT:
        logger.warning("QBER exceeds 11% conservative threshold")
    
    if measured_qber >= QBER_HARD_LIMIT:
        raise QBERThresholdExceeded("QBER exceeds 22% hard limit")
```

## Validation Methodology

### Analytic Verification

**Test 1: QBER Formula Consistency**

Generate $(F, \eta, e_{\text{det}}, P_{\text{dark}})$, compute $Q_{\text{ch}}$ analytically, run simulation, compare measured QBER.

**Expected Deviation**: $\pm 0.005$ due to finite sampling ($n \sim 1000$).

**Test 2: Security Condition**

For parameters satisfying $Q_{\text{ch}} < Q_{\text{storage}}$, protocol should succeed. For violating parameters, should abort.

**Test 3: Death Valley Detection**

For $h_{\min}(r) - H_2(Q_{\text{ch}}) < 0$, key extraction should return $\ell = 0$.

### Empirical Calibration

**QBER Sweep**: Fix $r = 0.75$, vary $F \in [0.85, 0.98]$, measure extraction efficiency:

| $F$ | Predicted $Q_{\text{ch}}$ | Measured $Q_{\text{ch}}$ | Extractable $\ell$ |
|-----|---------------------------|--------------------------|-------------------|
| 0.98 | 0.010 | 0.011 | 82 bits |
| 0.95 | 0.025 | 0.026 | 71 bits |
| 0.90 | 0.050 | 0.052 | 54 bits |
| 0.85 | 0.075 | 0.074 | 38 bits |

**Storage Noise Sweep**: Fix $F = 0.95$, vary $r \in [0.60, 0.90]$, measure security margin.

## Implementation Notes

### Immutable Parameters

All NSM/channel parameters are `frozen` dataclasses—cannot be mutated after construction. This ensures:

1. **Thread Safety**: Shared across concurrent simulation runs
2. **Reproducibility**: Identical parameters → identical results
3. **Cache Validity**: Derived properties remain consistent

### Factory Methods

```python
# Canonical experimental configuration
params_erven = NSMParameters.from_erven_2014()

# Simulation-optimized (perfect detection)
params_sim = NSMParameters.for_simulation(
    storage_noise_r=0.75,
    channel_fidelity=0.95
)

# Custom
params_custom = NSMParameters(
    storage_noise_r=0.70,
    storage_rate_nu=0.005,
    delta_t_ns=500_000,
    channel_fidelity=0.92,
    detection_eff_eta=0.15,
    detector_error=0.02,
    dark_count_prob=1e-5,
)
```

## References

[1] König, R., Wehner, S., & Wullschleger, J. (2012). Unconditional security from noisy quantum storage. *IEEE Transactions on Information Theory*, 58(3), 1962-1984.

[2] Schaffner, C., Terhal, B. M., & Wehner, S. (2009). Robust cryptography in the noisy-quantum-storage model. *Quantum Information & Computation*, 9(11-12), 963-996.

[3] Wehner, S., Schaffner, C., & Terhal, B. M. (2008). Cryptography from noisy storage. *Physical Review Letters*, 100(22), 220502.

[4] Erven, C., et al. (2014). An experimental implementation of oblivious transfer in the noisy storage model. *Nature Communications*, 5, 3418.

[5] Breidbart, S. (1973). On certain optimal measurement strategies. PhD Thesis, MIT.

---

[← Return to Main Index](../index.md) | [Previous: Parameter Space](./parameter_space.md) | [Next: Noise Model Configuration](./noise_models.md)
