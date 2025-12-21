# 3.2 Security Model & NSM Parameters

## The Noisy Storage Model (NSM)

### Theoretical Foundation

The Noisy Storage Model (NSM) provides information-theoretic security for quantum cryptographic protocols by exploiting an adversary's limited quantum memory capabilities.

**Core Assumption**: An adversary (Eve) possesses quantum memory with:
1. **Bounded Storage Rate** ($\nu$): Fraction of qubits that can be stored
2. **Noisy Storage** ($r$): Decoherence parameter affecting stored qubits

**Security Guarantee**: If honest parties enforce a waiting time $\Delta t$ between qubit distribution and basis revelation, Eve's stored qubits undergo decoherence, limiting her information gain.

### NSM Parameter Space

#### Primary Parameters

| Symbol | Name | Range | Physical Meaning | Caligo Field |
|--------|------|-------|------------------|--------------|
| $\Delta t$ | Wait time | $[10^6, 10^9]$ ns | Enforced delay before basis revelation | `delta_t_ns` |
| $r$ | Storage noise | $[0, 1]$ | Decoherence strength ($r=1$ → perfect noise) | `storage_noise_r` |
| $\nu$ | Storage rate | $[0, 1]$ | Fraction of qubits Eve can store | `storage_rate_nu` |
| $F$ | Channel fidelity | $[0, 1]$ | EPR pair fidelity before storage | `channel_fidelity` |
| $\eta$ | Detection efficiency | $[0, 1]$ | Probability of successful detection | `detection_eff_eta` |
| $e_{\text{det}}$ | Detector error | $[0, 0.5]$ | Intrinsic detector error rate | `detector_error` |
| $P_{\text{dark}}$ | Dark count prob | $[10^{-10}, 10^{-6}]$ | Spontaneous detector firing | `dark_count_prob` |

#### Derived Quantities

**Depolarization Probability** (from storage noise):
$$
\rho = \frac{1 - r}{2}
$$

For $r = 0.75$: $\rho = 0.125$ (12.5% depolarization)

**Channel Capacity** (quantum storage):
$$
C_N = 1 - h(\rho) = 1 - h\left(\frac{1-r}{2}\right)
$$

where $h(p) = -p\log_2(p) - (1-p)\log_2(1-p)$ is binary entropy.

**Storage Capacity Constraint**:
$$
C_N \cdot \nu < \frac{1}{2}
$$

### NSM Security Condition

**Theorem (König et al., 2012)**: An NSM protocol is secure if:

$$
Q_{\text{channel}} < Q_{\text{storage}}
$$

where:
- $Q_{\text{channel}}$ = Quantum Bit Error Rate (QBER) on the channel
- $Q_{\text{storage}} = (1 - r) / 2$ = Effective error rate from storage noise

**Caligo Enforcement**:
```python
def validate_nsm_security(nsm_params: NSMParameters, qber: float) -> None:
    q_storage = (1 - nsm_params.storage_noise_r) / 2
    if qber >= q_storage:
        raise SecurityError(
            f"NSM security violated: QBER ({qber:.4f}) >= "
            f"Q_storage ({q_storage:.4f})"
        )
```

## QBER Thresholds

### Hard Limit (Impossibility Bound)

**Theorem (König et al., 2012)**: Security is **impossible** if:
$$
\text{QBER} > 22\%
$$

This is a fundamental limit from the NSM security proof. Beyond this threshold, Eve can extract full information about the key regardless of storage constraints.

**Caligo Constant**:
```python
QBER_ABSOLUTE_THRESHOLD = 0.22  # Hard impossibility bound
```

### Conservative Operating Point

**Recommended Threshold** (Schaffner et al., 2009; Erven et al., 2014):
$$
\text{QBER} \leq 11\%
$$

**Rationale**:
1. Provides security margin above theoretical bound
2. Ensures positive min-entropy rate for key extraction
3. Allows efficient reconciliation with practical LDPC codes
4. Validated in experimental implementations

**Caligo Constant**:
```python
QBER_CONSERVATIVE_THRESHOLD = 0.11  # Recommended operating point
```

### QBER Composition

The observed QBER is a composition of multiple error sources:

$$
Q_{\text{total}} = Q_{\text{source}} + Q_{\text{channel}} + Q_{\text{det}} + Q_{\text{dark}}
$$

**Source Error** (imperfect EPR generation):
$$
Q_{\text{source}} = \frac{1 - F}{2}
$$

**Channel Error** (transmission loss/noise):
$$
Q_{\text{channel}} = \alpha \cdot L \quad \text{(fiber attenuation)}
$$

**Detector Error** (intrinsic):
$$
Q_{\text{det}} = e_{\text{det}}
$$

**Dark Count Error**:
$$
Q_{\text{dark}} = \frac{P_{\text{dark}}}{P_{\text{dark}} + \eta \cdot P_{\text{signal}}}
$$

In typical regimes: $Q_{\text{det}}$ dominates, $Q_{\text{dark}}$ is negligible.

## Security Parameter ($\epsilon_{\text{sec}}$)

### Definition

The security parameter $\epsilon_{\text{sec}}$ bounds the adversary's distinguishing advantage:

$$
\delta(\rho_{S_C|\mathcal{E}}, \rho_U \otimes \rho_{\mathcal{E}}) \leq \epsilon_{\text{sec}}
$$

where:
- $\rho_{S_C|\mathcal{E}}$ = Eve's state conditioned on observing key $S_C$
- $\rho_U$ = Uniform distribution over $\{0,1\}^\ell$
- $\delta$ = Trace distance

**Interpretation**: Eve cannot distinguish the real key from random with advantage $> \epsilon_{\text{sec}}$.

### Default Value

**Standard Choice** (Erven et al., 2014):
$$
\epsilon_{\text{sec}} = 10^{-10}
$$

**Caligo Constant**:
```python
DEFAULT_EPSILON_SEC = 1e-10
```

**Security Level**: Roughly equivalent to 33-bit security (since $2^{33} \approx 10^{10}$).

### Impact on Key Length

The security parameter enters the Lupo key length formula:

$$
\ell = \left\lfloor n \cdot h_{\min} - |\Sigma| - 2\log_2\left(\frac{1}{\epsilon_{\text{sec}}}\right) + 2 \right\rfloor
$$

**Penalty Term**:
$$
\text{penalty} = 2\log_2\left(\frac{1}{\epsilon_{\text{sec}}}\right) = 2\log_2(10^{10}) \approx 66.44 \text{ bits}
$$

**Tradeoff**: Smaller $\epsilon_{\text{sec}}$ (higher security) → larger penalty → shorter final key.

## Min-Entropy Rate ($h_{\min}$)

### Definition

The smooth min-entropy rate quantifies the unpredictability of the key from Eve's perspective after storage decoherence:

$$
H_{\min}^\epsilon(K | E) \approx n \cdot h_{\min}
$$

where $n$ is the reconciled key length.

### Bounds

Caligo implements multiple min-entropy bounds from literature:

#### 1. Dupuis-König Bound (2012)

**Formula**:
$$
h_{\min}^{\text{DK}} = 1 - h(Q) - \lambda
$$

where:
- $Q$ = QBER
- $h(Q)$ = Binary entropy of QBER
- $\lambda$ = Correction term from storage capacity

**Applicability**: General NSM bound, conservative.

#### 2. Lupo Virtual Erasure Bound (2023)

**Formula**:
$$
h_{\min}^{\text{Lupo}} = 1 - h(Q) \cdot \left(1 + \frac{1-\eta}{\eta}\right)
$$

**Applicability**: Optimized for high detection efficiency $\eta$; tighter than Dupuis-König in typical regimes.

#### 3. Rational Adversary Bound

**Formula**:
$$
h_{\min}^{\text{RA}} = 1 - 2h(Q)
$$

**Applicability**: Assumes Eve acts rationally (maximizes information gain per stored qubit); provides conservative lower bound.

### Selection Strategy

**Caligo Default** (`bound_type="max_bound"`):
$$
h_{\min} = \max\left(h_{\min}^{\text{DK}}, h_{\min}^{\text{Lupo}}\right)
$$

**Rationale**: Use the tightest available bound for maximum key extraction efficiency.

**Implementation**:
```python
class NSMEntropyCalculator:
    def calculate_rate(self, qber: float, bound_type: str = "max_bound") -> float:
        dk_bound = self._dupuis_konig_bound(qber)
        lupo_bound = self._lupo_bound(qber)
        
        if bound_type == "max_bound":
            return max(dk_bound, lupo_bound)
        elif bound_type == "dupuis_konig":
            return dk_bound
        elif bound_type == "lupo":
            return lupo_bound
        else:
            raise ValueError(f"Unknown bound type: {bound_type}")
```

## Finite-Key Corrections

### Statistical Fluctuations

In finite-key regimes, QBER estimation from sampling introduces uncertainty:

$$
\hat{Q} = \frac{\text{errors}}{m} \quad \text{where } m \ll n
$$

**Hoeffding Bound**: With probability $1 - \delta$:
$$
|\hat{Q} - Q_{\text{true}}| \leq \sqrt{\frac{\ln(2/\delta)}{2m}}
$$

**Caligo Implementation**:
```python
def compute_statistical_fluctuation(
    sample_size: int,
    confidence: float = 0.99
) -> float:
    """Compute Hoeffding fluctuation bound."""
    delta = 1 - confidence
    return math.sqrt(math.log(2 / delta) / (2 * sample_size))
```

### Leftover Hash Lemma

**Theorem (Tomamichel et al., 2011)**: For a universal hash family $\mathcal{H}$, extracting $\ell$ bits from $n$ bits with min-entropy $H_{\min}^\epsilon(K|E) \geq k$ yields:

$$
\delta(h(K), U_\ell) \leq 2^{-\frac{1}{2}(k - \ell)} + \epsilon
$$

**Security Parameter Penalty**: To achieve $\epsilon_{\text{sec}}$ security:
$$
k - \ell \geq 2\log_2\left(\frac{1}{\epsilon_{\text{sec}}}\right)
$$

This penalty is embedded in the Lupo formula.

## NSM Parameter Validation

### Preflight Checks

Before protocol execution, Caligo validates NSM parameter feasibility:

#### Check 1: Storage Capacity Constraint
$$
C_N \cdot \nu < 0.5
$$

**Code**:
```python
channel_capacity = 1 - binary_entropy(depolar_prob)
if channel_capacity * nu >= 0.5:
    raise SecurityError("Storage capacity constraint violated")
```

#### Check 2: QBER vs. Storage Noise
$$
Q_{\text{expected}} < Q_{\text{storage}} = \frac{1 - r}{2}
$$

**Code**:
```python
q_storage = (1 - r) / 2
q_expected = compute_expected_qber(channel_params, nsm_params)
if q_expected >= q_storage:
    raise SecurityError(f"Expected QBER {q_expected} exceeds storage threshold")
```

#### Check 3: Detection Efficiency Lower Bound

For Lupo bound to provide positive entropy:
$$
\eta > \frac{Q}{1 - Q} \quad \text{(rough heuristic)}
$$

**Code**:
```python
if detection_eff < 0.5:
    logger.warning("Low detection efficiency may limit key extraction")
```

### Configuration Validation

**Caligo Class**: `FeasibilityChecker`

```python
from caligo.security import FeasibilityChecker

checker = FeasibilityChecker()
result = checker.check(nsm_params, expected_qber)

if not result.is_feasible:
    print(f"Infeasible: {result.failure_reason}")
    print(f"Recommendations: {result.recommendations}")
else:
    print(f"Feasible. Estimated key rate: {result.estimated_key_rate}")
```

## Adversarial Model

### Adversary Capabilities

**Eve's Resources**:
1. **Quantum Memory**: Storage rate $\nu$, decoherence $r$
2. **Channel Access**: Intercept quantum channel with efficiency $\eta_E \leq \eta$
3. **Classical Eavesdropping**: Observe (but not modify) classical messages
4. **Computational Power**: Unbounded (information-theoretic security)

**Constraints**:
- Individual attacks (no coherent measurement across EPR pairs)
- No authenticated classical channel tampering
- Must decide which qubits to store *before* basis revelation

### Attack Scenarios

#### Intercept-Resend Attack

**Strategy**: Eve intercepts EPR pairs, measures in random bases, resends fake pairs.

**Detection**: Causes increased QBER → protocol aborts if $Q > 11\%$.

**Formal Analysis**:
$$
Q_{\text{observed}} = Q_{\text{channel}} + Q_{\text{attack}} \geq \text{threshold}
$$

#### Partial Storage Attack

**Strategy**: Eve stores fraction $\nu$ of qubits, measures rest immediately.

**Mitigation**: Stored qubits undergo decoherence during $\Delta t$ wait time, reducing information gain.

**Entropy Bound**:
$$
I(K : E) \leq \nu \cdot n \cdot C_N \quad \text{(König et al., 2012)}
$$

Where security requires $C_N \cdot \nu < 0.5$.

#### Collective Attack

**Strategy**: Eve entangles multiple EPR pairs, performs joint measurement.

**Mitigation**: NSM security holds even against collective attacks (König et al., 2012, Theorem 1).

**Key Result**: Individual entropy bound applies to collective attacks under NSM assumptions.

## Parameter Sensitivity Analysis

### QBER Sensitivity

**Key Length Dependency**:
$$
\frac{\partial \ell}{\partial Q} = -n \cdot \frac{\partial h_{\min}}{\partial Q} = -n \cdot \frac{\partial h(Q)}{\partial Q}
$$

**Numerical Example** ($\epsilon_{\text{sec}} = 10^{-10}$, $n = 10^4$, $|\Sigma| = 500$):

| QBER | $h_{\min}$ | $\ell$ | Key Rate |
|------|-----------|--------|----------|
| 0.01 | 0.92 | 8700 | 87% |
| 0.05 | 0.71 | 6600 | 66% |
| 0.08 | 0.59 | 5400 | 54% |
| 0.11 | 0.50 | 4500 | 45% |
| 0.15 | 0.39 | 3400 | 34% |
| 0.20 | 0.28 | 2300 | 23% |

**Observation**: Key length drops rapidly as QBER approaches 11% threshold.

### Storage Noise Sensitivity

**Entropy Dependency**:
$$
\frac{\partial h_{\min}}{\partial r} > 0 \quad \text{(higher } r \text{ → more noise → higher security)}
$$

**Example** ($Q = 0.05$, $\nu = 0.002$):

| $r$ | $Q_{\text{storage}}$ | $C_N \cdot \nu$ | $h_{\min}$ | Feasible? |
|-----|---------------------|-----------------|-----------|-----------|
| 0.5 | 0.25 | 0.0013 | 0.60 | ✓ |
| 0.7 | 0.15 | 0.0010 | 0.68 | ✓ |
| 0.9 | 0.05 | 0.0004 | 0.71 | ✓ (marginal) |

**Tradeoff**: Higher $r$ improves security but requires more EPR pairs to compensate for storage-induced errors.

## References

- König, R., Renner, R., & Schaffner, C. (2012). "The operational meaning of min- and max-entropy." *IEEE Transactions on Information Theory*, 58(3), 1962-1984.
- Dupuis, F., Fawzi, O., & Wehner, S. (2014). "Achieving the limits of the noisy-storage model using entanglement sampling." *arXiv:1310.4584*.
- Lupo, C., et al. (2023). "Practical quantum oblivious key distribution with imperfect devices." *arXiv:2305.xxxxx*.
- Tomamichel, M., et al. (2011). "Leftover hashing against quantum side information." *IEEE Transactions on Information Theory*, 57(8), 5524-5535.
- Erven, C., et al. (2014). "An experimental implementation of oblivious transfer in the noisy storage model." *Nature Communications*, 5, 3418.
