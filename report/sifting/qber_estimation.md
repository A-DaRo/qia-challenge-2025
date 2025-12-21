# 5.2 QBER Estimation

## Definition

**Quantum Bit Error Rate (QBER)**: Fraction of mismatched bits in sifted keys.

$$
Q = \frac{|\{i : K_A[i] \neq K_B[i]\}|}{|K_A|}
$$

## Estimation Procedure

### Sampling-Based Estimation

**Rationale**: Comparing full keys reveals all information → must sample subset.

**Algorithm**:
```
Input: K_A, K_B (sifted keys), m (sample size)
Output: Q̂ (QBER estimate)

1. Select m random indices: I = {i₁, i₂, ..., iₘ}
2. Count errors: e = |{i ∈ I : K_A[i] ≠ K_B[i]}|
3. Compute estimate: Q̂ = e / m
4. Remove sampled bits from keys (information reconciliation uses remainder)
```

### Implementation

```python
class QBEREstimator:
    def estimate(
        self,
        alice_key: bitarray,
        bob_key: bitarray,
        sample_size: int,
        confidence: float = 0.99,
    ) -> QBEREstimate:
        # Sample random positions
        n = len(alice_key)
        indices = random.sample(range(n), sample_size)
        
        # Count errors
        errors = sum(alice_key[i] != bob_key[i] for i in indices)
        qber = errors / sample_size
        
        # Compute Hoeffding bound
        delta = 1 - confidence
        epsilon = math.sqrt(math.log(2 / delta) / (2 * sample_size))
        
        bound = HoeffdingBound(
            lower_bound=max(0, qber - epsilon),
            upper_bound=min(1, qber + epsilon),
            confidence=confidence,
            sample_size=sample_size,
        )
        
        return QBEREstimate(
            qber=qber,
            num_errors=errors,
            num_compared=sample_size,
            hoeffding_bound=bound,
        )
```

## Statistical Confidence

### Hoeffding Bound

**Theorem** (Hoeffding, 1963): With probability $1 - \delta$,
$$
|\hat{Q} - Q_{\text{true}}| \leq \epsilon = \sqrt{\frac{\ln(2/\delta)}{2m}}
$$

**Confidence Interval**:
$$
Q_{\text{true}} \in [\hat{Q} - \epsilon, \hat{Q} + \epsilon]
$$

### Sample Size Selection

**Accuracy Requirement**: $\epsilon = 0.01$ (1% precision)

**Required Sample Size** ($\delta = 0.01$):
$$
m = \frac{\ln(2/\delta)}{2\epsilon^2} = \frac{\ln(200)}{2 \times 0.01^2} \approx 26,492
$$

**Practical Values**:
| Precision ($\epsilon$) | Confidence ($1-\delta$) | Sample Size ($m$) |
|------------------------|-------------------------|-------------------|
| 0.01 (1%)              | 99%                     | 26,492            |
| 0.01 (1%)              | 95%                     | 18,445            |
| 0.005 (0.5%)           | 99%                     | 105,966           |

## QBER Sources

### Error Composition

$$
Q_{\text{total}} = Q_{\text{source}} + Q_{\text{channel}} + Q_{\text{det}} + Q_{\text{dark}}
$$

**Source Error** (imperfect EPR state):
$$
Q_{\text{source}} = \frac{1 - F}{2}
$$

**Channel Error** (fiber attenuation, scattering):
$$
Q_{\text{channel}} = \text{function of } \alpha, L
$$

**Detector Error** (intrinsic detector noise):
$$
Q_{\text{det}} = e_{\text{det}}
$$

**Dark Count Error**:
$$
Q_{\text{dark}} \approx \frac{P_{\text{dark}}}{P_{\text{dark}} + \eta \cdot P_{\text{signal}}}
$$

### Dominant Terms

In typical regimes:
- **Erven et al. (2014)**: $Q_{\text{det}} = 0.0093$ dominates (η = 0.015)
- **High-η systems**: $Q_{\text{source}}$ or $Q_{\text{channel}}$ dominate

## Security Check

```python
if qber_estimate.qber > QBER_CONSERVATIVE_THRESHOLD:
    raise QBERThresholdExceeded(
        f"QBER {qber_estimate.qber:.4f} exceeds threshold "
        f"{QBER_CONSERVATIVE_THRESHOLD}"
    )
```

**Thresholds**:
- **Conservative**: 11% (Schaffner et al., 2009)
- **Absolute**: 22% (König et al., 2012 — impossibility bound)

## References

- Hoeffding, W. (1963). "Probability inequalities for sums of bounded random variables." *J. Amer. Statist. Assoc.*, 58(301), 13-30.
- Erven, C., et al. (2014). "An experimental implementation of oblivious transfer in the noisy storage model." *Nat. Commun.*, 5, 3418.
- Caligo implementation: [`caligo/sifting/qber.py`](../../caligo/caligo/sifting/qber.py)
