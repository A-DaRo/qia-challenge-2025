[← Return to Main Index](../index.md)

# 7.2 Extractable Length Calculation

## Introduction

The core challenge of privacy amplification is determining **how much key material** can be securely extracted from a partially secure reconciled string. Extract too much, and the adversary retains exploitable correlations; extract too little, and valuable entropy is wasted. This section derives the **Lupo key length formula** [1]—a finite-size security bound that accounts for both syndrome leakage and adversarial quantum information.

Unlike asymptotic analyses (which assume $n \to \infty$), this formula provides explicit guarantees for finite key lengths ($n \sim 10^3 - 10^6$), making it directly applicable to experimental implementations.

## Entropy-Theoretic Framework

### Min-Entropy

The **min-entropy** quantifies the adversary's maximal guessing probability for a random variable $X$ given quantum side information $E$:

$$
H_{\min}(X \mid E) = -\log_2 P_g(X \mid E)
$$

where $P_g(X \mid E) = \max_M \sum_x P_X(x) \langle M_x \rangle_{\rho_x^E}$ is the success probability of the optimal quantum measurement $M$ for guessing $x$.

**Interpretation**: If $H_{\min}(X \mid E) = k$, the adversary's best strategy succeeds with probability at most $2^{-k}$. For a uniformly distributed $n$-bit string with no leakage, $H_{\min}(X) = n$.

### Min-Entropy Rate

In the Noisy Storage Model, the adversary stores qubits through a depolarizing channel parameterized by $r \in [0, 1]$. The **min-entropy rate per bit** depends on which bound dominates [2]:

$$
h_{\min}(r) = \max\{h_{\text{DK}}(r), h_{\text{Lupo}}(r)\}
$$

where:

1. **Dupuis-König (Collision) Bound** [3]:
   $$
   h_{\text{DK}}(r) = \Gamma \left[1 - \log_2(1 + 3r^2)\right]
   $$
   with $\Gamma = 1 - \log_2(2 + \sqrt{2}) \approx 0.228$.

2. **Lupo Virtual Erasure Bound** [1]:
   $$
   h_{\text{Lupo}}(r) = 1 - r
   $$

**Crossover**: The bounds intersect at $r \approx 0.25$. For lower storage noise ($r < 0.25$), Dupuis-König dominates; for higher noise ($r \geq 0.25$), the simpler Lupo bound applies.

**Example**: For $r = 0.75$ (Erven et al. experimental regime [4]):

$$
\begin{aligned}
h_{\text{DK}}(0.75) &= 0.228 \times [1 - \log_2(1 + 3 \times 0.5625)] \\
&\approx 0.228 \times (-0.38) \approx -0.087 \quad (\text{negative, discard})\\
h_{\text{Lupo}}(0.75) &= 1 - 0.75 = 0.25
\end{aligned}
$$

Thus, $h_{\min}(0.75) = 0.25$ bits per raw bit.

### Smooth Min-Entropy

The **smooth min-entropy** $H_{\min}^\varepsilon(X \mid E)$ relaxes the strict min-entropy by allowing an $\varepsilon$-close state [5]:

$$
H_{\min}^\varepsilon(X \mid E) = \sup_{\bar{\rho}} \left\{ H_{\min}(X \mid \bar{E}) : \frac{1}{2} \|\rho_{XE} - \bar{\rho}_{XE}\|_1 \leq \varepsilon \right\}
$$

**Purpose**: Accounts for statistical fluctuations in parameter estimation (e.g., QBER) during finite-length runs. The smoothing parameter $\varepsilon$ quantifies the probability that the actual state deviates from the assumed model.

## Lupo Key Length Formula

### Derivation from Leftover Hash Lemma

Consider a reconciled key $X \in \{0, 1\}^n$ where:
- Alice and Bob agree on $X$ after error correction
- Adversary holds quantum state $\rho_E$ from eavesdropping
- Syndrome $\Sigma$ of length $|\Sigma|$ bits was transmitted during reconciliation

Applying the quantum Leftover Hash Lemma with a 2-universal hash function $f: \{0, 1\}^n \to \{0, 1\}^\ell$, the secrecy parameter satisfies:

$$
\varepsilon_{\text{sec}} \leq 2^{(\ell + |\Sigma|)/2} \cdot \frac{1}{\sqrt{2^{H_{\min}^\varepsilon(X \mid E)}}}
$$

Solving for $\ell$ with target $\varepsilon_{\text{sec}}$:

$$
\ell \leq H_{\min}^\varepsilon(X \mid E) - |\Sigma| - 2 \log_2(1/\varepsilon_{\text{sec}})
$$

**Floor Function**: Since $\ell$ must be an integer number of bits:

$$
\ell = \left\lfloor H_{\min}^\varepsilon(X \mid E) - |\Sigma| - 2 \log_2(1/\varepsilon_{\text{sec}}) + 2 \right\rfloor
$$

The $+2$ term is a minor adjustment from the finite-size correction in Lupo et al. [1] Eq. (43).

### Security Penalty

The term $\Delta_{\text{sec}} = 2 \log_2(1/\varepsilon_{\text{sec}}) - 2$ is the **security penalty**. For standard parameters:

| $\varepsilon_{\text{sec}}$ | $\Delta_{\text{sec}}$ (bits) |
|----------------------------|------------------------------|
| $10^{-6}$                  | 38                           |
| $10^{-8}$                  | 51                           |
| $10^{-10}$                 | 64                           |
| $10^{-12}$                 | 78                           |

**Physical Meaning**: This penalty accounts for the statistical distinguishability between the extracted key and a truly uniform key. Cryptographic applications typically require $\varepsilon_{\text{sec}} \leq 10^{-10}$, corresponding to a 64-bit overhead per extraction.

### Syndrome Leakage

Error correction reveals the syndrome $\Sigma$, which leaks information about the reconciled key:

$$
|\Sigma| = n - k
$$

where $k$ is the code dimension. For an LDPC code of rate $R$:

$$
|\Sigma| = n \cdot (1 - R)
$$

**Example**: A rate-0.5 LDPC code on a 1000-bit key leaks $|\Sigma| = 500$ bits. This directly subtracts from extractable entropy.

**Optimization Trade-off**:
- **High QBER**: Requires low code rate $R$ → large syndrome leakage
- **Low QBER**: Can use high $R$ → minimal leakage

Caligo's rate-compatible punctured LDPC codes adapt $R$ dynamically based on measured QBER to minimize leakage while ensuring error-free reconciliation.

## Implementation in Caligo

### SecureKeyLengthCalculator Class

The `SecureKeyLengthCalculator` encapsulates the Lupo formula:

```python
class SecureKeyLengthCalculator:
    def __init__(self, entropy_calculator: NSMEntropyCalculator, 
                 epsilon_sec: float = 1e-10):
        self._entropy_calc = entropy_calculator
        self._epsilon_sec = epsilon_sec
    
    def compute_final_length(self, reconciled_length: int, 
                             syndrome_leakage: int) -> int:
        h_min, _ = self._entropy_calc.max_bound_entropy_rate()
        entropy_available = h_min * reconciled_length
        security_penalty = 2 * math.log2(1 / self._epsilon_sec) - 2
        
        raw_length = entropy_available - syndrome_leakage - security_penalty
        return max(0, int(math.floor(raw_length)))
```

### Detailed Result Structure

The `compute_detailed()` method returns a `KeyLengthResult` dataclass with full breakdown:

```python
@dataclass
class KeyLengthResult:
    final_length: int             # Extractable key bits
    raw_length: int               # Input reconciled length
    entropy_available: float      # Total min-entropy (n × h_min)
    entropy_consumed: float       # Used entropy (ℓ + penalty)
    security_penalty: float       # Δ_sec term
    syndrome_leakage: int         # |Σ| bits
    is_viable: bool               # True if final_length > 0
    efficiency: float             # ℓ / n ratio
```

**Use Case**: Diagnostics during protocol optimization to identify bottlenecks (e.g., excessive syndrome leakage or insufficient storage noise).

## Death Valley Phenomenon

### Definition

"Death Valley" occurs when entropy depletion makes secure key extraction impossible:

$$
n \cdot h_{\min}(r) < |\Sigma| + \Delta_{\text{sec}}
$$

In this regime, $\ell = 0$—no key can be extracted.

### Critical Parameters

For $r = 0.75$ ($h_{\min} = 0.25$), $\varepsilon_{\text{sec}} = 10^{-10}$ ($\Delta_{\text{sec}} = 64$):

$$
|\Sigma| < 0.25n - 64
$$

**Example**: For $n = 1000$:
- **Available entropy**: $250$ bits
- **Security penalty**: $64$ bits
- **Maximum tolerable leakage**: $186$ bits
- **Corresponding code rate**: $R \geq 0.814$

If QBER necessitates $R < 0.814$ (e.g., $\text{QBER} > 8\%$), extraction fails.

### Mitigations

1. **Increase $n$**: Larger raw keys dilute the fixed penalty
   - For $n = 10{,}000$: maximum leakage $\approx 2436$ bits ($R \geq 0.76$)

2. **Improve Storage Noise**: Higher $r$ increases $h_{\min}$
   - At $r = 0.5$: $h_{\min} = 0.5$ → tolerable leakage $\approx 436$ bits for $n = 1000$

3. **Tolerate Higher $\varepsilon_{\text{sec}}$**: Reduces penalty but weakens security
   - $\varepsilon_{\text{sec}} = 10^{-6}$: $\Delta_{\text{sec}} = 38$ bits (vs. 64)

4. **Advanced Reconciliation**: Blind reconciliation reduces leakage by $\sim 5\%$ relative to rate-adaptive codes

## Finite-Size Corrections

### Statistical Fluctuations

The smooth min-entropy accounts for parameter estimation uncertainty. If QBER is estimated from a sample of size $m$, the Hoeffding bound gives:

$$
\Pr[|\hat{Q} - Q| > \delta] \leq 2 e^{-2m\delta^2}
$$

For confidence $1 - \varepsilon_{\text{est}}$:

$$
\delta = \sqrt{\frac{\log(2/\varepsilon_{\text{est}})}{2m}}
$$

**Example**: Estimating QBER within $\pm 1\%$ with $\varepsilon_{\text{est}} = 10^{-6}$ requires:

$$
m \geq \frac{\log(2 \times 10^6)}{2 \times 0.01^2} \approx 77{,}000 \text{ bits}
$$

This motivates using sufficiently large $n$ to avoid underestimating entropy.

### Minimum Input Length

To extract a target key length $\ell_{\text{target}}$, the required reconciled length is:

$$
n \geq \frac{\ell_{\text{target}} + \Delta_{\text{sec}}}{h_{\min} - f_{\text{leak}}}
$$

where $f_{\text{leak}} = |\Sigma| / n$ is the fractional leakage rate.

**Example**: For $\ell_{\text{target}} = 256$ bits, $h_{\min} = 0.25$, $f_{\text{leak}} = 0.2$ (code rate 0.8):

$$
n \geq \frac{256 + 64}{0.25 - 0.2} = \frac{320}{0.05} = 6400 \text{ bits}
$$

Caligo's `minimum_input_length()` method automates this calculation.

## Security Validation

### Composability

The extracted key $S = f(X)$ satisfies [6]:

$$
\frac{1}{2} \|\rho_{SE} - \omega_S \otimes \rho_E\|_1 \leq \varepsilon_{\text{sec}}
$$

where $\omega_S$ is the uniform state. This ensures:

1. **One-Time Pad Security**: $S$ can be used as a OTP key
2. **Protocol Composition**: $S$ can serve as input to $\binom{2}{1}$-OT without additional privacy amplification
3. **Non-Adaptive Attacks**: Security holds even if the adversary adaptively queries the key

### Practical Verification

Caligo's `SecurityVerifier` validates:

```python
class SecurityVerifier:
    def validate_extractable_length(self, params: NSMParameters,
                                    reconciled_length: int,
                                    syndrome_leakage: int) -> ValidationResult:
        calc = SecureKeyLengthCalculator(...)
        result = calc.compute_detailed(reconciled_length, syndrome_leakage)
        
        return ValidationResult(
            is_secure=(result.final_length > 0),
            extractable_bits=result.final_length,
            efficiency=result.efficiency,
            entropy_margin=result.entropy_available - result.entropy_consumed
        )
```

If `is_secure=False`, the protocol aborts (preferring no key over an insecure key).

## Comparison with QKD

| Aspect | QKD (BB84) | Caligo (NSM) |
|--------|------------|--------------|
| **Adversary Model** | Eve controls channel | Eve has noisy storage |
| **Min-Entropy Source** | Sifted key (post-basis matching) | Full raw key (basis-independent) |
| **Leakage** | QBER estimation + syndrome | Syndrome only (QBER implicit) |
| **Typical $h_{\min}$** | $\approx 1 - H_2(Q)$ | $\approx 0.25$ (for $r=0.75$) |
| **Efficiency** | 70-90% | 20-40% |

The NSM's lower efficiency is offset by its **no-quantum-memory assumption** for honest parties, making it implementally simpler than QKD with quantum storage.

## References

[1] Lupo, C., Peat, J. T., Andersson, E., & Kok, P. (2023). Error-tolerant oblivious transfer in the noisy-storage model. *Physical Review A*, 107(6), 062403.

[2] König, R., Wehner, S., & Wullschleger, J. (2012). Unconditional security from noisy quantum storage. *IEEE Transactions on Information Theory*, 58(3), 1962-1984.

[3] Dupuis, F., Fawzi, O., & Wehner, S. (2013). Entanglement sampling and applications. *IEEE Transactions on Information Theory*, 61(2), 1093-1112.

[4] Erven, C., et al. (2014). An experimental implementation of oblivious transfer in the noisy storage model. *Nature Communications*, 5, 3418.

[5] Renner, R., & Wolf, S. (2005). Simple and tight bounds for information reconciliation and privacy amplification. In *Advances in Cryptology—ASIACRYPT 2005* (pp. 199-216). Springer.

[6] Tomamichel, M., Lim, C. C. W., Gisin, N., & Renner, R. (2012). Tight finite-key analysis for quantum cryptography. *Nature Communications*, 3, 634.

---

[← Return to Main Index](../index.md) | [Previous: Toeplitz Hashing](./toeplitz_hashing.md) | [Next: Key Derivation](./key_derivation.md)
