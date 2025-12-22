[← Return to Main Index](../index.md)

# 6.5 Leakage Accounting: Security Impact of Error Correction

## The Leakage-Security Tradeoff

### Fundamental Constraint

In the Noisy Storage Model, information reconciliation directly reduces the adversary's uncertainty about the honest party's raw string. Unlike passive eavesdropping scenarios, reconciliation leakage in NSM-OT goes **directly to Bob**—the potential adversary attempting to learn Alice's complementary input $X_{1-c}$.

The extractable secure key length is bounded by [1]:

$$
\ell \leq H_{\min}^\varepsilon(X | E) - \text{leak}_{\text{EC}} - 2\log_2\left(\frac{1}{\varepsilon_{\text{sec}}}\right) + 2
$$

**Critical implication**: If $\text{leak}_{\text{EC}} \to H_{\min}^\varepsilon(X|E)$, then $\ell \to 0$. The protocol cannot extract any secure key.

### The Safety Cap

To ensure positive key extraction with margin for statistical fluctuations, define a **safety cap**:

$$
\text{SafetyCap} = \alpha \cdot \left[ n \cdot h_{\min}(r) - 2\log_2\left(\frac{1}{\varepsilon_{\text{sec}}}\right) + 2 \right]
$$

where $\alpha \in [0.7, 0.9]$ accounts for:
1. Finite-size estimation uncertainties
2. Privacy amplification overhead
3. Implementation margin

If cumulative leakage exceeds this cap, reconciliation must abort.

---

## Leakage Component Analysis

### Decomposition

$$
\text{leak}_{\text{EC}} = \text{leak}_{\text{syndrome}} + \text{leak}_{\text{verification}} + \text{leak}_{\text{interaction}}
$$

### Syndrome Leakage

For syndrome-based reconciliation using code $\mathcal{C}(n, k)$ with rate $R = k/n$:

$$
\text{leak}_{\text{syndrome}} = (1 - R) \cdot n
$$

**Key property**: Syndrome length depends only on the **mother code rate** $R_0$, not the effective rate achieved through puncturing/shortening. This is because:

$$
|\mathbf{s}| = \text{rank}(H) = n - k = n(1 - R_0)
$$

regardless of how many symbols are punctured.

**Example**: For $R_0 = 0.5$, $n = 4096$:
$$
\text{leak}_{\text{syndrome}} = 2048 \text{ bits per block}
$$

### Verification Hash Leakage

To detect decoder convergence to incorrect codewords (undetected errors), a verification hash is transmitted:

$$
\text{leak}_{\text{verification}} = |h|
$$

The hash provides error detection probability $1 - 2^{-|h|}$:

| Hash length | Undetected error probability |
|-------------|------------------------------|
| 32 bits | $2.3 \times 10^{-10}$ |
| 64 bits | $5.4 \times 10^{-20}$ |
| 128 bits | $2.9 \times 10^{-39}$ |

**Tradeoff**: Longer hashes improve error detection but increase leakage.

### Interaction Leakage

**Baseline Protocol**:
$$
\text{leak}_{\text{interaction}}^{\text{Baseline}} = m_{\text{sample}}
$$
where $m_{\text{sample}}$ bits are disclosed for QBER estimation.

**Blind Protocol**:
$$
\text{leak}_{\text{interaction}}^{\text{Blind}} = \sum_{i=2}^{t} \Delta_i
$$
where $\Delta_i$ is the number of punctured symbols revealed in iteration $i$.

---

## Min-Entropy Estimation

### Conservative Bound

The smooth min-entropy conditioned on adversarial side information is bounded by [2]:

$$
H_{\min}^\varepsilon(X|E) \geq n \cdot h_{\min}(r) - 2\log_2\left(\frac{1}{\varepsilon_{\text{min-ent}}}\right)
$$

where $h_{\min}(r)$ is the per-bit min-entropy rate under storage noise parameter $r$:

$$
h_{\min}(r) = \max\left\{ \Gamma[1 - \log_2(1 + 3r^2)], \; 1 - r \right\}
$$

with $\Gamma = 1 - \log_2(2 + \sqrt{2}) \approx 0.228$.

### Finite-Size Correction

For parameter estimation from $m$ sample bits:

$$
\Delta_{\text{finite-size}} = \sqrt{\frac{\log(1/\varepsilon_{\text{est}})}{2m}}
$$

The estimated min-entropy becomes:

$$
\hat{H}_{\min}^\varepsilon(X|E) \geq n \cdot [h_{\min}(r) - \Delta_{\text{finite-size}}] - 2\log_2\left(\frac{1}{\varepsilon_{\text{min-ent}}}\right)
$$

**Example** ($n = 10^6$, $r = 0.75$, $\varepsilon = 10^{-10}$):
$$
\hat{H}_{\min}^\varepsilon \geq 10^6 \times 0.25 - 67 \approx 249,933 \text{ bits}
$$

---

## Protocol-Specific Analysis

### Baseline Reconciliation

For rate-$R$ LDPC code with $m$ sample bits and 64-bit hash:

$$
\text{leak}_{\text{Baseline}} = n(1 - R) + 64 + m
$$

**Efficiency constraint**: Achieves efficiency $f = \frac{1-R}{h(Q)}$ only if $R$ matches true QBER $Q$.

### Blind Reconciliation

Expected leakage depends on iteration distribution:

$$
\mathbb{E}[\text{leak}_{\text{Blind}}] = n(1 - R_0) + 64 + \sum_{i=2}^{t} \Delta_i \cdot \Pr[\text{reach iteration } i]
$$

For well-designed codes in the low-QBER regime ($Q \ll Q_{\max}$):
$$
\Pr[\text{iteration 1 success}] \approx 1 - F^{(1)}(Q) \approx 1
$$

yielding leakage close to the syndrome-only baseline.

### Hybrid Puncturing

Combines predetermined patterns with runtime adaptation. Total leakage:

$$
\text{leak}_{\text{Hybrid}} = n(1 - R_0) + 64 + |\mathcal{P}_{\text{revealed}}|
$$

where $|\mathcal{P}_{\text{revealed}}|$ is the number of pattern indices revealed (implicit in shortening).

---

## Security Implications

### Leakage-to-Entropy Ratio

Define the **leakage efficiency**:

$$
\eta_{\text{leak}} = \frac{\text{leak}_{\text{EC}}}{H_{\min}^\varepsilon(X|E)}
$$

Security requires $\eta_{\text{leak}} < 1$. In practice, we enforce:

$$
\eta_{\text{leak}} \leq \frac{1}{\alpha} - \frac{2\log_2(1/\varepsilon_{\text{sec}}) + 2}{H_{\min}^\varepsilon(X|E)}
$$

to ensure positive key extraction.

### Death Valley Boundary

The **Death Valley** regime occurs when:

$$
n \cdot h_{\min}(r) < (1 - R) \cdot n + 2\log_2(1/\varepsilon_{\text{sec}})
$$

Rearranging:

$$
R < 1 - h_{\min}(r) + \frac{2\log_2(1/\varepsilon_{\text{sec}})}{n}
$$

For $r = 0.75$ ($h_{\min} = 0.25$), $\varepsilon_{\text{sec}} = 10^{-10}$, $n = 4096$:

$$
R_{\text{critical}} \approx 1 - 0.25 + 0.016 = 0.766
$$

Codes with $R < 0.766$ (i.e., high QBER requiring low rate) cannot yield positive key.

---

## Accounting Implementation

### Real-Time Tracking

The simulation tracks cumulative leakage across protocol phases:

1. **Quantum phase**: No leakage
2. **Sifting phase**: $\text{leak} += m_{\text{sample}}$ (Baseline only)
3. **Reconciliation phase**: $\text{leak} += (1-R_0) \cdot n + |h| + \sum \Delta_i$
4. **Amplification phase**: Leakage subtracted in key length calculation

### Abort Conditions

Reconciliation aborts if:

$$
\text{leak}_{\text{cumulative}} > \text{SafetyCap}
$$

This prevents security violations from propagating to privacy amplification.

---

## References

[1] C. Lupo, F. Ottaviani, R. Ferrara, and S. Pirandola, "Performance of Practical Quantum Oblivious Key Distribution," *PRX Quantum*, vol. 3, 020353, 2023.

[2] R. König, S. Wehner, and J. Wullschleger, "Unconditional Security from Noisy Quantum Storage," *IEEE Trans. Inf. Theory*, vol. 58, no. 3, pp. 1962–1984, 2012.

[3] J. Martinez-Mateo, D. Elkouss, and V. Martin, "Blind Reconciliation," *Quantum Inf. Comput.*, vol. 12, pp. 791–812, 2012.

---

[← Return to Main Index](../index.md) | [← Previous: Hybrid Puncturing](./hybrid_puncturing.md) | [Next: Toeplitz Hashing →](../amplification/toeplitz_hashing.md)