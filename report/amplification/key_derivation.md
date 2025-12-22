[← Return to Main Index](../index.md)

# 7.3 Privacy Amplification and Key Derivation

## Introduction

Privacy amplification transforms a partially secret string $X \in \{0,1\}^n$—about which an adversary possesses partial information—into a shorter, uniformly random string $S \in \{0,1\}^\ell$ that is statistically independent of the adversary's knowledge. This transformation is the final step in establishing information-theoretic security for NSM-based protocols.

The theoretical foundation is the **Leftover Hash Lemma** (LHL) [1], which guarantees that hashing with a randomly chosen function from a 2-universal family extracts nearly all the min-entropy from $X$.

## Information-Theoretic Framework

### Adversary's Knowledge Model

After the quantum phase, sifting, and reconciliation, the adversary (dishonest Bob or Eve) possesses a quantum system $B$ correlated with Alice's classical string $X$. The joint state is described by the classical-quantum state:

$$
\rho_{XB} = \sum_{x \in \{0,1\}^n} P_X(x) |x\rangle\langle x| \otimes \rho_B^x
$$

The adversary's knowledge is quantified by the **conditional min-entropy**:

$$
H_{\min}(X|B) = -\log_2 \max_{\{M_x\}} \sum_x P_X(x) \cdot \text{Tr}[M_x \rho_B^x]
$$

where the maximum is over all POVMs $\{M_x\}$. This equals the negative log of the adversary's maximum guessing probability.

### Smooth Min-Entropy

For finite-size analysis, the **smooth min-entropy** [2] is defined as:

$$
H_{\min}^\varepsilon(X|B) = \max_{\rho'_{XB} \in B^\varepsilon(\rho_{XB})} H_{\min}(X|B)_{\rho'}
$$

where $B^\varepsilon(\rho)$ is the $\varepsilon$-ball of states with trace distance $\|\rho' - \rho\|_1 \leq \varepsilon$.

The smoothing allows for a small probability $\varepsilon$ of deviation from the idealized scenario.

## The Leftover Hash Lemma

### 2-Universal Hash Families

**Definition**: A family $\mathcal{H} = \{h : \{0,1\}^n \to \{0,1\}^\ell\}$ is **2-universal** if for all $x \neq x'$:

$$
P_{h \leftarrow \mathcal{H}}[h(x) = h(x')] \leq 2^{-\ell}
$$

**Interpretation**: Collisions occur with probability no greater than for truly random functions.

### Classical LHL

**Theorem** (Leftover Hash Lemma) [1]: Let $X$ be a random variable with min-entropy $H_{\min}(X) \geq k$, and let $h$ be chosen uniformly from a 2-universal family $\mathcal{H}$. Then:

$$
\frac{1}{2}\|P_{h(X), h} - U_\ell \times P_h\|_1 \leq \frac{1}{2}\sqrt{2^{\ell - k}}
$$

where $U_\ell$ is the uniform distribution on $\{0,1\}^\ell$.

**Corollary**: For security parameter $\varepsilon$, choose output length:

$$
\ell \leq k - 2\log_2(1/\varepsilon)
$$

to achieve statistical distance $\varepsilon$ from uniform.

### Quantum LHL

**Theorem** (Quantum LHL) [3]: Let $\rho_{XB}$ be a classical-quantum state with $H_{\min}^\varepsilon(X|B) \geq k$. Then for $h \leftarrow \mathcal{H}$ (2-universal):

$$
\frac{1}{2}\|\rho_{h(X), h, B} - U_\ell \otimes \rho_{h, B}\|_1 \leq \varepsilon + \frac{1}{2}\sqrt{2^{\ell - k}}
$$

**Security Guarantee**: The extracted key $S = h(X)$ is $(\varepsilon + 2^{(\ell-k)/2})$-close to being uniformly random and independent of the adversary's system.

## Extractable Key Length Formula

### Lupo Formula

Combining the NSM min-entropy bound with the quantum LHL, the secure key length is [4]:

$$
\ell \leq n \cdot h_{\min}(r) - \text{leak}_{\text{EC}} - 2\log_2(1/\varepsilon_{\text{sec}}) + 2
$$

where:
- $n$ is the reconciled string length
- $h_{\min}(r)$ is the min-entropy rate for storage parameter $r$
- $\text{leak}_{\text{EC}}$ is the error correction leakage (syndrome bits)
- $\varepsilon_{\text{sec}}$ is the security parameter

### Component Analysis

**Min-Entropy Term**: For depolarizing storage with parameter $r$:

$$
h_{\min}(r) = \max\left\{\Gamma[1 - \log_2(1 + 3r^2)], \; 1 - r\right\}
$$

where $\Gamma \approx 0.228$ is the Dupuis-König constant [5].

**Leakage Term**: Error correction reveals $\text{leak}_{\text{EC}} = n(1 - R_{\text{eff}}) \approx n \cdot f(Q) \cdot h(Q)$ bits.

**Security Penalty**: The term $2\log_2(1/\varepsilon_{\text{sec}}) - 2$ accounts for:
- Smoothing of min-entropy ($\log_2(1/\varepsilon)$)
- LHL security guarantee ($\log_2(1/\varepsilon)$)

### Death Valley Threshold

Setting $\ell = 0$ determines the minimum viable block length:

$$
n_{\min} = \frac{2\log_2(1/\varepsilon_{\text{sec}}) - 2}{h_{\min}(r) - f(Q) \cdot h(Q)}
$$

For $r = 0.75$, $Q = 0.05$, $f = 1.10$, $\varepsilon_{\text{sec}} = 10^{-10}$:

$$
n_{\min} \approx \frac{64}{0.25 - 0.315} \approx -1000 \quad \text{(infeasible)}
$$

This illustrates the **Death Valley** phenomenon: the parameter regime where finite-size penalties preclude positive key rates.

## Toeplitz Hashing Implementation

### Construction

A **Toeplitz matrix** $T \in \{0,1\}^{\ell \times n}$ is defined by its first row and column, requiring only $n + \ell - 1$ random bits:

$$
T_{ij} = t_{i-j+n}
$$

where $\{t_1, \ldots, t_{n+\ell-1}\}$ are the seed bits.

### 2-Universality Proof

**Theorem**: The family of Toeplitz matrices is 2-universal.

**Proof**: For $x \neq x'$, let $d = x \oplus x'$. Then:

$$
P[Tx = Tx'] = P[Td = 0] = \prod_{i=1}^{\ell} P[\langle t^{(i)}, d \rangle = 0]
$$

where $t^{(i)}$ is row $i$ of $T$. Since $d \neq 0$, each inner product is uniform on $\{0,1\}$:

$$
P[Tx = Tx'] = 2^{-\ell} \quad \square
$$

### Computational Efficiency

**Direct computation**: $O(n \cdot \ell)$ bit operations.

**FFT acceleration**: Toeplitz matrix-vector multiplication reduces to circular convolution:

$$
Tx = \text{IFFT}(\text{FFT}(\bar{t}) \odot \text{FFT}(\bar{x}))
$$

Complexity: $O((n + \ell) \log(n + \ell))$ — a significant speedup for $\ell \sim n$.

## Security Composition

### Seed Distribution

The Toeplitz seed $t$ must be:
1. **Authenticated**: Bob verifies the seed originates from Alice
2. **Public**: No confidentiality required (consistent with Kerckhoffs' principle)

The seed is transmitted after reconciliation over the authenticated classical channel.

### Composable Security

For composable security analysis [6], the final key satisfies:

$$
\|\rho_{S, E} - U_\ell \otimes \rho_E\|_1 \leq \varepsilon_{\text{PA}} + \varepsilon_{\text{PE}} + \varepsilon_{\text{EC}}
$$

where:
- $\varepsilon_{\text{PA}}$: Privacy amplification failure probability
- $\varepsilon_{\text{PE}}$: Parameter estimation failure probability  
- $\varepsilon_{\text{EC}}$: Error correction failure probability

---

## References

[1] C. H. Bennett, G. Brassard, C. Crépeau, and U. M. Maurer, "Generalized privacy amplification," *IEEE Trans. Inf. Theory*, vol. 41, no. 6, pp. 1915–1923, 1995.

[2] R. Renner, "Security of Quantum Key Distribution," Ph.D. thesis, ETH Zürich, 2005.

[3] R. Renner and R. König, "Universally composable privacy amplification against quantum adversaries," *Proc. TCC 2005*, LNCS 3378, pp. 407–425, 2005.

[4] C. Lupo, F. Ottaviani, R. Ferrara, and S. Pirandola, "Performance of Practical Quantum Oblivious Key Distribution," *PRX Quantum*, vol. 3, 020353, 2023.

[5] F. Dupuis, O. Fawzi, and R. Renner, "Entropy accumulation," *Commun. Math. Phys.*, vol. 379, pp. 867–913, 2020.

[6] M. Tomamichel, C. C. W. Lim, N. Gisin, and R. Renner, "Tight finite-key analysis for quantum cryptography," *Nat. Commun.*, vol. 3, 634, 2012.

---

[← Return to Main Index](../index.md)
