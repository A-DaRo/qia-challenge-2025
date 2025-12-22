[← Return to Main Index](../index.md)

# 7.1 Two-Universal Hashing with Toeplitz Matrices

## The Privacy Amplification Problem

### Physical Motivation

After error correction, Alice and Bob share a reconciled string $X \in \{0,1\}^n$. However, an adversary (dishonest Bob) holds quantum side information $\rho_E$ correlated with $X$. The challenge is to extract a shorter string $K$ that is:

1. **Statistically uniform**: $K$ is within trace distance $\varepsilon_{\text{sec}}$ of the uniform distribution
2. **Independent of $\rho_E$**: The adversary's quantum state provides negligible advantage in guessing $K$

This is achieved through **privacy amplification**—a compression procedure that destroys the adversary's partial information.

---

## Two-Universal Hash Families

### Definition

A family $\mathcal{H}$ of hash functions $h: \{0,1\}^n \to \{0,1\}^\ell$ is **2-universal** if for all distinct $x, y \in \{0,1\}^n$:

$$
\Pr_{h \leftarrow \mathcal{H}}[h(x) = h(y)] \leq 2^{-\ell}
$$

This collision bound is optimal—it equals the probability that two random $\ell$-bit strings coincide.

### The Leftover Hash Lemma

The security of privacy amplification rests on the **quantum Leftover Hash Lemma** (Tomamichel et al. [1]):

**Theorem**: Let $\rho_{XE}$ be a classical-quantum state where $X$ is classical and $E$ is a quantum register held by an adversary. Let $\mathcal{H}$ be a 2-universal hash family. Then for uniformly random $H \in \mathcal{H}$:

$$
\frac{1}{2} \left\| \rho_{H(X)HE} - \frac{\mathbb{I}}{2^\ell} \otimes \rho_{HE} \right\|_1 \leq \varepsilon_{\text{sec}}
$$

provided:

$$
\ell \leq H_{\min}(X | E) - 2\log_2\left(\frac{1}{\varepsilon_{\text{sec}}}\right) + 2
$$

**Physical interpretation**: The hashed output $H(X)$ is $\varepsilon_{\text{sec}}$-close to uniform and independent of the adversary's quantum state, even when the hash function description $H$ is publicly revealed.

---

## Toeplitz Matrix Construction

### Algebraic Structure

A **Toeplitz matrix** $T \in \{0,1\}^{\ell \times n}$ has constant entries along each diagonal:

$$
T_{i,j} = T_{i-1,j-1} \quad \forall i \in [2,\ell], j \in [2,n]
$$

The matrix is fully determined by its first row $(r_1, \ldots, r_n)$ and first column $(c_1, \ldots, c_\ell)$, requiring only $n + \ell - 1$ random bits rather than $n\ell$.

Explicitly:

$$
T = \begin{pmatrix}
c_1 & r_2 & r_3 & \cdots & r_n \\
c_2 & c_1 & r_2 & \cdots & r_{n-1} \\
c_3 & c_2 & c_1 & \cdots & r_{n-2} \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
c_\ell & c_{\ell-1} & c_{\ell-2} & \cdots & r_{n-\ell+1}
\end{pmatrix}
$$

### Hash Function

The hash of input $x \in \{0,1\}^n$ is:

$$
h_T(x) = T \cdot x \pmod{2}
$$

where arithmetic is over the binary field $\mathbb{F}_2$. Each output bit is a parity check:

$$
h_T(x)_i = \bigoplus_{j=1}^{n} T_{i,j} \cdot x_j
$$

### Two-Universality Proof

**Claim**: The set $\{h_T : T \text{ Toeplitz}\}$ is 2-universal.

**Proof**: For $x \neq y$, let $d = x \oplus y \neq 0$. Then:
$$
h_T(x) = h_T(y) \iff T \cdot d = 0 \pmod{2}
$$

Since $d \neq 0$, there exists $j^*$ with $d_{j^*} = 1$. The $i$-th row of $T \cdot d$ depends linearly on $T_{i,j^*}$, which (in a random Toeplitz matrix) is uniformly distributed over $\{0,1\}$ and independent across rows. Thus:
$$
\Pr[T \cdot d = 0] = 2^{-\ell}
$$
as required. $\square$

---

## Computational Efficiency

### Direct Multiplication

Naive matrix-vector multiplication requires $O(n\ell)$ bit operations. For $n = 10^6$, $\ell = 10^5$, this is $10^{11}$ operations—prohibitive.

### FFT Acceleration

The Toeplitz structure enables efficient multiplication via the Fast Fourier Transform [2]:

1. **Embed** $T$ into a circulant matrix $C$ of size $2^k \geq n + \ell - 1$
2. **Compute** FFTs of the defining column of $C$ and the padded input $x$
3. **Multiply** element-wise in the frequency domain
4. **Apply** inverse FFT and extract the first $\ell$ bits

**Complexity**: $O(n \log n)$ using FFT, a dramatic improvement for large keys.

### Binary Field Optimization

Over $\mathbb{F}_2$, the FFT-based approach requires careful handling:
- Use the **Number Theoretic Transform** (NTT) with appropriate prime modulus
- Alternatively, use **bitwise convolution** with Karatsuba multiplication

In practice, the $O(n\ell)$ direct method suffices for $n \lesssim 10^4$; FFT is essential for $n \gtrsim 10^5$.

---

## Security Analysis in the NSM Context

### Min-Entropy Under Storage Noise

In the Noisy Storage Model, the adversary's min-entropy is determined by the storage channel:

$$
H_{\min}(X | E) \geq n \cdot h_{\min}(r)
$$

where $h_{\min}(r) = \max\{\Gamma[1 - \log_2(1 + 3r^2)], 1 - r\}$ is the per-bit bound under depolarizing noise with parameter $r$ (see [§2.1](../foundations/nsm_model.md)).

**Example** ($r = 0.75$, $n = 10^4$):
$$
H_{\min}(X|E) \geq 10^4 \times 0.25 = 2500 \text{ bits}
$$

### Extractable Length

Applying the Leftover Hash Lemma with security parameter $\varepsilon_{\text{sec}} = 10^{-10}$:

$$
\ell \leq 2500 - 2\log_2(10^{10}) + 2 = 2500 - 66.4 + 2 \approx 2436 \text{ bits}
$$

After subtracting syndrome leakage $|\Sigma| = n(1-R)$, the final key is:
$$
\ell_{\text{final}} = 2436 - |\Sigma| - |\text{hash}|
$$

---

## Protocol Integration

### Seed Distribution

The Toeplitz matrix is specified by a **seed** of $n + \ell - 1$ random bits. This seed must be:

1. **Cryptographically random**: Generated from `/dev/urandom` or equivalent
2. **Shared authentically**: Transmitted over the classical authenticated channel
3. **Public**: The seed is not secret; security relies only on the min-entropy bound

Alice generates the seed and transmits it to Bob after reconciliation. Both parties independently reconstruct the identical Toeplitz matrix.

### Deterministic Reconstruction

For reproducibility and debugging, a **pseudo-random seed** (32-256 bits) can expand to the full $n + \ell - 1$ bits using a cryptographic PRG. This reduces classical communication while maintaining 2-universality.

### Composable Security

Toeplitz hashing achieves **composable security** [3]: the output key can be used in subsequent cryptographic protocols (e.g., OT) without security degradation. The $\varepsilon_{\text{sec}}$-bound composes additively across protocol stages.

---

## Comparison with Alternative Hash Families

| Hash Family | Description | Seed Size | Computation |
|-------------|-------------|-----------|-------------|
| **Toeplitz** | Diagonal-constant matrices | $n + \ell - 1$ | $O(n \log n)$ with FFT |
| **Random linear** | Uniformly random matrices | $n \ell$ | $O(n \ell)$ |
| **Polynomial** | Evaluation of random polynomial | $O(\ell)$ | $O(n \log^2 n)$ |
| **Trevisan** | Combinatorial extractor | $O(\log^2 n)$ | $O(n \text{poly}(\log n))$ |

Toeplitz matrices offer the best balance of seed compactness, computational efficiency, and implementation simplicity for practical QKD and NSM applications.

---

## References

[1] M. Tomamichel, C. Schaffner, A. Smith, and R. Renner, "Leftover Hashing Against Quantum Side Information," *IEEE Trans. Inf. Theory*, vol. 57, no. 8, pp. 5524–5535, 2011.

[2] P. Beelen and T. Høholdt, "The decoding of algebraic geometry codes," in *Advances in Algebraic Geometry Codes*, pp. 49–98, 2008.

[3] R. Renner, "Security of Quantum Key Distribution," Ph.D. thesis, ETH Zürich, 2005.

---

[← Return to Main Index](../index.md) | [Next: Extractable Length →](./extractable_length.md)
