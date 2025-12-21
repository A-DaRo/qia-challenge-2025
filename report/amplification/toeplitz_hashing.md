[← Return to Main Index](../index.md)

# 7.1 Toeplitz Hashing

## Introduction

Privacy amplification is the final cryptographic transformation in the Caligo protocol, converting a partially secure key (about which an adversary has limited information) into a uniformly random key with information-theoretic security. This section examines the use of **Toeplitz matrices** as 2-universal hash functions—a construction that combines theoretical elegance, computational efficiency, and provable security guarantees.

The challenge addressed by privacy amplification is fundamental: after quantum key distribution and error correction, Alice and Bob share a reconciled bit string, but an adversary may possess correlated quantum information that leaks partial knowledge. Privacy amplification compresses this string to a shorter output where the adversary's residual information is exponentially suppressed.

## Theoretical Foundation

### 2-Universal Hashing

A class $\mathcal{F}$ of hash functions $f: \{0, 1\}^n \to \{0, 1\}^\ell$ is **2-universal** if for all distinct inputs $x \neq y \in \{0, 1\}^n$ and for a randomly selected $f \in \mathcal{F}$:

$$
\Pr[f(x) = f(y)] \leq 2^{-\ell}
$$

This collision-resistance property is critical for security: it bounds the probability that two different input strings map to the same output, preventing an adversary from exploiting hash collisions.

**Example**: The set of all affine transformations from $\{0, 1\}^n$ to $\{0, 1\}^\ell$ (linear maps plus constants) forms a 2-universal family [1].

### Leftover Hash Lemma

The security of privacy amplification rests on the **Leftover Hash Lemma** (LHL), which quantifies how hashing increases the privacy of a random variable against a quantum adversary. The quantum formulation by Tomamichel et al. [2] establishes:

Let $\rho_{XE}$ be a classical-quantum (cq) state where $X$ is a classical random variable and $E$ is a quantum system held by an adversary. Let $\mathcal{F}$ be a 2-universal hash function family. Then for a uniformly random $F \in \mathcal{F}$, the hashed output $F(X)$ satisfies:

$$
d(F(X) \mid F D \rho_E) \leq 2^{(\ell + k)/2} \cdot \frac{1}{\sqrt{P_g(X \mid \rho_E)}}
$$

where:
- $d(\cdot)$ is the **nonuniformity** (half the trace distance from uniform)
- $k$ is additional classical leakage (e.g., syndrome bits)
- $P_g(X \mid \rho_E)$ is the adversary's maximal guessing probability
- $\ell$ is the output length

**Intuition**: If the adversary's guessing probability is low (high min-entropy), then the hashed output is statistically close to uniform and independent of the adversary's quantum state, even when the hash function and leakage $D$ are revealed.

**Finite-Size Corrections**: Unlike asymptotic proofs, the LHL provides explicit finite-$n$ bounds, crucial for experimental implementations where key lengths are limited ($n \sim 10^3 - 10^6$).

## Toeplitz Matrix Construction

### Matrix Structure

A **Toeplitz matrix** $T \in \{0, 1\}^{m \times n}$ has constant entries along each diagonal:

$$
T_{i,j} = T_{i-1,j-1} \quad \text{for all valid } i, j
$$

This constraint means the matrix is fully specified by its first row $r \in \{0, 1\}^n$ and first column $c \in \{0, 1\}^m$:

$$
T = \begin{bmatrix}
c_1 & r_2 & r_3 & \cdots & r_n \\
c_2 & c_1 & r_2 & \cdots & r_{n-1} \\
c_3 & c_2 & c_1 & \cdots & r_{n-2} \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
c_m & c_{m-1} & c_{m-2} & \cdots & r_{n-m+1}
\end{bmatrix}
$$

**Compact Representation**: Only $n + m - 1$ random bits are required (the combined vector $[c_1, c_2, \ldots, c_m, r_2, r_3, \ldots, r_n]$), compared to $m \times n$ for a general matrix.

### Hashing Operation

The hash function maps an input $x \in \{0, 1\}^n$ to an output $h \in \{0, 1\}^m$ via matrix-vector multiplication over $\mathbb{F}_2$ (the binary field):

$$
h = T \cdot x \pmod{2}
$$

Each output bit $h_i$ is the parity (XOR) of input bits selected by the $i$-th row of $T$:

$$
h_i = \bigoplus_{j=1}^{n} T_{i,j} \cdot x_j \pmod{2}
$$

**2-Universality**: Toeplitz matrices form a 2-universal hash family. For any $x \neq y$, the probability that $T \cdot x = T \cdot y$ over uniformly random choice of $T$ is exactly $2^{-m}$ [3].

## Implementation in Caligo

### Computational Complexity

**Direct Multiplication**: Naïve matrix-vector multiplication requires $O(mn)$ bit operations, feasible for moderate key lengths but potentially expensive for $n \sim 10^6$.

**FFT Acceleration**: Toeplitz matrix multiplication can be embedded into **circulant matrix multiplication**, which is diagonalized by the Discrete Fourier Transform (DFT). Using the Fast Fourier Transform (FFT), this reduces complexity to $O(n \log n)$ [4].

**Algorithm Outline**:
1. Pad the Toeplitz matrix to a circulant matrix of size $2^k \geq n + m - 1$
2. Compute FFTs of the first column (defining the circulant) and the input vector
3. Perform element-wise multiplication in the frequency domain
4. Apply inverse FFT and extract the first $m$ bits

**Caligo Implementation**: The `ToeplitzHasher` class automatically selects FFT-based computation for $n > 64$, providing significant speedups for practical key lengths:

```python
class ToeplitzHasher:
    def __init__(self, input_length: int, output_length: int, 
                 use_fft: bool = True):
        self._n = input_length
        self._m = output_length
        self._use_fft = use_fft
        # Generate (n + m - 1) random bits
        self._random_bits = self._generate_random_bits(
            input_length + output_length - 1
        )
```

### Cryptographic Randomness

**Seed Generation**: The random bits defining the Toeplitz matrix are generated using `secrets.token_bytes()`, which provides cryptographically secure randomness from the operating system's entropy pool (e.g., `/dev/urandom` on Linux).

**Deterministic Replay**: A seed can be specified to enable deterministic hashing (critical for Alice and Bob to use the same hash function):

```python
hasher = ToeplitzHasher(
    input_length=1000, 
    output_length=512, 
    seed=shared_random_seed
)
```

The seed is typically generated by Alice, transmitted to Bob over the authenticated classical channel, and used to reconstruct the identical Toeplitz matrix.

## Security Analysis

### Min-Entropy Requirement

The Leftover Hash Lemma guarantees security provided the input has sufficient **min-entropy**:

$$
H_{\min}(X \mid E) \geq \ell + 2 \log_2(1/\varepsilon_{\text{sec}}) + k
$$

where:
- $H_{\min}(X \mid E)$ is the adversary-conditioned min-entropy
- $\ell$ is the desired output length
- $\varepsilon_{\text{sec}}$ is the security parameter (typically $10^{-10}$)
- $k$ is the syndrome leakage from error correction

**Example**: For $\varepsilon_{\text{sec}} = 10^{-10}$, the security penalty is $2 \log_2(10^{10}) \approx 66$ bits. If the input has 1000 bits with min-entropy rate 0.7 (700 bits available) and syndrome leakage of 120 bits, the extractable length is:

$$
\ell = 700 - 66 - 120 = 514 \text{ bits}
$$

### NSM-Specific Security

In the Noisy Storage Model, the adversary's min-entropy depends on the storage noise parameter $r$:

$$
H_{\min}(X_{\bar{C}} \mid \rho_E) \geq n \cdot h_{\min}(r)
$$

where $h_{\min}(r)$ is the **Max Bound** entropy rate (see [Section 7.2](./extractable_length.md)). For depolarizing storage noise with $r = 0.75$:

$$
h_{\min}(0.75) = \max\{\Gamma[1 - \log_2(1 + 3 \cdot 0.75^2)], 1 - 0.75\} = 0.25
$$

Thus, a 1000-bit reconciled key provides $\approx 250$ bits of extractable entropy after accounting for the adversary's quantum storage.

### Composable Security

Toeplitz hashing satisfies the **composable security** framework [5]: the output key can be safely used in subsequent cryptographic protocols (e.g., $\binom{2}{1}$-OT) without degrading security. The $\varepsilon_{\text{sec}}$-secrecy bound ensures the key is:

1. **Statistically uniform**: Within $\varepsilon_{\text{sec}}$ trace distance of the uniform distribution
2. **Independent of adversary**: The adversary's quantum state $\rho_E$ provides negligible advantage

This eliminates the need for further post-processing (e.g., AES encryption) and enables direct application to information-theoretic primitives.

## Practical Considerations

### Memory Efficiency

The Toeplitz matrix is never explicitly constructed in memory. Only the random bit vector of size $n + m - 1$ is stored, yielding:

- **Storage**: $O(n)$ bits (vs. $O(mn)$ for dense matrix)
- **Transmission**: The seed (typically 32-256 bytes) is transmitted, from which Alice and Bob independently reconstruct the matrix

### Numerical Stability

Binary operations (XOR) are exact—no floating-point rounding errors arise, unlike continuous-alphabet privacy amplification schemes (e.g., Gaussian modulation). This ensures bit-perfect reproducibility across platforms.

### Parameter Selection

The output length $\ell$ must be chosen conservatively:
- **Too large**: Insufficient compression, adversary retains significant information ($\varepsilon_{\text{sec}}$ blows up)
- **Too small**: Over-compression, wasted key material

Caligo uses the `SecureKeyLengthCalculator` to compute optimal $\ell$ from measured QBER, syndrome leakage, and NSM parameters, ensuring $\varepsilon_{\text{sec}} \leq 10^{-10}$ while maximizing efficiency.

## References

[1] Carter, L., & Wegman, M. N. (1979). Universal classes of hash functions. *Journal of Computer and System Sciences*, 18(2), 143-154.

[2] Tomamichel, M., Lim, C. C. W., Gisin, N., & Renner, R. (2012). Tight finite-key analysis for quantum cryptography. *Nature Communications*, 3(1), 634.

[3] Krawczyk, H. (1994). LFSR-based hashing and authentication. In *Advances in Cryptology—CRYPTO '94* (pp. 129-139). Springer.

[4] Shparlinski, I., & Winterhof, A. (2005). Fast multiplication of Toeplitz matrices. *Applicable Algebra in Engineering, Communication and Computing*, 16(1), 5-17.

[5] Renner, R. (2008). Security of Quantum Key Distribution. *International Journal of Quantum Information*, 6(1), 1-127.

---

[← Return to Main Index](../index.md) | [Next: Extractable Length Calculation](./extractable_length.md)
