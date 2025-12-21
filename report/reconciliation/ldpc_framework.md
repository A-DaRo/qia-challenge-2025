[← Return to Main Index](../index.md)

# 6.1 Rate-Compatible LDPC Framework

## Introduction

Phase III of the Caligo protocol implements information reconciliation—the critical stage where Alice and Bob eliminate discrepancies between their correlated strings using error correction codes. This phase is particularly demanding in the Noisy Storage Model (NSM) context, where syndrome information leaks directly to Bob (the potential adversary), rather than to an external eavesdropper as in traditional QKD protocols.

The reconciliation framework in Caligo is built on Low-Density Parity-Check (LDPC) codes with **rate-compatible puncturing and shortening**, enabling efficient operation across a wide range of Quantum Bit Error Rates (QBER) without requiring multiple pre-compiled matrices.

## Theoretical Foundation

### The Channel Model

Information reconciliation addresses the problem of **source coding with side information** [1]. Alice holds a string $\mathbf{x} \in \{0,1\}^m$ and Bob holds a correlated string $\mathbf{y} \in \{0,1\}^m$, where the correlation can be modeled as if $\mathbf{y}$ were received through a Binary Symmetric Channel (BSC) with crossover probability $p = \text{QBER}$.

The Slepian-Wolf theorem establishes that the minimum information Alice must disclose to enable Bob to recover $\mathbf{x}$ is:

$$
I_{\text{opt}} = H(X|Y) = n \cdot h(p)
$$

where $h(p) = -p\log_2(p) - (1-p)\log_2(1-p)$ is the binary entropy function.

### LDPC Codes as Reconciliation Tools

An LDPC code is defined by a sparse parity-check matrix $H \in \{0,1\}^{m \times n}$, where $m = (1-R_0) \cdot n$ check equations constrain $n$ codeword bits at rate $R_0 = k/n$. The code is represented as a bipartite **Tanner graph**:
- **Variable nodes** (symbol nodes): represent codeword bits
- **Check nodes**: represent parity-check constraints
- **Edges**: connect variables participating in each constraint

For information reconciliation via **syndrome coding** [2]:
1. Alice computes the syndrome $\mathbf{s} = H \cdot \mathbf{x} \mod 2$ and sends it to Bob
2. Bob uses his received string $\mathbf{y}$ as channel observations to decode
3. Bob runs belief-propagation (BP) decoding to find $\hat{\mathbf{x}}$ satisfying $H \cdot \hat{\mathbf{x}} = \mathbf{s}$

The information disclosed is:

$$
|\Sigma| = (1 - R_0) \cdot n \text{ bits (syndrome)} + h \text{ bits (verification hash)}
$$

### Reconciliation Efficiency

The **reconciliation efficiency** quantifies how close the actual disclosure is to the theoretical optimum:

$$
f = \frac{|\Sigma| + |H|}{n \cdot h(\text{QBER})} = \frac{(1-R_0) \cdot n + h}{n \cdot h(\text{QBER})}
$$

For perfect reconciliation, $f = 1$. Practical LDPC-based protocols achieve $f \in [1.05, 1.20]$ for QBER ranges typical in quantum protocols [3, 4].

## Rate Adaptation via Puncturing and Shortening

### The Core Challenge

A fixed-rate code optimized for QBER $= p_0$ performs poorly when the actual QBER deviates from $p_0$. Traditional solutions require maintaining a library of many pre-compiled codes, each optimized for a narrow QBER range [5].

**Rate-compatible coding** solves this by dynamically adapting a single **mother code** to different effective rates using two complementary techniques:

#### Puncturing

**Puncturing** increases the code rate by deleting $p$ symbols from the codeword, converting $\mathcal{C}(n, k) \to \mathcal{C}(n-p, k)$:

$$
R_{\text{punct}} = \frac{k}{n-p} = \frac{R_0}{1 - \pi}
$$

where $\pi = p/n$ is the puncturing fraction.

**Operational meaning:** Punctured positions are filled with pseudo-random padding (unknown to the decoder). The decoder initializes these positions with **zero LLR** (erasure), representing complete uncertainty.

#### Shortening

**Shortening** decreases the code rate by fixing the values of $s$ symbols at positions known to both parties, converting $\mathcal{C}(n, k) \to \mathcal{C}(n-s, k-s)$:

$$
R_{\text{short}} = \frac{k-s}{n-s} = \frac{R_0 - \sigma}{1 - \sigma}
$$

where $\sigma = s/n$ is the shortening fraction.

**Operational meaning:** Shortened positions are filled with pseudo-random values generated from a synchronized seed. The decoder initializes these positions with **infinite LLR** ($\pm \infty$), representing perfect knowledge.

### Combined Modulation

The **inverse puncturing and shortening** protocol [1] applies both techniques to a frame of fixed length $n$:

1. **Payload positions** ($m = n - d$): filled with the correlated string
2. **Punctured positions** ($p$): filled with random padding (LLR = 0)
3. **Shortened positions** ($s$): filled with known values (LLR = $\pm\infty$)

where $d = p + s$ is the **modulation parameter**.

The **effective rate** becomes:

$$
R_{\text{eff}} = \frac{k - s}{n - p - s} = \frac{R_0 - \sigma}{1 - \pi - \sigma}
$$

The achievable rate range for modulation fraction $\delta = d/n$ is:

$$
R_{\min} = \frac{R_0 - \delta}{1 - \delta} \leq R_{\text{eff}} \leq \frac{R_0}{1 - \delta} = R_{\max}
$$

### Frame Construction Function

The frame construction function $g(\mathbf{x}, \sigma, \pi)$ deterministically maps a payload string $\mathbf{x}$ to a codeword frame $\mathbf{x}^+$ of length $n$:

$$
\mathbf{x}^+ = g(\mathbf{x}, \sigma, \pi) = 
\begin{cases}
x_{\text{payload}[i]} & \text{if } i \in \text{PayloadSet} \\
\text{PRNG}(i) & \text{if } i \in \text{PunctureSet} \\
\text{PRNG}_{\text{known}}(i) & \text{if } i \in \text{ShortenSet}
\end{cases}
$$

**Critical requirement:** The position sets and shortened values must be **reproducible** from a synchronized pseudo-random generator (PRG) to avoid additional communication overhead [1].

## Leakage Accounting in the NSM Context

### NSM Security Constraint

Under the Noisy Storage Model, the extractable secure key length is bounded by:

$$
\ell \leq n \cdot \left[ H_{\min}^{\epsilon}(X|E) - \text{leak}_{\text{EC}} - \log_2\left(\frac{2}{\epsilon^2}\right) \right]
$$

The reconciliation leakage $\text{leak}_{\text{EC}}$ directly reduces the extractable output. Therefore, **minimizing syndrome length while maintaining decoding reliability** is paramount.

### Leakage Components

For syndrome-based reconciliation:

$$
\text{leak}_{\text{EC}} = |\Sigma| + |H| + |\text{Revealed}|
$$

where:
- $|\Sigma| = (1-R_0) \cdot n$: syndrome bits (fixed by mother matrix)
- $|H|$: verification hash bits (typically 32-128)
- $|\text{Revealed}|$: additional revealed bits (blind protocol only)

**Key observation:** The syndrome length is **independent of the effective rate** because it is computed with the fixed mother matrix. Rate adaptation via puncturing/shortening modifies decoding behavior without changing syndrome size.

### Circuit Breaker Pattern

Caligo implements a **circuit breaker** [6] that immediately aborts reconciliation if the cumulative leakage exceeds the NSM-derived safety cap:

```python
if cumulative_leakage > safety_cap:
    raise LeakageBudgetExceeded(
        f"Leakage {cumulative_leakage} exceeds cap {safety_cap}"
    )
```

This prevents security violations from propagating through the protocol pipeline.

## Progressive Edge-Growth (PEG) Mother Code Construction

### Graph Construction Algorithm

Caligo uses the PEG algorithm [7] to construct mother codes with maximized local girth (shortest cycle length). The algorithm builds the Tanner graph edge-by-edge, placing each edge to maximize the distance to existing cycles around the current variable node.

**Key properties:**
1. **Girth maximization**: Longer cycles improve BP convergence
2. **Irregular degree distributions**: Optimized $\lambda(x)$ and $\rho(x)$ polynomials for specific channels
3. **Edge-perspective design**: Degrees specified from edge viewpoint for density evolution analysis

### Mother Code Parameters

For Caligo Phase III:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Frame size** $n$ | 4096 | Balance between performance and hardware feasibility [2] |
| **Mother rate** $R_0$ | 0.50 | Symmetric starting point for wide rate coverage |
| **Variable degree dist.** | Irregular (PEG-optimized) | Threshold optimization for BSC |
| **Check degree dist.** | Irregular (PEG-optimized) | Balanced connectivity |
| **Girth** | $\geq 6$ | Avoid 4-cycles that trap BP messages |

### Compilation and Storage

Mother matrices are:
1. Generated offline using PEG with optimized degree distributions
2. Stored in compressed sparse row (CSR) format (`.npz` files)
3. Compiled to adjacency-list representation for fast syndrome computation
4. Checksum-verified for Alice-Bob synchronization

```python
# Matrix loading with checksum verification
matrix = sp.load_npz(matrix_path).tocsr().astype(np.uint8)
checksum = hashlib.sha256(matrix.data.tobytes()).hexdigest()
```

## Belief-Propagation Decoding

### Sum-Product Algorithm

Bob decodes using iterative belief propagation in the log-likelihood ratio (LLR) domain. Messages are exchanged between variable and check nodes:

**Variable-to-check messages:**

$$
\mu_{v \to c}^{(t)} = \lambda_v + \sum_{c' \in \mathcal{N}(v) \setminus \{c\}} \mu_{c' \to v}^{(t-1)}
$$

where $\lambda_v$ is the channel LLR.

**Check-to-variable messages** (tanh-domain update):

$$
\mu_{c \to v}^{(t)} = 2 \cdot \text{arctanh}\left( \prod_{v' \in \mathcal{N}(c) \setminus \{v\}} \tanh\left(\frac{\mu_{v' \to c}^{(t)}}{2}\right) \right)
$$

**Hard decision:**

$$
\hat{x}_v = \begin{cases}
0 & \text{if } \lambda_v + \sum_{c \in \mathcal{N}(v)} \mu_{c \to v}^{(t)} \geq 0 \\
1 & \text{otherwise}
\end{cases}
$$

### Convergence Criteria

Decoding terminates when either:
1. **Success:** $H \cdot \hat{\mathbf{x}} = \mathbf{s}$ (syndrome match)
2. **Iteration limit:** Maximum iterations reached without convergence
3. **Stagnation:** Messages stable but syndrome mismatch persists

### Three-State LLR Initialization

The decoder initializes LLRs based on position type:

| Position Type | Initial LLR | Interpretation |
|---------------|-------------|----------------|
| **Payload** | $\alpha \cdot (1 - 2y_i)$ where $\alpha = \ln\frac{1-p}{p}$ | Soft channel information |
| **Punctured** | $0$ | Complete erasure (unknown) |
| **Shortened** | $\pm 100$ (saturated) | Known value (certain) |

This **three-state initialization** is critical for rate-compatible decoding [1, 2].

## Implementation Architecture

### Module Hierarchy

```
caligo.reconciliation/
├── matrix_manager.py         # LDPC matrix loading and caching
├── ldpc_encoder.py           # Syndrome computation (Alice)
├── ldpc_decoder.py           # Belief-propagation (Bob)
├── compiled_matrix.py        # Fast adjacency-list representation
├── rate_selector.py          # (p,s) selection from QBER
├── hash_verifier.py          # ε-universal hash verification
├── leakage_tracker.py        # NSM leakage accounting
└── orchestrator.py           # Phase III coordinator
```

### Key Abstractions

**`MatrixManager`**: Singleton providing thread-safe matrix access with checksum verification

**`CompiledParityCheckMatrix`**: Pre-processed adjacency lists for $O(|\text{edges}|)$ syndrome computation

**`BeliefPropagationDecoder`**: Optimized BP kernel with message buffer reuse

**`LeakageTracker`**: Circuit-breaker pattern enforcing NSM safety cap

## Performance Considerations

### Computational Complexity

- **Syndrome computation** (Alice): $O(|\text{edges}|) \approx O(n \cdot \bar{d}_v)$ where $\bar{d}_v$ is average variable degree
- **BP decoding** (Bob): $O(I \cdot |\text{edges}|)$ where $I$ is iteration count (typically $I \leq 50$)
- **Frame construction**: $O(n)$ (linear scan)

### Memory Footprint

- **Mother matrix** (CSR): $\approx 3 \times |\text{edges}| \times 4$ bytes (indices + data)
- **Compiled adjacency**: $\approx 2 \times |\text{edges}| \times 4$ bytes
- **Decoder buffers**: $2 \times |\text{edges}| \times 8$ bytes (float64 messages)
- **Total** (per mother code): $\approx 100$-$200$ KB for $n=4096$

## References

[1] D. Elkouss, J. Martinez-Mateo, D. Lancho, and V. Martin, "Rate Compatible Protocol for Information Reconciliation: An Application to QKD," 2010.

[2] J. Martinez-Mateo, D. Elkouss, and V. Martin, "Blind Reconciliation," *Quantum Information and Computation*, Vol. 12, No. 9&10, pp. 791-812, 2012.

[3] D. Elkouss, A. Leverrier, R. Alléaume, and J. J. Boutros, "Efficient reconciliation protocol for discrete-variable quantum key distribution," *arXiv:0901.2140*, 2009.

[4] E. Kiktenko et al., "Post-processing procedure for industrial quantum key distribution systems," *J. Phys.: Conf. Ser.* 741, 012081, 2016.

[5] W. T. Buttler et al., "Winnow reconciliation protocol for quantum key distribution," *Applied Optics*, 2003.

[6] M. Nygard, "Release It! Design and Deploy Production-Ready Software," 2nd ed., 2018.

[7] X. Y. Hu, E. Eleftheriou, and D. M. Arnold, "Regular and irregular progressive edge-growth tanner graphs," *IEEE Trans. Info. Theory*, Vol. 51, No. 1, pp. 386-398, 2005.

---

[← Return to Main Index](../index.md) | [Next: Baseline Strategy →](./baseline_strategy.md)
