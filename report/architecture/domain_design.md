[← Return to Main Index](../index.md)

# 3.3 Implementation Architecture

## 3.3.1 Modular Structure

The Caligo implementation separates concerns into distinct modules corresponding to protocol phases and cross-cutting functionality.

### Module Organization

| Module | Responsibility | Key Functions |
|--------|----------------|---------------|
| `quantum/` | EPR generation, measurement | BB84 state preparation |
| `sifting/` | Basis comparison, QBER estimation | Index partitioning |
| `reconciliation/` | LDPC encoding/decoding | Syndrome computation |
| `privacy/` | Privacy amplification | Toeplitz hashing |
| `core/` | Protocol orchestration | Phase sequencing |
| `utils/` | Shared utilities | Bit operations, logging |

### Data Flow

```
┌─────────────┐    ┌─────────────┐    ┌───────────────────┐    ┌─────────────┐
│   quantum/  │───▶│  sifting/   │───▶│  reconciliation/  │───▶│  privacy/   │
│             │    │             │    │                   │    │             │
│ (X, Θ)      │    │ (K_A, K_B)  │    │ K_rec             │    │ (S₀,S₁,S_C) │
└─────────────┘    └─────────────┘    └───────────────────┘    └─────────────┘
```

## 3.3.2 Protocol Execution Model

### Generator-Based Execution

The protocol executes as coroutines, enabling interleaved operation between Alice and Bob:

**Alice's Protocol:**
$$
\mathcal{A}: \text{State}_A \times \text{Message} \to \text{State}_A \times \text{Message}
$$

**Bob's Protocol:**
$$
\mathcal{B}: \text{State}_B \times \text{Message} \to \text{State}_B \times \text{Message}
$$

**Execution:** The simulator interleaves:
1. Alice generates message $m_1$
2. Bob processes $m_1$, generates $m_2$
3. Alice processes $m_2$, generates $m_3$
4. ...continue until termination

### Phase Boundaries

Each phase produces well-defined outputs consumed by the next phase:

**Phase I → II:** Raw measurements $(X, \Theta)$ of length $n_{\text{raw}}$

**Phase II → III:** Sifted keys $(K_A, K_B)$ of length $n_{\text{sifted}}$, QBER estimate $\hat{Q}$

**Phase III → IV:** Reconciled key $K_{\text{rec}}$ of length $n$, leakage $|\Sigma|$

**Phase IV → Output:** Oblivious keys $(S_0, S_1)$ for Alice, $S_C$ for Bob

## 3.3.3 Quantum Layer Abstraction

### EPR Interface

The quantum layer abstracts over the simulation framework:

**Input:** Number of pairs $n$, generation mode (sequential/parallel)

**Output:** Measurement outcomes $X \in \{0,1\}^n$, basis choices $\Theta \in \{+,\times\}^n$

**Noise Model:** Werner state fidelity $F$ maps to QBER:
$$
Q = \frac{1-F}{2}
$$

### Timing Enforcement

The implementation enforces the NSM timing constraint:

1. **Mark:** Record timestamp when quantum phase completes
2. **Wait:** Block classical communication until $\Delta t$ elapses
3. **Release:** Allow basis revelation after delay

## 3.3.4 Reconciliation Engine

### LDPC Code Interface

**Input:** Block $K \in \{0,1\}^n$, code rate $R$

**Output:** Syndrome $\Sigma = H \cdot K \in \{0,1\}^{(1-R)n}$

### Decoding Interface

**Input:** Noisy block $K_B$, syndrome $\Sigma$, channel LLR

**Output:** Decoded block $\hat{K}_A$, success flag

**Algorithm:** Sum-product belief propagation:
$$
L_i^{(t+1)} = L_i^{(0)} + \sum_{j \in \mathcal{N}(i)} \phi\left(\sum_{k \in \mathcal{N}(j) \setminus i} \phi(L_k^{(t)})\right) \cdot \prod_{k \in \mathcal{N}(j) \setminus i} \text{sign}(L_k^{(t)})
$$

where $\phi(x) = -\log\tanh(|x|/2)$ is the boxplus kernel.

### Rate Adaptation

The blind reconciliation protocol adapts to unknown QBER:

1. **Initial:** Assume optimistic QBER, transmit syndrome
2. **Decode:** Attempt belief propagation
3. **Adapt:** If failure, reveal additional bits (puncturing)
4. **Iterate:** Until success or maximum iterations

## 3.3.5 Privacy Amplification Engine

### Toeplitz Matrix Generation

**Input:** Seed $s \in \{0,1\}^{n+\ell-1}$

**Output:** Matrix $T \in \{0,1\}^{\ell \times n}$ with constant diagonals

**Property:** $T_{ij} = s_{n+i-j}$ for $i \in [0,\ell)$, $j \in [0,n)$

### Hashing Operation

**Input:** Key $K \in \{0,1\}^n$, matrix $T$

**Output:** Compressed key $S = T \cdot K \in \{0,1\}^\ell$

**Complexity:** $O(n \cdot \ell)$ via direct multiplication; $O(n \log n)$ via FFT for large blocks.

### Key Length Calculation

The implementation computes:
$$
\ell = \max\left\{0, \left\lfloor n \cdot h_{\min}(r) - |\Sigma| - 2\log_2(1/\varepsilon_{\text{sec}}) + 2 \right\rfloor\right\}
$$

where $h_{\min}(r)$ is computed from the Dupuis-König or Lupo bound (whichever is tighter).

## 3.3.6 Configuration Parameters

### Simulation Parameters

| Parameter | Type | Default | Range |
|-----------|------|---------|-------|
| `n_epr_pairs` | int | 1000 | $[100, 10^6]$ |
| `epr_fidelity` | float | 0.95 | $[0.5, 1.0]$ |
| `generation_mode` | enum | parallel | sequential, parallel |

### Security Parameters

| Parameter | Type | Default | Range |
|-----------|------|---------|-------|
| `epsilon_sec` | float | $10^{-10}$ | $[10^{-15}, 10^{-6}]$ |
| `qber_threshold` | float | 0.11 | $[0.05, 0.22]$ |
| `storage_noise_r` | float | 0.3 | $[0.1, 0.7]$ |

### Reconciliation Parameters

| Parameter | Type | Default | Range |
|-----------|------|---------|-------|
| `ldpc_rate` | float | 0.5 | $[0.1, 0.9]$ |
| `max_bp_iterations` | int | 100 | $[10, 1000]$ |
| `hash_length` | int | 64 | $[32, 256]$ |

## 3.3.7 Error Handling

### Protocol Aborts

The protocol aborts under the following conditions:

| Condition | Threshold | Action |
|-----------|-----------|--------|
| QBER > hard limit | $Q > 0.22$ | Immediate abort |
| Reconciliation failure | After max retries | Abort with diagnostic |
| Negative key length | $\ell \leq 0$ | Abort (Death Valley) |

### Diagnostic Information

On abort, the implementation provides:
- Estimated QBER with confidence interval
- Reconciliation attempt history
- Min-entropy calculation breakdown
- Suggested parameter adjustments

## 3.3.8 Validation Properties

The implementation maintains invariants throughout execution:

**Length Conservation:**
$$
n_{\text{sifted}} \leq n_{\text{raw}}, \quad \ell \leq n_{\text{rec}}
$$

**Security Consistency:**
$$
\ell > 0 \implies Q < Q_{\max} \land |\Sigma| < n \cdot h_{\min}(r)
$$

**Completeness:**
$$
K_A^{\text{(rec)}} = K_B^{\text{(rec)}} \text{ (after successful reconciliation)}
$$

---

[← Return to Main Index](../index.md) | [Next: Quantum Layer →](../quantum/epr_generation.md)
