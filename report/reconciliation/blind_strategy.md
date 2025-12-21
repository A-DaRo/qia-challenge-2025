[← Return to Main Index](../index.md)

# 6.3 Blind Reconciliation Strategy

## Introduction

The **Blind reconciliation strategy** implements the Martinez-Mateo et al. (2012) protocol [1], which eliminates *a priori* QBER estimation by adaptively revealing previously punctured symbols until decoding succeeds. This approach minimizes unnecessary information disclosure when the channel is better than expected, while gracefully degrading to lower rates when errors are high.

The protocol is called "blind" because it discovers the effective rate iteratively without requiring explicit channel parameter estimation. This has profound implications for NSM security, as it avoids the sampling overhead inherent in Baseline reconciliation.

## Theoretical Foundation

### The Information-Theoretic Argument

**Theorem (Syndrome Reuse, Martinez-Mateo et al. [1]):** In the Blind protocol, the syndrome $\mathbf{s} = H \cdot \mathbf{x}^+$ is transmitted **once**. Subsequent iterations reveal only the values of previously punctured symbols. The total information leakage is:

$$
\text{leak}_{\text{Blind}} = |\mathbf{s}| + |h| + \sum_{i=2}^{t} \Delta_i
$$

where $\Delta_i$ is the number of symbols revealed (converted from punctured to shortened) in iteration $i$.

**Proof Sketch:**

Consider the frame partition $\mathbf{x}^+ = [\mathbf{x}_{\text{payload}}, \mathbf{x}_{\text{punct}}, \mathbf{x}_{\text{short}}]$. The syndrome satisfies:

$$
\mathbf{s} = H_{\text{payload}} \cdot \mathbf{x}_{\text{payload}} + H_{\text{punct}} \cdot \mathbf{x}_{\text{punct}} + H_{\text{short}} \cdot \mathbf{x}_{\text{short}} \mod 2
$$

After revealing shortened values, Bob knows:
- Payload observations: $\mathbf{y}_{\text{payload}}$ (from quantum channel)
- Revealed shortened values: $\mathbf{x}_{\text{short}}^{(i)}$
- Syndrome: $\mathbf{s}$

The key insight: **punctured symbols are never observed by Bob** (they are internal padding), so revealing their values is the **only** way to reduce uncertainty about them. Each revealed symbol provides exactly 1 bit of information. ∎

### Iteration Monotonicity

**Lemma (Information Monotonicity):** Converting a punctured bit to a shortened bit provides **maximal information** to the decoder.

| Transition | LLR Change | Information Gain |
|------------|------------|------------------|
| Punctured → Payload | $0 \to \alpha$ | None (no channel observation) |
| Punctured → Shortened | $0 \to \pm\infty$ | 1 bit (known value) |
| Payload → Shortened | $\alpha \to \pm\infty$ | Resolves uncertainty |

Therefore, iteratively shortening punctured symbols **strictly increases** decoder knowledge without redundant communication.

### The Revelation Order Security Requirement

**Critical Design Constraint:** The order in which punctured symbols are revealed must be **independent of Bob's quantum side information** and **fixed at protocol setup**.

**Rationale:** If the revelation order were adaptive (chosen based on decoder feedback), it could leak side-channel information beyond the explicit bit values. For NSM security, we must ensure that:

$$
I(X_{1-c}; \text{RevelationOrder}) = 0
$$

where $c$ is Bob's choice bit in the OT protocol and $X_{1-c}$ is the input Bob should not learn.

**Implementation:** Caligo uses a **pre-computed deterministic revelation order** derived from the hybrid puncturing pattern:

```python
revelation_order = hybrid_puncturing_pattern  # Fixed at setup
revealed_indices = revelation_order[iteration * delta : (iteration+1) * delta]
```

This makes the pattern public and reproducible, with adaptivity limited only to the **continuation decision** (whether to proceed to the next iteration), which is a public Boolean dependent only on syndrome matching.

## Protocol Flow

### Setup Phase

**Parameters:**
- Mother code: $\mathcal{C}(n, k)$ with rate $R_0 = 0.5$
- Frame size: $n = 4096$
- Modulation parameter: $d = \lfloor \delta \cdot n \rfloor$ where $\delta \in [0.1, 0.5]$
- Maximum iterations: $t$ (typically 3)
- Revelation step size: $\Delta = \lceil d / t \rceil$

**Modulation Range:**

For $\delta = 0.44$ and $R_0 = 0.5$:

$$
R_{\min} = \frac{0.5 - 0.44}{1 - 0.44} = 0.107, \quad R_{\max} = \frac{0.5}{1 - 0.44} = 0.893
$$

This provides wide coverage from high-QBER ($\approx 0.45$) to low-QBER ($\approx 0.05$) regimes.

### Iteration 1: Optimistic Attempt

**Alice's Actions:**

1. **Initial configuration:**
   - Punctured: $p_1 = d$ (all modulation symbols punctured)
   - Shortened: $s_1 = 0$ (no shortening initially)
   - **Effective rate:** $R_{\text{eff}} = R_{\max} = 0.893$ (optimistic)

2. **Frame construction:**

$$
x^+_i = \begin{cases}
x_j & \text{if } i \in \mathcal{I}_{\text{payload}} \\
\text{PRNG}(\text{seed}, i) & \text{if } i \in \mathcal{I}_{\text{modulation}}
\end{cases}
$$

All $d$ modulation positions are filled with pseudo-random padding (unknown to decoder).

3. **Syndrome computation:**

$$
\mathbf{s} = H_{\text{mother}} \cdot \mathbf{x}^+ \mod 2
$$

4. **Verification hash:**

$$
h = \text{PolyHash}(\mathbf{x}_{\text{payload}}, \text{block\_id})
$$

5. **Transmission:**

```python
alice_message = {
    "type": "blind_init",
    "syndrome": s,        # 2048 bits
    "hash": h,            # 64 bits
    "delta": delta,       # Modulation fraction
    "max_iterations": t,  # Iteration budget
}
```

**Bob's Actions:**

1. **Frame construction:** Reproduce $\mathbf{y}^+$ with same pseudo-random padding

2. **LLR initialization:**

$$
\lambda_i = \begin{cases}
\ln\frac{1-p_{\text{est}}}{p_{\text{est}}} \cdot (1 - 2y_i) & \text{if } i \in \mathcal{I}_{\text{payload}} \\
0 & \text{if } i \in \mathcal{I}_{\text{modulation}}
\end{cases}
$$

where $p_{\text{est}}$ is an initial LLR magnitude heuristic (often $0.05$ for low-noise channels).

3. **Belief propagation:** Run BP decoder with target syndrome $\mathbf{s}$

4. **Check convergence:**

$$
\text{Success}_1 = (H \cdot \hat{\mathbf{x}}^{(1)} = \mathbf{s}) \land (\text{hash}(\hat{\mathbf{x}}) = h)
$$

If $\text{Success}_1 = \text{True}$: **Protocol terminates successfully** with zero additional leakage.

### Iteration $i \geq 2$: Adaptive Revelation

**Trigger:** Bob's decoder failed to converge in iteration $i-1$.

**Alice's Actions:**

1. **Select revelation batch:**

$$
\mathcal{S}_i = \{\text{revelation\_order}[(i-2)\Delta + 1 : (i-1)\Delta]\}
$$

This selects the next $\Delta$ positions from the pre-determined hybrid puncturing pattern.

2. **Reveal values:**

$$
\text{RevealedBits}_i = \{(j, x^+_j) : j \in \mathcal{S}_i\}
$$

3. **Update configuration:**
   - $p \leftarrow p - \Delta$ (fewer punctured)
   - $s \leftarrow s + \Delta$ (more shortened)
   - **Effective rate:** $R_{\text{eff}} = \frac{k - s}{n - p - s}$ (lower rate)

4. **Transmission:**

```python
alice_message_i = {
    "type": "blind_reveal",
    "revealed_bits": RevealedBits_i,  # Δ bits
    "iteration": i,
}
```

**Bob's Actions:**

1. **Update LLRs** (hot-start from previous iteration):

$$
\lambda_j \leftarrow \begin{cases}
+100 & \text{if } j \in \mathcal{S}_i \text{ and } x^+_j = 0 \\
-100 & \text{if } j \in \mathcal{S}_i \text{ and } x^+_j = 1 \\
\lambda_j^{(i-1)} & \text{otherwise (preserve messages)}
\end{cases}
$$

The **hot-start optimization** reuses message buffers from iteration $i-1$, avoiding re-initialization. This is critical for computational efficiency [1].

2. **Resume BP decoding:** Continue from previous state with updated LLRs

3. **Check convergence:** Test syndrome match and hash verification

### Termination Conditions

**Success:** Decoder converges and passes verification at iteration $i$

$$
\text{leak}_{\text{total}} = |\mathbf{s}| + |h| + \sum_{j=2}^{i} \Delta_j
$$

**Failure:** All $d$ modulation symbols revealed ($p = 0$, $s = d$) but decoder still fails

$$
\text{leak}_{\text{total}} = |\mathbf{s}| + |h| + d
$$

In the failure case, the protocol **aborts** and flags an error (QBER likely exceeded reconciliation threshold).

## NSM-Gated Variant: Heuristic QBER for Optimization

While the canonical Blind protocol [1] requires no QBER estimation, Caligo implements an **NSM-gated variant** that uses an optional heuristic QBER estimate (derived from trusted simulation parameters via `compute_qber_erven`) for two conservative optimizations:

### Optimization 1: Permissive Starting-Rate Cap

**Problem:** Starting at the absolute maximum rate ($p = d, s = 0$) is wasteful when the channel is clearly noisy.

**Solution:** Allow a small initial shortening $s_1 > 0$ to bring the first-iteration rate below a conservative cap:

$$
R^{(1)}_{\text{cap}} = 1 - f(\widehat{\text{QBER}}_{\text{FK}}) \cdot h(\widehat{\text{QBER}}_{\text{FK}})
$$

where $\widehat{\text{QBER}}_{\text{FK}}$ is a conservative estimate including finite-size correction.

**Implementation:**

```python
if qber_heuristic is not None:
    qber_fk = qber_heuristic + confidence_radius
    r_cap = 1 - efficiency_model(qber_fk) * binary_entropy(qber_fk)
    s_1 = max(0, int((R0 - r_cap * (1 - delta)) * n))
    p_1 = d - s_1
else:
    s_1, p_1 = 0, d  # Canonical blind protocol
```

**Leakage:** The $s_1$ values are sent alongside the syndrome in the first message, so:

$$
\text{leak}^{(1)} = |\mathbf{s}| + |h| + s_1
$$

This is still a **one-round protocol** (no extra interaction), but with a small upfront leakage to avoid a predictable first-iteration failure.

### Optimization 2: Restrictive Iteration Budget

**Problem:** Fine-grained revelation ($\Delta$ small, $t$ large) improves average efficiency but increases worst-case leakage and interaction complexity.

**Solution:** Use $t = 3$ by default (coarse steps, low overhead), and allow $t > 3$ only when $\widehat{\text{QBER}}_{\text{FK}}$ is near the protocol threshold (where finer adaptation materially improves success probability).

**Rationale:** This balances efficiency (small $\Delta$ helps when QBER is marginal) against complexity (large $t$ is overkill for low-QBER channels).

## Average Efficiency Analysis

### Expected Leakage

The average leakage of Blind reconciliation depends on the distribution of decoding successes across iterations.

Define:
- $F^{(i)}$: Frame Error Rate when using adapted rate at iteration $i$
- $a_i = \frac{F^{(i-1)} - F^{(i)}}{1 - F^{(t)}}$: Fraction of codewords corrected at iteration $i$

The average revealed bits are:

$$
\mathbb{E}[|\text{Revealed}|] = \sum_{i=2}^{t} a_i \cdot \sum_{j=2}^{i} \Delta
$$

**Example:** For $t = 3$, $\Delta = d/3$, and $F^{(1)} = 0.6$, $F^{(2)} = 0.1$, $F^{(3)} = 0.01$:

$$
a_1 = \frac{1 - 0.6}{1 - 0.01} = 0.404, \quad a_2 = \frac{0.6 - 0.1}{0.99} = 0.505, \quad a_3 = \frac{0.1 - 0.01}{0.99} = 0.091
$$

$$
\mathbb{E}[|\text{Revealed}|] = a_1 \cdot 0 + a_2 \cdot \frac{d}{3} + a_3 \cdot \frac{2d}{3} = 0.505 \cdot \frac{d}{3} + 0.091 \cdot \frac{2d}{3} \approx 0.229d
$$

For $d = 1800$ (corresponding to $\delta = 0.44$), this yields $\approx 412$ revealed bits, compared to the worst-case $1800$ bits.

### Efficiency Calculation

The average efficiency is:

$$
\bar{f}_{\text{Blind}} = \frac{|\mathbf{s}| + |h| + \mathbb{E}[|\text{Revealed}|]}{n \cdot h(\text{QBER})}
$$

For $n = 4096$, $R_0 = 0.5$, $h = 64$, QBER $= 0.08$:

$$
\bar{f}_{\text{Blind}} = \frac{2048 + 64 + 412}{4096 \times 0.408} \approx 1.51
$$

This is competitive with Baseline when accounting for the elimination of sampling overhead.

### Comparison with Fully Granular Blind ($\Delta = 1$)

Martinez-Mateo et al. [1] derive the theoretical limit for $\Delta \to 1$:

$$
\bar{R}_{\text{fine}} \approx R_{\max} - \frac{\beta}{t-1} \sum_{i=1}^{t-1} F^{(i)}
$$

where $\beta = \delta / (1 - \delta)$.

**Key finding:** Using $t = 3$ captures $\approx 80\%$ of the efficiency gain of fully granular revelation ($\Delta = 1$, $t = d$), while drastically reducing implementation complexity [1].

## Implementation Details

### Hot-Start Decoder State

The Blind protocol requires **persistent decoder state** across iterations. This is implemented using message buffer preservation:

```python
class BlindDecoder:
    def __init__(self):
        self.r_messages = None  # Check-to-variable messages
        self.q_messages = None  # Variable-to-check messages
    
    def decode_iteration(self, llr_updated, syndrome, iteration):
        if iteration == 1:
            # Cold start: initialize from channel LLRs
            self.r_messages = np.zeros(self.edge_count)
            self.q_messages = llr_updated.copy()
        else:
            # Hot start: update only revealed positions
            for idx in revealed_indices:
                self.q_messages[idx] = llr_updated[idx]
        
        # Continue BP from current state
        for bp_iter in range(self.max_iterations):
            self._update_check_to_var()
            self._update_var_to_check()
            
            if self._check_syndrome(syndrome):
                return DecodeResult(success=True, iterations=bp_iter)
        
        return DecodeResult(success=False, iterations=self.max_iterations)
```

### Freeze Optimization

**Concept:** Shortened symbols (LLR $= \pm\infty$) produce deterministic messages. Their contributions can be **precomputed** and excluded from iterative updates [1].

**Implementation:**

```python
def _update_check_to_var_frozen(self, frozen_mask):
    """Update check messages, skipping frozen (shortened) symbols."""
    for edge_idx in range(self.edge_count):
        if not frozen_mask[edge_idx]:
            # Standard tanh-domain update
            self.r_messages[edge_idx] = self._tanh_product(edge_idx)
        # else: message is deterministic, skip computation
```

This reduces computational complexity by $\approx s/n$ per iteration (5-20% savings for typical $s$ values).

## Failure Modes and Recovery

### Failure Mode 1: Early Exhaustion

**Symptom:** Decoder fails at iteration $t$ with all symbols shortened ($p = 0$)

**Cause:** Actual QBER exceeds the rate range covered by $\delta$

**Handling:**
1. Abort reconciliation for this block
2. Log failure with diagnostic info (final FER, QBER estimate)
3. Increment failure counter (may trigger protocol-level abort if threshold exceeded)

### Failure Mode 2: Verification Hash Mismatch

**Symptom:** Decoder converges (syndrome match) but hash verification fails

**Cause:** BP trapped in incorrect fixed point (rare with well-designed codes and sufficient iterations)

**Handling:**
1. Continue to next revelation iteration (do not terminate)
2. If persists after all revelations: abort with decoder error flag

### Failure Mode 3: Revelation Order Corruption

**Symptom:** Alice and Bob reveal different bit values for same indices

**Cause:** PRG seed desynchronization or implementation bug

**Detection:**
- Alice logs hash of revelation order at setup
- Bob compares against expected pattern
- Mismatch triggers immediate abort

**Prevention:**
- Explicit seed exchange and verification during protocol initialization
- Deterministic PRG (no OS entropy sources during reconciliation)

## Performance Characteristics

### Communication Rounds

| Scenario | Rounds | Bandwidth |
|----------|--------|-----------|
| **Success (iteration 1)** | 1 | $2048 + 64 = 2112$ bits |
| **Success (iteration 2)** | 2 | $2112 + \Delta$ bits |
| **Success (iteration 3)** | 3 | $2112 + 2\Delta$ bits |
| **Failure (all iterations)** | $t$ | $2112 + (t-1)\Delta$ bits |

For $\Delta = 600$, worst-case is $\approx 3312$ bits, compared to Baseline's $\approx 2212$ bits (including sampling overhead).

### Computational Complexity

**Iteration 1:**
- Alice: Syndrome computation ($O(|\text{edges}|)$)
- Bob: BP decoding ($O(I \cdot |\text{edges}|)$)

**Iteration $i > 1$:**
- Alice: Reveal extraction ($O(\Delta)$)
- Bob: Hot-start BP ($O(I \cdot |\text{edges}|)$ but with faster convergence)

**Total (worst-case):** $\approx t \times 100$ ms = $300$ ms for $t = 3$

### Memory Footprint

**Additional overhead over Baseline:**
- Message buffers: $2 \times |\text{edges}| \times 8$ bytes $\approx 100$ KB
- Revelation history: $d \times 9$ bytes (index + value) $\approx 18$ KB
- **Total:** $\approx 120$ KB extra

## When to Use Blind

**Ideal scenarios:**
1. QBER is unknown or highly variable
2. Minimize worst-case leakage (avoid sampling overhead)
3. NSM security critical (no explicit sampling disclosure)
4. Low-latency communication (1-3 rounds acceptable)

**Avoid when:**
1. QBER is stable and well-characterized (Baseline more efficient)
2. Strict one-round requirement (use Baseline)
3. Hot-start decoder complexity infeasible (limited hardware)

## Comparison Summary

| Metric | Baseline | Blind ($t=3$) |
|--------|----------|---------------|
| **QBER Estimation** | Required ($t$ bits) | Not required |
| **Worst-case Leakage** | $2048 + 100 + 64$ | $2048 + 1200 + 64$ |
| **Average Leakage (QBER=0.05)** | $2212$ bits | $2200$ bits (no sampling) |
| **Average Leakage (QBER=0.08)** | $2212$ bits | $2524$ bits |
| **Rounds** | 1 | 1-3 (adaptive) |
| **Decoding Time** | $50$ ms | $150$ ms (worst-case) |

## References

[1] J. Martinez-Mateo, D. Elkouss, and V. Martin, "Blind Reconciliation," *Quantum Information and Computation*, Vol. 12, No. 9&10, pp. 791-812, 2012.

[2] D. Elkouss, J. Martinez-Mateo, D. Lancho, and V. Martin, "Rate Compatible Protocol for Information Reconciliation: An Application to QKD," 2010.

---

[← Return to Main Index](../index.md) | [Previous: Baseline Strategy](./baseline_strategy.md) | [Next: Hybrid Puncturing →](./hybrid_puncturing.md)
