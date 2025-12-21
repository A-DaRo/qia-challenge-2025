[← Return to Main Index](../index.md)

# 6.2 Baseline Reconciliation Strategy

## Introduction

The **Baseline reconciliation strategy** implements the Elkouss et al. (2010) rate-compatible protocol [1], which requires *a priori* QBER estimation followed by a single-shot syndrome transmission. This approach optimizes for minimal interactivity at the cost of requiring accurate channel parameter knowledge.

In the Caligo NSM-OT context, Baseline reconciliation trades the disclosure of $t$ sampling bits (used for QBER estimation) for a one-shot decoding attempt with near-optimal rate matching. The protocol is particularly efficient when the QBER is stable and can be estimated accurately with small sample sizes.

## Protocol Flow

### Phase A: QBER Estimation via Sampling

Before reconciliation begins, Alice and Bob estimate the channel crossover probability using a random test sample.

**Step 1: Sample Selection** (Bob)

Bob randomly selects $t$ positions from his received string $\mathbf{y}$:

$$
\mathcal{T} = \{\text{random positions}\} \subset \{1, 2, \ldots, m\}, \quad |\mathcal{T}| = t
$$

Bob sends:
1. The positions: $\text{pos}(\mathbf{y}) = \mathcal{T}$
2. The bit values: $m(\mathbf{y}) = \{y_i : i \in \mathcal{T}\}$

**Step 2: QBER Computation** (Alice)

Alice extracts the corresponding bits from her string and computes the empirical error rate:

$$
p^* = \frac{1}{t} \sum_{i \in \mathcal{T}} \mathbf{1}[x_i \neq y_i]
$$

This is the **raw QBER estimate**.

**Step 3: Conservative Adjustment**

For finite-size security, the estimate must be adjusted upward to account for statistical uncertainty. Using Hoeffding's inequality [2], the $(1-\varepsilon_{\text{PE}})$-confidence upper bound is:

$$
p^*_{\text{FK}} = p^* + \sqrt{\frac{\ln(1/\varepsilon_{\text{PE}})}{2t}}
$$

where $\varepsilon_{\text{PE}}$ is the parameter estimation failure probability (typically $10^{-10}$).

**Example:** For $t = 100$ and $\varepsilon_{\text{PE}} = 10^{-10}$:

$$
p^*_{\text{FK}} = p^* + \sqrt{\frac{23.03}{200}} \approx p^* + 0.034
$$

This conservative adjustment ensures that the selected code rate is sufficient with high probability.

### Phase B: Rate Selection and Frame Construction

**Step 4: Efficiency Model Lookup**

Alice uses an empirically calibrated efficiency function $f(p)$ to determine the target effective rate:

$$
R_{\text{target}} = 1 - f(p^*_{\text{FK}}) \cdot h(p^*_{\text{FK}})
$$

where:
- $h(p) = -p\log_2(p) - (1-p)\log_2(1-p)$ is the binary entropy function
- $f(p) \geq 1$ is the reconciliation efficiency (code-family dependent)

**Typical efficiency model** [1, 3]:

$$
f(p) = \begin{cases}
1.10 & \text{if } 0.01 \leq p \leq 0.05 \\
1.10 + |p - 0.10| & \text{if } 0.05 < p \leq 0.11 \\
\text{infeasible} & \text{if } p > 0.11
\end{cases}
$$

This model reflects that LDPC efficiency degrades near threshold and becomes infeasible beyond the Shannon capacity of practical codes.

**Step 5: Puncturing and Shortening Allocation**

Given the target rate $R_{\text{target}}$, mother code rate $R_0$, frame size $n$, and modulation parameter $d = \lfloor \delta \cdot n \rfloor$, Alice computes:

$$
s = \left\lceil \left( R_0 - R_{\text{target}} \left(1 - \frac{d}{n}\right) \right) \cdot n \right\rceil
$$

$$
p = d - s
$$

**Constraint verification:**
- $s \geq 0$ (non-negative shortening)
- $p \geq 0$ (non-negative puncturing)
- $s + p = d$ (total modulation constraint)

If $s < 0$, the requested rate is too high for the chosen $\delta$; if $p < 0$, the requested rate is too low. In either case, the protocol aborts with an error indicating insufficient modulation range.

**Step 6: Frame Construction**

Alice constructs the full codeword frame using the deterministic function $g(\mathbf{x}, \sigma, \pi)$:

1. **Load puncturing pattern:** Retrieve the hybrid puncturing pattern for rate $R_{\text{eff}}$, which specifies:
   - $\mathcal{I}_{\text{payload}}$: payload positions
   - $\mathcal{I}_{\text{punct}}$: punctured positions (size $p$)
   - $\mathcal{I}_{\text{short}}$: shortened positions (size $s$)

2. **Fill frame:**

$$
x^+_i = \begin{cases}
x_{\text{payload}[j]} & \text{if } i \in \mathcal{I}_{\text{payload}} \\
\text{PRNG}(\text{seed}, i) & \text{if } i \in \mathcal{I}_{\text{punct}} \\
\text{PRNG}_{\text{short}}(\text{seed}, i) & \text{if } i \in \mathcal{I}_{\text{short}}
\end{cases}
$$

The PRNG seed is synchronized at protocol initialization to ensure Alice and Bob generate identical padding.

**Step 7: Syndrome Computation**

Alice computes the syndrome using the mother parity-check matrix:

$$
\mathbf{s} = H_{\text{mother}} \cdot \mathbf{x}^+ \mod 2
$$

The syndrome length is:

$$
|\mathbf{s}| = m_{\text{checks}} = (1 - R_0) \cdot n
$$

For $R_0 = 0.5$ and $n = 4096$, this yields $|\mathbf{s}| = 2048$ bits.

**Step 8: Verification Hash Computation**

To detect decoder errors (wrong codeword with correct syndrome), Alice computes a polynomial hash [4]:

$$
h(\mathbf{x}) = \left( \sum_{i=0}^{m-1} x_i \cdot \alpha^i \right) \mod P(\alpha)
$$

where $P(\alpha)$ is an irreducible polynomial of degree $h_{\text{bits}}$ (typically 32-128 bits).

### Phase C: Transmission and Decoding

**Step 9: Public Message** (Alice → Bob)

Alice sends:
1. Syndrome: $\mathbf{s}$ (2048 bits for $n=4096$, $R_0=0.5$)
2. Rate identifier: $p^*_{\text{FK}}$ or $(p, s)$ tuple (small overhead, typically 8 bytes)
3. Verification hash: $h(\mathbf{x})$ (32-128 bits)

**Total disclosure:**

$$
|\text{Message}| = |\mathbf{s}| + \log_2(\text{rate params}) + |h| \approx (1-R_0) \cdot n + 32 + h_{\text{bits}}
$$

**Step 10: Frame Construction** (Bob)

Bob reconstructs his frame $\mathbf{y}^+$ using the same procedure:

1. Reproduce $(p, s)$ from received $p^*_{\text{FK}}$ (or directly from message)
2. Load the same puncturing pattern
3. Fill frame with $\mathbf{y}$, pseudo-random puncture padding, and known shortened values

**Step 11: Decoder Initialization** (Bob)

Bob initializes LLRs using the **three-state rule**:

$$
\lambda_i = \begin{cases}
\ln\frac{1-p^*_{\text{FK}}}{p^*_{\text{FK}}} \cdot (1 - 2y_i) & \text{if } i \in \mathcal{I}_{\text{payload}} \\
0 & \text{if } i \in \mathcal{I}_{\text{punct}} \\
+100 & \text{if } i \in \mathcal{I}_{\text{short}} \text{ and } \text{short\_val}_i = 0 \\
-100 & \text{if } i \in \mathcal{I}_{\text{short}} \text{ and } \text{short\_val}_i = 1
\end{cases}
$$

The value $\pm 100$ approximates $\pm\infty$ without numerical overflow in tanh computations.

**Step 12: Belief Propagation Decoding**

Bob runs iterative BP decoding (see [LDPC Framework](./ldpc_framework.md)) until:
- **Success:** $H \cdot \hat{\mathbf{x}} = \mathbf{s}$ (syndrome match)
- **Failure:** Maximum iterations reached

**Step 13: Verification**

If decoding succeeds, Bob computes $h(\hat{\mathbf{x}})$ and compares with the received hash:

$$
\text{Verified} = \begin{cases}
\text{True} & \text{if } h(\hat{\mathbf{x}}) = h(\mathbf{x}) \\
\text{False} & \text{otherwise}
\end{cases}
$$

Verification failures indicate a **decoder error**: BP converged to a codeword with matching syndrome but incorrect payload.

## Leakage Analysis

### Disclosed Information

The Baseline protocol reveals:

1. **Sampling disclosure:** $t$ bits from $\mathbf{x}$ (positions + values)
2. **Syndrome:** $(1 - R_0) \cdot n$ bits
3. **Verification hash:** $h_{\text{bits}}$ bits
4. **Rate parameter:** $\approx 8$ bytes (negligible)

**Total leakage:**

$$
\text{leak}_{\text{Baseline}} = t + (1-R_0) \cdot n + h_{\text{bits}}
$$

### NSM Budget Impact

In the NSM finite-size key rate equation:

$$
\ell \leq n \cdot \left[ H_{\min}^{\epsilon}(X|E) - \text{leak}_{\text{Baseline}} - \log_2\left(\frac{2}{\epsilon^2}\right) \right]
$$

the Baseline leakage **directly reduces** the extractable secure output by $\text{leak}_{\text{Baseline}}$ bits.

**Critical observation:** The syndrome length is **independent of the effective rate** because it is computed with the fixed mother matrix $H_{\text{mother}}$. Rate adaptation via puncturing/shortening modifies decoder behavior without changing syndrome size.

### Efficiency Optimization

For a given QBER $p$, the reconciliation efficiency is:

$$
f = \frac{\text{leak}_{\text{Baseline}}}{n \cdot h(p)} = \frac{t + (1-R_0) \cdot n + h_{\text{bits}}}{n \cdot h(p)}
$$

**Minimizing leakage:**
1. **Reduce $t$:** Use conservative confidence bounds to minimize sampling size
2. **Optimize $R_0$:** Choose mother rate balancing coverage and syndrome length
3. **Minimize $h_{\text{bits}}$:** Use smallest hash providing acceptable collision probability

**Typical values:**
- $t = 100$ (sampling)
- $h_{\text{bits}} = 64$ (verification)
- $n = 4096$, $R_0 = 0.5$ → syndrome = 2048 bits
- **Total:** $\approx 2212$ bits

For QBER = 0.05, $h(0.05) \approx 0.286$:

$$
f = \frac{2212}{4096 \times 0.286} \approx 1.89
$$

This is substantially higher than the optimal $f \approx 1.10$ because the sampling overhead ($t$) is amortized over a small block. For larger block sizes ($n \geq 10^5$), the efficiency approaches the code's intrinsic efficiency.

## Implementation Details

### Code Structure

```python
# caligo/reconciliation/strategies/baseline.py

class BaselineStrategy(ReconciliationStrategy):
    """
    Elkouss et al. (2010) rate-compatible reconciliation.
    
    Single-shot syndrome transmission with QBER-based rate selection.
    """
    
    def alice_reconcile_block(
        self,
        payload: np.ndarray,
        ctx: ReconciliationContext,
        block_id: int,
    ) -> Generator[Dict[str, Any], Dict[str, Any], BlockResult]:
        """
        Alice performs baseline reconciliation for one block.
        
        Yields
        ------
        Dict
            {"type": "syndrome", "data": syndrome, "rate": (p, s), "hash": h}
        
        Returns
        -------
        BlockResult
            Success/failure status with leakage accounting.
        """
        # Step 1: Extract QBER from context (pre-estimated in Phase II)
        qber = ctx.qber_measured
        qber_fk = qber + self._confidence_adjustment(ctx.sample_size)
        
        # Step 2: Rate selection
        r_target = 1 - self._efficiency_model(qber_fk) * binary_entropy(qber_fk)
        s = int(np.ceil((self.R0 - r_target * (1 - self.delta)) * self.n))
        p = self.modulation_budget - s
        
        if s < 0 or p < 0:
            raise ValueError(f"Rate infeasible: s={s}, p={p}")
        
        # Step 3: Load puncturing pattern
        pattern = self._mother_code.get_pattern(effective_rate=r_target)
        
        # Step 4: Construct frame
        frame = self._construct_frame(payload, pattern, p, s)
        
        # Step 5: Compute syndrome
        syndrome = self._mother_code.compute_syndrome(frame)
        
        # Step 6: Compute verification hash
        hash_val = self._hash_verifier.compute_hash(payload, block_id)
        
        # Step 7: Send to Bob
        message = {
            "type": "syndrome",
            "data": syndrome,
            "p": p,
            "s": s,
            "hash": hash_val,
        }
        response = yield message
        
        # Step 8: Record leakage
        self._leakage_tracker.record_block(
            syndrome_bits=len(syndrome),
            hash_bits=self._hash_bits,
            n_shortened=s,
            frame_size=self.n,
            block_id=block_id,
        )
        
        # Step 9: Return result
        return BlockResult(
            success=response["verified"],
            iterations=response["iterations"],
            leakage=len(syndrome) + self._hash_bits,
        )
```

### Bob's Decoder

```python
def bob_reconcile_block(
    self,
    received: np.ndarray,
    ctx: ReconciliationContext,
    block_id: int,
) -> Generator[Dict[str, Any], Dict[str, Any], BlockResult]:
    """
    Bob decodes using syndrome from Alice.
    """
    # Step 1: Receive Alice's message
    alice_msg = yield {"type": "ready"}
    
    syndrome = alice_msg["data"]
    p, s = alice_msg["p"], alice_msg["s"]
    hash_expected = alice_msg["hash"]
    
    # Step 2: Reconstruct frame
    pattern = self._mother_code.get_pattern_from_ps(p, s)
    frame = self._construct_frame(received, pattern, p, s)
    
    # Step 3: Initialize LLRs
    qber = ctx.qber_measured
    llr = build_three_state_llr(frame, qber, pattern.puncture_mask, pattern.shorten_mask)
    
    # Step 4: Decode
    result = self._decoder.decode(
        llr=llr,
        target_syndrome=syndrome,
        max_iterations=self._max_iterations,
    )
    
    # Step 5: Verify
    if result.converged:
        corrected_payload = self._extract_payload(result.corrected_bits, pattern)
        hash_actual = self._hash_verifier.compute_hash(corrected_payload, block_id)
        verified = (hash_actual == hash_expected)
    else:
        verified = False
    
    # Step 6: Report to Alice
    yield {
        "verified": verified,
        "iterations": result.iterations,
    }
    
    return BlockResult(
        success=verified,
        iterations=result.iterations,
        corrected_bits=corrected_payload if verified else None,
    )
```

## Performance Characteristics

### Decoding Success Probability

For a well-designed LDPC code and accurate QBER estimate, the **Frame Error Rate** (FER) as a function of actual QBER follows a waterfall curve:

$$
\text{FER}(p) \approx \begin{cases}
10^{-6} & \text{if } p < p_{\text{threshold}} \\
\text{rapid increase} & \text{if } p \approx p_{\text{threshold}} \\
1 & \text{if } p > p_{\text{threshold}}
\end{cases}
$$

The threshold $p_{\text{threshold}}$ is determined by the effective rate and code structure (typically via density evolution analysis).

### Computational Complexity

**Alice:**
- Frame construction: $O(n)$
- Syndrome computation: $O(|\text{edges}|) \approx O(n \cdot \bar{d}_v)$
- **Total:** $\approx 10$ ms for $n=4096$

**Bob:**
- Frame construction: $O(n)$
- LLR initialization: $O(n)$
- BP decoding: $O(I \cdot |\text{edges}|)$ where $I \approx 20$-$50$
- **Total:** $\approx 50$-$100$ ms for $n=4096$

### Communication Rounds

**Advantage:** Only **one round** of classical communication (Alice → Bob).

**Comparison with Cascade:** Traditional Cascade requires $O(\log n)$ rounds with binary search for each error, making it highly interactive [5].

## Failure Modes and Mitigation

### Failure Mode 1: QBER Underestimation

**Symptom:** Decoder fails to converge because selected rate is too high

**Cause:** Sampling estimate $p^*$ below actual channel error rate

**Mitigation:**
1. Use conservative confidence bound $p^*_{\text{FK}}$
2. Increase sample size $t$ (at cost of more leakage)
3. Implement retry with lower rate (transitions to multi-round protocol)

### Failure Mode 2: Decoder Error

**Symptom:** BP converges to wrong codeword with matching syndrome

**Cause:** Decoder trapped in incorrect fixed point (common in high-error regime)

**Detection:** Verification hash mismatch

**Mitigation:**
1. Use sufficiently long hash ($h_{\text{bits}} \geq 64$)
2. Abort reconciliation and flag protocol failure
3. Require re-sifting or abort entire protocol run

### Failure Mode 3: Infeasible Rate

**Symptom:** Rate selection yields $s < 0$ or $p < 0$

**Cause:** QBER outside the range covered by modulation parameter $\delta$

**Mitigation:**
1. Verify QBER against protocol thresholds before reconciliation
2. Use NSM security check: QBER must satisfy $Q_{\text{channel}} < Q_{\text{storage}}$
3. Abort with explicit error message

## Comparison with Alternative Strategies

| Aspect | Baseline (Elkouss) | Cascade | Blind (Martinez-Mateo) |
|--------|-------------------|---------|------------------------|
| **Interactivity** | 1 round | $O(\log n)$ rounds | 1-3 rounds |
| **QBER estimation** | Required ($t$ bits) | Implicit | Not required |
| **Leakage** | $(1-R_0)n + t + h$ | Variable, often high | $(1-R_0)n + h + \sum\Delta_i$ |
| **Complexity** | Low (single decode) | High (iterative search) | Medium (hot-start decode) |
| **Efficiency** | $f \approx 1.10$ (large $n$) | $f \approx 1.05$ | $f \approx 1.12$ (small $t$) |
| **Hardware suitability** | Excellent | Poor | Good |

## When to Use Baseline

**Ideal scenarios:**
1. QBER is stable and predictable
2. Minimal interactivity is critical (e.g., satellite links)
3. Large block sizes ($n \geq 10^4$) amortize sampling overhead
4. Hardware resources allow single-shot BP decoding

**Avoid when:**
1. QBER is highly variable (wastes efficiency)
2. Very small block sizes ($n < 10^3$) make sampling overhead dominant
3. Blind protocol's slight overhead is acceptable for QBER-free operation

## References

[1] D. Elkouss, J. Martinez-Mateo, D. Lancho, and V. Martin, "Rate Compatible Protocol for Information Reconciliation: An Application to QKD," 2010.

[2] W. Hoeffding, "Probability Inequalities for Sums of Bounded Random Variables," *Journal of the American Statistical Association*, Vol. 58, No. 301, pp. 13-30, 1963.

[3] D. Elkouss, A. Leverrier, R. Alléaume, and J. J. Boutros, "Efficient reconciliation protocol for discrete-variable quantum key distribution," *arXiv:0901.2140*, 2009.

[4] E. Kiktenko et al., "Post-processing procedure for industrial quantum key distribution systems," *J. Phys.: Conf. Ser.* 741, 012081, 2016.

[5] G. Brassard and L. Salvail, "Secret-Key Reconciliation by Public Discussion," *EUROCRYPT*, 1993.

---

[← Return to Main Index](../index.md) | [Previous: LDPC Framework](./ldpc_framework.md) | [Next: Blind Strategy →](./blind_strategy.md)
