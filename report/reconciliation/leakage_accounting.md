[← Return to Main Index](../index.md)

# 6.5 Leakage Accounting and Safety Cap Enforcement

## Introduction

In the Noisy Storage Model (NSM), information leakage during reconciliation **directly reduces** the extractable secure key length. Unlike traditional QKD where syndrome information leaks to a passive eavesdropper, in Caligo's $\binom{2}{1}$-OT protocol, reconciliation leakage goes directly to Bob—the potential adversary attempting to learn Alice's alternative input.

The **LeakageTracker** module implements a **circuit breaker pattern** [1] that enforces the NSM-derived safety cap, immediately aborting reconciliation if cumulative leakage exceeds the secure threshold. This prevents security violations from propagating through the protocol pipeline.

## NSM Security Framework

### Finite-Size Key Rate Equation

The extractable secure key length after reconciliation is bounded by:

$$
\ell_{\text{secure}} \leq n_{\text{sifted}} \cdot \left[ H_{\min}^{\epsilon}(X|E) - \text{leak}_{\text{EC}} - \log_2\left(\frac{2}{\epsilon_{\text{sec}}^2}\right) \right]
$$

where:
- $n_{\text{sifted}}$: Number of bits after basis sifting
- $H_{\min}^{\epsilon}(X|E)$: Smooth min-entropy of Alice's string given Bob's quantum side information
- $\text{leak}_{\text{EC}}$: Total information revealed during error correction
- $\epsilon_{\text{sec}}$: Security parameter (typically $10^{-10}$)

**Critical constraint:** If $\text{leak}_{\text{EC}}$ approaches $H_{\min}^{\epsilon}(X|E)$, the extractable length collapses to zero. The **safety cap** is set conservatively below this threshold:

$$
\text{SafetyCap} = \alpha \cdot \left[ n_{\text{sifted}} \cdot H_{\min}^{\epsilon}(X|E) - \log_2\left(\frac{2}{\epsilon_{\text{sec}}^2}\right) \right]
$$

where $\alpha \in [0.7, 0.9]$ is a safety margin accounting for:
1. Finite-size estimation uncertainties
2. Privacy amplification overhead
3. Protocol implementation margin

### Min-Entropy Estimation

The smooth min-entropy depends on:
1. **Channel QBER** ($Q_{\text{channel}}$): Observable from sifting
2. **Storage noise** ($Q_{\text{storage}}$): Determined by NSM parameters $(\nu, \Delta t)$
3. **Finite-size correction**: Statistical fluctuation bounds

**Conservative bound** (from König et al. [2]):

$$
H_{\min}^{\epsilon}(X|E) \geq n \cdot \left( 1 - h(Q_{\text{channel}}) - \Delta_{\text{finite-size}} \right) - 2\log_2\left(\frac{1}{\epsilon_{\text{min-ent}}}\right)
$$

where $\Delta_{\text{finite-size}} \approx \sqrt{\frac{\log(1/\epsilon_{\text{min-ent}})}{2n}}$ accounts for sampling uncertainty.

**Example calculation:**
- $n = 10^6$ sifted bits
- $Q_{\text{channel}} = 0.05$ → $h(0.05) = 0.286$
- $\Delta_{\text{finite-size}} \approx 0.003$
- $\epsilon_{\text{min-ent}} = 10^{-10}$ → correction $\approx 67$ bits

$$
H_{\min}^{\epsilon}(X|E) \geq 10^6 \times (1 - 0.286 - 0.003) - 67 \approx 710,933 \text{ bits}
$$

With $\alpha = 0.8$ safety margin:

$$
\text{SafetyCap} \approx 0.8 \times 710,933 = 568,746 \text{ bits}
$$

## Leakage Components

### Reconciliation Leakage Breakdown

$$
\text{leak}_{\text{EC}} = \text{leak}_{\text{syndrome}} + \text{leak}_{\text{verification}} + \text{leak}_{\text{interaction}}
$$

#### 1. Syndrome Leakage

For syndrome-based reconciliation using mother code with rate $R_0$:

$$
\text{leak}_{\text{syndrome}} = (1 - R_0) \cdot n_{\text{block}}
$$

**Key property:** Syndrome length is **independent of effective rate** (determined solely by mother matrix).

**Example:** For $R_0 = 0.5$ and $n_{\text{block}} = 4096$:
$$
\text{leak}_{\text{syndrome}} = 0.5 \times 4096 = 2048 \text{ bits per block}
$$

#### 2. Verification Hash Leakage

To detect decoder errors (BP converging to wrong codeword), a verification hash is transmitted:

$$
\text{leak}_{\text{verification}} = h_{\text{bits}}
$$

**Typical values:**
- 32 bits: Collision probability $\approx 2^{-32}$ per block
- 64 bits: Collision probability $\approx 2^{-64}$ per block (recommended)
- 128 bits: Collision probability $\approx 2^{-128}$ per block (high security)

**Trade-off:** Longer hashes provide stronger verification but increase leakage.

#### 3. Interaction Leakage

**Baseline Protocol:** 
$$
\text{leak}_{\text{interaction}}^{\text{Baseline}} = t_{\text{sample}}
$$
where $t_{\text{sample}}$ is the number of disclosed bits for QBER estimation.

**Blind Protocol:**
$$
\text{leak}_{\text{interaction}}^{\text{Blind}} = \sum_{i=2}^{t} \Delta_i + \text{leak}_{\text{shortening}}
$$

where:
- $\Delta_i$: Number of bits revealed in iteration $i$
- $\text{leak}_{\text{shortening}}$: Position information leakage (discussed below)

### Shortening Position Information

When Alice reveals shortened bit positions, she implicitly leaks combinatorial information:

$$
\text{leak}_{\text{shortening}} = \log_2 \binom{n}{s} \approx s \cdot \log_2\left(\frac{n}{s}\right)
$$

**Conservative approximation:**

For $s \ll n$:
$$
\text{leak}_{\text{shortening}} \approx s \cdot \log_2(n/s)
$$

**Example:** For $n = 4096$, $s = 100$:
$$
\text{leak}_{\text{shortening}} = 100 \times \log_2(4096/100) \approx 100 \times 5.36 = 536 \text{ bits}
$$

**Mitigation:** Use a **pre-agreed deterministic pattern** (derived from public seed) so position indices need not be transmitted. In this case:
$$
\text{leak}_{\text{shortening}} \to 0 \text{ (positions implicit)}
$$

Caligo implements this optimization: shortening positions follow the hybrid puncturing pattern order, which is public and reproducible.

## LeakageTracker Implementation

### Core Architecture

```python
@dataclass
class LeakageRecord:
    """
    Record of a single leakage event.
    
    Attributes
    ----------
    syndrome_bits : int
        Syndrome bits transmitted.
    hash_bits : int
        Verification hash bits.
    retry_penalty_bits : int
        Conservative penalty for retry/interaction metadata.
    shortening_bits : float
        Shortening position leakage (log2 combinatorial bound).
    block_id : int
        Associated block identifier.
    iteration : int
        Blind iteration number (1-indexed).
    """
    syndrome_bits: int
    hash_bits: int = 64  # Default verification hash
    retry_penalty_bits: int = 0
    shortening_bits: float = 0.0
    block_id: int = 0
    iteration: int = 1
    
    @property
    def total_leakage(self) -> int:
        """Total leakage for this record."""
        return int(math.ceil(
            self.syndrome_bits + self.hash_bits + 
            self.retry_penalty_bits + self.shortening_bits
        ))
```

### Circuit Breaker Pattern

The circuit breaker is the **critical security mechanism** that prevents leakage budget violations:

```python
class LeakageTracker:
    """
    Accumulate and enforce reconciliation leakage bounds with circuit breaker.
    
    Parameters
    ----------
    safety_cap : int
        Maximum allowed total leakage in bits.
    abort_on_exceed : bool, optional
        If True (default), immediately raise exception when cap exceeded.
    """
    
    def __init__(self, safety_cap: int, abort_on_exceed: bool = True):
        if safety_cap < 0:
            raise ValueError("safety_cap must be non-negative")
        
        self.safety_cap = safety_cap
        self.abort_on_exceed = abort_on_exceed
        self.records: List[LeakageRecord] = []
        self._cumulative_leakage = 0
    
    @property
    def total_leakage(self) -> int:
        """Current cumulative leakage across all records."""
        return sum(record.total_leakage for record in self.records)
    
    @property
    def remaining_budget(self) -> int:
        """Remaining leakage budget before cap."""
        return max(0, self.safety_cap - self.total_leakage)
    
    def record(self, event: LeakageRecord) -> None:
        """
        Add a leakage event with immediate circuit breaker enforcement.
        
        Raises
        ------
        LeakageBudgetExceeded
            If cumulative leakage exceeds safety_cap and abort_on_exceed=True.
        """
        self.records.append(event)
        current_leakage = self.total_leakage
        
        logger.debug(
            "Leakage recorded: block=%d, syndrome=%d, hash=%d, "
            "short=%.1f, iter=%d, total=%d/%d",
            event.block_id, event.syndrome_bits, event.hash_bits,
            event.shortening_bits, event.iteration,
            current_leakage, self.safety_cap
        )
        
        # CIRCUIT BREAKER: Immediate enforcement
        if self.abort_on_exceed and current_leakage > self.safety_cap:
            margin = current_leakage - self.safety_cap
            
            logger.error(
                "Leakage budget EXCEEDED: %d > %d (margin: %d bits)",
                current_leakage, self.safety_cap, margin
            )
            
            raise LeakageBudgetExceeded(
                f"Cumulative leakage {current_leakage} bits exceeds "
                f"safety cap {self.safety_cap} bits (exceeded by {margin} bits)",
                actual_leakage=current_leakage,
                max_allowed=self.safety_cap,
            )
```

### Exception Hierarchy

```python
class LeakageBudgetExceeded(SecurityError):
    """
    Raised when cumulative leakage exceeds NSM safety cap.
    
    This is a FATAL error that should abort the entire protocol run.
    """
    
    def __init__(
        self,
        message: str,
        actual_leakage: int,
        max_allowed: int,
    ):
        super().__init__(message)
        self.actual_leakage = actual_leakage
        self.max_allowed = max_allowed
        self.margin = actual_leakage - max_allowed
```

## Integration with Reconciliation Flow

### Baseline Protocol Integration

```python
class BaselineStrategy:
    def alice_reconcile_block(
        self,
        payload: np.ndarray,
        ctx: ReconciliationContext,
        block_id: int,
    ) -> BlockResult:
        """Alice performs baseline reconciliation with leakage tracking."""
        
        # ... frame construction, syndrome computation ...
        
        # Record leakage BEFORE transmission
        try:
            self._leakage_tracker.record(LeakageRecord(
                syndrome_bits=len(syndrome),
                hash_bits=self._hash_bits,
                block_id=block_id,
            ))
        except LeakageBudgetExceeded as e:
            logger.error(f"Block {block_id}: {e}")
            raise  # Propagate to protocol coordinator
        
        # Transmission happens AFTER leakage is verified acceptable
        response = yield {
            "syndrome": syndrome,
            "hash": verification_hash,
        }
        
        return BlockResult(success=response["verified"])
```

### Blind Protocol Integration

```python
class BlindStrategy:
    def alice_reconcile_block(
        self,
        payload: np.ndarray,
        ctx: ReconciliationContext,
        block_id: int,
    ) -> Generator:
        """Alice performs blind reconciliation with iterative leakage tracking."""
        
        # Iteration 1: Initial syndrome (no revealed bits yet)
        try:
            self._leakage_tracker.record(LeakageRecord(
                syndrome_bits=len(syndrome),
                hash_bits=self._hash_bits,
                shortening_bits=0.0,  # Positions implicit in pattern
                block_id=block_id,
                iteration=1,
            ))
        except LeakageBudgetExceeded:
            raise  # Abort before transmission
        
        response = yield {"syndrome": syndrome, "hash": hash}
        
        # Iterations 2+: Revealed bits
        iteration = 2
        while not response["success"] and iteration <= self._max_iterations:
            # Check if revealing Δ more bits would exceed budget
            projected_leakage = self._leakage_tracker.total_leakage + self._delta
            
            if projected_leakage > self._leakage_tracker.safety_cap:
                logger.warning(
                    f"Block {block_id}: Revealing {self._delta} bits would "
                    f"exceed budget ({projected_leakage} > {self.safety_cap}). "
                    "Aborting blind iterations."
                )
                break  # Stop revealing, accept failure
            
            # Reveal next batch
            revealed_batch = self._get_revelation_batch(iteration)
            
            try:
                self._leakage_tracker.record(LeakageRecord(
                    syndrome_bits=0,  # Syndrome sent once
                    hash_bits=0,       # Hash sent once
                    shortening_bits=0.0,  # Positions implicit
                    retry_penalty_bits=len(revealed_batch),
                    block_id=block_id,
                    iteration=iteration,
                ))
            except LeakageBudgetExceeded:
                raise
            
            response = yield {"revealed": revealed_batch}
            iteration += 1
```

## Safety Cap Calculation Strategies

### Conservative Fixed Fraction

**Approach:** Set safety cap as a fixed fraction of estimated min-entropy.

$$
\text{SafetyCap} = \alpha \cdot n_{\text{sifted}} \cdot (1 - h(Q_{\text{channel}}))
$$

**Typical values:** $\alpha \in [0.7, 0.85]$

**Advantages:**
- Simple to compute
- Conservative for all block sizes

**Disadvantages:**
- May be overly conservative for large $n$
- Ignores finite-size effects

### Finite-Size Aware

**Approach:** Include explicit finite-size correction.

$$
\text{SafetyCap} = \alpha \cdot \left[ n \cdot (1 - h(Q_{\text{channel}}) - \Delta_{\text{FS}}) - \kappa_{\text{sec}} \right]
$$

where:
- $\Delta_{\text{FS}} = \sqrt{\frac{\log(1/\epsilon)}{2n}}$: Finite-size correction
- $\kappa_{\text{sec}} = \log_2(2/\epsilon_{\text{sec}}^2)$: Security parameter overhead

**Advantages:**
- Tighter for large $n$
- Accounts for statistical uncertainty

**Disadvantages:**
- More complex
- Requires careful $\epsilon$ calibration

### Adaptive Per-Block

**Approach:** Adjust cap dynamically based on observed QBER trajectory.

```python
def compute_adaptive_cap(
    block_history: List[Tuple[int, float]],  # (n_sifted, qber)
    target_security: float = 1e-10,
    safety_margin: float = 0.8,
) -> int:
    """
    Compute adaptive safety cap based on block history.
    
    Uses exponentially weighted moving average for QBER estimation.
    """
    # Estimate QBER with EWMA
    qber_estimate = 0.0
    alpha_ewma = 0.3
    
    for n, qber in block_history[-10:]:  # Last 10 blocks
        qber_estimate = alpha_ewma * qber + (1 - alpha_ewma) * qber_estimate
    
    # Add conservative confidence bound
    qber_conservative = qber_estimate + 0.01  # +1% margin
    
    # Total sifted bits
    total_n = sum(n for n, _ in block_history)
    
    # Compute cap
    h_qber = binary_entropy(qber_conservative)
    finite_size_correction = np.sqrt(np.log(1/target_security) / (2*total_n))
    
    min_entropy_estimate = total_n * (1 - h_qber - finite_size_correction)
    security_overhead = np.log2(2 / target_security**2)
    
    cap = int(safety_margin * (min_entropy_estimate - security_overhead))
    
    return max(0, cap)
```

## Leakage Monitoring and Diagnostics

### Real-Time Monitoring

```python
class LeakageMonitor:
    """Monitor leakage accumulation during protocol execution."""
    
    def __init__(self, tracker: LeakageTracker):
        self.tracker = tracker
        self.checkpoints: List[Tuple[int, int]] = []  # (timestamp, leakage)
    
    def checkpoint(self, label: str = ""):
        """Record current leakage state."""
        timestamp = time.time_ns()
        leakage = self.tracker.total_leakage
        self.checkpoints.append((timestamp, leakage))
        
        utilization = leakage / self.tracker.safety_cap
        logger.info(
            f"Leakage checkpoint [{label}]: {leakage}/{self.tracker.safety_cap} "
            f"({utilization:.1%} utilization)"
        )
    
    def get_leakage_rate(self) -> float:
        """Compute leakage accumulation rate (bits/second)."""
        if len(self.checkpoints) < 2:
            return 0.0
        
        t0, l0 = self.checkpoints[0]
        t1, l1 = self.checkpoints[-1]
        
        time_elapsed_sec = (t1 - t0) / 1e9
        leakage_delta = l1 - l0
        
        return leakage_delta / time_elapsed_sec if time_elapsed_sec > 0 else 0.0
    
    def estimate_remaining_blocks(self, avg_block_leakage: int) -> int:
        """Estimate how many more blocks can be reconciled."""
        remaining_budget = self.tracker.remaining_budget
        
        if avg_block_leakage <= 0:
            return float('inf')
        
        return remaining_budget // avg_block_leakage
```

### Post-Execution Analysis

```python
def analyze_leakage_distribution(tracker: LeakageTracker) -> Dict:
    """
    Analyze leakage distribution across records.
    
    Returns
    -------
    Dict
        Statistics: total, per_component breakdown, efficiency
    """
    total_syndrome = sum(r.syndrome_bits for r in tracker.records)
    total_hash = sum(r.hash_bits for r in tracker.records)
    total_interaction = sum(r.retry_penalty_bits for r in tracker.records)
    total_shortening = sum(r.shortening_bits for r in tracker.records)
    
    total = tracker.total_leakage
    
    return {
        "total_leakage": total,
        "breakdown": {
            "syndrome": total_syndrome,
            "hash": total_hash,
            "interaction": total_interaction,
            "shortening": total_shortening,
        },
        "percentages": {
            "syndrome": 100 * total_syndrome / total,
            "hash": 100 * total_hash / total,
            "interaction": 100 * total_interaction / total,
            "shortening": 100 * total_shortening / total,
        },
        "utilization": total / tracker.safety_cap,
        "margin": tracker.remaining_budget,
    }
```

**Example output:**

```python
{
    'total_leakage': 456234,
    'breakdown': {
        'syndrome': 409600,  # 200 blocks × 2048 bits
        'hash': 12800,       # 200 blocks × 64 bits
        'interaction': 32456,
        'shortening': 1378,
    },
    'percentages': {
        'syndrome': 89.77%,
        'hash': 2.81%,
        'interaction': 7.11%,
        'shortening': 0.30%,
    },
    'utilization': 0.803,  # 80.3% of cap used
    'margin': 111766,      # 111766 bits remaining
}
```

## Optimization Strategies

### Minimizing Syndrome Leakage

**Strategy 1: Higher Mother Rate**

Use $R_0 = 0.6$ instead of $R_0 = 0.5$:
$$
\text{leak}_{\text{syndrome}} = (1 - 0.6) \times 4096 = 1638 \text{ bits} \quad (\downarrow 20\%)
$$

**Trade-off:** Narrower effective rate range:
$$
R_{\max} = \frac{0.6}{1 - 0.44} = 1.07 \quad (\text{infeasible!})
$$

For $R_{\max} = 0.9$, need $\delta = 1 - 0.6/0.9 = 0.333$ (reduced coverage).

**Strategy 2: Larger Block Size**

Use $n = 8192$ instead of $n = 4096$ (amortizes overhead):
$$
f_{\text{overhead}} = \frac{64 + 100}{4096 \times h(\text{QBER})} \to \frac{64 + 100}{8192 \times h(\text{QBER})} \quad (\downarrow 50\%)
$$

**Trade-off:** Higher computational complexity and memory footprint.

### Minimizing Interaction Leakage

**Baseline:**
- Reduce sample size $t$ (increases QBER estimation uncertainty)
- Use conservative confidence bounds (avoid under-estimation)

**Blind:**
- Use larger revelation steps $\Delta$ (fewer iterations)
- Employ NSM-gated variant with heuristic QBER (avoid worst-case revelation)

## Failure Recovery

### Leakage Budget Exhaustion

When leakage approaches the safety cap mid-protocol:

```python
if tracker.remaining_budget < min_block_leakage:
    logger.warning(
        "Insufficient leakage budget for additional blocks. "
        f"Remaining: {tracker.remaining_budget} bits, "
        f"Required: {min_block_leakage} bits/block"
    )
    
    # Option 1: Abort remaining blocks
    raise InsufficientLeakageBudget(
        "Cannot complete reconciliation within safety cap"
    )
    
    # Option 2: Attempt partial reconciliation
    # (reconcile as many complete blocks as budget allows)
    reconcilable_blocks = tracker.remaining_budget // min_block_leakage
    logger.info(f"Attempting partial reconciliation: {reconcilable_blocks} blocks")
```

### Post-Reconciliation Verification

After reconciliation completes, verify leakage budget compliance:

```python
def verify_leakage_compliance(
    tracker: LeakageTracker,
    min_entropy: float,
    target_output_length: int,
) -> bool:
    """
    Verify that leakage permits extracting target output length.
    
    Returns
    -------
    bool
        True if extraction is secure, False otherwise.
    """
    effective_entropy = min_entropy - tracker.total_leakage
    security_overhead = np.log2(2 / (1e-10)**2)
    
    max_extractable = int(effective_entropy - security_overhead)
    
    if max_extractable < target_output_length:
        logger.error(
            f"Insufficient entropy for extraction: "
            f"need {target_output_length}, have {max_extractable}"
        )
        return False
    
    logger.info(
        f"Leakage compliance verified: "
        f"{max_extractable} bits extractable (target: {target_output_length})"
    )
    return True
```

## Testing and Validation

### Unit Tests

```python
def test_circuit_breaker():
    """Test that circuit breaker aborts on cap violation."""
    tracker = LeakageTracker(safety_cap=1000, abort_on_exceed=True)
    
    # Should succeed
    tracker.record(LeakageRecord(syndrome_bits=500, hash_bits=64))
    assert tracker.total_leakage == 564
    
    # Should fail (1064 > 1000)
    with pytest.raises(LeakageBudgetExceeded) as exc_info:
        tracker.record(LeakageRecord(syndrome_bits=500, hash_bits=64))
    
    assert exc_info.value.actual_leakage == 1128
    assert exc_info.value.margin == 128


def test_shortening_leakage():
    """Test shortening position leakage calculation."""
    # For n=4096, s=100: leak ≈ 100 * log2(4096/100) ≈ 536
    leak = compute_shortening_leakage(n=4096, s=100)
    assert 535 <= leak <= 537
    
    # For s=0: no leakage
    leak_zero = compute_shortening_leakage(n=4096, s=0)
    assert leak_zero == 0
```

### Integration Tests

```python
@pytest.mark.integration
def test_baseline_leakage_enforcement():
    """Test that Baseline reconciliation respects safety cap."""
    # Setup: artificially low cap to trigger failure
    safety_cap = 1000  # Will be exceeded by first block
    tracker = LeakageTracker(safety_cap=safety_cap)
    
    strategy = BaselineStrategy(
        mother_code=load_mother_code(),
        leakage_tracker=tracker,
    )
    
    # Attempt reconciliation (should abort)
    with pytest.raises(LeakageBudgetExceeded):
        result = strategy.alice_reconcile_block(
            payload=np.random.randint(0, 2, 2048),
            ctx=ReconciliationContext(qber_measured=0.05),
            block_id=0,
        )
```

## References

[1] M. Nygard, "Release It! Design and Deploy Production-Ready Software," 2nd ed., Pragmatic Bookshelf, 2018.

[2] R. König, S. Wehner, and J. Wullschleger, "Unconditional Security From Noisy Quantum Storage," *IEEE Transactions on Information Theory*, Vol. 58, No. 3, pp. 1962-1984, 2012.

[3] M. Tomamichel, C. C. W. Lim, N. Gisin, and R. Renner, "Tight Finite-Key Analysis for Quantum Cryptography," *Nature Communications*, Vol. 3, Article 634, 2012.

[4] E. Kiktenko et al., "Post-processing procedure for industrial quantum key distribution systems," *J. Phys.: Conf. Ser.* 741, 012081, 2016.

---

[← Return to Main Index](../index.md) | [Previous: Hybrid Puncturing](./hybrid_puncturing.md)
