# Implementation Report v2: Architectural Blueprint for Rate-Compatible Reconciliation in Caligo

**Version:** 2.1  
**Date:** December 2025  
**Module:** `caligo.reconciliation`  
**Based on:** Theoretical Report v2 (Elkouss/Martinez-Mateo)

---

## Executive Summary

This report provides a comprehensive architectural blueprint for transitioning the Caligo reconciliation layer from its current state—exhibiting multiple anti-patterns and a mathematically flawed random puncturing scheme—to a robust **Hybrid Rate-Compatible Architecture**. The refactoring enables two distinct, rate-compatible protocols (**Baseline** and **Blind**) to operate on a single, optimized $R_0=0.5$ ACE-PEG matrix with a **Hybrid Pattern Library**.

**Key Architectural Principles:**

1. **Hybrid Puncturing Strategy:** As established in Theoretical Report v2 (Theorem 2.2), strict untainted puncturing saturates at $R_{\text{eff}} \approx 0.625$. The architecture implements a **two-regime puncturing strategy**:
   - **Regime A (Untainted, $R \leq 0.625$):** Strict untainted puncturing guaranteeing stopping-set protection [3]
   - **Regime B (ACE-Guided, $R > 0.625$):** ACE/EMD-guided intentional puncturing for high-rate operation [4]

2. **Numba-First Computation Model:** NSM timing constraints require JIT-compiled kernels. Python handles control flow; Numba handles encoding/decoding.

3. **Fine-Grained Rate Adaptation:** A step size of $\Delta R = 0.01$ maximizes efficiency $f$ relative to the Shannon limit.

The implementation addresses critical security requirements under the Noisy Storage Model (NSM) by enforcing strict leakage accounting at architectural boundaries with circuit-breaker patterns.

---

## Part I: Current State Analysis

### 1. Architectural Overview

The current reconciliation subsystem is organized as follows:

```
caligo/reconciliation/
├── orchestrator.py          # "God Object" - mixed concerns
├── block_reconciler.py      # Single-block logic (partial separation)
├── matrix_manager.py        # Multi-matrix pool management
├── ldpc_encoder.py          # Syndrome computation
├── ldpc_decoder.py          # Belief propagation decoder
├── rate_selector.py         # Rate selection (degraded to single rate)
├── leakage_tracker.py       # Leakage accounting
├── hash_verifier.py         # Verification hash
├── blind_manager.py         # Blind state machine (incomplete)
├── factory.py               # Strategy factory (minimal)
└── constants.py             # Configuration constants
```

### 2. Identified Anti-Patterns and Architectural Flaws

#### 2.1 Anti-Pattern: "God Object" Orchestrator

**File:** [orchestrator.py](caligo/reconciliation/orchestrator.py)

The `ReconciliationOrchestrator` class violates the **Single Responsibility Principle** by conflating multiple concerns:

| Concern | Lines | Description |
|---------|-------|-------------|
| Block Partitioning | 210-245 | `partition_key()`, key slicing logic |
| Component Instantiation | 108-140 | Creates decoder, hash_verifier, block_reconciler |
| Decode Orchestration | 270-330 | `decode_with_retry()` retry logic |
| Leakage Enforcement | 319, `should_abort()` | Leakage cap checking |
| Configuration Management | 45-85 | `ReconciliationOrchestratorConfig` |

**Code Evidence:**

```python
# orchestrator.py:108-140 - Mixing construction with coordination
class ReconciliationOrchestrator:
    def __init__(self, matrix_manager, leakage_tracker, config, safety_cap):
        self.config = config or ReconciliationOrchestratorConfig()
        # Direct instantiation couples to concrete implementations
        self.decoder = BeliefPropagationDecoder(max_iterations=self.config.max_iterations)
        self.hash_verifier = PolynomialHashVerifier(hash_bits=self.config.hash_bits)
        self._block_reconciler = BlockReconciler(...)  # Another layer of composition
```

**Impact:** 
- Difficult to test in isolation
- Cannot switch between Baseline/Blind without modifying this class
- Protocol logic leaks into orchestration layer

#### 2.2 Critical Flaw: Random Puncturing in `ldpc_encoder.py`

**File:** [ldpc_encoder.py](caligo/reconciliation/ldpc_encoder.py#L66-L108)

The file defines `apply_puncture_pattern()` (correct approach) but the codebase historically used random padding that does not guarantee the untainted property.

**Current Implementation (Correct but Incomplete):**

```python
# ldpc_encoder.py:66-108
def apply_puncture_pattern(
    payload: np.ndarray,
    pattern: np.ndarray,
    frame_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct LDPC frame using untainted puncturing pattern.
    ...
    Notes: This implements the Elkouss et al. (2012) untainted puncturing scheme.
    """
```

**Critical Problem (Theoretical Report v2, Theorem 2.2):** The pattern loading in `matrix_manager.py` expects patterns in `puncture_patterns/` subdirectory, but:

1. **Untainted Saturation Limit:** Strict untainted puncturing saturates at approximately 20% of symbols ($\pi \approx 0.2$), corresponding to $R_{\text{eff}} \approx 0.625$ for an $R_0 = 0.5$ mother code. This is insufficient for the target range $R_{\text{eff}} \in [0.5, 0.9]$.

2. **Missing Regime B (ACE-Guided):** Per Theoretical Report v2 §2.2.3, to achieve $R_{\text{eff}} > 0.625$, the system must transition to ACE/EMD-guided puncturing (Liu & de Lamare, 2014 [4]).

3. **Rate selector hardcoded to R=0.5:** Due to convergence failures, [rate_selector.py#L110-L125](caligo/reconciliation/rate_selector.py#L110-L125) bypasses rate selection:

```python
# rate_selector.py:110-125 - CRITICAL FIX comment reveals the underlying issue
#
# CRITICAL FIX: Until we have better LDPC matrices or decoder tuning,
# we ALWAYS use rate 0.5 for all QBER levels. This sacrifices efficiency
# (more syndrome bits leaked) but ensures >90% decoder success rate.
#
MOST_RELIABLE_RATE = 0.5
return float(MOST_RELIABLE_RATE)  # Ignores rate selection entirely!
```

**Required Fix:** Implement **Hybrid Pattern Library** with fine-grained rate steps ($\Delta R = 0.01$) covering $R_{\text{eff}} \in [0.51, 0.90]$.

#### 2.3 Coupling Analysis: Protocol Logic in Role Classes

**Files:** [alice.py](caligo/protocol/alice.py), [bob.py](caligo/protocol/bob.py)

The `AliceProgram` and `BobProgram` classes contain embedded reconciliation logic that tightly couples them to the Baseline protocol message sequence.

**Alice Phase III Implementation (alice.py:285-400):**

```python
def _phase3_reconcile(self, alice_bits, qber_observed, qber_adjusted):
    # Direct matrix manager instantiation
    matrix_manager = MatrixManager.from_directory(recon_constants.LDPC_MATRICES_PATH)
    
    # Protocol-specific logic embedded in role class
    is_blind = self.params.reconciliation.reconciliation_type == ReconciliationType.BLIND
    
    # Block reconciler instantiation duplicated between Alice and Bob
    block_reconciler = BlockReconciler(...)
    
    # Baseline-specific message construction
    for block_id, alice_block in enumerate(alice_blocks):
        syndrome_block = encode_block_from_payload(...)
        yield from self._ordered_socket.send(
            MessageType.SYNDROME,
            {
                "kind": "baseline",  # Hardcoded protocol type
                "payload_length": payload_len,
                "syndrome": syndrome_block.syndrome.tobytes().hex(),
                ...
            },
        )
```

**Impact:**
- Adding Blind protocol requires modifying both `alice.py` and `bob.py`
- Message sequence is implicit and scattered across role classes
- No clear interface between protocol logic and network I/O

**Bob Phase III Implementation (bob.py:130-200):**

```python
def _run_protocol(self, context):
    # Duplicated matrix manager instantiation
    matrix_manager = MatrixManager.from_directory(recon_constants.LDPC_MATRICES_PATH)
    
    for block_id, start in enumerate(...):
        msg0 = yield from self._ordered_socket.recv(MessageType.SYNDROME)
        kind = str(msg0.get("kind"))
        
        if kind == "baseline":
            # Baseline-specific decoding logic embedded here
            ...
        # No "blind" case implemented!
```

**Observation:** Bob's code has no handler for `kind == "blind"`, making Blind protocol impossible without significant refactoring.

#### 2.4 Incomplete Blind State Machine

**File:** [blind_manager.py](caligo/reconciliation/blind_manager.py)

The `BlindReconciliationManager` class exists but is:
1. Not integrated with the orchestrator
2. Uses a simplified interface incompatible with the on-wire protocol
3. Does not implement the syndrome reuse pattern from Martinez-Mateo (Theoretical Report v2, Theorem 4.1)

**Evidence:**

```python
# blind_manager.py:100-115 - State tracks converged/syndrome_errors but not LLRs
@dataclass
class BlindIterationState:
    iteration: int = 0
    n_punctured: int = 0
    n_shortened: int = 0
    shortened_values: List[np.ndarray] = field(default_factory=list)
    # Missing: persistent LLR array for decoder state across iterations
```

**Critical Missing Requirements (from Theoretical Report v2 §4):**

1. **Syndrome Reuse:** The syndrome $\mathbf{s} = H \cdot \mathbf{x}^+$ is computed and transmitted **exactly once** (Theorem 4.1). Subsequent iterations only reveal shortened values.

2. **LLR Persistence:** Bob must maintain decoder LLR state across network round-trips. Revealed bits update LLRs from $0 \to \pm\infty$ without recomputing the syndrome.

3. **Leakage Accounting:** Per Corollary 4.1:
   $$
   \mathsf{leak}_{\mathsf{Blind}} = (1 - R_0) \cdot n + h + \sum_{i=2}^{t} \Delta_i
   $$
   where $\Delta_i$ is the number of symbols revealed in iteration $i$.

4. **NSM-Gated Heuristic:** The Blind protocol should accept an optional heuristic QBER estimate from `utils/math.py::compute_qber_erven` for:
   - Permissive starting-rate cap (small pre-shortening $s_1 \geq 0$)
   - Restrictive iteration budget ($t=3$ default, $t>3$ only for high QBER)

#### 2.5 Summary: Current State Deficiencies

| Deficiency | Severity | Files Affected | Theoretical Reference |
|------------|----------|----------------|----------------------|
| Random puncturing causes high-rate failures | Critical | `ldpc_encoder.py`, `matrix_manager.py` | §2.2.2c [3] |
| Missing Regime B (ACE-Guided) puncturing | Critical | `generate_puncture_patterns.py` | §2.2.3 [4] |
| Untainted saturation limit ($R \approx 0.625$) not addressed | Critical | `matrix_manager.py` | Theorem 2.2 |
| Rate selector bypassed to single rate | High | `rate_selector.py` | §3.2 [1] |
| God Object orchestrator | Medium | `orchestrator.py` | - |
| Protocol logic in role classes | High | `alice.py`, `bob.py` | - |
| Blind protocol not integrated | High | `blind_manager.py` | §4 [2] |
| No circuit-breaker for leakage | Medium | `leakage_tracker.py` | §1.2 |
| Missing syndrome reuse (Blind) | High | `blind_manager.py` | Theorem 4.1 |
| Slow Python decoding (NSM timing) | High | `ldpc_decoder.py` | - |

---

## Part II: Target Architecture

### 3. Design Principles

1. **Single Mother Code:** One $R_0=0.5$ ACE-PEG matrix with offline-generated **Hybrid Pattern Library**

2. **Hybrid-Native & JIT-First:** The architecture explicitly acknowledges the **Untainted Saturation Limit** ($R \approx 0.625$, Theoretical Report v2 Theorem 2.2). The `MotherCodeManager` manages a Hybrid Pattern Library (Untainted for Regime A + ACE-Guided for Regime B). All heavy computation (Encoding/Decoding) is offloaded to Numba JIT kernels; Python handles only control flow.

3. **Strategy Pattern:** Protocol-agnostic orchestration with injectable Baseline/Blind strategies

4. **Explicit State Machines:** Blind decoder maintains LLR state across iterations with "Hot-Start" capability

5. **Strict Leakage Boundaries:** Every component that leaks information reports to a centralized tracker with circuit-breaker enforcement

6. **Fine-Grained Rate Steps:** $\Delta R = 0.01$ (approximately 40 pattern files) maximizes reconciliation efficiency $f$

7. **Testability:** Small, focused classes with clear interfaces

### 4. Component Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         CALIGO PROTOCOL LAYER                           │
│  ┌─────────────────┐                           ┌─────────────────┐      │
│  │  AliceProgram   │                           │   BobProgram    │      │
│  │  (Role)         │                           │   (Role)        │      │
│  └────────┬────────┘                           └────────┬────────┘      │
│           │ delegates to                                │ delegates to  │
│           ▼                                             ▼               │
│  ┌────────────────────────────────────────────────────────────────┐     │
│  │              ReconciliationSession (Context)                   │     │
│  │  - Manages per-session state                                   │     │
│  │  - Coordinates strategy execution                              │     │
│  │  - Tracks cumulative leakage                                   │     │
│  └────────┬───────────────────────────────────────────┬───────────┘     │
│           │                                           │                 │
│           │ ┌─────────────────────────────────────────┼───────────┐     │
│           │ │        ReconciliationStrategy (ABC)     │           │     │
│           ▼ │                                         ▼           │     │
│  ┌─────────────────┐                       ┌─────────────────────┐│     │
│  │ BaselineStrategy│                       │   BlindStrategy     ││     │
│  │                 │                       │                     ││     │
│  │ - Single-shot   │                       │ - Iterative reveal  ││     │
│  │ - QBER required │                       │ - LLR persistence   ││     │
│  └────────┬────────┘                       └─────────┬───────────┘│     │
│           │                                          │            │     │
│           └──────────────────┬───────────────────────┘            │     │
│                              │                                    │     │
└──────────────────────────────┼────────────────────────────────────┘     │
                               │                                          │
┌──────────────────────────────┼──────────────────────────────────────────│
│                              │                                          │
│                  RECONCILIATION ENGINE LAYER                            │
│                              │                                          │
│           ┌──────────────────┴──────────────────┐                       │
│           ▼                                     ▼                       │
│  ┌─────────────────────┐             ┌─────────────────────┐            │
│  │  MotherCodeManager  │             │   LeakageTracker    │            │
│  │                     │             │   (Circuit Breaker) │            │
│  │  - Single R=0.5 H   │             │                     │            │
│  │  - Hybrid patterns  │             │  - Cumulative sum   │            │
│  │  - Numba topology   │             │  - Budget check     │            │
│  └──────────┬──────────┘             │  - SecurityError    │            │
│             │                        └─────────────────────┘            │
│             ▼                                                           │
│  ┌─────────────────────────────────────────────────────────┐            │
│  │                    LDPCCodec                            │            │
│  │  (Thin JIT Interface Facade)                            │            │
│  │  - encode_block(): Numba bit-packed SpMV kernel         │            │
│  │  - decode_baseline(): Virtual Graph decoder kernel      │            │
│  │  - decode_blind(): Hot-Start stateful decoder kernel    │            │
│  └─────────────────────────────────────────────────────────┘            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.1 Numba-Accelerated Online Architecture

To satisfy the strict timing constraints of the Noisy Storage Model (NSM) and the throughput requirements of real-time OT, the Python layer must act purely as a **control plane**. The heavy computational lifting for reconciliation—specifically graph traversal, GF(2) arithmetic, and belief propagation—must be offloaded to **Just-In-Time (JIT) compiled kernels** via Numba.

#### 4.1.1 The "Baked" Mother Code Strategy

The core architectural shift is treating the Mother Code not as a generic matrix object, but as **static, compile-time data**.

*   **Pre-Compilation:** Upon instantiation, the `MotherCodeManager` does not merely load the CSR matrix; it "bakes" the graph topology into flat, structure-of-arrays (SoA) buffers optimized for cache locality. These arrays (e.g., `check_to_var_indices`, `var_to_check_indices`) are pinned in memory.
*   **Kernel Injection:** All Numba kernels accept these topology arrays as read-only arguments. Because the mother code matrix never changes during a session, the JIT compiler can perform aggressive loop unrolling and vectorization specific to the graph's edge distribution.

#### 4.1.2 Shared Component: High-Throughput Encoding (Bit-Packed SpMV)

The generic `scipy.sparse.dot` operation is a major bottleneck because it performs integer arithmetic followed by a modulo operation, wasting memory bandwidth and CPU cycles.

*   **Bit-Packed Kernel:** The encoder will use a custom Numba kernel that operates on **bit-packed integers** (`uint64`).
    *   **Input:** The 4096-bit payload is compressed into an array of 64 `uint64` words.
    *   **Logic:** The sparse matrix-vector multiplication (SpMV) is rewritten to use bitwise `AND` instructions combined with hardware `POPCNT` (population count) instructions. Parity is determined by `popcount(row_mask & data_chunk) & 1`.
    *   **Gain:** This approach processes 64 bits per CPU cycle per stream, offering a 10x-50x speedup over standard arithmetic, crucial for high-rate syndrome computation.

#### 4.1.3 Protocol-Specific Optimizations

The Numba architecture allows us to implement the distinct logic of Baseline and Blind protocols as specialized kernel pathways without recompiling the underlying graph structure.

**A. Baseline Protocol: The "Virtual Graph" Kernel**

In Baseline reconciliation, the effective code rate changes per block ($0.5 \to 0.9$). Naive implementations would slice the matrix, forcing expensive re-allocation or re-compilation.

*   **Virtual Graph Approach:** We use a single kernel that always operates on the full $n=4096$ mother graph.
*   **Mask-Based LLR Initialization:** Instead of reshaping the matrix, we pass the **Puncturing Pattern** (a boolean mask) to the kernel.
    *   The kernel initializes the Log-Likelihood Ratios (LLRs).
    *   If `pattern[i] == 1` (Punctured): The LLR is set to `0.0` (Erasure).
    *   If `pattern[i] == 0` (Payload): The LLR is set to the channel value derived from QBER.
*   **Execution:** The BP decoder runs on the full graph. Punctured nodes naturally act as "neutral" message carriers. This allows instantaneous rate switching with zero overhead—rate adaptation becomes purely a memory initialization step.

**B. Blind Protocol: The Stateful "Hot-Start" Kernel**

Blind reconciliation requires decoding the same block multiple times as Alice reveals bits. Standard decoders reset internal state (messages) to zero every time, discarding valuable convergence progress.

*   **Message Persistence (Hot-Start):** The Blind kernel signature is modified to accept `in_out_messages` as an argument.
    *   **Iteration 1:** Messages are initialized to zero.
    *   **Iteration $i > 1$:** The kernel receives the *output messages* from Iteration $i-1$. The Belief Propagation algorithm resumes exactly where it left off, but with updated "priors" for the newly revealed bits. This drastically reduces the number of iterations required for convergence in later steps.
*   **Dynamic Pruning (The "Freeze" Optimization):**
    *   As bits are revealed (shortened), their LLRs become $\pm \infty$.
    *   The kernel includes a "Freeze Check": if a node's intrinsic reliability exceeds a threshold (indicating it is a revealed shortened bit), the kernel **skips the expensive `tanh/arctanh` updates** for that node's outgoing edges.
    *   **Impact:** As the Blind protocol progresses and reveals more bits, the effective computational load *decreases*, countering the latency cost of multiple round-trips.

### 5. Strategy Pattern Interface

#### 5.1 Abstract Base Class

```python
# caligo/reconciliation/strategies/base.py

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generator, Any, Optional, Dict
import numpy as np


@dataclass(frozen=True)
class ReconciliationContext:
    """
    Immutable context passed to strategies.
    
    Attributes
    ----------
    session_id : int
        Unique session identifier for logging/debugging.
    frame_size : int
        LDPC frame size (n = 4096).
    mother_rate : float
        Mother code rate (R_0 = 0.5).
    max_iterations : int
        BP decoder max iterations.
    hash_bits : int
        Verification hash length (typically 64 bits).
    f_crit : float
        Target efficiency threshold (f ∈ [1.05, 1.2]).
    qber_prior : float
        NSM heuristic QBER for Blind (from compute_qber_erven).
    modulation_delta : float
        Modulation budget δ = (p+s)/n (determines R_eff range).
    """
    session_id: int
    frame_size: int
    mother_rate: float
    max_iterations: int
    hash_bits: int
    f_crit: float
    qber_prior: float
    modulation_delta: float = 0.1


@dataclass
class BlockResult:
    """
    Result of single block reconciliation.
    
    Per Theoretical Report v2 §1.2, leakage accounting must be exact:
    leak_EC = syndrome_leakage + hash_leakage + revealed_leakage
    
    Attributes
    ----------
    corrected_payload : np.ndarray
        Corrected bits (payload only, excluding padding).
    verified : bool
        True if hash verification passed.
    converged : bool
        True if BP decoder converged.
    iterations_used : int
        BP iterations until convergence (or max).
    syndrome_leakage : int
        Syndrome bits leaked = (1 - R_0) × n (constant per block).
    revealed_leakage : int
        Revealed bits leaked (Blind only, 0 for Baseline).
    hash_leakage : int
        Hash bits leaked.
    retry_count : int
        Number of blind iterations (1 for Baseline).
    effective_rate : float
        Effective code rate R_eff used for this block.
    """
    corrected_payload: np.ndarray
    verified: bool
    converged: bool
    iterations_used: int
    syndrome_leakage: int
    revealed_leakage: int
    hash_leakage: int
    retry_count: int
    effective_rate: float = 0.5
    
    @property
    def total_leakage(self) -> int:
        """
        Total leakage for this block.
        
        Per Theoretical Report v2 Eq. (leak_EC):
        leak = |Σ| + |Hash| + |Revealed|
        """
        return self.syndrome_leakage + self.revealed_leakage + self.hash_leakage


class ReconciliationStrategy(ABC):
    """
    Abstract base for reconciliation strategies.
    
    Implements the Strategy Pattern to support both Baseline [1] and
    Blind [2] reconciliation protocols with a unified interface.
    
    Strategies are stateless protocol implementations. Per-block state
    (e.g., LLRs across Blind iterations) is managed via yielded messages
    to the network layer.
    
    The generator-based interface supports SquidASM's cooperative scheduling.
    
    References
    ----------
    [1] Elkouss et al. (2010), "Rate Compatible Protocol for Information Reconciliation"
    [2] Martinez-Mateo et al. (2012), "Blind Reconciliation"
    """
    
    @property
    @abstractmethod
    def requires_qber_estimation(self) -> bool:
        """
        Whether this strategy requires QBER pre-estimation.
        
        Per Theoretical Report v2:
        - Baseline (§3): Requires explicit QBER sampling (t disclosed bits)
        - Blind (§4): Uses optional heuristic from compute_qber_erven
        
        Returns
        -------
        bool
            True for Baseline (needs rate selection), False for Blind.
        """
        pass
    
    @abstractmethod
    def alice_reconcile_block(
        self,
        payload: np.ndarray,
        ctx: ReconciliationContext,
        block_id: int,
        qber_estimate: Optional[float] = None,
    ) -> Generator[Dict[str, Any], Dict[str, Any], BlockResult]:
        """
        Alice-side block reconciliation.
        
        Parameters
        ----------
        payload : np.ndarray
            Alice's payload bits (uint8).
        ctx : ReconciliationContext
            Session context.
        block_id : int
            Block identifier.
        qber_estimate : float, optional
            QBER estimate (required for Baseline, optional heuristic for Blind).
            
        Yields
        ------
        Dict[str, Any]
            Outgoing message to Bob (send to network).
            
        Receives
        --------
        Dict[str, Any]
            Response from Bob (received from network).
            
        Returns
        -------
        BlockResult
            Reconciliation result for this block.
        """
        pass
    
    @abstractmethod
    def bob_reconcile_block(
        self,
        payload: np.ndarray,
        ctx: ReconciliationContext,
        block_id: int,
    ) -> Generator[Dict[str, Any], Dict[str, Any], BlockResult]:
        """
        Bob-side block reconciliation.
        
        Parameters
        ----------
        payload : np.ndarray
            Bob's received payload bits (uint8).
        ctx : ReconciliationContext
            Session context.
        block_id : int
            Block identifier.
            
        Yields
        ------
        Dict[str, Any]
            Outgoing message to Alice (ACK/NACK for Blind).
            
        Receives
        --------
        Dict[str, Any]
            Incoming message from Alice (syndrome, revealed bits).
            
        Returns
        -------
        BlockResult
            Reconciliation result for this block.
        """
        pass
```

#### 5.2 Baseline Strategy Implementation

```python
# caligo/reconciliation/strategies/baseline.py

class BaselineStrategy(ReconciliationStrategy):
    """
    Elkouss et al. (2010) rate-compatible reconciliation.
    
    Single-shot syndrome transmission with QBER-based rate selection.
    
    Per Theoretical Report v2 §3:
    - Requires explicit QBER estimation via sampling (t disclosed bits)
    - Rate selection: R = 1 - f(p*) × h(p*)
    - Frame construction via g(x, σ, π) function
    - Fixed syndrome leakage: |Σ| = (1 - R_0) × n
    
    The QBER estimate must be obtained from sifting/qber.py BEFORE
    calling this strategy. The estimate is passed via qber_estimate parameter.
    """
    
    def __init__(
        self,
        mother_code: MotherCodeManager,
        codec: LDPCCodec,
        leakage_tracker: LeakageTracker,
    ) -> None:
        self._mother_code = mother_code
        self._codec = codec
        self._leakage_tracker = leakage_tracker
    
    @property
    def requires_qber_estimation(self) -> bool:
        return True
    
    def alice_reconcile_block(
        self,
        payload: np.ndarray,
        ctx: ReconciliationContext,
        block_id: int,
        qber_estimate: Optional[float] = None,
    ) -> Generator[Dict[str, Any], Dict[str, Any], BlockResult]:
        """
        Alice computes syndrome and sends to Bob.
        
        Per Theoretical Report v2 §3.2-3.3:
        1. Select rate based on QBER: R = 1 - f(p*) × h(p*)
        2. Compute (s, p) from (R, δ) using Eq. (7)
        3. Construct frame x+ = g(x, σ, π)
        4. Compute syndrome s = H × x+ mod 2
        """
        if qber_estimate is None:
            raise ValueError(
                "Baseline requires QBER estimate from sifting phase. "
                "Ensure qber.py::estimate_qber() was called."
            )
        
        # 1. Select effective rate based on QBER
        # Per Theoretical Report v2 Eq. (3.2): R = 1 - f(p*) × h(p*)
        effective_rate = self._select_rate(qber_estimate, ctx.f_crit)
        
        # 2. Get hybrid puncture pattern for this rate
        # Pattern comes from Hybrid Library (Untainted for R ≤ 0.625, ACE for R > 0.625)
        pattern = self._mother_code.get_pattern(effective_rate)
        
        # 3. Construct frame x+ = g(x, σ, π)
        # Per Theoretical Report v2 §3.2 Step 4
        frame = self._construct_frame(payload, pattern, ctx.frame_size)
        
        # 4. Compute syndrome using Numba bit-packed SpMV kernel
        syndrome = self._codec.encode(frame, pattern)
        
        # 5. Compute verification hash
        hash_value = compute_hash(payload, seed=block_id)
        
        # 6. Record leakage: |Σ| + |Hash|
        # Per Theoretical Report v2 Theorem 3.1
        self._leakage_tracker.record(
            block_id=block_id,
            syndrome_bits=len(syndrome),
            hash_bits=ctx.hash_bits,
            revealed_bits=0,
        )
        
        # 7. Send syndrome message to Bob
        response = yield {
            "kind": "baseline",
            "block_id": block_id,
            "syndrome": syndrome,
            "pattern_id": effective_rate,  # Bob looks up pattern by rate
            "payload_length": len(payload),
            "hash_value": hash_value,
            "qber_channel": qber_estimate,  # For Bob's LLR construction
        }
        
        # 8. Check Bob's response
        verified = response.get("verified", False)
        
        return BlockResult(
            corrected_payload=payload,  # Alice already has correct bits
            verified=verified,
            converged=True,
            iterations_used=0,
            syndrome_leakage=len(syndrome),
            revealed_leakage=0,
            hash_leakage=ctx.hash_bits,
            retry_count=1,
            effective_rate=effective_rate,
        )
    
    def _select_rate(self, qber: float, f_crit: float) -> float:
        """
        Select effective rate using reconciliation efficiency model.
        
        Per Theoretical Report v2 §3.2:
        R = 1 - f(p*) × h(p*)
        
        where h(·) is binary entropy and f(·) ≥ 1 is efficiency.
        """
        from caligo.utils.math import binary_entropy
        
        if qber <= 0 or qber >= 0.5:
            return 0.5  # Fallback to mother rate
        
        h_qber = binary_entropy(qber)
        target_rate = 1.0 - f_crit * h_qber
        
        # Clamp to achievable range [R_min, R_max]
        # Per Theoretical Report v2 Definition 2.2
        delta = 0.1  # Default modulation parameter
        r_min = (0.5 - delta) / (1 - delta)
        r_max = 0.9  # Hybrid library upper limit
        
        return max(r_min, min(target_rate, r_max))
    
    def bob_reconcile_block(
        self,
        payload: np.ndarray,
        ctx: ReconciliationContext,
        block_id: int,
    ) -> Generator[Dict[str, Any], Dict[str, Any], BlockResult]:
        """
        Bob receives syndrome and decodes using Virtual Graph kernel.
        
        Per Theoretical Report v2 §3.3:
        - Construct frame y+ = g(y, σ, π)
        - Initialize LLRs with three-state builder
        - Decode using BP against syndrome s
        """
        # 1. Receive syndrome message from Alice
        msg = yield {}  # Initial yield to receive first message
        
        if msg.get("kind") != "baseline":
            raise ProtocolError(f"Expected baseline, got {msg.get('kind')}")
        
        syndrome = np.array(msg["syndrome"], dtype=np.uint8)
        pattern_id = msg["pattern_id"]
        payload_length = msg["payload_length"]
        expected_hash = msg["hash_value"]
        qber_channel = msg["qber_channel"]
        
        # 2. Get hybrid puncture pattern
        pattern = self._mother_code.get_pattern(pattern_id)
        
        # 3. Construct frame and build LLRs
        frame = self._construct_frame(payload, pattern, ctx.frame_size)
        llr = build_three_state_llr(
            received_bits=frame,
            qber=qber_channel,
            puncture_mask=pattern,
            shorten_mask=None,
            shorten_values=None,
        )
        
        # 4. Decode using Virtual Graph kernel (full mother graph, pattern as mask)
        result = self._codec.decode_baseline(syndrome, llr, pattern)
        
        # 5. Extract corrected payload
        corrected_payload = result.corrected_bits[:payload_length]
        
        # 6. Verify hash
        computed_hash = compute_hash(corrected_payload, seed=block_id)
        verified = (computed_hash == expected_hash)
        
        # 7. Send response to Alice
        yield {"verified": verified, "converged": result.converged}
        
        return BlockResult(
            corrected_payload=corrected_payload,
            verified=verified,
            converged=result.converged,
            iterations_used=result.iterations,
            syndrome_leakage=len(syndrome),
            revealed_leakage=0,
            hash_leakage=ctx.hash_bits,
            retry_count=1,
            effective_rate=pattern_id,
        )
```

#### 5.3 Blind Strategy Implementation with State Machine

```python
# caligo/reconciliation/strategies/blind.py

@dataclass
class BlindDecoderState:
    """
    Persistent decoder state across Blind iterations.
    
    Per Theoretical Report v2 §4.1 (Theorem 4.1), the syndrome is computed
    ONCE and reused. This state maintains LLR arrays and messages across
    network round-trips for the "Hot-Start" decoder kernel.
    
    Attributes
    ----------
    llr : np.ndarray
        Current LLR array (updated as bits are revealed).
    messages : np.ndarray
        BP edge messages for Hot-Start continuation.
    puncture_indices : np.ndarray
        Originally punctured positions (from hybrid pattern).
    shortened_indices : np.ndarray
        Currently shortened positions (grows each iteration).
    shortened_values : np.ndarray
        Values at shortened positions (grows each iteration).
    frozen_mask : np.ndarray
        Boolean mask for "freeze" optimization (revealed bits).
    iteration : int
        Current iteration number.
    syndrome : np.ndarray
        Fixed syndrome (computed once by Alice).
    """
    llr: np.ndarray
    messages: np.ndarray  # For Hot-Start persistence
    puncture_indices: np.ndarray
    shortened_indices: np.ndarray
    shortened_values: np.ndarray
    frozen_mask: np.ndarray
    iteration: int
    syndrome: np.ndarray


class BlindStrategy(ReconciliationStrategy):
    """
    Martinez-Mateo et al. (2012) blind reconciliation.
    
    Iterative protocol without QBER pre-estimation. Bob maintains
    decoder state across iterations; Alice progressively reveals
    punctured values (shortening them).
    
    Per Theoretical Report v2 §4:
    - Syndrome computed and transmitted ONCE (Theorem 4.1)
    - Leakage: leak_Blind = (1-R_0)×n + h + Σ Δ_i (Corollary 4.1)
    - Revelation order fixed by hybrid pattern (security requirement)
    - Optional heuristic QBER from compute_qber_erven for gating
    
    Architecture Notes:
    - Uses "Hot-Start" kernel: messages persist across iterations
    - Uses "Freeze" optimization: revealed bits skip tanh/arctanh
    """
    
    def __init__(
        self,
        mother_code: MotherCodeManager,
        codec: LDPCCodec,
        leakage_tracker: LeakageTracker,
        max_blind_iterations: int = 3,
        modulation_fraction: float = 0.1,
    ) -> None:
        self._mother_code = mother_code
        self._codec = codec
        self._leakage_tracker = leakage_tracker
        self._max_iterations = max_blind_iterations
        self._delta = modulation_fraction
    
    @property
    def requires_qber_estimation(self) -> bool:
        # Per Theoretical Report v2 §4: Blind does NOT require QBER pre-estimation
        # but can accept an optional heuristic for NSM-gated optimization
        return False
    
    def alice_reconcile_block(
        self,
        payload: np.ndarray,
        ctx: ReconciliationContext,
        block_id: int,
        qber_estimate: Optional[float] = None,
    ) -> Generator[Dict[str, Any], Dict[str, Any], BlockResult]:
        """
        Alice sends syndrome once, then iteratively reveals punctured values.
        
        Per Theoretical Report v2 §4.3:
        - Iteration 1: p=d, s=0 (all modulation bits punctured)
        - Iteration i≥2: Reveal Δ bits, update p←p-Δ, s←s+Δ
        - Termination: Success OR p=0
        
        NSM-Gated Variant (from §4.3):
        - If qber_estimate provided (from compute_qber_erven), use for:
          a) Permissive starting-rate cap (small s_1 > 0)
          b) Restrictive iteration budget (t=3 default, t>3 for high QBER)
        """
        # 1. Compute modulation parameters
        d = int(self._delta * ctx.frame_size)  # Total modulation bits
        
        # 2. NSM-gated starting adjustment (optional)
        # Per Theoretical Report v2 §4.3: permissive starting-rate cap
        initial_shortened = 0
        if qber_estimate is not None and qber_estimate > 0.05:
            # Small pre-shortening for high-QBER channels
            initial_shortened = self._compute_initial_shortening(
                qber_estimate, ctx.f_crit, d
            )
        
        # 3. Get revelation order from hybrid pattern (deterministic)
        # Per Theoretical Report v2 §4.3: order fixed at setup time
        puncture_indices = self._mother_code.get_modulation_indices(d)
        
        # 4. Compute step size: Δ = d/t
        delta_step = max(1, d // self._max_iterations)
        
        # 5. Construct frame with all punctured positions (highest rate)
        frame = self._construct_frame_with_padding(
            payload, puncture_indices, ctx.frame_size
        )
        padding_values = frame[puncture_indices]  # Save for potential reveal
        
        # 6. Compute syndrome ONCE
        # Per Theoretical Report v2 Theorem 4.1: syndrome reuse
        syndrome = self._codec.encode(frame, np.zeros(ctx.frame_size, dtype=np.uint8))
        hash_value = compute_hash(payload, seed=block_id)
        
        # 7. Record initial syndrome leakage
        self._leakage_tracker.record(
            block_id=block_id,
            syndrome_bits=len(syndrome),
            hash_bits=ctx.hash_bits,
            revealed_bits=initial_shortened,
        )
        
        # 8. Send initial syndrome + any initial shortening
        revealed_indices = puncture_indices[:initial_shortened] if initial_shortened > 0 else np.array([], dtype=np.int64)
        revealed_values = padding_values[:initial_shortened] if initial_shortened > 0 else np.array([], dtype=np.uint8)
        
        response = yield {
            "kind": "blind",
            "block_id": block_id,
            "syndrome": syndrome,
            "puncture_indices": puncture_indices,
            "payload_length": len(payload),
            "hash_value": hash_value,
            "qber_prior": ctx.qber_prior,
            "iteration": 1,
            "revealed_indices": revealed_indices,
            "revealed_values": revealed_values,
        }
        
        # 9. Iterative reveal loop
        iteration = 1
        total_revealed = initial_shortened
        
        while not response.get("verified") and iteration < self._max_iterations:
            if response.get("converged") and not response.get("verified"):
                # Converged but wrong codeword - verification failed
                break
            
            iteration += 1
            
            # Select next batch of punctured bits to reveal (shorten)
            # Per Theoretical Report v2 §4.3: fixed order from P_ord
            reveal_start = initial_shortened + (iteration - 1) * delta_step
            reveal_end = min(initial_shortened + iteration * delta_step, d)
            reveal_indices = puncture_indices[reveal_start:reveal_end]
            reveal_values = padding_values[reveal_start:reveal_end]
            
            total_revealed += len(reveal_indices)
            
            # Record additional leakage (Δ_i bits)
            # Per Theoretical Report v2 Corollary 4.1
            self._leakage_tracker.record_reveal(
                block_id=block_id,
                iteration=iteration,
                revealed_bits=len(reveal_indices),
            )
            
            # Send reveal message (NO new syndrome!)
            response = yield {
                "kind": "blind_reveal",
                "block_id": block_id,
                "iteration": iteration,
                "revealed_indices": reveal_indices,
                "revealed_values": reveal_values,
            }
        
        verified = response.get("verified", False)
        converged = response.get("converged", False)
        
        return BlockResult(
            corrected_payload=payload,
            verified=verified,
            converged=converged,
            iterations_used=0,
            syndrome_leakage=len(syndrome),
            revealed_leakage=total_revealed,
            hash_leakage=ctx.hash_bits,
            retry_count=iteration,
            effective_rate=self._compute_effective_rate(d, total_revealed),
        )
    
    def _compute_initial_shortening(
        self, qber: float, f_crit: float, d: int
    ) -> int:
        """
        Compute permissive initial shortening s_1 for NSM-gated variant.
        
        Per Theoretical Report v2 §4.3: choose s_1 as smallest value
        that brings initial effective rate below conservative cap.
        """
        from caligo.utils.math import binary_entropy
        
        h_qber = binary_entropy(qber)
        target_rate = 1.0 - f_crit * h_qber
        
        # Very conservative: start closer to target rate
        # This avoids obviously infeasible first attempts
        return min(int(d * 0.1), d // self._max_iterations)
    
    def _compute_effective_rate(self, d: int, shortened: int) -> float:
        """Compute effective rate given shortening."""
        # R_eff = (R_0 - s/n) / (1 - p/n - s/n)
        # With p + s = d (initially), as we shorten: p decreases, s increases
        p = d - shortened
        s = shortened
        n = 4096  # Frame size
        return (0.5 - s/n) / (1 - (p + s)/n)
    
    def bob_reconcile_block(
        self,
        payload: np.ndarray,
        ctx: ReconciliationContext,
        block_id: int,
    ) -> Generator[Dict[str, Any], Dict[str, Any], BlockResult]:
        """
        Bob decodes iteratively, updating LLRs as Alice reveals values.
        
        Per Theoretical Report v2 §4:
        - Uses Hot-Start kernel: messages persist across iterations
        - Uses Freeze optimization: revealed bits (LLR=±∞) skip updates
        - Same syndrome used for all iterations (Theorem 4.1)
        """
        # 1. Receive initial syndrome message
        msg = yield {}
        
        if msg.get("kind") != "blind":
            raise ProtocolError(f"Expected blind, got {msg.get('kind')}")
        
        syndrome = np.array(msg["syndrome"], dtype=np.uint8)
        puncture_indices = np.array(msg["puncture_indices"], dtype=np.int64)
        payload_length = msg["payload_length"]
        expected_hash = msg["hash_value"]
        qber_prior = msg["qber_prior"]
        initial_revealed_indices = np.array(msg.get("revealed_indices", []), dtype=np.int64)
        initial_revealed_values = np.array(msg.get("revealed_values", []), dtype=np.uint8)
        
        # 2. Construct Bob's frame (with his noisy payload + erasures at puncture positions)
        frame = np.zeros(ctx.frame_size, dtype=np.uint8)
        non_puncture_mask = np.ones(ctx.frame_size, dtype=bool)
        non_puncture_mask[puncture_indices] = False
        frame[non_puncture_mask] = payload
        
        # 3. Initialize state for Hot-Start decoder
        state = BlindDecoderState(
            llr=build_three_state_llr(
                received_bits=frame,
                qber=qber_prior,
                puncture_mask=np.isin(np.arange(ctx.frame_size), puncture_indices),
                shorten_mask=np.isin(np.arange(ctx.frame_size), initial_revealed_indices) if len(initial_revealed_indices) > 0 else None,
                shorten_values=initial_revealed_values if len(initial_revealed_values) > 0 else None,
            ),
            messages=np.zeros(self._codec.num_edges, dtype=np.float64),  # Initialize messages
            puncture_indices=puncture_indices,
            shortened_indices=initial_revealed_indices.copy(),
            shortened_values=initial_revealed_values.copy(),
            frozen_mask=np.zeros(ctx.frame_size, dtype=bool),
            iteration=1,
            syndrome=syndrome,
        )
        
        # Mark initially revealed bits as frozen
        if len(initial_revealed_indices) > 0:
            state.frozen_mask[initial_revealed_indices] = True
        
        # 4. Initial decode attempt using Hot-Start kernel
        result = self._codec.decode_blind(
            syndrome=syndrome,
            llr=state.llr,
            messages=state.messages,
            frozen_mask=state.frozen_mask,
        )
        state.messages = result.messages  # Preserve for next iteration
        
        corrected_payload = result.corrected_bits[:payload_length]
        computed_hash = compute_hash(corrected_payload, seed=block_id)
        verified = (computed_hash == expected_hash)
        
        if result.converged and verified:
            # Success on first try!
            yield {"converged": True, "verified": True}
            return BlockResult(
                corrected_payload=corrected_payload,
                verified=True,
                converged=True,
                iterations_used=result.iterations,
                syndrome_leakage=len(syndrome),
                revealed_leakage=len(initial_revealed_indices),
                hash_leakage=ctx.hash_bits,
                retry_count=1,
                effective_rate=0.5 / (1 - len(puncture_indices)/ctx.frame_size),
            )
        
        # 5. Iterative reveal loop
        total_revealed = len(initial_revealed_indices)
        
        while True:
            # Send NACK and wait for reveal
            msg = yield {"converged": result.converged, "verified": verified}
            
            if msg.get("kind") != "blind_reveal":
                break  # Protocol ended (max iterations)
            
            # Update state with revealed values
            # Per Theoretical Report v2 §4.2: LLR update 0 → ±∞
            new_indices = np.array(msg["revealed_indices"], dtype=np.int64)
            new_values = np.array(msg["revealed_values"], dtype=np.uint8)
            
            state.shortened_indices = np.concatenate([state.shortened_indices, new_indices])
            state.shortened_values = np.concatenate([state.shortened_values, new_values])
            state.iteration = msg["iteration"]
            total_revealed += len(new_indices)
            
            # Update LLRs for revealed positions (0 → ±∞)
            # Update frozen mask for Freeze optimization
            LLR_INFINITY = 100.0
            for idx, val in zip(new_indices, new_values):
                state.llr[idx] = (+LLR_INFINITY if val == 0 else -LLR_INFINITY)
                state.frozen_mask[idx] = True
            
            # Re-decode with Hot-Start kernel (same syndrome, preserved messages)
            # Per Theoretical Report v2 Theorem 4.1: syndrome reuse
            result = self._codec.decode_blind(
                syndrome=syndrome,
                llr=state.llr,
                messages=state.messages,  # Resume from previous state
                frozen_mask=state.frozen_mask,
            )
            state.messages = result.messages
            
            corrected_payload = result.corrected_bits[:payload_length]
            computed_hash = compute_hash(corrected_payload, seed=block_id)
            verified = (computed_hash == expected_hash)
            
            if result.converged and verified:
                yield {"converged": True, "verified": True}
                break
        
        return BlockResult(
            corrected_payload=corrected_payload,
            verified=verified,
            converged=result.converged,
            iterations_used=result.iterations,
            syndrome_leakage=len(syndrome),
            revealed_leakage=total_revealed,
            hash_leakage=ctx.hash_bits,
            retry_count=state.iteration,
            effective_rate=self._compute_effective_rate(len(puncture_indices), total_revealed),
        )
```

---

### 5.4 QBER Estimation Wiring: Sifting → Baseline Protocol

This section documents the integration between the sifting phase QBER estimation ([sifting/qber.py](caligo/sifting/qber.py)) and the Baseline reconciliation strategy.

#### 5.4.1 Key Constraint: QBER Estimation is Baseline-Only

**Critical Requirement:** QBER estimation via test bit disclosure is **only performed for the Baseline protocol**. The Blind protocol explicitly skips QBER estimation—this is its core advantage (Martinez-Mateo et al. 2012 [2]).

The gating logic is already implemented in the existing codebase:

**File:** [reconciliation/factory.py](caligo/reconciliation/factory.py#L91-L121)

```python
class ReconciliationType(Enum):
    BASELINE = "baseline"
    BLIND = "blind"
    INTERACTIVE = "interactive"

    @property
    def requires_qber_estimation(self) -> bool:
        """
        Whether this reconciliation type requires prior QBER estimation.
        
        Returns True for BASELINE and INTERACTIVE, False for BLIND.
        """
        return self != ReconciliationType.BLIND
```

**Protocol Layer Enforcement:** [alice.py](caligo/protocol/alice.py#L220-L260)

```python
# _phase2_sifting()
if self.params.reconciliation.requires_qber_estimation:
    # BASELINE: Perform test bit exchange and QBER estimation
    test_indices, key_indices = self._sifter.select_test_subset(...)
    
    yield from self._ordered_socket.send(MessageType.INDEX_LISTS, {...})
    test_outcomes_msg = yield from self._ordered_socket.recv(MessageType.TEST_OUTCOMES)
    
    qber = self._qber_estimator.estimate(
        alice_test_bits=alice_test_bits,
        bob_test_bits=bob_test_bits,
        key_size=len(key_indices),
    )
    qber_estimate = float(qber.observed_qber)
    qber_adjusted = float(qber.adjusted_qber)
else:
    # BLIND: Use heuristic from NSM parameters (no test bit disclosure)
    qber_estimate = float(self.params.nsm_params.qber_conditional)
    qber_adjusted = qber_estimate
```

#### 5.4.2 QBER Estimation Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        QBER ESTIMATION DATA FLOW                            │
│                        (Baseline Protocol Only)                             │
└─────────────────────────────────────────────────────────────────────────────┘

                                  Phase II: Sifting
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  1. Sifter.compute_sifted_key() → SiftingResult                             │
│     • Computes matching_indices (where Alice & Bob bases agree)             │
│     • Partitions into I₀/I₁ for OT                                          │
│                                                                             │
│  2. [BASELINE ONLY] Sifter.select_test_subset() → (test_indices, key_indices)│
│     • Randomly selects ~10% of matching_indices for testing                 │
│     • Remaining indices form the secret key                                 │
│                                                                             │
│  3. [BASELINE ONLY] QBEREstimator.estimate() → QBEREstimate                 │
│     • Input: alice_test_bits, bob_test_bits, key_size                       │
│     • Output:                                                               │
│       ├── observed_qber: e_obs = errors / n_test                            │
│       ├── mu_penalty: μ = √[(n+k)/(n·k) · (k+1)/k · ln(4/ε_sec)]            │
│       └── adjusted_qber: e_adj = e_obs + μ (for rate selection)             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      │ qber_adjusted (float)
                                      ▼
                             Phase III: Reconciliation
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  4. BaselineStrategy.alice_reconcile_block(qber_estimate=qber_adjusted)     │
│     • Rate selection: R = 1 - f × h(qber_adjusted)                          │
│     • Pattern lookup: MotherCodeManager.get_pattern(R)                      │
│     • Syndrome computation: LDPCCodec.encode(frame, pattern)                │
│                                                                             │
│  5. Message sent to Bob: SYNDROME {qber_channel: qber_adjusted, ...}        │
│     • Bob uses qber_channel for LLR construction                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 5.4.3 QBEREstimate Dataclass (Existing Implementation)

**File:** [sifting/qber.py](caligo/sifting/qber.py#L28-L64)

```python
@dataclass
class QBEREstimate:
    """
    Result of QBER estimation.
    
    Attributes
    ----------
    observed_qber : float
        Observed error rate from test bits: e_obs = errors / n_test.
    adjusted_qber : float
        QBER with finite-size penalty: e_adj = e_obs + μ.
        This is the value used for rate selection in Baseline.
    mu_penalty : float
        Statistical penalty μ from Erven et al. (2014) Eq. (2):
        μ = √[(n+k)/(n·k) · (k+1)/k · ln(4/ε_sec)]
    num_test_bits : int
        Number of bits used for estimation (|T|).
    num_errors : int
        Number of errors observed in test set.
    exceeds_hard_limit : bool
        True if adjusted QBER > 22% (security impossible).
    exceeds_warning_limit : bool
        True if adjusted QBER > 11% (conservative limit).
    """
    observed_qber: float
    adjusted_qber: float
    mu_penalty: float
    num_test_bits: int
    num_errors: int
    confidence_level: float = 1.0 - 1e-10
    exceeds_hard_limit: bool = False
    exceeds_warning_limit: bool = False
```

#### 5.4.4 Rate Selection Using adjusted_qber

The `adjusted_qber` (not `observed_qber`) is used for rate selection to ensure conservative operation under finite-size effects:

**File:** [alice.py](caligo/protocol/alice.py#L348-L365)

```python
# _phase3_reconcile()

# Get rate based on adjusted QBER (includes finite-size penalty)
rate = select_rate(
    qber_estimate=qber_adjusted,  # ← Uses adjusted, not observed
    available_rates=matrix_manager.rates,
    f_crit=config.f_crit,
)
```

**File:** [reconciliation/rate_selector.py](caligo/reconciliation/rate_selector.py)

```python
def select_rate(
    qber_estimate: float,
    available_rates: List[float],
    f_crit: float = 1.1,
) -> float:
    """
    Select optimal code rate for given QBER.
    
    Per Theoretical Report v2 §3.2:
    R = 1 - f × h(qber)
    
    where f is the efficiency target (typically 1.05-1.2).
    """
    from caligo.utils.math import binary_entropy
    
    h_qber = binary_entropy(qber_estimate)
    target_rate = 1.0 - f_crit * h_qber
    
    # Find closest available rate
    closest = min(available_rates, key=lambda r: abs(r - target_rate))
    return closest
```

#### 5.4.5 Refactoring Recommendation: Strategy-Level QBER Injection

The current implementation embeds QBER estimation logic in `alice.py`. The refactored architecture should inject QBER via the `ReconciliationContext`:

```python
# Proposed: caligo/reconciliation/strategies/base.py

@dataclass(frozen=True)
class ReconciliationContext:
    """Immutable context passed to strategies."""
    
    # ... existing fields ...
    
    # QBER fields (Baseline-specific, from sifting/qber.py)
    qber_observed: Optional[float] = None   # e_obs (for logging)
    qber_adjusted: Optional[float] = None   # e_adj (for rate selection)
    mu_penalty: Optional[float] = None      # μ (for leakage accounting)
    
    # Heuristic QBER (Blind-specific, from utils/math.py)
    qber_heuristic: Optional[float] = None  # From compute_qber_erven()
    
    def validate_for_baseline(self) -> None:
        """Ensure Baseline has required QBER estimate."""
        if self.qber_adjusted is None:
            raise ValueError(
                "Baseline strategy requires qber_adjusted from sifting phase. "
                "Call QBEREstimator.estimate() before reconciliation."
            )
```

---

### 5.5 Heuristic QBER Wiring: NSM Parameters → Blind Protocol

This section documents the **optional** heuristic QBER integration for the Blind protocol. Unlike Baseline, Blind does not require QBER pre-estimation, but can benefit from an NSM-derived heuristic for **permissive starting-rate optimization**.

#### 5.5.1 The NSM-Gated Variant (Theoretical Report v2 §4.3)

The standard Blind protocol starts with maximum rate (all modulation bits punctured) and iteratively reveals bits until convergence. The **NSM-Gated variant** uses a heuristic QBER to:

1. **Permissive Starting-Rate Cap:** Pre-shorten some bits in iteration 1 to avoid obviously infeasible high rates
2. **Iteration Budget Gating:** Allow extended iterations ($t > 3$) only for high-QBER channels

**Key Distinction:**
- **Baseline:** QBER estimation is **mandatory** (test bits must be disclosed)
- **Blind:** Heuristic QBER is **optional** (no test bits disclosed; uses physical model parameters)

#### 5.5.2 Heuristic QBER Source: `compute_qber_erven()`

**File:** [utils/math.py](caligo/utils/math.py#L177-L220)

The `compute_qber_erven()` function computes expected QBER from NSM physical parameters **without disclosing any test bits**:

```python
def compute_qber_erven(
    fidelity: float,
    detector_error: float,
    detection_efficiency: float,
    dark_count_prob: float,
) -> float:
    """
    Compute total QBER using Erven et al. (2014) formula.

    The total QBER combines three error sources:

    1. **Source errors**: From imperfect Bell state preparation
       Q_source = (1 - F) / 2

    2. **Detector errors**: Intrinsic measurement errors
       Q_det = e_det

    3. **Dark count errors**: False detections when no photon arrives
       Q_dark = (1 - η) × P_dark / 2

    Returns
    -------
    float
        Total QBER = Q_source + Q_det + Q_dark.
    """
    q_source = (1.0 - fidelity) / 2.0
    q_det = detector_error
    q_dark = (1.0 - detection_efficiency) * dark_count_prob / 2.0
    return q_source + q_det + q_dark
```

#### 5.5.3 NSMParameters Integration

**File:** [simulation/physical_model.py](caligo/simulation/physical_model.py)

The `NSMParameters` dataclass provides the `qber_conditional` property, which is the pre-computed heuristic QBER available to the Blind protocol:

```python
@dataclass(frozen=True)
class NSMParameters:
    """
    Noisy Storage Model physical parameters.
    
    Attributes
    ----------
    channel_fidelity : float
        EPR source fidelity F ∈ (0.5, 1].
    detection_eff_eta : float
        Detection efficiency η ∈ (0, 1].
    dark_count_prob : float
        Dark count probability P_dark.
    detector_error : float
        Intrinsic detector error e_det.
    storage_noise_r : float
        Adversary's storage noise parameter r ∈ [0, 1].
    delta_t_ns : int
        Waiting time Δt in nanoseconds.
    """
    
    @property
    def qber_conditional(self) -> float:
        """
        Heuristic QBER from physical parameters (no test bit disclosure).
        
        This is the value used by the Blind protocol for NSM-gated
        starting-rate optimization (Theoretical Report v2 §4.3).
        """
        from caligo.utils.math import compute_qber_erven
        
        return compute_qber_erven(
            fidelity=self.channel_fidelity,
            detector_error=self.detector_error,
            detection_efficiency=self.detection_eff_eta,
            dark_count_prob=self.dark_count_prob,
        )
```

#### 5.5.4 Blind Protocol Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    HEURISTIC QBER DATA FLOW                                 │
│                    (Blind Protocol - Optional)                              │
└─────────────────────────────────────────────────────────────────────────────┘

                               YAML Configuration
┌─────────────────────────────────────────────────────────────────────────────┐
│  nsm:                                                                       │
│    channel_fidelity: 0.98      # F                                          │
│    detection_eff_eta: 0.85     # η                                          │
│    dark_count_prob: 1e-6       # P_dark                                     │
│    detector_error: 0.01        # e_det                                      │
│    storage_noise_r: 0.95       # r (adversary)                              │
│    delta_t_ns: 100000          # Δt                                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
                           NSMParameters.from_yaml()
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  nsm_params = NSMParameters(...)                                            │
│                                                                             │
│  # Heuristic QBER computed on-demand (no test bits!)                        │
│  qber_heuristic = nsm_params.qber_conditional                               │
│    = compute_qber_erven(F, e_det, η, P_dark)                                │
│    = (1-F)/2 + e_det + (1-η)×P_dark/2                                       │
│    ≈ 0.02 for typical parameters                                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      │ qber_heuristic (float)
                                      ▼
                            Phase III: Blind Reconciliation
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  # BlindStrategy uses heuristic for NSM-gated optimization ONLY             │
│  # This is NOT used for rate selection (Blind discovers rate iteratively)   │
│                                                                             │
│  class BlindStrategy:                                                       │
│      def alice_reconcile_block(self, ..., qber_estimate=None):              │
│          # qber_estimate is the OPTIONAL heuristic                          │
│          if qber_estimate is not None and qber_estimate > 0.05:             │
│              # NSM-gated: Pre-shorten some bits for high-QBER channels      │
│              initial_shortened = self._compute_initial_shortening(...)      │
│          else:                                                              │
│              initial_shortened = 0  # Standard Blind: start at max rate     │
│                                                                             │
│  # Key insight: Blind NEVER discloses test bits for QBER estimation         │
│  # The heuristic only affects STARTING point, not the iterative protocol    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 5.5.5 Usage in BlindStrategy (Existing Implementation)

**File:** [alice.py](caligo/protocol/alice.py#L247-L260)

```python
# _phase2_sifting() - Blind branch
else:
    # Blind reconciliation: provide a prior from NSM parameters.
    # This is the HEURISTIC, not a measured QBER!
    qber_estimate = float(self.params.nsm_params.qber_conditional)
    qber_adjusted = qber_estimate  # No μ penalty (no test bits used)
    finite_size_penalty = 0.0      # No finite-size penalty
    test_set_size = 0              # No test set disclosed
```

#### 5.5.6 Refactoring Recommendation: Explicit Heuristic Injection

The proposed `ReconciliationContext` should explicitly distinguish measured vs heuristic QBER:

```python
# Proposed: caligo/reconciliation/strategies/base.py

@dataclass(frozen=True)
class ReconciliationContext:
    """Immutable context passed to strategies."""
    
    # ... other fields ...
    
    # Measured QBER (Baseline only, from sifting/qber.py)
    qber_measured: Optional[float] = None
    
    # Heuristic QBER (Blind optional, from NSMParameters.qber_conditional)
    qber_heuristic: Optional[float] = None
    
    @property
    def qber_for_blind_gating(self) -> float:
        """
        QBER for Blind NSM-gating (optional starting-rate optimization).
        
        Falls back to conservative default if no heuristic available.
        """
        return self.qber_heuristic if self.qber_heuristic is not None else 0.05


class BlindStrategy(ReconciliationStrategy):
    @property
    def requires_qber_estimation(self) -> bool:
        return False  # Core advantage: no test bits disclosed
    
    def alice_reconcile_block(self, ..., ctx: ReconciliationContext, ...):
        # Use heuristic for NSM-gated optimization ONLY
        # This does NOT replace the iterative rate discovery
        if ctx.qber_for_blind_gating > 0.05:
            initial_shortened = self._compute_initial_shortening(
                qber=ctx.qber_for_blind_gating,
                f_crit=ctx.f_crit,
                d=self._modulation_budget,
            )
        else:
            initial_shortened = 0
```

#### 5.5.7 Security Note: Heuristic vs Measured QBER

| Property | Measured QBER (Baseline) | Heuristic QBER (Blind) |
|----------|--------------------------|------------------------|
| **Source** | Test bit disclosure | Physical model parameters |
| **Accuracy** | Exact (with μ penalty) | Approximate estimate |
| **Information Leakage** | Yes (test bits disclosed) | No (derived from setup) |
| **Rate Selection** | Mandatory for rate choice | Optional optimization only |
| **Security Proof** | Per Theoretical Report v2 §3 | Per Theoretical Report v2 §4 |

The heuristic QBER enables practical optimization without compromising the Blind protocol's core security advantage: **no test bits are ever disclosed**.

### 6. MotherCodeManager Specification

```python
# caligo/reconciliation/mother_code_manager.py

@dataclass
class NumbaGraphTopology:
    """
    Pre-compiled graph topology arrays for Numba kernels.
    
    Structure-of-Arrays (SoA) format optimized for cache locality.
    These arrays are pinned in memory and passed as read-only arguments
    to JIT-compiled kernels.
    """
    # Check-to-variable edges (CSR format)
    check_row_ptr: np.ndarray      # uint32[m+1]: Row pointers
    check_col_idx: np.ndarray      # uint32[nnz]: Column indices
    
    # Variable-to-check edges (CSC format for efficient transpose)
    var_col_ptr: np.ndarray        # uint32[n+1]: Column pointers
    var_row_idx: np.ndarray        # uint32[nnz]: Row indices
    
    # Edge indexing for message passing
    edge_c2v: np.ndarray           # uint32[nnz]: Check→Var edge indices
    edge_v2c: np.ndarray           # uint32[nnz]: Var→Check edge indices
    
    # Metadata
    n_checks: int
    n_vars: int
    n_edges: int


class MotherCodeManager:
    """
    Singleton manager for R=0.5 Mother Code with Hybrid Pattern Library.
    
    Per Theoretical Report v2 §2.2 and §5.1, this class:
    1. Manages a single R_0=0.5 mother matrix
    2. Provides Hybrid Pattern Library (Untainted + ACE-Guided)
    3. Serves as the 'Static Data' provider for Numba kernels
    
    The Hybrid Pattern Library covers R_eff ∈ [0.51, 0.90] with Δ R = 0.01:
    - Regime A (R ≤ 0.625): Untainted puncturing patterns [3]
    - Regime B (R > 0.625): ACE-guided puncturing patterns [4]
    """
    
    _instance: Optional["MotherCodeManager"] = None
    
    def __init__(
        self,
        matrix_path: Path,
        pattern_dir: Path,
    ) -> None:
        """
        Load mother matrix and Hybrid Pattern Library.
        
        Parameters
        ----------
        matrix_path : Path
            Path to R=0.5 mother matrix (.npz format).
        pattern_dir : Path
            Directory containing hybrid pattern files.
        """
        # Load single R=0.5 matrix
        self._H_csr = sp.load_npz(matrix_path).tocsr().astype(np.uint8)
        
        # Verify R=0.5
        n, m = self._H_csr.shape[1], self._H_csr.shape[0]
        rate = 1.0 - m / n
        if abs(rate - 0.5) > 0.01:
            raise ValueError(f"Mother code rate {rate:.3f} != 0.5")
        
        # PRE-COMPILE FOR NUMBA:
        # Flatten CSR arrays to contiguous uint32/uint64 buffers for 
        # direct access by JIT kernels.
        self._compiled_topology = self._compile_topology()
        
        # Load Hybrid Pattern Library (Step = 0.01)
        # Dictionary structure: {0.51: mask_A, ... 0.62: mask_B, ... 0.90: mask_Z}
        # Masks are pre-converted to boolean arrays for the decoder kernel.
        self._patterns = self._load_hybrid_library(pattern_dir)
        
        # Pre-computed modulation indices for Blind protocol
        # Per Theoretical Report v2 §4.3: fixed order from hybrid puncturing order
        self._modulation_indices: Optional[np.ndarray] = None
    
    @classmethod
    def get_instance(cls, **kwargs) -> "MotherCodeManager":
        """Singleton accessor."""
        if cls._instance is None:
            cls._instance = cls(**kwargs)
        return cls._instance
    
    @property
    def frame_size(self) -> int:
        return self._H_csr.shape[1]
    
    @property
    def compiled_topology(self) -> NumbaGraphTopology:
        return self._compiled_topology
    
    @property
    def num_edges(self) -> int:
        return self._compiled_topology.n_edges
    
    def get_pattern(self, effective_rate: float) -> np.ndarray:
        """
        Get hybrid puncturing pattern for target effective rate.
        
        Per Theoretical Report v2 §2.2:
        - R ≤ 0.625: Untainted pattern (Regime A)
        - R > 0.625: ACE-guided pattern (Regime B)
        
        Returns
        -------
        np.ndarray
            Binary mask where 1 indicates punctured position.
        """
        # Find closest available rate (Δ R = 0.01 step)
        available = sorted(self._patterns.keys())
        closest = min(available, key=lambda r: abs(r - effective_rate))
        return self._patterns[closest].copy()
    
    def get_modulation_indices(self, d: int) -> np.ndarray:
        """
        Get d hybrid modulation indices for Blind protocol.
        
        Per Theoretical Report v2 §4.3: the revelation order is fixed
        at setup time using the hybrid puncturing order (Phase I untainted
        first, then Phase II ACE-guided).
        
        Parameters
        ----------
        d : int
            Number of modulation positions (punctured + shortened budget).
            
        Returns
        -------
        np.ndarray
            Ordered indices for modulation positions.
        """
        if self._modulation_indices is None:
            self._modulation_indices = self._compute_hybrid_indices()
        return self._modulation_indices[:d].copy()
    
    def _compile_topology(self) -> NumbaGraphTopology:
        """
        Convert CSR matrix to Numba-friendly SoA format.
        
        This is the "baking" step that converts the sparse matrix
        representation into flat arrays optimized for JIT compilation.
        """
        H = self._H_csr
        H_csc = H.tocsc()
        
        # CSR format (check-to-variable)
        check_row_ptr = H.indptr.astype(np.uint32)
        check_col_idx = H.indices.astype(np.uint32)
        
        # CSC format (variable-to-check)
        var_col_ptr = H_csc.indptr.astype(np.uint32)
        var_row_idx = H_csc.indices.astype(np.uint32)
        
        # Edge indexing (for message arrays)
        n_edges = H.nnz
        edge_c2v = np.arange(n_edges, dtype=np.uint32)
        edge_v2c = np.arange(n_edges, dtype=np.uint32)
        
        return NumbaGraphTopology(
            check_row_ptr=check_row_ptr,
            check_col_idx=check_col_idx,
            var_col_ptr=var_col_ptr,
            var_row_idx=var_row_idx,
            edge_c2v=edge_c2v,
            edge_v2c=edge_v2c,
            n_checks=H.shape[0],
            n_vars=H.shape[1],
            n_edges=n_edges,
        )
    
    def _load_hybrid_library(self, pattern_dir: Path) -> Dict[float, np.ndarray]:
        """
        Load Hybrid Pattern Library from directory.
        
        Expected file naming: pattern_rate0.51.npy, pattern_rate0.52.npy, ...
        Covers R_eff ∈ [0.51, 0.90] with Δ R = 0.01 (~40 files).
        """
        patterns = {}
        for path in pattern_dir.glob("*.npy"):
            # Parse rate from filename: pattern_rate0.65.npy
            rate_str = path.stem.split("rate")[-1]
            try:
                rate = float(rate_str)
                patterns[rate] = np.load(path).astype(np.uint8)
            except ValueError:
                continue  # Skip malformed filenames
        
        if len(patterns) < 10:
            raise ValueError(
                f"Insufficient patterns in {pattern_dir}. "
                f"Expected ~40, found {len(patterns)}. "
                "Run generate_hybrid_patterns.py to create Hybrid Pattern Library."
            )
        
        return patterns
    
    def _compute_hybrid_indices(self) -> np.ndarray:
        """
        Compute ordered modulation indices from hybrid puncturing order.
        
        Per Theoretical Report v2 §2.2.3: Phase I (untainted) first,
        then Phase II (ACE-guided), maintaining nesting property for
        rate-compatibility.
        """
        # Load highest-rate pattern (R=0.9) which contains all modulation indices
        max_rate = max(self._patterns.keys())
        max_pattern = self._patterns[max_rate]
        
        # Extract punctured indices (pattern[i] == 1)
        indices = np.where(max_pattern == 1)[0]
        
        # Note: The pattern generation script ensures indices are in
        # hybrid puncturing order (untainted first, ACE-guided second)
        return indices
```

---

### 6.2 Pattern Generation Script Wiring

This section documents the integration between the existing pattern generation infrastructure and the proposed Hybrid Pattern Library.

#### 6.2.1 Existing Script: `generate_puncture_patterns.py`

**File:** [scripts/generate_puncture_patterns.py](caligo/scripts/generate_puncture_patterns.py)

The current implementation provides:

| Component | Status | Description |
|-----------|--------|-------------|
| `UntaintedPuncturingGenerator` | ✅ Implemented | Elkouss et al. (2012) Regime A algorithm |
| `PuncturingPattern` dataclass | ✅ Implemented | Pattern storage with metadata |
| `_is_untainted()` | ✅ Implemented | Check if node is in untainted set |
| `_compute_untainted_set()` | ✅ Implemented | Compute X∞ for current punctured set |
| `_forced_puncture_select()` | ⚠️ Partial | Falls back when X∞ exhausted, but doesn't use ACE scores |
| ACE-guided puncturing | ❌ Missing | Regime B for R > 0.625 |
| Hybrid two-regime strategy | ❌ Missing | Automatic regime transition |

**Key Existing Classes:**

```python
# scripts/generate_puncture_patterns.py

class UntaintedPuncturingGenerator:
    """
    Generator for untainted puncturing patterns using Elkouss et al. algorithm.
    
    Current Limitation: Uses forced heuristic (minimize dead checks) when
    untainted set exhausted. Does NOT implement ACE-guided puncturing.
    """
    
    def _is_untainted(self, v: int, punctured: Set[int]) -> bool:
        """Check if variable v is untainted (no punctured neighbors in 2-hop)."""
        ...
    
    def _forced_puncture_select(self, punctured: Set[int]) -> int:
        """
        CURRENT: Select next node via forced puncturing heuristic.
        REQUIRED: Replace with ACE-guided selection for Regime B.
        """
        # Minimizes dead check nodes (not ACE-optimal)
        ...
```

#### 6.2.2 Required Extension: Hybrid Pattern Generation

The Appendix C script (`generate_hybrid_patterns.py`) extends the existing infrastructure with:

**New Functions Required:**

```python
# scripts/generate_hybrid_patterns.py (proposed extension)

def compute_ace_score(
    H: csr_matrix,
    symbol_idx: int,
    punctured: Set[int],
    max_cycle_length: int = 12,
) -> float:
    """
    Compute ACE (Approximate Cycle Extrinsic message degree) score.
    
    Per Liu & de Lamare [4]:
    - High ACE = Better connectivity = Safer to puncture
    - Low ACE = Weakly connected = Risky to puncture
    
    This replaces _forced_puncture_select() for Regime B.
    """
    pass

def run_ace_phase(
    H: csr_matrix,
    state: PuncturingState,
    target_fraction: float,
) -> bool:
    """
    Run ACE-guided puncturing (Regime B).
    
    Called when untainted set is exhausted (R > 0.625).
    Ranks candidates by ACE score (highest first).
    """
    pass
```

#### 6.2.3 Pattern Directory Structure

The Hybrid Pattern Library will be stored in a well-defined directory structure:

```
caligo/configs/ldpc_matrices/
├── mother_code_4096_R0.5.npz       # Single mother code matrix
└── hybrid_patterns/                 # Hybrid Pattern Library
    ├── modulation_indices.npy       # Ordered indices for Blind revelation
    ├── pattern_rate0.51.npy         # R_eff = 0.51 (Regime A)
    ├── pattern_rate0.52.npy         # R_eff = 0.52 (Regime A)
    ├── ...
    ├── pattern_rate0.62.npy         # R_eff = 0.62 (Regime A - near saturation)
    ├── pattern_rate0.63.npy         # R_eff = 0.63 (Regime B - ACE kicks in)
    ├── ...
    └── pattern_rate0.90.npy         # R_eff = 0.90 (Regime B)
```

**Saturation Transition (~R = 0.625):**
- Patterns 0.51-0.62: Pure untainted puncturing (Regime A)
- Patterns 0.63-0.90: ACE-guided puncturing (Regime B)
- `modulation_indices.npy`: Ordered puncturing sequence for Blind protocol

#### 6.2.4 Integration with MatrixManager

**File:** [reconciliation/matrix_manager.py](caligo/reconciliation/matrix_manager.py)

The existing `MatrixManager` must be updated to:

1. Load patterns from `hybrid_patterns/` directory
2. Provide pattern lookup by effective rate
3. Ensure deterministic ordering for Blind protocol

**Current Implementation (Partial):**

```python
# reconciliation/matrix_manager.py

class MatrixManager:
    """
    Manages LDPC parity-check matrices for reconciliation.
    
    Current: Loads multiple pre-built rate matrices
    Required: Single mother matrix + Hybrid Pattern Library
    """
    
    def get_puncture_pattern(self, rate: float) -> Optional[np.ndarray]:
        """
        Get puncture pattern for target effective rate.
        
        Current: Returns None if pattern not found
        Required: Raise error or return closest available pattern
        """
        pattern_path = self._pattern_dir / f"pattern_rate{rate:.2f}.npy"
        if pattern_path.exists():
            return np.load(pattern_path)
        return None
```

**Proposed Enhancement:**

```python
# reconciliation/matrix_manager.py (enhanced)

class MatrixManager:
    def get_puncture_pattern(self, rate: float) -> np.ndarray:
        """
        Get puncture pattern for target effective rate.
        
        Per Theoretical Report v2 §2.2, patterns cover R_eff ∈ [0.51, 0.90]
        with Δ R = 0.01. Returns closest available pattern.
        
        Raises
        ------
        ValueError
            If Hybrid Pattern Library is not generated.
        """
        available_rates = sorted(self._patterns.keys())
        
        if not available_rates:
            raise ValueError(
                "Hybrid Pattern Library not found. "
                "Run: python -m caligo.scripts.generate_hybrid_patterns"
            )
        
        closest_rate = min(available_rates, key=lambda r: abs(r - rate))
        return self._patterns[closest_rate].copy()
```

#### 6.2.5 Generation Pipeline

The pattern generation is a **build-time** step, not runtime:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     PATTERN GENERATION PIPELINE                             │
│                     (Offline Build Step)                                    │
└─────────────────────────────────────────────────────────────────────────────┘

Step 1: Generate Mother Code
┌─────────────────────────────────────────────────────────────────────────────┐
│  python -m caligo.scripts.generate_ace_mother_code                          │
│    → configs/ldpc_matrices/mother_code_4096_R0.5.npz                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
Step 2: Generate Hybrid Pattern Library
┌─────────────────────────────────────────────────────────────────────────────┐
│  python -m caligo.scripts.generate_hybrid_patterns \                        │
│      --matrix configs/ldpc_matrices/mother_code_4096_R0.5.npz \             │
│      --output configs/ldpc_matrices/hybrid_patterns/ \                      │
│      --rate-min 0.51 --rate-max 0.90                                        │
│                                                                             │
│  Output:                                                                    │
│    → configs/ldpc_matrices/hybrid_patterns/pattern_rate0.51.npy             │
│    → configs/ldpc_matrices/hybrid_patterns/pattern_rate0.52.npy             │
│    → ...                                                                    │
│    → configs/ldpc_matrices/hybrid_patterns/pattern_rate0.90.npy             │
│    → configs/ldpc_matrices/hybrid_patterns/modulation_indices.npy           │
│                                                                             │
│  Console output shows regime transition:                                    │
│    "Untainted saturation at π=0.20, R_eff=0.625"                            │
│    "Switching to ACE-guided puncturing (Regime B)"                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
Step 3: Runtime Pattern Loading
┌─────────────────────────────────────────────────────────────────────────────┐
│  # In protocol execution:                                                   │
│  mother_mgr = MotherCodeManager(                                            │
│      matrix_path="configs/ldpc_matrices/mother_code_4096_R0.5.npz",         │
│      pattern_dir="configs/ldpc_matrices/hybrid_patterns/",                  │
│  )                                                                          │
│                                                                             │
│  # Pattern lookup by rate:                                                  │
│  pattern = mother_mgr.get_pattern(effective_rate=0.75)  # Returns Regime B  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 7. LeakageTracker as Circuit Breaker

```python
# caligo/reconciliation/leakage_tracker.py (enhanced)

class LeakageTracker:
    """
    Cumulative leakage tracker with circuit breaker pattern.
    
    If total leakage exceeds the NSM budget, immediately raises
    SecurityError to abort the protocol before violating security.
    """
    
    def __init__(
        self,
        safety_cap: int,
        abort_on_exceed: bool = True,
    ) -> None:
        self._safety_cap = safety_cap
        self._abort_on_exceed = abort_on_exceed
        self._total_leakage = 0
        self._records: List[LeakageRecord] = []
    
    def record(
        self,
        block_id: int,
        syndrome_bits: int,
        hash_bits: int,
        revealed_bits: int,
        **kwargs
    ) -> None:
        """
        Record leakage event and check safety cap.
        
        Raises
        ------
        LeakageBudgetExceeded
            If cumulative leakage exceeds safety_cap and abort_on_exceed=True.
        """
        event = LeakageRecord(
            block_id=block_id,
            syndrome_bits=syndrome_bits,
            hash_bits=hash_bits,
            revealed_bits=revealed_bits,
        )
        self._records.append(event)
        self._total_leakage += event.total
        
        if self._abort_on_exceed and self._total_leakage > self._safety_cap:
            raise LeakageBudgetExceeded(
                f"Cumulative leakage {self._total_leakage} exceeds "
                f"safety cap {self._safety_cap}",
                actual_leakage=self._total_leakage,
                max_allowed=self._safety_cap,
            )
    
    def record_reveal(
        self,
        block_id: int,
        iteration: int,
        revealed_bits: int,
    ) -> None:
        """Record Blind iteration reveal (syndrome already counted)."""
        self.record(
            block_id=block_id,
            syndrome_bits=0,
            hash_bits=0,
            revealed_bits=revealed_bits,
        )
    
    @property
    def total_leakage(self) -> int:
        return self._total_leakage
    
    def should_abort(self) -> bool:
        return self._total_leakage > self._safety_cap
```

### 8. LDPCCodec: JIT Interface Facade

```python
# caligo/reconciliation/ldpc_codec.py

from caligo.scripts.numba_kernels import (
    encode_bitpacked_kernel,
    decode_bp_virtual_graph_kernel,
    decode_bp_hotstart_kernel,
)


@dataclass
class DecoderResult:
    """Result from BP decoder kernel."""
    corrected_bits: np.ndarray
    converged: bool
    iterations: int
    messages: np.ndarray  # For Hot-Start persistence


class LDPCCodec:
    """
    Thin wrapper around JIT-compiled Numba kernels.
    
    Per the "Python Control, Numba Engine" architecture, this class:
    1. Accepts high-level Python objects (numpy.ndarray, ReconciliationContext)
    2. Efficiently packs bits and prepares aligned C-contiguous buffers
    3. Invokes the appropriate numba.njit(nogil=True) kernel
    4. Unpacks the result back to Python-friendly structures
    
    The complex, high-performance C-style logic is contained entirely
    within the kernels, while Python maintains readability and manages
    the protocol state machine.
    """
    
    def __init__(self, mother_code: MotherCodeManager) -> None:
        self._topo = mother_code.compiled_topology
        self._frame_size = mother_code.frame_size
    
    @property
    def num_edges(self) -> int:
        return self._topo.n_edges
    
    def encode(self, frame: np.ndarray, pattern: np.ndarray) -> np.ndarray:
        """
        Compute syndrome using bit-packed SpMV kernel.
        
        Per §4.1.2: Uses bitwise AND + POPCNT for 10x-50x speedup.
        
        Parameters
        ----------
        frame : np.ndarray
            Full LDPC frame (n bits).
        pattern : np.ndarray
            Puncturing pattern (for validation only; syndrome uses full frame).
            
        Returns
        -------
        np.ndarray
            Syndrome bits (m = (1-R_0) × n bits).
        """
        # Bit-pack frame into uint64 words
        packed_frame = self._bitpack(frame)
        
        # Call Numba kernel
        packed_syndrome = encode_bitpacked_kernel(
            packed_frame,
            self._topo.check_row_ptr,
            self._topo.check_col_idx,
            self._topo.n_checks,
        )
        
        # Unpack syndrome
        return self._bitunpack(packed_syndrome, self._topo.n_checks)
    
    def decode_baseline(
        self,
        syndrome: np.ndarray,
        llr: np.ndarray,
        pattern: np.ndarray,
        max_iterations: int = 60,
    ) -> DecoderResult:
        """
        Baseline decoding using Virtual Graph kernel.
        
        Per §4.1.3A: Single kernel operates on full mother graph.
        Pattern is used only for LLR initialization (punctured → 0).
        Rate adaptation is purely a memory initialization step.
        
        Parameters
        ----------
        syndrome : np.ndarray
            Syndrome from Alice.
        llr : np.ndarray
            Three-state LLR array (payload, punctured=0, shortened=±∞).
        pattern : np.ndarray
            Puncturing pattern (used for mask validation).
        max_iterations : int
            Maximum BP iterations.
            
        Returns
        -------
        DecoderResult
            Decoded bits and convergence status.
        """
        # Ensure C-contiguous arrays
        llr = np.ascontiguousarray(llr, dtype=np.float64)
        syndrome = np.ascontiguousarray(syndrome, dtype=np.uint8)
        
        # Initialize messages to zero
        messages = np.zeros(self._topo.n_edges * 2, dtype=np.float64)
        
        # Call Virtual Graph decoder kernel
        corrected_bits, converged, iterations = decode_bp_virtual_graph_kernel(
            llr,
            syndrome,
            messages,
            self._topo.check_row_ptr,
            self._topo.check_col_idx,
            self._topo.var_col_ptr,
            self._topo.var_row_idx,
            max_iterations,
        )
        
        return DecoderResult(
            corrected_bits=corrected_bits,
            converged=converged,
            iterations=iterations,
            messages=messages,
        )
    
    def decode_blind(
        self,
        syndrome: np.ndarray,
        llr: np.ndarray,
        messages: np.ndarray,
        frozen_mask: np.ndarray,
        max_iterations: int = 60,
    ) -> DecoderResult:
        """
        Blind decoding using Hot-Start kernel with Freeze optimization.
        
        Per §4.1.3B:
        - Messages persist across iterations (Hot-Start)
        - Frozen bits (LLR=±∞) skip expensive tanh/arctanh updates
        
        Parameters
        ----------
        syndrome : np.ndarray
            Syndrome from Alice (computed once, reused).
        llr : np.ndarray
            Current LLR array (updated as bits are revealed).
        messages : np.ndarray
            Edge messages from previous iteration (in/out).
        frozen_mask : np.ndarray
            Boolean mask for revealed/shortened positions.
        max_iterations : int
            Maximum BP iterations for this round.
            
        Returns
        -------
        DecoderResult
            Decoded bits, convergence status, and updated messages.
        """
        # Ensure C-contiguous arrays
        llr = np.ascontiguousarray(llr, dtype=np.float64)
        syndrome = np.ascontiguousarray(syndrome, dtype=np.uint8)
        messages = np.ascontiguousarray(messages, dtype=np.float64)
        frozen_mask = np.ascontiguousarray(frozen_mask, dtype=np.bool_)
        
        # Call Hot-Start decoder kernel with Freeze optimization
        corrected_bits, converged, iterations, out_messages = decode_bp_hotstart_kernel(
            llr,
            syndrome,
            messages,  # Resume from previous state
            frozen_mask,  # Skip updates for frozen positions
            self._topo.check_row_ptr,
            self._topo.check_col_idx,
            self._topo.var_col_ptr,
            self._topo.var_row_idx,
            max_iterations,
        )
        
        return DecoderResult(
            corrected_bits=corrected_bits,
            converged=converged,
            iterations=iterations,
            messages=out_messages,
        )
    
    def _bitpack(self, bits: np.ndarray) -> np.ndarray:
        """Pack bits into uint64 words for SIMD operations."""
        n = len(bits)
        n_words = (n + 63) // 64
        packed = np.zeros(n_words, dtype=np.uint64)
        
        for i, bit in enumerate(bits):
            if bit:
                packed[i // 64] |= np.uint64(1) << (i % 64)
        
        return packed
    
    def _bitunpack(self, packed: np.ndarray, n_bits: int) -> np.ndarray:
        """Unpack uint64 words back to bit array."""
        bits = np.zeros(n_bits, dtype=np.uint8)
        
        for i in range(n_bits):
            if packed[i // 64] & (np.uint64(1) << (i % 64)):
                bits[i] = 1
        
        return bits
```

### 8.1 Three-State LLR Builder

```python
# caligo/reconciliation/ldpc_decoder.py (enhanced)

def build_three_state_llr(
    received_bits: np.ndarray,
    qber: float,
    puncture_mask: np.ndarray,
    shorten_mask: Optional[np.ndarray] = None,
    shorten_values: Optional[np.ndarray] = None,
    llr_shortened: float = 100.0,
) -> np.ndarray:
    """
    Build LLR array with three distinct belief states.
    
    Parameters
    ----------
    received_bits : np.ndarray
        Bob's received/constructed frame.
    qber : float
        Channel error probability for payload LLRs.
    puncture_mask : np.ndarray
        Boolean mask where True indicates punctured position.
    shorten_mask : np.ndarray, optional
        Boolean mask where True indicates shortened position.
    shorten_values : np.ndarray, optional
        Known values at shortened positions.
    llr_shortened : float
        Magnitude of LLR for shortened bits (±∞ approximation).
        
    Returns
    -------
    np.ndarray
        LLR array with:
        - Payload positions: channel LLR based on QBER
        - Punctured positions: 0 (erasure/no information)
        - Shortened positions: ±llr_shortened (perfect knowledge)
    """
    n = len(received_bits)
    llr = np.zeros(n, dtype=np.float64)
    
    # 1. Payload bits: channel information
    if qber > 0 and qber < 1:
        channel_llr = np.log((1 - qber) / qber)
    else:
        channel_llr = 10.0  # Fallback for edge cases
    
    payload_mask = ~puncture_mask
    if shorten_mask is not None:
        payload_mask = payload_mask & ~shorten_mask
    
    # LLR = ln(P(y|x=0) / P(y|x=1)) = (1 - 2y) * ln((1-p)/p)
    llr[payload_mask] = (1 - 2 * received_bits[payload_mask]) * channel_llr
    
    # 2. Punctured bits: erasure (zero LLR)
    llr[puncture_mask] = 0.0
    
    # 3. Shortened bits: perfect knowledge
    if shorten_mask is not None and shorten_values is not None:
        # LLR = +∞ for bit=0, -∞ for bit=1
        shortened_llr = (1 - 2 * shorten_values) * llr_shortened
        llr[shorten_mask] = shortened_llr
    
    return llr
```

---

## Part III: Protocol-Level Message Sequence

### 9. Baseline Protocol Sequence

```
Alice                                          Bob
  │                                              │
  │ ──────── Phase II: QBER Estimation ────────  │
  │                                              │
  │  [Test subset exchange, compute QBER]        │
  │                                              │
  │ ─────────── Phase III: Baseline ───────────  │
  │                                              │
  ├──────────────────────────────────────────────┤
  │           For each block b:                  │
  ├──────────────────────────────────────────────┤
  │                                              │
  │  Alice:                                      │
  │   1. Select rate based on QBER               │
  │   2. Get untainted pattern for rate          │
  │   3. Construct frame                         │
  │   4. Compute syndrome (once)                 │
  │   5. Compute hash                            │
  │                                              │
  │  SYNDROME(block_id, syndrome, pattern_id,    │
  │           payload_len, hash, qber)           │
  │ ────────────────────────────────────────────►│
  │                                              │  Bob:
  │                                              │   1. Get pattern by id
  │                                              │   2. Construct frame
  │                                              │   3. Build LLRs
  │                                              │   4. Decode (BP)
  │                                              │   5. Verify hash
  │                                              │
  │  ACK(block_id, verified)                     │
  │ ◄────────────────────────────────────────────│
  │                                              │
  └──────────────────────────────────────────────┘
```

### 10. Blind Protocol Sequence

```
Alice                                          Bob
  │                                              │
  │ ─────────── Phase III: Blind ──────────────  │
  │                                              │
  ├──────────────────────────────────────────────┤
  │           For each block b:                  │
  ├──────────────────────────────────────────────┤
  │                                              │
  │  Alice:                                      │
  │   1. Set p=d, s=0 (all punctured)            │
  │   2. Construct frame with padding            │
  │   3. Compute syndrome ONCE                   │
  │   4. Compute hash                            │
  │                                              │
  │  BLIND_INIT(block_id, syndrome,              │
  │             puncture_indices, hash, qber)    │
  │ ────────────────────────────────────────────►│
  │                                              │  Bob:
  │                                              │   1. Initialize LLRs
  │                                              │      (punctured = 0)
  │                                              │   2. Decode attempt #1
  │                                              │
  │  If converged & verified:                    │
  │  ACK(block_id, converged=True)               │
  │ ◄────────────────────────────────────────────│
  │  DONE                                        │
  │                                              │
  │  Else (NACK):                                │
  │  NACK(block_id, converged=False)             │
  │ ◄────────────────────────────────────────────│
  │                                              │
  │  Alice:                                      │
  │   - Select Δ bits to reveal                  │
  │   - Update: p -= Δ, s += Δ                   │
  │                                              │
  │  REVEAL(block_id, iter=2,                    │
  │         revealed_indices, revealed_values)   │
  │ ────────────────────────────────────────────►│
  │                                              │  Bob:
  │                                              │   - Update LLRs at
  │                                              │     revealed positions
  │                                              │     (0 → ±∞)
  │                                              │   - Re-decode (same
  │                                              │     syndrome!)
  │                                              │
  │  [Repeat until success or p=0]               │
  │                                              │
  └──────────────────────────────────────────────┘
```

---

### 10.1 Message Envelope Wiring for Blind Protocol

This section documents the required extensions to [connection/envelope.py](caligo/connection/envelope.py) to support the Blind protocol's iterative message exchange.

#### 10.1.1 Current MessageType Enum

**File:** [connection/envelope.py](caligo/connection/envelope.py#L18-L40)

```python
class MessageType(str, Enum):
    """Protocol message type discriminators."""

    # Phase II
    DETECTION_COMMITMENT = "DETECTION_COMMITMENT"
    BASIS_REVEAL = "BASIS_REVEAL"
    COMMITMENT_OPENING = "COMMITMENT_OPENING"

    # Sifting
    INDEX_LISTS = "INDEX_LISTS"
    TEST_OUTCOMES = "TEST_OUTCOMES"

    # Reconciliation
    SYNDROME = "SYNDROME"
    SYNDROME_RESPONSE = "SYNDROME_RESPONSE"

    # Amplification
    TOEPLITZ_SEED = "TOEPLITZ_SEED"

    # Control
    ACK = "ACK"
    ABORT = "ABORT"
```

#### 10.1.2 Required Extensions for Blind Protocol

The current `SYNDROME` and `SYNDROME_RESPONSE` message types are used for **both** Baseline and Blind protocols, with the `kind` field in the payload distinguishing them:

| Message Type | Baseline Payload | Blind Payload |
|--------------|------------------|---------------|
| `SYNDROME` | `{kind: "baseline", syndrome, pattern_id, ...}` | `{kind: "blind", syndrome, puncture_indices, ...}` |
| `SYNDROME_RESPONSE` | `{kind: "baseline", verified, converged, ...}` | `{kind: "blind", verified, converged, ...}` |

**Current Design Decision:** The existing implementation uses a single `SYNDROME` type with a `kind` discriminator to avoid proliferating message types. This is acceptable but requires careful validation.

**Proposed Extension:** Add explicit Blind message types for clearer protocol separation:

```python
# connection/envelope.py (proposed extension)

class MessageType(str, Enum):
    """Protocol message type discriminators."""

    # ... existing types ...

    # Reconciliation - Baseline
    SYNDROME = "SYNDROME"              # Baseline: single-shot syndrome
    SYNDROME_RESPONSE = "SYNDROME_RESPONSE"

    # Reconciliation - Blind (NEW)
    BLIND_INIT = "BLIND_INIT"          # Blind: initial syndrome + puncture indices
    BLIND_REVEAL = "BLIND_REVEAL"      # Blind: iteration reveal (Δ bits)
    BLIND_ACK = "BLIND_ACK"            # Blind: success acknowledgment
    BLIND_NACK = "BLIND_NACK"          # Blind: request more reveals

    # ... rest unchanged ...
```

#### 10.1.3 Blind Protocol Message Flow (with Types)

```
Alice                                          Bob
  │                                              │
  │  BLIND_INIT                                  │
  │  {                                           │
  │    block_id: 0,                              │
  │    syndrome: [...],                          │
  │    puncture_indices: [...],                  │
  │    payload_length: 4096,                     │
  │    hash_value: 0xABCD,                       │
  │    qber_prior: 0.02,                         │
  │    iteration: 1,                             │
  │    revealed_indices: [],  // empty initially │
  │    revealed_values: [],                      │
  │  }                                           │
  │ ────────────────────────────────────────────►│
  │                                              │
  │  BLIND_NACK (decode failed, need more info)  │
  │  {                                           │
  │    block_id: 0,                              │
  │    converged: false,                         │
  │    verified: false,                          │
  │  }                                           │
  │ ◄────────────────────────────────────────────│
  │                                              │
  │  BLIND_REVEAL                                │
  │  {                                           │
  │    block_id: 0,                              │
  │    iteration: 2,                             │
  │    revealed_indices: [100, 200, 300],        │
  │    revealed_values: [0, 1, 0],               │
  │  }                                           │
  │ ────────────────────────────────────────────►│
  │                                              │
  │  BLIND_ACK (success!)                        │
  │  {                                           │
  │    block_id: 0,                              │
  │    converged: true,                          │
  │    verified: true,                           │
  │  }                                           │
  │ ◄────────────────────────────────────────────│
```

#### 10.1.4 OrderedSocket Integration

**File:** [connection/ordered_socket.py](caligo/connection/ordered_socket.py)

The `OrderedSocket` class provides send/recv generators that work with SquidASM's cooperative scheduling. No changes required to the core socket logic—only the message types need extension.

**Current Usage (alice.py/bob.py):**

```python
# Alice sends syndrome
yield from self._ordered_socket.send(
    MessageType.SYNDROME,
    {"kind": "blind", "block_id": block_id, ...},
)

# Bob receives and responds
msg = yield from self._ordered_socket.recv(MessageType.SYNDROME)
yield from self._ordered_socket.send(
    MessageType.SYNDROME_RESPONSE,
    {"kind": "blind", "verified": verified, ...},
)
```

**Proposed Usage (with explicit types):**

```python
# Alice sends Blind init
yield from self._ordered_socket.send(
    MessageType.BLIND_INIT,
    {"block_id": block_id, "syndrome": syndrome, ...},
)

# Bob receives and responds
msg = yield from self._ordered_socket.recv(MessageType.BLIND_INIT)
yield from self._ordered_socket.send(
    MessageType.BLIND_NACK,  # or BLIND_ACK
    {"verified": verified, "converged": converged},
)

# Alice sends reveal
yield from self._ordered_socket.send(
    MessageType.BLIND_REVEAL,
    {"iteration": 2, "revealed_indices": [...], ...},
)
```

#### 10.1.5 Backward Compatibility

The current design using `{kind: "baseline"|"blind"}` in payloads is **acceptable** for the initial implementation. The explicit message types (`BLIND_INIT`, `BLIND_REVEAL`, etc.) are a **recommended enhancement** for:

1. Clearer protocol separation in logs and debugging
2. Stronger type safety at the envelope layer
3. Easier extension for future protocols (e.g., INTERACTIVE)

**Migration Path:**

1. **Phase 1 (Current):** Use `SYNDROME`/`SYNDROME_RESPONSE` with `kind` discriminator
2. **Phase 2 (Optional):** Add explicit `BLIND_*` types when refactoring strategies
3. **Deprecation:** Remove `kind` field after all consumers migrate

---

### 10.2 Protocol Layer Wiring for Dual Architecture

This section documents the integration between the protocol layer ([protocol/alice.py](caligo/protocol/alice.py), [protocol/bob.py](caligo/protocol/bob.py)) and the reconciliation strategy system with YAML runtime switching.

#### 10.2.1 Current Protocol Layer Structure

**Alice's Phase III Implementation:** [alice.py](caligo/protocol/alice.py#L285-L480)

The current implementation embeds protocol logic directly in `_phase3_reconcile()`:

```python
# alice.py (current structure)

def _phase3_reconcile(self, alice_bits, qber_observed, qber_adjusted):
    # 1. Direct component instantiation
    matrix_manager = MatrixManager.from_directory(...)
    block_reconciler = BlockReconciler(...)
    
    # 2. Protocol branching embedded in role class
    is_blind = self.params.reconciliation.reconciliation_type == ReconciliationType.BLIND
    
    # 3. For each block: inline protocol logic
    for block_id, alice_block in enumerate(alice_blocks):
        if is_blind:
            # Blind-specific message construction
            yield from self._ordered_socket.send(MessageType.SYNDROME, {"kind": "blind", ...})
        else:
            # Baseline-specific message construction
            yield from self._ordered_socket.send(MessageType.SYNDROME, {"kind": "baseline", ...})
```

**Problem:** Protocol logic is tightly coupled to role classes, making it difficult to:
- Test protocols in isolation
- Add new protocols without modifying alice.py/bob.py
- Reuse protocol logic in different contexts

#### 10.2.2 YAML Configuration for Runtime Switching

**File:** [reconciliation/factory.py](caligo/reconciliation/factory.py)

The `ReconciliationConfig` dataclass enables YAML-based protocol selection:

```python
@dataclass
class ReconciliationConfig:
    """Configuration for reconciliation protocol execution."""
    
    reconciliation_type: ReconciliationType = ReconciliationType.BASELINE
    frame_size: int = 4096
    max_iterations: int = 50
    max_blind_rounds: int = 3
    use_nsm_informed_start: bool = True
    
    @property
    def requires_qber_estimation(self) -> bool:
        """Delegate to ReconciliationType."""
        return self.reconciliation_type.requires_qber_estimation
```

**YAML Configuration:** [configs/reconciliation/master.yaml](caligo/configs/reconciliation/master.yaml)

```yaml
reconciliation:
  # Switch protocol at runtime without code changes
  type: "baseline"  # or "blind"
  
  frame_size: 4096
  max_iterations: 60
  
  # Baseline-specific
  baseline:
    efficiency_target: 1.1
    
  # Blind-specific
  blind:
    max_iterations: 3
    use_nsm_informed_start: true
```

#### 10.2.3 Protocol Parameters Integration

**File:** [protocol/base.py](caligo/protocol/base.py#L65-L100)

The `ProtocolParameters` dataclass carries reconciliation config to both roles:

```python
@dataclass(frozen=True)
class ProtocolParameters:
    """Parameters for a Caligo protocol run."""
    
    session_id: str
    nsm_params: NSMParameters
    num_pairs: int
    num_qubits: int = 10
    precomputed_epr: Optional[PrecomputedEPRData] = None
    
    # Reconciliation configuration (from YAML)
    reconciliation: ReconciliationConfig = field(default_factory=ReconciliationConfig)
```

#### 10.2.4 Protocol Switching Logic

The current implementation uses `params.reconciliation.reconciliation_type` to branch:

**File:** [alice.py](caligo/protocol/alice.py#L220-L260)

```python
# Phase II Sifting - QBER estimation gating
if self.params.reconciliation.requires_qber_estimation:
    # BASELINE: Perform test bit exchange
    test_indices, key_indices = self._sifter.select_test_subset(...)
    qber = self._qber_estimator.estimate(...)
else:
    # BLIND: Use heuristic from NSM parameters
    qber_estimate = float(self.params.nsm_params.qber_conditional)
```

**File:** [alice.py](caligo/protocol/alice.py#L335-L365)

```python
# Phase III Reconciliation - protocol branching
is_blind = self.params.reconciliation.reconciliation_type == ReconciliationType.BLIND

if is_blind:
    yield from self._ordered_socket.send(
        MessageType.SYNDROME, {"kind": "blind", ...}
    )
else:
    yield from self._ordered_socket.send(
        MessageType.SYNDROME, {"kind": "baseline", ...}
    )
```

#### 10.2.5 Proposed Refactoring: Strategy Injection

The proposed architecture replaces embedded protocol logic with injected strategies:

```python
# Proposed: protocol/alice.py

def _phase3_reconcile(self, alice_bits, qber_observed, qber_adjusted):
    # 1. Create strategy via factory (from YAML config)
    strategy = create_strategy(
        config=self.params.reconciliation,
        mother_code=self._mother_code,
        codec=self._codec,
        leakage_tracker=self._leakage_tracker,
    )
    
    # 2. Create context with all required data
    ctx = ReconciliationContext(
        session_id=self.params.session_id,
        frame_size=self.params.reconciliation.frame_size,
        qber_adjusted=qber_adjusted if strategy.requires_qber_estimation else None,
        qber_heuristic=self.params.nsm_params.qber_conditional,
    )
    
    # 3. Delegate to strategy (no protocol-specific logic here)
    for block_id, alice_block in enumerate(alice_blocks):
        result = yield from strategy.alice_reconcile_block(
            payload=alice_block,
            ctx=ctx,
            block_id=block_id,
            qber_estimate=qber_adjusted,
        )
        reconciled_parts.append(result.corrected_payload)
```

#### 10.2.6 Strategy Factory

**Proposed:** [reconciliation/factory.py](caligo/reconciliation/factory.py)

```python
def create_strategy(
    config: ReconciliationConfig,
    mother_code: MotherCodeManager,
    codec: LDPCCodec,
    leakage_tracker: LeakageTracker,
) -> ReconciliationStrategy:
    """
    Factory function for creating reconciliation strategies.
    
    Enables YAML-based protocol switching without code changes.
    """
    if config.reconciliation_type == ReconciliationType.BASELINE:
        return BaselineStrategy(
            mother_code=mother_code,
            codec=codec,
            leakage_tracker=leakage_tracker,
        )
    elif config.reconciliation_type == ReconciliationType.BLIND:
        return BlindStrategy(
            mother_code=mother_code,
            codec=codec,
            leakage_tracker=leakage_tracker,
            max_blind_iterations=config.max_blind_rounds,
        )
    else:
        raise ValueError(f"Unsupported reconciliation type: {config.reconciliation_type}")
```

#### 10.2.7 Runtime Protocol Switching Example

```yaml
# configs/session_baseline.yaml
reconciliation:
  type: "baseline"
  frame_size: 4096
  baseline:
    efficiency_target: 1.1

# configs/session_blind.yaml
reconciliation:
  type: "blind"
  frame_size: 4096
  blind:
    max_iterations: 3
```

```python
# Python usage
from caligo.reconciliation.factory import ReconciliationConfig

# Load from YAML (baseline)
config_baseline = ReconciliationConfig.from_yaml("configs/session_baseline.yaml")
assert config_baseline.requires_qber_estimation == True

# Load from YAML (blind)
config_blind = ReconciliationConfig.from_yaml("configs/session_blind.yaml")
assert config_blind.requires_qber_estimation == False
```

---

## Part IV: Refactoring Plan

### 11. Phase 1: Foundation (Week 1)

| Task | Priority | Owner | Deliverable |
|------|----------|-------|-------------|
| Implement `MotherCodeManager` | P0 | - | New file replacing multi-matrix logic |
| Generate untainted patterns | P0 | - | Pattern files for rates 0.5-0.9 |
| Create `ReconciliationContext` dataclass | P1 | - | Shared context structure |
| Create `BlockResult` dataclass | P1 | - | Unified result structure |

### 12. Phase 2: Strategy Layer (Week 2)

| Task | Priority | Owner | Deliverable |
|------|----------|-------|-------------|
| Define `ReconciliationStrategy` ABC | P0 | - | Base class with generator interface |
| Implement `BaselineStrategy` | P0 | - | Port existing logic to strategy |
| Implement `BlindStrategy` | P0 | - | New implementation with LLR persistence |
| Implement `BlindDecoderState` | P0 | - | State machine for Bob |

### 13. Phase 3: Integration (Week 3)

| Task | Priority | Owner | Deliverable |
|------|----------|-------|-------------|
| Create `ReconciliationSession` | P0 | - | Replaces orchestrator |
| Refactor `alice.py` to delegate | P0 | - | Remove embedded protocol logic |
| Refactor `bob.py` to delegate | P0 | - | Add Blind message handler |
| Enhance `LeakageTracker` circuit breaker | P1 | - | Strict budget enforcement |

### 14. Phase 4: Testing & Validation (Week 4)

| Task | Priority | Owner | Deliverable |
|------|----------|-------|-------------|
| Unit tests for strategy classes | P0 | - | pytest suite |
| Integration tests with mock network | P0 | - | End-to-end flow tests |
| Convergence tests for untainted patterns | P0 | - | Statistical validation |
| Leakage accounting tests | P1 | - | Verify exact leakage tracking |

---

### 14.1 Reconciliation Module Wiring Summary

This section provides a comprehensive mapping of all reconciliation modules to the proposed refactoring architecture.

#### 14.1.1 Module Inventory

| Module | Current Role | Proposed Fate | Integration Point |
|--------|--------------|---------------|-------------------|
| [orchestrator.py](caligo/reconciliation/orchestrator.py) | "God Object" coordinator | **Deprecated** | Replace with `ReconciliationSession` |
| [block_reconciler.py](caligo/reconciliation/block_reconciler.py) | Single-block logic | **Retained** | Used by both strategies |
| [matrix_manager.py](caligo/reconciliation/matrix_manager.py) | Multi-matrix pool | **Refactored** | Single mother code + hybrid patterns |
| [ldpc_encoder.py](caligo/reconciliation/ldpc_encoder.py) | Syndrome computation | **Enhanced** | Wire to Numba bit-packed kernel |
| [ldpc_decoder.py](caligo/reconciliation/ldpc_decoder.py) | BP decoding | **Enhanced** | Wire to Numba Virtual Graph / Hot-Start |
| [rate_selector.py](caligo/reconciliation/rate_selector.py) | Rate selection (bypassed) | **Restored** | Use adjusted QBER for rate selection |
| [leakage_tracker.py](caligo/reconciliation/leakage_tracker.py) | Leakage accounting | **Enhanced** | Add circuit-breaker pattern |
| [hash_verifier.py](caligo/reconciliation/hash_verifier.py) | Verification hash | **Retained** | No changes required |
| [blind_manager.py](caligo/reconciliation/blind_manager.py) | Blind state machine | **Refactored** | Integrate into `BlindStrategy` |
| [factory.py](caligo/reconciliation/factory.py) | YAML config loading | **Enhanced** | Add `create_strategy()` factory |
| [constants.py](caligo/reconciliation/constants.py) | Configuration constants | **Retained** | Add Numba-related constants |

#### 14.1.2 Current Module Responsibilities

**orchestrator.py** (Lines ~367):
```python
class ReconciliationOrchestrator:
    """Current responsibilities (to be distributed):
    - Block partitioning → ReconciliationSession
    - Component instantiation → Strategy Factory
    - Decode orchestration → Strategy classes
    - Leakage enforcement → LeakageTracker (enhanced)
    """
```

**block_reconciler.py** (Lines ~321):
```python
class BlockReconciler:
    """Single-block reconciliation (RETAINED).
    
    Used internally by both BaselineStrategy and BlindStrategy.
    Provides: encode, decode, verify for one LDPC block.
    """
    
@dataclass
class BlockResult:
    """Unified result structure (RETAINED).
    
    Fields: corrected_payload, verified, converged, error_count,
    syndrome_length, retry_report, leakage.
    """
```

**blind_manager.py** (Lines ~372):
```python
@dataclass
class BlindConfig:
    """Blind configuration (TO BE INTEGRATED into BlindStrategy)."""
    max_iterations: int = 3
    modulation_fraction: float = 0.1
    frame_size: int = 4096

@dataclass
class BlindIterationState:
    """Iteration state (TO BE REPLACED by BlindDecoderState).
    
    Missing in current: LLR persistence, message persistence, frozen mask.
    """
```

#### 14.1.3 Refactoring Dependency Graph

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    RECONCILIATION MODULE DEPENDENCIES                       │
│                    (Current → Proposed)                                     │
└─────────────────────────────────────────────────────────────────────────────┘

                            CURRENT ARCHITECTURE
                            ─────────────────────
                            
alice.py/bob.py
      │
      ▼
┌─────────────────┐
│  Orchestrator   │  ◄── "God Object": owns everything
│  (orchestrator) │
└────────┬────────┘
         │ creates
         ├──────────────────────────────────────────┐
         ▼                                          ▼
┌─────────────────┐                      ┌─────────────────┐
│ BlockReconciler │                      │  BlindManager   │ (not integrated)
│(block_reconciler│                      │ (blind_manager) │
└────────┬────────┘                      └─────────────────┘
         │ uses
         ├─────────────┬──────────────┬─────────────┐
         ▼             ▼              ▼             ▼
┌─────────────┐ ┌─────────────┐ ┌────────────┐ ┌────────────┐
│MatrixManager│ │  Decoder    │ │  Encoder   │ │HashVerifier│
│(matrix_mgr) │ │(ldpc_decode)│ │(ldpc_encode│ │(hash_verify│
└─────────────┘ └─────────────┘ └────────────┘ └────────────┘


                            PROPOSED ARCHITECTURE
                            ─────────────────────

alice.py/bob.py
      │
      │ delegates to strategy
      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      ReconciliationSession (NEW)                            │
│  - Manages per-session state                                                │
│  - Injects strategy via factory                                             │
│  - Tracks cumulative leakage                                                │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
         ┌───────────────────────┴───────────────────────┐
         │ ReconciliationStrategy (ABC)                  │
         │ - alice_reconcile_block()                     │
         │ - bob_reconcile_block()                       │
         ├───────────────────────┬───────────────────────┤
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│BaselineStrategy │    │  BlindStrategy  │    │ (Future: LDPC   │
│                 │    │                 │    │  Rateless, etc) │
│ - QBER required │    │ - No QBER       │    │                 │
│ - Single-shot   │    │ - Iterative     │    │                 │
│ - Virtual Graph │    │ - Hot-Start     │    │                 │
└────────┬────────┘    └────────┬────────┘    └─────────────────┘
         │                      │
         └──────────┬───────────┘
                    │ shared components
                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SHARED INFRASTRUCTURE                               │
├─────────────────┬───────────────────┬───────────────────┬───────────────────┤
│ MotherCodeMgr   │ LDPCCodec         │ LeakageTracker    │ HashVerifier      │
│ (NEW)           │ (NEW JIT facade)  │ (enhanced)        │ (unchanged)       │
│                 │                   │                   │                   │
│ - Single R=0.5  │ - encode()        │ - record()        │ - compute_hash()  │
│ - Hybrid pattrn │ - decode_base()   │ - record_reveal() │ - verify()        │
│ - Numba topol.  │ - decode_blind()  │ - should_abort()  │                   │
└─────────────────┴───────────────────┴───────────────────┴───────────────────┘
                    │
                    ▼ wraps
┌─────────────────────────────────────────────────────────────────────────────┐
│                         NUMBA KERNELS (scripts/numba_kernels.py)            │
├─────────────────────────────────────────────────────────────────────────────┤
│ encode_bitpacked_kernel()       - Bit-packed SpMV syndrome                  │
│ decode_bp_virtual_graph_kernel() - Baseline decoder (full graph, pattern)   │
│ decode_bp_hotstart_kernel()     - Blind decoder (message persist, freeze)   │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 14.1.4 File-by-File Wiring Instructions

**1. orchestrator.py → ReconciliationSession**

```python
# Current: caligo/reconciliation/orchestrator.py
class ReconciliationOrchestrator:
    def __init__(self, matrix_manager, leakage_tracker, config, safety_cap): ...
    
# Proposed: caligo/reconciliation/session.py (NEW FILE)
class ReconciliationSession:
    """Thin session coordinator, delegates to strategies."""
    
    def __init__(
        self,
        config: ReconciliationConfig,
        mother_code: MotherCodeManager,
        leakage_tracker: LeakageTracker,
    ) -> None:
        self._strategy = create_strategy(config, mother_code, ...)
        self._leakage_tracker = leakage_tracker
    
    def reconcile(self, ...):
        """Delegate to strategy.alice_reconcile_block() or bob_reconcile_block()."""
        ...
```

**2. matrix_manager.py → MotherCodeManager**

```python
# Current: Loads multiple rate matrices
class MatrixManager:
    def get_matrix(self, rate: float) -> sp.csr_matrix: ...

# Proposed: Single mother code + pattern lookup
class MotherCodeManager:
    def __init__(self, matrix_path, pattern_dir): ...
    def get_pattern(self, effective_rate: float) -> np.ndarray: ...
    def get_modulation_indices(self, d: int) -> np.ndarray: ...
```

**3. ldpc_encoder.py + ldpc_decoder.py → LDPCCodec**

```python
# Current: Separate encoder/decoder classes
class BeliefPropagationDecoder: ...
def encode_block_from_payload(...): ...

# Proposed: Unified JIT facade
class LDPCCodec:
    def encode(self, frame, pattern) -> np.ndarray: ...
    def decode_baseline(self, syndrome, llr, pattern) -> DecoderResult: ...
    def decode_blind(self, syndrome, llr, messages, frozen_mask) -> DecoderResult: ...
```

**4. blind_manager.py → BlindStrategy**

```python
# Current: Separate manager class
class BlindReconciliationManager: ...

# Proposed: Integrated into strategy
class BlindStrategy(ReconciliationStrategy):
    """Contains BlindDecoderState and iterative logic."""
    ...
```

**5. leakage_tracker.py → Enhanced Circuit Breaker**

```python
# Current: Tracks leakage, no enforcement
class LeakageTracker:
    def record(...): ...
    def should_abort(self) -> bool: ...

# Proposed: Immediate enforcement
class LeakageTracker:
    def record(self, ...):
        """Record and IMMEDIATELY raise if budget exceeded."""
        self._total_leakage += event.total
        if self._abort_on_exceed and self._total_leakage > self._safety_cap:
            raise LeakageBudgetExceeded(...)
```

---

### 14.2 Exception and Contract Wiring

This section documents the integration of reconciliation-specific exceptions and phase contracts with the proposed architecture.

#### 14.2.1 Exception Hierarchy

**File:** [types/exceptions.py](caligo/types/exceptions.py)

The existing exception hierarchy supports the refactored architecture:

```
CaligoError
├── SecurityError
│   ├── QBERThresholdExceeded       # QBER > 22% (hard limit)
│   ├── NSMViolationError           # Noisy Storage Model violation
│   ├── FeasibilityError            # Pre-flight check failure
│   ├── EntropyDepletedError        # No extractable entropy (Death Valley)
│   ├── CommitmentVerificationError # Commitment hash failure
│   └── SynchronizationError        # Alice/Bob metadata mismatch
├── ProtocolError
│   ├── PhaseOrderViolation
│   ├── ContractViolation           # Phase contract invariant violated
│   └── ReconciliationError
│       ├── DecodingFailure         # Single block decode failure
│       ├── DecodingFailed          # All retries exhausted
│       ├── LeakageBudgetExceeded   # Leakage > safety cap ← CIRCUIT BREAKER
│       ├── LeakageCapExceeded      # Alias for compatibility
│       ├── HashVerificationFailed  # Block hash mismatch
│       └── MatrixSynchronizationError
└── SimulationError
    └── ...
```

#### 14.2.2 LeakageBudgetExceeded for Circuit Breaker

**File:** [types/exceptions.py](caligo/types/exceptions.py#L178-L198)

```python
class LeakageBudgetExceeded(ReconciliationError):
    """
    Fatal error raised when reconciliation leakage exceeds its budget.
    
    This is the CIRCUIT BREAKER exception raised by LeakageTracker
    when cumulative leakage exceeds the safety cap.
    
    Attributes
    ----------
    actual_leakage : int
        Total leakage accumulated (bits).
    max_allowed : int
        Configured maximum leakage (bits).
    """
    
    def __init__(
        self,
        message: str,
        actual_leakage: int = 0,
        max_allowed: int = 0,
    ) -> None:
        super().__init__(message)
        self.actual_leakage = actual_leakage
        self.max_allowed = max_allowed
```

**Integration with LeakageTracker:**

```python
# reconciliation/leakage_tracker.py

class LeakageTracker:
    def record(self, block_id, syndrome_bits, hash_bits, revealed_bits, **kwargs):
        event = LeakageRecord(...)
        self._records.append(event)
        self._total_leakage += event.total
        
        # CIRCUIT BREAKER: Immediate enforcement
        if self._abort_on_exceed and self._total_leakage > self._safety_cap:
            raise LeakageBudgetExceeded(
                f"Cumulative leakage {self._total_leakage} exceeds "
                f"safety cap {self._safety_cap}",
                actual_leakage=self._total_leakage,
                max_allowed=self._safety_cap,
            )
```

#### 14.2.3 Phase Contract: ReconciliationPhaseResult

**File:** [types/phase_contracts.py](caligo/types/phase_contracts.py#L370-L450)

The `ReconciliationPhaseResult` contract defines the Phase III → Phase IV data transfer:

```python
@dataclass
class ReconciliationPhaseResult:
    """
    Contract: Phase III → Phase IV data transfer.
    
    Post-conditions:
    - POST-R-001: total_syndrome_bits ≤ leakage_cap (if cap set)
    - POST-R-002: hash_verified == True (else would abort)
    """
    
    reconciled_key: bitarray           # Error-corrected key
    num_blocks: int                    # Total LDPC blocks processed
    blocks_succeeded: int              # Blocks passing verification
    blocks_failed: int                 # Blocks discarded
    total_syndrome_bits: int           # |Σ| leakage in bits
    effective_rate: float              # R = (n - |Σ|) / n
    hash_verified: bool = True         # Final verification status
    leakage_within_cap: bool = True    # |Σ| ≤ L_max
    leakage_cap: Optional[int] = None  # Maximum allowed leakage
```

#### 14.2.4 Strategy → Contract Mapping

The strategy classes should produce `ReconciliationPhaseResult` or compatible `BlockResult` objects:

```python
# reconciliation/strategies/base.py

@dataclass
class BlockResult:
    """
    Result of single block reconciliation.
    
    Maps to ReconciliationPhaseResult when aggregated across blocks.
    
    Per Theoretical Report v2 §1.2, leakage accounting must be exact:
    leak_EC = syndrome_leakage + hash_leakage + revealed_leakage
    """
    corrected_payload: np.ndarray
    verified: bool
    converged: bool
    iterations_used: int
    syndrome_leakage: int       # |Σ| for this block
    revealed_leakage: int       # Blind only: revealed bits
    hash_leakage: int           # Hash bits leaked
    retry_count: int
    effective_rate: float = 0.5
    
    @property
    def total_leakage(self) -> int:
        """Per Theoretical Report v2 Eq. (leak_EC)."""
        return self.syndrome_leakage + self.revealed_leakage + self.hash_leakage
```

**Aggregation in ReconciliationSession:**

```python
# reconciliation/session.py (proposed)

def _aggregate_block_results(
    self,
    block_results: List[BlockResult],
) -> ReconciliationPhaseResult:
    """Aggregate BlockResults into ReconciliationPhaseResult contract."""
    
    succeeded = [r for r in block_results if r.verified]
    failed = [r for r in block_results if not r.verified]
    
    reconciled_bits = np.concatenate([r.corrected_payload for r in succeeded])
    total_syndrome = sum(r.syndrome_leakage for r in block_results)
    total_revealed = sum(r.revealed_leakage for r in block_results)
    total_hash = sum(r.hash_leakage for r in block_results)
    
    return ReconciliationPhaseResult(
        reconciled_key=bitarray_from_numpy(reconciled_bits),
        num_blocks=len(block_results),
        blocks_succeeded=len(succeeded),
        blocks_failed=len(failed),
        total_syndrome_bits=total_syndrome + total_revealed + total_hash,
        effective_rate=np.mean([r.effective_rate for r in succeeded]),
        hash_verified=all(r.verified for r in succeeded),
        leakage_within_cap=self._leakage_tracker.total_leakage <= self._safety_cap,
        leakage_cap=self._safety_cap,
    )
```

#### 14.2.5 Contract Validation Points

| Contract | Validation Point | Exception on Violation |
|----------|------------------|------------------------|
| `QuantumPhaseResult` | POST-Q-001..005 | `ContractViolation` |
| `SiftingPhaseResult` | POST-S-001..003 | `ContractViolation`, `QBERThresholdExceeded` |
| `SiftedKeyMaterial` | POST-SKM-001 | `ContractViolation` |
| `ChannelEstimate` | POST-CE-001..002 | `ContractViolation` |
| `ReconciliationPhaseResult` | POST-R-001..002 | `ContractViolation`, `LeakageBudgetExceeded` |
| `AmplificationPhaseResult` | POST-AMP-001..002 | `ContractViolation`, `EntropyDepletedError` |

---

## Appendix A: Configuration Schema

```yaml
# configs/reconciliation/master.yaml
#
# Master configuration for runtime protocol selection.
# Allows switching between Baseline and Blind via YAML.

reconciliation:
  # Protocol selection: "baseline" or "blind"
  # Switched at runtime without code changes
  strategy: "baseline"
  
  # Mother code settings (shared by both protocols)
  frame_size: 4096
  mother_rate: 0.5
  mother_matrix_path: "caligo/configs/ldpc_matrices/ldpc_ace_peg/matrix_4096_R0.5.npz"
  
  # Hybrid Pattern Library (Δ R = 0.01, ~40 patterns)
  # Contains both Untainted (R ≤ 0.625) and ACE-guided (R > 0.625) patterns
  pattern_directory: "caligo/configs/ldpc_matrices/hybrid_patterns/"
  
  # Numba kernel settings
  numba:
    cache_dir: ".numba_cache"
    parallel: true
    fastmath: true
  
  # Decoder settings (shared)
  decoder:
    max_iterations: 60
    convergence_threshold: 1e-6
  
  # Verification hash
  verification:
    hash_bits: 64
    hash_algorithm: "polynomial"
  
  # NSM leakage budget
  leakage:
    safety_cap_bits: 1000000
    abort_on_exceed: true


# configs/reconciliation/baseline.yaml (inherits from master)
reconciliation:
  strategy: "baseline"
  
  baseline:
    # Efficiency target for rate selection: R = 1 - f × h(QBER)
    efficiency_target: 1.1  # f ∈ [1.05, 1.2]
    
    # QBER estimation settings (from sifting phase)
    qber_estimation:
      # Source: qber.py::estimate_qber()
      sample_fraction: 0.1
      confidence_adjustment: 0.02  # Conservative uplift


# configs/reconciliation/blind.yaml (inherits from master)
reconciliation:
  strategy: "blind"
  
  blind:
    # Maximum iterations (Δ ≈ d/t)
    max_iterations: 3
    
    # Modulation budget δ = (p+s)/n
    # Per Theoretical Report v2 Remark 2.1:
    # For R_eff=0.9, need δ ≈ 0.44
    modulation_fraction: 0.44
    
    # NSM-gated variant settings
    nsm_gating:
      # Heuristic QBER from compute_qber_erven
      use_heuristic: true
      
      # Permissive starting-rate cap
      initial_shortening_fraction: 0.1
      
      # Iteration budget gating
      high_qber_threshold: 0.08
      extended_iterations: 5
```

---

## Appendix B: Migration Checklist

### Phase 0: Numba Kernel Foundation
- [ ] **Kernel Infrastructure**
  - [ ] Implement `encode_bitpacked_kernel` in `scripts/numba_kernels.py`
  - [ ] Implement `decode_bp_virtual_graph_kernel` for Baseline
  - [ ] Implement `decode_bp_hotstart_kernel` for Blind with Freeze optimization
  - [ ] Add kernel unit tests with known test vectors
  - [ ] Benchmark kernel performance vs scipy.sparse baseline

### Phase 1: Data Layer
- [ ] **Mother Code & Patterns**
  - [ ] Generate R=0.5 ACE-PEG mother matrix (existing `generate_ace_mother_code.py`)
  - [ ] Implement Hybrid Pattern Generation (`generate_hybrid_patterns.py`)
  - [ ] Generate Hybrid Pattern Library (Δ R = 0.01, ~40 patterns)
  - [ ] Validate untainted saturation at R ≈ 0.625
  - [ ] Deprecate multi-rate matrix loading in `matrix_manager.py`

- [ ] **Numba Topology**
  - [ ] Implement `NumbaGraphTopology` dataclass
  - [ ] Add topology compilation in `MotherCodeManager._compile_topology()`
  - [ ] Verify SoA format correctness

### Phase 2: Strategy Layer
- [ ] **Core Components**
  - [ ] Implement `ReconciliationStrategy` ABC with generator interface
  - [ ] Implement `ReconciliationContext` with all required fields
  - [ ] Implement `BlockResult` with leakage accounting
  - [ ] Create `StrategyFactory` for YAML-based instantiation

- [ ] **Baseline Strategy**
  - [ ] Implement `BaselineStrategy` with QBER requirement enforcement
  - [ ] Wire to `sifting/qber.py::estimate_qber()` for QBER source
  - [ ] Implement rate selection: R = 1 - f × h(QBER)
  - [ ] Use `LDPCCodec.decode_baseline()` (Virtual Graph kernel)

- [ ] **Blind Strategy**
  - [ ] Implement `BlindStrategy` with NSM-gated variant
  - [ ] Implement `BlindDecoderState` with message persistence
  - [ ] Wire heuristic QBER from `utils/math.py::compute_qber_erven()`
  - [ ] Use `LDPCCodec.decode_blind()` (Hot-Start kernel with Freeze)

- [ ] **LDPCCodec**
  - [ ] Implement `LDPCCodec` as JIT Interface Facade
  - [ ] Add bit-packing/unpacking utilities
  - [ ] Wire to Numba kernels

### Phase 3: Integration Layer
- [ ] **Session Management**
  - [ ] Implement `ReconciliationSession` replacing `ReconciliationOrchestrator`
  - [ ] Add strategy injection via factory
  - [ ] Wire `LeakageTracker` circuit breaker

- [ ] **Protocol Layer Refactoring**
  - [ ] Refactor `alice.py._phase3_reconcile()` to delegate to session
  - [ ] Refactor `bob.py` Phase III to delegate to session
  - [ ] Remove embedded reconciliation logic from role classes
  - [ ] Add Blind message types to `connection/envelope.py`
  - [ ] Ensure ACK/NACK handling via `OrderedSocket`

### Phase 4: Types & Exceptions
- [ ] **Type System**
  - [ ] Add `ReconciliationType` enum to `types/`
  - [ ] Add `LeakageBudgetExceeded` exception to `types/exceptions.py`
  - [ ] Add `ReconciliationPhaseContract` to `types/phase_contracts.py`
  - [ ] Ensure type hints throughout reconciliation module

### Phase 5: Testing & Validation
- [ ] **Unit Tests**
  - [ ] Strategy classes (mock network)
  - [ ] LDPCCodec encode/decode correctness
  - [ ] Numba kernel correctness
  - [ ] Leakage accounting exactness

- [ ] **Integration Tests**
  - [ ] Baseline protocol end-to-end
  - [ ] Blind protocol end-to-end
  - [ ] Protocol switching via YAML
  - [ ] Circuit breaker activation

- [ ] **Convergence Tests**
  - [ ] Hybrid pattern library validation
  - [ ] FER curves for Regime A vs Regime B
  - [ ] Hot-Start performance improvement measurement

- [ ] **Performance Tests**
  - [ ] Numba kernel throughput benchmarks
  - [ ] NSM timing constraint validation

---

## Appendix C: Hybrid Puncturing Pattern Generation Script

Per Theoretical Report v2 §2.2.3, the architecture requires a **Hybrid Pattern Library** that combines:
- **Phase I (Untainted / Regime A):** Strict untainted puncturing [3] until saturation ($R \approx 0.625$)
- **Phase II (ACE-Guided / Regime B):** ACE/EMD-based puncturing [4] for higher rates

This script replaces the previous single-regime approach.

```python
# scripts/generate_hybrid_patterns.py
"""
Hybrid Puncturing Pattern Generation for Caligo Reconciliation.

Implements the two-regime strategy from Theoretical Report v2 §2.2:
- Regime A: Untainted puncturing (Elkouss et al. 2012 [3])
- Regime B: ACE-guided puncturing (Liu & de Lamare 2014 [4])

Usage:
    python generate_hybrid_patterns.py --matrix path/to/mother.npz --output path/to/patterns/

References:
    [3] Elkouss et al., "Untainted Puncturing for Irregular LDPC Codes"
    [4] Liu & de Lamare, "Rate-Compatible LDPC Codes Based on Puncturing"
"""

import numpy as np
from scipy.sparse import csr_matrix
from pathlib import Path
from typing import Set, List, Tuple, Dict
from dataclasses import dataclass


# ============================================================================
# CONSTANTS
# ============================================================================

MOTHER_RATE = 0.5
FRAME_SIZE = 4096
RATE_STEP = 0.01  # Δ R = 0.01 for fine-grained rate adaptation
RATE_MIN = 0.51
RATE_MAX = 0.90

# Saturation threshold for Regime A → Regime B transition
# Per Theoretical Report v2 Theorem 2.2
UNTAINTED_SATURATION_RATE = 0.625


@dataclass
class PuncturingState:
    """Track state during hybrid pattern generation."""
    pattern: np.ndarray           # Current puncturing pattern (1=punctured)
    untainted_set: Set[int]       # Remaining untainted candidates
    punctured_order: List[int]    # Ordered list of punctured indices
    current_rate: float           # Current effective rate
    regime: str                   # 'untainted' or 'ace'


# ============================================================================
# GRAPH ANALYSIS FUNCTIONS
# ============================================================================

def compute_depth2_neighborhood(H: csr_matrix, symbol_idx: int) -> Set[int]:
    """
    Compute N²(v) - all symbols within 2 hops of symbol v.
    
    Per Theoretical Report v2 Definition 2.3:
    N²(v) = {v} ∪ {all symbols sharing a check with v}
    """
    # Get checks connected to symbol (1 hop)
    check_indices = H.getcol(symbol_idx).nonzero()[0]
    
    # Get all symbols connected to those checks (2 hops)
    neighbors = {symbol_idx}
    for check_idx in check_indices:
        symbol_indices = H.getrow(check_idx).nonzero()[1]
        neighbors.update(symbol_indices)
    
    return neighbors


def compute_ace_score(
    H: csr_matrix,
    symbol_idx: int,
    punctured: Set[int],
    max_cycle_length: int = 12,
) -> float:
    """
    Compute ACE (Approximate Cycle Extrinsic message degree) score.
    
    Per Theoretical Report v2 §2.2.3 and Liu & de Lamare [4]:
    - High ACE = Better connectivity = Safer to puncture
    - Low ACE = Weakly connected = Risky to puncture
    
    Parameters
    ----------
    H : csr_matrix
        Parity-check matrix.
    symbol_idx : int
        Symbol node to evaluate.
    punctured : Set[int]
        Already punctured symbol indices.
    max_cycle_length : int
        Maximum cycle length to consider.
        
    Returns
    -------
    float
        ACE score (higher is better for puncturing).
    """
    # Get checks connected to this symbol
    check_indices = H.getcol(symbol_idx).nonzero()[0]
    
    if len(check_indices) == 0:
        return 0.0
    
    # Compute extrinsic connectivity for each check
    ace_scores = []
    
    for check_idx in check_indices:
        # Get all symbols connected to this check
        neighbor_symbols = set(H.getrow(check_idx).nonzero()[1])
        neighbor_symbols.discard(symbol_idx)  # Exclude self
        
        # Count non-punctured neighbors (extrinsic connections)
        extrinsic_count = len(neighbor_symbols - punctured)
        
        # Check degree minus 1 (excluding the current symbol)
        check_degree = len(neighbor_symbols)
        
        if check_degree > 0:
            ace_scores.append(extrinsic_count / check_degree)
    
    # Return minimum ACE across all connected checks
    # Per [4]: puncture nodes with HIGH minimum ACE first
    return min(ace_scores) if ace_scores else 0.0


def compute_n2_size(H: csr_matrix, symbol_idx: int) -> int:
    """Compute |N²(v)| for tie-breaking in untainted selection."""
    return len(compute_depth2_neighborhood(H, symbol_idx))


# ============================================================================
# REGIME A: UNTAINTED PUNCTURING
# ============================================================================

def run_untainted_phase(
    H: csr_matrix,
    state: PuncturingState,
    target_fraction: float,
) -> bool:
    """
    Run strict untainted puncturing (Regime A).
    
    Per Theoretical Report v2 §2.2.2 Algorithm:
    1. Select candidates with smallest |N²(v)|
    2. Puncture one (random tie-breaking)
    3. Remove N²(v) from untainted set
    4. Repeat until target or saturation
    
    Parameters
    ----------
    H : csr_matrix
        Parity-check matrix.
    state : PuncturingState
        Current puncturing state (modified in-place).
    target_fraction : float
        Target puncturing fraction π.
        
    Returns
    -------
    bool
        True if target reached, False if saturated early.
    """
    n = H.shape[1]
    target_punctured = int(target_fraction * n)
    
    # Precompute N² sizes for remaining candidates
    n2_sizes = {v: compute_n2_size(H, v) for v in state.untainted_set}
    
    while state.untainted_set and len(state.punctured_order) < target_punctured:
        # Step 1: Find candidates with smallest N² size
        min_size = min(n2_sizes[v] for v in state.untainted_set)
        candidates = [v for v in state.untainted_set if n2_sizes[v] == min_size]
        
        # Step 2: Select one (random tie-breaking for determinism, use sorted)
        selected = min(candidates)  # Deterministic: choose smallest index
        
        # Step 3: Puncture selected
        state.pattern[selected] = 1
        state.punctured_order.append(selected)
        
        # Step 4: Remove N²(selected) from untainted set
        n2_selected = compute_depth2_neighborhood(H, selected)
        state.untainted_set -= n2_selected
        
        # Clean up n2_sizes for removed nodes
        for v in n2_selected:
            n2_sizes.pop(v, None)
    
    # Check if target reached
    return len(state.punctured_order) >= target_punctured


# ============================================================================
# REGIME B: ACE-GUIDED PUNCTURING
# ============================================================================

def run_ace_phase(
    H: csr_matrix,
    state: PuncturingState,
    target_fraction: float,
) -> bool:
    """
    Run ACE-guided puncturing (Regime B).
    
    Per Theoretical Report v2 §2.2.3:
    - Used when untainted set is exhausted
    - Rank candidates by ACE score (highest first)
    - Puncture to preserve graph connectivity
    
    Parameters
    ----------
    H : csr_matrix
        Parity-check matrix.
    state : PuncturingState
        Current puncturing state (modified in-place).
    target_fraction : float
        Target puncturing fraction π.
        
    Returns
    -------
    bool
        True if target reached.
    """
    n = H.shape[1]
    target_punctured = int(target_fraction * n)
    punctured_set = set(state.punctured_order)
    
    state.regime = 'ace'
    
    while len(state.punctured_order) < target_punctured:
        # Get remaining unpunctured symbols
        remaining = [i for i in range(n) if state.pattern[i] == 0]
        
        if not remaining:
            break
        
        # Compute ACE scores for all remaining candidates
        ace_scores = {
            v: compute_ace_score(H, v, punctured_set)
            for v in remaining
        }
        
        # Select candidate with HIGHEST ACE score
        # Per [4]: high ACE = well-connected = safer to puncture
        selected = max(remaining, key=lambda v: (ace_scores[v], -v))
        
        # Puncture selected
        state.pattern[selected] = 1
        state.punctured_order.append(selected)
        punctured_set.add(selected)
    
    return len(state.punctured_order) >= target_punctured


# ============================================================================
# MAIN GENERATION LOGIC
# ============================================================================

def generate_hybrid_library(
    H: csr_matrix,
    output_dir: Path,
    rate_min: float = RATE_MIN,
    rate_max: float = RATE_MAX,
    rate_step: float = RATE_STEP,
) -> Dict[float, np.ndarray]:
    """
    Generate Hybrid Pattern Library covering R_eff ∈ [rate_min, rate_max].
    
    Per Theoretical Report v2 §2.2.3:
    1. Phase I: Untainted puncturing until saturation (~R=0.625)
    2. Phase II: ACE-guided puncturing for higher rates
    3. Rate-compatibility: patterns are nested (truncatable)
    
    Parameters
    ----------
    H : csr_matrix
        R=0.5 mother code parity-check matrix.
    output_dir : Path
        Directory to save pattern files.
    rate_min : float
        Minimum effective rate (default 0.51).
    rate_max : float
        Maximum effective rate (default 0.90).
    rate_step : float
        Rate step size (default 0.01).
        
    Returns
    -------
    Dict[float, np.ndarray]
        Dictionary of {rate: pattern} mappings.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    n = H.shape[1]
    
    # Initialize state
    state = PuncturingState(
        pattern=np.zeros(n, dtype=np.uint8),
        untainted_set=set(range(n)),
        punctured_order=[],
        current_rate=MOTHER_RATE,
        regime='untainted',
    )
    
    patterns = {}
    
    # Generate rates from rate_min to rate_max
    current_rate = rate_min
    while current_rate <= rate_max + 1e-6:
        # Compute required puncture fraction for this rate
        # R_eff = R_0 / (1 - π) => π = 1 - R_0 / R_eff
        target_puncture_fraction = 1.0 - MOTHER_RATE / current_rate
        
        if target_puncture_fraction <= 0:
            current_rate += rate_step
            continue
        
        # Try untainted puncturing first (Regime A)
        if state.regime == 'untainted':
            reached = run_untainted_phase(H, state, target_puncture_fraction)
            
            if not reached and not state.untainted_set:
                # Untainted set exhausted → transition to ACE (Regime B)
                print(f"  Untainted saturation at π={len(state.punctured_order)/n:.3f}, "
                      f"R_eff={MOTHER_RATE/(1-len(state.punctured_order)/n):.3f}")
                run_ace_phase(H, state, target_puncture_fraction)
        else:
            # Already in Regime B (ACE)
            run_ace_phase(H, state, target_puncture_fraction)
        
        # Compute actual achieved rate
        actual_punctured = state.pattern.sum()
        actual_rate = MOTHER_RATE / (1 - actual_punctured / n)
        
        # Save pattern
        filename = f"pattern_rate{actual_rate:.2f}.npy"
        np.save(output_dir / filename, state.pattern.copy())
        patterns[actual_rate] = state.pattern.copy()
        
        print(f"Generated {filename}: π={actual_punctured/n:.3f}, "
              f"R_eff={actual_rate:.3f}, regime={state.regime}")
        
        state.current_rate = actual_rate
        current_rate += rate_step
    
    # Save ordered indices for Blind protocol revelation order
    indices_path = output_dir / "modulation_indices.npy"
    np.save(indices_path, np.array(state.punctured_order, dtype=np.int64))
    print(f"Saved modulation indices to {indices_path}")
    
    return patterns


if __name__ == "__main__":
    import argparse
    import scipy.sparse as sp
    
    parser = argparse.ArgumentParser(
        description="Generate Hybrid Puncturing Pattern Library"
    )
    parser.add_argument(
        "--matrix", "-m",
        type=Path,
        default=Path("configs/ldpc_matrices/mother_code_4096_R0.5.npz"),
        help="Path to R=0.5 mother matrix (.npz)",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("configs/ldpc_matrices/hybrid_patterns/"),
        help="Output directory for pattern files",
    )
    parser.add_argument(
        "--rate-min",
        type=float,
        default=RATE_MIN,
        help=f"Minimum effective rate (default {RATE_MIN})",
    )
    parser.add_argument(
        "--rate-max",
        type=float,
        default=RATE_MAX,
        help=f"Maximum effective rate (default {RATE_MAX})",
    )
    args = parser.parse_args()
    
    print(f"Loading mother code from {args.matrix}")
    H = sp.load_npz(args.matrix).tocsr()
    print(f"Matrix shape: {H.shape}, rate: {1 - H.shape[0]/H.shape[1]:.3f}")
    
    print(f"\nGenerating Hybrid Pattern Library:")
    print(f"  Rate range: [{args.rate_min}, {args.rate_max}]")
    print(f"  Rate step: {RATE_STEP}")
    print(f"  Output: {args.output}\n")
    
    patterns = generate_hybrid_library(
        H,
        args.output,
        rate_min=args.rate_min,
        rate_max=args.rate_max,
    )
    
    print(f"\nGenerated {len(patterns)} patterns.")
```

**Key Features:**

1. **Two-Regime Strategy:** Automatically transitions from Untainted (Regime A) to ACE-guided (Regime B) when the untainted candidate set is exhausted.

2. **Rate-Compatibility (Nesting):** The `punctured_order` list maintains the puncturing sequence, ensuring patterns at higher rates are supersets of lower-rate patterns. This enables the Blind protocol to reveal bits in a deterministic order.

3. **Fine-Grained Steps:** $\Delta R = 0.01$ generates approximately 40 pattern files, maximizing reconciliation efficiency $f$.

4. **Modulation Indices:** Saves the ordered puncturing indices for the Blind protocol's fixed revelation order (per Theoretical Report v2 §4.3).

---

## Appendix D: Final Implementation Plan and Definition of Done

This appendix provides the authoritative implementation roadmap, testing strategy, and definition of done for the Caligo reconciliation refactoring.

### D.1 Implementation Roadmap

#### D.1.1 Phase 0: Numba Kernel Foundation (Week 0-1)

| Task ID | Task | Inputs | Outputs | Dependencies |
|---------|------|--------|---------|--------------|
| P0.1 | Implement `encode_bitpacked_kernel` | CSR topology arrays | Syndrome (bit-packed) | None |
| P0.2 | Implement `decode_bp_virtual_graph_kernel` | LLR, syndrome, pattern mask | Decoded bits | P0.1 |
| P0.3 | Implement `decode_bp_hotstart_kernel` | LLR, syndrome, messages, frozen_mask | Decoded bits + messages | P0.2 |
| P0.4 | Unit test kernels with known vectors | Test matrices | Pass/fail | P0.1-P0.3 |
| P0.5 | Benchmark vs scipy.sparse baseline | Benchmark script | Speedup metrics | P0.4 |

**Exit Criteria:**
- [ ] All three kernels compile without errors
- [ ] Unit tests pass with 100% coverage of kernel code paths
- [ ] Measured speedup ≥ 5x vs scipy.sparse for encode, ≥ 3x for decode

#### D.1.2 Phase 1: Data Layer (Week 1-2)

| Task ID | Task | Inputs | Outputs | Dependencies |
|---------|------|--------|---------|--------------|
| P1.1 | Generate R=0.5 ACE-PEG mother matrix | ACE parameters | `mother_code_4096_R0.5.npz` | None |
| P1.2 | Implement `generate_hybrid_patterns.py` | Mother matrix | Hybrid Pattern Library | P1.1 |
| P1.3 | Generate Hybrid Pattern Library | Mother matrix, script | 40 pattern files | P1.2 |
| P1.4 | Implement `MotherCodeManager` | Matrix path, pattern dir | Singleton manager | P1.3 |
| P1.5 | Implement `NumbaGraphTopology` | CSR matrix | SoA topology arrays | P1.4 |
| P1.6 | Validate untainted saturation | Pattern library | R_sat ≈ 0.625 ± 0.02 | P1.3 |

**Exit Criteria:**
- [ ] Mother matrix passes ACE validation (minimum ACE ≥ 2)
- [ ] Pattern library covers R_eff ∈ [0.51, 0.90] with Δ R = 0.01
- [ ] Saturation point documented and matches Theorem 2.2
- [ ] `MotherCodeManager` singleton loads in < 100ms

#### D.1.3 Phase 2: Strategy Layer (Week 2-3)

| Task ID | Task | Inputs | Outputs | Dependencies |
|---------|------|--------|---------|--------------|
| P2.1 | Define `ReconciliationStrategy` ABC | Spec | Abstract base class | None |
| P2.2 | Define `ReconciliationContext` | Spec | Immutable dataclass | None |
| P2.3 | Define `BlockResult` | Spec | Result dataclass | None |
| P2.4 | Implement `LDPCCodec` JIT facade | Numba kernels | Codec class | P0.1-P0.3 |
| P2.5 | Implement `BaselineStrategy` | ABC, Codec | Baseline class | P2.1-P2.4 |
| P2.6 | Implement `BlindStrategy` | ABC, Codec | Blind class | P2.1-P2.4 |
| P2.7 | Implement `BlindDecoderState` | Spec | State dataclass | P2.6 |
| P2.8 | Implement strategy factory | Config, strategies | Factory function | P2.5-P2.6 |

**Exit Criteria:**
- [ ] `BaselineStrategy` passes unit tests with mock network
- [ ] `BlindStrategy` passes unit tests with mock network
- [ ] Factory correctly instantiates strategy from YAML config
- [ ] Generator interface compatible with SquidASM scheduling

#### D.1.4 Phase 3: Integration (Week 3-4)

| Task ID | Task | Inputs | Outputs | Dependencies |
|---------|------|--------|---------|--------------|
| P3.1 | Implement `ReconciliationSession` | Strategy, LeakageTracker | Session class | P2.5-P2.8 |
| P3.2 | Refactor `alice.py` to delegate | Strategy factory | Clean role class | P3.1 |
| P3.3 | Refactor `bob.py` to delegate | Strategy factory | Clean role class | P3.1 |
| P3.4 | Enhance `LeakageTracker` circuit breaker | Spec | Enhanced tracker | None |
| P3.5 | Wire QBER estimation (Baseline only) | `sifting/qber.py` | Integrated flow | P3.2-P3.3 |
| P3.6 | Wire heuristic QBER (Blind optional) | `utils/math.py` | Integrated flow | P3.2-P3.3 |
| P3.7 | Update `ReconciliationConfig` YAML schema | Spec | Updated schema | P3.1 |

**Exit Criteria:**
- [ ] `alice.py` has no embedded protocol logic (< 10 lines per protocol)
- [ ] `bob.py` handles both Baseline and Blind message types
- [ ] LeakageTracker raises `LeakageBudgetExceeded` on overflow
- [ ] YAML-based protocol switching works without code changes

#### D.1.5 Phase 4: Testing & Validation (Week 4-5)

| Task ID | Task | Inputs | Outputs | Dependencies |
|---------|------|--------|---------|--------------|
| P4.1 | Unit tests for strategy classes | Pytest | Test suite | P2.5-P2.7 |
| P4.2 | Integration tests with mock network | Pytest | Test suite | P3.1-P3.3 |
| P4.3 | Convergence tests for hybrid patterns | Statistical validation | FER curves | P1.3 |
| P4.4 | Leakage accounting tests | Pytest | Test suite | P3.4 |
| P4.5 | NSM timing validation | Simulation | Timing report | P3.1 |
| P4.6 | Performance benchmarks | Benchmark script | Performance report | P0.5, P3.1 |
| P4.7 | End-to-end integration test | Full simulation | Pass/fail | P3.1-P3.6 |

**Exit Criteria:**
- [ ] All unit tests pass with ≥ 90% coverage
- [ ] Integration tests pass for both protocols
- [ ] FER < 10% at R_eff = 0.8 for QBER = 0.05
- [ ] NSM timing constraints satisfied (Δt enforced)
- [ ] Throughput meets minimum 10 Kb/s for n=4096

---

### D.2 Testing Strategy

#### D.2.1 Test Levels

| Level | Scope | Tools | Coverage Target |
|-------|-------|-------|-----------------|
| **Unit** | Individual classes/functions | pytest, unittest.mock | ≥ 90% |
| **Component** | Module interactions | pytest, fixtures | ≥ 80% |
| **Integration** | End-to-end protocol flow | SquidASM simulation | Critical paths |
| **Performance** | Throughput, latency | pytest-benchmark, cProfile | All Numba kernels |

#### D.2.2 Key Test Cases

**Baseline Strategy:**
1. Single-block encode/decode with known QBER
2. Rate selection matches Shannon limit within f_crit
3. Hash verification failure triggers block exclusion
4. Leakage accounting matches syndrome length
5. YAML config switches to Baseline correctly

**Blind Strategy:**
1. Multi-iteration convergence from maximum rate
2. LLR persistence across iterations (Hot-Start)
3. Revelation order matches modulation_indices.npy
4. Leakage includes revealed bits + syndrome + hash
5. NSM-gated optimization reduces average iterations

**Hybrid Patterns:**
1. Regime A patterns satisfy untainted property
2. Regime B patterns have higher ACE scores
3. Rate-nesting property: R_high ⊃ R_low
4. Saturation transition at R ≈ 0.625

**Numba Kernels:**
1. Bit-packed encode matches scalar reference
2. Virtual Graph decode matches reference BP
3. Hot-Start decode converges faster with prior messages
4. Freeze optimization skips frozen nodes

#### D.2.3 Regression Test Suite

```bash
# Run full test suite
pytest tests/ -v --cov=caligo.reconciliation --cov-report=html

# Run specific test categories
pytest tests/test_strategies.py -v
pytest tests/test_numba_kernels.py -v
pytest tests/test_integration.py -v

# Performance benchmarks
pytest tests/test_performance.py --benchmark-only
```

---

### D.3 Definition of Done

#### D.3.1 Feature-Level DoD

A feature is **DONE** when:

1. **Code Complete:**
   - [ ] Implementation matches specification in this report
   - [ ] Type hints on all public functions
   - [ ] Numpydoc docstrings on all public functions
   - [ ] No `# TODO` or `# FIXME` comments in shipped code

2. **Tested:**
   - [ ] Unit tests pass with ≥ 90% coverage
   - [ ] Integration tests pass
   - [ ] Performance benchmarks meet targets

3. **Documented:**
   - [ ] Docstrings complete and accurate
   - [ ] Implementation Report updated if design changed
   - [ ] CHANGELOG entry added

4. **Reviewed:**
   - [ ] Code review passed
   - [ ] Security review for NSM-critical code

#### D.3.2 Module-Level DoD

| Module | DoD Criteria |
|--------|-------------|
| `MotherCodeManager` | Single matrix, hybrid patterns load, topology compiled |
| `LDPCCodec` | All three kernels benchmarked, speedup documented |
| `BaselineStrategy` | QBER→rate selection verified, leakage accounting exact |
| `BlindStrategy` | Hot-Start measured, Freeze optimization benchmarked |
| `ReconciliationSession` | Both protocols pass E2E test, YAML switching works |
| `LeakageTracker` | Circuit breaker tested, exception attributes correct |

#### D.3.3 System-Level DoD

The refactoring is **COMPLETE** when:

1. **Functional Requirements:**
   - [ ] Baseline protocol produces correct OT output
   - [ ] Blind protocol produces correct OT output
   - [ ] Protocol switching via YAML works at runtime
   - [ ] Hybrid Pattern Library covers R_eff ∈ [0.51, 0.90]

2. **Performance Requirements:**
   - [ ] Throughput ≥ 10 Kb/s for n=4096, QBER=0.05
   - [ ] Numba kernels ≥ 5x faster than scipy baseline
   - [ ] Memory usage < 1 GB for full simulation

3. **Security Requirements:**
   - [ ] Leakage accounting exact (no silent overflows)
   - [ ] LeakageBudgetExceeded raised before cap exceeded
   - [ ] NSM timing constraint (Δt) enforced

4. **Quality Requirements:**
   - [ ] All tests pass (unit, integration, performance)
   - [ ] Code coverage ≥ 90% for new code
   - [ ] No critical linter warnings (ruff, mypy)
   - [ ] Documentation complete and reviewed

---

### D.4 Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Numba kernel compilation fails on target platform | Medium | High | CI on multiple Python versions, test on SquidASM image |
| Hybrid patterns don't converge at high rates | Medium | High | ACE threshold tuning, FER curve validation |
| Hot-Start doesn't reduce iteration count | Low | Medium | Fall back to standard BP, accept higher latency |
| YAML config migration breaks existing tests | High | Low | Backward-compatible defaults, deprecation warnings |
| LeakageTracker race condition in parallel blocks | Low | High | Single-threaded leakage recording, explicit locks |

---

### D.5 Dependencies and Prerequisites

| Dependency | Version | Purpose |
|------------|---------|---------|
| NumPy | ≥ 1.24 | Array operations |
| SciPy | ≥ 1.10 | Sparse matrix operations |
| Numba | ≥ 0.57 | JIT compilation |
| PyYAML | ≥ 6.0 | Configuration loading |
| SquidASM | ≥ 0.13 | Quantum network simulation |
| pytest | ≥ 7.0 | Testing framework |
| pytest-cov | ≥ 4.0 | Coverage reporting |
| pytest-benchmark | ≥ 4.0 | Performance testing |

---

*End of Implementation Report v2*
