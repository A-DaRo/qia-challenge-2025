# Implementation Report v2: Architectural Blueprint for Rate-Compatible Reconciliation in Caligo

**Version:** 2.0  
**Date:** December 2025  
**Module:** `caligo.reconciliation`  
**Based on:** Theoretical Report v2 (Elkouss/Martinez-Mateo)

---

## Executive Summary

This report provides a comprehensive architectural blueprint for transitioning the Caligo reconciliation layer from its current state—exhibiting multiple anti-patterns and a mathematically flawed random puncturing scheme—to a robust **Unified Mother Code Architecture**. The refactoring enables two distinct, rate-compatible protocols (**Baseline** and **Blind**) to operate on a single, optimized R=0.5 ACE-PEG matrix with untainted puncturing patterns.

The implementation addresses critical security requirements under the Noisy Storage Model (NSM) by enforcing strict leakage accounting at architectural boundaries.

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

**Problem:** The pattern loading in `matrix_manager.py` expects patterns in `puncture_patterns/` subdirectory, but:

1. **Patterns are not guaranteed to be untainted:** The generation script must implement the untainted algorithm from Section 2.2.2 of the Theoretical Report.

2. **Rate selector hardcoded to R=0.5:** Due to convergence failures, [rate_selector.py#L110-L125](caligo/reconciliation/rate_selector.py#L110-L125) bypasses rate selection:

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

**Root Cause Analysis:** The convergence failures at higher rates (0.6, 0.7, 0.8) documented in the comments are symptomatic of:
1. **Tainted puncturing patterns:** Random or naive patterns create stopping sets
2. **Insufficient pattern generation:** No implementation of the untainted algorithm
3. **Multi-matrix architecture mismatch:** Loading multiple physical matrices instead of one mother + patterns

**Evidence from Terminal History:** The last terminal commands show failed attempts to generate matrices:
```
python -m caligo.scripts.generate_ace_mother_code --block-length 65536 --irregular --output-path /ca
ligo/configs/ldpc_matrices/ldpc_ace_peg
Exit Code: 1

python -m caligo.scripts.generate_ace_mother_code --block-length 1024 --irregular --output-path /tmp/test_ldpc.npz 
Exit Code: 1
```

This confirms the matrix generation tooling is incomplete or broken, preventing proper untainted pattern generation.

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
3. Does not implement the syndrome reuse pattern from Martinez-Mateo

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

The fundamental requirement of Blind reconciliation—**maintaining decoder LLR state across network round-trips**—is not addressed.

#### 2.5 Summary: Current State Deficiencies

| Deficiency | Severity | Files Affected |
|------------|----------|----------------|
| Random puncturing causes high-rate failures | Critical | `ldpc_encoder.py`, `matrix_manager.py` |
| Rate selector bypassed to single rate | High | `rate_selector.py` |
| God Object orchestrator | Medium | `orchestrator.py` |
| Protocol logic in role classes | High | `alice.py`, `bob.py` |
| Blind protocol not integrated | High | `blind_manager.py` |
| No circuit-breaker for leakage | Medium | `leakage_tracker.py` |

---

## Part II: Target Architecture

### 3. Design Principles

1. **Single Mother Code:** One R=0.5 ACE-PEG matrix with offline-generated untainted puncturing patterns
2. **Strategy Pattern:** Protocol-agnostic orchestration with injectable Baseline/Blind strategies
3. **Explicit State Machines:** Blind decoder maintains LLR state across iterations
4. **Strict Leakage Boundaries:** Every component that leaks information reports to a centralized tracker
5. **Testability:** Small, focused classes with clear interfaces

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
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │              ReconciliationSession (Context)                    │    │
│  │  - Manages per-session state                                    │    │
│  │  - Coordinates strategy execution                               │    │
│  │  - Tracks cumulative leakage                                    │    │
│  └────────┬───────────────────────────────────────────┬───────────┘    │
│           │                                           │                 │
│           │ ┌─────────────────────────────────────────┼───────────┐    │
│           │ │        ReconciliationStrategy (ABC)     │           │    │
│           ▼ │                                         ▼           │    │
│  ┌─────────────────┐                       ┌─────────────────────┐│    │
│  │ BaselineStrategy│                       │   BlindStrategy     ││    │
│  │                 │                       │                     ││    │
│  │ - Single-shot   │                       │ - Iterative reveal  ││    │
│  │ - QBER required │                       │ - LLR persistence   ││    │
│  └────────┬────────┘                       └─────────┬───────────┘│    │
│           │                                          │            │    │
│           └──────────────────┬───────────────────────┘            │    │
│                              │                                    │    │
└──────────────────────────────┼────────────────────────────────────┘    │
                               │                                          │
┌──────────────────────────────┼──────────────────────────────────────────┘
│                              │                                          
│                  RECONCILIATION ENGINE LAYER                            
│                              │                                          
│           ┌──────────────────┴──────────────────┐                       
│           ▼                                     ▼                       
│  ┌─────────────────────┐             ┌─────────────────────┐           
│  │  MotherCodeManager  │             │   LeakageTracker    │           
│  │                     │             │   (Circuit Breaker) │           
│  │  - Single R=0.5 H   │             │                     │           
│  │  - Pattern library  │             │  - Cumulative sum   │           
│  │  - Compiled cache   │             │  - Budget check     │           
│  └──────────┬──────────┘             │  - SecurityError    │           
│             │                        └─────────────────────┘           
│             ▼                                                          
│  ┌─────────────────────────────────────────────────────────┐           
│  │                    LDPCCodec                             │           
│  │  - encode_block(): payload + pattern → syndrome          │           
│  │  - decode_block(): syndrome + LLRs → corrected_bits      │           
│  │  - Three-state LLR builder                               │           
│  └─────────────────────────────────────────────────────────┘           
│                                                                         
└─────────────────────────────────────────────────────────────────────────
```

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
        LDPC frame size (n).
    mother_rate : float
        Mother code rate (R_0 = 0.5).
    max_iterations : int
        BP decoder max iterations.
    hash_bits : int
        Verification hash length.
    f_crit : float
        Target efficiency threshold.
    qber_prior : float
        NSM prior QBER (for LLR initialization in Blind).
    """
    session_id: int
    frame_size: int
    mother_rate: float
    max_iterations: int
    hash_bits: int
    f_crit: float
    qber_prior: float


@dataclass
class BlockResult:
    """
    Result of single block reconciliation.
    
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
        Syndrome bits leaked (constant per block).
    revealed_leakage : int
        Revealed bits leaked (Blind only, 0 for Baseline).
    hash_leakage : int
        Hash bits leaked.
    retry_count : int
        Number of blind iterations (1 for Baseline).
    """
    corrected_payload: np.ndarray
    verified: bool
    converged: bool
    iterations_used: int
    syndrome_leakage: int
    revealed_leakage: int
    hash_leakage: int
    retry_count: int
    
    @property
    def total_leakage(self) -> int:
        """Total leakage for this block."""
        return self.syndrome_leakage + self.revealed_leakage + self.hash_leakage


class ReconciliationStrategy(ABC):
    """
    Abstract base for reconciliation strategies.
    
    Strategies are stateless protocol implementations. Per-block state
    (e.g., LLRs across Blind iterations) is managed via yielded messages
    to the network layer.
    
    The generator-based interface supports SquidASM's cooperative scheduling.
    """
    
    @property
    @abstractmethod
    def requires_qber_estimation(self) -> bool:
        """
        Whether this strategy requires QBER pre-estimation.
        
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
            QBER estimate (required for Baseline).
            
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
    """
    
    def __init__(
        self,
        mother_code: MotherCodeManager,
        leakage_tracker: LeakageTracker,
    ) -> None:
        self._mother_code = mother_code
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
        """
        if qber_estimate is None:
            raise ValueError("Baseline requires QBER estimate")
        
        # 1. Select effective rate based on QBER
        effective_rate = self._select_rate(qber_estimate, ctx.f_crit)
        
        # 2. Get untainted puncture pattern for this rate
        pattern = self._mother_code.get_pattern(effective_rate)
        
        # 3. Construct frame and compute syndrome
        frame = self._construct_frame(payload, pattern, ctx.frame_size)
        syndrome = self._mother_code.compute_syndrome(frame)
        
        # 4. Compute verification hash
        hash_value = compute_hash(payload, seed=block_id)
        
        # 5. Record leakage
        self._leakage_tracker.record(
            block_id=block_id,
            syndrome_bits=len(syndrome),
            hash_bits=ctx.hash_bits,
            revealed_bits=0,
        )
        
        # 6. Send syndrome message to Bob
        response = yield {
            "kind": "baseline",
            "block_id": block_id,
            "syndrome": syndrome,
            "pattern_id": effective_rate,  # Bob looks up pattern by rate
            "payload_length": len(payload),
            "hash_value": hash_value,
            "qber_channel": qber_estimate,  # For Bob's LLR construction
        }
        
        # 7. Check Bob's response
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
        )
    
    def bob_reconcile_block(
        self,
        payload: np.ndarray,
        ctx: ReconciliationContext,
        block_id: int,
    ) -> Generator[Dict[str, Any], Dict[str, Any], BlockResult]:
        """
        Bob receives syndrome and decodes.
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
        
        # 2. Get puncture pattern
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
        
        # 4. Decode
        decoder = BeliefPropagationDecoder(max_iterations=ctx.max_iterations)
        result = decoder.decode(llr, syndrome, self._mother_code.compiled_H)
        
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
        )
```

#### 5.3 Blind Strategy Implementation with State Machine

```python
# caligo/reconciliation/strategies/blind.py

@dataclass
class BlindDecoderState:
    """
    Persistent decoder state across Blind iterations.
    
    This is the KEY architectural difference from Baseline:
    Bob maintains LLR state and incrementally updates it as Alice
    reveals shortened values.
    """
    llr: np.ndarray                    # Current LLR array
    puncture_indices: np.ndarray       # Originally punctured positions
    shortened_indices: np.ndarray      # Currently shortened positions (grows)
    shortened_values: np.ndarray       # Values at shortened positions (grows)
    iteration: int                     # Current iteration number
    syndrome: np.ndarray               # Fixed syndrome (computed once by Alice)


class BlindStrategy(ReconciliationStrategy):
    """
    Martinez-Mateo et al. (2012) blind reconciliation.
    
    Iterative protocol without QBER pre-estimation. Bob maintains
    decoder state across iterations; Alice progressively reveals
    punctured values.
    """
    
    def __init__(
        self,
        mother_code: MotherCodeManager,
        leakage_tracker: LeakageTracker,
        max_blind_iterations: int = 3,
        modulation_fraction: float = 0.1,
    ) -> None:
        self._mother_code = mother_code
        self._leakage_tracker = leakage_tracker
        self._max_iterations = max_blind_iterations
        self._delta = modulation_fraction
    
    @property
    def requires_qber_estimation(self) -> bool:
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
        """
        # 1. Compute modulation parameters
        d = int(self._delta * ctx.frame_size)  # Total modulation bits
        delta_step = max(1, d // self._max_iterations)  # Bits per iteration
        
        # 2. Initially: all d bits are punctured (highest rate attempt)
        puncture_indices = self._mother_code.get_modulation_indices(d)
        
        # 3. Construct frame with all punctured positions
        frame = self._construct_frame_with_padding(
            payload, puncture_indices, ctx.frame_size
        )
        padding_values = frame[puncture_indices]  # Save for potential reveal
        
        # 4. Compute syndrome ONCE
        syndrome = self._mother_code.compute_syndrome(frame)
        hash_value = compute_hash(payload, seed=block_id)
        
        # 5. Record initial syndrome leakage
        self._leakage_tracker.record(
            block_id=block_id,
            syndrome_bits=len(syndrome),
            hash_bits=ctx.hash_bits,
            revealed_bits=0,
        )
        
        # 6. Send initial syndrome (no shortened values yet)
        response = yield {
            "kind": "blind",
            "block_id": block_id,
            "syndrome": syndrome,
            "puncture_indices": puncture_indices,
            "payload_length": len(payload),
            "hash_value": hash_value,
            "qber_prior": ctx.qber_prior,
            "iteration": 1,
            "revealed_indices": np.array([], dtype=np.int64),
            "revealed_values": np.array([], dtype=np.uint8),
        }
        
        # 7. Iterative reveal loop
        iteration = 1
        total_revealed = 0
        n_shortened = 0
        
        while not response.get("converged") and iteration < self._max_iterations:
            iteration += 1
            
            # Select next batch of punctured bits to reveal (shorten)
            reveal_start = (iteration - 1) * delta_step
            reveal_end = min(iteration * delta_step, d)
            reveal_indices = puncture_indices[reveal_start:reveal_end]
            reveal_values = padding_values[reveal_start:reveal_end]
            
            n_shortened += len(reveal_indices)
            total_revealed += len(reveal_indices)
            
            # Record additional leakage
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
        
        return BlockResult(
            corrected_payload=payload,
            verified=verified,
            converged=response.get("converged", False),
            iterations_used=0,
            syndrome_leakage=len(syndrome),
            revealed_leakage=total_revealed,
            hash_leakage=ctx.hash_bits,
            retry_count=iteration,
        )
    
    def bob_reconcile_block(
        self,
        payload: np.ndarray,
        ctx: ReconciliationContext,
        block_id: int,
    ) -> Generator[Dict[str, Any], Dict[str, Any], BlockResult]:
        """
        Bob decodes iteratively, updating LLRs as Alice reveals values.
        
        CRITICAL: Decoder state persists across yields!
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
        
        # 2. Construct Bob's frame (with his noisy payload + erasures at puncture positions)
        frame = np.zeros(ctx.frame_size, dtype=np.uint8)
        non_puncture_mask = np.ones(ctx.frame_size, dtype=bool)
        non_puncture_mask[puncture_indices] = False
        frame[non_puncture_mask] = payload
        
        # 3. Initialize LLRs with all punctured positions as erasures
        state = BlindDecoderState(
            llr=build_three_state_llr(
                received_bits=frame,
                qber=qber_prior,
                puncture_mask=np.isin(np.arange(ctx.frame_size), puncture_indices),
                shorten_mask=None,
                shorten_values=None,
            ),
            puncture_indices=puncture_indices,
            shortened_indices=np.array([], dtype=np.int64),
            shortened_values=np.array([], dtype=np.uint8),
            iteration=1,
            syndrome=syndrome,
        )
        
        # 4. Initial decode attempt
        decoder = BeliefPropagationDecoder(max_iterations=ctx.max_iterations)
        result = decoder.decode(state.llr, syndrome, self._mother_code.compiled_H)
        
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
                revealed_leakage=0,
                hash_leakage=ctx.hash_bits,
                retry_count=1,
            )
        
        # 5. Iterative reveal loop
        total_revealed = 0
        
        while True:
            # Send NACK and wait for reveal
            msg = yield {"converged": result.converged, "verified": verified}
            
            if msg.get("kind") != "blind_reveal":
                break  # Protocol ended (max iterations)
            
            # Update state with revealed values
            new_indices = np.array(msg["revealed_indices"], dtype=np.int64)
            new_values = np.array(msg["revealed_values"], dtype=np.uint8)
            
            state.shortened_indices = np.concatenate([state.shortened_indices, new_indices])
            state.shortened_values = np.concatenate([state.shortened_values, new_values])
            state.iteration = msg["iteration"]
            total_revealed += len(new_indices)
            
            # Update LLRs for revealed positions
            # CRITICAL: Set revealed positions to ±∞ (perfect knowledge)
            for idx, val in zip(new_indices, new_values):
                state.llr[idx] = (+100.0 if val == 0 else -100.0)
            
            # Re-decode with updated LLRs (same syndrome!)
            result = decoder.decode(state.llr, syndrome, self._mother_code.compiled_H)
            
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
        )
```

### 6. MotherCodeManager Specification

```python
# caligo/reconciliation/mother_code_manager.py

class MotherCodeManager:
    """
    Singleton manager for R=0.5 mother code with untainted patterns.
    
    This class enforces the architectural constraint that reconciliation
    uses exactly ONE parity-check matrix with dynamically selected
    puncturing patterns.
    """
    
    _instance: Optional["MotherCodeManager"] = None
    
    def __init__(
        self,
        matrix_path: Path,
        pattern_dir: Path,
    ) -> None:
        """
        Load mother matrix and pattern library.
        
        Parameters
        ----------
        matrix_path : Path
            Path to R=0.5 mother matrix (.npz format).
        pattern_dir : Path
            Directory containing untainted pattern files.
        """
        self._H = sp.load_npz(matrix_path).tocsr().astype(np.uint8)
        self._compiled_H = compile_parity_check_matrix(self._H)
        self._patterns: Dict[float, np.ndarray] = self._load_patterns(pattern_dir)
        self._modulation_indices: Optional[np.ndarray] = None
        
        # Verify R=0.5
        n, m = self._H.shape[1], self._H.shape[0]
        rate = 1.0 - m / n
        if abs(rate - 0.5) > 0.01:
            raise ValueError(f"Mother code rate {rate:.3f} != 0.5")
    
    @classmethod
    def get_instance(cls, **kwargs) -> "MotherCodeManager":
        """Singleton accessor."""
        if cls._instance is None:
            cls._instance = cls(**kwargs)
        return cls._instance
    
    @property
    def frame_size(self) -> int:
        return self._H.shape[1]
    
    @property
    def compiled_H(self) -> CompiledParityCheckMatrix:
        return self._compiled_H
    
    def get_pattern(self, effective_rate: float) -> np.ndarray:
        """
        Get untainted puncturing pattern for target effective rate.
        
        Returns
        -------
        np.ndarray
            Binary mask where 1 indicates punctured position.
        """
        # Find closest available rate
        available = sorted(self._patterns.keys())
        closest = min(available, key=lambda r: abs(r - effective_rate))
        return self._patterns[closest].copy()
    
    def get_modulation_indices(self, d: int) -> np.ndarray:
        """
        Get d untainted modulation indices for Blind protocol.
        
        The indices are pre-computed to satisfy the untainted property
        and are deterministic for Alice-Bob synchronization.
        """
        if self._modulation_indices is None:
            self._modulation_indices = self._compute_untainted_indices()
        return self._modulation_indices[:d].copy()
    
    def compute_syndrome(self, frame: np.ndarray) -> np.ndarray:
        """Compute syndrome s = H·x mod 2."""
        return self._compiled_H.compute_syndrome(frame)
    
    def _load_patterns(self, pattern_dir: Path) -> Dict[float, np.ndarray]:
        """Load all .npy pattern files from directory."""
        patterns = {}
        for path in pattern_dir.glob("*.npy"):
            # Parse rate from filename: pattern_rate0.65.npy
            rate_str = path.stem.split("rate")[-1]
            rate = float(rate_str)
            patterns[rate] = np.load(path).astype(np.uint8)
        return patterns
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

### 8. Three-State LLR Builder

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
  │ ──────── Phase II: QBER Estimation ──────── │
  │                                              │
  │  [Test subset exchange, compute QBER]        │
  │                                              │
  │ ─────────── Phase III: Baseline ─────────── │
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
  │  SYNDROME(block_id, syndrome, pattern_id,   │
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
  │ ─────────── Phase III: Blind ────────────── │
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
  │  BLIND_INIT(block_id, syndrome,             │
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
  │  REVEAL(block_id, iter=2,                   │
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

## Appendix A: Configuration Schema

```yaml
# configs/reconciliation/baseline.yaml
reconciliation:
  strategy: "baseline"
  frame_size: 4096
  mother_rate: 0.5
  mother_matrix_path: "configs/ldpc_matrices/mother_code_4096_R0.5.npz"
  pattern_directory: "configs/ldpc_matrices/untainted_patterns/"
  
  decoder:
    max_iterations: 60
    convergence_threshold: 1e-6
    
  verification:
    hash_bits: 64
    hash_algorithm: "polynomial"
    
  leakage:
    safety_cap_bits: 1000000
    abort_on_exceed: true


# configs/reconciliation/blind.yaml
reconciliation:
  strategy: "blind"
  frame_size: 4096
  mother_rate: 0.5
  mother_matrix_path: "configs/ldpc_matrices/mother_code_4096_R0.5.npz"
  pattern_directory: "configs/ldpc_matrices/untainted_patterns/"
  
  blind:
    max_iterations: 3
    modulation_fraction: 0.1  # δ = 10%
    
  decoder:
    max_iterations: 60
    convergence_threshold: 1e-6
    
  verification:
    hash_bits: 64
    hash_algorithm: "polynomial"
    
  leakage:
    safety_cap_bits: 1000000
    abort_on_exceed: true
```

---

## Appendix B: Migration Checklist

- [ ] **Data Layer**
  - [ ] Generate untainted patterns using Algorithm from §2.2.2 of Theoretical Report
  - [ ] Create R=0.5 mother matrix (ACE-PEG)
  - [ ] Deprecate multi-rate matrix loading

- [ ] **Strategy Layer**
  - [ ] Implement `ReconciliationStrategy` ABC
  - [ ] Implement `BaselineStrategy`
  - [ ] Implement `BlindStrategy` with `BlindDecoderState`
  - [ ] Create strategy factory

- [ ] **Session Layer**
  - [ ] Implement `ReconciliationSession`
  - [ ] Wire leakage tracker circuit breaker
  - [ ] Remove `ReconciliationOrchestrator` God Object

- [ ] **Protocol Layer**
  - [ ] Refactor `alice.py._phase3_reconcile` to use session
  - [ ] Refactor `bob.py` Phase III handler to use session
  - [ ] Add Blind message types to envelope

- [ ] **Testing**
  - [ ] Unit tests for each strategy
  - [ ] Integration tests for both protocols
  - [ ] Convergence regression tests
  - [ ] Leakage accounting verification

---

## Appendix C: Untainted Puncturing Pattern Generation Script

The following pseudocode can be used to generate untainted puncturing patterns offline:

```python
# scripts/generate_untainted_patterns.py

import numpy as np
from scipy.sparse import csr_matrix
from pathlib import Path


def compute_depth2_neighborhood(H: csr_matrix, symbol_idx: int) -> set:
    """
    Compute N²(v) - all symbols within 2 hops of symbol v.
    
    N²(v) = {v} ∪ {all symbols sharing a check with v}
    """
    # Get checks connected to symbol
    check_indices = H.getcol(symbol_idx).nonzero()[0]
    
    # Get all symbols connected to those checks
    neighbors = set()
    for check_idx in check_indices:
        symbol_indices = H.getrow(check_idx).nonzero()[1]
        neighbors.update(symbol_indices)
    
    return neighbors


def generate_untainted_pattern(
    H: csr_matrix,
    target_puncture_fraction: float,
) -> np.ndarray:
    """
    Generate untainted puncturing pattern using greedy algorithm.
    
    Parameters
    ----------
    H : csr_matrix
        Mother code parity-check matrix (m x n).
    target_puncture_fraction : float
        Target π = p/n (may not be fully achievable).
        
    Returns
    -------
    np.ndarray
        Binary pattern where 1 = punctured position.
    """
    n = H.shape[1]  # Number of symbol nodes
    
    # Initialize: all symbols are untainted
    untainted = set(range(n))
    punctured = []
    
    # Precompute neighborhood sizes for efficiency
    n2_sizes = {}
    for v in range(n):
        n2_sizes[v] = len(compute_depth2_neighborhood(H, v))
    
    max_punctured = int(target_puncture_fraction * n)
    
    while untainted and len(punctured) < max_punctured:
        # Step 1: Find candidates with smallest N² size
        min_size = min(n2_sizes[v] for v in untainted)
        candidates = [v for v in untainted if n2_sizes[v] == min_size]
        
        # Step 2: Select one (random tie-breaking)
        selected = np.random.choice(candidates)
        punctured.append(selected)
        
        # Step 3: Remove selected and its N² from untainted set
        n2_selected = compute_depth2_neighborhood(H, selected)
        untainted -= n2_selected
    
    # Build pattern array
    pattern = np.zeros(n, dtype=np.uint8)
    pattern[punctured] = 1
    
    return pattern


def generate_pattern_library(
    H: csr_matrix,
    output_dir: Path,
    rates: list = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80],
    mother_rate: float = 0.5,
) -> None:
    """
    Generate untainted patterns for multiple effective rates.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    n = H.shape[1]
    
    for target_rate in rates:
        # Compute required puncture fraction
        # R_eff = R_0 / (1 - π)  =>  π = 1 - R_0 / R_eff
        puncture_fraction = 1.0 - mother_rate / target_rate
        
        if puncture_fraction <= 0:
            print(f"Skipping rate {target_rate}: no puncturing needed")
            continue
        
        pattern = generate_untainted_pattern(H, puncture_fraction)
        actual_punctured = pattern.sum()
        actual_rate = mother_rate / (1 - actual_punctured / n)
        
        filename = f"pattern_rate{actual_rate:.2f}.npy"
        np.save(output_dir / filename, pattern)
        
        print(f"Generated {filename}: {actual_punctured} punctured bits, "
              f"effective rate {actual_rate:.3f}")


if __name__ == "__main__":
    import scipy.sparse as sp
    
    # Load mother code
    H = sp.load_npz("configs/ldpc_matrices/mother_code_4096_R0.5.npz").tocsr()
    
    # Generate pattern library
    generate_pattern_library(
        H,
        Path("configs/ldpc_matrices/untainted_patterns/"),
        rates=[0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90],
    )
```

**Note:** The maximum achievable rate depends on the code structure. For an n=4096, R=0.5 irregular LDPC code, typical untainted puncturing achieves up to ~20% of symbols punctured (corresponding to $R_{\text{eff}} \approx 0.625$). Higher rates require modified algorithms (e.g., Ha et al. 2006) that trade off some untainted guarantees.

---

*End of Implementation Report v2*
