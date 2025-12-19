# Parallel EPR Generation for Simulation Acceleration

**Technical Assessment and Implementation Proposal**

---

## Executive Summary

This document assesses the feasibility of parallelizing EPR pair generation and measurement in SquidASM simulations to accelerate the Caligo QKD protocol. We analyze whether this "simulation cheat" preserves physical validity and propose a detailed implementation strategy.

**Key Finding**: Parallel EPR generation is a **physically valid simulation optimization** that does not compromise experimental correctness, provided proper statistical aggregation is maintained. The approach treats multiple independent simulation runs as a Monte Carlo ensemble, which is mathematically equivalent to sequential generation under the i.i.d. (independent and identically distributed) assumption that underlies all QKD security proofs.

---

## Table of Contents

1. [Physical Validity Assessment](#1-physical-validity-assessment)
2. [SquidASM Architecture Analysis](#2-squidasm-architecture-analysis)
3. [Implementation Proposal](#3-implementation-proposal)
4. [Security Considerations](#4-security-considerations)
5. [Performance Projections](#5-performance-projections)
6. [Code Examples](#6-code-examples)
7. [Appendix: Mathematical Justification](#appendix-mathematical-justification)

---

## 1. Physical Validity Assessment

### 1.1 The Core Question

**Question**: Does running multiple EPR generation+measurement instances in parallel violate physical principles, thereby invalidating experimental findings?

**Answer**: **No**, parallel execution is physically valid for the following reasons:

### 1.2 Independence of EPR Pairs

In both theoretical QKD analysis and physical implementations, EPR pairs are treated as **independent and identically distributed (i.i.d.)** random variables. This is a fundamental assumption in:

- **BB84 Security Proofs** (Bennett & Brassard, 1984; Shor & Preskill, 2000)
- **E91 Protocol Analysis** (Ekert, 1991)
- **NSM Security Model** (König et al., 2012; Wehner et al., 2010)
- **Finite-Key Analysis** (Tomamichel et al., 2012; Lim et al., 2014)

The security of QKD protocols depends on:
1. The statistical properties of each EPR pair (fidelity, QBER contribution)
2. The aggregate statistics over many pairs (overall QBER, key rate)

**Neither depends on the temporal ordering of generation.**

### 1.3 Mathematical Equivalence

Let $\{(\text{outcome}_i^A, \text{outcome}_i^B, \text{basis}_i^A, \text{basis}_i^B)\}_{i=1}^{N}$ be the sequence of EPR measurement outcomes.

**Sequential Generation**:
$$P(\text{data}|\text{sequential}) = \prod_{i=1}^{N} P(\text{pair}_i | \text{noise model}, \text{channel})$$

**Parallel Generation** (assuming i.i.d.):
$$P(\text{data}|\text{parallel}) = \prod_{i=1}^{N} P(\text{pair}_i | \text{noise model}, \text{channel})$$

These are **identical** because:
- Each pair is generated from the same noise model configuration
- There are no memory effects between pairs (Markovian assumption)
- Measurement outcomes are determined at generation time

### 1.4 What Parallel Execution Does NOT Affect

| Physical Property | Affected by Parallelization? | Reason |
|-------------------|------------------------------|--------|
| **Fidelity** | ❌ No | Determined by noise model parameters |
| **QBER** | ❌ No | Statistical aggregate of per-pair errors |
| **Sifting Efficiency** | ❌ No | Depends only on basis choice distribution |
| **Key Rate** | ❌ No | Function of QBER and sifted key length |
| **Decoherence** | ❌ No | Modeled per-pair, not across pairs |
| **Dark Counts** | ❌ No | Probabilistic per detection event |

### 1.5 What Parallel Execution DOES Affect (Acceptably)

| Simulation Property | How It Changes | Is This OK? |
|---------------------|----------------|-------------|
| **Wall-clock time** | Reduced by ~1/workers | ✅ Yes |
| **Memory usage** | Increased (parallel state) | ✅ Manageable |
| **Random seed state** | Different sequence | ✅ Must use independent seeds |
| **Exact ordering** | Lost (aggregated) | ✅ Ordering is not physical |

### 1.6 Physical Scenarios Where Parallelization Would Be INVALID

Parallel generation would **violate physics** only if:

1. **Correlated pairs**: If the generation of pair $i+1$ depends on pair $i$ (non-Markovian memory)
2. **Shared resources**: If qubits share quantum memory causing cross-talk
3. **Timing-sensitive protocols**: If the security proof requires specific temporal ordering
4. **Adversarial timing attacks**: If Eve can exploit generation order

**None of these apply to our scenario** because:
- SquidASM models each pair independently
- The NSM security model assumes Markovian storage (no correlations)
- Caligo's timing barrier ($\Delta t$) operates on classical messages, not EPR ordering
- Eve's attack model is already i.i.d.-based

### 1.7 Conclusion: Validity Assessment

**Verdict**: Parallel EPR generation is a **valid simulation acceleration technique** that:
- Preserves all physically relevant statistics
- Is mathematically equivalent to sequential generation under i.i.d. assumption
- Does not affect security analysis or experimental validity
- Is analogous to Monte Carlo ensemble methods in computational physics

---

## 2. SquidASM Architecture Analysis

### 2.1 Current Execution Model

SquidASM uses a **discrete-event simulation** powered by NetSquid. The execution model is:

```
                    ┌─────────────────────────────────────────┐
                    │           NetSquid Scheduler            │
                    │      (Single-threaded event loop)       │
                    └──────────────────┬──────────────────────┘
                                       │
        ┌──────────────────────────────┼──────────────────────────────┐
        ▼                              ▼                              ▼
┌───────────────┐              ┌───────────────┐              ┌───────────────┐
│   Alice Node  │              │  Quantum Link │              │   Bob Node    │
│   (QNodeOS)   │◄────────────►│ (MagicDistrib)│◄────────────►│   (QNodeOS)   │
└───────────────┘              └───────────────┘              └───────────────┘
```

**Key Constraint**: NetSquid's discrete-event simulator is **inherently single-threaded**. Each simulation run processes events sequentially.

### 2.2 EPR Generation Flow

From `squidasm/nqasm/netstack.py` and `netqasm/sdk/epr_socket.py`:

```python
# Alice initiates
qubit = epr_socket.create_keep()[0]  # Registers EPR request

# NetSquid processes:
# 1. Alice's request → Link Layer Service
# 2. MagicDistributor creates entangled state
# 3. State distributed to Alice & Bob quantum memories
# 4. Bob's recv_keep() unblocks

yield from context.connection.flush()  # Synchronization point
```

**Timing Model** (`squidasm/sim/network/network.py`):
```python
# Link configuration determines generation time
model_params = DepolariseModelParameters(
    cycle_time=state_delay,      # Time per EPR attempt (ns)
    prob_success=success_prob,   # Success probability
    prob_max_mixed=noise         # Noise level
)
```

### 2.3 Bottleneck Analysis

Current performance bottlenecks in Caligo's quantum phase:

| Operation | Sequential Time | Bottleneck Type |
|-----------|-----------------|-----------------|
| EPR generation | O(N × t_cycle) | Simulation time |
| Qubit measurement | O(N × t_measure) | Simulation time |
| Basis selection | O(N) | CPU (negligible) |
| Classical messages | O(round_trips) | Event scheduling |
| NetSquid events | O(N × events_per_pair) | Single-threaded |

For N = 100,000 pairs with t_cycle = 10ns:
- Simulated time: ~1ms
- **Wall-clock time**: Minutes to hours (event processing overhead)

### 2.4 Multiprocessing Opportunities

SquidASM already supports some parallelism via `ThreadPool`:

```python
# From squidasm/run/multithread/runtime_mgr.py
with ThreadPool(len(programs) + 1) as executor:
    program_futures = []
    for program in programs:
        future = executor.apply_async(program.entry, kwds=inputs)
        program_futures.append(future)
```

However, this is for **running different programs in parallel**, not for parallelizing EPR generation within a single protocol run.

---

## 3. Implementation Proposal

### 3.1 Architectural Overview

We propose a **parallel worker pool** architecture that:
1. Divides the total EPR pairs into independent batches
2. Runs each batch in a separate process with isolated NetSquid instance
3. Aggregates results while preserving statistical properties

```
                        ┌────────────────────────────┐
                        │   ParallelEPROrchestrator  │
                        │    (Main Coordinator)      │
                        └────────────┬───────────────┘
                                     │
           ┌─────────────────────────┼─────────────────────────┐
           │                         │                         │
           ▼                         ▼                         ▼
   ┌───────────────┐         ┌───────────────┐         ┌───────────────┐
   │ EPRWorker #1  │         │ EPRWorker #2  │         │ EPRWorker #N  │
   │ (Process 1)   │         │ (Process 2)   │         │ (Process N)   │
   │               │         │               │         │               │
   │ ┌───────────┐ │         │ ┌───────────┐ │         │ ┌───────────┐ │
   │ │ NetSquid  │ │         │ │ NetSquid  │ │         │ │ NetSquid  │ │
   │ │ Instance  │ │         │ │ Instance  │ │         │ │ Instance  │ │
   │ └───────────┘ │         │ └───────────┘ │         │ └───────────┘ │
   │               │         │               │         │               │
   │ Batch: 0-999  │         │ Batch:1K-1.9K │         │ Batch: ...    │
   └───────┬───────┘         └───────┬───────┘         └───────┬───────┘
           │                         │                         │
           └─────────────────────────┼─────────────────────────┘
                                     │
                        ┌────────────▼───────────────┐
                        │    ResultAggregator        │
                        │  (Combines batch results)  │
                        └────────────────────────────┘
```

### 3.2 Design Principles

#### 3.2.1 Process Isolation

Each worker runs in a **separate process** (not thread) because:
- NetSquid is not thread-safe (global state in `ns.sim_reset()`)
- Python's GIL limits threading benefits for CPU-bound work
- Process isolation prevents state contamination

#### 3.2.2 Statistical Correctness

To maintain equivalence with sequential generation:

1. **Independent RNG Seeds**: Each worker uses a unique, deterministic seed
   ```python
   worker_seed = base_seed + worker_id * SEED_OFFSET
   ```

2. **Identical Noise Models**: All workers use the same noise configuration
   ```python
   # Shared across workers (serialized)
   noise_config = {
       'fidelity': 0.95,
       't_cycle': 10.0,
       'depolar_prob': 0.001
   }
   ```

3. **Preserving Pair Ordering**: Results are tagged with global indices
   ```python
   global_idx = batch_start + local_idx
   ```

#### 3.2.3 Memory Efficiency

Use shared memory for large arrays to avoid serialization overhead:

```python
from multiprocessing import shared_memory

# Create shared array for results
shm = shared_memory.SharedMemory(
    create=True,
    size=N * 2  # outcomes + bases (uint8)
)
```

### 3.3 Component Design

#### 3.3.1 ParallelEPRConfig

```python
@dataclass(frozen=True)
class ParallelEPRConfig:
    """Configuration for parallel EPR generation.
    
    Parameters
    ----------
    total_pairs : int
        Total number of EPR pairs to generate.
    num_workers : int
        Number of parallel worker processes.
    pairs_per_batch : int
        Pairs per worker batch (affects memory usage).
    base_seed : int
        Base random seed for reproducibility.
    noise_params : Dict[str, Any]
        Noise model parameters (shared across workers).
    aggregation_mode : str
        How to combine results: 'concatenate' or 'streaming'.
    """
    total_pairs: int
    num_workers: int = 4
    pairs_per_batch: int = 10000
    base_seed: int = 42
    noise_params: Dict[str, Any] = field(default_factory=dict)
    aggregation_mode: str = 'concatenate'
    
    def __post_init__(self):
        if self.total_pairs <= 0:
            raise ValueError("total_pairs must be positive")
        if self.num_workers <= 0:
            raise ValueError("num_workers must be positive")
```

#### 3.3.2 EPRWorkerTask

```python
@dataclass
class EPRWorkerTask:
    """Task specification for a single worker.
    
    Parameters
    ----------
    worker_id : int
        Unique worker identifier.
    batch_start_idx : int
        Global index of first pair in this batch.
    batch_size : int
        Number of pairs to generate.
    seed : int
        RNG seed for this worker.
    noise_config : StackNetworkConfig
        Network configuration (noise, fidelity, etc.).
    """
    worker_id: int
    batch_start_idx: int
    batch_size: int
    seed: int
    noise_config: Dict[str, Any]
```

#### 3.3.3 EPRWorkerResult

```python
@dataclass
class EPRWorkerResult:
    """Result from a single worker.
    
    Parameters
    ----------
    worker_id : int
        Worker that produced this result.
    batch_start_idx : int
        Global index of first pair.
    outcomes : np.ndarray
        Measurement outcomes (uint8).
    bases : np.ndarray
        Measurement bases (uint8).
    generation_times : np.ndarray
        Per-pair generation times (float64).
    success : bool
        Whether the worker completed successfully.
    error_message : Optional[str]
        Error details if success=False.
    statistics : Dict[str, float]
        Worker-level statistics (mean QBER, etc.).
    """
    worker_id: int
    batch_start_idx: int
    outcomes: np.ndarray
    bases: np.ndarray
    generation_times: np.ndarray
    success: bool
    error_message: Optional[str] = None
    statistics: Dict[str, float] = field(default_factory=dict)
```

### 3.4 Worker Implementation

#### 3.4.1 Minimal EPR Generation Program

For parallel execution, we use a **minimal program** that only performs EPR generation and measurement, without the full protocol overhead:

```python
class MinimalAliceProgram(Program):
    """Minimal Alice program for parallel EPR generation."""
    
    PEER = "Bob"
    
    def __init__(self, num_pairs: int, batch_id: int):
        self._num_pairs = num_pairs
        self._batch_id = batch_id
    
    @property
    def meta(self) -> ProgramMeta:
        return ProgramMeta(
            name=f"parallel_alice_{self._batch_id}",
            csockets=[self.PEER],
            epr_sockets=[self.PEER],
            max_qubits=10  # Memory constraint
        )
    
    def run(self, context: ProgramContext) -> Generator[Any, None, Dict]:
        epr_socket = context.epr_sockets[self.PEER]
        
        outcomes = np.zeros(self._num_pairs, dtype=np.uint8)
        bases = np.zeros(self._num_pairs, dtype=np.uint8)
        
        for i in range(self._num_pairs):
            # Random basis selection
            basis = np.random.randint(0, 2)
            bases[i] = basis
            
            # Generate EPR pair
            q = epr_socket.create_keep(1)[0]
            
            # Apply basis rotation if needed
            if basis == 1:  # X basis
                q.H()
            
            # Measure
            outcome = q.measure()
            yield from context.connection.flush()
            
            outcomes[i] = int(outcome)
        
        return {
            "outcomes": outcomes.tobytes(),
            "bases": bases.tobytes(),
            "batch_id": self._batch_id
        }
```

#### 3.4.2 Worker Process Function

```python
def epr_worker_process(task: EPRWorkerTask, result_queue: Queue) -> None:
    """Worker process that runs isolated EPR generation.
    
    This function runs in a separate process and creates its own
    NetSquid simulation instance.
    
    Parameters
    ----------
    task : EPRWorkerTask
        Work specification.
    result_queue : Queue
        Queue to put results into.
    """
    try:
        # Set random seed for this worker
        np.random.seed(task.seed)
        
        # Import NetSquid fresh (process-local)
        import netsquid as ns
        ns.sim_reset()
        ns.set_random_state(rng=np.random.default_rng(task.seed))
        
        # Build network configuration
        from squidasm.run.stack.config import StackNetworkConfig
        config = build_config_from_dict(task.noise_config)
        
        # Create minimal programs
        alice = MinimalAliceProgram(task.batch_size, task.worker_id)
        bob = MinimalBobProgram(task.batch_size, task.worker_id)
        
        # Run simulation
        from squidasm.run.stack.run import run
        results = run(
            config=config,
            programs={"Alice": alice, "Bob": bob},
            num_times=1
        )
        
        # Extract results
        alice_result = results[0][0]  # First stack, first iteration
        
        result = EPRWorkerResult(
            worker_id=task.worker_id,
            batch_start_idx=task.batch_start_idx,
            outcomes=np.frombuffer(alice_result["outcomes"], dtype=np.uint8),
            bases=np.frombuffer(alice_result["bases"], dtype=np.uint8),
            generation_times=np.zeros(task.batch_size),  # Simplified
            success=True,
            statistics=compute_batch_statistics(alice_result)
        )
        
    except Exception as e:
        result = EPRWorkerResult(
            worker_id=task.worker_id,
            batch_start_idx=task.batch_start_idx,
            outcomes=np.array([], dtype=np.uint8),
            bases=np.array([], dtype=np.uint8),
            generation_times=np.array([]),
            success=False,
            error_message=str(e)
        )
    
    result_queue.put(result)
```

### 3.5 Orchestrator Implementation

```python
class ParallelEPROrchestrator:
    """Coordinates parallel EPR generation across multiple processes.
    
    This class manages the distribution of work, process spawning,
    result collection, and aggregation.
    
    Parameters
    ----------
    config : ParallelEPRConfig
        Parallel generation configuration.
    
    Examples
    --------
    >>> config = ParallelEPRConfig(
    ...     total_pairs=100000,
    ...     num_workers=8,
    ...     noise_params={'fidelity': 0.95}
    ... )
    >>> orchestrator = ParallelEPROrchestrator(config)
    >>> result = orchestrator.run()
    >>> print(f"Generated {len(result.outcomes)} pairs")
    """
    
    def __init__(self, config: ParallelEPRConfig):
        self._config = config
        self._logger = get_logger(__name__)
    
    def run(self) -> AggregatedEPRResult:
        """Execute parallel EPR generation.
        
        Returns
        -------
        AggregatedEPRResult
            Combined results from all workers.
        """
        # Create task distribution
        tasks = self._create_tasks()
        
        # Execute in parallel
        with ProcessPoolExecutor(max_workers=self._config.num_workers) as pool:
            result_queue = Queue()
            
            # Submit all tasks
            futures = []
            for task in tasks:
                future = pool.submit(epr_worker_process, task, result_queue)
                futures.append(future)
            
            # Collect results
            results = []
            for _ in range(len(tasks)):
                result = result_queue.get(timeout=3600)  # 1 hour timeout
                results.append(result)
                self._logger.info(
                    f"Worker {result.worker_id} completed: "
                    f"{len(result.outcomes)} pairs"
                )
        
        # Aggregate results
        return self._aggregate_results(results)
    
    def _create_tasks(self) -> List[EPRWorkerTask]:
        """Divide work into tasks for workers."""
        tasks = []
        pairs_per_worker = self._config.total_pairs // self._config.num_workers
        remainder = self._config.total_pairs % self._config.num_workers
        
        current_idx = 0
        for worker_id in range(self._config.num_workers):
            batch_size = pairs_per_worker + (1 if worker_id < remainder else 0)
            
            task = EPRWorkerTask(
                worker_id=worker_id,
                batch_start_idx=current_idx,
                batch_size=batch_size,
                seed=self._config.base_seed + worker_id * 1000000,
                noise_config=self._config.noise_params
            )
            tasks.append(task)
            current_idx += batch_size
        
        return tasks
    
    def _aggregate_results(
        self, results: List[EPRWorkerResult]
    ) -> AggregatedEPRResult:
        """Combine results from all workers.
        
        Results are sorted by batch_start_idx to maintain
        deterministic ordering (though ordering doesn't affect
        physical validity).
        """
        # Sort by batch start index
        sorted_results = sorted(results, key=lambda r: r.batch_start_idx)
        
        # Check for failures
        failed = [r for r in sorted_results if not r.success]
        if failed:
            raise RuntimeError(
                f"{len(failed)} workers failed: "
                f"{[r.error_message for r in failed]}"
            )
        
        # Concatenate arrays
        all_outcomes = np.concatenate([r.outcomes for r in sorted_results])
        all_bases = np.concatenate([r.bases for r in sorted_results])
        
        # Compute aggregate statistics
        aggregate_stats = self._compute_aggregate_statistics(sorted_results)
        
        return AggregatedEPRResult(
            outcomes=all_outcomes,
            bases=all_bases,
            total_pairs=len(all_outcomes),
            num_workers=len(sorted_results),
            statistics=aggregate_stats
        )
    
    def _compute_aggregate_statistics(
        self, results: List[EPRWorkerResult]
    ) -> Dict[str, float]:
        """Compute aggregate statistics across all workers."""
        # Per-worker stats
        worker_stats = [r.statistics for r in results]
        
        return {
            'total_pairs': sum(len(r.outcomes) for r in results),
            'mean_qber_per_worker': np.mean([
                s.get('estimated_qber', 0.0) for s in worker_stats
            ]),
            'std_qber_per_worker': np.std([
                s.get('estimated_qber', 0.0) for s in worker_stats
            ]),
        }
```

### 3.6 Integration with Caligo Protocol

#### 3.6.1 Modified Protocol Flow

```python
class CaligoParallelProgram(CaligoProgram):
    """Caligo protocol with parallel quantum phase."""
    
    def __init__(
        self,
        params: ProtocolParameters,
        parallel_config: Optional[ParallelEPRConfig] = None
    ):
        super().__init__(params)
        self._parallel_config = parallel_config or ParallelEPRConfig(
            total_pairs=params.num_pairs,
            num_workers=4
        )
    
    def _run_protocol(self, context) -> Generator[Any, None, Dict]:
        # Phase I: Parallel Quantum Generation (ACCELERATED)
        alice_data, bob_data = self._phase1_parallel_quantum()
        
        # Phase II: Sifting (unchanged)
        sifting_result = yield from self._phase2_sifting(
            alice_outcomes=alice_data.outcomes,
            alice_bases=alice_data.bases,
        )
        
        # Phase III: Reconciliation (unchanged)
        # Phase IV: Amplification (unchanged)
        # ...
    
    def _phase1_parallel_quantum(self) -> Tuple[EPRData, EPRData]:
        """Execute quantum phase with parallel workers.
        
        NOTE: This runs OUTSIDE the SquidASM event loop, before
        the main protocol starts. Results are pre-generated.
        """
        orchestrator = ParallelEPROrchestrator(self._parallel_config)
        
        # This call blocks but runs in parallel processes
        result = orchestrator.run()
        
        # Package for protocol use
        alice_data = EPRData(
            outcomes=result.alice_outcomes,
            bases=result.alice_bases
        )
        bob_data = EPRData(
            outcomes=result.bob_outcomes,
            bases=result.bob_bases
        )
        
        return alice_data, bob_data
```

#### 3.6.2 Alternative: Pre-Generation Mode

For even better performance, generate EPR results **before** the protocol run:

```python
def pregenerate_quantum_data(
    config: ParallelEPRConfig
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pre-generate all quantum measurement data.
    
    Returns
    -------
    Tuple containing:
        - alice_outcomes : np.ndarray
        - alice_bases : np.ndarray
        - bob_outcomes : np.ndarray  
        - bob_bases : np.ndarray
    """
    orchestrator = ParallelEPROrchestrator(config)
    result = orchestrator.run()
    
    return (
        result.alice_outcomes,
        result.alice_bases,
        result.bob_outcomes,
        result.bob_bases
    )


# Usage in test/benchmark
def run_caligo_with_pregenerated_data():
    # Step 1: Generate quantum data (parallel, slow but faster)
    config = ParallelEPRConfig(total_pairs=1_000_000, num_workers=16)
    alice_out, alice_bas, bob_out, bob_bas = pregenerate_quantum_data(config)
    
    # Step 2: Run classical protocol (uses pre-generated data)
    # This is now MUCH faster since no simulation needed
    result = run_classical_protocol(
        alice_outcomes=alice_out,
        alice_bases=alice_bas,
        bob_outcomes=bob_out,
        bob_bases=bob_bas
    )
```

---

## 4. Security Considerations

### 4.1 Preserved Security Properties

| Security Property | Status | Justification |
|-------------------|--------|---------------|
| **QBER correctness** | ✅ Preserved | Same noise model, same statistics |
| **Key rate bounds** | ✅ Preserved | Depends only on QBER and length |
| **NSM assumption** | ✅ Preserved | No correlation between pairs |
| **Finite-key effects** | ✅ Preserved | Statistical sampling unchanged |
| **Commitment binding** | ✅ Preserved | Classical layer unchanged |
| **Δt timing barrier** | ⚠️ Modified | See below |

### 4.2 Timing Barrier Considerations

The NSM timing barrier ($\Delta t$) ensures that:
1. Bob commits to his measurement record before Alice reveals bases
2. Adversarial storage decoheres during the wait

**In parallel mode**: The timing barrier operates at the **aggregated result** level, not per-pair. This is acceptable because:
- The barrier's purpose is to ensure commitment ordering in classical messages
- Parallel generation happens **before** any classical exchange
- Bob still commits before Alice reveals, just with pre-generated data

```
Sequential:           Parallel:
─────────────         ─────────────
EPR-1 gen             [Worker 1: EPR batch]
EPR-1 meas            [Worker 2: EPR batch]  } Pre-generation
EPR-2 gen             [Worker 3: EPR batch]
EPR-2 meas            [Worker 4: EPR batch]
  ...                 ─────────────
EPR-N meas            Aggregate results
─────────────         ─────────────
Bob commits           Bob commits          } Same timing
Wait Δt               Wait Δt              } relationship
Alice reveals         Alice reveals
```

### 4.3 Random Number Generation

**Critical**: Workers must use **independent, non-overlapping** random streams.

```python
def create_worker_rng(base_seed: int, worker_id: int) -> np.random.Generator:
    """Create independent RNG for worker.
    
    Uses SeedSequence to generate independent streams.
    """
    ss = np.random.SeedSequence(base_seed)
    worker_seeds = ss.spawn(num_workers)
    return np.random.default_rng(worker_seeds[worker_id])
```

### 4.4 Deterministic Reproducibility

For debugging and verification:

```python
class DeterministicParallelConfig(ParallelEPRConfig):
    """Configuration that guarantees reproducible results."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Fix all sources of randomness
        self._seed_sequence = np.random.SeedSequence(self.base_seed)
    
    def get_worker_seed(self, worker_id: int) -> int:
        """Get deterministic seed for worker."""
        return self._seed_sequence.spawn(self.num_workers)[worker_id]
```

---

## 5. Performance Projections

### 5.1 Expected Speedup

Theoretical speedup with $W$ workers for $N$ pairs:

$$\text{Speedup} \approx \min\left(W, \frac{T_{\text{quantum}}}{T_{\text{overhead}}}\right)$$

Where:
- $T_{\text{quantum}}$: Time for EPR generation + measurement
- $T_{\text{overhead}}$: Process spawning + result aggregation

### 5.2 Benchmark Projections

| N (pairs) | Sequential (est.) | 4 Workers | 8 Workers | 16 Workers |
|-----------|-------------------|-----------|-----------|------------|
| 10,000 | 30s | 10s | 8s | 8s |
| 100,000 | 5min | 90s | 50s | 35s |
| 1,000,000 | 50min | 15min | 8min | 5min |
| 10,000,000 | 8hr | 2.5hr | 1.3hr | 45min |

*Note: Estimates based on typical SquidASM performance. Actual results depend on hardware and noise model complexity.*

### 5.3 Memory Requirements

Per worker:
- NetSquid state: ~100 MB baseline
- Results buffer: ~2 bytes × pairs_per_worker
- Overhead: ~50 MB

Total for 16 workers with 100K pairs each:
$$16 \times (100 + 0.2 + 50) \text{ MB} \approx 2.4 \text{ GB}$$

### 5.4 Scaling Limits

| Limiting Factor | Symptom | Mitigation |
|-----------------|---------|------------|
| Memory | OOM errors | Reduce pairs_per_worker |
| CPU saturation | No further speedup | Match workers to cores |
| I/O bottleneck | Slow aggregation | Use shared memory |
| Process startup | High overhead for small N | Batch multiple runs |

---

## 6. Code Examples

### 6.1 Minimal Working Example

```python
"""Minimal example of parallel EPR generation."""

import numpy as np
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Dict, Tuple

@dataclass
class WorkerResult:
    outcomes: np.ndarray
    bases: np.ndarray
    worker_id: int

def generate_epr_batch(
    batch_size: int,
    seed: int,
    fidelity: float,
    worker_id: int
) -> WorkerResult:
    """Generate a batch of EPR pairs in isolated process."""
    # Set random state
    np.random.seed(seed)
    
    # Import SquidASM (process-local)
    import netsquid as ns
    ns.sim_reset()
    
    from squidasm.run.stack.config import StackNetworkConfig
    from squidasm.run.stack.run import run
    
    # Create minimal config
    config = create_minimal_config(fidelity)
    
    # Create programs
    alice = MinimalAlice(batch_size)
    bob = MinimalBob(batch_size)
    
    # Run
    results = run(config, {"Alice": alice, "Bob": bob}, num_times=1)
    
    return WorkerResult(
        outcomes=np.array(results[0][0]["outcomes"]),
        bases=np.array(results[0][0]["bases"]),
        worker_id=worker_id
    )

def parallel_epr_generation(
    total_pairs: int,
    num_workers: int = 4,
    fidelity: float = 0.95,
    base_seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate EPR pairs using parallel workers."""
    
    pairs_per_worker = total_pairs // num_workers
    
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        futures = []
        for i in range(num_workers):
            future = pool.submit(
                generate_epr_batch,
                batch_size=pairs_per_worker,
                seed=base_seed + i * 1000000,
                fidelity=fidelity,
                worker_id=i
            )
            futures.append(future)
        
        results = [f.result() for f in futures]
    
    # Aggregate
    all_outcomes = np.concatenate([r.outcomes for r in results])
    all_bases = np.concatenate([r.bases for r in results])
    
    return all_outcomes, all_bases

# Usage
if __name__ == "__main__":
    outcomes, bases = parallel_epr_generation(
        total_pairs=100000,
        num_workers=8,
        fidelity=0.95
    )
    print(f"Generated {len(outcomes)} pairs")
    print(f"Estimated QBER: {np.mean(outcomes != 0):.4f}")
```

### 6.2 Integration with Existing Caligo Tests

```python
"""Integration test for parallel EPR generation."""

import pytest
import numpy as np
from caligo.parallel import ParallelEPROrchestrator, ParallelEPRConfig
from caligo.sifting import Sifter
from caligo.sifting.qber import QBEREstimator

class TestParallelGeneration:
    """Test suite for parallel EPR generation."""
    
    @pytest.fixture
    def parallel_config(self):
        return ParallelEPRConfig(
            total_pairs=10000,
            num_workers=4,
            base_seed=42,
            noise_params={'fidelity': 0.95}
        )
    
    def test_generates_correct_number_of_pairs(self, parallel_config):
        """Verify total pair count."""
        orchestrator = ParallelEPROrchestrator(parallel_config)
        result = orchestrator.run()
        
        assert len(result.outcomes) == parallel_config.total_pairs
        assert len(result.bases) == parallel_config.total_pairs
    
    def test_qber_within_expected_range(self, parallel_config):
        """Verify QBER matches noise model expectations."""
        orchestrator = ParallelEPROrchestrator(parallel_config)
        result = orchestrator.run()
        
        # Expected QBER for fidelity=0.95: ~2.5%
        # Q = (1 - F) / 2 = 0.025
        expected_qber = (1 - 0.95) / 2
        
        # Estimate actual QBER from matching bases
        matching = result.alice_bases == result.bob_bases
        errors = result.alice_outcomes[matching] != result.bob_outcomes[matching]
        actual_qber = np.mean(errors)
        
        # Allow 3-sigma tolerance
        tolerance = 3 * np.sqrt(expected_qber * (1 - expected_qber) / np.sum(matching))
        
        assert abs(actual_qber - expected_qber) < tolerance
    
    def test_reproducibility_with_same_seed(self, parallel_config):
        """Verify deterministic results with same seed."""
        orchestrator1 = ParallelEPROrchestrator(parallel_config)
        orchestrator2 = ParallelEPROrchestrator(parallel_config)
        
        result1 = orchestrator1.run()
        result2 = orchestrator2.run()
        
        np.testing.assert_array_equal(result1.outcomes, result2.outcomes)
        np.testing.assert_array_equal(result1.bases, result2.bases)
    
    def test_different_seeds_give_different_results(self):
        """Verify different seeds produce different data."""
        config1 = ParallelEPRConfig(total_pairs=1000, base_seed=42)
        config2 = ParallelEPRConfig(total_pairs=1000, base_seed=123)
        
        result1 = ParallelEPROrchestrator(config1).run()
        result2 = ParallelEPROrchestrator(config2).run()
        
        # Should be statistically different
        assert not np.array_equal(result1.outcomes, result2.outcomes)
    
    def test_integrates_with_sifting(self, parallel_config):
        """Verify parallel results work with sifting phase."""
        result = ParallelEPROrchestrator(parallel_config).run()
        
        sifter = Sifter()
        alice_sift, bob_sift = sifter.compute_sifted_key(
            alice_bases=result.alice_bases,
            alice_outcomes=result.alice_outcomes,
            bob_bases=result.bob_bases,
            bob_outcomes=result.bob_outcomes
        )
        
        # Sifting should produce ~50% of pairs (basis matching)
        expected_sifted = parallel_config.total_pairs * 0.5
        actual_sifted = len(alice_sift.matching_indices)
        
        # Allow 10% tolerance
        assert abs(actual_sifted - expected_sifted) < expected_sifted * 0.1
```

---

## Appendix: Mathematical Justification

### A.1 I.I.D. Assumption in QKD

The security of QKD protocols relies on the i.i.d. assumption for EPR pairs:

**Definition (i.i.d. EPR pairs)**: A sequence of EPR pairs $\{|\psi_i\rangle\}_{i=1}^N$ is i.i.d. if:
1. Each $|\psi_i\rangle$ is drawn from the same distribution $\mathcal{D}$
2. The pairs are mutually independent: $\rho_{1...N} = \rho_1 \otimes \rho_2 \otimes ... \otimes \rho_N$

### A.2 Sequential vs. Parallel: Statistical Equivalence

**Theorem**: Under the i.i.d. assumption, sequential and parallel EPR generation produce statistically equivalent data.

**Proof sketch**:

Let $X_i = (o_i^A, o_i^B, b_i^A, b_i^B)$ be the outcome tuple for pair $i$.

For sequential generation:
$$P(X_1, X_2, ..., X_N) = \prod_{i=1}^N P(X_i | \text{noise model})$$

For parallel generation with $W$ workers, each generating $N/W$ pairs:
$$P(X_1, ..., X_N) = \prod_{w=1}^W \prod_{j=1}^{N/W} P(X_{w,j} | \text{noise model})$$

Since multiplication is commutative and associative:
$$\prod_{w=1}^W \prod_{j=1}^{N/W} P(X_{w,j}) = \prod_{i=1}^N P(X_i) \quad \blacksquare$$

### A.3 Markovian Storage Model Compatibility

The NSM assumes adversary storage follows a Markovian channel $\mathcal{N}_r$:

$$\mathcal{N}_r(\rho) = r\rho + (1-r)\frac{I}{2}$$

**Key property**: The channel acts independently on each qubit:
$$\mathcal{N}_r^{\otimes N}(\rho_1 \otimes ... \otimes \rho_N) = \mathcal{N}_r(\rho_1) \otimes ... \otimes \mathcal{N}_r(\rho_N)$$

This factorization is preserved regardless of generation order.

### A.4 Security Parameter Preservation

The security parameter $\epsilon$ bounds the distinguishing advantage between real and ideal protocols:

$$\epsilon = \epsilon_{\text{PE}} + \epsilon_{\text{EC}} + \epsilon_{\text{PA}}$$

Each component depends only on:
- $n$: Number of sifted pairs (unchanged by parallelization)
- $Q$: QBER estimate (statistically equivalent)
- $\ell$: Final key length (determined by above)

None depend on generation ordering. $\blacksquare$

---

## 7. Implementation Plan

This section provides a practical roadmap for integrating parallel EPR generation into the Caligo codebase, including architectural considerations, new module specifications, refactoring strategies, and comprehensive testing plans.

### 7.1 Current Architecture Analysis

The existing Caligo codebase follows a modular structure:

```
caligo/
├── quantum/
│   ├── epr.py              # EPR generation logic (AliceEPRProgram, BobEPRProgram)
│   ├── batching.py         # Batch processing utilities
│   ├── basis.py            # Measurement basis definitions
│   └── measurement.py      # Measurement operations
├── protocol/
│   ├── alice.py            # Alice's QKD protocol implementation
│   ├── bob.py              # Bob's QKD protocol implementation
│   ├── base.py             # Base protocol classes
│   └── orchestrator.py     # Protocol coordination
├── simulation/
│   ├── network_builder.py  # Quantum network topology setup
│   ├── physical_model.py   # Physical noise models
│   └── timing.py           # Timing configurations
├── sifting/
│   └── core.py             # Basis reconciliation
├── reconciliation/
│   └── cascade.py          # Error correction
└── amplification/
    └── privacy.py          # Privacy amplification
```

**Key integration points:**
- **[caligo/quantum/epr.py](caligo/quantum/epr.py)**: Contains `AliceEPRProgram` and `BobEPRProgram` NetQASM programs
- **[caligo/protocol/alice.py](caligo/protocol/alice.py)**: Orchestrates EPR generation via `epr_generation_phase()`
- **[caligo/protocol/bob.py](caligo/protocol/bob.py)**: Coordinates with Alice, stores EPR outcomes
- **[caligo/simulation/network_builder.py](caligo/simulation/network_builder.py)**: Sets up simulation environment

### 7.2 Proposed Module Structure

Three new modules will be added to support parallel generation:

#### 7.2.1 `caligo/quantum/parallel.py`

**Purpose**: Orchestrates parallel EPR generation using multiprocessing.

**Core Classes:**
```python
@dataclass
class ParallelEPRConfig:
    """Configuration for parallel EPR generation.
    
    Attributes
    ----------
    num_workers : int
        Number of parallel worker processes (default: cpu_count() - 1).
    pairs_per_batch : int
        EPR pairs generated per worker batch.
    isolation_level : Literal["process", "thread"]
        Concurrency model (process recommended for NetSquid).
    prefetch_batches : int
        Number of batches to prefetch (default: 2).
    """
    num_workers: int = field(default_factory=lambda: max(1, cpu_count() - 1))
    pairs_per_batch: int = 1000
    isolation_level: Literal["process", "thread"] = "process"
    prefetch_batches: int = 2


class ParallelEPROrchestrator:
    """Manages parallel EPR pair generation across worker processes.
    
    Coordinates worker lifecycle, batch distribution, and result aggregation
    while maintaining statistical equivalence to sequential generation.
    
    Parameters
    ----------
    config : ParallelEPRConfig
        Parallel execution configuration.
    alice_program : Type[AliceEPRProgram]
        NetQASM program class for Alice.
    bob_program : Type[BobEPRProgram]
        NetQASM program class for Bob.
    network_config : NetworkConfig
        Quantum network topology and noise parameters.
    
    Attributes
    ----------
    _executor : ProcessPoolExecutor
        Worker pool for isolated simulations.
    _result_queue : Queue
        Thread-safe queue for collecting worker outputs.
    _logger : Logger
        Instance logger via LogManager.
    """
    
    def __init__(
        self,
        config: ParallelEPRConfig,
        alice_program: Type[AliceEPRProgram],
        bob_program: Type[BobEPRProgram],
        network_config: NetworkConfig,
    ):
        self._config = config
        self._alice_program = alice_program
        self._bob_program = bob_program
        self._network_config = network_config
        self._executor = ProcessPoolExecutor(max_workers=config.num_workers)
        self._result_queue: Queue = Queue()
        self._logger = LogManager.get_stack_logger(__name__)
    
    def generate_parallel(
        self, total_pairs: int
    ) -> tuple[list[int], list[int], list[int], list[int]]:
        """Generate EPR pairs using parallel workers.
        
        Distributes generation load across workers, aggregates results,
        and ensures random shuffling to preserve i.i.d. assumption.
        
        Parameters
        ----------
        total_pairs : int
            Total number of EPR pairs to generate.
        
        Returns
        -------
        alice_outcomes : list[int]
            Alice's measurement results (0 or 1).
        alice_bases : list[int]
            Alice's measurement bases (0=Z, 1=X).
        bob_outcomes : list[int]
            Bob's measurement results (0 or 1).
        bob_bases : list[int]
            Bob's measurement bases (0=Z, 1=X).
        
        Raises
        ------
        SimulationError
            If worker processes fail or timeout.
        
        Notes
        -----
        Results are shuffled to eliminate ordering artifacts from batching.
        """
        num_batches = math.ceil(total_pairs / self._config.pairs_per_batch)
        futures = []
        
        self._logger.info(
            f"Launching {num_batches} batches across {self._config.num_workers} workers"
        )
        
        for batch_idx in range(num_batches):
            pairs_in_batch = min(
                self._config.pairs_per_batch,
                total_pairs - batch_idx * self._config.pairs_per_batch
            )
            future = self._executor.submit(
                _worker_generate_epr,
                self._alice_program,
                self._bob_program,
                self._network_config,
                pairs_in_batch,
                batch_idx,
            )
            futures.append(future)
        
        # Aggregate results
        all_results = []
        for future in as_completed(futures):
            try:
                result = future.result(timeout=300)  # 5 min timeout
                all_results.append(result)
            except Exception as e:
                self._logger.error(f"Worker failed: {e}")
                self._executor.shutdown(wait=False)
                raise SimulationError(f"Parallel generation failed: {e}") from e
        
        # Concatenate and shuffle to preserve i.i.d.
        alice_outcomes = sum((r["alice_outcomes"] for r in all_results), [])
        alice_bases = sum((r["alice_bases"] for r in all_results), [])
        bob_outcomes = sum((r["bob_outcomes"] for r in all_results), [])
        bob_bases = sum((r["bob_bases"] for r in all_results), [])
        
        # Shuffle all lists with same seed
        indices = list(range(len(alice_outcomes)))
        random.shuffle(indices)
        alice_outcomes = [alice_outcomes[i] for i in indices]
        alice_bases = [alice_bases[i] for i in indices]
        bob_outcomes = [bob_outcomes[i] for i in indices]
        bob_bases = [bob_bases[i] for i in indices]
        
        self._logger.info(f"Generated {len(alice_outcomes)} pairs across {num_batches} batches")
        return alice_outcomes, alice_bases, bob_outcomes, bob_bases
    
    def shutdown(self):
        """Gracefully shutdown worker pool."""
        self._executor.shutdown(wait=True)
        self._logger.debug("Worker pool shutdown complete")
```

#### 7.2.2 `caligo/quantum/workers.py`

**Purpose**: Defines minimal worker programs for isolated simulation contexts.

**Key Functions:**
```python
def _worker_generate_epr(
    alice_program_cls: Type[AliceEPRProgram],
    bob_program_cls: Type[BobEPRProgram],
    network_config: NetworkConfig,
    num_pairs: int,
    batch_id: int,
) -> dict[str, list[int]]:
    """Worker function for isolated EPR generation.
    
    Runs in separate process with independent NetSquid simulator instance.
    
    Parameters
    ----------
    alice_program_cls : Type[AliceEPRProgram]
        Alice's NetQASM program class.
    bob_program_cls : Type[BobEPRProgram]
        Bob's NetQASM program class.
    network_config : NetworkConfig
        Network topology and noise parameters (serialized).
    num_pairs : int
        Number of pairs to generate in this batch.
    batch_id : int
        Unique batch identifier for logging.
    
    Returns
    -------
    dict[str, list[int]]
        Dictionary with keys:
        - "alice_outcomes": Alice's measurement outcomes
        - "alice_bases": Alice's measurement bases
        - "bob_outcomes": Bob's measurement outcomes
        - "bob_bases": Bob's measurement bases
        - "batch_id": Original batch identifier
    
    Notes
    -----
    This function must be picklable and contain no closures.
    NetSquid state is completely isolated per process.
    """
    # Initialize fresh NetSquid simulator
    ns.sim_reset()
    logger = LogManager.get_stack_logger(__name__)
    logger.debug(f"Worker {batch_id} starting {num_pairs} pairs")
    
    # Reconstruct network in this process
    cfg = NetworkBuilder.build_from_config(network_config)
    alice_node = cfg.get_node("Alice")
    bob_node = cfg.get_node("Bob")
    
    # Instantiate programs
    alice_program = alice_program_cls(alice_node, bob_node.name, num_pairs)
    bob_program = bob_program_cls(bob_node, alice_node.name, num_pairs)
    
    # Run simulation
    alice_program.start()
    bob_program.start()
    ns.sim_run()
    
    # Extract results
    alice_results = alice_program.get_results()
    bob_results = bob_program.get_results()
    
    logger.debug(f"Worker {batch_id} completed")
    
    return {
        "alice_outcomes": alice_results["outcomes"],
        "alice_bases": alice_results["bases"],
        "bob_outcomes": bob_results["outcomes"],
        "bob_bases": bob_results["bases"],
        "batch_id": batch_id,
    }


class MinimalAliceWorkerProgram(AliceEPRProgram):
    """Lightweight Alice program for worker processes.
    
    Strips unnecessary overhead from full protocol implementation,
    focusing solely on EPR generation and measurement.
    
    Parameters
    ----------
    node : QuantumNode
        Alice's quantum network node.
    peer : str
        Bob's node name.
    num_pairs : int
        Number of EPR pairs to generate.
    
    Yields
    ------
    EventExpression
        NetQASM events for EPR operations.
    """
    
    def __init__(self, node: QuantumNode, peer: str, num_pairs: int):
        super().__init__(node, peer, num_pairs)
        self._outcomes: list[int] = []
        self._bases: list[int] = []
    
    def run(self) -> Generator[EventExpression, None, None]:
        """Execute EPR generation loop.
        
        Yields
        ------
        EventExpression
            Quantum operations (create_epr, measure).
        """
        for _ in range(self._num_pairs):
            # Create EPR pair
            epr_socket = self._connection.get_epr_socket(self._peer)
            qubit = yield from epr_socket.create_keep()[0]
            
            # Random basis choice
            basis = random.choice([0, 1])  # 0=Z, 1=X
            self._bases.append(basis)
            
            # Measure
            if basis == 1:  # X-basis
                qubit.H()
            outcome = yield from qubit.measure()
            self._outcomes.append(int(outcome))
    
    def get_results(self) -> dict[str, list[int]]:
        """Return measurement outcomes and bases.
        
        Returns
        -------
        dict[str, list[int]]
            Keys: "outcomes", "bases".
        """
        return {"outcomes": self._outcomes, "bases": self._bases}
```

#### 7.2.3 `caligo/quantum/factory.py`

**Purpose**: Factory pattern for seamless switching between sequential/parallel modes.

**Implementation:**
```python
class EPRGenerationStrategy(Protocol):
    """Protocol for EPR generation strategies.
    
    Enables polymorphic switching between sequential and parallel generation.
    """
    
    def generate(
        self, total_pairs: int
    ) -> tuple[list[int], list[int], list[int], list[int]]:
        """Generate EPR pairs using strategy-specific method.
        
        Parameters
        ----------
        total_pairs : int
            Number of EPR pairs to generate.
        
        Returns
        -------
        tuple[list[int], list[int], list[int], list[int]]
            (alice_outcomes, alice_bases, bob_outcomes, bob_bases)
        """
        ...


class SequentialEPRStrategy:
    """Sequential EPR generation (original implementation)."""
    
    def __init__(
        self,
        alice_program: AliceEPRProgram,
        bob_program: BobEPRProgram,
    ):
        self._alice = alice_program
        self._bob = bob_program
    
    def generate(
        self, total_pairs: int
    ) -> tuple[list[int], list[int], list[int], list[int]]:
        """Generate pairs sequentially using existing code path."""
        # Delegate to original implementation
        self._alice.set_num_pairs(total_pairs)
        self._bob.set_num_pairs(total_pairs)
        self._alice.start()
        self._bob.start()
        ns.sim_run()
        return (
            self._alice.outcomes,
            self._alice.bases,
            self._bob.outcomes,
            self._bob.bases,
        )


class ParallelEPRStrategy:
    """Parallel EPR generation using multiprocessing."""
    
    def __init__(self, orchestrator: ParallelEPROrchestrator):
        self._orchestrator = orchestrator
    
    def generate(
        self, total_pairs: int
    ) -> tuple[list[int], list[int], list[int], list[int]]:
        """Generate pairs in parallel using worker pool."""
        return self._orchestrator.generate_parallel(total_pairs)


class EPRGenerationFactory:
    """Factory for creating EPR generation strategies.
    
    Centralizes strategy selection based on configuration or runtime conditions.
    
    Parameters
    ----------
    config : CaligoConfig
        Global configuration object.
    network_config : NetworkConfig
        Network topology configuration.
    
    Methods
    -------
    create_strategy() -> EPRGenerationStrategy
        Returns appropriate strategy based on config.parallel_enabled.
    
    Examples
    --------
    >>> factory = EPRGenerationFactory(config, network_config)
    >>> strategy = factory.create_strategy()
    >>> results = strategy.generate(total_pairs=10000)
    """
    
    def __init__(self, config: CaligoConfig, network_config: NetworkConfig):
        self._config = config
        self._network_config = network_config
    
    def create_strategy(self) -> EPRGenerationStrategy:
        """Create appropriate EPR generation strategy.
        
        Returns
        -------
        EPRGenerationStrategy
            Sequential or parallel strategy based on config.
        """
        if self._config.parallel_config.enabled:
            orchestrator = ParallelEPROrchestrator(
                config=self._config.parallel_config,
                alice_program=MinimalAliceWorkerProgram,
                bob_program=MinimalBobWorkerProgram,
                network_config=self._network_config,
            )
            return ParallelEPRStrategy(orchestrator)
        else:
            # Use existing sequential programs
            alice = AliceEPRProgram(
                self._network_config.alice_node,
                self._network_config.bob_node.name,
                0,  # Will be set via set_num_pairs()
            )
            bob = BobEPRProgram(
                self._network_config.bob_node,
                self._network_config.alice_node.name,
                0,
            )
            return SequentialEPRStrategy(alice, bob)
```

### 7.3 Integration with Existing Modules

#### 7.3.1 Refactor `caligo/quantum/epr.py`

**Changes:**
1. Extract `AliceEPRProgram.run()` core logic into reusable method `_generate_single_pair()`
2. Add `get_results()` method for consistent result extraction
3. Make measurement basis selection injectable for testing

**Diff:**
```python
# Before
class AliceEPRProgram(NetQASMProgram):
    def run(self):
        for i in range(self._num_pairs):
            # Inline EPR generation + measurement
            ...

# After
class AliceEPRProgram(NetQASMProgram):
    def __init__(self, node, peer, num_pairs, basis_strategy=None):
        super().__init__(node, peer)
        self._num_pairs = num_pairs
        self._basis_strategy = basis_strategy or RandomBasisStrategy()
    
    def _generate_single_pair(self) -> tuple[int, int]:
        """Generate one EPR pair and measure.
        
        Returns
        -------
        tuple[int, int]
            (outcome, basis)
        """
        epr_socket = self._connection.get_epr_socket(self._peer)
        qubit = yield from epr_socket.create_keep()[0]
        basis = self._basis_strategy.choose()
        if basis == 1:
            qubit.H()
        outcome = yield from qubit.measure()
        return int(outcome), basis
    
    def run(self):
        """Generate all EPR pairs using _generate_single_pair."""
        for _ in range(self._num_pairs):
            outcome, basis = yield from self._generate_single_pair()
            self._outcomes.append(outcome)
            self._bases.append(basis)
    
    def get_results(self) -> dict[str, list[int]]:
        """Return measurement outcomes and bases."""
        return {"outcomes": self._outcomes, "bases": self._bases}
```

#### 7.3.2 Update `caligo/protocol/base.py`

**Add configuration support:**
```python
@dataclass
class CaligoConfig:
    """Master configuration for Caligo QKD protocol.
    
    Attributes
    ----------
    num_epr_pairs : int
        Total EPR pairs to generate.
    parallel_config : ParallelEPRConfig
        Parallel generation settings.
    network_config : NetworkConfig
        Quantum network parameters.
    security_params : SecurityParams
        NSM security parameters.
    """
    num_epr_pairs: int
    parallel_config: ParallelEPRConfig = field(default_factory=ParallelEPRConfig)
    network_config: NetworkConfig = field(default_factory=NetworkConfig)
    security_params: SecurityParams = field(default_factory=SecurityParams)
```

#### 7.3.3 Modify `caligo/protocol/alice.py`

**Inject factory-created strategy:**
```python
class AliceProtocol:
    def __init__(self, config: CaligoConfig):
        self._config = config
        self._factory = EPRGenerationFactory(config, config.network_config)
        self._epr_strategy = self._factory.create_strategy()
        self._logger = LogManager.get_stack_logger(__name__)
    
    def epr_generation_phase(self) -> EPRResults:
        """Execute EPR generation using configured strategy.
        
        Returns
        -------
        EPRResults
            Dataclass containing all raw EPR data.
        """
        self._logger.info(
            f"Starting EPR generation: {self._config.num_epr_pairs} pairs, "
            f"mode={'parallel' if self._config.parallel_config.enabled else 'sequential'}"
        )
        
        alice_outcomes, alice_bases, bob_outcomes, bob_bases = \
            self._epr_strategy.generate(self._config.num_epr_pairs)
        
        return EPRResults(
            alice_outcomes=alice_outcomes,
            alice_bases=alice_bases,
            bob_outcomes=bob_outcomes,
            bob_bases=bob_bases,
        )
```

### 7.4 CLI and Configuration

#### 7.4.1 Command-Line Arguments

**Extend `caligo/cli.py`:**
```python
def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser with parallel generation options.
    
    Returns
    -------
    argparse.ArgumentParser
        Configured parser.
    """
    parser = argparse.ArgumentParser(description="Caligo QKD Protocol")
    
    # Existing arguments
    parser.add_argument("--num-pairs", type=int, default=10000)
    parser.add_argument("--config", type=str, help="Path to YAML config")
    
    # New parallel arguments
    parallel_group = parser.add_argument_group("parallel generation")
    parallel_group.add_argument(
        "--parallel",
        action="store_true",
        help="Enable parallel EPR generation"
    )
    parallel_group.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes (default: CPU count - 1)"
    )
    parallel_group.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="EPR pairs per worker batch"
    )
    
    return parser


def load_config(args: argparse.Namespace) -> CaligoConfig:
    """Load configuration from CLI args and/or YAML file.
    
    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.
    
    Returns
    -------
    CaligoConfig
        Fully configured protocol parameters.
    """
    if args.config:
        with open(args.config) as f:
            yaml_config = yaml.safe_load(f)
    else:
        yaml_config = {}
    
    # Merge CLI args with YAML (CLI takes precedence)
    parallel_config = ParallelEPRConfig(
        enabled=args.parallel or yaml_config.get("parallel", {}).get("enabled", False),
        num_workers=args.workers or yaml_config.get("parallel", {}).get("workers"),
        pairs_per_batch=args.batch_size or yaml_config.get("parallel", {}).get("batch_size", 1000),
    )
    
    return CaligoConfig(
        num_epr_pairs=args.num_pairs,
        parallel_config=parallel_config,
        # ... other configs
    )
```

#### 7.4.2 YAML Configuration

**Example `../example_configs/parallel.yaml`:** (outside of caligo codebase)
```yaml
num_epr_pairs: 50000

parallel:
  enabled: true
  workers: 8
  batch_size: 5000
  isolation_level: process
  prefetch_batches: 2

network:
  distance_km: 10
  fiber_loss_db_per_km: 0.2
  detector_efficiency: 0.85
  dark_count_rate: 1e-6

security:
  nsm_storage_rate: 0.5
  error_correction_efficiency: 1.16
  target_security_parameter: 1e-9
```

### 7.5 Testing Strategy

#### 7.5.1 Unit Tests

**File: `tests/quantum/test_parallel.py`**

```python
"""Unit tests for parallel EPR generation module."""

import pytest
from unittest.mock import Mock, patch
from caligo.quantum.parallel import ParallelEPROrchestrator, ParallelEPRConfig
from caligo.quantum.workers import _worker_generate_epr


class TestParallelEPRConfig:
    """Test ParallelEPRConfig dataclass."""
    
    def test_default_workers(self):
        """Test default worker count is CPU count - 1."""
        config = ParallelEPRConfig()
        assert config.num_workers >= 1
        assert config.num_workers == max(1, cpu_count() - 1)
    
    def test_custom_workers(self):
        """Test custom worker count."""
        config = ParallelEPRConfig(num_workers=4)
        assert config.num_workers == 4
    
    def test_isolation_level_validation(self):
        """Test isolation_level only accepts valid values."""
        config = ParallelEPRConfig(isolation_level="process")
        assert config.isolation_level == "process"
        
        with pytest.raises(ValueError):
            ParallelEPRConfig(isolation_level="invalid")


class TestParallelEPROrchestrator:
    """Test ParallelEPROrchestrator class."""
    
    @pytest.fixture
    def mock_network_config(self):
        """Mock NetworkConfig fixture."""
        return Mock(spec=NetworkConfig)
    
    @pytest.fixture
    def orchestrator(self, mock_network_config):
        """Create orchestrator with mock config."""
        config = ParallelEPRConfig(num_workers=2, pairs_per_batch=100)
        return ParallelEPROrchestrator(
            config=config,
            alice_program=Mock,
            bob_program=Mock,
            network_config=mock_network_config,
        )
    
    def test_batch_distribution(self, orchestrator):
        """Test batch count calculation."""
        total_pairs = 250
        num_batches = math.ceil(250 / orchestrator._config.pairs_per_batch)
        assert num_batches == 3  # 100 + 100 + 50
    
    @patch("caligo.quantum.parallel._worker_generate_epr")
    def test_generate_parallel_submission(self, mock_worker, orchestrator):
        """Test worker task submission."""
        mock_worker.return_value = {
            "alice_outcomes": [0] * 100,
            "alice_bases": [0] * 100,
            "bob_outcomes": [0] * 100,
            "bob_bases": [0] * 100,
            "batch_id": 0,
        }
        
        results = orchestrator.generate_parallel(total_pairs=200)
        assert len(results[0]) == 200  # alice_outcomes
        assert mock_worker.call_count == 2  # 2 batches
    
    def test_result_shuffling(self, orchestrator):
        """Test results are shuffled to preserve i.i.d."""
        with patch("caligo.quantum.parallel._worker_generate_epr") as mock_worker:
            # Worker returns sequential outcomes
            mock_worker.return_value = {
                "alice_outcomes": list(range(100)),
                "alice_bases": [0] * 100,
                "bob_outcomes": list(range(100)),
                "bob_bases": [0] * 100,
                "batch_id": 0,
            }
            
            results = orchestrator.generate_parallel(total_pairs=100)
            alice_outcomes = results[0]
            
            # Results should not be strictly sequential
            assert alice_outcomes != list(range(100))


class TestWorkerFunction:
    """Test _worker_generate_epr worker function."""
    
    @pytest.mark.integration
    def test_worker_isolation(self):
        """Test worker runs in isolated NetSquid context."""
        # This requires actual NetSquid setup
        mock_config = Mock(spec=NetworkConfig)
        
        result = _worker_generate_epr(
            alice_program_cls=MinimalAliceWorkerProgram,
            bob_program_cls=MinimalBobWorkerProgram,
            network_config=mock_config,
            num_pairs=10,
            batch_id=0,
        )
        
        assert "alice_outcomes" in result
        assert len(result["alice_outcomes"]) == 10
        assert result["batch_id"] == 0
```

**File: `tests/quantum/test_factory.py`**

```python
"""Test EPR generation factory pattern."""

import pytest
from caligo.quantum.factory import (
    EPRGenerationFactory,
    SequentialEPRStrategy,
    ParallelEPRStrategy,
)


class TestEPRGenerationFactory:
    """Test EPRGenerationFactory strategy creation."""
    
    @pytest.fixture
    def sequential_config(self):
        """Config with parallel disabled."""
        config = Mock(spec=CaligoConfig)
        config.parallel_config.enabled = False
        return config
    
    @pytest.fixture
    def parallel_config(self):
        """Config with parallel enabled."""
        config = Mock(spec=CaligoConfig)
        config.parallel_config.enabled = True
        config.parallel_config.num_workers = 4
        return config
    
    def test_create_sequential_strategy(self, sequential_config):
        """Test factory returns sequential strategy when disabled."""
        factory = EPRGenerationFactory(sequential_config, Mock())
        strategy = factory.create_strategy()
        assert isinstance(strategy, SequentialEPRStrategy)
    
    def test_create_parallel_strategy(self, parallel_config):
        """Test factory returns parallel strategy when enabled."""
        factory = EPRGenerationFactory(parallel_config, Mock())
        strategy = factory.create_strategy()
        assert isinstance(strategy, ParallelEPRStrategy)


class TestStrategyInterface:
    """Test strategy implementations follow protocol."""
    
    def test_sequential_strategy_generate(self):
        """Test SequentialEPRStrategy.generate() signature."""
        mock_alice = Mock()
        mock_bob = Mock()
        mock_alice.outcomes = [0, 1]
        mock_alice.bases = [0, 0]
        mock_bob.outcomes = [0, 1]
        mock_bob.bases = [0, 0]
        
        strategy = SequentialEPRStrategy(mock_alice, mock_bob)
        results = strategy.generate(total_pairs=2)
        
        assert len(results) == 4  # (alice_out, alice_bases, bob_out, bob_bases)
        assert len(results[0]) == 2
```

#### 7.5.2 Integration Tests

**File: `tests/integration/test_parallel_protocol.py`**

```python
"""Integration tests for parallel generation in full protocol."""

import pytest
from caligo.protocol.alice import AliceProtocol
from caligo.protocol.bob import BobProtocol


@pytest.mark.integration
class TestParallelProtocolIntegration:
    """Test parallel generation integrated with full QKD protocol."""
    
    @pytest.fixture
    def parallel_config(self):
        """Create config with parallel enabled."""
        return CaligoConfig(
            num_epr_pairs=1000,
            parallel_config=ParallelEPRConfig(
                enabled=True,
                num_workers=2,
                pairs_per_batch=500,
            ),
        )
    
    def test_full_protocol_run(self, parallel_config):
        """Test complete QKD protocol with parallel EPR generation."""
        alice = AliceProtocol(parallel_config)
        bob = BobProtocol(parallel_config)
        
        # EPR generation phase
        epr_results = alice.epr_generation_phase()
        assert len(epr_results.alice_outcomes) == 1000
        
        # Sifting phase (uses EPR results)
        sifted_key = alice.sift_phase(epr_results)
        assert len(sifted_key) < 1000  # Some bases mismatch
        
        # Remaining phases should work identically
        reconciled_key = alice.reconciliation_phase(sifted_key)
        final_key = alice.privacy_amplification_phase(reconciled_key)
        
        assert len(final_key) > 0
    
    def test_sequential_vs_parallel_equivalence(self):
        """Test sequential and parallel modes produce statistically equivalent results."""
        base_config = CaligoConfig(num_epr_pairs=5000)
        
        # Run sequential
        seq_config = base_config
        seq_config.parallel_config.enabled = False
        alice_seq = AliceProtocol(seq_config)
        epr_seq = alice_seq.epr_generation_phase()
        
        # Run parallel
        par_config = base_config
        par_config.parallel_config.enabled = True
        par_config.parallel_config.num_workers = 4
        alice_par = AliceProtocol(par_config)
        epr_par = alice_par.epr_generation_phase()
        
        # Statistical equivalence tests
        qber_seq = calculate_qber(epr_seq.alice_outcomes, epr_par.bob_outcomes)
        qber_par = calculate_qber(epr_par.alice_outcomes, epr_par.bob_outcomes)
        
        # QBER should be within 1% (statistical noise)
        assert abs(qber_seq - qber_par) < 0.01
```

#### 7.5.3 End-to-End Tests

**File: `tests/e2e/test_parallel_simulation.py`**

```python
"""End-to-end tests with real NetSquid simulation."""

import pytest
import netsquid as ns


@pytest.mark.e2e
@pytest.mark.slow
class TestParallelSimulationE2E:
    """Test parallel generation with actual NetSquid simulator."""
    
    def test_10k_pairs_parallel(self):
        """Generate 10k EPR pairs using 4 workers."""
        config = CaligoConfig(
            num_epr_pairs=10000,
            parallel_config=ParallelEPRConfig(
                enabled=True,
                num_workers=4,
                pairs_per_batch=2500,
            ),
        )
        
        factory = EPRGenerationFactory(config, config.network_config)
        strategy = factory.create_strategy()
        
        results = strategy.generate(total_pairs=10000)
        alice_outcomes, alice_bases, bob_outcomes, bob_bases = results
        
        # Validate results
        assert len(alice_outcomes) == 10000
        assert len(set(alice_bases)) == 2  # Both Z and X bases used
        assert 0.4 < sum(alice_outcomes) / len(alice_outcomes) < 0.6  # ~50% ones
    
    def test_realistic_noise_model(self):
        """Test parallel generation with realistic channel noise."""
        # 10 km fiber, 0.2 dB/km loss
        network_config = NetworkConfig(
            distance_km=10,
            fiber_loss_db_per_km=0.2,
            detector_efficiency=0.85,
        )
        
        config = CaligoConfig(
            num_epr_pairs=5000,
            parallel_config=ParallelEPRConfig(enabled=True, num_workers=2),
            network_config=network_config,
        )
        
        factory = EPRGenerationFactory(config, network_config)
        strategy = factory.create_strategy()
        results = strategy.generate(5000)
        
        # Calculate empirical QBER
        qber = calculate_qber(results[0], results[2])
        assert 0.01 < qber < 0.15  # Realistic QBER range
```

#### 7.5.4 Performance Benchmarks

**File: `tests/performance/test_parallel_speedup.py`**

```python
"""Performance benchmarks for parallel generation."""

import pytest
import time


@pytest.mark.performance
class TestParallelSpeedup:
    """Measure speedup from parallelization."""
    
    @pytest.mark.parametrize("num_pairs", [10000, 50000, 100000])
    @pytest.mark.parametrize("num_workers", [1, 2, 4, 8])
    def test_scaling_efficiency(self, num_pairs, num_workers):
        """Test speedup vs worker count."""
        config = CaligoConfig(
            num_epr_pairs=num_pairs,
            parallel_config=ParallelEPRConfig(
                enabled=(num_workers > 1),
                num_workers=num_workers,
                pairs_per_batch=num_pairs // num_workers,
            ),
        )
        
        factory = EPRGenerationFactory(config, config.network_config)
        strategy = factory.create_strategy()
        
        start = time.perf_counter()
        results = strategy.generate(num_pairs)
        elapsed = time.perf_counter() - start
        
        # Log results for analysis
        print(f"Pairs: {num_pairs}, Workers: {num_workers}, Time: {elapsed:.2f}s")
        
        # Basic sanity check (parallel should not be slower for large N)
        if num_workers > 1 and num_pairs > 10000:
            baseline_time = self._get_baseline_time(num_pairs)
            assert elapsed < baseline_time * 1.2  # At most 20% overhead
    
    def _get_baseline_time(self, num_pairs: int) -> float:
        """Run sequential baseline."""
        config = CaligoConfig(
            num_epr_pairs=num_pairs,
            parallel_config=ParallelEPRConfig(enabled=False),
        )
        factory = EPRGenerationFactory(config, config.network_config)
        strategy = factory.create_strategy()
        
        start = time.perf_counter()
        strategy.generate(num_pairs)
        return time.perf_counter() - start
```

#### 7.5.5 Security Validation Tests

**File: `tests/security/test_parallel_iid.py`**

```python
"""Validate i.i.d. assumption preservation."""

import pytest
from scipy.stats import chi2_contingency


@pytest.mark.security
class TestParallelIIDPreservation:
    """Test parallel generation maintains i.i.d. properties."""
    
    def test_basis_independence(self):
        """Test measurement bases are independent across pairs."""
        config = CaligoConfig(
            num_epr_pairs=10000,
            parallel_config=ParallelEPRConfig(enabled=True, num_workers=4),
        )
        
        factory = EPRGenerationFactory(config, config.network_config)
        strategy = factory.create_strategy()
        _, alice_bases, _, _ = strategy.generate(10000)
        
        # Chi-squared test for independence of consecutive bases
        contingency_table = [
            [sum(1 for i in range(len(alice_bases)-1) if alice_bases[i] == 0 and alice_bases[i+1] == 0),
             sum(1 for i in range(len(alice_bases)-1) if alice_bases[i] == 0 and alice_bases[i+1] == 1)],
            [sum(1 for i in range(len(alice_bases)-1) if alice_bases[i] == 1 and alice_bases[i+1] == 0),
             sum(1 for i in range(len(alice_bases)-1) if alice_bases[i] == 1 and alice_bases[i+1] == 1)],
        ]
        
        chi2, p_value, _, _ = chi2_contingency(contingency_table)
        assert p_value > 0.05  # Fail to reject independence (good)
    
    def test_batch_boundary_mixing(self):
        """Test no correlation at batch boundaries."""
        config = CaligoConfig(
            num_epr_pairs=4000,
            parallel_config=ParallelEPRConfig(
                enabled=True,
                num_workers=4,
                pairs_per_batch=1000,  # 4 batches
            ),
        )
        
        factory = EPRGenerationFactory(config, config.network_config)
        strategy = factory.create_strategy()
        alice_outcomes, _, _, _ = strategy.generate(4000)
        
        # Extract outcomes at batch boundaries (indices 999, 1000, 1999, 2000, etc.)
        boundary_indices = [999, 1000, 1999, 2000, 2999, 3000]
        boundary_outcomes = [alice_outcomes[i] for i in boundary_indices]
        
        # Should have ~50% 0s and 50% 1s (no systematic bias)
        assert 0.3 < sum(boundary_outcomes) / len(boundary_outcomes) < 0.7
```

### 7.6 Implementation Checklist

**Week 1: Core Infrastructure**
- [ ] Create `caligo/quantum/parallel.py` with `ParallelEPROrchestrator`
- [ ] Create `caligo/quantum/workers.py` with `_worker_generate_epr`
- [ ] Create `caligo/quantum/factory.py` with strategy pattern
- [ ] Add `ParallelEPRConfig` to `caligo/protocol/base.py`
- [ ] Write unit tests for new modules

**Week 2: Integration**
- [ ] Refactor `caligo/quantum/epr.py` to extract reusable methods
- [ ] Update `caligo/protocol/alice.py` to use factory pattern
- [ ] Update `caligo/protocol/bob.py` to handle parallel results
- [ ] Write integration tests

**Week 3: CLI and Configuration**
- [ ] Extend `caligo/cli.py` with parallel arguments
- [ ] Implement YAML config loading in `load_config()`
- [ ] Create example configs in `config/parallel.yaml`
- [ ] Test CLI argument precedence

**Week 4: Testing and Validation**
- [ ] Write E2E tests with real NetSquid
- [ ] Run performance benchmarks (10k, 50k, 100k pairs)
- [ ] Validate i.i.d. preservation with statistical tests
- [ ] Verify QBER equivalence between sequential/parallel

**Week 5: Documentation and Deployment**
- [ ] Update [README.md](caligo/README.md) with parallel usage examples
- [ ] Add docstrings to all new modules (Numpydoc format)
- [ ] Create migration guide for existing users
- [ ] Profile memory usage and optimize if needed

---

## References

1. Bennett, C. H., & Brassard, G. (1984). Quantum cryptography: Public key distribution and coin tossing.
2. König, R., Renner, R., & Schaffner, C. (2012). The operational meaning of min- and max-entropy.
3. Wehner, S., Schaffner, C., & Terhal, B. M. (2010). Cryptography from noisy storage.
4. Erven, C., et al. (2014). Experimental three-particle quantum nonlocality under strict locality conditions.
5. Tomamichel, M., Lim, C. C. W., Gisin, N., & Renner, R. (2012). Tight finite-key analysis for quantum cryptography.
6. SquidASM Documentation: https://squidasm.readthedocs.io/
7. NetSquid Documentation: https://netsquid.org/

---

*Document Version: 1.0*  
*Last Updated: 2025-12-19*
