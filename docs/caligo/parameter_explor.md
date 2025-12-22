# Implementation Plan: Automated High-Dimensional Parameter Exploration for Caligo (v3.0)

**Target System:** Caligo QKD Framework
**Persistence:** HDF5 (with State Preservation)
**Architecture:** Full-Stack Injection (Quantum + Classical)

---

## 1. Executive Summary

This document specifies the architecture for an autonomous adversarial stress-testing suite for the Caligo QKD protocol. This system moves beyond theoretical bounds checking to empirical verification by executing the **full protocol stack**—from quantum state generation to privacy amplification.

It features a **Resilient Loop Architecture** capable of pausing and resuming long-running exploration campaigns (e.g., multi-day runs). It utilizes **HDF5** for efficient high-dimensional data storage and employs a **Hybrid Active-Surrogate** strategy to map the 9-dimensional "Security Cliff."

---

## 2. Architecture: The "Injection" Pattern

To validate the full pipeline without incurring the massive overhead of sequential discrete-event simulation for every layer, we adopt a **Pre-computation + Injection** architecture.

```
┌─────────────────────────┐       ┌──────────────────────────────┐
│  Exploration Controller │       │   Parallel EPR Orchestrator  │
│ (LHS / Bayesian Optim)  │──────►│    (Multiprocessing Pool)    │
└────────────┬────────────┘       └──────────────┬───────────────┘
             │                                   │
             │ 1. Params                         │ 2. Raw Quantum Data
             │                                   │ (Outcomes + Bases)
             ▼                                   ▼
┌────────────────────────────────────────────────────────────────┐
│                     Simulation Harness                         │
│ ┌──────────────────────┐           ┌──────────────────────┐    │
│ │    AliceProtocol     │           │     BobProtocol      │    │
│ │  (Real Logic/State)  │           │  (Real Logic/State)  │    │
│ └──────────┬───────────┘           └───────────┬──────────┘    │
│            │                                   │               │
│            ▼                                   ▼               │
│ ┌──────────────────────┐           ┌──────────────────────┐    │
│ │ PrecomputedBackend   │◄─────────►│ PrecomputedBackend   │    │
│ └──────────────────────┘           └──────────────────────┘    │
└────────────────────────────┬───────────────────────────────────┘
                             │
                             │ 3. Full Protocol Metrics
                             │ (Success/Fail, Leakage, CPU Time)
                             ▼
                  ┌──────────────────────┐
                  │   Surrogate Model    │
                  └──────────────────────┘
```

---

## 3. The 9-Dimensional Parameter Space

| Dim | Parameter | Symbol | Range | Sampling |
|:---:|:---|:---:|:---|:---|
| **1** | Storage Noise | $r$ | $[0.0, 1.0]$ | Linear |
| **2** | Storage Rate | $\nu$ | $[0.0, 1.0]$ | Log-Uniform |
| **3** | Wait Time | $\Delta t$ | $[10^5, 10^9]$ ns | Log-Uniform |
| **4** | Channel Fidelity | $F$ | $[0.5, 1.0]$ | Beta Dist |
| **5** | Detection Eff. | $\eta$ | $[10^{-3}, 1.0]$ | Log-Uniform |
| **6** | Detector Error | $e_{det}$ | $[0.0, 0.1]$ | Linear |
| **7** | Dark Counts | $P_{dark}$ | $[10^{-8}, 10^{-3}]$ | Log-Uniform |
| **8** | Input EPR Pairs | $N$ | $[10^4, 10^6]$ | Log-Uniform |
| **9** | Strategy | $S$ | $\{Base, Blind\}$ | Categorical |

---

## 4. Data Storage & Persistence Layer

**Objective:** Robust storage of high-dimensional simulation data with fault tolerance.

### 4.1 Dependencies
*   **`h5py`:** For HDF5 file manipulation.
*   **`dill` (or `pickle`):** For serializing the State Manager and Surrogate Models.

### 4.2 HDF5 Schema (`exploration_data.h5`)
The file is structured to separate sampling phases while allowing efficient appending.

*   **Group: `/lhs_warmup`**
    *   Dataset: `inputs` (Shape: $M \times 9$, Float32)
    *   Dataset: `metrics` (Shape: $M \times 4$, Float32) - [NetEfficiency, QBER, Rounds, Duration]
    *   Dataset: `metadata` (Shape: $M$, String) - JSON dump of error codes/exceptions.
*   **Group: `/active_learning`**
    *   (Same structure as above, appended iteratively).

### 4.3 State Management
**Module: `caligo.exploration.persistence`**

*   **Class: `StateManager`**
    *   **Responsibility:** Tracks the "Cursor" of the exploration.
    *   **Attributes:**
        *   `current_phase`: Enum (LHS, TRAINING, ACTIVE).
        *   `samples_collected`: Integer count.
        *   `rng_state`: The serialized state of the Numpy/Scipy random number generator (crucial for deterministic replay).
        *   `surrogate_model_path`: Path to the latest serialized GP model.
    *   **Methods:**
        *   `checkpoint()`: Atomically saves state to `state.pkl`.
        *   `load()`: Restores state, checking for file integrity.

---

## 5. Architecture: The Injection Harness

To simulate the full pipeline efficiently, we bypass the NetSquid network layer during the classical phase and inject the pre-computed quantum data directly into the protocols.

### 5.1 New Modules

**`caligo.exploration.harness`**

*   **Class: `MockConnection`**
    *   Simulates the Classical Channel (`NetQASMConnection`).
    *   Uses Python `queue.Queue` to pass messages instantaneously between Alice and Bob instances in memory.

*   **Class: `PrecomputedEPRSocket`**
    *   Replaces the standard `EPRSocket`.
    *   **Input:** Arrays of Outcomes and Bases (from Parallel Orchestrator).
    *   **Behavior:** When `create_keep()` or `measure()` is called, it yields the next value from the arrays instead of invoking the quantum simulator.

*   **Class: `ProtocolHarness`**
    *   **Responsibility:** The "Sandbox."
    *   **Workflow:**
        1.  Initialize `AliceProtocol` and `BobProtocol` with `PrecomputedEPRSocket`.
        2.  **Quantum Phase (Mocked):** "Fast-forward" the EPR generation by loading the pre-computed arrays into the sockets.
        3.  **Classical Phase (Real):** Execute `alice.run_classical()` and `bob.run_classical()`.
        4.  **Capture:** Monitor internal states for `LeakageBudgetExceeded`, `FrameError`, etc.
        5.  **Output:** Return a standardized `SimulationResult`.

---

## 6. Phase 1: Data Generation (LHS)

**Objective:** Populate the `lhs_warmup` group in HDF5.

### 6.1 Architecture Integration

Phase 1 leverages three existing Caligo modules to build the exploration infrastructure:

#### 6.1.1 Quantum Generation Layer (`caligo.quantum.parallel`)

**Existing Infrastructure:**
- [ParallelEPROrchestrator](caligo/caligo/quantum/parallel.py#L245-L350): Production-ready parallel EPR generation
- [EPRWorkerResult](caligo/caligo/quantum/parallel.py#L155-L195): Structured batch results
- [ParallelEPRConfig](caligo/caligo/quantum/parallel.py#L64-L145): Configurable parallelization settings
- [_worker_generate_epr()](caligo/caligo/quantum/parallel.py#L457-L560): Process-isolated worker function

**What We Need to Build:**
```python
# caligo/exploration/epr_batcher.py

@dataclass
class ExplorationSample:
    """Single parameter configuration for exploration."""
    sample_id: int
    storage_noise_r: float
    storage_rate_nu: float
    delta_t_ns: float
    channel_fidelity: float
    detection_eff_eta: float
    detector_error: float
    dark_count_prob: float
    num_pairs: int
    strategy: str  # "baseline" or "blind"


class BatchedEPROrchestrator:
    """
    Wraps ParallelEPROrchestrator to generate quantum data for multiple
    parameter configurations sequentially.
    
    This class iterates over exploration samples and generates EPR pairs
    for each configuration using the existing parallel infrastructure.
    """
    
    def __init__(self, parallel_config: ParallelEPRConfig):
        self._parallel_config = parallel_config
    
    def generate_for_samples(
        self, 
        samples: List[ExplorationSample]
    ) -> Dict[int, PrecomputedEPRData]:
        """
        Generate EPR data for all samples.
        
        For each sample:
        1. Build NetworkConfig from NSMParameters
        2. Create ParallelEPROrchestrator instance
        3. Call generate_parallel()
        4. Wrap results in PrecomputedEPRData
        5. Store in dictionary keyed by sample_id
        
        Returns
        -------
        Dict[int, PrecomputedEPRData]
            Mapping from sample_id to EPR measurement results.
        """
        results = {}
        
        for sample in samples:
            # Build NSMParameters from exploration sample
            nsm_params = NSMParameters(
                storage_noise_r=sample.storage_noise_r,
                storage_rate_nu=sample.storage_rate_nu,
                delta_t_ns=sample.delta_t_ns,
                channel_fidelity=sample.channel_fidelity,
                detection_eff_eta=sample.detection_eff_eta,
                detector_error=sample.detector_error,
                dark_count_prob=sample.dark_count_prob,
            )
            
            # Serialize to dict for worker processes
            network_config = {
                "fidelity": nsm_params.channel_fidelity,
                "noise": 1.0 - nsm_params.channel_fidelity,
                "detection_eff": nsm_params.detection_eff_eta,
                "dark_count_prob": nsm_params.dark_count_prob,
            }
            
            # Generate EPR pairs in parallel
            orchestrator = ParallelEPROrchestrator(
                config=self._parallel_config,
                network_config=network_config,
            )
            
            try:
                alice_out, alice_bases, bob_out, bob_bases = \
                    orchestrator.generate_parallel(sample.num_pairs)
                
                results[sample.sample_id] = PrecomputedEPRData(
                    alice_outcomes=alice_out,
                    alice_bases=alice_bases,
                    bob_outcomes=bob_out,
                    bob_bases=bob_bases,
                )
            finally:
                orchestrator.shutdown()
        
        return results
```

#### 6.1.2 Classical Protocol Execution Layer (`caligo.protocol`)

**Existing Infrastructure:**
- [ProtocolParameters](caligo/caligo/protocol/base.py#L69-L101): Dataclass with `precomputed_epr` hook
- [PrecomputedEPRData](caligo/caligo/protocol/base.py#L42-L66): Injection container for EPR data
- [AliceProgram](caligo/caligo/protocol/alice.py#L38-L515): Full sender protocol with EPR bypass
- [BobProgram](caligo/caligo/protocol/bob.py#L28-L395): Full receiver protocol with EPR bypass
- [run_protocol()](caligo/caligo/protocol/orchestrator.py#L19-L142): SquidASM runner

**Key Injection Points:**
```python
# From caligo/protocol/alice.py:141-157
def _phase1_quantum(self, context) -> Generator[Any, None, Tuple[np.ndarray, np.ndarray]]:
    """Generate and measure EPR pairs (Alice side)."""
    
    if self.params.precomputed_epr is not None:  # <-- INJECTION HOOK
        n = int(self.params.num_pairs)
        outcomes = np.asarray(
            self.params.precomputed_epr.alice_outcomes, 
            dtype=np.uint8
        )
        bases = np.asarray(
            self.params.precomputed_epr.alice_bases, 
            dtype=np.uint8
        )
        if len(outcomes) != n or len(bases) != n:
            raise ValueError("precomputed_epr length mismatch")
        self._timing_barrier.mark_quantum_complete()
        return outcomes, bases  # <-- Bypass EPR socket operations
    
    # Normal SquidASM EPR generation path...
```

**What We Need to Build:**
```python
# caligo/exploration/harness.py

class ProtocolHarness:
    """
    Executes full Caligo protocol (Phases I-IV) using precomputed quantum data.
    
    This is the critical performance optimization: instead of running
    NetSquid discrete-event simulation for EVERY parameter configuration,
    we pre-generate quantum measurement outcomes in parallel workers and
    inject them into the protocol via PrecomputedEPRData.
    
    The classical protocol (sifting, reconciliation, amplification) still
    runs in full, preserving all LDPC decoding, hash verification, and
    security computations.
    """
    
    def __init__(self, num_workers: int = 1):
        """
        Parameters
        ----------
        num_workers : int
            Number of parallel protocol executors. Use 1 for single-threaded
            execution (GIL-bound). Use ProcessPoolExecutor for true parallelism.
        """
        self._num_workers = num_workers
        self._executor = None
    
    def execute_batch(
        self,
        samples: List[ExplorationSample],
        epr_data: Dict[int, PrecomputedEPRData],
    ) -> Dict[int, ProtocolResult]:
        """
        Execute protocol for all samples using precomputed EPR data.
        
        Parameters
        ----------
        samples : List[ExplorationSample]
            Parameter configurations to test.
        epr_data : Dict[int, PrecomputedEPRData]
            Precomputed quantum measurements keyed by sample_id.
        
        Returns
        -------
        Dict[int, ProtocolResult]
            Protocol outcomes keyed by sample_id.
        """
        results = {}
        
        for sample in samples:
            result = self._execute_single(sample, epr_data[sample.sample_id])
            results[sample.sample_id] = result
        
        return results
    
    def _execute_single(
        self,
        sample: ExplorationSample,
        epr_data: PrecomputedEPRData,
    ) -> ProtocolResult:
        """
        Execute protocol for a single parameter configuration.
        
        This method builds the ProtocolParameters, injects the EPR data,
        and calls run_protocol(). The protocol runs WITHOUT SquidASM
        quantum simulation—only classical processing.
        """
        # Build NSMParameters
        nsm_params = NSMParameters(
            storage_noise_r=sample.storage_noise_r,
            storage_rate_nu=sample.storage_rate_nu,
            delta_t_ns=sample.delta_t_ns,
            channel_fidelity=sample.channel_fidelity,
            detection_eff_eta=sample.detection_eff_eta,
            detector_error=sample.detector_error,
            dark_count_prob=sample.dark_count_prob,
        )
        
        # Build ReconciliationConfig based on strategy
        if sample.strategy == "baseline":
            recon_config = ReconciliationConfig(
                reconciliation_type=ReconciliationType.BASELINE,
                frame_size=4096,
            )
        else:  # blind
            recon_config = ReconciliationConfig(
                reconciliation_type=ReconciliationType.BLIND,
                frame_size=4096,
                max_blind_rounds=10,
            )
        
        # Build ProtocolParameters with precomputed EPR injection
        params = ProtocolParameters(
            session_id=f"exploration_{sample.sample_id}",
            nsm_params=nsm_params,
            num_pairs=sample.num_pairs,
            num_qubits=10,
            precomputed_epr=epr_data,  # <-- INJECTION
            reconciliation=recon_config,
        )
        
        try:
            # Run full protocol (sifting, reconciliation, amplification)
            # This calls AliceProgram and BobProgram which will consume
            # precomputed_epr and bypass SquidASM EPR socket operations.
            ot_output, raw_results = run_protocol(
                params=params,
                bob_choice_bit=0,
                network_config=None,  # Not needed with precomputed EPR
            )
            
            # Extract metrics
            return ProtocolResult(
                sample_id=sample.sample_id,
                success=ot_output.protocol_succeeded,
                key_length=ot_output.final_key_length,
                entropy_rate=ot_output.entropy_rate,
                qber=raw_results["Alice"]["qber"],
                timing_compliant=raw_results["Alice"]["timing_compliant"],
                net_efficiency=self._compute_efficiency(ot_output, sample),
                error_code=None,
            )
        
        except Exception as e:
            # Catch protocol failures (e.g., QBER too high, LDPC divergence)
            logger.error(f"Protocol failed for sample {sample.sample_id}: {e}")
            return ProtocolResult(
                sample_id=sample.sample_id,
                success=False,
                key_length=0,
                entropy_rate=0.0,
                qber=0.5,
                timing_compliant=False,
                net_efficiency=0.0,
                error_code=str(type(e).__name__),
            )
    
    def _compute_efficiency(
        self,
        ot_output: ObliviousTransferOutput,
        sample: ExplorationSample,
    ) -> float:
        """
        Compute net efficiency: key_length / num_pairs.
        
        This is the primary metric for exploration. It captures:
        - Sifting efficiency (basis matching)
        - Reconciliation overhead (syndrome bits)
        - Privacy amplification compression
        - Security parameter penalties
        """
        return ot_output.final_key_length / sample.num_pairs
```

#### 6.1.3 Parameter Sampling (`caligo.exploration.sampler`)

**New Module Required:**
```python
# caligo/exploration/sampler.py

import numpy as np
from scipy.stats.qmc import LatinHypercube
from typing import List, Optional

class LHSSampler:
    """
    Latin Hypercube Sampling for the 9-dimensional NSM parameter space.
    
    This class generates a space-filling design across:
    1. storage_noise_r ∈ [0.0, 1.0] (linear)
    2. storage_rate_nu ∈ [0.0, 1.0] (log-uniform, avoiding 0)
    3. delta_t_ns ∈ [1e5, 1e9] (log-uniform)
    4. channel_fidelity ∈ [0.5, 1.0] (beta distribution)
    5. detection_eff_eta ∈ [1e-3, 1.0] (log-uniform)
    6. detector_error ∈ [0.0, 0.1] (linear)
    7. dark_count_prob ∈ [1e-8, 1e-3] (log-uniform)
    8. num_pairs ∈ [1e4, 1e6] (log-uniform, rounded)
    9. strategy ∈ {baseline, blind} (categorical)
    """
    
    def __init__(self, seed: Optional[int] = None):
        self._rng = np.random.default_rng(seed)
        # 8 continuous dimensions (strategy handled separately)
        self._lhs = LatinHypercube(d=8, seed=seed)
    
    def sample(self, n_samples: int) -> List[ExplorationSample]:
        """
        Generate n_samples LHS configurations.
        
        Parameters
        ----------
        n_samples : int
            Number of parameter configurations to generate.
        
        Returns
        -------
        List[ExplorationSample]
            Sampled configurations with proper transformations applied.
        """
        # Generate LHS samples in [0, 1]^8
        unit_samples = self._lhs.random(n=n_samples)
        
        samples = []
        for idx, unit in enumerate(unit_samples):
            # Transform unit hypercube to parameter domains
            
            # 1. storage_noise_r: linear [0, 1]
            r = unit[0]
            
            # 2. storage_rate_nu: log-uniform [1e-4, 1.0]
            nu = 10 ** (unit[1] * (-4))  # log10(nu) ∈ [-4, 0]
            
            # 3. delta_t_ns: log-uniform [1e5, 1e9]
            delta_t = 10 ** (unit[2] * 4 + 5)  # log10(Δt) ∈ [5, 9]
            
            # 4. channel_fidelity: Beta(2, 2) shifted to [0.5, 1.0]
            # Beta(2,2) has mode at 0.5, biasing toward high fidelity
            from scipy.stats import beta
            F = 0.5 + 0.5 * beta.ppf(unit[3], 2, 2)
            
            # 5. detection_eff_eta: log-uniform [1e-3, 1.0]
            eta = 10 ** (unit[4] * (-3))  # log10(η) ∈ [-3, 0]
            
            # 6. detector_error: linear [0, 0.1]
            e_det = unit[5] * 0.1
            
            # 7. dark_count_prob: log-uniform [1e-8, 1e-3]
            p_dark = 10 ** (unit[6] * (-5) - 3)  # log10(P_d) ∈ [-8, -3]
            
            # 8. num_pairs: log-uniform [1e4, 1e6], rounded
            N = int(10 ** (unit[7] * 2 + 4))  # log10(N) ∈ [4, 6]
            
            # 9. strategy: categorical (alternate or randomize)
            strategy = "baseline" if idx % 2 == 0 else "blind"
            
            samples.append(ExplorationSample(
                sample_id=idx,
                storage_noise_r=r,
                storage_rate_nu=nu,
                delta_t_ns=delta_t,
                channel_fidelity=F,
                detection_eff_eta=eta,
                detector_error=e_det,
                dark_count_prob=p_dark,
                num_pairs=N,
                strategy=strategy,
            ))
        
        return samples
```

### 6.2 Complete Phase 1 Implementation

**Module:** `caligo/exploration/lhs_executor.py`

```python
import h5py
import dill
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from caligo.exploration.sampler import LHSSampler, ExplorationSample
from caligo.exploration.epr_batcher import BatchedEPROrchestrator
from caligo.exploration.harness import ProtocolHarness, ProtocolResult
from caligo.quantum.parallel import ParallelEPRConfig
from caligo.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Phase1State:
    """Persistent state for Phase 1 LHS sampling."""
    samples_collected: int = 0
    rng_state: Optional[dict] = None
    current_phase: str = "LHS"


class Phase1Executor:
    """
    Executor for Phase 1: Latin Hypercube Sampling warmup.
    
    This class coordinates the LHS sampling loop:
    1. Generate parameter configurations (LHSSampler)
    2. Generate quantum data in parallel (BatchedEPROrchestrator)
    3. Execute classical protocol (ProtocolHarness)
    4. Persist results to HDF5
    5. Checkpoint state for fault tolerance
    """
    
    def __init__(
        self,
        output_path: Path,
        state_path: Path,
        target_samples: int = 2000,
        batch_size: int = 50,
    ):
        """
        Parameters
        ----------
        output_path : Path
            Path to HDF5 file (e.g., exploration_data.h5).
        state_path : Path
            Path to state checkpoint file (e.g., state.pkl).
        target_samples : int
            Total number of LHS samples to collect.
        batch_size : int
            Number of samples to process per batch (memory management).
        """
        self.output_path = output_path
        self.state_path = state_path
        self.target_samples = target_samples
        self.batch_size = batch_size
        
        self._init_hdf5()
        self._load_or_init_state()
    
    def _init_hdf5(self) -> None:
        """Create HDF5 file with schema if it doesn't exist."""
        if not self.output_path.exists():
            with h5py.File(self.output_path, "w") as f:
                grp = f.create_group("lhs_warmup")
                
                # Datasets (initially empty, will be appended)
                grp.create_dataset(
                    "inputs",
                    shape=(0, 9),
                    maxshape=(None, 9),
                    dtype="float32",
                    compression="gzip",
                )
                grp.create_dataset(
                    "net_efficiency",
                    shape=(0,),
                    maxshape=(None,),
                    dtype="float32",
                    compression="gzip",
                )
                grp.create_dataset(
                    "key_length",
                    shape=(0,),
                    maxshape=(None,),
                    dtype="int32",
                    compression="gzip",
                )
                grp.create_dataset(
                    "metadata",
                    shape=(0,),
                    maxshape=(None,),
                    dtype=h5py.string_dtype(),
                )
    
    def _load_or_init_state(self) -> None:
        """Load checkpoint or initialize fresh state."""
        if self.state_path.exists():
            with open(self.state_path, "rb") as f:
                self.state = dill.load(f)
            logger.info(f"Resuming from checkpoint: {self.state.samples_collected} samples")
        else:
            self.state = Phase1State()
    
    def run(self) -> None:
        """Execute Phase 1 LHS sampling loop."""
        sampler = LHSSampler(seed=42)
        
        # Configure parallel EPR generation (16 workers on high-core server)
        epr_config = ParallelEPRConfig(
            enabled=True,
            num_workers=16,
            pairs_per_batch=5000,
            shuffle_results=True,
        )
        epr_orchestrator = BatchedEPROrchestrator(epr_config)
        
        # Configure protocol harness (single-threaded for now; GIL-bound)
        harness = ProtocolHarness(num_workers=1)
        
        while self.state.samples_collected < self.target_samples:
            remaining = self.target_samples - self.state.samples_collected
            batch_n = min(self.batch_size, remaining)
            
            logger.info(f"Generating batch of {batch_n} samples...")
            
            # 1. Sample parameter configurations
            samples = sampler.sample(n_samples=batch_n)
            
            # 2. Generate quantum data (parallel across EPR pairs)
            logger.info("Generating EPR pairs in parallel...")
            epr_data = epr_orchestrator.generate_for_samples(samples)
            
            # 3. Execute classical protocol
            logger.info("Executing protocol harness...")
            results = harness.execute_batch(samples, epr_data)
            
            # 4. Write to HDF5
            self._append_hdf5(samples, results)
            
            # 5. Update checkpoint
            self.state.samples_collected += batch_n
            self._save_state()
            
            logger.info(
                f"Progress: {self.state.samples_collected}/{self.target_samples}"
            )
    
    def _append_hdf5(
        self,
        samples: List[ExplorationSample],
        results: Dict[int, ProtocolResult],
    ) -> None:
        """Append batch results to HDF5."""
        with h5py.File(self.output_path, "a") as f:
            grp = f["lhs_warmup"]
            
            # Convert samples to input matrix
            input_matrix = np.array([
                [
                    s.storage_noise_r,
                    s.storage_rate_nu,
                    s.delta_t_ns,
                    s.channel_fidelity,
                    s.detection_eff_eta,
                    s.detector_error,
                    s.dark_count_prob,
                    s.num_pairs,
                    1.0 if s.strategy == "blind" else 0.0,  # Categorical encoding
                ]
                for s in samples
            ], dtype=np.float32)
            
            net_eff = np.array([results[s.sample_id].net_efficiency for s in samples], dtype=np.float32)
            key_len = np.array([results[s.sample_id].key_length for s in samples], dtype=np.int32)
            metadata = [results[s.sample_id].error_code or "success" for s in samples]
            
            # Resize and append
            old_size = grp["inputs"].shape[0]
            new_size = old_size + len(samples)
            
            grp["inputs"].resize(new_size, axis=0)
            grp["inputs"][old_size:new_size] = input_matrix
            
            grp["net_efficiency"].resize(new_size, axis=0)
            grp["net_efficiency"][old_size:new_size] = net_eff
            
            grp["key_length"].resize(new_size, axis=0)
            grp["key_length"][old_size:new_size] = key_len
            
            grp["metadata"].resize(new_size, axis=0)
            grp["metadata"][old_size:new_size] = metadata
    
    def _save_state(self) -> None:
        """Checkpoint state to disk."""
        with open(self.state_path, "wb") as f:
            dill.dump(self.state, f)
```

### 6.3 Fault Tolerance and Resumption

**Critical Feature:** Phase 1 must support pause/resume for multi-day runs.

**Test Scenario:**
```bash
# Terminal 1: Start exploration (LHS campaign)
$ python -m caligo.exploration.lhs_executor

# Terminal 2: Simulate crash after 500 samples
$ pkill -9 python  # Kill process mid-batch

# Terminal 1: Resume
$ python -m caligo.exploration.lhs_executor
# Output: "Resuming from checkpoint: 500 samples"
# Continue from sample 501...
```

**Implementation Notes:**
1. State checkpointing happens **after** each batch is written to HDF5
2. HDF5 writes are atomic per batch (no partial batch corruption)
3. RNG state is captured in checkpoint for reproducibility

---

## 7. Phase 2: Surrogate Modeling

**Objective:** Train Twin GPs and persist them.

### 7.1 Architecture: Twin Gaussian Processes

We train **two independent GPs** to model the efficiency landscape:
- $GP_{baseline}$: Models baseline reconciliation strategy
- $GP_{blind}$: Models blind reconciliation strategy

**Rationale:** The two strategies exhibit different behavior across the parameter space:
- **Baseline** requires QBER estimation (Phase II overhead) but benefits from accurate rate adaptation
- **Blind** skips QBER estimation but uses conservative LDPC rates

The security cliff location differs between strategies, necessitating separate models.

### 7.2 Model Architecture

**Module:** `caligo/exploration/surrogate.py`

```python
import numpy as np
import h5py
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional
import dill

from caligo.utils.logging import get_logger

logger = get_logger(__name__)


class EfficiencyLandscape:
    """
    Twin Gaussian Process models for net efficiency prediction.
    
    This class maintains two independent GP regressors:
    1. GP_baseline: For baseline reconciliation strategy
    2. GP_blind: For blind reconciliation strategy
    
    The models predict net_efficiency given 8-dimensional continuous
    parameters (strategy is handled by model selection).
    
    Architecture
    ------------
    - Kernel: Matern(nu=2.5) + WhiteKernel
        - Matern provides smoothness with finite differentiability
        - WhiteKernel accounts for protocol stochasticity
    - Normalization: StandardScaler on inputs (Z-score)
    - Targets: No transformation (efficiency ∈ [0, 1] naturally scaled)
    
    References
    ----------
    - Rasmussen & Williams (2006): GP for Machine Learning
    - Forrester et al. (2008): Engineering Design via Surrogate Modelling
    """
    
    def __init__(self):
        self._gp_baseline: Optional[GaussianProcessRegressor] = None
        self._gp_blind: Optional[GaussianProcessRegressor] = None
        self._scaler: Optional[StandardScaler] = None
        self._fitted = False
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        strategies: np.ndarray,
    ) -> None:
        """
        Fit twin GPs on exploration data.
        
        Parameters
        ----------
        X : np.ndarray
            Input features (N x 8): [r, ν, Δt, F, η, e_det, P_dark, N].
        y : np.ndarray
            Net efficiency targets (N,).
        strategies : np.ndarray
            Strategy indicators (N,): 0=baseline, 1=blind.
        
        Notes
        -----
        - Filters failed samples (y=0) before fitting
        - Normalizes inputs using StandardScaler
        - Uses default GP hyperparameters (sklearn optimization)
        """
        # Filter failures
        valid_mask = y > 0
        X_valid = X[valid_mask]
        y_valid = y[valid_mask]
        strategies_valid = strategies[valid_mask]
        
        logger.info(f"Fitting GPs on {len(X_valid)} valid samples ({np.sum(~valid_mask)} failures excluded)")
        
        # Normalize inputs
        self._scaler = StandardScaler()
        X_norm = self._scaler.fit_transform(X_valid)
        
        # Split by strategy
        baseline_mask = strategies_valid == 0
        blind_mask = strategies_valid == 1
        
        X_baseline = X_norm[baseline_mask]
        y_baseline = y_valid[baseline_mask]
        
        X_blind = X_norm[blind_mask]
        y_blind = y_valid[blind_mask]
        
        logger.info(f"Baseline samples: {len(X_baseline)}, Blind samples: {len(X_blind)}")
        
        # Define kernel
        # Length scale ~ 1.0 in normalized space
        kernel = (
            ConstantKernel(1.0, (1e-3, 1e3)) *
            Matern(length_scale=1.0, length_scale_bounds=(0.1, 10.0), nu=2.5) +
            WhiteKernel(noise_level=0.01, noise_level_bounds=(1e-5, 1e-1))
        )
        
        # Fit baseline GP
        self._gp_baseline = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=5,
            alpha=1e-6,
            normalize_y=False,  # We normalize X, not y
        )
        self._gp_baseline.fit(X_baseline, y_baseline)
        logger.info(f"Baseline GP fitted. Kernel: {self._gp_baseline.kernel_}")
        
        # Fit blind GP
        self._gp_blind = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=5,
            alpha=1e-6,
            normalize_y=False,
        )
        self._gp_blind.fit(X_blind, y_blind)
        logger.info(f"Blind GP fitted. Kernel: {self._gp_blind.kernel_}")
        
        self._fitted = True
    
    def predict(
        self,
        X: np.ndarray,
        strategy: str,
        return_std: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict net efficiency for query points.
        
        Parameters
        ----------
        X : np.ndarray
            Query points (M x 8) in original scale.
        strategy : str
            "baseline" or "blind".
        return_std : bool
            If True, return posterior standard deviation.
        
        Returns
        -------
        mean : np.ndarray
            Predicted net efficiency (M,).
        std : Optional[np.ndarray]
            Posterior standard deviation (M,) if return_std=True.
        """
        if not self._fitted:
            raise RuntimeError("Models not fitted. Call fit() first.")
        
        # Normalize inputs
        X_norm = self._scaler.transform(X)
        
        # Select GP
        if strategy == "baseline":
            gp = self._gp_baseline
        elif strategy == "blind":
            gp = self._gp_blind
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # Predict
        if return_std:
            mean, std = gp.predict(X_norm, return_std=True)
            return mean, std
        else:
            mean = gp.predict(X_norm, return_std=False)
            return mean, None
    
    def save(self, path: str) -> None:
        """Serialize models to disk using dill."""
        with open(path, "wb") as f:
            dill.dump({
                "gp_baseline": self._gp_baseline,
                "gp_blind": self._gp_blind,
                "scaler": self._scaler,
                "fitted": self._fitted,
            }, f)
        logger.info(f"Surrogate models saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> "EfficiencyLandscape":
        """Load serialized models from disk."""
        with open(path, "rb") as f:
            data = dill.load(f)
        
        instance = cls()
        instance._gp_baseline = data["gp_baseline"]
        instance._gp_blind = data["gp_blind"]
        instance._scaler = data["scaler"]
        instance._fitted = data["fitted"]
        
        logger.info(f"Surrogate models loaded from {path}")
        return instance
```

### 7.3 Phase 2 Executor

**Module:** `caligo/exploration/surrogate_trainer.py`

```python
import h5py
import numpy as np
from pathlib import Path

from caligo.exploration.surrogate import EfficiencyLandscape
from caligo.utils.logging import get_logger

logger = get_logger(__name__)


class Phase2Executor:
    """
    Executor for Phase 2: Surrogate model training.
    
    This class:
    1. Loads data from HDF5 (lhs_warmup and optionally active_learning)
    2. Preprocesses data (filtering failures, normalizing)
    3. Trains twin GPs
    4. Persists models to disk
    """
    
    def __init__(
        self,
        hdf5_path: Path,
        output_dir: Path,
    ):
        """
        Parameters
        ----------
        hdf5_path : Path
            Path to exploration_data.h5.
        output_dir : Path
            Directory to save surrogate models.
        """
        self.hdf5_path = hdf5_path
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def run(self, epoch: int = 0) -> EfficiencyLandscape:
        """
        Train surrogate models.
        
        Parameters
        ----------
        epoch : int
            Training epoch (for versioning saved models).
        
        Returns
        -------
        EfficiencyLandscape
            Trained surrogate model.
        """
        logger.info("Loading data from HDF5...")
        X, y, strategies = self._load_data()
        
        logger.info(f"Training on {len(X)} samples...")
        landscape = EfficiencyLandscape()
        landscape.fit(X, y, strategies)
        
        # Save model
        model_path = self.output_dir / f"surrogate_v{epoch}.dill"
        landscape.save(str(model_path))
        
        # Log diagnostics
        self._log_diagnostics(landscape, X, y, strategies)
        
        return landscape
    
    def _load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load training data from HDF5.
        
        Returns
        -------
        X : np.ndarray
            Input features (N x 8): [r, ν, Δt, F, η, e_det, P_dark, N].
        y : np.ndarray
            Net efficiency targets (N,).
        strategies : np.ndarray
            Strategy indicators (N,): 0=baseline, 1=blind.
        """
        with h5py.File(self.hdf5_path, "r") as f:
            # Load from lhs_warmup
            inputs = f["lhs_warmup/inputs"][:]  # (N, 9)
            net_eff = f["lhs_warmup/net_efficiency"][:]  # (N,)
            
            # Optionally append active_learning data if it exists
            if "active_learning" in f:
                inputs_active = f["active_learning/inputs"][:]
                net_eff_active = f["active_learning/net_efficiency"][:]
                inputs = np.vstack([inputs, inputs_active])
                net_eff = np.concatenate([net_eff, net_eff_active])
        
        # Split inputs: first 8 columns are continuous, last is strategy
        X = inputs[:, :8].astype(np.float32)
        strategies = inputs[:, 8].astype(np.int32)  # 0=baseline, 1=blind
        y = net_eff.astype(np.float32)
        
        return X, y, strategies
    
    def _log_diagnostics(
        self,
        landscape: EfficiencyLandscape,
        X: np.ndarray,
        y: np.ndarray,
        strategies: np.ndarray,
    ) -> None:
        """Log model diagnostics."""
        # Compute R² on training set
        baseline_mask = strategies == 0
        blind_mask = strategies == 1
        
        y_pred_baseline, _ = landscape.predict(X[baseline_mask], "baseline")
        y_pred_blind, _ = landscape.predict(X[blind_mask], "blind")
        
        from sklearn.metrics import r2_score
        
        r2_baseline = r2_score(y[baseline_mask], y_pred_baseline)
        r2_blind = r2_score(y[blind_mask], y_pred_blind)
        
        logger.info(f"Training R²: Baseline={r2_baseline:.3f}, Blind={r2_blind:.3f}")
        
        # Check for negative predictions (indicates underfitting)
        if np.any(y_pred_baseline < 0) or np.any(y_pred_blind < 0):
            logger.warning("Negative predictions detected. Consider kernel tuning.")
```

### 7.4 Model Validation and Retraining

**Critical Edge Case: Model Divergence**

If the GP starts predicting unrealistic values (e.g., net_efficiency > 1.0 or < 0.0), the model has diverged. This can happen if:
1. Hyperparameter optimization converges to a bad local minimum
2. Training data contains outliers (protocol crashes with corrupted outputs)
3. Kernel choice is inappropriate for the landscape topology

**Recovery Strategy:**
```python
def detect_divergence(
    landscape: EfficiencyLandscape,
    X_test: np.ndarray,
    strategy: str,
) -> bool:
    """
    Check if model predictions are physically plausible.
    
    Returns True if divergence detected.
    """
    y_pred, y_std = landscape.predict(X_test, strategy, return_std=True)
    
    # Check for out-of-bounds predictions
    if np.any(y_pred < -0.1) or np.any(y_pred > 1.1):
        return True
    
    # Check for excessive uncertainty (>50% of parameter space)
    if np.mean(y_std) > 0.5:
        return True
    
    return False


def retrain_with_reset(
    hdf5_path: Path,
    output_dir: Path,
    epoch: int,
) -> EfficiencyLandscape:
    """
    Retrain GPs with fresh hyperparameter initialization.
    
    This function:
    1. Reloads data from HDF5
    2. Filters outliers more aggressively
    3. Uses fixed kernel hyperparameters (no optimization)
    4. Saves retrained model with incremented epoch
    """
    # ... implementation following Phase2Executor.run() 
    # but with fixed kernel hyperparameters
    pass
```

---

## 8. Phase 3: Active Stress Testing (Bayesian Loop)

**Objective:** The autonomous "Red Team" loop.

### 8.1 Acquisition Function: Straddle Strategy

We use a **custom acquisition function** designed to find the security cliff (boundary between success and failure):

$$
A(x) = 1.96\sigma(x) - |E_{\text{net}}(x)|
$$

Where:
- $\sigma(x)$: GP posterior standard deviation (exploration term)
- $E_{\text{net}}(x)$: Predicted net efficiency (exploitation term)
- $1.96$: Targets 95% confidence interval boundary

**Intuition:** This function rewards:
1. **High uncertainty** $\sigma(x)$ → Explore unknown regions
2. **Near-zero efficiency** $|E_{\text{net}}(x)| \approx 0$ → Targets the security cliff

### 8.2 Batch Acquisition: Kriging Believer

To generate batches of $N$ candidates (for parallel execution), we use the **Kriging Believer** strategy:

```python
# caligo/exploration/active.py

import numpy as np
from scipy.optimize import differential_evolution
from typing import List, Tuple

from caligo.exploration.surrogate import EfficiencyLandscape
from caligo.utils.logging import get_logger

logger = get_logger(__name__)


class BayesianOptimizer:
    """
    Bayesian optimization with Straddle acquisition function.
    
    This class implements the active learning loop for finding
    the security cliff in the 9D parameter space.
    
    References
    ----------
    - Ginsbourger et al. (2010): Kriging Believer
    - Gramacy & Lee (2010): Optimization under constraints
    """
    
    def __init__(
        self,
        landscape: EfficiencyLandscape,
        bounds: List[Tuple[float, float]],
    ):
        """
        Parameters
        ----------
        landscape : EfficiencyLandscape
            Trained surrogate models.
        bounds : List[Tuple[float, float]]
            Parameter bounds for 8 continuous dimensions.
            Example: [(0, 1), (1e-4, 1), (1e5, 1e9), ...]
        """
        self.landscape = landscape
        self.bounds = bounds
    
    def _straddle_acquisition(
        self,
        x: np.ndarray,
        strategy: str,
    ) -> float:
        """
        Straddle acquisition function.
        
        Parameters
        ----------
        x : np.ndarray
            Single query point (8,).
        strategy : str
            "baseline" or "blind".
        
        Returns
        -------
        float
            Acquisition value (maximize to find next query point).
        """
        # Reshape for GP prediction
        X = x.reshape(1, -1)
        
        # Get mean and std
        mean, std = self.landscape.predict(X, strategy, return_std=True)
        
        # Straddle: reward uncertainty and near-zero efficiency
        acquisition = 1.96 * std[0] - np.abs(mean[0])
        
        return acquisition
    
    def suggest_batch(
        self,
        n_workers: int,
        strategy: str = "baseline",
    ) -> np.ndarray:
        """
        Suggest batch of n_workers query points using Kriging Believer.
        
        Parameters
        ----------
        n_workers : int
            Number of parallel workers (batch size).
        strategy : str
            Strategy to optimize for.
        
        Returns
        -------
        np.ndarray
            Batch of query points (n_workers x 8).
        
        Algorithm
        ---------
        1. Optimize acquisition function to find x1*
        2. Predict y1* = E[f(x1*)] (Kriging Believer: treat prediction as truth)
        3. Augment training set with (x1*, y1*)
        4. Repeat for x2*, ..., xN*
        """
        batch = []
        
        # Temporarily store "believed" points
        believed_X = []
        believed_y = []
        
        for i in range(n_workers):
            logger.debug(f"Suggesting point {i+1}/{n_workers}...")
            
            # Optimize acquisition function
            def neg_acq(x):
                return -self._straddle_acquisition(x, strategy)
            
            result = differential_evolution(
                func=neg_acq,
                bounds=self.bounds,
                maxiter=100,
                seed=i,
                workers=1,
            )
            
            x_next = result.x
            batch.append(x_next)
            
            # Kriging Believer: predict outcome and augment model
            y_pred, _ = self.landscape.predict(
                x_next.reshape(1, -1),
                strategy,
                return_std=False,
            )
            
            believed_X.append(x_next)
            believed_y.append(y_pred[0])
            
            # Temporarily augment GP (refit with believed points)
            # This is computationally expensive; alternative: use GP conditioning
            # For now, we skip refitting within the batch (approximation)
        
        return np.array(batch)
```

### 8.3 Main Loop Implementation

**Module:** `caligo/exploration/active_executor.py`

```python
import h5py
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

from caligo.exploration.active import BayesianOptimizer
from caligo.exploration.surrogate import EfficiencyLandscape
from caligo.exploration.epr_batcher import BatchedEPROrchestrator
from caligo.exploration.harness import ProtocolHarness
from caligo.exploration.sampler import ExplorationSample
from caligo.quantum.parallel import ParallelEPRConfig
from caligo.utils.logging import get_logger

logger = get_logger(__name__)


class Phase3Executor:
    """
    Executor for Phase 3: Active stress testing loop.
    
    This is the autonomous "Red Team" that iteratively finds
    adversarial parameter configurations at the security cliff.
    """
    
    def __init__(
        self,
        hdf5_path: Path,
        surrogate_path: Path,
        state_path: Path,
        max_iterations: int = 100,
        batch_size: int = 16,
    ):
        """
        Parameters
        ----------
        hdf5_path : Path
            Path to exploration_data.h5.
        surrogate_path : Path
            Path to trained surrogate model (.dill).
        state_path : Path
            Path to state checkpoint.
        max_iterations : int
            Maximum number of active learning iterations.
        batch_size : int
            Number of parallel evaluations per iteration.
        """
        self.hdf5_path = hdf5_path
        self.surrogate_path = surrogate_path
        self.state_path = state_path
        self.max_iterations = max_iterations
        self.batch_size = batch_size
        
        # Load surrogate
        self.landscape = EfficiencyLandscape.load(str(surrogate_path))
        
        # Initialize optimizer
        bounds = [
            (0.0, 1.0),      # storage_noise_r
            (1e-4, 1.0),     # storage_rate_nu
            (1e5, 1e9),      # delta_t_ns
            (0.5, 1.0),      # channel_fidelity
            (1e-3, 1.0),     # detection_eff_eta
            (0.0, 0.1),      # detector_error
            (1e-8, 1e-3),    # dark_count_prob
            (1e4, 1e6),      # num_pairs
        ]
        self.optimizer = BayesianOptimizer(self.landscape, bounds)
        
        # Execution engines
        epr_config = ParallelEPRConfig(
            enabled=True,
            num_workers=16,
            pairs_per_batch=5000,
        )
        self.epr_orchestrator = BatchedEPROrchestrator(epr_config)
        self.harness = ProtocolHarness(num_workers=1)
    
    def run(self) -> None:
        """Execute active learning loop."""
        for iteration in range(self.max_iterations):
            logger.info(f"=== Active Learning Iteration {iteration+1}/{self.max_iterations} ===")
            
            # Check stop signal (external file flag for graceful shutdown)
            if self._check_stop_signal():
                logger.info("Stop signal detected. Saving state and exiting.")
                break
            
            # 1. Suggest parameter batch
            logger.info("Suggesting batch via Bayesian optimization...")
            X_batch = self.optimizer.suggest_batch(
                n_workers=self.batch_size,
                strategy="baseline",  # Alternate or optimize both
            )
            
            # Convert to ExplorationSamples
            samples = []
            for i, x in enumerate(X_batch):
                samples.append(ExplorationSample(
                    sample_id=iteration * self.batch_size + i,
                    storage_noise_r=x[0],
                    storage_rate_nu=x[1],
                    delta_t_ns=x[2],
                    channel_fidelity=x[3],
                    detection_eff_eta=x[4],
                    detector_error=x[5],
                    dark_count_prob=x[6],
                    num_pairs=int(x[7]),
                    strategy="baseline",
                ))
            
            # 2. Generate quantum data
            logger.info("Generating EPR pairs...")
            epr_data = self.epr_orchestrator.generate_for_samples(samples)
            
            # 3. Execute protocol
            logger.info("Executing protocol...")
            results = self.harness.execute_batch(samples, epr_data)
            
            # 4. Persist results
            self._append_hdf5(samples, results)
            
            # 5. Update surrogate model
            logger.info("Updating surrogate model...")
            self._update_surrogate(samples, results)
            
            logger.info(f"Iteration {iteration+1} complete. Checkpoint saved.")
    
    def _append_hdf5(self, samples, results) -> None:
        """Append results to active_learning group in HDF5."""
        with h5py.File(self.hdf5_path, "a") as f:
            if "active_learning" not in f:
                # Create group on first iteration
                grp = f.create_group("active_learning")
                grp.create_dataset("inputs", shape=(0, 9), maxshape=(None, 9), dtype="float32")
                grp.create_dataset("net_efficiency", shape=(0,), maxshape=(None,), dtype="float32")
                grp.create_dataset("key_length", shape=(0,), maxshape=(None,), dtype="int32")
                grp.create_dataset("metadata", shape=(0,), maxshape=(None,), dtype=h5py.string_dtype())
            
            grp = f["active_learning"]
            
            # Build arrays
            input_matrix = np.array([
                [s.storage_noise_r, s.storage_rate_nu, s.delta_t_ns,
                 s.channel_fidelity, s.detection_eff_eta, s.detector_error,
                 s.dark_count_prob, s.num_pairs,
                 1.0 if s.strategy == "blind" else 0.0]
                for s in samples
            ], dtype=np.float32)
            
            net_eff = np.array([results[s.sample_id].net_efficiency for s in samples], dtype=np.float32)
            key_len = np.array([results[s.sample_id].key_length for s in samples], dtype=np.int32)
            metadata = [results[s.sample_id].error_code or "success" for s in samples]
            
            # Append
            old_size = grp["inputs"].shape[0]
            new_size = old_size + len(samples)
            
            grp["inputs"].resize(new_size, axis=0)
            grp["inputs"][old_size:new_size] = input_matrix
            
            grp["net_efficiency"].resize(new_size, axis=0)
            grp["net_efficiency"][old_size:new_size] = net_eff
            
            grp["key_length"].resize(new_size, axis=0)
            grp["key_length"][old_size:new_size] = key_len
            
            grp["metadata"].resize(new_size, axis=0)
            grp["metadata"][old_size:new_size] = metadata
    
    def _update_surrogate(self, samples, results) -> None:
        """
        Incrementally update surrogate models with new data.
        
        Options:
        1. Online GP update (conditioning)
        2. Periodic retraining from scratch (every 10 iterations)
        3. Hybrid: online for small batches, retrain for large updates
        """
        # For simplicity: retrain from scratch every 10 iterations
        # TODO: Implement efficient online GP conditioning
        pass
    
    def _check_stop_signal(self) -> bool:
        """Check for external stop signal (e.g., .stop file in working directory)."""
        stop_file = Path.cwd() / ".stop_exploration"
        return stop_file.exists()
```

### 8.4 Performance Optimization: Protocol Parallelization

**Current Bottleneck:** The `ProtocolHarness` executes protocols sequentially due to Python's GIL.

**Solution:** Use `ProcessPoolExecutor` for classical protocol execution:

```python
# caligo/exploration/harness.py (updated)

class ProtocolHarness:
    def __init__(self, num_workers: int = 1):
        self._num_workers = num_workers
        self._executor = ProcessPoolExecutor(max_workers=num_workers) if num_workers > 1 else None
    
    def execute_batch(self, samples, epr_data):
        if self._executor is None:
            # Sequential execution
            return {s.sample_id: self._execute_single(s, epr_data[s.sample_id]) for s in samples}
        else:
            # Parallel execution
            futures = {
                self._executor.submit(self._execute_single, s, epr_data[s.sample_id]): s.sample_id
                for s in samples
            }
            results = {}
            for future in as_completed(futures):
                sample_id = futures[future]
                results[sample_id] = future.result()
            return results
```

**Expected Speedup:** On a 16-core server:
- Quantum generation: 16x (already parallelized)
- Classical protocol: 8-12x (LDPC decoding is CPU-bound, embarrassingly parallel)

**Total wall-clock time estimate:** 100,000 protocol runs in ~48 hours.

---

## 9. Implementation Roadmap

### Step 1: Infrastructure & Harness (Foundation)

**Milestone: Precomputed EPR Injection Working End-to-End**

**Tasks:**
1. **Create exploration package structure:**
   ```bash
   mkdir -p caligo/exploration
   touch caligo/exploration/{__init__.py,sampler.py,epr_batcher.py,harness.py,persistence.py,lhs_executor.py,surrogate_trainer.py,active_executor.py}
   ```

2. **Implement `ExplorationSample` and `ProtocolResult` dataclasses** ([caligo/exploration/sampler.py](caligo/caligo/exploration/sampler.py)):
   - Document with Numpydoc style
   - Add invariant validation (`__post_init__`)
   - Write unit tests for edge cases

3. **Implement `ProtocolHarness._execute_single()`** ([caligo/exploration/harness.py](caligo/caligo/exploration/harness.py)):
   - Integrate with existing [ProtocolParameters](caligo/caligo/protocol/base.py#L69-L101)
   - Use [PrecomputedEPRData](caligo/caligo/protocol/base.py#L42-L66) injection hook
   - Test with [run_protocol()](caligo/caligo/protocol/orchestrator.py#L19-L142)
   
   **Verification Test:**
   ```python
   # tests/exploration/test_harness.py
   def test_precomputed_injection():
       """Verify protocol runs with injected EPR data, bypassing SquidASM."""
       # Generate synthetic EPR data
       epr_data = PrecomputedEPRData(
           alice_outcomes=[0, 1, 0, 1] * 1000,
           alice_bases=[0, 0, 1, 1] * 1000,
           bob_outcomes=[0, 1, 0, 1] * 1000,
           bob_bases=[0, 0, 1, 1] * 1000,
       )
       
       sample = ExplorationSample(
           sample_id=0,
           storage_noise_r=0.9,
           storage_rate_nu=0.002,
           delta_t_ns=1e6,
           channel_fidelity=0.95,
           detection_eff_eta=0.9,
           detector_error=0.01,
           dark_count_prob=1e-6,
           num_pairs=4000,
           strategy="baseline",
       )
       
       harness = ProtocolHarness(num_workers=1)
       result = harness._execute_single(sample, epr_data)
       
       assert result.success is True
       assert result.key_length > 0
       assert 0.0 <= result.net_efficiency <= 1.0
   ```

4. **Implement `BatchedEPROrchestrator`** ([caligo/exploration/epr_batcher.py](caligo/caligo/exploration/epr_batcher.py)):
   - Wrapper around existing [ParallelEPROrchestrator](caligo/caligo/quantum/parallel.py#L245-L350)
   - Convert `ExplorationSample` → `NetworkConfig` dict
   - Map results to `PrecomputedEPRData`
   
   **Integration Point:** The orchestrator must serialize NSM parameters for worker processes:
   ```python
   def _build_network_config(self, sample: ExplorationSample) -> Dict[str, Any]:
       """Convert exploration sample to worker-compatible config."""
       return {
           "fidelity": sample.channel_fidelity,
           "noise": 1.0 - sample.channel_fidelity,
           "detection_eff": sample.detection_eff_eta,
           "dark_count_prob": sample.dark_count_prob,
           # Worker processes use _worker_generate_epr() which accepts these keys
       }
   ```

5. **Build HDF5 schema and StateManager** ([caligo/exploration/persistence.py](caligo/caligo/exploration/persistence.py)):
   - Implement `StateManager` with `dill` serialization
   - Create HDF5 schema in `_init_hdf5()`
   - Write pause/resume integration test:
     ```bash
     # Kill process after 50 samples, resume from checkpoint
     pytest tests/exploration/test_persistence.py::test_pause_resume
     ```

**Deliverable:** End-to-end test executing 100 protocol runs with precomputed EPR data, writing to HDF5, and resuming from checkpoint.

---

### Step 2: Phase 1 (LHS) & Visualization

**Milestone: 2,000 LHS Samples Collected**

**Tasks:**
1. **Implement `LHSSampler.sample()`** ([caligo/exploration/sampler.py](caligo/caligo/exploration/sampler.py)):
   - Use `scipy.stats.qmc.LatinHypercube` for 8D continuous space
   - Apply transformations (log-uniform, beta distribution) per parameter
   - Handle categorical strategy via alternation or stratification
   
   **Validation:**
   ```python
   def test_lhs_coverage():
       """Verify LHS samples cover parameter space uniformly."""
       sampler = LHSSampler(seed=42)
       samples = sampler.sample(n_samples=1000)
       
       # Check marginal distributions
       r_values = [s.storage_noise_r for s in samples]
       assert 0.0 <= min(r_values) < 0.1  # Lower bound coverage
       assert 0.9 < max(r_values) <= 1.0  # Upper bound coverage
       
       # Check log-uniform spacing for delta_t_ns
       delta_t_log = np.log10([s.delta_t_ns for s in samples])
       hist, _ = np.histogram(delta_t_log, bins=10)
       assert np.std(hist) < 50  # Roughly uniform in log-space
   ```

2. **Implement `Phase1Executor.run()`** ([caligo/exploration/lhs_executor.py](caligo/caligo/exploration/lhs_executor.py)):
   - Integrate `LHSSampler`, `BatchedEPROrchestrator`, `ProtocolHarness`
   - Add batch processing loop with checkpointing
   - Handle protocol failures gracefully (log error, set net_efficiency=0)

3. **Run Production LHS Campaign:**
   ```bash
   # On high-core server (16+ cores)
   python -m caligo.exploration.lhs_executor \
       --output exploration_data.h5 \
       --target-samples 2000 \
       --batch-size 50 \
       --num-epr-workers 16
   ```
   
   **Expected Runtime:** ~12 hours (2000 samples × 5 min/sample ÷ 16 workers)

4. **Build Dashboard Script** ([scripts/visualize_exploration.py](caligo/caligo/scripts/visualize_exploration.py)):
   - Load HDF5 data
   - Generate 2D projections of efficiency landscape
   - Identify parameter correlations
   
   **Key Visualizations:**
   - **QBER vs. Net Efficiency:** Verify security cliff at Q ≈ 11%
   - **N (num_pairs) vs. Key Length:** Validate finite-key effects
   - **Strategy Comparison:** Baseline vs. Blind efficiency distributions

**Deliverable:** HDF5 file with 2,000 LHS samples + dashboard plots confirming physical validity.

---

### Step 3: Phase 2 & 3 (The Brain)

**Milestone: Active Learning Loop Operational**

**Tasks:**
1. **Implement `EfficiencyLandscape`** ([caligo/exploration/surrogate.py](caligo/caligo/exploration/surrogate.py)):
   - Twin GP architecture (baseline + blind)
   - Matern kernel with WhiteKernel for noise
   - Implement `fit()`, `predict()`, `save()`, `load()`
   
   **Validation:**
   ```python
   def test_gp_interpolation():
       """GP should interpolate training data with low error."""
       # Create synthetic data
       X_train = np.random.rand(100, 8)
       y_train = np.sin(X_train[:, 0]) * np.cos(X_train[:, 1])  # Smooth function
       
       landscape = EfficiencyLandscape()
       landscape.fit(X_train, y_train, strategies=np.zeros(100))
       
       # Check interpolation error
       y_pred, _ = landscape.predict(X_train, "baseline")
       rmse = np.sqrt(np.mean((y_train - y_pred)**2))
       assert rmse < 0.05, "GP should interpolate training data accurately"
   ```

2. **Implement `Phase2Executor.run()`** ([caligo/exploration/surrogate_trainer.py](caligo/caligo/exploration/surrogate_trainer.py)):
   - Load LHS data from HDF5
   - Train twin GPs
   - Log R² and diagnostics
   - Save surrogate models

3. **Implement `BayesianOptimizer.suggest_batch()`** ([caligo/exploration/active.py](caligo/caligo/exploration/active.py)):
   - Straddle acquisition function
   - Kriging Believer batching
   - Use `scipy.optimize.differential_evolution` for acquisition maximization
   
   **Validation:**
   ```python
   def test_acquisition_targets_cliff():
       """Acquisition function should target security cliff region."""
       # Create landscape with known cliff at QBER=0.11 (F≈0.78)
       # ... setup GP with synthetic data ...
       
       optimizer = BayesianOptimizer(landscape, bounds)
       batch = optimizer.suggest_batch(n_workers=10, strategy="baseline")
       
       # Check that suggested points cluster near cliff
       fidelities = batch[:, 3]  # channel_fidelity column
       assert 0.75 < np.mean(fidelities) < 0.85, "Should target QBER cliff"
   ```

4. **Implement `Phase3Executor.run()`** ([caligo/exploration/active_executor.py](caligo/caligo/exploration/active_executor.py)):
   - Active learning loop
   - Append results to `active_learning` group in HDF5
   - Model update strategy (periodic retraining)

5. **Integration Test: 50-Iteration Loop**
   ```bash
   # Train surrogate on LHS data
   python -m caligo.exploration.surrogate_trainer \
       --hdf5 exploration_data.h5 \
       --output-dir models/
   
   # Run active learning
   python -m caligo.exploration.active_executor \
       --hdf5 exploration_data.h5 \
       --surrogate models/surrogate_v0.dill \
       --max-iterations 50 \
       --batch-size 16
   ```
   
   **Verification:**
   - Monitor acquisition value over iterations (should decrease as cliff is mapped)
   - Check that active samples have lower mean efficiency than LHS samples
   - Visualize query point distribution (should cluster near cliff)

**Deliverable:** Functional active learning loop executing 50 iterations, targeting security cliff.

---

### Step 4: Production Run

**Milestone: 100,000 Protocol Executions**

**Tasks:**
1. **Deploy to High-Core Server:**
   - Provision machine with 32+ cores, 64+ GB RAM
   - Install dependencies (`squidasm`, `h5py`, `scikit-learn`, `dill`)
   - Transfer codebase and LHS checkpoint

2. **Run Extended Active Learning Campaign:**
   ```bash
   nohup python -m caligo.exploration.active_executor \
       --hdf5 exploration_data.h5 \
       --surrogate models/surrogate_v0.dill \
       --max-iterations 5000 \
       --batch-size 32 \
       > active_executor.log 2>&1 &
   ```
   
   **Expected Runtime:** 48-72 hours continuous execution

3. **Monitor Progress:**
   - Check HDF5 file size growth (`watch -n 60 ls -lh exploration_data.h5`)
   - Tail logs for errors (`tail -f active_executor.log`)
   - Generate interim dashboard plots

4. **Final Analysis:**
   - Load full dataset (LHS + active learning)
   - Build lookup tables for $N_{\text{crit}}(\text{QBER}, \text{strategy})$
   - Generate "Safety Manual" report
   - Identify blind vs. baseline crossover points

**Deliverable:** Complete parameter exploration dataset (100K+ samples) + Safety Manual document.

---

## 10. Error Handling & Edge Cases

### 10.1 Protocol Failures (Expected)

**Failure Modes:**
1. **QBER too high (>11%):** Reconciliation aborts due to entropy depletion
2. **LDPC decoder divergence:** Maximum iterations exceeded, verification fails
3. **Zero sifted key:** No basis matches (catastrophic channel noise)
4. **Timing violation:** Bob opens commitment before Δt expires (NSM security violated)

**Handling Strategy:**
```python
# In ProtocolHarness._execute_single()
try:
    ot_output, raw_results = run_protocol(params, bob_choice_bit=0)
    # ... extract metrics ...
except EntropyDepletedError as e:
    logger.warning(f"Sample {sample.sample_id}: Entropy depleted (QBER too high)")
    return ProtocolResult(
        sample_id=sample.sample_id,
        success=False,
        key_length=0,
        net_efficiency=0.0,
        error_code="EntropyDepleted",
    )
except SecurityError as e:
    logger.warning(f"Sample {sample.sample_id}: Security violation ({e})")
    return ProtocolResult(
        sample_id=sample.sample_id,
        success=False,
        key_length=0,
        net_efficiency=0.0,
        error_code="SecurityViolation",
    )
except Exception as e:
    logger.error(f"Sample {sample.sample_id}: Unexpected error: {e}", exc_info=True)
    return ProtocolResult(
        sample_id=sample.sample_id,
        success=False,
        key_length=0,
        net_efficiency=0.0,
        error_code=type(e).__name__,
    )
```

**Impact on Surrogate Modeling:**
- Failed samples (net_efficiency=0) are **excluded** during GP training
- Failure rate is a secondary metric (tracked separately)
- If failure rate > 50%, adjust parameter bounds or sampling strategy

### 10.2 Memory Management

**Problem:** Each worker process maintains NetSquid state. With 16 workers × 10K pairs, memory usage can exceed 32 GB.

**Solution: Worker Process Recycling**
```python
# In Phase1Executor.__init__()
epr_config = ParallelEPRConfig(
    enabled=True,
    num_workers=16,
    pairs_per_batch=5000,
    max_tasks_per_child=10,  # <-- Recycle workers every 10 tasks
)
```

**Monitoring:**
```bash
# Check memory usage during run
watch -n 5 'ps aux | grep python | awk "{sum+=\$6} END {print sum/1024\" MB\"}"'
```

### 10.3 Model Divergence Detection

**Symptom:** GP predictions become unrealistic (negative efficiency, >100% efficiency).

**Detection Mechanism:**
```python
# In Phase3Executor._update_surrogate()
def validate_model(landscape: EfficiencyLandscape) -> bool:
    """Check model sanity on held-out test set."""
    # Use 10% of LHS data as validation set
    X_val, y_val, strat_val = load_validation_set()
    
    for strategy in ["baseline", "blind"]:
        mask = (strat_val == (1 if strategy == "blind" else 0))
        y_pred, y_std = landscape.predict(X_val[mask], strategy, return_std=True)
        
        # Check bounds
        if np.any(y_pred < -0.1) or np.any(y_pred > 1.1):
            logger.error(f"Model divergence: {strategy} GP predicts out-of-bounds values")
            return False
        
        # Check uncertainty
        if np.mean(y_std) > 0.5:
            logger.warning(f"High uncertainty in {strategy} GP (mean σ={np.mean(y_std):.3f})")
    
    return True
```

**Recovery:** If divergence detected, retrain with:
1. Fixed kernel hyperparameters (no optimization)
2. More aggressive outlier filtering
3. Reduced parameter space (focus on high-density regions)

### 10.4 Simulation Crashes (NetSquid/SquidASM)

**Cause:** Rare edge cases in quantum simulator (e.g., divide-by-zero in noise model).

**Handling:**
```python
# In BatchedEPROrchestrator.generate_for_samples()
for sample in samples:
    try:
        orchestrator = ParallelEPROrchestrator(config, network_config)
        alice_out, alice_bases, bob_out, bob_bases = \
            orchestrator.generate_parallel(sample.num_pairs)
        results[sample.sample_id] = PrecomputedEPRData(...)
    except SimulationError as e:
        logger.error(f"Simulation failed for sample {sample.sample_id}: {e}")
        # Generate dummy data (all zeros) to allow protocol to fail gracefully
        results[sample.sample_id] = PrecomputedEPRData(
            alice_outcomes=[0] * sample.num_pairs,
            alice_bases=[0] * sample.num_pairs,
            bob_outcomes=[1] * sample.num_pairs,  # Mismatched outcomes
            bob_bases=[0] * sample.num_pairs,
        )
    finally:
        orchestrator.shutdown()
```

**Log Analysis:** After production run, grep for `SimulationError` to identify problematic parameter regions.

---

## 11. Integration with Existing Caligo Architecture

### 11.1 Module Dependencies

```
caligo.exploration (NEW)
├── sampler.py       → scipy.stats.qmc
├── epr_batcher.py   → caligo.quantum.parallel
├── harness.py       → caligo.protocol.orchestrator
├── persistence.py   → h5py, dill
├── surrogate.py     → sklearn.gaussian_process
├── active.py        → scipy.optimize
├── phase1.py        → ALL OF THE ABOVE
├── surrogate_trainer.py        → persistence + surrogate
└── active_executor.py        → ALL OF THE ABOVE

EXISTING MODULES MODIFIED:
- caligo.protocol.base (already has PrecomputedEPRData hook) ✓
- caligo.protocol.alice (already uses precomputed_epr) ✓
- caligo.protocol.bob (already uses precomputed_epr) ✓
- caligo.quantum.parallel (already production-ready) ✓
```

**Key Insight:** The exploration framework requires **zero modifications** to existing protocol code. It leverages the precomputed EPR injection hook that was designed for test acceleration.

### 11.2 Configuration Management

**Approach:** YAML configuration files for exploration campaigns.

```yaml
# configs/exploration/lhs_campaign.yaml
campaign:
  name: "lhs_warmup_2k"
  output_hdf5: "exploration_data.h5"
  state_checkpoint: "state.pkl"

phase1:
  target_samples: 2000
  batch_size: 50
  sampler:
    seed: 42

epr_parallel:
  enabled: true
  num_workers: 16
  pairs_per_batch: 5000
  shuffle_results: true

harness:
  num_workers: 1  # Sequential for Phase 1 (GIL bottleneck)

parameters:
  storage_noise_r: [0.0, 1.0]
  storage_rate_nu: [1e-4, 1.0]
  delta_t_ns: [1e5, 1e9]
  channel_fidelity: [0.5, 1.0]
  detection_eff_eta: [1e-3, 1.0]
  detector_error: [0.0, 0.1]
  dark_count_prob: [1e-8, 1e-3]
  num_pairs: [1e4, 1e6]
  strategies: ["baseline", "blind"]
```

**Usage:**
```bash
python -m caligo.exploration.phase1 --config configs/exploration/lhs_campaign.yaml
```

### 11.3 Testing Strategy

**Unit Tests:**
- `tests/exploration/test_sampler.py`: LHS coverage, transformations
- `tests/exploration/test_harness.py`: Precomputed injection, error handling
- `tests/exploration/test_surrogate.py`: GP fitting, predictions, serialization

**Integration Tests:**
- `tests/exploration/test_phase1_integration.py`: Full LHS loop (10 samples)
- `tests/exploration/test_active_integration.py`: Active learning loop (5 iterations)

**Smoke Test:**
```bash
# Quick validation on CI/CD
pytest tests/exploration/ -m "not slow"
```