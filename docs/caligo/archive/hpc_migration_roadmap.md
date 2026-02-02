# HPC Migration Roadmap: Caligo on AMD Genoa

## Executive Summary

This document outlines the technical strategy for migrating the Caligo quantum network simulation framework from a GPU-accelerated workstation environment to a CPU-only HPC node with the following specifications:

| Resource | Specification |
|----------|---------------|
| **CPU** | 2× AMD EPYC 9654 (Genoa) — 192 cores total @ 2.4GHz |
| **Memory** | 384 GiB DDR5-4800 (2 GiB/core) |
| **Storage** | 6.4TB NVMe local scratch |
| **Network** | 200Gbps NDR InfiniBand |
| **GPU** | ❌ **None** |

The migration requires addressing three critical challenges:

1. **Surrogate Model Replatforming**: The `gpu_surrogate.py` module relies on PyTorch CUDA. We must migrate to a CPU-optimized backend without catastrophic performance loss.
2. **Massive Parallelism**: 192 cores demand careful GIL avoidance and worker management.
3. **Memory Discipline**: 2GB/core is tight for quantum simulations with LDPC matrices.

---

## 1. System Architecture Analysis

### 1.1 Current Architecture (Workstation)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Current Caligo Architecture                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐     ┌──────────────────────────────────────────────┐  │
│  │  main_explor.py │────▶│  Phase Executors (lhs_executor, active_...)  │  │
│  └─────────────────┘     └──────────────────────────────────────────────┘  │
│                                        │                                    │
│                                        ▼                                    │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                    ProcessPoolExecutor (N workers)                   │  │
│  │    ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐              │  │
│  │    │ Worker 0 │ │ Worker 1 │ │ Worker 2 │ │ Worker N │  ...         │  │
│  │    │ SquidASM │ │ SquidASM │ │ SquidASM │ │ SquidASM │              │  │
│  │    └──────────┘ └──────────┘ └──────────┘ └──────────┘              │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                        │                                    │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │                  SharedMemoryArena (Zero-Copy IPC)                 │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                        │                                    │
│                                        ▼                                    │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                      GPyTorchLandscape (GPU)                         │  │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐   │  │
│  │  │  ExactGPModel   │  │  ExactGPModel   │  │  BoTorch Optimizer  │   │  │
│  │  │  (Baseline GP)  │  │   (Blind GP)    │  │    (CUDA Accel)     │   │  │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────────┘   │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                 ▲                                           │
│                                 │                                           │
│                      ┌──────────┴──────────┐                                │
│                      │   NVIDIA GPU (CUDA) │                                │
│                      └─────────────────────┘                                │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Gap Analysis

| Component | Current State | Gap | Impact |
|-----------|---------------|-----|--------|
| **gpu_surrogate.py** | GPyTorch + CUDA, Float32, CG solver | No CUDA on target | 🔴 Critical — Surrogate training/inference broken |
| **ProcessPoolExecutor** | ~16 workers | 192 cores available | 🟡 Suboptimal — ~12× underutilization |
| **SharedMemoryArena** | Designed for ~8-16 slots | 192 slots × 1M pairs = potential fragmentation | 🟡 Needs tuning |
| **LDPC Decoder** | Pure NumPy, no SIMD hints | AVX-512 available | 🟡 Missing 2-4× speedup |
| **HDF5 I/O** | Network filesystem | Local NVMe available | 🟡 Missing bandwidth |
| **Memory** | ~16GB/worker assumed | 2GB/core limit | 🔴 Risk of OOM |
| **NumPy/BLAS** | MKL or OpenBLAS (unoptimized) | AMD Zen4 micro-arch | 🟡 Suboptimal BLAS |

### 1.3 Target Architecture (Genoa HPC)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        Target Caligo HPC Architecture                           │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────┐     ┌────────────────────────────────────────────────┐    │
│  │  main_explor.py │────▶│  Phase Executors (Hierarchical Parallelism)    │    │
│  │  (SLURM Job)    │     │  - Outer: Sample batches (MPI-ready)           │    │
│  └─────────────────┘     │  - Inner: Per-sample workers (ProcessPool)      │    │
│                          └────────────────────────────────────────────────┘    │
│                                         │                                       │
│                                         ▼                                       │
│  ┌────────────────────────────────────────────────────────────────────────┐    │
│  │          ProcessPoolExecutor (N=180 workers, maxtasksperchild=50)      │    │
│  │  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐       │    │
│  │  │ W0  │ │ W1  │ │ W2  │ │ W3  │ │ ... │ │W178 │ │W179 │ │ +12 │ spare │    │
│  │  └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘       │    │
│  └────────────────────────────────────────────────────────────────────────┘    │
│                                         │                                       │
│  ┌────────────────────────────────────────────────────────────────────────┐    │
│  │               SharedMemoryArena (Tiered: 64 hot + overflow)            │    │
│  └────────────────────────────────────────────────────────────────────────┘    │
│                                         │                                       │
│                                         ▼                                       │
│  ┌────────────────────────────────────────────────────────────────────────┐    │
│  │                  CPUOptimizedLandscape (CPU Backend)                   │    │
│  │  ┌───────────────────┐  ┌───────────────────┐  ┌──────────────────┐   │    │
│  │  │ GPyTorch (CPU)    │  │ GPyTorch (CPU)    │  │  LBFGS Optimizer │   │    │
│  │  │ + KeOps CPU       │  │ + KeOps CPU       │  │  (scipy.optimize)│   │    │
│  │  │ + Lanczos Solver  │  │ + Lanczos Solver  │  │                  │   │    │
│  │  └───────────────────┘  └───────────────────┘  └──────────────────┘   │    │
│  └────────────────────────────────────────────────────────────────────────┘    │
│                                         │                                       │
│                                         ▼                                       │
│  ┌────────────────────────────────────────────────────────────────────────┐    │
│  │                         AMD EPYC 9654 (Genoa)                          │    │
│  │  ┌───────────────────────────────────────────────────────────────────┐ │    │
│  │  │           192 Cores × AVX-512 + OpenBLAS (AOCL-tuned)             │ │    │
│  │  └───────────────────────────────────────────────────────────────────┘ │    │
│  └────────────────────────────────────────────────────────────────────────┘    │
│                                                                                 │
│  ┌────────────────────────────────────────────────────────────────────────┐    │
│  │                    /scratch-node (6.4TB NVMe)                          │    │
│  │   - exploration_results/                                               │    │
│  │   - checkpoint files                                                    │    │
│  │   - HDF5 data store                                                    │    │
│  └────────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Critical Refactoring Paths

### 2.1 Path A: CPU-Optimized Surrogate Model

#### 2.1.1 Problem Statement

The current `gpu_surrogate.py` ([source](caligo/exploration/gpu_surrogate.py)) is tightly coupled to CUDA:

```python
# gpu_surrogate.py:88
device: str = "cuda" if torch.cuda.is_available() else "cpu"
```

While there's a CPU fallback, the code is optimized for GPU memory patterns:
- Float32 quantization (VRAM efficiency, not CPU-optimal)
- CG solver tuned for GPU parallelism
- KeOps symbolic kernels (designed for GPU memory bandwidth)

#### 2.1.2 Solution: `cpu_optimized_surrogate.py`

Create a new module that provides a drop-in replacement optimized for CPU:

```python
# caligo/exploration/cpu_optimized_surrogate.py
"""
CPU-optimized surrogate modeling for HPC environments.

This module provides CPU-optimized Gaussian Process surrogates using:
1. GPyTorch with OpenMP-parallelized Lanczos/CG solvers
2. Float64 precision (better cache utilization on CPU)
3. LOVE (Lanczos Variance Estimates) for fast predictive variance
4. Chunked inference to stay within 2GB/core memory budget
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import torch
from numpy.typing import NDArray

# Set OpenMP threads BEFORE importing GPyTorch
os.environ.setdefault("OMP_NUM_THREADS", "4")  # Per-GP parallelism
os.environ.setdefault("MKL_NUM_THREADS", "4")

import gpytorch
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.models import ExactGP
from gpytorch.settings import (
    fast_computations,
    max_cholesky_size,
    max_lanczos_quadrature_iterations,
    use_toeplitz,
)

from caligo.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class CPUSurrogateConfig:
    """
    Configuration optimized for AMD Genoa CPUs.
    
    Parameters
    ----------
    dtype : torch.dtype
        Float64 for CPU cache efficiency (vs Float32 for GPU VRAM).
    num_threads : int
        OpenMP threads per GP operation. Default 4 balances
        parallelism vs memory bandwidth on Zen4.
    max_cholesky_size : int
        Use Lanczos/CG above this matrix size. 2000 is optimal
        for 2GB/core memory budget.
    lanczos_iterations : int
        Lanczos iterations for variance estimation.
    use_love : bool
        Use LOVE for O(1) predictive variance (highly recommended).
    chunk_size : int
        Inference chunk size to limit memory footprint.
    training_iterations : int
        Adam iterations for hyperparameter optimization.
    learning_rate : float
        Learning rate for Adam.
    """
    
    dtype: torch.dtype = torch.float64
    num_threads: int = 4
    max_cholesky_size: int = 2000
    lanczos_iterations: int = 100
    use_love: bool = True
    chunk_size: int = 5000
    training_iterations: int = 150
    learning_rate: float = 0.05


class CPUExactGP(ExactGP):
    """Exact GP with CPU-optimized settings."""
    
    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: GaussianLikelihood,
        ard_num_dims: int,
    ) -> None:
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(
            MaternKernel(nu=2.5, ard_num_dims=ard_num_dims)
        )
    
    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


class CPUOptimizedLandscape:
    """
    CPU-optimized twin GP surrogate for HPC deployment.
    
    Key optimizations:
    - Float64 for better CPU cache utilization
    - Lanczos solver for O(N²) scaling
    - LOVE for O(1) variance estimation
    - Chunked inference for memory efficiency
    - OpenMP threading tuned for Zen4 CCDs
    """
    
    def __init__(self, config: Optional[CPUSurrogateConfig] = None) -> None:
        self.config = config or CPUSurrogateConfig()
        
        # Set thread counts
        torch.set_num_threads(self.config.num_threads)
        
        self._baseline_gp: Optional[CPUExactGP] = None
        self._blind_gp: Optional[CPUExactGP] = None
        self._baseline_likelihood: Optional[GaussianLikelihood] = None
        self._blind_likelihood: Optional[GaussianLikelihood] = None
        
        # Normalization statistics
        self._X_mean: Optional[torch.Tensor] = None
        self._X_std: Optional[torch.Tensor] = None
        
        logger.info(
            f"Initialized CPUOptimizedLandscape: "
            f"threads={self.config.num_threads}, "
            f"dtype={self.config.dtype}, "
            f"max_cholesky={self.config.max_cholesky_size}"
        )
    
    def fit(
        self,
        X_baseline: NDArray[np.floating],
        y_baseline: NDArray[np.floating],
        X_blind: NDArray[np.floating],
        y_blind: NDArray[np.floating],
    ) -> "CPUOptimizedLandscape":
        """Fit twin GPs with CPU-optimized settings."""
        
        # Compute normalization stats on combined data
        X_all = np.vstack([X_baseline, X_blind])
        self._X_mean = torch.tensor(X_all.mean(axis=0), dtype=self.config.dtype)
        self._X_std = torch.tensor(X_all.std(axis=0) + 1e-8, dtype=self.config.dtype)
        
        # Configure GPyTorch for CPU
        with max_cholesky_size(self.config.max_cholesky_size), \
             max_lanczos_quadrature_iterations(self.config.lanczos_iterations):
            
            self._baseline_gp, self._baseline_likelihood = self._fit_single(
                X_baseline, y_baseline, "baseline"
            )
            self._blind_gp, self._blind_likelihood = self._fit_single(
                X_blind, y_blind, "blind"
            )
        
        return self
    
    def _fit_single(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.floating],
        name: str,
    ) -> Tuple[CPUExactGP, GaussianLikelihood]:
        """Fit a single GP."""
        
        X_t = self._normalize(torch.tensor(X, dtype=self.config.dtype))
        y_t = torch.tensor(y, dtype=self.config.dtype)
        
        likelihood = GaussianLikelihood()
        model = CPUExactGP(X_t, y_t, likelihood, ard_num_dims=X.shape[1])
        
        model.train()
        likelihood.train()
        
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        mll = ExactMarginalLogLikelihood(likelihood, model)
        
        for i in range(self.config.training_iterations):
            optimizer.zero_grad()
            output = model(X_t)
            loss = -mll(output, y_t)
            loss.backward()
            optimizer.step()
            
            if (i + 1) % 50 == 0:
                logger.debug(f"{name} GP iter {i+1}: loss={loss.item():.4f}")
        
        logger.info(f"Fitted {name} GP: {len(X)} samples, final_loss={loss.item():.4f}")
        return model, likelihood
    
    def _normalize(self, X: torch.Tensor) -> torch.Tensor:
        """Normalize features using stored statistics."""
        return (X - self._X_mean) / self._X_std
    
    def predict_baseline(
        self,
        X: NDArray[np.floating],
        return_std: bool = False,
    ) -> Tuple[NDArray[np.floating], Optional[NDArray[np.floating]]]:
        """Predict with chunked inference for memory efficiency."""
        return self._predict_chunked(
            self._baseline_gp, self._baseline_likelihood, X, return_std
        )
    
    def predict_blind(
        self,
        X: NDArray[np.floating],
        return_std: bool = False,
    ) -> Tuple[NDArray[np.floating], Optional[NDArray[np.floating]]]:
        """Predict with chunked inference for memory efficiency."""
        return self._predict_chunked(
            self._blind_gp, self._blind_likelihood, X, return_std
        )
    
    def _predict_chunked(
        self,
        model: CPUExactGP,
        likelihood: GaussianLikelihood,
        X: NDArray[np.floating],
        return_std: bool,
    ) -> Tuple[NDArray[np.floating], Optional[NDArray[np.floating]]]:
        """Chunked prediction to limit memory footprint."""
        
        model.eval()
        likelihood.eval()
        
        n_samples = len(X)
        chunk_size = self.config.chunk_size
        
        means = []
        stds = [] if return_std else None
        
        with torch.no_grad(), gpytorch.settings.fast_pred_var(self.config.use_love):
            for start in range(0, n_samples, chunk_size):
                end = min(start + chunk_size, n_samples)
                X_chunk = torch.tensor(X[start:end], dtype=self.config.dtype)
                X_norm = self._normalize(X_chunk)
                
                pred = likelihood(model(X_norm))
                means.append(pred.mean.numpy())
                
                if return_std:
                    stds.append(pred.stddev.numpy())
        
        mean_arr = np.concatenate(means).astype(np.float32)
        std_arr = np.concatenate(stds).astype(np.float32) if return_std else None
        
        return mean_arr, std_arr
```

#### 2.1.3 Integration Strategy

Modify `active_executor.py` to use the CPU-optimized surrogate on HPC:

```python
# In active_executor.py, add environment detection:

import os

def _select_landscape_backend():
    """Select surrogate backend based on environment."""
    
    # Check for HPC environment marker (set in SLURM job script)
    if os.environ.get("CALIGO_HPC_MODE") == "1":
        from caligo.exploration.cpu_optimized_surrogate import (
            CPUOptimizedLandscape,
            CPUSurrogateConfig,
        )
        logger.info("HPC mode: using CPUOptimizedLandscape")
        return CPUOptimizedLandscape, CPUSurrogateConfig
    
    # Check for GPU availability
    try:
        import torch
        if torch.cuda.is_available():
            from caligo.exploration.gpu_surrogate import (
                GPyTorchLandscape,
                GPyTorchConfig,
            )
            logger.info("GPU detected: using GPyTorchLandscape")
            return GPyTorchLandscape, GPyTorchConfig
    except ImportError:
        pass
    
    # Fallback to sklearn-based surrogate
    from caligo.exploration.surrogate import EfficiencyLandscape, GPConfig
    logger.info("Fallback: using sklearn EfficiencyLandscape")
    return EfficiencyLandscape, GPConfig
```

---

### 2.2 Path B: Massive Parallelism Optimization

#### 2.2.1 Current Worker Pool Analysis

From [lhs_executor.py](caligo/exploration/lhs_executor.py) and [epr_batcher.py](caligo/exploration/epr_batcher.py):

```python
# Current: Default to cpu_count() or config-specified
max_workers = exec_cfg.get("num_workers", 16)
```

**Problems with 192 workers:**
1. **Memory Explosion**: Each SquidASM worker can consume 500MB–2GB
2. **IPC Overhead**: 192 workers × pickle serialization = bottleneck
3. **NetSquid State Isolation**: Each worker needs fresh NetSquid state

#### 2.2.2 Solution: Hierarchical Worker Architecture

```python
# caligo/exploration/hpc_worker_pool.py
"""
HPC-optimized worker pool for AMD Genoa nodes.

Architecture:
- Outer tier: 12 "super-workers" (one per CCD)
- Inner tier: 15 workers per super-worker (one per core in CCD)
- Memory budget: 32GB per super-worker (16 cores × 2GB)
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Optional

from caligo.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class HPCPoolConfig:
    """
    Configuration for HPC worker pools.
    
    Parameters
    ----------
    total_cores : int
        Total available cores (default: auto-detect).
    cores_per_ccd : int
        Cores per Core Complex Die on Genoa (16 for 9654).
    reserved_cores : int
        Cores reserved for orchestration and I/O.
    max_tasks_per_child : int
        Worker recycling to prevent memory leaks.
    use_spawn : bool
        Use 'spawn' start method (required for CUDA compat and clean state).
    memory_limit_gb : float
        Soft memory limit per worker (for monitoring).
    """
    
    total_cores: int = 192
    cores_per_ccd: int = 16
    reserved_cores: int = 12
    max_tasks_per_child: int = 50
    use_spawn: bool = True
    memory_limit_gb: float = 1.8  # Leave headroom within 2GB


def get_optimal_worker_count(config: Optional[HPCPoolConfig] = None) -> int:
    """
    Calculate optimal worker count for Genoa node.
    
    Strategy:
    - Reserve cores for main process, I/O, and BLAS threads
    - Account for hyperthreading (not beneficial for compute-bound)
    - Leave headroom for memory pressure
    """
    config = config or HPCPoolConfig()
    
    available = config.total_cores - config.reserved_cores
    
    # For compute-bound quantum simulation, ~90% core utilization is optimal
    optimal = int(available * 0.9)
    
    logger.info(
        f"HPC worker calculation: {config.total_cores} total, "
        f"{config.reserved_cores} reserved → {optimal} workers"
    )
    
    return optimal


def create_hpc_executor(config: Optional[HPCPoolConfig] = None) -> ProcessPoolExecutor:
    """
    Create a ProcessPoolExecutor optimized for Genoa HPC.
    
    Key settings:
    - max_workers: Calculated from core count
    - max_tasks_per_child: Recycle workers to prevent memory leaks
    - mp_context: Use 'spawn' for clean NetSquid state isolation
    """
    import multiprocessing as mp
    
    config = config or HPCPoolConfig()
    n_workers = get_optimal_worker_count(config)
    
    # Use spawn to ensure clean process state (critical for NetSquid)
    ctx = mp.get_context("spawn") if config.use_spawn else None
    
    executor = ProcessPoolExecutor(
        max_workers=n_workers,
        max_tasks_per_child=config.max_tasks_per_child,
        mp_context=ctx,
    )
    
    logger.info(
        f"Created HPC executor: {n_workers} workers, "
        f"max_tasks_per_child={config.max_tasks_per_child}"
    )
    
    return executor
```

#### 2.2.3 Batch Size Tuning

Update `qia_challenge_config.yaml` for HPC:

```yaml
# explor_configs/hpc_genoa_config.yaml

execution:
  # HPC: Use 90% of 192 cores, leave room for orchestration
  num_workers: 172
  
  # Longer timeout for complex simulations
  timeout_seconds: 600.0
  
  # Worker recycling to prevent memory leaks (critical!)
  max_tasks_per_child: 50

phase1:
  # Larger batches to amortize IPC overhead
  batch_size: 172  # One sample per worker
  
  # More aggressive checkpointing for fault tolerance
  checkpoint_interval: 3

phase3:
  # Match batch size to worker count for full utilization
  batch_size: 172
  
  # Retrain less frequently (expensive on CPU)
  retrain_interval: 10
```

---

### 2.3 Path C: Vectorized LDPC Reconciliation

#### 2.3.1 Current LDPC Performance Bottleneck

From [ldpc_decoder.py](caligo/reconciliation/ldpc_decoder.py#L230-L280):

```python
# Current: Loop-based check node update
for c in range(m):
    start = int(compiled.check_ptr[c])
    end = int(compiled.check_ptr[c + 1])
    # ... per-check operations
```

This loop-based approach cannot leverage AVX-512.

#### 2.3.2 Solution: Numba-JIT Vectorized Decoder

```python
# caligo/reconciliation/ldpc_decoder_avx.py
"""
AVX-512 optimized LDPC decoder using Numba.

This module provides a belief propagation decoder optimized for AMD Genoa's
AVX-512 capabilities using Numba's SIMD vectorization.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange
from numba.typed import List

from caligo.reconciliation.compiled_matrix import CompiledParityCheckMatrix
from caligo.reconciliation.ldpc_decoder import DecodeResult
from caligo.utils.logging import get_logger

logger = get_logger(__name__)


@njit(parallel=True, fastmath=True, cache=True)
def _check_node_update_vectorized(
    q: np.ndarray,
    r: np.ndarray,
    check_ptr: np.ndarray,
    check_var: np.ndarray,
    target_syndrome: np.ndarray,
    m: int,
) -> None:
    """
    Vectorized check-to-variable message update.
    
    Uses Numba's parallel loop and fastmath for AVX-512 auto-vectorization.
    The tanh-domain update is computed in a SIMD-friendly manner.
    """
    for c in prange(m):
        start = check_ptr[c]
        end = check_ptr[c + 1]
        degree = end - start
        
        if degree == 0:
            continue
        
        # Compute tanh(q/2) for all edges in this check
        tanh_vals = np.empty(degree, dtype=np.float64)
        for i in range(degree):
            tanh_vals[i] = np.tanh(q[start + i] * 0.5)
        
        # Compute product excluding each index
        total_prod = 1.0
        for i in range(degree):
            total_prod *= tanh_vals[i]
        
        # Compute messages
        sign_flip = -1.0 if target_syndrome[c] == 1 else 1.0
        
        for i in range(degree):
            if abs(tanh_vals[i]) > 1e-10:
                prod_excl = total_prod / tanh_vals[i]
            else:
                # Handle zero case: product of others
                prod_excl = 1.0
                for j in range(degree):
                    if j != i:
                        prod_excl *= tanh_vals[j]
            
            # Clip and compute arctanh
            prod_clipped = max(-0.999999, min(0.999999, prod_excl))
            r[start + i] = sign_flip * 2.0 * np.arctanh(prod_clipped)


@njit(parallel=True, fastmath=True, cache=True)
def _variable_node_update_vectorized(
    llr: np.ndarray,
    q: np.ndarray,
    r: np.ndarray,
    var_ptr: np.ndarray,
    var_edges: np.ndarray,
    decoded: np.ndarray,
    n: int,
) -> None:
    """
    Vectorized variable-to-check message update and hard decision.
    
    Computes total LLR per variable and updates outgoing messages.
    """
    for v in prange(n):
        start = var_ptr[v]
        end = var_ptr[v + 1]
        
        # Sum incoming check messages
        sum_r = 0.0
        for i in range(start, end):
            edge_idx = var_edges[i]
            sum_r += r[edge_idx]
        
        # Total LLR and hard decision
        total_llr = llr[v] + sum_r
        decoded[v] = 1 if total_llr < 0.0 else 0
        
        # Update outgoing messages (exclude incoming r)
        for i in range(start, end):
            edge_idx = var_edges[i]
            q[edge_idx] = total_llr - r[edge_idx]


class AVX512BeliefPropagationDecoder:
    """
    AVX-512 optimized belief propagation decoder.
    
    Uses Numba JIT compilation with parallel=True and fastmath=True
    to generate SIMD-vectorized code for AMD Genoa's AVX-512 units.
    
    Performance characteristics:
    - 2-4× speedup over pure NumPy on Genoa
    - Automatic parallelization across cores
    - Cache-friendly memory access patterns
    """
    
    def __init__(
        self,
        max_iterations: int = 100,
        threshold: float = 1e-8,
    ) -> None:
        self.max_iterations = max_iterations
        self.threshold = threshold
        
        # Pre-compile Numba functions on first use
        self._warmup_done = False
    
    def _warmup(self) -> None:
        """Trigger Numba JIT compilation with small test data."""
        if self._warmup_done:
            return
        
        # Minimal compilation trigger
        small_q = np.zeros(10, dtype=np.float64)
        small_r = np.zeros(10, dtype=np.float64)
        small_ptr = np.array([0, 2, 5, 10], dtype=np.int32)
        small_var = np.array([0, 1, 0, 1, 2, 0, 1, 2, 3, 4], dtype=np.int32)
        small_syn = np.zeros(3, dtype=np.uint8)
        
        _check_node_update_vectorized(
            small_q, small_r, small_ptr, small_var, small_syn, 3
        )
        
        self._warmup_done = True
        logger.debug("AVX-512 decoder Numba warmup complete")
    
    def decode(
        self,
        llr: np.ndarray,
        target_syndrome: np.ndarray,
        compiled: CompiledParityCheckMatrix,
        max_iterations: int = None,
    ) -> DecodeResult:
        """
        Decode using AVX-512 optimized belief propagation.
        
        Parameters
        ----------
        llr : np.ndarray
            Initial log-likelihood ratios, shape (n,).
        target_syndrome : np.ndarray
            Target syndrome vector, shape (m,).
        compiled : CompiledParityCheckMatrix
            Precompiled parity-check matrix.
        max_iterations : int, optional
            Override default max iterations.
        
        Returns
        -------
        DecodeResult
            Decoding result with corrected bits.
        """
        self._warmup()
        
        m, n = compiled.m, compiled.n
        max_iters = max_iterations or self.max_iterations
        
        # Ensure contiguous float64 arrays
        llr = np.ascontiguousarray(llr, dtype=np.float64)
        target_u8 = np.ascontiguousarray(target_syndrome, dtype=np.uint8)
        
        # Allocate message arrays
        r = np.zeros(compiled.edge_count, dtype=np.float64)
        q = np.zeros(compiled.edge_count, dtype=np.float64)
        decoded = np.zeros(n, dtype=np.uint8)
        
        # Initialize q with channel LLR
        for v in range(n):
            start = compiled.var_ptr[v]
            end = compiled.var_ptr[v + 1]
            for i in range(start, end):
                q[compiled.var_edges[i]] = llr[v]
        
        converged = False
        iteration = 0
        
        for iteration in range(1, max_iters + 1):
            # Check node update (parallel)
            _check_node_update_vectorized(
                q, r,
                compiled.check_ptr.astype(np.int64),
                compiled.check_var.astype(np.int64),
                target_u8,
                m,
            )
            
            # Variable node update (parallel)
            _variable_node_update_vectorized(
                llr, q, r,
                compiled.var_ptr.astype(np.int64),
                compiled.var_edges.astype(np.int64),
                decoded,
                n,
            )
            
            # Check convergence
            syndrome_errors = compiled.count_syndrome_errors(decoded, target_u8)
            if syndrome_errors == 0:
                converged = True
                break
        
        return DecodeResult(
            corrected_bits=decoded.copy(),
            converged=converged,
            iterations=iteration,
            syndrome_errors=int(syndrome_errors) if not converged else 0,
        )
```

#### 2.3.3 BLAS Configuration for AMD Genoa

Create an environment setup script:

```bash
#!/bin/bash
# scripts/setup_hpc_env.sh
# Environment configuration for Caligo on AMD Genoa HPC nodes

# ============================================================================
# BLAS/LAPACK Configuration
# ============================================================================

# Use AMD Optimizing CPU Libraries (AOCL) if available
if [ -d "/opt/AMD/aocl" ]; then
    export LD_LIBRARY_PATH="/opt/AMD/aocl/lib:$LD_LIBRARY_PATH"
    export BLAS="/opt/AMD/aocl/lib/libblis.so"
    export LAPACK="/opt/AMD/aocl/lib/libflame.so"
    echo "Using AMD AOCL for BLAS/LAPACK"
fi

# OpenBLAS thread configuration
export OPENBLAS_NUM_THREADS=4
export GOTO_NUM_THREADS=4

# MKL configuration (if present, optimize for AMD)
export MKL_DEBUG_CPU_TYPE=5  # Pretend to be Intel for better codepath
export MKL_NUM_THREADS=4

# ============================================================================
# OpenMP Configuration
# ============================================================================

# Optimal for Zen4 CCDs
export OMP_NUM_THREADS=4
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# ============================================================================
# Python/NumPy Configuration
# ============================================================================

# Ensure NumPy uses optimized BLAS
export NPY_NUM_BUILD_JOBS=16

# Disable NumPy internal threading (we manage parallelism)
export OPENBLAS_MAIN_FREE=1

# ============================================================================
# Caligo HPC Mode
# ============================================================================

export CALIGO_HPC_MODE=1
export CALIGO_SCRATCH_DIR="${SCRATCH_NODE:-/scratch-node}/${USER}/caligo"

# Create scratch directory
mkdir -p "$CALIGO_SCRATCH_DIR"

echo "Caligo HPC environment configured"
echo "  BLAS threads: $OPENBLAS_NUM_THREADS"
echo "  OMP threads: $OMP_NUM_THREADS"
echo "  Scratch dir: $CALIGO_SCRATCH_DIR"
```

---

## 3. Deployment Configuration

### 3.1 SLURM Job Script

```bash
#!/bin/bash
#SBATCH --job-name=caligo-explor
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --mem=380G
#SBATCH --time=48:00:00
#SBATCH --partition=tcn
#SBATCH --output=caligo_%j.out
#SBATCH --error=caligo_%j.err

# ============================================================================
# Caligo Exploration Campaign - HPC Job Script
# ============================================================================

echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Start time: $(date)"
echo "============================================"

# Load environment
source /path/to/caligo/scripts/setup_hpc_env.sh

# Activate virtual environment
source /path/to/qia/bin/activate

# Set scratch directory for this job
export CALIGO_OUTPUT_DIR="${CALIGO_SCRATCH_DIR}/run_${SLURM_JOB_ID}"
mkdir -p "$CALIGO_OUTPUT_DIR"

# Navigate to project
cd /path/to/qia-challenge-2025/caligo

# Run exploration with HPC configuration
python main_explor.py \
    --config explor_configs/hpc_genoa_config.yaml \
    --workers 172 \
    --output-dir "$CALIGO_OUTPUT_DIR"

# Copy results back to persistent storage
RESULTS_DIR="/home/${USER}/caligo_results/run_${SLURM_JOB_ID}"
mkdir -p "$RESULTS_DIR"
cp -r "$CALIGO_OUTPUT_DIR"/* "$RESULTS_DIR/"

echo "============================================"
echo "Job complete: $(date)"
echo "Results: $RESULTS_DIR"
echo "============================================"
```

### 3.2 HPC Configuration File

```yaml
# explor_configs/hpc_genoa_config.yaml
# Configuration optimized for AMD Genoa 192-core nodes

output:
  base_dir: "${CALIGO_OUTPUT_DIR:-exploration_results}"
  campaign_name: "qia_challenge_hpc"
  timestamped: true

execution:
  # HPC: 172 workers (180 usable - 8 reserved for orchestration)
  num_workers: 172
  
  # Longer timeout for complex simulations
  timeout_seconds: 600.0
  
  # Random seed for reproducibility
  random_seed: 42
  
  # HPC-specific settings
  max_tasks_per_child: 50  # Worker recycling
  use_spawn_context: true   # Clean process state

phase1:
  num_samples: 10000  # Larger warmup for 192-core throughput
  batch_size: 172     # One sample per worker
  checkpoint_interval: 3

phase2:
  gp:
    kernel_type: "matern52"
    n_restarts_optimizer: 20  # More restarts (CPU time is cheaper)
    normalize_y: true
  
  # CPU-optimized surrogate settings
  cpu_surrogate:
    enabled: true
    num_threads: 4
    max_cholesky_size: 2000
    use_love: true

phase3:
  num_iterations: 200
  batch_size: 172
  retrain_interval: 10  # Less frequent retraining (expensive on CPU)
  checkpoint_interval: 5
  
  acquisition:
    type: "straddle"
    kappa: 1.96
    target_threshold: 0.0

bounds:
  storage_noise_r:
    min: 0.60
    max: 0.95
  storage_rate_nu:
    min: 0.001
    max: 1.0
  wait_time_ns:
    min_log: 5.0
    max_log: 9.0
  channel_fidelity:
    min: 0.501
    max: 1.0
  detection_efficiency:
    min_log: -3.0
    max_log: 0.0
  detector_error:
    min: 0.0
    max: 0.1
  dark_count_prob:
    min_log: -8.0
    max_log: -3.0
  num_pairs:
    min_log: 4.0
    max_log: 6.0
```

### 3.3 Environment Variables Reference

| Variable | Recommended Value | Purpose |
|----------|-------------------|---------|
| `CALIGO_HPC_MODE` | `1` | Enable HPC-optimized code paths |
| `CALIGO_SCRATCH_DIR` | `/scratch-node/$USER/caligo` | Fast local I/O |
| `OMP_NUM_THREADS` | `4` | Threads per OpenMP region |
| `OMP_PROC_BIND` | `close` | Thread-core affinity |
| `OMP_PLACES` | `cores` | Bind to physical cores |
| `OPENBLAS_NUM_THREADS` | `4` | BLAS parallelism |
| `MKL_DEBUG_CPU_TYPE` | `5` | MKL AMD optimization hint |
| `NUMBA_NUM_THREADS` | `4` | Numba parallelism |

---

## 4. Implementation Roadmap

### Phase 1: Foundation (Week 1-2)

| Task | Priority | Effort | Dependencies |
|------|----------|--------|--------------|
| Create `cpu_optimized_surrogate.py` | 🔴 Critical | 3 days | None |
| Create `hpc_worker_pool.py` | 🔴 Critical | 2 days | None |
| Create `ldpc_decoder_avx.py` | 🟡 High | 3 days | None |
| Create `setup_hpc_env.sh` | 🟡 High | 1 day | None |
| Create `hpc_genoa_config.yaml` | 🟡 High | 1 day | None |

### Phase 2: Integration (Week 3)

| Task | Priority | Effort | Dependencies |
|------|----------|--------|--------------|
| Modify `active_executor.py` for backend selection | 🔴 Critical | 1 day | cpu_optimized_surrogate |
| Modify `lhs_executor.py` for HPC pool | 🔴 Critical | 1 day | hpc_worker_pool |
| Add `--hpc` flag to `main_explor.py` | 🟡 High | 0.5 days | All above |
| Update `SharedMemoryArena` for 64+ slots | 🟡 High | 1 day | None |

### Phase 3: Testing & Optimization (Week 4)

| Task | Priority | Effort | Dependencies |
|------|----------|--------|--------------|
| Unit tests for CPU surrogate | 🟡 High | 2 days | cpu_optimized_surrogate |
| Benchmark LDPC decoder variants | 🟡 High | 1 day | ldpc_decoder_avx |
| Memory profiling under 2GB/core | 🔴 Critical | 2 days | All integration |
| Create SLURM job template | 🟡 High | 0.5 days | setup_hpc_env.sh |

### Phase 4: Production Deployment (Week 5)

| Task | Priority | Effort | Dependencies |
|------|----------|--------|--------------|
| Dry-run on HPC node (1 hour job) | 🔴 Critical | 1 day | All above |
| Tune batch sizes based on profiling | 🟡 High | 1 day | Dry-run |
| Full 48-hour production run | 🟢 Normal | 2 days | Tuning |
| Document lessons learned | 🟢 Normal | 1 day | Production run |

---

## 5. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| CPU surrogate 10× slower than GPU | Medium | High | Use LOVE for O(1) variance; reduce retrain frequency |
| Memory OOM with 172 workers | Medium | High | Strict `max_tasks_per_child=50`; monitor with `psutil` |
| NetSquid state leaks | Low | High | Use `spawn` context; worker recycling |
| Numba compilation slow on first run | High | Low | Warmup in job prologue |
| HDF5 lock contention | Medium | Medium | Use local NVMe scratch |

---

## 6. Expected Performance

### Throughput Estimates

| Metric | Current (16-core workstation) | Target (192-core Genoa) | Speedup |
|--------|-------------------------------|-------------------------|---------|
| Phase 1 samples/hour | ~200 | ~2,000 | 10× |
| Phase 3 iterations/hour | ~10 | ~50 | 5× |
| Total campaign time | 48 hours | 12 hours | 4× |

### Memory Budget

| Component | Per-Worker | Total (172 workers) |
|-----------|------------|---------------------|
| Python interpreter | 50 MB | 8.6 GB |
| NumPy arrays | 100 MB | 17.2 GB |
| SquidASM simulation | 500 MB | 86 GB |
| LDPC matrices | 200 MB | 34.4 GB |
| **Headroom** | — | **237 GB** |

---

## 7. Conclusion

Migrating Caligo to the AMD Genoa HPC node is feasible with targeted refactoring. The primary challenges are:

1. **GPU → CPU Surrogate**: Addressed via GPyTorch CPU backend with LOVE variance estimation
2. **Scaling to 192 cores**: Addressed via hierarchical worker pools with strict memory discipline
3. **LDPC Vectorization**: Addressed via Numba-JIT with AVX-512 auto-vectorization

The proposed architecture maintains the existing codebase structure while adding HPC-specific modules that can be conditionally enabled. This allows continued development on workstations while supporting production runs on HPC infrastructure.

**Next Steps:**
1. Review this roadmap with the team
2. Prioritize Phase 1 tasks
3. Request a short allocation on the Genoa node for initial testing
