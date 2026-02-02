#!/bin/bash
# ==============================================================================
# Caligo HPC Environment Setup Script
# Optimized for AMD EPYC 9654 (Genoa) nodes with 192 cores
# ==============================================================================

set -euo pipefail

echo "============================================"
echo "Caligo HPC Environment Configuration"
echo "Target: AMD EPYC 9654 (Genoa) - 192 cores"
echo "============================================"

# ============================================================================
# BLAS/LAPACK Configuration
# ============================================================================

# Prefer AMD Optimizing CPU Libraries (AOCL) if available
if [ -d "/opt/AMD/aocl" ]; then
    export LD_LIBRARY_PATH="/opt/AMD/aocl/lib:${LD_LIBRARY_PATH:-}"
    export BLAS="/opt/AMD/aocl/lib/libblis.so"
    export LAPACK="/opt/AMD/aocl/lib/libflame.so"
    echo "✓ Using AMD AOCL for BLAS/LAPACK"
elif [ -d "/opt/OpenBLAS" ]; then
    export LD_LIBRARY_PATH="/opt/OpenBLAS/lib:${LD_LIBRARY_PATH:-}"
    echo "✓ Using OpenBLAS"
else
    echo "⚠ No optimized BLAS found - using system default"
fi

# Thread configuration for BLAS libraries
# 4 threads per operation balances parallelism vs memory bandwidth on Zen4
export OPENBLAS_NUM_THREADS=4
export GOTO_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4

# MKL configuration (if present)
# MKL_DEBUG_CPU_TYPE=5 hints to use optimized codepaths on AMD
export MKL_DEBUG_CPU_TYPE=5
export MKL_NUM_THREADS=4
export MKL_DYNAMIC=FALSE

# ============================================================================
# OpenMP Configuration (Critical for Numba/GPyTorch)
# ============================================================================

# 4 threads per OpenMP parallel region
# This allows ~48 independent GP evaluations with 4 threads each
export OMP_NUM_THREADS=4

# Bind threads to cores for cache locality
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# Stack size for deeply recursive operations
export OMP_STACKSIZE=16M

# Disable dynamic thread adjustment
export OMP_DYNAMIC=FALSE

# ============================================================================
# Numba Configuration
# ============================================================================

# Match OpenMP thread count
export NUMBA_NUM_THREADS=4

# Enable parallel target
export NUMBA_THREADING_LAYER=omp

# Cache compiled functions
export NUMBA_CACHE_DIR="${HOME}/.cache/numba"
mkdir -p "$NUMBA_CACHE_DIR"

# ============================================================================
# PyTorch Configuration (CPU-only)
# ============================================================================

# Disable CUDA (not available on this node)
export CUDA_VISIBLE_DEVICES=""

# PyTorch inter-op parallelism (between independent operations)
export OMP_NUM_THREADS=4

# PyTorch intra-op parallelism (within single operation)
# Set via torch.set_num_threads() in Python

# ============================================================================
# Python/Process Configuration
# ============================================================================

# Use 'spawn' multiprocessing start method for clean process isolation
export PYTHONSTARTUP=""
export PYTHONHASHSEED=42

# Disable Python's GIL debugging (minor overhead)
export PYTHONMALLOC=default

# ============================================================================
# Caligo-Specific Configuration
# ============================================================================

# Enable HPC mode in Caligo
export CALIGO_HPC_MODE=1

# Use local NVMe scratch for I/O-intensive operations
if [ -d "/scratch-node" ]; then
    export CALIGO_SCRATCH_DIR="/scratch-node/${USER:-caligo}/caligo_${SLURM_JOB_ID:-$$}"
elif [ -d "/tmp" ]; then
    export CALIGO_SCRATCH_DIR="/tmp/caligo_${USER:-caligo}_${SLURM_JOB_ID:-$$}"
else
    export CALIGO_SCRATCH_DIR="${HOME}/caligo_scratch"
fi

# Create scratch directories
mkdir -p "$CALIGO_SCRATCH_DIR"/{checkpoints,hdf5,logs}

# Set HDF5 to use local scratch
export HDF5_USE_FILE_LOCKING=FALSE

# ============================================================================
# Memory Configuration
# ============================================================================

# Limit per-process memory to leave headroom (340GB for workers, 44GB reserved)
# This is enforced via ulimit, not cgroups
# ulimit -v $((340 * 1024 * 1024))  # Uncomment if needed

# ============================================================================
# Summary
# ============================================================================

echo ""
echo "Environment configured:"
echo "  BLAS threads:     $OPENBLAS_NUM_THREADS"
echo "  OpenMP threads:   $OMP_NUM_THREADS"
echo "  Numba threads:    $NUMBA_NUM_THREADS"
echo "  Scratch dir:      $CALIGO_SCRATCH_DIR"
echo "  HPC mode:         $CALIGO_HPC_MODE"
echo ""
echo "Memory layout (384 GiB total):"
echo "  Workers (172×):   ~344 GiB @ 2GB/worker"
echo "  Orchestrator:     ~20 GiB"
echo "  System/headroom:  ~20 GiB"
echo ""
echo "============================================"
