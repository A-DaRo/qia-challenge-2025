#!/bin/bash
#SBATCH --job-name=caligo-explor
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --mem=380G
#SBATCH --time=48:00:00
#SBATCH --partition=tcn
#SBATCH --output=logs/caligo_%j.out
#SBATCH --error=logs/caligo_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=${USER}@example.com

# ==============================================================================
# Caligo Exploration Campaign - SLURM Job Script
# Target: AMD EPYC 9654 (Genoa) node - 192 cores, 384 GiB RAM
# ==============================================================================

set -euo pipefail

# ============================================================================
# Job Information
# ============================================================================

echo "============================================"
echo "Caligo HPC Exploration Campaign"
echo "============================================"
echo "Job ID:        $SLURM_JOB_ID"
echo "Job Name:      $SLURM_JOB_NAME"
echo "Node:          $SLURM_NODELIST"
echo "CPUs:          $SLURM_CPUS_PER_TASK"
echo "Memory:        $SLURM_MEM_PER_NODE MB"
echo "Partition:     $SLURM_JOB_PARTITION"
echo "Start Time:    $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"

# ============================================================================
# Path Configuration (EDIT THESE)
# ============================================================================

# Project root directory
PROJECT_ROOT="/path/to/qia-challenge-2025/caligo"

# Python virtual environment
VENV_PATH="/path/to/qia/bin/activate"

# Persistent results directory (network storage)
RESULTS_BASE="/home/${USER}/caligo_results"

# ============================================================================
# Environment Setup
# ============================================================================

echo ""
echo "[1/6] Setting up environment..."

# Load any required modules (site-specific)
# module load python/3.10
# module load aocl/4.0

# Source HPC environment configuration
source "${PROJECT_ROOT}/scripts/setup_hpc_env.sh"

# Activate Python environment
source "$VENV_PATH"

# Verify Python and key packages
echo "Python: $(which python) ($(python --version 2>&1))"
echo "NumPy BLAS: $(python -c 'import numpy; print(numpy.show_config())' 2>&1 | grep -i blas | head -1 || echo 'unknown')"

# ============================================================================
# Scratch Directory Setup
# ============================================================================

echo ""
echo "[2/6] Setting up scratch directory..."

# CALIGO_SCRATCH_DIR is set by setup_hpc_env.sh
echo "Scratch directory: $CALIGO_SCRATCH_DIR"

# Verify NVMe scratch is available and has space
if [ -d "/scratch-node" ]; then
    SCRATCH_FREE=$(df -BG /scratch-node | tail -1 | awk '{print $4}' | tr -d 'G')
    echo "Scratch free space: ${SCRATCH_FREE}G"
    
    if [ "$SCRATCH_FREE" -lt 100 ]; then
        echo "WARNING: Less than 100GB free on scratch!"
    fi
fi

# ============================================================================
# Pre-flight Checks
# ============================================================================

echo ""
echo "[3/6] Running pre-flight checks..."

cd "$PROJECT_ROOT"

# Verify config exists
CONFIG_FILE="explor_configs/hpc_genoa_config.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi
echo "✓ Config file: $CONFIG_FILE"

# Quick Python import test
python -c "
import caligo
from caligo.exploration.lhs_executor import Phase1Executor
from caligo.exploration.active_executor import Phase3Executor
print('✓ Caligo imports OK')
"

# Test NumPy threading
python -c "
import numpy as np
import os
print(f'✓ NumPy threads: OMP={os.environ.get(\"OMP_NUM_THREADS\", \"unset\")}')
"

# Memory check
TOTAL_MEM_GB=$(free -g | awk '/^Mem:/{print $2}')
echo "✓ Total memory: ${TOTAL_MEM_GB}G"

# ============================================================================
# Numba Warmup (Optional but Recommended)
# ============================================================================

echo ""
echo "[4/6] Warming up Numba JIT cache..."

python -c "
# Pre-compile Numba functions to avoid compilation during campaign
import numpy as np
from caligo.exploration.sampler import check_feasibility_batch

# Trigger compilation with small data
test_samples = np.random.rand(10, 9).astype(np.float32)
_ = check_feasibility_batch(test_samples)
print('✓ Numba warmup complete')
" || echo "⚠ Numba warmup skipped (non-critical)"

# ============================================================================
# Run Exploration Campaign
# ============================================================================

echo ""
echo "[5/6] Starting exploration campaign..."
echo "============================================"

# Set output directory for this job
export CALIGO_OUTPUT_DIR="${CALIGO_SCRATCH_DIR}"

# Run with HPC configuration
# --workers overrides config if specified
python main_explor.py \
    --config "$CONFIG_FILE" \
    --workers 172 \
    2>&1 | tee "${CALIGO_SCRATCH_DIR}/logs/campaign.log"

CAMPAIGN_EXIT_CODE=${PIPESTATUS[0]}

echo "============================================"
echo "Campaign exit code: $CAMPAIGN_EXIT_CODE"

# ============================================================================
# Results Transfer
# ============================================================================

echo ""
echo "[6/6] Transferring results to persistent storage..."

# Create results directory
RESULTS_DIR="${RESULTS_BASE}/run_${SLURM_JOB_ID}"
mkdir -p "$RESULTS_DIR"

# Copy results (preserve timestamps)
cp -rp "${CALIGO_SCRATCH_DIR}"/* "$RESULTS_DIR/" || {
    echo "WARNING: Results copy failed, attempting rsync..."
    rsync -av "${CALIGO_SCRATCH_DIR}/" "$RESULTS_DIR/"
}

# Verify copy
if [ -f "${RESULTS_DIR}/exploration_data.h5" ]; then
    echo "✓ Results transferred to: $RESULTS_DIR"
    
    # Show summary
    ls -lh "${RESULTS_DIR}/"
    
    # Show HDF5 file size
    HDF5_SIZE=$(du -h "${RESULTS_DIR}/exploration_data.h5" | cut -f1)
    echo "  HDF5 data: $HDF5_SIZE"
else
    echo "WARNING: HDF5 file not found in results!"
fi

# ============================================================================
# Cleanup
# ============================================================================

# Optionally clean scratch (comment out to keep for debugging)
# rm -rf "$CALIGO_SCRATCH_DIR"

# ============================================================================
# Summary
# ============================================================================

echo ""
echo "============================================"
echo "Job Complete"
echo "============================================"
echo "End Time:      $(date '+%Y-%m-%d %H:%M:%S')"
echo "Exit Code:     $CAMPAIGN_EXIT_CODE"
echo "Results:       $RESULTS_DIR"
echo "============================================"

exit $CAMPAIGN_EXIT_CODE
