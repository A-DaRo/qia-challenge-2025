#!/usr/bin/env python3
"""Test post-processing on recent exploration data."""

from pathlib import Path
import numpy as np
from caligo.exploration.persistence import HDF5Writer, hdf5_arrays_to_training_data, GROUP_LHS_WARMUP, GROUP_ACTIVE_LEARNING
from caligo.vis_tables import generate_all_figures, generate_all_tables

# Load data from most recent run
data_path = Path("exploration_results/dry_run_with_viz_20251222_091754/exploration_data.h5")
output_path = Path("exploration_results/dry_run_with_viz_20251222_091754")

print(f"Loading data from: {data_path}")

with HDF5Writer(data_path, mode="r") as reader:
    # Load LHS warmup data
    inputs_lhs, outputs_lhs, outcomes_lhs, _ = reader.read_group(GROUP_LHS_WARMUP)
    X_lhs, y_lhs = hdf5_arrays_to_training_data(inputs_lhs, outputs_lhs, outcomes_lhs)
    
    # Load active learning data if available
    try:
        inputs_al, outputs_al, outcomes_al, _ = reader.read_group(GROUP_ACTIVE_LEARNING)
        X_al, y_al = hdf5_arrays_to_training_data(inputs_al, outputs_al, outcomes_al)
    except (KeyError, Exception) as e:
        print(f"No active learning data: {e}")
        X_al, y_al = np.array([]).reshape(0, 9), np.array([])

# Combine all data
X = np.vstack([X_lhs, X_al]) if len(X_al) > 0 else X_lhs
y = np.concatenate([y_lhs, y_al]) if len(y_al) > 0 else y_lhs

print(f"Loaded {len(X)} samples for post-processing")

# Generate visualizations
print("\n" + "="*70)
print("GENERATING VISUALIZATIONS")
print("="*70)

try:
    figures = generate_all_figures(
        X=X,
        y=y,
        output_path=output_path,
        phase_metrics=None,
        show=False,
    )
    print(f"✓ Generated {len(figures)} figures")
except Exception as e:
    print(f"✗ Visualization failed: {e}")
    import traceback
    traceback.print_exc()

# Generate tables
print("\n" + "="*70)
print("GENERATING TABLES")
print("="*70)

try:
    tables = generate_all_tables(
        X=X,
        y=y,
        output_path=output_path,
        config={
            "formats": ["csv", "markdown"],
            "qber_levels": [0.01, 0.03, 0.05, 0.08, 0.10],
            "storage_noise_levels": [0.90, 0.85, 0.80, 0.75],
        },
    )
    print(f"✓ Generated {len(tables)} tables")
except Exception as e:
    print(f"✗ Table generation failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print(f"Results saved to: {output_path}")
print("="*70)
