"""
Table generation utilities for Caligo Exploration results.

This module provides functions to generate publication-quality tables
summarizing the exploration campaign findings.

Tables Generated
----------------
1. Death Valley Table: Minimum viable block size per QBER level
2. Zone of Feasibility Table: Maximum distance per adversary strength
3. Strategy Selection Table: Recommended strategy per scenario
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from caligo.vis_tables.metrics import (
    compute_qber_from_array,
    compute_theoretical_min_n,
    compute_max_distance_km,
    samples_to_dataframe,
)
from caligo.simulation.constants import (
    QBER_CONSERVATIVE_LIMIT,
)


# =============================================================================
# Death Valley Table (Minimum Block Size)
# =============================================================================


def generate_death_valley_table(
    df: pd.DataFrame,
    qber_levels: List[float] = None,
    output_path: Optional[Path] = None,
    formats: List[str] = None,
) -> pd.DataFrame:
    """
    Generate the Death Valley lookup table.

    This table maps QBER levels to minimum viable block sizes,
    comparing theoretical bounds with empirical Caligo results.

    Parameters
    ----------
    df : pd.DataFrame
        Results DataFrame from samples_to_dataframe.
    qber_levels : List[float]
        QBER levels to evaluate (e.g., [0.01, 0.03, 0.05, 0.08, 0.10]).
    output_path : Optional[Path]
        Directory to save output files.
    formats : List[str]
        Output formats: csv, markdown, latex.

    Returns
    -------
    pd.DataFrame
        Death Valley table with columns:
        - qber: Channel QBER
        - min_n_theoretical: Theoretical minimum N
        - min_n_empirical: Empirical minimum N from Caligo
        - safety_factor: Ratio of empirical to theoretical
        - success_rate: Success rate at this QBER
    """
    if qber_levels is None:
        qber_levels = [0.01, 0.03, 0.05, 0.08, 0.10]
    
    if formats is None:
        formats = ["csv", "markdown"]
    
    results = []
    
    for qber_target in qber_levels:
        # Theoretical minimum
        min_n_theory = compute_theoretical_min_n(qber_target)
        
        # Filter samples near this QBER
        qber_tol = 0.005
        mask = (df["qber"] >= qber_target - qber_tol) & (df["qber"] < qber_target + qber_tol)
        subset = df[mask]
        
        if len(subset) == 0:
            results.append({
                "qber_percent": qber_target * 100,
                "min_n_theoretical": min_n_theory if min_n_theory else "N/A",
                "min_n_empirical": "NO DATA",
                "safety_factor": "N/A",
                "success_rate_percent": "N/A",
            })
            continue
        
        # Find empirical minimum N for success
        successful = subset[subset["is_success"]]
        
        if len(successful) == 0:
            min_n_empirical = "FAIL"
            safety_factor = "N/A"
        else:
            min_n_empirical = int(successful["num_pairs"].min())
            if min_n_theory and min_n_theory > 0:
                safety_factor = f"{min_n_empirical / min_n_theory:.1f}x"
            else:
                safety_factor = "N/A"
        
        success_rate = len(successful) / len(subset) * 100
        
        results.append({
            "qber_percent": qber_target * 100,
            "min_n_theoretical": f"{min_n_theory:,}" if min_n_theory else "N/A",
            "min_n_empirical": f"{min_n_empirical:,}" if isinstance(min_n_empirical, int) else min_n_empirical,
            "safety_factor": safety_factor,
            "success_rate_percent": f"{success_rate:.1f}%",
        })
    
    table = pd.DataFrame(results)
    
    # Save outputs
    if output_path:
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if "csv" in formats:
            table.to_csv(output_path / "death_valley_table.csv", index=False)
        
        if "markdown" in formats:
            _save_markdown_table(
                table,
                output_path / "death_valley_table.md",
                title="Death Valley Lookup Table (Minimum Viable Block Size)",
            )
        
        if "latex" in formats:
            table.to_latex(output_path / "death_valley_table.tex", index=False)
    
    return table


# =============================================================================
# Zone of Feasibility Table
# =============================================================================


def generate_feasibility_table(
    df: pd.DataFrame,
    storage_noise_levels: List[float] = None,
    dark_count_levels: List[float] = None,
    output_path: Optional[Path] = None,
    formats: List[str] = None,
) -> pd.DataFrame:
    """
    Generate the Zone of Feasibility matrix.

    This table shows maximum permissible distance for given
    adversary strength (storage noise) and detector quality.

    Parameters
    ----------
    df : pd.DataFrame
        Results DataFrame.
    storage_noise_levels : List[float]
        Storage noise r values (e.g., [0.90, 0.85, 0.80, 0.75]).
    dark_count_levels : List[float]
        Dark count probabilities (e.g., [1e-6, 1e-5, 1e-4]).
    output_path : Optional[Path]
        Directory to save output files.
    formats : List[str]
        Output formats.

    Returns
    -------
    pd.DataFrame
        Feasibility table with columns:
        - storage_noise_r: Adversary storage noise
        - dark_count_prob: Detector dark count rate
        - max_distance_km: Maximum feasible distance
        - limiting_factor: What limits the distance
    """
    if storage_noise_levels is None:
        storage_noise_levels = [0.90, 0.85, 0.80, 0.75]
    
    if dark_count_levels is None:
        dark_count_levels = [1e-6, 1e-5, 1e-4]
    
    if formats is None:
        formats = ["csv", "markdown"]
    
    results = []
    
    for r in storage_noise_levels:
        for p_dark in dark_count_levels:
            # Filter data near these parameters
            r_tol = 0.05
            p_tol_log = 0.5
            
            r_mask = (df["storage_noise_r"] >= r - r_tol) & (df["storage_noise_r"] < r + r_tol)
            p_log = np.log10(p_dark)
            p_mask = (np.log10(df["dark_count_prob"]) >= p_log - p_tol_log) & \
                     (np.log10(df["dark_count_prob"]) < p_log + p_tol_log)
            
            subset = df[r_mask & p_mask]
            
            # Analyze success by distance
            if len(subset) > 0:
                successful = subset[subset["is_success"]]
                if len(successful) > 0:
                    max_dist = successful["distance_km"].max()
                    
                    # Determine limiting factor
                    failed = subset[~subset["is_success"]]
                    if len(failed) > 0:
                        avg_qber_fail = failed["qber"].mean()
                        if avg_qber_fail > QBER_CONSERVATIVE_LIMIT:
                            limiting = "QBER Threshold"
                        elif r < 0.8:
                            limiting = "NSM Security Bound"
                        else:
                            limiting = "Signal-to-Noise"
                    else:
                        limiting = "Not Limited"
                else:
                    max_dist = 0
                    limiting = "Always Insecure"
            else:
                # Use theoretical estimate
                max_dist = compute_max_distance_km(
                    storage_noise_r=r,
                    dark_count_prob=p_dark,
                )
                limiting = "Theoretical"
            
            # Format adversary strength
            adv_strength = "Weak" if r >= 0.85 else ("Moderate" if r >= 0.75 else "Strong")
            
            results.append({
                "storage_noise_r": f"{r:.2f}",
                "adversary_strength": adv_strength,
                "dark_count_prob": f"{p_dark:.0e}",
                "max_distance_km": f"{max_dist:.0f} km" if max_dist > 0 else "0 km (Insecure)",
                "limiting_factor": limiting,
            })
    
    table = pd.DataFrame(results)
    
    # Save outputs
    if output_path:
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if "csv" in formats:
            table.to_csv(output_path / "feasibility_table.csv", index=False)
        
        if "markdown" in formats:
            _save_markdown_table(
                table,
                output_path / "feasibility_table.md",
                title="Zone of Feasibility Matrix",
            )
        
        if "latex" in formats:
            table.to_latex(output_path / "feasibility_table.tex", index=False)
    
    return table


# =============================================================================
# Strategy Selection Table
# =============================================================================


def generate_strategy_table(
    df: pd.DataFrame,
    output_path: Optional[Path] = None,
    formats: List[str] = None,
) -> pd.DataFrame:
    """
    Generate strategy selection guidelines.

    This table recommends baseline vs blind strategy based on
    scenario conditions derived from the exploration data.

    Parameters
    ----------
    df : pd.DataFrame
        Results DataFrame.
    output_path : Optional[Path]
        Output directory.
    formats : List[str]
        Output formats.

    Returns
    -------
    pd.DataFrame
        Strategy selection table.
    """
    if formats is None:
        formats = ["csv", "markdown"]
    
    results = []
    
    # Scenario 1: Low latency (small N)
    small_n_mask = df["num_pairs"] < 5000
    if small_n_mask.any():
        small_n_data = df[small_n_mask]
        baseline_eff = small_n_data[small_n_data["strategy"] == "baseline"]["net_efficiency"].mean()
        blind_eff = small_n_data[small_n_data["strategy"] == "blind"]["net_efficiency"].mean()
        
        if blind_eff > baseline_eff:
            gain = (blind_eff - baseline_eff) / baseline_eff * 100 if baseline_eff > 0 else 0
            results.append({
                "scenario": "Low Latency (N < 5000)",
                "recommended_strategy": "Blind",
                "expected_gain": f"+{gain:.0f}% Efficiency" if gain > 0 else "Similar",
                "rationale": "Blind adapts better to high QBER variance",
            })
        else:
            results.append({
                "scenario": "Low Latency (N < 5000)",
                "recommended_strategy": "Baseline",
                "expected_gain": "Guaranteed bounds",
                "rationale": "Static rate more stable",
            })
    
    # Scenario 2: High fidelity
    high_f_mask = df["channel_fidelity"] > 0.98
    if high_f_mask.any():
        high_f_data = df[high_f_mask]
        baseline_eff = high_f_data[high_f_data["strategy"] == "baseline"]["net_efficiency"].mean()
        blind_eff = high_f_data[high_f_data["strategy"] == "blind"]["net_efficiency"].mean()
        
        winner = "Baseline" if baseline_eff >= blind_eff else "Blind"
        gain = abs(baseline_eff - blind_eff) / max(baseline_eff, blind_eff, 0.001) * 100
        
        results.append({
            "scenario": "High Fidelity (F > 0.98)",
            "recommended_strategy": winner,
            "expected_gain": f"+{gain:.0f}% Efficiency",
            "rationale": "Low QBER regime favors static rates" if winner == "Baseline" else "Adaptive still beneficial",
        })
    
    # Scenario 3: Burst noise (high dark counts)
    burst_mask = df["dark_count_prob"] > 1e-4
    if burst_mask.any():
        burst_data = df[burst_mask]
        baseline_success = (burst_data["strategy"] == "baseline") & burst_data["is_success"]
        blind_success = (burst_data["strategy"] == "blind") & burst_data["is_success"]
        
        baseline_rate = baseline_success.sum() / (burst_data["strategy"] == "baseline").sum() if (burst_data["strategy"] == "baseline").sum() > 0 else 0
        blind_rate = blind_success.sum() / (burst_data["strategy"] == "blind").sum() if (burst_data["strategy"] == "blind").sum() > 0 else 0
        
        if blind_rate > baseline_rate:
            results.append({
                "scenario": f"Burst Noise (P_dark > 10^-4)",
                "recommended_strategy": "Blind",
                "expected_gain": "Prevents Abort",
                "rationale": "Adaptive handles dark count bursts",
            })
        else:
            results.append({
                "scenario": f"Burst Noise (P_dark > 10^-4)",
                "recommended_strategy": "Either",
                "expected_gain": "Similar performance",
                "rationale": "Both strategies robust here",
            })
    
    # Scenario 4: Critical safety (low margin)
    critical_mask = (df["net_efficiency"] > 0) & (df["net_efficiency"] < 0.05)
    if critical_mask.any():
        critical_data = df[critical_mask]
        baseline_count = (critical_data["strategy"] == "baseline").sum()
        blind_count = (critical_data["strategy"] == "blind").sum()
        
        if baseline_count > blind_count:
            results.append({
                "scenario": "Critical Safety (Margin < 5%)",
                "recommended_strategy": "Baseline",
                "expected_gain": "Guaranteed Leakage Cap",
                "rationale": "Deterministic error correction",
            })
        else:
            results.append({
                "scenario": "Critical Safety (Margin < 5%)",
                "recommended_strategy": "Blind",
                "expected_gain": "Better success rate",
                "rationale": "Adaptive recovery",
            })
    
    table = pd.DataFrame(results)
    
    # Save outputs
    if output_path:
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if "csv" in formats:
            table.to_csv(output_path / "strategy_table.csv", index=False)
        
        if "markdown" in formats:
            _save_markdown_table(
                table,
                output_path / "strategy_table.md",
                title="Strategy Selection Guidelines",
            )
        
        if "latex" in formats:
            table.to_latex(output_path / "strategy_table.tex", index=False)
    
    return table


# =============================================================================
# Summary Statistics Table
# =============================================================================


def generate_summary_table(
    df: pd.DataFrame,
    output_path: Optional[Path] = None,
    formats: List[str] = None,
) -> pd.DataFrame:
    """
    Generate exploration summary statistics.

    Parameters
    ----------
    df : pd.DataFrame
        Results DataFrame.
    output_path : Optional[Path]
        Output directory.
    formats : List[str]
        Output formats.

    Returns
    -------
    pd.DataFrame
        Summary statistics table.
    """
    if formats is None:
        formats = ["csv", "markdown"]
    
    total = len(df)
    successful = df["is_success"].sum()
    
    summary = {
        "Total Samples": total,
        "Successful Executions": successful,
        "Success Rate": f"{successful/total*100:.1f}%",
        "Mean Efficiency (All)": f"{df['net_efficiency'].mean():.4f}",
        "Mean Efficiency (Success)": f"{df[df['is_success']]['net_efficiency'].mean():.4f}" if successful > 0 else "N/A",
        "Max Efficiency": f"{df['net_efficiency'].max():.4f}",
        "Mean QBER": f"{df['qber'].mean():.4f}",
        "QBER Range": f"[{df['qber'].min():.4f}, {df['qber'].max():.4f}]",
        "Baseline Samples": (df["strategy"] == "baseline").sum(),
        "Blind Samples": (df["strategy"] == "blind").sum(),
        "Max Distance (Success)": f"{df[df['is_success']]['distance_km'].max():.1f} km" if successful > 0 else "N/A",
    }
    
    table = pd.DataFrame([summary]).T.reset_index()
    table.columns = ["Metric", "Value"]
    
    if output_path:
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if "csv" in formats:
            table.to_csv(output_path / "summary_table.csv", index=False)
        
        if "markdown" in formats:
            _save_markdown_table(
                table,
                output_path / "summary_table.md",
                title="Exploration Summary Statistics",
            )
    
    return table


# =============================================================================
# Master Table Generator
# =============================================================================


def generate_all_tables(
    X: NDArray[np.floating],
    y: NDArray[np.floating],
    output_path: Path,
    config: Dict[str, Any] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Generate all result tables from exploration data.

    Parameters
    ----------
    X : NDArray[np.floating]
        Input features.
    y : NDArray[np.floating]
        Target values.
    output_path : Path
        Directory for output files.
    config : Dict[str, Any]
        Configuration dictionary with:
        - formats: List of output formats
        - qber_levels: QBER levels for Death Valley
        - storage_noise_levels: Storage noise levels for feasibility

    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary of generated tables.
    """
    if config is None:
        config = {}
    
    output_path = Path(output_path) / "tables"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Convert to DataFrame
    df = samples_to_dataframe(X, y, compute_derived=True)
    
    # Generate all tables
    tables = {}
    
    tables["summary"] = generate_summary_table(
        df=df,
        output_path=output_path,
        formats=config.get("formats", ["csv", "markdown"]),
    )
    
    tables["death_valley"] = generate_death_valley_table(
        df=df,
        qber_levels=config.get("qber_levels", [0.01, 0.03, 0.05, 0.08, 0.10]),
        output_path=output_path,
        formats=config.get("formats", ["csv", "markdown"]),
    )
    
    tables["feasibility"] = generate_feasibility_table(
        df=df,
        storage_noise_levels=config.get("storage_noise_levels", [0.90, 0.85, 0.80, 0.75]),
        output_path=output_path,
        formats=config.get("formats", ["csv", "markdown"]),
    )
    
    tables["strategy"] = generate_strategy_table(
        df=df,
        output_path=output_path,
        formats=config.get("formats", ["csv", "markdown"]),
    )
    
    return tables


# =============================================================================
# Helper Functions
# =============================================================================


def _save_markdown_table(
    df: pd.DataFrame,
    filepath: Path,
    title: str = "",
) -> None:
    """Save DataFrame as markdown table."""
    with open(filepath, "w") as f:
        if title:
            f.write(f"# {title}\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n")
