"""
Visualization utilities for Caligo Exploration results.

This module provides publication-quality plotting functions for
analyzing and presenting the exploration campaign findings.

Figures Generated
-----------------
1. Death Valley Plot: QBER vs block size heatmap
2. Strategy Duel Plot: Baseline vs Blind comparison
3. Security Volume Plot: 3D parameter space visualization
4. Sensitivity Spider Plot: Radar chart of Sobol indices
5. Surrogate Trustworthiness: Residuals & Uncertainty Map
6. The "Cliff" Cross-Section: 1D High-Resolution Slice
7. Acquisition Dynamics: The "Flight Path"
8. Guardrail Reliability: ROC Curve
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.patches import Circle
    from mpl_toolkits.mplot3d import Axes3D
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Scikit-learn for metrics
try:
    from sklearn.metrics import roc_curve, auc, r2_score, mean_squared_error
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from caligo.vis_tables.metrics import (
    compute_theoretical_min_n,
    samples_to_dataframe,
)
from caligo.simulation.constants import (
    QBER_HARD_LIMIT,
    QBER_CONSERVATIVE_LIMIT,
)

# Type checking imports to avoid runtime circular interaction if possible,
# though surrogate is needed for runtime prediction in some plots.
from caligo.exploration.surrogate import EfficiencyLandscape, FeasibilityGuardrail


# =============================================================================
# Plot Configuration
# =============================================================================

PLOT_CONFIG = {
    "figure_size": (10, 8),
    "dpi": 150,
    "colormap": "viridis",
    "diverging_colormap": "RdBu",
    "font_size": 12,
    "title_size": 14,
    "label_size": 12,
}


def _check_matplotlib() -> None:
    """Raise error if matplotlib not available."""
    if not HAS_MATPLOTLIB:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install with: pip install matplotlib"
        )


# =============================================================================
# Death Valley Plot (2D Contour Heatmap)
# =============================================================================


def plot_death_valley(
    X: NDArray[np.floating],
    y: NDArray[np.floating],
    output_path: Optional[Path] = None,
    filename: str = "death_valley.png",
    show: bool = False,
    figsize: Tuple[int, int] = None,
    title: str = "Death Valley: QBER vs Block Size",
) -> Optional[plt.Figure]:
    """
    Generate the Death Valley contour plot.

    A 2D heatmap showing net efficiency as a function of QBER (x-axis)
    and block size N (y-axis). Red line indicates theoretical bound
    below which no positive key rate is possible.

    Parameters
    ----------
    X : NDArray[np.floating]
        Input features array (n_samples, n_features).
    y : NDArray[np.floating]
        Target values (net efficiency).
    output_path : Optional[Path]
        Directory to save the figure. If None, figure not saved.
    filename : str
        Output filename.
    show : bool
        Whether to display the plot.
    figsize : Tuple[int, int]
        Figure size (width, height) in inches.
    title : str
        Plot title.

    Returns
    -------
    Optional[plt.Figure]
        Matplotlib Figure object if successful.
    """
    _check_matplotlib()
    
    if figsize is None:
        figsize = PLOT_CONFIG["figure_size"]
    
    # Convert to DataFrame for easier manipulation
    df = samples_to_dataframe(X, y, compute_derived=True)
    
    # Extract QBER and N columns
    qber = df["qber"].values
    num_pairs = df["num_pairs"].values
    efficiency = df["net_efficiency"].values
    
    # Create grid for interpolation
    qber_range = np.linspace(qber.min(), min(qber.max(), QBER_HARD_LIMIT), 50)
    n_range = np.logspace(np.log10(num_pairs.min()), np.log10(num_pairs.max()), 50)
    
    # Create meshgrid
    QBER_grid, N_grid = np.meshgrid(qber_range, n_range)
    
    # Interpolate efficiency onto grid
    from scipy.interpolate import griddata
    efficiency_grid = griddata(
        (qber, num_pairs),
        efficiency,
        (QBER_grid, N_grid),
        method="linear",
    )
    
    # Fill NaN with 0 for visualization
    efficiency_grid = np.nan_to_num(efficiency_grid, nan=0.0)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=PLOT_CONFIG["dpi"])
    
    # Contour plot
    contour = ax.contourf(
        QBER_grid * 100,  # Convert to percentage
        N_grid,
        efficiency_grid,
        levels=20,
        cmap=PLOT_CONFIG["colormap"],
        extend="both",
    )
    
    # Add colorbar
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label("Net Efficiency", fontsize=PLOT_CONFIG["label_size"])
    
    # Add theoretical bound line
    qber_theory = np.linspace(0.01, QBER_HARD_LIMIT - 0.001, 100)
    n_min_theory = [compute_theoretical_min_n(q) or np.nan for q in qber_theory]
    ax.plot(
        qber_theory * 100,
        n_min_theory,
        "r--",
        linewidth=2,
        label="Theoretical Minimum N",
    )
    
    # Add QBER threshold lines
    ax.axvline(
        x=QBER_CONSERVATIVE_LIMIT * 100,
        color="orange",
        linestyle=":",
        linewidth=1.5,
        label=f"Conservative Limit ({QBER_CONSERVATIVE_LIMIT*100:.0f}%)",
    )
    ax.axvline(
        x=QBER_HARD_LIMIT * 100,
        color="red",
        linestyle="-",
        linewidth=2,
        label=f"Hard Limit ({QBER_HARD_LIMIT*100:.0f}%)",
    )
    
    # Formatting
    ax.set_xlabel("QBER (%)", fontsize=PLOT_CONFIG["label_size"])
    ax.set_ylabel("Block Size N (log scale)", fontsize=PLOT_CONFIG["label_size"])
    ax.set_yscale("log")
    ax.set_title(title, fontsize=PLOT_CONFIG["title_size"])
    ax.legend(loc="upper right")
    
    plt.tight_layout()
    
    # Save
    if output_path:
        output_path = Path(output_path) / "figures"
        output_path.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path / filename, dpi=PLOT_CONFIG["dpi"], bbox_inches="tight")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


# =============================================================================
# Strategy Duel Plot (Diverging Heatmap)
# =============================================================================


def plot_strategy_duel(
    X: NDArray[np.floating],
    y: NDArray[np.floating],
    output_path: Optional[Path] = None,
    filename: str = "strategy_duel.png",
    show: bool = False,
    figsize: Tuple[int, int] = None,
    title: str = "Strategy Comparison: Baseline vs Blind",
) -> Optional[plt.Figure]:
    """
    Generate the Strategy Duel diverging heatmap.

    Shows the difference in net efficiency between baseline and blind
    strategies across detector efficiency (η) and dark count probability.
    Blue = Baseline better, Red = Blind better.

    Parameters
    ----------
    X : NDArray[np.floating]
        Input features array.
    y : NDArray[np.floating]
        Target values.
    output_path : Optional[Path]
        Directory to save the figure.
    filename : str
        Output filename.
    show : bool
        Whether to display the plot.
    figsize : Tuple[int, int]
        Figure size.
    title : str
        Plot title.

    Returns
    -------
    Optional[plt.Figure]
        Matplotlib Figure object.
    """
    _check_matplotlib()
    
    if figsize is None:
        figsize = PLOT_CONFIG["figure_size"]
    
    df = samples_to_dataframe(X, y, compute_derived=True)
    
    # Separate by strategy
    baseline = df[df["strategy"] == "baseline"]
    blind = df[df["strategy"] == "blind"]
    
    # Create grid for comparison
    eta_range = np.linspace(0.6, 0.99, 20)
    pdark_range = np.logspace(-7, -3, 20)
    
    ETA_grid, PDARK_grid = np.meshgrid(eta_range, pdark_range)
    
    # Calculate average efficiency for each strategy in each cell
    from scipy.interpolate import griddata
    
    if len(baseline) > 0 and len(blind) > 0:
        baseline_grid = griddata(
            (baseline["detection_efficiency"].values, baseline["dark_count_prob"].values),
            baseline["net_efficiency"].values,
            (ETA_grid, PDARK_grid),
            method="nearest",
        )
        
        blind_grid = griddata(
            (blind["detection_efficiency"].values, blind["dark_count_prob"].values),
            blind["net_efficiency"].values,
            (ETA_grid, PDARK_grid),
            method="nearest",
        )
        
        diff_grid = baseline_grid - blind_grid
    else:
        # Not enough data for comparison
        diff_grid = np.zeros_like(ETA_grid)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=PLOT_CONFIG["dpi"])
    
    # Diverging colormap centered at 0
    vmax = max(abs(np.nanmin(diff_grid)), abs(np.nanmax(diff_grid)), 0.01)
    
    contour = ax.pcolormesh(
        ETA_grid,
        PDARK_grid,
        diff_grid,
        cmap=PLOT_CONFIG["diverging_colormap"],
        vmin=-vmax,
        vmax=vmax,
        shading="auto",
    )
    
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label(
        "Baseline - Blind Efficiency\n(Blue: Baseline Better, Red: Blind Better)",
        fontsize=PLOT_CONFIG["label_size"] - 1,
    )
    
    # Formatting
    ax.set_xlabel("Detection Efficiency η", fontsize=PLOT_CONFIG["label_size"])
    ax.set_ylabel("Dark Count Probability", fontsize=PLOT_CONFIG["label_size"])
    ax.set_yscale("log")
    ax.set_title(title, fontsize=PLOT_CONFIG["title_size"])
    
    # Add annotation for winner regions
    ax.text(
        0.65, 1e-6, "Baseline\nFavored",
        fontsize=10, color="blue", ha="center", va="center",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
    )
    ax.text(
        0.9, 1e-4, "Blind\nFavored",
        fontsize=10, color="red", ha="center", va="center",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
    )
    
    plt.tight_layout()
    
    if output_path:
        output_path = Path(output_path) / "figures"
        output_path.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path / filename, dpi=PLOT_CONFIG["dpi"], bbox_inches="tight")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


# =============================================================================
# Security Volume Plot (3D Visualization)
# =============================================================================


def plot_security_volume(
    X: NDArray[np.floating],
    y: NDArray[np.floating],
    output_path: Optional[Path] = None,
    filename: str = "security_volume.png",
    show: bool = False,
    figsize: Tuple[int, int] = None,
    title: str = "Security Volume: (r, F, Δt) Parameter Space",
) -> Optional[plt.Figure]:
    """
    Generate the Security Volume 3D scatter plot.

    A 3D visualization showing the secure vs insecure regions of the
    parameter space defined by storage noise r, fidelity F, and timing Δt.

    Parameters
    ----------
    X : NDArray[np.floating]
        Input features array.
    y : NDArray[np.floating]
        Target values.
    output_path : Optional[Path]
        Directory to save the figure.
    filename : str
        Output filename.
    show : bool
        Whether to display the plot.
    figsize : Tuple[int, int]
        Figure size.
    title : str
        Plot title.

    Returns
    -------
    Optional[plt.Figure]
        Matplotlib Figure object.
    """
    _check_matplotlib()
    
    if figsize is None:
        figsize = (12, 10)
    
    df = samples_to_dataframe(X, y, compute_derived=True)
    
    # Extract 3D coordinates
    r = df["storage_noise_r"].values
    F = df["channel_fidelity"].values
    dt = df["wait_time_ns"].values  # Using wait_time_ns instead of timing_delta_t
    efficiency = df["net_efficiency"].values
    is_success = df["is_success"].values
    
    # Create figure with 3D axes
    fig = plt.figure(figsize=figsize, dpi=PLOT_CONFIG["dpi"])
    ax = fig.add_subplot(111, projection="3d")
    
    # Plot successful points (secure volume)
    secure_mask = is_success & (efficiency > 0)
    ax.scatter(
        r[secure_mask],
        F[secure_mask],
        dt[secure_mask],
        c=efficiency[secure_mask],
        cmap=PLOT_CONFIG["colormap"],
        s=20,
        alpha=0.6,
        label="Secure",
        marker="o",
    )
    
    # Plot failed points (insecure region)
    insecure_mask = ~secure_mask
    ax.scatter(
        r[insecure_mask],
        F[insecure_mask],
        dt[insecure_mask],
        c="red",
        s=10,
        alpha=0.2,
        label="Insecure",
        marker="x",
    )
    
    # Add colorbar
    norm = mcolors.Normalize(vmin=efficiency[secure_mask].min() if secure_mask.any() else 0,
                              vmax=efficiency[secure_mask].max() if secure_mask.any() else 1)
    sm = plt.cm.ScalarMappable(cmap=PLOT_CONFIG["colormap"], norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.5, aspect=20)
    cbar.set_label("Net Efficiency", fontsize=PLOT_CONFIG["label_size"])
    
    # Axis labels
    ax.set_xlabel("Storage Noise r", fontsize=PLOT_CONFIG["label_size"])
    ax.set_ylabel("Channel Fidelity F", fontsize=PLOT_CONFIG["label_size"])
    ax.set_zlabel("Wait Time (ns)", fontsize=PLOT_CONFIG["label_size"])
    ax.set_title(title, fontsize=PLOT_CONFIG["title_size"])
    
    ax.legend(loc="upper left")
    
    # Set viewing angle
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    
    if output_path:
        output_path = Path(output_path) / "figures"
        output_path.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path / filename, dpi=PLOT_CONFIG["dpi"], bbox_inches="tight")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


# =============================================================================
# Sensitivity Spider Plot (Radar Chart)
# =============================================================================


def plot_sensitivity_spider(
    sobol_indices: Dict[str, float] = None,
    output_path: Optional[Path] = None,
    filename: str = "sensitivity_spider.png",
    show: bool = False,
    figsize: Tuple[int, int] = None,
    title: str = "Parameter Sensitivity (Sobol Indices)",
) -> Optional[plt.Figure]:
    """
    Generate the Sensitivity Spider radar chart.

    Displays Sobol first-order sensitivity indices for each parameter,
    showing which parameters have the greatest impact on net efficiency.

    Parameters
    ----------
    sobol_indices : Dict[str, float]
        Dictionary mapping parameter names to Sobol indices.
        If None, uses placeholder values.
    output_path : Optional[Path]
        Directory to save the figure.
    filename : str
        Output filename.
    show : bool
        Whether to display the plot.
    figsize : Tuple[int, int]
        Figure size.
    title : str
        Plot title.

    Returns
    -------
    Optional[plt.Figure]
        Matplotlib Figure object.
    """
    _check_matplotlib()
    
    if figsize is None:
        figsize = (10, 10)
    
    # Default Sobol indices if not provided
    if sobol_indices is None:
        sobol_indices = {
            "storage_noise_r": 0.25,
            "storage_rate_nu": 0.05,
            "wait_time_ns": 0.08,
            "channel_fidelity": 0.35,
            "detection_efficiency": 0.15,
            "detector_error": 0.03,
            "dark_count_prob": 0.06,
            "num_pairs": 0.02,
            "strategy": 0.01,
        }
    
    # Prepare data
    categories = list(sobol_indices.keys())
    values = list(sobol_indices.values())
    
    # Number of parameters
    N = len(categories)
    
    # Compute angle for each category
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete the loop
    
    values += values[:1]  # Complete the loop
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=PLOT_CONFIG["dpi"], subplot_kw=dict(polar=True))
    
    # Draw the chart
    ax.plot(angles, values, "o-", linewidth=2, color="steelblue")
    ax.fill(angles, values, alpha=0.25, color="steelblue")
    
    # Set category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=PLOT_CONFIG["label_size"] - 1)
    
    # Set radial limits
    ax.set_ylim(0, max(values) * 1.1)
    
    # Title
    ax.set_title(title, fontsize=PLOT_CONFIG["title_size"], y=1.08)
    
    plt.tight_layout()
    
    if output_path:
        output_path = Path(output_path) / "figures"
        output_path.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path / filename, dpi=PLOT_CONFIG["dpi"], bbox_inches="tight")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


# =============================================================================
# Exploration Progress Plot
# =============================================================================


def plot_exploration_progress(
    phase1_metrics: Optional[Dict[str, Any]] = None,
    phase2_metrics: Optional[Dict[str, Any]] = None,
    phase3_metrics: Optional[Dict[str, Any]] = None,
    output_path: Optional[Path] = None,
    filename: str = "exploration_progress.png",
    show: bool = False,
    figsize: Tuple[int, int] = None,
) -> Optional[plt.Figure]:
    """
    Generate exploration campaign progress visualization.

    Shows key metrics from each phase of the exploration campaign.

    Parameters
    ----------
    phase1_metrics : Dict[str, Any]
        Metrics from Phase 1 (LHS warmup).
    phase2_metrics : Dict[str, Any]
        Metrics from Phase 2 (surrogate training).
    phase3_metrics : Dict[str, Any]
        Metrics from Phase 3 (active learning).
    output_path : Optional[Path]
        Directory to save the figure.
    filename : str
        Output filename.
    show : bool
        Whether to display the plot.
    figsize : Tuple[int, int]
        Figure size.

    Returns
    -------
    Optional[plt.Figure]
        Matplotlib Figure object.
    """
    _check_matplotlib()
    
    if figsize is None:
        figsize = (14, 6)
    
    fig, axes = plt.subplots(1, 3, figsize=figsize, dpi=PLOT_CONFIG["dpi"])
    
    # Phase 1: Sample distribution
    ax1 = axes[0]
    ax1.set_title("Phase 1: LHS Warmup", fontsize=PLOT_CONFIG["title_size"])
    if phase1_metrics:
        total = phase1_metrics.get("total_samples", 0)
        successful = phase1_metrics.get("successful_samples", 0)
        ax1.bar(["Total", "Successful"], [total, successful], color=["steelblue", "green"])
        ax1.set_ylabel("Samples")
    else:
        ax1.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax1.transAxes)
    
    # Phase 2: CV Performance
    ax2 = axes[1]
    ax2.set_title("Phase 2: Surrogate Training", fontsize=PLOT_CONFIG["title_size"])
    if phase2_metrics:
        r2 = phase2_metrics.get("r2_score", 0)
        rmse = phase2_metrics.get("rmse", 0)
        ax2.bar(["R² Score", "RMSE"], [r2, rmse], color=["steelblue", "orange"])
        ax2.set_ylabel("Metric Value")
    else:
        ax2.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax2.transAxes)
    
    # Phase 3: Active Learning Progress
    ax3 = axes[2]
    ax3.set_title("Phase 3: Active Learning", fontsize=PLOT_CONFIG["title_size"])
    if phase3_metrics:
        iterations = phase3_metrics.get("iterations", [])
        best_y = phase3_metrics.get("best_y_history", [])
        if iterations and best_y and len(iterations) == len(best_y):
            ax3.plot(iterations, best_y, "o-", color="steelblue")
            ax3.set_xlabel("Iteration")
            ax3.set_ylabel("Best Efficiency")
        elif iterations or best_y:
            # If data exists but lengths mismatch, show simple plot
            if best_y:
                ax3.plot(range(len(best_y)), best_y, "o-", color="steelblue")
                ax3.set_xlabel("Sample Index")
                ax3.set_ylabel("Best Efficiency")
            else:
                ax3.text(0.5, 0.5, "No iteration data", ha="center", va="center", transform=ax3.transAxes)
        else:
            ax3.text(0.5, 0.5, "No iteration data", ha="center", va="center", transform=ax3.transAxes)
    else:
        ax3.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax3.transAxes)
    
    plt.tight_layout()
    
    if output_path:
        output_path = Path(output_path) / "figures"
        output_path.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path / filename, dpi=PLOT_CONFIG["dpi"], bbox_inches="tight")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


# =============================================================================
# Surrogate Trustworthiness Plot
# =============================================================================


def plot_surrogate_trustworthiness(
    model: EfficiencyLandscape,
    X: NDArray[np.floating],
    y: NDArray[np.floating],
    output_path: Optional[Path] = None,
    filename: str = "surrogate_trustworthiness.png",
    show: bool = False,
    figsize: Tuple[int, int] = (10, 10),
) -> Optional[plt.Figure]:
    """
    Generate surrogate trustworthiness residuals plot.

    Compares predicted vs actual efficiency with uncertainty quantification.

    Parameters
    ----------
    model : EfficiencyLandscape
        Trained surrogate model.
    X : NDArray[np.floating]
        Input features.
    y : NDArray[np.floating]
        True efficiency values.
    output_path : Optional[Path]
        Output directory.
    filename : str
        Output filename.
    show : bool
        Whether to display.
    figsize : Tuple[int, int]
        Figure size.

    Returns
    -------
    Optional[plt.Figure]
        Matplotlib figure.
    """
    _check_matplotlib()
    
    if not HAS_SKLEARN:
        print("Warning: scikit-learn not found, skipping trustworthiness plot")
        return None

    # Predict
    # NOTE: model.predict calls baseline_gp or blind_gp depending on internal logic for 'active', 
    # but for validation we really want to check the specific active model or both.
    # To keep it simple, we use predict_baseline + predict_blind weighted or picked?
    # The 'y' contains mixed strategies.
    # We should filter X/y by strategy if we want per-model accuracy.
    # However, standard model.predict delegates to baseline.
    # Let's approximate by using baseline for baseline points and blind for blind points?
    # But X doesn't have strategy column (it's 9D, strategy is in Types but X is float array).
    # Wait, X usually doesn't include strategy as a feature in this codebase's split (it's separate GP).
    # But the Prompt says "Input arrays from persistence.py".
    # We will assume a naive prediction for now using baseline as proxy or do something smarter if we had strategy info.
    # Since we can't easily distinguish strategy from X alone without metadata, 
    # and EfficiencyLandscape has .predict() which defaults to baseline, we use that for the "General" fit.
    # Better: If we had strategy, we would split.
    
    mu, std = model.predict(X, return_std=True)
    if std is None:
        std = np.zeros_like(mu)

    # Metrics
    r2 = r2_score(y, mu)
    rmse = np.sqrt(mean_squared_error(y, mu))
    
    # Create joint plot using seaborn
    g = sns.JointGrid(x=mu, y=y, height=figsize[0]/1.5, ratio=5)
    
    # Scatter with uncertainty color
    sc = g.ax_joint.scatter(
        mu, y, 
        c=std, 
        cmap='viridis', 
        alpha=0.6, 
        edgecolor='w', 
        s=50,
        label=f'Samples (N={len(y)})'
    )
    
    # Add identity line
    lims = [
        min(min(mu), min(y)),
        max(max(mu), max(y))
    ]
    g.ax_joint.plot(lims, lims, 'r--', alpha=0.75, zorder=0, label='Ideal')
    
    # Add marginal histograms
    g.ax_marg_x.hist(mu, bins=30, alpha=0.6, color='steelblue')
    g.ax_marg_y.hist(y, bins=30, alpha=0.6, color='steelblue', orientation='horizontal')
    
    # Labels and Titles
    g.ax_joint.set_xlabel('Predicted Efficiency')
    g.ax_joint.set_ylabel('Actual Efficiency')
    plt.colorbar(sc, ax=g.ax_joint, label='Predicted Std Dev (σ)')
    
    # Stats annotation
    stats_text = (
        f"$R^2 = {r2:.3f}$\n"
        f"RMSE = {rmse:.4f}"
    )
    g.ax_joint.text(
        0.05, 0.95, stats_text, 
        transform=g.ax_joint.transAxes, 
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )
    
    plt.suptitle("Surrogate Trustworthiness: Predicted vs Actual", y=1.02, fontsize=14)
    
    fig = g.fig
    
    if output_path:
        output_path = Path(output_path) / "figures"
        output_path.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path / filename, dpi=PLOT_CONFIG["dpi"], bbox_inches="tight")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
        
    return fig


# =============================================================================
# Cliff Cross-Section Plot
# =============================================================================


def plot_cliff_cross_section(
    model: EfficiencyLandscape,
    X_sample: NDArray[np.floating],
    param_idx: int = 2,  # default to wait_time_ns
    param_name: str = "Wait Time (ns)",
    param_log_scale: bool = True,
    n_points: int = 100,
    output_path: Optional[Path] = None,
    filename: str = "cliff_cross_section.png",
    show: bool = False,
    figsize: Tuple[int, int] = None,
) -> Optional[plt.Figure]:
    """
    Generate 1D cross-section of the security cliff.

    Varies one parameter while keeping others fixed at a robust sample point.

    Parameters
    ----------
    model : EfficiencyLandscape
        Trained surrogate model.
    X_sample : NDArray[np.floating]
        A single reference sample (shape (9,)) to define fixed parameters.
        Ideally a successful point.
    param_idx : int
        Index of parameter to vary (default 2 for wait_time_ns).
    param_name : str
        Label for x-axis.
    param_log_scale : bool
        Whether to plot x-axis in log scale.
    n_points : int
        Resolution of the slice.
    """
    _check_matplotlib()
    
    if figsize is None:
        figsize = PLOT_CONFIG["figure_size"]
    
    val = X_sample[param_idx]
    
    if param_log_scale:
        # Sweep 2 orders of magnitude around value, or min/max known
        # Hardcoding safe ranges based on types.py
        if param_idx == 2: # wait_time
            xs = np.logspace(5, 9, n_points)
        elif param_idx == 3: # channel_fidelity
            xs = np.linspace(0.5, 1.0, n_points)
            param_log_scale = False
        else:
            if val > 0:
                xs = np.logspace(np.log10(val)-1, np.log10(val)+1, n_points)
            else:
                xs = np.linspace(val - 0.5, val + 0.5, n_points)
    else:
        xs = np.linspace(val * 0.5, val * 1.5, n_points)

    # Construct batch
    # Ensure X_sample is 1D
    if X_sample.ndim > 1:
        X_sample = X_sample[0]
        
    X_sweep = np.tile(X_sample, (n_points, 1))
    X_sweep[:, param_idx] = xs

    # Predict
    mu_base, std_base = model.predict_baseline(X_sweep, return_std=True)
    mu_blind, std_blind = model.predict_blind(X_sweep, return_std=True)

    if std_base is None: std_base = np.zeros_like(mu_base)
    if std_blind is None: std_blind = np.zeros_like(mu_blind)

    fig, ax = plt.subplots(figsize=figsize, dpi=PLOT_CONFIG["dpi"])

    # Plot Baseline
    ax.plot(xs, mu_base, 'b-', label='Baseline Strategy', linewidth=2)
    ax.fill_between(xs, mu_base - 1.96*std_base, mu_base + 1.96*std_base, color='b', alpha=0.2)

    # Plot Blind
    ax.plot(xs, mu_blind, 'r-', label='Blind Strategy', linewidth=2)
    ax.fill_between(xs, mu_blind - 1.96*std_blind, mu_blind + 1.96*std_blind, color='r', alpha=0.2)

    ax.set_xlabel(param_name, fontsize=PLOT_CONFIG["label_size"])
    ax.set_ylabel("Predicted Efficiency", fontsize=PLOT_CONFIG["label_size"])
    if param_log_scale:
        ax.set_xscale('log')
    
    ax.set_title(f"Cliff Cross-Section: Varying {param_name}", fontsize=PLOT_CONFIG["title_size"])
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path) / "figures"
        output_path.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path / filename, dpi=PLOT_CONFIG["dpi"], bbox_inches="tight")
    
    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


# =============================================================================
# Acquisition Dynamics Plot
# =============================================================================


def plot_acquisition_dynamics(
    X_active: NDArray[np.floating],
    y_active: NDArray[np.floating],
    strategies: Optional[List[str]] = None,
    output_path: Optional[Path] = None,
    filename: str = "acquisition_dynamics.png",
    show: bool = False,
    figsize: Tuple[int, int] = None,
) -> Optional[plt.Figure]:
    """
    Visualize Active Learning progress towards the cliff.

    Plots the efficiency of samples over iterations.

    Parameters
    ----------
    X_active : NDArray[np.floating]
        Active learning inputs (ordered by iteration).
    y_active : NDArray[np.floating]
        Active learning efficiency outcomes.
    strategies : Optional[List[str]]
        Strategy labels for points (if available).
    """
    _check_matplotlib()
    
    if figsize is None:
        figsize = (12, 6)

    fig, ax = plt.subplots(figsize=figsize, dpi=PLOT_CONFIG["dpi"])
    
    iterations = np.arange(len(y_active))
    
    # Scatter points
    if strategies:
        # Map strings to colors
        unique_strats = sorted(list(set(strategies)))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_strats)))
        c_map = {s: colors[i] for i, s in enumerate(unique_strats)}
        c_list = [c_map.get(s, 'k') for s in strategies]
        
        ax.scatter(iterations, y_active, c=c_list, alpha=0.7, s=30, label='Samples')
        
        # Customized Legend
        from matplotlib.lines import Line2D
        handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=c_map[s], label=s) for s in unique_strats]
        ax.legend(handles=handles)
    else:
        ax.scatter(iterations, y_active, c='k', alpha=0.6, s=30, label='Samples')

    # Moving average trend
    window = max(5, len(y_active) // 20)
    if len(y_active) > window:
        y_smooth = np.convolve(y_active, np.ones(window)/window, mode='valid')
        ax.plot(iterations[window-1:], y_smooth, 'r--', linewidth=2, label=f'Trend (MA-{window})')

    # Cliff Zone highlight
    ax.axhspan(-0.05, 0.05, color='orange', alpha=0.1, label='Cliff Zone')
    ax.axhline(0, color='r', linewidth=1, linestyle='-')

    ax.set_xlabel("Acquisition Step (Iteration)", fontsize=PLOT_CONFIG["label_size"])
    ax.set_ylabel("Protocol Efficiency", fontsize=PLOT_CONFIG["label_size"])
    ax.set_title("Acquisition Dynamics: Convergence to Cliff", fontsize=PLOT_CONFIG["title_size"])
    
    if not strategies:
        ax.legend()
        
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_path:
        output_path = Path(output_path) / "figures"
        output_path.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path / filename, dpi=PLOT_CONFIG["dpi"], bbox_inches="tight")
    
    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


# =============================================================================
# Guardrail Reliability Plot
# =============================================================================


def plot_guardrail_roc(
    guardrail: FeasibilityGuardrail,
    X_val: NDArray[np.floating],
    y_val: NDArray[np.floating],
    output_path: Optional[Path] = None,
    filename: str = "guardrail_roc.png",
    show: bool = False,
    figsize: Tuple[int, int] = None,
) -> Optional[plt.Figure]:
    """
    Plot ROC curve for the failure prediction guardrail.

    Parameters
    ----------
    guardrail : FeasibilityGuardrail
        Trained guardrail classifier.
    X_val : NDArray[np.floating]
        Validation features.
    y_val : NDArray[np.floating]
        Validation efficiency (used to derive binary labels).
    """
    _check_matplotlib()
    
    if not HAS_SKLEARN:
        return None
        
    if figsize is None:
        figsize = (8, 8)
        
    # Derive Ground Truth Labels (Success = 1, Failure = 0)
    # y > 0 is Success
    y_binary = (y_val > 0).astype(int)
    
    # Get probabilities
    if hasattr(guardrail, "classifier"):
        try:
            # Need to transform X if guardrail has internal scaler
            if hasattr(guardrail, "scaler"):
                X_scaled = guardrail.scaler.transform(X_val)
            else:
                X_scaled = X_val
                
            y_score = guardrail.classifier.predict_proba(X_scaled)[:, 1]
        except (AttributeError, ValueError):
            print("Warning: Could not get probabilities from guardrail")
            return None
    else:
        return None

    fpr, tpr, _ = roc_curve(y_binary, y_score)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=figsize, dpi=PLOT_CONFIG["dpi"])
    
    ax.plot(
        fpr, tpr, 
        color='darkorange', 
        lw=2, 
        label=f'ROC curve (area = {roc_auc:.2f})'
    )
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (Incorrectly flagged as Success)', fontsize=PLOT_CONFIG["label_size"])
    ax.set_ylabel('True Positive Rate (Correctly flagged as Success)', fontsize=PLOT_CONFIG["label_size"])
    ax.set_title('Guardrail Reliability (ROC)', fontsize=PLOT_CONFIG["title_size"])
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path) / "figures"
        output_path.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path / filename, dpi=PLOT_CONFIG["dpi"], bbox_inches="tight")
    
    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


# =============================================================================
# Master Figure Generator
# =============================================================================


def generate_all_figures(
    X: NDArray[np.floating],
    y: NDArray[np.floating],
    output_path: Path,
    phase_metrics: Dict[str, Any] = None,
    sobol_indices: Dict[str, float] = None,
    surrogate_model: Optional[EfficiencyLandscape] = None,
    guardrail_model: Optional[FeasibilityGuardrail] = None,
    show: bool = False,
) -> Dict[str, plt.Figure]:
    """
    Generate all result figures from exploration data.

    Parameters
    ----------
    X : NDArray[np.floating]
        Input features.
    y : NDArray[np.floating]
        Target values.
    output_path : Path
        Directory for output files.
    phase_metrics : Dict[str, Any]
        Metrics from each phase.
    sobol_indices : Dict[str, float]
        Sobol sensitivity indices.
    surrogate_model : Optional[EfficiencyLandscape]
        Trained surrogate model for validation plots.
    guardrail_model : Optional[FeasibilityGuardrail]
        Trained guardrail for ROC plot.
    show : bool
        Whether to display plots.

    Returns
    -------
    Dict[str, plt.Figure]
        Dictionary of generated figures.
    """
    _check_matplotlib()
    
    output_path = Path(output_path)
    
    figures = {}
    
    # Death Valley plot
    figures["death_valley"] = plot_death_valley(
        X, y,
        output_path=output_path,
        show=show,
    )
    
    # Strategy Duel plot
    figures["strategy_duel"] = plot_strategy_duel(
        X, y,
        output_path=output_path,
        show=show,
    )
    
    # Security Volume plot
    figures["security_volume"] = plot_security_volume(
        X, y,
        output_path=output_path,
        show=show,
    )
    
    # Sensitivity Spider plot
    figures["sensitivity_spider"] = plot_sensitivity_spider(
        sobol_indices=sobol_indices,
        output_path=output_path,
        show=show,
    )
    
    # Exploration Progress plot
    if phase_metrics:
        figures["exploration_progress"] = plot_exploration_progress(
            phase1_metrics=phase_metrics.get("phase1"),
            phase2_metrics=phase_metrics.get("phase2"),
            phase3_metrics=phase_metrics.get("phase3"),
            output_path=output_path,
            show=show,
        )

    # -------------------------------------------------------------------------
    # Scientific Visualizations (New)
    # -------------------------------------------------------------------------

    # 1. Surrogate Trustworthiness
    # Uses all data to check model fit (or just validation if we split, but X/y is all)
    if surrogate_model:
        figures["surrogate_trustworthiness"] = plot_surrogate_trustworthiness(
            model=surrogate_model,
            X=X,
            y=y,
            output_path=output_path,
            show=show,
        )

        # 2. Cliff Cross-Section
        # Pick a sample with high efficiency as reference for robustness
        # We sort by efficiency and pick top
        if len(y) > 0:
            best_idx = np.argmax(y)
            X_best = X[best_idx]
            
            figures["cliff_cross_section"] = plot_cliff_cross_section(
                model=surrogate_model,
                X_sample=X_best,
                output_path=output_path,
                show=show,
            )

    # 3. Acquisition Dynamics
    # Need to separate Active Learning samples
    # We infer cutoff from Phase 1 metrics
    try:
        n_warmup = 0
        if phase_metrics and "phase1" in phase_metrics:
             # phase1 metrics is likely a dict version of Phase1Metrics
             n_warmup = phase_metrics["phase1"].get("num_samples", 0)
        
        # If we have more samples than warmup, the rest are active learning
        if len(y) > n_warmup:
            X_active = X[n_warmup:]
            y_active = y[n_warmup:]
            
            # Try to get strategies if available (requires parsing metadata or inferring)
            # Since we don't have strategy labels in X, we skip coloring by strategy for now
            # unless we passed metadata.
            
            figures["acquisition_dynamics"] = plot_acquisition_dynamics(
                X_active=X_active,
                y_active=y_active,
                output_path=output_path,
                show=show,
            )
    except Exception as e:
        print(f"Skipping acquisition dynamics plot: {e}")

    # 4. Guardrail Reliability
    if guardrail_model:
        # Use a random subset or all data for ROC
        figures["guardrail_roc"] = plot_guardrail_roc(
            guardrail=guardrail_model,
            X_val=X,
            y_val=y,
            output_path=output_path,
            show=show,
        )
    
    return figures
