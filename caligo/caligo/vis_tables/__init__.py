"""
Visualization and table generation utilities for Caligo Exploration.

This package provides tools for generating publication-quality figures
and result tables from the exploration campaign data.

Modules
-------
visualizations
    Matplotlib-based plotting functions for exploration results.
tables
    Pandas-based table generation for result summaries.
metrics
    Metric computation utilities bridging exploration and simulation.
"""

from caligo.vis_tables.visualizations import (
    plot_death_valley,
    plot_strategy_duel,
    plot_security_volume,
    plot_sensitivity_spider,
    generate_all_figures,
)

from caligo.vis_tables.tables import (
    generate_death_valley_table,
    generate_feasibility_table,
    generate_strategy_table,
    generate_all_tables,
)

from caligo.vis_tables.metrics import (
    compute_qber_from_sample,
    compute_theoretical_min_n,
    compute_max_distance_km,
    samples_to_dataframe,
)

__all__ = [
    # Visualizations
    "plot_death_valley",
    "plot_strategy_duel",
    "plot_security_volume",
    "plot_sensitivity_spider",
    "generate_all_figures",
    # Tables
    "generate_death_valley_table",
    "generate_feasibility_table",
    "generate_strategy_table",
    "generate_all_tables",
    # Metrics
    "compute_qber_from_sample",
    "compute_theoretical_min_n",
    "compute_max_distance_km",
    "samples_to_dataframe",
]
