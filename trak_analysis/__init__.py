"""
TRAK Analysis for News Recommendation Models.

This module provides tools for TRAK (Tracing with the Randomly-projected After Kernel)
scoring to understand which training examples influence model predictions.
"""

from .trak_scorer import (
    TRAKScorer,
    compute_trak_scores,
    rank_training_samples,
)

from .data_loader import load_train_data, get_data_statistics_fast

from .visualize import (
    plot_score_distribution,
    plot_top_influential_samples,
    plot_score_comparison,
)

__all__ = [
    "TRAKScorer",
    "compute_trak_scores",
    "rank_training_samples",
    "load_train_data",
    "get_data_statistics_fast",
    "plot_score_distribution",
    "plot_top_influential_samples",
    "plot_score_comparison",
]
