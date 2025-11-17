"""
Attribution Analysis for News Recommendation Models.

This module provides tools for axiomatic attribution analysis using
Integrated Gradients to understand which words affect model predictions
for real vs fake news classification.
"""

from .attribution import (
    IntegratedGradients,
    extract_attributions_for_dataset,
    analyze_word_importance,
    plot_word_importance,
    plot_attribution_heatmap,
    compare_attributions,
)

from .data_loader import load_test_data, get_data_statistics_fast

__all__ = [
    "IntegratedGradients",
    "extract_attributions_for_dataset",
    "analyze_word_importance",
    "plot_word_importance",
    "plot_attribution_heatmap",
    "compare_attributions",
    "load_test_data",
    "get_data_statistics_fast",
]
