"""
Core utilities for head-level attention plasticity analysis.
"""

from .data_utils import fit_bucket_stats, orientation_from_keys, stack_oriented_examples
from .head_analysis import analyze_head
from .plasticity import NUM_PAIRS_PER_BUCKET, compute_attention_plasticity
from .statistics import (
    compute_first_component_residual_stats,
    compute_noise_normality_stats,
    compute_scalar_position_predictability,
    fit_multioutput_linear_regressor,
    make_pos_rotation,
)

__all__ = [
    "analyze_head",
    "compute_attention_plasticity",
    "compute_first_component_residual_stats",
    "compute_noise_normality_stats",
    "compute_scalar_position_predictability",
    "fit_bucket_stats",
    "fit_multioutput_linear_regressor",
    "make_pos_rotation",
    "NUM_PAIRS_PER_BUCKET",
    "orientation_from_keys",
    "stack_oriented_examples",
]
