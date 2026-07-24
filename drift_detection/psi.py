"""
Population Stability Index (PSI).

Industry-standard measure of how much a distribution has shifted between
two samples. Reference-quantile binning: build `n_bins` bins from the
REFERENCE distribution's quantiles (so each reference bin holds ~equal
mass by construction), then apply those same bin edges to the CURRENT
distribution and compare proportions:

    PSI = sum_i (current_pct_i - reference_pct_i) * ln(current_pct_i / reference_pct_i)

Conventional thresholds (used throughout credit-risk and ML-monitoring
practice, adopted here as-is rather than reinvented):
    PSI < 0.10             : no significant shift
    0.10 <= PSI < 0.25      : moderate shift, worth investigating (WARNING)
    PSI >= 0.25             : significant shift (CRITICAL)

A zero-probability bin (current or reference) would make the log term
undefined — each bin's proportion is floored at a small epsilon before
the log, which is the standard fix and has negligible effect on the
overall statistic for any bin that actually has some mass.
"""

from __future__ import annotations

import numpy as np

_EPSILON = 1e-6


def population_stability_index(
    reference: np.ndarray, current: np.ndarray, n_bins: int = 10
) -> float:
    if len(reference) < n_bins:
        raise ValueError(f"reference sample (n={len(reference)}) too small for {n_bins} bins")
    if len(current) == 0:
        raise ValueError("current sample must not be empty")

    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(reference, quantiles))
    if len(edges) < 2:
        return 0.0

    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    ref_pct = np.maximum(ref_counts / len(reference), _EPSILON)
    cur_pct = np.maximum(cur_counts / len(current), _EPSILON)

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)
