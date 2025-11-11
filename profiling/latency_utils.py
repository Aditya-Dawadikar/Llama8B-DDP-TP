"""Utility functions to compute latency percentiles."""

from typing import Sequence, Tuple
import numpy as np


def latency_percentiles(durations: Sequence[float], percentiles=(50, 95, 99)) -> Tuple[float, ...]:
    """Return the requested percentiles of a list of durations.

    Args:
        durations: Iterable of latency values (seconds or milliseconds).
        percentiles: Iterable of percentile values (0–100).

    Returns:
        A tuple of percentile values in the same order as `percentiles`.
    """
    arr = np.asarray(list(durations))
    return tuple(float(np.percentile(arr, p)) for p in percentiles)
