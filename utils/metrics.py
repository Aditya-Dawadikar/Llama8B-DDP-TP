"""Evaluation metrics for GLUE tasks.

This module implements a minimal set of metrics used in GLUE.  It currently
supports accuracy for classification tasks and Pearson/Spearman correlation for
the STS‑B regression task.  For full evaluation, consider using the `evaluate`
package or the Hugging Face `datasets.load_metric` API.
"""

from typing import Sequence, Tuple
import numpy as np
from scipy.stats import pearsonr, spearmanr


def accuracy(preds: Sequence[int], labels: Sequence[int]) -> float:
    """Compute classification accuracy."""
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    return float((preds == labels).mean())


def stsb_corr(preds: Sequence[float], labels: Sequence[float]) -> Tuple[float, float]:
    """Compute Pearson and Spearman correlation for STS-B."""
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    pearson, _ = pearsonr(preds, labels)
    spearman, _ = spearmanr(preds, labels)
    return float(pearson), float(spearman)
