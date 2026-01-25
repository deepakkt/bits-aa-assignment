"""
Evaluation metrics for voice conversion outputs.

The metrics implemented here align with the assignment requirements:
- Mel Cepstral Distortion (MCD) with DTW alignment
- F0 correlation with NaN masking
- Formant RMSE across the first three formants
"""
from __future__ import annotations

import numpy as np
import librosa

from .io_utils import get_logger

logger = get_logger(__name__)


def _validate_mfcc_pair(converted_mfcc: np.ndarray, target_mfcc: np.ndarray):
    c = np.asarray(converted_mfcc, dtype=np.float32)
    t = np.asarray(target_mfcc, dtype=np.float32)
    if c.ndim != 2 or t.ndim != 2:
        raise ValueError(f"MFCC inputs must be 2D, got {c.shape} and {t.shape}")
    if c.shape[0] != t.shape[0]:
        raise ValueError(f"MFCC dimension mismatch: {c.shape[0]} vs {t.shape[0]}")
    if c.shape[1] == 0 or t.shape[1] == 0:
        raise ValueError("MFCC inputs must have at least one frame.")
    return c, t


def mcd(converted_mfcc: np.ndarray, target_mfcc: np.ndarray) -> float:
    """
    Compute Mel Cepstral Distortion (dB) between two MFCC sequences.

    Uses DTW alignment on L2 frame distances. Returns 0.0 if alignment fails to
    avoid NaNs in downstream aggregations.
    """
    try:
        c, t = _validate_mfcc_pair(converted_mfcc, target_mfcc)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Invalid MFCC inputs for MCD: %s", exc)
        return 0.0

    _, path = librosa.sequence.dtw(X=c, Y=t, metric="euclidean")
    if path is None or len(path) == 0:
        logger.warning("DTW returned an empty path for MCD computation.")
        return 0.0

    ordered = np.array(path[::-1], dtype=np.int32)
    aligned_c = c[:, ordered[:, 0]]
    aligned_t = t[:, ordered[:, 1]]

    diff = aligned_c - aligned_t
    # Frame-wise L2 distances.
    dists = np.sqrt(np.sum(diff * diff, axis=0))
    if dists.size == 0:
        return 0.0
    mcd_const = 10.0 / np.log(10) * np.sqrt(2.0)
    return float(mcd_const * np.mean(dists))


def calculate_f0correlation(converted_f0: np.ndarray, target_f0: np.ndarray) -> float:
    """
    Pearson correlation between converted and target F0 contours on voiced frames.

    NaNs/unvoiced frames are masked out. Returns 0.0 if insufficient overlap.
    The value is clamped to [0, 1] to satisfy the assignment contract.
    """
    if converted_f0 is None or target_f0 is None:
        return 0.0
    c = np.asarray(converted_f0, dtype=np.float32)
    t = np.asarray(target_f0, dtype=np.float32)
    length = min(c.size, t.size)
    if length == 0:
        return 0.0

    c = c[:length]
    t = t[:length]
    mask = np.isfinite(c) & np.isfinite(t)
    if np.count_nonzero(mask) < 2:
        return 0.0
    c_voiced = c[mask]
    t_voiced = t[mask]
    if np.std(c_voiced) == 0 or np.std(t_voiced) == 0:
        return 0.0
    corr = float(np.corrcoef(c_voiced, t_voiced)[0, 1])
    if not np.isfinite(corr):
        return 0.0
    return float(np.clip(corr, 0.0, 1.0))


def calculate_formant_rmse(
    converted_formants: np.ndarray, target_formants: np.ndarray
) -> float:
    """
    Compute RMSE across F1-F3 formants.

    Inputs are expected to be shape (3,) with ascending frequencies. Returns 0.0
    if inputs are invalid.
    """
    if converted_formants is None or target_formants is None:
        return 0.0
    c = np.asarray(converted_formants, dtype=np.float32).reshape(-1)
    t = np.asarray(target_formants, dtype=np.float32).reshape(-1)
    if c.size != 3 or t.size != 3:
        logger.warning("Formant vectors must have length 3, got %d and %d", c.size, t.size)
        return 0.0
    diff = c - t
    return float(np.sqrt(np.mean(diff * diff)))


__all__ = ["mcd", "calculate_f0correlation", "calculate_formant_rmse"]
