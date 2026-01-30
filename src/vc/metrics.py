"""Objective evaluation metrics for the voice conversion assignment (Part D).

The functions here intentionally avoid external dependencies beyond NumPy and
reuse the existing DTW helper for alignment. All inputs are sanitized to keep
the metrics stable inside the virtual lab environment.
"""
from __future__ import annotations

import numpy as np

from vc import alignment

# Constant factor for MCD calculation: 10 / ln(10) * sqrt(2)
_MCD_CONST = float(10.0 / np.log(10.0) * np.sqrt(2.0))


def _sanitize_matrix(arr: np.ndarray) -> np.ndarray:
    """Return a 2D float32 matrix (feat_dim, frames) with NaNs/Infs zeroed."""

    mat = np.asarray(arr, dtype=np.float32)
    mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
    if mat.ndim == 1:
        mat = mat[None, :]
    if mat.ndim != 2:
        raise ValueError("Expected 1D or 2D array for feature matrix")
    return mat


def _match_feature_dims(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Trim feature dimensions to the minimum shared size."""

    dim = min(a.shape[0], b.shape[0])
    return a[:dim], b[:dim]


def calculate_mcd(converted_mfcc: np.ndarray, target_mfcc: np.ndarray) -> float:
    """Compute Mel-Cepstral Distortion (MCD) between two MFCC sequences.

    The sequences are aligned with DTW to handle duration mismatches, then the
    mean Euclidean distance across the alignment path is scaled by the
    canonical constant ``10 / ln(10) * sqrt(2)``.
    """

    conv = _sanitize_matrix(converted_mfcc)
    tgt = _sanitize_matrix(target_mfcc)
    if conv.shape[1] == 0 or tgt.shape[1] == 0:
        return float("nan")

    conv, tgt = _match_feature_dims(conv, tgt)
    path = alignment.align_features_dtw(conv, tgt)
    if path.size == 0:
        return float("nan")

    conv_frames = conv[:, path[:, 0]]
    tgt_frames = tgt[:, path[:, 1]]
    if conv_frames.shape[1] == 0:
        return float("nan")

    dists = np.linalg.norm(conv_frames - tgt_frames, axis=0)
    if dists.size == 0:
        return float("nan")

    return float(_MCD_CONST * float(np.mean(dists)))


def _fill_and_resample(array: np.ndarray, length: int) -> np.ndarray:
    """Fill NaNs by linear interpolation and resample to a fixed length."""

    arr = np.asarray(array, dtype=np.float32).ravel()
    if length <= 0:
        return np.zeros(0, dtype=np.float32)
    if arr.size == 0:
        return np.zeros(length, dtype=np.float32)

    if not np.isfinite(arr).all():
        valid = np.flatnonzero(np.isfinite(arr))
        if valid.size == 0:
            arr = np.zeros_like(arr)
        else:
            arr = arr.copy()
            arr[~np.isfinite(arr)] = np.interp(
                np.flatnonzero(~np.isfinite(arr)),
                valid,
                arr[valid],
            )

    x_old = np.linspace(0.0, 1.0, num=arr.size)
    x_new = np.linspace(0.0, 1.0, num=length)
    return np.interp(x_new, x_old, arr).astype(np.float32)


def calculate_f0_correlation(converted_f0: np.ndarray, target_f0: np.ndarray) -> float:
    """Pearson correlation between F0 contours (voiced frames only).

    Contours are resampled to equal length. Frames with non-positive values
    are treated as unvoiced and excluded from the correlation.
    """

    conv = np.asarray(converted_f0, dtype=np.float32).ravel()
    tgt = np.asarray(target_f0, dtype=np.float32).ravel()
    if conv.size == 0 or tgt.size == 0:
        return 0.0

    length = min(conv.size, tgt.size)
    conv_r = _fill_and_resample(conv, length)
    tgt_r = _fill_and_resample(tgt, length)

    mask = (conv_r > 0) & (tgt_r > 0) & np.isfinite(conv_r) & np.isfinite(tgt_r)
    if np.count_nonzero(mask) < 2:
        return 0.0

    corr = np.corrcoef(conv_r[mask], tgt_r[mask])[0, 1]
    if not np.isfinite(corr):
        return 0.0
    return float(np.clip(corr, -1.0, 1.0))


def calculate_formant_rmse(converted_formants: np.ndarray, target_formants: np.ndarray) -> float:
    """Root-mean-square error between the first three formants (Hz)."""

    conv = np.nan_to_num(np.asarray(converted_formants, dtype=np.float32), nan=0.0)[:3]
    tgt = np.nan_to_num(np.asarray(target_formants, dtype=np.float32), nan=0.0)[:3]
    if conv.size == 0 or tgt.size == 0:
        return 0.0

    dim = min(conv.size, tgt.size)
    diff = conv[:dim] - tgt[:dim]
    if diff.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(diff))))


__all__ = [
    "calculate_mcd",
    "calculate_f0_correlation",
    "calculate_formant_rmse",
]
