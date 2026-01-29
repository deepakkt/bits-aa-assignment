"""Feature alignment utilities (DTW) for the voice conversion pipeline.

Implements a lightweight Dynamic Time Warping helper used in Part 5 to align
time-varying feature sequences (e.g., MFCCs) between source and target
utterances. The alignment path is returned as integer index pairs that remain
stable across runs (deterministic for fixed inputs).
"""
from __future__ import annotations

import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean


def _to_frame_matrix(features: np.ndarray) -> np.ndarray:
    """Convert arbitrary input to a 2D (feature_dim, frames) float32 matrix.

    The DTW implementation expects a sequence of frame vectors; we treat the
    second dimension as time. NaNs/Infs are replaced with zeros to keep the
    distance computation stable.
    """

    arr = np.asarray(features, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    if arr.ndim == 1:
        arr = arr[None, :]  # shape -> (1, frames)
    if arr.ndim != 2:
        raise ValueError("features must be 1D or 2D array-like")

    return arr


def align_features_dtw(source_features: np.ndarray, target_features: np.ndarray) -> np.ndarray:
    """Align features using Dynamic Time Warping.

    Parameters
    ----------
    source_features : np.ndarray
        Array shaped ``(feature_dim, frames_s)``.
    target_features : np.ndarray
        Array shaped ``(feature_dim, frames_t)``.

    Returns
    -------
    np.ndarray
        Alignment path as an array of shape ``(path_len, 2)`` where each row is
        ``(src_idx, tgt_idx)``. Empty arrays yield an empty path.
    """

    src = _to_frame_matrix(source_features)
    tgt = _to_frame_matrix(target_features)

    # Handle degenerate inputs early
    if src.shape[1] == 0 or tgt.shape[1] == 0:
        return np.zeros((0, 2), dtype=np.int64)

    # DTW expects sequences of frame vectors; transpose to (frames, dim).
    src_seq = src.T
    tgt_seq = tgt.T

    # fastdtw provides an efficient approximation while preserving monotonic
    # alignment; euclidean distance matches MFCC-based alignment well.
    _, path = fastdtw(src_seq, tgt_seq, dist=euclidean)

    if not path:
        return np.zeros((0, 2), dtype=np.int64)

    return np.asarray(path, dtype=np.int64)


__all__ = ["align_features_dtw"]
