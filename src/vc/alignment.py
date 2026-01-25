"""
Dynamic Time Warping alignment utilities.

DTW is used to align source and target MFCC frame sequences so that a mapping
model can learn frame-level correspondences despite local timing differences.
The returned path is monotonic and indexes source/target frames in forward
order (earliest to latest).
"""
from __future__ import annotations

from typing import Tuple

import librosa
import numpy as np

from .io_utils import get_logger

logger = get_logger(__name__)


def _validate_pair(
    source_features: np.ndarray, target_features: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Ensure feature matrices are compatible for DTW."""
    src = np.asarray(source_features, dtype=np.float32)
    tgt = np.asarray(target_features, dtype=np.float32)
    if src.ndim != 2 or tgt.ndim != 2:
        raise ValueError(
            f"Expected 2D feature matrices, got shapes {src.shape} and {tgt.shape}"
        )
    if src.shape[0] != tgt.shape[0]:
        raise ValueError(
            f"Feature dimension mismatch: {src.shape[0]} vs {tgt.shape[0]}"
        )
    if src.shape[1] == 0 or tgt.shape[1] == 0:
        raise ValueError("Cannot align empty feature sequences.")
    return src, tgt


def align_features_dtw(
    source_features: np.ndarray, target_features: np.ndarray
) -> np.ndarray:
    """
    Align two feature sequences using Dynamic Time Warping.

    Args:
        source_features: Array shaped (D, T_src) such as source MFCCs.
        target_features: Array shaped (D, T_tgt) such as target MFCCs.

    Returns:
        np.ndarray with shape (L, 2) where each row is (src_idx, tgt_idx) giving
        the warping path from start to end. The path is reversed from librosa's
        output so that it runs forward in time.
    """
    src, tgt = _validate_pair(source_features, target_features)
    # librosa.sequence.dtw returns the cumulative cost matrix and a path that
    # runs from end to start; reversing it makes downstream alignment simpler.
    _, path = librosa.sequence.dtw(X=src, Y=tgt, metric="euclidean")
    ordered_path = np.array(path[::-1], dtype=np.int32)
    if ordered_path.size == 0:
        raise RuntimeError("DTW returned an empty path; check feature inputs.")
    return ordered_path


__all__ = ["align_features_dtw"]
