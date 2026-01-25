"""
Feature mapping models for voice conversion.

We train a simple linear regression baseline on DTW-aligned MFCC frames. This
keeps the pipeline lightweight while providing a reasonable spectral envelope
mapping. The mapping object wraps scikit-learn's LinearRegression so it can be
serialized and reused in downstream conversion steps.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LinearRegression

from .io_utils import get_logger

logger = get_logger(__name__)


@dataclass
class FeatureMappingModel:
    """
    Container for a frame-wise linear mapping.

    Attributes:
        regressor: Fitted LinearRegression that maps source->target MFCC frames.
        n_features: Dimensionality of the feature vectors (e.g., 13 for MFCC).
    """

    regressor: LinearRegression
    n_features: int

    def transform(self, source_features: np.ndarray) -> np.ndarray:
        """
        Apply the mapping to a sequence of source features.

        Args:
            source_features: Array shaped (D, T).

        Returns:
            Array shaped (D, T) containing converted features.
        """
        src = np.asarray(source_features, dtype=np.float32)
        if src.ndim != 2:
            raise ValueError(f"Expected 2D features, got shape {src.shape}")
        if src.shape[0] != self.n_features:
            raise ValueError(
                f"Model expects {self.n_features} dims, got {src.shape[0]}"
            )
        preds = self.regressor.predict(src.T)
        return np.asarray(preds, dtype=np.float32).T


def _to_feature_list(features: Sequence[np.ndarray]) -> List[np.ndarray]:
    """
    Normalize inputs to a list of 2D feature matrices.
    """
    if isinstance(features, np.ndarray):
        return [features]
    if isinstance(features, Iterable):
        return [np.asarray(f, dtype=np.float32) for f in features]
    raise TypeError("Features must be a numpy array or an iterable of arrays.")


def _stack_aligned_pairs(
    source_list: List[np.ndarray], target_list: List[np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stack aligned feature pairs into frame-wise matrices for regression.
    """
    if len(source_list) != len(target_list):
        raise ValueError("Source and target feature lists must be the same length.")

    X_frames = []
    Y_frames = []
    for idx, (src, tgt) in enumerate(zip(source_list, target_list)):
        if src.ndim != 2 or tgt.ndim != 2:
            raise ValueError(
                f"Pair {idx} has invalid shapes {src.shape} and {tgt.shape}"
            )
        if src.shape != tgt.shape:
            raise ValueError(
                f"Aligned pair {idx} shapes differ: {src.shape} vs {tgt.shape}"
            )
        if src.shape[1] == 0:
            continue
        X_frames.append(src.T)  # frames as rows
        Y_frames.append(tgt.T)

    if not X_frames:
        return np.empty((0, source_list[0].shape[0]), dtype=np.float32), np.empty(
            (0, target_list[0].shape[0]), dtype=np.float32
        )

    X = np.vstack(X_frames)
    Y = np.vstack(Y_frames)
    return X.astype(np.float32), Y.astype(np.float32)


def train_feature_mapping(
    source_features: Sequence[np.ndarray], target_features: Sequence[np.ndarray]
) -> FeatureMappingModel:
    """
    Train a linear regression mapping on aligned source/target feature frames.

    Args:
        source_features: Iterable of aligned source feature arrays (D, T).
        target_features: Iterable of aligned target feature arrays (D, T).

    Returns:
        FeatureMappingModel wrapping the trained regressor.
    """
    src_list = _to_feature_list(source_features)
    tgt_list = _to_feature_list(target_features)
    if not src_list:
        raise ValueError("No source features provided for training.")

    X, Y = _stack_aligned_pairs(src_list, tgt_list)
    n_features = src_list[0].shape[0]

    reg = LinearRegression()
    if X.shape[0] == 0:
        # Degenerate fallback to identity to keep the pipeline running.
        logger.warning("No frames available; falling back to identity mapping.")
        reg.fit(np.eye(n_features), np.eye(n_features))
    else:
        reg.fit(X, Y)

    return FeatureMappingModel(regressor=reg, n_features=n_features)


def convert_features(model: FeatureMappingModel, source_features: np.ndarray) -> np.ndarray:
    """
    Convert source feature sequence using a trained mapping model.

    Args:
        model: FeatureMappingModel returned by train_feature_mapping.
        source_features: Array shaped (D, T) to convert.

    Returns:
        Converted feature array shaped (D, T).
    """
    if not isinstance(model, FeatureMappingModel):
        raise TypeError("model must be a FeatureMappingModel instance.")
    return model.transform(source_features)


__all__ = ["FeatureMappingModel", "train_feature_mapping", "convert_features"]
