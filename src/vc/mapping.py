"""Feature mapping models for voice conversion (Part 5).

Implements a lightweight linear regression baseline that maps source MFCC
frames to target MFCC frames after DTW alignment. The model is intentionally
simple and deterministic to keep runtime low in the virtual lab.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from sklearn.linear_model import LinearRegression

from vc import alignment


@dataclass
class FeatureMappingModel:
    """Container for a fitted regression model and metadata."""

    regressor: LinearRegression
    feature_dim: int
    source_speaker: str
    target_speaker: str

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict target-space features for input frames.

        Parameters
        ----------
        features : np.ndarray
            Array of shape ``(feature_dim, frames)``.
        """

        feats = np.asarray(features, dtype=np.float32)
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        if feats.ndim == 1:
            feats = feats[None, :]
        if feats.shape[0] != self.feature_dim:
            raise ValueError(
                f"Expected feature_dim={self.feature_dim}, got {feats.shape[0]}"
            )
        # sklearn expects (n_samples, n_features); transpose frames-first.
        preds = self.regressor.predict(feats.T)
        return np.asarray(preds, dtype=np.float32).T


def _prepare_aligned_frames(
    source_features: np.ndarray, target_features: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Align feature sequences and return paired frames."""

    path = alignment.align_features_dtw(source_features, target_features)
    if path.size == 0:
        return (
            np.zeros((0, source_features.shape[0]), dtype=np.float32),
            np.zeros((0, target_features.shape[0]), dtype=np.float32),
        )

    src = np.asarray(source_features, dtype=np.float32)
    tgt = np.asarray(target_features, dtype=np.float32)

    X = src[:, path[:, 0]]  # shape (feat_dim, frames_aligned)
    Y = tgt[:, path[:, 1]]

    # Transpose to (frames, feat_dim) for sklearn
    return X.T, Y.T


def train_feature_mapping(
    source_features: np.ndarray, target_features: np.ndarray
) -> FeatureMappingModel:
    """Train feature mapping model (linear regression baseline)."""

    X, Y = _prepare_aligned_frames(source_features, target_features)
    if X.shape[0] == 0 or Y.shape[0] == 0:
        raise ValueError("Cannot train mapping: empty aligned frame set")

    reg = LinearRegression()
    reg.fit(X, Y)

    feature_dim = source_features.shape[0] if source_features.ndim > 1 else 1
    return FeatureMappingModel(
        regressor=reg,
        feature_dim=feature_dim,
        source_speaker="unknown",
        target_speaker="unknown",
    )


def convert_features(model: FeatureMappingModel, source_features: np.ndarray) -> np.ndarray:
    """Convert source features using trained mapping model."""

    return model.predict(source_features)


__all__ = [
    "FeatureMappingModel",
    "train_feature_mapping",
    "convert_features",
]
