"""
Auto-graded API surface for the voice conversion assignment.

Each part of the project will progressively populate this module with the
required functions. Keeping the file present from the start prevents import
errors in notebooks or scripts that preemptively reference it.
"""

from __future__ import annotations

from .audio_preproc import (
    compute_f0stats,
    compute_rms_energy,
    load_speaker_data,
    preprocess_audio,
)
from .alignment import align_features_dtw
from .features import (
    calculate_pitch_shift_ratio,
    extract_f0,
    extract_formants,
    extract_mfcc,
)
from .mapping import convert_features, train_feature_mapping
from .conversion import convert_spectral_envelope, shift_pitch
from .metrics import mcd, calculate_f0correlation, calculate_formant_rmse

__all__ = [
    "load_speaker_data",
    "compute_f0stats",
    "compute_rms_energy",
    "preprocess_audio",
    "extract_f0",
    "calculate_pitch_shift_ratio",
    "extract_mfcc",
    "extract_formants",
    "align_features_dtw",
    "train_feature_mapping",
    "convert_features",
    "shift_pitch",
    "convert_spectral_envelope",
    "mcd",
    "calculate_f0correlation",
    "calculate_formant_rmse",
]

# Later parts will extend this namespace with feature extraction, alignment,
# mapping, conversion, and evaluation helpers while preserving these exports.
