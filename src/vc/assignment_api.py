"""Auto-graded API surface for the BITS voice conversion assignment.

This module exposes all functions referenced in the grading rubric. Parts that
are not yet implemented deliberately raise NotImplementedError to make gaps
explicit. Part A functions are fully implemented here by delegating to
`vc.audio_preproc`.
"""
from __future__ import annotations

import numpy as np

from vc import audio_preproc, features, alignment, mapping


# -----------------------------------------------------------------------------
# Part A - Preprocessing
# -----------------------------------------------------------------------------
def load_speaker_data(speaker_id: str, data_path: str) -> list:
    """Load all audio files for a speaker.

    Returns
    -------
    list
        List of (audio_array, sample_rate) tuples
    """

    return audio_preproc.load_speaker_data(speaker_id, data_path)


def preprocess_audio(audio: np.ndarray, sr: int) -> tuple:
    """Preprocess audio: resample, normalize, pre-emphasis.

    Returns
    -------
    tuple
        (processed_audio, new_sr)
    """

    return audio_preproc.preprocess_audio(audio, sr)


def compute_f0_stats(audio: np.ndarray, sr: int) -> dict:
    """Compute F0 statistics.

    Returns
    -------
    dict
        {'mean_f0','std_f0','min_f0','max_f0'}
    """

    return audio_preproc.compute_f0_stats(audio, sr)


def compute_rms_energy(audio: np.ndarray) -> float:
    """Compute RMS energy of audio signal.

    Returns
    -------
    float
        RMS energy value
    """

    return audio_preproc.compute_rms_energy(audio)


# -----------------------------------------------------------------------------
# Part B - Feature Extraction
# -----------------------------------------------------------------------------
def extract_f0(audio: np.ndarray, sr: int) -> np.ndarray:
    return features.extract_f0(audio, sr)


def extract_mfcc(audio: np.ndarray, sr: int, n_mfcc: int = 13) -> np.ndarray:
    return features.extract_mfcc(audio, sr, n_mfcc=n_mfcc)


def extract_formants(audio: np.ndarray, sr: int) -> np.ndarray:
    return features.extract_formants(audio, sr)


def calculate_pitch_shift_ratio(source_f0: np.ndarray, target_f0: np.ndarray) -> float:
    return features.calculate_pitch_shift_ratio(source_f0, target_f0)


def align_features_dtw(source_features: np.ndarray, target_features: np.ndarray) -> np.ndarray:
    return alignment.align_features_dtw(source_features, target_features)


def train_feature_mapping(source_features: np.ndarray, target_features: np.ndarray):
    return mapping.train_feature_mapping(source_features, target_features)


def convert_features(model, source_features: np.ndarray) -> np.ndarray:
    return mapping.convert_features(model, source_features)


# -----------------------------------------------------------------------------
# Part C - Voice Conversion (stubs)
# -----------------------------------------------------------------------------
def shift_pitch(audio: np.ndarray, sr: int, pitch_ratio: float) -> np.ndarray:
    raise NotImplementedError("Part C not yet implemented")


def convert_spectral_envelope(audio: np.ndarray, sr: int, mapping_model) -> np.ndarray:
    raise NotImplementedError("Part C not yet implemented")


def voice_conversion_pipeline(
    source_audio: np.ndarray, sr: int, mapping_model, pitch_ratio: float
) -> np.ndarray:
    raise NotImplementedError("Part C not yet implemented")


# -----------------------------------------------------------------------------
# Part D - Evaluation (stubs)
# -----------------------------------------------------------------------------
def calculate_mcd(converted_mfcc: np.ndarray, target_mfcc: np.ndarray) -> float:
    raise NotImplementedError("Part D not yet implemented")


def calculate_f0_correlation(converted_f0: np.ndarray, target_f0: np.ndarray) -> float:
    raise NotImplementedError("Part D not yet implemented")


def calculate_formant_rmse(converted_formants: np.ndarray, target_formants: np.ndarray) -> float:
    raise NotImplementedError("Part D not yet implemented")
