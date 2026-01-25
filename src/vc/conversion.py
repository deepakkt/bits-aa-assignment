"""
Conversion helpers for voice conversion.

This module contains lightweight pitch shifting and spectral envelope mapping
utilities used by the conversion and evaluation scripts. Implementations favor
determinism and guard against extreme ratios to keep artifacts manageable on
the 16 kHz CMU Arctic data.
"""
from __future__ import annotations

import numpy as np
import librosa

from .config import TARGET_SR
from .features import HOP_LENGTH, N_FFT, extract_mfcc
from .mapping import FeatureMappingModel, convert_features
from .io_utils import get_logger

logger = get_logger(__name__)


def _validate_audio(audio: np.ndarray, sr: int) -> np.ndarray:
    if sr != TARGET_SR:
        raise ValueError(f"Expected sample rate {TARGET_SR}, got {sr}")
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1:
        raise ValueError(f"Expected mono audio, got shape {audio.shape}")
    if audio.size == 0:
        raise ValueError("Audio array is empty.")
    return audio


def shift_pitch(audio: np.ndarray, sr: int, pitch_ratio: float) -> np.ndarray:
    """
    Shift the pitch of an audio signal without significantly altering duration.

    The ratio is clamped to [0.5, 2.0] to avoid extreme artifacts. The output
    is normalized to stay within [-1, 1].
    """
    audio = _validate_audio(audio, sr)
    if pitch_ratio is None or not np.isfinite(pitch_ratio):
        pitch_ratio = 1.0
    ratio = float(np.clip(pitch_ratio, 0.5, 2.0))

    # librosa pitch_shift preserves duration to within numerical tolerance.
    n_steps = 12.0 * np.log2(ratio)
    shifted = librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)

    max_abs = np.max(np.abs(shifted)) if shifted.size else 1.0
    if max_abs > 1.0:
        shifted = shifted / max_abs
    return np.asarray(shifted, dtype=np.float32)


def _mfcc_to_linear_spectrogram(mfcc: np.ndarray, sr: int) -> np.ndarray:
    """
    Approximate a linear magnitude spectrogram from MFCCs.

    This uses the inverse mel transform followed by a mel->STFT magnitude
    projection. While lossy, it provides a reasonable spectral envelope for
    Griffin-Lim reconstruction.
    """
    mel_spec = librosa.feature.inverse.mfcc_to_mel(mfcc)
    # mel_to_stft returns a linear magnitude matrix; power=1 for magnitude.
    linear_mag = librosa.feature.inverse.mel_to_stft(
        mel_spec, sr=sr, n_fft=N_FFT, power=1.0
    )
    return np.asarray(np.abs(linear_mag), dtype=np.float32)


def convert_spectral_envelope(
    audio: np.ndarray, sr: int, mapping_model: FeatureMappingModel
) -> np.ndarray:
    """
    Convert the spectral envelope of an audio signal using a trained mapping.

    Args:
        audio: Preprocessed mono waveform at TARGET_SR.
        sr: Sample rate (must equal TARGET_SR).
        mapping_model: Trained FeatureMappingModel.

    Returns:
        Linear magnitude spectrogram (frequency bins x frames) after mapping the
        MFCC representation through the provided model.
    """
    audio = _validate_audio(audio, sr)
    mfcc = extract_mfcc(audio, sr)
    converted_mfcc = convert_features(mapping_model, mfcc)
    spectrogram = _mfcc_to_linear_spectrogram(converted_mfcc, sr)
    return spectrogram


__all__ = ["shift_pitch", "convert_spectral_envelope"]
