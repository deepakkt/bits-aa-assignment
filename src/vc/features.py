"""
Feature extraction utilities for the voice conversion pipeline.

This module centralizes F0, MFCC, and formant estimation with deterministic
parameters so DTW alignment and mapping stay consistent across runs.
"""
from __future__ import annotations

import numpy as np
import librosa

from .config import MFCC_N, TARGET_SR
from .io_utils import get_logger

logger = get_logger(__name__)

# Shared analysis parameters to keep frame alignment consistent.
HOP_LENGTH = 256
N_FFT = 1024
MEL_BANDS = 128
PYIN_FRAME_LENGTH = 1024  # keeps a 4x hop stride; stable for 16 kHz speech
LPC_ORDER = 16
FORMANT_DEFAULTS = np.array([500.0, 1500.0, 2500.0], dtype=np.float32)


def _validate_audio(audio: np.ndarray, sr: int) -> np.ndarray:
    """Basic validation to keep downstream estimates stable."""
    if audio is None:
        raise ValueError("Audio array is None.")
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        raise ValueError("Audio array is empty.")
    if sr != TARGET_SR:
        raise ValueError(f"Expected sample rate {TARGET_SR}, got {sr}")
    return audio


def extract_f0(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Estimate the fundamental frequency contour using librosa.pyin.

    Returns:
        Array of F0 values per frame with ``np.nan`` for unvoiced regions. The
        hop length matches MFCC extraction so DTW can align both spaces.
    """
    audio = _validate_audio(audio, sr)
    try:
        f0, _, _ = librosa.pyin(
            audio,
            fmin=50,
            fmax=500,
            sr=sr,
            frame_length=PYIN_FRAME_LENGTH,
            hop_length=HOP_LENGTH,
        )
    except Exception as exc:  # noqa: BLE001
        # Returning NaNs keeps shapes consistent without silently hiding issues.
        logger.warning("F0 extraction failed; returning NaNs. Error: %s", exc)
        frames = max(1, 1 + (audio.size - 1) // HOP_LENGTH)
        return np.full(frames, np.nan, dtype=np.float32)
    return np.asarray(f0, dtype=np.float32)


def calculate_pitch_shift_ratio(source_f0: np.ndarray, target_f0: np.ndarray) -> float:
    """
    Compute a stable pitch shift ratio based on voiced-frame means.

    The ratio is clamped to [0.5, 2.0] to avoid extreme time-stretch artifacts.
    If no voiced overlap is found, a neutral ratio of 1.0 is returned.
    """
    if source_f0 is None or target_f0 is None:
        return 1.0
    src = np.asarray(source_f0, dtype=np.float32)
    tgt = np.asarray(target_f0, dtype=np.float32)
    if src.size == 0 or tgt.size == 0:
        return 1.0

    length = min(src.size, tgt.size)
    src = src[:length]
    tgt = tgt[:length]
    valid = np.isfinite(src) & np.isfinite(tgt)
    if not np.any(valid):
        return 1.0

    src_mean = float(np.mean(src[valid]))
    tgt_mean = float(np.mean(tgt[valid]))
    if src_mean <= 0:
        return 1.0
    ratio = tgt_mean / src_mean
    return float(np.clip(ratio, 0.5, 2.0))


def extract_mfcc(audio: np.ndarray, sr: int, n_mfcc: int = MFCC_N) -> np.ndarray:
    """
    Compute MFCCs with a fixed hop for downstream DTW alignment.

    Args:
        audio: Preprocessed mono waveform at TARGET_SR.
        sr: Sample rate (must equal TARGET_SR).
        n_mfcc: Number of coefficients to return (default MFCC_N).

    Returns:
        MFCC array of shape (n_mfcc, T) as float32.
    """
    if n_mfcc <= 0:
        raise ValueError(f"n_mfcc must be positive, got {n_mfcc}")
    if n_mfcc > MEL_BANDS:
        raise ValueError(f"n_mfcc ({n_mfcc}) cannot exceed mel bands ({MEL_BANDS})")
    audio = _validate_audio(audio, sr)
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=N_FFT,
        center=False,
        n_mels=MEL_BANDS,
        power=2.0,
    )
    # Natural log avoids the dB scaling blow-up that inflates MCD; floor to keep values finite.
    log_mel = np.log(np.maximum(mel_spec, np.finfo(np.float32).eps))
    mfcc = librosa.feature.mfcc(
        S=log_mel,
        n_mfcc=n_mfcc,
        dct_type=2,
        norm="ortho",
    )
    # Remove implicit sqrt(n_mels) scaling so MFCC magnitudes stay comparable across configs.
    mfcc = mfcc / np.sqrt(MEL_BANDS)
    return mfcc.astype(np.float32)


def _lpc_formants(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Estimate formants via LPC roots.

    We work on a centered window to reduce edge effects; unstable roots are
    filtered by bandwidth and frequency limits.
    """
    # Use a focused window to stabilize LPC on short utterances.
    window = min(audio.size, 4096)
    start = max(0, (audio.size // 2) - (window // 2))
    segment = audio[start : start + window]
    segment = segment - np.mean(segment)

    max_order = max(2, segment.size - 1)
    order = min(LPC_ORDER, max(4, segment.size // 2 - 1))
    order = int(np.clip(order, 2, max_order))
    coeffs = librosa.lpc(segment, order=order)
    roots = np.roots(coeffs)
    roots = roots[np.imag(roots) >= 0]

    angles = np.angle(roots)
    freqs = angles * (sr / (2 * np.pi))
    bandwidths = -0.5 * (sr / np.pi) * np.log(np.abs(roots))

    # Reject roots that imply unrealistically wide resonances.
    mask = (freqs > 90) & (freqs < sr / 2) & (bandwidths < 400)
    return np.sort(freqs[mask].real)


def extract_formants(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Estimate the first three formants (F1, F2, F3) using LPC analysis.

    Returns:
        np.ndarray of shape (3,) containing ascending formant frequencies in Hz.
        If estimation is unstable or insufficient, plausible defaults are
        returned to keep downstream metrics finite.
    """
    audio = _validate_audio(audio, sr)
    if np.allclose(audio, 0):
        return FORMANT_DEFAULTS.copy()

    try:
        formants = _lpc_formants(audio, sr)
    except Exception as exc:  # noqa: BLE001
        # LPC can blow up on degenerate segments; fall back to defaults.
        logger.warning("Formant extraction failed; using defaults. Error: %s", exc)
        return FORMANT_DEFAULTS.copy()

    if formants.size < 3:
        # Pad with defaults to avoid NaNs while keeping sorted order.
        padded = np.concatenate([formants, FORMANT_DEFAULTS[: 3 - formants.size]])
        return np.sort(padded).astype(np.float32)
    return np.sort(formants[:3]).astype(np.float32)


__all__ = [
    "extract_f0",
    "calculate_pitch_shift_ratio",
    "extract_mfcc",
    "extract_formants",
    "MEL_BANDS",
    "HOP_LENGTH",
    "N_FFT",
]
