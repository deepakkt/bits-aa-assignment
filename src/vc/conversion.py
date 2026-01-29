"""Voice conversion utilities (Part C).

Currently implements pitch shifting for the assignment's Part 6. Spectral
envelope conversion and the full pipeline will be added in later parts.

Design goals for `shift_pitch`:
- Clamp the user-provided ratio to a safe range [0.5, 2.0].
- Preserve duration (pad/trim to the original length if needed).
- Avoid NaNs/Infs and excessive clipping; fall back gracefully if librosa's
  pitch shifter fails.
"""
from __future__ import annotations

import logging
from typing import Tuple

import librosa
import numpy as np

from vc import config

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _sanitize_audio(audio: np.ndarray) -> np.ndarray:
    """Ensure float32 mono audio with NaNs/Infs replaced by zeros."""

    audio = np.asarray(audio, dtype=np.float32)
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    return audio


def _match_length(audio: np.ndarray, target_len: int) -> np.ndarray:
    """Pad or trim to the desired length (helper to preserve duration)."""

    if target_len <= 0:
        return np.zeros(0, dtype=np.float32)
    if audio.size == target_len:
        return audio
    if audio.size > target_len:
        return audio[:target_len]
    pad = target_len - audio.size
    return np.pad(audio, (0, pad))


# -----------------------------------------------------------------------------
# Public Part C API
# -----------------------------------------------------------------------------
def shift_pitch(audio: np.ndarray, sr: int, pitch_ratio: float) -> np.ndarray:
    """Apply pitch shifting while preserving duration.

    Parameters
    ----------
    audio : np.ndarray
        Input waveform (mono or multi-channel). Values are assumed to be in
        the range [-1, 1] as produced by preprocessing, but the function is
        defensive against out-of-range inputs.
    sr : int
        Sample rate of the input audio.
    pitch_ratio : float
        Desired pitch multiplier; values are clamped to [0.5, 2.0]. For
        example, 1.0 leaves pitch unchanged; 2.0 raises by one octave; 0.5
        lowers by one octave.

    Returns
    -------
    np.ndarray
        Pitch-shifted audio with roughly the same duration as the input.
    """

    if sr <= 0:
        raise ValueError("Sample rate must be positive")

    ratio = float(pitch_ratio) if np.isfinite(pitch_ratio) else 1.0
    ratio = float(np.clip(ratio, 0.5, 2.0))

    audio = _sanitize_audio(audio)
    if audio.size == 0:
        return np.zeros(0, dtype=np.float32)

    target_len = audio.shape[0]
    orig_rms = float(np.sqrt(np.mean(np.square(audio)))) if target_len else 0.0

    # If ratio is effectively unity, return a safe copy without extra work.
    if np.isclose(ratio, 1.0, atol=1e-3):
        return audio.copy()

    n_steps = 12.0 * np.log2(ratio)  # semitone shift for librosa

    try:
        shifted = librosa.effects.pitch_shift(
            y=audio,
            sr=sr,
            n_steps=n_steps,
            res_type="kaiser_best",
        )
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.warning("librosa pitch_shift failed (%s); falling back to resample+stretch", exc)
        # Fallback: change pitch by resampling, then time-stretch back to the
        # original duration using a phase vocoder. This keeps length stable.
        target_sr = max(int(sr * ratio), 1)
        resampled = librosa.resample(audio, orig_sr=sr, target_sr=target_sr, res_type="kaiser_best")
        # time_stretch rate <1 slows down (longer); rate >1 speeds up (shorter)
        stretch_rate = max(resampled.size / float(target_len), 1e-3)
        shifted = librosa.effects.time_stretch(resampled, rate=stretch_rate)

    shifted = np.nan_to_num(shifted, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    shifted = _match_length(shifted, target_len)

    # Guard against numerical blow-up or near-silence.
    max_abs = float(np.max(np.abs(shifted))) if shifted.size else 0.0
    if max_abs > 1.0:
        shifted = shifted / max_abs * 0.99

    shifted_rms = float(np.sqrt(np.mean(np.square(shifted)))) if shifted.size else 0.0
    if shifted_rms < 1e-6 and orig_rms > 0:
        # If the operation produced silence, fall back to the original audio.
        shifted = audio.copy()

    return shifted.astype(np.float32)


def convert_spectral_envelope(audio: np.ndarray, sr: int, mapping_model) -> np.ndarray:
    """Placeholder for Part 7 spectral envelope conversion."""

    raise NotImplementedError("Spectral conversion will be implemented in Part 7")


def voice_conversion_pipeline(
    source_audio: np.ndarray, sr: int, mapping_model, pitch_ratio: float
) -> np.ndarray:
    """Placeholder for Part 7 full voice conversion pipeline."""

    raise NotImplementedError("Voice conversion pipeline will be implemented in Part 7")

