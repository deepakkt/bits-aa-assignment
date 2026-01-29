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

import librosa
import numpy as np
from scipy.signal import lfilter

from vc import audio_preproc, config, features

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


def _de_emphasize(audio: np.ndarray, coeff: float) -> np.ndarray:
    """Undo pre-emphasis with a simple inverse filter."""

    if audio.size == 0:
        return audio.astype(np.float32)
    restored = lfilter([1.0], [1.0, -float(coeff)], audio)
    return restored.astype(np.float32)


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
    """Convert spectral envelope using a trained MFCC mapping model.

    Steps:
    1) Preprocess the input (resample, normalize, pre-emphasize).
    2) Extract MFCCs.
    3) Map source MFCCs to target space via the provided model.
    4) Reconstruct waveform with Griffin-Lim and remove pre-emphasis.

    The function is defensive against empty inputs and model failures.
    """

    if sr <= 0:
        raise ValueError("Sample rate must be positive")

    audio = _sanitize_audio(audio)
    if audio.size == 0:
        return np.zeros(0, dtype=np.float32)

    # Preprocess to target sampling rate / emphasis domain
    proc_audio, proc_sr = audio_preproc.preprocess_audio(audio, sr)
    if proc_audio.size == 0:
        return np.zeros(0, dtype=np.float32)

    # Feature extraction
    mfcc = features.extract_mfcc(proc_audio, proc_sr, n_mfcc=config.MFCC_N)

    # Apply mapping if available; fall back gracefully
    converted_mfcc = mfcc
    if mapping_model is not None:
        try:
            if hasattr(mapping_model, "predict"):
                converted_mfcc = mapping_model.predict(mfcc)
            else:  # pragma: no cover - unlikely alt API
                converted_mfcc = mapping_model(mfcc)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Mapping model failed (%s); using source MFCCs", exc)
            converted_mfcc = mfcc

    converted_mfcc = np.nan_to_num(converted_mfcc, nan=0.0, posinf=0.0, neginf=0.0).astype(
        np.float32
    )

    n_fft = 1024
    hop_length = 256
    try:
        recon = librosa.feature.inverse.mfcc_to_audio(
            converted_mfcc,
            sr=proc_sr,
            n_fft=n_fft,
            hop_length=hop_length,
            window="hann",
            center=True,
            n_mels=128,
            dct_type=2,
            norm="ortho",
            ref=1.0,
            n_iter=32,
        )
    except Exception as exc:  # pragma: no cover - rare numeric failure
        logger.warning("mfcc_to_audio failed (%s); using mel->audio fallback", exc)
        mel = librosa.feature.inverse.mfcc_to_mel(
            converted_mfcc, n_mels=128, dct_type=2, norm="ortho", ref=1.0
        )
        recon = librosa.feature.inverse.mel_to_audio(
            mel,
            sr=proc_sr,
            n_fft=n_fft,
            hop_length=hop_length,
            window="hann",
            center=True,
            power=1.0,
            n_iter=32,
        )

    recon = np.nan_to_num(recon, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    recon = _match_length(recon, proc_audio.shape[0])
    recon = _de_emphasize(recon, config.PREEMPH)

    # Final safety normalization
    max_abs = float(np.max(np.abs(recon))) if recon.size else 0.0
    if max_abs > 1.0 and max_abs > 0:
        recon = recon / max_abs

    return recon.astype(np.float32)


def voice_conversion_pipeline(
    source_audio: np.ndarray, sr: int, mapping_model, pitch_ratio: float
) -> np.ndarray:
    """Full voice conversion: spectral envelope mapping + pitch shift."""

    if sr <= 0:
        raise ValueError("Sample rate must be positive")

    spectral_audio = convert_spectral_envelope(source_audio, sr, mapping_model)
    if spectral_audio.size == 0:
        return spectral_audio

    ratio = pitch_ratio
    if ratio is None or not np.isfinite(ratio):
        ratio = 1.0

    converted = shift_pitch(spectral_audio, config.TARGET_SR, ratio)
    converted = np.nan_to_num(converted, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    max_abs = float(np.max(np.abs(converted))) if converted.size else 0.0
    if max_abs > 1.0 and max_abs > 0:
        converted = converted / max_abs

    return converted.astype(np.float32)
