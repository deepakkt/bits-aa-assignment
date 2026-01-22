"""
Audio loading and preprocessing utilities for the voice conversion pipeline.

This module centralizes resampling, normalization, silence trimming, and
pre-emphasis so the behavior stays deterministic across scripts and notebooks.
All processed audio is returned as float32 at TARGET_SR with values constrained
to [-1, 1].
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import librosa
import numpy as np
import soundfile as sf

from .config import PREEMPH, TARGET_SR
from .io_utils import get_logger

logger = get_logger(__name__)


def _safe_to_mono(audio: np.ndarray) -> np.ndarray:
    """Collapse multi-channel audio to mono while preserving dtype."""
    if audio.ndim == 1:
        return audio
    if audio.ndim == 2:
        return np.mean(audio, axis=1)
    raise ValueError(f"Expected 1D or 2D audio, got shape {audio.shape}")


def _trim_silence(audio: np.ndarray, top_db: float = 25.0) -> np.ndarray:
    """
    Trim leading/trailing silence conservatively. If trimming would remove most
    of the content, fall back to the original to avoid accidental deletion.
    """
    if audio.size == 0:
        return audio
    trimmed, idx = librosa.effects.trim(
        audio, top_db=top_db, frame_length=2048, hop_length=512
    )
    # Avoid over-aggressive trimming by requiring at least 20% of frames remain.
    if trimmed.size < max(1, int(0.2 * audio.size)):
        return audio
    # librosa returns int indices; if nothing was trimmed they match full span.
    if idx[0] == 0 and idx[1] == audio.size:
        return audio
    return trimmed


def _normalize(audio: np.ndarray) -> np.ndarray:
    """Scale audio to lie within [-1, 1]; return zeros if input is silent."""
    if audio.size == 0:
        return audio.astype(np.float32)
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak
    return audio.astype(np.float32)


def _apply_preemphasis(audio: np.ndarray, coeff: float = PREEMPH) -> np.ndarray:
    """
    Apply first-order pre-emphasis y[n] = x[n] - coeff * x[n-1].

    Pre-emphasis amplifies higher frequencies, which improves MFCC stability on
    voiced speech. We renormalize afterwards to keep values within [-1, 1].
    """
    if audio.size == 0:
        return audio.astype(np.float32)
    emphasized = np.empty_like(audio, dtype=np.float32)
    emphasized[0] = audio[0]
    emphasized[1:] = audio[1:] - coeff * audio[:-1]
    max_abs = np.max(np.abs(emphasized))
    if max_abs > 1.0:
        emphasized = emphasized / max_abs
    return emphasized.astype(np.float32)


def preprocess_audio(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Resample, trim silence, normalize, and apply pre-emphasis.

    Args:
        audio: Input waveform (mono or multi-channel).
        sr: Original sample rate.

    Returns:
        Preprocessed mono waveform at TARGET_SR as float32, normalized to [-1, 1]
        with pre-emphasis applied.
    """
    if sr is None or sr <= 0:
        raise ValueError(f"Invalid sample rate provided: {sr}")
    if audio is None:
        raise ValueError("Audio array is None.")

    audio = np.asarray(audio, dtype=np.float32)
    audio = _safe_to_mono(np.nan_to_num(audio))
    if audio.size == 0:
        raise ValueError("Audio array is empty after loading.")

    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)

    audio = _trim_silence(audio, top_db=25.0)
    audio = _normalize(audio)
    audio = _apply_preemphasis(audio, coeff=PREEMPH)
    return audio


def compute_f0stats(audio: np.ndarray, sr: int) -> dict:
    """
    Compute summary statistics of the fundamental frequency (F0) contour.

    NaN values from unvoiced frames are ignored. If no voiced frames are found,
    zeros are returned to keep downstream code finite.
    """
    if audio is None or audio.size == 0:
        return {"mean_f0": 0.0, "std_f0": 0.0, "min_f0": 0.0, "max_f0": 0.0}
    try:
        f0, _, _ = librosa.pyin(
            audio,
            fmin=50,
            fmax=500,
            sr=sr,
            frame_length=2048,
            hop_length=256,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("F0 estimation failed: %s", exc)
        return {"mean_f0": 0.0, "std_f0": 0.0, "min_f0": 0.0, "max_f0": 0.0}

    f0_arr = np.asarray(f0, dtype=np.float32)
    valid = np.isfinite(f0_arr)
    if not np.any(valid):
        return {"mean_f0": 0.0, "std_f0": 0.0, "min_f0": 0.0, "max_f0": 0.0}

    voiced = f0_arr[valid]
    return {
        "mean_f0": float(np.mean(voiced)),
        "std_f0": float(np.std(voiced)),
        "min_f0": float(np.min(voiced)),
        "max_f0": float(np.max(voiced)),
    }


def compute_rms_energy(audio: np.ndarray) -> float:
    """Compute RMS energy of a waveform; returns 0.0 for empty input."""
    if audio is None or audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio))))


def _resolve_wav_dir(speaker_id: str, data_root: Path) -> Path:
    """
    Locate a speaker's wav directory across common CMU Arctic layouts.
    """
    candidates = [
        data_root / f"cmu_us_{speaker_id}_arctic" / "wav",
        data_root / f"cmu_us_{speaker_id}_arctic-0.95-release" / "wav",
        data_root / "cmu_arctic" / f"cmu_us_{speaker_id}_arctic" / "wav",
        data_root / "cmu_arctic" / f"cmu_us_{speaker_id}_arctic-0.95-release" / "wav",
        data_root / speaker_id / "wav",
        data_root / speaker_id,
    ]
    release_variants = sorted(data_root.glob(f"cmu_us_{speaker_id}_arctic*-release"))
    for variant in release_variants:
        candidates.extend([variant / "wav", variant])
    for cand in candidates:
        if cand.is_dir():
            if cand.name == "wav":
                return cand
            nested = cand / "wav"
            if nested.is_dir():
                return nested
    raise FileNotFoundError(
        f"Could not locate wav directory for speaker '{speaker_id}' under {data_root}"
    )


def load_speaker_data(speaker_id: str, data_path: str) -> List[np.ndarray]:
    """
    Load and preprocess all utterances for a speaker.

    Args:
        speaker_id: CMU Arctic speaker ID (e.g., 'bdl').
        data_path: Root directory containing speaker folders.

    Returns:
        List of preprocessed waveforms as float32 arrays.
    """
    root = Path(data_path)
    wav_dir = _resolve_wav_dir(speaker_id, root)
    wav_paths = sorted(p for p in wav_dir.glob("*.wav") if p.is_file())
    if not wav_paths:
        raise FileNotFoundError(f"No wav files found for speaker '{speaker_id}' in {wav_dir}")

    audio_list: List[np.ndarray] = []
    for wav_path in wav_paths:
        audio, sr = sf.read(wav_path, dtype="float32", always_2d=False)
        audio = _safe_to_mono(audio)
        processed = preprocess_audio(audio, sr)
        audio_list.append(processed)
    return audio_list
