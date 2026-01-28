"""Audio loading and preprocessing utilities for the voice conversion pipeline.

Implements the Part A functions required by the assignment rubric:
- loading per-speaker waveforms
- preprocessing (resample -> normalize -> pre-emphasize -> trim)
- RMS energy and F0 statistics helpers

Design goals:
- Deterministic, idempotent transforms
- Safe handling of edge cases (silent audio, short clips, NaNs)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import librosa
import numpy as np
from scipy.signal import lfilter

from vc import config, io_utils

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------

def _to_mono(audio: np.ndarray) -> np.ndarray:
    """Convert multi-channel audio to mono by averaging channels."""
    if audio.ndim > 1:
        return np.mean(audio, axis=1)
    return audio


def _resolve_speaker_audio_dir(data_path: Path, speaker_id: str) -> Path:
    """Best-effort resolution of a speaker's audio directory.

    Supports common CMU Arctic layouts and falls back to a recursive search.
    """
    candidates = [
        data_path / f"cmu_us_{speaker_id}_arctic" / "wav",
        data_path / f"cmu_us_{speaker_id}_arctic",
        data_path / speaker_id,
        data_path,
    ]

    for cand in candidates:
        if not cand.exists():
            continue
        if (cand / "wav").is_dir():
            return cand / "wav"
        # If this dir already holds wav/flac files, use it directly
        if any(p.is_file() and p.suffix.lower() in {".wav", ".flac"} for p in cand.iterdir()):
            return cand

    # Fallback: recursive search for a directory containing the speaker id
    for path in data_path.rglob("*"):
        if path.is_dir() and speaker_id in path.name:
            if (path / "wav").is_dir():
                return path / "wav"
            if any(p.is_file() and p.suffix.lower() in {".wav", ".flac"} for p in path.iterdir()):
                return path

    raise FileNotFoundError(f"Could not locate audio directory for speaker '{speaker_id}' under {data_path}")


def _trim_silence(audio: np.ndarray, top_db: float = 20.0) -> np.ndarray:
    """Trim leading/trailing silence; return original if trimming fails."""
    try:
        trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
        if trimmed.size > 0:
            return trimmed
    except Exception as exc:  # pragma: no cover - defensive against rare librosa errors
        logger.warning("Silence trim skipped: %s", exc)
    return audio


def _normalize(audio: np.ndarray) -> np.ndarray:
    """Normalize to [-1, 1] (no-op for all-zero signals)."""
    max_abs = float(np.max(np.abs(audio))) if audio.size else 0.0
    if max_abs > 0.0:
        audio = audio / max_abs
    return np.clip(audio.astype(np.float32), -1.0, 1.0)


def _pre_emphasize(audio: np.ndarray, coeff: float) -> np.ndarray:
    """Apply first-order pre-emphasis filter y[n] = x[n] - a*x[n-1]."""
    emphasized = lfilter([1.0, -coeff], [1.0], audio)
    return emphasized.astype(np.float32)


# -----------------------------------------------------------------------------
# Public API (Part A)
# -----------------------------------------------------------------------------

def load_speaker_data(speaker_id: str, data_path: str) -> List[Tuple[np.ndarray, int]]:
    """Load all audio files for a speaker.

    Returns
    -------
    list
        List of ``(audio_array, sample_rate)`` tuples for the speaker.
    """

    root = Path(data_path)
    audio_dir = _resolve_speaker_audio_dir(root, speaker_id)
    files = io_utils.list_audio_files(audio_dir, exts=(".wav", ".flac"))
    if not files:
        raise FileNotFoundError(f"No audio files found for speaker '{speaker_id}' in {audio_dir}")

    loaded: List[Tuple[np.ndarray, int]] = []
    for path in files:
        audio, sr = io_utils.load_audio(path)
        loaded.append((audio, sr))

    logger.info("Loaded %d files for speaker %s from %s", len(loaded), speaker_id, audio_dir)
    return loaded


def preprocess_audio(audio: np.ndarray, sr: int) -> Tuple[np.ndarray, int]:
    """Preprocess audio: resample, normalize, pre-emphasis.

    Returns
    -------
    tuple
        (processed_audio, new_sr)
    """

    if sr <= 0:
        raise ValueError("Sample rate must be positive")

    audio = np.asarray(audio, dtype=np.float32)
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    audio = _to_mono(audio)

    # Optional silence trimming (helps later feature extraction)
    audio = _trim_silence(audio, top_db=20.0)

    # Resample to target sample rate
    target_sr = config.TARGET_SR
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr, res_type="kaiser_best")
        sr = target_sr
    else:
        sr = target_sr

    # Normalize then pre-emphasize
    audio = _normalize(audio)
    audio = _pre_emphasize(audio, config.PREEMPH)
    audio = _normalize(audio)  # ensure final range stays within [-1, 1]

    return audio, sr


def compute_f0_stats(audio: np.ndarray, sr: int) -> dict:
    """Compute F0 statistics.

    Returns
    -------
    dict
        ``{'mean_f0','std_f0','min_f0','max_f0'}``
    """

    audio = np.asarray(audio, dtype=np.float32)
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    audio = _to_mono(audio)

    if audio.size == 0 or sr <= 0:
        return {"mean_f0": 0.0, "std_f0": 0.0, "min_f0": 0.0, "max_f0": 0.0}

    frame_length = 2048
    hop_length = 256
    if audio.size < frame_length:
        audio = np.pad(audio, (0, frame_length - audio.size), mode="constant")

    f0 = librosa.yin(audio, fmin=50, fmax=500, sr=sr, frame_length=frame_length, hop_length=hop_length)
    f0 = np.asarray(f0, dtype=np.float32)
    voiced = np.isfinite(f0) & (f0 > 0)

    if not np.any(voiced):
        return {"mean_f0": 0.0, "std_f0": 0.0, "min_f0": 0.0, "max_f0": 0.0}

    voiced_f0 = f0[voiced]
    stats = {
        "mean_f0": float(np.mean(voiced_f0)),
        "std_f0": float(np.std(voiced_f0)),
        "min_f0": float(np.min(voiced_f0)),
        "max_f0": float(np.max(voiced_f0)),
    }
    return stats


def compute_rms_energy(audio: np.ndarray) -> float:
    """Compute RMS energy of audio signal.

    Returns
    -------
    float
        RMS energy value
    """

    audio = np.asarray(audio, dtype=np.float32)
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    audio = _to_mono(audio)

    if audio.size == 0:
        return 0.0

    rms = float(np.sqrt(np.mean(np.square(audio))))
    return rms

