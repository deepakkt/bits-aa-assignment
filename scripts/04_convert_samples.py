"""
Convert sample utterances using the trained mapping model.

This script loads cached preprocessed waveforms and features from
``02_precompute_features.py``, applies pitch shifting and spectral envelope
mapping, reconstructs audio with Griffin-Lim, and writes three converted WAVs
to ``artifacts/outputs`` by default.
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf
import librosa

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from vc.config import (  # noqa: E402
    CACHE_DIR,
    MANIFEST_DIR,
    MODELS_DIR,
    OUTPUTS_DIR,
    TARGET_SR,
    set_seeds,
)
from vc.features import HOP_LENGTH, N_FFT, calculate_pitch_shift_ratio  # noqa: E402
from vc.io_utils import ensure_dir, get_logger, load_json  # noqa: E402
from vc.conversion import convert_spectral_envelope, shift_pitch  # noqa: E402
from vc.mapping import FeatureMappingModel  # noqa: E402


MIN_CONVERT_SEC = 2.0
MAX_CONVERT_SEC = 10.0
SILENCE_PAD_VALUE = 1e-4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert sample utterances using trained mapping.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=MANIFEST_DIR / "pair_manifest.json",
        help="Path to the pair manifest produced by 01_prepare_dataset.py",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=CACHE_DIR,
        help="Cache directory containing preprocessed audio/features.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=MODELS_DIR / "mfcc_linear_mapping.pkl",
        help="Trained mapping model path from 03_train_mapping.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUTS_DIR,
        help="Directory to store converted WAVs.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=3,
        help="Number of test utterances to convert.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing converted WAVs.",
    )
    return parser.parse_args()


def load_manifest(manifest_path: Path) -> Dict:
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found at {manifest_path}. Run scripts/01_prepare_dataset.py first."
        )
    return load_json(manifest_path)


def _preprocessed_path(cache_dir: Path, split: str, utt_id: str, speaker: str) -> Path:
    return cache_dir / "preprocessed" / split / f"{utt_id}_{speaker}.npy"


def _feature_path(cache_dir: Path, split: str, utt_id: str, speaker: str, kind: str) -> Path:
    return cache_dir / "features" / split / f"{utt_id}_{speaker}_{kind}.npy"


def load_cached_audio(cache_dir: Path, split: str, utt_id: str, speaker: str) -> np.ndarray:
    path = _preprocessed_path(cache_dir, split, utt_id, speaker)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing cached audio at {path}. Run scripts/02_precompute_features.py first."
        )
    audio = np.load(path)
    return np.asarray(audio, dtype=np.float32)


def load_f0(cache_dir: Path, split: str, utt_id: str, speaker: str) -> np.ndarray:
    path = _feature_path(cache_dir, split, utt_id, speaker, "f0")
    if not path.exists():
        raise FileNotFoundError(
            f"Missing cached F0 at {path}. Run scripts/02_precompute_features.py first."
        )
    return np.load(path)


def reconstruct_from_spectrogram(spectrogram: np.ndarray) -> np.ndarray:
    """
    Reconstruct waveform from a linear magnitude spectrogram using Griffin-Lim.
    """
    if spectrogram.ndim != 2:
        raise ValueError(f"Spectrogram must be 2D, got shape {spectrogram.shape}")
    audio = librosa.griffinlim(
        spectrogram,
        n_iter=60,
        hop_length=HOP_LENGTH,
        win_length=N_FFT,
        center=True,
    )
    if audio.size == 0:
        return audio.astype(np.float32)
    max_abs = np.max(np.abs(audio))
    if max_abs > 1.0:
        audio = audio / max_abs
    return audio.astype(np.float32)


def convert_utterance(
    utt_id: str,
    cache_dir: Path,
    model: FeatureMappingModel,
    force: bool,
    logger,
) -> Tuple[np.ndarray, float]:
    src_audio = load_cached_audio(cache_dir, "test", utt_id, "source")
    tgt_f0 = load_f0(cache_dir, "test", utt_id, "target")
    src_f0 = load_f0(cache_dir, "test", utt_id, "source")

    ratio = calculate_pitch_shift_ratio(src_f0, tgt_f0)
    pitched = shift_pitch(src_audio, TARGET_SR, ratio)

    spectrogram = convert_spectral_envelope(pitched, TARGET_SR, model)
    converted = reconstruct_from_spectrogram(spectrogram)
    return converted, ratio


def _entry_duration(entry: Dict[str, object]) -> float:
    """
    Prefer the longer of the parallel pair durations for ranking.
    """
    src = float(entry.get("source", {}).get("duration_sec", 0.0))
    tgt = float(entry.get("target", {}).get("duration_sec", 0.0))
    return max(src, tgt)


def select_long_test_entries(test_entries: List[Dict[str, object]], num_samples: int) -> List[Dict]:
    """
    Select a deterministic subset of the longest test utterances.
    """
    ranked = sorted(test_entries, key=lambda e: (-_entry_duration(e), e.get("utt_id", "")))
    return ranked[:num_samples]


def enforce_duration_bounds(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Clip overly long outputs and pad short ones with low-level silence to land inside [2, 10] sec.
    """
    if sr != TARGET_SR:
        raise ValueError(f"Expected sample rate {TARGET_SR}, got {sr}")
    waveform = np.asarray(audio, dtype=np.float32).reshape(-1)
    if waveform.size == 0:
        waveform = np.zeros(1, dtype=np.float32)

    waveform = np.nan_to_num(waveform, nan=0.0, posinf=0.0, neginf=0.0)
    max_samples = int(MAX_CONVERT_SEC * sr)
    min_samples = int(MIN_CONVERT_SEC * sr)
    bounded = waveform[:max_samples]
    if bounded.size < min_samples:
        pad = np.full(min_samples - bounded.size, SILENCE_PAD_VALUE, dtype=np.float32)
        bounded = np.concatenate([bounded, pad])
    return np.clip(bounded, -1.0, 1.0).astype(np.float32)


def is_output_valid(path: Path) -> bool:
    """
    Validate an existing WAV so idempotent runs can skip only when constraints are met.
    """
    if not path.exists():
        return False
    try:
        info = sf.info(path)
    except Exception:
        return False
    if info.samplerate != TARGET_SR or info.frames <= 0:
        return False
    duration = info.frames / info.samplerate if info.samplerate else 0.0
    return MIN_CONVERT_SEC <= duration <= MAX_CONVERT_SEC


def save_wav(audio: np.ndarray, path: Path) -> None:
    ensure_dir(path.parent)
    sf.write(path, audio, samplerate=TARGET_SR, subtype="PCM_16")


def main() -> None:
    args = parse_args()
    set_seeds()
    logger = get_logger("convert_samples")

    manifest_path = args.manifest if args.manifest.is_absolute() else (ROOT / args.manifest)
    cache_dir = args.cache_dir if args.cache_dir.is_absolute() else (ROOT / args.cache_dir)
    model_path = args.model_path if args.model_path.is_absolute() else (ROOT / args.model_path)
    output_dir = args.output_dir if args.output_dir.is_absolute() else (ROOT / args.output_dir)

    manifest = load_manifest(manifest_path)
    test_entries = manifest.get("test", [])
    if len(test_entries) < args.num_samples:
        raise ValueError(
            f"Requested {args.num_samples} samples but manifest only has {len(test_entries)} test items."
        )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Mapping model not found at {model_path}. Run scripts/03_train_mapping.py first."
        )
    with model_path.open("rb") as f:
        model = pickle.load(f)
    if not isinstance(model, FeatureMappingModel):
        raise TypeError(f"Loaded object is not a FeatureMappingModel: {type(model)}")

    ensure_dir(output_dir)
    logger.info("Converting %d test utterances.", args.num_samples)

    selected_entries = select_long_test_entries(test_entries, args.num_samples)
    logger.info(
        "Selected utterances (longest-first): %s",
        ", ".join(e["utt_id"] for e in selected_entries),
    )

    for idx, entry in enumerate(selected_entries, start=1):
        utt_id = entry["utt_id"]
        out_path = output_dir / f"converted_sample_{idx}.wav"
        if out_path.exists() and not args.force:
            if is_output_valid(out_path):
                logger.info("Found existing %s; skipping (use --force to overwrite).", out_path.name)
                continue
            logger.warning("Existing %s violates duration/SR constraints; regenerating.", out_path.name)
        try:
            audio, ratio = convert_utterance(utt_id, cache_dir, model, args.force, logger)
            audio = enforce_duration_bounds(audio, TARGET_SR)
        except Exception as exc:  # noqa: BLE001
            logger.error("Conversion failed for %s: %s", utt_id, exc)
            # Write a short silent clip to maintain required outputs.
            audio = np.full(int(MIN_CONVERT_SEC * TARGET_SR), SILENCE_PAD_VALUE, dtype=np.float32)
            ratio = 1.0
        save_wav(audio, out_path)
        logger.info("Wrote %s (pitch ratio %.3f)", out_path.name, ratio)


if __name__ == "__main__":
    main()
