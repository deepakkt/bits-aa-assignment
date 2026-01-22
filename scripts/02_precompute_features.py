"""
Precompute preprocessed waveforms and basic statistics for all utterances.

This script reads the deterministic manifest produced in Part-2, applies the
preprocessing pipeline (resample -> trim -> normalize -> pre-emphasize), and
caches the results under ``artifacts/cache/preprocessed``. It also records
simple RMS and F0 statistics to help sanity-check audio quality.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from vc.audio_preproc import compute_f0stats, compute_rms_energy, preprocess_audio  # noqa: E402
from vc.config import CACHE_DIR, MANIFEST_DIR, TARGET_SR, set_seeds  # noqa: E402
from vc.io_utils import ensure_dir, get_logger, load_json, save_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess CMU Arctic waveforms and cache them for feature extraction."
    )
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
        help="Directory to store cached preprocessed audio.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute preprocessing even if cached outputs already exist.",
    )
    return parser.parse_args()


def load_manifest(manifest_path: Path) -> Dict:
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found at {manifest_path}. Run scripts/01_prepare_dataset.py first."
        )
    return load_json(manifest_path)


def preprocess_and_cache(
    wav_path: Path,
    split: str,
    utt_id: str,
    speaker_label: str,
    cache_dir: Path,
    force: bool,
):
    """
    Preprocess a single wav and cache it to disk.

    Returns:
        cached_path: Location of the saved numpy file.
        rms: RMS energy of the processed audio.
        f0_stats: Summary stats of the F0 contour.
    """
    out_dir = ensure_dir(cache_dir / "preprocessed" / split)
    cached_path = out_dir / f"{utt_id}_{speaker_label}.npy"

    if cached_path.exists() and not force:
        audio = np.load(cached_path)
    else:
        audio_raw, sr = sf.read(wav_path, dtype="float32", always_2d=False)
        if audio_raw.ndim == 2:
            audio_raw = np.mean(audio_raw, axis=1)
        audio = preprocess_audio(audio_raw, sr)
        np.save(cached_path, audio)

    rms = compute_rms_energy(audio)
    f0_stats = compute_f0stats(audio, TARGET_SR)
    return cached_path, rms, f0_stats


def process_split(
    entries: List[Dict],
    split: str,
    cache_dir: Path,
    force: bool,
    logger,
) -> List[Dict]:
    processed_entries: List[Dict] = []
    for entry in entries:
        utt_id = entry["utt_id"]
        for speaker_label in ("source", "target"):
            wav_path = Path(entry[speaker_label]["path"])
            cached_path, rms, f0_stats = preprocess_and_cache(
                wav_path, split, utt_id, speaker_label, cache_dir, force
            )
            processed_entries.append(
                {
                    "utt_id": utt_id,
                    "speaker": speaker_label,
                    "cached_path": str(cached_path.resolve()),
                    "rms": rms,
                    "f0_stats": f0_stats,
                }
            )
        logger.info("Processed %s pair %s", split, utt_id)
    return processed_entries


def main() -> None:
    args = parse_args()
    set_seeds()
    logger = get_logger("precompute_features")

    manifest_path = args.manifest if args.manifest.is_absolute() else (ROOT / args.manifest)
    manifest = load_manifest(manifest_path)

    cache_dir = args.cache_dir if args.cache_dir.is_absolute() else (ROOT / args.cache_dir)
    ensure_dir(cache_dir)

    stats_path = cache_dir / "preprocess_stats.json"

    logger.info("Preprocessing %d train and %d test pairs.", len(manifest["train"]), len(manifest["test"]))
    train_stats = process_split(manifest["train"], "train", cache_dir, args.force, logger)
    test_stats = process_split(manifest["test"], "test", cache_dir, args.force, logger)

    save_json({"train": train_stats, "test": test_stats}, stats_path)
    logger.info("Saved preprocessing stats to %s", stats_path)


if __name__ == "__main__":
    main()
