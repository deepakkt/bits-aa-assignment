"""Precompute and cache features for all paired utterances (Part 4).

This script reads the deterministic manifest produced in Part 2 and stores
F0, MFCC, and formant features for each source/target utterance under
``artifacts/cache/features/<split>/<speaker>/<utt_id>.npz``.

Idempotent: existing cache files are reused unless ``--force`` is provided.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from tqdm import tqdm

from vc import assignment_api as api, config, io_utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute features (Part 4)")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=config.MANIFEST_DIR / "pair_manifest.json",
        help="Path to paired manifest JSON",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=config.CACHE_DIR / "features",
        help="Root directory for cached feature files",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute features even if cache files already exist",
    )
    return parser.parse_args()


def load_manifest(manifest_path: Path) -> Dict:
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found at {manifest_path}. Run scripts/01_prepare_dataset.py first."
        )
    return json.loads(manifest_path.read_text())


def _feature_entry(
    utt_id: str,
    speaker: str,
    split: str,
    cache_path: Path,
    audio_path: Path,
    f0: np.ndarray,
    mfcc: np.ndarray,
    formants: np.ndarray,
) -> Dict:
    return {
        "utt_id": utt_id,
        "speaker": speaker,
        "split": split,
        "cache_path": str(cache_path),
        "audio_path": str(audio_path),
        "f0_frames": int(f0.shape[0]),
        "mfcc_shape": [int(mfcc.shape[0]), int(mfcc.shape[1])],
        "formants_hz": [float(x) for x in formants[:3]],
    }


def compute_and_save(
    utt_id: str,
    speaker: str,
    split: str,
    audio_path: Path,
    cache_root: Path,
    force: bool,
) -> Tuple[str, Dict]:
    cache_path = cache_root / split / speaker / f"{utt_id}.npz"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists() and not force:
        # Reuse metadata from disk to keep index complete
        try:
            with np.load(cache_path, allow_pickle=False) as npz:
                f0 = npz["f0"]
                mfcc = npz["mfcc"]
                formants = npz["formants"]
            return "skipped", _feature_entry(
                utt_id, speaker, split, cache_path, audio_path, f0, mfcc, formants
            )
        except Exception:
            logging.info("Cache %s unreadable; recomputing", cache_path)

    audio, sr = io_utils.load_audio(audio_path)
    audio_proc, sr_proc = api.preprocess_audio(audio, sr)

    f0 = api.extract_f0(audio_proc, sr_proc)
    mfcc = api.extract_mfcc(audio_proc, sr_proc, n_mfcc=config.MFCC_N)
    formants = api.extract_formants(audio_proc, sr_proc)

    np.savez(
        cache_path,
        f0=f0,
        mfcc=mfcc,
        formants=formants,
        sr=sr_proc,
        utt_id=utt_id,
        speaker=speaker,
        split=split,
        audio_path=str(audio_path),
    )
    return "computed", _feature_entry(
        utt_id, speaker, split, cache_path, audio_path, f0, mfcc, formants
    )


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    np.random.seed(42)
    config.ensure_directories()

    manifest = load_manifest(args.manifest)
    cache_root: Path = args.cache_dir
    cache_root.mkdir(parents=True, exist_ok=True)

    entries = []
    counts = {"computed": 0, "skipped": 0}

    pairs = sorted(manifest["pairs"], key=lambda p: p["utt_id"])
    for split in ("train", "test"):
        split_pairs = [p for p in pairs if p["split"] == split]
        if not split_pairs:
            continue
        logging.info("Processing %d %s pairs", len(split_pairs), split)
        for pair in tqdm(split_pairs, desc=f"{split} pairs", ncols=80):
            for role, speaker in [
                ("source", manifest["source_speaker"]),
                ("target", manifest["target_speaker"]),
            ]:
                audio_path = Path(pair[f"{role}_path"])
                status, entry = compute_and_save(
                    utt_id=pair["utt_id"],
                    speaker=speaker,
                    split=split,
                    audio_path=audio_path,
                    cache_root=cache_root,
                    force=args.force,
                )
                counts[status] += 1
                entries.append(entry)

    index = {
        "cache_dir": str(cache_root),
        "computed": counts["computed"],
        "reused": counts["skipped"],
        "total_entries": len(entries),
        "splits": {
            "train": len([e for e in entries if e["split"] == "train"]),
            "test": len([e for e in entries if e["split"] == "test"]),
        },
        "mfcc_n": config.MFCC_N,
        "target_sr": config.TARGET_SR,
        "entries": entries,
    }

    index_path = cache_root / "index.json"
    index_path.write_text(json.dumps(index, indent=2))
    logging.info(
        "Feature cache complete. computed=%d reused=%d -> %s",
        counts["computed"],
        counts["skipped"],
        index_path,
    )


if __name__ == "__main__":
    main()
