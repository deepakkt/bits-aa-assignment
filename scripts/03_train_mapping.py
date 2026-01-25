"""
Train a frame-wise MFCC mapping using DTW-aligned parallel utterances.

This script:
- loads cached MFCC features produced by ``02_precompute_features.py``
- aligns each train pair with DTW and caches the path
- trains a linear regression mapping from source->target MFCCs
- saves the model under ``artifacts/models`` along with a small metadata JSON
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from vc.alignment import align_features_dtw  # noqa: E402
from vc.config import CACHE_DIR, MANIFEST_DIR, MODELS_DIR, set_seeds  # noqa: E402
from vc.io_utils import ensure_dir, get_logger, load_json, save_json  # noqa: E402
from vc.mapping import FeatureMappingModel, train_feature_mapping  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train MFCC mapping model using DTW-aligned parallel data."
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
        help="Cache directory containing precomputed features.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=MODELS_DIR / "mfcc_linear_mapping.pkl",
        help="Output path for the trained mapping model.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Retrain even if a saved model already exists.",
    )
    return parser.parse_args()


def load_manifest(manifest_path: Path) -> Dict:
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found at {manifest_path}. Run scripts/01_prepare_dataset.py first."
        )
    return load_json(manifest_path)


def _feature_path(cache_dir: Path, split: str, utt_id: str, speaker: str) -> Path:
    return cache_dir / "features" / split / f"{utt_id}_{speaker}_mfcc.npy"


def load_mfcc(cache_dir: Path, split: str, utt_id: str, speaker: str) -> np.ndarray:
    path = _feature_path(cache_dir, split, utt_id, speaker)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing MFCC cache at {path}. Run scripts/02_precompute_features.py first."
        )
    mfcc = np.load(path)
    if mfcc.ndim != 2:
        raise ValueError(f"MFCC array at {path} has invalid shape {mfcc.shape}")
    return mfcc.astype(np.float32)


def load_or_compute_path(
    src_mfcc: np.ndarray,
    tgt_mfcc: np.ndarray,
    cache_dir: Path,
    split: str,
    utt_id: str,
    force: bool,
) -> np.ndarray:
    """
    Cache DTW paths to avoid recomputation across runs.
    """
    align_dir = ensure_dir(cache_dir / "alignment" / split)
    path_file = align_dir / f"{utt_id}_dtw_path.npy"
    if path_file.exists() and not force:
        path = np.load(path_file)
    else:
        path = align_features_dtw(src_mfcc, tgt_mfcc)
        np.save(path_file, path)
    return np.asarray(path, dtype=np.int32)


def build_aligned_training_data(
    manifest: Dict,
    cache_dir: Path,
    force: bool,
    logger,
) -> Tuple[List[np.ndarray], List[np.ndarray], int]:
    """
    Produce aligned frame sequences for all training pairs.

    Returns aligned source list, aligned target list, and total frame count.
    """
    aligned_src: List[np.ndarray] = []
    aligned_tgt: List[np.ndarray] = []
    total_frames = 0

    for entry in manifest.get("train", []):
        utt_id = entry["utt_id"]
        src_mfcc = load_mfcc(cache_dir, "train", utt_id, "source")
        tgt_mfcc = load_mfcc(cache_dir, "train", utt_id, "target")

        path = load_or_compute_path(src_mfcc, tgt_mfcc, cache_dir, "train", utt_id, force)
        aligned_src.append(src_mfcc[:, path[:, 0]])
        aligned_tgt.append(tgt_mfcc[:, path[:, 1]])

        frame_count = aligned_src[-1].shape[1]
        total_frames += frame_count
        logger.info(
            "Aligned %s: %d->%d frames via DTW",
            utt_id,
            src_mfcc.shape[1],
            frame_count,
        )
    return aligned_src, aligned_tgt, total_frames


def save_model(model: FeatureMappingModel, path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("wb") as f:
        pickle.dump(model, f)


def main() -> None:
    args = parse_args()
    set_seeds()
    logger = get_logger("train_mapping")

    manifest_path = args.manifest if args.manifest.is_absolute() else (ROOT / args.manifest)
    model_path = args.model_path if args.model_path.is_absolute() else (ROOT / args.model_path)
    cache_dir = args.cache_dir if args.cache_dir.is_absolute() else (ROOT / args.cache_dir)

    if model_path.exists() and not args.force:
        logger.info("Model already exists at %s; skipping. Use --force to retrain.", model_path)
        return

    manifest = load_manifest(manifest_path)
    if not manifest.get("train"):
        raise ValueError("Manifest contains no training pairs.")

    aligned_src, aligned_tgt, total_frames = build_aligned_training_data(
        manifest, cache_dir, args.force, logger
    )
    if total_frames == 0:
        raise RuntimeError("No aligned frames available for training.")

    logger.info("Training linear regression on %d aligned frames.", total_frames)
    model = train_feature_mapping(aligned_src, aligned_tgt)
    save_model(model, model_path)

    metadata = {
        "model_path": str(model_path.resolve()),
        "manifest": str(manifest_path.resolve()),
        "cache_dir": str(cache_dir.resolve()),
        "pairs": len(aligned_src),
        "total_aligned_frames": total_frames,
        "status": "ok",
    }
    save_json(metadata, model_path.with_suffix(".json"))
    logger.info("Saved mapping model and metadata under %s", model_path.parent)


if __name__ == "__main__":
    main()
