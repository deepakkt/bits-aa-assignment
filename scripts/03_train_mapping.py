"""Train feature mapping model (Part 5: alignment + mapping).

This script:
1) Loads the deterministic manifest (50 paired utterances).
2) Loads cached MFCC features produced by scripts/02_precompute_features.py.
3) Aligns source/target MFCC sequences for the 40 training pairs using DTW.
4) Fits a linear regression mapping from source MFCC -> target MFCC.
5) Saves the model to artifacts/models/mapping_linear.joblib (idempotent).

Re-run with --force to retrain/overwrite.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

from vc import assignment_api as api, config, mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MFCC mapping model (Part 5)")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=config.MANIFEST_DIR / "pair_manifest.json",
        help="Paired manifest produced by Part 2",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=config.CACHE_DIR / "features",
        help="Feature cache root produced by Part 4",
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=config.MODELS_DIR / "mapping_linear.joblib",
        help="Output path for the trained mapping model",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing model file",
    )
    return parser.parse_args()


def load_manifest(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Manifest not found at {path}. Run scripts/01_prepare_dataset.py first."
        )
    return json.loads(path.read_text())


def load_cached_mfcc(cache_dir: Path, split: str, speaker: str, utt_id: str) -> np.ndarray:
    cache_path = cache_dir / split / speaker / f"{utt_id}.npz"
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Missing cache file {cache_path}. Run scripts/02_precompute_features.py."
        )
    with np.load(cache_path, allow_pickle=False) as npz:
        mfcc = np.asarray(npz["mfcc"], dtype=np.float32)
    return mfcc


def collect_aligned_frames(
    pairs: List[Dict],
    cache_dir: Path,
    source_speaker: str,
    target_speaker: str,
) -> Tuple[np.ndarray, np.ndarray, int]:
    X_list: List[np.ndarray] = []
    Y_list: List[np.ndarray] = []
    total_pairs = 0

    for pair in tqdm(pairs, desc="align+stack", ncols=80):
        utt_id = pair["utt_id"]
        src_mfcc = load_cached_mfcc(cache_dir, "train", source_speaker, utt_id)
        tgt_mfcc = load_cached_mfcc(cache_dir, "train", target_speaker, utt_id)

        path = api.align_features_dtw(src_mfcc, tgt_mfcc)
        if path.shape[0] == 0:
            logging.warning("Empty alignment for %s; skipping", utt_id)
            continue

        X = src_mfcc[:, path[:, 0]]  # shape (feat_dim, frames_aligned)
        Y = tgt_mfcc[:, path[:, 1]]

        X_list.append(X.T)  # to (frames, feat_dim)
        Y_list.append(Y.T)
        total_pairs += 1

    if not X_list:
        raise ValueError("No aligned frames collected; ensure cache is present and non-empty.")

    X_all = np.vstack(X_list)
    Y_all = np.vstack(Y_list)
    return X_all, Y_all, total_pairs


def train_mapping(
    X: np.ndarray,
    Y: np.ndarray,
    source_speaker: str,
    target_speaker: str,
):
    reg = LinearRegression()
    reg.fit(X, Y)
    feature_dim = X.shape[1]
    return mapping.FeatureMappingModel(
        regressor=reg,
        feature_dim=feature_dim,
        source_speaker=source_speaker,
        target_speaker=target_speaker,
    )


def save_model(model, path: Path, meta: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "metadata": meta}, path)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    np.random.seed(42)
    config.ensure_directories()

    if args.model_out.exists() and not args.force:
        logging.info("Model already exists at %s (use --force to retrain)", args.model_out)
        return

    manifest = load_manifest(args.manifest)
    train_pairs = [p for p in manifest["pairs"] if p["split"] == "train"]
    train_pairs = sorted(train_pairs, key=lambda p: p["utt_id"])

    logging.info("Collected %d training pairs", len(train_pairs))
    X, Y, used_pairs = collect_aligned_frames(
        train_pairs,
        cache_dir=args.cache_dir,
        source_speaker=manifest["source_speaker"],
        target_speaker=manifest["target_speaker"],
    )
    logging.info("Aligned frames: %d (from %d pairs)", X.shape[0], used_pairs)

    model = train_mapping(X, Y, manifest["source_speaker"], manifest["target_speaker"])
    preds = model.predict(X.T).T  # predict on training data for a quick sanity metric
    mse = float(np.mean(np.square(preds - Y)))

    meta = {
        "source_speaker": manifest["source_speaker"],
        "target_speaker": manifest["target_speaker"],
        "train_pairs": used_pairs,
        "aligned_frames": int(X.shape[0]),
        "feature_dim": X.shape[1],
        "mfcc_n": config.MFCC_N,
        "target_sr": config.TARGET_SR,
        "train_mse": mse,
    }

    save_model(model, args.model_out, meta)
    logging.info("Saved model -> %s (train MSE %.4f)", args.model_out, mse)


if __name__ == "__main__":
    main()
