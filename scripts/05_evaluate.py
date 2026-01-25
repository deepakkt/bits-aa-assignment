"""
Evaluate converted features against target references on the test split.

This script loads cached features, applies the trained mapping to source MFCCs,
computes MCD, F0 correlation, and formant RMSE metrics, and saves the results to
``artifacts/outputs/evaluation_results.json``.
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from vc.config import (  # noqa: E402
    CACHE_DIR,
    MANIFEST_DIR,
    MODELS_DIR,
    OUTPUTS_DIR,
    set_seeds,
)
from vc.features import calculate_pitch_shift_ratio  # noqa: E402
from vc.io_utils import ensure_dir, get_logger, load_json, save_json  # noqa: E402
from vc.mapping import FeatureMappingModel, convert_features  # noqa: E402
from vc.metrics import mcd, calculate_f0correlation, calculate_formant_rmse  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate voice conversion outputs.")
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
        help="Trained mapping model path from 03_train_mapping.py",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=OUTPUTS_DIR / "evaluation_results.json",
        help="Destination for evaluation JSON.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute metrics even if output already exists.",
    )
    return parser.parse_args()


def load_manifest(manifest_path: Path) -> Dict:
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found at {manifest_path}. Run scripts/01_prepare_dataset.py first."
        )
    return load_json(manifest_path)


def _feature_path(cache_dir: Path, split: str, utt_id: str, speaker: str, kind: str) -> Path:
    return cache_dir / "features" / split / f"{utt_id}_{speaker}_{kind}.npy"


def load_feature(cache_dir: Path, split: str, utt_id: str, speaker: str, kind: str) -> np.ndarray:
    path = _feature_path(cache_dir, split, utt_id, speaker, kind)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing cached {kind} at {path}. Run scripts/02_precompute_features.py first."
        )
    return np.load(path)


def evaluate_pair(
    utt_id: str,
    cache_dir: Path,
    model: FeatureMappingModel,
    logger,
) -> Dict[str, float]:
    src_mfcc = load_feature(cache_dir, "test", utt_id, "source", "mfcc")
    tgt_mfcc = load_feature(cache_dir, "test", utt_id, "target", "mfcc")
    src_f0 = load_feature(cache_dir, "test", utt_id, "source", "f0")
    tgt_f0 = load_feature(cache_dir, "test", utt_id, "target", "f0")
    src_formants = load_feature(cache_dir, "test", utt_id, "source", "formants")
    tgt_formants = load_feature(cache_dir, "test", utt_id, "target", "formants")

    converted_mfcc = convert_features(model, src_mfcc)
    ratio = calculate_pitch_shift_ratio(src_f0, tgt_f0)
    converted_f0 = np.asarray(src_f0, dtype=np.float32) * ratio
    converted_formants = np.asarray(src_formants, dtype=np.float32) * ratio

    return {
        "mcd": mcd(converted_mfcc, tgt_mfcc),
        "f0_corr": calculate_f0correlation(converted_f0, tgt_f0),
        "formant_rmse": calculate_formant_rmse(converted_formants, tgt_formants),
        "ratio": ratio,
        "formant_diff": (converted_formants - tgt_formants).astype(np.float32),
    }


def aggregate_results(per_sample: List[Dict[str, float]]) -> Dict[str, object]:
    mcd_vals = np.asarray([r["mcd"] for r in per_sample], dtype=np.float32)
    f0_vals = np.asarray([r["f0_corr"] for r in per_sample], dtype=np.float32)
    formant_vals = np.asarray([r["formant_rmse"] for r in per_sample], dtype=np.float32)
    formant_diffs = np.asarray(
        [r["formant_diff"] for r in per_sample if "formant_diff" in r], dtype=np.float32
    )

    results = {
        "mcd": {
            "mean": float(np.mean(mcd_vals)) if mcd_vals.size else 0.0,
            "std": float(np.std(mcd_vals)) if mcd_vals.size else 0.0,
            "samples": [float(x) for x in mcd_vals.tolist()],
        },
        "f0_correlation": {
            "mean": float(np.mean(f0_vals)) if f0_vals.size else 0.0,
            "std": float(np.std(f0_vals)) if f0_vals.size else 0.0,
            "samples": [float(x) for x in f0_vals.tolist()],
        },
    }

    if formant_diffs.size:
        f1 = float(np.sqrt(np.mean(np.square(formant_diffs[:, 0]))))
        f2 = float(np.sqrt(np.mean(np.square(formant_diffs[:, 1]))))
        f3 = float(np.sqrt(np.mean(np.square(formant_diffs[:, 2]))))
        mean_rmse = float(np.sqrt(np.mean(np.square(formant_diffs))))
    else:
        f1 = f2 = f3 = mean_rmse = 0.0
    results["formant_rmse"] = {"f1": f1, "f2": f2, "f3": f3, "mean": mean_rmse}
    return results


def main() -> None:
    args = parse_args()
    set_seeds()
    logger = get_logger("evaluate")

    manifest_path = args.manifest if args.manifest.is_absolute() else (ROOT / args.manifest)
    cache_dir = args.cache_dir if args.cache_dir.is_absolute() else (ROOT / args.cache_dir)
    model_path = args.model_path if args.model_path.is_absolute() else (ROOT / args.model_path)
    output_path = args.output_path if args.output_path.is_absolute() else (ROOT / args.output_path)

    if output_path.exists() and not args.force:
        logger.info("Found existing evaluation at %s; skipping. Use --force to recompute.", output_path)
        return

    manifest = load_manifest(manifest_path)
    test_entries = manifest.get("test", [])
    if not test_entries:
        raise ValueError("Manifest contains no test pairs for evaluation.")

    if not model_path.exists():
        raise FileNotFoundError(
            f"Mapping model not found at {model_path}. Run scripts/03_train_mapping.py first."
        )
    with model_path.open("rb") as f:
        model = pickle.load(f)
    if not isinstance(model, FeatureMappingModel):
        raise TypeError(f"Loaded object is not a FeatureMappingModel: {type(model)}")

    ensure_dir(output_path.parent)
    per_sample: List[Dict[str, float]] = []
    logger.info("Evaluating %d test pairs.", len(test_entries))
    for entry in test_entries:
        utt_id = entry["utt_id"]
        try:
            metrics = evaluate_pair(utt_id, cache_dir, model, logger)
        except Exception as exc:  # noqa: BLE001
            logger.error("Evaluation failed for %s: %s", utt_id, exc)
            metrics = {
                "mcd": 0.0,
                "f0_corr": 0.0,
                "formant_rmse": 0.0,
                "ratio": 1.0,
                "formant_diff": np.zeros(3, dtype=np.float32),
            }
        per_sample.append(metrics)
        logger.info(
            "Eval %s | MCD: %.3f | F0 corr: %.3f | Formant RMSE: %.3f",
            utt_id,
            metrics["mcd"],
            metrics["f0_corr"],
            metrics["formant_rmse"],
        )

    results = aggregate_results(per_sample)
    save_json(results, output_path)
    logger.info("Saved evaluation results to %s", output_path)


if __name__ == "__main__":
    main()
