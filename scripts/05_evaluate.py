"""Compute objective metrics for converted samples (Part 8)."""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
from tqdm import tqdm

from vc import assignment_api as api, config, io_utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate converted audio (Part 8)")
    parser.add_argument(
        "--conversion-manifest",
        type=Path,
        default=config.OUTPUTS_DIR / "converted" / "conversion_manifest.json",
        help="Path to conversion_manifest.json produced by Part 7",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=config.CACHE_DIR / "features",
        help="Feature cache root (uses target cache if available)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=config.OUTPUTS_DIR / "evaluation_results.json",
        help="Destination JSON file for metrics",
    )
    parser.add_argument(
        "--split",
        choices=["train", "test", "all"],
        default="test",
        help="Which split from the conversion manifest to evaluate",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of items to evaluate",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output JSON",
    )
    return parser.parse_args()


def load_conversion_manifest(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Conversion manifest not found at {path}. Run scripts/04_convert_samples.py first."
        )
    return json.loads(path.read_text())


def extract_features(audio_path: Path) -> Dict:
    audio, sr = io_utils.load_audio(audio_path)
    proc_audio, proc_sr = api.preprocess_audio(audio, sr)
    return {
        "sr": int(proc_sr),
        "f0": api.extract_f0(proc_audio, proc_sr),
        "mfcc": api.extract_mfcc(proc_audio, proc_sr, n_mfcc=config.MFCC_N),
        "formants": api.extract_formants(proc_audio, proc_sr),
    }


def load_target_features(
    cache_dir: Path, split: str, target_speaker: str, utt_id: str, target_path: Path
) -> Dict:
    cache_path = cache_dir / split / target_speaker / f"{utt_id}.npz"
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as npz:
            return {
                "sr": int(npz.get("sr", config.TARGET_SR)),
                "f0": np.asarray(npz["f0"], dtype=np.float32),
                "mfcc": np.asarray(npz["mfcc"], dtype=np.float32),
                "formants": np.asarray(npz["formants"], dtype=np.float32),
                "cache_path": str(cache_path),
                "from_cache": True,
            }
    feats = extract_features(target_path)
    feats["cache_path"] = None
    feats["from_cache"] = False
    return feats


def aggregate(values: List[float]) -> Dict:
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return {"mean": None, "std": None, "count": 0, "min": None, "max": None}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "count": int(arr.size),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    np.random.seed(42)
    config.ensure_directories()

    if args.output.exists() and not args.force:
        logging.info("Output %s exists; use --force to overwrite", args.output)
        return

    conv_manifest = load_conversion_manifest(args.conversion_manifest)
    items = conv_manifest.get("items", [])
    if args.split != "all":
        items = [i for i in items if i.get("split") == args.split]
    items = sorted(items, key=lambda r: r.get("utt_id", ""))
    if args.limit:
        items = items[: args.limit]

    logging.info("Evaluating %d items (split=%s)", len(items), args.split)

    per_item: List[Dict] = []
    metrics_pool = {"mcd": [], "f0_correlation": [], "formant_rmse": []}
    counts = {"evaluated": 0, "skipped": 0, "missing": 0}

    tgt_speaker = conv_manifest.get("target_speaker", "")

    for item in tqdm(items, desc="evaluate", ncols=80):
        rec = {
            "utt_id": item.get("utt_id"),
            "split": item.get("split"),
            "output_path": item.get("output_path"),
            "target_path": item.get("target_path"),
            "pitch_ratio": item.get("pitch_ratio"),
        }

        if item.get("status") != "converted":
            counts["skipped"] += 1
            rec["status"] = "skipped"
            rec["reason"] = "not converted"
            per_item.append(rec)
            continue

        out_path = Path(item["output_path"])
        tgt_path = Path(item["target_path"])
        if not out_path.exists() or not tgt_path.exists():
            counts["missing"] += 1
            rec["status"] = "missing"
            rec["reason"] = "missing output or target"
            per_item.append(rec)
            continue

        target_feats = load_target_features(
            cache_dir=args.cache_dir,
            split=item.get("split", "test"),
            target_speaker=tgt_speaker,
            utt_id=item["utt_id"],
            target_path=tgt_path,
        )
        converted_feats = extract_features(out_path)

        mcd = api.calculate_mcd(converted_feats["mfcc"], target_feats["mfcc"])
        f0_corr = api.calculate_f0_correlation(converted_feats["f0"], target_feats["f0"])
        formant_rmse = api.calculate_formant_rmse(converted_feats["formants"], target_feats["formants"])

        rec.update(
            {
                "status": "evaluated",
                "mcd": float(mcd) if np.isfinite(mcd) else None,
                "f0_correlation": float(f0_corr) if np.isfinite(f0_corr) else None,
                "formant_rmse": float(formant_rmse) if np.isfinite(formant_rmse) else None,
                "target_cache_path": target_feats.get("cache_path"),
                "target_from_cache": target_feats.get("from_cache", False),
            }
        )

        metrics_pool["mcd"].append(rec["mcd"])
        metrics_pool["f0_correlation"].append(rec["f0_correlation"])
        metrics_pool["formant_rmse"].append(rec["formant_rmse"])
        counts["evaluated"] += 1
        per_item.append(rec)

    summary = {
        "source_speaker": conv_manifest.get("source_speaker"),
        "target_speaker": tgt_speaker,
        "conversion_manifest": str(args.conversion_manifest),
        "model_path": conv_manifest.get("model_path"),
        "cache_dir": str(args.cache_dir),
        "output_path": str(args.output),
        "split": args.split,
        "limit": args.limit,
        "counts": counts,
        "metrics": {k: aggregate(v) for k, v in metrics_pool.items()},
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "items": per_item,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))
    logging.info("Evaluation complete -> %s", args.output)


if __name__ == "__main__":
    main()
