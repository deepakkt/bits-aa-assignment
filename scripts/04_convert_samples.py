"""Convert held-out samples using the trained mapping model (Part 7)."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
from tqdm import tqdm

from vc import assignment_api as api, config, io_utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert samples with mapping model (Part 7)")
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
        "--model",
        type=Path,
        default=config.MODELS_DIR / "mapping_linear.joblib",
        help="Trained mapping model produced by Part 5",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=config.OUTPUTS_DIR / "converted",
        help="Directory to store converted waveforms",
    )
    parser.add_argument(
        "--split",
        choices=["train", "test", "all"],
        default="test",
        help="Which manifest split to convert (default: test)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of utterances to convert",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing outputs",
    )
    return parser.parse_args()


def load_manifest(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Manifest not found at {path}. Run scripts/01_prepare_dataset.py first."
        )
    return json.loads(path.read_text())


def load_model(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Mapping model not found at {path}. Train it via scripts/03_train_mapping.py."
        )
    bundle = joblib.load(path)
    return bundle["model"], bundle.get("metadata", {})


def load_cached_f0(cache_dir: Path, split: str, speaker: str, utt_id: str):
    cache_path = cache_dir / split / speaker / f"{utt_id}.npz"
    if not cache_path.exists():
        return None
    with np.load(cache_path, allow_pickle=False) as npz:
        return np.asarray(npz["f0"], dtype=np.float32)


def pitch_ratio_from_cache(cache_dir: Path, split: str, pair: Dict, src_spk: str, tgt_spk: str) -> float:
    src_f0 = load_cached_f0(cache_dir, split, src_spk, pair["utt_id"])
    tgt_f0 = load_cached_f0(cache_dir, split, tgt_spk, pair["utt_id"])
    if src_f0 is None or tgt_f0 is None:
        return 1.0
    return float(api.calculate_pitch_shift_ratio(src_f0, tgt_f0))


def convert_pair(
    pair: Dict,
    model,
    cache_dir: Path,
    split: str,
    out_dir: Path,
    src_spk: str,
    tgt_spk: str,
    force: bool,
) -> Tuple[str, Dict]:
    utt_id = pair["utt_id"]
    out_path = out_dir / f"{utt_id}_converted.wav"
    pitch_ratio = pitch_ratio_from_cache(cache_dir, split, pair, src_spk, tgt_spk)

    meta = {
        "utt_id": utt_id,
        "source_path": pair["source_path"],
        "target_path": pair["target_path"],
        "output_path": str(out_path),
        "split": pair["split"],
        "pitch_ratio": float(pitch_ratio),
        "target_sr": config.TARGET_SR,
    }

    if out_path.exists() and not force:
        return "skipped", meta

    audio, sr = io_utils.load_audio(pair["source_path"])
    converted = api.voice_conversion_pipeline(audio, sr, model, pitch_ratio)
    io_utils.save_audio(out_path, converted, config.TARGET_SR)
    return "converted", meta


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    np.random.seed(42)
    config.ensure_directories()

    manifest = load_manifest(args.manifest)
    model, model_meta = load_model(args.model)

    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs: List[Dict] = manifest["pairs"]
    if args.split != "all":
        pairs = [p for p in pairs if p["split"] == args.split]
    pairs = sorted(pairs, key=lambda p: p["utt_id"])
    if args.limit:
        pairs = pairs[: args.limit]

    logging.info(
        "Converting %d utterances (%s split) using model %s",
        len(pairs),
        args.split,
        args.model,
    )

    records: List[Dict] = []
    counts = {"converted": 0, "skipped": 0}

    for pair in tqdm(pairs, desc="convert", ncols=80):
        status, rec = convert_pair(
            pair,
            model=model,
            cache_dir=args.cache_dir,
            split=pair["split"],
            out_dir=out_dir,
            src_spk=manifest["source_speaker"],
            tgt_spk=manifest["target_speaker"],
            force=args.force,
        )
        counts[status] += 1
        records.append({"status": status, **rec})

    summary = {
        "source_speaker": manifest["source_speaker"],
        "target_speaker": manifest["target_speaker"],
        "model_path": str(args.model),
        "model_meta": model_meta,
        "manifest_path": str(args.manifest),
        "cache_dir": str(args.cache_dir),
        "output_dir": str(out_dir),
        "split": args.split,
        "converted": counts["converted"],
        "skipped": counts["skipped"],
        "total": len(records),
        "items": records,
    }

    manifest_out = out_dir / "conversion_manifest.json"
    manifest_out.write_text(json.dumps(summary, indent=2))
    logging.info(
        "Done. converted=%d skipped=%d -> %s",
        counts["converted"],
        counts["skipped"],
        manifest_out,
    )


if __name__ == "__main__":
    main()
