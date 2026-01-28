"""Prepare CMU Arctic dataset and deterministic pair manifest (Part 2).

This script is idempotent and will:
- Ensure CMU Arctic source/target speakers are present (uses existing extract, tarball, or optional download).
- Generate a deterministic manifest with 50 parallel utterances (40 train / 10 test).
- Print basic duration statistics for quick validation.

Default speakers: bdl (source) -> slt (target). Override with CLI flags.
"""
from __future__ import annotations

import argparse
import json
import logging
import tarfile
import urllib.request
from pathlib import Path
from typing import Dict, Tuple

import soundfile as sf

from vc import config, io_utils

NUM_PAIRS = 50
TRAIN_SPLIT = 40  # Remaining 10 are test


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare CMU Arctic dataset and manifest")
    parser.add_argument("--data-root", type=Path, default=config.DATA_ROOT, help="Dataset root directory")
    parser.add_argument("--source", default=config.DEFAULT_SOURCE_SPK, help="Source speaker ID (e.g., bdl)")
    parser.add_argument("--target", default=config.DEFAULT_TARGET_SPK, help="Target speaker ID (e.g., slt)")
    parser.add_argument(
        "--manifest-out",
        type=Path,
        default=config.MANIFEST_DIR / "pair_manifest.json",
        help="Output manifest path",
    )
    parser.add_argument("--download", action="store_true", help="Download tarballs if missing")
    parser.add_argument("--force", action="store_true", help="Regenerate manifest even if it exists")
    return parser.parse_args()


def find_extracted_dir(data_root: Path, speaker_id: str) -> Path | None:
    """Return the extracted CMU Arctic directory for a speaker, if present."""
    pattern = f"cmu_us_{speaker_id}_arctic*"
    for candidate in sorted(data_root.glob(pattern)):
        wav_dir = candidate / "wav"
        if wav_dir.is_dir():
            return candidate
    return None


def find_tarball(data_root: Path, speaker_id: str) -> Path | None:
    """Return the tar.bz2 path for a speaker if present."""
    pattern = f"cmu_us_{speaker_id}_arctic*.tar.bz2"
    for candidate in sorted(data_root.glob(pattern)):
        if candidate.is_file():
            return candidate
    return None


def download_tarball(data_root: Path, speaker_id: str) -> Path:
    """Download CMU Arctic tarball for the given speaker to data_root."""
    filename = f"cmu_us_{speaker_id}_arctic-0.95-release.tar.bz2"
    url = f"https://cmu-arctic.speech.cs.cmu.edu/{filename}"
    dest = data_root / filename
    if dest.exists():
        logging.info("Tarball already exists: %s", dest)
        return dest
    logging.info("Downloading %s -> %s", url, dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)
    return dest


def extract_tarball(tar_path: Path, data_root: Path) -> Path:
    """Extract tarball to data_root and return extracted directory path."""
    logging.info("Extracting %s", tar_path.name)
    with tarfile.open(tar_path, "r:bz2") as tar:
        members = tar.getmembers()
        tar.extractall(data_root)
    top_level = next(iter({Path(member.name).parts[0] for member in members}))
    return data_root / top_level


def ensure_dataset(data_root: Path, speaker_id: str, allow_download: bool) -> Path:
    """Ensure the dataset for a speaker exists; return path to wav directory."""
    extracted = find_extracted_dir(data_root, speaker_id)
    if extracted:
        logging.info("Found extracted dataset for %s at %s", speaker_id, extracted)
        return extracted / "wav"

    tar_path = find_tarball(data_root, speaker_id)
    if not tar_path:
        if not allow_download:
            raise FileNotFoundError(
                f"Missing dataset for speaker '{speaker_id}'. Place tarball under {data_root} or pass --download."
            )
        tar_path = download_tarball(data_root, speaker_id)

    extracted_dir = extract_tarball(tar_path, data_root)
    wav_dir = extracted_dir / "wav"
    if not wav_dir.exists():
        raise FileNotFoundError(f"Extracted directory missing wav/: {wav_dir}")
    return wav_dir


def collect_utterances(wav_dir: Path) -> Dict[str, Path]:
    """Collect utterance id -> path mapping from a wav directory."""
    files = io_utils.list_audio_files(wav_dir, exts=(".wav",))
    mapping: Dict[str, Path] = {}
    for path in files:
        utt_id = path.stem
        mapping[utt_id] = path
    return mapping


def file_duration_sec(path: Path) -> float:
    info = sf.info(str(path))
    if info.samplerate <= 0:
        return 0.0
    return float(info.frames) / float(info.samplerate)


def build_manifest(
    src_map: Dict[str, Path], tgt_map: Dict[str, Path], src_id: str, tgt_id: str, data_root: Path
) -> Tuple[dict, list[dict]]:
    """Create manifest dict and per-pair entries (ordered)."""
    common_ids = sorted(set(src_map) & set(tgt_map))
    if len(common_ids) < NUM_PAIRS:
        raise ValueError(f"Need at least {NUM_PAIRS} parallel utterances; found {len(common_ids)}")

    selected = common_ids[:NUM_PAIRS]
    pairs: list[dict] = []
    stats = {
        "source_total_duration_sec": 0.0,
        "target_total_duration_sec": 0.0,
        "train_duration_sec": 0.0,
        "test_duration_sec": 0.0,
    }

    for idx, utt_id in enumerate(selected):
        split = "train" if idx < TRAIN_SPLIT else "test"
        src_path = src_map[utt_id]
        tgt_path = tgt_map[utt_id]
        src_dur = file_duration_sec(src_path)
        tgt_dur = file_duration_sec(tgt_path)
        stats["source_total_duration_sec"] += src_dur
        stats["target_total_duration_sec"] += tgt_dur
        if split == "train":
            stats["train_duration_sec"] += src_dur
        else:
            stats["test_duration_sec"] += src_dur

        pairs.append(
            {
                "utt_id": utt_id,
                "source_path": str(src_path.resolve()),
                "target_path": str(tgt_path.resolve()),
                "split": split,
                "source_duration_sec": round(src_dur, 3),
                "target_duration_sec": round(tgt_dur, 3),
            }
        )

    manifest = {
        "source_speaker": src_id,
        "target_speaker": tgt_id,
        "num_pairs": NUM_PAIRS,
        "train_pairs": TRAIN_SPLIT,
        "test_pairs": NUM_PAIRS - TRAIN_SPLIT,
        "data_root": str(data_root.resolve()),
        "pairs": pairs,
        "stats": {k: round(v, 3) for k, v in stats.items()},
    }
    return manifest, pairs


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    config.ensure_directories()

    if args.manifest_out.exists() and not args.force:
        logging.info("Manifest already exists at %s; use --force to regenerate", args.manifest_out)
        return

    src_wav = ensure_dataset(args.data_root, args.source, allow_download=args.download)
    tgt_wav = ensure_dataset(args.data_root, args.target, allow_download=args.download)

    src_map = collect_utterances(src_wav)
    tgt_map = collect_utterances(tgt_wav)
    logging.info(
        "Found %d source files and %d target files; intersecting for parallel set",
        len(src_map),
        len(tgt_map),
    )

    manifest, pairs = build_manifest(src_map, tgt_map, args.source, args.target, args.data_root)

    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest_out.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logging.info("Wrote manifest with %d pairs (train=%d, test=%d) -> %s", NUM_PAIRS, TRAIN_SPLIT, NUM_PAIRS - TRAIN_SPLIT, args.manifest_out)

    # Duration summary (source side)
    train_ids = [p for p in pairs if p["split"] == "train"]
    test_ids = [p for p in pairs if p["split"] == "test"]
    train_dur = sum(p["source_duration_sec"] for p in train_ids)
    test_dur = sum(p["source_duration_sec"] for p in test_ids)
    logging.info("Train duration (source): %.1fs | Test duration (source): %.1fs", train_dur, test_dur)
    logging.info("Example pair: %s -> %s", pairs[0]["source_path"], pairs[0]["target_path"])


if __name__ == "__main__":
    main()
