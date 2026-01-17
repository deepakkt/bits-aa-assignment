"""
Prepare the CMU Arctic dataset and build a deterministic parallel manifest.

This script:
- Ensures the two required speaker subsets are present (downloads if allowed).
- Selects 40 train and 10 test parallel utterances with identical IDs.
- Records basic stats (sample rate, duration) in a JSON manifest.

The manifest is written to ``artifacts/manifests/pair_manifest.json`` and is
idempotent: it will skip regeneration when a valid manifest already exists,
unless ``--force`` is provided.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

SPEAKER_ARCHIVES = {
    "bdl": "http://festvox.org/cmu_arctic/cmu_arctic/packed/cmu_us_bdl_arctic-0.95-release.tar.bz2",
    "awb": "http://festvox.org/cmu_arctic/cmu_arctic/packed/cmu_us_awb_arctic-0.95-release.tar.bz2",
    "clb": "http://festvox.org/cmu_arctic/cmu_arctic/packed/cmu_us_clb_arctic-0.95-release.tar.bz2",
    "jmk": "http://festvox.org/cmu_arctic/cmu_arctic/packed/cmu_us_jmk_arctic-0.95-release.tar.bz2",
    "ksp": "http://festvox.org/cmu_arctic/cmu_arctic/packed/cmu_us_ksp_arctic-0.95-release.tar.bz2",
    "rms": "http://festvox.org/cmu_arctic/cmu_arctic/packed/cmu_us_rms_arctic-0.95-release.tar.bz2",
    "slt": "http://festvox.org/cmu_arctic/cmu_arctic/packed/cmu_us_slt_arctic-0.95-release.tar.bz2",
}

from vc.config import (  # noqa: E402
    ARTIFACTS_ROOT,
    DATA_ROOT,
    DEFAULT_SEED,
    MANIFEST_DIR,
    SOURCE_SPK,
    TARGET_SPK,
    TEST_UTT,
    TRAIN_UTT,
)
from vc.io_utils import ensure_dir, get_logger, load_json, save_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare CMU Arctic dataset and manifest.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DATA_ROOT,
        help="Root directory containing (or to contain) CMU Arctic speaker folders.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=SOURCE_SPK,
        help="Source speaker ID (e.g., bdl).",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=TARGET_SPK,
        help="Target speaker ID (e.g., slt).",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=TRAIN_UTT,
        help="Number of training utterance pairs to include in the manifest.",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=TEST_UTT,
        help="Number of test utterance pairs to include in the manifest.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recreate the manifest even if it already exists.",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Skip download attempts; require data to be present locally.",
    )
    return parser.parse_args()


def is_manifest_valid(manifest_path: Path, expected_train: int, expected_test: int) -> bool:
    """
    Quick sanity check for an existing manifest so we only skip when it is usable.
    """
    if not manifest_path.exists():
        return False
    try:
        manifest = load_json(manifest_path)
    except Exception:
        return False
    if manifest.get("status") == "placeholder":
        return False
    train = manifest.get("train", [])
    test = manifest.get("test", [])
    if len(train) != expected_train or len(test) != expected_test:
        return False
    for entry in train + test:
        for side in ("source", "target"):
            path_str = entry.get(side, {}).get("path")
            if not path_str:
                return False
            if not Path(path_str).exists():
                return False
    return True


def download_with_progress(url: str, dest: Path, logger, chunk_size: int = 8192) -> None:
    """
    Download a URL to dest while reporting size and 5%% progress blocks.
    """
    with urllib.request.urlopen(url) as response, dest.open("wb") as f:
        total_bytes = response.getheader("Content-Length")
        total_bytes = int(total_bytes) if total_bytes else None
        if total_bytes:
            logger.info("Dataset size: %.2f MB", total_bytes / (1024 * 1024))
        else:
            logger.info("Dataset size: unknown (missing Content-Length header)")

        downloaded = 0
        next_progress = 5
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if total_bytes:
                percent = (downloaded / total_bytes) * 100
                while percent >= next_progress:
                    logger.info("Download progress: %d%%", next_progress)
                    next_progress += 5

        if total_bytes:
            logger.info("Download completed: 100%% (%.2f MB)", downloaded / (1024 * 1024))
        else:
            logger.info("Download completed: %.2f MB", downloaded / (1024 * 1024))


def download_speaker_if_missing(
    speaker_id: str, data_root: Path, logger
) -> Optional[Path]:
    """
    Attempt to download and extract a CMU Arctic speaker archive.

    Returns the extracted speaker directory path if successful, otherwise None.
    """
    speaker_dir = data_root / f"cmu_us_{speaker_id}_arctic"
    if speaker_dir.exists():
        return speaker_dir

    archive_url = SPEAKER_ARCHIVES.get(speaker_id.lower())
    if not archive_url:
        logger.warning(
            "No known download URL for speaker %s. Please download manually and place the archive under %s.",
            speaker_id,
            data_root,
        )
        return None

    archive_name = Path(archive_url).name
    archive_path = data_root / archive_name
    ensure_dir(data_root)

    if not archive_path.exists():
        try:
            logger.info("Downloading %s", archive_url)
            download_with_progress(archive_url, archive_path, logger)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Could not download speaker %s automatically (%s). "
                "Please download %s manually and place it under %s.",
                speaker_id,
                exc,
                archive_name,
                data_root,
            )
            return None
    else:
        logger.info("Found existing archive at %s; skipping download.", archive_path)

    try:
        logger.info("Extracting %s", archive_path)
        # r:* auto-detects gzip/bzip2 so we can handle both archive types.
        with tarfile.open(archive_path, "r:*") as tar:
            tar.extractall(path=data_root)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to extract %s: %s", archive_path, exc)
        return None

    resolved = resolve_wav_dir(speaker_id, data_root)
    if resolved:
        return resolved.parent if resolved.name == "wav" else resolved
    return speaker_dir if speaker_dir.exists() else None


def resolve_wav_dir(speaker_id: str, data_root: Path) -> Optional[Path]:
    """
    Locate the wav directory for a given speaker, probing a few common layouts.
    """
    candidates = [
        data_root / f"cmu_us_{speaker_id}_arctic" / "wav",
        data_root / f"cmu_us_{speaker_id}_arctic-0.95-release" / "wav",
        data_root / "cmu_arctic" / f"cmu_us_{speaker_id}_arctic" / "wav",
        data_root / "cmu_arctic" / f"cmu_us_{speaker_id}_arctic-0.95-release" / "wav",
        data_root / f"cmu_us_{speaker_id}_arctic-0.95-release",
        data_root / speaker_id / "wav",
        data_root / f"cmu_us_{speaker_id}_arctic",
        data_root / speaker_id,  # allows pointing directly to wav directory
    ]
    release_variants = sorted(data_root.glob(f"cmu_us_{speaker_id}_arctic*-release"))
    for variant in release_variants:
        candidates.extend([variant / "wav", variant])
    for cand in candidates:
        if not cand.is_dir():
            continue
        if cand.name == "wav":
            return cand
        nested = cand / "wav"
        if nested.is_dir():
            return nested
        if any(cand.glob("*.wav")):
            return cand
    return None


def ensure_speaker_wavs(
    speaker_id: str, data_root: Path, logger, allow_download: bool
) -> Path:
    """
    Ensure a speaker's wav directory is available, downloading if permitted.
    """
    wav_dir = resolve_wav_dir(speaker_id, data_root)
    if wav_dir:
        return wav_dir
    if allow_download:
        downloaded = download_speaker_if_missing(speaker_id, data_root, logger)
        if downloaded:
            wav_dir = resolve_wav_dir(speaker_id, data_root)
            if wav_dir:
                return wav_dir
    raise FileNotFoundError(
        f"Could not locate wav directory for speaker '{speaker_id}' under {data_root}. "
        "Place the extracted CMU Arctic folder manually or allow downloads."
    )


def list_wavs(wav_dir: Path) -> List[Path]:
    return sorted(p for p in wav_dir.glob("*.wav") if p.is_file())


def compute_duration_stats(paths: Iterable[Path]) -> Dict[str, float]:
    durations = []
    samplerates = []
    for p in paths:
        info = sf.info(p)
        if info.samplerate:
            samplerates.append(info.samplerate)
            durations.append(info.frames / info.samplerate if info.frames else 0.0)
    stats: Dict[str, float] = {"count": len(durations)}
    if durations:
        stats["duration_min_sec"] = float(np.min(durations))
        stats["duration_max_sec"] = float(np.max(durations))
        stats["duration_mean_sec"] = float(np.mean(durations))
    if samplerates:
        mode_sr = Counter(samplerates).most_common(1)[0][0]
        stats["samplerate_mode"] = int(mode_sr)
    return stats


def select_parallel_ids(
    source_paths: List[Path], target_paths: List[Path], total_needed: int, seed: int
) -> List[str]:
    source_ids = {p.stem for p in source_paths}
    target_ids = {p.stem for p in target_paths}
    common = sorted(source_ids & target_ids)
    if len(common) < total_needed:
        raise RuntimeError(
            f"Not enough parallel utterances shared between speakers "
            f"(found {len(common)}, need {total_needed})."
        )
    rng = np.random.default_rng(seed)
    selected = np.array(common)
    rng.shuffle(selected)
    return selected[:total_needed].tolist()


def build_split_entries(
    utt_ids: List[str], source_map: Dict[str, Path], target_map: Dict[str, Path]
) -> List[Dict[str, object]]:
    entries: List[Dict[str, object]] = []
    for utt_id in utt_ids:
        src = source_map[utt_id]
        tgt = target_map[utt_id]
        info_src = sf.info(src)
        info_tgt = sf.info(tgt)
        entries.append(
            {
                "utt_id": utt_id,
                "source": {
                    "path": str(src.resolve()),
                    "samplerate": int(info_src.samplerate),
                    "duration_sec": float(info_src.frames / info_src.samplerate)
                    if info_src.samplerate
                    else 0.0,
                },
                "target": {
                    "path": str(tgt.resolve()),
                    "samplerate": int(info_tgt.samplerate),
                    "duration_sec": float(info_tgt.frames / info_tgt.samplerate)
                    if info_tgt.samplerate
                    else 0.0,
                },
            }
        )
    return entries


def main() -> None:
    args = parse_args()
    logger = get_logger("prepare_dataset")
    data_root = args.data_root if args.data_root.is_absolute() else (ROOT / args.data_root)
    ensure_dir(ARTIFACTS_ROOT)
    ensure_dir(MANIFEST_DIR)

    manifest_path = MANIFEST_DIR / "pair_manifest.json"
    if not args.force and is_manifest_valid(manifest_path, args.train_size, args.test_size):
        logger.info("Found existing manifest at %s; skipping regeneration.", manifest_path)
        return

    total_needed = args.train_size + args.test_size
    logger.info("Preparing dataset with source=%s target=%s", args.source, args.target)

    wav_dir_src = ensure_speaker_wavs(args.source, data_root, logger, allow_download=not args.no_download)
    wav_dir_tgt = ensure_speaker_wavs(args.target, data_root, logger, allow_download=not args.no_download)

    source_paths = list_wavs(wav_dir_src)
    target_paths = list_wavs(wav_dir_tgt)
    logger.info("Found %d source wavs and %d target wavs.", len(source_paths), len(target_paths))

    stats_source = compute_duration_stats(source_paths)
    stats_target = compute_duration_stats(target_paths)

    selected_ids = select_parallel_ids(source_paths, target_paths, total_needed, seed=DEFAULT_SEED)
    source_map = {p.stem: p for p in source_paths}
    target_map = {p.stem: p for p in target_paths}

    train_ids = selected_ids[: args.train_size]
    test_ids = selected_ids[args.train_size :]

    manifest = {
        "dataset": "CMU Arctic",
        "data_root": str(data_root.resolve()),
        "source_speaker": args.source,
        "target_speaker": args.target,
        "train_size": args.train_size,
        "test_size": args.test_size,
        "train": build_split_entries(train_ids, source_map, target_map),
        "test": build_split_entries(test_ids, source_map, target_map),
        "stats": {"source": stats_source, "target": stats_target},
    }

    save_json(manifest, manifest_path)
    logger.info("Wrote manifest to %s", manifest_path)
    logger.info("Train pairs: %d | Test pairs: %d", len(train_ids), len(test_ids))
    logger.info(
        "Duration stats (sec) - source mean: %.2f, target mean: %.2f",
        stats_source.get("duration_mean_sec", 0.0),
        stats_target.get("duration_mean_sec", 0.0),
    )


if __name__ == "__main__":
    main()
