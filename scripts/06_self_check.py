"""
Self-check the voice conversion pipeline artifacts against assignment DoD.

This script validates:
- Required API functions exist
- Manifest split (40 train / 10 test) and SR=16 kHz inputs
- Preprocessing invariants: normalization to [-1, 1] and pre-emphasis applied
- Cached feature shapes and NaN handling
- Model artifact presence and type
- Converted WAV constraints (3 files, 2–10s, 16 kHz, PCM16)
- evaluation_results.json schema and finite values

It is read-only and exits non-zero on any violation.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from vc import assignment_api as api  # noqa: E402
from vc.audio_preproc import preprocess_audio  # noqa: E402
from vc.config import (  # noqa: E402
    ARTIFACTS_ROOT as CFG_ARTIFACTS_ROOT,
    CACHE_DIR as CFG_CACHE_DIR,
    DATA_ROOT as CFG_DATA_ROOT,
    MANIFEST_DIR as CFG_MANIFEST_DIR,
    MFCC_N,
    MODELS_DIR as CFG_MODELS_DIR,
    OUTPUTS_DIR as CFG_OUTPUTS_DIR,
    TARGET_SR,
    TEST_UTT,
    TRAIN_UTT,
)
from vc.io_utils import get_logger, load_json  # noqa: E402
from vc.mapping import FeatureMappingModel  # noqa: E402

DATA_ROOT = ROOT / CFG_DATA_ROOT
ARTIFACTS_ROOT = ROOT / CFG_ARTIFACTS_ROOT
MANIFEST_DIR = ROOT / CFG_MANIFEST_DIR
CACHE_DIR = ROOT / CFG_CACHE_DIR
MODELS_DIR = ROOT / CFG_MODELS_DIR
OUTPUTS_DIR = ROOT / CFG_OUTPUTS_DIR

REQUIRED_FUNCS = [
    "load_speaker_data",
    "compute_f0stats",
    "compute_rms_energy",
    "preprocess_audio",
    "extract_f0",
    "calculate_pitch_shift_ratio",
    "extract_mfcc",
    "extract_formants",
    "align_features_dtw",
    "train_feature_mapping",
    "convert_features",
    "shift_pitch",
    "convert_spectral_envelope",
    "mcd",
    "calculate_f0correlation",
    "calculate_formant_rmse",
]

MIN_WAV_SEC = 2.0
MAX_WAV_SEC = 10.0
MODEL_FILE = "mfcc_linear_mapping.pkl"
EVAL_FILE = "evaluation_results.json"


def record_error(errors: List[str], logger, message: str) -> None:
    errors.append(message)
    logger.error(message)


def _preprocessed_path(split: str, utt_id: str, speaker: str) -> Path:
    return CACHE_DIR / "preprocessed" / split / f"{utt_id}_{speaker}.npy"


def _feature_path(split: str, utt_id: str, speaker: str, kind: str) -> Path:
    return CACHE_DIR / "features" / split / f"{utt_id}_{speaker}_{kind}.npy"


def check_directories(logger, errors: List[str]) -> None:
    required = [
        DATA_ROOT,
        ARTIFACTS_ROOT,
        MANIFEST_DIR,
        CACHE_DIR,
        MODELS_DIR,
        OUTPUTS_DIR,
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        record_error(errors, logger, f"Missing required paths: {', '.join(missing)}")
    else:
        logger.info("All core directories are present.")


def check_api_surface(logger, errors: List[str]) -> None:
    missing_funcs = [name for name in REQUIRED_FUNCS if not hasattr(api, name)]
    if missing_funcs:
        record_error(errors, logger, f"Missing required API functions: {', '.join(missing_funcs)}")
    else:
        logger.info("API surface OK (%d functions).", len(REQUIRED_FUNCS))


def load_manifest(logger, errors: List[str]) -> Optional[Dict]:
    manifest_path = MANIFEST_DIR / "pair_manifest.json"
    if not manifest_path.exists():
        record_error(errors, logger, f"Manifest missing at {manifest_path}")
        return None
    try:
        manifest = load_json(manifest_path)
    except Exception as exc:  # noqa: BLE001
        record_error(errors, logger, f"Failed to load manifest {manifest_path}: {exc}")
        return None

    train = manifest.get("train", [])
    test = manifest.get("test", [])
    if len(train) != TRAIN_UTT or len(test) != TEST_UTT:
        record_error(
            errors,
            logger,
            f"Manifest split mismatch (train={len(train)}, test={len(test)}; expected {TRAIN_UTT}/{TEST_UTT})",
        )

    seen_ids = set()
    for split_name, entries in (("train", train), ("test", test)):
        for entry in entries:
            utt_id = entry.get("utt_id")
            if not utt_id:
                record_error(errors, logger, f"Missing utt_id in {split_name} entry: {entry}")
                continue
            if utt_id in seen_ids:
                record_error(errors, logger, f"Duplicate utt_id {utt_id} across manifest splits.")
            seen_ids.add(utt_id)
            for speaker in ("source", "target"):
                info = entry.get(speaker, {})
                wav_path = Path(info.get("path", ""))
                sr = info.get("samplerate")
                if sr != TARGET_SR:
                    record_error(
                        errors,
                        logger,
                        f"{split_name}/{utt_id}/{speaker} samplerate={sr} (expected {TARGET_SR})",
                    )
                if not wav_path.exists():
                    record_error(errors, logger, f"WAV path missing for {split_name}/{utt_id}/{speaker}: {wav_path}")
    if not errors:
        logger.info("Manifest validated with %d train and %d test pairs.", len(train), len(test))
    return manifest


def check_preprocessed_audio(
    manifest: Dict,
    logger,
    errors: List[str],
    warnings: List[str],
) -> None:
    for split_name in ("train", "test"):
        for entry in manifest.get(split_name, []):
            utt_id = entry["utt_id"]
            for speaker in ("source", "target"):
                path = _preprocessed_path(split_name, utt_id, speaker)
                if not path.exists():
                    record_error(errors, logger, f"Cached preprocessed audio missing at {path}")
                    continue
                try:
                    audio = np.load(path)
                except Exception as exc:  # noqa: BLE001
                    record_error(errors, logger, f"Failed to load {path}: {exc}")
                    continue
                if audio.ndim != 1:
                    record_error(errors, logger, f"Preprocessed audio at {path} has invalid shape {audio.shape}")
                if audio.size == 0:
                    record_error(errors, logger, f"Preprocessed audio at {path} is empty.")
                if not np.isfinite(audio).all():
                    record_error(errors, logger, f"Non-finite values in preprocessed audio at {path}")
                max_abs = float(np.max(np.abs(audio))) if audio.size else 0.0
                if max_abs > 1.0005:
                    record_error(errors, logger, f"Preprocessed audio exceeds [-1,1] at {path} (max={max_abs:.4f})")
                elif max_abs == 0.0:
                    warnings.append(f"Preprocessed audio at {path} is silent after normalization.")

    # Re-run preprocessing on a representative pair to verify pre-emphasis/normalization.
    sample_entry = manifest.get("train", [None])[0]
    if sample_entry:
        utt_id = sample_entry["utt_id"]
        for speaker in ("source", "target"):
            raw_path = Path(sample_entry[speaker]["path"])
            cached_path = _preprocessed_path("train", utt_id, speaker)
            if not raw_path.exists() or not cached_path.exists():
                continue
            raw_audio, sr = sf.read(raw_path, dtype="float32", always_2d=False)
            if raw_audio.ndim == 2:
                raw_audio = np.mean(raw_audio, axis=1)
            expected = preprocess_audio(raw_audio, sr)
            cached = np.load(cached_path)
            if expected.shape != cached.shape or not np.allclose(
                expected, cached, rtol=1e-3, atol=1e-4
            ):
                record_error(
                    errors,
                    logger,
                    f"Preprocessing mismatch for {utt_id}/{speaker}; pre-emphasis or normalization may be incorrect.",
                )
            else:
                logger.info("Preprocessing (SR/normalize/pre-emphasis) verified for %s/%s.", utt_id, speaker)


def check_feature_caches(
    manifest: Dict,
    logger,
    errors: List[str],
    warnings: List[str],
) -> None:
    for split_name in ("train", "test"):
        for entry in manifest.get(split_name, []):
            utt_id = entry["utt_id"]
            for speaker in ("source", "target"):
                f0_path = _feature_path(split_name, utt_id, speaker, "f0")
                mfcc_path = _feature_path(split_name, utt_id, speaker, "mfcc")
                formant_path = _feature_path(split_name, utt_id, speaker, "formants")

                if not f0_path.exists():
                    record_error(errors, logger, f"Missing F0 cache at {f0_path}")
                    continue
                f0 = np.load(f0_path)
                if f0.ndim != 1 or f0.size == 0:
                    record_error(errors, logger, f"F0 cache at {f0_path} has invalid shape {f0.shape}")
                if np.isinf(f0).any():
                    record_error(errors, logger, f"F0 cache at {f0_path} contains inf values")
                if not np.isfinite(f0).any():
                    warnings.append(f"F0 at {f0_path} is entirely NaN; pitch ratio will default to 1.0.")

                if not mfcc_path.exists():
                    record_error(errors, logger, f"Missing MFCC cache at {mfcc_path}")
                    continue
                mfcc = np.load(mfcc_path)
                if mfcc.ndim != 2 or mfcc.shape[0] != MFCC_N or mfcc.shape[1] == 0:
                    record_error(errors, logger, f"MFCC cache at {mfcc_path} has invalid shape {mfcc.shape}")
                if not np.isfinite(mfcc).all():
                    record_error(errors, logger, f"MFCC cache at {mfcc_path} contains NaN/inf values")

                if not formant_path.exists():
                    record_error(errors, logger, f"Missing formant cache at {formant_path}")
                    continue
                formants = np.load(formant_path).reshape(-1)
                if formants.size != 3:
                    record_error(errors, logger, f"Formant cache at {formant_path} has invalid shape {formants.shape}")
                if not np.isfinite(formants).all():
                    record_error(errors, logger, f"Formant cache at {formant_path} contains NaN/inf values")
                if formants.size == 3 and not np.all(np.diff(formants) >= 0):
                    record_error(errors, logger, f"Formants at {formant_path} are not sorted ascending: {formants}")


def check_model_artifact(logger, errors: List[str]) -> Optional[Path]:
    model_path = MODELS_DIR / MODEL_FILE
    if not model_path.exists():
        record_error(errors, logger, f"Mapping model missing at {model_path}")
        return None
    try:
        with model_path.open("rb") as f:
            model = pickle.load(f)
    except Exception as exc:  # noqa: BLE001
        record_error(errors, logger, f"Failed to load model at {model_path}: {exc}")
        return None
    if not isinstance(model, FeatureMappingModel):
        record_error(errors, logger, f"Loaded object is not FeatureMappingModel: {type(model)}")
    elif getattr(model, "n_features", None) != MFCC_N:
        record_error(errors, logger, f"Model expects {getattr(model, 'n_features', None)} features (expected {MFCC_N})")
    else:
        logger.info("Model artifact loaded and validated at %s", model_path)
    return model_path


def check_converted_outputs(logger, errors: List[str], warnings: List[str]) -> None:
    for idx in range(1, 4):
        path = OUTPUTS_DIR / f"converted_sample_{idx}.wav"
        if not path.exists():
            record_error(errors, logger, f"Converted sample missing: {path}")
            continue
        try:
            info = sf.info(path)
        except Exception as exc:  # noqa: BLE001
            record_error(errors, logger, f"Could not read {path}: {exc}")
            continue
        if info.samplerate != TARGET_SR:
            record_error(errors, logger, f"{path.name} has SR={info.samplerate} (expected {TARGET_SR})")
        duration = info.frames / info.samplerate if info.samplerate else 0.0
        if duration < MIN_WAV_SEC or duration > MAX_WAV_SEC:
            record_error(errors, logger, f"{path.name} duration {duration:.2f}s outside [{MIN_WAV_SEC}, {MAX_WAV_SEC}]s")
        if "PCM_16" not in info.subtype:
            record_error(errors, logger, f"{path.name} subtype {info.subtype} is not PCM16")

        audio, _ = sf.read(path, dtype="float32", always_2d=False)
        if audio.ndim == 2:
            audio = np.mean(audio, axis=1)
        if audio.size == 0:
            record_error(errors, logger, f"{path.name} is empty after load.")
        if not np.isfinite(audio).all():
            record_error(errors, logger, f"{path.name} contains NaN/inf samples.")
        max_abs = float(np.max(np.abs(audio))) if audio.size else 0.0
        if max_abs > 1.02:
            warnings.append(f"{path.name} amplitude exceeds [-1,1] (max {max_abs:.3f}).")
        logger.info("%s validated: SR=%d, dur=%.2fs, subtype=%s", path.name, info.samplerate, duration, info.subtype)


def _validate_numeric(value) -> bool:
    return isinstance(value, (int, float, np.floating)) and np.isfinite(value)


def _validate_samples(
    name: str,
    samples: Sequence,
    expected_len: int,
    errors: List[str],
    logger,
) -> None:
    if not isinstance(samples, list):
        record_error(errors, logger, f"{name} samples is not a list.")
        return
    if len(samples) != expected_len:
        record_error(errors, logger, f"{name} samples has length {len(samples)} (expected {expected_len}).")
    for idx, val in enumerate(samples):
        if not _validate_numeric(val):
            record_error(errors, logger, f"{name} sample {idx} is non-finite: {val}")


def check_evaluation_results(manifest: Optional[Dict], logger, errors: List[str]) -> None:
    path = OUTPUTS_DIR / EVAL_FILE
    if not path.exists():
        record_error(errors, logger, f"Evaluation JSON missing at {path}")
        return
    try:
        data = load_json(path)
    except Exception as exc:  # noqa: BLE001
        record_error(errors, logger, f"Failed to parse evaluation JSON: {exc}")
        return

    expected_samples = len(manifest.get("test", [])) if manifest else TEST_UTT

    for metric in ("mcd", "f0_correlation"):
        if metric not in data or not isinstance(data[metric], dict):
            record_error(errors, logger, f"Missing metric block '{metric}' in evaluation JSON.")
            continue
        block = data[metric]
        for key in ("mean", "std"):
            if key not in block or not _validate_numeric(block[key]):
                record_error(errors, logger, f"{metric}.{key} is missing or non-finite: {block.get(key)}")
        _validate_samples(metric, block.get("samples", []), expected_samples, errors, logger)

    formant = data.get("formant_rmse")
    if not isinstance(formant, dict):
        record_error(errors, logger, "formant_rmse block missing or invalid.")
    else:
        for key in ("f1", "f2", "f3", "mean"):
            if key not in formant or not _validate_numeric(formant[key]):
                record_error(errors, logger, f"formant_rmse.{key} is missing or non-finite: {formant.get(key)}")

    if not errors:
        logger.info("Evaluation JSON schema and values validated at %s", path)


def main() -> None:
    logger = get_logger("self_check")
    errors: List[str] = []
    warnings: List[str] = []

    check_directories(logger, errors)
    check_api_surface(logger, errors)
    manifest = load_manifest(logger, errors)

    if manifest:
        check_preprocessed_audio(manifest, logger, errors, warnings)
        check_feature_caches(manifest, logger, errors, warnings)
    else:
        logger.warning("Skipping cache checks because manifest is unavailable.")

    check_model_artifact(logger, errors)
    check_converted_outputs(logger, errors, warnings)
    check_evaluation_results(manifest, logger, errors)

    if errors:
        logger.error("SELF-CHECK FAILED with %d error(s).", len(errors))
        for msg in errors:
            logger.error(" - %s", msg)
        if warnings:
            logger.warning("Warnings encountered: %s", "; ".join(warnings))
        sys.exit(1)

    if warnings:
        logger.warning("Self-check completed with %d warning(s): %s", len(warnings), "; ".join(warnings))
    logger.info("PASS: all self-checks succeeded.")


if __name__ == "__main__":
    main()
