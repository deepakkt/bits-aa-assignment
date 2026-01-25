"""
Placeholder self-check script for the voice conversion assignment.

Later parts will expand this to validate the full pipeline end-to-end. For now
it only verifies that the expected directory layout exists.
"""
from __future__ import annotations

import sys
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from vc.config import (  # noqa: E402
    ARTIFACTS_ROOT,
    CACHE_DIR,
    DATA_ROOT,
    MANIFEST_DIR,
    MODELS_DIR,
    OUTPUTS_DIR,
    TARGET_SR,
)
from vc.io_utils import get_logger  # noqa: E402
from vc import assignment_api as api  # noqa: E402


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


def main() -> None:
    logger = get_logger("self_check")
    required = [
        DATA_ROOT,
        ARTIFACTS_ROOT,
        MANIFEST_DIR,
        CACHE_DIR,
        MODELS_DIR,
        OUTPUTS_DIR,
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        logger.warning("Missing required paths: %s", ", ".join(str(p) for p in missing))
    else:
        logger.info("All core directories are present.")

    missing_funcs = [name for name in REQUIRED_FUNCS if not hasattr(api, name)]
    if missing_funcs:
        logger.error("Missing required API functions: %s", ", ".join(missing_funcs))
    else:
        logger.info("API surface OK (%d functions).", len(REQUIRED_FUNCS))

    manifest = MANIFEST_DIR / "pair_manifest.json"
    if manifest.exists():
        logger.info("Found manifest at %s", manifest)
    else:
        logger.error("Manifest missing at %s", manifest)

    model_path = MODELS_DIR / "mfcc_linear_mapping.pkl"
    if model_path.exists():
        logger.info("Found mapping model at %s", model_path)
    else:
        logger.warning("Mapping model missing at %s", model_path)

    outputs = [
        OUTPUTS_DIR / "converted_sample_1.wav",
        OUTPUTS_DIR / "converted_sample_2.wav",
        OUTPUTS_DIR / "converted_sample_3.wav",
        OUTPUTS_DIR / "evaluation_results.json",
    ]
    for p in outputs:
        if not p.exists():
            logger.warning("Expected output missing: %s", p)
        elif p.suffix == ".wav":
            info = sf.info(p)
            if info.samplerate != TARGET_SR:
                logger.error("Output %s has wrong sample rate %s (expected %d)", p.name, info.samplerate, TARGET_SR)
            else:
                logger.info("Output %s present with SR=%d", p.name, info.samplerate)
        else:
            logger.info("Found evaluation JSON at %s", p)

    logger.info("Self-check complete.")


if __name__ == "__main__":
    main()
