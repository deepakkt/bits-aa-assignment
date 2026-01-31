#!/usr/bin/env python3
"""End-to-end walkthrough of `vc.assignment_api` on one utterance.

This script mirrors the graded API surface (Parts A–D) on the pair
bdl->slt for `arctic_b0041.wav`, following the same stages as the
staged scripts:
01_prepare_dataset -> 02_precompute_features -> 03_train_mapping ->
04_convert_samples -> 05_evaluate.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import soundfile as sf

from vc import assignment_api as api, config


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def describe_audio(label: str, audio: np.ndarray, sr: int) -> None:
    print(
        f"{label}: sr={sr}, len={audio.size}, "
        f"min={float(np.min(audio)):.3f}, max={float(np.max(audio)):.3f}, "
        f"rms={api.compute_rms_energy(audio):.4f}"
    )


def main() -> None:
    src_path = Path("data/cmu_us_bdl_arctic/wav/arctic_b0041.wav")
    tgt_path = Path("data/cmu_us_slt_arctic/wav/arctic_b0041.wav")
    if not src_path.exists() or not tgt_path.exists():
        raise SystemExit("Expected arctic_b0041.wav under both bdl and slt wav folders.")

    # ------------------------------------------------------------------
    # Part 1 analogue: create a tiny data root so load_speaker_data only
    # touches the single demo file.
    # ------------------------------------------------------------------
    mini_root = config.ARTIFACT_ROOT / "demo_single_pair"
    bdl_wav = mini_root / "cmu_us_bdl_arctic" / "wav"
    slt_wav = mini_root / "cmu_us_slt_arctic" / "wav"
    bdl_wav.mkdir(parents=True, exist_ok=True)
    slt_wav.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, bdl_wav / src_path.name)
    shutil.copy2(tgt_path, slt_wav / tgt_path.name)

    section("Part A: load_speaker_data")
    bdl_loaded = api.load_speaker_data("bdl", str(mini_root))
    slt_loaded = api.load_speaker_data("slt", str(mini_root))
    print(f"bdl files loaded={len(bdl_loaded)} | slt files loaded={len(slt_loaded)}")
    src_raw, src_sr = bdl_loaded[0]
    tgt_raw, tgt_sr = slt_loaded[0]

    section("Part A: preprocess_audio")
    src_proc, src_sr = api.preprocess_audio(src_raw, src_sr)
    tgt_proc, tgt_sr = api.preprocess_audio(tgt_raw, tgt_sr)
    describe_audio("source_preproc", src_proc, src_sr)
    describe_audio("target_preproc", tgt_proc, tgt_sr)

    section("Part A: stats")
    print("source_f0_stats:", api.compute_f0_stats(src_proc, src_sr))
    print("target_f0_stats:", api.compute_f0_stats(tgt_proc, tgt_sr))
    print("source_rms:", api.compute_rms_energy(src_proc))
    print("target_rms:", api.compute_rms_energy(tgt_proc))

    # ------------------------------------------------------------------
    # Part 2 analogue: feature extraction for the single pair.
    # ------------------------------------------------------------------
    section("Part B: feature extraction")
    src_f0 = api.extract_f0(src_proc, src_sr)
    tgt_f0 = api.extract_f0(tgt_proc, tgt_sr)
    src_mfcc = api.extract_mfcc(src_proc, src_sr)
    tgt_mfcc = api.extract_mfcc(tgt_proc, tgt_sr)
    src_formants = api.extract_formants(src_proc, src_sr)
    tgt_formants = api.extract_formants(tgt_proc, tgt_sr)
    print(f"src_f0 len={src_f0.size} | tgt_f0 len={tgt_f0.size}")
    print(f"src_mfcc shape={src_mfcc.shape} | tgt_mfcc shape={tgt_mfcc.shape}")
    print(f"src_formants={src_formants} | tgt_formants={tgt_formants}")

    section("Part B: pitch shift ratio")
    pitch_ratio = api.calculate_pitch_shift_ratio(src_f0, tgt_f0)
    print(f"pitch_ratio={pitch_ratio:.3f}")

    section("Part B: DTW alignment")
    path = api.align_features_dtw(src_mfcc, tgt_mfcc)
    print(
        f"alignment path length={path.shape[0]} | "
        f"src_frames={src_mfcc.shape[1]} | tgt_frames={tgt_mfcc.shape[1]}"
    )

    section("Part B: train_feature_mapping")
    model = api.train_feature_mapping(src_mfcc, tgt_mfcc)
    print(f"model.feature_dim={model.feature_dim}")

    section("Part B: convert_features")
    converted_mfcc = api.convert_features(model, src_mfcc)
    print(f"converted_mfcc shape={converted_mfcc.shape}")

    # ------------------------------------------------------------------
    # Part 3+4 analogue: spectral envelope + pitch conversion.
    # ------------------------------------------------------------------
    section("Part C: spectral conversion")
    spectral_audio = api.convert_spectral_envelope(src_raw, src_sr, model)
    describe_audio("spectral_only", spectral_audio, config.TARGET_SR)

    section("Part C: pitch shift")
    shifted_audio = api.shift_pitch(spectral_audio, config.TARGET_SR, pitch_ratio)
    describe_audio("spectral_plus_pitch", shifted_audio, config.TARGET_SR)

    section("Part C: voice_conversion_pipeline")
    converted_audio = api.voice_conversion_pipeline(src_raw, src_sr, model, pitch_ratio)
    describe_audio("full_pipeline", converted_audio, config.TARGET_SR)

    out_path = config.OUTPUTS_DIR / "converted" / "arctic_b0041_manual.wav"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, converted_audio, config.TARGET_SR)
    print(f"wrote {out_path}")

    # ------------------------------------------------------------------
    # Part 5 analogue: objective metrics.
    # ------------------------------------------------------------------
    section("Part D: metrics vs target")
    conv_mfcc = api.extract_mfcc(converted_audio, config.TARGET_SR)
    conv_f0 = api.extract_f0(converted_audio, config.TARGET_SR)
    conv_formants = api.extract_formants(converted_audio, config.TARGET_SR)
    mcd = api.calculate_mcd(conv_mfcc, tgt_mfcc)
    f0_corr = api.calculate_f0_correlation(conv_f0, tgt_f0)
    frmse = api.calculate_formant_rmse(conv_formants, tgt_formants)
    print(f"MCD={mcd:.3f}")
    print(f"F0 correlation={f0_corr:.3f}")
    print(f"Formant RMSE={frmse:.3f} Hz")

    section("Completed")
    print("Conversion finished; metrics reported and audio saved.")


if __name__ == "__main__":
    main()
