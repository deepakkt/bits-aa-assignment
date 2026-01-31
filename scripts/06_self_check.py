"""End-to-end self-check harness (Part 10).

This script performs lightweight, deterministic validations to catch common
autograder failures before submission. It only reads existing artifacts; no
files are modified.

Checks performed:
- Required public function signatures (assignment API)
- Preprocessing invariants (16 kHz, normalization, pre-emphasis)
- Feature extraction shapes and NaN handling
- Mapping / conversion sanity (DTW, regression, pitch shift, full pipeline)
- Required output WAVs (sample rate, PCM16, duration, not silent)
- evaluation_results.json schema and finiteness

Exit status is non-zero on any failure.
"""
from __future__ import annotations

import inspect
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import lfilter

from vc import assignment_api as api, config


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str
    fix: str | None = None


def _log_results(results: List[CheckResult]) -> bool:
    failed = [r for r in results if not r.ok]
    for r in results:
        status = "PASS" if r.ok else "FAIL"
        msg = f"[{status}] {r.name}: {r.detail}"
        if r.ok:
            logging.info(msg)
        else:
            logging.error(msg)
            if r.fix:
                logging.error("       fix: %s", r.fix)
    if failed:
        logging.error("Self-check failed (%d issues). See above fixes.", len(failed))
        return False
    logging.info("All self-checks passed (%d items).", len(results))
    return True


# ---------------------------------------------------------------------------
# Signature checks
# ---------------------------------------------------------------------------
EXPECTED_SIGNATURES = {
    "load_speaker_data": ["speaker_id", "data_path"],
    "preprocess_audio": ["audio", "sr"],
    "compute_f0_stats": ["audio", "sr"],
    "compute_rms_energy": ["audio"],
    "extract_f0": ["audio", "sr"],
    "extract_mfcc": ["audio", "sr", "n_mfcc"],
    "extract_formants": ["audio", "sr"],
    "calculate_pitch_shift_ratio": ["source_f0", "target_f0"],
    "align_features_dtw": ["source_features", "target_features"],
    "train_feature_mapping": ["source_features", "target_features"],
    "convert_features": ["model", "source_features"],
    "shift_pitch": ["audio", "sr", "pitch_ratio"],
    "convert_spectral_envelope": ["audio", "sr", "mapping_model"],
    "voice_conversion_pipeline": ["source_audio", "sr", "mapping_model", "pitch_ratio"],
    "calculate_mcd": ["converted_mfcc", "target_mfcc"],
    "calculate_f0_correlation": ["converted_f0", "target_f0"],
    "calculate_formant_rmse": ["converted_formants", "target_formants"],
}


def check_signatures() -> List[CheckResult]:
    results: List[CheckResult] = []
    for func_name, expected_params in EXPECTED_SIGNATURES.items():
        fn = getattr(api, func_name, None)
        if fn is None:
            results.append(
                CheckResult(
                    name=f"signature:{func_name}",
                    ok=False,
                    detail="function missing",
                    fix="Implement in vc.assignment_api with the exact signature.",
                )
            )
            continue

        sig = inspect.signature(fn)
        actual_params = list(sig.parameters.keys())
        ok_params = actual_params == expected_params
        results.append(
            CheckResult(
                name=f"signature:{func_name}",
                ok=ok_params,
                detail=f"params={actual_params}, expected={expected_params}",
                fix=None if ok_params else "Update parameter names/order to match the rubric.",
            )
        )

        if func_name == "extract_mfcc":
            default_n = sig.parameters["n_mfcc"].default
            ok_default = default_n == config.MFCC_N or default_n == 13
            results.append(
                CheckResult(
                    name="signature:extract_mfcc_default",
                    ok=ok_default,
                    detail=f"default n_mfcc={default_n}",
                    fix=None if ok_default else "Set default n_mfcc to 13.",
                )
            )
    return results


# ---------------------------------------------------------------------------
# Synthetic helpers
# ---------------------------------------------------------------------------
def _sine_wave(sr: int, seconds: float, freq: float = 220.0) -> np.ndarray:
    t = np.arange(int(sr * seconds))
    return np.sin(2 * np.pi * freq * t / sr).astype(np.float32)


def _mean_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return float("inf")
    n = min(a.size, b.size)
    return float(np.mean(np.abs(a[:n] - b[:n])))


# ---------------------------------------------------------------------------
# Preprocessing checks
# ---------------------------------------------------------------------------
def check_preprocessing() -> List[CheckResult]:
    results: List[CheckResult] = []
    sr_in = 22050
    audio = 0.6 * _sine_wave(sr_in, 1.0, 220.0) + 0.1 * _sine_wave(sr_in, 1.0, 440.0)

    proc, sr_out = api.preprocess_audio(audio, sr_in)
    max_abs = float(np.max(np.abs(proc))) if proc.size else 0.0
    finite = np.isfinite(proc).all()

    results.append(
        CheckResult(
            name="preprocess:sr",
            ok=sr_out == config.TARGET_SR,
            detail=f"sr_out={sr_out}, expected={config.TARGET_SR}",
            fix="Ensure resampling to TARGET_SR inside preprocess_audio.",
        )
    )
    results.append(
        CheckResult(
            name="preprocess:finite",
            ok=finite,
            detail="all finite" if finite else "NaNs/Infs present",
            fix="Replace NaNs/Infs before returning from preprocess_audio.",
        )
    )
    results.append(
        CheckResult(
            name="preprocess:normalized",
            ok=max_abs <= 1.01 and max_abs > 1e-4,
            detail=f"max_abs={max_abs:.4f}",
            fix="Normalize to [-1,1] and avoid returning silence.",
        )
    )

    # Verify pre-emphasis was applied (compare to manual pipeline)
    resamp = librosa.resample(audio, orig_sr=sr_in, target_sr=config.TARGET_SR, res_type="kaiser_best")
    resamp = resamp.astype(np.float32)
    if resamp.size:
        resamp = resamp / float(np.max(np.abs(resamp)))
    expected = lfilter([1.0, -config.PREEMPH], [1.0], resamp)
    if expected.size:
        expected = expected / float(np.max(np.abs(expected)))
    diff = _mean_abs_diff(proc, expected)
    results.append(
        CheckResult(
            name="preprocess:preemphasis",
            ok=diff < 5e-2,
            detail=f"mean_abs_diff_to_expected={diff:.4f}",
            fix="Order should be resample -> normalize -> pre-emphasize -> normalize.",
        )
    )

    rms = api.compute_rms_energy(proc)
    results.append(
        CheckResult(
            name="preprocess:rms",
            ok=rms > 0.0,
            detail=f"rms={rms:.4f}",
            fix="compute_rms_energy should return positive value for voiced audio.",
        )
    )

    stats = api.compute_f0_stats(proc, sr_out)
    mean_f0 = stats.get("mean_f0", 0.0)
    stats_finite = all(np.isfinite(list(stats.values())))
    results.append(
        CheckResult(
            name="preprocess:f0_stats",
            ok=stats_finite and mean_f0 > 50.0,
            detail=f"stats={stats}",
            fix="Use librosa.yin/pyin and handle unvoiced frames gracefully.",
        )
    )

    return results


# ---------------------------------------------------------------------------
# Feature / mapping checks
# ---------------------------------------------------------------------------
def check_features_and_mapping() -> List[CheckResult]:
    results: List[CheckResult] = []
    sr = config.TARGET_SR
    audio = _sine_wave(sr, 1.0, 220.0)

    f0 = api.extract_f0(audio, sr)
    mfcc = api.extract_mfcc(audio, sr, n_mfcc=config.MFCC_N)
    formants = api.extract_formants(audio, sr)

    results.append(
        CheckResult(
            name="features:f0_len",
            ok=f0.size > 0,
            detail=f"len={f0.size}",
            fix="extract_f0 should return contour with length > 0.",
        )
    )
    results.append(
        CheckResult(
            name="features:f0_finite_fraction",
            ok=np.mean(np.isfinite(f0)) > 0.1,
            detail=f"finite_fraction={np.mean(np.isfinite(f0)):.2f}",
            fix="Ensure pyin/yin returns voiced estimates; keep NaN for unvoiced.",
        )
    )
    results.append(
        CheckResult(
            name="features:mfcc_shape",
            ok=mfcc.shape[0] == config.MFCC_N and mfcc.shape[1] > 0 and np.isfinite(mfcc).all(),
            detail=f"shape={mfcc.shape}",
            fix="MFCCs must be (13, T>0) with finite values.",
        )
    )
    results.append(
        CheckResult(
            name="features:formants",
            ok=formants.shape == (3,) and np.all(np.diff(formants) > 0) and np.isfinite(formants).all(),
            detail=f"formants={formants}",
            fix="Return ascending F1<F2<F3 and fall back to defaults if LPC fails.",
        )
    )

    ratio = api.calculate_pitch_shift_ratio(
        np.array([100.0, np.nan, 120.0], dtype=np.float32),
        np.array([150.0, 140.0, np.nan], dtype=np.float32),
    )
    results.append(
        CheckResult(
            name="features:pitch_ratio",
            ok=0.5 <= ratio <= 2.0,
            detail=f"ratio={ratio:.3f}",
            fix="Use voiced means and clamp ratio to [0.5,2.0].",
        )
    )

    # DTW + regression mapping on random data
    np.random.seed(42)
    src = np.random.randn(config.MFCC_N, 30).astype(np.float32)
    tgt = np.random.randn(config.MFCC_N, 28).astype(np.float32)
    path = api.align_features_dtw(src, tgt)
    mono_ok = path.size > 0 and np.all(np.diff(path[:, 0]) >= 0) and np.all(np.diff(path[:, 1]) >= 0)
    results.append(
        CheckResult(
            name="alignment:dtw_path",
            ok=mono_ok,
            detail=f"path_len={path.shape[0]}",
            fix="align_features_dtw should return monotonic index pairs.",
        )
    )

    model = api.train_feature_mapping(src, tgt)
    converted = api.convert_features(model, src)
    conv_ok = converted.shape == src.shape and np.isfinite(converted).all()
    results.append(
        CheckResult(
            name="mapping:convert_shape",
            ok=conv_ok,
            detail=f"converted_shape={converted.shape}",
            fix="Ensure convert_features preserves shape and finiteness.",
        )
    )

    return results


# ---------------------------------------------------------------------------
# Conversion checks
# ---------------------------------------------------------------------------
def check_conversion() -> List[CheckResult]:
    results: List[CheckResult] = []
    sr = config.TARGET_SR
    audio = _sine_wave(sr, 1.0, 220.0)

    shifted = api.shift_pitch(audio, sr, 1.5)
    dur_diff = abs(len(shifted) - len(audio)) / max(1, len(audio))
    results.append(
        CheckResult(
            name="conversion:shift_pitch_duration",
            ok=dur_diff <= 0.1,
            detail=f"duration_diff={dur_diff:.3f}",
            fix="Preserve duration in shift_pitch (pad/trim as needed).",
        )
    )
    results.append(
        CheckResult(
            name="conversion:shift_pitch_bounds",
            ok=np.all(np.isfinite(shifted)) and np.max(np.abs(shifted)) <= 1.01,
            detail=f"max_abs={float(np.max(np.abs(shifted))):.4f}",
            fix="Clamp amplitudes to avoid clipping and ensure finiteness.",
        )
    )

    full = api.voice_conversion_pipeline(audio, sr, mapping_model=None, pitch_ratio=1.1)
    full_dur_diff = abs(len(full) - len(audio)) / max(1, len(audio))
    results.append(
        CheckResult(
            name="conversion:pipeline_duration",
            ok=full_dur_diff <= 0.1 and full.size > 0,
            detail=f"duration_diff={full_dur_diff:.3f}, len={full.size}",
            fix="Ensure voice_conversion_pipeline returns audio close to input length.",
        )
    )
    results.append(
        CheckResult(
            name="conversion:pipeline_finite",
            ok=np.isfinite(full).all() and np.max(np.abs(full)) > 1e-4,
            detail=f"max_abs={float(np.max(np.abs(full))):.4f}",
            fix="Avoid NaNs/silence in pipeline output.",
        )
    )

    return results


# ---------------------------------------------------------------------------
# Artifact checks
# ---------------------------------------------------------------------------
def _wav_checks(path: Path) -> List[CheckResult]:
    results: List[CheckResult] = []
    if not path.exists():
        results.append(
            CheckResult(
                name=f"artifact:{path.name}",
                ok=False,
                detail="missing",
                fix="Run scripts/04_convert_samples.py to generate required WAVs.",
            )
        )
        return results

    info = sf.info(path)
    audio, sr = sf.read(path, always_2d=False)
    duration = audio.shape[0] / float(sr) if sr else 0.0
    max_abs = float(np.max(np.abs(audio))) if audio.size else 0.0

    results.append(
        CheckResult(
            name=f"artifact:{path.name}:sr_pcm",
            ok=sr == config.TARGET_SR and "PCM" in info.subtype,
            detail=f"sr={sr}, subtype={info.subtype}",
            fix="Save WAVs as 16 kHz PCM16 in scripts/04_convert_samples.py.",
        )
    )
    results.append(
        CheckResult(
            name=f"artifact:{path.name}:duration",
            ok=2.0 <= duration <= 10.0,
            detail=f"duration={duration:.2f}s",
            fix="Ensure output length is between 2–10 seconds.",
        )
    )
    results.append(
        CheckResult(
            name=f"artifact:{path.name}:finite_nonzero",
            ok=np.isfinite(audio).all() and max_abs > 1e-4,
            detail=f"max_abs={max_abs:.4f}",
            fix="Avoid NaNs and silence in generated WAVs.",
        )
    )
    return results


def check_required_wavs() -> List[CheckResult]:
    results: List[CheckResult] = []
    required = [config.OUTPUTS_DIR / f"converted_sample_{i}.wav" for i in (1, 2, 3)]
    for path in required:
        results.extend(_wav_checks(path))
    return results


def check_evaluation_json() -> List[CheckResult]:
    results: List[CheckResult] = []
    path = config.OUTPUTS_DIR / "evaluation_results.json"
    if not path.exists():
        results.append(
            CheckResult(
                name="evaluation_results.json",
                ok=False,
                detail="missing",
                fix="Run scripts/05_evaluate.py to generate evaluation_results.json.",
            )
        )
        return results

    data = json.loads(path.read_text())
    required_top = ["mcd", "f0_correlation", "formant_rmse", "pitch_shift_ratio", "conversion_summary"]
    for key in required_top:
        ok = key in data
        results.append(
            CheckResult(
                name=f"eval:key:{key}",
                ok=ok,
                detail="present" if ok else "missing",
                fix="Ensure evaluation_results.json follows the rubric schema.",
            )
        )
        if not ok:
            return results  # schema missing; other checks would error

    def _finite_list(values: Iterable[float]) -> bool:
        vals = list(values)
        return len(vals) > 0 and all(np.isfinite(v) for v in vals)

    mcd = data["mcd"]
    f0c = data["f0_correlation"]
    frm = data["formant_rmse"]

    results.append(
        CheckResult(
            name="eval:mcd_finite",
            ok=_finite_list([mcd.get("mean"), mcd.get("std")]) and _finite_list(mcd.get("samples", [])),
            detail=f"mean={mcd.get('mean')} std={mcd.get('std')}",
            fix="Populate mcd mean/std/samples with finite floats (length=10).",
        )
    )
    results.append(
        CheckResult(
            name="eval:f0_corr_range",
            ok=_finite_list([f0c.get("mean"), f0c.get("std")])
            and _finite_list(f0c.get("samples", []))
            and all(0.0 <= float(v) <= 1.0 for v in f0c.get("samples", [])),
            detail=f"mean={f0c.get('mean')} std={f0c.get('std')}",
            fix="calculate_f0_correlation must clamp to [0,1]; rerun scripts/05_evaluate.py.",
        )
    )
    results.append(
        CheckResult(
            name="eval:formant_rmse_finite",
            ok=_finite_list([frm.get("f1"), frm.get("f2"), frm.get("f3"), frm.get("average")]),
            detail=f"formant_rmse={frm}",
            fix="Include f1/f2/f3/average as finite floats.",
        )
    )

    pitch_ratio = data.get("pitch_shift_ratio", None)
    results.append(
        CheckResult(
            name="eval:pitch_ratio",
            ok=pitch_ratio is not None and np.isfinite(pitch_ratio) and pitch_ratio > 0,
            detail=f"pitch_shift_ratio={pitch_ratio}",
            fix="Write pitch_shift_ratio as a positive finite float.",
        )
    )

    cs = data.get("conversion_summary", {})
    cs_keys_ok = all(k in cs for k in ["source_speaker", "target_speaker", "num_test_samples", "avg_conversion_quality"])
    results.append(
        CheckResult(
            name="eval:conversion_summary",
            ok=cs_keys_ok,
            detail=f"keys={list(cs.keys())}",
            fix="conversion_summary must include source_speaker, target_speaker, num_test_samples, avg_conversion_quality.",
        )
    )

    return results


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    np.random.seed(42)
    config.ensure_directories()

    results: List[CheckResult] = []
    results.extend(check_signatures())
    results.extend(check_preprocessing())
    results.extend(check_features_and_mapping())
    results.extend(check_conversion())
    results.extend(check_required_wavs())
    results.extend(check_evaluation_json())

    ok = _log_results(results)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
