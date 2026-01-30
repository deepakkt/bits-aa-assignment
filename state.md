# Project State Snapshot (2026-01-30)

This file captures the current implementation state to speed up future parts.

## Parts completion
- [x] Part 1: Env/deps/skeleton
- [x] Part 2: Dataset prep + manifest (bdl -> slt, CMU Arctic)
- [x] Part 3: Preprocessing functions (load, resample->normalize->preemphasize, RMS, F0 stats)
- [x] Part 4: Feature extraction + caching
- [x] Part 5: Alignment + mapping (DTW + linear regression)
- [x] Part 6: Pitch modification
- [x] Part 7: Spectral conversion + pipeline
- [x] Part 8: Metrics + evaluation JSON
- [ ] Part 9: Notebook + report
- [ ] Part 10: Self-check harness

## Key constants (vc.config)
- `TARGET_SR = 16000`
- `PREEMPH = 0.97`
- `MFCC_N = 13`
- Default speakers: `bdl` (source) -> `slt` (target)
- Paths: `data/`, `artifacts/{manifests,cache,models,outputs}`, `notebooks/`

## Implemented modules
- `src/vc/config.py`: paths/constants + `ensure_directories()`.
- `src/vc/io_utils.py`: list/load/save audio helpers.
- `src/vc/audio_preproc.py`: Part A logic (load_speaker_data, preprocess_audio, compute_f0_stats, compute_rms_energy) with silence trim, resample, normalize, pre-emphasis.
- `src/vc/features.py`: Part B feature extractors (F0 via pyin fallback to yin, MFCC, LPC formants) + pitch shift ratio helper.
- `src/vc/alignment.py`: DTW alignment using fastdtw, returns path indices.
- `src/vc/mapping.py`: LinearRegression-based FeatureMappingModel + predict/convert helpers.
- `src/vc/conversion.py`: Pitch shifting + spectral envelope conversion + full voice conversion pipeline (MFCC mapping -> Griffin-Lim -> pitch shift).
- `src/vc/metrics.py`: Part D metrics (MCD, F0 correlation, formant RMSE) with DTW alignment and robust resampling.
- `src/vc/assignment_api.py`: exposes Parts A–D (preprocess, features, DTW, mapping, spectral conversion, pipeline, metrics).
- `src/vc/__init__.py`: exports config, io_utils, audio_preproc, features, metrics, assignment_api.

## Scripts status
- `scripts/01_prepare_dataset.py`: implemented; builds deterministic 50-pair manifest (40 train / 10 test) under `artifacts/manifests/pair_manifest.json`; idempotent; optional `--download`.
- `scripts/02_precompute_features.py`: implemented; caches F0/MFCC/formants per utterance into `artifacts/cache/features/...`, idempotent with `--force`.
- `scripts/03_train_mapping.py`: implemented; loads cached MFCCs, aligns via DTW, trains LinearRegression mapping, saves to `artifacts/models/mapping_linear.joblib` (idempotent with `--force`).
- `scripts/04_convert_samples.py`: implemented; runs full conversion pipeline on chosen split (default test), saves WAVs and `conversion_manifest.json` under `artifacts/outputs/converted`.
- `scripts/05_evaluate.py`: implemented; evaluates converted outputs, reuses cached target features, writes aggregate + per-utt metrics to `artifacts/outputs/evaluation_results.json`.
  - Latest run (2026-01-30, test split): evaluated=10, mcd≈900.1, f0_corr≈-0.027, formant_rmse≈447.8.
- `scripts/06_self_check.py`: placeholder pending future parts.

## Current run order
1. `python scripts/01_prepare_dataset.py`  # regenerates manifest if missing (use `--force` to overwrite)
2. `python scripts/02_precompute_features.py`  # caches F0/MFCC/formants (use `--force` to recompute)
3. `python scripts/03_train_mapping.py`  # align MFCCs with DTW, train LinearRegression mapping
4. `python scripts/04_convert_samples.py`  # run full Part 7 pipeline on test split (or specify --split/--limit)
5. `PYTHONPATH=src python scripts/05_evaluate.py --force`  # compute Part 8 metrics, writes evaluation_results.json

## Quick smoke test (Part 3)
```bash
python - <<'PY'
import json, pathlib
import soundfile as sf
import numpy as np
from vc import assignment_api as api, config

manifest = json.loads(pathlib.Path("artifacts/manifests/pair_manifest.json").read_text())
first = manifest["pairs"][0]["source_path"]
audio, sr = sf.read(first)
proc, new_sr = api.preprocess_audio(audio, sr)
print("orig_sr", sr, "new_sr", new_sr, "len", len(proc))
print("rms", api.compute_rms_energy(proc))
print("f0 stats", api.compute_f0_stats(proc, new_sr))
assert new_sr == config.TARGET_SR
assert np.max(np.abs(proc)) <= 1.01
PY
```

## Quick sanity (Part 5)
Requires manifest + cached features:
```bash
python scripts/03_train_mapping.py --force
PYTHONPATH=src python - <<'PY'
import joblib, json, pathlib, numpy as np
from vc import assignment_api as api, config

bundle = joblib.load(config.MODELS_DIR / "mapping_linear.joblib")
model = bundle["model"]
manifest = json.loads(pathlib.Path("artifacts/manifests/pair_manifest.json").read_text())
utt = manifest["pairs"][0]["utt_id"]
src = np.load(f"artifacts/cache/features/train/{manifest['source_speaker']}/{utt}.npz")["mfcc"]
tgt = np.load(f"artifacts/cache/features/train/{manifest['target_speaker']}/{utt}.npz")["mfcc"]
path = api.align_features_dtw(src, tgt)
print("alignment len", len(path))
converted = api.convert_features(model, src[:, path[:,0]])
print("converted shape", converted.shape, "finite", np.isfinite(converted).all())
PY
```

## Quick sanity (Part 7)
Requires manifest, cache, and trained model:
```bash
python scripts/04_convert_samples.py --limit 2 --force
python - <<'PY'
import json, pathlib
root = pathlib.Path("artifacts/outputs/converted")
meta = json.loads((root / "conversion_manifest.json").read_text())
print(meta["converted"], "converted;", "example:", meta["items"][0]["output_path"])
PY
```
Expected: converted files written and listed in the conversion manifest.

## Pending next steps
- Notebook/report (Part 9) and self-check harness (Part 10).

## Dependency note
- Added `resampy==0.4.3` to requirements to satisfy librosa pitch shifting backend.
- Extend conversion + evaluation scripts once mapping is stable; keep README.md updated with commands and artifacts.
