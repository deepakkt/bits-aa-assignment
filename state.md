# Project State Snapshot (2026-01-29)

This file captures the current implementation state to speed up future parts.

## Parts completion
- [x] Part 1: Env/deps/skeleton
- [x] Part 2: Dataset prep + manifest (bdl -> slt, CMU Arctic)
- [x] Part 3: Preprocessing functions (load, resample->normalize->preemphasize, RMS, F0 stats)
- [x] Part 4: Feature extraction + caching
- [x] Part 5: Alignment + mapping (DTW + linear regression)
- [x] Part 6: Pitch modification
- [ ] Part 7: Spectral conversion + pipeline
- [ ] Part 8: Metrics + evaluation JSON
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
- `src/vc/conversion.py`: Part C pitch shifting implementation (ratio clamp, duration preservation, fallback when librosa fails).
- `src/vc/assignment_api.py`: exposes Parts A–B, DTW, mapping, and Part C `shift_pitch`; remaining Part C/D functions still stubbed.
- `src/vc/__init__.py`: exports config, io_utils, audio_preproc, features, assignment_api.

## Scripts status
- `scripts/01_prepare_dataset.py`: implemented; builds deterministic 50-pair manifest (40 train / 10 test) under `artifacts/manifests/pair_manifest.json`; idempotent; optional `--download`.
- `scripts/02_precompute_features.py`: implemented; caches F0/MFCC/formants per utterance into `artifacts/cache/features/...`, idempotent with `--force`.
- `scripts/03_train_mapping.py`: implemented; loads cached MFCCs, aligns via DTW, trains LinearRegression mapping, saves to `artifacts/models/mapping_linear.joblib` (idempotent with `--force`).
- `scripts/04_convert_samples.py` .. `06_self_check.py`: placeholders pending future parts.

## Current run order
1. `python scripts/01_prepare_dataset.py`  # regenerates manifest if missing (use `--force` to overwrite)
2. `python scripts/02_precompute_features.py`  # caches F0/MFCC/formants (use `--force` to recompute)
3. `python scripts/03_train_mapping.py`  # align MFCCs with DTW, train LinearRegression mapping

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

## Pending next steps
- Implement spectral conversion + full pipeline (Part 7), metrics + evaluation JSON (Part 8), notebook/report (Part 9), self-check (Part 10).

## Dependency note
- Added `resampy==0.4.3` to requirements to satisfy librosa pitch shifting backend.
- Extend conversion + evaluation scripts once mapping is stable; keep README.md updated with commands and artifacts.
