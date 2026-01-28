# Project State Snapshot (2026-01-28)

This file captures the current implementation state to speed up future parts.

## Parts completion
- [x] Part 1: Env/deps/skeleton
- [x] Part 2: Dataset prep + manifest (bdl -> slt, CMU Arctic)
- [x] Part 3: Preprocessing functions (load, resample->normalize->preemphasize, RMS, F0 stats)
- [ ] Part 4: Feature extraction + caching
- [ ] Part 5: Alignment + mapping
- [ ] Part 6: Pitch modification
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
- `src/vc/assignment_api.py`: exposes Part A functions; Parts B–D stubs raise NotImplementedError.
- `src/vc/__init__.py`: exports config, io_utils, audio_preproc, assignment_api.

## Scripts status
- `scripts/01_prepare_dataset.py`: implemented; builds deterministic 50-pair manifest (40 train / 10 test) under `artifacts/manifests/pair_manifest.json`; idempotent; optional `--download`.
- `scripts/02_precompute_features.py` .. `06_self_check.py`: placeholders pending future parts.

## Current run order
1. `python scripts/01_prepare_dataset.py`  # regenerates manifest if missing (use `--force` to overwrite)

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

## Pending next steps
- Implement Part 4 feature extractors in `src/vc/features.py` and wire into `assignment_api.py`; add caching script `02_precompute_features.py`.
- Keep README.md updated as parts land; ensure idempotent `--force` flags on new scripts.
