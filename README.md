# BITS Voice Conversion Assignment (CMU Arctic, Virtual Lab)

Target platform: Python 3.11 in BITS Pilani Virtual Lab. This repository contains a staged, idempotent pipeline for the voice conversion assignment. Parts 1–10 will be implemented incrementally; Parts 1–5 are complete in this snapshot.

## Setup (run once per machine)
1) Create and activate a virtual environment (Python 3.11):
   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
2) (Optional) ensure project is on the module path when running scripts:
   ```bash
   export PYTHONPATH=src
   ```

## Repository layout
- `requirements.txt` pinned dependencies for Python 3.11.
- `src/vc/` package skeleton with shared config and I/O helpers.
- `scripts/01_prepare_dataset.py` .. `scripts/06_self_check.py`: staged pipeline; 01 is implemented for Part 2.
- `artifacts/` (manifests, cache, models, outputs) and `data/` directories are pre-created for determinism.
- `notebooks/` reserved for the final submission notebook.

## Run order (updated as parts land)
1. `python scripts/01_prepare_dataset.py`  # Part 2 (dataset + manifest) — implemented
2. `python scripts/02_precompute_features.py`  # Part 4 (feature caching) — implemented
3. `python scripts/03_train_mapping.py`  # Part 5 (alignment + mapping) — implemented
4. `python scripts/04_convert_samples.py`  # Part 7 (pending)
5. `python scripts/05_evaluate.py`  # Part 8 (pending)
6. `python scripts/06_self_check.py`  # Part 10 (pending)

## Part status checklist
- [x] Part-1: Environment, dependencies, skeleton
- [x] Part-2: Dataset preparation + deterministic manifest
- [x] Part-3: Preprocessing functions
- [x] Part-4: Feature extraction + caching
- [x] Part-5: Alignment + mapping
- [ ] Part-6: Pitch modification
- [ ] Part-7: Spectral conversion + pipeline
- [ ] Part-8: Metrics + evaluation JSON
- [ ] Part-9: Notebook + report
- [ ] Part-10: Self-check harness

## Quick smoke tests
After activating the venv and installing requirements:
```bash
python - <<'PY'
import numpy, scipy, librosa, soundfile, sklearn, matplotlib, tqdm
from vc import config, io_utils
print('Imports OK; TARGET_SR=', config.TARGET_SR)
PY
```
Expected: a short confirmation line and no ImportError.

### Part 2: dataset + manifest
Dataset placement (manual, recommended):
- Place CMU Arctic tarballs under `data/`, e.g. `data/cmu_us_bdl_arctic-0.95-release.tar.bz2` and `data/cmu_us_slt_arctic-0.95-release.tar.bz2`.
- Alternatively extract them to `data/cmu_us_bdl_arctic/` and `data/cmu_us_slt_arctic/` yourself. The script is idempotent and will reuse existing extractions.

Run Part 2 (bdl -> slt default):
```bash
python scripts/01_prepare_dataset.py
# optional: override speakers or paths
# python scripts/01_prepare_dataset.py --source bdl --target slt --data-root data --manifest-out artifacts/manifests/pair_manifest.json --force
```
Expected outputs:
- `artifacts/manifests/pair_manifest.json` with 50 parallel utterances (train=40, test=10), absolute paths, and duration stats.
- Log lines showing file counts and train/test duration summary.

Quick check:
```bash
python - <<'PY'
import json, pathlib
manifest = json.loads(pathlib.Path("artifacts/manifests/pair_manifest.json").read_text())
print(manifest["num_pairs"], "pairs;", manifest["train_pairs"], "train;", manifest["test_pairs"], "test")
print("First pair IDs:", manifest["pairs"][0]["utt_id"])
PY
```

### Part 3: preprocessing functions (load, resample->normalize->preemphasize, RMS, F0 stats)
Run a quick smoke test on the first source utterance:
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
stats = api.compute_f0_stats(proc, new_sr)
print("f0 stats", stats)
assert new_sr == config.TARGET_SR
assert np.max(np.abs(proc)) <= 1.01
PY
```
Expected: new_sr == 16000, RMS > 0 for voiced files, F0 stats with finite numbers (0 if unvoiced).

### Part 4: feature extraction + caching (F0, MFCC, formants)
Run feature precomputation (uses the manifest from Part 2):
```bash
python scripts/02_precompute_features.py
# optional: change cache location or recompute
# python scripts/02_precompute_features.py --cache-dir artifacts/cache/features --force
```
Expected outputs:
- Cached feature files under `artifacts/cache/features/<split>/<speaker>/<utt_id>.npz`
- `artifacts/cache/features/index.json` summarizing all cached entries

Quick check (after running the script):
```bash
PYTHONPATH=src python - <<'PY'
import json, pathlib, numpy as np
from pathlib import Path

index = json.loads(Path("artifacts/cache/features/index.json").read_text())
print("entries", index["total_entries"], "computed", index["computed"])
one = Path(index["entries"][0]["cache_path"])
with np.load(one, allow_pickle=False) as npz:
    print("mfcc shape", npz["mfcc"].shape, "f0 len", npz["f0"].shape[0], "formants", npz["formants"])
PY
```
Expected: 100 entries (source + target for 50 pairs); MFCC shape `(13, T>0)`; F0 length > 0 with some NaNs for unvoiced; formants finite and ascending.

### Part 5: alignment + mapping (DTW + linear regression)
Train the MFCC mapping model (requires Part 4 cache):
```bash
python scripts/03_train_mapping.py
# optional: override locations or retrain
# python scripts/03_train_mapping.py --manifest artifacts/manifests/pair_manifest.json \\
#   --cache-dir artifacts/cache/features --model-out artifacts/models/mapping_linear.joblib --force
```
Expected output:
- `artifacts/models/mapping_linear.joblib` containing a pickled `FeatureMappingModel` and metadata.
- Logs showing aligned frames and training MSE (on the aligned training set).

Quick check (after training):
```bash
PYTHONPATH=src python - <<'PY'
import joblib, numpy as np, pathlib, json
from vc import assignment_api as api, config

bundle = joblib.load(config.MODELS_DIR / "mapping_linear.joblib")
model = bundle["model"]
meta = bundle["metadata"]
print("model feature_dim", model.feature_dim, "speakers", meta["source_speaker"], "->", meta["target_speaker"])
manifest = json.loads(pathlib.Path("artifacts/manifests/pair_manifest.json").read_text())
utt = manifest["pairs"][0]["utt_id"]
src = np.load(f"artifacts/cache/features/train/{manifest['source_speaker']}/{utt}.npz")["mfcc"]
tgt = np.load(f"artifacts/cache/features/train/{manifest['target_speaker']}/{utt}.npz")["mfcc"]
aligned = api.align_features_dtw(src, tgt)
converted = api.convert_features(model, src[:, aligned[:,0]])
print("converted shape", converted.shape)
PY
```
Expected: feature_dim == 13, reasonable aligned path length, converted MFCC shape `(13, frames_aligned)` with finite values.

## Notes
- All dependencies are pinned for reproducibility on Python 3.11; `fastdtw==0.3.4` is the latest PyPI release that supports Py3.11.
- Directories under `artifacts/` and `data/` are created on import via `vc.config.ensure_directories()`.
- Later parts will introduce idempotent scripts with `--force` flags; see future README updates.
