# BITS Voice Conversion Assignment (CMU Arctic, Virtual Lab)

Target platform: Python 3.11 in BITS Pilani Virtual Lab. This repository contains a staged, idempotent pipeline for the voice conversion assignment. Parts 1–10 will be implemented incrementally; only Part 1 (environment + skeleton) is complete in this snapshot.

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
   export PYTHONPATH=.
   ```

## Repository layout
- `requirements.txt` pinned dependencies for Python 3.11.
- `src/vc/` package skeleton with shared config and I/O helpers.
- `scripts/01_prepare_dataset.py` .. `scripts/06_self_check.py`: placeholders for upcoming parts.
- `artifacts/` (manifests, cache, models, outputs) and `data/` directories are pre-created for determinism.
- `notebooks/` reserved for the final submission notebook.

## Run order (will be filled as parts land)
1. `python scripts/01_prepare_dataset.py`  # Part 2 (pending)
2. `python scripts/02_precompute_features.py`  # Part 4 (pending)
3. `python scripts/03_train_mapping.py`  # Part 5 (pending)
4. `python scripts/04_convert_samples.py`  # Part 7 (pending)
5. `python scripts/05_evaluate.py`  # Part 8 (pending)
6. `python scripts/06_self_check.py`  # Part 10 (pending)

## Part status checklist
- [x] Part-1: Environment, dependencies, skeleton
- [ ] Part-2: Dataset preparation + deterministic manifest
- [ ] Part-3: Preprocessing functions
- [ ] Part-4: Feature extraction + caching
- [ ] Part-5: Alignment + mapping
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

## Notes
- All dependencies are pinned for reproducibility on Python 3.11; `fastdtw==0.3.4` is the latest PyPI release that supports Py3.11.
- Directories under `artifacts/` and `data/` are created on import via `vc.config.ensure_directories()`.
- Later parts will introduce idempotent scripts with `--force` flags; see future README updates.
