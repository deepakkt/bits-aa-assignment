# Voice Conversion Assignment (BITS AIML ZG527)

This repository implements a classical voice conversion pipeline on the CMU Arctic dataset using paired utterances from two speakers (default: `bdl` as source, `slt` as target). The project is built part-by-part to stay idempotent and runnable in constrained environments.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Repository layout
- `src/vc/`: core library modules (configuration, IO utilities, APIs)
- `scripts/`: entrypoints for dataset prep, feature caching, model training, conversion, evaluation, self-checks
- `artifacts/`: manifests, caches, models, and outputs (created automatically)
- `data/`: CMU Arctic dataset root (place manually or let scripts download when available)
- `notebooks/`: submission notebook and report template

## Run order (when all parts are implemented)
1. `python scripts/01_prepare_dataset.py`
2. `python scripts/02_precompute_features.py`
3. `python scripts/03_train_mapping.py`
4. `python scripts/04_convert_samples.py`
5. `python scripts/05_evaluate.py`
6. `python scripts/06_self_check.py`

Each script is designed to be idempotent and accepts a `--force` flag (added in later parts) to recompute outputs.
