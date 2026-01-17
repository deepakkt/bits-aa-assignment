# AGENTS.md — Voice Conversion Assignment Builder Agent Guide

This document is the operating manual for an AI agent that generates a **full-credit, self-contained** voice conversion experiment and submission (Jupyter notebook + required artifacts) for the **BITS AIML ZG527 Voice Conversion assignment**, adapted to use **CMU Arctic** due to compute/storage constraints.

The user will request work **part-by-part** (Part-1, Part-2, …). Assume all earlier parts are already completed and present on disk. Each part must be atomic, runnable, and as idempotent as possible.

---

## 1) Mission

Build a classical voice conversion pipeline that:
- Uses **paired parallel utterances** from **CMU Arctic** (e.g., `bdl` ↔ `slt`) to enable DTW-aligned feature mapping.
- Implements all required **auto-graded API functions** with exact names/signatures.
- Produces required deliverables:
  - `converted_sample_1.wav`
  - `converted_sample_2.wav`
  - `converted_sample_3.wav`
  - `evaluation_results.json`
- Includes notebook-based analysis with required plots, metrics, and write-up.

**Success = evaluator can run** `Restart & Run All` and observe:
- Files created in expected locations
- Audio playable and sensible
- Metrics computed and JSON schema correct
- Report present with insightful analysis
- All required API functions exist and behave correctly

---

## 2) Non-negotiable constraints

### Dataset constraint
- Do **NOT** use the 10 GiB dataset (VCTK full). Use **CMU Arctic** “multiple single-speaker English”.
- Use **two speakers** with **parallel prompt IDs** (same utterance IDs), e.g.:
  - Source: `bdl`
  - Target: `slt`
- Use **50 paired utterances total**:
  - **40 train**
  - **10 test**
- Cache and operate only on these subsets (not full corpora).

### Audio & preprocessing constraint
- All processed audio must be:
  - **16 kHz**
  - normalized to **[-1, 1]**
  - pre-emphasis: `y[n] = x[n] - 0.97*x[n-1]`
- Silence trimming must not destroy content (conservative thresholding).

### Deliverable constraint
- Final submission must be a **single self-contained Jupyter notebook** that:
  - imports from `src/vc`
  - runs scripts / functions end-to-end
  - displays figures using **matplotlib only**
  - embeds audio players for source/target/converted examples

### Idempotency constraint
- Every script must:
  - create needed directories
  - skip recomputation if outputs exist
  - accept `--force` to override
- Use deterministic splitting and cached manifests.

### Charting constraint
- Use **matplotlib** only.
- Avoid hardcoding colors unless asked.
- One figure per chart (no subplots unless explicitly required).

---

## 3) Repository layout (fixed)

Do not rename folders. Create missing paths if needed.

```
./
  requirements.txt
  README.md
  AGENTS.md
  src/vc/
      __init__.py
      config.py
      io_utils.py
      audio_preproc.py
      features.py
      alignment.py
      mapping.py
      conversion.py
      metrics.py
      assignment_api.py
  scripts/
      01_prepare_dataset.py
      02_precompute_features.py
      03_train_mapping.py
      04_convert_samples.py
      05_evaluate.py
      06_self_check.py
  data/                 # dataset root (user-managed)
  artifacts/
      manifests/
      cache/
      models/
      outputs/
  notebooks/
      VoiceConversion_Submission.ipynb
      report_template.md
```

---

## 4) Execution model

### Virtual environment
Assume user creates a venv and installs requirements from `requirements.txt`.

### Part-by-part workflow
- The user will ask for “Part-X”.
- The agent must generate **only Part-X** outputs, based on the state created by Parts 1..X-1.

### Output format in responses (agent behavior)
For every part response:

1) For each file created or modified:
- `FILE: relative/path/to/file`
- one fenced code block containing **full file contents** (not a diff)

2) Then:
- `RUN:` bullet list of commands (assume venv active)
- `CHECKS:` bullet list of quick sanity checks

Avoid extra narrative.

---

## 5) Auto-graded API contract (must match exactly)

All required functions live in: `src/vc/assignment_api.py`

### Part A (data + preprocessing)
- `load_speaker_data(speaker_id: str, data_path: str) -> list`
- `compute_f0stats(audio, sr) -> dict`  
  keys: `mean_f0, std_f0, min_f0, max_f0` (ignore NaNs)
- `compute_rms_energy(audio) -> float`
- `preprocess_audio(audio, sr)` -> preprocessed audio array (SR fixed at 16 kHz)

### Part B (features + alignment + mapping)
- `extract_f0(audio, sr) -> np.ndarray`
- `calculate_pitch_shift_ratio(source_f0, target_f0) -> float` (clamp [0.5, 2.0])
- `extract_mfcc(audio, sr, n_mfcc=13) -> np.ndarray` shape (13, T)
- `extract_formants(audio, sr) -> np.ndarray` shape (3,), sorted (F1<F2<F3)
- `align_features_dtw(source_features, target_features)` -> alignment path
- `train_feature_mapping(source_features, target_features)` -> trained model
- `convert_features(model, source_features)` -> converted features array

### Part C (conversion)
- `shift_pitch(audio, sr, pitch_ratio) -> np.ndarray` (duration within ±5%)
- `convert_spectral_envelope(audio, sr, mapping_model) -> np.ndarray` (converted spectrogram)

### Part D (evaluation)
- `mcd(converted_mfcc, target_mfcc) -> float`
- `calculate_f0correlation(converted_f0, target_f0) -> float` in [0,1] (handle NaNs)
- `calculate_formant_rmse(converted_formants, target_formants) -> float`

**Important:** If the assignment spec differs in return types (e.g., returning `(audio, sr)`), adapt strictly to the PDF spec—but keep internal helpers stable.

---

## 6) Algorithmic expectations (what “correct” means)

### Feature extraction
- **MFCC**: 13 coefficients, consistent hop length across pipeline
- **F0**: robust extraction (e.g., `librosa.pyin`), NaNs for unvoiced frames
- **Formants**: LPC-based estimate; stable and sorted; explain limitations

### Alignment
- DTW uses MFCC distance (e.g., L2) and returns a valid path mapping frames.
- Path must be non-empty and monotonic.

### Mapping
- **Baseline mapping must exist**: Linear Regression from aligned MFCC frames.
- Optional: GMM mapping if stable, but do not risk fragility.

### Conversion
- Provide ablations in notebook:
  1. no conversion
  2. pitch-only
  3. spectral-only
  4. full system
- Reconstruct waveform with Griffin-Lim (acceptable but slow—cache outputs).

---

## 7) Metrics & JSON schema (must be exact)

Write: `artifacts/outputs/evaluation_results.json`

Schema:

```json
{
  "mcd": {"mean": 0.0, "std": 0.0, "samples": [0.0]},
  "f0_correlation": {"mean": 0.0, "std": 0.0, "samples": [0.0]},
  "formant_rmse": { "f1": 0.0, "f2": 0.0, "f3": 0.0, "mean": 0.0 }
}
```

- `samples` arrays correspond to **10 test utterances**.
- All values must be finite (no NaNs/inf).
- If additional keys are added, keep these keys intact and correct.

---

## 8) Notebook requirements (what earns full credit)

The notebook must include:
- Dataset stats and sanity checks
- Printed pitch ratio and split summary
- Audio playback cells for:
  - Source
  - Target
  - Converted (3 required samples)
- 6 figures (matplotlib), each with:
  - title, axes labels (units), legends where applicable
  - a caption that includes **one observation + one interpretation**
- An analysis/report section (Part E) that includes:
  - quantitative results interpretation (MCD, F0 corr, formant RMSE)
  - success & failure cases with concrete reasons
  - limitations + improvement ideas

---

## 9) Coding standards and consistency rules

### Determinism
- Set seeds once in `config.py`.
- Manifest split must be deterministic and stored to disk.

### Performance
- Cache intermediate features in `artifacts/cache/`.
- Avoid recomputing MFCC/F0/formants on every run.
- Keep notebook runtime reasonable.

### Error handling
- Fail loudly with clear messages.
- Validate inputs: SR, dtype, shape, NaN checks.

### Comments
- Comment “hard parts” in plain English:
  - DTW path interpretation
  - MCD computation and why DTW alignment is used
  - LPC formant estimation pitfalls
  - MFCC → mel → linear magnitude approximations

### Matplotlib
- Use `plt.figure()` per plot.
- Do not use seaborn.
- Do not set global styles or colors unless asked.

---

## 10) Change control (how to modify earlier parts)

Avoid breaking changes. If a later part requires modifying earlier files:
- Keep changes minimal.
- Maintain public function signatures.
- If you must change file outputs/paths, provide a migration note and ensure old paths still work (symlink or compatibility loader) where feasible.

---

## 11) Definition of Done (DoD) for the whole project

A complete run passes:

1. `python scripts/01_prepare_dataset.py` creates manifest with 40/10 split.
2. `python scripts/02_precompute_features.py` caches MFCC/F0/formants.
3. `python scripts/03_train_mapping.py` saves mapping model.
4. `python scripts/04_convert_samples.py` writes 3 converted WAVs (16kHz, PCM16, 2–10s).
5. `python scripts/05_evaluate.py` writes `evaluation_results.json` with correct schema.
6. `python scripts/06_self_check.py` prints PASS for:
   - API function presence
   - SR/normalization/preemphasis
   - shapes and NaN handling
   - output file existence and schema correctness
7. Notebook `VoiceConversion_Submission.ipynb` runs top-to-bottom with plots + audio + report.

---

## 12) Quick pitfalls checklist (avoid these)

- Returning MFCC shape (T,13) instead of (13,T)
- Forgetting pre-emphasis or using wrong coefficient
- Pitch ratio computed from medians but reported as means (be consistent)
- F0 correlation computed including NaNs/unvoiced frames (must mask)
- Formant extraction unstable (returning zeros/NaNs); must guard and explain
- Griffin-Lim too slow (cache and keep sample count small)
- JSON schema mismatch (keys or nesting wrong)
- WAVs saved as float instead of PCM16; wrong SR

---

## 13) How to respond when user asks for a part

- Generate ONLY the requested part’s files + RUN + CHECKS.
- Assume earlier artifacts exist.
- If prerequisites are missing, fail gracefully:
  - explain exactly what file is missing
  - provide a minimal command to create it (do not rebuild everything)

End of AGENTS.md
