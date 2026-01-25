# Voice Conversion Report Template

## 1. Dataset and Preprocessing
- Describe CMU Arctic speakers used (source/target), total utterances, and the deterministic 40/10 split.
- Note sampling rate (16 kHz), normalization range, pre-emphasis setting, and silence trimming approach.
- Mention any data quality observations (noise, clipping) and how missing files were handled.

## 2. Feature Extraction
- Summarize F0, MFCC, and formant extraction settings (hop length, n_fft, n_mfcc, pyin, LPC order).
- Include rationale for parameter choices and any caching strategy.
- Call out validation steps to ensure shapes, NaN handling, and alignment consistency.

## 3. Alignment and Mapping
- Explain DTW configuration, distance metric, and how alignment paths are validated.
- Detail the feature mapping model (e.g., linear regression) and training data (aligned MFCC pairs).
- Note regularization or stability measures taken to keep mapping deterministic.

## 4. Conversion Experiments
- List the four ablations: no conversion baseline, pitch-only, spectral-only, full system.
- Describe how pitch shifting and spectral envelope conversion are combined; mention duration handling.
- Reference where the three converted samples are stored and how Griffin-Lim (or equivalent) is configured.

## 5. Evaluation
- Present metrics: MCD, F0 correlation, formant RMSE. Include the required JSON schema path.
- For each metric, add one observation and one interpretation tied to the test set.
- Discuss any anomalies (e.g., NaNs avoided, outlier utterances) and how they were mitigated.

## 6. Analysis and Discussion
- Highlight success cases and failure cases with concrete reasons tied to acoustic properties.
- Outline limitations (data size, model simplicity, LPC instability, vocoder artifacts).
- Propose improvement ideas and expected impact on the metrics.

## 7. Figures and Audio
- List the six matplotlib figures with captions that include one observation and one interpretation each.
- Indicate where audio players appear for source, target, and converted samples (three required).
- Note any reproducibility settings (seed values) relevant to the figures.

## 8. Reproducibility Notes
- Record commands or notebook cells needed to regenerate artifacts end-to-end.
- Include seed values, versions of key libraries, and manifest/model/output paths.
