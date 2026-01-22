"""
Auto-graded API surface for the voice conversion assignment.

Each part of the project will progressively populate this module with the
required functions. Keeping the file present from the start prevents import
errors in notebooks or scripts that preemptively reference it.
"""

from __future__ import annotations

from .audio_preproc import (
    compute_f0stats,
    compute_rms_energy,
    load_speaker_data,
    preprocess_audio,
)
from .features import (
    calculate_pitch_shift_ratio,
    extract_f0,
    extract_formants,
    extract_mfcc,
)

__all__ = [
    "load_speaker_data",
    "compute_f0stats",
    "compute_rms_energy",
    "preprocess_audio",
    "extract_f0",
    "calculate_pitch_shift_ratio",
    "extract_mfcc",
    "extract_formants",
]

# Later parts will extend this namespace with feature extraction, alignment,
# mapping, conversion, and evaluation helpers while preserving these exports.
