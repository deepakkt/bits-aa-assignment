"""Voice conversion toolkit skeleton for BITS Virtual Lab assignment."""

import warnings

# Suppress noisy pkg_resources deprecation warning emitted by librosa dependency chain.
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API",
    category=UserWarning,
)

# Export features module after Part 4 so `vc.features` is available to callers.
from . import config, io_utils, audio_preproc, features, metrics, assignment_api

__all__ = [
    "config",
    "io_utils",
    "audio_preproc",
    "features",
    "metrics",
    "assignment_api",
]

__version__ = "0.1.0"
