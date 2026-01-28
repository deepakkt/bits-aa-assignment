"""Common I/O helpers for audio files and filesystem utilities."""
from pathlib import Path
from typing import Iterable, List, Tuple, Union
import logging

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


PathLike = Union[str, Path]


def list_audio_files(root: PathLike, exts: Iterable[str] = (".wav", ".flac")) -> List[Path]:
    """Recursively list audio files under ``root`` with matching extensions."""
    root_path = Path(root)
    if not root_path.exists():
        logger.warning("Audio root %s does not exist", root_path)
        return []
    exts_l = tuple(e.lower() for e in exts)
    return sorted(
        p for p in root_path.rglob("*") if p.is_file() and p.suffix.lower() in exts_l
    )


def load_audio(path: PathLike) -> Tuple[np.ndarray, int]:
    """Load an audio file as float32 array in range [-1, 1]."""
    data, sr = sf.read(str(path), always_2d=False)
    if data.dtype.kind == "i":
        max_val = np.iinfo(data.dtype).max
        data = data.astype(np.float32) / max_val
    data = np.asarray(data, dtype=np.float32)
    return data, sr


def save_audio(path: PathLike, audio: np.ndarray, sr: int) -> None:
    """Save audio to ``path`` as 16-bit PCM WAV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio.astype(np.float32), sr, subtype="PCM_16")


def ensure_parent(path: PathLike) -> None:
    """Create parent directories for a file path."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
