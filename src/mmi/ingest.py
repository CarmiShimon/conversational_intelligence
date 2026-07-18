"""Stage 0 -- Ingest.

Responsibilities:
* Resolve the input (local path or Google Drive link -> download via gdown).
* Probe media metadata with ffprobe.
* Extract a 16 kHz mono WAV (the contract expected by WhisperX / ASR).

The video file itself is left untouched; the vision stage reads it directly.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .utils import get_logger

_LOG = get_logger("mmi.ingest")

_DRIVE_RE = re.compile(r"drive\.google\.com")


@dataclass
class MediaInputs:
    video_path: Path
    audio_path: Path
    duration_sec: Optional[float]


def _require(binary: str) -> None:
    if shutil.which(binary) is None:
        raise RuntimeError(
            f"'{binary}' not found on PATH. Install ffmpeg (which provides "
            f"ffmpeg and ffprobe) and ensure it is on PATH."
        )


def resolve_input(source: str, input_dir: Path) -> Path:
    """Return a local video path, downloading from Google Drive if needed."""
    if _DRIVE_RE.search(source):
        return _download_drive(source, input_dir)
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Input video not found: {source}")
    return path


def _download_drive(url: str, input_dir: Path) -> Path:
    input_dir.mkdir(parents=True, exist_ok=True)
    dest = input_dir / "clip.mp4"
    if dest.exists() and dest.stat().st_size > 0:
        _LOG.info("Using cached download: %s", dest)
        return dest
    try:
        import gdown  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "gdown is required to download the Google Drive clip. "
            "Install it (`pip install gdown`) or download the file manually "
            f"and place it at {dest}."
        ) from exc

    _LOG.info("Downloading clip from Google Drive ...")
    out = gdown.download(url=url, output=str(dest), quiet=False, fuzzy=True)
    if not out or not Path(out).exists():
        raise RuntimeError(
            "Google Drive download failed (link may require manual "
            f"confirmation). Download the file manually to {dest} and re-run."
        )
    return Path(out)


def probe_duration(video_path: Path) -> Optional[float]:
    _require("ffprobe")
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "json", str(video_path),
            ],
            capture_output=True, text=True, check=True,
        )
        return float(json.loads(result.stdout)["format"]["duration"])
    except Exception as exc:  # pragma: no cover
        _LOG.warning("Could not probe duration: %s", exc)
        return None


def extract_audio(video_path: Path, cache_dir: Path) -> Path:
    """Extract a 16 kHz mono PCM WAV for ASR (cached)."""
    _require("ffmpeg")
    cache_dir.mkdir(parents=True, exist_ok=True)
    audio_path = cache_dir / f"{video_path.stem}.16k.wav"
    if audio_path.exists() and audio_path.stat().st_size > 0:
        _LOG.info("Using cached audio: %s", audio_path)
        return audio_path
    _LOG.info("Extracting 16 kHz mono audio ...")
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(video_path),
            "-vn", "-ac", "1", "-ar", "16000",
            "-acodec", "pcm_s16le", str(audio_path),
        ],
        capture_output=True, text=True, check=True,
    )
    return audio_path


def ingest(source: str, input_dir: Path, cache_dir: Path) -> MediaInputs:
    video_path = resolve_input(source, input_dir)
    _LOG.info("Video: %s", video_path)
    audio_path = extract_audio(video_path, cache_dir)
    duration = probe_duration(video_path)
    return MediaInputs(video_path, audio_path, duration)
