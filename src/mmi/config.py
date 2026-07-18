"""Central configuration.

Values are layered: built-in defaults <- config.yaml <- environment variables.
Secrets (API keys / HF token) are read *only* from the environment, never from
the YAML file, so nothing sensitive is committed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - yaml is optional at runtime
    yaml = None


@dataclass
class SpeechConfig:
    # Default to a small model: the reference GPU (T1200) has only 4 GB VRAM.
    whisper_model: str = "small"
    # float16 on GPU, int8 fallback keeps memory low; overridable per machine.
    compute_type: str = "float16"
    device: str = "cuda"
    batch_size: int = 8
    language: Optional[str] = None  # None -> auto-detect
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None


@dataclass
class VisionConfig:
    scene_threshold: float = 27.0  # PySceneDetect content threshold
    min_scene_len_sec: float = 2.0
    ocr_engine: str = "easyocr"  # easyocr | paddleocr | tesseract
    ocr_languages: tuple = ("en",)
    ocr_min_confidence: float = 0.3
    max_scenes: int = 200  # safety cap for very long inputs


@dataclass
class IntelligenceConfig:
    model: str = "gpt-4o"
    temperature: float = 0.0
    max_window_chars: int = 24000  # chunking threshold for long meetings
    request_timeout: int = 120
    # Vision-capable scene understanding: attach representative keyframe
    # images to the final structured LLM call so it can "see" slides/gallery
    # layout directly, instead of relying solely on OCR text.
    include_keyframes: bool = True
    max_keyframe_images: int = 40  # cost/latency cap; evenly sampled if exceeded


@dataclass
class Paths:
    input_dir: Path = Path("data/input")
    output_dir: Path = Path("outputs")
    cache_dir: Path = Path("outputs/cache")
    keyframe_dir: Path = Path("outputs/keyframes")


@dataclass
class Config:
    speech: SpeechConfig = field(default_factory=SpeechConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    intelligence: IntelligenceConfig = field(default_factory=IntelligenceConfig)
    paths: Paths = field(default_factory=Paths)

    # --- secrets (env only) -------------------------------------------------
    @property
    def openai_api_key(self) -> Optional[str]:
        return os.getenv("OPENAI_API_KEY")

    @property
    def hf_token(self) -> Optional[str]:
        return os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")

    # --- loading ------------------------------------------------------------
    @classmethod
    def load(cls, path: Optional[str | os.PathLike] = None) -> "Config":
        cfg = cls()
        data: Dict[str, Any] = {}
        if path and Path(path).exists():
            if yaml is None:
                raise RuntimeError(
                    "pyyaml is required to read config.yaml but is not installed "
                    "(`pip install pyyaml`)."
                )
            with open(path, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
        _apply(cfg.speech, data.get("speech", {}))
        _apply(cfg.vision, data.get("vision", {}))
        _apply(cfg.intelligence, data.get("intelligence", {}))
        _apply(cfg.paths, {k: Path(v) for k, v in data.get("paths", {}).items()})
        return cfg

    def ensure_dirs(self) -> None:
        for p in (
            self.paths.output_dir,
            self.paths.cache_dir,
            self.paths.keyframe_dir,
        ):
            Path(p).mkdir(parents=True, exist_ok=True)

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Paths -> str for JSON friendliness
        d["paths"] = {k: str(v) for k, v in d["paths"].items()}
        return d


def _apply(obj: Any, overrides: Dict[str, Any]) -> None:
    for key, value in overrides.items():
        if hasattr(obj, key) and value is not None:
            setattr(obj, key, value)
