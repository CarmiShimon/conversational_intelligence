"""Part B -- Computer Vision: visual context extraction.

Strategy (efficiency first -- we do NOT process every frame):

    detect scene/slide changes (PySceneDetect content detector)
      -> one representative keyframe per scene (scene midpoint)
      -> OCR the keyframe (on-screen text)
      -> VisualContext

The output is a small set of timestamped, de-duplicated visual segments that a
downstream LLM can cheaply consume.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

from .config import VisionConfig
from .schemas import OcrLine, SceneSegment, VisualContext
from .utils import get_logger, stage_timer

_LOG = get_logger("mmi.vision")


def extract_visual_context(
    video_path: Path, cfg: VisionConfig, keyframe_dir: Path
) -> VisualContext:
    scenes = _detect_scenes(video_path, cfg)
    _LOG.info("Detected %d scene(s).", len(scenes))
    if len(scenes) > cfg.max_scenes:
        _LOG.warning("Capping scenes at %d (had %d).", cfg.max_scenes, len(scenes))
        scenes = scenes[: cfg.max_scenes]

    reader = _build_ocr(cfg)
    keyframe_dir.mkdir(parents=True, exist_ok=True)

    import cv2  # type: ignore

    segments: List[SceneSegment] = []
    # Reuse a single VideoCapture across all keyframes instead of
    # opening/closing the video file once per scene (cheaper for long
    # inputs with many scenes).
    cap = cv2.VideoCapture(str(video_path))
    try:
        with stage_timer("vision.keyframe_ocr"):
            for idx, (start, end) in enumerate(scenes):
                mid = start + (end - start) / 2.0
                kf_path = keyframe_dir / f"scene_{idx:03d}.jpg"
                frame = _grab_frame(cap, mid, kf_path)
                ocr_lines = _ocr_frame(reader, frame, cfg) if frame is not None else []
                segments.append(
                    SceneSegment(
                        scene_id=idx,
                        start=round(start, 3),
                        end=round(end, 3),
                        keyframe_path=str(kf_path) if frame is not None else None,
                        keyframe_time=round(mid, 3),
                        ocr_lines=ocr_lines,
                    )
                )
    finally:
        cap.release()
    return VisualContext(scenes=segments)


def _detect_scenes(video_path: Path, cfg: VisionConfig) -> List[Tuple[float, float]]:
    from scenedetect import ContentDetector, SceneManager, open_video  # type: ignore

    with stage_timer("vision.scene_detect"):
        video = open_video(str(video_path))
        fps = video.frame_rate or 30.0
        min_len_frames = max(1, int(cfg.min_scene_len_sec * fps))
        manager = SceneManager()
        manager.add_detector(
            ContentDetector(threshold=cfg.scene_threshold, min_scene_len=min_len_frames)
        )
        manager.detect_scenes(video, show_progress=False)
        scene_list = manager.get_scene_list()

    if not scene_list:
        # No cuts detected -> treat whole video as one scene.
        duration = _video_duration(video_path)
        return [(0.0, duration)]
    return [(s.get_seconds(), e.get_seconds()) for s, e in scene_list]


def _video_duration(video_path: Path) -> float:
    import cv2  # type: ignore

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()
    return float(frames / fps) if fps else 0.0


def _grab_frame(cap, t_sec: float, out_path: Path):
    """Seek an already-open capture to ``t_sec`` and save that frame as a JPEG."""
    import cv2  # type: ignore

    cap.set(cv2.CAP_PROP_POS_MSEC, t_sec * 1000.0)
    ok, frame = cap.read()
    if not ok or frame is None:
        _LOG.warning("Failed to read frame at %.2fs", t_sec)
        return None
    cv2.imwrite(str(out_path), frame)
    return frame


# --------------------------------------------------------------------------- #
# OCR backends
# --------------------------------------------------------------------------- #
def _build_ocr(cfg: VisionConfig):
    engine = cfg.ocr_engine.lower()
    if engine == "easyocr":
        import easyocr  # type: ignore

        try:
            import torch  # type: ignore

            gpu = torch.cuda.is_available()
        except Exception:
            gpu = False
        _LOG.info("Loading EasyOCR (gpu=%s) ...", gpu)
        return ("easyocr", easyocr.Reader(list(cfg.ocr_languages), gpu=gpu))
    if engine == "tesseract":
        return ("tesseract", None)
    if engine == "paddleocr":
        from paddleocr import PaddleOCR  # type: ignore

        return ("paddleocr", PaddleOCR(use_angle_cls=True, lang=cfg.ocr_languages[0]))
    raise ValueError(f"Unknown OCR engine: {cfg.ocr_engine}")


def _ocr_frame(reader, frame, cfg: VisionConfig) -> List[OcrLine]:
    engine, obj = reader
    try:
        if engine == "easyocr":
            return _ocr_easyocr(obj, frame, cfg)
        if engine == "tesseract":
            return _ocr_tesseract(frame, cfg)
        if engine == "paddleocr":
            return _ocr_paddle(obj, frame, cfg)
    except Exception as exc:  # pragma: no cover
        _LOG.warning("OCR failed on a frame: %s", exc)
    return []


def _ocr_easyocr(reader, frame, cfg: VisionConfig) -> List[OcrLine]:
    lines: List[OcrLine] = []
    for bbox, text, conf in reader.readtext(frame):
        if conf < cfg.ocr_min_confidence or not text.strip():
            continue
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        lines.append(
            OcrLine(
                text=text.strip(),
                confidence=float(conf),
                bbox=[float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))],
            )
        )
    return lines


def _ocr_tesseract(frame, cfg: VisionConfig) -> List[OcrLine]:
    import pytesseract  # type: ignore
    from PIL import Image  # type: ignore
    import cv2  # type: ignore

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    text = pytesseract.image_to_string(Image.fromarray(rgb))
    return [OcrLine(text=ln.strip(), confidence=1.0) for ln in text.splitlines() if ln.strip()]


def _ocr_paddle(ocr, frame, cfg: VisionConfig) -> List[OcrLine]:
    result = ocr.ocr(frame, cls=True)
    lines: List[OcrLine] = []
    for block in result or []:
        for box, (text, conf) in block:
            if conf < cfg.ocr_min_confidence or not text.strip():
                continue
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            lines.append(
                OcrLine(
                    text=text.strip(),
                    confidence=float(conf),
                    bbox=[float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))],
                )
            )
    return lines
