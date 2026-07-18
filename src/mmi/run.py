"""Pipeline entry point.

    python -m mmi.run --input data/input/clip.mp4 --output outputs/

Orchestrates the cascade: ingest -> speech (A) -> vision (B) -> fusion +
intelligence (C). Each stage's artifact is cached to disk so re-runs skip
completed work. Stages can be toggled off for partial runs / debugging.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from .config import Config
from .fusion import build_timeline
from .schemas import (
    MeetingIntelligence,
    PipelineMetadata,
    PipelineOutput,
    Transcript,
    VisualContext,
)
from .utils import (
    STAGE_TIMINGS,
    configure_utf8_stdio,
    get_logger,
    load_dotenv_if_available,
    read_json,
    reset_stage_timings,
    stage_timer,
    write_json,
)

_LOG = get_logger("mmi.run")


def run_pipeline(
    source: str,
    cfg: Config,
    skip_speech: bool = False,
    skip_vision: bool = False,
    skip_llm: bool = False,
    force: bool = False,
) -> PipelineOutput:
    cfg.ensure_dirs()
    reset_stage_timings()
    cache = Path(cfg.paths.cache_dir)

    # --- Stage 0: ingest --------------------------------------------------- #
    from .ingest import ingest

    with stage_timer("ingest"):
        media = ingest(source, Path(cfg.paths.input_dir), cache)

    # --- Stage A: speech --------------------------------------------------- #
    transcript = _run_speech(media, cfg, cache, skip_speech, force)

    # --- Stage B: vision --------------------------------------------------- #
    visual = _run_vision(media, cfg, cache, skip_vision, force)

    # --- Stage C: fusion + intelligence ------------------------------------ #
    intelligence: Optional[MeetingIntelligence] = None
    llm_error: Optional[str] = None
    events = build_timeline(transcript, visual)
    write_json([e.model_dump(mode="json") for e in events], cache / "timeline.json")

    if not skip_llm:
        from .intelligence import generate_intelligence

        try:
            intelligence = generate_intelligence(
                events,
                language=transcript.language,
                participants=transcript.speakers,
                cfg=cfg.intelligence,
                api_key=cfg.openai_api_key,
                visual=visual,
            )
        except Exception as exc:
            # Don't lose the (expensive) transcript + visual work if the LLM
            # stage fails (e.g. missing quota / network). Emit a partial
            # artifact with intelligence=null and record the reason.
            _LOG.error("Intelligence stage failed; writing partial output: %s", exc)
            llm_error = str(exc)

    # --- Assemble final artifact ------------------------------------------- #
    output = PipelineOutput(
        metadata=PipelineMetadata(
            source=str(media.video_path),
            duration_sec=media.duration_sec,
            language=transcript.language,
            models={
                "asr": cfg.speech.whisper_model,
                "ocr": cfg.vision.ocr_engine,
                "llm": cfg.intelligence.model if not skip_llm else None,
            },
            llm_error=llm_error,
            stage_seconds={k: round(v, 2) for k, v in STAGE_TIMINGS.items()},
        ),
        transcript=transcript,
        visual=visual,
        intelligence=intelligence,
    )
    out_path = Path(cfg.paths.output_dir) / "result.json"
    write_json(output.model_dump(mode="json"), out_path)
    _LOG.info("Wrote final output -> %s", out_path)
    return output


def _run_speech(media, cfg, cache: Path, skip: bool, force: bool) -> Transcript:
    cache_file = cache / "transcript.json"
    if skip:
        cached = read_json(cache_file)
        if cached:
            _LOG.info("Speech skipped; loaded cached transcript.")
            return Transcript.model_validate(cached)
        _LOG.warning("Speech skipped but no cache; using empty transcript.")
        return Transcript(language="unknown", segments=[])
    if not force and (cached := read_json(cache_file)):
        _LOG.info("Using cached transcript.")
        return Transcript.model_validate(cached)

    from .speech import transcribe

    transcript = transcribe(media.audio_path, cfg.speech, cfg.hf_token)
    write_json(transcript.model_dump(mode="json"), cache_file)
    return transcript


def _run_vision(media, cfg, cache: Path, skip: bool, force: bool) -> VisualContext:
    cache_file = cache / "visual.json"
    if skip:
        cached = read_json(cache_file)
        if cached:
            _LOG.info("Vision skipped; loaded cached visual context.")
            return VisualContext.model_validate(cached)
        _LOG.warning("Vision skipped but no cache; using empty visual context.")
        return VisualContext(scenes=[])
    if not force and (cached := read_json(cache_file)):
        _LOG.info("Using cached visual context.")
        return VisualContext.model_validate(cached)

    from .vision import extract_visual_context

    visual = extract_visual_context(
        media.video_path, cfg.vision, Path(cfg.paths.keyframe_dir)
    )
    write_json(visual.model_dump(mode="json"), cache_file)
    return visual


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mmi", description="Multimodal Meeting Intelligence Pipeline"
    )
    p.add_argument(
        "--input", "-i", required=True,
        help="Path to a local video file or a Google Drive share link.",
    )
    p.add_argument("--output", "-o", default=None, help="Output directory.")
    p.add_argument("--config", "-c", default="config.yaml", help="Config YAML path.")
    p.add_argument(
        "--language", "-l", default=None,
        help="Force a language code (e.g. 'en'). Default: auto-detect.",
    )
    p.add_argument("--skip-speech", action="store_true")
    p.add_argument("--skip-vision", action="store_true")
    p.add_argument("--skip-llm", action="store_true", help="Skip the LLM stage.")
    p.add_argument(
        "--force", action="store_true", help="Ignore cached stage artifacts."
    )
    return p


def main(argv: Optional[list] = None) -> int:
    configure_utf8_stdio()
    load_dotenv_if_available()
    args = build_arg_parser().parse_args(argv)
    cfg = Config.load(args.config)
    if args.output:
        cfg.paths.output_dir = Path(args.output)
        cfg.paths.cache_dir = Path(args.output) / "cache"
        cfg.paths.keyframe_dir = Path(args.output) / "keyframes"
    if args.language:
        cfg.speech.language = args.language

    try:
        run_pipeline(
            args.input,
            cfg,
            skip_speech=args.skip_speech,
            skip_vision=args.skip_vision,
            skip_llm=args.skip_llm,
            force=args.force,
        )
    except Exception as exc:
        _LOG.error("Pipeline failed: %s", exc, exc_info=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
