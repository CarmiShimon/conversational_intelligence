"""Part A -- Speech: transcription & speaker attribution.

Pipeline within this stage (the segmentation -> ASR -> diarization -> merge
contract that the brief asks us to design):

    load audio
      -> WhisperX VAD segmentation + faster-whisper ASR (auto language ID)
      -> word-level alignment (wav2vec2)
      -> pyannote diarization
      -> assign words/segments to speakers
      -> Transcript

Because the reference GPU has only ~4 GB of VRAM, each heavyweight model is
loaded, used, and unloaded (freeing CUDA memory) before the next one runs.
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import List, Optional

from .config import SpeechConfig
from .schemas import Transcript, TranscriptSegment, Word
from .utils import get_logger, stage_timer

_LOG = get_logger("mmi.speech")


def _release_gpu_memory() -> None:
    """Run GC and clear the CUDA cache.

    Callers must ``del`` their own reference to the model/tensor *before*
    calling this: a helper that receives the object as an argument only
    holds a local binding, so ``del`` inside the helper does not drop the
    caller's reference and the memory is never actually freed.
    """
    gc.collect()
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _resolve_device(requested: str) -> str:
    try:
        import torch  # type: ignore

        if requested == "cuda" and not torch.cuda.is_available():
            _LOG.warning("CUDA requested but unavailable; falling back to CPU.")
            return "cpu"
    except Exception:
        return "cpu"
    return requested


def transcribe(audio_path: Path, cfg: SpeechConfig, hf_token: Optional[str]) -> Transcript:
    """Run the full speech stage and return a speaker-attributed Transcript."""
    import whisperx  # type: ignore

    device = _resolve_device(cfg.device)
    compute_type = cfg.compute_type if device == "cuda" else "int8"

    with stage_timer("speech.load_audio"):
        audio = whisperx.load_audio(str(audio_path))

    # --- ASR + language ID ------------------------------------------------- #
    with stage_timer("speech.asr"):
        asr_model = _load_asr(whisperx, cfg, device, compute_type)
        language = cfg.language or _detect_language(asr_model, audio, cfg.batch_size)
        result = asr_model.transcribe(
            audio, batch_size=cfg.batch_size, language=language
        )
        language = result["language"]
        _LOG.info("Using language: %s", language)
        del asr_model
        _release_gpu_memory()

    # --- Word-level alignment --------------------------------------------- #
    with stage_timer("speech.align"):
        try:
            align_model, metadata = whisperx.load_align_model(
                language_code=language, device=device
            )
            result = whisperx.align(
                result["segments"], align_model, metadata, audio, device,
                return_char_alignments=False,
            )
            del align_model
            _release_gpu_memory()
        except Exception as exc:
            _LOG.warning("Alignment unavailable for '%s': %s", language, exc)

    # --- Diarization + speaker assignment --------------------------------- #
    diarized = False
    if hf_token:
        with stage_timer("speech.diarize"):
            try:
                diarize_segments = _diarize(whisperx, audio, cfg, device, hf_token)
                result = whisperx.assign_word_speakers(diarize_segments, result)
                diarized = True
                del diarize_segments
                _release_gpu_memory()
            except Exception as exc:
                _LOG.warning("Diarization failed: %s", exc)
    else:
        _LOG.warning("No HuggingFace token; skipping diarization (single speaker).")

    del audio
    _release_gpu_memory()
    return _to_transcript(result["segments"], language, diarized)


def _load_asr(whisperx, cfg: SpeechConfig, device: str, compute_type: str):
    try:
        return whisperx.load_model(cfg.whisper_model, device, compute_type=compute_type)
    except Exception as exc:
        _LOG.warning("load_model(%s) failed (%s); retrying with int8.", compute_type, exc)
        return whisperx.load_model(cfg.whisper_model, device, compute_type="int8")


def _detect_language(asr_model, audio, batch_size: int, sr: int = 16000):
    """Detect language by majority vote over a few mid-meeting windows.

    Detecting on only the first 30 s is fragile: intros often contain music,
    applause, or roll-call noise that Whisper mis-attributes (e.g. English
    meetings detected as Maori). Sampling at 25/50/75 % of the audio is far
    more robust and stays language-agnostic (no hard-coded language).
    """
    n = len(audio)
    if n < sr * 5:  # very short clip -> let transcribe() auto-detect
        return None
    votes: dict[str, int] = {}
    for frac in (0.25, 0.5, 0.75):
        start = int(n * frac)
        window = audio[start : min(n, start + 30 * sr)]
        if len(window) < sr:
            continue
        try:
            res = asr_model.transcribe(window, batch_size=batch_size)
            lang = res.get("language")
            if lang:
                votes[lang] = votes.get(lang, 0) + 1
        except Exception as exc:  # pragma: no cover
            _LOG.warning("Language sample failed: %s", exc)
    if not votes:
        return None
    best = max(votes, key=votes.get)
    _LOG.info("Language votes: %s -> %s", votes, best)
    return best


def _diarize(whisperx, audio, cfg: SpeechConfig, device: str, hf_token: str):
    # DiarizationPipeline moved between whisperx versions; support both.
    try:
        from whisperx.diarize import DiarizationPipeline  # type: ignore
    except Exception:
        DiarizationPipeline = whisperx.DiarizationPipeline  # type: ignore
    pipeline = DiarizationPipeline(use_auth_token=hf_token, device=device)
    return pipeline(
        audio, min_speakers=cfg.min_speakers, max_speakers=cfg.max_speakers
    )


def _to_transcript(segments: List[dict], language: str, diarized: bool) -> Transcript:
    out: List[TranscriptSegment] = []
    for seg in segments:
        words = [
            Word(
                start=w.get("start"),
                end=w.get("end"),
                word=w.get("word", ""),
                score=w.get("score"),
                speaker=w.get("speaker"),
            )
            for w in seg.get("words", [])
        ]
        out.append(
            TranscriptSegment(
                start=float(seg.get("start", 0.0)),
                end=float(seg.get("end", 0.0)),
                text=str(seg.get("text", "")).strip(),
                speaker=seg.get("speaker", "SPEAKER_00" if not diarized else "UNKNOWN"),
                words=words,
            )
        )
    return Transcript(language=language, segments=out)
