"""Part C (2/2) -- Conversation intelligence.

Turns the fused timeline into a structured ``MeetingIntelligence`` object via an
OpenAI model, with:

* an enforced output schema (OpenAI structured outputs / Pydantic parsing),
* grounding rules (every item must cite timestamp / speaker / scene evidence
  drawn only from the provided timeline -- no outside knowledge),
* long-input handling via map-reduce (chunk -> notes -> single structured pass),
* vision-capable scene understanding: representative keyframe images are
  attached to the final structured call (when the configured model supports
  it) so slide content / on-screen layout can be read directly instead of
  relying solely on OCR text, which can miss real slide content entirely.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import List, Optional

from .config import IntelligenceConfig
from .fusion import render_timeline
from .schemas import MeetingIntelligence, SceneSegment, TimelineEvent, VisualContext
from .utils import get_logger, stage_timer

_LOG = get_logger("mmi.intelligence")

_SYSTEM = (
    "You are a meeting-intelligence analyst. You are given a time-ordered "
    "transcript of a meeting fused with on-screen (OCR) text. Each line is "
    "prefixed with a [HH:MM:SS] timestamp and either a (SPEAKER) tag or an "
    "<ON-SCREEN scene#N> tag. You may also be given representative keyframe "
    "images after the transcript, each labeled 'scene#N keyframe'. Use those "
    "images to describe on-screen content (slide text, layout, people/gallery "
    "view) more accurately than the OCR text, and prefer what you see in the "
    "image if it conflicts with the OCR text.\n"
    "Rules:\n"
    "1. Use ONLY information present in the provided content (text and/or "
    "images). Do not invent facts, names, numbers, or decisions.\n"
    "2. Ground every topic, action item, and decision with evidence: cite the "
    "timestamp, and the speaker or scene# it came from, plus a short quote.\n"
    "3. If something is uncertain or absent, omit it rather than guessing.\n"
    "4. Keep the summary concise and factual."
)

_MAP_SYSTEM = (
    "You are summarising ONE chunk of a longer meeting transcript (with "
    "on-screen text). Produce concise bullet notes capturing: key discussion "
    "points, any decisions, and any action items -- each with its [HH:MM:SS] "
    "timestamp and speaker/scene reference. Use only the provided content."
)


def generate_intelligence(
    events: List[TimelineEvent],
    language: str,
    participants: List[str],
    cfg: IntelligenceConfig,
    api_key: Optional[str],
    visual: Optional[VisualContext] = None,
) -> MeetingIntelligence:
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Export it or run with --skip-llm."
        )
    from openai import OpenAI  # type: ignore

    client = OpenAI(api_key=api_key, timeout=cfg.request_timeout)
    timeline_text = render_timeline(events)

    with stage_timer("intelligence.llm"):
        if len(timeline_text) <= cfg.max_window_chars:
            content = timeline_text
        else:
            _LOG.info(
                "Timeline too long (%d chars); using map-reduce.", len(timeline_text)
            )
            content = _map_reduce_notes(client, timeline_text, cfg)
        return _structured_call(client, content, language, participants, cfg, visual)


def _chunk(text: str, max_chars: int) -> List[str]:
    lines = text.splitlines()
    chunks, buf, size = [], [], 0
    for ln in lines:
        if size + len(ln) > max_chars and buf:
            chunks.append("\n".join(buf))
            buf, size = [], 0
        buf.append(ln)
        size += len(ln) + 1
    if buf:
        chunks.append("\n".join(buf))
    return chunks


def _map_reduce_notes(client, timeline_text: str, cfg: IntelligenceConfig) -> str:
    """Map step: summarise each chunk into notes; return concatenated notes."""
    chunks = _chunk(timeline_text, cfg.max_window_chars)
    notes: List[str] = []
    for i, chunk in enumerate(chunks):
        _LOG.info("Summarising chunk %d/%d ...", i + 1, len(chunks))
        resp = client.chat.completions.create(
            model=cfg.model,
            temperature=cfg.temperature,
            messages=[
                {"role": "system", "content": _MAP_SYSTEM},
                {"role": "user", "content": chunk},
            ],
        )
        notes.append(resp.choices[0].message.content or "")
    return "\n\n".join(f"# Notes (part {i + 1})\n{n}" for i, n in enumerate(notes))


def _encode_image_data_uri(path: Path) -> Optional[str]:
    """Base64-encode a keyframe JPEG as a data URI for the OpenAI vision API."""
    try:
        data = Path(path).read_bytes()
    except OSError as exc:
        _LOG.warning("Could not read keyframe %s: %s", path, exc)
        return None
    return f"data:image/jpeg;base64,{base64.b64encode(data).decode('ascii')}"


def _select_keyframes(
    visual: Optional[VisualContext], cfg: IntelligenceConfig
) -> List[SceneSegment]:
    """Pick which scene keyframes to attach, capped and evenly sampled.

    Sending every scene's image on a long meeting would be slow and costly, so
    if there are more scenes than ``max_keyframe_images`` we take an evenly
    spaced subset (rather than just the first N) to keep coverage across the
    whole meeting instead of only its start.
    """
    if not visual or not cfg.include_keyframes:
        return []
    scenes = [s for s in visual.scenes if s.keyframe_path]
    cap = cfg.max_keyframe_images
    if cap <= 0:
        return []
    if len(scenes) <= cap:
        return scenes
    step = len(scenes) / cap
    return [scenes[int(i * step)] for i in range(cap)]


def _structured_call(
    client,
    content: str,
    language: str,
    participants: List[str],
    cfg: IntelligenceConfig,
    visual: Optional[VisualContext] = None,
) -> MeetingIntelligence:
    user_text = (
        f"Detected language: {language}\n"
        f"Known speaker labels: {', '.join(participants) or 'unknown'}\n\n"
        f"MEETING CONTENT:\n{content}"
    )

    keyframes = _select_keyframes(visual, cfg)
    if not keyframes:
        user_content = user_text
    else:
        _LOG.info("Attaching %d keyframe image(s) to the LLM call.", len(keyframes))
        parts: List[dict] = [{"type": "text", "text": user_text}]
        for scene in keyframes:
            data_uri = _encode_image_data_uri(Path(scene.keyframe_path))
            if data_uri is None:
                continue
            parts.append({"type": "text", "text": f"scene#{scene.scene_id} keyframe:"})
            parts.append(
                {"type": "image_url", "image_url": {"url": data_uri, "detail": "low"}}
            )
        user_content = parts

    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": user_content},
    ]

    # Preferred path: OpenAI structured outputs with Pydantic parsing.
    try:
        completion = client.beta.chat.completions.parse(
            model=cfg.model,
            temperature=cfg.temperature,
            messages=messages,
            response_format=MeetingIntelligence,
        )
        parsed = completion.choices[0].message.parsed
        if parsed is not None:
            if not parsed.language:
                parsed.language = language
            return parsed
    except Exception as exc:
        _LOG.warning("Structured parse unavailable (%s); using JSON mode.", exc)

    # Fallback: JSON mode + manual validation.
    completion = client.chat.completions.create(
        model=cfg.model,
        temperature=cfg.temperature,
        response_format={"type": "json_object"},
        messages=messages
        + [
            {
                "role": "system",
                "content": (
                    "Return a JSON object with keys: language (str), summary "
                    "(str), participants (str[]), topics, action_items, "
                    "decisions. topics/decisions are objects with 'text' and "
                    "'evidence' (list of {timestamp, speaker, scene_id, quote}). "
                    "action_items additionally allow 'owner' and 'due'."
                ),
            }
        ],
    )
    raw = completion.choices[0].message.content or "{}"
    data = MeetingIntelligence.model_validate_json(raw)
    if not data.language:
        data.language = language
    return data
