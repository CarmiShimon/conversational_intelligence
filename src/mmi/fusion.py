"""Part C (1/2) -- Fusion.

Merge the speaker-attributed transcript (Part A) and the visual context
(Part B) onto a single, time-ordered timeline. This shared timeline is what we
serialise into the LLM prompt, so speech and on-screen text stay aligned in
time and can be cross-referenced as grounding evidence.
"""

from __future__ import annotations

from typing import List

from .schemas import EventType, TimelineEvent, Transcript, VisualContext
from .utils import get_logger

_LOG = get_logger("mmi.fusion")


def build_timeline(
    transcript: Transcript, visual: VisualContext
) -> List[TimelineEvent]:
    events: List[TimelineEvent] = []

    for seg in transcript.segments:
        events.append(
            TimelineEvent(
                t_start=seg.start,
                t_end=seg.end,
                type=EventType.SPEECH,
                speaker=seg.speaker,
                text=seg.text,
            )
        )

    for scene in visual.scenes:
        text = scene.ocr_text
        if not text:
            continue  # skip scenes with no readable on-screen text
        events.append(
            TimelineEvent(
                t_start=scene.start,
                t_end=scene.end,
                type=EventType.VISUAL,
                text=text,
                scene_id=scene.scene_id,
            )
        )

    events.sort(key=lambda e: (e.t_start, e.type.value))
    _LOG.info(
        "Timeline: %d events (%d speech, %d visual).",
        len(events),
        sum(1 for e in events if e.type is EventType.SPEECH),
        sum(1 for e in events if e.type is EventType.VISUAL),
    )
    return events


def _fmt_ts(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def render_timeline(events: List[TimelineEvent]) -> str:
    """Human/LLM-readable transcript of the fused timeline."""
    lines: List[str] = []
    for e in events:
        ts = _fmt_ts(e.t_start)
        if e.type is EventType.SPEECH:
            lines.append(f"[{ts}] ({e.speaker}) {e.text}")
        else:
            snippet = " | ".join(e.text.splitlines())
            lines.append(f"[{ts}] <ON-SCREEN scene#{e.scene_id}> {snippet}")
    return "\n".join(lines)
