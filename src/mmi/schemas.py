"""Data contracts shared between pipeline stages.

Every stage of the cascade consumes and produces one of these Pydantic models.
They are the *explicit contract* between components, so any stage can be swapped
for another implementation as long as it honours the same schema.

The models are grouped by stage:

* Part A (Speech)  -> ``Word``, ``TranscriptSegment``, ``Transcript``
* Part B (Vision)  -> ``OcrLine``, ``SceneSegment``, ``VisualContext``
* Part C (Fusion)  -> ``TimelineEvent``, ``Evidence``, ``IntelligenceItem``,
                      ``MeetingIntelligence``
* Final artifact   -> ``PipelineOutput``
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# Part A -- Speech: transcription & speaker attribution
# --------------------------------------------------------------------------- #
class Word(BaseModel):
    """A single aligned word with timing and confidence."""

    start: Optional[float] = Field(None, description="Word start time (s).")
    end: Optional[float] = Field(None, description="Word end time (s).")
    word: str = Field(..., description="The word text.")
    score: Optional[float] = Field(None, description="Alignment/ASR confidence.")
    speaker: Optional[str] = Field(None, description="Diarized speaker label.")


class TranscriptSegment(BaseModel):
    """A contiguous utterance attributed to a single speaker."""

    start: float = Field(..., description="Segment start time (s).")
    end: float = Field(..., description="Segment end time (s).")
    text: str = Field(..., description="Transcribed text for the segment.")
    speaker: str = Field("UNKNOWN", description="Speaker label, e.g. SPEAKER_00.")
    words: List[Word] = Field(default_factory=list)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


class Transcript(BaseModel):
    """Full speaker-attributed transcript (output of Part A)."""

    language: str = Field(..., description="ISO code of the detected language.")
    segments: List[TranscriptSegment] = Field(default_factory=list)

    @property
    def speakers(self) -> List[str]:
        return sorted({s.speaker for s in self.segments})


# --------------------------------------------------------------------------- #
# Part B -- Computer Vision: visual context
# --------------------------------------------------------------------------- #
class OcrLine(BaseModel):
    """One line of on-screen text detected by OCR."""

    text: str
    confidence: float = 0.0
    bbox: Optional[List[float]] = Field(
        None, description="[x0, y0, x1, y1] pixel bounding box."
    )


class SceneSegment(BaseModel):
    """A visually stable span of video (e.g. a single slide)."""

    scene_id: int
    start: float = Field(..., description="Scene start time (s).")
    end: float = Field(..., description="Scene end time (s).")
    keyframe_path: Optional[str] = Field(
        None, description="Path to the representative keyframe image."
    )
    keyframe_time: Optional[float] = Field(
        None, description="Timestamp of the representative keyframe (s)."
    )
    ocr_lines: List[OcrLine] = Field(default_factory=list)

    @property
    def ocr_text(self) -> str:
        return "\n".join(line.text for line in self.ocr_lines).strip()


class VisualContext(BaseModel):
    """All extracted visual context (output of Part B)."""

    scenes: List[SceneSegment] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Part C -- Fusion & conversation intelligence
# --------------------------------------------------------------------------- #
class EventType(str, Enum):
    SPEECH = "speech"
    VISUAL = "visual"


class TimelineEvent(BaseModel):
    """A single event on the shared A+B timeline (output of fusion)."""

    t_start: float
    t_end: float
    type: EventType
    speaker: Optional[str] = None
    text: str = ""
    scene_id: Optional[int] = None


class Evidence(BaseModel):
    """A grounding reference back into the source signals."""

    timestamp: Optional[float] = Field(None, description="Approx. time (s).")
    speaker: Optional[str] = None
    scene_id: Optional[int] = Field(None, description="Visual scene reference.")
    quote: Optional[str] = Field(None, description="Supporting quote/snippet.")


class IntelligenceItem(BaseModel):
    """A grounded intelligence element (topic / action item / decision)."""

    text: str
    evidence: List[Evidence] = Field(default_factory=list)


class ActionItem(IntelligenceItem):
    owner: Optional[str] = Field(None, description="Who is responsible, if stated.")
    due: Optional[str] = Field(None, description="Deadline if mentioned.")


class MeetingIntelligence(BaseModel):
    """Structured LLM output (output of Part C)."""

    language: str
    summary: str
    participants: List[str] = Field(default_factory=list)
    topics: List[IntelligenceItem] = Field(default_factory=list)
    action_items: List[ActionItem] = Field(default_factory=list)
    decisions: List[IntelligenceItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Final pipeline artifact
# --------------------------------------------------------------------------- #
class PipelineMetadata(BaseModel):
    source: str
    duration_sec: Optional[float] = None
    language: Optional[str] = None
    generated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    models: dict = Field(default_factory=dict)
    llm_error: Optional[str] = Field(
        None, description="Reason the intelligence stage was skipped, if any."
    )
    stage_seconds: dict = Field(
        default_factory=dict,
        description="Wall-clock seconds spent per stage (latency/resource notes).",
    )


class PipelineOutput(BaseModel):
    """The single structured artifact the pipeline emits."""

    metadata: PipelineMetadata
    transcript: Transcript
    visual: VisualContext
    intelligence: Optional[MeetingIntelligence] = None
