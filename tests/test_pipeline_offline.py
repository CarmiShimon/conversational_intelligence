"""Offline smoke test -- exercises the data contracts, fusion, and evaluation
without any heavyweight ML dependency (no GPU / models / network needed).

Run:  python tests/test_pipeline_offline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mmi.evaluate import eval_ocr, eval_speaker, eval_wer  # noqa: E402
from mmi.fusion import build_timeline, render_timeline  # noqa: E402
from mmi.schemas import (  # noqa: E402
    OcrLine,
    PipelineMetadata,
    PipelineOutput,
    SceneSegment,
    Transcript,
    TranscriptSegment,
    VisualContext,
)


def _fake_output() -> PipelineOutput:
    transcript = Transcript(
        language="en",
        segments=[
            TranscriptSegment(start=0.0, end=3.0, text="Hello team welcome",
                              speaker="SPEAKER_00"),
            TranscriptSegment(start=3.0, end=6.0, text="Let us review the roadmap",
                              speaker="SPEAKER_01"),
        ],
    )
    visual = VisualContext(
        scenes=[
            SceneSegment(scene_id=0, start=0.0, end=6.0, keyframe_time=3.0,
                         ocr_lines=[OcrLine(text="Q3 Roadmap", confidence=0.9)]),
        ]
    )
    return PipelineOutput(
        metadata=PipelineMetadata(source="fake.mp4", duration_sec=6.0, language="en"),
        transcript=transcript,
        visual=visual,
    )


def test_fusion_orders_events():
    out = _fake_output()
    events = build_timeline(out.transcript, out.visual)
    assert len(events) == 3  # 2 speech + 1 visual
    assert [e.t_start for e in events] == sorted(e.t_start for e in events)
    rendered = render_timeline(events)
    assert "ON-SCREEN" in rendered and "SPEAKER_00" in rendered


def test_schema_roundtrip():
    out = _fake_output()
    dumped = out.model_dump(mode="json")
    again = PipelineOutput.model_validate(dumped)
    assert again.transcript.language == "en"
    assert again.visual.scenes[0].ocr_text == "Q3 Roadmap"


def test_evaluation_metrics():
    out = _fake_output()
    wer = eval_wer(out, "hello team welcome let us review the roadmap")
    assert wer["wer"] == 0.0

    spk = eval_speaker(
        out,
        [
            {"start": 0.0, "end": 3.0, "speaker": "A"},
            {"start": 3.0, "end": 6.0, "speaker": "B"},
        ],
    )
    assert spk["accuracy"] == 1.0

    ocr = eval_ocr(out, ["Q3 Roadmap", "Budget"])
    assert ocr["keyword_recall"] == 0.5


def main() -> int:
    tests = [test_fusion_orders_events, test_schema_roundtrip, test_evaluation_metrics]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nAll {len(tests)} offline tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
