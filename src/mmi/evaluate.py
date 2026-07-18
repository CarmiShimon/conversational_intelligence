"""Part D -- Evaluation.

Computes quality metrics against a small hand-labelled reference:

* ASR:     Word Error Rate (jiwer) on normalised text.
* Speaker: time-weighted attribution accuracy with optimal label mapping
           (a lightweight diarization-accuracy proxy).
* OCR:     keyword recall -- fraction of expected on-screen terms recovered.
* LLM:     a rubric scaffold (optionally scored by an LLM judge).

Usage:
    python -m mmi.evaluate --result outputs/result.json \
        --reference data/reference/reference.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .schemas import PipelineOutput
from .utils import (
    configure_utf8_stdio,
    get_logger,
    load_dotenv_if_available,
    read_json,
    write_json,
)

_LOG = get_logger("mmi.evaluate")


def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# --------------------------------------------------------------------------- #
# ASR
# --------------------------------------------------------------------------- #
def eval_wer(
    output: PipelineOutput,
    ref_text: str,
    ref_window: Optional[Tuple[float, float]] = None,
) -> Dict:
    import jiwer  # type: ignore

    if ref_window is not None:
        w0, w1 = ref_window
        hyp_text = " ".join(
            s.text for s in output.transcript.segments if s.start < w1 and s.end > w0
        )
    else:
        hyp_text = " ".join(s.text for s in output.transcript.segments)
    hyp = _normalize(hyp_text)
    ref = _normalize(ref_text)
    if not ref:
        return {"wer": None, "note": "empty reference"}
    measures = jiwer.compute_measures(ref, hyp)
    return {
        "wer": round(measures["wer"], 4),
        "substitutions": measures["substitutions"],
        "deletions": measures["deletions"],
        "insertions": measures["insertions"],
        "ref_words": len(ref.split()),
        "hyp_words": len(hyp.split()),
        "ref_window": list(ref_window) if ref_window else None,
    }


# --------------------------------------------------------------------------- #
# Speaker attribution
# --------------------------------------------------------------------------- #
def eval_speaker(output: PipelineOutput, ref_segments: List[dict], step: float = 0.1) -> Dict:
    if not ref_segments:
        return {"accuracy": None, "note": "no reference speaker segments"}

    end = max(s["end"] for s in ref_segments)
    n = int(end / step) + 1

    ref_labels = _grid(ref_segments, n, step, key="speaker")
    hyp_segments = [
        {"start": s.start, "end": s.end, "speaker": s.speaker}
        for s in output.transcript.segments
    ]
    hyp_labels = _grid(hyp_segments, n, step, key="speaker")

    ref_set = sorted({l for l in ref_labels if l})
    hyp_set = sorted({l for l in hyp_labels if l})
    if not ref_set or not hyp_set:
        return {"accuracy": None, "note": "insufficient labels"}

    best_acc, best_map = 0.0, {}
    # Brute-force best mapping of hyp labels -> ref labels (small speaker counts).
    for perm in itertools.permutations(ref_set, min(len(hyp_set), len(ref_set))):
        mapping = dict(zip(hyp_set, perm))
        correct = sum(
            1
            for r, h in zip(ref_labels, hyp_labels)
            if r and h and mapping.get(h) == r
        )
        scored = sum(1 for r, h in zip(ref_labels, hyp_labels) if r and h)
        acc = correct / scored if scored else 0.0
        if acc > best_acc:
            best_acc, best_map = acc, mapping
    return {
        "accuracy": round(best_acc, 4),
        "label_mapping": best_map,
        "ref_speakers": ref_set,
        "hyp_speakers": hyp_set,
    }


def _grid(segments: List[dict], n: int, step: float, key: str) -> List[Optional[str]]:
    grid: List[Optional[str]] = [None] * n
    for seg in segments:
        i0 = int(seg["start"] / step)
        i1 = min(n, int(seg["end"] / step))
        for i in range(max(0, i0), i1):
            grid[i] = seg.get(key)
    return grid


# --------------------------------------------------------------------------- #
# OCR
# --------------------------------------------------------------------------- #
def eval_ocr(output: PipelineOutput, keywords: List[str]) -> Dict:
    if not keywords:
        return {"keyword_recall": None, "note": "no reference keywords"}
    corpus = _normalize(
        " ".join(sc.ocr_text for sc in output.visual.scenes)
    )
    found = [k for k in keywords if _normalize(k) in corpus]
    return {
        "keyword_recall": round(len(found) / len(keywords), 4),
        "found": found,
        "missing": [k for k in keywords if k not in found],
        "scenes_with_text": sum(1 for sc in output.visual.scenes if sc.ocr_text),
        "total_scenes": len(output.visual.scenes),
    }


# --------------------------------------------------------------------------- #
# LLM rubric (scaffold + optional LLM judge)
# --------------------------------------------------------------------------- #
_RUBRIC = [
    "Summary is faithful to the transcript (no invented facts).",
    "Key topics reflect what was actually discussed.",
    "Action items are real commitments, with owners where stated.",
    "Each item is grounded with a plausible timestamp / speaker / scene.",
    "Output is well-structured and free of hallucinated entities.",
]


def eval_llm(output: PipelineOutput, judge_model: Optional[str], api_key: Optional[str]) -> Dict:
    if output.intelligence is None:
        return {"note": "LLM stage was skipped"}
    result: Dict = {
        "rubric": _RUBRIC,
        "counts": {
            "topics": len(output.intelligence.topics),
            "action_items": len(output.intelligence.action_items),
            "decisions": len(output.intelligence.decisions),
        },
        "grounded_items": _grounded_fraction(output),
    }
    if judge_model and api_key:
        result["judge"] = _llm_judge(output, judge_model, api_key)
    return result


def _grounded_fraction(output: PipelineOutput) -> Dict:
    items = (
        output.intelligence.topics
        + output.intelligence.action_items
        + output.intelligence.decisions
    )
    if not items:
        return {"fraction": None}
    grounded = sum(1 for it in items if it.evidence)
    return {"fraction": round(grounded / len(items), 4), "total_items": len(items)}


def _llm_judge(output: PipelineOutput, model: str, api_key: str) -> Dict:
    from openai import OpenAI  # type: ignore

    client = OpenAI(api_key=api_key)
    prompt = (
        "Score the following meeting-intelligence JSON from 1-5 on each rubric "
        "item. Return JSON {scores:[{criterion,score,reason}], overall:int}.\n\n"
        "RUBRIC:\n- " + "\n- ".join(_RUBRIC) + "\n\nOUTPUT JSON:\n"
        f"{output.intelligence.model_dump_json(indent=2)}"
    )
    resp = client.chat.completions.create(
        model=model,
        temperature=0.0,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}],
    )
    return json.loads(resp.choices[0].message.content or "{}")


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def evaluate(result_path: Path, reference_path: Path, judge: bool = False) -> Dict:
    output = PipelineOutput.model_validate(read_json(result_path))
    reference = read_json(reference_path) or {}

    # If speaker_segments are provided, use their span as the ASR time window,
    # so WER is not distorted by comparing a snippet reference against a full
    # transcript.
    ref_segs = reference.get("speaker_segments", [])
    ref_window = None
    if ref_segs:
        starts = [s["start"] for s in ref_segs]
        ends = [s["end"] for s in ref_segs]
        if max(ends) > min(starts):
            ref_window = (min(starts), max(ends))

    report = {
        "asr": eval_wer(output, reference.get("transcript", ""), ref_window),
        "speaker": eval_speaker(output, ref_segs),
        "ocr": eval_ocr(output, reference.get("ocr_keywords", [])),
        "llm": eval_llm(
            output,
            judge_model="gpt-4o-mini" if judge else None,
            api_key=os.getenv("OPENAI_API_KEY"),
        ),
    }
    return report


def main(argv: Optional[list] = None) -> int:
    configure_utf8_stdio()
    load_dotenv_if_available()

    p = argparse.ArgumentParser(description="Evaluate pipeline output.")
    p.add_argument("--result", default="outputs/result.json")
    p.add_argument("--reference", default="data/reference/reference.json")
    p.add_argument("--out", default="outputs/eval.json")
    p.add_argument("--judge", action="store_true", help="Use an LLM judge for Part C.")
    args = p.parse_args(argv)

    report = evaluate(Path(args.result), Path(args.reference), judge=args.judge)
    write_json(report, args.out)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    _LOG.info("Wrote evaluation -> %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
