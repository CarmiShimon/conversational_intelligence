"""Render the pipeline architecture diagram as a PNG image (no external
diagram tools such as mermaid-cli/graphviz required).

Mirrors docs/diagram.mmd, so if that file changes this should be updated too.

Run:  python scripts/render_diagram.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "docs" / "diagram.png"

# (label, x, y, width, height, facecolor)
BOXES = {
    "in": ("Meeting video\n(audio + on-screen content)", 0.5, 9.4, 3.4, 0.7, "#e8e8e8"),
    "ingest": ("Ingest\nffmpeg -> WAV + video", 0.5, 8.1, 3.4, 0.7, "#dfe7f5"),
    "asr": ("A: ASR\nfaster-whisper + lang ID", 0.2, 6.6, 3.0, 0.7, "#dff0d8"),
    "align": ("A: word alignment\n(wav2vec2)", 0.2, 5.5, 3.0, 0.7, "#dff0d8"),
    "diar": ("A: diarization\n(pyannote 3.1)", 0.2, 4.4, 3.0, 0.7, "#dff0d8"),
    "scene": ("B: scene/slide-cut detection\n(PySceneDetect)", 4.0, 6.6, 3.2, 0.7, "#fdebd0"),
    "key": ("B: representative keyframe\nper scene", 4.0, 5.5, 3.2, 0.7, "#fdebd0"),
    "ocr": ("B: OCR on-screen text\n(EasyOCR)", 4.0, 4.4, 3.2, 0.7, "#fdebd0"),
    "fuse": ("C: fuse Transcript + VisualContext\ninto one time-ordered timeline", 0.9, 3.0, 6.2, 0.8, "#f2e2f5"),
    "llm": ("C: OpenAI structured call (gpt-4o)\n+ keyframe images; map-reduce if long", 0.9, 1.8, 6.2, 0.8, "#f2e2f5"),
    "out": ("result.json (PipelineOutput)\nsummary / topics / action items + evidence", 0.9, 0.6, 6.2, 0.7, "#e8e8e8"),
    "eval": ("D: Evaluation\nWER / speaker acc / OCR recall / LLM rubric", 7.9, 0.6, 3.4, 0.7, "#eaeaea"),
}

ARROWS = [
    ("in", "ingest"),
    ("ingest", "asr"),
    ("asr", "align"),
    ("align", "diar"),
    ("ingest", "scene"),
    ("scene", "key"),
    ("key", "ocr"),
    ("diar", "fuse"),
    ("ocr", "fuse"),
    ("fuse", "llm"),
    ("llm", "out"),
    ("out", "eval"),
]


def center(name: str):
    _, x, y, w, h, _ = BOXES[name]
    return x + w / 2, y + h / 2


def anchor(name: str, other: str):
    """Pick a sane edge point on box `name` facing `other`'s center."""
    _, x, y, w, h, _ = BOXES[name]
    ox, oy = center(other)
    cx, cy = x + w / 2, y + h / 2
    if abs(oy - cy) >= abs(ox - cx):
        return (cx, y + h) if oy > cy else (cx, y)
    return (x + w, cy) if ox > cx else (x, cy)


def main() -> int:
    fig, ax = plt.subplots(figsize=(11, 8.2))
    ax.set_xlim(0, 11.5)
    ax.set_ylim(0, 10.3)
    ax.axis("off")
    ax.set_title(
        "Multimodal Meeting Intelligence Pipeline -- Architecture",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )

    # Group backdrops for Part A / B / C so the cascade reads visually.
    ax.add_patch(FancyBboxPatch((0.05, 4.2), 3.3, 3.3, boxstyle="round,pad=0.02",
                                 linewidth=1, edgecolor="#5b8c3e", facecolor="none"))
    ax.text(0.15, 7.65, "Part A -- Speech", fontsize=9, color="#5b8c3e", fontweight="bold")
    ax.add_patch(FancyBboxPatch((3.85, 4.2), 3.5, 3.3, boxstyle="round,pad=0.02",
                                 linewidth=1, edgecolor="#c9821a", facecolor="none"))
    ax.text(3.95, 7.65, "Part B -- Vision", fontsize=9, color="#c9821a", fontweight="bold")
    ax.add_patch(FancyBboxPatch((0.75, 1.55), 6.55, 2.45, boxstyle="round,pad=0.02",
                                 linewidth=1, edgecolor="#8e44ad", facecolor="none"))
    ax.text(0.85, 4.15, "Part C -- Fusion & Intelligence", fontsize=9, color="#8e44ad", fontweight="bold")

    for name, (label, x, y, w, h, color) in BOXES.items():
        box = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=1.1, edgecolor="#444", facecolor=color,
        )
        ax.add_patch(box)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=8.3, wrap=True)

    for src, dst in ARROWS:
        p0 = anchor(src, dst)
        p1 = anchor(dst, src)
        arrow = FancyArrowPatch(
            p0, p1, arrowstyle="-|>", mutation_scale=12,
            linewidth=1.1, color="#333", shrinkA=2, shrinkB=2,
        )
        ax.add_patch(arrow)

    fig.tight_layout()
    fig.savefig(DEST, dpi=170)
    print(f"Wrote {DEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
