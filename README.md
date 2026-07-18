# Multimodal Meeting Intelligence Pipeline

An end-to-end pipeline that ingests a recorded meeting video and produces a
single, structured, **speaker-attributed, visually-grounded, LLM-enriched**
output. It is a cascade of independent, swappable stages with explicit data
contracts (Pydantic models in [`src/mmi/schemas.py`](src/mmi/schemas.py)):

```
video ─▶ ingest ─▶ [A] speech ─┐
                                ├─▶ [C] fusion ─▶ [C] intelligence ─▶ result.json
        └──────▶ [B] vision ────┘
```

| Stage | Module | What it does | Model / API |
|-------|--------|--------------|-------------|
| A. Speech | [`speech.py`](src/mmi/speech.py) | Segment → ASR (auto language ID) → word alignment → diarization → merge | WhisperX (faster-whisper) + pyannote |
| B. Vision | [`vision.py`](src/mmi/vision.py) | Scene/slide-change detection → representative keyframe → OCR | PySceneDetect + EasyOCR |
| C. Fusion | [`fusion.py`](src/mmi/fusion.py) | Merge A + B onto one timeline | — |
| C. Intelligence | [`intelligence.py`](src/mmi/intelligence.py) | Structured summary / topics / action items / decisions, grounded in evidence | OpenAI (gpt-4o, vision-capable) |
| D. Evaluation | [`evaluate.py`](src/mmi/evaluate.py) | WER, speaker accuracy, OCR recall, LLM rubric | jiwer + optional LLM judge |

## Requirements

- Python 3.10–3.12 (tested on 3.10)
- `ffmpeg` + `ffprobe` on `PATH`
- NVIDIA GPU + CUDA recommended (falls back to CPU)
- An OpenAI API key; a free Hugging Face token for diarization

## Setup

```powershell
# 1. Create and activate a virtual environment
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install PyTorch with the CUDA wheel that matches your driver FIRST
pip install torch==2.3.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121

# 3. Install the pipeline + remaining dependencies
pip install -r requirements.txt
pip install -e .            # exposes the `mmi` package and console script

# 4. Configure secrets
copy .env.example .env      # then edit .env and add your keys
```

Accept the pyannote model terms once (needed for diarization):
<https://hf.co/pyannote/speaker-diarization-3.1> and
<https://hf.co/pyannote/segmentation-3.0>.

## Run

```powershell
# From a local file:
python -m mmi.run --input data\input\clip.mp4 --output outputs

# Or straight from the provided Google Drive link (auto-downloads via gdown):
python -m mmi.run --input "https://drive.google.com/file/d/1OrMgA5-RXQKL4Db728WAMQXhYb-Dzdw7/view"
```

The final artifact lands at **`outputs/result.json`**. Intermediate stage
outputs are cached under `outputs/cache/` and keyframes under
`outputs/keyframes/`, so re-runs skip completed stages (use `--force` to redo).

### Useful flags

| Flag | Effect |
|------|--------|
| `--language en` | Force ASR language (skips auto-detect). Recommended for known-language clips. |
| `--skip-llm` | Run A + B only (no OpenAI call) |
| `--skip-speech` / `--skip-vision` | Reuse cached stage / debug one branch |
| `--force` | Ignore caches and recompute |
| `--config path.yaml` | Use a different config |

### On Windows: force UTF-8

Some third-party progress bars print box-drawing characters that legacy
Windows codepages (e.g. cp1255) can't encode. The CLI reconfigures stdout to
UTF-8 automatically, but for belt-and-suspenders you can also set:

```powershell
$env:PYTHONUTF8="1"; $env:PYTHONIOENCODING="utf-8"
```

## Evaluate

Fill in [`data/reference/reference.json`](data/reference/reference.json) with a
small hand-labelled reference (transcript text, a few speaker segments, and
expected on-screen keywords), then:

```powershell
python -m mmi.evaluate --result outputs\result.json --reference data\reference\reference.json
# add --judge to also score the LLM output with an LLM-as-judge
```

## Configuration

All non-secret settings live in [`config.yaml`](config.yaml) (model sizes,
scene sensitivity, OCR engine, LLM model, chunking threshold, paths). Secrets
come only from the environment / `.env`.

## Output shape

See [`src/mmi/schemas.py`](src/mmi/schemas.py). Top level:

```jsonc
{
  "metadata":     { "source", "duration_sec", "language", "models", "generated_at" },
  "transcript":   { "language", "segments": [ { "start","end","text","speaker","words" } ] },
  "visual":       { "scenes":   [ { "scene_id","start","end","keyframe_path","ocr_lines" } ] },
  "intelligence": { "summary","participants","topics","action_items","decisions" }
}
```

## Notes

- Nothing is hard-coded to the sample clip: language, speaker count, and slide
  timings are all detected at runtime.
- The design, trade-offs, evaluation, and production considerations are written
  up in [`docs/report.md`](docs/report.md); the architecture diagram is in
  [`docs/diagram.mmd`](docs/diagram.mmd).
