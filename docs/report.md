# Multimodal Meeting Intelligence Pipeline — Report

## 1. Architecture & key design decisions

The system is a **linear cascade of independent, swappable stages** connected by
explicit Pydantic data contracts ([`src/mmi/schemas.py`](../src/mmi/schemas.py)).
Each stage reads a typed object and emits a typed object, so any component can be
replaced (a different ASR, a hosted OCR, another LLM) without touching its
neighbours — the same shape a production speech/AI stack takes.

See the system block diagram in a separate document:
[`docs/diagram.png`](diagram.png) (rendered) /
[`docs/diagram.mmd`](diagram.mmd) (Mermaid source) — it shows every stage,
the data contract flowing between them, and which model/API sits where.

**Data contracts (the important part).** The brief explicitly asks us to design
the contract between stages, so this is where most of the design effort went:

- `Transcript` = `language` + `TranscriptSegment[]`, where each segment carries
  `start/end/text/speaker` and optional word-level timings. This is the merge
  point of the segmentation → ASR → diarization sub-pipeline.
- `VisualContext` = `SceneSegment[]`, each with a time span, a representative
  keyframe path, and OCR lines (text + confidence + bbox).
- `TimelineEvent[]` is the *fused* representation: speech and visual events
  interleaved on one clock, which is exactly what the LLM consumes.
- `MeetingIntelligence` is the enforced LLM output: `summary`, `topics`,
  `action_items`, `decisions`, each item carrying `Evidence`
  (timestamp / speaker / scene / quote) for grounding.

**Orchestration.** [`run.py`](../src/mmi/run.py) is the single entry point. Each
stage's artifact is cached to `outputs/cache/` and reloaded on re-run, so an
expensive ASR pass isn't repeated while iterating on prompts. Stages can be
toggled (`--skip-llm`, `--skip-vision`, …) for cheap partial runs. Secrets are
read only from the environment.

**What we rejected.**

- *WhisperX vs. rolling our own faster-whisper + pyannote glue.* We kept
  WhisperX because its VAD-batched inference and word-level alignment give
  materially better speaker attribution at segment boundaries, and it already
  wires pyannote in — but we still expose the merge as our own `Transcript`
  contract rather than leaking WhisperX's dict format downstream.
- *Processing every frame for CV.* Rejected on cost; scene-change detection +
  one keyframe per scene captures slide content at a tiny fraction of the frames.
- *Free-form LLM output.* Rejected in favour of a strict schema (OpenAI
  structured outputs / Pydantic parse) so the result is machine-consumable and
  every claim is forced to carry evidence.
- *One giant prompt for long meetings.* Rejected; we chunk and map-reduce above a
  configurable character budget.

## 2. Model / API choices and trade-offs

| Stage | Choice | Why | Trade-off |
|-------|--------|-----|-----------|
| ASR | WhisperX (`small` default) | Fast CTranslate2 backend, auto language ID, word alignment | `small` trades WER for fitting a 4 GB GPU; `medium`/`large-v3` configurable |
| Diarization | pyannote 3.1 | State-of-the-art open diarizer, integrates with WhisperX | Needs a HF token; sensitive to overlapping speech |
| Scene detection | PySceneDetect content detector | Cheap, robust to slide/scene cuts | Threshold tuning; gradual fades can be missed |
| OCR | EasyOCR (GPU) | Simple install, decent on slide text, GPU-accelerated | Weaker on dense/stylised text than PaddleOCR (swappable via config) |
| LLM | OpenAI `gpt-4o` (vision-capable) | Native multimodal input (reads keyframes directly), strong structured-output support | Hosted dependency, higher cost/latency than `gpt-4o-mini` (configurable) |

**Cost/latency/accuracy stance.** We default to the *cheap, fast* end of the
ASR/vision axes (small Whisper, one keyframe/scene) because the reference
machine is a 4 GB laptop GPU, but spend more on the LLM (`gpt-4o` rather than
`gpt-4o-mini`) because grounded reasoning over fused text+image evidence is
where quality matters most and tokens are comparatively cheap next to a bad
summary. Every knob is in [`config.yaml`](../config.yaml) to retune the
balance. GPU memory is the binding constraint on Part A, so the speech stage
loads/uses/frees each model (ASR → align → diarize) sequentially rather than
holding them all resident.

## 3. Evaluation

Metrics are implemented in [`evaluate.py`](../src/mmi/evaluate.py) and run against
a small **bootstrap** reference ([`data/reference/reference.json`](../data/reference/reference.json))
— not an independent hand-labelled reference; see the caveat below the results table:

- **ASR — Word Error Rate** (jiwer) on normalised text, with S/D/I breakdown.
  When the reference is a snippet (has `speaker_segments`), WER is scoped to
  that time window so it isn't distorted by comparing 60 s of reference against
  the full ~26 min hypothesis.
- **Speaker attribution accuracy** — a diarization proxy: both reference and
  hypothesis are rasterised onto a 0.1 s grid, and we brute-force the optimal
  hypothesis→reference label mapping (speaker counts are small) and report
  time-weighted agreement. This sidesteps the arbitrary-label problem.
- **OCR keyword recall** — fraction of expected on-screen terms recovered across
  all keyframes; also reports how many scenes yielded any text.
- **LLM output** — a rubric (faithfulness, topic relevance, real action items,
  grounding, no hallucinated entities) plus an automatic *grounded-fraction*
  (share of items carrying evidence) and an optional LLM-as-judge (`--judge`).

**Results on the provided clip** (26 min council meeting, `--language en`,
`whisper-small` on a 4 GB T1200, LLM stage: `gpt-4o` with 16 keyframe images,
full run — see [`outputs/result.json`](../outputs/result.json) /
[`outputs/eval.json`](../outputs/eval.json)):

<table>
<colgroup><col style="width:30%"><col style="width:14%"><col style="width:56%"></colgroup>
<thead><tr><th>Metric</th><th>Value</th><th>Notes</th></tr></thead>
<tbody>
<tr><td>ASR WER (first 60 s vs bootstrap reference)</td><td><strong>6.67 %</strong></td><td>8 insertions / 0 subs / 0 dels. <strong>Harness validation, not an accuracy claim</strong> &mdash; see caveat below.</td></tr>
<tr><td>Speaker attribution accuracy (first 60 s)</td><td><strong>100 %</strong></td><td>2 speakers in window; label map <code>SPEAKER_01&rarr;chair, SPEAKER_03&rarr;andrew</code>; 6 speakers globally. <strong>Harness validation, not an accuracy claim</strong> &mdash; see caveat below.</td></tr>
<tr><td>OCR keyword recall</td><td><strong>100 %</strong> (1/1)</td><td>Only meaningful term is &quot;zoom&quot;; see below</td></tr>
<tr><td>Scenes with OCR text</td><td><strong>15 / 16</strong></td><td>&nbsp;</td></tr>
<tr><td>Distinct diarized speakers</td><td><strong>6</strong></td><td>Plausible for a small council meeting</td></tr>
<tr><td>Detected language</td><td><strong>en</strong></td><td>Auto-detect chose <code>mi</code> (Maori) on the first 30 s intro; overridden by <code>--language en</code>. Robust auto-detect (majority vote at 25/50/75 %) is implemented in <code>speech.py</code> for future runs.</td></tr>
<tr><td>LLM output (topics / action items / decisions)</td><td><strong>7 / 1 / 1</strong></td><td>All grounded (see next row)</td></tr>
<tr><td>LLM grounded-evidence fraction</td><td><strong>100 %</strong> (9/9)</td><td>Every topic/action-item/decision cites a timestamp + speaker or scene; see the known gap noted below</td></tr>
</tbody>
</table>

**Caveat: no independent manual labels exist for this clip.** The reference
transcript *and* speaker segments in `data/reference/reference.json` are a
**bootstrap** — both derived from the pipeline's own first-minute output
rather than an independent human listen (see the file's own `_comment`). That
makes **both** the ASR WER and the speaker-attribution accuracy circular: they
measure whether the harness computes the metric correctly, not whether the
transcript or diarization is actually right. The one exception is **OCR
keyword recall** — the single expected keyword ("zoom") was noted by eye from
the video, independent of the pipeline's own OCR output, so that number is a
genuine (if trivial) check. See "Evaluation methodology — future work" below
for how we'd replace the bootstrap with a real, non-circular reference.

**Failure analysis (where it breaks and why).**

- *Language ID on noisy intros* → Whisper mis-detected English as Maori on the
  first 30 s (which contains roll-call cross-talk and short utterances),
  producing a garbage transcript of "maitha, maitha, maitha…" for the first
  segment on that run. Fixed by (a) adding a `--language` CLI override and
  (b) a majority-vote language detector that samples windows at 25/50/75 % of
  the audio; both live in `speech.py`.
- *Zoom meetings have no slides* → OCR faithfully returns just the word "zoom"
  from the Zoom UI overlay across 15 of 16 scenes. The visual channel adds
  essentially no downstream signal on this particular clip. For a
  presentation/slide meeting the same OCR pipeline would yield rich content —
  this is a property of the input, not a pipeline bug.
- *Diarization initially failed* due to a `speechbrain 1.1 + pyannote 3.3.2`
  lazy-import bug (`speechbrain.integrations.k2_fsa` cannot be resolved).
  Pinning `speechbrain==1.0.2` fixed it and produced 6 plausible speakers.
- *LLM stage is a single external dependency* — during development it failed
  once with `429 insufficient_quota`. The pipeline catches that, writes a
  valid `result.json` with `intelligence=null` and `metadata.llm_error`
  populated instead of losing the (expensive) transcript/visual work, and a
  re-run retries it (idempotent). The committed `result.json` is a clean run
  where the LLM stage succeeded.
- *Windows console encoding* — EasyOCR / gdown progress bars use box-drawing
  characters that cp1255 can't encode; forced UTF-8 for stdout in the CLI.
- *Diarization on overlap / short backchannels* → still a general risk; the
  biggest driver of speaker-accuracy loss on longer meetings.
- *LLM grounding* → the schema forces evidence, but the model can still cite a
  plausible-but-wrong timestamp; the grounded-fraction metric flags *whether*
  evidence exists, not that it's correct — a known gap.

**Evaluation methodology — future work (no manual labels currently exist).**
We have no independently human-labelled reference for this clip, so ASR WER
and speaker-attribution accuracy above should be read as harness validation,
not accuracy claims (see the caveat above). Three concrete ways to get a real,
non-circular signal, roughly in increasing cost:

- *Silver-standard reference model.* Run `whisper-large-v3` (or the OpenAI
  Whisper API) once on the same clip and score `small`'s output against it
  instead of against itself. Not truly independent (same model family), but a
  standard, fully-automatic proxy for "does the cheap model degrade relative
  to the expensive one" — no human time required.
- *A few minutes of real human listening.* The current reference only spans
  the first 60 s. Listening to that one window ourselves (or running it
  through an unrelated service, e.g. YouTube auto-captions) and correcting it
  by ear turns the bootstrap into a genuine, independent reference at minimal
  cost — no need to hand-label the full 26 minutes.
- *Multimodal cross-validation for diarization.* This is a Zoom recording,
  and Zoom highlights the active speaker in its UI. The vision stage already
  OCRs keyframes; detecting that highlight box would give an automatically-
  extracted, independent "who was speaking when" signal from the *visual*
  channel to score diarization against, instead of comparing the audio
  pipeline against a label bootstrapped from the same audio pipeline.

## 4. Production thinking

- **Scaling.** Stages are independent and stateless given their inputs, so each
  maps to its own horizontally-scaled worker/queue. A 2-hour recording is
  handled by (a) VAD-chunked ASR already, (b) scene-bounded CV that is
  independent of duration, and (c) LLM **map-reduce**: summarise windows in
  parallel, then reduce — implemented in [`intelligence.py`](../src/mmi/intelligence.py)
  above a configurable budget.
- **Latency & memory.** The bottleneck is Part A on GPU; on 4 GB we serialise
  model loads and prefer `int8`/`small`. In production ASR and diarization run on
  right-sized GPUs; CV and the LLM call are CPU/network-bound and cheap.
- **Cost.** Dominated by LLM tokens; the timeline is compact (deduplicated OCR,
  one line per utterance) and chunked, keeping cost roughly linear in meeting
  length. Model tier is a config switch.
- **Monitoring & regression.** Cache the per-stage artifacts (already done) and
  run the eval harness on a labelled golden set in CI to catch drift; track WER,
  DER/speaker-accuracy, OCR recall, and LLM grounded-fraction over time. Log
  detected language, speaker count, and scene count per run as cheap health
  signals.
- **Robustness.** Missing HF token → diarization is skipped (single speaker)
  rather than crashing; CUDA absent → CPU fallback; OCR/alignment failures are
  caught per-item so one bad frame/segment can't fail the run.

**With one more week:** add active-speaker detection (face + lip-motion) to fuse
*who is on screen* with diarization; a real DER via `pyannote.metrics`; PaddleOCR
+ slide-title heuristics for cleaner visual grounding; a retrieval step so the
LLM cites exact source spans; and a golden-set CI job.

## 5. Use of AI assistants

AI coding assistants were used to scaffold boilerplate (argparse wiring, Pydantic
models, ffmpeg/OpenCV/EasyOCR call patterns) and to draft this report. The
**design decisions were made deliberately**: the stage decomposition and data
contracts, the choice to keep WhisperX but hide its format behind our own
`Transcript`, the GPU-memory-driven sequential model loading for a 4 GB card, the
grounding-via-schema approach, the map-reduce long-input strategy, and the
evaluation metric definitions (especially the label-mapping speaker-accuracy
proxy). Every one of these is defensible in follow-up discussion.
