import json

d = json.load(open("outputs/result.json", encoding="utf-8"))
m, t, v = d["metadata"], d["transcript"], d["visual"]
print("language:", t["language"], "| duration_s:", round(m["duration_sec"]))
print("segments:", len(t["segments"]), "| scenes:", len(v["scenes"]))
spk = sorted(set(s["speaker"] for s in t["segments"]))
print("speakers:", spk)
print("llm_error:", m.get("llm_error"))
print("\n--- first 6 transcript segments ---")
for s in t["segments"][:6]:
    print(f"[{s['start']:6.1f}-{s['end']:6.1f}] {s['speaker']}: {s['text'][:90]}")
print("\n--- scenes with OCR text (up to 6) ---")
shown = 0
for sc in v["scenes"]:
    lines = sc.get("ocr_lines", [])
    if lines and shown < 6:
        text = " | ".join(l["text"] for l in lines)[:110]
        print(f"scene {sc['scene_id']} @{sc['start']:.0f}s: {text}")
        shown += 1
print(f"\nscenes with any OCR: {sum(1 for s in v['scenes'] if s.get('ocr_lines'))}/{len(v['scenes'])}")
