#!/usr/bin/env python3
"""`windows.py <wav> <out dir>`: the encoder's windows for a clip as the API will send them (`windows.f32`,
`[N][128][3000]` f32, the processor's input_features) and the transcription prompt's ids (`prompt.json`), without
the model: the end-to-end gate's input for a clip the oracle has not run on."""
import json, os, sys
from transformers import VoxtralProcessor
wav, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
snap = os.environ.get("VOXTRAL_SNAPSHOT") or sys.argv[3]
proc = VoxtralProcessor.from_pretrained(snap)
req = proc.apply_transcription_request(language="en", audio=wav, model_id=snap, sampling_rate=16000, return_tensors="pt")
feats = req["input_features"]
feats.float().contiguous().numpy().tofile(os.path.join(out, "windows.f32"))
ids = req["input_ids"][0].tolist()
json.dump({"ids": ids, "windows": feats.shape[0], "audio": wav}, open(os.path.join(out, "prompt.json"), "w"))
print(f"{out}: {feats.shape[0]} windows, {len(ids)} prompt ids")
