#!/usr/bin/env python3
"""`edge_files.py <snapshot dir> <out dir>`: the Voxtral bundle's edge files the checkpoint does not ship
(docs/queue/094): `tokenizer.json`, converted from the checkpoint's `tekken.json` by transformers'
`convert_tekken_tokenizer` (the same ids as mistral-common's, checked below on the control tokens and
`lang:en`), and `mel_filters.f32`, the Whisper log-mel filterbank `[128 bins][201 freqs]` f32 little-endian
(transformers stores it `[201][128]`; the edge and the node read the transposed form). `vocab.bin` is the
builder's, from `tokenizer.json`, as for every model."""
import os, sys
import numpy as np
from transformers import WhisperFeatureExtractor
from transformers.integrations.mistral import convert_tekken_tokenizer

snap, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
tok = convert_tekken_tokenizer(os.path.join(snap, "tekken.json"))
tok.save_pretrained(out)
for f in os.listdir(out):
    if f not in ("tokenizer.json",) and f.startswith(("special_tokens", "tokenizer_config", "chat_template")):
        os.remove(os.path.join(out, f))  # the manifest lists tokenizer.json alone; the edge needs no template for transcription
ids = {t: tok.convert_tokens_to_ids(t) for t in ["<s>", "</s>", "[INST]", "[/INST]", "[AUDIO]", "[BEGIN_AUDIO]", "[TRANSCRIBE]"]}
want = {"<s>": 1, "</s>": 2, "[INST]": 3, "[/INST]": 4, "[AUDIO]": 24, "[BEGIN_AUDIO]": 25, "[TRANSCRIBE]": 34}
assert ids == want, ids
lang = tok.encode("lang:en", add_special_tokens=False)
assert lang == [9909, 1058, 1262], lang
assert len(tok) == 131072, len(tok)
fe = WhisperFeatureExtractor(feature_size=128, sampling_rate=16000, n_fft=400, hop_length=160)
mel = np.ascontiguousarray(np.asarray(fe.mel_filters, dtype=np.float32).T)  # [128][201]
assert mel.shape == (128, 201), mel.shape
mel.tofile(os.path.join(out, "mel_filters.f32"))
print(f"{out}: tokenizer.json ({os.path.getsize(os.path.join(out, 'tokenizer.json'))} bytes, {len(tok)} ids), mel_filters.f32 {mel.shape}")
