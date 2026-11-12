# -*- coding: utf-8 -*-
"""Check FastQwenTTS against reference forward and measure speed (.venv_tts python)."""
import sys
import time
import wave
from pathlib import Path

import numpy as np
import torch
from qwen_tts import Qwen3TTSModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fast_tts import FastQwenTTS  # noqa: E402

tts = Qwen3TTSModel.from_pretrained(str(ROOT / "models" / "qwen3-tts"), device_map="cuda:0",
                                    dtype=torch.bfloat16, attn_implementation="sdpa")
spk = "aiden"
fast = FastQwenTTS(tts)

# 1) numeric check: my talker prefill vs HF talker.model
with torch.inference_mode():
    kw = fast._prepare("Проверка совпадения.", spk, "russian")
    emb = kw["inputs_embeds"]
    ref = fast.talker.model(inputs_embeds=emb, attention_mask=kw["attention_mask"], use_cache=False).last_hidden_state
    mine = fast._talker(emb.to(fast.dtype), fast.kpos[:emb.shape[1]])
    print("talker prefill max diff", (ref - mine).abs().max().item(), "ref scale", ref.abs().max().item())
    # code predictor check on random input
    x = torch.randn(1, 2, fast.tc.hidden_size, device=fast.dev, dtype=fast.dtype)
    refc = fast.cp.model(inputs_embeds=fast.cp.small_to_mtp_projection(x), use_cache=False).last_hidden_state
    minec = fast._cp(fast.cp.small_to_mtp_projection(x), fast.cpos[:2], fast.cp_masks[0])
    print("predictor max diff", (refc - minec).abs().max().item(), "ref scale", refc.abs().max().item())

texts = ["Привет, это проверка голоса.",
         "Ну слушай, я вообще не понимаю, зачем ты это делаешь, но ладно, давай попробуем ещё раз."]
list(fast.stream(texts[0], spk, "russian"))   # capture + warmup
for i, t in enumerate(texts):
    torch.cuda.synchronize()
    t0 = time.time()
    first = None
    chunks = []
    for c in fast.stream(t, spk, "russian"):
        if first is None:
            first = time.time() - t0
        chunks.append(c)
    dt = time.time() - t0
    audio = np.concatenate(chunks)
    dur = len(audio) / 24000
    print(f"first audio {first * 1000:.0f} ms, total {dt:.2f}s for {dur:.2f}s audio, RTF={dt / dur:.2f}")
    with wave.open(str(ROOT / "cache" / "tmp" / f"fast_{i}.wav"), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())
print(f"VRAM peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GB, reserved {torch.cuda.memory_reserved() / 2**30:.2f} GB")
