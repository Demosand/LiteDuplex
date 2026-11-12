# -*- coding: utf-8 -*-
"""Does Qwen3-TTS 0.6B CustomVoice react to emotion instructions? (.venv_tts)

The qwen_tts wrapper drops `instruct` for 0.6B models; we bypass that and measure
pitch / loudness / speed differences between instructions.
"""
import sys
import wave
from pathlib import Path

import numpy as np
import torch
from qwen_tts import Qwen3TTSModel

ROOT = Path(__file__).resolve().parents[1]
tts = Qwen3TTSModel.from_pretrained(str(ROOT / "models" / "qwen3-tts"), device_map="cuda:0",
                                    dtype=torch.bfloat16, attn_implementation="sdpa")
tts.model.tts_model_size = "1b7"   # обходим `if tts_model_size in "0b6": instruct = None`
text = "Ты представляешь, мы вчера встретили в лесу настоящего медведя!"
instr = {"none": "", "happy": "Говори очень радостно, возбуждённо и громко.",
         "sad": "Говори грустно, тихо и медленно, с печалью в голосе.",
         "angry": "Говори раздражённо и зло."}


def f0(a, sr):
    """Median and spread of pitch by autocorrelation over voiced 40 ms frames."""
    out = []
    w = int(sr * 0.04)
    for i in range(0, len(a) - w, w // 2):
        x = a[i:i + w] - a[i:i + w].mean()
        if np.sqrt((x ** 2).mean()) < 0.02:
            continue
        c = np.correlate(x, x, "full")[w - 1:]
        lo, hi = int(sr / 400), int(sr / 70)
        k = lo + int(np.argmax(c[lo:hi]))
        if c[k] > 0.3 * c[0]:
            out.append(sr / k)
    return (np.median(out), np.std(out)) if out else (0, 0)


torch.manual_seed(0)
for name, ins in instr.items():
    durs, f0s, sds, rms = [], [], [], []
    for _ in range(3):
        wavs, sr = tts.generate_custom_voice(text=text, speaker="aiden", language="russian", instruct=ins)
        a = np.asarray(wavs[0], dtype=np.float32)
        m, s = f0(a, sr)
        durs.append(len(a) / sr); f0s.append(m); sds.append(s); rms.append(np.sqrt((a ** 2).mean()))
    with wave.open(str(ROOT / "cache" / "tmp" / f"instr_{name}.wav"), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes((np.clip(a, -1, 1) * 32767).astype(np.int16).tobytes())
    print(f"{name:6}: dur {np.mean(durs):.2f}s  F0 {np.mean(f0s):5.0f} Hz ±{np.mean(sds):4.0f}  "
          f"rms {np.mean(rms):.3f}", flush=True)
