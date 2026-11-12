# -*- coding: utf-8 -*-
"""Baseline Qwen3-TTS speed (run with .venv_tts python)."""
import time
from pathlib import Path

import torch
from qwen_tts import Qwen3TTSModel

ROOT = Path(__file__).resolve().parents[1]
tts = Qwen3TTSModel.from_pretrained(str(ROOT / "models" / "qwen3-tts"), device_map="cuda:0",
                                    dtype=torch.bfloat16, attn_implementation="sdpa")
spk = tts.get_supported_speakers()[0]
print("speakers", tts.get_supported_speakers())
texts = ["Привет, это проверка голоса.",
         "Ну слушай, я вообще не понимаю, зачем ты это делаешь, но ладно, давай попробуем ещё раз."]
tts.generate_custom_voice(text=texts[0], speaker=spk, language="russian")
for t in texts:
    torch.cuda.synchronize(); t0 = time.time()
    wavs, sr = tts.generate_custom_voice(text=t, speaker=spk, language="russian")
    torch.cuda.synchronize(); dt = time.time() - t0
    dur = len(wavs[0]) / sr
    print(f"{dt:.2f}s for {dur:.2f}s audio, RTF={dt / dur:.2f}")
print(f"VRAM peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")
