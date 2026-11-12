# -*- coding: utf-8 -*-
"""Chunk schedule vs first-audio latency and playback underruns, with LLM load in parallel (.venv_tts)."""
import json
import sys
import threading
import time
import urllib.request
from pathlib import Path

import torch
from qwen_tts import Qwen3TTSModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fast_tts import FastQwenTTS  # noqa: E402

tts = Qwen3TTSModel.from_pretrained(str(ROOT / "models" / "qwen3-tts"), device_map="cuda:0",
                                    dtype=torch.bfloat16, attn_implementation="sdpa")
fast = FastQwenTTS(tts)
spk, lang = "aiden", "russian"
text = "Причём стоял медведь прямо на тропинке, а вы на него смотрели и не знали, что делать дальше."
list(fast.stream(text, spk, lang))


def llm_load():
    body = json.dumps({"model": "x", "max_tokens": 200, "messages": [
        {"role": "user", "content": "Расскажи длинную историю про медведя."}]}).encode()
    req = urllib.request.Request("http://127.0.0.1:8765/v1/chat/completions", body,
                                 {"Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=120).read()


def run(first, chunk, load):
    th = None
    if load:
        th = threading.Thread(target=llm_load)
        th.start()
        time.sleep(0.15)
    t0 = time.time()
    first_t, play_end, gaps, audio = None, None, 0.0, 0.0
    for wav in fast.stream(text, spk, lang, first_frames=first, chunk_frames=chunk):
        now = time.time() - t0
        if first_t is None:
            first_t, play_end = now, now
        if now > play_end:        # плеер уже доиграл предыдущее — слышна пауза
            gaps += now - play_end
            play_end = now
        play_end += len(wav) / fast.sr
        audio += len(wav) / fast.sr
    if th:
        th.join()
    return first_t, gaps, (time.time() - t0) / audio


for load in (False, True):
    for first, chunk in [(4, 12), (2, 4), (3, 4), (2, 6), (3, 6)]:
        r = [run(first, chunk, load) for _ in range(3)]
        f = sum(x[0] for x in r) / 3
        g = max(x[1] for x in r)
        rtf = sum(x[2] for x in r) / 3
        print(f"llm_load={load!s:5} first={first} chunk={chunk:2}: first audio {f * 1000:4.0f} ms, "
              f"worst gaps {g * 1000:4.0f} ms, RTF {rtf:.2f}", flush=True)
