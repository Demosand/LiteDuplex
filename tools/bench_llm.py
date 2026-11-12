# -*- coding: utf-8 -*-
"""Measure TTFT and tokens/s of an OpenAI-compatible server on :8765 (.venv python)."""
import json
import sys
import time

import httpx

URL = "http://127.0.0.1:8765"
for _ in range(180):
    try:
        if httpx.get(URL + "/health", timeout=2).status_code == 200:
            break
    except Exception:
        pass
    time.sleep(1)

SYS = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "D:/LiteDuplex/config.json", encoding="utf-8"))["system_prompt"]
qs = ["Привет, как дела?", "Расскажи, что ты думаешь про путешествия в горы, только подробно.",
      "Слушай, а ты вообще кто такой?"]
for q in qs * 2:
    t0 = time.time()
    first, text, n = None, "", 0
    with httpx.stream("POST", URL + "/v1/chat/completions", timeout=120, json={
            "model": "x", "stream": True, "max_tokens": 120, "temperature": 0.8, "top_p": 0.95,
            "messages": [{"role": "system", "content": SYS}, {"role": "user", "content": q}]}) as r:
        for line in r.iter_lines():
            if not line.startswith("data: ") or line.endswith("[DONE]"):
                continue
            d = json.loads(line[6:])["choices"][0]["delta"].get("content")
            if d:
                first = first or time.time() - t0
                text += d
                n += 1
    dt = time.time() - t0
    print(f"TTFT {first * 1000:.0f} ms, {n} chunks, {n / max(dt - first, 1e-3):.1f} tok/s | {text.strip()[:150]}")
