# -*- coding: utf-8 -*-
"""Scripted multi-turn Russian chat against the server on :8765; prints TTFT, tok/s and replies.

Usage (.venv python): python tools/bench_dialog.py <label>
"""
import json
import sys
import time

import httpx

URL = "http://127.0.0.1:8765"
label = sys.argv[1] if len(sys.argv) > 1 else "model"
sys.stdout.reconfigure(encoding="utf-8")
for _ in range(240):
    try:
        if httpx.get(URL + "/health", timeout=2).status_code == 200:
            break
    except Exception:
        pass
    time.sleep(1)

cfg = json.load(open("D:/LiteDuplex/config.json", encoding="utf-8"))
SYS = cfg["system_prompt"]
turns = [
    "Привет! Как дела?",
    "Да так, с работы пришёл, устал как собака.",
    "Я программист, весь день баги чинил. Начальник опять орал.",
    "А ты бы что сделал на моём месте?",
    "Слушай, а как меня зовут? Я Миша, кстати.",
    "Ну и как меня зовут?",
    "Расскажи какой-нибудь анекдот.",
    "Не смешно вообще.",
    "Что думаешь про кошек?",
    "А ты вообще кто такой?",
    "Ты тупой, что ли?",
    "Ладно, извини. Посоветуй фильм на вечер.",
    "Я уже смотрел. Давай что-нибудь другое.",
    "Окей, спасибо. Чем займёшься вечером?",
    "Ну всё, пока!",
]
msgs = [{"role": "system", "content": SYS}]
ttfts, rates = [], []
for q in turns:
    msgs.append({"role": "user", "content": q})
    t0 = time.time()
    first, text, n = None, "", 0
    with httpx.stream("POST", URL + "/v1/chat/completions", timeout=120, json={
            "model": "x", "stream": True, "max_tokens": cfg["max_new_tokens"],
            "temperature": cfg["temperature"], "top_p": cfg["top_p"], "messages": msgs}) as r:
        for line in r.iter_lines():
            if not line.startswith("data: ") or line.endswith("[DONE]"):
                continue
            ch = json.loads(line[6:])["choices"]
            d = ch[0]["delta"].get("content") if ch else None
            if d:
                first = first or time.time() - t0
                text += d
                n += 1
    dt = time.time() - t0
    msgs.append({"role": "assistant", "content": text.strip()})
    ttfts.append(first or dt)
    if first and n > 3:
        rates.append(n / max(dt - first, 1e-3))
    print(f"[{(first or 0) * 1000:4.0f} ms] Я: {q}\n          ИИ: {text.strip()}", flush=True)
ttfts.sort()
print(f"\n== {label}: TTFT median {ttfts[len(ttfts) // 2] * 1000:.0f} ms, "
      f"~{sum(rates) / max(len(rates), 1):.0f} chunks/s (≈tok/s)")
