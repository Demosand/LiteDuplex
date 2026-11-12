# -*- coding: utf-8 -*-
"""Fake backend for UI work (.venv python): same WebSocket events as duplex.UIBridge, no models.

Loading statuses -> ready; after "start": user speaks (levels) -> bot says two phrases word-timed.
"""
import asyncio
import json
import math
import sys
import time

import websockets

clients = set()
state = {"ready": False, "active": False}


# --en: English phrases (README recordings, tools/record/)
PHRASES = (["Honestly, that's a great idea,", "let's try it right now and see what happens!"] if "--en" in sys.argv
           else ["Слушай, это вообще-то отличная идея,", "давай попробуем прямо сейчас и посмотрим, что выйдет!"])


async def send(obj):
    for ws in list(clients):
        try:
            await ws.send(json.dumps(obj, ensure_ascii=False))
        except Exception:
            clients.discard(ws)


async def conversation():
    while True:
        await asyncio.sleep(0.5)
        if not state["active"]:
            continue
        await send({"t": "user", "v": True})
        t0 = time.time()
        while time.time() - t0 < 2.6 and state["active"]:
            t = time.time() - t0
            await send({"t": "level", "v": round(abs(math.sin(t * 7)) * 0.5 + abs(math.sin(t * 2.3)) * 0.4, 3)})
            await asyncio.sleep(0.033)
        await send({"t": "level", "v": 0})
        await send({"t": "user", "v": False})
        await asyncio.sleep(0.35)
        if not state["active"]:
            continue
        await send({"t": "bot", "v": True})
        sid = 0
        for text in PHRASES:
            sid += 1
            dur = len(text) / 13
            await send({"t": "say", "id": sid, "words": text.split(), "at": int(time.time() * 1000), "dur": dur})
            await asyncio.sleep(dur)
        await send({"t": "bot", "v": False})
        await asyncio.sleep(3.0)


async def handler(ws):
    clients.add(ws)
    await ws.send(json.dumps({"t": "status", "text": "Загружаю голос…" if not state["ready"] else "Готово",
                              "ready": state["ready"], "active": state["active"]}, ensure_ascii=False))
    try:
        async for msg in ws:
            cmd = json.loads(msg).get("t")
            state["active"] = cmd == "start"
            await send({"t": "active", "v": state["active"]})
            if cmd == "stop":
                await send({"t": "interrupt"})
    finally:
        clients.discard(ws)


async def main():
    async with websockets.serve(handler, "127.0.0.1", 8767):
        for s in ["Запускаю голос и языковую модель…", "Прогреваю языковую модель…", "Загружаю голос…"]:
            await send({"t": "status", "text": s, "ready": False, "active": False})
            await asyncio.sleep(1.2)
        state["ready"] = True
        await send({"t": "status", "text": "Готово", "ready": True, "active": False})
        await conversation()


asyncio.run(main())
