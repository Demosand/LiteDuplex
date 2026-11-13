# -*- coding: utf-8 -*-
"""End-to-end duplex test without a microphone (.venv python; llama-server and TTS server must run).

Scenario, fed as a live "microphone" (20 ms frames, silence between):
  1. a sad story with a pause mid-sentence      -> semantic backchannel ("эх"), then the bot answers
  2. "ага" while the bot is speaking              -> must NOT interrupt (overlap)
  3. "стоп, подожди, у меня вопрос" while speaking -> must interrupt
The sink plays the role of the speaker: it emits BotStarted/StoppedSpeaking by audio duration.
Writes cache/tmp/e2e_out.wav (bot audio on the timeline).
"""
import asyncio
import sys
import time
import wave
from pathlib import Path

import httpx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import app  # noqa: E402

app.add_cuda_dll_dirs()
cfg = app.load_config()
cfg["backchannel_prob"] = 1.0   # для теста: поддакивать всегда
cfg["filler_prob"] = 0.0

from pipecat.frames.frames import (BotStartedSpeakingFrame, BotStoppedSpeakingFrame, EndFrame,  # noqa: E402
                                   InputAudioRawFrame, InterimTranscriptionFrame, InterruptionFrame,
                                   LLMContextFrame, LLMTextFrame, OutputAudioRawFrame, TranscriptionFrame,
                                   TTSAudioRawFrame, TTSTextFrame, UserStartedSpeakingFrame,
                                   UserStoppedSpeakingFrame, VADUserStoppedSpeakingFrame)
from pipecat.pipeline.pipeline import Pipeline  # noqa: E402
from pipecat.pipeline.runner import PipelineRunner  # noqa: E402
from pipecat.pipeline.task import PipelineParams, PipelineTask  # noqa: E402
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor  # noqa: E402

http = httpx.Client(timeout=60)


def say(text):
    a = np.frombuffer(http.post(f"http://127.0.0.1:{cfg['tts_port']}/tts", json={"text": text}).content,
                      dtype=np.int16).astype(np.float32)
    return np.interp(np.arange(0, len(a), 1.5), np.arange(len(a)), a).astype(np.int16)   # 24k -> 16k


def cut_mid(a, frac=0.5):
    """Quietest 40 ms window near the middle — a pause between words, not at a phrase end."""
    w = 640
    e = np.array([np.abs(a[i:i + w].astype(np.float32)).mean() for i in range(0, len(a) - w, 160)])
    lo, hi = int(len(e) * (frac - 0.15)), int(len(e) * (frac + 0.15))
    return (lo + int(np.argmin(e[lo:hi]))) * 160 + w // 2


sil = lambda s: np.zeros(int(16000 * s), dtype=np.int16)  # noqa: E731
story = say("Представляешь, я вчера потерял кошелёк со всеми документами и деньгами, и теперь вообще не знаю, что мне делать")
c = cut_mid(story, 0.55)
part1 = np.concatenate([story[:c], sil(1.4), story[c:]])
aga = say("Ага, да, понятно.")
stop = say("Стоп, подожди, у меня вопрос.")
# long monologue, one sentence with breath pauses mid-phrase (Smart Turn: not finished) —
# the bot should cut in by itself (CutIn) before it ends
mono = say("Вот скажи мне честно как ты думаешь мне стоит увольняться или нет потому что я уже месяц об этом "
           "думаю и жена говорит одно а мама говорит совсем другое а начальник вообще молчит и я не понимаю")
for f in (0.8, 0.6, 0.4, 0.2):   # с конца, чтобы индексы впереди не сдвигались
    c = cut_mid(mono, f)
    mono = np.concatenate([mono[:c], sil(0.5), mono[c:]])

T0 = [0.0]
out = []
bot = {"speaking": False, "started": None, "end": 0.0}


def ev(msg):
    print(f"{time.time() - T0[0]:6.2f}s  {msg}", flush=True)


class Mic(FrameProcessor):
    """Live microphone: plays queued clips, silence otherwise."""

    def __init__(self):
        super().__init__()
        self.q = []

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)

    async def run(self, script):
        await asyncio.sleep(1.0)
        T0[0] = time.time()
        buf = np.zeros(0, np.int16)
        task = asyncio.create_task(script(self))
        t_next = time.time()
        while not task.done() or len(buf) or self.q:
            if not len(buf) and self.q:
                buf = self.q.pop(0)
            chunk, buf = (buf[:320], buf[320:]) if len(buf) else (np.zeros(320, np.int16), buf)
            if len(chunk) < 320:
                chunk = np.concatenate([chunk, np.zeros(320 - len(chunk), np.int16)])
            await self.push_frame(InputAudioRawFrame(audio=chunk.tobytes(), sample_rate=16000, num_channels=1))
            t_next += 0.02
            await asyncio.sleep(max(0, t_next - time.time()))
        await self.push_frame(EndFrame())


async def script(mic):
    mic.q.append(part1)
    ev(f"USER: story ({len(part1) / 16000:.1f}s, pause 1.4s mid-sentence)")
    await asyncio.sleep(len(part1) / 16000 + 0.3)
    bot["started"].clear()          # ждём ответ на всю историю, а не на её первую половину
    await bot["started"].wait()
    await asyncio.sleep(0.6)
    ev("USER: 'Ага, да, понятно.' over the bot (overlap)")
    mic.q.append(aga)
    await asyncio.sleep(1.2)
    if bot["speaking"]:
        ev("USER: 'Стоп, подожди, у меня вопрос.' over the bot")
        mic.q.append(stop)
    else:
        ev("bot already finished — can't test interruption by stop-phrase")
    await asyncio.sleep(9.0)
    mic.q.append(mono)
    ev(f"USER: long monologue ({len(mono) / 16000:.1f}s, breath pauses) — expect '[Перебиваю]' before it ends")
    await asyncio.sleep(len(mono) / 16000 + 6.0)


class Speaker(FrameProcessor):
    """Stands in for the audio output transport: timeline + BotStarted/StoppedSpeaking."""

    async def _watch_end(self):
        while time.time() < bot["end"]:
            await asyncio.sleep(0.05)
        bot["speaking"] = False
        ev("bot stopped speaking (audio ended)")
        await self.push_frame(BotStoppedSpeakingFrame(), FrameDirection.UPSTREAM)

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        now = time.time() - T0[0]
        if isinstance(frame, TTSAudioRawFrame):
            if not bot["speaking"]:
                bot["speaking"] = True
                bot["started"].set()
                ev("BOT VOICE starts")
                await self.push_frame(BotStartedSpeakingFrame(), FrameDirection.UPSTREAM)
                bot["end"] = time.time()
                asyncio.create_task(self._watch_end())
            dur = len(frame.audio) / 2 / frame.sample_rate
            start = max(now, bot["end"] - T0[0])
            bot["end"] = T0[0] + start + dur
            out.append((start, frame.audio, frame.sample_rate))
        elif isinstance(frame, OutputAudioRawFrame):
            ev(f"CLIP played ({len(frame.audio) / 2 / frame.sample_rate:.2f}s)")
            out.append((now, frame.audio, frame.sample_rate))
        elif isinstance(frame, InterruptionFrame):
            if bot["speaking"]:
                ev("*** INTERRUPTED — bot goes silent")
                bot["end"] = time.time()
                out[:] = [o for o in out if o[0] <= now]
        elif isinstance(frame, (UserStartedSpeakingFrame, UserStoppedSpeakingFrame, VADUserStoppedSpeakingFrame)) \
                and direction == FrameDirection.DOWNSTREAM:
            ev(type(frame).__name__)
        elif isinstance(frame, TTSTextFrame):
            ev(f"   spoken: {frame.text!r}")
        await self.push_frame(frame, direction)


class Tap(FrameProcessor):
    def __init__(self):
        super().__init__()
        self.first = True
        self.last_interim = ""

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, InterimTranscriptionFrame) and frame.text != self.last_interim:
            self.last_interim = frame.text
            ev(f"stt interim: {frame.text!r}")
        elif isinstance(frame, TranscriptionFrame):
            ev(f"STT final: {frame.text!r}")
        elif isinstance(frame, LLMContextFrame):
            ev(f"LLM request, last user msg: {frame.context.messages[-1]['content']!r}")
            self.first = True
        elif isinstance(frame, LLMTextFrame) and self.first:
            ev("LLM first token")
            self.first = False
        await self.push_frame(frame, direction)


async def main():
    bot["started"] = asyncio.Event()
    stt, usr, bc, llm, tts, assistant = app.build_processors(cfg)
    app.load_clips(cfg, bc, stt._model)
    orig = bc._reaction

    async def spy(text):
        k = await orig(text)
        ev(f"backchannel kind chosen by LLM: {k} (for {text[-60:]!r})")
        return k
    bc._reaction = spy
    orig_tag = bc._tag_context

    def tag_spy(frame):
        orig_tag(frame)
        ev(f"LLM gets user msg: {frame.context.messages[-1]['content']!r}")
    bc._tag_context = tag_spy
    mic = Mic()
    pipeline = Pipeline([mic, stt, Tap(), usr, bc, llm, Tap(), tts, Speaker(), assistant])
    task = PipelineTask(pipeline, params=PipelineParams(audio_in_sample_rate=16000, audio_out_sample_rate=24000,
                                                        enable_metrics=True), idle_timeout_secs=None)
    asyncio.get_event_loop().create_task(mic.run(script))
    await PipelineRunner(handle_sigint=False).run(task)

    total = np.zeros(int(24000 * 40), dtype=np.float32)
    for t, pcm, sr in out:
        a = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
        s = int(t * 24000)
        total[s:s + len(a)] += a[:len(total) - s]
    with wave.open(str(ROOT / "cache" / "tmp" / "e2e_out.wav"), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000)
        w.writeframes(np.clip(total, -32767, 32767).astype(np.int16).tobytes())


asyncio.run(main())
