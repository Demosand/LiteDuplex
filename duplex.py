# -*- coding: utf-8 -*-
"""Duplex-поведение LiteDuplex поверх Pipecat (работает в .venv, импортируется из app.py).

- StreamingWhisperSTT — текст появляется, пока ты говоришь; финальный текст берётся из уже готового
  промежуточного, если тот успел покрыть конец речи (спекулятивный финал, −0.2 c).
- VoiceMood — громкость/высота/темп твоего голоса относительно твоей обычной речи -> «[голос: …]» для LLM.
- SmartInterrupt — пока бот говорит, перебивают только ≥N слов или «стоп/подожди»; «ага» поверх его речи
  не перебивает и не попадает в разговор (перекрытия речи).
- QwenTTSService — потоковый TTS; первая фраза режется по запятой; «!» звучит бодрее, «…» — спокойнее.
- Backchannel — «ага/ого/эх/ха-ха» в паузах твоей речи (тип выбирает LLM по смыслу), «хм» после реплики,
  метки голоса в контекст LLM.
"""
import asyncio
import random
import re
import time

import httpx
import numpy as np
from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import (BotStartedSpeakingFrame, BotStoppedSpeakingFrame, ErrorFrame,
                                   InterimTranscriptionFrame, InterruptionFrame, LLMContextFrame,
                                   OutputAudioRawFrame, TranscriptionFrame, TTSAudioRawFrame,
                                   UserStartedSpeakingFrame, UserStoppedSpeakingFrame,
                                   VADUserStartedSpeakingFrame, VADUserStoppedSpeakingFrame)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService
from pipecat.services.whisper.stt import WhisperSTTService
from pipecat.transcriptions.language import Language
from pipecat.turns.user_start import MinWordsUserTurnStartStrategy
from pipecat.utils.text.base_text_aggregator import Aggregation, TextType
from pipecat.utils.text.simple_text_aggregator import SimpleTextAggregator
from pipecat.utils.time import time_now_iso8601


# ----------------------------------------------------------------------------- голос пользователя
def voice_features(pcm: bytes, sr: int):
    """Громкость (дБ), высота (Гц), разброс высоты (полутона) и длительность речи. ~2 мс на 5 c звука."""
    a = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    w = int(sr * 0.04)
    hop = w // 2
    n = (len(a) - w) // hop
    if n < 10:
        return None
    frames = a[np.arange(w)[None, :] + hop * np.arange(n)[:, None]]
    frames = frames - frames.mean(1, keepdims=True)
    rms = np.sqrt((frames ** 2).mean(1))
    voiced = rms > max(0.01, np.percentile(rms, 90) * 0.15)
    if voiced.sum() < 8:
        return None
    ac = np.fft.irfft(np.abs(np.fft.rfft(frames[voiced], 2 * w)) ** 2)[:, :w]   # автокорреляция через FFT
    lo, hi = sr // 400, sr // 70
    k = lo + ac[:, lo:hi].argmax(1)
    strong = ac[np.arange(len(k)), k] > 0.3 * np.maximum(ac[:, 0], 1e-9)
    f0 = sr / k[strong]
    med = float(np.median(f0)) if len(f0) > 4 else None
    return {"db": float(20 * np.log10(np.median(rms[voiced]) + 1e-9)),
            "f0": med, "f0_sd": float(np.std(12 * np.log2(f0 / med))) if med else None,
            "dur": float(voiced.sum() * hop / sr)}


class VoiceMood:
    """Сравнивает реплику с обычной манерой речи этого человека (медианы прошлых реплик)."""

    def __init__(self):
        self.hist = []

    def describe(self, feats: list, words: int) -> str:
        feats = [f for f in feats if f]
        if not feats:
            return ""
        dur = sum(f["dur"] for f in feats)
        cur = {"db": np.average([f["db"] for f in feats], weights=[f["dur"] for f in feats]),
               "rate": words / max(dur, 0.3),
               "f0": np.median([f["f0"] for f in feats if f["f0"]] or [0]) or None,
               "f0_sd": np.median([f["f0_sd"] for f in feats if f["f0_sd"] is not None] or [0]) or None}
        parts = []
        if len(self.hist) >= 2 and dur >= 0.8:
            base = {k: np.median([h[k] for h in self.hist if h[k]] or [0]) for k in cur}
            if cur["db"] - base["db"] > 5:
                parts.append("громко")
            elif cur["db"] - base["db"] < -5:
                parts.append("тихо")
            if base["rate"] and words >= 3:
                if cur["rate"] > base["rate"] * 1.3:
                    parts.append("быстро")
                elif cur["rate"] < base["rate"] * 0.75:
                    parts.append("медленно")
            if cur["f0"] and base["f0"]:
                if cur["f0"] > base["f0"] * 1.15:
                    parts.append("выше обычного, взволнованно")
                elif cur["f0"] < base["f0"] * 0.88:
                    parts.append("ниже обычного")
            if cur["f0_sd"] and base["f0_sd"]:
                if cur["f0_sd"] > base["f0_sd"] * 1.4:
                    parts.append("эмоционально")
                elif cur["f0_sd"] < base["f0_sd"] * 0.6:
                    parts.append("монотонно")
        if dur >= 0.8:
            self.hist = (self.hist + [cur])[-20:]
        return f"[голос: {', '.join(parts)}]" if parts else ""


# ----------------------------------------------------------------------------- потоковый STT
class StreamingWhisperSTT(WhisperSTTService):
    """Whisper, который распознаёт и пока ты говоришь.

    Промежуточный текст (InterimTranscriptionFrame) нужен для быстрого перебивания и для выбора «ага/ого».
    Когда VAD сообщает о конце речи (через 0.2 c тишины), промежуточное распознавание часто уже покрыло
    всю фразу — тогда оно и становится финальным, без второго прохода Whisper.
    """

    def __init__(self, *args, interim_step: float = 0.35, **kwargs):
        super().__init__(*args, **kwargs)
        self._step = interim_step
        self._loop_task = None
        self._job = None        # (сегмент, байт аудио, future) — идущее сейчас распознавание
        self._cache = None      # (сегмент, байт аудио, текст) — последнее готовое
        self._bot_speaking = False
        self._preroll = 0       # байт тишины до начала речи в буфере (Segmented держит ~1 c «до»)
        self.latest_text = ""   # промежуточный текст текущего куска речи (до паузы)
        self.turn_text = []     # финальные тексты кусков текущей реплики
        self.turn_feats = []    # характеристики голоса за реплику (для VoiceMood)

    def _transcribe(self, pcm: bytes) -> str:
        a = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        a = np.concatenate([a, np.zeros(int(self.sample_rate * 0.3), np.float32)])
        segs, _ = self._model.transcribe(a, language="ru")
        text = " ".join(s.text.strip() for s in segs if s.no_speech_prob < 0.4).strip()
        words = re.findall(r"\w+", text.lower())
        trigrams = list(zip(words, words[1:], words[2:]))
        # «что вы видите, что вы видите…» — галлюцинация Whisper на обрывке звука, не пускаем дальше
        return "" if len(trigrams) != len(set(trigrams)) else text

    async def _interim_loop(self, seg: int):
        bps = self.sample_rate * 2
        done = 0
        while self._user_speaking and self._segment_count == seg:
            n = len(self._audio_buffer)
            step = self._step * (1.5 if self._bot_speaking else 1.0)   # бот говорит — меньше мешаем TTS
            if n - self._preroll < 0.5 * bps or n - done < step * bps:
                await asyncio.sleep(0.03)
                continue
            fut = asyncio.ensure_future(asyncio.to_thread(self._transcribe, bytes(self._audio_buffer[:n])))
            self._job = (seg, n, fut)
            text = await fut
            done, self._cache = n, (seg, n, text)
            if text and text != self.latest_text and self._segment_count == seg:
                self.latest_text = text
                await self.push_frame(InterimTranscriptionFrame(text, self._user_id, time_now_iso8601(),
                                                                Language.RU))

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, (BotStartedSpeakingFrame, BotStoppedSpeakingFrame)):
            self._bot_speaking = isinstance(frame, BotStartedSpeakingFrame)
        elif isinstance(frame, VADUserStartedSpeakingFrame):
            self.latest_text = ""
            self._preroll = max(0, len(self._audio_buffer) - int(self.sample_rate * 0.2) * 2)
            self._loop_task = asyncio.create_task(self._interim_loop(self._segment_count))

    async def run_stt(self, audio: bytes):
        seg = self._transcribing_segment
        speech = len(audio) - len(self._trailing_silence())
        need = speech - int(self.sample_rate * 0.25) * 2   # VAD замечает тишину через 0.2 c после речи
        self.turn_feats.append(voice_features(audio[:speech], self.sample_rate))
        text = None
        if self._cache and self._cache[0] == seg and self._cache[1] >= need:
            text = self._cache[2]
        elif self._job and self._job[0] == seg and self._job[1] >= need:
            text = await self._job[2]
        if text is None:
            async for f in super().run_stt(audio):
                if isinstance(f, TranscriptionFrame):
                    self.turn_text.append(f.text.strip())
                yield f
            return
        if text:
            self.turn_text.append(text)
            yield TranscriptionFrame(text, self._user_id, time_now_iso8601(), Language.RU)

    def turn_words(self) -> int:
        """Сколько слов сказано в текущей реплике (готовые куски + текущий)."""
        return len(" ".join(self.turn_text + [self.latest_text]).split())

    async def cleanup(self):
        if self._loop_task:
            self._loop_task.cancel()
        await super().cleanup()


# ----------------------------------------------------------------------------- перебивание
# Поддакивание не перебивает: Whisper часто пишет «ага» как «Ага, да.» — это не два содержательных слова.
BACKCHANNEL_WORDS = {"ага", "угу", "ну", "да", "ок", "окей", "ясно", "понятно", "понял", "поняла", "хорошо",
                     "ладно", "так", "м", "мм", "хм", "ах", "ох", "эх", "о", "ого", "вау", "ха", "хаха", "конечно",
                     "верно", "точно", "ага-ага", "угу-угу", "да-да", "ну-ну", "ой", "ааа", "эм", "угум", "мгм"}
# Классические галлюцинации Whisper на дыхании и шуме (наследие субтитров в обучающих данных)
WHISPER_GHOSTS = ("продолжение следует", "спасибо за просмотр", "субтитр", "подписывайтесь", "редактор",
                  "корректор", "dimatorzok", "amara", "благодарю за внимание", "до новых встреч")


def _words(text: str) -> list:
    return re.findall(r"[а-яёa-z]+(?:-[а-яёa-z]+)*", text.lower().replace("ё", "е"))


class SmartInterrupt(MinWordsUserTurnStartStrategy):
    """Пока бот говорит, он замолкает только от ≥min_words содержательных слов или от стоп-слова.
    Поддакивание и галлюцинации Whisper не перебивают и в разговор не попадают."""

    STOP = {"стоп", "хватит", "подожди", "погоди", "замолчи", "стой", "остановись", "прекрати"}

    async def _handle_transcription(self, frame):
        text = frame.text.strip()
        if any(g in text.lower() for g in WHISPER_GHOSTS):
            print(f"[Перебивание] шум/галлюцинация Whisper, пропускаю: «{text}»", flush=True)
            await self.trigger_reset_aggregation()
            return
        if not self._bot_speaking:
            await super()._handle_transcription(frame)
            return
        words = _words(text)
        if any(w.split("-")[0] in self.STOP for w in words):   # «подожди-ка»
            print(f"[Перебивание] стоп-слово: «{text}»", flush=True)
            await self.report_user_turn_started()
            return
        content = [w for w in words if w not in BACKCHANNEL_WORDS]
        if len(content) >= self._min_words:
            print(f"[Перебивание] ты заговорил ({len(content)} сл.): «{text}»", flush=True)
            await self.report_user_turn_started()
            return
        if isinstance(frame, TranscriptionFrame):
            print(f"[Перебивание] нет — поддакивание поверх речи бота: «{text}»", flush=True)
        await self.trigger_reset_aggregation()


# ----------------------------------------------------------------------------- TTS
class FirstClauseAggregator(SimpleTextAggregator):
    """Первую фразу ответа отдаём в TTS уже на запятой (если до неё ≥2 слов) — голос начинается раньше."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self._first = True

    async def aggregate(self, text):
        async for a in super().aggregate(text):
            self._first = False
            yield a
        if self._first and self._aggregation_type == TextType.SENTENCE:
            for m in re.finditer(r"[,;:—–]\s", self._text):
                if len(self._text[:m.start()].split()) >= 2:
                    self._first = False
                    yield Aggregation(text=self._text[:m.end()].strip(), type=TextType.SENTENCE)
                    self._text = self._text[m.end():]
                    break

    async def handle_interruption(self):
        self._first = True
        await super().handle_interruption()

    async def reset(self):   # вызывается и из flush() в конце ответа
        self._first = True
        await super().reset()


def speech_rate(text: str) -> float:
    """Интонация без нагрузки на GPU: «!» — на 5% быстрее и выше, «…» — на 5% медленнее и ниже."""
    if "!" in text:
        return 1.05
    if "…" in text or "..." in text:
        return 0.95
    return 1.0


class QwenTTSService(TTSService):
    """Потоково получает звук от сервера Qwen3-TTS и отдаёт его в Pipecat кусками по 100 мс."""

    def __init__(self, url: str, expressive: bool = True, **kwargs):
        super().__init__(push_start_frame=True, push_stop_frames=True,
                         settings=TTSSettings(model=None, voice=None, language=None), **kwargs)
        old = self._text_aggregator
        self._text_aggregator = FirstClauseAggregator(aggregation_type=old.aggregation_type, language=old.language)
        self._url = url
        self._expressive = expressive
        self._http = httpx.AsyncClient(timeout=60)

    def can_generate_metrics(self) -> bool:
        return True

    async def run_tts(self, text: str, context_id: str):
        # ремарки вроде «(смеётся)», метки «[голос: …]» и эмодзи (маленькая модель их иногда пишет) не озвучиваем
        text = re.sub(r"\[[^\]]*\]|\([^)]*\)|\*[^*]*\*|[\U0001F000-\U0001FAFF☀-➿️]", "", text).strip()
        if not re.search(r"\w", text):
            return
        rate = speech_rate(text) if self._expressive else 1.0
        resampler = create_stream_resampler()
        step = self.sample_rate // 10 * 2   # 100 мс, 16 бит
        try:
            await self.start_tts_usage_metrics(text)
            async with self._http.stream("POST", self._url, json={"text": text}) as r:
                r.raise_for_status()
                sr = int(int(r.headers.get("X-Sample-Rate", "24000")) * rate)
                rest, first = b"", True
                async for chunk in r.aiter_bytes():
                    await self.stop_ttfb_metrics()
                    chunk = rest + chunk
                    cut = len(chunk) // 2 * 2
                    chunk, rest = chunk[:cut], chunk[cut:]
                    audio = await resampler.resample(chunk, sr, self.sample_rate)
                    for i in range(0, len(audio), step):
                        f = TTSAudioRawFrame(audio=audio[i:i + step], sample_rate=self.sample_rate,
                                             num_channels=1, context_id=context_id)
                        if first:   # окно узнает, какие слова звучат с этого кадра
                            f.metadata["ui_text"] = (text, rate)
                            first = False
                        yield f
        except Exception as e:
            yield ErrorFrame(error=f"Qwen3-TTS: {e}")
        finally:
            await self.stop_ttfb_metrics()


# ----------------------------------------------------------------------------- поддакивание и эмоции
REACTIONS = {"1": "нейтрально", "2": "удивление", "3": "сочувствие", "4": "смех"}
REACTION_PROMPT = ("Человек рассказывает тебе что-то и сделал паузу. Его слова: «{text}». "
                   "Какой короткой реакцией слушателя лучше поддержать его сейчас? "
                   "1 — «ага» (просто слушаю), 2 — «ого» (удивительно), 3 — «эх» (грустно, сочувствую), "
                   "4 — «ха-ха» (смешно). Ответь одной цифрой.")


class Backchannel(FrameProcessor):
    """Поддакивает в паузах, пока говорит пользователь, вставляет «хм» после его реплики
    и дописывает в его реплику метку голоса для LLM.

    Клипы идут обычным OutputAudioRawFrame: транспорт не считает это речью бота
    (нет BotStartedSpeaking -> нет ложных перебиваний), и в контекст LLM они не попадают.
    """

    def __init__(self, cfg: dict, stt=None):
        super().__init__()
        self._cfg = cfg
        self._stt = stt
        self._mood = VoiceMood()
        self._src, self._clips = {}, {}
        self._timer = None
        self._in_turn = False
        self._bot = False
        self._turn_t = 0.0
        self._last_bc = 0.0
        self._http = httpx.AsyncClient(timeout=1.0)
        self._llm = f"http://127.0.0.1:{cfg['llm_port']}/v1/chat/completions"

    def set_clips(self, kind, clips):
        self._src[kind] = clips

    async def setup(self, setup):
        await super().setup(setup)
        sr = setup.audio_out_sample_rate
        for k, lst in self._src.items():
            self._clips[k] = []
            for pcm, src_sr in lst:
                rs = create_stream_resampler()
                pcm += b"\0" * (src_sr // 30 * 2)   # 33 мс хвоста, чтобы ресемплер отдал весь клип
                self._clips[k].append((await rs.resample(pcm, src_sr, sr), sr))

    async def _play(self, kind):
        clips = self._clips.get(kind) or self._clips.get("нейтрально") or []
        if clips:
            pcm, sr = random.choice(clips)
            await self.push_frame(OutputAudioRawFrame(audio=pcm, sample_rate=sr, num_channels=1))

    async def _reaction(self, text: str) -> str:
        """LLM выбирает тип реакции одной цифрой (грамматика не даёт ответить иначе), ~20–60 мс."""
        try:
            r = await self._http.post(self._llm, json={
                "model": "x", "max_tokens": 1, "temperature": 0, "grammar": "root ::= [1-4]",
                "id_slot": 1,   # отдельный слот: не вытесняет кэш основного разговора
                "messages": [{"role": "user", "content": REACTION_PROMPT.format(text=text[-300:])}]})
            return REACTIONS.get(r.json()["choices"][0]["message"]["content"].strip()[:1], "нейтрально")
        except Exception:
            return "нейтрально"

    async def _after_pause(self):
        cfg = self._cfg
        text = self._stt.latest_text if self._stt is not None else ""
        words = self._stt.turn_words() if self._stt is not None else 99   # без потокового STT слов не знаем
        kind = asyncio.ensure_future(self._reaction(text)) if text and cfg.get("backchannel_smart", True) else None
        await asyncio.sleep(float(cfg.get("backchannel_pause", 0.5)))
        now = time.time()
        if (self._in_turn and not self._bot and now - self._turn_t > 2.0
                and words >= int(cfg.get("backchannel_min_words", 6))
                and now - self._last_bc > float(cfg.get("backchannel_min_gap", 5.0))
                and random.random() < float(cfg.get("backchannel_prob", 0.5))):
            self._last_bc = now
            await self._play(kind.result() if kind and kind.done() else "нейтрально")
        elif kind:
            kind.cancel()

    def _cancel_timer(self):
        if self._timer:
            self._timer.cancel()
            self._timer = None

    def _tag_context(self, frame):
        """Дописывает «[голос: …]» к последней реплике пользователя."""
        if self._stt is None or not self._cfg.get("voice_emotion", True):
            return
        feats, self._stt.turn_feats = self._stt.turn_feats, []
        msgs = frame.context.messages
        if not msgs or msgs[-1].get("role") != "user" or not isinstance(msgs[-1].get("content"), str):
            return
        tag = self._mood.describe(feats, len(msgs[-1]["content"].split()))
        if tag:
            msgs[-1]["content"] += " " + tag

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        cfg = self._cfg
        if isinstance(frame, (BotStartedSpeakingFrame, BotStoppedSpeakingFrame)):
            self._bot = isinstance(frame, BotStartedSpeakingFrame)
        elif direction == FrameDirection.DOWNSTREAM:
            if isinstance(frame, UserStartedSpeakingFrame):
                self._in_turn, self._turn_t = True, time.time()
            elif isinstance(frame, (VADUserStartedSpeakingFrame, InterruptionFrame)):
                self._cancel_timer()
            elif isinstance(frame, VADUserStoppedSpeakingFrame) and self._in_turn and cfg.get("backchannel"):
                self._cancel_timer()
                self._timer = asyncio.create_task(self._after_pause())
            elif isinstance(frame, LLMContextFrame) and not frame.speculation:
                self._tag_context(frame)
            elif isinstance(frame, UserStoppedSpeakingFrame):
                self._in_turn = False
                self._cancel_timer()
                await self.push_frame(frame, direction)
                words = len(" ".join(self._stt.turn_text).split()) if self._stt is not None else 99
                if self._stt is not None:
                    self._stt.turn_text = []
                if (cfg.get("filler") and time.time() - self._last_bc > 1.5
                        and words >= int(cfg.get("filler_min_words", 8))
                        and random.random() < float(cfg.get("filler_prob", 0.35))):
                    await self._play("fill")
                return
        await self.push_frame(frame, direction)

    async def cleanup(self):
        self._cancel_timer()
        await self._http.aclose()
        await super().cleanup()


# ----------------------------------------------------------------------------- окно (Electron UI)
class UIBridge:
    """WebSocket-мост к окну приложения (ui/). Живёт в своём потоке, поэтому окно видит статус
    загрузки ещё до старта Pipecat. Окно присылает {"t":"start"|"stop"}, получает события разговора."""

    def __init__(self, port: int):
        import threading
        self.port = port
        self.active = False         # голосовой режим включён кнопкой
        self.ready = False
        self.last_status = "Запускаю…"
        self.on_stop = None         # корутина-функция из конвейера: прервать бота
        self.pipeline_loop = None
        self._clients = set()
        self._loop = asyncio.new_event_loop()
        threading.Thread(target=self._loop.run_forever, daemon=True).start()
        asyncio.run_coroutine_threadsafe(self._serve(), self._loop).result()

    async def _serve(self):
        import websockets
        self._server = await websockets.serve(self._handler, "127.0.0.1", self.port)

    async def _handler(self, ws):
        import json
        self._clients.add(ws)
        try:
            await ws.send(json.dumps({"t": "status", "text": self.last_status, "ready": self.ready,
                                      "active": self.active}, ensure_ascii=False))
            async for msg in ws:
                cmd = json.loads(msg).get("t")
                if cmd in ("start", "stop"):
                    self.active = cmd == "start" and self.ready
                    if not self.active and self.on_stop and self.pipeline_loop:
                        asyncio.run_coroutine_threadsafe(self.on_stop(), self.pipeline_loop)
                    self.send({"t": "active", "v": self.active})
        except Exception:
            pass
        finally:
            self._clients.discard(ws)

    def send(self, obj: dict):
        """Потокобезопасно разослать событие всем окнам."""
        import json
        data = json.dumps(obj, ensure_ascii=False)

        def go():
            for ws in list(self._clients):
                asyncio.ensure_future(self._safe(ws, data))
        self._loop.call_soon_threadsafe(go)

    async def _safe(self, ws, data):
        try:
            await ws.send(data)
        except Exception:
            self._clients.discard(ws)

    def status(self, text: str, ready: bool = False):
        print(f"[UI] {text}", flush=True)
        self.last_status, self.ready = text, ready or self.ready
        self.send({"t": "status", "text": text, "ready": self.ready, "active": self.active})


class MicGate(FrameProcessor):
    """Сразу после микрофона: пока голосовой режим выключен — звук не идёт дальше.
    Пока включён — шлёт в окно громкость голоса (для свечения) ~30 раз в секунду."""

    def __init__(self, bridge: UIBridge):
        super().__init__()
        self._b = bridge
        self._last = 0.0
        self._peak = 0.0
        bridge.on_stop = self.broadcast_interruption   # «стоп» в окне обрывает речь бота

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        from pipecat.frames.frames import InputAudioRawFrame
        if isinstance(frame, InputAudioRawFrame):
            if not self._b.active:
                return
            a = np.frombuffer(frame.audio, dtype=np.int16).astype(np.float32) / 32768.0
            self._peak = max(self._peak, float(np.sqrt((a ** 2).mean())) if len(a) else 0.0)
            now = time.time()
            if now - self._last >= 0.033:
                self._b.send({"t": "level", "v": round(min(1.0, self._peak * 6), 3)})
                self._last, self._peak = now, 0.0
        await self.push_frame(frame, direction)


class UIEvents(FrameProcessor):
    """Перед динамиком: знает, когда реально зазвучит каждая фраза бота, и шлёт окну её слова
    с моментом начала и длительностью — окно проявляет слова в такт речи."""

    def __init__(self, bridge: UIBridge):
        super().__init__()
        self._b = bridge
        self._q_end = 0.0       # когда доиграет уже отправленный в динамик звук
        self._sid = 0
        self._cur = None        # (id, накопленная длительность)

    def _close(self):
        if self._cur:
            self._b.send({"t": "dur", "id": self._cur[0], "dur": round(self._cur[1], 3)})
            self._cur = None

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        from pipecat.frames.frames import TTSStoppedFrame
        now = time.time()
        if isinstance(frame, TTSAudioRawFrame):
            dur = len(frame.audio) / 2 / frame.sample_rate
            start = max(now, self._q_end)
            ui = frame.metadata.get("ui_text")
            if ui:
                self._close()
                self._sid += 1
                text, rate = ui
                self._cur = [self._sid, 0.0]
                self._b.send({"t": "say", "id": self._sid, "words": text.split(), "at": int(start * 1000),
                              "dur": round(len(text) / (13.0 * rate), 3)})
            if self._cur:
                self._cur[1] += dur
            self._q_end = start + dur
        elif isinstance(frame, TTSStoppedFrame):
            self._close()
        elif isinstance(frame, InterruptionFrame):
            self._q_end, self._cur = now, None
            self._b.send({"t": "interrupt"})
        elif isinstance(frame, (BotStartedSpeakingFrame, BotStoppedSpeakingFrame)):
            self._b.send({"t": "bot", "v": isinstance(frame, BotStartedSpeakingFrame)})
        elif isinstance(frame, (UserStartedSpeakingFrame, UserStoppedSpeakingFrame)) \
                and direction == FrameDirection.DOWNSTREAM:
            self._b.send({"t": "user", "v": isinstance(frame, UserStartedSpeakingFrame)})
        await self.push_frame(frame, direction)
