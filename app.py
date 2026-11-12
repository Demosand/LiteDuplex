# -*- coding: utf-8 -*-
r"""
Русский голосовой собеседник: Pipecat + Smart Turn + Whisper + Gemma 4 (с твоей LoRA) + Qwen3-TTS.

Запуск:  python app.py      (или двойной клик по start.bat)

Всё ставится и скачивается ТОЛЬКО в папку, где лежит этот файл:
  .venv\          основной venv: Pipecat, Whisper, Gemma + LoRA
  .venv_tts\      отдельный venv для голоса Qwen3-TTS (ему нужен старый transformers)
  models\         Gemma, Whisper, Qwen3-TTS
  cache\          кэши pip / Hugging Face / torch / временные файлы
  config.json     настройки (личность, голос, громкость ответов) — можно править

LoRA: положи папку с LoRA (ту, где лежит adapter_config.json) в эту же папку.
Программа найдёт её сама.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
VENV_TTS = ROOT / ".venv_tts"
MODELS = ROOT / "models"
CACHE = ROOT / "cache"
CONFIG_PATH = ROOT / "config.json"

TOOLS = ROOT / "tools"
LLAMA_DIR = TOOLS / "llama"          # готовые бинарники llama.cpp (CUDA)
LLAMA_BUILD = "b12284"
LLAMA_ZIPS = [f"llama-{LLAMA_BUILD}-bin-win-cuda-13.5-x64.zip", "cudart-llama-bin-win-cuda-13.5-x64.zip"]
TORCH_INDEX = "https://download.pytorch.org/whl/cu128"   # CUDA 12.8 — нужна для RTX 50xx (Blackwell)
DEFAULT_GEMMA = "google/gemma-4-12B-it-qat-q4_0-unquantized"
WHISPER_REPO = "deepdml/faster-whisper-large-v3-turbo-ct2"
TTS_REPO = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"

# Примеры тона важнее правил: маленькая модель копирует стиль примеров лучше, чем следует запретам.
SYSTEM_PROMPT = (
    "Ты живой человек в голосовом разговоре, а не ассистент и не ИИ-помощник. "
    "Отвечай на том языке, на котором говорит собеседник. "
    "Говори коротко: обычно одна фраза, максимум две, как в болтовне с другом. "
    "У тебя есть характер: ироничный, прямой, можешь подколоть или быть грубоватым. "
    "У тебя есть свои мнения, вкусы и настроение — высказывай их, не уходи в нейтральность. "
    "Иногда задавай встречный вопрос, но не в каждой реплике. "
    "Никогда не говори «Конечно!», «Отличный вопрос», «Чем могу помочь?», «Как ИИ…» и не предлагай помощь. "
    "Никаких списков, markdown, эмодзи и смайликов: всё, что ты пишешь, будет произнесено вслух.\n"
    "Примеры тона:\n"
    "— Привет, как дела? — Да нормально, сижу туплю. А у тебя что?\n"
    "— Я опять проспал работу. — Опять? Слушай, пора менять будильник. Или себя.\n"
    "— Посоветуй фильм. — Пересмотри «Интерстеллар», я его каждый год смотрю и не надоедает."
)

DEFAULT_CONFIG = {
    "system_prompt": SYSTEM_PROMPT,
    # "eco" — экономия видеопамяти (~10.5 ГБ на всё приложение с Gemma 4 12B), "fast" — быстрее (~12 ГБ).
    # Переключается в окне приложения.
    "vram_mode": "eco",
    "max_new_tokens": 120,
    "temperature": 0.8,
    "top_p": 0.95,
    "repetition_penalty": 1.1,
    # "qwen" — Qwen3-TTS (живее, но медленный на Windows); "piper" — Piper (очень быстрый, звучит проще)
    "tts_engine": "qwen",
    "piper_voice": "ru_RU-denis-medium",   # другие: ru_RU-dmitri-medium, ru_RU-irina-medium, ru_RU-ruslan-medium
    "tts_speaker": "",          # пусто = первый доступный голос; список голосов печатается при запуске
    "tts_language": "russian",
    "whisper_device": "cuda",   # если Whisper не заработает на видеокарте — поставь "cpu"
    # "llamacpp" — быстрый llama.cpp с GGUF-моделью ниже; "hf" — старый вариант: Gemma + LoRA через transformers
    "llm_backend": "llamacpp",
    # GGUF-модель для llama.cpp и откуда её скачать, если файла нет. Gemma 4 12B (QAT Q4_0 от Google):
    # многоязычная, Apache 2.0, живой разговорный тон. Нужен --reasoning off (иначе шаблон включает «размышления»).
    # Полегче: "llm_gguf": "models/qwen3-4b/Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
    #          "llm_gguf_repo": "unsloth/Qwen3-4B-Instruct-2507-GGUF" (~3 ГБ, но заметно глупее)
    "llm_gguf": "models/gemma4-12b/gemma-4-12b-it-qat-q4_0.gguf",
    "llm_gguf_repo": "google/gemma-4-12B-it-qat-q4_0-gguf",
    "llm_ctx": 4096,
    # Для "hf": готовая сжатая модель с вшитой LoRA на Hugging Face; если пусто — Gemma (~24 ГБ) + папка LoRA.
    "merged_model": "",
    # Перебивание: пока бот говорит, он замолкает, только если ты сказал ≥N слов (или «стоп», «подожди»…).
    # Твои «ага», «угу» поверх его речи не перебивают и не попадают в разговор.
    "interrupt_min_words": 2,
    # Если Smart Turn сомневается, договорил ли ты, бот всё равно ответит после стольких секунд тишины
    "turn_max_silence": 2.0,
    # Потоковое распознавание: текст появляется, пока ты говоришь (быстрее ответ и перебивание)
    "stream_stt": True,
    # Эмоции голоса: громкость, высота, темп твоей речи → подсказка для ИИ «[голос: тихо, медленно]»
    "voice_emotion": True,
    # Интонация ответа: «!» — чуть быстрее и выше, «…» — медленнее и ниже
    "expressive_tts": True,
    # Поддакивание, пока ты говоришь. Тип реакции выбирает ИИ по смыслу сказанного.
    "backchannel": True,
    # Qwen-TTS плохо произносит междометия («эх» -> «аааа»), поэтому «удивление» и «сочувствие» — словами
    "backchannel_clips": {"нейтрально": ["ага", "угу"], "удивление": ["ничего себе", "ого!"],
                          "сочувствие": ["блин", "жаль"], "смех": ["ха-ха"]},
    "backchannel_prob": 0.5,        # вероятность поддакнуть в паузе
    "backchannel_min_words": 6,     # поддакивать, только если ты уже сказал столько слов (не на «привет»)
    "backchannel_pause": 0.5,       # сколько секунд тишины (после 0.2 c VAD) ждать перед «ага»
    "backchannel_min_gap": 5.0,     # не чаще, чем раз в N секунд
    "backchannel_volume": 0.6,
    # Короткая заглушка («хм», «так») сразу после твоей реплики — пока думает ИИ
    "filler": True,
    "filler_words": ["хм", "так"],
    "filler_prob": 0.35,
    "filler_min_words": 8,          # «хм» только после длинной реплики: на короткую бот и так отвечает за ~0.3 c
    "llm_port": 8765,
    "tts_port": 8766,
    "ui_port": 8767,   # окно приложения (ui/) подключается сюда по WebSocket
}


# --------------------------------------------------------------------------------------
# Всё внутри папки проекта: кэши и временные файлы
# --------------------------------------------------------------------------------------
def local_env() -> dict:
    env = os.environ.copy()
    tmp = CACHE / "tmp"
    dirs = {
        "HF_HOME": CACHE / "huggingface",
        "HF_HUB_CACHE": CACHE / "huggingface" / "hub",
        "TORCH_HOME": CACHE / "torch",
        "PIP_CACHE_DIR": CACHE / "pip",
        "XDG_CACHE_HOME": CACHE / "xdg",
        "CUDA_CACHE_PATH": CACHE / "cuda",
        "TRITON_CACHE_DIR": CACHE / "triton",
        "NUMBA_CACHE_DIR": CACHE / "numba",
        "MPLCONFIGDIR": CACHE / "matplotlib",
        "PYTHONPYCACHEPREFIX": CACHE / "pycache",
        "PYTHONUSERBASE": CACHE / "userbase",
        "TMP": tmp, "TEMP": tmp, "TMPDIR": tmp,
    }
    for k, v in dirs.items():
        Path(v).mkdir(parents=True, exist_ok=True)
        env[k] = str(v)
    env["PIP_NO_INPUT"] = "1"
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"   # логи серверов сразу видны
    env["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    return env


os.environ.update(local_env())
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


def venv_python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def run(cmd, **kw):
    print("  >", " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], check=True, env=local_env(), cwd=str(ROOT), **kw)


def load_config() -> dict:
    cfg = dict(DEFAULT_CONFIG)
    if CONFIG_PATH.exists():
        cfg.update(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
    CONFIG_PATH.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    return cfg


def find_lora(required: bool = True):
    skip = {".venv", ".venv_tts", "models", "cache"}
    found = []
    for p in ROOT.rglob("adapter_config.json"):
        if not skip.intersection(p.relative_to(ROOT).parts):
            found.append(p.parent)
    if not found:
        if not required:
            return None
        sys.exit(
            "\nНе нашёл LoRA.\n"
            "Скачай с Google Диска папку ru_voice/lora (в ней лежит adapter_config.json)\n"
            f"и положи её сюда: {ROOT}\n"
        )
    found.sort(key=lambda d: (d / "adapter_config.json").stat().st_mtime, reverse=True)
    if len(found) > 1:
        print("Нашёл несколько LoRA, беру самую свежую:", found[0])
    return found[0]


def base_model_id(lora_dir: Path) -> str:
    cfg = json.loads((lora_dir / "adapter_config.json").read_text(encoding="utf-8"))
    name = cfg.get("base_model_name_or_path") or ""
    # Unsloth иногда пишет сюда свою копию модели или локальный путь — берём оригинал Google
    if not name.startswith("google/gemma-4"):
        return DEFAULT_GEMMA
    return name


# --------------------------------------------------------------------------------------
# Установка
# --------------------------------------------------------------------------------------
def make_venv(venv: Path):
    if not venv_python(venv).exists():
        print(f"\n=== Создаю {venv.name} ===")
        run([sys.executable, "-m", "venv", venv])
        run([venv_python(venv), "-m", "pip", "install", "-U", "pip", "wheel", "setuptools"])


def install_main():
    marker = VENV / ".installed"
    if marker.exists():
        return
    make_venv(VENV)
    py = venv_python(VENV)
    print("\n=== Ставлю PyTorch (CUDA 12.8) в .venv ===")
    run([py, "-m", "pip", "install", "torch", "torchaudio", "torchvision", "--index-url", TORCH_INDEX])
    print("\n=== Ставлю Pipecat, Whisper, transformers, LoRA-библиотеки в .venv ===")
    run([py, "-m", "pip", "install",
         "pipecat-ai[whisper,silero,openai,local]",
         "transformers>=5.17", "peft", "accelerate", "bitsandbytes",
         "fastapi", "uvicorn", "huggingface_hub", "httpx", "numpy", "sentencepiece", "protobuf",
         "--extra-index-url", TORCH_INDEX])
    marker.write_text("ok")


def install_tts():
    marker = VENV_TTS / ".installed"
    if marker.exists():
        return
    make_venv(VENV_TTS)
    py = venv_python(VENV_TTS)
    print("\n=== Ставлю PyTorch (CUDA 12.8) в .venv_tts ===")
    run([py, "-m", "pip", "install", "torch", "torchaudio", "--index-url", TORCH_INDEX])
    print("\n=== Ставлю Qwen3-TTS в .venv_tts ===")
    run([py, "-m", "pip", "install", "qwen-tts", "fastapi", "uvicorn", "numpy",
         "--extra-index-url", TORCH_INDEX])
    marker.write_text("ok")


def llm_gguf(cfg: dict) -> Path:
    return ROOT / cfg["llm_gguf"]


def download_models(cfg: dict, lora_dir):
    jobs = [(WHISPER_REPO, str(MODELS / "whisper"), False), (TTS_REPO, str(MODELS / "qwen3-tts"), False)]
    gemma_id = ""
    if cfg.get("llm_backend", "llamacpp") == "hf":
        merged = (cfg.get("merged_model") or "").strip()
        gemma_id = merged or base_model_id(lora_dir)
        jobs.insert(0, (gemma_id, str(MODELS / ("gemma-merged" if merged else "gemma")), True))
    elif not llm_gguf(cfg).exists():
        repo = (cfg.get("llm_gguf_repo") or "").strip()
        if not repo:
            sys.exit(f"Нет файла модели {llm_gguf(cfg)}, и в config.json не задан llm_gguf_repo.")
        jobs.insert(0, (repo, str(llm_gguf(cfg)), False))
    script = f"""
import os, sys
from pathlib import Path
from huggingface_hub import snapshot_download, hf_hub_download, get_token, login

jobs = [(r, Path(d), g) for r, d, g in {jobs!r}]
if any(gated and not (d / ".done").exists() for _, d, gated in jobs) and not get_token():
    print()
    print("Для Gemma нужен токен Hugging Face:")
    print("  1) зайди на https://huggingface.co/{gemma_id} и прими лицензию")
    print("  2) Settings -> Access Tokens -> New token (тип Read)")
    tok = input("Вставь токен и нажми Enter: ").strip()
    login(token=tok)   # токен сохранится в папке cache/huggingface, не на диск C

for repo, dst, _ in jobs:
    if dst.suffix == ".gguf":   # одиночный GGUF-файл для llama.cpp
        print(f"\\n=== Скачиваю {{repo}}/{{dst.name}} -> {{dst.parent}} ===")
        hf_hub_download(repo, dst.name, local_dir=str(dst.parent))
        continue
    if (dst / ".done").exists():
        continue
    print(f"\\n=== Скачиваю {{repo}} -> {{dst}} ===")
    snapshot_download(repo_id=repo, local_dir=str(dst))
    (dst / ".done").write_text("ok")
print("\\nВсе модели на месте.")
"""
    run([venv_python(VENV), "-c", script])


def llama_exe(name: str) -> Path:
    return LLAMA_DIR / (name + (".exe" if os.name == "nt" else ""))


def install_llamacpp():
    if llama_exe("llama-server").exists():
        return
    import urllib.request
    import zipfile
    LLAMA_DIR.mkdir(parents=True, exist_ok=True)
    for z in LLAMA_ZIPS:
        dst = CACHE / "tmp" / z
        url = f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_BUILD}/{z}"
        print(f"\n=== Скачиваю {url} ===")
        urllib.request.urlretrieve(url, dst)
        with zipfile.ZipFile(dst) as f:
            f.extractall(LLAMA_DIR)
        dst.unlink()


def setup_and_launch():
    v = sys.version_info
    if not ((3, 10) <= v[:2] <= (3, 12)):
        sys.exit(f"Нужен Python 3.10–3.12, а у тебя {v.major}.{v.minor}. Поставь Python 3.12 и запусти снова.")
    cfg = load_config()
    lora = None
    if cfg.get("llm_backend", "llamacpp") == "hf":
        merged = bool((cfg.get("merged_model") or "").strip())
        lora = find_lora(required=not merged)
        print("Модель:", cfg["merged_model"] if merged else f"Gemma + LoRA из {lora}")
    else:
        print("Модель:", cfg["llm_gguf"])
    install_main()
    install_tts()
    download_models(cfg, lora)
    if cfg.get("llm_backend", "llamacpp") == "llamacpp":
        install_llamacpp()
    print("\n=== Запускаю собеседника ===")
    extra = ["--ui"] if "--ui" in sys.argv else []
    sys.exit(subprocess.call([str(venv_python(VENV)), str(Path(__file__)), "--run", *extra],
                             env=local_env(), cwd=str(ROOT)))


# --------------------------------------------------------------------------------------
# Сервер голоса (работает в .venv_tts)
# --------------------------------------------------------------------------------------
def tts_server():
    import threading
    import numpy as np
    import torch
    import uvicorn
    import asyncio
    from fastapi import FastAPI, Request
    from fastapi.responses import Response, StreamingResponse
    from qwen_tts import Qwen3TTSModel
    sys.path.insert(0, str(ROOT))
    from fast_tts import FastQwenTTS

    cfg = load_config()
    print("[TTS] загружаю Qwen3-TTS...")
    tts = Qwen3TTSModel.from_pretrained(
        str(MODELS / "qwen3-tts"), device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa")

    speakers = tts.get_supported_speakers() or []
    langs = tts.get_supported_languages() or []
    print("[TTS] голоса:", ", ".join(speakers) or "—")
    speaker = (cfg.get("tts_speaker") or "").lower()
    if speakers and speaker not in speakers:
        speaker = speakers[0]
    language = (cfg.get("tts_language") or "auto").lower()
    if langs and language not in [x.lower() for x in langs]:
        language = "auto"
    print(f"[TTS] голос: {speaker}, язык: {language}  (сменить: tts_speaker в config.json)")
    eco = cfg.get("vram_mode", "eco") == "eco"
    if eco:
        # Таблица текстовых эмбеддингов (~600 МБ) нужна раз на фразу, не на каждый кадр — держим её в RAM.
        # HF грузит веса talker одним блоком, и освобождённый кусок не вернуть драйверу —
        # поэтому сначала перекладываем talker целиком (каждый тензор — отдельный блок).
        talker, dev = tts.model.talker, tts.model.talker.device
        talker.to("cpu")
        torch.cuda.empty_cache()
        talker.to(dev)
        emb = talker.get_text_embeddings()
        emb.to("cpu")
        torch.cuda.empty_cache()
        emb.register_forward_pre_hook(lambda m, args: (args[0].cpu(),))
        emb.register_forward_hook(lambda m, args, out: out.to(dev))
    # Eco: кэш на 512 кадров (~40 с речи на фразу) вместо 2048 — −170 МБ
    fast = FastQwenTTS(tts, max_len=512 if eco else 2048)
    assert fast.dev.type == "cuda", fast.dev
    t0 = time.time()
    for _ in range(2):   # первый раз — захват CUDA Graph
        list(fast.stream("Привет, это проверка голоса.", speaker, language))
    if eco:
        torch.cuda.empty_cache()   # вернуть драйверу память, оставшуюся после прогрева
    print(f"[TTS] прогрев: {time.time() - t0:.1f} c, режим {'Eco' if eco else 'Fast'}")
    free, total = torch.cuda.mem_get_info()
    print(f"[TTS] видеопамять: свободно {free / 2**30:.1f} из {total / 2**30:.1f} ГБ")

    app = FastAPI()

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.post("/tts")
    async def synth(req: Request):
        text = (await req.json()).get("text", "").strip()
        if not text:
            return Response(b"", headers={"X-Sample-Rate": str(fast.sr)})
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()
        cancel = threading.Event()

        def work():
            t0, first, samples = time.time(), None, 0
            gen = fast.stream(text, speaker, language, cancel=cancel)
            try:
                for wav in gen:
                    first = first or time.time() - t0
                    samples += len(wav)
                    pcm = (np.clip(wav, -1, 1) * 32767).astype(np.int16).tobytes()
                    loop.call_soon_threadsafe(q.put_nowait, pcm)
            except Exception as e:
                print(f"[TTS] ошибка: {e}")
            finally:
                gen.close()
                dt, dur = time.time() - t0, samples / fast.sr
                print(f"[TTS] первый звук {(first or 0) * 1000:.0f} мс, {dt:.1f} c на {dur:.1f} c речи"
                      f"{' (прервано)' if cancel.is_set() else ''}")
                loop.call_soon_threadsafe(q.put_nowait, None)

        threading.Thread(target=work, daemon=True).start()

        async def body():
            try:
                while (chunk := await q.get()) is not None:
                    yield chunk
            finally:
                cancel.set()   # клиент отключился (перебивание) — останавливаем генерацию

        return StreamingResponse(body(), media_type="application/octet-stream",
                                 headers={"X-Sample-Rate": str(fast.sr)})

    print(f"[TTS] готов на порту {cfg['tts_port']}")
    uvicorn.run(app, host="127.0.0.1", port=cfg["tts_port"], log_level="warning")


# --------------------------------------------------------------------------------------
# Локальный LLM-сервер: Gemma 4 + LoRA, OpenAI-совместимый (работает в .venv)
# --------------------------------------------------------------------------------------
def start_llm_server(cfg: dict, lora_dir):
    import asyncio
    import threading
    import torch
    import uvicorn
    from fastapi import FastAPI, Request
    from fastapi.responses import StreamingResponse
    from peft import PeftModel
    from transformers import (AutoProcessor, AutoTokenizer, BitsAndBytesConfig, StoppingCriteria,
                              StoppingCriteriaList, TextIteratorStreamer)
    import transformers

    merged = bool((cfg.get("merged_model") or "").strip())
    if merged:
        print("[LLM] загружаю сжатую Gemma 4 с вшитой LoRA (1–2 минуты)...")
        base = str(MODELS / "gemma-merged")
        extra = {}   # модель уже в 4 битах, настройки сжатия лежат в её config.json
    else:
        print("[LLM] загружаю Gemma 4 в 4 битах + LoRA (1–3 минуты)...")
        base = str(MODELS / "gemma")
        extra = {"quantization_config": BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)}
    model, last_err = None, None
    for cls_name in ("AutoModelForMultimodalLM", "AutoModelForImageTextToText", "AutoModelForCausalLM"):
        cls = getattr(transformers, cls_name, None)
        if cls is None:
            continue
        try:
            model = cls.from_pretrained(base, device_map="cuda:0", dtype=torch.bfloat16, **extra)
            break
        except Exception as e:
            last_err = e
    if model is None:
        raise RuntimeError(f"Не удалось загрузить Gemma: {last_err}")
    if not merged:
        model = PeftModel.from_pretrained(model, str(lora_dir))
    model.eval()

    # Нам нужен только текст, поэтому берём токенизатор (без обработчика картинок, ему нужен torchvision)
    src = base if merged else str(lora_dir)
    try:
        processor = AutoTokenizer.from_pretrained(src)
    except Exception:
        src = base
        processor = AutoTokenizer.from_pretrained(src)
    if not getattr(processor, "chat_template", None):
        for d in (Path(src), Path(base)):
            jinja, js = d / "chat_template.jinja", d / "chat_template.json"
            if jinja.exists():
                processor.chat_template = jinja.read_text(encoding="utf-8")
                break
            if js.exists():
                processor.chat_template = json.loads(js.read_text(encoding="utf-8"))["chat_template"]
                break
    if not getattr(processor, "chat_template", None):
        raise RuntimeError("В папке модели нет шаблона чата (chat_template.jinja)")
    tokenizer = getattr(processor, "tokenizer", processor)
    gen_lock = threading.Lock()
    free, total = torch.cuda.mem_get_info()
    print(f"[LLM] готов. Видеопамять: свободно {free / 2**30:.1f} из {total / 2**30:.1f} ГБ")

    def to_text(content):
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
        return ""

    class Cancel(StoppingCriteria):
        def __init__(self, ev):
            self.ev = ev

        def __call__(self, *a, **k):
            return self.ev.is_set()

    app = FastAPI()

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.post("/v1/chat/completions")
    async def chat(req: Request):
        body = await req.json()
        msgs = []
        for m in body.get("messages", []):
            role = {"developer": "system"}.get(m.get("role"), m.get("role"))
            if role not in ("system", "user", "assistant"):
                continue
            text = to_text(m.get("content"))
            if not text:
                continue
            if msgs and msgs[-1]["role"] == role and role != "system":
                msgs[-1]["content"] += " " + text
            else:
                msgs.append({"role": role, "content": text})

        try:
            prompt = processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False,
                                                   enable_thinking=False)
        except TypeError:
            prompt = processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=[prompt], return_tensors="pt", add_special_tokens=False).to("cuda:0")

        cancel = threading.Event()
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=120)
        kwargs = dict(**inputs, streamer=streamer, do_sample=True,
                      max_new_tokens=cfg["max_new_tokens"], temperature=cfg["temperature"],
                      top_p=cfg["top_p"], repetition_penalty=cfg["repetition_penalty"],
                      stopping_criteria=StoppingCriteriaList([Cancel(cancel)]))

        def generate():
            with gen_lock, torch.inference_mode():
                model.generate(**kwargs)

        threading.Thread(target=generate, daemon=True).start()
        cid = f"chatcmpl-{int(time.time() * 1000)}"

        def chunk(delta, finish=None):
            return "data: " + json.dumps({
                "id": cid, "object": "chat.completion.chunk", "created": int(time.time()),
                "model": "gemma-lora",
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }, ensure_ascii=False) + "\n\n"

        async def events():
            try:
                yield chunk({"role": "assistant"})
                it = iter(streamer)
                while True:
                    if await req.is_disconnected():   # пользователь перебил — останавливаем генерацию
                        cancel.set()
                        return
                    piece = await asyncio.to_thread(next, it, None)
                    if piece is None:
                        break
                    if piece:
                        yield chunk({"content": piece})
                yield chunk({}, "stop")
                yield "data: [DONE]\n\n"
            finally:
                cancel.set()

        return StreamingResponse(events(), media_type="text/event-stream")

    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=cfg["llm_port"], log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()


def tie_children_to_me():
    """Windows Job Object: все процессы, запущенные после этого (llama.cpp, сервер голоса), Windows убьёт
    вместе с этим процессом — даже если его закрыли крестиком консоли или он упал. Иначе они остаются
    висеть, держат порты и ~10 ГБ видеопамяти, а следующий запуск молча работает с ними."""
    if os.name != "nt":
        return
    import ctypes
    from ctypes import wintypes as w

    class Basic(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", w.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", w.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", w.DWORD), ("SchedulingClass", w.DWORD)]

    class Extended(ctypes.Structure):
        _fields_ = [("Basic", Basic), ("Io", ctypes.c_uint64 * 6), ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t)]

    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.CreateJobObjectW.restype = w.HANDLE
    k.GetCurrentProcess.restype = w.HANDLE
    job = k.CreateJobObjectW(None, None)
    info = Extended()
    info.Basic.LimitFlags = 0x2000   # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    ok = job and k.SetInformationJobObject(w.HANDLE(job), 9, ctypes.byref(info), ctypes.sizeof(info))         and k.AssignProcessToJobObject(w.HANDLE(job), w.HANDLE(k.GetCurrentProcess()))
    if not ok:
        print(f"[Запуск] не удалось привязать дочерние процессы (код {ctypes.get_last_error()})")
    globals()["_JOB"] = job   # держим handle до конца жизни процесса


def free_ports(*ports):
    """Убивает старые llama-server / серверы голоса, оставшиеся на наших портах от прошлых запусков."""
    if os.name != "nt":
        return
    out = subprocess.run(["netstat", "-ano", "-p", "TCP"], capture_output=True, text=True, errors="ignore").stdout
    pids = {line.split()[-1] for line in out.splitlines()
            if "LISTENING" in line and any(line.split()[1].endswith(f":{p}") for p in ports)}
    for pid in pids:
        info = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                              capture_output=True, text=True, errors="ignore").stdout.lower()
        if "llama-server" in info or "python" in info:
            print(f"[Запуск] закрываю старый сервер на порту (PID {pid})")
            subprocess.run(["taskkill", "/PID", pid, "/T", "/F"], capture_output=True)


def exit_with_window():
    """Режим окна: окно (Electron) передаёт свой PID. Пропало окно (закрыли, упало, убили) — выходим,
    а вместе с нами Job Object закроет и серверы. Ждём сам процесс окна, а не stdin: блокирующее
    чтение stdin в потоке на Windows вешает загрузку DLL (numpy) в главном потоке."""
    pid = int(os.environ.get("LITEDUPLEX_WINDOW_PID") or 0)
    if os.name != "nt" or not pid:
        return
    import ctypes
    import threading
    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.OpenProcess.restype = ctypes.c_void_p
    handle = k.OpenProcess(0x00100000, False, pid)   # SYNCHRONIZE
    if not handle:
        return

    def watch():
        k.WaitForSingleObject(ctypes.c_void_p(handle), 0xFFFFFFFF)
        os._exit(0)
    threading.Thread(target=watch, daemon=True).start()


def wait_http(url: str, what: str, proc=None, timeout=900):
    import httpx
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc is not None and proc.poll() is not None:
            sys.exit(f"{what} упал при запуске — смотри ошибку выше.")
        try:
            if httpx.get(url, timeout=2).status_code == 200:
                return
        except Exception:
            pass
        time.sleep(1)
    sys.exit(f"{what} не запустился за {timeout} секунд.")


# --------------------------------------------------------------------------------------
# Голосовой бот (работает в .venv)
# --------------------------------------------------------------------------------------
def add_cuda_dll_dirs():
    """На Windows Whisper (ctranslate2) ищет библиотеки CUDA — подсовываем те, что пришли с torch."""
    if os.name != "nt":
        return
    import importlib.util
    spec = importlib.util.find_spec("torch")
    dirs = []
    if spec and spec.origin:
        dirs.append(Path(spec.origin).parent / "lib")
    sp = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    if sp.exists():
        dirs += [p for p in sp.glob("*/bin")]
    for d in dirs:
        if d.exists():
            os.add_dll_directory(str(d))
            os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")


def run_bot():
    import asyncio
    add_cuda_dll_dirs()
    cfg = load_config()
    free_ports(cfg["llm_port"], cfg["tts_port"], cfg.get("ui_port", 8767))
    bridge = None
    if "--ui" in sys.argv:
        exit_with_window()
        sys.path.insert(0, str(ROOT))
        from duplex import UIBridge
        bridge = UIBridge(int(cfg.get("ui_port", 8767)))
    say = bridge.status if bridge else (lambda text: None)
    say("Запускаю голос и языковую модель…")
    lora = None
    if cfg.get("llm_backend", "llamacpp") == "hf":
        lora = find_lora(required=not (cfg.get("merged_model") or "").strip())

    use_qwen = cfg.get("tts_engine", "qwen") == "qwen"
    if not use_qwen:
        ensure_piper()
    tie_children_to_me()
    procs = []
    if use_qwen:
        procs.append(subprocess.Popen([str(venv_python(VENV_TTS)), str(Path(__file__)), "--tts-server"],
                                      env=local_env(), cwd=str(ROOT)))
    llm_proc = None
    try:
        if cfg.get("llm_backend", "llamacpp") == "llamacpp":
            llm_proc = start_llamacpp(cfg)
            procs.append(llm_proc)
        else:
            start_llm_server(cfg, lora)
        wait_http(f"http://127.0.0.1:{cfg['llm_port']}/health", "LLM-сервер", llm_proc)
        say("Прогреваю языковую модель…")
        warmup_llm(cfg)
        if use_qwen:
            say("Загружаю голос…")
            wait_http(f"http://127.0.0.1:{cfg['tts_port']}/health", "Сервер голоса", procs[0])
        say("Загружаю распознавание речи…")
        asyncio.run(bot_main(cfg, bridge))
    finally:
        for p in procs:
            p.terminate()


def start_llamacpp(cfg: dict):
    print(f"[LLM] запускаю llama.cpp: {llm_gguf(cfg).name}")
    log = open(CACHE / "tmp" / "llama-server.log", "w", encoding="utf-8")
    ctx = int(cfg.get("llm_ctx", 4096))
    # Eco: KV-кэш в 8 битах и батч 256 — ~0.8 ГБ меньше видеопамяти, промпт на ~7% медленнее; Fast: f16, батч 512
    kv = (["-ctk", "q8_0", "-ctv", "q8_0", "-ub", "256", "-b", "256"]
          if cfg.get("vram_mode", "eco") == "eco" else [])
    return subprocess.Popen([
        str(llama_exe("llama-server")), "-m", str(llm_gguf(cfg)), "--host", "127.0.0.1", "--port", str(cfg["llm_port"]),
        # 2 слота: разговор и быстрые служебные вопросы («какую реакцию вставить?») не вытесняют кэш друг друга
        "-ngl", "99", "-c", str(ctx * 2), "-np", "2",
        "-fa", "on", *kv,
        "--jinja", "--reasoning", "off",                       # без скрытых «размышлений»: сразу ответ
        "-n", str(cfg["max_new_tokens"]), "--temp", str(cfg["temperature"]), "--top-p", str(cfg["top_p"]),
        "--repeat-penalty", str(cfg["repetition_penalty"]),
    ], env=local_env(), cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT)


def ensure_piper():
    try:
        import piper  # noqa: F401
    except ImportError:
        print("\n=== Ставлю Piper (быстрый голос) в .venv ===")
        run([venv_python(VENV), "-m", "pip", "install", "pipecat-ai[piper]"])


def warmup_llm(cfg: dict):
    """Первый ответ видеокарты всегда медленный — делаем его заранее, до начала разговора."""
    import httpx
    t0 = time.time()
    with httpx.stream("POST", f"http://127.0.0.1:{cfg['llm_port']}/v1/chat/completions", timeout=300,
                      json={"model": "x", "stream": True,
                            "messages": [{"role": "user", "content": "Привет"}]}) as r:
        for _ in r.iter_lines():
            pass
    print(f"[LLM] прогрев: {time.time() - t0:.1f} c")


def _letters(s: str) -> str:
    """'Ммм, да-да!' -> 'мда': только буквы, повторы подряд схлопнуты."""
    import re
    return re.sub(r"(.)\1+", r"\1", re.sub(r"[^а-яё]", "", s.lower()))


def clip_ok(word: str, heard: str, dur: float) -> bool:
    """Клип годится, если он короткий и Whisper слышит в нём это слово («хм» ~ «мм», «ха-ха» ~ «ха-ха-ха»)."""
    w, h = _letters(word), _letters(heard)
    return bool(h) and dur <= 0.5 + 0.1 * len(w) and (h == w or (len(set(w)) <= 2 and set(h) <= set(w)))


def fetch_clips(cfg: dict, words: list, whisper=None, attempts: int = 8, keep: int = 3) -> list:
    """Озвучивает короткие реплики («ага», «хм») голосом бота; возвращает [(pcm int16, sr)].

    TTS часто произносит такие слова странно («хм» -> «ах, ах», «ну» -> «но») или тянет 2 секунды,
    поэтому каждый вариант проверяется Whisper'ом. Удачные клипы сохраняются в cache/clips/<голос>/
    и при следующих запусках берутся оттуда. Удали папку, чтобы озвучить заново.
    """
    import wave
    import httpx
    import numpy as np
    vol = float(cfg.get("backchannel_volume", 0.6))
    folder = CACHE / "clips" / (cfg.get("tts_speaker") or "default")
    folder.mkdir(parents=True, exist_ok=True)
    clips = []
    for w in words:
        saved = sorted(folder.glob(f"{w}_*.wav"))
        if not saved:
            for _ in range(attempts):
                if len(saved) >= keep:
                    break
                r = httpx.post(f"http://127.0.0.1:{cfg['tts_port']}/tts", json={"text": w}, timeout=60)
                sr = int(r.headers.get("X-Sample-Rate", "24000"))
                a = np.frombuffer(r.content, dtype=np.int16)
                loud = np.nonzero(np.abs(a) > max(600, np.abs(a).max() * 0.06))[0]
                if len(loud) == 0:
                    continue
                a = a[max(loud[0] - sr // 50, 0):loud[-1] + sr // 50]
                if whisper is not None:
                    a16 = np.interp(np.arange(0, len(a), sr / 16000), np.arange(len(a)), a / 32768).astype(np.float32)
                    heard = " ".join(s.text for s in whisper.transcribe(a16, language="ru")[0])
                    if not clip_ok(w, heard, len(a) / sr):
                        continue
                path = folder / f"{w}_{len(saved)}.wav"
                with wave.open(str(path), "wb") as f:
                    f.setnchannels(1); f.setsampwidth(2); f.setframerate(sr); f.writeframes(a.tobytes())
                saved.append(path)
        for path in saved:
            with wave.open(str(path), "rb") as f:
                a = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)
                clips.append(((a * vol).astype(np.int16).tobytes(), f.getframerate()))
    return clips


def build_processors(cfg: dict):
    """Всё, кроме микрофона и динамика: stt, user, backchannel, llm, tts, assistant."""
    import functools
    import numpy as np
    from loguru import logger
    from pipecat.audio.vad.silero import SileroVADAnalyzer
    from pipecat.processors.aggregators.llm_context import LLMContext
    from pipecat.processors.aggregators.llm_response_universal import (
        LLMContextAggregatorPair, LLMUserAggregatorParams)
    from pipecat.services.openai.llm import OpenAILLMService
    from pipecat.services.whisper.stt import WhisperSTTService
    from pipecat.transcriptions.language import Language
    from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
    from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
    from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy
    from pipecat.turns.user_turn_strategies import UserTurnStrategies
    sys.path.insert(0, str(ROOT))
    from duplex import Backchannel, QwenTTSService, SmartInterrupt, StreamingWhisperSTT

    stt_cls = StreamingWhisperSTT if cfg.get("stream_stt", True) else WhisperSTTService
    whisper_settings = WhisperSTTService.Settings(model=str(MODELS / "whisper"), language=Language.RU)
    try:
        if cfg["whisper_device"] != "cuda":
            raise RuntimeError("cpu по настройке")
        # int8_float16: на ~1.2 ГБ меньше видеопамяти, чем float16, при той же скорости и точности
        stt = stt_cls(settings=whisper_settings, device="cuda", compute_type="int8_float16")
    except Exception as e:
        logger.warning(f"Whisper на видеокарте не запустился ({e}), перехожу на процессор")
        stt = stt_cls(settings=whisper_settings, device="cpu", compute_type="int8")
    # Жадный поиск (−50 мс). Без перезапусков с повышенной температурой: на «ага» Whisper иногда пишет
    # «аааа…», и стандартный faster-whisper переспрашивает себя до 6 раз — модель занята секунды.
    stt._model.transcribe = functools.partial(
        stt._model.transcribe, beam_size=1, temperature=0.0, without_timestamps=True, max_new_tokens=96,
        condition_on_previous_text=False)

    t0 = time.time()
    try:
        segs, _ = stt._model.transcribe(np.zeros(16000, dtype=np.float32), language="ru")
        list(segs)
        print(f"[Whisper] прогрев: {time.time() - t0:.1f} c")
    except Exception as e:
        logger.warning(f"Прогрев Whisper не удался: {e}")

    llm = OpenAILLMService(api_key="local", base_url=f"http://127.0.0.1:{cfg['llm_port']}/v1",
                           settings=OpenAILLMService.Settings(model="local"))
    qwen = cfg.get("tts_engine", "qwen") == "qwen"
    if qwen:
        tts = QwenTTSService(url=f"http://127.0.0.1:{cfg['tts_port']}/tts",
                             expressive=bool(cfg.get("expressive_tts", True)))
    else:
        from pipecat.services.piper.tts import PiperTTSService
        tts = PiperTTSService(settings=PiperTTSService.Settings(voice=cfg["piper_voice"]),
                              download_dir=MODELS / "piper")

    prompt = cfg["system_prompt"]
    if cfg.get("filler") and qwen:
        prompt += " Не начинай ответ с междометий вроде «хм», «так», «ну»."
    if cfg.get("voice_emotion", True) and isinstance(stt, StreamingWhisperSTT):
        prompt += (" Иногда в конце реплики собеседника в квадратных скобках написано, как звучал его голос,"
                   " например [голос: тихо, медленно] — учитывай это, но сам квадратные скобки не пиши.")
    if cfg.get("expressive_tts", True) and qwen:
        prompt += " Передавай эмоции словами и знаками: «!», «...», «ох», «ха»."
    context = LLMContext(messages=[{"role": "system", "content": prompt}])
    # Silero VAD слышит речь, Smart Turn v3 решает, что ты договорил. Пока бот молчит — ход начинается
    # с первого слова; пока бот говорит — только с interrupt_min_words слов (или «стоп»), иначе он не замолкает.
    aggregators = LLMContextAggregatorPair(
        context, user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(),
            user_turn_strategies=UserTurnStrategies(
                start=[SmartInterrupt(min_words=int(cfg.get("interrupt_min_words", 2)))],
                stop=[TurnAnalyzerUserTurnStopStrategy(turn_analyzer=LocalSmartTurnAnalyzerV3(
                    params=SmartTurnParams(stop_secs=float(cfg.get("turn_max_silence", 2.0)))))]),
            empty_user_turn=None))   # не просить «повтори», если Whisper ничего не разобрал
    bc = Backchannel(cfg, stt if isinstance(stt, StreamingWhisperSTT) else None)
    return stt, aggregators.user(), bc, llm, tts, aggregators.assistant()


def load_clips(cfg: dict, bc, whisper):
    """Озвученные заранее «ага/ого/эх/ха-ха» по категориям и заглушки «хм/так»."""
    if cfg.get("tts_engine", "qwen") != "qwen":
        return
    t0 = time.time()
    if cfg.get("backchannel"):
        for kind, words in (cfg.get("backchannel_clips") or {}).items():
            bc.set_clips(kind, fetch_clips(cfg, words, whisper))
    if cfg.get("filler"):
        bc.set_clips("fill", fetch_clips(cfg, cfg.get("filler_words") or [], whisper))
    print("[Голос] клипы: " + ", ".join(f"{k} {len(v)}" for k, v in bc._src.items())
          + f" ({time.time() - t0:.1f} c)")


async def bot_main(cfg: dict, bridge=None):
    import asyncio
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.runner import PipelineRunner
    from pipecat.pipeline.task import PipelineParams, PipelineTask
    from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams

    stt, user, bc, llm, tts, assistant = build_processors(cfg)
    if bridge:
        bridge.status("Готовлю короткие реплики («ага», «ничего себе»)…")
    load_clips(cfg, bc, stt._model)

    transport = LocalAudioTransport(LocalAudioTransportParams(audio_in_enabled=True, audio_out_enabled=True))
    if bridge:   # окно: микрофон открывается кнопкой, слова и громкость уходят в окно
        from duplex import MicGate, UIEvents
        bridge.pipeline_loop = asyncio.get_running_loop()
        steps = [transport.input(), MicGate(bridge), stt, user, bc, llm, tts, UIEvents(bridge),
                 transport.output(), assistant]
    else:
        steps = [transport.input(), stt, user, bc, llm, tts, transport.output(), assistant]
    pipeline = Pipeline(steps)
    # idle_timeout_secs=None: не завершать разговор после 5 минут тишины
    task = PipelineTask(pipeline, params=PipelineParams(enable_metrics=True), idle_timeout_secs=None)
    if bridge:
        bridge.status("Готово", ready=True)

    print("\n" + "=" * 60)
    print("  Готово! Говори в микрофон. Лучше в НАУШНИКАХ —")
    print("  иначе бот услышит сам себя и будет себя перебивать.")
    print("  Выход: Ctrl+C")
    print("=" * 60 + "\n")
    await PipelineRunner(handle_sigint=True).run(task)


# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    if "--tts-server" in sys.argv:
        tts_server()
    elif "--run" in sys.argv:
        run_bot()
    else:
        setup_and_launch()
