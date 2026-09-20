import os
import re
import json
import glob
import time
import shutil
import tempfile
import subprocess
from concurrent.futures import ThreadPoolExecutor

import requests
from requests.adapters import HTTPAdapter
from flask import Flask, request, jsonify, Response

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_STT_URL = "https://api.groq.com/openai/v1/audio/transcriptions"

DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
DEFAULT_WHISPER_MODEL = "whisper-large-v3-turbo"
WHISPER_MODELS = {"whisper-large-v3-turbo", "whisper-large-v3"}

GROQ_FILE_LIMIT = 24 * 1024 * 1024   # Groq upload limit is ~25MB (free tier)
SEGMENT_SECONDS = 600                # audio is cut into 10-minute pieces (needs ffmpeg)
STT_PARALLEL = 3                     # pieces transcribed at the same time
MAX_ITEMS_PER_REQUEST = 200
MAX_TEXT_LEN = 2000
MAX_GLOSSARY_LEN = 2000

FFMPEG = shutil.which("ffmpeg")

# One shared session = connections are reused (faster for many small requests)
SESSION = requests.Session()
SESSION.mount("https://", HTTPAdapter(pool_connections=8, pool_maxsize=16))

TONE_DESCRIPTIONS = {
    'natural': 'natural spoken conversational style suitable for movie/drama subtitles (သဘာဝကျကျ စကားပြောဟန်)',
    'formal': 'polite, elegant literary/formal style suitable for documentaries or official media (ယဉ်ကျေးသပ်ရပ်သော စာဟန်ပေဟန်)',
    'explaining': 'clear, easy-to-understand instructional/educational style as if explaining to an audience (နားလည်လွယ်အောင် သေချာရှင်းပြသလိုဟန်)',
    'casual': 'relaxed, trendy youthful casual style with modern slangs (ပေါ့ပေါ့ပါးပါး လူငယ်သုံး စကားပြောဟန်)',
}

MODEL_NAME_RE = re.compile(r'^[A-Za-z0-9._\-]{1,64}$')
LANG_RE = re.compile(r'^[A-Za-z ()\-]{2,30}$')


# ---------------------------------------------------------------------------
# Errors
#   kind = auth   -> key/permission problem  (client stops everything)
#          bad    -> bad request/model name  (client stops everything)
#          rate   -> 429                     (client waits, then retries)
#          server -> 5xx / timeout / network (client retries)
#          blocked-> empty / blocked answer  (client retries, then skips chunk)
# ---------------------------------------------------------------------------
class ApiError(Exception):
    def __init__(self, message, kind="server", status=502, retry_after=None):
        super().__init__(message)
        self.message = message
        self.kind = kind
        self.status = status
        self.retry_after = retry_after


def error_response(err):
    body = {"error": err.message, "kind": err.kind}
    if err.retry_after:
        body["retryAfter"] = err.retry_after
    return jsonify(body), err.status


def extract_retry_after(res, err_obj):
    header = res.headers.get("retry-after")
    if header:
        try:
            return min(float(header), 90.0)
        except ValueError:
            pass
    details = err_obj.get("details") if isinstance(err_obj, dict) else None
    for d in details or []:
        if isinstance(d, dict) and d.get("retryDelay"):
            m = re.match(r"([\d.]+)s", str(d["retryDelay"]))
            if m:
                return min(float(m.group(1)), 90.0)
    return None


def upstream_error(res, provider):
    try:
        data = res.json()
    except ValueError:
        data = {}
    err = data.get("error", {}) if isinstance(data, dict) else {}
    if not isinstance(err, dict):
        err = {"message": str(err)}
    message = err.get("message") or f"{provider} API Error ({res.status_code})"
    code = res.status_code
    retry_after = extract_retry_after(res, err)
    if code == 429:
        return ApiError(message, "rate", 429, retry_after)
    if code in (401, 403):
        return ApiError(message, "auth", 401)
    if code >= 500:
        return ApiError(message, "server", 502)
    return ApiError(message, "bad", 400)


def post_json(url, provider, **kwargs):
    """requests.post with our own (key-free) error messages."""
    try:
        return SESSION.post(url, timeout=(10, 180), **kwargs)
    except requests.Timeout:
        raise ApiError(f"{provider} က အချိန်ကြာလွန်းလို့ မပြန်လာပါ (timeout)", "server", 504)
    except requests.RequestException:
        raise ApiError(f"{provider} နှင့် ချိတ်ဆက်မရပါ", "server", 502)


# ---------------------------------------------------------------------------
# Translation helpers
# ---------------------------------------------------------------------------
def build_system_prompt(target_lang, tone_style, glossary):
    tone = TONE_DESCRIPTIONS.get(tone_style, TONE_DESCRIPTIONS['natural'])
    prompt = (
        "You are a professional audiovisual subtitle translator. "
        f"Translate the subtitles into {target_lang}. "
        f"Translation Style Instruction: {tone}. "
        "Maintain emotional nuances, natural flow, and match subtitle reading speed. "
        "Return exactly one translation for every id in 'translate'. "
        "Never merge, split, skip or reorder subtitles. "
        "Keep tags such as <i></i> and line breaks where they make sense. "
        "'context_before' and 'context_after' are only for understanding the scene: "
        "never translate them and never return them. "
        "Output STRICTLY valid JSON in this shape: "
        '{"translations": [{"id": 1, "translatedText": "မင်္ဂလာပါ ခင်ဗျာ"}]}'
    )
    if glossary:
        prompt += (
            "\nGlossary - always use exactly these translations for these names/terms:\n"
            + glossary
        )
    return prompt


def clean_items(items, limit):
    out = []
    for s in (items or [])[:limit]:
        if not isinstance(s, dict):
            continue
        try:
            sid = int(s.get("id"))
        except (TypeError, ValueError):
            continue
        out.append({"id": sid, "text": str(s.get("originalText", ""))[:MAX_TEXT_LEN]})
    return out


_ITEM_RE = re.compile(
    r'\{\s*"id"\s*:\s*(\d+)\s*,\s*"translatedText"\s*:\s*("(?:[^"\\]|\\.)*")\s*\}', re.S
)


def parse_translations(raw):
    """Turn the model's answer into [{id, translatedText}], even if it is a bit broken."""
    text = (raw or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()

    data = None
    try:
        data = json.loads(text)
    except ValueError:
        pass

    items = None
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        for key in ("translations", "subtitles", "items", "result"):
            if isinstance(data.get(key), list):
                items = data[key]
                break

    result = []
    for it in items or []:
        if not isinstance(it, dict) or "id" not in it:
            continue
        try:
            rid = int(it["id"])
        except (TypeError, ValueError):
            continue
        t = it.get("translatedText", it.get("text"))
        if isinstance(t, str):
            result.append({"id": rid, "translatedText": t.strip()})

    if not result:  # salvage what we can from truncated / malformed JSON
        for m in _ITEM_RE.finditer(text):
            try:
                result.append({"id": int(m.group(1)), "translatedText": json.loads(m.group(2)).strip()})
            except ValueError:
                continue
    return result


def thinking_config_for(model):
    """Translation needs no deep reasoning: switch thinking off/minimal for speed."""
    m = model.lower()
    if m.startswith("gemini-3") and "flash" in m:
        return {"thinkingLevel": "minimal"}
    if m.startswith("gemini-2.5") and "flash" in m:
        return {"thinkingBudget": 0}
    return None


GEMINI_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "translations": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "id": {"type": "INTEGER"},
                    "translatedText": {"type": "STRING"},
                },
                "required": ["id", "translatedText"],
                "propertyOrdering": ["id", "translatedText"],
            },
        }
    },
    "required": ["translations"],
}


def extract_gemini_text(data):
    cands = data.get("candidates") or []
    if not cands:
        reason = (data.get("promptFeedback") or {}).get("blockReason", "empty response")
        raise ApiError(f"Gemini က ဘာသာပြန်မပေးပါ ({reason})", "blocked", 422)
    parts = (cands[0].get("content") or {}).get("parts") or []
    text = "".join(
        p.get("text", "") for p in parts if isinstance(p, dict) and not p.get("thought")
    )
    if not text.strip():
        raise ApiError(f"Gemini က အဖြေအလွတ် ပြန်ပို့ပါသည် ({cands[0].get('finishReason', '')})", "blocked", 422)
    return text


def call_gemini(api_key, model, system_prompt, user_json):
    url = f"{GEMINI_BASE}/{model}:generateContent"
    headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}  # key in header, not URL

    full_cfg = {
        "responseMimeType": "application/json",
        "responseSchema": GEMINI_SCHEMA,
        "temperature": 0.25,
    }
    think = thinking_config_for(model)
    if think:
        full_cfg["thinkingConfig"] = think
    plain_cfg = {"responseMimeType": "application/json", "temperature": 0.25}

    last_err = None
    for cfg in (full_cfg, plain_cfg):
        body = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_json}]}],
            "generationConfig": cfg,
        }
        res = post_json(url, "Gemini", headers=headers, json=body)
        if res.status_code == 200:
            return extract_gemini_text(res.json())
        last_err = upstream_error(res, "Gemini")
        # 400 may just mean "this model doesn't support schema/thinking option" -> retry once, plain
        if res.status_code == 400 and cfg is full_cfg and "api key" not in last_err.message.lower():
            continue
        break
    raise last_err


def call_groq(api_key, model, system_prompt, user_json):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_json},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.25,
    }
    res = post_json(GROQ_CHAT_URL, "Groq", headers=headers, json=body)
    if res.status_code != 200:
        raise upstream_error(res, "Groq")
    try:
        return res.json()["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError, ValueError):
        raise ApiError("Groq က အဖြေအလွတ် ပြန်ပို့ပါသည်", "blocked", 422)


@app.route('/api/translate', methods=['POST'])
def translate():
    req = request.get_json(silent=True) or {}
    provider = req.get('provider', 'gemini')
    api_key = (req.get('apiKey') or '').strip()

    if provider not in ('gemini', 'groq'):
        return error_response(ApiError("Provider မှားနေပါသည်", "bad", 400))
    if not api_key:
        return error_response(ApiError(
            f"{provider.upper()} API Key မရှိသေးပါ။ Menu -> Settings ထဲတွင် ထည့်သွင်းပေးပါ", "auth", 400))

    default_model = DEFAULT_GEMINI_MODEL if provider == 'gemini' else DEFAULT_GROQ_MODEL
    model = (req.get('modelName') or default_model).strip()
    if not MODEL_NAME_RE.match(model):
        return error_response(ApiError("Model name မမှန်ပါ", "bad", 400))

    items = clean_items(req.get('subtitles'), MAX_ITEMS_PER_REQUEST)
    if not items:
        return error_response(ApiError("ဘာသာပြန်ရန် Subtitle မပါပါ", "bad", 400))

    target_lang = req.get('targetLang', 'Burmese')
    if not isinstance(target_lang, str) or not LANG_RE.match(target_lang):
        target_lang = 'Burmese'
    glossary = str(req.get('glossary') or '').strip()[:MAX_GLOSSARY_LEN]

    system_prompt = build_system_prompt(target_lang, req.get('toneStyle', 'natural'), glossary)
    payload = {"translate": items}
    before = clean_items(req.get('contextBefore'), 10)
    after = clean_items(req.get('contextAfter'), 10)
    if before:
        payload["context_before"] = before
    if after:
        payload["context_after"] = after
    user_json = json.dumps(payload, ensure_ascii=False)

    try:
        if provider == 'gemini':
            raw = call_gemini(api_key, model, system_prompt, user_json)
        else:
            raw = call_groq(api_key, model, system_prompt, user_json)

        wanted = {i["id"] for i in items}
        translations = [t for t in parse_translations(raw) if t["id"] in wanted]
        if not translations:
            raise ApiError("Model ပြန်ပေးတဲ့ အဖြေကို ဖတ်မရပါ", "server", 502)
        return jsonify({"translations": translations})
    except ApiError as e:
        return error_response(e)
    except Exception as e:
        return jsonify({"error": str(e).replace(api_key, "***"), "kind": "server"}), 500


# ---------------------------------------------------------------------------
# Transcription (Groq Whisper)
# ---------------------------------------------------------------------------
def fmt_ts(sec):
    ms = int(round(max(sec, 0) * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def groq_stt(path, mimetype, api_key, model):
    headers = {"Authorization": f"Bearer {api_key}"}
    for attempt in range(3):
        with open(path, 'rb') as f:
            try:
                res = SESSION.post(
                    GROQ_STT_URL,
                    headers=headers,
                    files={'file': (os.path.basename(path), f, mimetype)},
                    data={'model': model, 'response_format': 'verbose_json'},
                    timeout=(10, 300),
                )
            except requests.Timeout:
                raise ApiError("Groq က အချိန်ကြာလွန်းလို့ မပြန်လာပါ (timeout)", "server", 504)
            except requests.RequestException:
                raise ApiError("Groq နှင့် ချိတ်ဆက်မရပါ", "server", 502)
        if res.status_code == 200:
            return res.json()
        err = upstream_error(res, "Groq")
        if err.kind in ("rate", "server") and attempt < 2:
            time.sleep(min(err.retry_after or 5 * (attempt + 1), 30))
            continue
        raise err


def extract_audio_segments(src, workdir):
    """Video/audio -> small mono 16kHz mp3 pieces (uses ffmpeg)."""
    pattern = os.path.join(workdir, "seg_%04d.mp3")
    cmd = [
        FFMPEG, "-nostdin", "-y", "-loglevel", "error",
        "-i", src, "-vn", "-map", "0:a:0?",
        "-ac", "1", "-ar", "16000", "-c:a", "libmp3lame", "-b:a", "32k",
        "-f", "segment", "-segment_time", str(SEGMENT_SECONDS), "-reset_timestamps", "1",
        pattern,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    except subprocess.TimeoutExpired:
        raise ApiError("ဖိုင်ကို အသံပြောင်းတာ အချိန်ကြာလွန်းပါသည်", "bad", 400)
    files = sorted(glob.glob(os.path.join(workdir, "seg_*.mp3")))
    if proc.returncode != 0 or not files:
        tail = (proc.stderr or "").strip().splitlines()[-1:] or ["no audio stream?"]
        raise ApiError(f"ဖိုင်ထဲက အသံကို ထုတ်မရပါ (ffmpeg: {tail[0][:200]})", "bad", 400)
    return [(p, i * SEGMENT_SECONDS) for i, p in enumerate(files)]


def transcribe_part(job):
    path, offset, mimetype, api_key, model = job
    data = groq_stt(path, mimetype, api_key, model)
    out = []
    for seg in data.get('segments') or []:
        text = (seg.get('text') or '').strip()
        if not text:
            continue
        # Whisper's own "this is silence/noise" rule -> avoids hallucinated lines
        if seg.get('no_speech_prob', 0) > 0.6 and seg.get('avg_logprob', 0) < -1.0:
            continue
        out.append((seg['start'] + offset, seg['end'] + offset, text))
    return out


@app.route('/api/transcribe', methods=['POST'])
def transcribe():
    file = request.files.get('file')
    api_key = (request.form.get('apiKey') or '').strip()
    model = request.form.get('model') or DEFAULT_WHISPER_MODEL
    if model not in WHISPER_MODELS:
        model = DEFAULT_WHISPER_MODEL
    if not file or not api_key:
        return error_response(ApiError("Video/Audio transcribe လုပ်ရန် Groq API Key လိုအပ်ပါသည်", "auth", 400))

    try:
        with tempfile.TemporaryDirectory() as tmp:
            ext = os.path.splitext(file.filename or '')[1].lower()
            ext = re.sub(r'[^a-z0-9.]', '', ext)[:8]
            src = os.path.join(tmp, "input" + ext)
            file.save(src)  # streamed to disk, not held in RAM
            size = os.path.getsize(src)

            if FFMPEG:
                parts = extract_audio_segments(src, tmp)
                mimetype = "audio/mpeg"
            elif size <= GROQ_FILE_LIMIT:
                parts = [(src, 0)]
                mimetype = file.content_type or "application/octet-stream"
            else:
                mb = size // (1024 * 1024)
                raise ApiError(
                    f"ဖိုင်ကြီးလွန်းပါသည် ({mb}MB)။ Groq ကန့်သတ်ချက်က ~25MB ဖြစ်ပြီး server မှာ ffmpeg မရှိလို့ "
                    "အလိုအလျောက် မချုံ့နိုင်ပါ။ အသံကို mp3 အဖြစ် ချုံ့ပြီးတင်ပါ (သို့) server မှာ ffmpeg သွင်းပါ။",
                    "bad", 400)

            jobs = [(p, off, mimetype, api_key, model) for p, off in parts]
            with ThreadPoolExecutor(max_workers=STT_PARALLEL) as pool:
                results = list(pool.map(transcribe_part, jobs))  # keeps order

        items = []
        for seg_list in results:
            for start, end, text in seg_list:
                items.append({
                    "id": len(items) + 1,
                    "startTime": fmt_ts(start),
                    "endTime": fmt_ts(end),
                    "originalText": text,
                    "translatedText": "",
                })
        if not items:
            raise ApiError("အသံထဲမှာ စကားပြောသံ မတွေ့ပါ", "bad", 400)
        return jsonify({"subtitles": items})
    except ApiError as e:
        return error_response(e)
    except Exception as e:
        return jsonify({"error": str(e).replace(api_key, "***"), "kind": "server"}), 500


# ---------------------------------------------------------------------------
# Web page
# ---------------------------------------------------------------------------
HTML_PAGE = r"""<!DOCTYPE html>
<html lang="my" class="h-full">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Thiri's Koko — AI Subtitle Studio</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=Padauk:wght@400;700&display=swap" rel="stylesheet">
  <style>
    body {
      background: radial-gradient(circle at 50% 0%, #290d1f 0%, #11050c 100%);
      font-family: 'Plus Jakarta Sans', 'Padauk', sans-serif;
      -webkit-tap-highlight-color: transparent;
    }
    .romantic-card {
      background: rgba(36, 14, 26, 0.78);
      backdrop-filter: blur(14px);
      -webkit-backdrop-filter: blur(14px);
      border: 1px solid rgba(243, 146, 189, 0.18);
    }
    /* Subtitle cards: no blur (hundreds of blurred cards make phones lag) + skip off-screen rendering */
    .sub-card {
      background: rgba(36, 14, 26, 0.85);
      content-visibility: auto;
      contain-intrinsic-size: auto 150px;
    }
    .romantic-glow {
      box-shadow: 0 4px 20px -2px rgba(219, 39, 119, 0.35);
    }
    .btn-stop {
      background-image: linear-gradient(to right, #9f1239, #7f1d1d) !important;
    }
    .drawer-transition {
      transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    /* 16px fields on touch screens stop iOS from zooming in when a field is focused */
    @media (pointer: coarse) {
      input[type="text"], input[type="password"], input[type="number"], textarea, select {
        font-size: 16px !important;
      }
    }
  </style>
</head>
<body class="text-rose-100 min-h-full flex flex-col selection:bg-rose-500 selection:text-white pb-36">

  <!-- Header -->
  <header class="sticky top-0 z-40 romantic-card border-b border-rose-900/40 px-4 py-3 flex items-center justify-between">
    <div class="flex items-center gap-2.5">
      <div class="w-8 h-8 rounded-xl bg-gradient-to-tr from-rose-600 to-pink-500 flex items-center justify-center text-white font-bold shadow-md shadow-rose-900/40">
        ❤
      </div>
      <div>
        <h1 class="text-sm font-bold text-white tracking-tight">Thiri's Koko</h1>
        <p class="text-[10px] text-rose-300/80">AI Subtitle Studio</p>
      </div>
    </div>

    <!-- 3-Line Hamburger Menu -->
    <button onclick="toggleMenu(true)" class="p-2 rounded-xl bg-rose-900/40 border border-rose-800/50 text-rose-200 hover:text-white hover:bg-rose-800/60 transition active:scale-95" aria-label="Menu">
      <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6h16M4 12h16M4 18h16"/>
      </svg>
    </button>
  </header>

  <!-- Status Notification Pill (with progress bar) -->
  <div id="statusPill" class="hidden mx-4 mt-2.5 p-2.5 rounded-xl text-xs romantic-card border border-rose-500/40 text-rose-200 flex flex-col items-stretch gap-2 shadow-sm">
    <div class="flex items-center justify-center gap-2">
      <div class="w-2 h-2 rounded-full bg-rose-400 animate-ping"></div>
      <span id="statusText">Processing...</span>
    </div>
    <div id="progressWrap" class="hidden w-full h-1.5 rounded-full bg-rose-950 overflow-hidden">
      <div id="progressBar" class="h-full rounded-full bg-gradient-to-r from-rose-500 to-pink-400 transition-all duration-300" style="width:0%"></div>
    </div>
  </div>

  <!-- Main Container -->
  <main class="flex-1 px-4 py-3 max-w-3xl mx-auto w-full flex flex-col gap-3">

    <!-- 1. Upload Section -->
    <div class="romantic-card rounded-2xl p-4 border border-rose-800/40 flex flex-col gap-3">
      <div class="flex items-center justify-between">
        <div class="flex items-center gap-2.5">
          <div class="w-9 h-9 rounded-xl bg-rose-500/10 text-rose-300 flex items-center justify-center">
            <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"/>
            </svg>
          </div>
          <div>
            <h2 class="text-xs font-semibold text-rose-100">Upload SRT / Video / Audio</h2>
            <p class="text-[10px] text-rose-400">Total Loaded: <span id="subCount">0</span> subtitles</p>
          </div>
        </div>

        <input type="file" id="fileInput" accept=".srt,video/*,audio/*" class="hidden"/>
        <label for="fileInput" class="cursor-pointer bg-gradient-to-r from-rose-600 to-pink-500 hover:from-rose-500 hover:to-pink-400 text-white text-xs font-semibold py-2 px-3.5 rounded-xl shadow transition active:scale-95">
          Choose File
        </label>
      </div>

      <!-- 2. Target Language & Tone Style Options -->
      <div class="pt-3 border-t border-rose-900/60 grid grid-cols-1 md:grid-cols-2 gap-2.5">
        <div>
          <label class="block text-[11px] font-medium text-rose-300 mb-1">Target Language (ပြောင်းမည့် ဘာသာစကား)</label>
          <select id="targetLang" onchange="handleLangChange()" class="w-full bg-rose-950/90 border border-rose-800/60 text-xs text-rose-100 rounded-xl p-2 focus:outline-none focus:border-rose-400">
            <option value="Burmese">မြန်မာစာ (Burmese)</option>
            <option value="English">English</option>
            <option value="Thai">ภาษาไทย (Thai)</option>
            <option value="Japanese">日本語 (Japanese)</option>
            <option value="Chinese">中文 (Chinese)</option>
          </select>
        </div>

        <div id="toneContainer">
          <label class="block text-[11px] font-medium text-rose-300 mb-1">စကားပြောပုံစံ / အသုံးအနှုန်းဟန်</label>
          <select id="toneStyle" class="w-full bg-rose-950/90 border border-rose-800/60 text-xs text-rose-100 rounded-xl p-2 focus:outline-none focus:border-rose-400">
            <option value="natural">🗣️ သဘာဝကျ စကားပြောဟန် (ရုပ်ရှင်/ဇာတ်လမ်း)</option>
            <option value="formal">📖 စာဟန်ပေဟန် (ယဉ်ကျေး/မှတ်တမ်းတင်)</option>
            <option value="explaining">🎓 ရှင်းပြသလိုဟန် (သေချာနားလည်လွယ်အောင်)</option>
            <option value="casual">🎭 ပေါ့ပေါ့ပါးပါး လူငယ်သုံး (Modern Slangs)</option>
          </select>
        </div>
      </div>

      <!-- Glossary + Overwrite -->
      <div class="pt-3 border-t border-rose-900/60 flex flex-col gap-2">
        <div>
          <label class="block text-[11px] font-medium text-rose-300 mb-1">Glossary — နာမည်/စကားလုံး ဘာသာပြန်ပုံ (မထည့်လည်းရသည်)</label>
          <textarea id="glossary" rows="2" oninput="saveGlossary()"
            placeholder="Tony = တိုနီ&#10;Seoul = ဆိုးလ်"
            class="w-full bg-rose-950/90 border border-rose-800/60 focus:border-rose-400 rounded-xl p-2 text-xs text-rose-50 placeholder-rose-400/40 focus:outline-none resize-none leading-relaxed"></textarea>
        </div>
        <label class="flex items-center gap-2 text-[11px] text-rose-300 cursor-pointer select-none">
          <input type="checkbox" id="overwrite" class="accent-rose-500">
          <span>ဘာသာပြန်ပြီးသားတွေကိုပါ အစကပြန်ပြန်မည် (Overwrite)</span>
        </label>
      </div>
    </div>

    <!-- Subtitle Cards List -->
    <div id="subList" class="space-y-2.5">
      <div class="romantic-card rounded-2xl p-10 text-center border border-rose-900/40 text-rose-300/80 my-2">
        <div class="w-12 h-12 rounded-full bg-rose-500/10 text-rose-400 mx-auto flex items-center justify-center mb-3">
          <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/>
          </svg>
        </div>
        <h3 class="text-sm font-semibold text-rose-100 mb-1">No Subtitles Loaded</h3>
        <p class="text-xs text-rose-400/80 max-w-xs mx-auto mb-3">
          ဗီဒီယို/အသံဖိုင် သို့မဟုတ် SRT ဖိုင်ကို Choose File နှိပ်၍ တင်ပေးပါ
        </p>
      </div>
    </div>
  </main>

  <!-- Sticky Bottom Controls: Engine Switcher & Export Actions -->
  <footer class="fixed bottom-0 left-0 right-0 z-30 romantic-card border-t border-rose-900/40 p-3 flex flex-col gap-2 max-w-lg mx-auto md:max-w-xl">

    <!-- Translation Engine Switcher with Gemini 3.5 Flash -->
    <div class="flex items-center justify-between text-xs px-1">
      <span class="text-[11px] text-rose-300 font-medium">Translate with:</span>
      <div class="flex items-center gap-2">
        <label class="flex items-center gap-1.5 cursor-pointer">
          <input type="radio" name="transEngine" value="gemini" checked onchange="saveActiveEngine('gemini')" class="accent-rose-500">
          <span class="text-xs text-indigo-300 font-semibold">Gemini 3.5 Flash</span>
        </label>
        <span class="text-rose-700">|</span>
        <label class="flex items-center gap-1.5 cursor-pointer">
          <input type="radio" name="transEngine" value="groq" onchange="saveActiveEngine('groq')" class="accent-rose-500">
          <span class="text-xs text-pink-300 font-semibold">Groq (Llama 3.3)</span>
        </label>
      </div>
    </div>

    <!-- Buttons Row -->
    <div class="flex items-center gap-2">
      <button onclick="translateAll()" id="btnTranslate" class="flex-1 bg-gradient-to-r from-rose-600 to-pink-500 hover:from-rose-500 hover:to-pink-400 text-white text-xs font-semibold py-3 px-3 rounded-xl flex items-center justify-center gap-1.5 transition romantic-glow active:scale-[0.98]">
        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 5h12M9 3v2m1.048 9.5A18.022 18.022 0 016.412 9m6.088 9h7M11 21l5-10 5 10M12.751 5C11.783 10.77 8.07 15.61 3 18.129"/>
        </svg>
        <span id="btnTranslateLabel">Translate All</span>
      </button>

      <button onclick="downloadOriginalSRT()" title="Download Original SRT" class="bg-rose-950/90 hover:bg-rose-900 text-rose-200 border border-rose-800/60 text-xs font-medium py-3 px-3 rounded-xl flex items-center gap-1 transition active:scale-[0.98]">
        <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"/>
        </svg>
        <span>Original .SRT</span>
      </button>

      <button onclick="downloadTranslatedSRT()" title="Download Translated SRT" class="bg-rose-900/80 hover:bg-rose-800 text-white border border-rose-700/60 text-xs font-medium py-3 px-3 rounded-xl flex items-center gap-1 transition active:scale-[0.98]">
        <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"/>
        </svg>
        <span>Translated .SRT</span>
      </button>
    </div>
  </footer>

  <!-- ================= ROMANTIC THEMED ALERT MODAL ================= -->
  <div id="customAlertModal" class="fixed inset-0 bg-black/70 backdrop-blur-md z-50 hidden flex items-center justify-center p-4">
    <div class="bg-rose-950 border border-rose-700/70 rounded-2xl w-full max-w-sm p-5 shadow-2xl space-y-4 romantic-card text-center">
      <div class="w-12 h-12 rounded-full bg-rose-500/20 text-rose-400 mx-auto flex items-center justify-center text-xl shadow">
        ✨
      </div>
      <div>
        <h3 id="customAlertTitle" class="text-sm font-bold text-white mb-1.5">သတိပေးချက်</h3>
        <p id="customAlertMessage" class="text-xs text-rose-200/90 leading-relaxed whitespace-pre-line"></p>
      </div>
      <button onclick="closeCustomAlert()" class="w-full bg-gradient-to-r from-rose-600 to-pink-500 hover:from-rose-500 text-white font-semibold py-2.5 rounded-xl text-xs shadow transition active:scale-95">
        နားလည်ပါပြီ
      </button>
    </div>
  </div>

  <!-- ================= RIGHT DRAWER MENU ================= -->
  <div id="menuOverlay" onclick="toggleMenu(false)" class="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 hidden transition-opacity"></div>
  <aside id="menuDrawer" class="fixed top-0 right-0 bottom-0 w-80 max-w-[85vw] bg-rose-950 border-l border-rose-800/60 z-50 transform translate-x-full drawer-transition flex flex-col p-5 shadow-2xl">
    <div class="flex items-center justify-between border-b border-rose-900/60 pb-4 mb-4">
      <div class="flex items-center gap-2">
        <span class="text-rose-400 font-bold text-lg">☰</span>
        <h2 class="text-sm font-bold text-white tracking-wide">Menu</h2>
      </div>
      <button onclick="toggleMenu(false)" class="p-1 rounded-lg text-rose-400 hover:text-white">✕</button>
    </div>

    <div class="flex-1 overflow-y-auto space-y-4 text-xs">
      <button onclick="toggleMenu(false); openSettingsModal();" class="w-full bg-rose-900/50 hover:bg-rose-900/80 border border-rose-800/60 rounded-xl p-3 text-left flex items-center justify-between text-rose-100 transition">
        <div class="flex items-center gap-2.5">
          <span class="text-base">⚙</span>
          <div>
            <div class="font-semibold text-white">Settings</div>
            <div class="text-[10px] text-rose-300/70">Groq & Gemini API Keys ထည့်ရန်</div>
          </div>
        </div>
        <span class="text-rose-400 font-bold">➔</span>
      </button>

      <button onclick="clearProject()" class="w-full bg-rose-900/30 hover:bg-rose-900/60 border border-rose-800/40 rounded-xl p-3 text-left flex items-center justify-between text-rose-100 transition">
        <div class="flex items-center gap-2.5">
          <span class="text-base">🗑</span>
          <div>
            <div class="font-semibold text-white">Clear saved project</div>
            <div class="text-[10px] text-rose-300/70">သိမ်းထားတဲ့ subtitle နဲ့ ဘာသာပြန်ချက်တွေ ဖျက်ရန်</div>
          </div>
        </div>
      </button>

      <!-- API Key Guide -->
      <div class="romantic-card rounded-2xl p-4 border border-rose-900/60 space-y-3">
        <div class="flex items-center gap-2 text-rose-300 font-semibold border-b border-rose-900/60 pb-2">
          <span>🔑</span>
          <span>API Key ယူနည်း (အခမဲ့)</span>
        </div>

        <div class="space-y-1.5">
          <a href="https://console.groq.com/keys" target="_blank" class="block font-bold text-pink-300 hover:underline flex items-center justify-between bg-rose-900/30 p-2 rounded-lg border border-rose-800/40">
            <span>👉 Groq API Key ယူရန်နှိပ်ပါ</span>
            <span class="text-[10px] bg-pink-500/20 text-pink-200 px-1.5 py-0.5 rounded">Console</span>
          </a>
          <p class="text-[11px] text-rose-200/80 leading-relaxed pl-1">
            <strong>ယူနည်း:</strong> Google Account ဖြင့် ဝင်၍ <strong>"Create API Key"</strong> နှိပ်ယူပါ။ (အသံဖိုင် စာတန်းထိုးထုတ်ရန် လိုအပ်ပါသည်)
          </p>
        </div>

        <div class="border-t border-rose-900/40 pt-2 space-y-1.5">
          <a href="https://aistudio.google.com/app/apikey" target="_blank" class="block font-bold text-indigo-300 hover:underline flex items-center justify-between bg-indigo-950/40 p-2 rounded-lg border border-indigo-800/40">
            <span>👉 Gemini API Key ယူရန်နှိပ်ပါ</span>
            <span class="text-[10px] bg-indigo-500/20 text-indigo-200 px-1.5 py-0.5 rounded">AI Studio</span>
          </a>
          <p class="text-[11px] text-rose-200/80 leading-relaxed pl-1">
            <strong>ယူနည်း:</strong> Google AI Studio တွင် <strong>"Create API key"</strong> နှိပ်၍ Copy ကူးယူပါ။
          </p>
        </div>
      </div>

      <div class="text-[10px] text-rose-400/60 text-center pt-2">
        Thiri's Koko Studio • Designed with ❤
      </div>
    </div>
  </aside>

  <!-- ================= SETTINGS MODAL ================= -->
  <div id="settingsModal" class="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 hidden flex items-center justify-center p-4">
    <div class="bg-rose-950 border border-rose-800/80 rounded-2xl w-full max-w-sm p-5 shadow-2xl space-y-4 max-h-[90vh] overflow-y-auto">
      <div class="flex items-center justify-between border-b border-rose-900/80 pb-2.5">
        <h3 class="text-sm font-bold text-white flex items-center gap-2">
          <span>⚙</span> API Settings
        </h3>
        <button onclick="closeSettingsModal()" class="text-rose-400 hover:text-white text-base">✕</button>
      </div>

      <div class="space-y-3.5 text-xs">
        <div>
          <div class="flex justify-between items-center mb-1">
            <label class="text-[11px] font-medium text-pink-300">Groq API Key (Whisper STT အတွက်)</label>
            <a href="https://console.groq.com/keys" target="_blank" class="text-[10px] text-pink-400 underline">ယူရန် ↗</a>
          </div>
          <input id="modalGroqKey" type="password" placeholder="gsk_..." class="w-full bg-rose-900/60 border border-rose-700/60 rounded-xl p-2 text-rose-100 font-mono text-xs focus:outline-none">
        </div>

        <div>
          <div class="flex justify-between items-center mb-1">
            <label class="text-[11px] font-medium text-indigo-300">Gemini API Key (Subtitle Translate အတွက်)</label>
            <a href="https://aistudio.google.com/app/apikey" target="_blank" class="text-[10px] text-indigo-400 underline">ယူရန် ↗</a>
          </div>
          <input id="modalGeminiKey" type="password" placeholder="AIzaSy..." class="w-full bg-rose-900/60 border border-rose-700/60 rounded-xl p-2 text-rose-100 font-mono text-xs focus:outline-none">
        </div>

        <div>
          <label class="block text-[11px] font-medium text-rose-300 mb-1">Gemini Model Name</label>
          <input id="modalGeminiModel" type="text" value="gemini-3.5-flash" class="w-full bg-rose-900/60 border border-rose-700/60 rounded-xl p-2 text-rose-100 font-mono text-xs focus:outline-none">
        </div>

        <div class="pt-2 border-t border-rose-900/60 space-y-3">
          <div class="text-[11px] font-semibold text-rose-300">Speed (မြန်နှုန်း)</div>
          <div class="grid grid-cols-2 gap-2.5">
            <div>
              <label class="block text-[11px] font-medium text-rose-300 mb-1">Chunk size (တစ်ခါပို့ line)</label>
              <input id="modalChunk" type="number" min="5" max="150" value="50" class="w-full bg-rose-900/60 border border-rose-700/60 rounded-xl p-2 text-rose-100 font-mono text-xs focus:outline-none">
            </div>
            <div>
              <label class="block text-[11px] font-medium text-rose-300 mb-1">Parallel (တပြိုင်နက်)</label>
              <input id="modalParallel" type="number" min="1" max="8" value="3" class="w-full bg-rose-900/60 border border-rose-700/60 rounded-xl p-2 text-rose-100 font-mono text-xs focus:outline-none">
            </div>
          </div>
          <p class="text-[10px] text-rose-300/70 leading-relaxed">Rate limit (429) မကြာခဏဖြစ်ရင် Parallel ကို လျှော့ပါ။ Gemini အတွက်သာ သက်ရောက်သည် (Groq က 20 line / 1 ခုစီ ပုံသေ)။</p>

          <div>
            <label class="block text-[11px] font-medium text-rose-300 mb-1">Whisper model (အသံထုတ်ရန်)</label>
            <select id="modalWhisper" class="w-full bg-rose-900/60 border border-rose-700/60 rounded-xl p-2 text-rose-100 text-xs focus:outline-none">
              <option value="whisper-large-v3-turbo">large-v3-turbo (ပိုမြန်)</option>
              <option value="whisper-large-v3">large-v3 (ပိုတိကျ)</option>
            </select>
          </div>
        </div>
      </div>

      <div class="pt-2">
        <button onclick="saveSettings()" class="w-full bg-gradient-to-r from-rose-600 to-pink-500 hover:from-rose-500 hover:to-pink-400 text-white font-semibold py-2.5 rounded-xl text-xs shadow transition active:scale-95">
          Save Keys & Close (အမြဲမှတ်ထားမည်)
        </button>
      </div>
    </div>
  </div>

  <script>
// ====================================================================
//  State
// ====================================================================
let subtitles = [];            // { id, startTime, endTime, originalText, translatedText }
let byId = new Map();          // id -> subtitle
let idxById = new Map();       // id -> position in `subtitles`
let fileBaseName = 'subtitles';

let isTranslating = false;
let stopRequested = false;
let abortCtl = null;
let cooldownUntil = 0;         // shared "rate limit" pause for all parallel workers
let run = { done: 0, total: 0, failed: 0 };

// localStorage keys (the first four are unchanged, so saved keys still work)
const KEY_GROQ = 'thiri_koko_groq_key';
const KEY_GEMINI = 'thiri_koko_gemini_key';
const KEY_TRANS_ENGINE = 'thiri_koko_trans_engine';
const KEY_GEMINI_MODEL = 'thiri_koko_gemini_model';
const KEY_WHISPER = 'thiri_koko_whisper_model';
const KEY_CHUNK = 'thiri_koko_gemini_chunk';
const KEY_PARALLEL = 'thiri_koko_gemini_parallel';
const KEY_GLOSSARY = 'thiri_koko_glossary';
const KEY_PROJECT = 'thiri_koko_project';

const DEFAULT_GEMINI_MODEL = 'gemini-3.5-flash';
const DEFAULT_WHISPER = 'whisper-large-v3-turbo';
const DEFAULT_CHUNK = 50;      // lines per Gemini request
const DEFAULT_PARALLEL = 3;    // Gemini requests at the same time
const GROQ_CHUNK = 20;         // Groq free tier has small token limits
const GROQ_PARALLEL = 1;
const MAX_ATTEMPTS = 4;        // tries per chunk
const CONTEXT_BEFORE = 3;      // neighbouring lines sent as read-only context
const CONTEXT_AFTER = 2;
const LANG_CODES = { Burmese: 'my', English: 'en', Thai: 'th', Japanese: 'ja', Chinese: 'zh' };

// ====================================================================
//  Helpers
// ====================================================================
const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

const store = {
  get(key, def = '') {
    try { const v = localStorage.getItem(key); return v === null ? def : v; } catch (e) { return def; }
  },
  set(key, val) {
    try { localStorage.setItem(key, val); return true; } catch (e) { return false; }
  },
  del(key) {
    try { localStorage.removeItem(key); } catch (e) { /* ignore */ }
  }
};

function numSetting(key, def, min, max) {
  const n = parseInt(store.get(key, ''), 10);
  return Number.isFinite(n) ? Math.min(max, Math.max(min, n)) : def;
}

// Escape text before putting it into innerHTML (stops broken layout / script injection from SRT files)
function esc(s) {
  return String(s ?? '').replace(/[&<>"']/g, c => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
  ));
}

async function readJson(res) {
  try { return await res.json(); } catch (e) { return { error: `Server error (${res.status})` }; }
}

function abortError() {
  const e = new Error('aborted');
  e.name = 'AbortError';
  return e;
}

// ====================================================================
//  UI helpers
// ====================================================================
window.addEventListener('DOMContentLoaded', () => {
  loadSettingsFields();
  document.getElementById('glossary').value = store.get(KEY_GLOSSARY, '');

  const engine = store.get(KEY_TRANS_ENGINE, 'gemini');
  const radio = document.querySelector(`input[name="transEngine"][value="${engine}"]`);
  if (radio) radio.checked = true;

  restoreProject();
});

window.addEventListener('beforeunload', (e) => {
  if (isTranslating) { e.preventDefault(); e.returnValue = ''; }
});

function showCustomAlert(msg, title = "သတိပေးချက်") {
  document.getElementById('customAlertTitle').innerText = title;
  document.getElementById('customAlertMessage').innerText = msg;
  document.getElementById('customAlertModal').classList.remove('hidden');
}

function closeCustomAlert() {
  document.getElementById('customAlertModal').classList.add('hidden');
}

function handleLangChange() {
  const lang = document.getElementById('targetLang').value;
  document.getElementById('toneContainer').style.display = (lang === 'Burmese') ? 'block' : 'none';
}

function saveActiveEngine(engine) {
  store.set(KEY_TRANS_ENGINE, engine);
}

function saveGlossary() {
  store.set(KEY_GLOSSARY, document.getElementById('glossary').value);
}

function toggleMenu(show) {
  const drawer = document.getElementById('menuDrawer');
  const overlay = document.getElementById('menuOverlay');
  if (show) {
    overlay.classList.remove('hidden');
    setTimeout(() => drawer.classList.remove('translate-x-full'), 10);
  } else {
    drawer.classList.add('translate-x-full');
    setTimeout(() => overlay.classList.add('hidden'), 300);
  }
}

function loadSettingsFields() {
  document.getElementById('modalGroqKey').value = store.get(KEY_GROQ);
  document.getElementById('modalGeminiKey').value = store.get(KEY_GEMINI);
  document.getElementById('modalGeminiModel').value = store.get(KEY_GEMINI_MODEL) || DEFAULT_GEMINI_MODEL;
  document.getElementById('modalChunk').value = numSetting(KEY_CHUNK, DEFAULT_CHUNK, 5, 150);
  document.getElementById('modalParallel').value = numSetting(KEY_PARALLEL, DEFAULT_PARALLEL, 1, 8);
  document.getElementById('modalWhisper').value = store.get(KEY_WHISPER) || DEFAULT_WHISPER;
}

function openSettingsModal() {
  loadSettingsFields();
  document.getElementById('settingsModal').classList.remove('hidden');
}

function closeSettingsModal() {
  document.getElementById('settingsModal').classList.add('hidden');
}

function saveSettings() {
  const clamp = (v, min, max, def) => {
    const n = parseInt(v, 10);
    return Number.isFinite(n) ? Math.min(max, Math.max(min, n)) : def;
  };
  store.set(KEY_GROQ, document.getElementById('modalGroqKey').value.trim());
  store.set(KEY_GEMINI, document.getElementById('modalGeminiKey').value.trim());
  store.set(KEY_GEMINI_MODEL, document.getElementById('modalGeminiModel').value.trim() || DEFAULT_GEMINI_MODEL);
  store.set(KEY_CHUNK, clamp(document.getElementById('modalChunk').value, 5, 150, DEFAULT_CHUNK));
  store.set(KEY_PARALLEL, clamp(document.getElementById('modalParallel').value, 1, 8, DEFAULT_PARALLEL));
  store.set(KEY_WHISPER, document.getElementById('modalWhisper').value);
  closeSettingsModal();
  setStatus("Settings မှတ်သားပြီးပါပြီ!", 3000);
}

let statusTimer = null;
function setStatus(text, duration = 0) {
  clearTimeout(statusTimer);
  document.getElementById('statusText').innerText = text;
  document.getElementById('statusPill').classList.remove('hidden');
  if (duration > 0) statusTimer = setTimeout(clearStatus, duration);
}

function clearStatus() {
  clearTimeout(statusTimer);
  document.getElementById('statusPill').classList.add('hidden');
  setProgress(0, 0);
}

function setProgress(done, total) {
  const wrap = document.getElementById('progressWrap');
  if (!total) { wrap.classList.add('hidden'); return; }
  wrap.classList.remove('hidden');
  document.getElementById('progressBar').style.width = Math.round((done / total) * 100) + '%';
}

function updateProgress() {
  if (!isTranslating) return;
  setStatus(`ဘာသာပြန်နေပါသည်: ${run.done}/${run.total}` + (run.failed ? ` (မအောင်မြင် ${run.failed})` : ''));
  setProgress(run.done, run.total);
}

function refreshTranslateLabel() {
  const label = document.getElementById('btnTranslateLabel');
  const btn = document.getElementById('btnTranslate');
  if (!label) return;
  btn.classList.toggle('btn-stop', isTranslating);
  if (isTranslating) { label.textContent = 'Stop'; return; }
  const total = subtitles.length;
  const done = subtitles.filter(s => s.translatedText && s.translatedText.trim()).length;
  label.textContent = (done > 0 && done < total) ? `Continue (${total - done} left)` : 'Translate All';
}

// ====================================================================
//  Subtitle list
// ====================================================================
function setSubtitles(items) {
  subtitles = items;
  byId = new Map(items.map(s => [s.id, s]));
  idxById = new Map(items.map((s, i) => [s.id, i]));
  renderList();
  saveProject();
}

function parseSRT(txt) {
  const clean = txt.replace(/^\uFEFF/, '').replace(/\r\n?/g, '\n').trim();
  const items = [];
  clean.split(/\n\s*\n/).forEach(block => {
    const l = block.trim().split('\n');
    if (l.length < 2) return;
    const tIdx = /^\d+$/.test(l[0].trim()) ? 1 : 0;
    if (l[tIdx] && l[tIdx].includes('-->')) {
      const [s, e] = l[tIdx].split('-->').map(x => x.trim());
      items.push({
        id: items.length + 1,
        startTime: s,
        endTime: e,
        originalText: l.slice(tIdx + 1).join('\n').trim(),
        translatedText: ''
      });
    }
  });
  setSubtitles(items);
}

function renderList() {
  const box = document.getElementById('subList');
  document.getElementById('subCount').innerText = subtitles.length;

  if (!subtitles.length) {
    box.innerHTML = `
      <div class="romantic-card rounded-2xl p-10 text-center border border-rose-900/40 text-rose-300/80 my-2">
        <h3 class="text-sm font-semibold text-rose-100 mb-1">No Subtitles Loaded</h3>
        <p class="text-xs text-rose-400/80">Choose File နှိပ်၍ SRT ဖိုင် သို့မဟုတ် ဗီဒီယို/အသံဖိုင် တင်ပါ</p>
      </div>`;
    refreshTranslateLabel();
    return;
  }

  box.innerHTML = subtitles.map(s => `
    <div class="sub-card rounded-2xl p-3.5 border border-rose-900/30 shadow-sm transition hover:border-rose-700/50">
      <div class="flex items-center justify-between text-[10px] font-mono text-rose-300/80 border-b border-rose-900/30 pb-1.5 mb-2">
        <span class="px-2 py-0.5 rounded-full bg-rose-500/10 text-rose-300 font-semibold">#${s.id}</span>
        <span>${esc(s.startTime.split(',')[0])} ➔ ${esc(s.endTime.split(',')[0])}</span>
      </div>

      <div class="text-xs text-rose-200/90 mb-2 leading-relaxed whitespace-pre-line">${esc(s.originalText)}</div>

      <div>
        <textarea
          id="tr-${s.id}"
          placeholder="ဘာသာပြန်စာသား..."
          oninput="onEdit(${s.id}, this.value)"
          rows="2"
          class="w-full bg-rose-950/80 border border-rose-800/50 focus:border-rose-400 rounded-xl p-2 text-xs text-rose-50 placeholder-rose-400/40 focus:outline-none resize-none leading-relaxed"
        >${esc(s.translatedText)}</textarea>
      </div>
    </div>
  `).join('');
  refreshTranslateLabel();
}

// Update one textarea without re-rendering the whole list
function setCard(id) {
  const ta = document.getElementById('tr-' + id);
  const s = byId.get(id);
  if (ta && s && ta.value !== s.translatedText) ta.value = s.translatedText;
}

function onEdit(id, val) {
  const s = byId.get(id);
  if (!s) return;
  s.translatedText = val;
  scheduleSave();
  refreshTranslateLabel();
}

// ====================================================================
//  Auto-save (so a refresh doesn't lose the work)
// ====================================================================
let saveTimer = null;
function scheduleSave() {
  clearTimeout(saveTimer);
  saveTimer = setTimeout(saveProject, 800);
}

function saveProject() {
  clearTimeout(saveTimer);
  if (!subtitles.length) { store.del(KEY_PROJECT); return; }
  store.set(KEY_PROJECT, JSON.stringify({ name: fileBaseName, subtitles }));
}

function restoreProject() {
  try {
    const raw = store.get(KEY_PROJECT, '');
    if (!raw) return;
    const p = JSON.parse(raw);
    if (!p || !Array.isArray(p.subtitles) || !p.subtitles.length) return;
    const items = p.subtitles
      .filter(s => s && typeof s.startTime === 'string' && typeof s.endTime === 'string')
      .map((s, i) => ({
        id: Number.isFinite(Number(s.id)) ? Number(s.id) : i + 1,
        startTime: s.startTime,
        endTime: s.endTime,
        originalText: String(s.originalText ?? ''),
        translatedText: String(s.translatedText ?? '')
      }));
    if (!items.length) return;
    fileBaseName = p.name || 'subtitles';
    setSubtitles(items);
    setStatus(`ယခင်အလုပ်ကို ပြန်ဖွင့်ပေးထားပါသည် (${items.length} lines)`, 4000);
  } catch (e) { /* ignore broken save */ }
}

function clearProject() {
  toggleMenu(false);
  if (isTranslating) { showCustomAlert('ဘာသာပြန်နေဆဲဖြစ်လို့ Stop အရင်နှိပ်ပါ'); return; }
  subtitles = [];
  byId = new Map();
  idxById = new Map();
  store.del(KEY_PROJECT);
  renderList();
  setStatus('သိမ်းထားတာတွေ ဖျက်လိုက်ပါပြီ', 3000);
}

// ====================================================================
//  File upload (SRT or audio/video)
// ====================================================================
document.getElementById('fileInput').addEventListener('change', async (e) => {
  const input = e.target;
  const file = input.files[0];
  if (!file) return;

  if (isTranslating) {
    showCustomAlert('ဘာသာပြန်နေဆဲဖြစ်လို့ Stop အရင်နှိပ်ပါ');
    input.value = '';
    return;
  }

  const ext = file.name.split('.').pop().toLowerCase();
  fileBaseName = file.name.replace(/\.[^.]+$/, '') || 'subtitles';

  try {
    if (ext === 'srt') {
      parseSRT(await file.text());
      if (subtitles.length) setStatus(`SRT loaded: ${subtitles.length} lines`, 3000);
      else showCustomAlert('ဒီ SRT ဖိုင်ထဲမှာ subtitle မတွေ့ပါ');
    } else {
      const groqKey = store.get(KEY_GROQ);
      if (!groqKey) {
        showCustomAlert('Audio/Video transcribe လုပ်ရန် Groq API Key လိုအပ်ပါသည်။ Menu -> Settings တွင် Groq API Key ထည့်သွင်းပေးပါခင်ဗျာ။');
        return;
      }

      setStatus("Groq Whisper ဖြင့် အသံဖိုင်မှ မူရင်းစာတန်းထိုး ထုတ်ယူနေပါသည် (ဖိုင်ကြီးရင် အချိန်ယူပါမည်)...");
      const fd = new FormData();
      fd.append('file', file);
      fd.append('apiKey', groqKey);
      fd.append('model', store.get(KEY_WHISPER) || DEFAULT_WHISPER);

      try {
        const res = await fetch('/api/transcribe', { method: 'POST', body: fd });
        const data = await readJson(res);
        if (!res.ok || data.error) throw new Error(data.error || `Server error (${res.status})`);
        setSubtitles(data.subtitles);
        setStatus("Original Subtitles ထုတ်ယူပြီးပါပြီ!", 3000);
      } catch (err) {
        showCustomAlert(err.message, "Transcription Error");
        clearStatus();
      }
    }
  } finally {
    input.value = '';   // lets the same file be chosen again
  }
});

// ====================================================================
//  Translation
// ====================================================================
function stopTranslate() {
  stopRequested = true;
  if (abortCtl) abortCtl.abort();
  setStatus('ရပ်နေပါသည်...');
}

async function sleepInterruptible(ms) {
  const end = Date.now() + ms;
  while (!stopRequested && Date.now() < end) await sleep(Math.min(250, end - Date.now()));
}

// When one request hits the rate limit, ALL workers pause together
async function waitCooldown() {
  let waited = false;
  while (!stopRequested && Date.now() < cooldownUntil) {
    waited = true;
    const left = Math.ceil((cooldownUntil - Date.now()) / 1000);
    setStatus(`Rate limit ဖြစ်နေလို့ ${left} စက္ကန့် စောင့်နေပါသည်...`);
    await sleep(Math.min(500, Math.max(50, cooldownUntil - Date.now())));
  }
  if (waited) updateProgress();
}

async function callTranslate(items, cfg) {
  const first = idxById.get(items[0].id);
  const last = idxById.get(items[items.length - 1].id);
  const ctx = (a, b) => subtitles.slice(Math.max(0, a), Math.max(0, b))
    .map(s => ({ id: s.id, originalText: s.originalText }));

  const body = {
    subtitles: items.map(s => ({ id: s.id, originalText: s.originalText })),
    contextBefore: ctx(first - CONTEXT_BEFORE, first),
    contextAfter: ctx(last + 1, last + 1 + CONTEXT_AFTER),
    targetLang: cfg.targetLang,
    toneStyle: cfg.toneStyle,
    glossary: cfg.glossary,
    provider: cfg.engine,
    modelName: cfg.modelName,
    apiKey: cfg.apiKey
  };

  let res;
  try {
    res = await fetch('/api/translate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal: abortCtl.signal
    });
  } catch (e) {
    if (e.name === 'AbortError') throw e;
    const err = new Error('Network ချိတ်ဆက်မရပါ');
    err.kind = 'server';
    throw err;
  }

  const data = await readJson(res);
  if (!res.ok || data.error) {
    const err = new Error(data.error || `Server error (${res.status})`);
    err.kind = data.kind || (res.status === 429 ? 'rate' : (res.status >= 500 ? 'server' : 'bad'));
    err.retryAfter = data.retryAfter;
    throw err;
  }
  return Array.isArray(data.translations) ? data.translations : [];
}

// Translates one chunk. Retries only the lines that are still missing.
async function translateChunk(chunk, cfg) {
  let pending = chunk.slice();
  let lastErr = null;

  for (let attempt = 0; attempt < MAX_ATTEMPTS && pending.length; attempt++) {
    if (stopRequested) throw abortError();
    await waitCooldown();
    if (stopRequested) throw abortError();

    try {
      const translations = await callTranslate(pending, cfg);
      const wanted = new Set(pending.map(s => s.id));
      for (const t of translations) {
        const id = Number(t.id);
        const text = typeof t.translatedText === 'string' ? t.translatedText.trim() : '';
        if (!wanted.has(id) || !text) continue;
        byId.get(id).translatedText = text;
        setCard(id);
      }
      const before = pending.length;
      pending = pending.filter(s => !s.translatedText.trim());
      run.done += before - pending.length;
      updateProgress();
      scheduleSave();

      if (pending.length) {
        lastErr = new Error(`Model က ${pending.length} line ကို ပြန်မပေးပါ`);
        await sleepInterruptible(800);
      }
    } catch (e) {
      if (e.name === 'AbortError' || stopRequested) throw abortError();
      if (e.kind === 'auth' || e.kind === 'bad') { e.fatal = true; throw e; }   // retrying can't help
      lastErr = e;
      if (e.kind === 'rate') {
        const base = e.retryAfter ? e.retryAfter * 1000 + 500 : 5000 * Math.pow(2, attempt);
        const wait = Math.min(90000, base) + Math.random() * 1000;
        cooldownUntil = Math.max(cooldownUntil, Date.now() + wait);
      } else {
        await sleepInterruptible(2000 * Math.pow(2, attempt));
      }
    }
  }

  if (pending.length) {
    const err = lastErr || new Error('ဘာသာပြန်အဖြေ မပြည့်စုံပါ');
    err.chunkFailed = pending.length;
    throw err;
  }
}

async function translateAll() {
  if (isTranslating) { stopTranslate(); return; }

  const engine = document.querySelector('input[name="transEngine"]:checked')?.value || 'gemini';
  const apiKey = store.get(engine === 'gemini' ? KEY_GEMINI : KEY_GROQ);

  if (!apiKey) {
    showCustomAlert(`${engine.toUpperCase()} API Key မရှိသေးပါ။ Menu -> Settings တွင် API Key အရင်ထည့်ပေးပါခင်ဗျာ။`);
    return;
  }
  if (!subtitles.length) {
    showCustomAlert("ဘာသာပြန်ရန် Subtitle မရှိသေးပါ");
    return;
  }

  // "Overwrite": forget the old translations first, so Continue works properly if we get interrupted
  const overwriteBox = document.getElementById('overwrite');
  if (overwriteBox.checked) {
    subtitles.forEach(s => { s.translatedText = ''; setCard(s.id); });
    overwriteBox.checked = false;
  }

  const todo = subtitles.filter(s => !s.translatedText.trim());
  if (!todo.length) {
    showCustomAlert('အားလုံး ဘာသာပြန်ပြီးသားပါ။ အစကပြန်ပြန်လိုရင် "Overwrite" ကို အမှန်ခြစ်ပြီး ပြန်နှိပ်ပါ။', 'ပြီးပြီးသားပါ');
    return;
  }

  const lang = document.getElementById('targetLang').value;
  const cfg = {
    engine,
    apiKey,
    modelName: engine === 'gemini' ? (store.get(KEY_GEMINI_MODEL) || DEFAULT_GEMINI_MODEL) : 'llama-3.3-70b-versatile',
    targetLang: lang,
    toneStyle: lang === 'Burmese' ? document.getElementById('toneStyle').value : 'natural',
    glossary: document.getElementById('glossary').value.trim()
  };

  const chunkSize = engine === 'gemini' ? numSetting(KEY_CHUNK, DEFAULT_CHUNK, 5, 150) : GROQ_CHUNK;
  const parallel = engine === 'gemini' ? numSetting(KEY_PARALLEL, DEFAULT_PARALLEL, 1, 8) : GROQ_PARALLEL;

  const chunks = [];
  for (let i = 0; i < todo.length; i += chunkSize) chunks.push(todo.slice(i, i + chunkSize));

  run = { done: 0, total: todo.length, failed: 0 };
  isTranslating = true;
  stopRequested = false;
  abortCtl = new AbortController();
  cooldownUntil = 0;
  refreshTranslateLabel();
  updateProgress();

  let fatal = null;
  let lastErr = null;
  let consecutiveFails = 0;
  let next = 0;

  // A small pool: `parallel` workers keep taking the next chunk until none are left
  const worker = async () => {
    while (!stopRequested && !fatal) {
      const i = next++;
      if (i >= chunks.length) return;
      try {
        await translateChunk(chunks[i], cfg);
        consecutiveFails = 0;
      } catch (e) {
        if (e.name === 'AbortError') return;
        if (e.fatal) { fatal = e; abortCtl.abort(); return; }
        run.failed += e.chunkFailed || chunks[i].length;
        lastErr = e;
        updateProgress();
        if (++consecutiveFails >= 3) {   // something is really wrong (quota, outage...) - stop early
          fatal = new Error(`ဆက်တိုက် မအောင်မြင်ပါ: ${e.message}`);
          abortCtl.abort();
          return;
        }
      }
    }
  };

  await Promise.all(Array.from({ length: Math.min(parallel, chunks.length) }, worker));

  isTranslating = false;
  abortCtl = null;
  refreshTranslateLabel();
  saveProject();

  const left = subtitles.filter(s => !s.translatedText.trim()).length;
  if (fatal) {
    clearStatus();
    const hint = engine === 'gemini' ? '\n(Groq Engine သို့ ပြောင်းလဲ၍လည်း စမ်းသပ်နိုင်ပါသည်)' : '';
    showCustomAlert(`${fatal.message}\n\nဘာသာပြန်ပြီးသားတွေ သိမ်းထားပါသည်။ ပြင်ပြီးရင် Continue နှိပ်ပါ။${hint}`, "ဘာသာပြန်ဆိုမှု အဆင်မပြေပါ");
  } else if (stopRequested) {
    setProgress(0, 0);
    setStatus(`ရပ်လိုက်ပါပြီ (ကျန်ရှိ ${left} line) — Continue နှိပ်ရင် ဆက်လုပ်ပါမည်`, 5000);
  } else if (left > 0) {
    clearStatus();
    showCustomAlert(`${left} line ဘာသာမပြန်နိုင်သေးပါ။ Continue ကို ပြန်နှိပ်ရင် ကျန်တာကိုသာ ပြန်လုပ်ပါမည်။\n\nနောက်ဆုံး error: ${lastErr ? lastErr.message : '-'}`, "တစ်စိတ်တစ်ပိုင်း ပြီးစီးပါသည်");
  } else {
    setProgress(run.total, run.total);
    setStatus("ဘာသာပြန်ဆိုခြင်း အောင်မြင်စွာ ပြီးစီးပါပြီ!", 4000);
  }
}

// ====================================================================
//  Export
// ====================================================================
function buildSRT(useTranslation) {
  return subtitles.map((s, i) => {
    const text = useTranslation ? (s.translatedText || s.originalText) : s.originalText;
    return `${i + 1}\n${s.startTime} --> ${s.endTime}\n${text}\n\n`;
  }).join('');
}

function saveTextFile(content, filename) {
  const blob = new Blob([content], { type: 'text/plain;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}

function downloadOriginalSRT() {
  if (!subtitles.length) {
    showCustomAlert("Export လုပ်ရန် Subtitle မရှိသေးပါ");
    return;
  }
  saveTextFile(buildSRT(false), `${fileBaseName}.original.srt`);
}

function downloadTranslatedSRT() {
  if (!subtitles.length) {
    showCustomAlert("Export လုပ်ရန် Subtitle မရှိသေးပါ");
    return;
  }
  const code = LANG_CODES[document.getElementById('targetLang').value] || 'tr';
  saveTextFile(buildSRT(true), `${fileBaseName}.${code}.srt`);
}
  </script>
</body>
</html>
"""


@app.route('/')
def index():
    resp = Response(HTML_PAGE, mimetype='text/html')
    resp.headers['Cache-Control'] = 'no-cache'
    return resp


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), threaded=True)
